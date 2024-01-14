import xml.etree.ElementTree as et
import numpy as np
from collections import deque

from chroma.detector import Detector
from chroma.transform import make_rotation_matrix
from chroma.geometry import Mesh, Solid
from chroma.log import logger
from copy import deepcopy

from . import gdml
from . import gen_mesh
import gmsh

# To convert length and angle units to cm and radians
units = gdml.units




from chroma.demo.optics import vacuum


def _default_volume_classifier(volume_ref, material_ref, parent_material_ref):
    '''This is an example volume classifier, primarily for visualization'''
    if 'OpDetSensitive' in volume_ref:
        return 'pmt', dict(material1=vacuum, material2=vacuum, color=0xA0A05000, surface=None, channel_type=0)
    elif material_ref == parent_material_ref:
        return 'omit', dict()
    else:
        return 'solid', dict(material1=vacuum, material2=vacuum, color=0xEEA0A0A0, surface=None)


class RATGeoLoader:
    '''
    This class supports loading a geometry from a GDML file by directly parsing 
    the XML. A subset of GDML is supported here, and exceptions will be raised
    if the GDML uses unsupported features.
    '''

    class Volume:
        '''
        Represents a GDML volume and the volumes placed inside it (physvol) as
        children. Keeps track of position and rotation of the GDML solid.
        '''

        def __init__(self, name, loader):
            self.name = name
            elem = loader.vol_map[name]
            self.material_ref = elem.find('materialref').get('ref')
            self.solid_ref = elem.find('solidref').get('ref')
            placements = elem.findall('physvol')
            self.children = []
            self.child_pos = []
            self.child_rot = []
            for placement in placements:
                vol = RATGeoLoader.Volume(placement.find('volumeref').get('ref'), loader)
                pos, rot = loader.get_pos_rot(placement)
                self.children.append(vol)
                self.child_pos.append(pos)
                self.child_rot.append(rot)

        def show_hierarchy(self, indent=''):
            print(indent + str(self), self.solid, self.material_ref)
            for child in self.children:
                child.show_hierarchy(indent=indent + ' ')

        def __str__(self):
            return self.name

        def __repr__(self):
            return str(self)

    def __init__(self, gdml_file, refinement_order=0):
        ''' 
        Read a geometry from the specified GDML file.
        '''
        # GDML mesh refinement order. This massively increases the number of triangles. Be careful!
        self.volume_to_tag_map = {}
        self.refinement_order = refinement_order
        self.gdml_file = gdml_file
        xml = et.parse(gdml_file)
        gdml_tree = xml.getroot()

        define = gdml_tree.find('define')
        self.pos_map = {pos.get('name'): pos for pos in define.findall('position')}
        self.rot_map = {rot.get('name'): rot for rot in define.findall('rotation')}

        solids = gdml_tree.find('solids')
        self.solid_map = {solid.get('name'): solid for solid in solids}

        structure = gdml_tree.find('structure')
        volumes = structure.findall('volume')
        self.vol_map = {v.get('name'): v for v in volumes}

        world_ref = gdml_tree.find('setup').find('world').get('ref')
        self.world = self.Volume(world_ref, self)
        # self.mesh_cache = {}
        self.vertex_positions = {vertex.get('name'): gdml.get_vals(vertex) for vertex in define.findall('position')}

        # Initialize gmsh
        gmsh.initialize()
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 32)  # number of meshes per 2*pi radian
        gmsh.option.setNumber('Mesh.MinimumCircleNodes', 32)  # number of nodes per circle
        # gmsh.option.setNumber('General.Verbosity', 2)
        gmsh.option.setNumber('General.NumThreads', 0)
        gmsh.option.setNumber('Geometry.ToleranceBoolean', 0.001)
        # gmsh.option.setNumber('Geometry.Tolerance', 0.001)
        gmsh.model.add(self.gdml_file)

    def get_pos_rot(self, elem, refs=('position', 'rotation')):
        ''' 
        Searches for position and rotation children of an Element. The found
        Elements are returned as a tuple as a tuple. Checks for elements
        defined inline as position, rotation tags, and dereferences 
        positionref, rotationref using defined values. Returns None if 
        neither inline nor ref is specified.
        '''
        pos_ref, rot_ref = refs
        pos = elem.find(pos_ref)
        if pos is None:
            pos = elem.find(pos_ref + 'ref')
            if pos is not None:
                pos = self.pos_map[pos.get('ref')]
        rot = elem.find(rot_ref)
        if rot is None:
            rot = elem.find(rot_ref + 'ref')
            if rot is not None:
                rot = self.rot_map[rot.get('ref')]
        return pos, rot

    def build_mesh(self, solid_ref):
        '''
        Build a mesh for the solid identified by solid_ref if the named
        solid has not been built. If it has been built, a cached mesh is returned.
        If the tag of the solid is not yet implemented, or it uses features not
        yet implemented, this will raise an exception.
        '''
        if self.solidsToIgnore(solid_ref):
            logger.info(f"Ignoring solid: {solid_ref}")
            return None
        logger.info(f"Generating Solid {solid_ref}")
        elem = self.solid_map[solid_ref]
        mesh_type = elem.tag
        if mesh_type in ('union', 'subtraction', 'intersection'):
            a = self.build_mesh(elem.find('first').get('ref'))
            b = self.build_mesh(elem.find('second').get('ref'))
            assert (not isinstance(a, Mesh)) and (not isinstance(b, Mesh)), \
                "Tessellated objects cannot be used for boolean operations!"
            fpos, frot = self.get_pos_rot(elem, refs=('firstposition', 'firstrotation'))
            pos, rot = self.get_pos_rot(elem)
            posrot_entries = (fpos, frot, pos, rot)
            posrot_vals = [None] * 4
            for i, entry in enumerate(posrot_entries):
                if entry is not None:
                    posrot_vals[i] = gdml.get_vals(entry)
            no_union = self.noUnionClassifier(solid_ref)
            logger.info(f"Performing {mesh_type} for {solid_ref}")
            mesh = gen_mesh.gdml_boolean(a, b, mesh_type, firstpos=posrot_vals[0], firstrot=posrot_vals[1],
                                         pos=posrot_vals[2], rot=posrot_vals[3], noUnion=no_union)
            return mesh
        dispatcher = {
            'box': gdml.box,
            'eltube': gdml.eltube,
            'ellipsoid': gdml.ellipsoid,
            'orb': gdml.orb,
            'polycone': gdml.polycone,
            'polyhedra': gdml.polyhedra,
            'sphere': gdml.sphere,
            'torus': gdml.torus,
            'tube': gdml.tube,
            'tessellated': lambda el: gdml.tessellated(el, self.vertex_positions),  # pass vertex cache to helper
            'torusstack': gdml.torusstack,
            'opticalsurface': gdml.ignore,
        }
        generator = dispatcher.get(mesh_type, gdml.notImplemented)
        mesh = generator(elem)
        return mesh

    def build_detector(self, detector=None, volume_classifier=_default_volume_classifier, solids_to_ignore=None,
                       no_union=None):
        '''
        Add the meshes defined by this GDML to the detector. If detector is not
        specified, a new detector will be created.
        
        The volume_classifier should be a function that returns a classification
        of the volume ('pmt','solid','omit') and kwargs passed to the Solid
        constructor for that volume: material1, material2, color, surface
        
        The different classifications have different behaviors:
        'pmt' should specify channel_type in the kwargs to identify the channel, calls add_pmt
        'solid' will add a normal solid to the Chroma geometry, calls add_solid
        'omit' will not add the Solid to the Chroma geometry
        '''
        if detector is None:
            detector = Detector(vacuum)
        if solids_to_ignore is None:  # by default ignore nothing
            self.solidsToIgnore = lambda _: False
        else:
            self.solidsToIgnore = solids_to_ignore

        if no_union is None:
            self.noUnionClassifier = lambda _: False
        else:
            self.noUnionClassifier = no_union
        q = deque()
        q.append([self.world, np.zeros(3), np.identity(3), None])
        gmsh.clear()
        while len(q):
            volume, pos, rot, parent_material_ref = q.pop()
            logger.debug(f"Generating volume {volume.name}\tsolid ref: {volume.solid_ref}")
            for child, c_pos, c_rot in zip(volume.children, volume.child_pos, volume.child_rot):
                c_pos = gdml.get_vals(c_pos) if c_pos is not None else np.zeros(3)
                c_rot = gdml.get_vals(c_rot) if c_rot is not None else np.identity(3)
                c_pos = (rot @ c_pos) + pos
                x_rot = make_rotation_matrix(c_rot[0], [1, 0, 0])
                y_rot = make_rotation_matrix(c_rot[1], [0, 1, 0])
                z_rot = make_rotation_matrix(c_rot[2], [0, 0, 1])
                c_rot = (rot @ x_rot @ y_rot @ z_rot)
                q.append([child, c_pos, c_rot, volume.material_ref])
            classification, kwargs = volume_classifier(volume.name, volume.material_ref, parent_material_ref)
            if classification == 'omit':
                logger.debug(f"Volume {volume.name} is omitted.")
                continue

            tag_or_mesh = self.build_mesh(volume.solid_ref)

            if tag_or_mesh is None or isinstance(tag_or_mesh, Mesh):
                # ignore for now FIXME
                continue
            self.volume_to_tag_map[volume.name] = tag_or_mesh
            gen_mesh.gdml_transform(tag_or_mesh, pos, rot)
            # mesh = gen_mesh.retrieve_mesh(tag_or_mesh, refinement_order=self.refinement_order)
            # FIXME: assign material
            # if classification == 'pmt':
            #     channel_type = kwargs.pop('channel_type', None)
            #     solid = Solid(mesh, **kwargs)
            #     detector.add_pmt(solid, displacement=pos, rotation=rot, channel_type=channel_type)
            # elif classification == 'solid':
            #     solid = Solid(mesh, **kwargs)
            #     detector.add_solid(solid, displacement=pos, rotation=rot)
            # else:
            #     raise Exception('Unknown volume classification: ' + classification)
        gen_mesh.retrieve_mesh(0)
        return detector

    def visualize(self):
        gmsh.fltk.run()
