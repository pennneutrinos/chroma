import xml.etree.ElementTree as et
from typing import Optional

import numpy as np
from collections import deque

import chroma.geometry
from chroma.detector import Detector
from chroma.transform import make_rotation_matrix
from chroma.geometry import Mesh, Solid
from chroma.log import logger
from copy import deepcopy

from chroma.rat import gdml
from chroma.rat import gen_mesh
import gmsh
from pathlib import Path
# To convert length and angle units to cm and radians
units = gdml.units




from chroma.demo.optics import vacuum

DEFAULT_SOLID_COLOR=0xEEA0A0A0
DEFAULT_PMT_COLOLR=0xA0A05000

def _default_volume_classifier(volume_ref, material_ref, parent_material_ref):
    '''This is an example volume classifier, primarily for visualization'''
    if 'OpDetSensitive' in volume_ref:
        return 'pmt', dict(material1=vacuum, material2=vacuum, color=0xA0A05000, surface=None, channel_type=0)
    elif material_ref == parent_material_ref:
        return 'omit', dict()
    elif 'leg' in volume_ref:
        return 'omit', dict()
    else:
        return 'solid', dict(material1=vacuum, material2=vacuum, color=0xEEA0A0A0, surface=None)


class Volume:
    '''
    Represents a GDML volume and the volumes placed inside it (physvol) as
    children. Keeps track of position and rotation of the GDML solid.
    '''

    def __init__(self, name: str, loader: "RATGeoLoader", phys_vol_name: str = '/BUILDROOT'):
        self.placementName = phys_vol_name
        self.name = name
        elem = loader.vol_xml_map[name]
        self.material_ref = elem.find('materialref').get('ref')
        self.solid_ref = elem.find('solidref').get('ref')
        self.in_gmsh_model = False
        self.gmsh_tag = -1

        placements = elem.findall('physvol')
        self.children = []
        self.child_pos = []
        self.child_rot = []
        for placement in placements:
            vol = Volume(placement.find('volumeref').get('ref'), loader,
                         self.placementName + '/' + placement.get('name'))
            pos, rot = loader.get_pos_rot(placement)
            self.children.append(vol)
            self.child_pos.append(pos)
            self.child_rot.append(rot)

    def show_hierarchy(self, indent=''):
        print(indent + str(self), self.solid_ref, self.material_ref)
        for child in self.children:
            child.show_hierarchy(indent=indent + ' ')

    def flat_view(self):
        """
        Returns a dict of all volumes in the hierarchy, keyed by placement name.
        Returns: dict[str, Volume]
        """
        placement_map = {self.placementName: self}
        for volume in self.children:
            placement_map.update(volume.flat_view())
        return placement_map

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class RATGeoLoader:
    '''
    This class supports loading a geometry from a GDML file by directly parsing 
    the XML. A subset of GDML is supported here, and exceptions will be raised
    if the GDML uses unsupported features.
    '''

    def __init__(self, gdml_file, refinement_order=0, override_worldref=None):
        ''' 
        Read a geometry from the specified GDML file.
        '''
        # GDML mesh refinement order. This massively increases the number of triangles. Be careful!

        self.refinement_order = refinement_order
        self.gdml_file = gdml_file
        xml = et.parse(gdml_file)
        gdml_tree = xml.getroot()

        define = gdml_tree.find('define')
        self.pos_map = {pos.get('name'): pos for pos in define.findall('position')}
        self.rot_map = {rot.get('name'): rot for rot in define.findall('rotation')}

        self.materials_used = []
        self.material_lookup = {}
        materials = gdml_tree.find('materials')
        for material_idx, material in enumerate(materials):
            self.materials_used.append(vacuum)
            self.material_lookup[material.get('name')] = material_idx
        # todo: add Material properties
        solids = gdml_tree.find('solids')
        self.solid_xml_map = {solid.get('name'): solid for solid in solids}

        structure = gdml_tree.find('structure')
        volumes = structure.findall('volume')
        self.vol_xml_map = {v.get('name'): v for v in volumes}

        world_ref = gdml_tree.find('setup').find('world').get('ref')
        if override_worldref is not None:
            world_ref = override_worldref
        self.world = Volume(world_ref, self)
        self.placement_to_volume_map = self.world.flat_view()
        # self.mesh_cache = {}
        self.vertex_positions = {vertex.get('name'): gdml.get_vals(vertex) for vertex in define.findall('position')}

        # Initialize gmsh
        gmsh.initialize()
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 32)  # number of meshes per 2*pi radian
        gmsh.option.setNumber('Mesh.MinimumCircleNodes', 32)  # number of nodes per circle
        gmsh.option.setNumber('General.Verbosity', 2)
        gmsh.option.setNumber('General.NumThreads', 0)
        gmsh.option.setNumber('Geometry.ToleranceBoolean', 0.0001)
        self.triangle_typeid = gmsh.model.mesh.getElementType('Triangle', 1)

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
        elem = self.solid_xml_map[solid_ref]
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
            logger.debug(f"Generating volume {volume.name}\tsolid ref: {volume.solid_ref}\tposition: {pos}")
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
            # self.placement_to_volume_map[volume.placementName] = tag_or_mesh
            gen_mesh.gdml_transform(tag_or_mesh, pos, rot)
            volume.in_gmsh_model = True
            volume.gmsh_tag = tag_or_mesh

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
        gen_mesh.conform_model(self.world)
        # detector = self.retrieve_mesh()

        return detector


    def retrieve_mesh(self):
        '''
        Processes solid into a chroma Mesh.
        tag_or_mesh is either a gmsh tag that relates to the root solid, or a chroma.Mesh object.
        If tag_or_mesh is a gmsh tag, apply downstream gmsh mesh generation routine and package the generated mesh into a
        chorma.Mesh object. If tag_or_mesh is already a mesh, do nothing. Simply return the mesh.
        Returns a chroma_mesh or None.
        '''
        occ = gmsh.model.occ
        mesh = gmsh.model.mesh
        # occ.synchronize()
        # mesh.generate(2)
        for _ in range(self.refinement_order):
            mesh.refine()
        node_tags, coords, _ = mesh.getNodes()
        node_tag_to_index = {node_tag: i for i, node_tag in enumerate(node_tags)}
        coords = np.reshape(coords, (-1, 3))

        nFaces = 0

        inside_materials = []
        outside_materials = []
        node_idx_per_face = []
        surface_tags_to_materialrefs = self.assign_material_to_surfaces()
        for surf_tag, (mref_in, mref_out) in surface_tags_to_materialrefs.items():
            logger.info(f"Processing surface {surf_tag} with materials {mref_in} and {mref_out}")
            face_tags_for_surf, node_tag_per_face_for_surf = mesh.getElementsByType(self.triangle_typeid, surf_tag)
            # face_tags_for_surf -= 1  # because tags are 1-indexed
            # node_tag_per_face_for_surf = np.reshape(node_tags_for_surf, (-1, 3))
            surf_material_idx_in = self.material_lookup[mref_in]
            surf_material_idx_out = self.material_lookup[mref_out]

            inside_materials.extend([surf_material_idx_in] * len(face_tags_for_surf))
            outside_materials.extend([surf_material_idx_out] * len(face_tags_for_surf))
            # fancy numpy magic for assigning node indices to faces
            node_idx_per_face_for_surf = np.vectorize(node_tag_to_index.get)(node_tag_per_face_for_surf)
            node_idx_per_face.extend(node_idx_per_face_for_surf)
            nFaces += len(face_tags_for_surf)
        node_idx_per_face = np.reshape(node_idx_per_face, (-1, 3))
        detector = chroma.detector.Detector(detector_material=chroma.geometry.Material("water"))  # TODO: figure out what to do with this. This is only used for G4. Remove?
        print(coords.shape)
        print(node_idx_per_face.shape)
        detector.mesh = Mesh(coords, node_idx_per_face, remove_duplicate_vertices=False, remove_null_triangles=False)
        detector.colors = np.ones(nFaces) * DEFAULT_SOLID_COLOR
        detector.solid_id = np.ones(nFaces, dtype=int)
        detector.unique_materials = self.materials_used
        detector.inner_material_index = np.asarray(inside_materials)
        detector.outer_material_index = np.asarray(outside_materials)
        detector.unique_surfaces = [None] #TODO
        detector.surface_index = np.zeros(nFaces, dtype=int)
        # TODO: add channels
        return detector
        #return node_idx_per_face, coords, inside_materials, outside_materials


    def assign_material_to_surfaces(self):
        surface_tags = [dimTag[1] for dimTag in gmsh.model.getEntities(2)]
        surface_tags_to_placementNames = {tag: ['', ''] for tag in surface_tags}  # inside, outside
        # Assign material to surface based on the volumes they bound
        for placement_name, volume in self.placement_to_volume_map.items():
            if volume.in_gmsh_model:
                volume_tag = volume.gmsh_tag
                boundaries = gmsh.model.getBoundary(gen_mesh.getDimTags(3, volume_tag))
                for surf in boundaries:
                    assert surf[0] == 2
                    surf_tag = np.abs(surf[1])
                    surf_sign = np.sign(surf[1])
                    side_idx = 0 if surf_sign == 1 else 1
                    assert surface_tags_to_placementNames[surf_tag][side_idx] == '', \
                        logger.critical(f"Surface {surf_tag} is assigned twice!")
                    surface_tags_to_placementNames[surf_tag][side_idx] = volume.placementName
        # Unassigned surfaces at this point neighbor the mother volume
        for surf_tag, placementNames in surface_tags_to_placementNames.items():
            if placementNames[0] == '':
                assert placementNames[1] != '', logger.critical(f"Surface {surf_tag} is not assigned!")
                placementNames[0] = Path(placementNames[1]).parent.as_posix()
            if placementNames[1] == '':
                placementNames[1] = Path(placementNames[0]).parent.as_posix()
        # Assign material to surfaces now
        surface_tags_to_materialrefs = {}
        for surf_tag, placementNames in surface_tags_to_placementNames.items():
            mrefs = []
            for pname in placementNames:
                if pname == '/':
                    mrefs.append(self.world.material_ref)
                else:
                    mrefs.append(self.placement_to_volume_map[pname].material_ref)
            surface_tags_to_materialrefs[surf_tag] = mrefs
        return surface_tags_to_materialrefs

    def visualize(self):
        gmsh.fltk.run()


