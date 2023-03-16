import xml.etree.ElementTree as et
import numpy as np
from collections import deque

from chroma.detector import Detector
from chroma.transform import make_rotation_matrix
from chroma.geometry import Mesh, Solid
from chroma.log import logger

#Using the PyMesh library instead of Chroma's internal meshes to support boolean
#operations like subtractions or unions. Perhaps Chroma's meshes should be
#replaced _entirely_ by PyMesh...
from chroma.gdml import loader_helper as helper
from chroma.gdml import gen_mesh
import gmsh

def mprint(*words):
    # Delete this after debugging
    print("\033[93mDebug>",*words,"\033[0m")
    
def generate_mesh_from_obj():
    elementTags, nodeTags = gmsh.model.mesh.getElementsByType(2)
    faces = np.asarray(nodeTags)
    faces = np.reshape(faces, (-1, 3))
    node_unique = list(set(nodeTags))
    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords)
    coords = np.reshape(coords, (-1, 3))

    coord_dict = {}
    for nodeTag, coord in zip(nodeTags, coords):
        coord_dict[nodeTag] = coord

    verts = np.zeros((int(np.max(node_unique)+1), 3))
    for node in node_unique:
        verts[node] = coord_dict.get(node)
    return verts, faces


#To convert length and angle units to cm and radians
units = { 'cm':10, 'mm':1, 'm':1000, 'deg':np.pi/180, 'rad':1 }

class Volume:
    '''
    Represents a GDML volume and the volumes placed inside it (physvol) as 
    childeren. Keeps track of position and rotation of the GDML solid.
    '''
    def __init__(self,name,gdml):
        self.name = name
        elem = gdml.vol_map[name]
        self.material_ref = elem.find('materialref').get('ref')
        self.solid_ref = elem.find('solidref').get('ref')
        placements = elem.findall('physvol')
        self.children = []
        self.child_pos = []
        self.child_rot = []
        for placement in placements:
            vol = Volume(placement.find('volumeref').get('ref'), gdml)
            pos, rot = gdml.get_pos_rot(placement)
            self.children.append(vol)
            self.child_pos.append(pos)
            self.child_rot.append(rot)
    def show_hierarchy(self,indent=''):
        print(indent+str(self), self.solid,self.material_ref)
        for child in self.children:
            child.show_hierarchy(indent=indent+' ')
    def __str__(self):
        return self.name
    def __repr__(self):
        return str(self)

from chroma.demo.optics import vacuum
def _default_volume_classifier(volume_ref, material_ref, parent_material_ref):
    '''This is an example volume classifier, primarily for visualization'''
    #if 'OpDetSensitive' in volume_ref:
    ## MORGAN: Adjust later, color here just for tests
    mprint(volume_ref)
    if 'pmts_body' in volume_ref:
        return 'pmt',dict(material1=vacuum, material2=vacuum, color=0xA0A05000, surface=None, channel_type=0)
    elif 'pmts_inner1' in volume_ref:
        return 'pmt',dict(material1=vacuum, material2=vacuum, color=0xA023a000, surface=None, channel_type=0)
    elif 'pmts_inner2' in volume_ref:
        return 'pmt',dict(material1=vacuum, material2=vacuum, color=0xA0008da0, surface=None, channel_type=0)
    elif 'pmts_dynode' in volume_ref:
        return 'pmt',dict(material1=vacuum, material2=vacuum, color=0xA0A05000, surface=None, channel_type=0)
    elif material_ref == parent_material_ref:
        return 'omit',dict()
    else:
        return 'solid',dict(material1=vacuum, material2=vacuum, color=0xEEA0A0A0, surface=None)
        

class GDMLLoader:
    '''
    This class supports loading a geometry from a GDML file by directly parsing 
    the XML. A subset of GDML is supported here, and exceptions will be raised
    if the GDML uses unsupported features.
    '''
    
    def __init__(self, gdml_file, refinement_order=0):
        ''' 
        Read a geometry from the specified GDML file.
        '''
        self.refinement_order = refinement_order
        
        self.gdml_file = gdml_file
        xml = et.parse(gdml_file)
        gdml = xml.getroot()
        
        define = gdml.find('define')
        self.pos_map = { pos.get('name'):pos for pos in define.findall('position') }
        self.rot_map = { rot.get('name'):rot for rot in define.findall('rotation') }
        
        solids = gdml.find('solids')
        self.solid_map = { solid.get('name'):solid for solid in solids }
        mprint(self.solid_map.keys())
        
        structure = gdml.find('structure')
        volumes = structure.findall('volume')
        self.vol_map = { v.get('name'):v for v in volumes }
        mprint(self.vol_map.keys())
        
        world_ref = gdml.find('setup').find('world').get('ref')
        self.world = Volume(world_ref, self)
        self.mesh_cache = {}

        ## Initialize gmsh
        gmsh.initialize()
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 32) # number of meshes per 2*pi radian
        gmsh.option.setNumber('Mesh.MinimumCircleNodes', 32) # number of nodes per circle
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
        return pos,rot
        
    def get_vals(self, elem, value_attr=['x', 'y', 'z'], unit_attr='unit'):
        '''
        Calls get_val for a list of attributes (value_attr). The result 
        values are scaled from the unit specified in the unit_attrib attribute.
        '''
        scale = units[elem.get(unit_attr)] if unit_attr is not None else 1.0
        return [self.get_val(elem, attr) * scale for attr in value_attr]
        
    def get_val(self, elem, attr, default=None):
        '''
        Calls eval on the value of the attribute attr if it exists and return 
        it. Otherwise return the default specified. If there is no default 
        specifed, raise an exception.
        '''
        txt = elem.get(attr, default=None)
        assert txt is not None or default is not None, 'Missing attribute: '+attr
        return eval(txt, {}, {}) if txt is not None else default
        
    def get_mesh(self,solid_ref):
        '''
        Build a PyMesh mesh for the solid identified by solid_ref if the named
        solid has not been built. If it has been built, a cached mesh is returned.
        If the tag of the solid is not yet implemented, or it uses features not
        yet implemented, this will raise an exception.
        '''
        mprint("get_mesh",solid_ref)
        if self.solidsToIgnore(solid_ref):
            logger.info(f"Ignoring solid: {solid_ref}")
            return None
        logger.info(f"Generating Solid {solid_ref}")
        elem = self.solid_map[solid_ref]
        mesh_type = elem.tag
        mprint(f'mesh_type: {mesh_type}')
        if mesh_type in ('union', 'subtraction', 'intersection'):
            a = self.get_mesh(elem.find('first').get('ref'))
            b = self.get_mesh(elem.find('second').get('ref'))
            fpos, frot = self.get_pos_rot(elem, refs=('firstposition', 'firstrotation'))
            pos, rot = self.get_pos_rot(elem)
            posrot_entries = (fpos, frot, pos, rot)
            posrot_vals = [None]*4
            for i, entry in enumerate(posrot_entries):
                if entry is not None:
                    posrot_vals[i] = helper.get_vals(entry)
            noUnion = self.noUnionClassifier(solid_ref)
            mesh = gen_mesh.gdml_boolean(a, b, mesh_type, firstpos=posrot_vals[0], firstrot=posrot_vals[1], pos=posrot_vals[2], rot=posrot_vals[3], noUnion=noUnion)
            return mesh
        dispatcher = {
            'box':              helper.box,
            'eltube':           helper.eltube,
            'orb':              helper.orb,
            'polycone':         helper.polycone,
            'polyhedra':        helper.polyhedra,
            'sphere':           helper.sphere,
            'torus':            helper.torus,
            'tube':             helper.tube,
            'ellipsoid':        helper.ellipsoid,
            'opticalsurface':   helper.ignore,
        }
        mprint("one")
        generator = dispatcher.get(mesh_type, helper.notImplemented)
        mprint("two")
        mesh = generator(elem)
        mprint("three")
        return mesh
        
    def build_detector(self, detector=None, volume_classifier=_default_volume_classifier, solidsToIgnore=None, noUnion=None):
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
        mprint("build_detector called on", detector)
        if detector is None:
            detector = Detector(vacuum)
        if solidsToIgnore is None: # by default ignore nothing
            self.solidsToIgnore = lambda _: False
        else:
            self.solidsToIgnore=solidsToIgnore

        if noUnion is None:
            self.noUnionClassifier = lambda _: False
        else:
            self.noUnionClassifier = noUnion
        q = deque()
        q.append([self.world, np.zeros(3), np.identity(3), None])
        while len(q):
            v, pos, rot, parent_material_ref = q.pop()
            for child, c_pos, c_rot in zip(v.children, v.child_pos, v.child_rot):
                c_pos = self.get_vals(c_pos) if c_pos is not None else np.zeros(3)
                c_rot = self.get_vals(c_rot) if c_rot is not None else np.identity(3)
                c_pos = np.matmul(c_pos,rot)+pos
                x_rot = make_rotation_matrix(c_rot[0], [1, 0, 0])
                y_rot = make_rotation_matrix(c_rot[1], [0, 1, 0])
                z_rot = make_rotation_matrix(c_rot[2], [0, 0, 1])
                c_rot = np.matmul(rot, np.matmul(x_rot, np.matmul(y_rot, z_rot))) #FIXME verify this order
                q.append([child, c_pos, c_rot, v.material_ref])
            classification, kwargs = volume_classifier(v.name, v.material_ref, parent_material_ref)
            mprint(f" build_detector -> {v.name}, class:{classification}")
            if classification == 'omit':
                logger.debug(f"Volume {v.name} is omitted.")
                continue
            if v.solid_ref in self.mesh_cache:
                logger.info(f"Using cache of solid {v.solid_ref} for volume {v.name}")
                mesh = self.mesh_cache[v.solid_ref]
            else:
                gmsh.clear()
                obj = self.get_mesh(v.solid_ref)
                gmsh.model.occ.synchronize()
                gmsh.model.mesh.generate(2)
                for _ in range(self.refinement_order):
                    gmsh.model.mesh.refine()
                verts, faces = generate_mesh_from_obj()
                mesh = Mesh(verts, faces)
                self.mesh_cache[v.solid_ref] = mesh
            if mesh is None:
                continue
            if classification == 'pmt':
                channel_type = kwargs.pop('channel_type',None)
                solid = Solid(mesh, **kwargs)
                detector.add_pmt(solid, displacement=pos, rotation=rot, channel_type=channel_type)   
            elif classification == 'solid':
                solid = Solid(mesh, **kwargs)
                detector.add_solid(solid, displacement=pos, rotation=rot)   
            else:
                raise Exception('Unknown volume classification: '+classification)
        return detector
