import re
import xml.etree.ElementTree as et
from copy import deepcopy, copy
from typing import Optional

import numpy as np
from collections import deque

import chroma.geometry
from chroma.detector import Detector
from chroma.transform import make_rotation_matrix
from chroma.geometry import Mesh
from chroma.log import logger

from chroma.rat import gdml
from chroma.rat import gen_mesh
from .ratdb_parser import RatDBParser
import gmsh
from pathlib import Path

# To convert length and angle units to cm and radians

units = gdml.units

from chroma.demo.optics import vacuum

DEFAULT_SOLID_COLOR = 0xEEA0A0A0
DEFAULT_PMT_COLOR = 0xA0A05000


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

    def __init__(self, name: str, loader: "RATGeoLoader",
                 phys_vol_name: str = '/BUILDROOT',
                 parent_material_ref: Optional[str] = None,
                 absolute_pos: Optional[np.ndarray] = None,
                 absolute_rot: Optional[np.ndarray] = None):
        self.placementName = phys_vol_name
        self.name = name
        elem = loader.vol_xml_map[name]
        self.material_ref = elem.find('materialref').get('ref')
        self.parent_material_ref = parent_material_ref
        self.solid_ref = elem.find('solidref').get('ref')
        self.in_gmsh_model = False
        self.gmsh_tag = -1
        self.mesh = None
        self.subdetector: Optional[chroma.detector.Detector] = None
        self.skin_surface = None
        self.pmt_type = None
        self.pmt_channel = None
        self.absolute_pos = np.zeros(3) if absolute_pos is None else np.asarray(absolute_pos)
        self.absolute_rot = np.identity(3) if absolute_rot is None else np.asarray(absolute_rot)
        if absolute_pos is not None:
            self.absolute_pos = absolute_pos

        placements = elem.findall('physvol')
        self.children = []
        for placement in placements:
            c_pos, c_rot = loader.get_pos_rot(placement)
            c_pos = gdml.get_vals(c_pos) if c_pos is not None else np.zeros(3)
            c_rot = gdml.get_vals(c_rot) if c_rot is not None else np.identity(3)
            c_pos = (self.absolute_rot @ c_pos) + self.absolute_pos
            x_rot = make_rotation_matrix(c_rot[0], [1, 0, 0])
            y_rot = make_rotation_matrix(c_rot[1], [0, 1, 0])
            z_rot = make_rotation_matrix(c_rot[2], [0, 0, 1])
            c_rot = (self.absolute_rot @ x_rot @ y_rot @ z_rot)

            vol = Volume(placement.find('volumeref').get('ref'), loader,
                         self.placementName + '/' + placement.get('name'),
                         parent_material_ref=self.material_ref,
                         absolute_pos=c_pos, absolute_rot=c_rot)

            self.children.append(vol)

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
    """
    This class supports loading a geometry from a GDML file by directly parsing
    the XML. A subset of GDML is supported here, and exceptions will be raised
    if the GDML uses unsupported features.
    """

    def __init__(self, gdml_file, refinement_order=0,
                 ratdb_file=None,
                 override_worldref=None,
                 outside_material_ref=None):
        ''' 
        Read a geometry from the specified GDML file.
        '''
        self.nPMTs = 0
        self.pmt_index_to_position = None
        self.pmt_index_to_type = None
        self.ratdb_parser = None
        if ratdb_file is not None:
            self.add_ratdb(ratdb_file)
        else:
            logger.warn("No RATDB file is provided. No PMT Channel info will be loaded.")

        # GDML mesh refinement order. This massively increases the number of triangles. Be careful!
        self.refinement_order = refinement_order
        self.gdml_file = gdml_file
        xml = et.parse(gdml_file)
        gdml_tree = xml.getroot()

        # definitions
        define = gdml_tree.find('define')
        self.pos_map = {pos.get('name'): pos for pos in define.findall('position')}
        self.rot_map = {rot.get('name'): rot for rot in define.findall('rotation')}
        self.matrix_map = {matrix.get('name'): matrix for matrix in define.findall('matrix')}

        self.materials_used = []
        self.material_lookup = {}

        # materials
        materials = gdml_tree.find('materials')
        for material_xml in materials:
            if material_xml.tag != 'material':
                continue
            self.materials_used.append(gdml.create_material(self.matrix_map, material_xml))
            self.material_lookup[material_xml.get('name')] = len(self.materials_used) - 1
        solids = gdml_tree.find('solids')
        self.solid_xml_map = {solid.get('name'): solid for solid in solids}
        surfaces = solids.findall('opticalsurface')
        self.surfaces_used = [None]
        self.surface_lookup = {None: -1}
        for surface_idx, surface_xml in enumerate(surfaces, start=1):  # 0 is reserved for no surface
            self.surfaces_used.append(gdml.create_surface(self.matrix_map, surface_xml))
            self.surface_lookup[surface_xml.get('name')] = surface_idx

        # volumes
        structure = gdml_tree.find('structure')
        volumes = structure.findall('volume')
        self.vol_xml_map = {v.get('name'): v for v in volumes}
        world_ref = gdml_tree.find('setup').find('world').get('ref')
        if override_worldref is not None:
            world_ref = override_worldref
        self.world = Volume(world_ref, self)
        self.placement_to_volume_map = self.world.flat_view()
        if outside_material_ref is None:
            self.outside_material = self.world.material_ref
        else:
            self.outside_material = outside_material_ref
        assert self.outside_material in self.material_lookup, \
            f"Outside material {self.outside_material} not found in materials"
        # surfaces
        skin_surfaces_xml = structure.findall('skinsurface')
        skin_surfaces_map = {skin.find('volumeref').get('ref'): skin.get('surfaceproperty') for skin in
                             skin_surfaces_xml}
        for volume in self.placement_to_volume_map.values():
            if volume.name in skin_surfaces_map:
                volume.skin_surface = skin_surfaces_map[volume.name]
        border_surfaces_xml = structure.findall('bordersurface')
        self.border_surfaces = []
        for border_surface in border_surfaces_xml:
            surface_ref = border_surface.get('surfaceproperty')
            placement_names = [physvolref.get('ref') for physvolref in border_surface.findall('physvolref')]
            self.border_surfaces.append({'surface': surface_ref,
                                         'placement_names': placement_names})
        self.fix_orphaned_border_surfaces()
        self.vertex_positions = {vertex.get('name'): gdml.get_vals(vertex) for vertex in define.findall('position')}

        # PMT info
        if self.ratdb_parser is not None:
            self.add_pmt_info()

        # Initialize gmsh
        gmsh.initialize()
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 32)  # number of meshes per 2*pi radian
        gmsh.option.setNumber('Mesh.MinimumCircleNodes', 32)  # number of nodes per circle
        gmsh.option.setNumber('General.Verbosity', 2)
        gmsh.option.setNumber('General.NumThreads', 0)
        gmsh.option.setNumber('Geometry.ToleranceBoolean', 0.0001)
        gmsh.option.setNumber('Geometry.OCCParallel', 1)
        gmsh.option.setNumber('Mesh.MeshSizeMax', 100)
        self.triangle_typeid = gmsh.model.mesh.getElementType('Triangle', 1)

        # gmsh.option.setNumber('Geometry.Tolerance', 0.001)
        gmsh.model.add(self.gdml_file)
    
    def import_subdetector(self, volume_regex, subdetector):
        """
        Import a section of the detector that is pre-built. Useful for repeating structures like PMT arrays.
        Note: Applying pre-built subdetectors will bypass mesh conformation. This means that the subdetector element
        have no contact with any other detector elements.
        Args:
            volume_regex: str - Regular expression to match the volume name to apply the subdetector to.
            subdetector: chroma.detector.Detector
        """
        for placement, volume in self.placement_to_volume_map.items():
            if bool(re.search(volume_regex, volume.name)):
                logger.info(f"Applying subdetector to {placement}")
                volume.subdetector = copy(subdetector)
                volume.subdetector.mesh = deepcopy(subdetector.mesh)
                volume.subdetector.mesh.vertices = (
                        np.inner(volume.subdetector.mesh.vertices, volume.absolute_rot) + volume.absolute_pos
                )



    def add_ratdb(self, ratdb_file):
        self.ratdb_parser = RatDBParser(ratdb_file)

    def get_pos_rot(self, elem, refs=('position', 'rotation')):
        """
        Searches for position and rotation children of an Element. The found
        Elements are returned as a tuple as a tuple. Checks for elements
        defined inline as position, rotation tags, and dereferences
        positionref, rotationref using defined values. Returns None if
        neither inline nor ref is specified.
        """
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
        """
        Build a mesh for the solid identified by solid_ref if the named
        solid has not been built. If it has been built, a cached mesh is returned.
        If the tag of the solid is not yet implemented, or it uses features not
        yet implemented, this will raise an exception.
        """
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
        """
        Add the meshes defined by this GDML to the detector. If detector is not
        specified, a new detector will be created.

        The volume_classifier should be a function that returns a classification
        of the volume ('pmt','solid','omit') and kwargs passed to the Solid
        constructor for that volume: material1, material2, color, surface

        The different classifications have different behaviors:
        'pmt' should specify channel_type in the kwargs to identify the channel, calls add_pmt
        'solid' will add a normal solid to the Chroma geometry, calls add_solid
        'omit' will not add the Solid to the Chroma geometry
        """
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
        q.append(self.world)
        gmsh.clear()
        while len(q):
            volume: Volume = q.pop()
            if volume.subdetector is not None:
                continue
            pos = volume.absolute_pos
            rot = volume.absolute_rot
            parent_material_ref = volume.parent_material_ref
            logger.debug(f"Generating volume {volume.name}\tsolid ref: {volume.solid_ref}\tposition: {pos}")
            for child in volume.children:
                q.append(child)
            classification, kwargs = volume_classifier(volume.name, volume.material_ref, parent_material_ref)
            if classification == 'omit':
                logger.warn(f"Volume {volume.name} is omitted.")
                continue

            tag_or_mesh = self.build_mesh(volume.solid_ref)

            if tag_or_mesh is None:
                continue
            elif isinstance(tag_or_mesh, Mesh):
                volume.mesh = tag_or_mesh
                volume.mesh.vertices = np.inner(volume.mesh.vertices, rot) + pos
            else:
                gen_mesh.gdml_transform(tag_or_mesh, pos, rot)
                volume.in_gmsh_model = True
                volume.gmsh_tag = tag_or_mesh

        gen_mesh.conform_model(self.world)
        detector = self.retrieve_mesh()

        return detector

    def retrieve_mesh(self):
        """
        Processes solid into a chroma Mesh.
        tag_or_mesh is either a gmsh tag that relates to the root solid, or a chroma.Mesh object.
        If tag_or_mesh is a gmsh tag, apply downstream gmsh mesh generation routine and package the generated mesh into a
        chorma.Mesh object. If tag_or_mesh is already a mesh, do nothing. Simply return the mesh.
        Returns a chroma_mesh or None.
        """
        occ = gmsh.model.occ
        mesh = gmsh.model.mesh
        occ.synchronize()
        mesh.generate(2)
        for _ in range(self.refinement_order):
            mesh.refine()
        node_tags, coords, _ = mesh.getNodes()
        node_tag_to_index = {node_tag: i for i, node_tag in enumerate(node_tags)}
        coords = np.reshape(coords, (-1, 3))

        nFaces = 0
        nVtxs = len(coords)

        inside_materials = []
        outside_materials = []
        surfaces = []
        node_idx_per_face = []
        solid_ids = []  # id=0: not a PMT. id>0: PMT id + 1
        colors = []
        surface_tag_to_properties = self.assign_surface_properties()
        for surf_tag, prop in surface_tag_to_properties.items():
            vol_a, vol_b = prop['volumes']
            surface_ref = prop['surface']
            pmt_channel = prop['pmt_channel']
            solid_id = 0 if pmt_channel is None else pmt_channel + 1
            color = DEFAULT_PMT_COLOR if pmt_channel is not None else DEFAULT_SOLID_COLOR
            vol_a_name = vol_a.name if vol_a is not None else 'OUTSIDE'  # outside of world volume
            vol_b_name = vol_b.name if vol_b is not None else 'OUTSIDE'
            vol_a_mref = vol_a.material_ref if vol_a is not None else self.outside_material
            vol_b_mref = vol_b.material_ref if vol_b is not None else self.outside_material
            logger.info(f"Processing surface {surf_tag} between {vol_a_name} and {vol_b_name}")
            face_tags_for_surf, node_tag_per_face_for_surf = mesh.getElementsByType(self.triangle_typeid, surf_tag)
            # fancy numpy magic for assigning node indices to faces
            node_idx_per_face_for_surf = np.vectorize(node_tag_to_index.get)(node_tag_per_face_for_surf)
            orientation = gen_mesh.surface_orientation(node_idx_per_face_for_surf, coords, prop['volumes'])
            node_idx_per_face.extend(node_idx_per_face_for_surf)
            # material assignment
            if orientation == 1:
                mref_in, mref_out = vol_a_mref, vol_b_mref
            else:
                mref_in, mref_out = vol_b_mref, vol_a_mref
            surf_material_idx_in = self.material_lookup[mref_in]
            surf_material_idx_out = self.material_lookup[mref_out]
            surf_surface_idx = self.surface_lookup[surface_ref]
            inside_materials.extend([surf_material_idx_in] * len(face_tags_for_surf))
            outside_materials.extend([surf_material_idx_out] * len(face_tags_for_surf))
            surfaces.extend([surf_surface_idx] * len(face_tags_for_surf))
            solid_ids.extend([solid_id] * len(face_tags_for_surf))
            colors.extend([color] * len(face_tags_for_surf))

            nFaces += len(face_tags_for_surf)
        node_idx_per_face = np.reshape(node_idx_per_face, (-1, 3))
        detector = Detector(detector_material=chroma.geometry.Material(
            "water"))  # TODO: figure out what to do with this. This is only used for G4. Remove?
        logger.info(f"GMSH Model Exported. {nFaces} Triangles, {nVtxs} vertices")
        detector.mesh = Mesh(coords, node_idx_per_face, remove_duplicate_vertices=False, remove_null_triangles=False)
        detector.colors = np.asarray(colors)
        detector.solid_id = np.asarray(solid_ids)
        detector.unique_materials = self.materials_used
        detector.inner_material_index = np.asarray(inside_materials)
        detector.outer_material_index = np.asarray(outside_materials)
        detector.unique_surfaces = self.surfaces_used
        detector.surface_index = np.asarray(surfaces)
        detector.channel_index_to_channel_type = self.pmt_index_to_type
        detector.channel_index_to_position = self.pmt_index_to_position
        detector.solid_id_to_channel_index = np.arange(-1, self.nPMTs, dtype=int)
        detector.channel_index_to_solid_id = np.arange(self.nPMTs, dtype=int) + 1

        # add all elements outside the gmsh model
        concat_meshes = [detector.mesh]
        concat_colors = [detector.colors]
        concat_solid_id = [detector.solid_id]
        concat_inner_material_index = [detector.inner_material_index]
        concat_outer_material_index = [detector.outer_material_index]
        concat_surface_index = [detector.surface_index]
        for volume in self.placement_to_volume_map.values():
            if volume.mesh is not None:
                logger.info(f"Adding tessellated solid {volume.name} to the geometry")
                concat_meshes.append(volume.mesh)
                concat_colors.append(np.ones(len(volume.mesh.triangles)) * DEFAULT_SOLID_COLOR)
                concat_solid_id.append(np.zeros(len(volume.mesh.triangles), dtype=int))
                inner_material = self.material_lookup[volume.material_ref]
                outer_material = self.material_lookup[volume.parent_material_ref]
                skin_surface = self.surface_lookup[volume.skin_surface]
                concat_inner_material_index.append(np.ones(len(volume.mesh.triangles), dtype=int) * inner_material)
                concat_outer_material_index.append(np.ones(len(volume.mesh.triangles), dtype=int) * outer_material)
                concat_surface_index.append(np.ones(len(volume.mesh.triangles), dtype=int) * skin_surface)
                nFaces += len(volume.mesh.triangles)
                nVtxs += len(volume.mesh.vertices)
            elif volume.subdetector is not None:
                logger.info(f"Adding subdetector {volume.name} to the geometry")
                concat_meshes.append(volume.subdetector.mesh)
                concat_colors.append(volume.subdetector.colors)
                solid_id = 0 if volume.pmt_channel is None else volume.pmt_channel + 1
                concat_solid_id.append(np.ones(len(volume.subdetector.mesh.triangles), dtype=int)*solid_id)
                concat_inner_material_index.append(volume.subdetector.inner_material_index)
                concat_outer_material_index.append(volume.subdetector.outer_material_index)
                concat_surface_index.append(volume.subdetector.surface_index)
                nFaces += len(volume.subdetector.mesh.triangles)
                nVtxs += len(volume.subdetector.mesh.vertices)
        logger.info(f"Total after adding Tess Objects: {nFaces} triangles and {nVtxs} vertices")
        n_vertex_cumulative = np.cumsum([len(mesh.vertices) for mesh in concat_meshes])
        n_vertex_cumulative = np.concatenate([np.zeros(1), n_vertex_cumulative[:-1]])
        detector.mesh = Mesh(
            np.concatenate([mesh.vertices for mesh in concat_meshes]),
            np.concatenate([mesh.triangles + n_vertex_cumulative[i] for i, mesh in enumerate(concat_meshes)]),
            remove_duplicate_vertices=False, remove_null_triangles=False
        )
        detector.colors = np.concatenate(concat_colors)
        detector.solid_id = np.concatenate(concat_solid_id)
        detector.inner_material_index = np.concatenate(concat_inner_material_index)
        detector.outer_material_index = np.concatenate(concat_outer_material_index)
        detector.surface_index = np.concatenate(concat_surface_index)
        return detector
        # return node_idx_per_face, coords, inside_materials, outside_materials

    def assign_surface_properties(self):
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
            properties = {'volumes': [], 'surface': None, 'pmt_channel': None}
            for pname in placementNames:
                if pname == '/':  # outside of world volume
                    properties['volumes'].append(None)
                    continue
                else:
                    properties['volumes'].append(self.placement_to_volume_map[pname])
                    if self.placement_to_volume_map[pname].skin_surface is not None:
                        assert properties['surface'] is None, f"Surface is assigned twice for {pname}"
                        properties['surface'] = self.placement_to_volume_map[pname].skin_surface
                if self.placement_to_volume_map[pname].pmt_type is not None:
                    assert properties['pmt_channel'] is None, f"PMT channel is assigned twice for {pname}"
                    properties['pmt_channel'] = self.placement_to_volume_map[pname].pmt_channel
            physvol_names = [Path(pname).name for pname in placementNames]
            for border_surface in self.border_surfaces:
                if set(border_surface['placement_names']) == set(physvol_names):
                    properties['surface'] = border_surface['surface']
            surface_tags_to_materialrefs[surf_tag] = properties
        return surface_tags_to_materialrefs

    def visualize(self):
        gmsh.fltk.run()

    def fix_orphaned_border_surfaces(self):
        """
            RAT-PAC2 currently have a bug where one of the physical volumes that a border surface is assigned to does
            not exist. When this happens, the border was supposed to be assigned between the other physical volume and
            its mother.
        """
        all_physvols = set(Path(placement).name for placement in self.placement_to_volume_map.keys())
        border_surface_copy = deepcopy(self.border_surfaces)
        self.border_surfaces = []
        for border_surface in border_surface_copy:
            needs_fixing = False
            for i, physvol_name in enumerate(border_surface['placement_names']):
                if physvol_name not in all_physvols:
                    needs_fixing = True
                    logger.warning(f"Border surface {border_surface} has an orphaned physical volume {physvol_name}. "
                                   f"Fixing by replacing the orphan with the mother volume.")
                    other_physvol_name = border_surface['placement_names'][1 - i]
                    for placement in self.placement_to_volume_map.keys():
                        if Path(placement).name == other_physvol_name:
                            self.border_surfaces.append({
                                'surface': border_surface['surface'],
                                'placement_names': [Path(placement).parent.name, other_physvol_name]
                            })
                            logger.info(f"Fixed border surface {border_surface} by replacing "
                                        f"{physvol_name} with {Path(placement).parent.name}")
                    break
            if not needs_fixing:
                self.border_surfaces.append(border_surface)

    def add_pmt_info(self):
        pmt_arrays = self.ratdb_parser.get_matching_entries(
            content_match=lambda entry: entry['name'] == 'GEO' and entry['type'] == 'pmtarray'
        )
        pmt_volume_names = [table['index'] + '_body_log' for table in pmt_arrays]
        pmtinfo_tables = [self.ratdb_parser.get_entry(table['pos_table'], '') for table in pmt_arrays]
        pmt_array_positions = [np.array([table['x'], table['y'], table['z']]).T
                               for table in pmtinfo_tables]
        # pmt_array_positions = np.concatenate(pmt_array_positions)
        pmt_types = [table['type'] for table in pmtinfo_tables]
        # pmt_types = np.concatenate(pmt_types)
        self.nPMTs = 0
        self.pmt_index_to_type = []
        self.pmt_index_to_position = []
        for placement, volume in self.placement_to_volume_map.items():
            for pmt_array_idx, vol_name in enumerate(pmt_volume_names):
                if volume.name.startswith(vol_name):
                    pmt_idx_in_array = np.argwhere(
                        np.all(np.isclose(volume.absolute_pos, pmt_array_positions[pmt_array_idx]), axis=1))
                    assert pmt_idx_in_array.size == 1, \
                        (f"PMT {volume.name} in PMT Array {vol_name} can't be found or is not unique. "
                         f"Indices: {pmt_idx_in_array}")
                    pmt_idx_in_array = pmt_idx_in_array.item()
                    volume.pmt_type = pmt_types[pmt_array_idx][pmt_idx_in_array]
                    volume.pmt_channel = self.nPMTs
                    self.pmt_index_to_type.append(volume.pmt_type)
                    self.pmt_index_to_position.append(pmt_array_positions[pmt_array_idx][pmt_idx_in_array])
                    self.nPMTs += 1
                    logger.info(f"Assigned PMT Channel {volume.pmt_channel} to {placement}, Type {volume.pmt_type}")
                    break
        logger.info(f"Assigned {self.nPMTs} PMT Channels")
