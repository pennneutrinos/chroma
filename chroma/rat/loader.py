import xml.etree.ElementTree as et
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
from scipy import constants

# To convert length and angle units to cm and radians

units = gdml.units
TwoPiHbarC = constants.value('reduced Planck constant times c in MeV fm') * 1e-6 * 2 * np.pi  # MeV * nm

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
    '''
    This class supports loading a geometry from a GDML file by directly parsing 
    the XML. A subset of GDML is supported here, and exceptions will be raised
    if the GDML uses unsupported features.
    '''

    def __init__(self, gdml_file, refinement_order=0, ratdb_file=None, override_worldref=None):
        ''' 
        Read a geometry from the specified GDML file.
        '''
        self.nPMTs = None
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
            self.materials_used.append(self.create_material(material_xml))
            self.material_lookup[material_xml.get('name')] = len(self.materials_used) - 1
        solids = gdml_tree.find('solids')
        self.solid_xml_map = {solid.get('name'): solid for solid in solids}
        surfaces = solids.findall('opticalsurface')
        self.surfaces_used = [None]
        self.surface_lookup = {None: 0}
        for surface_idx, surface_xml in enumerate(surfaces, start=1):  # 0 is reserved for no surface
            self.surfaces_used.append(self.create_surface(surface_xml))
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
        self.triangle_typeid = gmsh.model.mesh.getElementType('Triangle', 1)

        # gmsh.option.setNumber('Geometry.Tolerance', 0.001)
        gmsh.model.add(self.gdml_file)

    def add_ratdb(self, ratdb_file):
        self.ratdb_parser = RatDBParser(ratdb_file)

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
        q.append(self.world)
        gmsh.clear()
        while len(q):
            volume: Volume = q.pop()
            pos = volume.absolute_pos
            rot = volume.absolute_rot
            parent_material_ref = volume.parent_material_ref
            logger.debug(f"Generating volume {volume.name}\tsolid ref: {volume.solid_ref}\tposition: {pos}")
            for child in volume.children:
                q.append(child)
            classification, kwargs = volume_classifier(volume.name, volume.material_ref, parent_material_ref)
            if classification == 'omit':
                logger.debug(f"Volume {volume.name} is omitted.")
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
        '''
        Processes solid into a chroma Mesh.
        tag_or_mesh is either a gmsh tag that relates to the root solid, or a chroma.Mesh object.
        If tag_or_mesh is a gmsh tag, apply downstream gmsh mesh generation routine and package the generated mesh into a
        chorma.Mesh object. If tag_or_mesh is already a mesh, do nothing. Simply return the mesh.
        Returns a chroma_mesh or None.
        '''
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
            mref_in, mref_out = prop['mrefs']
            surface_ref = prop['surface']
            pmt_channel = prop['pmt_channel']
            solid_id = 0 if pmt_channel is None else pmt_channel + 1
            color = DEFAULT_PMT_COLOR if pmt_channel is not None else DEFAULT_SOLID_COLOR
            logger.info(f"Processing surface {surf_tag} with materials {mref_in} and {mref_out}")
            face_tags_for_surf, node_tag_per_face_for_surf = mesh.getElementsByType(self.triangle_typeid, surf_tag)
            # face_tags_for_surf -= 1  # because tags are 1-indexed
            # node_tag_per_face_for_surf = np.reshape(node_tags_for_surf, (-1, 3))
            surf_material_idx_in = self.material_lookup[mref_in]
            surf_material_idx_out = self.material_lookup[mref_out]
            surf_surface_idx = self.surface_lookup[surface_ref]
            inside_materials.extend([surf_material_idx_in] * len(face_tags_for_surf))
            outside_materials.extend([surf_material_idx_out] * len(face_tags_for_surf))
            surfaces.extend([surf_surface_idx] * len(face_tags_for_surf))
            solid_ids.extend([solid_id] * len(face_tags_for_surf))
            colors.extend([color] * len(face_tags_for_surf))
            # fancy numpy magic for assigning node indices to faces
            node_idx_per_face_for_surf = np.vectorize(node_tag_to_index.get)(node_tag_per_face_for_surf)
            node_idx_per_face.extend(node_idx_per_face_for_surf)
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

        # add tessellated solids. Put all of them in lists fist and concat them at the end because numpy concat is slow
        concat_meshes = [detector.mesh]
        concat_colors = [detector.colors]
        concat_solid_id = [detector.solid_id]
        concat_inner_material_index = [detector.inner_material_index]
        concat_outer_material_index = [detector.outer_material_index]
        concat_surface_index = [detector.surface_index]
        for volume in self.placement_to_volume_map.values():
            logger.info(f"Adding tessellated solid {volume.name} to the geometry")
            if volume.mesh is not None:
                concat_meshes.append(volume.mesh)
                concat_colors.append(np.ones(len(volume.mesh.triangles)) * DEFAULT_SOLID_COLOR)
                concat_solid_id.append(np.ones(len(volume.mesh.triangles), dtype=int))
                inner_material = self.material_lookup[volume.material_ref]
                outer_material = self.material_lookup[volume.parent_material_ref]
                skin_surface = self.surface_lookup[volume.skin_surface]
                concat_inner_material_index.append(np.ones(len(volume.mesh.triangles), dtype=int) * inner_material)
                concat_outer_material_index.append(np.ones(len(volume.mesh.triangles), dtype=int) * outer_material)
                concat_surface_index.append(np.ones(len(volume.mesh.triangles), dtype=int) * skin_surface)
                nFaces += len(volume.mesh.triangles)
                nVtxs += len(volume.mesh.vertices)
                # logger.info(f"Added {len(volume.mesh.triangles)} triangles and {len(volume.mesh.vertices)} vertices")
        logger.info(f"Total after adding Tess Objects: {nFaces} triangles and {nVtxs} vertices")
        n_vertex_cumulative = np.cumsum([len(mesh.vertices) for mesh in concat_meshes])
        n_vertex_cumulative = np.concatenate([[0], n_vertex_cumulative[:-1]])
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
            properties = {'mrefs': [], 'surface': None, 'pmt_channel': None}
            for pname in placementNames:
                if pname == '/':
                    properties['mrefs'].append(self.world.material_ref)
                    continue
                else:
                    properties['mrefs'].append(self.placement_to_volume_map[pname].material_ref)
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

    def create_material(self, material_xml) -> chroma.geometry.Material:
        name = material_xml.get('name')
        material = chroma.geometry.Material(name)
        name_nouid = name.split('0x')[0]
        density = gdml.get_val(material_xml.find('D'), attr='value')
        density *= units.get(material_xml.find('D').get('unit'), 1.0)
        material.density = density
        material.set('refractive_index', 1.0)
        material.set('absorption_length', 1e6)
        material.set('scattering_length', 1e6)
        for comp in material_xml.findall('fraction'):
            element = comp.get('ref').split('0x')[0]
            fraction = gdml.get_val(comp, attr='n')
            material.composition[element] = fraction

        # Material-wise properties
        num_comp = 0
        optical_props = material_xml.findall('property')
        for optical_prop in optical_props:
            data_ref = optical_prop.get('ref')
            data = gdml.get_matrix(self.matrix_map[data_ref])
            property_name = optical_prop.get('name')
            if property_name == 'RINDEX':
                material.refractive_index = _convert_to_wavelength(data)
            elif property_name == 'ABSLENGTH':
                material.absorption_length = _convert_to_wavelength(data)
            elif property_name == 'RSLENGTH':
                material.scattering_length = _convert_to_wavelength(data)
            elif property_name == "SCINTILLATION":
                material.scintillation_spectrum = _convert_to_wavelength(data)
            elif property_name == "SCINT_RISE_TIME":
                material.scintillation_rise_time = data.item()
            elif property_name == "LIGHT_YIELD":
                material.scintillation_light_yield = data.item()
            elif property_name.startswith('SCINTWAVEFORM'):
                if material.scintillation_waveform is None:
                    material.scintillation_waveform = {}
                # extract the property name from the SCINTWAVEFORM prefix
                material.scintillation_waveform[property_name[len('SCINTWAVEFORM'):]] = data
            elif property_name.startswith('SCINTMOD'):
                if material.scintillation_mod is None:
                    material.scintillation_mod = {}
                # extract the property name from the SCINTMOD prefix
                material.scintillation_mod[property_name[len('SCINTMOD'):]] = data
            elif property_name == 'NUM_COMP':
                num_comp = int(data.item())

        # Component wise properties.
        reemission_spectrum = None
        # RAT does not support component-wise reemission spectra. All components share the
        # same spectrum.
        if num_comp > 0:
            for prop_name in ['SCINTILLATION_WLS', 'SCINTILLATION']:
                reemission_spectrum = self._find_property(prop_name, optical_props)
                if reemission_spectrum is not None:
                    reemission_spectrum = _convert_to_wavelength(reemission_spectrum)
                    reemission_spectrum = _pdf_to_cdf(reemission_spectrum)
                    break
            assert reemission_spectrum is not None, f"No reemission spectrum found for material {name}"
        for i_comp in range(num_comp):
            reemission_prob = self._find_property('REEMISSION_PROB' + str(i_comp), optical_props)
            if reemission_prob is not None:
                reemission_prob = _convert_to_wavelength(reemission_prob)
                material.comp_reemission_prob.append(reemission_prob)
            else:
                material.comp_reemission_prob.append(np.column_stack((
                    chroma.geometry.standard_wavelengths,
                    np.zeros(chroma.geometry.standard_wavelengths.size))))
            material.comp_reemission_wvl_cdf.append(reemission_spectrum)

            reemission_waveform = self._find_property('REEMITWAVEFORM' + str(i_comp), optical_props)
            if reemission_waveform is not None:
                if reemission_waveform.flatten()[0] < 0:
                    reemission_waveform = _exp_decay_cdf(reemission_waveform) #FIXME: use scintillation rise time?
                else:
                    reemission_waveform = _pdf_to_cdf(reemission_waveform)
            else:
                reemission_waveform = np.column_stack(([0, 1], [0, 0]))  # dummy waveform
            material.comp_reemission_time_cdf.append(reemission_waveform)

            absorption_length = self._find_property('ABSLENGTH' + str(i_comp), optical_props)
            assert absorption_length is not None, "No component-wise absorption length found for material"
            material.comp_absorption_length.append(_convert_to_wavelength(absorption_length))
        return material

    def create_surface(self, surface_xml) -> chroma.geometry.Surface:
        name = surface_xml.get('name')
        surface = chroma.geometry.Surface(name)
        model = gdml.get_val(surface_xml, attr='model')
        surface_type = gdml.get_val(surface_xml, attr='type')
        finish = gdml.get_val(surface_xml, attr='finish')
        value = gdml.get_val(surface_xml, attr='value')
        assert model == 0 or model == 1 or model == 4, "Only glisur, unified, and dichroic models are supported"
        assert surface_type == 0 or surface_type == 4, "Only dielectric_metal and dichroic surfaces are supported"
        assert finish == 0 or finish == 1 or finish == 3, \
            "Only polished, ground, and polishedfrontpainted are supported"
        specular_component = value if model == 0 else 1 - value  # this is a hack, because chroma does not support the
        # same time of diffusive reflection
        if finish == 1:
            surface.transmissive = False
        abslength = None
        for optical_prop in surface_xml.findall('property'):
            data_ref = optical_prop.get('ref')
            property_name = optical_prop.get('name')
            data = gdml.get_matrix(self.matrix_map[data_ref])
            if property_name == 'REFLECTIVITY':
                reflectivity = _convert_to_wavelength(data)
                reflectivity_specular = reflectivity
                reflectivity_specular[:, 1] *= specular_component
                reflectivity_diffuse = reflectivity
                reflectivity_diffuse[:, 1] *= (1 - specular_component)
                surface.reflect_specular = _convert_to_wavelength(reflectivity_specular)
                surface.reflect_diffuse = _convert_to_wavelength(reflectivity_diffuse)
            if property_name == 'THICKNESS':
                thicknesses = data[:, 1]
                if not np.allclose(thicknesses, thicknesses[0]):
                    logger.warning(f"Surface {name} has non-uniform thicknesses. Average will be taken")
                surface.thickness = np.mean(thicknesses)
            if property_name == 'RINDEX':
                surface.eta = _convert_to_wavelength(data)
            if property_name == 'KINDEX':
                surface.k = _convert_to_wavelength(data)
                surface.model = 1  # if k index is specified, we have a complex surface model
            if property_name == 'EFFICIENCY':
                surface.detect = _convert_to_wavelength(data)
            if property_name == "ABSLENGTH":
                abslength = _convert_to_wavelength(data)
        if abslength is not None:
            surface.absorb = abslength
            surface.absorb[:, 1] = 1 - np.exp(-surface.thickness / surface.absorb[:, 1])
        if model == 4 and surface_type == 4:
            assert surface_xml.find('dichroic_data') is not None, "Dichroic surfaces must have dichroic_data"
            surface.model = 3  # CUDA dichroic model
            dichroic_data = surface_xml.find('dichroic_data')
            x_length = gdml.get_val(dichroic_data, attr='x_length')
            y_length = gdml.get_val(dichroic_data, attr='y_length')
            x_val_elem = dichroic_data.find('x')
            wvls = gdml.get_vector(x_val_elem)
            y_val_elem = dichroic_data.find('y')
            angles = gdml.get_vector(y_val_elem)
            data_elem = dichroic_data.find('data')
            transmission_data = gdml.get_vector(data_elem).reshape(x_length, y_length)
            reflection_data = 1 - transmission_data
            angles = np.deg2rad(angles)
            transmits = [np.asarray([wvls, transmission_data[:, i]]).T for i in range(y_length)]
            reflects = [np.asarray([wvls, reflection_data[:, i]]).T for i in range(y_length)]
            surface.dichroic_props = chroma.geometry.DichroicProps(angles, transmits, reflects)
        return surface

    def fix_orphaned_border_surfaces(self):
        """
            RAT-PAC2 currently have a bug where one of the physical volumes that a border surface is assigned to does
            not exist. When this happens, the border was supposed to be assigned between the other physical volume and
            its mother.
        """
        all_physvols = set(Path(placement).name for placement in self.placement_to_volume_map.keys())
        for border_surface in self.border_surfaces:
            for i, physvol_name in enumerate(border_surface['placement_names']):
                if physvol_name not in all_physvols:
                    logger.warning(f"Border surface {border_surface} has an orphaned physical volume {physvol_name}. "
                                   f"Attempting to fix")
                    other_physvol_name = border_surface['placement_names'][1 - i]
                    for placement in self.placement_to_volume_map.keys():
                        if other_physvol_name == Path(placement).name:
                            border_surface['placement_names'][i] = Path(placement).parent.name
                            logger.warning(
                                f"Fixed border surface {border_surface} by changing {physvol_name} to {Path(placement).parent.name}")
                            break
                    break

    def add_pmt_info(self):
        pmtinfo_tables = self.ratdb_parser.get_matching_entries(
            table_name_match=lambda name: name.startswith('PMTINFO'),
        )
        pmt_array_names = [table['name'] for table in pmtinfo_tables]
        pmt_volume_names = ['pmts_' + name[len('PMTINFO_'):].lower() + '_body_log'
                            for name in pmt_array_names]
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
                    self.pmt_index_to_position.append(volume.absolute_pos)
                    self.nPMTs += 1
                    logger.info(f"Assigned PMT Channel {volume.pmt_channel} to {placement}, Type {volume.pmt_type}")
                    break
        logger.info(f"Assigned {self.nPMTs} PMT Channels")

    def _find_property(self, name, properties):
        for prop in properties:
            if prop.get('name') == name:
                data_ref = prop.get('ref')
                data = gdml.get_matrix(self.matrix_map[data_ref])
                return data
        return None

def _convert_to_wavelength(arr):
    arr[:, 0] = TwoPiHbarC / arr[:, 0]
    return arr[::-1]


def _pdf_to_cdf(arr):
    x, y = arr.T
    yc = np.cumsum((y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    yc = np.concatenate([[0], yc])
    if yc[-1] != 0:
        yc /= yc[-1]
    return np.column_stack([x, yc])


def _exp_decay_cdf(arr, t_rise=0):
    decays = np.exp(-arr[:, 0])
    weights = np.exp(arr[:, 1])
    max_time = 3.0 * np.max(decays)
    min_time = np.min(decays)
    bin_width = min_time / 100
    times = np.arange(0, max_time + bin_width / 2, bin_width)
    if t_rise == 0:
        cdf = np.sum([a * (t * (1.0 - np.exp(-times / t))) / (t) for t, a in zip(decays, weights)], axis=0)
    else:
        cdf = np.sum(
            [a * (t * (1.0 - np.exp(-times / t)) + t_rise * (np.exp(-times / t_rise) - 1)) / (t - t_rise) for t, a in
             zip(decays, weights)], axis=0)
    return np.column_stack([times, cdf])