import itertools
import xml.etree.ElementTree as et
from typing import Dict, Optional, List

import numpy as np
from collections import deque

from chroma.rat import gen_mesh
from chroma.geometry import Surface, Material, Mesh, DichroicProps, standard_wavelengths
from chroma.log import logger
from copy import deepcopy
from scipy import constants

units = {'cm': 10, 'mm': 1, 'm': 1000, 'deg': np.pi / 180, 'rad': 1, 'g/cm3': 1}
TwoPiHbarC = constants.value('reduced Planck constant times c in MeV fm') * 1e-6 * 2 * np.pi  # MeV * nm


def get_vals(elem, value_attr=None, default_vals=None, unit_attr='unit'):
    """
    Calls get_val for a list of attributes (value_attr). The result
    values are scaled from the unit specified in the unit_attrib attribute.
    """
    if value_attr is None:
        value_attr = ['x', 'y', 'z']
    if default_vals is None:
        default_vals = [None] * len(value_attr)  # no default value by default
    assert len(value_attr) == len(default_vals), 'length of attributes does not equal to number of default values'
    scale = units[elem.get(unit_attr)] if unit_attr is not None else 1.0
    return [get_val(elem, attr, default) * scale for (attr, default) in zip(value_attr, default_vals)]


def get_val(elem, attr, default=None):
    """
    Calls eval on the value of the attribute attr if it exists and return
    it. Otherwise, return the default specified. If there is no default
    specified, raise an exception.
    """
    txt = elem.get(attr, default=None)
    assert txt is not None or default is not None, 'Missing attribute: ' + attr
    return eval(txt, {}, {}) if txt is not None else default


def get_matrix(elem):
    '''
    Return the correctly shaped matrix from the text of the element, as a numpy array.
    '''
    assert elem.tag == 'matrix', 'Element is not a matrix'
    coldim = int(elem.get('coldim'))
    values = get_vector(elem)
    return values.reshape(-1, coldim)


def get_vector(elem, attr='values', dtype=float):
    return np.asarray(elem.get(attr).split(), dtype=dtype)


def get_daughters_as_dict(elem, tag='zplane', unit_attr='lunit', add_rmin=True):
    """Return the children elements with the `tag` as an attribute dictionary """
    scale = units[elem.get(unit_attr)] if unit_attr is not None else 1.0
    planes = elem.findall(tag)
    result = deepcopy([plane.attrib for plane in planes])
    for r in result:
        r.update((x, float(y) * scale) for x, y in r.items())
        if add_rmin and 'rmin' not in r:
            r['rmin'] = 0
    return result


def box(elem):
    x, y, z = get_vals(elem, ['x', 'y', 'z'], unit_attr='lunit')
    return gen_mesh.gdml_box(x, y, z)


def ellipsoid(elem):
    ax, by, cz = get_vals(elem, ['ax', 'by', 'cz'], default_vals=[1.0, 1.0, 1.0], unit_attr='lunit')
    zcut1, zcut2 = get_vals(elem, ['zcut1', 'zcut2'], default_vals=[0.0, 0.0], unit_attr='lunit')
    return gen_mesh.gdml_ellipsoid(ax, by, cz, zcut1, zcut2)


def eltube(elem):
    dx, dy, dz = get_vals(elem, ['dx', 'dy', 'dz'], unit_attr='lunit')
    return gen_mesh.gdml_eltube(dx, dy, dz)


def orb(elem):
    r, = get_vals(elem, ['r'], unit_attr='lunit')
    return gen_mesh.gdml_orb(r)


def polycone(elem):
    startphi, deltaphi = get_vals(elem, ['startphi', 'deltaphi'], unit_attr='aunit')
    zplanes = get_daughters_as_dict(elem)
    return gen_mesh.gdml_polycone(startphi, deltaphi, zplanes)


def polyhedra(elem):
    startphi, deltaphi = get_vals(elem, ['startphi', 'deltaphi'], unit_attr='aunit')
    numsides = int(elem.get('numsides'))
    zplanes = get_daughters_as_dict(elem)
    return gen_mesh.gdml_polyhedra(startphi, deltaphi, numsides, zplanes)


def sphere(elem):
    rmin, rmax = get_vals(elem, ['rmin', 'rmax'], default_vals=[0.0, None], unit_attr='lunit')
    startphi, deltaphi, starttheta, deltatheta = get_vals(
        elem,
        ["startphi", "deltaphi", "starttheta", "deltatheta"],
        default_vals=[0.0, None, 0.0, None],
        unit_attr='aunit'
    )
    return gen_mesh.gdml_sphere(rmin, rmax, startphi, deltaphi, starttheta, deltatheta)


def tessellated(elem, all_vertex_positions):
    triangle_elements = elem.findall('triangular')
    triangle_vertex_tags = [[triangle.get('vertex1'), triangle.get('vertex2'), triangle.get('vertex3')]
                            for triangle in triangle_elements]
    vertex_tags_unique = list(set(itertools.chain(*triangle_vertex_tags)))
    vertex_positions = [all_vertex_positions[tag] for tag in vertex_tags_unique]
    triangles = [[vertex_tags_unique.index(tag) for tag in triangle] for triangle in triangle_vertex_tags]
    return Mesh(vertex_positions, triangles)


def torus(elem):
    rmin, rmax, rtor = get_vals(elem, ['rmin', 'rmax', 'rtor'], unit_attr='lunit')
    startphi, deltaphi = get_vals(elem, ['startphi', 'deltaphi'], unit_attr='aunit')
    return gen_mesh.gdml_torus(rmin, rmax, rtor, startphi, deltaphi)


def tube(elem):
    rmin, rmax, z = get_vals(elem, ['rmin', 'rmax', 'z'], default_vals=[0.0, None, 0.0], unit_attr='lunit')
    # if z < 1e-2:
    #     logger.warn(f"Very thin tube is found, with thickness of {z} mm. Skipping!")
    #     return
    startphi, deltaphi = get_vals(elem, ['startphi', 'deltaphi'], default_vals=[0.0, None], unit_attr='aunit')
    return gen_mesh.gdml_tube(rmin, rmax, z, startphi, deltaphi)


def torusstack(elem):
    edges = get_daughters_as_dict(elem, tag='edge', unit_attr='lunit', add_rmin=False)
    origins = get_daughters_as_dict(elem, tag='origin', unit_attr='lunit', add_rmin=False)
    rho_edges = [entry['rho'] for entry in edges]
    z_edges = [entry['z'] for entry in edges]
    z_origins = [entry['z'] for entry in origins]
    rho_origins = [entry['rho'] for entry in origins]
    outer_solid = gen_mesh.gdml_torusStack(rho_edges, z_edges, rho_origins, z_origins)
    inner_elem = elem.find('inner')
    if inner_elem is None:
        return outer_solid
    else:
        inner_solid = torusstack(inner_elem.find('torusstack'))
        return gen_mesh.gdml_boolean(outer_solid, inner_solid, 'subtraction')


def notImplemented(elem):
    raise NotImplementedError(f'{elem.tag} is not implemented')


def ignore(elem):
    return


def balanced_consecutive_subtraction(solids: deque):
    '''
    Take a deque of solids, perform balanced subtraction that is equivalent to solids[0] - solids[1] - solids[2]...
    '''
    # print("Current number of solids: ", len(solids))
    logger.debug('new layer')
    assert len(solids) != 0
    if len(solids) == 1:
        return solids[0]
    # subtraction for the first two
    next_level = deque()
    a = solids.popleft()
    b = solids.popleft()
    logger.debug("Subtracting")
    next_level.append(gen_mesh.gdml_boolean(a, b, 'subtraction'))
    logger.debug("Subtraction Done")
    while solids:
        if len(solids) == 1:
            next_level.append(solids.pop())
        else:
            x = solids.popleft()
            y = solids.popleft()
            next_level.append(gen_mesh.gdml_boolean(x, y, 'union'))
        logger.debug("Union Done")
    return balanced_consecutive_subtraction(next_level)


def subtraction_via_balanced_union(solids: deque):
    lhs = solids.popleft()
    logger.debug("Performing unions...")
    rhs = balanced_consecutive_union(solids)
    logger.debug("Performing subtraction...")
    result = gen_mesh.gdml_boolean(lhs, rhs, 'subtraction')
    logger.debug("DONE!")
    return result


def balanced_consecutive_union(solids: deque):
    assert len(solids) != 0
    if len(solids) == 1:
        return solids[0]
    next_level = deque()
    while solids:
        if len(solids) == 1:
            next_level.append(solids.pop())
        else:
            x = solids.popleft()
            y = solids.popleft()
            next_level.append(gen_mesh.gdml_boolean(x, y, 'union'))
    return balanced_consecutive_union(next_level)


def create_surface(matrix_map: Dict[Optional[str], et.Element], surface_xml: et.Element) -> Surface:
    name = surface_xml.get('name')
    surface = Surface(name)
    model = get_val(surface_xml, attr='model')
    surface_type = get_val(surface_xml, attr='type')
    finish = get_val(surface_xml, attr='finish')
    value = get_val(surface_xml, attr='value')
    assert model == 0 or model == 1 or model == 4, "Only glisur, unified, and dichroic models are supported"
    assert surface_type == 0 or surface_type == 4, "Only dielectric_metal and dichroic surfaces are supported"
    assert finish == 0 or finish == 1 or finish == 3, \
        "Only polished, ground, and polishedfrontpainted are supported"
    specular_component = value if model == 0 else 1 - value  # this is a hack, because chroma does not support the
    # same time of diffusive reflection
    if finish == 1:
        surface.transmissive = 0
    else:
        surface.transmissive = 1
    abslength = None
    for optical_prop in surface_xml.findall('property'):
        data_ref = optical_prop.get('ref')
        property_name = optical_prop.get('name')
        data = get_matrix(matrix_map[data_ref])
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
        x_length = get_val(dichroic_data, attr='x_length')
        y_length = get_val(dichroic_data, attr='y_length')
        x_val_elem = dichroic_data.find('x')
        wvls = get_vector(x_val_elem)
        y_val_elem = dichroic_data.find('y')
        angles = get_vector(y_val_elem)
        data_elem = dichroic_data.find('data')
        transmission_data = get_vector(data_elem).reshape(x_length, y_length)/100
        reflection_data = 1 - transmission_data
        angles = np.deg2rad(angles)
        transmits = [np.asarray([wvls, transmission_data[:, i]]).T for i in range(y_length)]
        reflects = [np.asarray([wvls, reflection_data[:, i]]).T for i in range(y_length)]
        surface.dichroic_props = DichroicProps(angles, reflect=reflects, transmit=transmits)
    return surface


def create_material(matrix_map: Dict[Optional[str], et.Element], material_xml: et.Element) -> Material:
    name = material_xml.get('name')
    material = Material(name)
    name_nouid = name.split('0x')[0]
    density = get_val(material_xml.find('D'), attr='value')
    density *= units.get(material_xml.find('D').get('unit'), 1.0)
    material.density = density
    material.set('refractive_index', 1.0)
    material.set('absorption_length', 1e6)
    material.set('scattering_length', 1e6)
    for comp in material_xml.findall('fraction'):
        element = comp.get('ref').split('0x')[0]
        fraction = get_val(comp, attr='n')
        material.composition[element] = fraction

    # Material-wise properties
    num_comp = 0
    optical_props = material_xml.findall('property')
    for optical_prop in optical_props:
        data_ref = optical_prop.get('ref')
        data = get_matrix(matrix_map[data_ref])
        property_name = optical_prop.get('name')
        if property_name == 'RINDEX':
            material.refractive_index = _convert_to_wavelength(data)
        elif property_name == 'ABSLENGTH':
            material.absorption_length = _convert_to_wavelength(data)
        elif property_name == 'RSLENGTH':
            material.scattering_length = _convert_to_wavelength(data)
        elif property_name == "SCINTILLATION":
            material.scintillation_spectrum = _convert_to_wavelength(data, dy_dwavelength=True)
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
            reemission_spectrum = _find_property(matrix_map, prop_name, optical_props)
            if reemission_spectrum is not None:
                reemission_spectrum = _convert_to_wavelength(reemission_spectrum, dy_dwavelength=True)
                reemission_spectrum = _pdf_to_cdf(reemission_spectrum)
                break
        assert reemission_spectrum is not None, f"No reemission spectrum found for material {name}"
    for i_comp in range(num_comp):
        reemission_prob = _find_property(matrix_map, 'REEMISSION_PROB' + str(i_comp), optical_props)
        if reemission_prob is not None:
            reemission_prob = _convert_to_wavelength(reemission_prob)
            material.comp_reemission_prob.append(reemission_prob)
        else:
            material.comp_reemission_prob.append(np.column_stack((
                standard_wavelengths,
                np.zeros(standard_wavelengths.size))))
        material.comp_reemission_wvl_cdf.append(reemission_spectrum)

        reemission_waveform = _find_property(matrix_map, 'REEMITWAVEFORM' + str(i_comp), optical_props)
        if reemission_waveform is not None:
            if reemission_waveform.flatten()[0] < 0:
                reemission_waveform = _exp_decay_cdf(reemission_waveform) # Reemission waveform have no rise time
            else:
                reemission_waveform = _pdf_to_cdf(reemission_waveform)
        else:
            reemission_waveform = np.column_stack(([0, 1], [0, 0]))  # dummy waveform
        material.comp_reemission_time_cdf.append(reemission_waveform)

        absorption_length = _find_property(matrix_map, 'ABSLENGTH' + str(i_comp), optical_props)
        assert absorption_length is not None, "No component-wise absorption length found for material"
        material.comp_absorption_length.append(_convert_to_wavelength(absorption_length))
    return material


def _convert_to_wavelength(arr, dy_dwavelength=False):
    arr[:, 0] = TwoPiHbarC / arr[:, 0]
    if dy_dwavelength:
        arr[:, 1] *= TwoPiHbarC / (arr[:, 0]**2)
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


def _find_property(matrix_map: Dict[Optional[str], et.Element],
                   prop_name: str, properties: List[et.Element]) -> Optional[np.ndarray]:
    for prop in properties:
        if prop.get('name') == prop_name:
            data_ref = prop.get('ref')
            data = get_matrix(matrix_map[data_ref])
            return data
    return None
