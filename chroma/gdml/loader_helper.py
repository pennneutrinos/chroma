import xml.etree.ElementTree as et
import numpy as np
from collections import deque

from chroma.gdml import gen_mesh
from chroma.log import logger

## Utility functions to connect the loader to gen_mesh
_units = { 'cm':1, 'mm':0.1, 'm':100, 'deg':np.pi/180, 'rad':1 }

def get_vals(elem, value_attr=['x', 'y', 'z'], default_vals=None, unit_attr='unit'):
    '''
    Calls get_val for a list of attributes (value_attr). The result 
    values are scaled from the unit specified in the unit_attrib attribute.
    '''
    if default_vals is None:
        default_vals = [None] * len(value_attr) # no default value by default
    assert len(value_attr) == len(default_vals), 'length of attributs does not equal to number of default values'
    scale = _units[elem.get(unit_attr)] if unit_attr is not None else 1.0
    return [get_val(elem, attr, default) * scale for (attr, default) in zip(value_attr, default_vals)]

def get_val(elem, attr, default=None):
    '''
    Calls eval on the value of the attribute attr if it exists and return 
    it. Otherwise return the default specified. If there is no default 
    specifed, raise an exception.
    '''
    txt = elem.get(attr, default=None)
    assert txt is not None or default is not None, 'Missing attribute: '+attr
    return eval(txt, {}, {}) if txt is not None else default

def get_zplanes(elem, tag='zplane', unit_attr='lunit'):
    '''Return the children elements with the `tag` as an attribute dictionary '''
    scale = _units[elem.get(unit_attr)] if unit_attr is not None else 1.0
    planes = elem.findall(tag)
    result = [plane.attrib for plane in planes]
    for r in result:
        r.update((x, float(y)*scale) for x, y in r.items())
        if 'rmin' not in r:
            r['rmin'] = 0
    return result


def box(elem):
    x, y, z = get_vals(elem, ['x', 'y', 'z'], unit_attr='lunit')
    return gen_mesh.gdml_box(x, y, z)


def eltube(elem):
    dx, dy, dz = get_vals(elem, ['dx', 'dy', 'dz'], unit_attr='lunit')
    return gen_mesh.gdml_eltube(dx, dy, dz)

def orb(elem):
    r, = get_vals(elem, ['r'], unit_attr='lunit')
    return gen_mesh.gdml_orb(r)

def polycone(elem):
    startphi, deltaphi = get_vals(elem, ['startphi', 'deltaphi'], unit_attr='aunit')
    zplanes = get_zplanes(elem)
    return gen_mesh.gdml_polycone(startphi, deltaphi, zplanes)

def polyhedra(elem):
    startphi, deltaphi = get_vals(elem, ['startphi', 'deltaphi'], unit_attr='aunit')
    numsides = int(elem.get('numsides'))
    zplanes = get_zplanes(elem)
    return gen_mesh.gdml_polyhedra(startphi, deltaphi, numsides, zplanes)

def sphere(elem):
    rmin, rmax = get_vals(elem, ['rmin', 'rmax'], default_vals=[0.0, None], unit_attr='lunit')
    startphi, deltaphi, starttheta, deltatheta = get_vals(
        elem, 
        ["startphi", "deltaphi", "starttheta", "deltatheta"],
        default_vals=[0.0, None , 0.0, None],
        unit_attr='aunit'
        )
    return gen_mesh.gdml_sphere(rmin, rmax, startphi, deltaphi, starttheta, deltatheta)

def torus(elem):
    rmin, rmax, rtor = get_vals(elem, ['rmin', 'rmax', 'rtor'], unit_attr='lunit')
    startphi, deltaphi = get_vals(elem, ['startphi', 'deltaphi'], unit_attr='aunit')
    return gen_mesh.gdml_torus(rmin, rmax, rtor, startphi, deltaphi)

def tube(elem):
    rmin, rmax, z = get_vals(elem, ['rmin', 'rmax', 'z'], default_vals=[0.0, None, 0.0],unit_attr='lunit')
    startphi, deltaphi = get_vals(elem, ['startphi', 'deltaphi'], default_vals=[0.0, None], unit_attr='aunit')
    return gen_mesh.gdml_tube(rmin, rmax, z, startphi, deltaphi)

def notImplemented(elem):
    raise NotImplementedError(f'{elem.tag} is not implemented')

def ignore(elem):
    return


def balanced_consecutive_subtraction(solids:deque):
    '''
    Take a deque of solids, perform balanced subtraction that is equivalent to solids[0] - solids[1] - solids[2]...
    '''
    # print("Current number of solids: ", len(solids))
    logger.debug('new layer')
    assert len(solids)!=0
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

def subtraction_via_balanced_union(solids:deque):
    lhs = solids.popleft()
    logger.debug("Performing unions...")
    rhs = balanced_consecutive_union(solids)
    logger.debug("Performing subtraction...")
    result = gen_mesh.gdml_boolean(lhs, rhs, 'subtraction')
    logger.debug("DONE!")
    return result

def balanced_consecutive_union(solids:deque):
    assert len(solids)!=0
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
