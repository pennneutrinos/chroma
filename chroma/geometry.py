import sys
import os
from itertools import chain
from hashlib import md5
import pickle as pickle
import gzip
import numpy as np
import time

from chroma.itertoolset import *
from chroma.tools import timeit, profile_if_possible, filled_array, \
    memoize_method_with_dictionary_arg
from chroma.log import logger

# all material/surface properties are interpolated at these
# wavelengths when they are sent to the gpu
standard_wavelengths = np.arange(60, 1000, 5).astype(np.float32)

class Mesh(object):
    "Triangle mesh object."
    def __init__(self, vertices, triangles, remove_duplicate_vertices=False, round=True, remove_null_triangles=True):
        vertices = np.asarray(vertices, dtype=np.float32)
        triangles = np.asarray(triangles, dtype=np.int32)

        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValueError('shape mismatch')

        if len(triangles.shape) != 2 or triangles.shape[1] != 3:
            raise ValueError('shape mismatch')

        if (triangles < 0).any():
            raise ValueError('indices in `triangles` must be positive.')

        if (triangles >= len(vertices)).any():
            raise ValueError('indices in `triangles` must be less than the '
                             'length of the vertex array.')

        self.vertices = vertices
        self.triangles = triangles
        if len(self.vertices) == 0:
            logger.warning("Generated mesh has no vertices.")
        if len(self.triangles) == 0:
            logger.warning("Generated mesh has no triangles.")
        if round:
            self.vertices = self.vertices.round(decimals=12)
        if remove_duplicate_vertices:
            self.remove_duplicate_vertices()
        if remove_null_triangles:
            self.remove_null_triangles()

    def get_triangle_centers(self):
        "Returns the x,y,z coordinate of the center of each triangle."
        return np.mean(self.assemble(),axis=1)

    def get_bounds(self):
        "Return the lower and upper bounds for the mesh as a tuple."
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def remove_duplicate_vertices(self):
        "Remove any duplicate vertices in the mesh."
        # view the vertices as a structured array in order to identify unique
        # rows, i.e. unique vertices
        unique_vertices, inverse = np.unique(self.vertices.view([('', self.vertices.dtype)]*self.vertices.shape[1]), return_inverse=True)
        # turn the structured vertex array back into a normal array
        self.vertices = unique_vertices.view(self.vertices.dtype).reshape((unique_vertices.shape[0], self.vertices.shape[1]))
        # remap the triangles to the unique vertices
        self.triangles = inverse[self.triangles]

    def remove_null_triangles(self):
        '''Remove any zero-area triangles from a mesh.
 
        Returns the mask of retained triangles, which may be applied to the
        material, surface, etc., arrays of an associated ``Solid``.
        '''
        if len(self.triangles) == 0:
            return
        mask = np.array([(len(set(x)) == 3) for x in self.triangles])
        self.triangles = self.triangles[mask]
        return mask

    def assemble(self, key=slice(None), group=True):
        """
        Return an assembled triangle mesh; i.e. return the vertex positions
        of every triangle. If `group` is True, the array returned will have
        an extra axis denoting the triangle number; i.e. if the mesh contains
        N triangles, the returned array will have the shape (N,3,3). If `group`
        is False, return just the vertex positions without any grouping; in
        this case the grouping is implied (i.e. elements [0:3] are the first
        triangle, [3:6] the second, and so on.

        The `key` argument is a slice object if you just want to assemble
        a certain range of the triangles, i.e. to get the assembled mesh for
        triangles 3 through 6, key = slice(3,7).
        """
        if group:
            vertex_indices = self.triangles[key]
        else:
            vertex_indices = self.triangles[key].flatten()

        return self.vertices[vertex_indices]

    def __add__(self, other):
        return Mesh(np.concatenate((self.vertices, other.vertices)), np.concatenate((self.triangles, other.triangles + len(self.vertices))))

    def md5(self):
        '''Return the MD5 hash of the vertices and triangles in this mesh as a 
        hexidecimal string.'''
        checksum = md5(self.vertices)
        checksum.update(self.triangles)
        return checksum.hexdigest()

def silly_unique(arr):
    return np.asarray(list(set(arr)))

class Solid(object):
    """Solid object attaches materials, surfaces, and colors to each triangle
    in a Mesh object."""
    def __init__(self, mesh,
                 inner_material=None, outer_material=None,
                 surface=None, color=0x33ffffff,
                 material1=None, material2=None  # deprecated
                 ):
        if material1 is not None or material2 is not None:
            logger.warn('material1 and material2 are deprecated.  Use inner_material and outer_material instead.')
            inner_material = material1
            outer_material = material2
        if inner_material is None:
            inner_material = []
        self.mesh = mesh
        if np.iterable(inner_material):
            if len(inner_material) != len(mesh.triangles):
                raise ValueError('shape mismatch')
            self.inner_material = np.array(inner_material, dtype=object)
        else:
            self.inner_material = np.tile(inner_material, len(self.mesh.triangles))

        if np.iterable(outer_material):
            if len(outer_material) != len(mesh.triangles):
                raise ValueError('shape mismatch')
            self.outer_material = np.array(outer_material, dtype=object)
        else:
            self.outer_material = np.tile(outer_material, len(self.mesh.triangles))

        if np.iterable(surface):
            if len(surface) != len(mesh.triangles):
                raise ValueError('shape mismatch')
            self.surface = np.array(surface, dtype=object)
        else:
            self.surface = np.tile(surface, len(self.mesh.triangles))

        if np.iterable(color):
            if len(color) != len(mesh.triangles):
                raise ValueError('shape mismatch')
            self.color = np.array(color, dtype=np.uint32)
        else:
            self.color = np.tile(color, len(self.mesh.triangles)).astype(np.uint32)

        self.unique_materials = \
            silly_unique(np.concatenate([self.inner_material, self.outer_material]))

        self.unique_surfaces = silly_unique(self.surface)

    def __add__(self, other):
        return Solid(self.mesh + other.mesh, np.concatenate((self.inner_material, other.inner_material)), np.concatenate((self.outer_material, other.outer_material)), np.concatenate((self.surface, other.surface)), np.concatenate((self.color, other.color)))

    def weld(self, other, shared_triangle_surface=None, shared_triangle_color=None):
        '''Merge this solid with another at any identical triangles.

        Triangles that are common to both solids will be reduced to one, with the
        surface and color properties of this one unless otherwise specified with
        ``shared_triangle_surface`` and/or ``shared_triangle_color``.

        Note that this is NOT a boolean union!
        '''
        # create arrays of sets of (three-tuple) points
        # use sets to take care of point order permutations, numpy element-wise-ness
        points_self = np.array([set(map(tuple, [self.mesh.vertices[x] for x in y])) for y in self.mesh.triangles], dtype=set)
        points_other = np.array([set(map(tuple, [other.mesh.vertices[x] for x in y])) for y in other.mesh.triangles], dtype=set)

        # find any triangles in other that also exist in self
        match = [np.where(points_other == x)[0] for x in points_self]
        mask = np.array([len(x)>0 for x in match])
        if mask.sum() == 0:
            raise Exception('cannot weld solids with no shared triangles')

        # get triangle ids of the duplicates
        duplicates = sorted(list(chain(*[np.where(points_other == x)[0] for x in points_self])))

        # create temporary second solid -- solid2 with duplicate triangles removed
        mesh = Mesh(other.mesh.vertices, np.delete(other.mesh.triangles, duplicates, 0))
        inner_material = np.delete(other.inner_material, duplicates, 0)
        outer_material = np.delete(other.outer_material, duplicates, 0)
        surface = np.delete(other.surface, duplicates, 0)
        color = np.delete(other.color, duplicates, 0)

        self.mesh = self.mesh + mesh
        self.inner_material = np.concatenate((self.inner_material, inner_material))
        self.outer_material = np.concatenate((self.outer_material, outer_material))
        self.surface = np.concatenate((self.surface, surface))
        self.color = np.concatenate((self.color, color))

        # set properties at interface
        self.outer_material[mask] = other.inner_material[0]
        if shared_triangle_surface is not None:
            self.surface[mask] = shared_triangle_surface
        if shared_triangle_color is not None:
            self.color[mask] = shared_triangle_color

    @memoize_method_with_dictionary_arg
    def inner_material_indices(self, material_lookup):
        return np.fromiter(map(material_lookup.get, self.inner_material), dtype=np.int32, count=len(self.inner_material))

    @memoize_method_with_dictionary_arg
    def outer_material_indices(self, material_lookup):
        return np.fromiter(map(material_lookup.get, self.outer_material), dtype=np.int32, count=len(self.outer_material))

    @memoize_method_with_dictionary_arg
    def surface_indices(self, surface_lookup):
        return np.fromiter(map(surface_lookup.get, self.surface), dtype=np.int32, count=len(self.surface))

class Material(object):
    """Material optical properties."""
    def __init__(self, name='none'):
        self.name = name

        self.refractive_index = None
        self.absorption_length = None
        self.scattering_length = None
        self.scintillation_spectrum = None
        self.scintillation_light_yield = None
        self.scintillation_rise_time = None
        self.scintillation_waveform = None
        self.scintillation_mod = None
        self.comp_reemission_prob = []
        self.comp_reemission_wvl_cdf = []
        self.comp_reemission_times = []  # FIXME: Not used?
        self.comp_reemission_time_cdf = []
        self.comp_absorption_length = []
        self.density = 0.0 # g/cm^3
        self.composition = {} # by mass

    def set(self, name, value, wavelengths=standard_wavelengths):
        if np.iterable(value):
            if len(value) != len(wavelengths):
                raise ValueError('shape mismatch')
        else:
            value = np.tile(value, len(wavelengths))

        self.__dict__[name] = np.array(list(zip(wavelengths, value)), dtype=np.float32)

# Empty material
vacuum = Material('vacuum')
vacuum.set('refractive_index', 1.0)
vacuum.set('absorption_length', 1e6)
vacuum.set('scattering_length', 1e6)

class DichroicProps(object):
    def __init__(self,angles,reflect,transmit):
        self.angles = np.asarray(angles) #[angle]
        self.dichroic_reflect = np.asarray(reflect) #[angle][point]
        self.dichroic_transmit = np.asarray(transmit) #[angle][point]

class Surface(object):
    """Surface optical properties."""
    def __init__(self, name='none', model=0):
        self.name = name
        self.model = model

        self.set('detect', 0)
        self.set('absorb', 0)
        self.set('reemit', 0)
        self.set('reflect_diffuse', 0)
        self.set('reflect_specular', 0)
        self.set('eta', 0)
        self.set('k', 0)
        self.set('reemission_cdf', 0)
        
        self.dichroic_props = None

        self.thickness = 0.0
        self.transmissive = 0

    def set(self, name, value, wavelengths=standard_wavelengths):
        if np.iterable(value):
            if len(value) != len(wavelengths):
                raise ValueError('shape mismatch')
        else:
            value = np.tile(value, len(wavelengths))

        if (np.asarray(value) < 0.0).any():
            raise Exception('all probabilities must be >= 0.0')

        self.__dict__[name] = np.array(list(zip(wavelengths, value)), dtype=np.float32)
    def __repr__(self):
        return '<Surface %s>' % self.name
        
class Geometry(object):
    "Geometry object."
    def __init__(self, detector_material=None):
        self.detector_material = detector_material
        self.solids = []
        self.solid_rotations = []
        self.solid_displacements = []
        self.bvh = None

    def add_solid(self, solid, rotation=None, displacement=None):
        """
        Add the solid `solid` to the geometry. When building the final triangle
        mesh, `solid` will be placed by rotating it with the rotation matrix
        `rotation` and displacing it by the vector `displacement`.
        """

        if rotation is None:
            rotation = np.identity(3)
        else:
            rotation = np.asarray(rotation, dtype=np.float32)

        if rotation.shape != (3,3):
            raise ValueError('rotation matrix has the wrong shape.')

        self.solid_rotations.append(rotation.astype(np.float32))

        if displacement is None:
            displacement = np.zeros(3)
        else:
            displacement = np.asarray(displacement, dtype=np.float32)

        if displacement.shape != (3,):
            raise ValueError('displacement vector has the wrong shape.')

        self.solid_displacements.append(displacement)

        self.solids.append(solid)

        return len(self.solids)-1

    def flatten(self):
        """
        Create the flat list of triangles (and triangle properties)
        from the list of solids in this geometry.

        This does not build the BVH!  If you want to use the geometry
        for rendering or simulation, you should call build() instead.
        """

        # Don't run this function twice!
        if hasattr(self, 'mesh'):
            return

        nv = np.cumsum([0] + [len(solid.mesh.vertices) for solid in self.solids])
        nt = np.cumsum([0] + [len(solid.mesh.triangles) for solid in self.solids])

        vertices = np.empty((nv[-1],3), dtype=np.float32)
        triangles = np.empty((nt[-1],3), dtype=np.uint32)
        

        logger.info('Flattening detector mesh...')
        logger.info('  triangles: %d' % len(triangles))
        logger.info('  vertices:  %d' % len(vertices))


        for i, solid in enumerate(self.solids):
            vertices[nv[i]:nv[i+1]] = \
                np.inner(solid.mesh.vertices, self.solid_rotations[i]) + self.solid_displacements[i]
            triangles[nt[i]:nt[i+1]] = solid.mesh.triangles + nv[i]

        # Different solids are very unlikely to share vertices, so this goes much faster
        self.mesh = Mesh(vertices, triangles, remove_duplicate_vertices=True, remove_null_triangles=False)

        self.colors = np.concatenate([solid.color for solid in self.solids])

        self.solid_id = np.concatenate([filled_array(i, shape=len(solid.mesh.triangles), dtype=np.uint32) for i, solid in enumerate(self.solids)])

        self.unique_materials = list(silly_unique(np.concatenate([solid.unique_materials for solid in self.solids])))

        material_lookup = dict(list(zip(self.unique_materials, list(range(len(self.unique_materials))))))

        self.inner_material_index = np.concatenate([solid.inner_material_indices(material_lookup) for solid in self.solids])

        self.outer_material_index = np.concatenate([solid.outer_material_indices(material_lookup) for solid in self.solids])

        self.unique_surfaces = list(silly_unique(np.concatenate([solid.unique_surfaces for solid in self.solids])))

        surface_lookup = dict(list(zip(self.unique_surfaces, list(range(len(self.unique_surfaces))))))

        self.surface_index = np.concatenate([solid.surface_indices(surface_lookup) for solid in self.solids])

        try:
            self.surface_index[self.surface_index == surface_lookup[None]] = -1
        except KeyError:
            pass

