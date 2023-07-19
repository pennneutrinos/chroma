import sys
import os
import time

from chroma.log import logger
from chroma.cache import Cache
from chroma.bvh import make_simple_bvh, make_recursive_grid_bvh
from chroma.geometry import Geometry, Solid, Mesh, vacuum
from chroma.detector import Detector
from chroma.stl import mesh_from_stl
from chroma.gpu import create_cuda_context
from chroma.gdml import GDMLLoader

def load_geometry_from_string(geometry_str, 
                              auto_build_bvh=True, read_bvh_cache=True,
                              update_bvh_cache=True, cache_dir=None,
                              cuda_device=None):
    '''Create or load a geometry and optionally load/build a BVH for it.

    This is a convenience interface to the geometry and BVH construction code,
    as well as the Chroma caching layer.  Most applications should use
    this function rather than manually building a Geometry and BVH.

    The geometry string passed to this function has several forms:

      "" (empty string) - Load the default geometry from the cache and
          the default BVH for that geometry.

      "filename.stl" or "filename.stl.bz2" - Create a geometry from a
          3D mesh on disk.  This model will not be cached, but the
          BVH can be, depending on whether update_bvh_cache is True.

      "geometry_name" - Load a geometry from the cache with this name
          and the default BVH for that geometry.

      "geometry_name:bvh_name" - Load a geometry from the cache and
          the requested BVH by name.
                                 
      "@chroma.models.lionsolid" - Run this function inside a Python
          module, found in the current $PYTHONPATH, to create the
          geometry, and load the default BVH.  For convenience, the
          current directory is also added to the $PYTHONPATH.

      "@chroma.models.lionsolid:bvh_name" - Run this function to
          create the Geometry and load a BVH by name.

    By default, the Chroma cache in the user's home directory is
    consulted for both the geometry and the BVH.  A different cache
    directory can be selected by passing the path in via the
    ``cache_dir`` parameter.

    If ``read_bvh_cache`` is set to False, then the BVH cache will not
    be inspected for BVH objects.

    If the requested BVH (default, or named) does not exist for this
    geometry (checked by MD5 hashing the geometry mesh) and
    ``auto_build_bvh`` is true, then a BVH will be automatically
    generated using the "simple" BVH algorithm.  The simple algorithm
    is very fast, but produces a poor quality BVH.

    Any newly created BVH will be saved in the Chroma cache if the
    ``update_cache_bvh`` parameter is True.
    
    BVH construction requires a GPU, so the CUDA device number can be
    specified with the ``cuda_device`` parameter.

    Returns: a Geometry object (or subclass) with the ``bvh`` property
      set if the options allow.
    '''
    # Find BVH id if given
    bvh_name = 'default'
    if ':' in geometry_str:
        geometry_id, bvh_name = geometry_str.split(':')
    else:
        geometry_id = geometry_str

    if cache_dir is None:
        cache = Cache()
    else:
        cache = Cache(cache_dir)

    # Where is the geometry coming from?
    if os.path.exists(geometry_id) and \
            geometry_id.lower().endswith(('.stl', '.bz2')):
        # Load from file
        mesh = mesh_from_stl(geometry_id)
        geometry = Geometry()
        geometry.add_solid(Solid(mesh, vacuum, vacuum, color=0x33ffffff))
        geometry.flatten()

    if os.path.exists(geometry_id) and \
            geometry_id.lower().endswith('.gdml'):
        gdml_load = GDMLLoader(geometry_id)
        geometry = gdml_load.build_detector()
        geometry.flatten()

    elif geometry_id.startswith('@'):
        # Load from function
        function_path = geometry_id[1:]

        module_name, obj_name = function_path.rsplit('.', 1)
        orig_sys_path = list(sys.path)
        try:
            sys.path.append('.')
            module = __import__(module_name, fromlist=[obj_name])
            sys.path = orig_sys_path
        except ImportError:
            sys.path = orig_sys_path
            raise

        obj = getattr(module, obj_name)

        geometry = create_geometry_from_obj(obj, bvh_name=bvh_name,
                                            auto_build_bvh=auto_build_bvh, 
                                            read_bvh_cache=read_bvh_cache,
                                            update_bvh_cache=update_bvh_cache,
                                            cache_dir=cache_dir,
                                            cuda_device=cuda_device)
        return geometry # RETURN EARLY HERE!  ALREADY GOT BVH

    else:
        # Load from cache
        if geometry_id == '':
            geometry = cache.load_default_geometry()
        else:
            geometry = cache.load_geometry(geometry_id)
        # Cached geometries are flattened already

    geometry.bvh = load_bvh(geometry, bvh_name=bvh_name,
                            auto_build_bvh=auto_build_bvh,
                            read_bvh_cache=read_bvh_cache,
                            update_bvh_cache=update_bvh_cache,
                            cache_dir=cache_dir,
                            cuda_device=cuda_device)

    return geometry

def load_bvh(geometry,  bvh_name="default", 
             auto_build_bvh=True, read_bvh_cache=False,
             update_bvh_cache=True, cache_dir=None,
             cuda_device=None):
    if cache_dir is None:
        cache = Cache()
    else:
        cache = Cache(cache_dir)

    mesh_hash = geometry.mesh.md5()
    bvh = None
    if read_bvh_cache and cache.exist_bvh(mesh_hash, bvh_name):
        logger.info('Loading BVH "%s" for geometry from cache.' % bvh_name)
        bvh = cache.load_bvh(mesh_hash, bvh_name)
    elif auto_build_bvh:
        logger.info('Building new BVH using recursive grid algorithm.')

        start = time.time()

        context = create_cuda_context(cuda_device)
        bvh = make_recursive_grid_bvh(geometry.mesh, target_degree=3)
        context.pop()

        logger.info('BVH generated in %1.1f seconds.' % (time.time() - start))

        if update_bvh_cache:
            logger.info('Saving BVH (%s:%s) to cache.' % (mesh_hash, bvh_name))
            cache.save_bvh(bvh, mesh_hash, bvh_name)

    return bvh

def create_geometry_from_obj(obj, bvh_name="default", 
                             auto_build_bvh=True, read_bvh_cache=True,
                             update_bvh_cache=True, cache_dir=None,
                             cuda_device=None):
    if callable(obj):
        obj = obj()

    if isinstance(obj, Detector):
        geometry = obj
    if isinstance(obj, Geometry):
        geometry = obj
    elif isinstance(obj, Solid):
        geometry = Geometry()
        geometry.add_solid(obj)
    elif isinstance(obj, Mesh):
        geometry = Geometry()
        geometry.add_solid(Solid(obj, vacuum, vacuum, color=0x33ffffff))
    else:
        raise TypeError('cannot build type %s' % type(obj))

    geometry.flatten()

    if geometry.bvh is None:
        geometry.bvh = load_bvh(geometry, auto_build_bvh=auto_build_bvh,
                                read_bvh_cache=read_bvh_cache,
                                update_bvh_cache=update_bvh_cache,
                                cache_dir=cache_dir,
                                cuda_device=cuda_device)

    return geometry
