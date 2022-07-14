import chroma.make as make
import pymesh
from chroma.geometry import Mesh
from chroma.transform import rotate, make_rotation_matrix
import numpy as np

from chroma.log import logger

## Generate meshes for GDML solids. 
## TODO add extra parameters for mesh resolution

def mesh_to_pymesh(mesh):
    return pymesh.form_mesh(mesh.vertices, mesh.triangles)

def pymesh_to_mesh(mesh_p):
    return Mesh(mesh_p.vertices, mesh_p.faces)

def gdml_boolean(mesh_1, mesh_2, op, engine='auto', pos=None, rot=None):
    # pymesh boolean method wrapper for chroma mesh
    if op == 'subtraction':
        op = 'difference' # difference is called subtraction in gdml
    if rot is not None:
        rot_matrix = np.identity(3)
        for idx, phi in enumerate(rot):
            axis = np.zeros(3)
            axis[idx] = 1
            rot_matrix = np.inner(rot_matrix, make_rotation_matrix(phi, axis))
        mesh_2.vertices = np.inner(mesh_2.vertices, rot_matrix)
    if pos is not None:
        mesh_2.vertices += pos
    m1_p = mesh_to_pymesh(mesh_1)
    m2_p = mesh_to_pymesh(mesh_2)
    result_p = pymesh.boolean(m1_p, m2_p, op, engine=engine)
    return Mesh(result_p.vertices, result_p.faces)

def gdml_box(dx, dy, dz):
    return make.box(dx, dy, dz)

def cylinder_segment(r, z, startphi, deltaphi):
    angles = np.linspace(startphi, startphi+deltaphi, 5, endpoint=True)
    vertices = np.vstack(
        [[[0, 0, -z/2]], 
        np.array([r * np.cos(angles), r * np.sin(angles), -(z/2)*np.ones(len(angles))]).transpose(), 
        [0, 0, z/2], 
        np.array([r * np.cos(angles), r * np.sin(angles), (z/2)*np.ones(len(angles))]).transpose()]
        )
    
    # connect the caps
    faces = [[0, v+1, v] for v in range(1, len(angles))] + \
            [[len(angles)+1, len(angles) + 1 + v, len(angles) + 2 + v] for v in range(1, len(angles))] 
    # make the sides
    for i in range(len(angles)):
        faces.extend(
            [[i, i+1, i+1+len(angles)], [i+1+len(angles), i+1, i+2+len(angles)]])
    faces.extend([[len(angles), 0, 2*len(angles)+1], [2*len(angles)+1, 0, len(angles)+1]])
    return Mesh(vertices, faces)

def gdml_polycone(startphi, deltaphi, zplane, nsteps=64):
    seg_list = []
    zplane = sorted(zplane, key=lambda p: p['z'])
    for pa, pb in zip(zplane, zplane[1:]):
        # zplane has elements rmin, rmax, z
        center_a = [0, 0, pa['z']]
        center_b = [0, 0, pb['z']]
        segment = pymesh.generate_tube(
            center_a, center_b, # centers
            pa['rmax'], pb['rmax'], # outer radii
            pa['rmin'], pb['rmin'], # inner radius
            num_segments=nsteps
        )
        seg_list.append(segment)
    # weld everything together
    result = seg_list[0]
    for seg in seg_list[1:]:
        result = pymesh.boolean(result, seg, 'union')
    result = pymesh_to_mesh(result)
    # Cut in phi
    if deltaphi < np.pi*2:
        max_radius = max([p['rmax'] for p in zplane])
        max_height = max([abs(p['z']) for p in zplane])
        segment = cylinder_segment(max_radius*1.5, max_height*1.5, startphi, deltaphi)
        result = gdml_boolean(result, segment, 'intersection')
    return result
    

def _solid_polyhedra(x_list, z_list, startphi, deltaphi, numsides):
    " Generate a solid polyhedra that has a leftover sides from startphi+delaphi to startphi"
    profile = np.array([x_list, np.zeros(len(x_list)), z_list]).transpose()
    if not profile[0, 0] == 0:
        cap_point = np.array([0, 0, profile[0, 2]])
        profile = np.insert(profile, 0, cap_point, axis=0)
    if not profile[-1, 0] == 0:
        cap_point = np.array([0, 0, profile[-1, 2]])
        profile = np.vstack([profile, cap_point])
    steps = np.linspace(startphi, startphi+deltaphi, numsides+1, endpoint=True)
    # ensure rotation is one full revolution by adding 90 degree faces
    if deltaphi < np.pi*2:
        leftover = np.pi*2 - deltaphi
        while leftover > np.pi*2/3:
            steps = np.append(steps, steps[-1] + np.pi/2)
            leftover -= np.pi/2
        steps = np.append(steps, startphi) 
    vertices = np.vstack([rotate(profile, angle, (0, 0, -1)) for angle in steps])
    triangles = make.mesh_grid(np.arange(len(vertices)).reshape((len(steps),len(profile))).transpose()[::-1])
    return Mesh(vertices, triangles, remove_duplicate_vertices=True)

def gdml_polyhedra(startphi, deltaphi, numsides, zplane):
    if deltaphi > np.pi*2:
        logger.warning("Polyhedra does not support deltaphi greater than 2*pi, proceeding with deltaphi = 2*pi")
        deltaphi = np.pi * 2
    zplane = sorted(zplane, key=lambda p: p['z'])
    z_list = [p['z'] for p in zplane]
    rmin_list = [p['rmin'] for p in zplane]
    rmax_list = [p['rmax'] for p in zplane]
    
    outer = _solid_polyhedra(rmax_list, z_list, startphi, deltaphi, numsides)
    inner = _solid_polyhedra(rmin_list, z_list, startphi, deltaphi, numsides)
    result = gdml_boolean(outer, inner, "difference")
    # Cut in phi
    if deltaphi < np.pi*2:
        max_radius = max([p['rmax'] for p in zplane])
        max_height = max([abs(p['z']) for p in zplane])
        segment = cylinder_segment(max_radius*1.5, max_height*1.5, startphi, deltaphi)
        result = gdml_boolean(result, segment, 'intersection')
    # cleanup
    result.vertices, result.triangles, _ = pymesh.collapse_short_edges_raw(result.vertices, result.triangles, abs_threshold = min(min([p['rmin'] for p in zplane])/20, 10.))
    return result


def gdml_tube(rmin, rmax, z, startphi, deltaphi):
    if deltaphi > np.pi * 2:
        deltaphi = np.pi*2
    full_tube = pymesh.generate_tube([0, 0, -z/2], [0, 0, z/2], 
        rmax, rmax, rmin, rmin,
        num_segments=64)
    segment = cylinder_segment(rmax*1.5, z*1.5, startphi, deltaphi)
    result = pymesh.boolean(full_tube, mesh_to_pymesh(segment), 'intersection')
    return pymesh_to_mesh(result)

def _sphere_segment_theta(r, starttheta, endtheta, nsteps=64):
    "make a sphere with theta extending from starttheta to endtheta"
    assert starttheta ==0 or endtheta == np.pi # can only generate closed solids.
    thetas = np.linspace(starttheta, endtheta, nsteps, endpoint=True)
    points = np.array([r * np.sin(thetas), r * np.cos(thetas)]).transpose()
    points = np.vstack([[0, 0], points, [0, 0]])
    result = make.rotate_extrude(points[:, 0], points[:, 1],nsteps)
    # Fix rotated shape's orientation, making it equivalent to spinning around z axis
    result.vertices[:, [1, 2]] = result.vertices[:, [2, 1]]
    result.vertices[:, 1] *= -1
    return result

def gdml_orb(r, order=4):
    result_p = pymesh.generate_icosphere(r, center=(0, 0, 0), refinement_order=order)
    return pymesh_to_mesh(result_p)

def gdml_sphere(rmin, rmax, startphi, deltaphi, starttheta, deltatheta):
    print(rmin, rmax, startphi, deltaphi, starttheta, deltatheta)
    # GDML Sphere can be an incomplete spherical shell
    assert (starttheta >= 0 and starttheta <= np.pi and starttheta + deltatheta >= 0 and starttheta + deltatheta <= np.pi), \
        logger.error("theta spec is not between [0, pi]")
    # make spherical shell
    result = gdml_orb(rmax)
    if rmin > 0:
        result = gdml_boolean(result, gdml_orb(rmin), "difference")
    # carve in phi direction
    if deltaphi < 2 * np.pi:
        segment = cylinder_segment(rmax*1.5, 2*rmax*1.5, startphi, deltaphi)
        result = gdml_boolean(result, segment, 'intersection')
    # cleanup
    # carve in theta
    endtheta = starttheta + deltatheta
    nsteps = 64
    ## Cut off [0, starttheta]
    if starttheta > 0:
        top = _sphere_segment_theta(rmax*1.5, 0, starttheta, nsteps)
        result = gdml_boolean(result, top, "difference", engine='cgal')
    ## Cut off [endtheta, pi]
    if endtheta < np.pi:
        bottom = _sphere_segment_theta(rmax*1.5, endtheta, np.pi, nsteps)
        result = gdml_boolean(result, bottom, "difference", engine='cgal')
    # result.vertices, result.triangles, _ = pymesh.split_long_edges_raw(result.vertices, result.triangles, 
    #     max_edge_length=0.1)
    # result.vertices, result.triangles, _ = pymesh.collapse_short_edges_raw(result.vertices, result.triangles, rel_threshold=0.25)
    result.vertices, result.triangles, _ = pymesh.collapse_short_edges_raw(result.vertices, result.triangles, abs_threshold = min(rmin/20, 10.))

    return result



def gdml_torus(rmin, rmax, rtor, startphi, deltaphi, nsteps=64):
    result = make.torus(rmax, rtor, nsteps=nsteps)
    # swap yz
    result.vertices[:, [1, 2]] = result.vertices[:, [2, 1]]
    result.vertices[:, 1] *= -1
    # Cut in phi
    if deltaphi < np.pi*2:
        max_radius = rtor + rmax
        max_height = rmax*2
        segment = cylinder_segment(max_radius*1.5, max_height*1.5, startphi, deltaphi)
        result = gdml_boolean(result, segment, 'intersection')
    if rmin != 0:
        inner = make.torus(rmin, rtor, nsteps=nsteps)
        # swap yz
        inner.vertices[:, [1, 2]] = inner.vertices[:, [2, 1]]
        inner.vertices[:, 1] *= -1
        result = gdml_boolean(result, inner, 'difference')
    
    return result


def gdml_eltube(dx, dy, dz, nsteps=64):
    ## NOTE: dz is HALF of total height! (don't ask why)
    angles = np.linspace(0, 2*np.pi, nsteps, endpoint=False)
    return make.linear_extrude(dx*np.cos(angles), dy*np.sin(angles), dz*2)
    