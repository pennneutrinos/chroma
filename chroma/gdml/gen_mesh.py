import chroma.make as make
import pymesh
from chroma.geometry import Mesh
from chroma.transform import rotate, rotate_matrix, make_rotation_matrix
from chroma.log import logger

import numpy as np
import copy 

## Generate meshes for GDML solids. 
## TODO add extra parameters for mesh resolution

def mesh_to_pymesh(mesh):
    return pymesh.form_mesh(mesh.vertices, mesh.triangles)

def pymesh_to_mesh(mesh_p):
    return Mesh(mesh_p.vertices, mesh_p.faces)

def transform(mesh, pos=None, rot=None):
    '''Perform translation and rotation for GDML solids.
    NOTE: GDML solids define rotation oppositely than chroma. Positive rotation is a clockwise rotation when looking
    towards infinity.
    '''
    mesh_result = copy.deepcopy(mesh)
    if rot is not None:
        for idx, phi in enumerate(rot):
            axis = np.zeros(3)
            axis[idx] = 1
            mesh_result.vertices = rotate_matrix(mesh_result.vertices, -phi, axis)
    if pos is not None:
        mesh_result.vertices += pos
    return mesh_result

def remove_zero_area_faces_raw(vertices, faces):
    new_faces = []
    for face in faces:
        a = vertices[face[0]]
        b = vertices[face[1]]
        c = vertices[face[2]]
        if np.array_equal(a, b) or np.array_equal(a, c) or np.array_equal(b, c): continue
        product = np.linalg.norm(np.cross(b-a, c-a))
        if product!=0:
            new_faces.append(face)
    new_faces = np.asarray(new_faces)
    new_vertices, new_faces, _ = pymesh.remove_isolated_vertices_raw(vertices, new_faces)
    return new_vertices, new_faces

def pymesh_remove_zero_area_faces(mesh_p):
    mesh = pymesh_to_mesh(mesh_p)
    mesh.vertices, mesh.triangles = remove_zero_area_faces_raw(mesh.vertices, mesh.triangles)
    return mesh_to_pymesh(mesh)
    
def gdml_boolean(mesh_1, mesh_2, op, engine='auto', firstpos = None, firstrot= None, pos=None, rot=None):
    # pymesh boolean method wrapper for chroma mesh
    if op == 'subtraction':
        op = 'difference' # difference is called subtraction in gdml
    mesh_1 = transform(mesh_1, firstpos, firstrot)
    mesh_2 = transform(mesh_2, pos, rot)
    
    m1_p = mesh_to_pymesh(mesh_1)
    m2_p = mesh_to_pymesh(mesh_2)
    result_p = pymesh.boolean(m1_p, m2_p, op, engine=engine)
    # if len(result_p.vertices) > (len(m1_p.vertices) + len(m2_p.vertices) + 1000):
    #     logger.info("Boolean op generated a lot more additional triangles, optimizing...")
    #     result_p, info = pymesh.collapse_short_edges(result_p, rel)
    result = pymesh_to_mesh(result_p)
    result_p = mesh_to_pymesh(result)
    # logger.debug(f"{result_p.vertices.shape}, {result_p.faces.shape}")
    # logger.debug("Cleaning up")
    cleaned, _ = pymesh.remove_duplicated_vertices(result_p, tol=1e-10)
    # cleaned, _ = pymesh.remove_degenerated_triangles(cleaned)
    cleaned, _ = pymesh.remove_duplicated_faces(cleaned)
    # cleaned, _ = pymesh.remove_degenerated_triangles(cleaned)
    # cleaned = pymesh.resolve_self_intersection(cleaned)
    # cleaned, _ = pymesh.remove_duplicated_faces(cleaned)
    # logger.debug(f"{cleaned.vertices.shape}, {cleaned.faces.shape}")
    return pymesh_to_mesh(result_p)

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
    # Apparently rmax and rmin are distances to the middle of the sides, not to the vertices
    theta_per_side = deltaphi / numsides
    x_list = [x / np.cos(theta_per_side/2) for x in x_list]
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

def _sphere_segment_theta(r, starttheta, endtheta, nsteps=16):
    "make a sphere with theta extending from starttheta to endtheta"
    assert starttheta ==0 or endtheta == np.pi # can only generate closed solids.
    thetas = np.linspace(endtheta, starttheta, nsteps, endpoint=True)
    points = np.array([r * np.sin(thetas), r * np.cos(thetas)]).transpose()
    points = np.vstack([[0, 0], points, [0, 0]])
    result = make.rotate_extrude(points[:, 0], points[:, 1],nsteps)
    # Fix rotated shape's orientation, making it equivalent to spinning around z axis
    result.vertices[:, [1, 2]] = result.vertices[:, [2, 1]]
    result.vertices[:, 1] *= -1
    return result

def gdml_orb(r, order=3):
    result_p = pymesh.generate_icosphere(r, center=(0, 0, 0), refinement_order=order)
    return pymesh_to_mesh(result_p)

def gdml_sphere(rmin, rmax, startphi, deltaphi, starttheta, deltatheta):
    # GDML Sphere can be an incomplete spherical shell
    assert (starttheta >= 0 and starttheta <= np.pi and starttheta + deltatheta >= 0 and starttheta + deltatheta <= np.pi), \
        logger.error("theta spec is not between [0, pi]")
    # make spherical shell
    result = gdml_orb(rmax, order=3)
    if rmin > 0:
        result = gdml_boolean(result, gdml_orb(rmin, order=3), "difference")
    # carve in phi direction
    if deltaphi < 2 * np.pi:
        segment = cylinder_segment(rmax*1.5, 2*rmax*1.5, startphi, deltaphi)
        result = gdml_boolean(result, segment, 'intersection')
    # cleanup
    # carve in theta
    endtheta = starttheta + deltatheta
    nsteps = 16

    if endtheta < np.pi:
        bottom = _sphere_segment_theta(rmax*1.5, 0, endtheta, nsteps)
        result = gdml_boolean(result, bottom, "intersection")
    if starttheta > 0:
        top = _sphere_segment_theta(rmax*1.5, starttheta, np.pi, nsteps)
        result = gdml_boolean(result, top, "intersection")
    # result.vertices, result.triangles, _ = pymesh.split_long_edges_raw(result.vertices, result.triangles, 
    #     max_edge_length=0.1)
    # result.vertices, result.triangles, _ = pymesh.collapse_short_edges_raw(result.vertices, result.triangles, rel_threshold=0.25)
    result.vertices, result.triangles, _ = pymesh.collapse_short_edges_raw(result.vertices, result.triangles, abs_threshold = min(rmin/20, 10.))

    return result



def gdml_torus(rmin, rmax, rtor, startphi, deltaphi, nsteps=64, circle_steps=64):
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
        inner = make.torus(rmin, rtor, nsteps=nsteps, circle_steps=circle_steps)
        # swap yz
        inner.vertices[:, [1, 2]] = inner.vertices[:, [2, 1]]
        inner.vertices[:, 1] *= -1
        result = gdml_boolean(result, inner, 'difference')
    
    return result


def gdml_eltube(dx, dy, dz, nsteps=64):
    ## NOTE: dz is HALF of total height! (don't ask why)
    angles = np.linspace(0, 2*np.pi, nsteps, endpoint=False)
    return make.linear_extrude(dx*np.cos(angles), dy*np.sin(angles), dz*2)
    