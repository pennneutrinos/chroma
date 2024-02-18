import gmsh

from chroma.geometry import Mesh
from chroma import transform

occ = gmsh.model.occ

import numpy as np

occ = gmsh.model.occ
from chroma.log import logger


def getTagsByDim(dimTags, dim):
    return [dimTag[1] for dimTag in dimTags if dimTag[0] == dim]


def getDimTagsByDim(dimTags, dim):
    return [dimTag for dimTag in dimTags if dimTag[0] == dim]


def getDimTags(dim, tags):
    if type(tags) == int:
        return [(dim, tags)]
    result = []
    for tag in tags:
        result.append((dim, tag))
    return result


def gdml_transform(obj, pos=None, rot=None):
    if pos is None:
        pos = [0., 0., 0.]
    if rot is None:
        rot = [0., 0., 0.]
    if np.ndim(rot) == 1:
        assert len(rot) == 3, "rotation defined in not 3 dimensions. Too much string theory?"
        for axis_idx, angle in enumerate(rot):
            # make the xyz axis
            axis = np.zeros(3)
            axis[axis_idx] = 1
            occ.rotate(getDimTags(3, obj), 0., 0., 0., axis[0], axis[1], axis[2], angle)

    else:  # a rotation matrix is specified:
        assert np.shape(rot) == (3, 3), f"rotation has shape {np.shape(rot)}. Too much string theory?"
        assert len(pos) == 3, f"translation has shape {np.shape(pos)}. Too much string theory?"
        # define the first 12 entries of a 4x4 transformation matrix.
        # https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
        rot_axis, rot_angle = transform.matrix_to_rotvec(rot)
        if rot_angle != 0:
            occ.rotate(getDimTags(3, obj), 0, 0, 0, *rot_axis, rot_angle)
    occ.translate(getDimTags(3, obj), pos[0], pos[1], pos[2])
    return obj


def gdml_boolean(a, b, op, pos=None, rot=None, firstpos=None, firstrot=None, deleteA=True, deleteB=True, noUnion=False):
    # Deal with all none objects
    if op == 'union':
        if a is None:
            return b
        if b is None:
            return a
    if op == 'subtraction':
        assert a is not None, "Subtraction requires first object to be not None"
        if b is None:
            return a  # Subtracting nothing is a no-op
    if op == 'intersection':
        assert a is not None and b is not None, "Intersection requires both objects to be not None"
    a = gdml_transform(a, pos=firstpos, rot=firstrot)
    b = gdml_transform(b, pos=pos, rot=rot)
    if op in ('subtraction', 'difference'):
        result = occ.cut(getDimTags(3, a), getDimTags(3, b), removeObject=deleteA, removeTool=deleteB)
    elif op in ('union'):
        if noUnion:
            result = getDimTags(3, a) + getDimTags(3, b), None
        else:
            result = occ.fuse(getDimTags(3, a), getDimTags(3, b), removeObject=deleteA, removeTool=deleteB)
    elif op in ('intersection'):
        occ.synchronize()
        result = occ.intersect(objectDimTags=getDimTags(3, a),
                               toolDimTags=getDimTags(3, b),
                               removeObject=deleteA, removeTool=deleteA)
        occ.synchronize()
    else:
        raise NotImplementedError(f'{op} is not implemented.')
    outDimTags, _ = result
    if len(outDimTags) == 0: return None
    if len(outDimTags) > 1:
        logger.info("Note: more than one object created by boolean operation.")
        return [DimTag[1] for DimTag in outDimTags]
    return outDimTags[0][1]


def gdml_box(dx, dy, dz):
    result = occ.addBox(-dx / 2, -dy / 2, -dz / 2, dx, dy, dz)
    return result


def genericCone(x, y, z, dx, dy, dz, r1, r2, tag=-1, angle=2 * np.pi):
    """Generate any cone, even if it is actually a cylinder"""
    if r1 == r2:
        return occ.addCylinder(x, y, z, dx, dy, dz, r1, tag=tag, angle=angle)
    return occ.addCone(x, y, z, dx, dy, dz, r1, r2, tag=tag, angle=angle)


def gdml_polycone(startphi, deltaphi, zplane):
    seg_list = []
    zplane = sorted(zplane, key=lambda p: p['z'])
    for pa, pb in zip(zplane, zplane[1:]):
        # zplane has elements rmin, rmax, z
        segment_out = genericCone(0, 0, pa['z'],
                                  0, 0, pb['z'] - pa['z'],
                                  pa['rmax'], pb['rmax'],
                                  angle=deltaphi)
        segment_in = genericCone(0, 0, pa['z'],
                                 0, 0, pb['z'] - pa['z'],
                                 pa['rmin'], pb['rmin'],
                                 angle=deltaphi)
        segment = gdml_boolean(segment_out, segment_in, 'subtraction')
        seg_list.append(segment)
    # weld everything together
    result = seg_list.pop()
    for seg in seg_list:
        result = gdml_boolean(result, seg, 'union')
    occ.rotate(getDimTags(3, result), 0, 0, 0, 0, 0, 1, startphi)
    return result


def make_face(lines):
    curve_loop = occ.addCurveLoop(lines)
    return occ.addPlaneSurface([curve_loop])


def solid_polyhedra(startphi, deltaphi, numsides, r_list, z_list):
    assert len(r_list) == len(z_list) == 2
    assert z_list[0] != z_list[-1]
    if r_list[0] == r_list[-1] == 0: return None
    dphi = deltaphi / numsides
    vertexLengthFactor = 1 / np.cos(dphi / 2)
    planes = []
    pointsPerBase = numsides if deltaphi == np.pi * 2 else numsides + 2  # number of points in the polygon
    # Create the bases
    for (r, z) in zip(r_list, z_list):
        vertices = []
        firstPoint = occ.addPoint(r * vertexLengthFactor, 0, z)
        if r == 0:
            vertices = [firstPoint] * pointsPerBase
        vertices.append(firstPoint)
        for i in range(numsides - 1):
            p_dimTag = occ.copy([(0, vertices[-1])])
            occ.rotate(p_dimTag,
                       0, 0, 0,
                       0, 0, 1,
                       dphi)
            vertices.append(p_dimTag[0][1])
        if deltaphi != np.pi * 2:  # need to add one more rotated point, as well as the origin
            p_dimTag = occ.copy([(0, vertices[-1])])
            occ.rotate(p_dimTag,
                       0, 0, 0,
                       0, 0, 1,
                       dphi)
            vertices.append(p_dimTag[0][1])
            origin = occ.addPoint(0, 0, z)
            vertices.append(origin)
        planes.append(vertices)

    planes = np.asarray(planes)
    bottom = planes[0]
    bottom_rolled = np.roll(bottom, -1)
    if r_list[0] == 0:
        bottom_lines = [None] * pointsPerBase
    else:
        bottom_lines = [occ.addLine(pa, pb) for pa, pb in zip(bottom, bottom_rolled)]
    top = planes[-1]
    top_rolled = np.roll(top, -1)
    if r_list[-1] == 0:
        top_lines = [None] * pointsPerBase
    else:
        top_lines = [occ.addLine(pa, pb) for pa, pb in zip(top, top_rolled)]
    side_lines = [occ.addLine(pa, pb) for pa, pb in zip(bottom, top)]
    side_lines_rolled = np.roll(side_lines, -1)

    faces = []
    for bline, lline, rline, tline in zip(bottom_lines, side_lines, side_lines_rolled, top_lines):
        boarder = []
        if bline: boarder.append(bline)
        boarder.append(rline)
        if tline: boarder.append(-tline)
        boarder.append(-lline)
        faces.append(make_face(boarder))
    # Add bottom and top
    if r_list[0] != 0:
        bottom_face = make_face(bottom_lines)
        faces.insert(0, bottom_face)
    if r_list[-1] != 0:
        top_face = make_face(top_lines)
        faces.insert(-1, top_face)
    surfaceLoop = occ.addSurfaceLoop(faces)
    result = occ.addVolume([surfaceLoop])
    occ.rotate(getDimTags(3, result), 0, 0, 0, 0, 0, 1, startphi)
    return result


def gdml_polyhedra(startphi, deltaphi, numsides, zplane):
    # First vertex is on the positive X half-axis.
    # Specified radius is distance from center to the middle of the edge
    zplane = sorted(zplane, key=lambda p: p['z'])
    segment_list = []
    for pa, pb in zip(zplane, zplane[1:]):
        rmax_list = pa['rmax'], pb['rmax']
        rmin_list = pa['rmin'], pb['rmin']
        z_list = pa['z'], pb['z']
        outer_solid = solid_polyhedra(startphi, deltaphi, numsides, rmax_list, z_list)
        inner_solid = solid_polyhedra(startphi, deltaphi, numsides, rmin_list, z_list)
        if inner_solid is None:
            segment_list.append(outer_solid)
        else:
            segment_list.append(gdml_boolean(outer_solid, inner_solid, op='subtraction'))
    result = segment_list[0]
    for segment in segment_list[1:]:
        result = gdml_boolean(result, segment, op='union')
    # occ.rotate(getDimTags(3, result), 0, 0, 0, 0, 0, 1, startphi)
    return result


def gdml_tube(rmin, rmax, z, startphi, deltaphi):
    pa = occ.addPoint(rmin, 0, -z / 2)
    pb = occ.addPoint(rmax, 0, -z / 2)
    baseArm = occ.addLine(pa, pb)
    occ.rotate(getDimTags(1, baseArm), 0, 0, 0, 0, 0, 1, startphi)
    base_dimTags = getDimTagsByDim(occ.revolve(getDimTags(1, baseArm), 0, 0, 0, 0, 0, 1, deltaphi), 2)
    # numElem = max(1, int(z//100))
    tube_dimTags = occ.extrude(base_dimTags, 0, 0, z)
    tube_tags_3d = getTagsByDim(tube_dimTags, 3)
    assert len(tube_tags_3d) == 1, f'Generated {len(tube_tags_3d)} solids instead of 1.'
    occ.remove(getDimTags(1, baseArm), recursive=True)
    return tube_tags_3d[0]


def gdml_orb(r):
    return occ.addSphere(0, 0, 0, r)


def gdml_sphere(rmin, rmax, startphi, deltaphi, starttheta, deltatheta):
    pa = occ.addPoint(0, 0, rmin)
    pb = occ.addPoint(0, 0, rmax)
    arm = occ.addLine(pa, pb)
    occ.rotate(getDimTags(1, arm), 0, 0, 0, 0, 1, 0, starttheta)
    theta_section_dimTags = getDimTagsByDim(occ.revolve(getDimTags(1, arm), 0, 0, 0, 0, 1, 0, deltatheta), 2)
    occ.rotate(theta_section_dimTags, 0, 0, 0, 0, 0, 1, startphi)
    sphere_dimTags = occ.revolve(theta_section_dimTags, 0, 0, 0, 0, 0, 1, deltaphi)
    sphere_tags_3d = getTagsByDim(sphere_dimTags, 3)
    assert len(sphere_tags_3d) == 1, f'Generated {len(sphere_tags_3d)} solids instead of 1.'
    return sphere_tags_3d[0]


def gdml_ellipsoid(ax, by, cz, zcut1, zcut2):
    base_ellipsoid = occ.addSphere(0, 0, 0, ax)
    squish_b, squish_c = by / ax, cz / ax
    occ.dilate(getDimTags(3, base_ellipsoid), 0, 0, 0, 1, squish_b, squish_c)
    kill_box = occ.addBox(-ax, -by, zcut1, 2 * ax, 2 * by, (zcut2 - zcut1))
    # Do the intersection and then delete. GMSH throws weird errors otherwise. Bug?
    ellipsoid_tags = gdml_boolean(base_ellipsoid, kill_box, 'intersection', deleteA=False, deleteB=False)
    occ.remove(getDimTags(3, [kill_box, base_ellipsoid]), recursive=True)
    return ellipsoid_tags


def gdml_torus(rmin, rmax, rtor, startphi, deltaphi):
    pa = occ.addPoint(rmin, 0, 0)
    pb = occ.addPoint(rmax, 0, 0)
    arm = occ.addLine(pa, pb)
    crossSection = getDimTagsByDim(occ.revolve(getDimTags(1, arm), 0, 0, 0, 0, 1, 0, np.pi * 2), 2)
    occ.translate(crossSection, rtor, 0, 0)
    occ.rotate(crossSection, 0, 0, 0, 0, 0, 1, startphi)
    torus_tags_3d = getTagsByDim(occ.revolve(crossSection, 0, 0, 0, 0, 0, 1, deltaphi), 3)
    occ.remove(getDimTags(1, arm), recursive=True)
    assert len(torus_tags_3d) == 1, f'Generated {len(torus_tags_3d)} solids instead of 1.'
    return torus_tags_3d[0]


def gdml_torusStack(rhoedge, zedge, rhoorigin, zorigin):
    assert len(zedge) == len(rhoedge), "zedge and rhoedge must have the same length"
    assert len(zedge) > 1, "must have at least one segment"
    assert len(zorigin) == len(zedge) - 1, "zorigin must have one less element than zedge"
    assert len(zorigin) == len(rhoorigin), "zorigin must have the same length as rhoorgin"

    if all(zedge[i] < zedge[i + 1] for i in range(len(zedge) - 1)):
        rhoedge.reverse()
        zedge.reverse()
        rhoorigin.reverse()
        zorigin.reverse()

    assert all(zedge[i] > zedge[i + 1] for i in range(len(zedge) - 1)), "zedge must be monotonically decreasing"
    # compute origin coordinates
    z0 = np.asarray(zorigin)
    r0 = np.asarray(rhoorigin)
    z1 = np.asarray(zedge[:-1])
    z2 = np.asarray(zedge[1:])
    r1 = np.asarray(rhoedge[:-1])
    r2 = np.asarray(rhoedge[1:])
    # Create side profile
    edges = [occ.addPoint(r, 0, z) for r, z in zip(rhoedge, zedge)]
    arcs = []
    for ro, zo, rs, re, zs, ze, edge1, edge2 in zip(r0, z0, r1, r2, z1, z2, edges[:-1], edges[1:]):
        if ro == rs and ro == re:
            arcs.append(occ.addLine(edge1, edge2))
        else:
            origin = occ.addPoint(ro, 0, zo)
            arcs.append(occ.addCircleArc(edge1, origin, edge2))
            occ.remove(getDimTags(0, origin))
    # profile is not closed loop, add a straight bottom section to close it off
    if rhoedge[0] != 0:
        firstpoint = occ.addPoint(0, 0, zedge[0])
        arcs.insert(0, occ.addLine(firstpoint, edges[0]))
        edges.insert(0, firstpoint)
    if rhoedge[-1] != 0:
        lastPoint = occ.addPoint(0, 0, zedge[-1])
        arcs.append(occ.addLine(edges[-1], lastPoint))
        edges.append(lastPoint)
    # close the loop by adding a center line
    bottomToTop = occ.addLine(edges[-1], edges[0])
    curveLoop = occ.addCurveLoop([bottomToTop, *arcs])
    plane = occ.addPlaneSurface([curveLoop])
    torus_stack_3d = getTagsByDim(
        occ.revolve([[2, plane]],
                    0, 0, 0,
                    0, 0, 1,
                    2 * np.pi),
        3)
    occ.remove(getDimTags(2, plane), recursive=True)
    return torus_stack_3d[0]


def gdml_eltube(dx, dy, dz):
    if dx >= dy:
        base_curve = occ.addEllipse(0, 0, -dz, dx, dy)
    else:
        base_curve = occ.addEllipse(0, 0, -dz, dy, dx, zAxis=[0, 0, 1], xAxis=[0, 1, 0])
    base_curveLoop = occ.addCurveLoop([base_curve])
    base = occ.addPlaneSurface([base_curveLoop])
    tube_tags_3d = getTagsByDim(occ.extrude([(2, base)], 0, 0, 2 * dz), 3)
    assert len(tube_tags_3d) == 1, f'Generate {len(tube_tags_3d)} solids instead of 1.'
    return tube_tags_3d[0]


def conform_model(root_volume):
    """
    Apply gmsh.model.occ.fragment on sibling volumes, therefore allow the volumes to share surfaces, making the mesh
    "conformal".
    Args:
        root_volume: tag of the root volume in the model.
    """
    logger.info(f"Conforming children of {root_volume.placementName}")
    if not root_volume.in_gmsh_model:
        return
    child_tags = []
    for child in root_volume.children:
        if child.in_gmsh_model:
            child_tags.append(child.gmsh_tag)
    logger.debug(f"There are {len(child_tags)} children")
    if len(child_tags) > 1:
        child_dimTags = getDimTags(3, child_tags)
        occ.synchronize()
        result_dimTags, result_mapping = occ.fragment(child_dimTags, child_dimTags)
        result_tags = getTagsByDim(result_dimTags, 3)
        if set(child_tags) != set(result_tags):
            logger.warn(f"Children of {root_volume.placementName} potentially overlaps.")
            for i, mapping in enumerate(result_mapping[:len(child_dimTags)]):
                if len(mapping) > 1:
                    for child in root_volume.children:
                        if child.gmsh_tag == child_dimTags[i][1]:
                            logger.warn(f"{child_dimTags[i]}:{child.placementName} -> {mapping}")
    logger.debug(f"Conforming children of {root_volume.placementName} complete!")
    for child in root_volume.children:
        conform_model(child)
