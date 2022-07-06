import numpy as np
from chroma.geometry import Mesh
from chroma.transform import rotate
from chroma.itertoolset import *

def mesh_grid(grid):
    begin = grid[:-1].flatten()
    end = grid[1:].flatten()
    begin_roll = np.roll(grid[:-1],-1,1).flatten()
    end_roll = np.roll(grid[1:],-1,1).flatten()
    
    mesh = np.empty(shape=(2*len(begin),3), dtype=begin.dtype)
    mesh[:len(begin),0] = begin
    mesh[:len(begin),1] = end
    mesh[:len(begin),2] = end_roll
    mesh[len(begin):,0] = begin
    mesh[len(begin):,1] = end_roll
    mesh[len(begin):,2] = begin_roll

    return mesh

def linear_extrude(x1, y1, height, x2=None, y2=None, center=None, endcaps=True):
    """
    Return the solid mesh formed by linearly extruding the polygon formed by
    the x and y points `x1` and `y1` by a distance `height`. If `x2` and `y2`
    are given extrude by connecting the points `x1` and `y1` to `x2` and `y2`;
    this allows the creation of tapered solids.  If `endcaps` is False, then
    the triangles on the endcaps will be left off, and the mesh will not be
    closed.

    .. note::
        The path traced by the points `x` and `y` should go counter-clockwise,
        otherwise the mesh will be inside out.

    Example:
        >>> # create a hexagon prism
        >>> angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        >>> m = linear_extrude(np.cos(angles), np.sin(angles), 2.0)
    """
    if len(x1) != len(y1):
        raise Exception('`x` and `y` arrays must have the same length.')

    if x2 is None:
        x2 = x1

    if y2 is None:
        y2 = y1

    if len(x2) != len(y2) or len(x2) != len(x1):
        raise Exception('`x` and `y` arrays must have the same length.')

    n = len(x1)

    vertex_iterators = [zip(x1,y1,repeat(-height/2.0,n)),
                        zip(x2,y2,repeat(height/2.0,n))]
    if endcaps:
        vertex_iterators = [zip(repeat(0,n),repeat(0,n),repeat(-height/2.0,n))] \
            + vertex_iterators \
            + [zip(repeat(0,n),repeat(0,n),repeat(height/2.0,n))]

    vertices = np.fromiter(flatten(roundrobin(*vertex_iterators)), float)
    vertices = vertices.reshape((len(vertices)//3,3))

    if center is not None:
        vertices += center

    triangles = mesh_grid(np.arange(len(vertices)).reshape((len(x1),len(vertices)//len(x1))).transpose()[::-1])

    return Mesh(vertices, triangles, remove_duplicate_vertices=True)

def rotate_extrude(x, y, nsteps=64):
    """
    Return the solid mesh formed by extruding the profile defined by the x and
    y points `x` and `y` around the y axis.

    .. note::
        The path traced by the points `x` and `y` should go counter-clockwise,
        otherwise the mesh will be inside out.

    Example:
        >>> # create a bipyramid
        >>> m = rotate_extrude([0,1,0], [-1,0,1], nsteps=4)
    """
    if len(x) != len(y):
        raise Exception('`x` and `y` arrays must have the same length.')

    points = np.array([x,y,np.zeros(len(x))]).transpose()

    steps = np.linspace(0, 2*np.pi, nsteps, endpoint=False)
    vertices = np.vstack([rotate(points,angle,(0,-1,0)) for angle in steps])
    triangles = mesh_grid(np.arange(len(vertices)).reshape((len(steps),len(points))).transpose()[::-1])

    return Mesh(vertices, triangles, remove_duplicate_vertices=True)

def box(dx, dy, dz, center=(0,0,0)):
    "Return a box with linear dimensions `dx`, `dy`, and `dz`."
    return linear_extrude([-dx/2.0,dx/2.0,dx/2.0,-dx/2.0],[-dy/2.0,-dy/2.0,dy/2.0,dy/2.0],height=dz,center=center)

def cube(size, height=None, center=(0,0,0)):
    "Return a cube mesh whose sides have length `size`."
    if height is None:
        height = size

    return linear_extrude([-size/2.0,size/2.0,size/2.0,-size/2.0],[-size/2.0,-size/2.0,size/2.0,size/2.0], height=size, center=center)

def cylinder_along_z(radius, height, points=100):
    angles = np.linspace(0, 2*np.pi, points, endpoint=False)
    return linear_extrude(radius*np.cos(angles), radius*np.sin(angles), height)

def cylinder(radius, height, radius2=None, nsteps=64):
    """
    Return a cylinder mesh with a radius of length `radius`, and a height of
    length `height`. If `radius2` is specified, return a cone shaped cylinder
    with bottom radius `radius`, and top radius `radius2`.
    """
    if radius2 is None:
        radius2 = radius

    return rotate_extrude([0,radius,radius2,0], [-height/2.0, -height/2.0, height/2.0, height/2.0], nsteps)

def segmented_cylinder(radius, height, nsteps=64, nsegments=100):
    """
    Return a cylinder mesh segmented into `nsegments` points along its profile.
    """
    nsegments_radius = int((nsegments*radius/(2*radius+height))/2)
    nsegments_height = int((nsegments*height/(2*radius+height))/2)
    x = np.concatenate([np.linspace(0,radius,nsegments_radius,endpoint=False),[radius]*nsegments_height,np.linspace(radius,0,nsegments_radius,endpoint=False),[0]])
    y = np.concatenate([[-height/2.0]*nsegments_radius,np.linspace(-height/2.0,height/2.0,nsegments_height,endpoint=False),[height/2.0]*(nsegments_radius+1)])
    return rotate_extrude(x, y, nsteps)

def sphere(radius, nsteps=64):
    "Return a sphere mesh."
    profile_angles = np.linspace(-np.pi/2, np.pi/2, nsteps)
    return rotate_extrude(radius*np.cos(profile_angles), radius*np.sin(profile_angles), nsteps)

def torus(radius, offset, nsteps=64, circle_steps=None):
    """
    Return a torus mesh. `offset` is the distance from the center of the torus
    to the center of the torus barrel. `radius` is the radius of the torus
    barrel. `nsteps` is the number of steps in the rotational extrusion of the
    circle. `circle_steps` if specified is the number of steps around the
    circumference of the torus barrel, else it defaults to `nsteps`.
    """
    if circle_steps is None:
        circle_steps = nsteps
    profile_angles = np.linspace(0, 2*np.pi, circle_steps)
    return rotate_extrude(radius * np.cos(profile_angles) + offset, radius * np.sin(profile_angles), nsteps)

def convex_polygon(x, y):
    """
    Return a polygon mesh in the x-y plane. `x` and `y` are the x and
    y coordinates for the points in the polygon.  The simple triangulation
    method used here requires that the polygon be convex and the points
    are specified in order.
    """
    vertices = np.column_stack( (x, y, np.zeros_like(x)) )
    # Every triangle includes 
    triangles = np.empty(shape=(len(vertices)-2,3), dtype=np.int32)
    triangles[:,0] = 0 # Every triangle includes vertex zero
    triangles[:,1] = np.arange(1, len(vertices)-1)
    triangles[:,2] = np.arange(2, len(vertices))
    
    return Mesh(vertices=vertices, triangles=triangles)
