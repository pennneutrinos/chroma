import numpy as np
from scipy.spatial.transform import Rotation

def get_perp(x):
    """Returns an arbitrary vector perpendicular to `x`."""
    a = np.zeros(3)
    a[np.argmin(abs(x))] = 1
    return np.cross(a,x)

def make_rotation_matrix(phi, n):
    """
    Make the rotation matrix to rotate points through an angle `phi`
    counter-clockwise around the axis `n` (when looking towards +infinity).

    Source: Weissten, Eric W. "Rotation Formula." Mathworld.

    Raises ValueError if n has zero magnitude
    """
    n = normalize(n)

    return np.cos(phi)*np.identity(3) + (1-np.cos(phi))*np.outer(n,n) + \
        np.sin(phi)*np.array([[0,n[2],-n[1]],[-n[2],0,n[0]],[n[1],-n[0],0]])


def matrix_to_rotvec(rot_matrix):
    r = Rotation.from_matrix(rot_matrix)
    rotvec = r.as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle == 0:
        return np.array([0, 0, 1]), 0
    return rotvec/angle, angle


def rotate(x, phi, n):
    """
    Rotate an array of points `x` through an angle phi counter-clockwise
    around the axis `n` (when looking towards +infinity).
    """
    n = normalize(n)
    x = np.atleast_2d(x)
    phi = np.atleast_1d(phi)

    return (x*np.cos(phi)[:,np.newaxis] + n*np.dot(x,n)[:,np.newaxis]*(1-np.cos(phi)[:,np.newaxis]) + np.cross(x,n)*np.sin(phi)[:,np.newaxis]).squeeze()

def rotate_matrix(x, phi, n):
    """Same as rotate() except uses a rotation matrix.

    Can only handle a single rotation angle `phi`."""
    rotation_matrix = make_rotation_matrix(phi, n)
    return np.inner(np.asarray(x),rotation_matrix)

def norm(x):
    "Returns the norm of the vector `x`."
    return np.sqrt((x*x).sum(-1))

def normalize(x):
    "Returns unit vectors in the direction of `x`."
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return (x/norm(x)[:,np.newaxis]).squeeze()
