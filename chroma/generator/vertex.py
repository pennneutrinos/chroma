import numpy as np
from itertools import count

from chroma.pi0 import pi0_decay
from chroma import event
from chroma.sample import uniform_sphere
from chroma.itertoolset import repeatfunc
from chroma.transform import norm

# generator parts for use with gun()

def from_histogram(h):
    "Yield values drawn randomly from the histogram `h` interpreted as a pdf."
    pdf = h.hist/h.hist.sum()
    cdf = np.cumsum(pdf)

    for x in repeatfunc(np.random.random_sample):
        yield h.bincenters[np.searchsorted(cdf, x)]

def constant(obj):
    while True:
        yield obj

def isotropic():
    while True:
        yield uniform_sphere()
        
def line_segment(point1, point2):
    while True:
        frac = np.random.uniform(0.0, 1.0)
        yield frac * point1 + (1.0 - frac) * point2

def fill_shell(center, radius):
    for direction in isotropic():
        r = radius * np.random.uniform(0.0, 1.0)**(1.0/3.0)
        yield r * direction

def flat(e_lo, e_hi):
    while True:
        yield np.random.uniform(e_lo, e_hi)

# vertex generators

def particle_gun(particle_name_iter, pos_iter, dir_iter, ke_iter, 
                 t0_iter=constant(0.0), start_id=0):
    for i, particle_name, pos, dir, ke, t0 in zip(count(start_id), particle_name_iter, pos_iter, dir_iter, ke_iter, t0_iter):
        dir = dir/norm(dir)
        print("dir", dir)
        vertex = event.Vertex(particle_name, pos, dir, ke, t0=t0)
        print("vertex", vertex)
        ev_vertex = event.Event(i, vertex, [vertex])
        print("ev_vertex", ev_vertex)
        yield ev_vertex

def pi0_gun(pos_iter, dir_iter, ke_iter, t0_iter=constant(0.0), start_id=0, gamma1_dir_iter=None):

    if gamma1_dir_iter is None:
        gamma1_dir_iter = isotropic()

    for i, pos, dir, ke, t0, gamma1_dir in zip(count(start_id), pos_iter, dir_iter, ke_iter, t0_iter, gamma1_dir_iter):
        dir = dir/norm(dir)
        primary_vertex = event.Vertex('pi0', pos, dir, ke, t0=t0)

        # In rest frame
        theta_rest = np.arccos(gamma1_dir[2])
        phi_rest = np.arctan2(gamma1_dir[1], gamma1_dir[0])

        # In lab frame
        (gamma1_e, gamma1_dir), (gamma2_e, gamma2_dir) = \
            pi0_decay(ke+134.9766, dir, theta_rest, phi_rest)

        gamma1_vertex = event.Vertex('gamma', pos, gamma1_dir, gamma1_e, t0=t0)
        gamma2_vertex = event.Vertex('gamma', pos, gamma2_dir, gamma2_e, t0=t0)

        ev_vertex = event.Event(i, primary_vertex, [gamma1_vertex, gamma2_vertex])

        yield ev_vertex


def constant_particle_gun(particle_name, pos, dir, ke, t0=0.0, start_id=0):
    '''Convenience wrapper around particle gun that assumes all
    arguments are constants, rather than generators.'''
    pos = np.asarray(pos)
    dir = np.asarray(dir)
    if (dir == 0.0).all():
        dir_gen = isotropic()
    else:
        dir_gen = constant(dir)
    
    if particle_name == 'pi0':
        return pi0_gun(constant(pos), dir_gen, constant(ke),
                       constant(t0), start_id=start_id)
    else:
        return particle_gun(constant(particle_name), constant(pos), dir_gen, constant(ke), constant(t0), start_id=start_id)
