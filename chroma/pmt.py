import numpy as np
from chroma.geometry import Solid
from chroma.make import rotate_extrude
from chroma.tools import read_csv, offset

def get_lc_profile(radii, a, b, d, rmin, rmax):
    c = -b*np.sqrt(1 - (rmin-d)**2/a**2)
    return -c - b*np.sqrt(1-(radii-d)**2/a**2)

def build_light_collector(pmt, a, b, d, rmin, rmax, surface, npoints=10):
    if not isinstance(pmt, Solid):
        raise Exception('`pmt` must be an instance of %s' % Solid)

    lc_radii = np.linspace(rmin, rmax, npoints)
    lc_profile = get_lc_profile(lc_radii, a, b, d, rmin, rmax)

    pmt_face_profile = pmt.profile[pmt.profile[:,1] > -1e-3]

    lc_offset = np.interp(lc_radii[0], list(reversed(pmt_face_profile[:,0])), list(reversed(pmt_face_profile[:,1])))

    lc_mesh = rotate_extrude(lc_radii, lc_profile + lc_offset, pmt.nsteps)

    return Solid(lc_mesh, pmt.outer_material, pmt.outer_material, surface=surface)

def build_light_collector_from_file(filename, outer_material,
                                    surface, nsteps=48):
    profile = read_csv(filename)
    
    mesh = rotate_extrude(profile[:,0], profile[:,1], nsteps)
    solid = Solid(mesh, outer_material, outer_material, surface=surface)
    return solid

def build_pmt_shell_from_file(filename, outer_material, glass, nsteps=16):
    profile = read_csv(filename)

    # slice profile in half
    profile = profile[profile[:,0] < 0]
    profile[:,0] = -profile[:,0]
    # order profile from base to face
    profile = profile[np.argsort(profile[:,1])]
    # set x coordinate to 0.0 for first and last profile along the profile
    # so that the mesh is closed
    profile[0,0] = 0.0
    profile[-1,0] = 0.0
    return build_pmt_shell(profile, outer_material, glass, nsteps)

def build_pmt_shell(profile, outer_material, glass, nsteps=16):
    return Solid(rotate_extrude(profile[:,0], profile[:,1], nsteps), glass, outer_material, color=0xeeffffff)

def build_pmt_from_file(filename, glass_thickness, outer_material, glass,
              vacuum, photocathode_surface, back_surface, nsteps=16):
    profile = read_csv(filename)

    # slice profile in half
    profile = profile[profile[:,0] < 0]
    profile[:,0] = -profile[:,0]
    # order profile from base to face
    profile = profile[np.argsort(profile[:,1])]
    # set x coordinate to 0.0 for first and last profile along the profile
    # so that the mesh is closed
    profile[0,0] = 0.0
    profile[-1,0] = 0.0

    return build_pmt(profile, glass_thickness, outer_material, glass, vacuum, photocathode_surface, back_surface, nsteps)

def build_pmt(profile, glass_thickness, outer_material, glass,
              vacuum, photocathode_surface, back_surface, nsteps=16):

    offset_profile = offset(profile, -glass_thickness)

    outer_envelope_mesh = rotate_extrude(profile[:,0], profile[:,1], nsteps)
    inner_envelope_mesh = rotate_extrude(offset_profile[:,0], offset_profile[:,1], nsteps)

    outer_envelope = Solid(outer_envelope_mesh, glass, outer_material)
    photocathode = np.mean(inner_envelope_mesh.assemble(), axis=1)[:,1] > 0
    inner_envelope = Solid(inner_envelope_mesh, vacuum, glass, surface=np.where(photocathode, photocathode_surface, back_surface), color=np.where(photocathode, 0xff00, 0xff0000))

    pmt = outer_envelope + inner_envelope

    # profile points, outer_material, and theta are used to construct the
    # light collector
    pmt.profile = profile
    pmt.outer_material = outer_material
    pmt.nsteps = nsteps

    return pmt
