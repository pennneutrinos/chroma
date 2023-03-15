from os.path import dirname

from chroma.pmt import build_pmt_from_file, build_light_collector_from_file

from chroma.demo.optics import water, glass, vacuum, shiny_surface, r7081hqe_photocathode

def build_8inch_pmt(outer_material=water, nsteps=24):
    return build_pmt_from_file(dirname(__file__) + '/sno_pmt.txt', 3.0, # 3 mm
                     outer_material=outer_material,
                     glass=glass, vacuum=vacuum,
                     photocathode_surface=r7081hqe_photocathode,
                     back_surface=shiny_surface,
                     nsteps=nsteps)

def build_8inch_pmt_with_lc(outer_material=water, nsteps=24):
    pmt = build_8inch_pmt(outer_material, nsteps)
    lc = build_light_collector_from_file(dirname(__file__) + '/sno_cone.txt',
                                         outer_material=outer_material,
                                         surface=shiny_surface, nsteps=nsteps)
    return pmt + lc
