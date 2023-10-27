from setuptools import setup, find_packages, Extension
import subprocess
import os

libraries = ['boost_python','boost_numpy']
extra_objects = []

if 'VIRTUAL_ENV' in os.environ:
    # use local copy of boost libs
    boost_lib = os.path.join(os.environ['VIRTUAL_ENV'],'lib','libboost_python.so')
    if os.path.exists(boost_lib):
        extra_objects.append(boost_lib)
        libraries.remove('boost_python')
    boost_lib = os.path.join(os.environ['VIRTUAL_ENV'],'lib','libboost_numpy.so')
    if os.path.exists(boost_lib):
        extra_objects.append(boost_lib)
        libraries.remove('boost_numpy')

def check_output(*popenargs, **kwargs):
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd, output=output)
    return output.decode('utf-8') if type(output) != str else output

geant4_cflags = check_output(['geant4-config','--cflags']).split()
geant4_libs = check_output(['geant4-config','--libs']).split()
# For GEANT4.9.4 built without cmake
try:
    clhep_libs = check_output(['clhep-config','--libs']).split()
except OSError:
    clhep_libs = []


include_dirs=['src']

#####

if 'VIRTUAL_ENV' in os.environ:
    include_dirs.append(os.path.join(os.environ['VIRTUAL_ENV'], 'include'))
try:
    import numpy.distutils
    include_dirs += numpy.distutils.misc_util.get_numpy_include_dirs()
except:
    pass # if numpy doesn't exist yet

setup(
    name = 'Chroma',
    version = '0.5',
    packages = find_packages(),
    include_package_data=True,

    scripts = ['bin/chroma-sim', 'bin/chroma-cam',
               'bin/chroma-geo', 'bin/chroma-bvh',
               'bin/chroma-server'],
    ext_modules = [
        Extension('chroma.generator._g4chroma',
                  ['src/G4chroma.cc','src/GLG4Scint.cc'],
                  include_dirs=include_dirs,
                  extra_compile_args=['--std=c++11']+geant4_cflags,
                  extra_link_args=geant4_libs+clhep_libs,
                  extra_objects=extra_objects,
                  libraries=libraries,
                  ),
        Extension('chroma.generator.mute',
                  ['src/mute.cc'],
                  include_dirs=include_dirs,
                  extra_compile_args=geant4_cflags,
                  extra_link_args=geant4_libs+clhep_libs,
                  extra_objects=extra_objects,
                  libraries=libraries),
        ],
 
    setup_requires = [],
    install_requires = ['uncertainties','pyzmq', 'pycuda','pytools==2022.1.2',
                        'numpy>=1.6', 'pygame', 'nose', 'sphinx', 'particle', 'uproot'],
    #test_suite = 'nose.collector',
    
)
