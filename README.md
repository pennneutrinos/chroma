# Chroma: Ultra-fast Photon Monte Carlo
Chroma is a high performance optical photon simulation for particle physics
detectors originally written by A. LaTorre and S. Seibert. It tracks individual
photons passing through a triangle-mesh detector geometry, simulating standard
physics processes like diffuse and specular reflections, refraction, Rayleigh
scattering and absorption.

With the assistance of a CUDA-enabled GPU, Chroma can propagate 2.5 million
photons per second in a detector with 29,000 photomultiplier tubes. This is
200x faster than the same simulation with GEANT4.

Check out the [Chroma whitepaper](doc/source/chroma.pdf) for information on how
Chroma works.

Information about the historical development of Chroma can be found at the
[bitbucket repository](https://chroma.bitbucket.io/index.html) this repository
was forked from.

This repository contains a modified version of Chroma that includes support for
wavelength shifters, scintillators, and dichroic filters.  This version of
Chroma has also been been updated for Python3, Geant4.10.05.p01, and Root
6.18.04, among other things.

## Chroma -> Ratpac-Two
Changes to base chroma are required to transition to Geant4.11+ and to make chroma
compatible with the ratpac-two library.
- Removed boost dependency using full pybind11
- Deprecated Geant4 python library replaced with geant4-pybind

## Container overview

The `installation` directory contains a collection of `Dockerfile`s to build an
ubuntu-derived image containing Chroma. This may be a useful reference for
other systems. Note that properly linking to boost_python and boost_numpy is
nontrivial on systems with both python2 and python3.

Containers are great for packaging dependencies, but CUDA throws an additional
constraint that the NVIDIA driver version inside the container must match the
host NVIDIA driver version that created NVIDIA device nodes. This is because
the device nodes created by the host are passed directly to the container.
Images will need to be built for each possible driver version, or the
`nvidia-docker` project https://github.com/NVIDIA/nvidia-docker must be
utilized to automatically synchronize the driver version. Singularity also
supports NVIDIA GPUs in a very graceful way, and has other nice features.

The images pushed to DockerHub are built from the subdirectories in the
`installation` directory. `installation/chroma3.deps` was used to build the
image `benland100/chroma3:deps` by running `docker build -t
benland100/chroma3:deps` and contains all dependencies for chroma except
nvidia-drivers, and can be used to build images for particular versions.
`installation/chroma.latest` builds an image using the default version of
NVIDIA drivers for Ubuntu-20.04 and is pushed to `benland100/chroma3:latest`.
The remaining directories create the images `benland100/chroma3:440` and
`benland100/chroma3:435` for other common versions of nvidia-drivers. If you
need another version for your host machine, you will have to create an
analogous `Dockerfile`. Finally the `benland100/chroma3:nvidia` image is built
from `chroma3.nvidia` which is derived from
`nvidia/cudagl:9.2-devel-ubuntu18.04` and features full CUDA and OpenGL support
with `nvidia-docker`.

To get a prebuilt image, run `docker pull benland100/chroma3:[tag]` where tag
identifies the image you want. 

## Docker usage

Connecting the container to the GPU requires either the `nvidia-docker` runtime
or passing the GPU device nodes manually to the container. See subsections for
details.

In general, the containers can be run with `docker run -it
benland100/chroma3:[tag]`, but to do something useful, you should mount some
host directory containing your analysis code and/or data storage to the
container with additional `-v host_path:container_path` flags, and work within
those directories. Paths not mounted from the host will not be saved when the
container exits. 

Consider adding `--net=host` to your run command and running `jupyter` within
the container. The default Python3 environment is setup for Chroma.

For running visualizations, you will need to allow the container to access your
X11 server. The easiest way to accomplish this is by adding these flags
`--net=host -v $HOME/.Xauthority:/root/.Xauthority:rw -e DISPLAY=$DISPLAY` to
the docker run command.

### With nvidia-docker

This tool must be installed on the host, and adds the `nvidia-docker` command,
which modifies the container on the fly to synchronize the NVIDIA drivers in
the container with the host. If it is available, it provides full CUDA and
OpenGL functionality for simulation and rendering. The
`benland100/chroma3:nvidia` image is derived from a base that supports this
functionality.

On my machine, the minimal docker command to launch a shell is:

`nvidia-docker run -it benland100/chroma3:nvidia`

### Without nvidia-docker

To use CUDA within a container, the host's NVIDIA device nodes must be passed
to the container. This will not enable OpenGL functionality, but is sufficient
for running Chroma on machines where `nvidia-docker` is unavailable. To see the
required device nodes run `grep /dev/*nvidia*`. Each must be passed to docker
with the `--device` flag as shown with `for dev in /dev/*nvidia*; do echo
--device $dev:$dev; done`

On my machine, this results in a very concise minimal docker run command:

`docker run --device /dev/nvidia-modeset:/dev/nvidia-modeset --device
/dev/nvidia-uvm:/dev/nvidia-uvm --device
/dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools --device /dev/nvidia0:/dev/nvidia0
--device /dev/nvidiactl:/dev/nvidiactl -it benland100/chroma3:440`

## Singularity usage

Singularity's `--nv` flag for detecting and synchronizing NVIDIA GPUs and
drivers within Singularity containers is very attractive. Singularity is more
likely to be supported on large compute clusters, as it does not require root
access. It also provides nice features like mounting your home directory in the
container and synchronizing aspects of the environment automatically.
Fortunately, Singularity can make use of the Docker containers described above.

A Singularity image may be derived from any Chroma Docker image with a simple
`Singularity` file as found in `installation/chroma3.nvidia/Singularity`.

The `:nvidia` tagged image used here is likely the best choice, as
Singularity's `--nv` flag is designed for the base it was derived from. 

Singularity can then be used to build an image: `sudo singularity build
chroma3.simg Singularity`

Running this image is pretty setraightforward. Your home directory will be
available within the image, but other directories can be mounted as desired.

`singularity run --nv chroma3.simg`

Visualization with OpenGL and simulation with CUDA will work in this container.

### Test drive

After deploying a container to a GPU-enabled host locally or via SSH with
XForwarding enabled, you should be able to run the container and execute 

`chroma-cam @chroma.models.lionsolid`

which should display a GPU-rendered visualization, ensuring everything is
working properly.
