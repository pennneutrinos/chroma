#!/bin/bash

singularity run --nv --contain --cleanenv\
  -B "/media/linux_store/hep/geant4_data/data/:/opt/geant4/share/Geant4-10.5.1/data,\
  /media/linux_store/hep/container_home/:/home/james,\
  /media/linux_store/hep/:/media/linux_store/hep" \
  /media/linux_store/hep/chroma3.simg
