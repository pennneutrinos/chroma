#!/bin/bash

cd $HEP
sudo -E APPTAINER_TMPDIR=/media/linux_store/tmp  singularity build --sandbox chroma3.simg $HEP/chroma/installation/Singularity
