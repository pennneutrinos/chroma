sudo cp $XAUTHORITY /media/linux_store/hep/container_home/.Xauthority
docker run -it --rm --gpus all\
  -v /media/linux_store/hep/container_home:/root \
  -v /media/linux_store/hep:/media/linux_store/hep \
  -v /media/linux_store/hep/chroma/chroma:/opt/chroma/chroma \
  -v /tmp/.X11-unix:/tmp/.X11-unix\
  --env="DISPLAY"\
  --network host \
  --workdir='/media/linux_store/hep/' \
  stjimmys/chroma3:nvidia.cuda10
