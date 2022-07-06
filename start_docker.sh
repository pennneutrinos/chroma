docker run -it --gpus all\
  -v /media/linux_store/hep/container_home:/root \
  -v /media/linux_store/hep:/media/linux_store/hep \
  -v /media/linux_store/hep/chroma/chroma:/opt/chroma/chroma \
  -v /tmp/.X11-unix:/tmp/.X11-unix\
  --network host \
  --workdir='/media/linux_store/hep/' \
  stjimmys/chroma3:nvidia.cuda10
