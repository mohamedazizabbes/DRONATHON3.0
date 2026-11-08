FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg lsb-release ca-certificates \
 && curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable noble main" > /etc/apt/sources.list.d/gazebo-stable.list \
 && apt-get update && apt-get install -y --no-install-recommends gz-harmonic python3-gz-transport13 python3-gz-msgs10 python3-numpy python3-opencv mesa-utils libegl1 libgl1-mesa-dri xvfb \
 && rm -rf /var/lib/apt/lists/*
ENV GZ_SIM_RESOURCE_PATH=/work/models
WORKDIR /work
