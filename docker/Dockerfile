# SKYFORGE onboard stack on ROS 2 Jazzy (Ubuntu 24.04, Python 3.12).
#   docker compose -f docker/compose.yaml build
# The PX4 SITL + Gazebo container is owned by Sim & Control (see compose.yaml).
FROM ros:jazzy-ros-base

ARG PX4_MSGS_BRANCH=release/1.15
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3-pip python3-numpy python3-yaml python3-psutil python3-matplotlib python3-pytest \
        ros-jazzy-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*
# Ubuntu 24.04 marks the system Python as externally managed
RUN pip3 install --no-cache-dir --break-system-packages "langgraph>=0.2"

WORKDIR /ws
# px4_msgs must match the PX4 firmware version used in SITL
RUN mkdir -p src && git clone --depth 1 -b ${PX4_MSGS_BRANCH} https://github.com/PX4/px4_msgs.git src/px4_msgs
RUN . /opt/ros/jazzy/setup.sh && colcon build --packages-select px4_msgs --cmake-args -DCMAKE_BUILD_TYPE=Release

COPY skyforge_ws/src/ src/
RUN . /opt/ros/jazzy/setup.sh && . install/setup.sh \
    && colcon build --packages-skip px4_msgs --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && . install/setup.sh \
    && python3 -m pytest -q src/skyforge_core/tests -k "not short_sim" -p no:cacheprovider

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "skyforge_bringup", "mission.launch.py"]
