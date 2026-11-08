"""Full SKYFORGE onboard stack (everything except the simulator itself).

  ros2 launch skyforge_bringup mission.launch.py                       # PX4 SITL backend
  ros2 launch skyforge_bringup mission.launch.py backend:=twist        # velocity-controlled Gazebo fallback
  ros2 launch skyforge_bringup mission.launch.py comm_mode:=pi pi_host:=192.168.1.50
  ros2 launch skyforge_bringup mission.launch.py team_model:=/models/team.onnx jury_model:=/models/jury.onnx
  ros2 launch skyforge_bringup mission.launch.py fault_script:=$(ros2 pkg prefix skyforge_bringup)/share/skyforge_bringup/config/faults_demo.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("skyforge_bringup")
    cfg = LaunchConfiguration("config")
    common = {"config_file": cfg}

    args = [
        DeclareLaunchArgument("config", default_value=os.path.join(share, "config", "skyforge.yaml")),
        DeclareLaunchArgument("backend", default_value="px4", description="px4 | twist"),
        DeclareLaunchArgument("comm_mode", default_value="local", description="local | pi"),
        DeclareLaunchArgument("pi_host", default_value="192.168.1.50"),
        DeclareLaunchArgument("gnss_source", default_value="px4", description="px4 | navsatfix"),
        DeclareLaunchArgument("jury_model", default_value=""),
        DeclareLaunchArgument("team_model", default_value=""),
        DeclareLaunchArgument("vision", default_value="true", description="start the two vision_node wrappers"),
        DeclareLaunchArgument("fault_script", default_value=""),
        DeclareLaunchArgument("viz", default_value="true", description="publish /viz/markers for Foxglove"),
        DeclareLaunchArgument("resume_snapshot", default_value="",
                              description="drone 2 after a MAYDAY: path to the station's mayday_snapshot.json"),
        DeclareLaunchArgument("use_sim_time", default_value="false",
                              description="true only if /clock is bridged from Gazebo, else all timers freeze"),
    ]
    sim_time = {"use_sim_time": PythonExpression(["'", LaunchConfiguration("use_sim_time"), "' == 'true'"])}

    def node(executable, name=None, extra=None, **kw):
        return Node(package="skyforge_ros", executable=executable, name=name or executable, output="screen",
                    parameters=[common, sim_time, extra or {}], **kw)

    nodes = [
        node("reflex_node", extra={"backend": LaunchConfiguration("backend")}),
        node("orchestrator_node", extra={"resume_snapshot": LaunchConfiguration("resume_snapshot")}),
        node("perception_agent_node", extra={"resume_snapshot": LaunchConfiguration("resume_snapshot")}),
        node("comm_node", extra={"mode": LaunchConfiguration("comm_mode"), "pi_host": LaunchConfiguration("pi_host")}),
        node("health_node"),
        node("nav_integrity_node"),
        node("metrics_node"),
        node("viz_node", condition=IfCondition(LaunchConfiguration("viz"))),
        node("fault_injector", extra={"gnss_source": LaunchConfiguration("gnss_source"),
                                      "script": LaunchConfiguration("fault_script")}),
        node("vision_node", name="jury_vision_node", condition=IfCondition(LaunchConfiguration("vision")),
             extra={"role": "jury", "model": LaunchConfiguration("jury_model")}),
        node("vision_node", name="team_vision_node", condition=IfCondition(LaunchConfiguration("vision")),
             extra={"role": "team", "model": LaunchConfiguration("team_model"), "thermal_topic": "/cam/thermal"}),
    ]
    return LaunchDescription(args + nodes)
