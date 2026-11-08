import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("skyforge_sim")
    bringup = get_package_share_directory("skyforge_bringup")
    models = os.path.join(share, "models")
    world = LaunchConfiguration("world")
    args = [
        DeclareLaunchArgument("world", default_value=os.path.join(share, "worlds", "coastal.sdf")),
        DeclareLaunchArgument("config", default_value=os.path.join(share, "config", "skyforge_gazebo.yaml")),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("stack", default_value="true"),
        DeclareLaunchArgument("jury_model", default_value=""),
        DeclareLaunchArgument("team_model", default_value=""),
        DeclareLaunchArgument("fault_script", default_value=""),
    ]
    env = [SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", models + os.pathsep + os.environ.get("GZ_SIM_RESOURCE_PATH", ""))]
    gz_server = ExecuteProcess(cmd=["gz", "sim", "-s", "-r", "--headless-rendering", world], output="screen")
    gz_gui = ExecuteProcess(cmd=["gz", "sim", "-g"], output="screen", condition=IfCondition(LaunchConfiguration("gui")))
    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", name="gz_bridge", output="screen",
                  parameters=[{"config_file": os.path.join(share, "config", "bridge.yaml"), "use_sim_time": True}])
    mover = Node(package="skyforge_sim", executable="mover_node", output="screen", parameters=[{"use_sim_time": True}])
    shim = Node(package="skyforge_sim", executable="heading_shim", output="screen", parameters=[{"use_sim_time": True}])
    stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup, "launch", "mission.launch.py")),
        launch_arguments={"backend": "twist", "use_sim_time": "true", "config": LaunchConfiguration("config"),
                          "jury_model": LaunchConfiguration("jury_model"), "team_model": LaunchConfiguration("team_model"),
                          "fault_script": LaunchConfiguration("fault_script")}.items(),
        condition=IfCondition(LaunchConfiguration("stack")),
    )
    return LaunchDescription(args + env + [gz_server, gz_gui, bridge, mover, shim, stack])
