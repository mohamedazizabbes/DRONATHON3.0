import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    share = get_package_share_directory("skyforge_sim")
    bringup = get_package_share_directory("skyforge_bringup")
    seed = LaunchConfiguration("seed").perform(context).strip()
    world = os.path.join(share, "worlds", "coastal.sdf")
    config = os.path.join(share, "config", "skyforge_gazebo.yaml")
    scenario = ""
    external = LaunchConfiguration("world").perform(context).strip()
    if external:
        world = external
    elif seed:
        import json
        from skyforge_core.params import DEFAULTS
        from skyforge_sim import make_world as mw
        out = os.path.join("/tmp", f"skyforge_world_{seed}")
        os.makedirs(out, exist_ok=True)
        sc, vessels = mw.random_scenario(int(seed), DEFAULTS)
        sc = dict(sc)
        sc["hostile_vessels"] = vessels
        world = os.path.join(out, "coastal.sdf")
        config = os.path.join(out, "skyforge_gazebo.yaml")
        scenario = os.path.join(out, "scenario.json")
        open(world, "w").write(mw.build(sc, DEFAULTS))
        mw.write_config(sc, os.path.join(bringup, "config", "skyforge.yaml"), config)
        open(scenario, "w").write(json.dumps(sc))
    extra = []
    if external:
        wname = LaunchConfiguration("world_name").perform(context).strip()
        extra.append(Node(package="ros_gz_sim", executable="create", output="screen",
                          arguments=["-world", wname, "-file", os.path.join(share, "models", "skyforge_drone", "model.sdf"),
                                     "-name", "skyforge", "-x", "0", "-y", "0", "-z", "0.3"]))
    gz_server = ExecuteProcess(cmd=["gz", "sim", "-s", "-r", "--headless-rendering", world], output="screen")
    gz_gui = ExecuteProcess(cmd=["gz", "sim", "-g", "--gui-config", os.path.join(share, "config", "gui.config")],
                            output="screen", condition=IfCondition(LaunchConfiguration("gui")))
    follow = TimerAction(period=12.0, condition=IfCondition(LaunchConfiguration("gui")), actions=[
        ExecuteProcess(cmd=["gz", "service", "-s", "/gui/follow/offset", "--reqtype", "gz.msgs.Vector3d", "--reptype", "gz.msgs.Boolean",
                            "--timeout", "3000", "--req", "x: -14, y: -6, z: 7"], output="screen"),
        ExecuteProcess(cmd=["gz", "service", "-s", "/gui/follow", "--reqtype", "gz.msgs.StringMsg", "--reptype", "gz.msgs.Boolean",
                            "--timeout", "3000", "--req", 'data: "skyforge"'], output="screen"),
    ])
    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", name="gz_bridge", output="screen",
                  parameters=[{"config_file": os.path.join(share, "config", "bridge.yaml"), "use_sim_time": True}])
    mover = Node(package="skyforge_sim", executable="mover_node", output="screen",
                 parameters=[{"use_sim_time": True, "scenario_file": scenario}])
    weather = Node(package="skyforge_sim", executable="weather_node", output="screen",
                   parameters=[{"use_sim_time": True, "scenario_file": scenario}])
    payload = Node(package="skyforge_sim", executable="payload_node", output="screen", parameters=[{"use_sim_time": True}])
    report = Node(package="skyforge_sim", executable="report_node", output="screen", parameters=[{"use_sim_time": True, "out_dir": "runs"}])
    shim = Node(package="skyforge_sim", executable="heading_shim", output="screen", parameters=[{"use_sim_time": True}])
    hud = Node(package="skyforge_sim", executable="hud_node", output="screen", condition=IfCondition(LaunchConfiguration("hud")))
    stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup, "launch", "mission.launch.py")),
        launch_arguments={"backend": "twist", "use_sim_time": "true", "config": config,
                          "jury_model": LaunchConfiguration("jury_model"), "team_model": LaunchConfiguration("team_model"),
                          "fault_script": LaunchConfiguration("fault_script")}.items(),
        condition=IfCondition(LaunchConfiguration("stack")),
    )
    return [gz_server, gz_gui, follow, bridge] + ([] if external else [mover]) + [weather, shim, payload, report, hud, stack] + extra


def generate_launch_description():
    share = get_package_share_directory("skyforge_sim")
    args = [
        DeclareLaunchArgument("seed", default_value=""),
        DeclareLaunchArgument("world", default_value=""),
        DeclareLaunchArgument("world_name", default_value="default"),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("hud", default_value="true"),
        DeclareLaunchArgument("stack", default_value="true"),
        DeclareLaunchArgument("jury_model", default_value=""),
        DeclareLaunchArgument("team_model", default_value=""),
        DeclareLaunchArgument("fault_script", default_value=""),
    ]
    env = [SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.join(share, "models") + os.pathsep + os.environ.get("GZ_SIM_RESOURCE_PATH", ""))]
    return LaunchDescription(args + env + [OpaqueFunction(function=setup)])
