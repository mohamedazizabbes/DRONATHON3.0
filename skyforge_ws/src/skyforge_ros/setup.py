from setuptools import setup

package_name = "skyforge_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SKYFORGE team",
    maintainer_email="team@skyforge.local",
    description="SKYFORGE ROS 2 nodes (thin wrappers around skyforge_core).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "orchestrator_node = skyforge_ros.orchestrator_node:main",
            "perception_agent_node = skyforge_ros.perception_agent_node:main",
            "reflex_node = skyforge_ros.reflex_node:main",
            "comm_node = skyforge_ros.comm_node:main",
            "health_node = skyforge_ros.monitor_nodes:health_main",
            "nav_integrity_node = skyforge_ros.monitor_nodes:nav_main",
            "metrics_node = skyforge_ros.monitor_nodes:metrics_main",
            "fault_injector = skyforge_ros.fault_injector:main",
            "vision_node = skyforge_ros.vision_node:main",
            "viz_node = skyforge_ros.viz_node:main",
            "obstacle_relay = skyforge_ros.obstacle_relay:main",
        ],
    },
)
