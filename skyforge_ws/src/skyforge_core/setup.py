from setuptools import find_packages, setup

package_name = "skyforge_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "pyyaml", "langgraph"],
    zip_safe=True,
    maintainer="SKYFORGE team",
    maintainer_email="team@skyforge.local",
    description="SKYFORGE pure-Python core: LangGraph orchestrator, PerceptionAgent, "
    "PlannerAgent libs, reflex controller, comm protocol, monitors, 2D mission sim.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "skyforge_sim = skyforge_core.sim.run:main",
            "skyforge_station = skyforge_core.comm.station_server:main",
        ],
    },
)
