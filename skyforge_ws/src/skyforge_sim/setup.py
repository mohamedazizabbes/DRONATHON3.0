import os
from glob import glob

from setuptools import setup

pkg = "skyforge_sim"


def tree(src):
    out = {}
    for root, _, files in os.walk(src):
        if files:
            out.setdefault(os.path.join("share", pkg, root), []).extend(os.path.join(root, f) for f in files)
    return list(out.items())


setup(
    name=pkg,
    version="0.1.0",
    packages=[pkg],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{pkg}"]),
        (f"share/{pkg}", ["package.xml"]),
        (f"share/{pkg}/launch", glob("launch/*.py")),
        (f"share/{pkg}/config", glob("config/*")),
        (f"share/{pkg}/worlds", glob("worlds/*")),
    ] + tree("models"),
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": ["mover_node = skyforge_sim.mover_node:main", "make_world = skyforge_sim.make_world:main"]},
)
