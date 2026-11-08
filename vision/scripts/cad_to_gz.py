import argparse
import glob
from pathlib import Path

import numpy as np
import trimesh

AIRFRAME = ["body-1", "mh32_wing_panel-1", "wing_right_mh32-1", "mh32_wingtip_cap", "wingtip_cap_right", "hstab_fixed", "vstab_fixed"]
MOTOR_KEEP = ["Bell", "Base", "UMAC 2807 1300KV.step-1.STL"]
CENTER = np.array([600.0, 100.0, 1671.5])
TO_GZ = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def to_gz(mesh):
    m = mesh.copy()
    m.vertices = ((m.vertices - CENTER) @ TO_GZ.T) / 1000.0
    return m


def simplify(mesh, target):
    if len(mesh.faces) <= target:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target)
    except Exception:
        return mesh.simplify_quadric_decimation(1.0 - target / len(mesh.faces))


def load(path):
    return trimesh.load(path, force="mesh")


PARTS = {
    "fuselage": ["body-1"],
    "wings": ["mh32_wing_panel-1", "wing_right_mh32-1"],
    "tips": ["mh32_wingtip_cap", "wingtip_cap_right"],
    "tail": ["hstab_fixed", "vstab_fixed"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    a = ap.parse_args()
    src = Path(a.src)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(src / "Assembly" / "*.STL")))
    for part, keys in PARTS.items():
        ms = [load(f) for f in files if any(k in Path(f).name for k in keys)]
        g = to_gz(simplify(trimesh.util.concatenate(ms), 12000))
        g.export(out / f"{part}.stl")
        print(part, len(g.faces))
    motors = [load(f) for f in files if "UMAC" in f and "Wire" not in f and "Magnets" not in f and "Stator" not in f]
    g = to_gz(simplify(trimesh.util.concatenate(motors), 9000))
    g.export(out / "motors.stl")
    print("motors", len(g.faces))
    prop = load(src / "Propeller.STL")
    prop.vertices -= np.array([26.4, 7.15, 193.15])
    prop = simplify(prop, 3000)
    rot_h = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    info = []
    for f in files:
        if "Bell" not in Path(f).name:
            continue
        inst = Path(f).name.split(".step-")[1][0]
        lo, hi = load(f).bounds
        c = (lo + hi) / 2
        axis = int(np.argmin(hi - lo))
        p = prop.copy()
        if axis == 1:
            hub = np.array([c[0], hi[1] + 4.0, c[2]])
        else:
            p.vertices = p.vertices @ rot_h.T
            hub = np.array([lo[0] - 4.0, c[1], c[2]])
        local = p.copy()
        local.vertices = (local.vertices @ TO_GZ.T) / 1000.0
        local.export(out / f"prop{inst}.stl")
        hub_gz = ((hub - CENTER) @ TO_GZ.T) / 1000.0
        axis_gz = TO_GZ @ np.eye(3)[axis]
        info.append((inst, hub_gz.round(4).tolist(), axis_gz.round(3).tolist()))
    for i in info:
        print("prop", i)


if __name__ == "__main__":
    main()
