import argparse
import math
from pathlib import Path

from skyforge_core.params import DEFAULTS
from skyforge_core.sim.world import DEFAULT_SCENARIO

HEAD = """<?xml version="1.0"?>
<sdf version="1.9">
<world name="skyforge">
  <physics name="p" type="ignored"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
  <gravity>0 0 -9.81</gravity>
  <atmosphere type="adiabatic"><temperature>{ambient}</temperature></atmosphere>
  <scene><ambient>0.55 0.55 0.55 1</ambient><background>0.62 0.76 0.9 1</background><shadows>false</shadows></scene>
  <light type="directional" name="sun"><cast_shadows>false</cast_shadows><pose>0 0 100 0 0 0</pose>
    <diffuse>0.95 0.93 0.88 1</diffuse><specular>0.3 0.3 0.3 1</specular><direction>0.3 0.2 -1</direction></light>
  <model name="sea"><static>true</static><pose>{sea_x} {sea_y} 0 0 0 0</pose><link name="l">
    <visual name="v"><geometry><box><size>{sea_w} {sea_h} 0.02</size></box></geometry>
    <material><ambient>0.05 0.3 0.45 1</ambient><diffuse>0.05 0.3 0.45 1</diffuse><specular>0.5 0.5 0.5 1</specular></material></visual></link></model>
  <model name="beach"><static>true</static><pose>{beach_x} {beach_y} 0 0 0 0</pose><link name="l">
    <collision name="c"><geometry><box><size>{beach_w} {beach_h} 0.02</size></box></geometry></collision>
    <visual name="v"><geometry><box><size>{beach_w} {beach_h} 0.02</size></box></geometry>
    <material><ambient>0.78 0.7 0.52 1</ambient><diffuse>0.78 0.7 0.52 1</diffuse></material></visual></link></model>
  <model name="home_pad"><static>true</static><pose>{hx} {hy} 0.015 0 0 0</pose><link name="l">
    <visual name="v"><geometry><cylinder><radius>1.5</radius><length>0.01</length></cylinder></geometry>
    <material><ambient>0.2 0.6 0.2 1</ambient><diffuse>0.2 0.6 0.2 1</diffuse></material></visual></link></model>\n"""

THERMAL = '<plugin filename="gz-sim-thermal-system" name="gz::sim::systems::Thermal"><temperature>{t}</temperature></plugin>'

INCLUDE = """  <include><uri>model://{m}</uri><name>{n}</name><static>true</static><pose>{x:.2f} {y:.2f} {z:.2f} {r:.4f} 0 {yaw:.4f}</pose>{plugins}</include>\n"""

PRIM = """  <model name="{n}"><static>true</static><pose>{x:.2f} {y:.2f} {z:.2f} 0 0 {yaw:.3f}</pose><link name="l">{visuals}</link>{plugins}</model>\n"""

VIS = """<visual name="{n}"><pose>{px} {py} {pz} {pr} {pp} 0</pose><geometry>{g}</geometry><material><ambient>{c} 1</ambient><diffuse>{c} 1</diffuse></material></visual>"""

COLL_CYL = """<collision name="c"><pose>0 0 {z} 0 0 0</pose><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry></collision>"""


def cyl(r, h):
    return f"<cylinder><radius>{r}</radius><length>{h}</length></cylinder>"


def box(x, y, z):
    return f"<box><size>{x} {y} {z}</size></box>"


def sphere(r):
    return f"<sphere><radius>{r}</radius></sphere>"


def vis(n, g, c, px=0, py=0, pz=0, pr=0, pp=0):
    return VIS.format(n=n, g=g, c=c, px=px, py=py, pz=pz, pr=pr, pp=pp)


def mine_visuals(size):
    r = size / 2
    v = [vis("body", sphere(r), "0.12 0.12 0.1", pz=r)]
    for i, (a, b) in enumerate([(0, 0), (math.pi / 2, 0), (0, math.pi / 2), (math.pi / 2, math.pi / 2)]):
        v.append(vis(f"spike{i}", cyl(0.03, size * 1.25), "0.2 0.2 0.18", pz=r, pr=a, pp=b))
    return "".join(v)


def weapon_visuals(size):
    return vis("stock", box(size, 0.08, 0.06), "0.08 0.08 0.08", pz=0.04) + vis("mag", box(0.08, 0.05, 0.18), "0.1 0.1 0.1", px=size * 0.1, pz=0.0)


def debris_visuals(size, k):
    cols = ["0.9 0.9 0.95", "0.2 0.5 0.9", "0.9 0.3 0.2", "0.95 0.85 0.2"]
    v = []
    for i in range(4):
        a = 2 * math.pi * i / 4 + k
        d = size * 0.35
        v.append(vis(f"p{i}", cyl(0.05, 0.28), cols[i % 4], px=round(d * math.cos(a), 2), py=round(d * math.sin(a), 2), pz=0.05, pr=math.pi / 2))
    v.append(vis("bag", box(size * 0.5, size * 0.35, 0.04), "0.85 0.85 0.85", pz=0.02))
    return "".join(v)


def obstacle(o):
    r, h = o["r"], o["h"]
    if o["id"] == "lighthouse":
        visuals = vis("tower", cyl(r * 0.6, h), "0.95 0.95 0.95", pz=h / 2) + vis("band", cyl(r * 0.62, h * 0.12), "0.8 0.1 0.1", pz=h * 0.6) + vis("top", cyl(r * 0.7, h * 0.08), "0.2 0.2 0.2", pz=h * 0.96)
    elif o["id"] == "rock":
        visuals = vis("rock", cyl(r, h), "0.38 0.36 0.33", pz=h / 2)
    else:
        visuals = vis("pole", cyl(r * 0.5, h), "0.6 0.6 0.6", pz=h / 2)
    coll = COLL_CYL.format(z=h / 2, r=r, h=h)
    return PRIM.format(n=o["id"], x=o["p"][0], y=o["p"][1], z=0, yaw=0, visuals=coll + visuals, plugins="")


def build(scenario, params):
    area = params["mission"]["area"]
    shore = max(p[1] for poly in params["water"]["polygons"] for p in poly)
    xmin, xmax = area["xmin"] - 60, area["xmax"] + 60
    ymin, ymax = area["ymin"] - 60, area["ymax"] + 60
    hx, hy = params["mission"]["home"]
    out = [HEAD.format(
        ambient=params["perception"]["ambient_k"],
        sea_x=(xmin + xmax) / 2, sea_y=(ymin + shore) / 2, sea_w=xmax - xmin, sea_h=shore - ymin,
        beach_x=(xmin + xmax) / 2, beach_y=(shore + ymax) / 2, beach_w=xmax - xmin, beach_h=ymax - shore,
        hx=hx, hy=hy)]
    k = 0
    for o in scenario["objects"]:
        k += 1
        x, y = o["p"]
        floating = o.get("floating", y < shore)
        yaw = (k * 1.37) % (2 * math.pi)
        cls = o["cls"]
        if cls == "injured":
            t = THERMAL.format(t=o.get("temp_k", 309.0))
            post = o.get("posture", "lying")
            if post == "sitting":
                out.append(INCLUDE.format(m="rescue_randy_sitting", n=o["id"], x=x, y=y, z=0.0, r=0, yaw=yaw, plugins=t))
            elif post in ("standing", "waving"):
                out.append(INCLUDE.format(m="male_visitor", n=o["id"], x=x, y=y, z=0.0, r=0, yaw=yaw, plugins=t))
            else:
                z = 0.05 if floating else 0.17
                out.append(INCLUDE.format(m="rescue_randy", n=o["id"], x=x, y=y, z=z, r=math.pi / 2, yaw=yaw, plugins=t))
        elif cls == "mine":
            out.append(PRIM.format(n=o["id"], x=x, y=y, z=-0.15 if floating else 0.0, yaw=yaw, visuals=mine_visuals(o.get("size", 0.7)), plugins=""))
        elif cls == "weapon":
            out.append(PRIM.format(n=o["id"], x=x, y=y, z=0.02, yaw=yaw, visuals=weapon_visuals(o.get("size", 1.0)), plugins=""))
        elif cls == "hostile_unit":
            out.append(INCLUDE.format(m="pickup", n=o["id"], x=x, y=y, z=0.0, r=0, yaw=yaw, plugins=""))
            out.append(INCLUDE.format(m="male_visitor", n=o["id"] + "_gunner", x=x + 3.0, y=y + 1.0, z=0.0, r=0, yaw=yaw, plugins=THERMAL.format(t=309.5)))
        elif cls == "debris":
            out.append(PRIM.format(n=o["id"], x=x, y=y, z=0.0, yaw=yaw, visuals=debris_visuals(o.get("size", 1.0), k), plugins=""))
    for o in scenario.get("obstacles", []):
        out.append(obstacle(o))
    for m in scenario.get("moving_obstacles", []):
        x, y = m["loop"][0]
        out.append(INCLUDE.format(m="patrol_boat", n=m["id"], x=x, y=y, z=0.0, r=0, yaw=0, plugins=""))
    out.append(INCLUDE.format(m="skyforge_drone", n="skyforge", x=hx, y=hy, z=0.12, r=0, yaw=0, plugins="").replace("<static>true</static>", "<static>false</static>"))
    out.append("</world>\n</sdf>\n")
    return "".join(out)


def write_config(scenario, base_yaml, out_yaml):
    import yaml
    cfg = yaml.safe_load(Path(base_yaml).read_text())
    root = cfg.setdefault("skyforge", {})
    root["obstacles"] = [{"id": o["id"], "p": list(o["p"]), "r": float(o["r"]), "h": float(o["h"])} for o in scenario.get("obstacles", [])]
    for section, values in scenario.get("params", {}).items():
        root.setdefault(section, {}).update(values)
    Path(out_yaml).write_text(yaml.safe_dump(cfg, sort_keys=False))


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(here / "worlds" / "coastal.sdf"))
    ap.add_argument("--base-config", default=str(here.parent / "skyforge_bringup" / "config" / "skyforge.yaml"))
    ap.add_argument("--out-config", default=str(here / "config" / "skyforge_gazebo.yaml"))
    a = ap.parse_args()
    Path(a.out).write_text(build(DEFAULT_SCENARIO, DEFAULTS))
    write_config(DEFAULT_SCENARIO, a.base_config, a.out_config)
    print(a.out, a.out_config)


if __name__ == "__main__":
    main()
