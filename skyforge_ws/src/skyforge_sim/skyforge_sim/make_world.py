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
  <scene><ambient>0.5 0.5 0.52 1</ambient><background>0.62 0.76 0.9 1</background><shadows>true</shadows><sky></sky></scene>
  <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 100 0 0 0</pose>
    <diffuse>0.95 0.93 0.88 1</diffuse><specular>0.3 0.3 0.3 1</specular><direction>0.3 0.2 -1</direction></light>
{terrain}\n"""

HOSTILE_VESSELS = [
    {"id": "H-vessel-1", "p": [35.0, -48.0], "yaw": 0.4},
    {"id": "H-vessel-2", "p": [215.0, -30.0], "yaw": 2.6, "patrol": [[215.0, -30.0], [215.0, -58.0]], "speed": 1.2},
]

SIM_OVERRIDES = {"health": {"odom_timeout_s": 2.0}, "mission": {"delivery_alt": 25.0}}

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


def pbr(tex, rough):
    return (f"<material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><specular>0.2 0.2 0.2 1</specular><pbr><metal>"
            f"<albedo_map>model://terrain/materials/textures/{tex}_albedo.png</albedo_map>"
            f"<normal_map>model://terrain/materials/textures/{tex}_normal.png</normal_map>"
            f"<roughness>{rough}</roughness><metalness>0.0</metalness></metal></pbr></material>")


def terrain(xmin, xmax, ymin, ymax, shore, tile=24.0):
    vis = []
    cols = int(math.ceil((xmax - xmin) / tile))
    y = shore - tile
    rows = []
    while y > ymin - tile:
        rows.append((y, "water_deep" if y < shore - 3 * tile else "water_shallow", 0.25))
        y -= tile
    y = shore
    while y < ymax:
        rows.append((y, "sand", 0.95))
        y += tile
    k = 0
    for y0, tex, rough in rows:
        for c in range(cols):
            x0 = xmin + c * tile
            k += 1
            vis.append(f'<visual name="t{k}"><pose>{x0 + tile / 2:.1f} {y0 + tile / 2:.1f} 0 0 0 0</pose><geometry><box><size>{tile} {tile} 0.02</size></box></geometry><cast_shadows>false</cast_shadows>{pbr(tex, rough)}</visual>')
    sea_h = shore - (ymin - tile)
    beach_h = ymax - shore
    coll = (f'<collision name="sea"><pose>{(xmin + xmax) / 2:.1f} {shore - sea_h / 2:.1f} 0 0 0 0</pose><geometry><box><size>{xmax - xmin + tile} {sea_h} 0.02</size></box></geometry></collision>'
            f'<collision name="beach"><pose>{(xmin + xmax) / 2:.1f} {shore + beach_h / 2:.1f} 0 0 0 0</pose><geometry><box><size>{xmax - xmin + tile} {beach_h} 0.02</size></box></geometry></collision>')
    return f'  <model name="terrain"><static>true</static><link name="l">{coll}{"".join(vis)}</link></model>'


HM_SIZE, HM_Z, HM_OFF = 1200.0, 201.66, (110.0, 760.0, -5.0)


class Terrain:
    def __init__(self, cx, cy):
        import numpy as np
        f = Path(__file__).resolve().parent.parent / "models" / "harmonic_terrain" / "materials" / "textures" / "Heightmap.png"
        if not f.exists():
            f = Path(__file__).resolve().parents[4] / "share" / "skyforge_sim" / "models" / "harmonic_terrain" / "materials" / "textures" / "Heightmap.png"
        try:
            from PIL import Image
            raw = np.asarray(Image.open(f))
        except ImportError:
            import cv2
            raw = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        self.h = raw.astype("float32") / 65535.0 * HM_Z + HM_OFF[2]
        self.n = self.h.shape[0] - 1
        self.cx, self.cy = cx, cy

    def height(self, x, y):
        c = (-(x - HM_OFF[0]) / HM_SIZE + 0.5) * self.n
        r = (0.5 + (y - HM_OFF[1]) / HM_SIZE) * self.n
        if not (0 <= c <= self.n and 0 <= r <= self.n):
            return None
        return float(self.h[int(round(r)), int(round(c))])


def horizon(xmin, xmax, ymin, ymax, shore):
    cx = (xmin + xmax) / 2
    far = (f'  <model name="open_sea"><static>true</static><pose>{cx:.1f} {shore - 1000:.1f} -0.05 0 0 0</pose><link name="l">'
           f'<visual name="v"><geometry><box><size>3000 2000 0.02</size></box></geometry><cast_shadows>false</cast_shadows>'
           f'<material><ambient>0.03 0.2 0.33 1</ambient><diffuse>0.03 0.22 0.36 1</diffuse><specular>0.5 0.5 0.5 1</specular></material></visual></link></model>\n')
    hy0 = TERRAIN_Y
    hills = INCLUDE.format(m="harmonic_terrain", n="hills", x=0.0, y=0.0, z=0.0, r=0, yaw=0.0, plugins="")
    land = (f'  <model name="hinterland"><static>true</static><pose>{cx:.1f} {ymax + 700:.1f} -0.05 0 0 0</pose><link name="l">'
            f'<visual name="v"><geometry><box><size>3000 1400 0.02</size></box></geometry><cast_shadows>false</cast_shadows>'
            f'<material><ambient>0.6 0.55 0.4 1</ambient><diffuse>0.62 0.57 0.42 1</diffuse></material></visual></link></model>\n')
    isles = "".join(INCLUDE.format(m="rock1" if i % 2 else "rock3", n=f"island_{i}", x=x, y=y, z=-2.0, r=0, yaw=i * 1.1, plugins="")
                    for i, (x, y) in enumerate([(-300, -500), (100, -700), (450, -550), (700, -800), (-600, -900)]))
    lake = []
    for i in range(4):
        for j in range(4):
            lx, ly = HM_OFF[0] - 450 + i * 300, HM_OFF[1] - 450 + j * 300
            lake.append(f'<visual name="w{i}{j}"><pose>{lx:.1f} {ly:.1f} 0 0 0 0</pose><geometry><box><size>300 300 0.02</size></box></geometry>'
                        f'<cast_shadows>false</cast_shadows>{pbr("water_deep", 0.25)}</visual>')
    lakes = f'  <model name="lagoon"><static>true</static><link name="l">{"".join(lake)}</link></model>\n'
    return far + hills + lakes + isles


TERRAIN_Y = 140.0


def nature(area, shore, rng):
    out = []
    k = 0
    ter = Terrain(0.0, 0.0)

    def inside(x, y, m=8.0):
        return area["xmin"] - m < x < area["xmax"] + m and area["ymin"] - m < y < area["ymax"] + m

    while k < 120:
        x, y = rng.uniform(area["xmin"] - 120, area["xmax"] + 120), rng.uniform(shore + 6, area["ymax"] + 60)
        if inside(x, y):
            continue
        hgt = ter.height(x, y)
        if hgt is not None and hgt > 0.3:
            continue
        k += 1
        m = "pine_tree" if rng.random() < 0.6 else "oak_tree"
        out.append(INCLUDE.format(m=m, n=f"tree_{k}", x=x, y=y, z=0.0, r=0, yaw=rng.uniform(0, 6.28), plugins=""))
    rocks = [(area["xmin"] - 45, shore - 6), (area["xmax"] + 45, shore - 8), (40, area["ymin"] - 45), (150, area["ymin"] - 60), (260, area["ymin"] - 40), (area["xmin"] - 30, shore + 12)]
    for i, (x, y) in enumerate(rocks):
        out.append(INCLUDE.format(m="rock1" if i % 2 == 0 else "rock3", n=f"outcrop_{i}", x=x, y=y, z=-1.0, r=0, yaw=rng.uniform(0, 6.28), plugins=""))
    return "".join(out)


def build(scenario, params):
    area = params["mission"]["area"]
    shore = max(p[1] for poly in params["water"]["polygons"] for p in poly)
    xmin, xmax = area["xmin"] - 220, area["xmax"] + 220
    ymin, ymax = area["ymin"] - 200, area["ymax"] + 70
    hx, hy = params["mission"]["home"]
    import random
    out = [HEAD.format(ambient=params["perception"]["ambient_k"], terrain=terrain(xmin, xmax, ymin, ymax, shore), hx=hx, hy=hy)]
    out.append(nature(area, shore, random.Random(11)))
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
    for v in scenario.get("hostile_vessels", HOSTILE_VESSELS):
        out.append(INCLUDE.format(m="patrol_boat", n=v["id"], x=v["p"][0], y=v["p"][1], z=0.0, r=0, yaw=v["yaw"], plugins=THERMAL.format(t=320.0)))
    out.append(horizon(xmin, xmax, ymin, ymax, shore))
    for m in scenario.get("moving_obstacles", []):
        x, y = m["loop"][0]
        out.append(INCLUDE.format(m="civil_boat", n=m["id"], x=x, y=y, z=0.0, r=0, yaw=0, plugins=""))
    out.append(INCLUDE.format(m="skyforge_drone", n="skyforge", x=hx, y=hy, z=0.12, r=0, yaw=0, plugins="").replace("<static>true</static>", "<static>false</static>"))
    out.append("</world>\n</sdf>\n")
    return "".join(out)


def random_scenario(seed, params):
    import copy
    import random
    rng = random.Random(seed)
    sc = copy.deepcopy(DEFAULT_SCENARIO)
    sc["name"] = f"random_{seed}"
    sc["seed"] = seed
    area = params["mission"]["area"]
    shore = max(p[1] for poly in params["water"]["polygons"] for p in poly)
    hx, hy = params["mission"]["home"]
    placed = []

    def spot(water, clear=12.0, home_clear=20.0):
        for _ in range(500):
            x = rng.uniform(area["xmin"] + 15, area["xmax"] - 10)
            y = rng.uniform(area["ymin"] + 8, shore - 6) if water else rng.uniform(shore + 6, area["ymax"] - 8)
            if math.hypot(x - hx, y - hy) < home_clear:
                continue
            if all(math.hypot(x - a, y - b) >= clear for a, b in placed):
                placed.append((x, y))
                return [round(x, 1), round(y, 1)]
        return [round(x, 1), round(y, 1)]

    objs = []
    cats = [("IMM", "lying", 309.5, False, False), ("IMM", "lying", 307.0, False, True), ("DEL", "sitting", 308.5, True, False),
            ("MIN", "standing", 309.0, True, False), ("EXP", "lying", 296.0, False, False), ("DEL", "lying", 308.8, True, False)]
    rng.shuffle(cats)
    for i, (cat, post, temp, moving, water) in enumerate(cats[:rng.randint(4, 6)]):
        o = {"id": f"T-{cat}-{i}", "cls": "injured", "p": spot(water), "size": 1.8 if post == "lying" else 1.2,
             "temp_k": temp, "posture": post, "moving": moving, "cat": cat}
        if water:
            o["floating"] = True
        objs.append(o)
    for i in range(rng.randint(2, 3)):
        fl = i == 0 or rng.random() < 0.4
        o = {"id": f"T-mine-{i}", "cls": "mine", "p": spot(fl, clear=18.0), "size": round(rng.uniform(0.6, 0.8), 2)}
        if fl:
            o["floating"] = True
        objs.append(o)
    for i in range(rng.randint(1, 2)):
        objs.append({"id": f"T-weapon-{i}", "cls": "weapon", "p": spot(False), "size": 1.0})
    for i in range(rng.randint(1, 2)):
        objs.append({"id": f"T-hostile-{i}", "cls": "hostile_unit", "p": spot(False, clear=30.0), "size": 2.0, "detection_radius": 20.0})
    for i in range(rng.randint(2, 4)):
        fl = rng.random() < 0.6
        o = {"id": f"T-debris-{i}", "cls": "debris", "p": spot(fl, clear=8.0), "size": round(rng.uniform(0.8, 1.2), 2)}
        if fl:
            o["floating"] = True
        objs.append(o)
    sc["objects"] = objs
    obst = []
    for oid, r, h in (("lighthouse", 5.0, 40.0), ("rock", 4.0, 12.0), ("pole", 1.0, 30.0)):
        obst.append({"id": oid, "p": spot(False, clear=20.0), "r": r, "h": h})
    sc["obstacles"] = obst
    sc["wind"] = [round(rng.uniform(-3, 4), 1), round(rng.uniform(-2, 3), 1)]
    vessels = []
    for i in range(2):
        p0 = spot(True, clear=35.0)
        v = {"id": f"H-vessel-{i + 1}", "p": p0, "yaw": round(rng.uniform(0, 6.28), 2)}
        if i == 1:
            v["patrol"] = [p0, [p0[0], max(area["ymin"] + 5, p0[1] - 25)]]
            v["speed"] = 1.2
        vessels.append(v)
    return sc, vessels


def write_config(scenario, base_yaml, out_yaml):
    import yaml
    cfg = yaml.safe_load(Path(base_yaml).read_text())
    root = cfg.setdefault("skyforge", {})
    root["obstacles"] = [{"id": o["id"], "p": list(o["p"]), "r": float(o["r"]), "h": float(o["h"])} for o in scenario.get("obstacles", [])]
    for section, values in scenario.get("params", {}).items():
        root.setdefault(section, {}).update(values)
    for section, values in SIM_OVERRIDES.items():
        root.setdefault(section, {}).update(values)
    Path(out_yaml).write_text(yaml.safe_dump(cfg, sort_keys=False))


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(here / "worlds" / "coastal.sdf"))
    ap.add_argument("--base-config", default=str(here.parent / "skyforge_bringup" / "config" / "skyforge.yaml"))
    ap.add_argument("--out-config", default=str(here / "config" / "skyforge_gazebo.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    import json
    if a.seed is None:
        sc, vessels = DEFAULT_SCENARIO, HOSTILE_VESSELS
    else:
        sc, vessels = random_scenario(a.seed, DEFAULTS)
    sc = dict(sc)
    sc["hostile_vessels"] = vessels
    Path(a.out).write_text(build(sc, DEFAULTS))
    write_config(sc, a.base_config, a.out_config)
    Path(a.out).with_name("hostile_vessels.json").write_text(json.dumps(vessels))
    Path(a.out).with_name("scenario.json").write_text(json.dumps(sc))
    print(a.out, a.out_config)


if __name__ == "__main__":
    main()
