import argparse
import json
import math
import os
import random
import signal
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node

LYING = ["rescue_randy", "standing_person", "male_visitor", "casual_female"]
SITTING = ["rescue_randy_sitting", "visitor_kid_sit"]
STANDING = ["standing_person", "walking_person", "male_visitor", "casual_female"]
VEHICLES = ["pickup", "suv", "hatchback"]
DISTRACT_WATER = ["buoy_orange", "buoy_red", "rc_boat"]
DISTRACT_BEACH = ["barrier", "buoy_orange"]
LABEL = {"lying": 1, "sitting": 2, "standing": 3, "water": 4, "hostile": 5, "mine": 6, "weapon": 7, "debris": 8}

PRIM = """  <model name="{n}"><static>true</static><pose>{x:.2f} {y:.2f} {z:.2f} 0 0 {yaw:.3f}</pose><link name="l">{visuals}</link>{lab}</model>\n"""

VIS = """<visual name="{n}"><pose>{px:.3f} {py:.3f} {pz:.3f} {pr:.3f} {pp:.3f} 0</pose><geometry>{g}</geometry><material><ambient>{c} 1</ambient><diffuse>{c} 1</diffuse></material></visual>"""


def vis(n, g, c, px=0, py=0, pz=0, pr=0, pp=0):
    return VIS.format(n=n, g=g, c=c, px=px, py=py, pz=pz, pr=pr, pp=pp)


def mine_visuals(size, dark):
    r = size / 2
    v = [vis("body", f"<sphere><radius>{r:.3f}</radius></sphere>", dark, pz=r)]
    for i, (a, b) in enumerate([(0, 0), (math.pi / 2, 0), (0, math.pi / 2), (math.pi / 2, math.pi / 2)]):
        v.append(vis(f"spike{i}", f"<cylinder><radius>0.03</radius><length>{size * 1.25:.3f}</length></cylinder>", "0.2 0.2 0.18", pz=r, pr=a, pp=b))
    return "".join(v)


def weapon_visuals(size):
    return (vis("stock", f"<box><size>{size:.2f} 0.08 0.06</size></box>", "0.08 0.08 0.08", pz=0.04)
            + vis("mag", "<box><size>0.08 0.05 0.18</size></box>", "0.1 0.1 0.1", px=size * 0.1))


def debris_visuals(size, rng):
    cols = ["0.9 0.9 0.95", "0.2 0.5 0.9", "0.9 0.3 0.2", "0.95 0.85 0.2", "0.3 0.7 0.3"]
    v = []
    for i in range(rng.randint(3, 6)):
        a = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(0, size * 0.45)
        v.append(vis(f"p{i}", f"<cylinder><radius>0.05</radius><length>{rng.uniform(0.2, 0.35):.2f}</length></cylinder>", rng.choice(cols),
                     px=d * math.cos(a), py=d * math.sin(a), pz=0.05, pr=math.pi / 2, pp=rng.uniform(0, 3)))
    v.append(vis("bag", f"<box><size>{size * 0.5:.2f} {size * 0.35:.2f} 0.04</size></box>", rng.choice(cols), pz=0.02))
    return "".join(v)
SEVERITY_TEMP = {"IMMEDIATE": 310.0, "DELAYED": 309.0, "MINOR": 309.5, "EXPECTANT": 297.0}

WORLD_HEAD = """<?xml version="1.0"?>
<sdf version="1.9">
<world name="gen">
  <physics name="p" type="ignored"><max_step_size>0.01</max_step_size><real_time_factor>1</real_time_factor></physics>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
  <gravity>0 0 0</gravity>
  <scene><ambient>{amb} {amb} {amb} 1</ambient><background>0.6 0.75 0.9 1</background><shadows>false</shadows></scene>
  <light type="directional" name="sun"><cast_shadows>false</cast_shadows><pose>0 0 50 0 0 0</pose>
    <diffuse>{sun} {sun} {sun2} 1</diffuse><specular>0.3 0.3 0.3 1</specular><direction>{dx} {dy} -1</direction></light>
  <model name="water"><static>true</static><link name="l"><visual name="v"><geometry><plane><normal>0 0 1</normal><size>600 600</size></plane></geometry>
    <material><ambient>{wr} {wg} {wb} 1</ambient><diffuse>{wr} {wg} {wb} 1</diffuse><specular>0.5 0.5 0.5 1</specular></material></visual></link></model>
  <model name="beach"><static>true</static><pose>{bx} {by} 0.03 0 0 {byaw}</pose><link name="l"><visual name="v"><geometry><box><size>{bw} 600 0.06</size></box></geometry>
    <material><ambient>{sr} {sg} {sb} 1</ambient><diffuse>{sr} {sg} {sb} 1</diffuse></material></visual></link></model>
  <model name="cam"><pose>0 0 30 0 1.5708 0</pose><link name="l"><gravity>false</gravity>
    <sensor name="bbox" type="boundingbox_camera"><topic>boxes</topic><update_rate>4</update_rate><always_on>1</always_on>
      <camera><box_type>2d</box_type><horizontal_fov>{fov}</horizontal_fov><image><width>{w}</width><height>{h}</height></image><clip><near>0.5</near><far>300</far></clip></camera>
    </sensor></link></model>\n"""

INCLUDE = """  <include><uri>model://{m}</uri><name>{n}</name><static>true</static><pose>{x:.2f} {y:.2f} {z:.2f} {r:.3f} {p:.3f} {yaw:.3f}</pose>{lab}</include>\n"""

LABEL_PLUGIN = '<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{}</label></plugin>'

RIFLE = """  <model name="{n}"><static>true</static><pose>{x:.2f} {y:.2f} 1.2 0 0.3 {yaw:.3f}</pose><link name="l"><visual name="v"><geometry><box><size>0.9 0.07 0.12</size></box></geometry>
    <material><ambient>0.05 0.05 0.05 1</ambient><diffuse>0.08 0.08 0.08 1</diffuse></material></visual></link></model>\n"""


def on_beach(x, y, bx, by, byaw, bw):
    c, s = math.cos(-byaw), math.sin(-byaw)
    lx = c * (x - bx) - s * (y - by)
    return abs(lx) < bw / 2 - 1.0


def make_world(seed, w, h, fov, extent):
    rng = random.Random(seed)
    amb = rng.uniform(0.35, 0.7)
    sun = rng.uniform(0.6, 1.0)
    byaw = rng.uniform(-0.6, 0.6)
    bw = rng.uniform(20, 45)
    side = rng.choice([-1, 1])
    bx, by = side * rng.uniform(extent * 0.55, extent * 0.85), 0.0
    head = WORLD_HEAD.format(
        amb=f"{amb:.2f}", sun=f"{sun:.2f}", sun2=f"{sun * rng.uniform(0.85, 1.0):.2f}",
        dx=f"{rng.uniform(-0.6, 0.6):.2f}", dy=f"{rng.uniform(-0.6, 0.6):.2f}",
        wr=f"{rng.uniform(0.02, 0.1):.2f}", wg=f"{rng.uniform(0.2, 0.38):.2f}", wb=f"{rng.uniform(0.35, 0.55):.2f}",
        sr=f"{rng.uniform(0.7, 0.85):.2f}", sg=f"{rng.uniform(0.62, 0.75):.2f}", sb=f"{rng.uniform(0.45, 0.58):.2f}",
        bx=f"{bx:.2f}", by=f"{by:.2f}", byaw=f"{byaw:.3f}", bw=f"{bw:.1f}", fov=f"{fov:.3f}", w=w, h=h,
    )
    body = []
    truth = []
    k = 0

    def place(beach):
        for _ in range(200):
            x, y = rng.uniform(-extent, extent), rng.uniform(-extent, extent)
            if on_beach(x, y, bx, by, byaw, bw) == beach:
                return x, y
        return rng.uniform(-extent, extent), rng.uniform(-extent, extent)

    def add(model, kind, x, y, z, r, p, yaw, label, extra=None):
        nonlocal k
        k += 1
        name = f"{kind}_{k}"
        lab = LABEL_PLUGIN.format(label) if label else ""
        body.append(INCLUDE.format(m=model, n=name, x=x, y=y, z=z, r=r, p=p, yaw=yaw, lab=lab))
        if label:
            truth.append({"name": name, "kind": kind, "label": label, "p": [round(x, 2), round(y, 2)], **(extra or {})})
        return name

    n_people = rng.randint(7, 12)
    for _ in range(n_people):
        u = rng.random()
        yaw = rng.uniform(-math.pi, math.pi)
        if u < 0.4:
            x, y = place(False)
            sev = "IMMEDIATE"
            add(rng.choice(LYING), "water", x, y, 0.09, math.pi / 2, 0, yaw, LABEL["water"],
                {"severity": sev, "temp_k": SEVERITY_TEMP[sev], "moving": False})
        elif u < 0.7:
            x, y = place(True)
            sev = rng.choices(["IMMEDIATE", "DELAYED", "EXPECTANT"], [0.5, 0.3, 0.2])[0]
            add(rng.choice(LYING), "lying", x, y, 0.2, math.pi / 2, 0, yaw, LABEL["lying"],
                {"severity": sev, "temp_k": SEVERITY_TEMP[sev], "moving": sev == "DELAYED"})
        elif u < 0.85:
            x, y = place(True)
            add(rng.choice(SITTING), "sitting", x, y, 0.06, 0, 0, yaw, LABEL["sitting"],
                {"severity": "DELAYED", "temp_k": SEVERITY_TEMP["DELAYED"], "moving": True})
        else:
            x, y = place(True)
            add(rng.choice(STANDING), "standing", x, y, 0.06, 0, 0, yaw, LABEL["standing"],
                {"severity": "MINOR", "temp_k": SEVERITY_TEMP["MINOR"], "moving": True})

    for _ in range(rng.randint(1, 3)):
        u = rng.random()
        yaw = rng.uniform(-math.pi, math.pi)
        if u < 0.6:
            x, y = place(False)
            add("patrol_boat", "hostile", x, y, 0.0, 0, 0, yaw, LABEL["hostile"], {"type": "patrol_boat"})
        elif u < 0.9:
            x, y = place(True)
            add(rng.choice(STANDING), "hostile", x, y, 0.06, 0, 0, yaw, LABEL["hostile"], {"type": "armed_figure"})
            body.append(RIFLE.format(n=f"rifle_{k}", x=x + 0.35 * math.cos(yaw + 1.2), y=y + 0.35 * math.sin(yaw + 1.2), yaw=yaw))
        else:
            x, y = place(True)
            add(rng.choice(VEHICLES), "hostile", x, y, 0.06, 0, 0, yaw, LABEL["hostile"], {"type": "vehicle"})

    def prim(kind, x, y, z, visuals, label, extra=None):
        nonlocal k
        k += 1
        name = f"{kind}_{k}"
        body.append(PRIM.format(n=name, x=x, y=y, z=z, yaw=rng.uniform(-3, 3), visuals=visuals, lab=LABEL_PLUGIN.format(label)))
        truth.append({"name": name, "kind": kind, "label": label, "p": [round(x, 2), round(y, 2)], **(extra or {})})

    for _ in range(rng.randint(2, 4)):
        floating = rng.random() < 0.5
        x, y = place(not floating)
        size = rng.uniform(0.5, 0.9)
        prim("mine", x, y, -0.15 if floating else 0.06, mine_visuals(size, rng.choice(["0.12 0.12 0.1", "0.2 0.18 0.12", "0.25 0.25 0.22"])), LABEL["mine"], {"floating": floating})
    for _ in range(rng.randint(1, 2)):
        x, y = place(True)
        prim("weapon", x, y, 0.08, weapon_visuals(rng.uniform(0.8, 1.1)), LABEL["weapon"])
    for _ in range(rng.randint(2, 4)):
        floating = rng.random() < 0.6
        x, y = place(not floating)
        prim("debris", x, y, 0.0 if floating else 0.06, debris_visuals(rng.uniform(0.8, 1.4), rng), LABEL["debris"])

    for _ in range(rng.randint(3, 7)):
        x, y = place(False)
        add(rng.choice(DISTRACT_WATER), "distract", x, y, 0.0, 0, 0, rng.uniform(-3, 3), 0)
    for _ in range(rng.randint(2, 5)):
        x, y = place(True)
        add(rng.choice(DISTRACT_BEACH), "distract", x, y, 0.06, 0, 0, rng.uniform(-3, 3), 0)

    return head + "".join(body) + "</world>\n</sdf>\n", truth


def stamp_key(h):
    return h.stamp.sec * 1_000_000_000 + h.stamp.nsec


def to_bgr(msg):
    a = np.frombuffer(msg.data, np.uint8)
    if len(a) == msg.width * msg.height * 3:
        return a.reshape(msg.height, msg.width, 3)[:, :, ::-1].copy()
    if len(a) == msg.width * msg.height * 4:
        return a.reshape(msg.height, msg.width, 4)[:, :, 2::-1].copy()
    return None


def euler_to_quat(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def run_scene(seed, out, views, w, h, fov, extent, lmap, settle, self_alt_lo=10.0, self_alt_hi=40.0, tilt_lo=0.0, tilt_hi=0.35):
    sdf, truth = make_world(seed, w, h, fov, extent)
    scene = f"g{seed:04d}"
    wpath = Path(out) / "worlds" / f"{scene}.sdf"
    wpath.parent.mkdir(parents=True, exist_ok=True)
    wpath.write_text(sdf)
    (Path(out) / "truth").mkdir(parents=True, exist_ok=True)
    (Path(out) / "truth" / f"{scene}.json").write_text(json.dumps(truth))
    (Path(out) / "images").mkdir(parents=True, exist_ok=True)
    (Path(out) / "labels").mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(["gz", "sim", "-s", "-r", "--headless-rendering", str(wpath)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    node = Node()
    lock = threading.Lock()
    state = {"img": {}, "box": {}}

    def on_img(m):
        with lock:
            state["img"][stamp_key(m.header)] = m

    def on_box(m):
        with lock:
            state["box"][stamp_key(m.header)] = m

    node.subscribe(Image, "/boxes_image", on_img)
    node.subscribe(AnnotatedAxisAligned2DBox_V, "/boxes", on_box)
    rng = random.Random(seed * 7 + 1)
    saved = 0
    try:
        t0 = time.time()
        while time.time() - t0 < 120:
            with lock:
                if state["img"]:
                    break
            time.sleep(0.5)
        else:
            print(f"{scene}: no frames from renderer", flush=True)
            return 0
        for v in range(views):
            alt = rng.uniform(self_alt_lo, self_alt_hi)
            tilt = rng.uniform(tilt_lo, tilt_hi)
            yaw = rng.uniform(-math.pi, math.pi)
            tgt = rng.choice(truth) if truth and rng.random() < 0.85 else {"p": [rng.uniform(-extent, extent), rng.uniform(-extent, extent)]}
            x, y = tgt["p"][0] + rng.uniform(-0.3, 0.3) * alt, tgt["p"][1] + rng.uniform(-0.3, 0.3) * alt
            q = euler_to_quat(0, math.pi / 2 - tilt, yaw)
            req = Pose()
            req.name = "cam"
            req.position.x, req.position.y, req.position.z = x, y, alt
            req.orientation.w, req.orientation.x, req.orientation.y, req.orientation.z = q
            ok, _ = node.request("/world/gen/set_pose", req, Pose, Boolean, 3000)
            if not ok:
                continue
            with lock:
                state["img"].clear()
                state["box"].clear()
            time.sleep(settle)
            pair = None
            t1 = time.time()
            while time.time() - t1 < 10 and pair is None:
                with lock:
                    common = sorted(set(state["img"]) & set(state["box"]))
                    if common:
                        k = common[-1]
                        pair = (state["img"][k], state["box"][k])
                time.sleep(0.1)
            if pair is None:
                continue
            frame = to_bgr(pair[0])
            if frame is None:
                continue
            lines = []
            for ab in pair[1].annotated_box:
                if ab.label not in lmap:
                    continue
                x0, y0 = max(0.0, ab.box.min_corner.x), max(0.0, ab.box.min_corner.y)
                x1, y1 = min(w, ab.box.max_corner.x), min(h, ab.box.max_corner.y)
                if x1 - x0 < 3 or y1 - y0 < 3:
                    continue
                lines.append(f"{lmap[ab.label]} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
            stem = f"{scene}_{v:03d}"
            cv2.imwrite(str(Path(out) / "images" / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (Path(out) / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            saved += 1
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        subprocess.run(["pkill", "-9", "-f", "gz sim"], check=False)
        time.sleep(1)
    print(f"{scene}: {saved} frames", flush=True)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--views", type=int, default=40)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=540)
    ap.add_argument("--fov", type=float, default=1.2)
    ap.add_argument("--extent", type=float, default=28.0)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--labels", default="1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7")
    ap.add_argument("--alt", default="10,40")
    ap.add_argument("--tilt", default="0,0.35")
    a = ap.parse_args()
    lmap = {int(k): int(v) for k, v in (p.split(":") for p in a.labels.split(","))}
    total = 0
    for s in range(a.start, a.start + a.scenes):
        al, ah = (float(v) for v in a.alt.split(","))
        tl, th = (float(v) for v in a.tilt.split(","))
        total += run_scene(s, a.out, a.views, a.w, a.h, a.fov, a.extent, lmap, a.settle, al, ah, tl, th)
    print(f"total {total}", flush=True)


if __name__ == "__main__":
    main()
