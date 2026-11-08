import json
import math

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from std_msgs.msg import String

from skyforge_core.params import DEFAULTS
from skyforge_core.sim.world import DEFAULT_SCENARIO


class Mover(Node):
    def __init__(self):
        super().__init__("sim_mover")
        self.declare_parameter("world", "skyforge")
        self.declare_parameter("rate", 10.0)
        self.declare_parameter("drift", True)
        self.declare_parameter("scenario_file", "")
        self.declare_parameter("wave_amp", 0.1)
        self.declare_parameter("wave_period", 4.0)
        self.t = 0.0
        sf = self.get_parameter("scenario_file").value
        sc = json.load(open(sf)) if sf else DEFAULT_SCENARIO
        shore = max(p[1] for poly in DEFAULTS["water"]["polygons"] for p in poly)
        self.wind = list(sc.get("wind", [0.0, 0.0]))
        cur = sc.get("current", DEFAULTS["drift"]["current"])
        k = sc.get("k_leeway", DEFAULTS["drift"]["k_leeway"])
        self.drift_v = (cur[0] + k * self.wind[0], cur[1] + k * self.wind[1])
        self.floaters = []
        if self.get_parameter("drift").value:
            for o in sc["objects"]:
                if o.get("floating", o["p"][1] < shore):
                    z = -0.15 if o["cls"] == "mine" else 0.05
                    rot = o["cls"] == "injured" and o.get("posture", "lying") == "lying"
                    self.floaters.append({"id": o["id"], "p": list(o["p"]), "z": z, "roll": math.pi / 2 if rot else 0.0,
                                          "phase": (len(self.floaters) * 1.7) % (2 * math.pi), "yaw": (len(self.floaters) * 2.3) % (2 * math.pi)})
        self.static = [{"p": list(o["p"]), "r": float(o["r"]), "h": float(o["h"]), "v": [0.0, 0.0]} for o in sc.get("obstacles", [])]
        self.movers = [{**m, "p": list(m["loop"][0]), "seg": 0, "v": [0.0, 0.0]} for m in sc.get("moving_obstacles", [])]
        vessels = sc.get("hostile_vessels")
        if vessels is None:
            try:
                from ament_index_python.packages import get_package_share_directory
                import os
                vessels = json.load(open(os.path.join(get_package_share_directory("skyforge_sim"), "worlds", "hostile_vessels.json")))
            except (OSError, ValueError):
                vessels = []
        for v in vessels:
            if "patrol" in v:
                self.movers.append({"id": v["id"], "loop": v["patrol"], "speed": v["speed"], "r": 0.0, "h": 0.0,
                                    "p": list(v["patrol"][0]), "seg": 0, "v": [0.0, 0.0], "hostile": True})
        w = self.get_parameter("world").value
        self.cli = self.create_client(SetEntityPose, f"/world/{w}/set_pose")
        self.pub_obs = self.create_publisher(String, "/skyforge/obstacles", 10)
        self.pub_truth = self.create_publisher(String, "/sim/truth", 10)
        self.truth_static = [{"id": o["id"], "cls": o["cls"], "p": list(o["p"]), "cat": o.get("cat", ""), "posture": o.get("posture", "")}
                             for o in sc["objects"] if not o.get("floating", o["p"][1] < shore)]
        self.truth_vessels = [{"id": v["id"], "cls": "hostile_unit", "p": list(v["p"])} for v in vessels if "patrol" not in v]
        self.truth_meta = {o["id"]: {"cls": o["cls"], "cat": o.get("cat", ""), "posture": o.get("posture", "")} for o in sc["objects"]}
        self.ticks = 0
        self.dt = 1.0 / float(self.get_parameter("rate").value)
        self.create_timer(self.dt, self.tick)

    def set_pose(self, name, x, y, z, roll, yaw, pitch=0.0):
        if not self.cli.service_is_ready():
            return
        r = SetEntityPose.Request()
        r.entity.name = name
        r.entity.type = Entity.MODEL
        r.pose.position.x, r.pose.position.y, r.pose.position.z = float(x), float(y), float(z)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        r.pose.orientation.w = cr * cp * cy + sr * sp * sy
        r.pose.orientation.x = sr * cp * cy - cr * sp * sy
        r.pose.orientation.y = cr * sp * cy + sr * cp * sy
        r.pose.orientation.z = cr * cp * sy - sr * sp * cy
        self.cli.call_async(r)

    def tick(self):
        dt = self.dt
        for m in self.movers:
            loop = m["loop"]
            tgt = loop[(m["seg"] + 1) % len(loop)]
            dx, dy = tgt[0] - m["p"][0], tgt[1] - m["p"][1]
            d = math.hypot(dx, dy)
            step = m["speed"] * dt
            if d <= step:
                m["p"] = list(tgt)
                m["seg"] = (m["seg"] + 1) % len(loop)
                m["v"] = [0.0, 0.0]
            else:
                m["v"] = [m["speed"] * dx / d, m["speed"] * dy / d]
                m["p"][0] += m["v"][0] * dt
                m["p"][1] += m["v"][1] * dt
            self.set_pose(m["id"], m["p"][0], m["p"][1], 0.0, 0.0, math.atan2(m["v"][1], m["v"][0]) if d > step else 0.0)
        self.t += dt
        amp = float(self.get_parameter("wave_amp").value)
        w = 2 * math.pi / float(self.get_parameter("wave_period").value)
        for f in self.floaters:
            f["p"][0] += self.drift_v[0] * dt
            f["p"][1] += self.drift_v[1] * dt
            ph = w * self.t + f["phase"]
            self.set_pose(f["id"], f["p"][0], f["p"][1], f["z"] + amp * math.sin(ph), f["roll"] + 0.08 * math.sin(ph + 0.7),
                          f["yaw"], 0.08 * math.cos(ph))
        obs = self.static + [{"p": [round(m["p"][0], 2), round(m["p"][1], 2)], "r": float(m["r"]), "h": float(m["h"]),
                              "v": [round(m["v"][0], 2), round(m["v"][1], 2)]} for m in self.movers if not m.get("hostile")]
        self.pub_obs.publish(String(data=json.dumps(obs, separators=(",", ":"))))
        self.ticks += 1
        if self.ticks % 5 == 0:
            truth = list(self.truth_static) + list(self.truth_vessels)
            for f in self.floaters:
                meta = self.truth_meta.get(f["id"], {"cls": "", "cat": "", "posture": ""})
                truth.append({"id": f["id"], "p": [round(f["p"][0], 1), round(f["p"][1], 1)], **meta})
            for mv in self.movers:
                if mv.get("hostile"):
                    truth.append({"id": mv["id"], "cls": "hostile_unit", "p": [round(mv["p"][0], 1), round(mv["p"][1], 1)]})
            self.pub_truth.publish(String(data=json.dumps(truth, separators=(",", ":"))))


def main():
    rclpy.init()
    n = Mover()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
