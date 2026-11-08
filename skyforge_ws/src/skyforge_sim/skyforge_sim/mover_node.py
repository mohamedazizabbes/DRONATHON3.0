import json
import math

import rclpy
from geometry_msgs.msg import Vector3
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
        sc = DEFAULT_SCENARIO
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
                    self.floaters.append({"id": o["id"], "p": list(o["p"]), "z": z, "roll": math.pi / 2 if rot else 0.0})
        self.static = [{"p": list(o["p"]), "r": float(o["r"]), "h": float(o["h"]), "v": [0.0, 0.0]} for o in sc.get("obstacles", [])]
        self.movers = [{**m, "p": list(m["loop"][0]), "seg": 0, "v": [0.0, 0.0]} for m in sc.get("moving_obstacles", [])]
        w = self.get_parameter("world").value
        self.cli = self.create_client(SetEntityPose, f"/world/{w}/set_pose")
        self.pub_obs = self.create_publisher(String, "/skyforge/obstacles", 10)
        self.pub_wind = self.create_publisher(Vector3, "/skyforge/wind", 10)
        self.dt = 1.0 / float(self.get_parameter("rate").value)
        self.create_timer(self.dt, self.tick)

    def set_pose(self, name, x, y, z, roll, yaw):
        if not self.cli.service_is_ready():
            return
        r = SetEntityPose.Request()
        r.entity.name = name
        r.entity.type = Entity.MODEL
        r.pose.position.x, r.pose.position.y, r.pose.position.z = float(x), float(y), float(z)
        cr, sr, cy, sy = math.cos(roll / 2), math.sin(roll / 2), math.cos(yaw / 2), math.sin(yaw / 2)
        r.pose.orientation.w, r.pose.orientation.x = cr * cy, sr * cy
        r.pose.orientation.y, r.pose.orientation.z = sr * sy, cr * sy
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
        for f in self.floaters:
            f["p"][0] += self.drift_v[0] * dt
            f["p"][1] += self.drift_v[1] * dt
            self.set_pose(f["id"], f["p"][0], f["p"][1], f["z"], f["roll"], 0.0)
        obs = self.static + [{"p": [round(m["p"][0], 2), round(m["p"][1], 2)], "r": float(m["r"]), "h": float(m["h"]),
                              "v": [round(m["v"][0], 2), round(m["v"][1], 2)]} for m in self.movers]
        self.pub_obs.publish(String(data=json.dumps(obs, separators=(",", ":"))))
        self.pub_wind.publish(Vector3(x=float(self.wind[0]), y=float(self.wind[1]), z=0.0))


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
