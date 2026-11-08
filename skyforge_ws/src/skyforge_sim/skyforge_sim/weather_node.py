import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from skyforge_core.sim.world import DEFAULT_SCENARIO

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=2)


class Weather(Node):
    def __init__(self):
        super().__init__("sim_weather")
        self.declare_parameter("scenario_file", "")
        self.declare_parameter("turbulence", 0.6)
        sf = self.get_parameter("scenario_file").value
        sc = json.load(open(sf)) if sf else DEFAULT_SCENARIO
        self.base_wind = [float(v) for v in sc.get("wind", [0.0, 0.0])]
        self.gusts = [f for f in sc.get("faults", []) if f.get("type") == "gust"]
        self.visibility = float(sc.get("visibility", 1.0))
        self.turb = float(self.get_parameter("turbulence").value)
        self.t0 = None
        self.rng = np.random.default_rng(5)
        self.noise = np.zeros(2)
        self.pub_wind = self.create_publisher(Vector3, "/skyforge/wind", 10)
        self.pub_truth = self.create_publisher(Vector3, "/sim/wind_truth", 10)
        self.pub_img = self.create_publisher(Image, "/cam/rgb", QOS)
        self.pub_vis = self.create_publisher(Float32, "/sim/visibility_state", 10)
        self.create_subscription(Image, "/cam/rgb_raw", self.on_img, QOS)
        self.create_subscription(Float32, "/sim/visibility", lambda m: setattr(self, "visibility", max(0.05, min(1.0, m.data))), 10)
        self.create_subscription(Vector3, "/sim/wind", lambda m: setattr(self, "base_wind", [m.x, m.y]), 10)
        self.create_timer(0.1, self.tick)

    def wind_now(self, t):
        w = list(self.base_wind)
        for g in self.gusts:
            if g["t"] <= t <= g["t"] + g.get("dur", 20):
                w = [float(v) for v in g["wind"]]
        self.noise = 0.95 * self.noise + 0.05 * self.rng.normal(0, self.turb, 2)
        return [w[0] + self.noise[0], w[1] + self.noise[1]]

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        w = self.wind_now(now - self.t0)
        self.pub_truth.publish(Vector3(x=float(w[0]), y=float(w[1]), z=0.0))
        self.pub_wind.publish(Vector3(x=float(self.base_wind[0]), y=float(self.base_wind[1]), z=0.0))
        self.pub_vis.publish(Float32(data=float(self.visibility)))

    def on_img(self, m):
        v = self.visibility
        if v >= 0.98:
            self.pub_img.publish(m)
            return
        ch = len(m.data) // (m.width * m.height)
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, ch).astype(np.float32)
        haze = 1.0 - v
        rows = np.linspace(1.0, 0.6, m.height, dtype=np.float32)[:, None, None]
        veil = np.array([205.0, 210.0, 215.0], np.float32)[None, None, :ch]
        k = np.clip(haze * rows * 0.9, 0, 0.92)
        a = a * (1 - k) + veil * k
        a += self.rng.normal(0, 6.0 * haze, a.shape).astype(np.float32)
        out = Image()
        out.header, out.height, out.width, out.encoding, out.step = m.header, m.height, m.width, m.encoding, m.step
        out.data = np.clip(a, 0, 255).astype(np.uint8).tobytes()
        self.pub_img.publish(out)


def main():
    rclpy.init()
    n = Weather()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
