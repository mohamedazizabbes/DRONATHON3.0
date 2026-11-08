"""Monitoring nodes: health (-> MAYDAY), navigation integrity, latency metrics.

health_node        /skyforge/odom + /skyforge/battery + /skyforge/fault (motor injection)
                   [+ PX4 /fmu/out/actuator_motors when available]  ->  /health/fault (String, "" = healthy)
nav_integrity_node /skyforge/odom (velocity) + /skyforge/gnss (sensor_msgs/NavSatFix, relayed by
                   fault_injector so spoofing can be injected)  ->  /skyforge/nav (JSON)
metrics_node       /mission/state.decision_ms, /reflex/latency_ms, /perception/latency_ms
                   ->  /metrics (Metric, 1 Hz per series) + metrics_<time>.json at shutdown
"""
from __future__ import annotations

import json
import math
import os
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, NavSatFix
from std_msgs.msg import Float32, String

from skyforge_core.monitor.health import HealthMonitor
from skyforge_core.monitor.metrics import LatencyTracker
from skyforge_core.monitor.nav_integrity import NavIntegrity
from skyforge_interfaces.msg import MissionState, Metric

from .conv import QOS_PX4, QOS_RELIABLE, jdump, node_params, now_s


class HealthNode(Node):
    def __init__(self):
        super().__init__("health_node")
        self.p = node_params(self)
        self.hm = HealthMonitor(self.p)
        self.last_odom = None
        self.motors = None
        self.injected_motor = False
        self.cell_v = None
        self.pub = self.create_publisher(String, "/health/fault", QOS_RELIABLE)
        self.create_subscription(Odometry, "/skyforge/odom", lambda m: setattr(self, "last_odom", now_s(self)), 20)
        self.create_subscription(BatteryState, "/skyforge/battery", self.on_bat, 10)
        self.create_subscription(String, "/skyforge/fault", self.on_fault, 10)
        try:
            from px4_msgs.msg import ActuatorMotors

            self.create_subscription(ActuatorMotors, "/fmu/out/actuator_motors",
                                     lambda m: setattr(self, "motors", [float(x) for x in m.control[:4] if not math.isnan(x)]), QOS_PX4)
        except ImportError:
            pass
        self.create_timer(0.1, self.tick)

    def on_bat(self, m: BatteryState):
        if m.cell_voltage:
            self.cell_v = min(m.cell_voltage)

    def on_fault(self, m: String):
        f = json.loads(m.data)
        if f.get("type") == "motor":
            self.injected_motor = True

    def tick(self):
        t = now_s(self)
        motors = [1.0, 0.6, 0.6, 0.6] if self.injected_motor else self.motors
        fault = self.hm.update(t, motors, self.last_odom, None, self.cell_v, None)
        self.pub.publish(String(data=fault or ""))


class NavIntegrityNode(Node):
    def __init__(self):
        super().__init__("nav_integrity_node")
        self.p = node_params(self)
        self.nav = NavIntegrity(self.p, self.p["mission"]["home"])
        self.declare_parameter("home_lat", 36.8065)  # set to the pad's coordinates
        self.declare_parameter("home_lon", 10.1815)
        self.lat0 = math.radians(self.get_parameter("home_lat").value)
        self.lon0 = math.radians(self.get_parameter("home_lon").value)
        self.pub = self.create_publisher(String, "/skyforge/nav", 10)
        self.create_subscription(Odometry, "/skyforge/odom", self.on_odom, 20)
        self.create_subscription(NavSatFix, "/skyforge/gnss", self.on_fix, 10)
        self.create_timer(0.1, self.tick)
        self._was_ok = True

    def on_odom(self, m: Odometry):
        v = m.twist.twist.linear
        self.nav.propagate(now_s(self), (v.x, v.y))

    def on_fix(self, m: NavSatFix):
        R = 6378137.0
        x = R * (math.radians(m.longitude) - self.lon0) * math.cos(self.lat0)
        y = R * (math.radians(m.latitude) - self.lat0)
        h = self.p["mission"]["home"]
        self.nav.gnss_fix(now_s(self), (x + h[0], y + h[1]), sats=12 if m.status.status >= 0 else 0)

    def tick(self):
        self.nav.check_timeout(now_s(self))
        x, y = self.nav.position()
        if self.nav.gnss_ok != self._was_ok:
            self._was_ok = self.nav.gnss_ok
            self.get_logger().warn(f"GNSS {'restored' if self.nav.gnss_ok else 'REJECTED'}: {self.nav.reason}")
        self.pub.publish(String(data=jdump({"gnss_ok": self.nav.gnss_ok, "source": self.nav.source, "x": x, "y": y,
                                            "reason": self.nav.reason, "innov_m": round(self.nav.last_d, 2)})))


class MetricsNode(Node):
    def __init__(self):
        super().__init__("metrics_node")
        self.p = node_params(self)
        self.tr = {k: LatencyTracker(k, 5000) for k in ("decision", "reflex", "perception", "total")}
        self._last = {"decision": 0.0, "reflex": 0.0, "perception": 0.0}
        self.pub = self.create_publisher(Metric, "/metrics", 10)
        self.create_subscription(MissionState, "/mission/state", lambda m: self._add("decision", m.decision_ms), 10)
        self.create_subscription(Float32, "/reflex/latency_ms", lambda m: self._add("reflex", m.data), 50)
        self.create_subscription(Float32, "/perception/latency_ms", lambda m: self._add("perception", m.data), 10)
        self.create_timer(1.0, self.publish)

    def _add(self, k, v):
        self.tr[k].add(float(v))
        self._last[k] = float(v)
        if k == "decision":  # chain: frame processed -> decision -> reflex command
            self.tr["total"].add(sum(self._last.values()))

    def publish(self):
        stamp = self.get_clock().now().to_msg()
        for k, t in self.tr.items():
            s = t.summary()
            m = Metric()
            m.header.stamp, m.name, m.n = stamp, k, s["n"]
            m.p50_ms, m.p95_ms, m.p99_ms, m.max_ms = s["p50_ms"], s["p95_ms"], s["p99_ms"], s["max_ms"]
            self.pub.publish(m)

    def dump(self):
        d = self.p["logging"]["dir"]
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"latency_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w", encoding="utf-8") as f:
            json.dump({k: t.summary() for k, t in self.tr.items()}, f, indent=1)


def _spin(cls):
    rclpy.init()
    n = cls()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(n, "dump"):
            n.dump()
        n.destroy_node()
        rclpy.try_shutdown()


def health_main():
    _spin(HealthNode)


def nav_main():
    _spin(NavIntegrityNode)


def metrics_main():
    _spin(MetricsNode)
