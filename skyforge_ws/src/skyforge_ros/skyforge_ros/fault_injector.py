"""fault_injector: live fault injection for the demo + GNSS relay.

Relays the real GNSS (/fmu/out/vehicle_gps_position for PX4, or NavSatFix on
`gnss_in`) to /skyforge/gnss, adding a spoofing offset when asked. Other
faults are broadcast on /skyforge/fault (comm_node: jam, health_node: motor).

Trigger from a terminal:
  ros2 topic pub --once /skyforge/fault_cmd std_msgs/String '{data: "{\\"type\\":\\"jam\\",\\"on\\":true}"}'
  types: jam{on}, spoof{dx,dy,ramp_s,dur_s}, motor, gust{wx,wy,dur_s}
Or run a scripted timeline:  --ros-args -p script:=/path/faults.yaml  (list of {t, type, ...})
"""
from __future__ import annotations

import json
import math

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String

from .conv import QOS_PX4, jdump, node_params, now_s


class FaultInjector(Node):
    def __init__(self):
        super().__init__("fault_injector")
        self.p = node_params(self)
        self.declare_parameter("gnss_source", "px4")  # px4 | navsatfix
        self.declare_parameter("gnss_in", "/gnss/fix")
        self.declare_parameter("script", "")
        self.spoof = None  # (t0, dx, dy, ramp, dur)
        self.t0 = now_s(self)
        self.pub_fault = self.create_publisher(String, "/skyforge/fault", 10)
        self.pub_gnss = self.create_publisher(NavSatFix, "/skyforge/gnss", 10)
        self.pub_wind = self.create_publisher(Vector3, "/skyforge/wind_cmd", 10)
        self.create_subscription(String, "/skyforge/fault_cmd", lambda m: self.fire(json.loads(m.data)), 10)
        if self.get_parameter("gnss_source").value == "px4":
            from px4_msgs.msg import SensorGps

            self.create_subscription(SensorGps, "/fmu/out/vehicle_gps_position", self.on_px4_gps, QOS_PX4)
        else:
            self.create_subscription(NavSatFix, self.get_parameter("gnss_in").value, self.on_fix, 10)
        self.timeline = []
        path = self.get_parameter("script").value
        if path:
            import yaml

            with open(path, encoding="utf-8") as f:
                self.timeline = sorted(yaml.safe_load(f) or [], key=lambda x: x["t"])
            self.create_timer(0.1, self.run_script)
            self.get_logger().info(f"fault script: {len(self.timeline)} faults")

    def run_script(self):
        t = now_s(self) - self.t0
        while self.timeline and self.timeline[0]["t"] <= t:
            self.fire(self.timeline.pop(0))

    def fire(self, f: dict):
        self.get_logger().warn(f"FAULT INJECTED: {f}")
        if f.get("type") == "spoof":
            self.spoof = (now_s(self), float(f.get("dx", 40)), float(f.get("dy", 0)), float(f.get("ramp_s", 8)), float(f.get("dur_s", 25)))
        if f.get("type") == "gust":  # forwarded to the world (P1's wind plugin bridge)
            self.pub_wind.publish(Vector3(x=float(f.get("wx", 5)), y=float(f.get("wy", 3)), z=float(f.get("dur_s", 30))))
        if f.get("type") == "jam" and "dur_s" in f:
            self.create_timer(float(f["dur_s"]), lambda: self.pub_fault.publish(String(data=jdump({"type": "jam", "on": False}))))
        self.pub_fault.publish(String(data=jdump(f)))

    def _offset_m(self):
        if not self.spoof:
            return 0.0, 0.0
        t0, dx, dy, ramp, dur = self.spoof
        dt = now_s(self) - t0
        if dt > dur:
            self.spoof = None
            return 0.0, 0.0
        k = min(1.0, dt / max(ramp, 1e-3))
        return dx * k, dy * k

    def _publish(self, lat, lon, alt, status=0):
        ox, oy = self._offset_m()
        m = NavSatFix()
        m.header.stamp = self.get_clock().now().to_msg()
        m.latitude = lat + math.degrees(oy / 6378137.0)
        m.longitude = lon + math.degrees(ox / (6378137.0 * math.cos(math.radians(lat))))
        m.altitude = alt
        m.status.status = status
        self.pub_gnss.publish(m)

    def on_px4_gps(self, m):
        self._publish(m.latitude_deg, m.longitude_deg, m.altitude_msl_m, 0 if m.fix_type >= 3 else -1)

    def on_fix(self, m: NavSatFix):
        self._publish(m.latitude, m.longitude, m.altitude, m.status.status)


def main():
    rclpy.init()
    n = FaultInjector()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
