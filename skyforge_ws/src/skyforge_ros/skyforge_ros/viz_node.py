"""viz_node: mission map for Foxglove / RViz (visualization_msgs/MarkerArray on /viz/markers, 2 Hz).

Draws: operating area, static obstacles, exclusion zones (red) and hostile
line-of-sight rings (yellow), provisional zones (orange outline), tracks
(casualties coloured by START: IMM red, DEL orange, MIN green, EXP black,
UNK grey; mines/weapons/hostiles red, debris brown), drift ellipses of
floating tracks (blue, radius 2 sigma), planned path (green), drone trail
(blue) and the drone with its phase label.

Foxglove: 3D panel -> /viz/markers; Plot panel -> /mission/state.battery_wh and
.rth_threshold_wh (the RTH gauge); Log panel -> /rosout (decision feed);
Plot -> /metrics.p95_ms; Raw Messages -> /comm/stats.
"""
from __future__ import annotations

import math
from collections import deque

import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from skyforge_interfaces.msg import MissionState, TrackArray, ZoneArray

from .conv import QOS_RELIABLE, node_params

START_COLORS = {"IMM": (0.9, 0.1, 0.1), "DEL": (1.0, 0.55, 0.0), "MIN": (0.1, 0.7, 0.2), "EXP": (0.1, 0.1, 0.1), "UNK": (0.6, 0.6, 0.6)}
CLASS_COLORS = {"mine": (0.85, 0.0, 0.0), "weapon": (0.85, 0.0, 0.4), "hostile_unit": (0.6, 0.0, 0.0), "debris": (0.55, 0.35, 0.15)}


def _color(rgb, a=1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(rgb[0]), float(rgb[1]), float(rgb[2]), float(a)
    return c


def _pt(x, y, z=0.0) -> Point:
    p = Point()
    p.x, p.y, p.z = float(x), float(y), float(z)
    return p


class VizNode(Node):
    def __init__(self):
        super().__init__("viz_node")
        self.p = node_params(self)
        self.tracks, self.zones, self.prov, self.path = [], [], [], []
        self.trail: deque = deque(maxlen=3000)
        self.pos = (0.0, 0.0, 0.0)
        self.phase = "INIT"
        self.create_subscription(TrackArray, "/perception/tracks", lambda m: setattr(self, "tracks", list(m.tracks)), 10)
        self.create_subscription(ZoneArray, "/perception/zones", lambda m: setattr(self, "zones", list(m.zones)), 10)
        self.create_subscription(ZoneArray, "/perception/provisional", lambda m: setattr(self, "prov", list(m.zones)), 10)
        self.create_subscription(Path, "/plan/waypoints", self.on_path, QOS_RELIABLE)
        self.create_subscription(Odometry, "/skyforge/odom", self.on_odom, 20)
        self.create_subscription(MissionState, "/mission/state", lambda m: setattr(self, "phase", m.phase_fr or m.phase), 10)
        self.pub = self.create_publisher(MarkerArray, "/viz/markers", 10)
        self.create_timer(0.5, self.publish)

    def on_path(self, m: Path):
        self.path = [(ps.pose.position.x, ps.pose.position.y, ps.pose.position.z) for ps in m.poses]

    def on_odom(self, m: Odometry):
        p = m.pose.pose.position
        self.pos = (p.x, p.y, p.z)
        if not self.trail or math.hypot(p.x - self.trail[-1][0], p.y - self.trail[-1][1]) > 1.0:
            self.trail.append((p.x, p.y, p.z))

    def _m(self, ns, mid, typ, stamp) -> Marker:
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = stamp
        m.ns, m.id, m.type, m.action = ns, mid, typ, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        return m

    def _disc(self, ns, mid, x, y, r, rgb, a, stamp, h=0.2) -> Marker:
        m = self._m(ns, mid, Marker.CYLINDER, stamp)
        m.pose.position = _pt(x, y, h / 2)
        m.scale.x = m.scale.y = 2.0 * float(r)
        m.scale.z = h
        m.color = _color(rgb, a)
        return m

    def _ring(self, ns, mid, x, y, r, rgb, a, stamp, z=0.3) -> Marker:
        m = self._m(ns, mid, Marker.LINE_STRIP, stamp)
        m.scale.x = 0.4
        m.color = _color(rgb, a)
        m.points = [_pt(x + r * math.cos(2 * math.pi * k / 48), y + r * math.sin(2 * math.pi * k / 48), z) for k in range(49)]
        return m

    def publish(self):
        stamp = self.get_clock().now().to_msg()
        out = []
        clear = self._m("all", 0, Marker.CUBE, stamp)
        clear.action = Marker.DELETEALL
        out.append(clear)
        a = self.p["mission"]["area"]
        area = self._m("area", 0, Marker.LINE_STRIP, stamp)
        area.scale.x = 0.5
        area.color = _color((0.4, 0.4, 0.4), 0.8)
        area.points = [_pt(x, y) for x, y in ((a["xmin"], a["ymin"]), (a["xmax"], a["ymin"]), (a["xmax"], a["ymax"]),
                                               (a["xmin"], a["ymax"]), (a["xmin"], a["ymin"]))]
        out.append(area)
        for i, o in enumerate(self.p.get("obstacles", [])):
            out.append(self._disc("obstacles", i, o["p"][0], o["p"][1], o["r"], (0.5, 0.5, 0.5), 0.9, stamp, h=float(o.get("h", 10.0))))
        for i, z in enumerate(self.zones):
            out.append(self._disc("zones", i, z.x, z.y, z.radius, (1.0, 0.0, 0.0), 0.25, stamp))
            if z.soft_radius > z.radius:
                out.append(self._ring("los", i, z.x, z.y, z.soft_radius, (1.0, 0.85, 0.0), 0.8, stamp))
        for i, z in enumerate(self.prov):
            out.append(self._ring("provisional", i, z.x, z.y, z.radius, (1.0, 0.5, 0.0), 0.6, stamp))
        for i, t in enumerate(self.tracks):
            if not t.confirmed:
                continue
            rgb = START_COLORS.get(t.category, START_COLORS["UNK"]) if t.cls == "injured" else CLASS_COLORS.get(t.cls, (0.5, 0.5, 0.5))
            s = self._m("tracks", i, Marker.SPHERE, stamp)
            s.pose.position = _pt(t.x, t.y, 0.8)
            s.scale.x = s.scale.y = s.scale.z = 2.0 if t.cls == "injured" else 1.2
            s.color = _color(rgb, 1.0)
            out.append(s)
            lbl = self._m("labels", i, Marker.TEXT_VIEW_FACING, stamp)
            lbl.pose.position = _pt(t.x, t.y, 3.0)
            lbl.scale.z = 2.0
            lbl.color = _color((0.0, 0.0, 0.0), 1.0)
            lbl.text = f"{t.id} {t.category}" if t.cls == "injured" else t.id
            out.append(lbl)
            if t.floating:
                out.append(self._disc("drift", i, t.x, t.y, max(2.0 * t.sigma, 0.5), (0.1, 0.3, 1.0), 0.2, stamp, h=0.1))
        if self.path:
            pm = self._m("path", 0, Marker.LINE_STRIP, stamp)
            pm.scale.x = 0.6
            pm.color = _color((0.0, 0.8, 0.2), 0.9)
            pm.points = [_pt(*self.pos)] + [_pt(*w) for w in self.path]
            out.append(pm)
        if len(self.trail) > 1:
            tm = self._m("trail", 0, Marker.LINE_STRIP, stamp)
            tm.scale.x = 0.4
            tm.color = _color((0.1, 0.2, 0.9), 0.8)
            tm.points = [_pt(*p) for p in self.trail]
            out.append(tm)
        d = self._m("drone", 0, Marker.CUBE, stamp)
        d.pose.position = _pt(*self.pos)
        d.scale.x = d.scale.y = 1.5
        d.scale.z = 0.5
        d.color = _color((0.0, 0.0, 0.0), 1.0)
        out.append(d)
        ph = self._m("phase", 0, Marker.TEXT_VIEW_FACING, stamp)
        ph.pose.position = _pt(self.pos[0], self.pos[1], self.pos[2] + 4.0)
        ph.scale.z = 3.0
        ph.color = _color((0.0, 0.0, 0.6), 1.0)
        ph.text = self.phase
        out.append(ph)
        arr = MarkerArray()
        arr.markers = out
        self.pub.publish(arr)


def main():
    rclpy.init()
    n = VizNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
