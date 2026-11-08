"""ROS <-> skyforge_core conversions and small shared helpers."""
from __future__ import annotations

import json
import os

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from skyforge_core.domain import Start, Track, Zone
from skyforge_core.params import load_params
from skyforge_interfaces.msg import Track as TrackMsg
from skyforge_interfaces.msg import TrackArray, ZoneArray
from skyforge_interfaces.msg import Zone as ZoneMsg

# PX4 /fmu/out/* topics are BEST_EFFORT
QOS_PX4 = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
QOS_SENSOR = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
QOS_RELIABLE = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
QOS_LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)


def node_params(node) -> dict:
    """Every node takes one parameter: config_file (plain SKYFORGE yaml)."""
    node.declare_parameter("config_file", os.environ.get("SKYFORGE_CONFIG", ""))
    path = node.get_parameter("config_file").get_parameter_value().string_value
    return load_params(path or None)


def now_s(node) -> float:
    return node.get_clock().now().nanoseconds * 1e-9


def track_to_msg(t: Track) -> TrackMsg:
    m = TrackMsg()
    m.id, m.cls = t.id, t.cls
    m.x, m.y = float(t.pos[0]), float(t.pos[1])
    m.vx, m.vy = float(t.vel[0]), float(t.vel[1])
    m.sigma, m.conf = float(t.sigma), float(t.conf)
    m.confirmed, m.floating = bool(t.confirmed), bool(t.floating)
    m.category = t.category.value if isinstance(t.category, Start) else str(t.category)
    m.urgency = float(t.urgency)
    m.first_seen, m.last_seen = float(t.first_seen), float(t.last_seen)
    return m


def msg_to_track(m: TrackMsg) -> Track:
    try:
        cat = Start(m.category)
    except ValueError:
        cat = Start.UNKNOWN
    return Track(m.id, m.cls, (m.x, m.y), m.sigma, (m.vx, m.vy), m.conf, 3, m.confirmed, m.first_seen, m.last_seen,
                 m.floating, cat, m.urgency)


def zone_to_msg(z: Zone) -> ZoneMsg:
    m = ZoneMsg()
    m.id, m.cls = z.id, z.cls
    m.x, m.y, m.radius = float(z.center[0]), float(z.center[1]), float(z.radius)
    m.vx, m.vy = float(z.vel[0]), float(z.vel[1])
    m.soft_radius, m.source = float(z.soft_radius), z.source
    return m


def msg_to_zone(m: ZoneMsg) -> Zone:
    return Zone(m.id, m.cls, (m.x, m.y), m.radius, (m.vx, m.vy), m.soft_radius, m.source)


def tracks_msg(tracks, stamp) -> TrackArray:
    a = TrackArray()
    a.header.stamp = stamp
    a.header.frame_id = "map"
    a.tracks = [track_to_msg(t) for t in tracks]
    return a


def zones_msg(zones, stamp) -> ZoneArray:
    a = ZoneArray()
    a.header.stamp = stamp
    a.header.frame_id = "map"
    a.zones = [zone_to_msg(z) for z in zones]
    return a


def jdump(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str, ensure_ascii=False)
