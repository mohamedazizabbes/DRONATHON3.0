"""Minimal in-process stand-in for rclpy + message packages.

Lets the real skyforge_ros nodes run (constructors, callbacks, timers,
conversions) on a machine without ROS 2, on a simulated clock and a
synchronous message bus. Not a ROS emulator: just enough API surface.
"""
from __future__ import annotations

import sys
import types
from collections import defaultdict


class Auto:
    """Message object: unknown attributes auto-create nested messages."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        v = Auto()
        object.__setattr__(self, name, v)
        return v


def msg_type(name, **defaults):
    def init(self, **kw):
        for k, v in defaults.items():
            setattr(self, k, list(v) if isinstance(v, list) else v)
        for k, v in kw.items():
            setattr(self, k, v)

    return type(name, (Auto,), {"__init__": init})


class Clock:
    t = 0.0


class _Time:
    def __init__(self, t):
        self.nanoseconds = int(t * 1e9)
        self._t = t

    def to_msg(self):
        return Auto(sec=int(self._t), nanosec=int((self._t % 1) * 1e9))


class Bus:
    subs = defaultdict(list)
    last = {}
    timers = []

    @classmethod
    def reset(cls):
        cls.subs = defaultdict(list)
        cls.last = {}
        cls.timers = []


class _Pub:
    def __init__(self, topic):
        self.topic = topic

    def publish(self, m):
        Bus.last[self.topic] = m
        for cb in list(Bus.subs[self.topic]):
            cb(m)


class _Param:
    def __init__(self, v):
        self.value = v

    def get_parameter_value(self):
        return Auto(string_value=str(self.value))


class _Logger:
    def __init__(self, name):
        self.name = name

    def info(self, s, **kw):
        pass

    def warn(self, s, **kw):
        print(f"[{self.name}] WARN {s}")

    def error(self, s, **kw):
        print(f"[{self.name}] ERROR {s}")


class _Timer:
    def __init__(self, period, cb):
        self.period, self.cb, self.next = period, cb, Clock.t + period

    def cancel(self):
        if self in Bus.timers:
            Bus.timers.remove(self)


class Node:
    overrides: dict = {}

    def __init__(self, name):
        self._name = name
        self._params = {}

    def declare_parameter(self, name, default=None):
        self._params[name] = Node.overrides.get(self._name, {}).get(name, default)

    def get_parameter(self, name):
        return _Param(self._params[name])

    def create_publisher(self, typ, topic, qos):
        return _Pub(topic)

    def create_subscription(self, typ, topic, cb, qos):
        Bus.subs[topic].append(cb)

    def create_timer(self, period, cb):
        t = _Timer(period, cb)
        Bus.timers.append(t)
        return t

    def get_clock(self):
        return Auto(now=lambda: _Time(Clock.t))

    def get_logger(self):
        return _Logger(self._name)

    def destroy_node(self):
        pass


def run_timers(until: float, step: float = 0.02, hook=None):
    while Clock.t < until - 1e-9:
        Clock.t = round(Clock.t + step, 6)
        if hook:
            hook(Clock.t)
        for tm in list(Bus.timers):
            if Clock.t + 1e-9 >= tm.next:
                tm.next += tm.period
                tm.cb()


def install():
    """Register fake modules in sys.modules (call before importing skyforge_ros)."""
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *a, **k: None
    rclpy.spin = lambda n: None
    rclpy.try_shutdown = lambda: None
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = Node
    qos = types.ModuleType("rclpy.qos")
    qos.QoSProfile = lambda **kw: kw
    for n in ("DurabilityPolicy", "HistoryPolicy", "ReliabilityPolicy"):
        setattr(qos, n, Auto(BEST_EFFORT=1, RELIABLE=2, TRANSIENT_LOCAL=1, VOLATILE=2, KEEP_LAST=1))
    rclpy.node, rclpy.qos = node_mod, qos
    sys.modules.update({"rclpy": rclpy, "rclpy.node": node_mod, "rclpy.qos": qos})

    def pkg(name, **types_):
        m = types.ModuleType(name)
        for k, v in types_.items():
            setattr(m, k, v)
        root = name.split(".")[0]
        if root not in sys.modules:
            sys.modules[root] = types.ModuleType(root)
        setattr(sys.modules[root], "msg", m)
        sys.modules[name] = m

    pkg("std_msgs.msg", String=msg_type("String", data=""), Bool=msg_type("Bool", data=False), ColorRGBA=msg_type("ColorRGBA"),
        Float32=msg_type("Float32", data=0.0), Header=msg_type("Header"))
    pkg("geometry_msgs.msg", Twist=msg_type("Twist"), Vector3=msg_type("Vector3", x=0.0, y=0.0, z=0.0),
        PoseStamped=msg_type("PoseStamped"), Point=msg_type("Point", x=0.0, y=0.0, z=0.0))
    marker = msg_type("Marker", points=[], text="")
    for k, v in dict(ARROW=0, CUBE=1, SPHERE=2, CYLINDER=3, LINE_STRIP=4, TEXT_VIEW_FACING=9, ADD=0, DELETE=2, DELETEALL=3).items():
        setattr(marker, k, v)
    pkg("visualization_msgs.msg", Marker=marker, MarkerArray=msg_type("MarkerArray", markers=[]))
    pkg("nav_msgs.msg", Odometry=msg_type("Odometry"), Path=msg_type("Path", poses=[]))
    pkg("sensor_msgs.msg", BatteryState=msg_type("BatteryState", percentage=1.0, charge=0.0, capacity=0.0, cell_voltage=[]),
        CameraInfo=msg_type("CameraInfo"), NavSatFix=msg_type("NavSatFix"), Image=msg_type("Image"),
        MagneticField=msg_type("MagneticField"))
    pkg("skyforge_interfaces.msg",
        Detection=msg_type("Detection"), DetectionArray=msg_type("DetectionArray", detections=[]),
        Track=msg_type("Track"), TrackArray=msg_type("TrackArray", tracks=[]),
        Zone=msg_type("Zone"), ZoneArray=msg_type("ZoneArray", zones=[]),
        Casualty=msg_type("Casualty"), MissionState=msg_type("MissionState", position=[0.0, 0.0, 0.0]),
        MissionEvent=msg_type("MissionEvent"), Metric=msg_type("Metric"))
    ament = types.ModuleType("ament_index_python.packages")
    ament.get_package_share_directory = lambda n: "."
    sys.modules["ament_index_python"] = types.ModuleType("ament_index_python")
    sys.modules["ament_index_python.packages"] = ament
