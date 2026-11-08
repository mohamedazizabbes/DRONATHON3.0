"""comm_node: ROS side of the Communication module (no agent).

mode:=local  the CommBridge (HMAC, ring buffer, link simulator, resync) runs in
             this process and talks UDP to the ground station.
mode:=pi     the CommBridge runs on the Raspberry Pi (F12 bonus); this node only
             forwards internal ops to it over UDP and relays goals/events back.

subscribes /mission/state (-> 'tel' at comm.tel_period_s)  /mission/events (-> det/dec/may)
           /mission/snapshot (sync source)  /skyforge/odom (link simulator position)
           /skyforge/fault (JSON {"type":"jam","on":true})
publishes  /station/goal_update (verified JSON)  /skyforge/link (Bool)
           /comm/stats (JSON, 1 Hz)  /mission/events (security/link events)
"""
from __future__ import annotations

import json
import socket

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String

from skyforge_core.comm.bridge import CommBridge
from skyforge_core.comm.messages import event_to_msg
from skyforge_interfaces.msg import MissionEvent, MissionState

from .conv import QOS_LATCHED, QOS_RELIABLE, jdump, node_params, now_s


class CommNode(Node):
    def __init__(self):
        super().__init__("comm_node")
        self.p = node_params(self)
        self.declare_parameter("mode", "local")
        self.declare_parameter("pi_host", "192.168.1.50")
        self.declare_parameter("pi_port", 14700)
        self.declare_parameter("reply_port", 14701)
        self.mode = self.get_parameter("mode").value
        udp = self.p["comm"]["udp"]
        self.pub_goal = self.create_publisher(String, "/station/goal_update", QOS_RELIABLE)
        self.pub_link = self.create_publisher(Bool, "/skyforge/link", 10)
        self.pub_stats = self.create_publisher(String, "/comm/stats", 10)
        self.pub_ev = self.create_publisher(MissionEvent, "/mission/events", QOS_RELIABLE)
        self.create_subscription(MissionState, "/mission/state", self.on_state, 10)
        self.create_subscription(MissionEvent, "/mission/events", self.on_event, QOS_RELIABLE)
        self.create_subscription(String, "/mission/snapshot", lambda m: self.op({"op": "snapshot", "body": json.loads(m.data)}), QOS_LATCHED)
        self.create_subscription(Odometry, "/skyforge/odom", self.on_odom, 10)
        self.create_subscription(String, "/skyforge/fault", self.on_fault, 10)
        self._last_tel = 0.0
        self._last_pos_op = 0.0
        self.last_stats = {}

        if self.mode == "pi":
            self.pi = (self.get_parameter("pi_host").value, int(self.get_parameter("pi_port").value))
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("0.0.0.0", int(self.get_parameter("reply_port").value)))
            self.sock.setblocking(False)
            self.bridge = None
            self.get_logger().info(f"comm_node: forwarding to Raspberry Pi bridge at {self.pi}")
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("0.0.0.0", int(udp["drone_port"])))
            self.sock.setblocking(False)
            self.station = (udp["station_host"], int(udp["station_port"]))
            self.bridge = CommBridge(self.p, self._to_station, self._from_bridge, clock=lambda: now_s(self))
            self.get_logger().info(f"comm_node: local bridge, station {self.station}, ring {self.bridge.comm.buffer.capacity}")
        self.create_timer(0.02, self.poll)
        self.create_timer(1.0, self.stats)

    # ---------------------------------------------------------- plumbing
    def _to_station(self, data: bytes) -> bool:
        try:
            self.sock.sendto(data, self.station)
            return True
        except OSError:
            return False

    def op(self, obj: dict) -> None:
        if self.bridge is not None:
            self.bridge.on_op(obj)
        else:
            try:
                self.sock.sendto(jdump(obj).encode(), self.pi)
            except OSError as e:
                self.get_logger().warn(f"pi bridge unreachable: {e}", throttle_duration_sec=5.0)

    def _from_bridge(self, obj: dict) -> None:
        k = obj.get("op")
        if k == "goal":
            self.pub_goal.publish(String(data=jdump(obj["g"])))
        elif k == "event":
            e = obj["e"]
            m = MissionEvent()
            m.header.stamp = self.get_clock().now().to_msg()
            m.type, m.phase, m.msg, m.json = e.get("type", "comm"), e.get("phase", ""), e.get("msg", ""), jdump(e)
            m.json = m.json if m.type != "link" else m.json
            self.pub_ev.publish(m)
        elif k == "stats":
            self.last_stats = obj["s"]

    def poll(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except (BlockingIOError, OSError):
                break
            if self.bridge is not None:
                self.bridge.on_station(data)
            else:
                try:
                    self._from_bridge(json.loads(data))
                except ValueError:
                    pass
        if self.bridge is not None:
            self.bridge._update_link(now_s(self))  # keeps link state fresh (not counted as a message)

    # ------------------------------------------------------------ inputs
    def on_state(self, m: MissionState):
        t = now_s(self)
        if t - self._last_tel >= float(self.p["comm"]["tel_period_s"]):
            self._last_tel = t
            self.op({"op": "send", "t": "tel", "body": {
                "p": [round(v, 1) for v in m.position], "ph": m.phase, "bat": round(m.battery_pct, 1),
                "rth": round(m.rth_threshold_wh, 1), "cov": round(m.coverage, 3), "nav": m.nav_source,
                "kit": m.kits, "tg": m.target_id or None}})

    def on_event(self, m: MissionEvent):
        if m.type in ("security",) or m.json.find('"comm:') >= 0:
            return  # our own events, already logged
        try:
            ev = json.loads(m.json) if m.json else {"type": m.type, "msg": m.msg}
        except ValueError:
            return
        tb = event_to_msg(ev)
        if tb:
            self.op({"op": "send", "t": tb[0], "body": tb[1]})

    def on_odom(self, m: Odometry):
        t = now_s(self)
        if t - self._last_pos_op >= 0.2:
            self._last_pos_op = t
            p = m.pose.pose.position
            self.op({"op": "pos", "p": [p.x, p.y]})

    def on_fault(self, m: String):
        f = json.loads(m.data)
        if f.get("type") == "jam":
            self.op({"op": "jam", "on": bool(f.get("on", True))})

    def stats(self):
        if self.bridge is not None:
            self.last_stats = self.bridge.metrics()
        self.pub_stats.publish(String(data=jdump(self.last_stats)))
        self.pub_link.publish(Bool(data=bool(self.last_stats.get("link_up", True))))


def main():
    rclpy.init()
    n = CommNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
