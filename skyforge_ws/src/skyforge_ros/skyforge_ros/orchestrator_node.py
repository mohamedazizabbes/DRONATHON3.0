"""orchestrator_node: LangGraph decision layer at 10 Hz (no LLM in the loop).

subscribes /skyforge/odom /skyforge/battery /skyforge/armed /skyforge/nav
           /perception/tracks /perception/zones /health/fault
           /station/goal_update (verified JSON from comm_node) /skyforge/link
           /skyforge/wind (geometry_msgs/Vector3, optional)
publishes  /plan/waypoints (nav_msgs/Path)  /mission/state  /mission/events
           /payload/release (std_msgs/String JSON)  /mission/snapshot (latched JSON)
writes     <logging.dir>/mission_log.jsonl
"""
from __future__ import annotations

import json
import math
import os
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Vector3
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32, String

from skyforge_core.brain.handover import maybe_resume
from skyforge_core.brain.orchestrator import Orchestrator
from skyforge_core.brain.state import snapshot
from skyforge_core.domain import PHASE_FR, Phase
from skyforge_core.mission_log import MissionLog
from skyforge_interfaces.msg import MissionEvent, MissionState, TrackArray, ZoneArray

from .conv import QOS_LATCHED, QOS_RELIABLE, jdump, msg_to_track, msg_to_zone, node_params, now_s


class OrchestratorNode(Node):
    def __init__(self):
        super().__init__("orchestrator_node")
        self.p = node_params(self)
        log_dir = self.p["logging"]["dir"]
        os.makedirs(log_dir, exist_ok=True)
        self.log = MissionLog(os.path.join(log_dir, f"mission_log_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"))
        self.orch = Orchestrator(self.p, self.log)
        # drone 2 after a MAYDAY: resume_snapshot:=<station's mayday_snapshot.json>
        self.declare_parameter("resume_snapshot", "")
        res = maybe_resume(self.get_parameter("resume_snapshot").value, self.orch, None, now_s(self), self.log)
        if res:
            self.get_logger().warn(f"HANDOVER: resuming mission from snapshot {res}")
        self.inputs = {"pos": (0.0, 0.0, 0.0), "vel": (0.0, 0.0, 0.0), "rpy": (0.0, 0.0, 0.0), "armed": False,
                       "battery_wh": self.p["energy"]["capacity_wh"], "battery_pct": 100.0, "link_up": True,
                       "gnss_ok": True, "nav_source": "gnss", "wind": tuple(self.p["weather"]["wind"]),
                       "visibility": float(self.p["weather"]["visibility"]), "casualties": [], "zones": [], "health_fault": None}
        self.nav_pos = None
        self.known_ids: set[str] = set()
        self.new_tracks = []
        self.goals: list[dict] = []
        self.last_wps = None
        self.all_tracks = []
        self.trajectory = []
        self.map_written = False
        self._last_snap_t = -1e9

        self.create_subscription(Odometry, "/skyforge/odom", self.on_odom, 20)
        self.create_subscription(BatteryState, "/skyforge/battery", self.on_battery, 10)
        self.create_subscription(Bool, "/skyforge/armed", lambda m: self.inputs.__setitem__("armed", m.data), 10)
        self.create_subscription(Bool, "/skyforge/link", lambda m: self.inputs.__setitem__("link_up", m.data), 10)
        self.create_subscription(String, "/skyforge/nav", self.on_nav, 10)
        self.create_subscription(Vector3, "/skyforge/wind", lambda m: self.inputs.__setitem__("wind", (m.x, m.y)), 10)
        self.create_subscription(Float32, "/skyforge/visibility", lambda m: self.inputs.__setitem__("visibility", float(m.data)), 10)
        self.pub_wind = self.create_publisher(Vector3, "/skyforge/wind", 10)
        self.pub_vis = self.create_publisher(Float32, "/skyforge/visibility", 10)
        self.create_subscription(TrackArray, "/perception/tracks", self.on_tracks, 10)
        self.create_subscription(ZoneArray, "/perception/zones", self.on_zones, 10)
        self.create_subscription(String, "/health/fault", lambda m: self.inputs.__setitem__("health_fault", m.data or None), QOS_RELIABLE)
        self.create_subscription(String, "/station/goal_update", self.on_goal, QOS_RELIABLE)
        self.create_subscription(String, "/perception/sonar_requests",
                                 lambda m: self.inputs.__setitem__("sonar_requests", [(i, tuple(p)) for i, p in json.loads(m.data)]), 10)
        self.pub_sonar = self.create_publisher(String, "/payload/sonar_cmd", QOS_RELIABLE)
        self.pub_path = self.create_publisher(Path, "/plan/waypoints", QOS_RELIABLE)
        self.pub_state = self.create_publisher(MissionState, "/mission/state", 10)
        self.pub_ev = self.create_publisher(MissionEvent, "/mission/events", QOS_RELIABLE)
        self.pub_release = self.create_publisher(String, "/payload/release", QOS_RELIABLE)
        self.pub_snap = self.create_publisher(String, "/mission/snapshot", QOS_LATCHED)
        self.create_timer(1.0 / float(self.p["orchestrator"]["rate_hz"]), self.tick)
        self.get_logger().info(f"orchestrator up: engine={self.orch.engine}, policy={self.p['mission']['triage_policy']}")

    # ------------------------------------------------------------ inputs
    def on_odom(self, m: Odometry):
        p, v = m.pose.pose.position, m.twist.twist.linear
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.inputs["pos"] = (p.x, p.y, p.z)
        if not self.trajectory or math.hypot(p.x - self.trajectory[-1][0], p.y - self.trajectory[-1][1]) > 2.0:
            self.trajectory.append((p.x, p.y))
        self.inputs["vel"] = (v.x, v.y, v.z)
        self.inputs["rpy"] = (m.twist.twist.angular.x, m.twist.twist.angular.y, yaw)

    def on_battery(self, m: BatteryState):
        self.inputs["battery_pct"] = 100.0 * m.percentage
        self.inputs["battery_wh"] = m.charge if m.charge > 0 else m.percentage * self.p["energy"]["capacity_wh"]

    def on_nav(self, m: String):
        d = json.loads(m.data)
        self.inputs["gnss_ok"] = bool(d.get("gnss_ok", True))
        self.inputs["nav_source"] = d.get("source", "gnss")
        self.nav_pos = (d["x"], d["y"]) if "x" in d else None

    def on_tracks(self, m: TrackArray):
        tracks = [msg_to_track(t) for t in m.tracks if t.confirmed]
        self.all_tracks = tracks
        for t in tracks:
            if t.id not in self.known_ids:
                self.known_ids.add(t.id)
                self.new_tracks.append(t)
        self.inputs["casualties"] = [t for t in tracks if t.cls == "injured"]

    def on_zones(self, m: ZoneArray):
        self.inputs["zones"] = [msg_to_zone(z) for z in m.zones]

    def on_goal(self, m: String):
        try:
            g = json.loads(m.data)
            g.setdefault("_recv_t", now_s(self))
            if g.get("type") == "weather":  # jury weather -> every node (perception drift, RTH, margins)
                if "wind" in g:
                    w = [float(v) for v in g["wind"]][:2]
                    self.pub_wind.publish(Vector3(x=w[0], y=w[1], z=0.0))
                if "visibility" in g:
                    self.pub_vis.publish(Float32(data=float(g["visibility"])))
            self.goals.append(g)
            self.tick()  # react in this callback: goal_update reaction < 100 ms
        except ValueError:
            self.get_logger().warn("bad goal_update JSON")

    # ------------------------------------------------------------- tick
    def tick(self):
        inp = dict(self.inputs)
        inp["t"] = now_s(self)
        if not inp["gnss_ok"] and self.nav_pos is not None:  # dead-reckoning estimate from nav_integrity
            inp["pos"] = (self.nav_pos[0], self.nav_pos[1], inp["pos"][2])
        inp["new_tracks"], self.new_tracks = self.new_tracks, []
        goals, self.goals = self.goals, []
        s = self.orch.tick(inp, goals)
        stamp = self.get_clock().now().to_msg()

        wps = s.get("waypoints") or []
        if wps != self.last_wps:
            self.last_wps = wps
        path = Path()
        path.header.stamp, path.header.frame_id = stamp, "map"
        for w in wps:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = float(w[0]), float(w[1]), float(w[2])
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)  # republished every tick = brain heartbeat for the reflex watchdog

        ms = MissionState()
        ms.header.stamp = stamp
        ph: Phase = s["phase"]
        ms.phase, ms.phase_fr = ph.value, PHASE_FR[ph]
        ms.position = [float(v) for v in s["pos"]]
        ms.battery_pct, ms.battery_wh = float(s.get("battery_pct", 0)), float(s.get("battery_wh", 0))
        ms.rth_threshold_wh, ms.coverage = float(s.get("rth_threshold_wh", 0)), float(s.get("coverage", 0))
        ms.link_up, ms.gnss_ok, ms.nav_source = bool(s.get("link_up", True)), bool(s.get("gnss_ok", True)), s.get("nav_source", "gnss")
        ms.kits, ms.target_id = int(s.get("kits", 0)), str((s.get("target") or {}).get("id") or "")
        ms.decision_ms = float(s.get("decision_ms", 0.0))
        self.pub_state.publish(ms)

        for ev in s.get("events", []):
            e = MissionEvent()
            e.header.stamp = stamp
            e.type, e.phase, e.msg, e.json = ev["type"], ev.get("phase", ""), ev.get("msg", ""), jdump(ev)
            self.pub_ev.publish(e)
            if ev["type"] in ("phase", "dec", "may"):
                self.get_logger().info(f"[{ev['phase']}] {ev['msg']}")
        if ph == Phase.LANDED and not self.map_written:
            self.map_written = True
            from skyforge_core.mission_map import export_mission_map, track_row

            out = export_mission_map(self.p["logging"]["dir"], self.orch.brain.area,
                                     [track_row(t, s.get("served_ids", [])) for t in self.all_tracks],
                                     [z.to_dict() for z in list(s.get("zones", [])) + list(s.get("manual_zones", []))],
                                     self.orch.brain.coverage, self.trajectory, s.get("served_ids", []), self.p["water"]["polygons"])
            self.get_logger().info(f"mission map written: {out}")
        for a in s.get("actions", []):
            if a["type"] == "release":
                self.pub_release.publish(String(data=jdump(a)))
            elif a["type"] == "sonar_ping":  # winch the transducer down, ping, publish /sensors/sonar
                self.pub_sonar.publish(String(data=jdump(a)))
            elif a["type"] == "mayday":
                self.pub_snap.publish(String(data=jdump(a["snapshot"])))
        if inp["t"] - self._last_snap_t >= 5.0:  # keep a fresh snapshot latched for sync
            self._last_snap_t = inp["t"]
            self.pub_snap.publish(String(data=jdump(snapshot(s))))


def main():
    rclpy.init()
    n = OrchestratorNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.log.close()
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
