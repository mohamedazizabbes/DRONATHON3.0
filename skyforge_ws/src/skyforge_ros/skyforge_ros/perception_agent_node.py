"""perception_agent_node: fuses jury + team vision, 2D -> 3D, tracking, drift KF,
START triage, exclusion zones. Runs on its own; the orchestrator never waits.

subscribes /vision/jury, /vision/team    std_msgs/String  detector JSON (see docs/INTERFACES.md)
           /vision/jury_array, /vision/team_array  DetectionArray (alternative)
           /cam/rgb/camera_info          sensor_msgs/CameraInfo (intrinsics override)
           /skyforge/odom                nav_msgs/Odometry (pose history for time sync)
           /skyforge/wind                geometry_msgs/Vector3 (optional)
publishes  /perception/tracks       TrackArray (all tracks, confirmed flag set)
           /perception/zones        ZoneArray  (confirmed danger -> hard zones)
           /perception/provisional  ZoneArray  (unconfirmed danger -> reflex inward veto only)
           /perception/latency_ms   std_msgs/Float32
"""
from __future__ import annotations

import bisect
import json
import math
import time
from collections import deque

import rclpy
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, MagneticField
from std_msgs.msg import Float32, String

from skyforge_core.domain import Detection, Zone
from skyforge_core.perception.adapter import map_class, parse_detections
from skyforge_core.perception.agent import PerceptionAgent
from skyforge_core.perception.projection import Camera
from skyforge_interfaces.msg import DetectionArray, TrackArray, ZoneArray

from .conv import QOS_SENSOR, node_params, now_s, tracks_msg, zones_msg


class PerceptionAgentNode(Node):
    def __init__(self):
        super().__init__("perception_agent_node")
        self.p = node_params(self)
        self.cam = Camera.from_params(self.p)
        self.agent = PerceptionAgent(self.p, self.cam)
        self.declare_parameter("resume_snapshot", "")
        snap_path = self.get_parameter("resume_snapshot").value
        if snap_path and snap_path not in ("none", "None"):
            from skyforge_core.brain.handover import load_snapshot
            from skyforge_core.domain import Cls, track_from_dict

            snap = load_snapshot(snap_path)
            tracks = []
            for c in snap.get("cas", []):
                tr = track_from_dict({**c, "cls": Cls.INJURED, "confirmed": True})
                if c.get("s"):
                    tr.attrs["marked"] = True
                tracks.append(tr)
            self.agent.seed_tracks(tracks, now_s(self))
            self.get_logger().warn(f"HANDOVER: {len(tracks)} casualties seeded from {snap_path}")
        self.wind = (0.0, 0.0)
        self.poses: deque = deque(maxlen=400)  # (t, pos, rpy)
        self.pending: dict[str, tuple[float, list[Detection]]] = {}
        self.create_subscription(String, "/vision/jury", lambda m: self.on_json(m, "jury"), QOS_SENSOR)
        self.create_subscription(String, "/vision/team", lambda m: self.on_json(m, "team"), QOS_SENSOR)
        self.create_subscription(DetectionArray, "/vision/jury_array", lambda m: self.on_array(m, "jury"), QOS_SENSOR)
        self.create_subscription(DetectionArray, "/vision/team_array", lambda m: self.on_array(m, "team"), QOS_SENSOR)
        self.create_subscription(CameraInfo, "/cam/rgb/camera_info", self.on_info, 1)
        self.create_subscription(Odometry, "/skyforge/odom", self.on_odom, 50)
        self.create_subscription(Vector3, "/skyforge/wind", lambda m: setattr(self, "wind", (m.x, m.y)), 10)
        self.pub_tracks = self.create_publisher(TrackArray, "/perception/tracks", 10)
        self.pub_zones = self.create_publisher(ZoneArray, "/perception/zones", 10)
        self.pub_prov = self.create_publisher(ZoneArray, "/perception/provisional", 10)
        self.pub_lat = self.create_publisher(Float32, "/perception/latency_ms", 10)
        # mine layers 2 (magnetometer) and 3 (dipping sonar)
        self.create_subscription(MagneticField, "/sensors/mag", self.on_mag, QOS_SENSOR)
        self.create_subscription(String, "/sensors/sonar", self.on_sonar, 10)
        self.pub_sonar_req = self.create_publisher(String, "/perception/sonar_requests", 10)
        self.pub_mag_req = self.create_publisher(String, "/perception/mag_requests", 10)
        self.create_subscription(String, "/perception/mag_inspect", self.on_mag_inspect, 10)
        self.create_timer(0.1, self.publish)  # 10 Hz outputs, prediction between frames
        self._info_logged = False

    def on_info(self, m: CameraInfo):
        self.cam = self.cam.with_K(list(m.k), m.width, m.height)
        self.agent.set_camera(self.cam)
        if not self._info_logged:
            self._info_logged = True
            self.get_logger().info(f"camera intrinsics from CameraInfo: fx={m.k[0]:.1f} {m.width}x{m.height}")

    def on_odom(self, m: Odometry):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.poses.append((t, (p.x, p.y, p.z), (m.twist.twist.angular.x, m.twist.twist.angular.y, yaw)))

    def pose_at(self, t: float):
        if not self.poses:
            return None
        ts = [x[0] for x in self.poses]
        i = bisect.bisect_left(ts, t)
        cands = [self.poses[j] for j in (i - 1, i) if 0 <= j < len(self.poses)]
        return min(cands, key=lambda x: abs(x[0] - t))

    def on_json(self, m: String, source: str):
        try:
            payload = json.loads(m.data)
        except ValueError:
            self.get_logger().warn(f"bad {source} JSON")
            return
        stamp = float(payload.get("stamp", now_s(self))) if isinstance(payload, dict) else now_s(self)
        dets = parse_detections(payload, self.cam.width, self.cam.height, source, stamp)
        self.ingest(dets, stamp)

    def on_array(self, m: DetectionArray, source: str):
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        dets = []
        for d in m.detections:
            cls = map_class(d.cls)
            if cls:
                attrs = json.loads(d.attrs_json) if d.attrs_json else {}
                dets.append(Detection(cls, d.conf, tuple(d.bbox), stamp, source, attrs))
        self.ingest(dets, stamp)

    def ingest(self, dets, stamp: float):
        t0 = time.perf_counter_ns()
        pose = self.pose_at(stamp)
        if pose is None:
            return
        _, pos, rpy = pose
        self.agent.process(dets, pos, rpy, stamp, self.wind)
        self.pub_lat.publish(Float32(data=(time.perf_counter_ns() - t0) / 1e6))

    def on_mag(self, m: MagneticField):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        pose = self.pose_at(t)
        if pose is None:
            return
        f = m.magnetic_field
        total_nt = math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z) * 1e9  # tesla -> nT
        tr = self.agent.add_mag_sample(t, pose[1], total_nt)
        if tr is not None:
            self.get_logger().info(f"magnetic contact -> {tr.id} at ({tr.pos[0]:.1f},{tr.pos[1]:.1f}) layers {tr.attrs.get('layers')}")

    def on_mag_inspect(self, m: String):
        """Orchestrator finished a close-range magnetic inspection hover: re-invert with all samples."""
        d = json.loads(m.data)
        tr = self.agent.mag_reinspect(d.get("track", ""), now_s(self))
        if tr is not None:
            self.get_logger().info(f"magnetic inspection -> {tr.id} confirmed={tr.confirmed} conf {tr.conf:.2f}")

    def on_sonar(self, m: String):
        """{"stamp": s, "contacts": [{"range": m, "bearing": rad ENU, "depth": m, "strength": 0..1}]}"""
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        t = float(d.get("stamp", now_s(self)))
        pose = self.pose_at(t)
        if pose is None:
            return
        for tr in self.agent.add_sonar_ping(pose[1][:2], d.get("contacts", []), t):
            self.get_logger().info(f"sonar contact -> {tr.id} depth {tr.attrs.get('depth')} m layers {tr.attrs.get('layers')}")

    def publish(self):
        t = now_s(self)
        self.agent.predict(t, self.wind)
        stamp = self.get_clock().now().to_msg()
        self.pub_tracks.publish(tracks_msg(self.agent.tracks(), stamp))
        self.pub_zones.publish(zones_msg(self.agent.zones(), stamp))
        prov = [Zone(f"P{i}", "provisional", c, r, source="provisional") for i, (c, r) in enumerate(self.agent.provisional_zones())]
        self.pub_prov.publish(zones_msg(prov, stamp))
        self.pub_sonar_req.publish(String(data=json.dumps([[i, list(p)] for i, p in self.agent.sonar_requests()])))
        self.pub_mag_req.publish(String(data=json.dumps([[i, list(p)] for i, p in self.agent.mag_requests()])))


def main():
    rclpy.init()
    n = PerceptionAgentNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
