"""reflex_node (50 Hz) + vehicle interface.

The ONLY node that knows about the autopilot. It normalises vehicle state for
everyone else and runs the classical reflex layer:

  publishes  /skyforge/odom      nav_msgs/Odometry   (ENU, map frame)
             /skyforge/battery   sensor_msgs/BatteryState (energy in Wh via capacity)
             /skyforge/armed     std_msgs/Bool
             /reflex/latency_ms  std_msgs/Float32
  subscribes /plan/waypoints     nav_msgs/Path
             /perception/zones   ZoneArray      (hard veto + escape)
             /perception/provisional ZoneArray  (inward veto only)
             /skyforge/obstacles std_msgs/String JSON [{"p":[x,y],"r":4,"v":[vx,vy],"h":9}]
             /mission/state      MissionState   (LANDED -> land + disarm)

backend:=px4   PX4 offboard over XRCE-DDS (px4_msgs), NED <-> ENU here only.
backend:=twist velocity-controlled Gazebo model (/cmd_vel + /odom), battery
               integrated from the energy model (fallback from the design doc).
"""
from __future__ import annotations

import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32, String

from skyforge_core.energy import EnergyModel
from skyforge_core.geometry import enu_to_ned, ned_to_enu, quat_ned_frd_to_enu_flu, yaw_enu_to_ned
from skyforge_core.reflex.controller import ReflexController
from skyforge_interfaces.msg import MissionState, ZoneArray

from .conv import QOS_PX4, QOS_RELIABLE, node_params, now_s


def _yaw_to_quat(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class ReflexNode(Node):
    def __init__(self):
        super().__init__("reflex_node")
        self.p = node_params(self)
        self.declare_parameter("backend", "px4")
        self.declare_parameter("px4_ns", "/fmu")
        self.declare_parameter("status_topic", "vehicle_status")
        self.declare_parameter("wind_topic", "wind")  # PX4 wind estimate (/fmu/out/wind if exported by the XRCE bridge)  # PX4 1.15; use vehicle_status_v1 with PX4 >= 1.16
        self.backend = self.get_parameter("backend").value
        self.ctrl = ReflexController(self.p)
        self.energy = EnergyModel.from_params(self.p)
        self.home = tuple(self.p["mission"]["home"])
        self.pos = [self.home[0], self.home[1], 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.rpy = (0.0, 0.0, 0.0)
        self.battery_pct = 100.0
        self.battery_wh = self.energy.capacity_wh
        self.armed = False
        self.offboard = False
        self.waypoints = []
        self.plan_t = -1e9
        self.zones = []
        self.provisional = []
        self.obstacles = []
        self.phase = "INIT"
        self._ticks = 0
        self._last_t = time.monotonic()

        self.pub_odom = self.create_publisher(Odometry, "/skyforge/odom", 10)
        self.pub_bat = self.create_publisher(BatteryState, "/skyforge/battery", 10)
        self.pub_armed = self.create_publisher(Bool, "/skyforge/armed", 10)
        self.pub_lat = self.create_publisher(Float32, "/reflex/latency_ms", 10)
        self.create_subscription(Path, "/plan/waypoints", self.on_path, QOS_RELIABLE)
        self.create_subscription(ZoneArray, "/perception/zones", lambda m: setattr(self, "zones", self._zones(m, True)), 10)
        self.create_subscription(ZoneArray, "/perception/provisional", lambda m: setattr(self, "provisional", self._zones(m, False)), 10)
        self.create_subscription(String, "/skyforge/obstacles", self.on_obstacles, 10)
        self.create_subscription(MissionState, "/mission/state", lambda m: setattr(self, "phase", m.phase), 10)

        if self.backend == "px4":
            self._setup_px4()
        else:
            self.pub_twist = self.create_publisher(Twist, "/cmd_vel", 10)
            self.create_subscription(Odometry, "/odom", self.on_twist_odom, 10)
        rate = float(self.p["reflex"]["rate_hz"])
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f"reflex_node up: backend={self.backend}, {rate:.0f} Hz")

    # ------------------------------------------------------------ inputs
    def _zones(self, m: ZoneArray, hard: bool):
        extra = self.p["planner"]["drone_radius"] + 1.0 if hard else 0.0
        return [((z.x, z.y), z.radius + extra) for z in m.zones]

    def on_path(self, m: Path):
        self.waypoints = [(ps.pose.position.x, ps.pose.position.y, ps.pose.position.z) for ps in m.poses]
        self.plan_t = now_s(self)

    def on_obstacles(self, m: String):
        try:
            self.obstacles = [(tuple(o["p"]), float(o["r"]), float(o.get("h", 100.0)), tuple(o.get("v", (0, 0))))
                              for o in json.loads(m.data)]
        except (ValueError, KeyError) as e:
            self.get_logger().warn(f"bad obstacles: {e}")

    # -------------------------------------------------------------- PX4
    def _setup_px4(self):
        from px4_msgs.msg import (BatteryStatus, OffboardControlMode, TrajectorySetpoint, VehicleCommand,
                                  VehicleOdometry, VehicleStatus)

        self.M = dict(OffboardControlMode=OffboardControlMode, TrajectorySetpoint=TrajectorySetpoint, VehicleCommand=VehicleCommand)
        ns = self.get_parameter("px4_ns").value
        self.pub_ocm = self.create_publisher(OffboardControlMode, f"{ns}/in/offboard_control_mode", QOS_PX4)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, f"{ns}/in/trajectory_setpoint", QOS_PX4)
        self.pub_cmd = self.create_publisher(VehicleCommand, f"{ns}/in/vehicle_command", QOS_PX4)
        self.create_subscription(VehicleOdometry, f"{ns}/out/vehicle_odometry", self.on_px4_odom, QOS_PX4)
        self.create_subscription(BatteryStatus, f"{ns}/out/battery_status", self.on_px4_battery, QOS_PX4)
        st = self.get_parameter("status_topic").value
        self.create_subscription(VehicleStatus, f"{ns}/out/{st}", self.on_px4_status, QOS_PX4)
        try:  # PX4's EKF wind estimate has priority over the station weather
            from px4_msgs.msg import Wind

            self.pub_wind = self.create_publisher(Vector3, "/skyforge/wind", 10)
            self.create_subscription(Wind, f"{ns}/out/{self.get_parameter('wind_topic').value}", self.on_px4_wind, QOS_PX4)
        except ImportError:
            pass

    def on_px4_odom(self, m):
        self.pos = list(ned_to_enu(m.position))
        self.pos[0] += self.home[0]
        self.pos[1] += self.home[1]
        self.vel = list(ned_to_enu(m.velocity))
        self.rpy = quat_ned_frd_to_enu_flu(m.q)
        self._publish_odom()

    def on_px4_wind(self, m):
        # NED (north, east) -> ENU (east, north)
        self.pub_wind.publish(Vector3(x=float(m.windspeed_east), y=float(m.windspeed_north), z=0.0))

    def on_px4_battery(self, m):
        if m.remaining >= 0:
            self.battery_pct = 100.0 * float(m.remaining)
            self.battery_wh = self.energy.capacity_wh * float(m.remaining)

    def on_px4_status(self, m):
        self.armed = m.arming_state == 2  # ARMING_STATE_ARMED
        self.offboard = m.nav_state == 14  # NAVIGATION_STATE_OFFBOARD

    def _cmd(self, command: int, p1: float = 0.0, p2: float = 0.0):
        c = self.M["VehicleCommand"]()
        c.command, c.param1, c.param2 = command, float(p1), float(p2)
        c.target_system = c.target_component = c.source_system = c.source_component = 1
        c.from_external = True
        c.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_cmd.publish(c)

    def _send_px4(self, vel_enu, yaw_enu):
        ts = int(self.get_clock().now().nanoseconds / 1000)
        ocm = self.M["OffboardControlMode"]()
        ocm.timestamp, ocm.position, ocm.velocity = ts, False, True
        self.pub_ocm.publish(ocm)
        sp = self.M["TrajectorySetpoint"]()
        sp.timestamp = ts
        nan = float("nan")
        sp.position = [nan, nan, nan]
        sp.velocity = [float(v) for v in enu_to_ned(vel_enu)]
        sp.yaw = float(yaw_enu_to_ned(yaw_enu))
        self.pub_sp.publish(sp)
        # offboard handshake: stream setpoints >= 1 s, then offboard + arm
        self._ticks += 1
        if self.phase not in ("LANDED",) and self._ticks > 50 and (not self.offboard or not self.armed) and self._ticks % 25 == 0:
            self._cmd(176, 1.0, 6.0)  # VEHICLE_CMD_DO_SET_MODE: custom, offboard
            self._cmd(400, 1.0)  # VEHICLE_CMD_COMPONENT_ARM_DISARM: arm

    # ------------------------------------------------- twist fallback
    def on_twist_odom(self, m: Odometry):
        p, v, q = m.pose.pose.position, m.twist.twist.linear, m.pose.pose.orientation
        self.pos = [p.x, p.y, p.z]
        self.vel = [v.x, v.y, v.z]
        self.rpy = (0.0, 0.0, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        self.armed = True
        self._publish_odom()

    # ------------------------------------------------------------- loop
    def _publish_odom(self):
        o = Odometry()
        o.header.stamp = self.get_clock().now().to_msg()
        o.header.frame_id, o.child_frame_id = "map", "base_link"
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = self.pos
        qx, qy, qz, qw = _yaw_to_quat(self.rpy[2])
        o.pose.pose.orientation.x, o.pose.pose.orientation.y, o.pose.pose.orientation.z, o.pose.pose.orientation.w = qx, qy, qz, qw
        # roll/pitch for the perception agent travel in the covariance-free twist.angular field
        o.twist.twist.angular.x, o.twist.twist.angular.y = float(self.rpy[0]), float(self.rpy[1])
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = self.vel
        self.pub_odom.publish(o)

    def tick(self):
        t0 = time.perf_counter_ns()
        t = now_s(self)
        dt = time.monotonic() - self._last_t
        self._last_t = time.monotonic()
        obst = [((c[0], c[1]), r + self.p["planner"]["drone_radius"], v) for c, r, h, v in self.obstacles if h > self.pos[2] - 2.0]
        out = self.ctrl.step(t, self.pos, self.vel, self.waypoints, self.plan_t, self.zones, obst,
                             battery_pct=self.battery_pct, provisional=self.provisional)
        vel = out.vel
        if self.phase == "LANDED":
            vel = (0.0, 0.0, -0.5) if self.pos[2] > 0.15 else (0.0, 0.0, 0.0)
        if self.backend == "px4":
            self._send_px4(vel, out.yaw)
            if self.phase == "LANDED" and self.armed and self.pos[2] < 0.3:
                self._cmd(400, 0.0)  # disarm on the pad
        else:
            tw = Twist()
            tw.linear.x, tw.linear.y, tw.linear.z = vel
            self.pub_twist.publish(tw)
            # battery integrated from the calibrated energy model
            if self.pos[2] > 0.05:
                self.battery_wh -= self.energy.power_w(math.hypot(self.vel[0], self.vel[1])) * dt / 3600.0
                self.battery_pct = 100.0 * self.battery_wh / self.energy.capacity_wh
        b = BatteryState()
        b.header.stamp = self.get_clock().now().to_msg()
        b.percentage = float(self.battery_pct) / 100.0
        b.capacity = float(self.energy.capacity_wh)  # Wh (documented convention)
        b.charge = float(self.battery_wh)  # Wh
        self.pub_bat.publish(b)
        self.pub_armed.publish(Bool(data=bool(self.armed)))
        self.pub_lat.publish(Float32(data=(time.perf_counter_ns() - t0) / 1e6))


def main():
    rclpy.init()
    n = ReflexNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
