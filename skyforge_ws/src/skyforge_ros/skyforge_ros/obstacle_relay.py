"""obstacle_relay: moving obstacles (Gazebo model poses) -> /skyforge/obstacles for the reflex.

Bridge each model's pose from Gazebo, e.g.
  ros2 run ros_gz_bridge parameter_bridge /model/boat/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose
then list the models (radius r and height h in metres, ENU map frame):
  ros2 run skyforge_ros obstacle_relay --ros-args -p models:='[{"name":"boat","topic":"/model/boat/pose","r":4.0,"h":9.0}]'

Velocity is estimated by finite differences (low-passed), so the reflex can
predict the obstacle `boat_horizon_s` ahead. Published at 10 Hz as JSON:
  [{"id":"boat","p":[x,y],"r":4.0,"h":9.0,"v":[vx,vy]}]
Stale models (no pose for 2 s) are dropped.
"""
from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String

from .conv import QOS_SENSOR, jdump, now_s


class ObstacleRelay(Node):
    def __init__(self):
        super().__init__("obstacle_relay")
        self.declare_parameter("models", "[]")
        self.models = json.loads(self.get_parameter("models").value)
        self.state = {}
        for m in self.models:
            self.create_subscription(PoseStamped, m.get("topic", f"/model/{m['name']}/pose"),
                                     lambda msg, m=m: self.on_pose(m, msg), QOS_SENSOR)
        self.pub = self.create_publisher(String, "/skyforge/obstacles", 10)
        self.create_timer(0.1, self.publish)
        self.get_logger().info(f"relaying {len(self.models)} moving obstacle(s): {[m['name'] for m in self.models]}")

    def on_pose(self, m: dict, msg: PoseStamped):
        t = now_s(self)
        x, y = msg.pose.position.x, msg.pose.position.y
        st = self.state.get(m["name"])
        v = (0.0, 0.0)
        if st is not None and t - st["t"] > 1e-3:
            raw = ((x - st["p"][0]) / (t - st["t"]), (y - st["p"][1]) / (t - st["t"]))
            v = (0.7 * st["v"][0] + 0.3 * raw[0], 0.7 * st["v"][1] + 0.3 * raw[1])
        self.state[m["name"]] = {"t": t, "p": (x, y), "v": v}

    def publish(self):
        t = now_s(self)
        out = []
        for m in self.models:
            st = self.state.get(m["name"])
            if st is None or t - st["t"] > 2.0:
                continue
            out.append({"id": m["name"], "p": [round(st["p"][0], 2), round(st["p"][1], 2)], "r": float(m.get("r", 3.0)),
                        "h": float(m.get("h", 10.0)), "v": [round(st["v"][0], 3), round(st["v"][1], 3)]})
        self.pub.publish(String(data=jdump(out)))


def main():
    rclpy.init()
    n = ObstacleRelay()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
