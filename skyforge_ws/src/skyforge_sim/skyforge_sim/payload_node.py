import json

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from ros_gz_interfaces.srv import SpawnEntity
from std_msgs.msg import String

KIT = """<?xml version="1.0"?>
<sdf version="1.9"><model name="{name}"><link name="l">
<inertial><mass>1.2</mass><inertia><ixx>0.02</ixx><iyy>0.02</iyy><izz>0.02</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
<collision name="c"><geometry><box><size>0.45 0.32 0.22</size></box></geometry></collision>
<visual name="case"><geometry><box><size>0.45 0.32 0.22</size></box></geometry><material><ambient>0.85 0.05 0.05 1</ambient><diffuse>0.9 0.08 0.08 1</diffuse><specular>0.4 0.4 0.4 1</specular></material></visual>
<visual name="cross_a"><pose>0 0 0.112 0 0 0</pose><geometry><box><size>0.26 0.07 0.005</size></box></geometry><material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><emissive>0.6 0.6 0.6 1</emissive></material></visual>
<visual name="cross_b"><pose>0 0 0.112 0 0 0</pose><geometry><box><size>0.07 0.26 0.005</size></box></geometry><material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><emissive>0.6 0.6 0.6 1</emissive></material></visual>
<visual name="beacon"><pose>0 0 0.16 0 0 0</pose><geometry><sphere><radius>0.04</radius></sphere></geometry><material><ambient>1 0.6 0 1</ambient><diffuse>1 0.6 0 1</diffuse><emissive>1 0.5 0 1</emissive></material></visual>
</link></model></sdf>"""

QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)


class Payload(Node):
    def __init__(self):
        super().__init__("sim_payload")
        self.declare_parameter("world", "skyforge")
        self.alt = 5.0
        self.n = 0
        w = self.get_parameter("world").value
        self.cli = self.create_client(SpawnEntity, f"/world/{w}/create")
        self.pub_ev = self.create_publisher(String, "/hud/event", 10)
        self.create_subscription(Odometry, "/skyforge/odom", lambda m: setattr(self, "alt", m.pose.pose.position.z), 10)
        self.create_subscription(String, "/payload/release", self.on_release, QOS)

    def on_release(self, m):
        try:
            a = json.loads(m.data)
        except ValueError:
            return
        x, y = float(a["p"][0]), float(a["p"][1])
        self.n += 1
        name = f"medkit_{self.n}_{a.get('id', 'X')}"
        if self.cli.service_is_ready():
            r = SpawnEntity.Request()
            r.entity_factory.name = name
            r.entity_factory.allow_renaming = True
            r.entity_factory.sdf = KIT.format(name=name)
            r.entity_factory.pose.position.x = x
            r.entity_factory.pose.position.y = y
            r.entity_factory.pose.position.z = max(1.0, self.alt - 0.6)
            r.entity_factory.pose.orientation.w = 1.0
            self.cli.call_async(r)
        self.pub_ev.publish(String(data=json.dumps({"kind": "kit", "id": a.get("id", "?"), "p": [round(x, 1), round(y, 1)]})))
        self.get_logger().info(f"medical kit dropped for {a.get('id')} at ({x:.1f},{y:.1f}) from {self.alt:.1f} m")


def main():
    rclpy.init()
    n = Payload()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
