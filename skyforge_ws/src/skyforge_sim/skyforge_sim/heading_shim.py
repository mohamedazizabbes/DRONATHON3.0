import math

import rclpy
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class HeadingShim(Node):
    def __init__(self):
        super().__init__("heading_shim")
        self.declare_parameter("min_speed", 0.8)
        self.declare_parameter("k_yaw", 1.5)
        self.declare_parameter("max_yaw_rate", 1.2)
        self.declare_parameter("wind_gain", 0.35)
        self.wind_gain = float(self.get_parameter("wind_gain").value)
        self.wind = (0.0, 0.0)
        self.min_speed = float(self.get_parameter("min_speed").value)
        self.k = float(self.get_parameter("k_yaw").value)
        self.max_rate = float(self.get_parameter("max_yaw_rate").value)
        self.yaw = 0.0
        self.target_yaw = 0.0
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel_gz", 10)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.create_subscription(Odometry, "/odom_gz", self.on_odom, 50)
        self.create_subscription(Vector3, "/sim/wind_truth", lambda m: setattr(self, "wind", (m.x, m.y)), 10)
        self.alt = 0.0

    def on_cmd(self, m: Twist):
        vx, vy, vz = m.linear.x, m.linear.y, m.linear.z
        if math.hypot(vx, vy) > self.min_speed:
            self.target_yaw = math.atan2(vy, vx)
        if self.alt > 0.5:
            vx += self.wind_gain * self.wind[0]
            vy += self.wind_gain * self.wind[1]
        rate = max(-self.max_rate, min(self.max_rate, self.k * wrap(self.target_yaw - self.yaw)))
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        out = Twist()
        out.linear.x = c * vx + s * vy
        out.linear.y = -s * vx + c * vy
        out.linear.z = vz
        out.angular.z = rate
        self.pub_cmd.publish(out)

    def on_odom(self, m: Odometry):
        self.yaw = yaw_of(m.pose.pose.orientation)
        self.alt = m.pose.pose.position.z
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        bx, by = m.twist.twist.linear.x, m.twist.twist.linear.y
        m.twist.twist.linear.x = c * bx - s * by
        m.twist.twist.linear.y = s * bx + c * by
        self.pub_odom.publish(m)


def main():
    rclpy.init()
    n = HeadingShim()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
