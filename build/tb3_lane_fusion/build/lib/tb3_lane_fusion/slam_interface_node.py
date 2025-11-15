#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from rclpy.duration import Duration

import tf2_ros
from tf2_ros import TransformException


class SlamInterfaceNode(Node):
    def __init__(self):
        super().__init__('slam_interface_node')

        # QoS for /scan (sensor data is usually best-effort)
        scan_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            scan_qos
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # TF buffer + listener (for map -> base_footprint)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Simple timer to log status once per second
        self.timer = self.create_timer(1.0, self.timer_callback)

        self.last_scan = None
        self.last_odom = None

        self.get_logger().info('slam_interface_node started')

    def scan_callback(self, msg: LaserScan):
        self.last_scan = msg

    def odom_callback(self, msg: Odometry):
        self.last_odom = msg

    def timer_callback(self):
        # 1) Log some info about the latest scan
        if self.last_scan is not None:
            num_ranges = len(self.last_scan.ranges)
            self.get_logger().info(
                f'/scan: {num_ranges} ranges, angle_min={self.last_scan.angle_min:.2f}, angle_max={self.last_scan.angle_max:.2f}'
            )
        else:
            self.get_logger().warn('No /scan received yet')

        # 2) Log some info about odometry
        if self.last_odom is not None:
            p = self.last_odom.pose.pose.position
            self.get_logger().info(
                f'/odom position: x={p.x:.2f}, y={p.y:.2f}'
            )
        else:
            self.get_logger().warn('No /odom received yet')

        # 3) Try TF lookup: map -> base_footprint
        from_frame = 'map'
        to_frame = 'base_footprint'  # change to base_link if needed

        try:
            now = Time()
            transform = self.tf_buffer.lookup_transform(
                from_frame,
                to_frame,
                now,
                timeout=Duration(seconds=0.1)
            )
            t = transform.transform.translation
            self.get_logger().info(
                f'TF {from_frame}->{to_frame}: x={t.x:.2f}, y={t.y:.2f}'
            )
        except TransformException as ex:
            self.get_logger().warn(
                f'Could not transform {from_frame} to {to_frame}: {ex}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = SlamInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
