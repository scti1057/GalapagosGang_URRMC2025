#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from cv_bridge import CvBridge
import cv2
import numpy as np

import tf2_ros
from tf2_ros import TransformException


class LaneMapNode(Node):
    def __init__(self):
        super().__init__('lane_map_node')

        # Parameters
        self.declare_parameter('bev_mask_topic', 'lane_bev/mask')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')

        # BEV physical extents in front of robot (meters)
        self.declare_parameter('x_near_m', 0.2)   # near distance (bottom of BEV)
        self.declare_parameter('x_far_m', 2.0)    # far distance (top of BEV)
        self.declare_parameter('y_left_m', -0.5)  # left edge
        self.declare_parameter('y_right_m', 0.5)  # right edge

        # Downsample factor for pixels
        self.declare_parameter('pixel_step', 4)

        self.bev_mask_topic = self.get_parameter('bev_mask_topic').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value

        self.x_near_m = float(self.get_parameter('x_near_m').get_parameter_value().double_value)
        self.x_far_m = float(self.get_parameter('x_far_m').get_parameter_value().double_value)
        self.y_left_m = float(self.get_parameter('y_left_m').get_parameter_value().double_value)
        self.y_right_m = float(self.get_parameter('y_right_m').get_parameter_value().double_value)

        self.pixel_step = int(self.get_parameter('pixel_step').get_parameter_value().integer_value)

        self.bridge = CvBridge()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Sub + pub
        self.mask_sub = self.create_subscription(
            Image,
            self.bev_mask_topic,
            self.mask_callback,
            10
        )

        self.marker_pub = self.create_publisher(Marker, 'lane_points', 10)

        self.get_logger().info(
            f'lane_map_node started, listening to {self.bev_mask_topic}, '
            f'base_frame={self.base_frame}, map_frame={self.map_frame}'
        )

        self.last_tf_warn_time = self.get_clock().now()

    def mask_callback(self, msg: Image):
        # Convert mask image to OpenCV
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        if mask is None:
            return

        H, W = mask.shape[:2]
        if H == 0 or W == 0:
            return

        # Check if TF is ready (latest transform)
        now = Time()
        if not self.tf_buffer.can_transform(
            self.map_frame,
            self.base_frame,
            now,
            timeout=Duration(seconds=0.1)
        ):
            # Limit warning rate to ~1 Hz
            current_time = self.get_clock().now()
            if (current_time - self.last_tf_warn_time).nanoseconds > 1e9:
                self.get_logger().warn(
                    f'No transform yet between {self.map_frame} and {self.base_frame}, '
                    'TF tree may not be ready.'
                )
                self.last_tf_warn_time = current_time
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                now,
                timeout=Duration(seconds=0.1)
            )
        except TransformException as ex:
            current_time = self.get_clock().now()
            if (current_time - self.last_tf_warn_time).nanoseconds > 1e9:
                self.get_logger().warn(
                    f'No transform {self.map_frame}->{self.base_frame}: {ex}'
                )
                self.last_tf_warn_time = current_time
            return

        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        q = transform.transform.rotation

        # Compute yaw from quaternion (assuming planar motion)
        yaw = self.quaternion_to_yaw(q)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # Prepare marker
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = msg.header.stamp  # time not super critical
        marker.ns = 'lane_points'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        marker.scale.x = 0.02  # point size
        marker.scale.y = 0.02

        # Yellow color
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.points = []

        # Find non-zero pixels and downsample
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            # publish empty marker to clear previous points
            self.marker_pub.publish(marker)
            return

        step = max(1, self.pixel_step)

        for i in range(0, len(xs), step):
            u = xs[i]  # column
            v = ys[i]  # row

            # Map pixels (u,v) to local BEV coordinates in base_frame
            # v: 0 (top, far) -> H-1 (bottom, near)
            alpha = (H - 1 - v) / float(H - 1)  # 0 = near, 1 = far
            x_local = self.x_near_m + alpha * (self.x_far_m - self.x_near_m)

            # u: 0 (left) -> W-1 (right)
            beta = u / float(W - 1)
            y_local = self.y_left_m + beta * (self.y_right_m - self.y_left_m)

            # Rotate + translate into map frame
            x_map = tx + x_local * cos_yaw - y_local * sin_yaw
            y_map = ty + x_local * sin_yaw + y_local * cos_yaw

            p = Point()
            p.x = x_map
            p.y = y_map
            p.z = 0.0
            marker.points.append(p)

        self.marker_pub.publish(marker)

    @staticmethod
    def quaternion_to_yaw(q):
        # q: geometry_msgs/Quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = LaneMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
