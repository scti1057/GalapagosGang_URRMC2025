#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class LaneCalibRectNode(Node):
    def __init__(self):
        super().__init__('lane_calib_rect_node')

        # Coordinate frame for the rectangles (e.g. "base_footprint" or your LiDAR frame)
        self.declare_parameter('frame_id', 'base_footprint')

        # Marker topic name
        self.declare_parameter('marker_topic', 'lane_calib_rect')

        # Geometry of the rectangles in front of the sensor (in meters)
        # Default: 14 cm..41.5 cm in x
        self.declare_parameter('x_near', 0.14)   # 14 cm from sensor
        self.declare_parameter('x_far', 0.415)   # 41.5 cm from sensor

        # First rectangle width: 19.5 cm
        self.declare_parameter('width', 0.195)

        # Second rectangle width: 24.5 cm
        self.declare_parameter('width2', 0.245)

        # Line thickness in RViz
        self.declare_parameter('line_width', 0.002)  # 1 cm

        marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        self.marker_pub = self.create_publisher(Marker, marker_topic, 1)

        # Publish regularly so the markers stay in RViz
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

        self.get_logger().info(
            f"LaneCalibRectNode started. Publishing rectangles on '{marker_topic}'."
        )

    def make_rectangle_marker(self, frame_id, x_near, x_far, width, line_width,
                              ns: str, marker_id: int):
        half_w = width / 2.0

        # Rectangle corners (closing back to the first point)
        points = [
            (x_near, -half_w, 0.0),
            (x_near,  half_w, 0.0),
            (x_far,   half_w, 0.0),
            (x_far,  -half_w, 0.0),
            (x_near, -half_w, 0.0),
        ]

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Line width (x is used for LINE_STRIP thickness)
        marker.scale.x = line_width

        # Color: green, fully opaque
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # No extra rotation; we define points directly in the frame
        marker.pose.orientation.w = 1.0

        marker.points = []
        for (x, y, z) in points:
            pt = Point()
            pt.x = x
            pt.y = y
            pt.z = z
            marker.points.append(pt)

        return marker

    def timer_callback(self):
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        x_near = self.get_parameter('x_near').get_parameter_value().double_value
        x_far = self.get_parameter('x_far').get_parameter_value().double_value
        width1 = self.get_parameter('width').get_parameter_value().double_value
        width2 = self.get_parameter('width2').get_parameter_value().double_value
        line_width = self.get_parameter('line_width').get_parameter_value().double_value

        # First rectangle (narrower, 19.5 cm default)
        marker1 = self.make_rectangle_marker(
            frame_id=frame_id,
            x_near=x_near,
            x_far=x_far,
            width=width1,
            line_width=line_width,
            ns="lane_calib_rect",
            marker_id=0
        )

        # Second rectangle (wider, 24.5 cm default)
        marker2 = self.make_rectangle_marker(
            frame_id=frame_id,
            x_near=x_near,
            x_far=x_far,
            width=width2,
            line_width=line_width,
            ns="lane_calib_rect",
            marker_id=1
        )

        self.marker_pub.publish(marker1)
        self.marker_pub.publish(marker2)


def main(args=None):
    rclpy.init(args=args)
    node = LaneCalibRectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
