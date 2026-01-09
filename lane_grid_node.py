#!/usr/bin/env python3
#test
import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from cv_bridge import CvBridge

import numpy as np

import tf2_ros
from tf2_ros import TransformException


class LaneGridNode(Node):
    def __init__(self):
        super().__init__('lane_grid_node')

        # Parameters
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('bev_mask_topic', 'lane_bev/mask')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')

        # BEV physical extents (must match lane_map_node!)
        self.declare_parameter('x_near_m', 0.2)
        self.declare_parameter('x_far_m', 2.0)
        self.declare_parameter('y_left_m', -0.5)
        self.declare_parameter('y_right_m', 0.5)

        self.declare_parameter('pixel_step', 4)

        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
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
        self.last_tf_warn_time = self.get_clock().now()

        # Map geometry & lane grid
        self.map_info: MapMetaData | None = None
        self.lane_grid = None  # will be a flat list of int8

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            10
        )

        self.mask_sub = self.create_subscription(
            Image,
            self.bev_mask_topic,
            self.mask_callback,
            10
        )

        # Publisher for lane occupancy grid
        self.lane_map_pub = self.create_publisher(OccupancyGrid, 'lane_map', 10)

        self.get_logger().info(
            f'lane_grid_node started. map_topic={self.map_topic}, '
            f'bev_mask_topic={self.bev_mask_topic}, base_frame={self.base_frame}'
        )

    def map_callback(self, msg: OccupancyGrid):
        # Initialize or update geometry if map changes
        info = msg.info
        if self.map_info is None:
            self.map_info = info
            self.init_lane_grid(info)
            self.get_logger().info(
                f'Initialized lane grid: {info.width}x{info.height}, res={info.resolution}'
            )
        else:
            # If geometry changes (rare with Cartographer), reinit
            if (info.width != self.map_info.width or
                    info.height != self.map_info.height or
                    info.resolution != self.map_info.resolution or
                    info.origin.position.x != self.map_info.origin.position.x or
                    info.origin.position.y != self.map_info.origin.position.y):
                self.get_logger().warn('Map geometry changed, reinitializing lane grid.')
                self.map_info = info
                self.init_lane_grid(info)

    def init_lane_grid(self, info: MapMetaData):
        size = info.width * info.height
        # -1 = unknown, 0 = observed no-lane, 100 = lane present
        self.lane_grid = [-1] * size

    def mask_callback(self, msg: Image):
        if self.map_info is None or self.lane_grid is None:
            # No map geometry yet
            return

        # Convert mask to CV image
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        H, W = mask.shape[:2]
        if H == 0 or W == 0:
            return

        # Get latest transform map <- base_frame
        now = Time()
        if not self.tf_buffer.can_transform(
            self.map_frame,
            self.base_frame,
            now,
            timeout=Duration(seconds=0.1)
        ):
            current_time = self.get_clock().now()
            if (current_time - self.last_tf_warn_time).nanoseconds > 1e9:
                self.get_logger().warn(
                    f'No transform yet between {self.map_frame} and {self.base_frame}'
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

        yaw = self.quaternion_to_yaw(q)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # 1) Clear FOV region in lane_grid (set to 0 = no lane)
        self.clear_fov_region(tx, ty, cos_yaw, sin_yaw)

        # 2) Paint lane cells from mask
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            step = max(1, self.pixel_step)
            for i in range(0, len(xs), step):
                u = xs[i]
                v = ys[i]

                # v: 0 (top, far) -> H-1 (bottom, near)
                alpha = (H - 1 - v) / float(H - 1)  # 0 = near, 1 = far
                x_local = self.x_near_m + alpha * (self.x_far_m - self.x_near_m)

                # u: 0 (left) -> W-1 (right)
                # Must match the flip logic from lane_map_node
                beta = 1.0 - (u / float(W - 1))
                y_local = self.y_left_m + beta * (self.y_right_m - self.y_left_m)

                # Into map frame
                x_map = tx + x_local * cos_yaw - y_local * sin_yaw
                y_map = ty + x_local * sin_yaw + y_local * cos_yaw

                self.set_lane_cell(x_map, y_map, 100)

        # 3) Publish lane occupancy grid
        self.publish_lane_map(msg.header.stamp)

    def clear_fov_region(self, tx, ty, cos_yaw, sin_yaw):
        """Clear (set to 0) all cells in the robot's current FOV rectangle."""

        info = self.map_info
        if info is None:
            return

        # FOV corners in base_frame
        corners_local = [
            (self.x_near_m, self.y_left_m),
            (self.x_near_m, self.y_right_m),
            (self.x_far_m, self.y_left_m),
            (self.x_far_m, self.y_right_m),
        ]

        xs_map = []
        ys_map = []
        for x_local, y_local in corners_local:
            x_map = tx + x_local * cos_yaw - y_local * sin_yaw
            y_map = ty + x_local * sin_yaw + y_local * cos_yaw
            xs_map.append(x_map)
            ys_map.append(y_map)

        min_x = min(xs_map)
        max_x = max(xs_map)
        min_y = min(ys_map)
        max_y = max(ys_map)

        # Convert bounding box to grid indices
        res = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        min_col = int((min_x - origin_x) / res)
        max_col = int((max_x - origin_x) / res)
        min_row = int((min_y - origin_y) / res)
        max_row = int((max_y - origin_y) / res)

        min_col = max(0, min_col)
        max_col = min(info.width - 1, max_col)
        min_row = max(0, min_row)
        max_row = min(info.height - 1, max_row)

        if min_col > max_col or min_row > max_row:
            return

        # Clear cells in this rectangle
        for row in range(min_row, max_row + 1):
            idx_base = row * info.width
            for col in range(min_col, max_col + 1):
                idx = idx_base + col
                # Only set to 0 if previously unknown or lane
                if self.lane_grid[idx] != 0:
                    self.lane_grid[idx] = 0

    def set_lane_cell(self, x_map, y_map, value):
        info = self.map_info
        if info is None:
            return

        res = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        col = int((x_map - origin_x) / res)
        row = int((y_map - origin_y) / res)

        if col < 0 or col >= info.width or row < 0 or row >= info.height:
            return

        idx = row * info.width + col
        self.lane_grid[idx] = value

    def publish_lane_map(self, stamp):
        if self.map_info is None or self.lane_grid is None:
            return

        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.info = self.map_info
        msg.data = self.lane_grid
        self.lane_map_pub.publish(msg)

    @staticmethod
    def quaternion_to_yaw(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = LaneGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
