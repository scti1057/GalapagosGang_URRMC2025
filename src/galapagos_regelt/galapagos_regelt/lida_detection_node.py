#!/usr/bin/env python3

import os
import math
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray

from ament_index_python.packages import get_package_share_directory


class LidaDetectionNode(Node):
    """
    Detects and tracks objects from TurtleBot3 LiDAR (LaserScan).

    - Subscribes: /scan (sensor_msgs/LaserScan) by default
    - Uses YAML config (lidar_detect.yaml) for:
        * min_range_m, max_range_m
        * scan_angle_min_deg, scan_angle_max_deg  (only look in this sector)
        * distance_jump_threshold_m
        * min_cluster_points
        * max_objects
        * degree_tolerance_deg
        * max_missed_iterations
    - Clusters contiguous scan beams where:
        * ranges in [min_range_m, max_range_m]
        * consecutive points differ by <= distance_jump_threshold_m
    - For each cluster/object:
        * Computes middle angle (deg, normalized to [-180, 180]) and mean distance (m)
    - Tracking / smoothing:
        * Keeps up to max_objects tracked objects.
        * Matching between frames via angle tolerance.
        * If an object disappears, it is kept for up to max_missed_iterations cycles.
    - Publishes: r_lidar (Float64MultiArray) with up to max_objects objects:
        [deg1, dist1, deg2, dist2, deg3, dist3, ...]
      Objects are sorted by angle. Unused slots are NaN.
    """

    def __init__(self):
        super().__init__('lida_detection_node')

        # Parameters
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('config_file', 'lidar_detect.yaml')
        self.declare_parameter('max_rate_hz', 10.0)

        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        max_rate = self.get_parameter('max_rate_hz').get_parameter_value().double_value

        self._min_period = 1.0 / float(max_rate)
        self._last_processed_time = self.get_clock().now()

        # Load YAML config
        pkg_share = get_package_share_directory('galapagos_regelt')
        self._config_path = os.path.join(pkg_share, 'config', config_file)
        self.get_logger().info(f'Using lidar config: {self._config_path}')

        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        cfg = self.conf.get('lidar', {})

        self.min_range_m = float(cfg.get('min_range_m', 0.12))
        self.max_range_m = float(cfg.get('max_range_m', 1.0))

        self.scan_angle_min_deg = float(cfg.get('scan_angle_min_deg', -180.0))
        self.scan_angle_max_deg = float(cfg.get('scan_angle_max_deg', 180.0))

        self.distance_jump_threshold = float(cfg.get('distance_jump_threshold_m', 0.05))
        self.min_cluster_points = int(cfg.get('min_cluster_points', 2))

        self.max_objects = int(cfg.get('max_objects', 3))
        self.degree_tolerance_deg = float(cfg.get('degree_tolerance_deg', 10.0))
        self.max_missed_iterations = int(cfg.get('max_missed_iterations', 3))

        self.get_logger().info(
            'Lidar params: '
            f'min_range={self.min_range_m} m, max_range={self.max_range_m} m, '
            f'scan_angle=[{self.scan_angle_min_deg}, {self.scan_angle_max_deg}] deg, '
            f'distance_jump_threshold={self.distance_jump_threshold} m, '
            f'min_cluster_points={self.min_cluster_points}, '
            f'max_objects={self.max_objects}, '
            f'degree_tolerance={self.degree_tolerance_deg} deg, '
            f'max_missed_iterations={self.max_missed_iterations}'
        )

        # Last scan
        self.latest_scan = None

        # Tracked objects: list of dicts with keys:
        # 'angle_deg', 'distance', 'num_points', 'missed'
        self.tracked_objects = []

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber
        self.sub_scan = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            sensor_qos
        )
        self.get_logger().info(f'Subscribing to LaserScan: {scan_topic}')

        # Publisher
        self.pub_r_lidar = self.create_publisher(Float64MultiArray, 'r_lidar', 10)

    # === Callbacks ===

    def scan_callback(self, msg: LaserScan):
        """Store latest scan and process it."""
        self.latest_scan = msg
        self.process_scan()

    # === Core processing ===

    def process_scan(self):
        if self.latest_scan is None:
            return

        # Rate limiting
        now = self.get_clock().now()
        elapsed = (now - self._last_processed_time).nanoseconds / 1e9
        if elapsed < self._min_period:
            return
        self._last_processed_time = now

        scan = self.latest_scan
        ranges = np.array(scan.ranges, dtype=np.float32)
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        # Extract raw objects (clusters) in the configured angle range
        detected_objects = self._extract_objects(ranges, angle_min, angle_increment)

        # Update tracks with smoothing
        self._update_tracks(detected_objects)

        # Build output from tracked objects, sorted by angle
        tracks_sorted = sorted(self.tracked_objects, key=lambda t: t['angle_deg'])

        data = []
        for track in tracks_sorted[: self.max_objects]:
            data.append(track['angle_deg'])
            data.append(track['distance'])

        # Pad with NaNs if fewer than max_objects
        while len(data) < 2 * self.max_objects:
            data.append(float('nan'))

        msg_out = Float64MultiArray()
        msg_out.data = data
        self.pub_r_lidar.publish(msg_out)

    def _extract_objects(self, ranges: np.ndarray, angle_min: float, angle_increment: float):
        """
        Cluster the scan ranges into objects using a simple 1D segmentation:
        - Only ranges in [min_range_m, max_range_m] are considered.
        - Only angles in [scan_angle_min_deg, scan_angle_max_deg] are kept.
        - Consecutive valid readings belong to the same cluster if their
          distance difference <= distance_jump_threshold.
        """
        objects = []

        current_indices = []
        current_ranges = []
        prev_idx = None
        prev_r = None

        n = len(ranges)

        for i in range(n):
            r = float(ranges[i])

            # Check if this reading is valid & in distance range
            if not math.isfinite(r) or r < self.min_range_m or r > self.max_range_m:
                # Close current cluster if any
                if current_indices:
                    obj = self._cluster_to_object(current_indices, current_ranges,
                                                  angle_min, angle_increment)
                    if self._angle_in_scan_window(obj['angle_deg']) and obj['num_points'] >= self.min_cluster_points:
                        objects.append(obj)
                    current_indices = []
                    current_ranges = []
                prev_idx = None
                prev_r = None
                continue

            # Valid reading
            if not current_indices:
                # Start new cluster
                current_indices = [i]
                current_ranges = [r]
                prev_idx = i
                prev_r = r
                continue

            # Check if contiguous and within jump threshold
            if i == prev_idx + 1 and abs(r - prev_r) <= self.distance_jump_threshold:
                current_indices.append(i)
                current_ranges.append(r)
                prev_idx = i
                prev_r = r
            else:
                # Close previous cluster
                obj = self._cluster_to_object(current_indices, current_ranges,
                                              angle_min, angle_increment)
                if self._angle_in_scan_window(obj['angle_deg']) and obj['num_points'] >= self.min_cluster_points:
                    objects.append(obj)
                # Start new cluster
                current_indices = [i]
                current_ranges = [r]
                prev_idx = i
                prev_r = r

        # Close last cluster
        if current_indices:
            obj = self._cluster_to_object(current_indices, current_ranges,
                                          angle_min, angle_increment)
            if self._angle_in_scan_window(obj['angle_deg']) and obj['num_points'] >= self.min_cluster_points:
                objects.append(obj)

        return objects

    def _cluster_to_object(self, indices, distances, angle_min, angle_increment):
        """
        Convert a cluster (list of indices + distances) into a single object:
          - middle angle (deg) = average of first and last beam index,
            converted from radians to degrees and normalized to [-180, 180]
          - distance (m) = mean of distances
        """
        idx_start = indices[0]
        idx_end = indices[-1]
        mid_idx = 0.5 * (idx_start + idx_end)

        angle_rad = angle_min + angle_increment * mid_idx
        angle_deg = math.degrees(angle_rad)

        # Normalize to [-180, 180)
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg <= -180.0:
            angle_deg += 360.0

        mean_dist = float(np.mean(distances))

        return {
            'angle_deg': angle_deg,
            'distance': mean_dist,
            'num_points': len(indices),
        }

    def _angle_in_scan_window(self, angle_deg: float) -> bool:
        """Check if angle is within configured [scan_angle_min_deg, scan_angle_max_deg]."""
        return self.scan_angle_min_deg <= angle_deg <= self.scan_angle_max_deg

    def _update_tracks(self, detected_objects):
        """
        Update tracked_objects list based on current detections.
        - Match detections to existing tracks by angle within degree_tolerance_deg.
        - If a track is not matched, increment its 'missed' counter.
        - Remove tracks with missed > max_missed_iterations.
        - Add new tracks for unmatched detections if slots are free.
        """

        # Sort detections by distance so we prefer nearer objects when adding new tracks
        detected_objects = sorted(detected_objects, key=lambda o: o['distance'])

        # Mark all tracks as not updated this iteration
        for track in self.tracked_objects:
            track['updated'] = False

        # 1) Match detections to existing tracks
        for obj in detected_objects:
            best_track_idx = None
            best_ang_diff = None

            for idx, track in enumerate(self.tracked_objects):
                ang_diff = abs(obj['angle_deg'] - track['angle_deg'])
                if ang_diff <= self.degree_tolerance_deg:
                    if best_ang_diff is None or ang_diff < best_ang_diff:
                        best_ang_diff = ang_diff
                        best_track_idx = idx

            if best_track_idx is not None:
                # Update existing track
                track = self.tracked_objects[best_track_idx]
                track['angle_deg'] = obj['angle_deg']
                track['distance'] = obj['distance']
                track['num_points'] = obj['num_points']
                track['missed'] = 0
                track['updated'] = True
            else:
                # Unmatched detection -> possible new track
                if len(self.tracked_objects) < self.max_objects:
                    self.tracked_objects.append({
                        'angle_deg': obj['angle_deg'],
                        'distance': obj['distance'],
                        'num_points': obj['num_points'],
                        'missed': 0,
                        'updated': True,
                    })
                else:
                    # No free slot: ignore this new detection (we keep existing tracks until they expire)
                    pass

        # 2) Increment missed for tracks not updated, and drop stale ones
        new_tracks = []
        for track in self.tracked_objects:
            if not track.get('updated', False):
                track['missed'] = track.get('missed', 0) + 1
            if track['missed'] <= self.max_missed_iterations:
                new_tracks.append(track)

        # Clean up helper flag
        for track in new_tracks:
            track.pop('updated', None)

        self.tracked_objects = new_tracks


def main(args=None):
    rclpy.init(args=args)
    node = LidaDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
