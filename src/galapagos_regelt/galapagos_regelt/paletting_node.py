#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import yaml
from typing import List, Dict, Optional

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration

from std_msgs.msg import Bool, Float64, Float64MultiArray
from vision_msgs.msg import Detection2DArray
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist


class PalettingNode(Node):
    """
    Paletting state machine.

    Subscribes:
      - pal_free (std_msgs/Bool): from blue_pal_detect_node (currently only monitored)
      - r_lidar (std_msgs/Float64MultiArray): [angle0, dist0, angle1, dist1, ...]
      - /yolo/sign_detections (Detection2DArray): YOLO signs; we use classes "1" and "2"

    Publishes:
      - yaw_init_pal (Float64): current angle of closest object [deg]
      - yaw_tar_pal  (Float64): desired angle [deg] (0° or 90°)
      - /electromagnet_cmd (Bool): electromagnet on/off (publisher only, no logic yet)

    Flow (first part):

      1. Wait until we see sign "1" or "2".
         If both are visible, store which one is closer (bigger bbox)
         as sign_order_initial = [first, second].

      2. Wait until the closest LiDAR object has |angle| >= start_angle_threshold_deg (e.g. 83°).

      3. Align that closest object to 0°:
           yaw_init_pal = closest_angle_deg
           yaw_tar_pal  = 0°
         until |angle| <= angle_tolerance_deg.

      4. Pause for pause_between_steps_s.

      5. Check YOLO:
           - If biggest sign is "1" -> DONE for now.
           - If biggest sign is "2" -> continue.

      6. Align closest object to 90°:
           yaw_init_pal = closest_angle_deg
           yaw_tar_pal  = 90°
         until |angle - 90°| <= angle_tolerance_deg.

      7. Pause for pause_after_90_s (default 2s).

      8. Wait again for closest LiDAR with |angle| >= start_angle_threshold_deg.

      9. Align again to 0° (same as 3) and pause for pause_between_steps_s.

     10. Check YOLO: if biggest sign is "1" -> DONE for now.

    All timings / thresholds come from paletting.yaml.
    """

    def __init__(self):
        super().__init__('paletting_node')

        # --- Parameters ---
        self.declare_parameter('config_file', 'paletting.yaml')
        self.declare_parameter('max_rate_hz', 10.0)
        self.declare_parameter('debug_visualization', True)
        self.declare_parameter('camera_topic', '/camera/image_raw/compressed')

        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        max_rate = self.get_parameter('max_rate_hz').get_parameter_value().double_value

        self._min_period = 1.0 / float(max_rate)
        self._last_step_time = self.get_clock().now()

        # Load YAML config
        try:
            pkg_share = get_package_share_directory('galapagos_regelt')
            self._config_path = os.path.join(pkg_share, 'config', config_file)
            self.get_logger().info(f'Using paletting config: {self._config_path}')
            with open(self._config_path, 'r') as f:
                self.conf = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().warn(
                f'Could not load paletting config "{config_file}", using defaults only: {e}'
            )
            self.conf = {}

        pal_cfg = self.conf.get('paletting', {})

        self.start_angle_threshold_deg = float(pal_cfg.get('start_angle_threshold_deg', 83.0))
        self.angle_tolerance_deg = float(pal_cfg.get('angle_tolerance_deg', 5.0))
        self.pause_between_steps_s = float(pal_cfg.get('pause_between_steps_s', 1.0))
        self.pause_after_90_s = float(pal_cfg.get('pause_after_90_s', 2.0))
        self.yaw_target0_deg = float(pal_cfg.get('yaw_target0_deg', 0.0))
        self.yaw_target90_deg = float(pal_cfg.get('yaw_target90_deg', 90.0))

        # --- NEW: pallet approach distances + x_pal values ---
        self.forward_dist_threshold_m = float(pal_cfg.get('forward_dist_threshold_m', 0.22))
        self.back_dist_threshold_m = float(pal_cfg.get('back_dist_threshold_m', 0.37))
        # x_pal value for "drive straight" (pixels, same semantics as x_tar)
        self.x_pal_forward = float(pal_cfg.get('x_pal_forward', 320.0))
        self.x_pal_backward = float(pal_cfg.get('x_pal_backward', 320.0))
        # wait time after turning the magnet on
        self.magnet_wait_s = float(pal_cfg.get('magnet_wait_s', 2.0))

        self.require_pal_free_for_start = bool(pal_cfg.get('require_pal_free_for_start', False))
        self.enabled = bool(pal_cfg.get('enabled', True))
        self.debug_logging = bool(pal_cfg.get('debug_logging', True))
        # --- Debug image stuff ---
        self.bridge = None
        self.debug_image = None
        self._debug_window = 'paletting_debug'

                # --- New geometry + pallet/drop parameters ---
        self.image_width_px = float(pal_cfg.get('image_width_px', 640.0))
        self.image_center_x = self.image_width_px / 2.0

        # Distances [m] used for both first pickup and second drop
        self.drop_forward_dist_threshold_m = float(
            pal_cfg.get('drop_forward_dist_threshold_m', 0.22)
        )
        self.drop_back_dist_threshold_m = float(
            pal_cfg.get('drop_back_dist_threshold_m', 0.37)
        )

        # Sideways motion along white lane after we picked up the pallet
        self.side_offset_px = float(pal_cfg.get('side_offset_px', 80.0))
        self.side_drive_duration_s = float(pal_cfg.get('side_drive_duration_s', 4.0))

        # Magnet timing
        self.magnet_after_grab_wait_s = float(
            pal_cfg.get('magnet_after_grab_wait_s', 2.0)
        )
        self.magnet_after_release_wait_s = float(
            pal_cfg.get('magnet_after_release_wait_s', 2.0)
        )

        # Backwards speed (negative -> reverse)
        self.back_linear_speed_mps = float(
            pal_cfg.get('back_linear_speed_mps', -0.08)
        )
        # Forward speed (positive) for straight approach / drop
        self.forward_linear_speed_mps = float(
            pal_cfg.get('forward_linear_speed_mps', 0.08)
        )


        dbg_vis_param = self.get_parameter('debug_visualization').get_parameter_value().bool_value
        self.debug_visualization = bool(pal_cfg.get('debug_visualization', dbg_vis_param))
        camera_topic_param = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.camera_topic = pal_cfg.get('camera_topic', camera_topic_param)

        # --- Internal state ---

        # latest sensor values
        self.pal_free: Optional[bool] = None
        self.lidar_objects: List[Dict[str, float]] = []
        self.yolo_areas: Dict[str, float] = {}  # '1'/'2' -> area
        self.latest_biggest_sign: Optional[str] = None

        # initial order of signs, e.g. [1, 2] or [2, 1] (for later use / debug)
        self.sign_order_initial: Optional[List[int]] = None

        # which side to go after pickup ('left' or 'right')
        self.lane_side: Optional[str] = None
        self.post_pick_twist_target_deg: Optional[float] = None
        self.side_drive_start_time = None

        # lane info for sideways motion
        self.x_white_far: Optional[float] = None

        # state machine
        self.state: str = 'IDLE'
        self.pause_until: Optional[Duration] = None
        self._pause_next_state: Optional[str] = None

        if self.debug_logging:
            self.get_logger().info(
                f'PalettingNode params: start_angle_threshold={self.start_angle_threshold_deg}deg, '
                f'angle_tolerance={self.angle_tolerance_deg}deg, '
                f'pause_between_steps={self.pause_between_steps_s}s, '
                f'pause_after_90={self.pause_after_90_s}s, enabled={self.enabled}, '
                f'require_pal_free_for_start={self.require_pal_free_for_start}, '
                f'debug_visualization={self.debug_visualization}, '
                f'camera_topic={self.camera_topic}'
            )

        # --- ROS wiring ---

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Debug camera subscription (optional)
        if self.debug_visualization:
            self.bridge = CvBridge()
            self.sub_debug_cam = self.create_subscription(
                CompressedImage,
                self.camera_topic,
                self.debug_image_callback,
                sensor_qos
            )
            self.get_logger().info(
                f'Paletting debug visualization enabled, subscribing to camera: {self.camera_topic}'
            )

        # Subscribers
        from std_msgs.msg import Bool as BoolMsg   # to avoid name clash with type hints
        self.sub_pal_free = self.create_subscription(
            BoolMsg,
            'pal_free',
            self.pal_free_callback,
            sensor_qos
        )

        self.sub_r_lidar = self.create_subscription(
            Float64MultiArray,
            'r_lidar',
            self.r_lidar_callback,
            sensor_qos
        )

        self.sub_yolo = self.create_subscription(
            Detection2DArray,
            '/yolo/sign_detections',
            self.yolo_callback,
            sensor_qos
        )

         # far white line for sideways motion
        self.sub_x_white_far = self.create_subscription(
            Float64,
            'lane/x_white_far',
            self.x_white_far_callback,
            sensor_qos
        )


        # Publishers
        self.pub_yaw_init_pal = self.create_publisher(Float64, 'yaw_init_pal', 10)
        self.pub_yaw_tar_pal = self.create_publisher(Float64, 'yaw_tar_pal', 10)
        self.pub_electromagnet = self.create_publisher(Bool, '/electromagnet_cmd', 10)
        # x_pal -> goes via control_node to drive_node
        self.pub_x_pal = self.create_publisher(Float64, 'x_pal', 10)
        # direct cmd_vel for backwards phases
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        # NEW: flag to mute DriveNode while PalettingNode sends cmd_vel directly
        self.pub_pal_control_active = self.create_publisher(Bool, 'pal_control_active', 10)



        # Main timer
        self.timer = self.create_timer(0.05, self.step)  # ~20 Hz, rate-limited by max_rate_hz

        self.get_logger().info('PalettingNode started (state machine: IDLE -> ... -> DONE).')

    # === Callbacks ===

    def debug_image_callback(self, msg: CompressedImage):
        """Store latest camera image for debug visualization."""
        if self.bridge is None:
            return
        try:
            self.debug_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().warn(f'Paletting: failed to convert debug image: {e}')

    def pal_free_callback(self, msg: Bool):
        self.pal_free = bool(msg.data)

    def x_white_far_callback(self, msg: Float64):
        self.x_white_far = float(msg.data)


    def r_lidar_callback(self, msg: Float64MultiArray):
        """Parse r_lidar into a list of objects [{'angle_deg': .., 'distance': ..}, ...]."""
        data = list(msg.data)
        objects: List[Dict[str, float]] = []
        for i in range(0, len(data), 2):
            if i + 1 >= len(data):
                break
            ang = data[i]
            dist = data[i + 1]
            if math.isnan(ang) or math.isnan(dist):
                continue
            objects.append({'angle_deg': float(ang), 'distance': float(dist)})
        self.lidar_objects = objects

    def yolo_callback(self, msg: Detection2DArray):
        """
        Track YOLO detections for classes "1" and "2".
        We keep the largest bbox area for each class and the overall largest.
        Also updates an OpenCV debug window if enabled.
        """
        areas: Dict[str, float] = {}
        boxes: List[Dict[str, float]] = []

        for det in msg.detections:
            if not det.results:
                continue
            cls_id = det.results[0].hypothesis.class_id
            if cls_id not in ('1', '2'):
                continue
            w = float(det.bbox.size_x)
            h = float(det.bbox.size_y)
            if w <= 0.0 or h <= 0.0:
                continue
            area = w * h
            prev = areas.get(cls_id)
            if (prev is None) or (area > prev):
                areas[cls_id] = area

            # For debug: compute pixel box corners
            cx = float(det.bbox.center.position.x)
            cy = float(det.bbox.center.position.y)
            x1 = int(cx - w / 2.0)
            y1 = int(cy - h / 2.0)
            x2 = int(cx + w / 2.0)
            y2 = int(cy + h / 2.0)
            score = float(det.results[0].hypothesis.score)

            boxes.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'cls_id': cls_id,
                'score': score,
            })

        self.yolo_areas = areas
        self.latest_biggest_sign = None
        if areas:
            self.latest_biggest_sign = max(areas.items(), key=lambda kv: kv[1])[0]

        self.debug_yolo_boxes = boxes

        # Debug imshow
        if self.debug_visualization and self.debug_image is not None:
            dbg = self.debug_image.copy()

            # Draw YOLO boxes
            for b in boxes:
                color = (0, 255, 0) if b['cls_id'] == '1' else (255, 0, 0)
                cv2.rectangle(dbg, (b['x1'], b['y1']), (b['x2'], b['y2']), color, 2)
                label = f"{b['cls_id']} {b['score']:.2f}"
                cv2.putText(
                    dbg,
                    label,
                    (b['x1'], max(0, b['y1'] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            # State/debug text
            y0 = 30
            dy = 20
            cv2.putText(
                dbg,
                f"state: {self.state}",
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                dbg,
                f"biggest: {self.latest_biggest_sign or '-'}",
                (10, y0 + dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                dbg,
                f"pal_free: {self.pal_free}",
                (10, y0 + 2 * dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                dbg,
                f"order_init: {self.sign_order_initial}",
                (10, y0 + 3 * dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            closest = self._get_closest_object()
            if closest is not None:
                cv2.putText(
                    dbg,
                    f"lidar angle: {closest['angle_deg']:.1f} deg",
                    (10, y0 + 4 * dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(self._debug_window, dbg)
            cv2.waitKey(1)

    # === Helpers ===

    def _get_closest_object(self) -> Optional[Dict[str, float]]:
        if not self.lidar_objects:
            return None
        return min(self.lidar_objects, key=lambda o: o['distance'])

    def _enter_pause(self, duration_s: float, next_state: str, now):
        self.state = 'PAUSE'
        self.pause_until = now + Duration(seconds=duration_s)
        self._pause_next_state = next_state
        if self.debug_logging:
            self.get_logger().info(
                f'Paletting: entering PAUSE {duration_s:.1f}s -> {next_state}.'
            )

    # === Main step ===

    def step(self):
        now = self.get_clock().now()
        elapsed = (now - self._last_step_time).nanoseconds / 1e9
        if elapsed < self._min_period:
            return
        self._last_step_time = now

        if not self.enabled:
            return

        # --- PAUSE state handling ---
        if self.state == 'PAUSE':
            if self.pause_until is not None and now < self.pause_until:
                # do nothing during pause
                return
            # end of pause
            next_state = self._pause_next_state or 'IDLE'
            if self.debug_logging:
                self.get_logger().info(f'Paletting: PAUSE done -> {next_state}.')
            self.state = next_state
            self.pause_until = None
            self._pause_next_state = None
            # fall through to new state in same cycle

        # === DONE: hold robot stopped and ignore DriveNode ===
        if self.state == 'DONE':
            # Keep DriveNode muted while we explicitly publish zero cmd_vel
            self.pub_pal_control_active.publish(Bool(data=True))

            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.pub_cmd_vel.publish(twist)

            if self.debug_logging:
                self.get_logger().debug(
                    'Paletting DONE: holding v=0, omega=0 with pal_control_active=True.'
                )
            return


        # === IDLE: wait for start condition ===
        if self.state == 'IDLE':
            # Need LiDAR and at least one of the signs 1 or 2
            if not self.lidar_objects:
                return
            if '1' not in self.yolo_areas and '2' not in self.yolo_areas:
                return
            if self.require_pal_free_for_start and not self.pal_free:
                return

            # Determine initial sign order (for debug): which is closer / visible
            area1 = self.yolo_areas.get('1')
            area2 = self.yolo_areas.get('2')

            if area1 is not None and area2 is not None:
                # both visible: record relative order
                if area1 >= area2:
                    self.sign_order_initial = [1, 2]
                else:
                    self.sign_order_initial = [2, 1]
            elif area1 is not None:
                # only "1" visible
                self.sign_order_initial = [1]
            elif area2 is not None:
                # only "2" visible
                self.sign_order_initial = [2]
            else:
                # should not happen because of the check above
                self.sign_order_initial = None

            if self.debug_logging:
                self.get_logger().info(
                    f'Paletting: start, sign(s) visible. '
                    f'sign_order_initial = {self.sign_order_initial}.'
                )

            self.state = 'WAIT_LIDAR_ANGLE_1'
            return

        # For all other states we need at least one LiDAR object
        closest = self._get_closest_object()
        if closest is None:
            if self.debug_logging:
                self.get_logger().debug('Paletting: no LiDAR objects, waiting...')
            return

        angle_deg = closest['angle_deg']
        dist_m = closest['distance']

        # === WAIT_LIDAR_ANGLE_1 ===
        if self.state == 'WAIT_LIDAR_ANGLE_1':
            if abs(angle_deg) >= self.start_angle_threshold_deg:
                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting: WAIT_LIDAR_ANGLE_1 satisfied (angle={angle_deg:.1f}deg) -> ALIGN_0_FIRST.'
                    )
                self.state = 'ALIGN_0_FIRST'
            return

        # === ALIGN_0_FIRST ===
        if self.state == 'ALIGN_0_FIRST':
            if abs(angle_deg) > self.angle_tolerance_deg:
                yaw_init = angle_deg
                yaw_tar = self.yaw_target0_deg
                self.pub_yaw_init_pal.publish(Float64(data=float(yaw_init)))
                self.pub_yaw_tar_pal.publish(Float64(data=float(yaw_tar)))

                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting ALIGN_0_FIRST: angle={angle_deg:.1f}deg -> target={yaw_tar:.1f}deg.'
                    )
            else:
                # Alignment complete -> pause, then check YOLO
                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting ALIGN_0_FIRST done (angle={angle_deg:.1f}deg) '
                        f'-> PAUSE -> CHECK_SIGN_AFTER_ALIGN0.'
                    )
                self._enter_pause(self.pause_between_steps_s,
                                  'CHECK_SIGN_AFTER_ALIGN0', now)
            return

        # === CHECK_SIGN_AFTER_ALIGN0 ===
        if self.state == 'CHECK_SIGN_AFTER_ALIGN0':
            biggest = self.latest_biggest_sign
            if biggest is None:
                if self.debug_logging:
                    self.get_logger().debug(
                        'Paletting: CHECK_SIGN_AFTER_ALIGN0, no YOLO 1/2 yet.'
                    )
                return

            if biggest == '1':
                # We saw "1" directly after first ALIGN_0.
                # This corresponds to order [1, 2] in the description.
                self.sign_order_initial = [1, 2]
                if self.debug_logging:
                    self.get_logger().info(
                        'Paletting: after first ALIGN_0, biggest sign is "1" '
                        '-> go to CHECK_PALLET.'
                    )
                self._enter_pause(self.pause_between_steps_s,
                                  'CHECK_PALLET', now)
                return

            if biggest == '2':
                # We saw "2" first -> order [2, 1].
                self.sign_order_initial = [2, 1]
                if self.debug_logging:
                    self.get_logger().info(
                        'Paletting: after first ALIGN_0, biggest sign is "2" '
                        '-> ALIGN_90.'
                    )
                self.state = 'ALIGN_90'
                return

            # Shouldn't happen (we only track 1/2), but just in case
            return

        # === ALIGN_90 ===
        if self.state == 'ALIGN_90':
            err = angle_deg - self.yaw_target90_deg
            if abs(err) > self.angle_tolerance_deg:
                yaw_init = angle_deg
                yaw_tar = self.yaw_target90_deg
                self.pub_yaw_init_pal.publish(Float64(data=float(yaw_init)))
                self.pub_yaw_tar_pal.publish(Float64(data=float(yaw_tar)))

                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting ALIGN_90: angle={angle_deg:.1f}deg -> target={yaw_tar:.1f}deg.'
                    )
            else:
                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting ALIGN_90 done (angle={angle_deg:.1f}deg) '
                        f'-> PAUSE -> WAIT_LIDAR_ANGLE_2.'
                    )
                self._enter_pause(self.pause_after_90_s,
                                  'WAIT_LIDAR_ANGLE_2', now)
            return

        # === WAIT_LIDAR_ANGLE_2 ===
        if self.state == 'WAIT_LIDAR_ANGLE_2':
            if abs(angle_deg) >= self.start_angle_threshold_deg:
                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting: WAIT_LIDAR_ANGLE_2 satisfied '
                        f'(angle={angle_deg:.1f}deg) -> ALIGN_0_SECOND.'
                    )
                self.state = 'ALIGN_0_SECOND'
            return

        # === ALIGN_0_SECOND ===
        if self.state == 'ALIGN_0_SECOND':
            if abs(angle_deg) > self.angle_tolerance_deg:
                yaw_init = angle_deg
                yaw_tar = self.yaw_target0_deg
                self.pub_yaw_init_pal.publish(Float64(data=float(yaw_init)))
                self.pub_yaw_tar_pal.publish(Float64(data=float(yaw_tar)))

                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting ALIGN_0_SECOND: angle={angle_deg:.1f}deg '
                        f'-> target={yaw_tar:.1f}deg.'
                    )
            else:
                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting ALIGN_0_SECOND done (angle={angle_deg:.1f}deg) '
                        f'-> PAUSE -> CHECK_SIGN_FINAL.'
                    )
                self._enter_pause(self.pause_between_steps_s,
                                  'CHECK_SIGN_FINAL', now)
            return

        # === CHECK_SIGN_FINAL ===
        if self.state == 'CHECK_SIGN_FINAL':
            biggest = self.latest_biggest_sign
            if biggest is None:
                if self.debug_logging:
                    self.get_logger().debug(
                        'Paletting: CHECK_SIGN_FINAL, no YOLO 1/2 yet.'
                    )
                return

            if biggest == '1':
                # Final situation: we now see "1" as the closest sign.
                # If we had "2" first, sign_order_initial will be [2, 1].
                if self.sign_order_initial is None:
                    self.sign_order_initial = [2, 1]
                if self.debug_logging:
                    self.get_logger().info(
                        'Paletting: final check, biggest sign is "1" '
                        '-> go to CHECK_PALLET.'
                    )
                self._enter_pause(self.pause_between_steps_s,
                                  'CHECK_PALLET', now)
            else:
                # If it's "2" or something else, we just wait for now.
                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting: final check, biggest sign is "{biggest}", '
                        f'waiting for "1"...'
                    )
            return

        # === CHECK_PALLET ===
        if self.state == 'CHECK_PALLET':
            # pal_free == True  -> pallet already free, nothing to pick
            # pal_free == False -> need to approach and pick up
            if self.pal_free is None:
                if self.debug_logging:
                    self.get_logger().info('Paletting: CHECK_PALLET, pal_free is None -> waiting...')
                return

            if self.pal_free:
                if self.debug_logging:
                    self.get_logger().info(
                        'Paletting: CHECK_PALLET -> pal_free=True, DONE.'
                    )
                # Immediately mute DriveNode and stop; DONE state will keep it that way
                self.pub_pal_control_active.publish(Bool(data=True))
                stop_twist = Twist()
                stop_twist.linear.x = 0.0
                stop_twist.angular.z = 0.0
                self.pub_cmd_vel.publish(stop_twist)

                self.state = 'DONE'

            else:
                if self.debug_logging:
                    self.get_logger().info(
                        'Paletting: CHECK_PALLET -> pal_free=False, '
                        'DRIVE_TO_PALLET_FORWARD.'
                    )
                self.state = 'DRIVE_TO_PALLET_FORWARD'
            return

        # === DRIVE_TO_PALLET_FORWARD ===
        if self.state == 'DRIVE_TO_PALLET_FORWARD':
            # PalettingNode takes over cmd_vel: mute DriveNode
            self.pub_pal_control_active.publish(Bool(data=True))

            if dist_m > self.drop_forward_dist_threshold_m:
                # Drive straight forward with fixed v, omega = 0
                twist = Twist()
                twist.linear.x = self.forward_linear_speed_mps
                twist.angular.z = 0.0
                self.pub_cmd_vel.publish(twist)

                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting DRIVE_TO_PALLET_FORWARD: dist={dist_m:.2f} m, '
                        f'v={self.forward_linear_speed_mps:.2f} m/s.'
                    )
            else:
                # Close enough -> stop, turn magnet ON, wait, then back out
                stop_twist = Twist()
                stop_twist.linear.x = 0.0
                stop_twist.angular.z = 0.0
                self.pub_cmd_vel.publish(stop_twist)

                self.pub_electromagnet.publish(Bool(data=True))
                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting: reached pallet (dist={dist_m:.2f} m), '
                        f'magnet ON -> PAUSE {self.magnet_after_grab_wait_s}s -> BACK_FROM_PALLET.'
                    )
                self._enter_pause(self.magnet_after_grab_wait_s,
                                  'BACK_FROM_PALLET', now)
            return


        # === BACK_FROM_PALLET ===
        if self.state == 'BACK_FROM_PALLET':
            # Still in direct-control phase: keep DriveNode muted
            self.pub_pal_control_active.publish(Bool(data=True))

            # Drive backwards (cmd_vel) until we are beyond the "back" threshold.
            if dist_m < self.drop_back_dist_threshold_m:
                twist = Twist()
                twist.linear.x = self.back_linear_speed_mps   # negative -> backwards
                twist.angular.z = 0.0
                self.pub_cmd_vel.publish(twist)

                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting BACK_FROM_PALLET: dist={dist_m:.2f} m, '
                        f'v={self.back_linear_speed_mps:.2f} m/s.'
                    )
            else:
                # Stop the robot
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.pub_cmd_vel.publish(twist)

                # From now on, DriveNode may be used again (for SIDEWAYS etc.)
                self.pub_pal_control_active.publish(Bool(data=False))

                # Decide which side the second pallet is on based on sign_order_initial
                if self.sign_order_initial and len(self.sign_order_initial) > 0:
                    first = self.sign_order_initial[0]
                    if first == 1:
                        # We saw "1" first -> "2" was to the right -> turn right (90°)
                        self.lane_side = 'left'
                        self.post_pick_twist_target_deg = self.yaw_target90_deg
                    else:
                        # We saw "2" first -> "1" was to the left -> turn left (+90°)
                        self.lane_side = 'right'
                        self.post_pick_twist_target_deg = -self.yaw_target90_deg
                else:
                    # Fallback
                    self.lane_side = 'right'
                    self.post_pick_twist_target_deg = self.yaw_target90_deg

                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting BACK_FROM_PALLET done (dist={dist_m:.2f} m), '
                        f'lane_side={self.lane_side}, '
                        f'twist_target={self.post_pick_twist_target_deg:.1f}deg '
                        f'-> PAUSE -> TWIST_AFTER_PICK.'
                    )

                self._enter_pause(self.pause_between_steps_s,
                                  'TWIST_AFTER_PICK', now)
            return


        # === TWIST_AFTER_PICK: rotate +/-90° depending on sign_order_initial ===
        if self.state == 'TWIST_AFTER_PICK':
            target_deg = self.post_pick_twist_target_deg
            if target_deg is None:
                target_deg = self.yaw_target90_deg

            err = angle_deg - target_deg
            if abs(err) > self.angle_tolerance_deg:
                self.pub_yaw_init_pal.publish(Float64(data=float(angle_deg)))
                self.pub_yaw_tar_pal.publish(Float64(data=float(target_deg)))

                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting TWIST_AFTER_PICK: angle={angle_deg:.1f}deg '
                        f'-> target={target_deg:.1f}deg.'
                    )
            else:
                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting TWIST_AFTER_PICK done (angle={angle_deg:.1f}deg) '
                        f'-> DRIVE_SIDEWAYS.'
                    )
                self.side_drive_start_time = now
                self.state = 'DRIVE_SIDEWAYS'
            return

        # === DRIVE_SIDEWAYS: follow lane using x_white_far +/- offset ===
        if self.state == 'DRIVE_SIDEWAYS':
            # Compute x_pal from x_white_far plus/minus offset
            if self.x_white_far is not None:
                if self.lane_side == 'right':
                    x_pal = self.x_white_far + self.side_offset_px
                else:
                    x_pal = self.x_white_far - self.side_offset_px + 147 #(needs hardcoded fixes)
            else:
                x_pal = self.image_center_x


            self.pub_x_pal.publish(Float64(data=float(x_pal)))

            dt_side = 0.0
            if self.side_drive_start_time is not None:
                dt_side = (now - self.side_drive_start_time).nanoseconds / 1e9

            if self.debug_logging:
                self.get_logger().info(
                    f'Paletting DRIVE_SIDEWAYS: t={dt_side:.2f}s, angle={angle_deg:.1f}deg, '
                    f'dist={dist_m:.2f}m, x_pal={x_pal:.1f}'
                )

            # First: ensure we drove at least side_drive_duration_s
            if dt_side < self.side_drive_duration_s:
                return

            # After that, additionally wait for |angle| >= start_angle_threshold_deg
            if abs(angle_deg) >= self.start_angle_threshold_deg:
                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting DRIVE_SIDEWAYS: angle={angle_deg:.1f}deg reached threshold '
                        f'-> PAUSE -> ALIGN_0_DROP.'
                    )
                self._enter_pause(self.pause_between_steps_s,
                                  'ALIGN_0_DROP', now)
            return

        # === ALIGN_0_DROP: rotate object back to 0° ===
        if self.state == 'ALIGN_0_DROP':
            if abs(angle_deg) > self.angle_tolerance_deg:
                yaw_init = angle_deg
                yaw_tar = self.yaw_target0_deg
                self.pub_yaw_init_pal.publish(Float64(data=float(yaw_init)))
                self.pub_yaw_tar_pal.publish(Float64(data=float(yaw_tar)))

                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting ALIGN_0_DROP: angle={angle_deg:.1f}deg -> target={yaw_tar:.1f}deg.'
                    )
            else:
                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting ALIGN_0_DROP done (angle={angle_deg:.1f}deg) '
                        f'-> PAUSE -> DRIVE_TO_DROP_FORWARD.'
                    )
                self._enter_pause(self.pause_between_steps_s,
                                  'DRIVE_TO_DROP_FORWARD', now)
            return

        # === DRIVE_TO_DROP_FORWARD: approach drop position ===
        if self.state == 'DRIVE_TO_DROP_FORWARD':
            # Straight approach is done by PalettingNode directly again
            self.pub_pal_control_active.publish(Bool(data=True))

            if dist_m > self.drop_forward_dist_threshold_m:
                twist = Twist()
                twist.linear.x = self.forward_linear_speed_mps
                twist.angular.z = 0.0
                self.pub_cmd_vel.publish(twist)

                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting DRIVE_TO_DROP_FORWARD: dist={dist_m:.2f} m, '
                        f'v={self.forward_linear_speed_mps:.2f} m/s.'
                    )
            else:
                # Close enough for dropping -> stop and go to magnet release
                stop_twist = Twist()
                stop_twist.linear.x = 0.0
                stop_twist.angular.z = 0.0
                self.pub_cmd_vel.publish(stop_twist)

                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting DRIVE_TO_DROP_FORWARD done (dist={dist_m:.2f} m) '
                        f'-> PAUSE -> MAGNET_RELEASE.'
                    )
                self._enter_pause(self.pause_between_steps_s,
                                  'MAGNET_RELEASE', now)
            return


        # === MAGNET_RELEASE: switch magnet off, then wait ===
        if self.state == 'MAGNET_RELEASE':
            # Turn magnet OFF once and immediately go into a pause before backing away.
            self.pub_electromagnet.publish(Bool(data=False))
            if self.debug_logging:
                self.get_logger().info(
                    f'Paletting MAGNET_RELEASE: magnet OFF -> PAUSE {self.magnet_after_release_wait_s}s '
                    f'-> BACK_FROM_DROP.'
                )
            self._enter_pause(self.magnet_after_release_wait_s,
                              'BACK_FROM_DROP', now)
            return

        # === BACK_FROM_DROP: reverse away from drop position ===
        if self.state == 'BACK_FROM_DROP':
            # Last direct-control phase: keep DriveNode muted
            self.pub_pal_control_active.publish(Bool(data=True))

            if dist_m < self.drop_back_dist_threshold_m:
                twist = Twist()
                twist.linear.x = self.back_linear_speed_mps
                twist.angular.z = 0.0
                self.pub_cmd_vel.publish(twist)

                if self.debug_logging:
                    self.get_logger().info(
                        f'Paletting BACK_FROM_DROP: dist={dist_m:.2f} m, '
                        f'v={self.back_linear_speed_mps:.2f} m/s.'
                    )
            else:
                # Stop robot and go to DONE, staying in direct-control mode.
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.pub_cmd_vel.publish(twist)

                # DO NOT re-enable DriveNode; DONE will keep pal_control_active=True
                if self.debug_logging:
                    self.get_logger().info(
                        'Paletting BACK_FROM_DROP done -> DONE (holding position).'
                    )
                self.state = 'DONE'
            return




    # === Cleanup ===

    def destroy_node(self):
        if getattr(self, 'debug_visualization', False):
            cv2.destroyAllWindows()
        self.get_logger().info('PalettingNode shutting down.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PalettingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
