#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
from pathlib import Path

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError

from ultralytics import YOLO
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose

import cv2
from ament_index_python.packages import get_package_share_directory


class YoloSignDetector(Node):
    """
    YOLOv8-Schild-Detektor:

    - subscribed auf *compressed* Kamera-Topic (sensor_msgs/CompressedImage)
    - führt YOLOv8-Inferenz aus
    - wählt pro Klasse nur das 'nächste' Schild (größte Bounding-Box-Fläche)
    - published:
        * Detection2DArray auf detections_topic
        * Debug-Image mit Bounding Boxes + Klassennamen auf debug_image_topic (sensor_msgs/Image)

    Konfiguration:
      - wird aus einem YAML-File geladen, z.B. share/galapagos_checked_yolo/config/yolo_sign_params.yaml
      - welches File benutzt wird, wird über den ROS-Parameter 'config_file' bestimmt
    """

    def __init__(self):
        super().__init__('yolo_sign_detector')

        # ------------------------------------------------------------------
        # Config-Datei laden (ähnlich wie beim Paletten-Detector)
        # ------------------------------------------------------------------
        self.declare_parameter('config_file', 'yolo_sign_params.yaml')
        config_file = (
            self.get_parameter('config_file')
            .get_parameter_value()
            .string_value
        )

        package_share = get_package_share_directory('galapagos_checked_yolo')
        self._config_path = os.path.join(package_share, 'config', config_file)
        self.get_logger().info(f'Using config file: {self._config_path}')

        try:
            with open(self._config_path, 'r') as f:
                conf_raw = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().error(f'Konnte Config nicht laden: {e}')
            conf_raw = {}

        # Node-Namen / ros__parameters ggf. „auspacken“
        cfg = conf_raw
        node_name = self.get_name()
        if isinstance(cfg, dict) and node_name in cfg:
            cfg = cfg[node_name]
        if isinstance(cfg, dict) and 'ros__parameters' in cfg:
            cfg = cfg['ros__parameters']
        if not isinstance(cfg, dict):
            cfg = {}
        self.conf = cfg

        def _cfg(key, default=None):
            return self.conf.get(key, default)

        self._cfg = _cfg

        # ------------------------------------------------------------------
        # Parameter aus YAML (mit sinnvollen Defaults)
        # ------------------------------------------------------------------
        default_weights = '/home/duckie6/GalapagosGang_URRMC2025/src/galapagos_checked_yolo/galapagos_checked_yolo/weights/20251123_200000.pt'

        image_topic = _cfg('image_topic', '/camera/image_raw/compressed')
        weights_path = _cfg('weights_path', default_weights)
        self.conf_threshold = float(_cfg('conf_threshold', 0.8))
        detections_topic = _cfg('detections_topic', '/yolo/sign_detections')

        self.debug_image = bool(_cfg('debug_image', False))
        debug_image_topic = _cfg('debug_image_topic', '/yolo/debug_image')

        class_names_cfg = _cfg(
            'class_names',
            ['1', '2', 'left', 'right', 'stop', 'tunnel']
        )
        # robust in Strings/Listen umsetzen
        if isinstance(class_names_cfg, (list, tuple)):
            self.class_names = [str(c) for c in class_names_cfg]
        elif isinstance(class_names_cfg, str):
            self.class_names = [s.strip() for s in class_names_cfg.split(',') if s.strip()]
        else:
            self.class_names = ['1', '2', 'left', 'right', 'stop', 'tunnel']

        # ------------------------------------------------------------------
        # YOLO laden
        # ------------------------------------------------------------------
        self.get_logger().info(f'YOLO-Weights laden: {weights_path}')
        self.model = YOLO(weights_path)

        # ------------------------------------------------------------------
        # Subscriber & Publisher
        # ------------------------------------------------------------------
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            CompressedImage,
            image_topic,
            self.image_callback,
            10
        )

        self.detections_pub = self.create_publisher(
            Detection2DArray,
            detections_topic,
            10
        )

        self.debug_image_pub = self.create_publisher(
            Image,
            debug_image_topic,
            10
        )

        log_info = (
            f"YoloSignDetector gestartet.\n"
            f"  Config: {self._config_path}\n"
            f"  Sub: image_topic={image_topic}\n"
            f"  Pub: detections_topic={detections_topic}\n"
            f"  YOLOv8-Weights: {weights_path}\n"
            f"  Confidence-Threshold: {self.conf_threshold:.2f}\n"
            f"  debug_image: {self.debug_image} -> {debug_image_topic}\n"
            f"  class_names: {self.class_names}\n"
            f"  config: {self.conf}"
        )
        self.get_logger().info(log_info)

    # ------------------------------------------------------------------

    def image_callback(self, msg: CompressedImage):
        # CompressedImage -> OpenCV
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Fehler: {e}")
            return

        # YOLO-Inferenz
        results = self.model.predict(
            source=cv_image,
            verbose=False
        )

        if not results:
            return

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return  # nichts erkannt

        # -------- pro Klasse: nur "nächstes" Schild behalten --------
        best_by_class = {}  # cls_id -> dict mit bbox + conf

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            area = w * h

            if w <= 0 or h <= 0:
                continue

            prev = best_by_class.get(cls_id)
            if prev is None or area > prev['area']:
                best_by_class[cls_id] = {
                    'conf': conf,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'w': w,
                    'h': h,
                    'area': area,
                }

        if not best_by_class:
            return

        # -------- Detection2DArray Message bauen --------
        det_array = Detection2DArray()
        det_array.header = msg.header  # Zeitstempel/Frame vom Kamera-Image übernehmen

        log_parts = []

        # Debug-Bild: Kopie vom Original
        debug_img = cv_image.copy()

        for cls_id, det in best_by_class.items():
            # Klassenname aus Liste
            if 0 <= cls_id < len(self.class_names):
                class_name = self.class_names[cls_id]
            else:
                class_name = f"cls_{cls_id}"

            # Detection2D für diese Klasse
            det_msg = Detection2D()
            det_msg.header = msg.header

            # Klasse & Score in hypothesis
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = class_name
            hyp.hypothesis.score = det['conf']
            det_msg.results.append(hyp)

            # Bounding Box
            bbox = BoundingBox2D()
            bbox.center.position.x = (det['x1'] + det['x2']) / 2.0
            bbox.center.position.y = (det['y1'] + det['y2']) / 2.0
            bbox.size_x = det['w']
            bbox.size_y = det['h']
            det_msg.bbox = bbox

            det_array.detections.append(det_msg)

            log_parts.append(
                f"{class_name} conf={det['conf']:.2f} "
                f"center=({bbox.center.position.x:.0f},{bbox.center.position.y:.0f}) "
                f"size=({bbox.size_x:.0f}x{bbox.size_y:.0f})"
            )

            # -------- Debug-Overlay zeichnen --------
            x1 = int(det['x1'])
            y1 = int(det['y1'])
            x2 = int(det['x2'])
            y2 = int(det['y2'])

            color = (0, 255, 0)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name} {det['conf']:.2f}"
            text_x = x1
            text_y = max(y1 - 5, 0)
            cv2.putText(
                debug_img,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

        # publish Detections
        self.detections_pub.publish(det_array)
        self.get_logger().info("Best per class: " + " | ".join(log_parts))

        # publish Debug-Image (nur wenn aktiviert)
        if self.debug_image:
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
                debug_msg.header = msg.header
                self.debug_image_pub.publish(debug_msg)
            except CvBridgeError as e:
                self.get_logger().error(f"CvBridge Fehler beim Debug-Image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloSignDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
