#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError

from ultralytics import YOLO
from pathlib import Path

from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose

import cv2


class YoloSignDetector(Node):
    """
    YOLOv8-Schild-Detektor:

    - subscribed auf ein *compressed* Kamera-Topic (sensor_msgs/CompressedImage)
    - führt YOLOv8-Inferenz aus
    - wählt pro Klasse nur das 'nächste' Schild (größte Bounding-Box-Fläche)
    - published:
        * Detection2DArray auf detections_topic
        * Debug-Image mit Bounding Boxes + Klassennamen auf debug_image_topic (sensor_msgs/Image)
    """

    def __init__(self):
        super().__init__('yolo_sign_detector')

        # -------- Parameter --------
        self.declare_parameter('image_topic', '/camera/image_raw/compressed')
        default_weights = str(
            Path(__file__).resolve().parent / 'weights' / '20251123_200000.pt'
        )
        self.declare_parameter('weights_path', default_weights)
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('detections_topic', '/yolo/sign_detections')

        # Debug-Flags / Topic
        self.declare_parameter('debug_image', True)
        self.declare_parameter('debug_image_topic', '/yolo/debug_image')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        weights_path = self.get_parameter('weights_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        detections_topic = self.get_parameter('detections_topic').get_parameter_value().string_value

        self.debug_image = self.get_parameter('debug_image').get_parameter_value().bool_value
        debug_image_topic = self.get_parameter('debug_image_topic').get_parameter_value().string_value

        # Klassen-Namen wie im Training
        # Index 0..5 -> string name
        self.class_names = ['1', '2', 'left', 'right', 'stop', 'tunnel']

        # -------- YOLO laden --------
        self.model = YOLO(weights_path)

        # -------- Subscriber & Publisher --------
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

        # Debug-Image Publisher (normales sensor_msgs/Image)
        self.debug_image_pub = self.create_publisher(
            Image,
            debug_image_topic,
            10
        )

        log_info = (
            f"YoloSignDetector gestartet.\n"
            f"  Sub: image_topic={image_topic}\n"
            f"  Pub: detections_topic={detections_topic}\n"
            f"  YOLOv8-Weights: {weights_path}\n"
            f"  Confidence-Threshold: {self.conf_threshold:.2f}\n"
            f"  debug_image: {self.debug_image} -> {debug_image_topic}"
        )
        self.get_logger().info(log_info)

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
            # Sicherer Klassenname (String)
            if 0 <= cls_id < len(self.class_names):
                class_name = self.class_names[cls_id]
            else:
                class_name = f"cls_{cls_id}"

            # Detection2D für diese Klasse
            det_msg = Detection2D()
            det_msg.header = msg.header

            # Klasse & Score in hypothesis
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = class_name   # <--- STRING!
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
