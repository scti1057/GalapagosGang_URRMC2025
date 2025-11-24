#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np

from ament_index_python.packages import get_package_share_directory


class PalletFreeDetector(Node):
    """
    Erkennung, ob eine blaue Palette / Fläche im Bildbereich vor dem Bot frei ist.

    - ROI ist ein Viereck (Trapez) mit 4 Punkten (x0,y0 ... x3,y3)
    - Nur innerhalb dieses ROI wird Blau gesucht.
    - Viel Blau  -> Fläche frei       -> pal_free = True
    - Wenig Blau -> Fläche blockiert  -> pal_free = False

    Ausgabe:
      - Topic "pal_free" (std_msgs/Bool)
        Bei jeder neuen Entscheidung wird 'burst_size'-mal derselbe Wert gepublished.
    """

    def __init__(self):
        super().__init__('pallet_free_detector')

        # ----------------------------------------------------------------------
        # --- Load config file from installed share/galapagos_regelt/config ---
        # ----------------------------------------------------------------------
        self.declare_parameter('config_file', 'pal_free_params.yaml')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value

        # Falls du das Paket umbenennst, hier anpassen:
        package_share = get_package_share_directory('galapagos_regelt')
        self._config_path = os.path.join(package_share, 'config', config_file)
        self.get_logger().info(f'Using config file: {self._config_path}')

        try:
            with open(self._config_path, 'r') as f:
                conf_raw = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().error(f'Konnte Config nicht laden: {e}')
            conf_raw = {}

        # Node-Name / ros__parameters ggf. „auspacken“,
        # damit sowohl flache YAMLs als auch ROS2-Param-Dateien funktionieren.
        cfg = conf_raw
        node_name = self.get_name()
        if isinstance(cfg, dict) and node_name in cfg:
            cfg = cfg[node_name]
        if isinstance(cfg, dict) and 'ros__parameters' in cfg:
            cfg = cfg['ros__parameters']

        if not isinstance(cfg, dict):
            cfg = {}

        self.conf = cfg

        # Kleine Helper-Funktion für Defaults
        def _cfg(key, default=None):
            return self.conf.get(key, default)

        self._cfg = _cfg  # merken für andere Methoden

        # ----------------------------------------------------------------------
        # Konfiguration aus YAML lesen
        # ----------------------------------------------------------------------
        image_topic = _cfg('image_topic', '/camera/image_raw/compressed')
        debug_image_topic = _cfg('debug_image_topic', '/pallet_free/debug_image')
        self.debug_image_enabled = bool(_cfg('debug_image', False))

        self.free_threshold = float(_cfg('free_threshold', 0.8))
        self.burst_size = int(_cfg('burst_size', 10))

        # HSV für Blau
        self.blue_h_min = int(_cfg('blue_h_min', 90))
        self.blue_h_max = int(_cfg('blue_h_max', 130))
        self.blue_s_min = int(_cfg('blue_s_min', 80))
        self.blue_v_min = int(_cfg('blue_v_min', 50))

        # ----------------------------------------------------------------------
        # ROS-Setup
        # ----------------------------------------------------------------------
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_sub = self.create_subscription(
            CompressedImage,
            image_topic,
            self.image_callback,
            qos_profile
        )

        self.pal_free_pub = self.create_publisher(Bool, 'pal_free', 10)
        self.debug_image_pub = self.create_publisher(Image, debug_image_topic, 1)
        self._window = "pallet_free_debug"
        #cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)

        self.bridge = CvBridge()
        self.last_is_free = None

        self.get_logger().info(
            f'PalletFreeDetector gestartet.\n'
            f'  Config: {self._config_path}\n'
            f'  Sub: image_topic = {image_topic}\n'
            f'  Pub: pal_free (std_msgs/Bool)\n'
            f'  Debug-Image: {self.debug_image_enabled} -> {debug_image_topic}\n'
            f'  free_threshold = {self.free_threshold}, burst_size = {self.burst_size}'
        )

    # ------------------------------------------------------------------

    def _get_roi_polygon(self, img_width: int, img_height: int) -> np.ndarray:
        """
        Liest die 4 ROI-Punkte aus self.conf, clipped sie ins Bild
        und gibt ein (4,2)-Array int32 zurück.
        """

        x0 = int(self._cfg('roi_x0', 200))
        y0 = int(self._cfg('roi_y0', 245))
        x1 = int(self._cfg('roi_x1', 425))
        y1 = int(self._cfg('roi_y1', 245))
        x2 = int(self._cfg('roi_x2', 535))
        y2 = int(self._cfg('roi_y2', 465))
        x3 = int(self._cfg('roi_x3', 100))
        y3 = int(self._cfg('roi_y3', 465))

        xs = np.array([x0, x1, x2, x3], dtype=np.int32)
        ys = np.array([y0, y1, y2, y3], dtype=np.int32)

        # In Bildgrenzen clippen
        xs = np.clip(xs, 0, img_width - 1)
        ys = np.clip(ys, 0, img_height - 1)

        pts = np.stack([xs, ys], axis=1)  # shape (4, 2)
        return pts.astype(np.int32)

    # ------------------------------------------------------------------

    def image_callback(self, msg: CompressedImage):
        # --- CompressedImage -> OpenCV-Bild ---
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Fehler: {e}')
            return

        h, w = cv_image.shape[:2]

        # ROI-Polygon holen
        pts = self._get_roi_polygon(w, h)

        if len(pts) < 3:
            self.get_logger().warn('ROI-Polygon hat weniger als 3 Punkte.')
            return

        # Maske für ROI erstellen
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [pts], 255)

        roi_pixel_count = int(np.count_nonzero(roi_mask))
        if roi_pixel_count == 0:
            self.get_logger().warn('ROI-Maske hat 0 Pixel – prüfe ROI-Koordinaten in der YAML.')
            return

        # --- Blau-Erkennung im Bild ---
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([self.blue_h_min, self.blue_s_min, self.blue_v_min], dtype=np.uint8)
        upper_blue = np.array([self.blue_h_max, 255, 255], dtype=np.uint8)

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Auf ROI einschränken
        blue_in_roi = cv2.bitwise_and(blue_mask, blue_mask, mask=roi_mask)

        blue_pixels = int(np.count_nonzero(blue_in_roi))
        ratio_blue = blue_pixels / float(roi_pixel_count)

        is_free = ratio_blue >= self.free_threshold

        # Nur bei Zustandsänderung loggen + Burst publishen
        if self.last_is_free is None or is_free != self.last_is_free:
            self.last_is_free = is_free
            state_str = 'FREI' if is_free else 'BLOCKIERT'
            self.get_logger().info(
                f'Palette: {state_str} | blue_ratio={ratio_blue:.3f} '
                f'(threshold={self.free_threshold:.3f}, roi_pixels={roi_pixel_count})'
            )

            out_msg = Bool()
            out_msg.data = is_free
            for _ in range(self.burst_size):
                self.pal_free_pub.publish(out_msg)

        # --- Debug-Image ---
        if self.debug_image_enabled:
            debug_img = cv_image.copy()

            # ROI einzeichnen
            cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Text in der Nähe von Punkt 0
            x0, y0 = int(pts[0, 0]), int(pts[0, 1])
            y_text = y0 - 10 if y0 - 10 > 10 else y0 + 20
            text = f'free={is_free} blue={ratio_blue:.2f}'
            cv2.putText(
                debug_img, text, (x0 + 5, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )

            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
                debug_msg.header = msg.header
                self.debug_image_pub.publish(debug_msg)
                # cv2.imshow(self._window, debug_img)
                # cv2.waitKey(1)
            except CvBridgeError as e:
                self.get_logger().error(f'CvBridge Fehler (debug image): {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PalletFreeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
