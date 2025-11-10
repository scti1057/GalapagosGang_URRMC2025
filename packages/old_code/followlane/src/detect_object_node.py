#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import os
import yaml
from collections import deque
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64MultiArray
from ultralytics import YOLO


class DetectParkingSlotNode(DTROS):
    def __init__(self, node_name):
        super(DetectParkingSlotNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        self._bridge = CvBridge()
        self.cv_image = None  # Latest camera image

        # History buffers for temporal smoothing
        self.history_duckie_lane = deque(maxlen=10)
        self.history_duckie_right = deque(maxlen=10)
        self.history_bot = deque(maxlen=10)
        self.history_freeSlot = deque(maxlen=10)
        self.history_occupiedSlot = deque(maxlen=10)

        self.timeout_sec = 0.1  # Max delay before clearing last object

        # Last known detections and timestamps
        self.last_duckie_lane = None
        self.last_duckie_lane_bbox = None
        self.last_duckie_lane_time = rospy.Time(0)

        self.last_duckie_lane_right = None
        self.last_duckie_lane_right_bbox = None
        self.last_duckie_lane_right_time = rospy.Time(0)

        self.last_bot = None
        self.last_bot_bbox = None
        self.last_bot_time = rospy.Time(0)

        self.last_freeSlot = None
        self.last_freeSlot_bbox = None
        self.last_freeSlot_time = rospy.Time(0)

        self.last_occupiedSlot = None
        self.last_occupiedSlot_bbox = None
        self.last_occupiedSlot_time = rospy.Time(0)

        # Load YOLO model for detection
        self._model = YOLO("packages/followlane/assets/model_detectDuckieBotSlot_V3.pt")

        # Load configuration for masks and thresholds
        with open('packages/followlane/config/detect_duckieBotSlot.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        # === Subscribers ===
        self.sub_image = rospy.Subscriber(f"/{self._vehicle_name}/camera_node/image/compressed",
                                          CompressedImage, self.cbDetectObjects, queue_size=1)

        # === Publishers ===
        self.pup_image = rospy.Publisher(f"/{self._vehicle_name}/detect/object/image", Image, queue_size=1)
        self.pup_duckieNearestBB = rospy.Publisher(f"/{self._vehicle_name}/detect/object/duckieNearestBB", Float64MultiArray, queue_size=1)
        self.pup_duckieNearestRightBB = rospy.Publisher(f"/{self._vehicle_name}/detect/object/duckieNearestRightBB", Float64MultiArray, queue_size=1)
        self.pup_botNearestBB = rospy.Publisher(f"/{self._vehicle_name}/detect/object/botNearestBB", Float64MultiArray, queue_size=1)
        self.pup_parkingBB = rospy.Publisher(f"/{self._vehicle_name}/detect/object/parkingBB", Float64MultiArray, queue_size=1)
        self.pup_parkingOccupiedBB = rospy.Publisher(f"/{self._vehicle_name}/detect/object/parkingOccupiedBB", Float64MultiArray, queue_size=1)

    # Callback: convert compressed image to OpenCV format
    def cbDetectObjects(self, image_msg):
        if image_msg is not None:
            self.cv_image = self._bridge.compressed_imgmsg_to_cv2(image_msg)

    # Retrieve polygon mask from config with optional offset
    def get_polygon(self, mask_name, shift_x=None, shift_y=None):
        mask = np.array([
            [self.conf[mask_name]['top_left_x'], self.conf[mask_name]['top_left_y']],
            [self.conf[mask_name]['top_right_x'], self.conf[mask_name]['top_right_y']],
            [self.conf[mask_name]['bottom_right_x'], self.conf[mask_name]['bottom_right_y']],
            [self.conf[mask_name]['bottom_left_x'], self.conf[mask_name]['bottom_left_y']],
        ], dtype=np.int32)

        if shift_x is not None:
            mask[:, 0] += shift_x
        if shift_y is not None:
            mask[:, 1] += shift_y

        return mask

    # Compute intersection-over-union (IoU) between two bounding boxes
    def compute_iou(self, box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return interArea / float(box1Area + box2Area - interArea + 1e-6)

    # Check if the center of a bounding box lies within a polygon
    def is_in_mask(self, bbox, mask_polygon):
        x_center = (bbox[0] + bbox[2]) // 2
        y_center = (bbox[1] + bbox[3]) // 2
        return cv2.pointPolygonTest(mask_polygon, (x_center, y_center), False) >= 0

    def run(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.cv_image is None:
                rate.sleep()
                continue

            now = rospy.Time.now()
            results = self._model(self.cv_image, verbose=False)[0]
            boxes = results.boxes

            duckies, bots, slots = [], [], []

            # === Filter YOLO detections by class and confidence ===
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self._model.names[cls]
                bbox = {'bbox': (x1, y1, x2, y2), 'conf': conf, 'label': label}

                if label == "Duckie" and conf > self.conf['conf_threshold_duckie']:
                    duckies.append(bbox)
                elif label == "Bot" and conf > self.conf['conf_threshold_bot']:
                    bots.append(bbox)
                elif "Slot" in label and conf > self.conf['conf_threshold_slot']:
                    slots.append(bbox)

            # === Slot occupancy classification ===
            occupiedSlots, freeSlots = [], []
            for slot in slots:
                box_slot = slot['bbox']
                max_iou = max([self.compute_iou(box_slot, obj['bbox']) for obj in duckies + bots], default=0.0)
                if max_iou > self.conf['iou_threshold']:
                    occupiedSlots.append({'bbox': box_slot, 'conf': slot['conf'], 'label': 'occupiedSlot'})
                else:
                    freeSlots.append({'bbox': box_slot, 'conf': slot['conf'], 'label': 'freeSlot'})

            # === Create and apply ROI masks ===
            mask_lane = self.get_polygon('mask_lane')
            mask_right = self.get_polygon('mask_lane', shift_x=180)
            mask_bot = self.get_polygon('mask')

            duckie_lane = [d['bbox'] for d in duckies if self.is_in_mask(d['bbox'], mask_lane)]
            duckie_right = [d['bbox'] for d in duckies if self.is_in_mask(d['bbox'], mask_right)]
            bot_lane = [b['bbox'] for b in bots if self.is_in_mask(b['bbox'], mask_bot)]

            # === Update history buffers ===
            self.history_duckie_lane.append(bool(duckie_lane))
            self.history_duckie_right.append(bool(duckie_right))
            self.history_bot.append(bool(bot_lane))
            self.history_freeSlot.append(bool(freeSlots))
            self.history_occupiedSlot.append(bool(occupiedSlots))

            # === Compute temporal confidence values ===
            conf_duckie = sum(self.history_duckie_lane) / len(self.history_duckie_lane)
            conf_duckie_right = sum(self.history_duckie_right) / len(self.history_duckie_right)
            conf_bot = sum(self.history_bot) / len(self.history_bot)
            conf_freeSlot = sum(self.history_freeSlot) / len(self.history_freeSlot)
            conf_occupiedSlot = sum(self.history_occupiedSlot) / len(self.history_occupiedSlot)

            # === Publish filtered detections with temporal filtering ===

            # Duckie lane (center)
            if duckie_lane and conf_duckie > self.conf["confidence_temporal"]:
                nearest_duckie = sorted(duckie_lane, key=lambda b: (b[1] + b[3]) // 2, reverse=True)[0]
                x1, y1, x2, y2 = map(int, nearest_duckie)
                msg_duckie_lane = Float64MultiArray(data=[x1, y1, x2, y2])
                self.last_duckie_lane = msg_duckie_lane
                self.last_duckie_lane_time = now
                self.last_duckie_lane_bbox = {'bbox': (x1, y1, x2, y2), 'label': 'duckie'}
            elif self.last_duckie_lane and (now - self.last_duckie_lane_time).to_sec() < self.timeout_sec:
                pass
            else:
                self.last_duckie_lane = None
                self.last_duckie_lane_bbox = None
                msg_duckie_lane = Float64MultiArray(data=[])
            self.pup_duckieNearestBB.publish(msg_duckie_lane)

            # Duckie right (for bypass)
            if duckie_right and conf_duckie_right > self.conf["confidence_temporal"]:
                nearest_duckie_right = sorted(duckie_right, key=lambda b: (b[1] + b[3]) // 2, reverse=True)[0]
                x1, y1, x2, y2 = map(int, nearest_duckie_right)
                msg_duckie_lane_right = Float64MultiArray(data=[x1, y1, x2, y2])
                self.last_duckie_lane_right = msg_duckie_lane_right
                self.last_duckie_lane_right_time = now
                self.last_duckie_lane_right_bbox = {'bbox': (x1, y1, x2, y2), 'label': 'duckie_right'}
            elif self.last_duckie_lane_right and (now - self.last_duckie_lane_right_time).to_sec() < self.timeout_sec:
                pass
            else:
                self.last_duckie_lane_right = None
                self.last_duckie_lane_right_bbox = None
                msg_duckie_lane_right = Float64MultiArray(data=[])
            self.pup_duckieNearestRightBB.publish(msg_duckie_lane_right)

            # Bot detection (multiple possible)
            if bot_lane and conf_bot > self.conf["confidence_temporal"]:
                sorted_bots = sorted(bot_lane, key=lambda b: (b[1] + b[3]) // 2, reverse=True)
                bbox_list = []
                for bot in sorted_bots:
                    x1, y1, x2, y2 = map(int, bot)
                    bbox_list.extend([x1, y1, x2, y2])
                msg_bot = Float64MultiArray(data=bbox_list)
                self.last_bot = msg_bot
                self.last_bot_time = now
                self.last_bot_bbox = [{'bbox': tuple(map(int, b)), 'label': 'bot'} for b in sorted_bots]
            elif self.last_bot and (now - self.last_bot_time).to_sec() < self.timeout_sec:
                pass
            else:
                self.last_bot = None
                self.last_bot_bbox = None
                msg_bot = Float64MultiArray(data=[])
            self.pup_botNearestBB.publish(msg_bot)

            # Free parking slot
            if freeSlots and conf_freeSlot > self.conf["confidence_temporal"]:
                nearest_free = sorted(freeSlots, key=lambda s: (s['bbox'][1] + s['bbox'][3]) // 2, reverse=True)[0]
                x1, y1, x2, y2 = map(int, nearest_free['bbox'])
                msg_freeSlot = Float64MultiArray(data=[x1, y1, x2, y2])
                self.last_freeSlot = msg_freeSlot
                self.last_freeSlot_time = now
                self.last_freeSlot_bbox = {'bbox': (x1, y1, x2, y2), 'label': 'freeSlot'}
            elif self.last_freeSlot and (now - self.last_freeSlot_time).to_sec() < self.timeout_sec:
                pass
            else:
                self.last_freeSlot = None
                self.last_freeSlot_bbox = None
                msg_freeSlot = Float64MultiArray(data=[])
            self.pup_parkingBB.publish(msg_freeSlot)

            # Occupied parking slot
            if occupiedSlots and conf_occupiedSlot > self.conf["confidence_temporal"]:
                nearest_occ = sorted(occupiedSlots, key=lambda s: (s['bbox'][1] + s['bbox'][3]) // 2, reverse=True)[0]
                x1, y1, x2, y2 = map(int, nearest_occ['bbox'])
                msg_occupiedSlot = Float64MultiArray(data=[x1, y1, x2, y2])
                self.last_occupiedSlot = msg_occupiedSlot
                self.last_occupiedSlot_time = now
                self.last_occupiedSlot_bbox = {'bbox': (x1, y1, x2, y2), 'label': 'occupiedSlot'}
            elif self.last_occupiedSlot and (now - self.last_occupiedSlot_time).to_sec() < self.timeout_sec:
                pass
            else:
                self.last_occupiedSlot = None
                self.last_occupiedSlot_bbox = None
                msg_occupiedSlot = Float64MultiArray(data=[])
            self.pup_parkingOccupiedBB.publish(msg_occupiedSlot)

            # === Visualization for debugging/monitoring ===
            annotated = self.cv_image.copy()
            for obj in [
                self.last_duckie_lane_bbox,
                self.last_duckie_lane_right_bbox,
                self.last_bot_bbox,
                self.last_freeSlot_bbox,
                self.last_occupiedSlot_bbox
            ]:
                if obj is None:
                    continue
                
                if isinstance(obj, list):
                    objs = obj
                else:
                    objs = [obj]

                for entry in objs:
                    box = entry['bbox']
                    label = entry['label']
                    color = {
                        'duckie': (0, 255, 255),
                        'duckie_right': (150, 150, 0),
                        'bot': (255, 0, 0),
                        'freeSlot': (0, 255, 0),
                        'occupiedSlot': (0, 0, 255)
                    }.get(label, (0, 0, 0))
                    cv2.rectangle(annotated, box[:2], box[2:], color, 2)
                    cv2.putText(annotated, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            img_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            self.pup_image.publish(img_msg)
            rate.sleep()


if __name__ == '__main__':
    node = DetectParkingSlotNode(node_name='detect_object_node')
    node.run()
