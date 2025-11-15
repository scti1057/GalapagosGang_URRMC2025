#!/usr/bin/env python3

class BirdseyeLidarNode(Node):
    def __init__(self):
        super().__init__('birdseye_lidar_node')

        # --- Konfiguration (Anpassen an deinen Turtlebot!) ---
        self.img_width = 640
        self.img_height = 480
        
        # 1. Homographie-Punkte für die Kamera (Trapez -> Rechteck)
        # Diese Punkte definieren den Bodenbereich im Kamerabild
        self.src_points = np.float32([
            [180, 300],  # Oben Links (im Bild)
            [460, 300],  # Oben Rechts
            [40, 470],   # Unten Links
            [600, 470]   # Unten Rechts
        ])
        
        # Zielpunkte in der Vogelperspektive (Rechteck)
        self.dst_points = np.float32([
            [200, 0],    # Oben Links
            [440, 0],    # Oben Rechts
            [200, 480],  # Unten Links
            [440, 480]   # Unten Rechts
        ])

        # 2. Skalierung (Wie viele Pixel entsprechen einem Meter in der Birdseye View?)
        # Dies muss mit dst_points übereinstimmen. Beispiel: Bild ist 4.8m hoch -> 100px/m
        self.pixels_per_meter = 100.0 
        self.birdseye_center_x = 320 # Wo ist der Roboter auf der X-Achse im Bild? (Mitte)
        self.birdseye_offset_y = 480 # Wo ist der Roboter auf der Y-Achse? (Ganz unten)

        # --- ROS Setup ---
        self.bridge = CvBridge()
        
        # Subscriber
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # Publisher
        self.pub_birdseye = self.create_publisher(Image, '/camera/birdseye_fused', 10)

        # Speicher für den letzten Scan
        self.latest_scan = None
        
        # Berechne die Transformationsmatrix einmalig beim Start
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.get_logger().info('Birdseye Node gestartet und Matrix berechnet.')

    def scan_callback(self, msg):
        # Wir speichern nur den Scan, die Verarbeitung passiert im Bild-Callback
        # um Synchronisationsprobleme einfach zu halten.
        self.latest_scan = msg

    def image_callback(self, msg):
        try:
            # 1. ROS Image zu OpenCV Image konvertieren
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 2. Birdseye View erstellen (Warp Perspective)
            birdseye_img = cv2.warpPerspective(cv_image, self.M, (self.img_width, self.img_height))
            
            # 3. LIDAR Daten einzeichnen (wenn vorhanden)
            if self.latest_scan is not None:
                self.draw_lidar_on_image(birdseye_img, self.latest_scan)

            # 4. Ergebnis publishen
            out_msg = self.bridge.cv2_to_imgmsg(birdseye_img, "bgr8")
            self.pub_birdseye.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Fehler im Image Processing: {e}')

    def draw_lidar_on_image(self, img, scan):
        # LIDAR Daten in kartesische Koordinaten umwandeln
        angle = scan.angle_min
        
        for r in scan.ranges:
            # Ungültige Messungen (inf/nan) ignorieren
            if not np.isinf(r) and not np.isnan(r) and r > scan.range_min and r < scan.range_max:
                
                # Polarkoordinaten zu Kartesisch (im Roboter-Frame)
                # x ist nach vorne, y ist nach links
                x_robot = r * math.cos(angle)
                y_robot = r * math.sin(angle)
                
                # Transformation in Bildkoordinaten (Pixel)
                # Im Bild: u (rechts), v (unten). 
                # Roboter x (vorne) -> Bild v (nach oben, also minus)
                # Roboter y (links) -> Bild u (nach links, also minus)
                
                # Achtung: Koordinatensystem muss zur warpPerspective passen!
                # Hier nehmen wir an: Roboter ist unten mittig und schaut nach oben.
                
                u = int(self.birdseye_center_x - (y_robot * self.pixels_per_meter))
                v = int(self.birdseye_offset_y - (x_robot * self.pixels_per_meter))

                # Zeichne Punkt, wenn er im Bildbereich liegt
                if 0 <= u < self.img_width and 0 <= v < self.img_height:
                    # Roter Kreis für LIDAR Punkt
                    cv2.circle(img, (u, v), 3, (0, 0, 255), -1)
            
            angle += scan.angle_increment

def main(args=None):
    rclpy.init(args=args)
    node = BirdseyeLidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()