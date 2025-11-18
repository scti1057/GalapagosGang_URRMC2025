class LaneErrorNode(Node):
    def __init__(self):
        super().__init__('lane_error_node')

        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/local_costmap/costmap',
            self.costmap_callback,
            10
        )

        self.lane_pose_pub = self.create_publisher(
            PoseStamped,
            '/lane_pose',
            10
        )

    def costmap_callback(self, msg):
        # TODO: hier aus der Lane-Schicht die linke/rechte Linie extrahieren
        # TODO: Spurmitte & Tangente berechnen
        # TODO: daraus y_err & yaw_err berechnen

        lane_pose = PoseStamped()
        lane_pose.header.frame_id = 'base_link'   # z.B.
        lane_pose.header.stamp = self.get_clock().now().to_msg()

        lane_pose.pose.position.y = y_err
        lane_pose.pose.orientation = yaw_to_quaternion(yaw_err)

        self.lane_pose_pub.publish(lane_pose)
