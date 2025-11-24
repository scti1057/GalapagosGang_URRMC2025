#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Common topics
    camera_topic = '/camera/image_raw/compressed'
    scan_topic = '/scan'

    # --- Lane detection (white/yellow, x_* topics) ---
    lane_detect_node = Node(
        package='galapagos_regelt',
        executable='lane_detect_node',
        name='lane_detect_node',
        output='screen',
        parameters=[{
            'camera_topic': camera_topic,
            'config_file': 'lane_detect.yaml',
            'max_rate_hz': 10.0,
            'debug_visualization': False,
        }],
    )

    # --- Blue pallet free-space detection (pal_free) ---
    blue_pal_detect_node = Node(
        package='galapagos_regelt',
        executable='blue_pal_detect_node',
        name='blue_pal_detect_node',
        output='screen',
        parameters=[{
            'camera_topic': camera_topic,
            'config_file': 'pal_free_params.yaml',
            'max_rate_hz': 10.0,
        }],
    )

    # --- LiDAR object detection -> r_lidar ---
    lida_detection_node = Node(
        package='galapagos_regelt',
        executable='lida_detection_node',  # adjust if your entry point name differs
        name='lida_detection_node',
        output='screen',
        parameters=[{
            'scan_topic': scan_topic,
            'config_file': 'lidar_detect.yaml',
            'max_rate_hz': 10.0,
        }],
    )


    # --- Paletting state machine (uses pal_free, r_lidar, /yolo/sign_detections) ---
    paletting_node = Node(
        package='galapagos_regelt',
        executable='paletting_node',  # adjust to your console_script name
        name='paletting_node',
        output='screen',
        parameters=[{
            'config_file': 'paletting.yaml',
            'max_rate_hz': 10.0,
        }],
    )

    # --- Control node (lane_following + paletting override) ---
    control_node = Node(
        package='galapagos_regelt',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{
            'image_width_px': 640.0,
            'max_rate_hz': 20.0,
            'debug_visualization': False,  # True if you want the OpenCV window
            'camera_topic': camera_topic,
            'config_file': 'control.yaml',
            # mode is hardcoded in the script; set self.mode = "paletting" there
        }],
    )

    # --- Drive node (PID on x_tar -> cmd_vel) ---
    drive_node = Node(
        package='galapagos_regelt',
        executable='drive_node',
        name='drive_node',
        output='screen',
        parameters=[{
            'config_file': 'drive.yaml',
            'max_rate_hz': 20.0,
        }],
    )

    # --- Yaw node (PID on yaw_init / yaw_tar -> cmd_vel) ---
    yaw_node = Node(
        package='galapagos_regelt',
        executable='yaw_node',
        name='yaw_node',
        output='screen',
        parameters=[{
            'config_file': 'yaw.yaml',
            'max_rate_hz': 20.0,
        }],
    )

    return LaunchDescription([
        lane_detect_node,
        blue_pal_detect_node,
        lida_detection_node,
        paletting_node,
        control_node,
        drive_node,
        yaw_node,
    ])
