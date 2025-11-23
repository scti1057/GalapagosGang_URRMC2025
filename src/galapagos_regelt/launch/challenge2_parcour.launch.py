#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Common camera topic
    camera_topic = '/camera/image_raw/compressed'

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

    # --- Lidar detection (r_lidar) ---
    lidar_detection_node = Node(
        package='galapagos_regelt',
        executable='lida_detection_node',   # adjust if your executable name differs
        name='lida_detection_node',
        output='screen',
        parameters=[{
            'config_file': 'lidar_detect.yaml',
            'max_rate_hz': 10.0,
        }],
    )

    # --- Red sign detection (x_red, red_sign_big) ---
    red_sign_detect_node = Node(
        package='galapagos_regelt',
        executable='red_sign_detect_node',
        name='red_sign_detect_node',
        output='screen',
        parameters=[{
            'camera_topic': camera_topic,
            'config_file': 'red_sign_detect.yaml',
            'max_rate_hz': 10.0,
            'debug_visualization': False,     # set True if you want red-sign debug window
        }],
    )

    # --- Parcour high-level planner (parcour) ---
    parcour_node = Node(
        package='galapagos_regelt',
        executable='parcour_node',
        name='parcour_node',
        output='screen',
        parameters=[{
            'camera_topic': camera_topic,
            'config_file': 'parcour.yaml',
            'max_rate_hz': 10.0,
            'debug_visualization': True,      # we keep this in debug mode as you requested
        }],
    )

    # --- Control node (bridge + mode logic) ---
    control_node = Node(
        package='galapagos_regelt',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{
            'image_width_px': 640.0,
            'max_rate_hz': 20.0,
            'debug_visualization': False,     # or True if you still want its debug view
            'camera_topic': camera_topic,
            # mode is hardcoded in the script; make sure self.mode = "parcour"
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
        lidar_detection_node,
        red_sign_detect_node,
        parcour_node,
        control_node,
        drive_node,
        yaw_node,
    ])
