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

    # --- Lane detection (white/yellow, x_* topics) ---
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
        blue_pal_detect_node,
        control_node,
        drive_node,
        yaw_node,
    ])
