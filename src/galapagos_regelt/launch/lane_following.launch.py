from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch all Galapagos lane-following nodes:

      - lane_detect_node
      - finish_line_node
      - control_node
      - drive_node
    """

    lane_detect = Node(
        package='galapagos_regelt',
        executable='lane_detect_node',
        name='lane_detect_node',
        output='screen',
        parameters=[{
            'camera_topic': '/camera/image_raw/compressed',
            'config_file': 'lane_detect.yaml',
            'max_rate_hz': 10.0,
        }]
    )

    finish_line = Node(
        package='galapagos_regelt',
        executable='finish_line_node',
        name='finish_line_node',
        output='screen',
        parameters=[{
            'camera_topic': '/camera/image_raw/compressed',
            'config_file': 'finish_line_detect.yaml',
            'max_rate_hz': 10.0,
        }]
    )

    control = Node(
        package='galapagos_regelt',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{
            'max_rate_hz': 20.0,
        }]
    )

    drive = Node(
        package='galapagos_regelt',
        executable='drive_node',
        name='drive_node',
        output='screen',
        parameters=[{
            'config_file': 'drive.yaml',
            'max_rate_hz': 20.0,
        }]
    )

    return LaunchDescription([
        lane_detect,
        finish_line,
        control,
        drive,
    ])
