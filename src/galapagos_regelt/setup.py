from setuptools import setup
import os
from glob import glob

package_name = 'galapagos_regelt'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # install config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='YOUR_NAME',
    maintainer_email='your@email.com',
    description='Lane detection for TurtleBot3 (GalapagosGang).',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detect_node = galapagos_regelt.lane_detect_node:main',
            'finish_line_node = galapagos_regelt.finish_line_node:main',
            'control_node = galapagos_regelt.control_node:main',
            'drive_node = galapagos_regelt.drive_node:main',
            'lida_detection_node = galapagos_regelt.lida_detection_node:main',
            'red_sign_detect_node = galapagos_regelt.red_sign_detect_node:main',
            'parcour_node = galapagos_regelt.parcour_node:main',
            'yaw_node = galapagos_regelt.yaw_node:main',
            'blue_pal_detect_node = galapagos_regelt.blue_pal_detect_node:main'
        ],
    },
)
