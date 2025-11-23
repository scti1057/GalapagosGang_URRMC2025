from setuptools import find_packages, setup
import os

package_name = 'tb3_maze'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            ['launch/tb3_nav_bringup.launch.py']),
        (os.path.join('share', package_name, 'config'),
            ['config/nav2_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='duckiebot1',
    maintainer_email='scme1025@h-ka.de',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'frontier_explorer = tb3_maze.frontier_explorer_node:main',
            'drive_in_box = tb3_maze.tb3_drive_in_box:main',
            'mission3_bt = tb3_maze.mission3_bt_node:main',
        ],
    },
)
