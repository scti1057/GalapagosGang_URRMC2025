from setuptools import find_packages, setup

package_name = 'tb3_lane_fusion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='duckie5',
    maintainer_email='duckie.town@web.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'test_node = tb3_lane_fusion.test_node:main',
        'slam_interface_node = tb3_lane_fusion.slam_interface_node:main',
        'lane_bev_node = tb3_lane_fusion.lane_bev_node:main',
        'lane_map_node = tb3_lane_fusion.lane_map_node:main',
        ],
    },
)
