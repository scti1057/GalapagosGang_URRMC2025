from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'galapagos_checked_yolo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'ultralytics'],
    zip_safe=True,
    maintainer='duckie6',
    maintainer_email='glpa1013@h-ka.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_detector_node = galapagos_checked_yolo.yolo_detector_node:main',
        ],
    },
)
