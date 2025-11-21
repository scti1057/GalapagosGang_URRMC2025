from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tb3_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Launchfiles installieren
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),

        # Config-Dateien installieren
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),

        # Behavior Trees installieren
        (os.path.join('share', package_name, 'behavior_trees'),
         glob('behavior_trees/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='duckiebot1',
    maintainer_email='scme1025@h-ka.de',
    description='Nav2 + Behavior Trees for Turtlebot3 / URRMC',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
