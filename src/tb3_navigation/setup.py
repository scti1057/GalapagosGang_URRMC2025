from setuptools import find_packages, setup

package_name = 'tb3_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/red_sign_detector_params.yaml']),
        ('share/' + package_name + '/config', ['config/reference_line_node_params.yaml']),
        ('share/' + package_name + '/config', ['config/red_sign_localizer_minimalized_params.yaml']),
        ('share/' + package_name + '/resource', ['resource/gg.jpg']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='duckie6',
    maintainer_email='glpa1013@h-ka.de',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'nav_goal_gui = tb3_navigation.nav_goal_gui:main',
            'red_sign_detector = tb3_navigation.red_sign_detector:main',
            'reference_line_node = tb3_navigation.reference_line_node:main',
            'red_sign_localizer = tb3_navigation.red_sign_localizer_minimalized:main',
            'red_sign_nav_client = tb3_navigation.red_sign_nav_client:main',
        ],
    },
)
