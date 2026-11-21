from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'warehouse_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'models', 'warehouse_robot'),
            glob('models/warehouse_robot/*')),
        (os.path.join('share', package_name, 'models', 'fire'),
            glob('models/fire/*')),
        (os.path.join('share', package_name, 'models', 'box'),
            glob('models/box/*')),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ethan Wang',
    maintainer_email='yichenwang2@gmail.com',
    description='Group 22 warehouse world and robot for Gazebo',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hazard_spawner = warehouse_sim.hazard_spawner:main',
            'fire_detection = warehouse_sim.fire_detection:main',
            'hazard_discovery = warehouse_sim.hazard_discovery:main',
            'thermal_ray_localisation = warehouse_sim.thermal_ray_localisation:main',
        ],
    },
)
