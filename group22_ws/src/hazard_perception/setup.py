from setuptools import find_packages, setup

package_name = 'hazard_perception'

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
    maintainer='Daniel Lawless',
    maintainer_email='daniellawless115@hotmail.com',
    description='Group 22 hazard perception: ROS-independent thermal-region extraction',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
)
