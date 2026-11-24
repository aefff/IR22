"""Start Jazzy Nav2 alongside a running warehouse simulation and live SLAM."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    """Use the installed Nav2 defaults with warehouse-specific overrides."""
    bringup = get_package_share_directory('nav2_bringup')
    params = RewrittenYaml(
        source_file=os.path.join(bringup, 'params', 'nav2_params.yaml'),
        param_rewrites={
            'collision_monitor.ros__parameters.base_frame_id': 'base_link',
            'robot_radius': '0.22',
            'allow_unknown': 'false',
            'controller_server.ros__parameters.FollowPath.vx_max': '0.35',
            'controller_server.ros__parameters.FollowPath.wz_max': '1.0',
            # Allow for intermittent map-to-odom delivery gaps observed under WSL.
            'controller_server.ros__parameters.FollowPath.transform_tolerance': '3.0',
            'controller_server.ros__parameters.FollowPath.visualize': 'false',
        },
        root_key=None,
        convert_types=True,
    )
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'params_file': params,
            'use_sim_time': 'true',
            'autostart': 'true',
            'use_composition': 'False',
        }.items(),
    )
    return LaunchDescription([navigation])
