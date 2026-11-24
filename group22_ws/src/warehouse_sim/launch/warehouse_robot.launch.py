"""
Start a warehouse world, spawn the robot and bridge its topics to ROS 2.

Build the workspace first, then:
    ros2 launch warehouse_sim warehouse_robot.launch.py
    ros2 launch warehouse_sim warehouse_robot.launch.py x:=-5.0 y:=-1.0 yaw:=1.57

All files are found through the installed package share directory, so it
works on any machine (Jazzy/Harmonic or Lyrical/Jetty) without editing.
"""
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_prefix, get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def static_tf(child, x, y, z):
    """Return a static transform for a fixed sensor pose relative to base_link (as model.sdf)."""
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=f'{child}_tf',
        arguments=['--x', str(x), '--y', str(y), '--z', str(z),
                   '--frame-id', 'base_link', '--child-frame-id', child],
        parameters=[{'use_sim_time': True}],
    )


def spawn_and_bridge(context, world_file, robot_sdf):
    """Target the SDF world explicitly for spawning and simulation time."""
    world_name = ET.parse(world_file.perform(context)).getroot().find('world').attrib['name']
    clock_topic = f'/world/{world_name}/clock'

    # Put the robot into the running world
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-world', world_name, '-name', 'warehouse_robot', '-file', robot_sdf,
                   '-x', LaunchConfiguration('x'),
                   '-y', LaunchConfiguration('y'),
                   '-z', '0.0',
                   '-Y', LaunchConfiguration('yaw')],
        output='screen',
    )

    # ]  ROS -> Gazebo,  [  Gazebo -> ROS
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            f'{clock_topic}@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/thermal/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/thermal/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/hazards/discovered@std_msgs/msg/String]gz.msgs.StringMsg'
        ],
        remappings=[(clock_topic, '/clock')],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )
    return [spawn, bridge]


def generate_launch_description():
    pkg_share = get_package_share_directory('warehouse_sim')
    plugin_prefix = get_package_prefix('warehouse_hazards')
    plugin_dir = os.path.join(plugin_prefix, 'lib')
    
    # Which world to load and where the robot starts
    args = [
        DeclareLaunchArgument('world', default_value='warehouse_a.sdf',
                              description='World file in the worlds/ folder'),
        DeclareLaunchArgument('spawn_hazards', default_value='true',
                              description='Spawn dynamic scenario hazards'),
        DeclareLaunchArgument('slam', default_value='false',
                              description='Build a live map with SLAM Toolbox'),
        DeclareLaunchArgument('x', default_value='-4.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        SetEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH',
                               [plugin_dir, ':', EnvironmentVariable(
                                   'GZ_SIM_SYSTEM_PLUGIN_PATH', default_value='')]),
    ]
    world = PathJoinSubstitution([pkg_share, 'worlds', LaunchConfiguration('world')])

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r ', world]}.items(),
    )

    hazard_spawner = Node(
        package='warehouse_sim',
        executable='hazard_spawner',
        condition=IfCondition(LaunchConfiguration('spawn_hazards')),
        output='screen'
    )
    
    fire_detection = Node(
        package='warehouse_sim',
        executable='fire_detection',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # Simulation only: uses ground-truth hazard poses to drive the heat manager.
    hazard_discovery = Node(
        package='warehouse_sim',
        executable='hazard_discovery',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # Camera rays in base_link from thermal observations, CameraInfo and TF.
    # The -0.5 px principal-point shift and REP 103 body convention were
    # validated in R1 for this robot's 320 x 240 thermal camera in gz-sim 8
    # (hazard_perception/test/data/r1_thermal_geometry); re-validate them for
    # any other camera or simulator version.
    thermal_ray_localisation = Node(
        package='warehouse_sim',
        executable='thermal_ray_localisation',
        parameters=[{
            'use_sim_time': True,
            'target_frame': 'base_link',
            'camera_frame': 'thermal_link',
            'principal_point_shift_px': -0.5,
            'camera_frame_convention': 'rep103_body',
        }],
        output='screen',
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('slam_toolbox'), 'launch', 'online_async_launch.py'])),
        condition=IfCondition(LaunchConfiguration('slam')),
        launch_arguments={
            'use_sim_time': 'true',
            'slam_params_file': os.path.join(pkg_share, 'config', 'warehouse_slam.yaml'),
        }.items(),
    )

    return LaunchDescription(args + [
        slam,
        gz_sim,
        OpaqueFunction(
            function=spawn_and_bridge,
            args=[world, os.path.join(pkg_share, 'models', 'warehouse_robot', 'model.sdf')],
        ),
        hazard_spawner,
        fire_detection,
        hazard_discovery,
        thermal_ray_localisation,
        static_tf('lidar_link', 0.0, 0.0, 0.21),
        static_tf('camera_link', 0.185, 0.04, 0.13),
        static_tf('thermal_link', 0.185, -0.04, 0.13),
    ])
