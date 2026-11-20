import os
import random
import subprocess
import tempfile

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseArray
import rclpy
from rclpy.node import Node


class HazardSpawner(Node):

    def __init__(self):

        super().__init__('hazard_spawner')
        self.package_path = get_package_share_directory('warehouse_sim')
        self.fire_model = os.path.join(self.package_path, 'models', 'fire', 'model.sdf')
        self.hazard_publisher = self.create_publisher(PoseArray, '/hazards/poses', 10)
        self.hazards = []
        self.get_logger().info('Hazard spawner started')

        # Random delay between 30 and 120 seconds
        delay = random.randint(30, 120)

        self.get_logger().info(f'Hazard will spawn in {delay} seconds')
        self._spawn_timer = self.create_timer(delay, self.spawn_fire)

    def spawn_fire(self):

        self._spawn_timer.cancel()

        # Random position
        x, y = random.uniform(-6.0, 6.0), random.uniform(-4.0, 4.0)
        name = f'fire_hazard_{len(self.hazards) + 1:03d}'
        temperature_c = random.uniform(150.0, 250.0)
        self.get_logger().info(f'Spawning {name} at x={x:.2f}, y={y:.2f}, {temperature_c:.1f} °C')

        # Each name is unique so the world heat manager can match discovery events.
        with open(self.fire_model, encoding='utf-8') as model_file:
            model_sdf = model_file.read().replace(
                '@INITIAL_TEMPERATURE_K@', str(temperature_c + 273.15))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', encoding='utf-8',
                                         delete=False) as temp_model:
            temp_model.write(model_sdf)
            model_path = temp_model.name
        try:
            result = subprocess.run([
                'ros2', 'run', 'ros_gz_sim', 'create', '-world', 'warehouse_a',
                '-file', model_path, '-name', name, '-x', str(x), '-y', str(y),
                '-z', '0.0', '-allow_renaming', 'false',
            ], check=False, capture_output=True, text=True)
        finally:
            os.unlink(model_path)
        if result.returncode:
            self.get_logger().error(f'Could not spawn {name}: {result.stderr.strip()}')
            return

        self.hazards.append((x, y))
        self._publish_hazards()
        self.get_logger().warning(f'HAZARD SPAWNED: {name}')

        # Preserve the original one-hazard-at-a-behaviour, with a new
        # random interval before each subsequent hazard.
        self._spawn_timer = self.create_timer(random.uniform(20.0, 60.0), self.spawn_fire)

    def _publish_hazards(self):
        message = PoseArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'world'
        for x, y in self.hazards:
            pose = Pose()
            pose.position.x, pose.position.y = x, y
            pose.orientation.w = 1.0
            message.poses.append(pose)
        self.hazard_publisher.publish(message)


def main(args=None):

    rclpy.init(args=args)
    node = HazardSpawner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
