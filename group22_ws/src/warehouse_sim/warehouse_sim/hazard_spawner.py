import rclpy
from rclpy.node import Node

import random
import subprocess

from ament_index_python.packages import get_package_share_directory
import os

class HazardSpawner(Node):

  def __init__(self):

    super().__init__('hazard_spawner')

    self.package_path = get_package_share_directory(
      'warehouse_sim'
    )

    self.fire_model = os.path.join(
      self.package_path,
      'models',
      'fire',
      'model.sdf'
    )

    self.get_logger().info('Hazard spawner started')

    # Random delay between 30 and 120 seconds
    delay = random.randint(30, 120)

    self.get_logger().info(
      f'Fire will spawn in {delay} seconds'
    )

    self.timer = self.create_timer(
      delay,
      self.spawn_fire
    )

  def spawn_fire(self):

    self.timer.cancel()

    # Random position
    x = random.uniform(-6.0, 6.0)
    y = random.uniform(-4.0, 4.0)

    self.get_logger().info(
      f'Spawning fire at x={x:.2f}, y={y:.2f}'
    )

    subprocess.Popen([
      'ros2', 'run', 'ros_gz_sim', 'create',
      '-world', 'warehouse_a',
      '-file', self.fire_model,
      '-name', 'fire_hazard',
      '-x', str(x),
      '-y', str(y),
      '-z', '0.0'
    ])

    self.get_logger().warning(
      'FIRE HAZARD SPAWNED!'
    )

def main(args=None):

  rclpy.init(args=args)
  node = HazardSpawner()
  rclpy.spin(node)
  node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__':
  main()
