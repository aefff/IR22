"""Preview frontier goals or drive to them one at a time with Nav2."""

import argparse
import math
import signal
import time

from action_msgs.msg import GoalStatus
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from tf2_ros import Buffer, TransformException, TransformListener

from .frontiers import GridFrame, select_goal


def yaw(q):
    """Return planar heading from a ROS quaternion."""
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


class FrontierExplorer(Node):
    """Run a sensor-map exploration session in a single ROS executor."""

    def __init__(self):
        """Subscribe to the live map and TF; create the Nav2 action client."""
        super().__init__('frontier_explorer', parameter_overrides=[
            Parameter('use_sim_time', value=True)])
        self.latest_map = None
        self.stopping = False
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, qos)
        self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')

    def on_map(self, msg):
        """Store the latest complete map."""
        self.latest_map = msg

    def wait(self, condition, seconds, description, allow_stop=True):
        """Spin until ready, using monotonic time for startup and cancellation deadlines."""
        deadline = time.monotonic() + seconds
        while not condition():
            if allow_stop and self.stopping:
                raise InterruptedError
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Timed out waiting for {description}.')
            rclpy.spin_once(self, timeout_sec=0.1)

    def choose(self, attempted):
        """Locate the robot at decision time and select a viewpoint in the map frame."""
        msg = self.latest_map
        frame = msg.header.frame_id
        self.wait(lambda: self.buffer.can_transform(frame, 'base_link', Time()),
                  10, f'{frame} to base_link')
        pose = self.buffer.lookup_transform(frame, 'base_link', Time()).transform.translation
        origin = msg.info.origin
        geometry = GridFrame(origin.position.x, origin.position.y,
                             yaw(origin.orientation), msg.info.resolution)
        grid = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        return frame, select_goal(grid, geometry, (pose.x, pose.y), attempted)

    def navigate(self, frame, point):
        """Execute one owned goal and cancel it if the operator stops exploration."""
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = frame
        goal.pose.pose.position.x = point.x
        goal.pose.pose.position.y = point.y
        goal.pose.pose.orientation.z = math.sin(point.yaw / 2)
        goal.pose.pose.orientation.w = math.cos(point.yaw / 2)
        if self.stopping:
            raise InterruptedError
        pending = self.nav.send_goal_async(goal)
        # Acquire the handle even if Ctrl+C arrives while acceptance is in flight.
        self.wait(pending.done, 15, 'Nav2 goal acknowledgement; check Nav2 status', False)
        handle = pending.result()
        if not handle.accepted:
            raise RuntimeError('Nav2 rejected the goal. Check its lifecycle state.')
        result = handle.get_result_async()
        try:
            self.wait(result.done, 240, 'Nav2 result')
            return result.result()
        finally:
            if not result.done():
                cancel = handle.cancel_goal_async()
                self.wait(cancel.done, 8, 'Nav2 cancellation acknowledgement', False)
                self.wait(result.done, 8, 'Nav2 to finish the canceled goal', False)

    def explore(self, execute, limit):
        """Preview once, or navigate until stopped, exhausted, or the attempt limit is hit."""
        attempted = []
        reached = 0
        self.wait(lambda: self.latest_map is not None, 15, 'a live /map')
        if execute:
            self.wait(self.nav.server_is_ready, 15, 'the Nav2 action server')
        while not self.stopping:
            frame, point = self.choose(attempted)
            if point is None:
                self.get_logger().info(f'No further eligible reachable frontier. '
                                       f'{reached}/{len(attempted)} goals reached.')
                return
            self.get_logger().info(
                f'Frontier approach in {frame}: x={point.x:.2f}, y={point.y:.2f}, '
                f'yaw={point.yaw:.2f}, grid route={point.route_m:.2f} m, '
                f'boundary={point.boundary_cells} cells.')
            if not execute:
                self.get_logger().info('Preview complete. No navigation goal sent.')
                return
            response = self.navigate(frame, point)
            attempted.append((point.x, point.y))
            code = response.result.error_code
            if response.status == GoalStatus.STATUS_SUCCEEDED:
                reached += 1
                self.get_logger().info(f'Reached frontier approach {len(attempted)}.')
            elif (response.status == GoalStatus.STATUS_ABORTED and
                  code in (ComputePathToPose.Result.NO_VALID_PATH,
                           ComputePathToPose.Result.GOAL_OCCUPIED)):
                self.get_logger().warn(f'Approach unavailable (Nav2 {code}); trying elsewhere.')
            else:
                raise RuntimeError(f'Nav2 stopped: status={response.status}, error={code}, '
                                   f'{response.result.error_msg}')
            if limit and len(attempted) >= limit:
                self.get_logger().info(f'Attempt limit reached: {reached}/{limit} goals reached.')
                return
            previous = self.latest_map
            self.wait(lambda: self.latest_map is not previous, 15, 'a map update after navigation')


def main(args=None):
    """Start the explorer; default to a read-only preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--navigate', action='store_true', help='send goals to Nav2')
    parser.add_argument('--max-goals', type=int, default=0,
                        help='maximum goal attempts during navigation (0 runs to exhaustion)')
    options = parser.parse_args(remove_ros_args(args=args)[1:])
    if options.max_goals < 0:
        parser.error('--max-goals must be nonnegative')
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = FrontierExplorer()

    def stop(signum, frame):
        node.stopping = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        node.explore(options.navigate, options.max_goals)
    except InterruptedError:
        node.get_logger().info('Exploration interrupted.')
    except (RuntimeError, TransformException, ValueError) as exc:
        node.get_logger().error(str(exc))
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0
