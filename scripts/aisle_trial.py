"""Compare one Nav2 aisle crossing with and without the scheduled warehouse A box."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path as Plan
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from aisle_box import box_present


def stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def wait_for(node, ready, seconds, description):
    deadline = time.monotonic() + seconds
    while not ready():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out waiting for {description}.")
        rclpy.spin_once(node, timeout_sec=0.1)


def robot_start(node, buffer):
    wait_for(node, lambda: buffer.can_transform("map", "base_link", Time()),
             10, "map to base_link")
    tf = buffer.lookup_transform("map", "base_link", Time()).transform
    q = tf.rotation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
    return [tf.translation.x, tf.translation.y, yaw]


def trial(node, mode, folder):
    client = ActionClient(node, NavigateToPose, "/navigate_to_pose")
    if not client.wait_for_server(timeout_sec=10):
        raise RuntimeError("Start Nav2 and wait for bt_navigator to become active.")
    if box_present():
        raise RuntimeError("The trial box is present. Clear it before either run.")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    baseline = folder / "clear.jsonl"
    output = folder / f"{mode}.jsonl"
    if output.exists():
        raise RuntimeError(f"{output} already exists. Use a fresh trial folder or archive the old run.")
    if mode == "clear":
        points = []
        def click(msg):
            if msg.header.frame_id == "map":
                points.append([msg.point.x, msg.point.y])
        node.create_subscription(PointStamped, "/clicked_point", click, 1)
        print("RViz: choose Publish Point and click the far side of the box site once.", flush=True)
        wait_for(node, lambda: bool(points), 120, "a Publish Point click in frame map")
        start = robot_start(node, buffer)
        goal = [*points[0], start[2]]
    else:
        with baseline.open() as source:
            first = json.loads(next(source, "{}"))
            last = first
            for line in source:
                last = json.loads(line)
        if first.get("event") != "setup" or last.get("outcome") != "SUCCEEDED":
            raise RuntimeError("Complete a successful clear run in this folder first.")
        goal = first["goal"]
        expected = first["start"]
        start = robot_start(node, buffer)
        angle = math.remainder(start[2] - expected[2], 2 * math.pi)
        if math.dist(start[:2], expected[:2]) > 0.30 or abs(angle) > math.radians(20):
            raise RuntimeError(f"Return to the baseline start (map x, y, yaw radians): {expected}")

    request = NavigateToPose.Goal()
    request.pose.header.frame_id = "map"
    pose = request.pose.pose
    pose.position.x, pose.position.y = goal[:2]
    pose.orientation.z, pose.orientation.w = math.sin(goal[2] / 2), math.cos(goal[2] / 2)
    with output.open("x", buffering=1) as log:
        def record(event, **data):
            log.write(json.dumps(dict(event=event, sim_s=node.get_clock().now().nanoseconds * 1e-9,
                                      **data), allow_nan=False) + "\n")

        record("setup", mode=mode, frame="map", start=start, goal=goal)
        print(f"Start {start}, goal {goal}. Recording to {output}", flush=True)
        plan_seen = False
        scan_seen = False
        def plan(msg):
            nonlocal plan_seen
            if msg.header.frame_id != "map" or not msg.poses:
                return
            xy = [[p.pose.position.x, p.pose.position.y] for p in msg.poses]
            record("plan", stamp=stamp(msg.header), xy=xy)
            if math.dist(xy[-1], goal[:2]) <= 0.5:
                plan_seen = True

        def scan(msg):
            nonlocal scan_seen
            if not msg.angle_increment or not math.isfinite(msg.angle_increment):
                return
            i = round(-msg.angle_min / msg.angle_increment)
            if 0 <= i < len(msg.ranges):
                r = msg.ranges[i]
                record("scan", stamp=stamp(msg.header), frame=msg.header.frame_id,
                       forward_m=r if math.isfinite(r) and msg.range_min <= r <= msg.range_max else None)
                scan_seen = True

        last_pose_stamp = None
        def feedback(msg):
            nonlocal last_pose_stamp
            p = msg.feedback.current_pose
            t = stamp(p.header)
            if p.header.frame_id == "map" and (last_pose_stamp is None or t >= last_pose_stamp + 0.2):
                record("pose", stamp=t, xy=[p.pose.position.x, p.pose.position.y])
                last_pose_stamp = t

        node.create_subscription(Plan, "/plan", plan, qos_profile_sensor_data)
        node.create_subscription(LaserScan, "/scan", scan, qos_profile_sensor_data)
        wait_for(node, lambda: node.count_publishers("/plan") > 0 and scan_seen,
                 10, "the Nav2 /plan publisher and a LiDAR scan")
        future = client.send_goal_async(request, feedback_callback=feedback)
        wait_for(node, future.done, 10, "goal acceptance")
        handle = future.result()
        if not handle.accepted:
            raise RuntimeError("Nav2 rejected the goal.")
        result = handle.get_result_async()
        spawner = None
        confirmed = False
        try:
            deadline = time.monotonic() + 240
            while not result.done():
                rclpy.spin_once(node, timeout_sec=0.1)
                if mode == "blocked" and plan_seen and spawner is None and not result.done():
                    spawner = subprocess.Popen(
                        [sys.executable, str(Path(__file__).with_name("aisle_box.py")), "spawn"],
                        stdout=subprocess.PIPE, text=True)
                    record("box_requested")
                if spawner is not None and not confirmed and spawner.poll() is not None:
                    report = spawner.communicate()[0]
                    if spawner.returncode:
                        raise RuntimeError("Box spawn failed. See the scenario script's terminal output.")
                    record("box_confirmed", scenario=json.loads(report))
                    confirmed = True
                if time.monotonic() >= deadline:
                    raise RuntimeError("Trial exceeded four minutes of wall time.")
            response = result.result()
            status = {GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED", GoalStatus.STATUS_ABORTED: "ABORTED",
                      GoalStatus.STATUS_CANCELED: "CANCELED"}.get(response.status, str(response.status))
            valid = plan_seen and last_pose_stamp is not None and (mode == "clear" or confirmed)
            record("result", outcome=status if valid else "INCONCLUSIVE", nav2_status=status,
                   error_code=response.result.error_code)
            print(f"Nav2 {status}. Trial {'recorded' if valid else 'inconclusive (missing plan, pose, or box confirmation)'}.")
        finally:
            try:
                if not result.done():
                    cancel = handle.cancel_goal_async()
                    wait_for(node, cancel.done, 5, "Nav2 cancellation")
            finally:
                if spawner is not None and spawner.poll() is None:
                    spawner.communicate(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["clear", "blocked"])
    parser.add_argument("folder", type=Path, help="One new folder per pair, in the same live map")
    args = parser.parse_args()
    args.folder.mkdir(exist_ok=True, parents=True)
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("aisle_navigation_trial",
                             parameter_overrides=[Parameter("use_sim_time", value=True)])
    try:
        trial(node, args.mode, args.folder)
    except (OSError, RuntimeError, ValueError, TransformException, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
