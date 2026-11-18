# R1 experiment driver (validation tooling, not production code). Places a hot target with Gazebo's
# set_pose service, records the operational detector's observations, and
# reads back true poses from Gazebo for OFFLINE evaluation.
import json
import math
import re
import statistics
import subprocess
import sys
import time

from hazard_interfaces.msg import ThermalFrameObservation
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage
import tf2_ros

WORLD = 'warehouse_a'
OUT = sys.argv[1]


def gz_pose_info():
    text = subprocess.run(['gz', 'topic', '-e', '-t', f'/world/{WORLD}/pose/info', '-n', '1'],
                          capture_output=True, text=True, timeout=30).stdout
    poses = {}
    for block in re.findall(r'pose \{(.*?)\n\}', text, re.S):
        name = re.search(r'name: "([^"]+)"', block).group(1)

        def field(section, key):
            m = re.search(section + r' \{([^}]*)\}', block)
            if not m:
                return 0.0
            k = re.search(rf'\b{key}: ([-0-9.e+]+)', m.group(1))
            return float(k.group(1)) if k else 0.0
        pos = [field('position', k) for k in 'xyz']
        q = [field('orientation', k) for k in 'xyz'] + [field('orientation', 'w') or 0.0]
        if not re.search(r'orientation \{[^}]*\bw:', block):
            q[3] = 0.0 if any(q[:3]) else 1.0
        poses.setdefault(name, (pos, q))  # model entries come before same-named links
    return poses


def set_pose(name, x, y, z, yaw=0.0):
    req = (f'name: "{name}", position: {{x: {x}, y: {y}, z: {z}}}, '
           f'orientation: {{z: {math.sin(yaw / 2)}, w: {math.cos(yaw / 2)}}}')
    r = subprocess.run(['gz', 'service', '-s', f'/world/{WORLD}/set_pose', '--reqtype',
                        'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean', '--timeout', '5000',
                        '--req', req], capture_output=True, text=True, timeout=30)
    assert 'data: true' in r.stdout, (name, r.stdout, r.stderr)


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Driver(Node):

    def __init__(self):
        super().__init__('r1_driver', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.observations, self.camera_info, self.clock, self.tf_samples = [], None, None, []
        self.create_subscription(ThermalFrameObservation, '/thermal/observations',
                                 self.observations.append, 50)
        self.create_subscription(CameraInfo, '/thermal/camera_info',
                                 lambda m: setattr(self, 'camera_info', m), 10)
        self.create_subscription(Clock, '/clock', lambda m: setattr(self, 'clock', m.clock), 10)
        self.create_subscription(TFMessage, '/tf', self.on_tf, 50)
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)

    def on_tf(self, msg):
        if self.clock is not None and len(self.tf_samples) < 50:
            for t in msg.transforms:
                self.tf_samples.append({
                    'parent': t.header.frame_id, 'child': t.child_frame_id,
                    'stamp': t.header.stamp.sec + t.header.stamp.nanosec * 1e-9,
                    'clock': self.clock.sec + self.clock.nanosec * 1e-9})

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def sim_now(self):
        return self.clock.sec + self.clock.nanosec * 1e-9

    def measure(self, settle_sim=0.4, frames=5, timeout=30):
        self.spin_for(0.2)
        after = self.sim_now() + settle_sim
        self.observations.clear()
        end = time.monotonic() + timeout
        picked = []
        while time.monotonic() < end and len(picked) < frames:
            rclpy.spin_once(self, timeout_sec=0.05)
            picked = [o for o in self.observations
                      if o.header.stamp.sec + o.header.stamp.nanosec * 1e-9 >= after]
        return picked


def summarise(obs):
    counts = [len(o.regions) for o in obs]
    big = [max(o.regions, key=lambda r: r.area_px) for o in obs if o.regions]
    return {
        'frames': len(obs), 'region_counts': counts,
        'stamps': [o.header.stamp.sec + o.header.stamp.nanosec * 1e-9 for o in obs],
        'frame_ids': sorted({o.header.frame_id for o in obs}),
        'u': statistics.median(r.centroid_x for r in big) if big else None,
        'v': statistics.median(r.centroid_y for r in big) if big else None,
        'u_spread': (max(r.centroid_x for r in big) - min(r.centroid_x for r in big))
        if big else None,
        'area': statistics.median(r.area_px for r in big) if big else None,
        'bbox': [big[-1].bbox_x, big[-1].bbox_y, big[-1].bbox_width, big[-1].bbox_height]
        if big else None,
        'touches_edge': big[-1].touches_image_boundary if big else None,
    }


def main():
    rclpy.init()
    d = Driver()
    while d.camera_info is None or d.clock is None:
        d.spin_for(0.2)
    d.spin_for(2.0)
    result = {'camera_info': {
        'frame_id': d.camera_info.header.frame_id, 'width': d.camera_info.width,
        'height': d.camera_info.height, 'distortion_model': d.camera_info.distortion_model,
        'd': list(d.camera_info.d), 'k': list(d.camera_info.k), 'r': list(d.camera_info.r),
        'p': list(d.camera_info.p)}}
    st = d.buffer.lookup_transform('base_link', 'thermal_link', rclpy.time.Time())
    result['tf_base_to_thermal'] = {
        't': [st.transform.translation.x, st.transform.translation.y, st.transform.translation.z],
        'q': [st.transform.rotation.x, st.transform.rotation.y, st.transform.rotation.z,
              st.transform.rotation.w],
        'stamp': st.header.stamp.sec + st.header.stamp.nanosec * 1e-9}
    result['tf_samples'] = d.tf_samples[:20]

    cases = json.loads(sys.argv[2])
    result['cases'] = []
    for case in cases:
        if case['kind'] == 'robot':
            set_pose('warehouse_robot', case['x'], case['y'], 0.0, case['yaw'])
            d.spin_for(1.5)
            continue
        if case['kind'] == 'park':
            set_pose(case['target'], *case['xyz'])
            continue
        target = case['target']
        poses = gz_pose_info()
        # Placement assumes the identity mount rotation (verified in R1);
        # evaluation uses the full recorded link pose.
        (rp, rq), lp = poses['warehouse_robot'], poses['thermal_link'][0]
        ryaw = yaw_of(rq)
        cam = (rp[0] + math.cos(ryaw) * lp[0] - math.sin(ryaw) * lp[1],
               rp[1] + math.sin(ryaw) * lp[0] + math.cos(ryaw) * lp[1], rp[2] + lp[2])
        az, el, dist = case['az'], case['el'], case['dist']
        gx = cam[0] + dist * math.cos(el) * math.cos(ryaw + az)
        gy = cam[1] + dist * math.cos(el) * math.sin(ryaw + az)
        gz_ = 0.0 if case.get('ground') else cam[2] + dist * math.sin(el)
        set_pose(target, gx, gy, gz_)
        obs = d.measure()
        after = gz_pose_info()
        result['cases'].append({**case, 'commanded_world': [gx, gy, gz_],
                                'robot_world': after['warehouse_robot'],
                                'thermal_link_in_model': after['thermal_link'],
                                'target_world': after[target][0],
                                'odom_hint': None, **summarise(obs)})
        print(case['label'], result['cases'][-1]['u'], result['cases'][-1]['v'], flush=True)
    json.dump(result, open(OUT, 'w'), indent=1)
    d.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
