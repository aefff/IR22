"""Exercise frontier action recovery without moving a robot."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from action_msgs.msg import GoalStatus
from nav2_msgs.action import FollowPath
from warehouse_sim.frontier_explorer import FrontierExplorer
from warehouse_sim.frontiers import FrontierGoal


class FrontierRecoveryTests(unittest.TestCase):
    """Check retry limits, map refresh, and frontier eligibility after a TF abort."""

    def setUp(self):
        """Build an explorer stand-in with deterministic actions and map updates."""
        self.node = MagicMock()
        self.node.stopping = False
        self.histories = []
        self.node.choose.side_effect = self.choose
        self.node.wait.side_effect = self.spin
        self.failure = SimpleNamespace(status=GoalStatus.STATUS_ABORTED, result=SimpleNamespace(
            error_code=FollowPath.Result.TF_ERROR, error_msg='transform failed'))
        self.success = SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED, result=SimpleNamespace(
            error_code=0, error_msg=''))

    def choose(self, attempted):
        """Record exclusions at each selection."""
        self.histories.append(list(attempted))
        return 'map', FrontierGoal(1., 2., 0., 3., 4)

    def spin(self, ready, seconds, description):
        """Publish a map when the explorer requests the next update."""
        if not ready():
            self.node.latest_map = object()
            self.assertTrue(ready())

    def test_recovers_without_excluding_target(self):
        """A fresh choice can revisit the failed approach within the send limit."""
        self.node.navigate.side_effect = [self.failure, self.success]
        FrontierExplorer.explore(self.node, True, 2)
        self.node.wait_for_tf_recovery.assert_called_once_with('map')
        self.assertEqual(self.histories, [[], []])
        self.assertEqual(self.node.navigate.call_count, 2)

    def test_consecutive_tf_failure_ends_the_run(self):
        """Persistent TF faults stop exploration instead of churning through goals."""
        self.node.navigate.side_effect = [self.failure, self.failure]
        with self.assertRaisesRegex(RuntimeError, 'no retry remaining'):
            FrontierExplorer.explore(self.node, True, 0)
        self.assertEqual(self.node.navigate.call_count, 2)

    def test_one_goal_limit_counts_aborts(self):
        """A one-attempt smoke test cannot send a second goal."""
        self.node.navigate.return_value = self.failure
        with self.assertRaisesRegex(RuntimeError, 'no retry remaining'):
            FrontierExplorer.explore(self.node, True, 1)
        self.node.wait_for_tf_recovery.assert_not_called()

    def test_recovery_timeout_ends_the_run(self):
        """Absent fresh transforms cannot trigger another action."""
        self.node.navigate.return_value = self.failure
        self.node.wait_for_tf_recovery.side_effect = RuntimeError('Timed out waiting for TF')
        with self.assertRaisesRegex(RuntimeError, 'Timed out'):
            FrontierExplorer.explore(self.node, True, 0)
        self.assertEqual(self.node.navigate.call_count, 1)

    def test_transform_check_uses_a_post_abort_stamp(self):
        """The latest cached transform alone cannot satisfy the recovery check."""
        clock = self.node.get_clock.return_value
        stamps = [MagicMock(nanoseconds=n) for n in (100, 110, 120)]
        stamps[1].__gt__.return_value = True
        clock.now.side_effect = stamps
        self.node.buffer.can_transform.return_value = True
        FrontierExplorer.wait_for_tf_recovery(self.node, 'map')
        self.node.buffer.can_transform.assert_called_once_with('map', 'base_link', stamps[2])
