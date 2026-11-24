"""Regression tests for map-derived frontier goals."""

import math
import unittest

import numpy as np

from warehouse_sim.frontiers import GridFrame, select_goal


class FrontierTests(unittest.TestCase):
    """Exercise footprint clearance, reachability and map frame handling."""

    def setUp(self):
        """Create a partially observed ten-centimetre grid."""
        self.frame = GridFrame(-1.0, 2.0, 0.0, 0.10)
        self.grid = np.full((40, 60), 100, dtype=np.int8)
        self.grid[2:-2, 2:40] = 0
        self.grid[2:-2, 40:-2] = -1
        self.robot = self.frame.point(20, 12)

    def test_approach_known_side_and_face_unknown(self):
        """Choose a known-free cell with clearance from the unseen area."""
        goal = select_goal(self.grid, self.frame, self.robot)
        row, col = self.frame.cell(goal.x, goal.y)
        self.assertEqual(row, 20)
        self.assertTrue(34 <= col <= 37)
        self.assertEqual(self.grid[row, col], 0)
        self.assertAlmostEqual(goal.yaw, 0)
        self.assertAlmostEqual(goal.route_m, (col - 12) * 0.10)
        self.assertEqual(goal.boundary_cells, 36)

    def test_disconnected_and_too_narrow(self):
        """A thin opening and a closed wall both disconnect the viewpoint."""
        self.grid[:, 25] = 100
        self.assertIsNone(select_goal(self.grid, self.frame, self.robot))
        self.grid[19:21, 25] = 0
        self.assertIsNone(select_goal(self.grid, self.frame, self.robot))
        self.grid[15:26, 25] = 0
        self.assertIsNotNone(select_goal(self.grid, self.frame, self.robot))

    def test_repeat_and_completed_map(self):
        """Skip recently attempted locations and report a completely known map."""
        first = select_goal(self.grid, self.frame, self.robot)
        other = select_goal(self.grid, self.frame, self.robot, [(first.x, first.y)])
        self.assertGreater(math.hypot(other.x - first.x, other.y - first.y), 0.50)
        self.grid[self.grid == -1] = 0
        self.assertIsNone(select_goal(self.grid, self.frame, self.robot))

    def test_rotated_origin(self):
        """Preserve the physical target cell and rotate the goal heading."""
        rotated = GridFrame(5.0, -4.0, math.pi / 2, 0.10)
        goal = select_goal(self.grid, rotated, rotated.point(20, 12))
        reference = select_goal(self.grid, self.frame, self.robot)
        self.assertEqual(rotated.cell(goal.x, goal.y), self.frame.cell(reference.x, reference.y))
        self.assertAlmostEqual(goal.yaw, math.pi / 2)

    def test_unsafe_start(self):
        """Report an invalid start instead of picking a disconnected replacement."""
        with self.assertRaisesRegex(ValueError, 'footprint'):
            select_goal(self.grid, self.frame, self.frame.point(20, 1))
