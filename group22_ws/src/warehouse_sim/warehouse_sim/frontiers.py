"""Select a reachable observation pose on the known side of a SLAM frontier."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class GridFrame:
    """Convert occupancy-grid cells and map-frame metres, including origin rotation."""

    x: float
    y: float
    yaw: float
    resolution: float

    def cell(self, x, y):
        """Return row and column for a map-frame point."""
        dx, dy = x - self.x, y - self.y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (math.floor((-s * dx + c * dy) / self.resolution),
                math.floor((c * dx + s * dy) / self.resolution))

    def point(self, row, col):
        """Return the map-frame centre of a cell."""
        dx, dy = (col + 0.5) * self.resolution, (row + 0.5) * self.resolution
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * dx - s * dy, self.y + s * dx + c * dy


@dataclass(frozen=True)
class FrontierGoal:
    """A known-free viewpoint facing a boundary between free and unknown space."""

    x: float
    y: float
    yaw: float
    route_m: float
    boundary_cells: int


def select_goal(grid, frame, robot_xy, attempted=(), radius=0.22):
    """Choose the nearest eligible approach by four-connected travel distance."""
    free = grid == 0
    edge = free & ndimage.binary_dilation(grid == -1)  # four-connected by default
    labels, count = ndimage.label(edge, structure=np.ones((3, 3)))
    if not count:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    edge = sizes[labels] >= max(1, math.ceil(0.20 / frame.resolution))
    if not np.any(edge):
        return None

    # Treat unknown cells and the outside of the map as obstacles for the footprint.
    clearance = ndimage.distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1]
    safe = (clearance - math.sqrt(0.5)) * frame.resolution >= radius
    start = frame.cell(*robot_xy)
    height, width = grid.shape
    if not (0 <= start[0] < height and 0 <= start[1] < width and safe[start]):
        raise ValueError('Robot pose lacks mapped footprint clearance. Check /map and TF.')

    distance, nearest = ndimage.distance_transform_edt(~edge, return_indices=True)
    candidates = safe & (distance * frame.resolution <= radius + 0.25)
    rows, cols = np.nonzero(candidates)
    # A thin wall can separate a goal from a nearby frontier. Check the sight line.
    er, ec = nearest[:, rows, cols]
    visible = np.ones(len(rows), dtype=bool)
    for t in np.linspace(0, 1, math.ceil(2 * (radius + 0.25) / frame.resolution) + 1):
        rr = np.rint(rows + t * (er - rows)).astype(int)
        cc = np.rint(cols + t * (ec - cols)).astype(int)
        visible &= free[rr, cc]
    xs, ys = frame.point(rows, cols)
    visible &= np.hypot(xs - robot_xy[0], ys - robot_xy[1]) >= 0.40
    for x, y in attempted:
        visible &= np.hypot(xs - x, ys - y) > 0.50
    candidates[rows, cols] = visible
    if not np.any(visible):
        return None

    # Breadth-first search avoids selecting free pockets behind walls or cutting corners.
    queue = deque([(start[0], start[1], 0)])
    safe[start] = False
    while queue:
        row, col, steps = queue.popleft()
        if candidates[row, col]:
            fr, fc = nearest[:, row, col]
            x, y = frame.point(row, col)
            fx, fy = frame.point(fr, fc)
            return FrontierGoal(x, y, math.atan2(fy - y, fx - x),
                                steps * frame.resolution, int(sizes[labels[fr, fc]]))
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < height and 0 <= nc < width and safe[nr, nc]:
                safe[nr, nc] = False
                queue.append((nr, nc, steps + 1))
    return None
