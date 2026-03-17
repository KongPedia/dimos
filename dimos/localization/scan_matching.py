# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numpy.typing import NDArray

from dimos.msgs.geometry_msgs import Pose, Quaternion, Vector3
from dimos.msgs.nav_msgs import OccupancyGrid
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues


def transform_scan_points(scan_points: NDArray[np.float32], pose: Pose) -> NDArray[np.float32]:
    scan_2d = scan_points[:, :2]
    yaw = pose.orientation.to_euler().z
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=scan_2d.dtype)
    transformed_points = (rotation_matrix @ scan_2d.T).T
    transformed_points[:, 0] += pose.position.x
    transformed_points[:, 1] += pose.position.y
    return transformed_points


def score_scan_against_map_pose(
    scan_points: NDArray[np.float32],
    reference_map: OccupancyGrid,
    pose: Pose,
) -> float:
    if scan_points.shape[0] == 0:
        return 0.0
    return _score_scan_against_map(transform_scan_points(scan_points, pose), reference_map)


def scan_to_map_2d(
    scan_points: NDArray[np.float32],
    reference_map: OccupancyGrid,
    initial_pose: Pose,
    max_iterations: int = 50,
    convergence_threshold: float = 0.01,
) -> tuple[Pose, float]:
    """2D scan-to-map matching using iterative closest point on occupancy grid.

    Args:
        scan_points: Nx3 array of 3D points from LiDAR (only x,y used)
        reference_map: Reference occupancy grid map
        initial_pose: Initial pose estimate
        max_iterations: Maximum optimization iterations
        convergence_threshold: Convergence threshold for pose delta

    Returns:
        Tuple of (corrected_pose, match_score)
    """
    if scan_points.shape[0] == 0:
        return initial_pose, 0.0

    # Initialize with initial pose
    current_x = initial_pose.position.x
    current_y = initial_pose.position.y
    current_yaw = initial_pose.orientation.to_euler().z

    current_pose = Pose(
        position=Vector3(current_x, current_y, initial_pose.position.z),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, current_yaw)),
    )
    best_score = score_scan_against_map_pose(scan_points, reference_map, current_pose)
    step_schedule = (0.25, 0.1, 0.05)

    for step in step_schedule:
        yaw_step = max(0.025, step)
        for _ in range(max_iterations):
            delta_x = 0.0
            delta_y = 0.0
            delta_yaw = 0.0
            perturbations = [
                (step, 0.0, 0.0),
                (-step, 0.0, 0.0),
                (0.0, step, 0.0),
                (0.0, -step, 0.0),
                (0.0, 0.0, yaw_step),
                (0.0, 0.0, -yaw_step),
            ]

            for dx, dy, dyaw in perturbations:
                test_pose = Pose(
                    position=Vector3(current_x + dx, current_y + dy, initial_pose.position.z),
                    orientation=Quaternion.from_euler(Vector3(0.0, 0.0, current_yaw + dyaw)),
                )
                test_score = score_scan_against_map_pose(scan_points, reference_map, test_pose)
                if test_score > best_score:
                    best_score = test_score
                    delta_x = dx
                    delta_y = dy
                    delta_yaw = dyaw

            current_x += delta_x
            current_y += delta_y
            current_yaw += delta_yaw
            delta_magnitude = np.sqrt(delta_x**2 + delta_y**2 + delta_yaw**2)
            if delta_magnitude < convergence_threshold:
                break

    # Build corrected pose
    corrected_pose = Pose(
        position=Vector3(current_x, current_y, initial_pose.position.z),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, current_yaw)),
    )

    return corrected_pose, best_score


def _score_scan_against_map(
    points: NDArray[np.float32], reference_map: OccupancyGrid
) -> float:
    """Score scan points against occupancy grid (vectorized).

    Args:
        points: Nx2 array of 2D points in map frame
        reference_map: Reference occupancy grid

    Returns:
        Match score (0.0 to 1.0, higher is better)
    """
    if points.shape[0] == 0:
        return 0.0

    grid_x = ((points[:, 0] - reference_map.origin.position.x) / reference_map.resolution).astype(np.int32)
    grid_y = ((points[:, 1] - reference_map.origin.position.y) / reference_map.resolution).astype(np.int32)

    valid_mask = (
        (grid_x >= 0) & (grid_x < reference_map.width) &
        (grid_y >= 0) & (grid_y < reference_map.height)
    )

    if not np.any(valid_mask):
        return 0.0

    grid_x_valid = grid_x[valid_mask]
    grid_y_valid = grid_y[valid_mask]
    cell_values = reference_map.grid[grid_y_valid, grid_x_valid]

    known_mask = cell_values != CostValues.UNKNOWN
    if not np.any(known_mask):
        return 0.0

    known_cells = cell_values[known_mask]
    hits = np.sum(known_cells > CostValues.FREE)
    known_total = len(known_cells)

    if known_total == 0:
        return 0.0

    match_ratio = float(hits) / float(known_total)

    feature_penalty = 1.0
    if hits < 10:
        feature_penalty = 0.5

    return match_ratio * feature_penalty
