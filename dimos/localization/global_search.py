from dataclasses import dataclass
import heapq
import math
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from dimos.localization.scan_matching import scan_to_map_2d, score_scan_against_map_pose
from dimos.localization.transform_utils import as_pose_stamped, estimate_map_from_odom
from dimos.msgs.geometry_msgs import Pose, PoseStamped
from dimos.msgs.nav_msgs import OccupancyGrid
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass(frozen=True)
class GlobalSearchConfig:
    xy_step: float = 1.0
    yaw_step: float = math.radians(30.0)
    max_candidates: int = 8
    max_scan_points: int = 1600
    search_radius: float | None = None


def search_global_map_from_odom(
    scan_points: NDArray[np.float32],
    reference_map: OccupancyGrid,
    odom_pose: PoseStamped,
    initial_map_from_odom: Pose,
    *,
    max_iterations: int = 50,
    convergence_threshold: float = 0.01,
    search_config: GlobalSearchConfig | None = None,
) -> tuple[Pose, float]:
    cfg = search_config or GlobalSearchConfig()
    sampled_points = _sample_scan_points(scan_points, cfg.max_scan_points)
    best_pose = initial_map_from_odom
    best_score = score_scan_against_map_pose(sampled_points, reference_map, initial_map_from_odom)
    candidate_heap: list[tuple[float, float, float, float, Pose]] = [
        (
            best_score,
            initial_map_from_odom.position.x,
            initial_map_from_odom.position.y,
            initial_map_from_odom.orientation.to_euler().z,
            initial_map_from_odom,
        )
    ]

    logger.info(
        "Global search started",
        sampled_scan_points=sampled_points.shape[0],
        xy_step=cfg.xy_step,
        yaw_step=cfg.yaw_step,
        max_candidates=cfg.max_candidates,
    )

    candidate_count = 0
    for candidate in _generate_coarse_candidates(reference_map, odom_pose, cfg):
        candidate_count += 1
        score = score_scan_against_map_pose(sampled_points, reference_map, candidate)
        item = (
            score,
            candidate.position.x,
            candidate.position.y,
            candidate.orientation.to_euler().z,
            candidate,
        )
        if len(candidate_heap) < cfg.max_candidates:
            heapq.heappush(candidate_heap, item)
            continue
        if item > candidate_heap[0]:
            heapq.heapreplace(candidate_heap, item)

    candidate_scores = [item[0] for item in candidate_heap]
    score_variance = float(np.var(candidate_scores)) if len(candidate_scores) > 1 else 0.0

    logger.info(
        "Global search coarse phase done",
        evaluated_candidates=candidate_count,
        top_candidates=len(candidate_heap),
        best_coarse_score=round(candidate_heap[-1][0] if candidate_heap else 0.0, 3),
        score_variance=round(score_variance, 4),
    )

    for _, _, _, _, candidate in sorted(candidate_heap, reverse=True):
        refined_pose, refined_score = scan_to_map_2d(
            sampled_points,
            reference_map,
            candidate,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        )
        if refined_score > best_score:
            best_pose = refined_pose
            best_score = refined_score

    logger.info(
        "Global search refinement done",
        final_score=round(best_score, 3),
        match_quality="low" if score_variance < 0.001 else "high",
    )
    return best_pose, best_score


def _sample_scan_points(scan_points: NDArray[np.float32], max_scan_points: int) -> NDArray[np.float32]:
    if scan_points.shape[0] <= max_scan_points:
        return scan_points
    step = max(1, scan_points.shape[0] // max_scan_points)
    return np.ascontiguousarray(scan_points[::step])


def _generate_coarse_candidates(
    reference_map: OccupancyGrid,
    odom_pose: PoseStamped,
    cfg: GlobalSearchConfig,
) -> Iterator[Pose]:
    step_cells = max(1, int(round(cfg.xy_step / reference_map.resolution)))

    if cfg.search_radius is not None:
        center_x = odom_pose.position.x
        center_y = odom_pose.position.y
        center_grid_x = int((center_x - reference_map.origin.position.x) / reference_map.resolution)
        center_grid_y = int((center_y - reference_map.origin.position.y) / reference_map.resolution)
        radius_cells = int(cfg.search_radius / reference_map.resolution)

        x_min = max(0, center_grid_x - radius_cells)
        x_max = min(reference_map.width, center_grid_x + radius_cells)
        y_min = max(0, center_grid_y - radius_cells)
        y_max = min(reference_map.height, center_grid_y + radius_cells)

        x_indices = np.arange(x_min, x_max, step_cells, dtype=np.int32)
        y_indices = np.arange(y_min, y_max, step_cells, dtype=np.int32)
    else:
        x_indices = np.arange(0, reference_map.width, step_cells, dtype=np.int32)
        y_indices = np.arange(0, reference_map.height, step_cells, dtype=np.int32)

    yaw_values = np.arange(-math.pi, math.pi, cfg.yaw_step, dtype=np.float32)

    for y_index in y_indices:
        for x_index in x_indices:
            world_x = reference_map.origin.position.x + (float(x_index) + 0.5) * reference_map.resolution
            world_y = reference_map.origin.position.y + (float(y_index) + 0.5) * reference_map.resolution
            for yaw in yaw_values:
                map_pose = as_pose_stamped(world_x, world_y, float(yaw), z=odom_pose.position.z, ts=odom_pose.ts)
                yield estimate_map_from_odom(map_pose, odom_pose)
