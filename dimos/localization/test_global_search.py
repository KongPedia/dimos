import math

import numpy as np

from dimos.localization.global_search import GlobalSearchConfig, search_global_map_from_odom
from dimos.localization.transform_utils import apply_map_from_odom, as_pose_stamped, estimate_map_from_odom
from dimos.msgs.geometry_msgs import Pose, Vector3
from dimos.msgs.nav_msgs import OccupancyGrid


def test_transform_roundtrip() -> None:
    odom_pose = as_pose_stamped(1.0, -0.5, 0.3, ts=123.0)
    map_pose = as_pose_stamped(4.0, 2.0, -0.2, ts=123.0)

    map_from_odom = estimate_map_from_odom(map_pose, odom_pose)
    reconstructed_pose = apply_map_from_odom(map_from_odom, odom_pose)

    assert reconstructed_pose.position.distance(map_pose.position) < 1e-6
    assert abs(reconstructed_pose.orientation.to_euler().z - map_pose.orientation.to_euler().z) < 1e-6


def test_global_search_recovers_pose() -> None:
    grid = np.zeros((20, 20), dtype=np.int8)
    occupied_world_points = np.array(
        [
            [3.0, 3.0, 0.0],
            [3.5, 3.0, 0.0],
            [4.0, 3.0, 0.0],
            [4.5, 3.5, 0.0],
            [4.0, 4.0, 0.0],
            [3.5, 4.5, 0.0],
            [3.0, 4.0, 0.0],
        ],
        dtype=np.float32,
    )

    for point in occupied_world_points:
        x = int(round(point[0]))
        y = int(round(point[1]))
        grid[y, x] = 100

    reference_map = OccupancyGrid(
        grid=grid,
        resolution=1.0,
        origin=Pose(position=Vector3(0.0, 0.0, 0.0)),
        frame_id="map",
    )
    odom_pose = as_pose_stamped(0.0, 0.0, 0.0, ts=123.0)
    true_map_pose = as_pose_stamped(2.0, 2.0, math.radians(30.0), ts=123.0)
    true_map_from_odom = estimate_map_from_odom(true_map_pose, odom_pose)

    cos_yaw = math.cos(-true_map_from_odom.orientation.to_euler().z)
    sin_yaw = math.sin(-true_map_from_odom.orientation.to_euler().z)
    scan_points = []
    for point in occupied_world_points:
        dx = point[0] - true_map_from_odom.position.x
        dy = point[1] - true_map_from_odom.position.y
        scan_points.append(
            [
                cos_yaw * dx - sin_yaw * dy,
                sin_yaw * dx + cos_yaw * dy,
                0.0,
            ]
        )
    scan_points_np = np.asarray(scan_points, dtype=np.float32)

    initial_map_from_odom = estimate_map_from_odom(as_pose_stamped(8.0, 8.0, 0.0, ts=123.0), odom_pose)
    recovered_map_from_odom, score = search_global_map_from_odom(
        scan_points_np,
        reference_map,
        odom_pose,
        initial_map_from_odom,
        max_iterations=20,
        convergence_threshold=0.001,
        search_config=GlobalSearchConfig(xy_step=1.0, yaw_step=math.radians(30.0), max_candidates=4, max_scan_points=256),
    )
    recovered_pose = apply_map_from_odom(recovered_map_from_odom, odom_pose)

    assert score > 0.9
    assert recovered_pose.position.distance(true_map_pose.position) < 0.6
    assert abs(recovered_pose.orientation.to_euler().z - true_map_pose.orientation.to_euler().z) < 0.3
