import time

import numpy as np

from dimos.msgs.geometry_msgs import Pose, PoseStamped, Quaternion, Vector3


def estimate_map_from_odom(map_pose: PoseStamped, odom_pose: PoseStamped) -> Pose:
    map_yaw = map_pose.orientation.to_euler().z
    odom_yaw = odom_pose.orientation.to_euler().z
    map_from_odom_yaw = map_yaw - odom_yaw

    cos_yaw = float(np.cos(map_from_odom_yaw))
    sin_yaw = float(np.sin(map_from_odom_yaw))

    tx = map_pose.position.x - (
        cos_yaw * odom_pose.position.x - sin_yaw * odom_pose.position.y
    )
    ty = map_pose.position.y - (
        sin_yaw * odom_pose.position.x + cos_yaw * odom_pose.position.y
    )

    return Pose(
        position=Vector3(tx, ty, 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, map_from_odom_yaw)),
    )


def apply_map_from_odom(
    map_from_odom: Pose,
    odom_pose: PoseStamped,
    *,
    ts: float | None = None,
) -> PoseStamped:
    map_from_odom_yaw = map_from_odom.orientation.to_euler().z
    odom_yaw = odom_pose.orientation.to_euler().z

    cos_yaw = float(np.cos(map_from_odom_yaw))
    sin_yaw = float(np.sin(map_from_odom_yaw))

    x = map_from_odom.position.x + (
        cos_yaw * odom_pose.position.x - sin_yaw * odom_pose.position.y
    )
    y = map_from_odom.position.y + (
        sin_yaw * odom_pose.position.x + cos_yaw * odom_pose.position.y
    )
    yaw = map_from_odom_yaw + odom_yaw

    return PoseStamped(
        position=Vector3(x, y, odom_pose.position.z),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
        frame_id="map",
        ts=odom_pose.ts if ts is None else ts,
    )


def pose_jump(current_pose: PoseStamped, proposed_pose: PoseStamped) -> tuple[float, float]:
    translation_jump = current_pose.position.distance(proposed_pose.position)
    current_yaw = current_pose.orientation.to_euler().z
    proposed_yaw = proposed_pose.orientation.to_euler().z
    yaw_jump = float(np.arctan2(np.sin(proposed_yaw - current_yaw), np.cos(proposed_yaw - current_yaw)))
    return translation_jump, abs(yaw_jump)


def as_pose_stamped(x: float, y: float, yaw: float, *, z: float = 0.0, ts: float | None = None) -> PoseStamped:
    return PoseStamped(
        position=Vector3(x, y, z),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
        frame_id="map",
        ts=time.time() if ts is None else ts,
    )
