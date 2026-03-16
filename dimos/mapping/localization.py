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

"""Continuous localization via scan matching against global map."""

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]
from reactivex import operators as ops

from dimos.core.core import rpc
from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs import (
    PoseStamped,
    Quaternion,
    Transform,
    Vector3,
)
from dimos.msgs.nav_msgs import OccupancyGrid
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.utils.logging_config import setup_logger
from dimos.utils.reactive import backpressure

logger = setup_logger()


@dataclass
class LocalizationConfig(ModuleConfig):
    alignment_frame: str = "map"
    update_rate: float = 2.0
    icp_max_correspondence_distance: float = 0.5
    icp_max_iterations: int = 30
    icp_relative_fitness_threshold: float = 1e-6
    icp_relative_rmse_threshold: float = 1e-6
    min_fitness_score: float = 0.3
    min_point_count: int = 100
    voxel_downsample_size: float = 0.1
    enable_continuous_localization: bool = True


class LocalizationModule(Module):
    default_config = LocalizationConfig
    config: LocalizationConfig
    
    lidar: In[PointCloud2]
    odom: In[PoseStamped]
    global_map: In[PointCloud2]
    global_costmap: In[OccupancyGrid]
    
    corrected_odom: Out[PoseStamped]
    map_to_odom: Out[Transform]
    
    _global_config: GlobalConfig
    _global_map_cloud: o3d.geometry.PointCloud | None = None
    _last_odom: PoseStamped | None = None
    _last_correction: Transform | None = None
    _last_update_time: float = 0.0
    _initial_pose_set: bool = False

    def __init__(
        self,
        cfg: GlobalConfig = global_config,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._global_config = cfg
        super().__init__(*args, **kwargs)

    @rpc
    def start(self) -> None:
        super().start()
        
        if not self.config.enable_continuous_localization:
            logger.info("Continuous localization disabled by config")
            return
        
        logger.info(
            f"LocalizationModule started: alignment_frame={self.config.alignment_frame}, "
            f"update_rate={self.config.update_rate}Hz"
        )
        
        self.global_map.subscribe(self._on_global_map)
        self.lidar.subscribe(self._on_lidar)
        self.odom.subscribe(self._on_odom)

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_global_map(self, pointcloud: PointCloud2) -> None:
        try:
            points = pointcloud.to_numpy()
            if len(points) < self.config.min_point_count:
                logger.warning(f"Global map too small: {len(points)} points")
                return
            
            self._global_map_cloud = o3d.geometry.PointCloud()
            self._global_map_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
            
            if self.config.voxel_downsample_size > 0:
                self._global_map_cloud = self._global_map_cloud.voxel_down_sample(
                    self.config.voxel_downsample_size
                )
            
            logger.info(
                f"Global map loaded: {len(self._global_map_cloud.points)} points "
                f"(downsampled from {len(points)})"
            )
        except Exception as e:
            logger.error(f"Failed to process global map: {e}")

    def _on_odom(self, odom: PoseStamped) -> None:
        self._last_odom = odom

    def _on_lidar(self, pointcloud: PointCloud2) -> None:
        if not self.config.enable_continuous_localization:
            return
        
        if self._global_map_cloud is None:
            return
        
        if self._last_odom is None:
            return
        
        current_time = time.time()
        if current_time - self._last_update_time < 1.0 / self.config.update_rate:
            return
        
        self._last_update_time = current_time
        
        try:
            correction = self._compute_localization(pointcloud, self._last_odom)
            if correction is not None:
                self._last_correction = correction
                self.map_to_odom.publish(correction)
                
                corrected_pose = self._apply_correction(self._last_odom, correction)
                self.corrected_odom.publish(corrected_pose)
        except Exception as e:
            logger.error(f"Localization failed: {e}")

    def _compute_localization(
        self,
        lidar_scan: PointCloud2,
        current_odom: PoseStamped,
    ) -> Transform | None:
        points = lidar_scan.to_numpy()
        if len(points) < self.config.min_point_count:
            return None
        
        source_cloud = o3d.geometry.PointCloud()
        source_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
        
        if self.config.voxel_downsample_size > 0:
            source_cloud = source_cloud.voxel_down_sample(
                self.config.voxel_downsample_size
            )
        
        if len(source_cloud.points) < self.config.min_point_count:
            return None
        
        initial_guess = self._pose_to_matrix(current_odom)
        
        if self._last_correction is not None and self._initial_pose_set:
            correction_matrix = self._transform_to_matrix(self._last_correction)
            initial_guess = correction_matrix @ initial_guess
        
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=self.config.icp_max_iterations,
            relative_fitness=self.config.icp_relative_fitness_threshold,
            relative_rmse=self.config.icp_relative_rmse_threshold,
        )
        
        result = o3d.pipelines.registration.registration_icp(
            source_cloud,
            self._global_map_cloud,
            self.config.icp_max_correspondence_distance,
            initial_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria,
        )
        
        if result.fitness < self.config.min_fitness_score:
            logger.warning(
                f"ICP fitness too low: {result.fitness:.3f} < {self.config.min_fitness_score}"
            )
            return None
        
        logger.debug(
            f"ICP converged: fitness={result.fitness:.3f}, "
            f"rmse={result.inlier_rmse:.4f}, "
            f"correspondences={len(result.correspondence_set)}"
        )
        
        self._initial_pose_set = True
        
        map_to_base = result.transformation
        odom_to_base = self._pose_to_matrix(current_odom)
        map_to_odom = map_to_base @ np.linalg.inv(odom_to_base)
        
        return self._matrix_to_transform(map_to_odom)

    def _pose_to_matrix(self, pose: PoseStamped) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        
        q = pose.orientation
        qw, qx, qy, qz = q.w, q.x, q.y, q.z
        
        T[0, 0] = 1 - 2*(qy**2 + qz**2)
        T[0, 1] = 2*(qx*qy - qz*qw)
        T[0, 2] = 2*(qx*qz + qy*qw)
        
        T[1, 0] = 2*(qx*qy + qz*qw)
        T[1, 1] = 1 - 2*(qx**2 + qz**2)
        T[1, 2] = 2*(qy*qz - qx*qw)
        
        T[2, 0] = 2*(qx*qz - qy*qw)
        T[2, 1] = 2*(qy*qz + qx*qw)
        T[2, 2] = 1 - 2*(qx**2 + qy**2)
        
        return T

    def _transform_to_matrix(self, transform: Transform) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ]
        
        q = transform.rotation
        qw, qx, qy, qz = q.w, q.x, q.y, q.z
        
        T[0, 0] = 1 - 2*(qy**2 + qz**2)
        T[0, 1] = 2*(qx*qy - qz*qw)
        T[0, 2] = 2*(qx*qz + qy*qw)
        
        T[1, 0] = 2*(qx*qy + qz*qw)
        T[1, 1] = 1 - 2*(qx**2 + qz**2)
        T[1, 2] = 2*(qy*qz - qx*qw)
        
        T[2, 0] = 2*(qx*qz - qy*qw)
        T[2, 1] = 2*(qy*qz + qx*qw)
        T[2, 2] = 1 - 2*(qx**2 + qy**2)
        
        return T

    def _matrix_to_transform(self, matrix: np.ndarray) -> Transform:
        translation = Vector3(
            float(matrix[0, 3]),
            float(matrix[1, 3]),
            float(matrix[2, 3]),
        )
        
        R = matrix[:3, :3]
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        rotation = Quaternion(float(qx), float(qy), float(qz), float(qw))
        
        return Transform(
            translation=translation,
            rotation=rotation,
            frame_id=self.config.alignment_frame,
            child_frame_id="odom",
        )

    def _apply_correction(
        self,
        odom: PoseStamped,
        correction: Transform,
    ) -> PoseStamped:
        odom_matrix = self._pose_to_matrix(odom)
        correction_matrix = self._transform_to_matrix(correction)
        corrected_matrix = correction_matrix @ odom_matrix
        
        corrected = PoseStamped(
            frame_id=self.config.alignment_frame,
            position=Vector3(
                float(corrected_matrix[0, 3]),
                float(corrected_matrix[1, 3]),
                float(corrected_matrix[2, 3]),
            ),
            orientation=self._matrix_to_quaternion(corrected_matrix[:3, :3]),
            ts=odom.ts,
        )
        
        return corrected

    def _matrix_to_quaternion(self, R: np.ndarray) -> Quaternion:
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        return Quaternion(float(qx), float(qy), float(qz), float(qw))

    @classmethod
    def blueprint(
        cls,
        alignment_frame: str = "map",
        update_rate: float = 2.0,
        enable: bool = True,
    ) -> ModuleConfig:
        return cls.default_config(
            alignment_frame=alignment_frame,
            update_rate=update_rate,
            enable_continuous_localization=enable,
        )


localization_module = LocalizationModule.blueprint

__all__ = ["LocalizationModule", "LocalizationConfig", "localization_module"]
