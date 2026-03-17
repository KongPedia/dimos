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

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time

import numpy as np
from reactivex import interval
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.localization.global_search import GlobalSearchConfig, search_global_map_from_odom
from dimos.localization.scan_matching import scan_to_map_2d, score_scan_against_map_pose
from dimos.localization.state import LocalizationState
from dimos.localization.transform_utils import (
    apply_map_from_odom,
    as_pose_stamped,
    estimate_map_from_odom,
    pose_jump,
)
from dimos.msgs.geometry_msgs import Pose, PoseStamped, Quaternion, Vector3
from dimos.msgs.nav_msgs import OccupancyGrid
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass
class Config(ModuleConfig):
    """Localization module configuration."""

    relocalization_interval: float = 2.0
    tracking_skip_relocalization: bool = True
    match_score_threshold: float = 0.3
    recovery_match_score_threshold: float = 0.2
    min_feature_hits: int = 15
    max_iterations: int = 50
    convergence_threshold: float = 0.01
    max_translation_jump: float = 1.5
    max_yaw_jump: float = 0.7
    initial_max_translation_jump: float = 4.0
    initial_local_xy_search_radius: float = 0.75
    initial_local_xy_search_step: float = 0.25
    initial_local_yaw_search_step: float = math.radians(10.0)
    lost_after_consecutive_misses: int = 3
    recovery_confirmations: int = 2
    recovery_candidate_translation_tolerance: float = 2.0
    recovery_candidate_yaw_tolerance: float = 0.8
    global_search_xy_step: float = 2.0
    global_search_yaw_step: float = math.radians(45.0)
    global_search_max_candidates: int = 6
    global_search_max_scan_points: int = 800
    global_search_radius: float | None = None


class LocalizationModule(Module):
    """Localization module for pose correction against a saved map.

    Provides initial pose setting and periodic relocalization to estimate
    the robot's pose within a pre-built map.
    """

    default_config = Config
    config: Config

    # Inputs
    lidar: In[PointCloud2]
    odom: In[PoseStamped]

    # Outputs
    corrected_pose: Out[PoseStamped]

    def __init__(self, cfg: GlobalConfig = global_config, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._global_config = cfg

        self._state_lock = threading.Lock()
        self._state: LocalizationState = LocalizationState.UNINITIALIZED
        self._reference_map: OccupancyGrid | None = None
        self._map_from_odom: Pose | None = None
        self._current_pose: PoseStamped | None = None
        self._last_odom: PoseStamped | None = None
        self._last_scan: PointCloud2 | None = None
        self._match_score: float = 0.0
        self._consecutive_misses: int = 0
        self._recovery_candidate: PoseStamped | None = None
        self._recovery_candidate_confirmations: int = 0
        self._recovery_candidate_score: float = 0.0
        self._initial_localization_pending: bool = True

    @rpc
    def start(self) -> None:
        logger.info("LocalizationModule start() called")
        super().start()

        logger.info(
            f"Starting LocalizationModule: enable_localization={self._global_config.enable_localization}, "
            f"map_path={self._global_config.map_path}"
        )

        # Load reference map if path provided
        if self._global_config.map_path:
            logger.info(f"Loading reference map from: {self._global_config.map_path}")
            self._load_reference_map()
            if self._reference_map:
                logger.info(
                    f"Reference map loaded: {self._reference_map.width}x{self._reference_map.height}, "
                    f"resolution={self._reference_map.resolution}m, "
                    f"origin=({self._reference_map.origin.position.x:.2f}, {self._reference_map.origin.position.y:.2f})"
                )
            else:
                logger.error("Reference map failed to load")

        # Subscribe to inputs
        self._disposables.add(Disposable(self.lidar.subscribe(self._on_lidar)))
        self._disposables.add(Disposable(self.odom.subscribe(self._on_odom)))
        logger.info("Subscribed to lidar and odom streams")

        # Set initial pose if configured
        if self._global_config.enable_localization:
            logger.info("Setting initial pose")
            self._set_initial_pose()
        else:
            logger.info("Localization disabled, skipping initial pose")

        # Start periodic relocalization
        if self._global_config.enable_localization and self._reference_map:
            relocalization_interval = (
                self.config.relocalization_interval
                if self.config.relocalization_interval > 0
                else self._global_config.relocalization_interval
            )
            logger.info(f"Starting periodic relocalization every {relocalization_interval}s")
            self._disposables.add(
                interval(relocalization_interval).subscribe(lambda _: self._relocalize())
            )
        else:
            logger.warning(
                f"Periodic relocalization NOT started: enable={self._global_config.enable_localization}, "
                f"has_map={self._reference_map is not None}"
            )

        logger.info(
            f"LocalizationModule started: map={self._global_config.map_path}, "
            f"state={self._state.value}, enable={self._global_config.enable_localization}"
        )

    @rpc
    def stop(self) -> None:
        super().stop()

    def _load_reference_map(self) -> None:
        """Load reference map from configured path."""
        try:
            map_path = Path(self._global_config.map_path)  # type: ignore[arg-type]
            if not map_path.exists():
                logger.error(f"Map path does not exist: {map_path}")
                return

            self._reference_map = OccupancyGrid.from_path(map_path)
            logger.info(
                f"Loaded reference map: {self._reference_map.width}x{self._reference_map.height}, "
                f"resolution={self._reference_map.resolution}m"
            )
        except Exception as e:
            logger.error(f"Failed to load reference map: {e}")

    def _set_initial_pose(self) -> None:
        """Set initial pose from global config."""
        with self._state_lock:
            initial_pose = as_pose_stamped(
                self._global_config.initial_pose_x,
                self._global_config.initial_pose_y,
                self._global_config.initial_pose_yaw,
                ts=time.time(),
            )
            self._map_from_odom = (
                estimate_map_from_odom(initial_pose, self._last_odom)
                if self._last_odom is not None
                else None
            )
            self._current_pose = initial_pose
            self._state = LocalizationState.INITIALIZED
            self._match_score = 0.0
            self._consecutive_misses = 0
            self._clear_recovery_candidate()
            self._initial_localization_pending = True

            logger.info(
                f"Initial pose set: x={self._global_config.initial_pose_x:.2f}, "
                f"y={self._global_config.initial_pose_y:.2f}, "
                f"yaw={self._global_config.initial_pose_yaw:.2f}, "
                f"orientation={self._format_quaternion(initial_pose.orientation)}"
            )

            # Publish initial corrected pose
            self.corrected_pose.publish(initial_pose)

    def _on_lidar(self, msg: PointCloud2) -> None:
        """Cache latest LiDAR scan for relocalization."""
        with self._state_lock:
            self._last_scan = msg

    def _on_odom(self, msg: PoseStamped) -> None:
        """Update current pose estimate from odometry."""
        with self._state_lock:
            if not self._global_config.enable_localization:
                self._last_odom = msg
                self.corrected_pose.publish(msg)
                return

            if not self._current_pose:
                self._last_odom = msg
                self.corrected_pose.publish(msg)
                return

            if self._map_from_odom is None:
                self._map_from_odom = estimate_map_from_odom(self._current_pose, msg)

            if self._last_odom is None:
                self._last_odom = msg
                corrected = apply_map_from_odom(self._map_from_odom, msg, ts=msg.ts)
                self._current_pose = corrected
                self.corrected_pose.publish(corrected)
                return

            corrected = apply_map_from_odom(self._map_from_odom, msg, ts=msg.ts)
            self._current_pose = corrected
            self._last_odom = msg
            self.corrected_pose.publish(corrected)

    def _relocalize(self) -> None:
        """Perform relocalization against reference map."""
        if not self._reference_map:
            return

        with self._state_lock:
            if self._state == LocalizationState.UNINITIALIZED:
                return

            if self._state == LocalizationState.TRACKING and self.config.tracking_skip_relocalization:
                return

            # Use current pose as initial guess
            if not self._current_pose:
                logger.warning("No current pose for relocalization")
                return

            if not self._last_odom:
                logger.debug("No odometry for relocalization")
                return

            # Need scan data
            if not self._last_scan:
                logger.debug("No scan data for relocalization")
                return

            # Extract point cloud as numpy array
            try:
                points, _ = self._last_scan.as_numpy()
                if points.shape[0] == 0:
                    logger.debug("Empty scan, skipping relocalization")
                    return

                map_from_odom = self._map_from_odom or estimate_map_from_odom(
                    self._current_pose,
                    self._last_odom,
                )
                logger.info(
                    "Relocalization attempt",
                    state=self._state.value,
                    scan_points=int(points.shape[0]),
                    odom_yaw=round(self._last_odom.orientation.to_euler().z, 3),
                    odom_orientation=self._format_quaternion(self._last_odom.orientation),
                    current_yaw=round(self._current_pose.orientation.to_euler().z, 3),
                    current_orientation=self._format_quaternion(self._current_pose.orientation),
                    map_from_odom_yaw=round(map_from_odom.orientation.to_euler().z, 3),
                    map_from_odom_orientation=self._format_quaternion(map_from_odom.orientation),
                )
                if self._initial_localization_pending:
                    corrected_map_from_odom, match_score = self._search_near_initial_pose(
                        points,
                        map_from_odom,
                    )
                elif self._state in {
                    LocalizationState.INITIALIZED,
                    LocalizationState.LOST,
                    LocalizationState.RECOVERING,
                }:
                    corrected_map_from_odom, match_score = search_global_map_from_odom(
                        points,
                        self._reference_map,
                        self._last_odom,
                        map_from_odom,
                        max_iterations=self.config.max_iterations,
                        convergence_threshold=self.config.convergence_threshold,
                        search_config=GlobalSearchConfig(
                            xy_step=self.config.global_search_xy_step,
                            yaw_step=self.config.global_search_yaw_step,
                            max_candidates=self.config.global_search_max_candidates,
                            max_scan_points=self.config.global_search_max_scan_points,
                            search_radius=self.config.global_search_radius,
                        ),
                    )
                else:
                    corrected_map_from_odom, match_score = scan_to_map_2d(
                        points,
                        self._reference_map,
                        map_from_odom,
                        max_iterations=self.config.max_iterations,
                        convergence_threshold=self.config.convergence_threshold,
                    )

                corrected = apply_map_from_odom(
                    corrected_map_from_odom,
                    self._last_odom,
                    ts=time.time(),
                )
                translation_jump, yaw_jump = pose_jump(self._current_pose, corrected)
                accepted = self._handle_relocalization_result(
                    corrected,
                    corrected_map_from_odom,
                    match_score,
                    translation_jump,
                    yaw_jump,
                )
                if not accepted:
                    return

                # Publish corrected pose
                self.corrected_pose.publish(corrected)

                logger.info(
                    "Relocalization result",
                    state=self._state.value,
                    match_score=round(match_score, 3),
                    corrected_x=round(corrected.position.x, 3),
                    corrected_y=round(corrected.position.y, 3),
                    corrected_yaw=round(corrected.orientation.to_euler().z, 3),
                    corrected_orientation=self._format_quaternion(corrected.orientation),
                    odom_yaw=round(self._last_odom.orientation.to_euler().z, 3),
                    odom_orientation=self._format_quaternion(self._last_odom.orientation),
                    map_from_odom_yaw=round(corrected_map_from_odom.orientation.to_euler().z, 3),
                    map_from_odom_orientation=self._format_quaternion(
                        corrected_map_from_odom.orientation
                    ),
                )

            except Exception as e:
                logger.error(f"Relocalization failed: {e}")

    def _format_quaternion(self, orientation: Quaternion) -> str:
        return (
            f"({orientation.x:.3f}, {orientation.y:.3f}, "
            f"{orientation.z:.3f}, {orientation.w:.3f})"
        )

    def _handle_relocalization_result(
        self,
        corrected: PoseStamped,
        corrected_map_from_odom: Pose,
        match_score: float,
        translation_jump: float,
        yaw_jump: float,
    ) -> bool:
        self._match_score = match_score

        if match_score < self.config.match_score_threshold:
            self._handle_relocalization_miss(match_score)
            logger.warning(
                "Relocalization rejected due to low score",
                state=self._state.value,
                match_score=round(match_score, 3),
                translation_jump=round(translation_jump, 3),
                yaw_jump=round(yaw_jump, 3),
            )
            return False

        if self._is_initial_jump_too_large(translation_jump, yaw_jump):
            self._handle_relocalization_miss(match_score)
            logger.warning(
                "Initial relocalization rejected due to excessive jump",
                state=self._state.value,
                match_score=round(match_score, 3),
                translation_jump=round(translation_jump, 3),
                yaw_jump=round(yaw_jump, 3),
                initial_max_translation_jump=round(self.config.initial_max_translation_jump, 3),
            )
            return False

        if self._initial_localization_pending:
            logger.info(
                "Initial relocalization accepted",
                state=self._state.value,
                match_score=round(match_score, 3),
                translation_jump=round(translation_jump, 3),
                yaw_jump=round(yaw_jump, 3),
            )
            self._accept_relocalization(
                corrected,
                corrected_map_from_odom,
                match_score,
                f"initial score={match_score:.3f}",
            )
            return True

        if not self._is_large_jump(translation_jump, yaw_jump):
            self._accept_relocalization(
                corrected,
                corrected_map_from_odom,
                match_score,
                f"score={match_score:.3f}",
            )
            return True

        if match_score < self.config.recovery_match_score_threshold:
            self._handle_relocalization_miss(match_score)
            logger.warning(
                "Relocalization rejected due to large jump below recovery threshold",
                state=self._state.value,
                match_score=round(match_score, 3),
                translation_jump=round(translation_jump, 3),
                yaw_jump=round(yaw_jump, 3),
            )
            return False

        self._transition_state(LocalizationState.RECOVERING, f"score={match_score:.3f}")

        if (
            self._recovery_candidate is None
            or not self._is_recovery_candidate_consistent(self._recovery_candidate, corrected)
        ):
            self._recovery_candidate = corrected
            self._recovery_candidate_confirmations = 1
            self._recovery_candidate_score = match_score
            logger.warning(
                "Recovery candidate registered",
                state=self._state.value,
                match_score=round(match_score, 3),
                translation_jump=round(translation_jump, 3),
                yaw_jump=round(yaw_jump, 3),
                candidate_x=round(corrected.position.x, 3),
                candidate_y=round(corrected.position.y, 3),
                candidate_yaw=round(corrected.orientation.to_euler().z, 3),
                confirmations=self._recovery_candidate_confirmations,
            )
            return False

        self._recovery_candidate = corrected
        self._recovery_candidate_confirmations += 1
        self._recovery_candidate_score = max(self._recovery_candidate_score, match_score)

        logger.warning(
            "Recovery candidate confirmed",
            state=self._state.value,
            match_score=round(match_score, 3),
            translation_jump=round(translation_jump, 3),
            yaw_jump=round(yaw_jump, 3),
            confirmations=self._recovery_candidate_confirmations,
            required_confirmations=self.config.recovery_confirmations,
        )

        if self._recovery_candidate_confirmations < self.config.recovery_confirmations:
            return False

        self._accept_relocalization(
            corrected,
            corrected_map_from_odom,
            match_score,
            f"recovered score={match_score:.3f}",
        )
        logger.warning(
            "Recovery relocalization accepted",
            state=self._state.value,
            match_score=round(match_score, 3),
            corrected_x=round(corrected.position.x, 3),
            corrected_y=round(corrected.position.y, 3),
            corrected_yaw=round(corrected.orientation.to_euler().z, 3),
        )
        return True

    def _handle_relocalization_miss(self, match_score: float) -> None:
        self._consecutive_misses += 1
        self._clear_recovery_candidate()
        if self._consecutive_misses >= self.config.lost_after_consecutive_misses:
            self._transition_state(LocalizationState.LOST, f"score={match_score:.3f}")
            return
        if self._state == LocalizationState.TRACKING:
            self._transition_state(LocalizationState.DEGRADED, f"score={match_score:.3f}")

    def _transition_state(self, new_state: LocalizationState, reason: str) -> None:
        if self._state == new_state:
            return
        old_state = self._state
        self._state = new_state
        logger.info(
            "Localization state changed",
            from_state=old_state.value,
            to_state=new_state.value,
            reason=reason,
        )

    def _clear_recovery_candidate(self) -> None:
        self._recovery_candidate = None
        self._recovery_candidate_confirmations = 0
        self._recovery_candidate_score = 0.0

    def _accept_relocalization(
        self,
        corrected: PoseStamped,
        corrected_map_from_odom: Pose,
        match_score: float,
        reason: str,
    ) -> None:
        self._match_score = match_score
        self._consecutive_misses = 0
        self._clear_recovery_candidate()
        self._map_from_odom = corrected_map_from_odom
        self._current_pose = corrected
        self._initial_localization_pending = False
        self._transition_state(LocalizationState.TRACKING, reason)

    def _is_large_jump(self, translation_jump: float, yaw_jump: float) -> bool:
        return (
            translation_jump > self.config.max_translation_jump
            or yaw_jump > self.config.max_yaw_jump
        )

    def _is_initial_jump_too_large(self, translation_jump: float, yaw_jump: float) -> bool:
        return (
            self._initial_localization_pending
            and translation_jump > self.config.initial_max_translation_jump
        )

    def _search_near_initial_pose(
        self,
        scan_points: np.ndarray,
        initial_map_from_odom: Pose,
    ) -> tuple[Pose, float]:
        if self._reference_map is None:
            return initial_map_from_odom, 0.0

        xy_radius = max(0.0, self.config.initial_local_xy_search_radius)
        xy_step = max(0.05, self.config.initial_local_xy_search_step)
        yaw_step = max(math.radians(2.0), self.config.initial_local_yaw_search_step)

        if xy_radius < xy_step:
            offsets = np.array([0.0], dtype=np.float32)
        else:
            offsets = np.arange(-xy_radius, xy_radius + xy_step * 0.5, xy_step, dtype=np.float32)

        yaw_offsets = np.arange(-math.pi, math.pi, yaw_step, dtype=np.float32)

        best_pose = initial_map_from_odom
        best_score = score_scan_against_map_pose(scan_points, self._reference_map, initial_map_from_odom)
        base_x = initial_map_from_odom.position.x
        base_y = initial_map_from_odom.position.y
        base_yaw = initial_map_from_odom.orientation.to_euler().z

        for dx in offsets:
            for dy in offsets:
                for yaw_offset in yaw_offsets:
                    candidate = Pose(
                        position=Vector3(base_x + float(dx), base_y + float(dy), initial_map_from_odom.position.z),
                        orientation=Quaternion.from_euler(
                            Vector3(0.0, 0.0, base_yaw + float(yaw_offset))
                        ),
                    )
                    score = score_scan_against_map_pose(scan_points, self._reference_map, candidate)
                    if score > best_score:
                        best_pose = candidate
                        best_score = score

        refined_pose, refined_score = scan_to_map_2d(
            scan_points,
            self._reference_map,
            best_pose,
            max_iterations=self.config.max_iterations,
            convergence_threshold=self.config.convergence_threshold,
        )
        logger.info(
            "Initial local relocalization done",
            best_score=round(refined_score, 3),
            xy_radius=round(xy_radius, 3),
            xy_step=round(xy_step, 3),
            yaw_step=round(yaw_step, 3),
        )
        return refined_pose, refined_score

    def _is_recovery_candidate_consistent(
        self, current_candidate: PoseStamped, proposed_pose: PoseStamped
    ) -> bool:
        translation_delta, yaw_delta = pose_jump(current_candidate, proposed_pose)
        return (
            translation_delta <= self.config.recovery_candidate_translation_tolerance
            and yaw_delta <= self.config.recovery_candidate_yaw_tolerance
        )

    @rpc
    def get_state(self) -> str:
        """Get current localization state."""
        with self._state_lock:
            return self._state.value

    @rpc
    def get_match_score(self) -> float:
        """Get last match score."""
        with self._state_lock:
            return self._match_score

    @rpc
    def reset(self) -> bool:
        """Reset localization to initial pose."""
        self._set_initial_pose()
        return True


localization_module = LocalizationModule.blueprint
