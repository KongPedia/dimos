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

from dataclasses import asdict, dataclass, field
from pathlib import Path
import time

from reactivex import operators as ops
import numpy as np

from dimos.core.core import rpc
from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.mapping.pointclouds.occupancy import (
    OCCUPANCY_ALGOS,
    GeneralOccupancyConfig,
    OccupancyConfig,
)
from dimos.msgs.nav_msgs import OccupancyGrid
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass
class Config(ModuleConfig):
    algo: str = "general"
    config: OccupancyConfig = field(default_factory=GeneralOccupancyConfig)
    update_interval: float = 0.2


class CostMapper(Module):
    default_config = Config
    config: Config

    global_map: In[PointCloud2]
    global_costmap: Out[OccupancyGrid]

    def __init__(self, cfg: GlobalConfig = global_config, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._global_config = cfg
        self._preloaded_costmap: OccupancyGrid | None = None
        self._reference_costmap: OccupancyGrid | None = None
        self._latest_costmap: OccupancyGrid | None = None

    @rpc
    def start(self) -> None:
        super().start()

        def _publish_costmap(grid: OccupancyGrid) -> None:
            self._latest_costmap = grid
            self.global_costmap.publish(grid)

        initial_costmap = self._get_initial_costmap()
        if initial_costmap is not None:
            _publish_costmap(initial_costmap)

        def _calculate_and_time(msg: PointCloud2) -> OccupancyGrid:
            _ = time.monotonic()
            start = time.perf_counter()
            grid = self._calculate_costmap(msg)
            _ = (time.perf_counter() - start) * 1000
            return grid

        stream = self.global_map.observable()  # type: ignore[no-untyped-call]
        if self.config.update_interval > 0:
            stream = stream.pipe(ops.sample(self.config.update_interval))

        self._disposables.add(
            stream.pipe(ops.map(_calculate_and_time)).subscribe(_publish_costmap)
        )

    @rpc
    def stop(self) -> None:
        super().stop()

    @rpc
    def has_latest_costmap(self) -> bool:
        return self._latest_costmap is not None

    @rpc
    def save_latest_costmap(self, path: str) -> bool:
        if self._latest_costmap is None:
            logger.warning("No latest costmap available to save")
            return False

        self._latest_costmap.to_path(Path(path))
        logger.info("Saved latest costmap", path=path)
        return True

    # @timed()  # TODO: fix thread leak in timed decorator
    def _calculate_costmap(self, msg: PointCloud2) -> OccupancyGrid:
        if self._global_config.mujoco_global_costmap_from_occupancy:
            if self._preloaded_costmap is None:
                path = Path(self._global_config.mujoco_global_costmap_from_occupancy)
                self._preloaded_costmap = OccupancyGrid.from_path(path)
            return self._preloaded_costmap

        fn = OCCUPANCY_ALGOS[self.config.algo]
        live_costmap = fn(msg, **asdict(self.config.config))
        reference_costmap = self._get_reference_costmap()
        if reference_costmap is None:
            return live_costmap
        return self._merge_costmaps(reference_costmap, live_costmap)

    def _get_initial_costmap(self) -> OccupancyGrid | None:
        if self._global_config.mujoco_global_costmap_from_occupancy:
            if self._preloaded_costmap is None:
                path = Path(self._global_config.mujoco_global_costmap_from_occupancy)
                self._preloaded_costmap = OccupancyGrid.from_path(path)
            return self._preloaded_costmap
        return self._get_reference_costmap()

    def _get_reference_costmap(self) -> OccupancyGrid | None:
        if not self._global_config.map_path:
            return None
        if self._reference_costmap is None:
            self._reference_costmap = OccupancyGrid.from_path(Path(self._global_config.map_path))
        return self._reference_costmap

    def _merge_costmaps(
        self, reference_costmap: OccupancyGrid, live_costmap: OccupancyGrid
    ) -> OccupancyGrid:
        merged_grid = np.array(reference_costmap.grid, copy=True)

        for live_y, live_x in np.argwhere(live_costmap.grid != -1):
            live_value = int(live_costmap.grid[live_y, live_x])
            world_point = live_costmap.grid_to_world((live_x + 0.5, live_y + 0.5, 0.0))
            ref_point = reference_costmap.world_to_grid(world_point)
            ref_x = int(round(ref_point.x - 0.5))
            ref_y = int(round(ref_point.y - 0.5))

            if ref_x < 0 or ref_x >= reference_costmap.width:
                continue
            if ref_y < 0 or ref_y >= reference_costmap.height:
                continue

            ref_value = int(merged_grid[ref_y, ref_x])
            if live_value > 0:
                # Remap live occupied cells to 150-250 range for visual distinction
                remapped_live = 150 + min(100, live_value)
                merged_grid[ref_y, ref_x] = remapped_live
            elif ref_value == -1:
                # Mark unknown areas in reference map as free when scanned by live LiDAR
                merged_grid[ref_y, ref_x] = 130

        return OccupancyGrid(
            grid=merged_grid,
            resolution=reference_costmap.resolution,
            origin=reference_costmap.origin,
            frame_id=reference_costmap.frame_id,
            ts=live_costmap.ts,
        )


cost_mapper = CostMapper.blueprint
