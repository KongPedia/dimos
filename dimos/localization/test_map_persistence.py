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

from pathlib import Path
import tempfile

import numpy as np
import pytest

from dimos.msgs.geometry_msgs import Pose, Quaternion, Vector3
from dimos.msgs.nav_msgs import OccupancyGrid


def test_save_and_load_yaml() -> None:
    """Test saving and loading occupancy grid as yaml+png."""
    # Create test grid
    grid_data = np.array([[0, 50, 100], [-1, 0, 50]], dtype=np.int8)
    origin = Pose(
        position=Vector3(-1.0, -2.0, 0.0),
        orientation=Quaternion.from_euler(Vector3(0.0, 0.0, 0.1)),
    )

    original = OccupancyGrid(
        grid=grid_data,
        resolution=0.05,
        origin=origin,
        frame_id="map",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "test_map.yaml"

        # Save
        original.to_path(yaml_path)

        # Verify files exist
        assert yaml_path.exists()
        assert (Path(tmpdir) / "test_map.png").exists()

        # Load
        loaded = OccupancyGrid.from_path(yaml_path)

        # Verify metadata
        assert loaded.width == original.width
        assert loaded.height == original.height
        assert loaded.resolution == pytest.approx(original.resolution)
        assert loaded.origin.position.x == pytest.approx(original.origin.position.x)
        assert loaded.origin.position.y == pytest.approx(original.origin.position.y)


def test_save_and_load_npy() -> None:
    """Test saving and loading occupancy grid as npy."""
    grid_data = np.random.randint(-1, 101, size=(10, 10), dtype=np.int8)
    original = OccupancyGrid(grid=grid_data, resolution=0.1)

    with tempfile.TemporaryDirectory() as tmpdir:
        npy_path = Path(tmpdir) / "test_map.npy"

        # Save
        original.to_path(npy_path)
        assert npy_path.exists()

        # Load
        loaded = OccupancyGrid.from_path(npy_path)

        # Verify grid data
        np.testing.assert_array_equal(loaded.grid, original.grid)


def test_save_png() -> None:
    """Test saving occupancy grid as standalone png."""
    grid_data = np.array([[0, 100], [50, -1]], dtype=np.int8)
    original = OccupancyGrid(grid=grid_data, resolution=0.05)

    with tempfile.TemporaryDirectory() as tmpdir:
        png_path = Path(tmpdir) / "test_map.png"

        # Save
        original.to_path(png_path)
        assert png_path.exists()

        # Load and verify it's an image
        from PIL import Image

        img = Image.open(png_path)
        assert img.size == (2, 2)
