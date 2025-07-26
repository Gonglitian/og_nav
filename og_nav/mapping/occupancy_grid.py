"""Occupancy grid map generation utilities."""

from typing import Tuple, Union, Optional

import cv2
import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.robots.tiago import Tiago


class OGMGenerator:
    """Occupancy Grid Map Generator for OmniGibson environments.

    This class provides functionality to generate occupancy grid maps
    from 3D environments and update traversability maps.
    """
    
    @staticmethod
    def get_default_cfg() -> dict:
        """Get default configuration for OGMGenerator."""
        return {
            'resolution': 0.1
        }

    def __init__(self, resolution: Optional[float] = None, config: Optional[dict] = None) -> None:
        """Initialize OGM generator.

        Priority order for parameters:
        1. Constructor arguments (highest priority)
        2. config dict values
        3. Default values (lowest priority)

        Args:
            resolution: Map resolution in meters per pixel
            config: OGM configuration dict
        """
        # Merge config with defaults following priority order
        default_config = self.get_default_cfg()
        merged_config = default_config.copy()
        if config is not None:
            merged_config.update(config)
        
        # Apply constructor arguments (highest priority)
        self.resolution = resolution if resolution is not None else merged_config['resolution']
        
        try:
            physx = lazy.omni.physx.acquire_physx_interface()
            stage_id = lazy.omni.usd.get_context().get_stage_id()
            self.generator = (
                lazy.omni.isaac.occupancy_map.bindings._occupancy_map.Generator(
                    physx, stage_id
                )
            )

            # Configure generator settings
            # Values 4, 5, 6 represent available, occupied, unknown respectively
            self.generator.update_settings(self.resolution, 4, 5, 6)

            print(f"OGM Generator initialized with resolution: {self.resolution}")

        except Exception as e:
            print(f"[Error] Failed to initialize OGM Generator: {e}")
            raise

    def generate_grid_map(
        self,
        map_center: Tuple[float, float, float] = (0, 0, 0),
        lower_bound: Tuple[float, float, float] = (0, 0, 0),
        upper_bound: Tuple[float, float, float] = (0, 0, 0),
        return_img: bool = False,
    ) -> Union[th.Tensor, np.ndarray]:
        """Generate occupancy grid map.

        Args:
            map_center: Center coordinates (x, y, z) of the map
            lower_bound: Lower bounds (x, y, z) of the map
            upper_bound: Upper bounds (x, y, z) of the map
            return_img: If True, return BGR image; if False, return tensor

        Returns:
            Generated map as tensor or BGR image array
        """
        try:
            self.generator.set_transform(map_center, lower_bound, upper_bound)
            self.generator.generate2d()

            dims = self.generator.get_dimensions()
            w, h, c = dims

            # Get colored buffer
            flat_buf = self.generator.get_colored_byte_buffer(
                (0, 0, 0, 255),  # Black for obstacles
                (255, 255, 255, 255),  # White for free space
                (128, 128, 128, 255),  # Gray for unknown
            )

            # Convert to numpy array
            byte_vals = [ord(c) for c in flat_buf]
            arr = np.array(byte_vals, dtype=np.uint8).reshape((h, w, 4))
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

            # Flip horizontally for alignment with OmniGibson
            img_bgr = cv2.flip(img_bgr, 1)

            # Store both formats
            self.tensor_map = self._bgr_to_tensor(img_bgr)
            self.bgr_map = img_bgr

            print(f"Generated map: {w}x{h}, center={map_center}")
            # note: save img for test
            # cv2.imwrite("map.png", img_bgr)

            return self.bgr_map if return_img else self.tensor_map

        except Exception as e:
            print(f"[Error] Error generating grid map: {e}")
            raise

    def update_env_trav_map(self, env: og.Environment) -> th.Tensor:
        """Update environment traversability map with generated occupancy grid.

        Args:
            env: OmniGibson environment

        Returns:
            Generated map tensor
        """
        try:
            # move robot to sky
            if env.robots:
                robot: Tiago = env.robots[0]
                pos, ori = robot.get_position_orientation()
                robot.set_position_orientation(
                    th.as_tensor([0, 0, 100], dtype=th.float32),
                )
                for _ in range(5):
                    env.step(th.zeros(robot.action_dim))
            # generate map
            map_tensor = self.generate_grid_map(
                map_center=(0, 0, 0),
                lower_bound=(-15, -15, 0.1),
                upper_bound=(15, 15, 0.5),
                return_img=False,
            )
            # put back robot
            if env.robots:
                robot: Tiago = env.robots[0]
                robot.set_position_orientation(pos, ori)
                for _ in range(5):
                    env.step(th.zeros(robot.action_dim))
            # update trav map
            env.scene.trav_map.floor_map[0] = map_tensor
            env.scene.trav_map.map_size = map_tensor.shape[0]

            print("Environment traversability map updated successfully")
            return map_tensor

        except Exception as e:
            print(f"[Error] Error updating environment traversability map: {e}")
            raise

    def _bgr_to_tensor(self, map_img: np.ndarray) -> th.Tensor:
        """Convert BGR image to grayscale tensor.

        Args:
            map_img: BGR image array

        Returns:
            Grayscale tensor
        """
        map_tensor = th.from_numpy(map_img)
        return map_tensor[:, :, 0]  # Use blue channel for grayscale
