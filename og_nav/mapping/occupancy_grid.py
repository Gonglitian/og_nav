"""Occupancy grid map generation utilities."""

from typing import Tuple, Union, Optional

import cv2
import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.robots.tiago import Tiago
from omnigibson.maps.traversable_map import TraversableMap

class OccupancyGridMap:
    """Occupancy Grid Map for OmniGibson environments.

    This class provides functionality to generate occupancy grid maps
    from 3D environments and update traversability maps.
    """
    
    @staticmethod
    def get_default_cfg() -> dict:
        """Get default configuration for OccupancyGridMap."""
        return {
            'resolution': 0.1,
            'world_size': 30
        }

    def __init__(self, resolution: Optional[float] = None, world_size: Optional[float] = None, config: Optional[dict] = None) -> None:
        """Initialize Occupancy Grid Map.

        Priority order for parameters:
        1. Constructor arguments (highest priority)
        2. config dict values
        3. Default values (lowest priority)

        Args:
            resolution: Map resolution in meters per pixel
            world_size: World coordinate range in meters (from -world_size/2 to +world_size/2)
            config: OGM configuration dict
        """
        # Merge config with defaults following priority order
        default_config = self.get_default_cfg()
        merged_config = default_config.copy()
        if config is not None:
            merged_config.update(config)
        
        # Apply constructor arguments (highest priority)
        self.resolution = resolution if resolution is not None else merged_config['resolution']
        self.world_size = world_size if world_size is not None else merged_config['world_size']
        self.trav_map: TraversableMap = None  # Will store reference to omnigibson traversable map
        
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

            self.map_tensor = None
            print(f"OGM Generator initialized with resolution: {self.resolution}")

        except Exception as e:
            print(f"[Error] Failed to initialize OGM Generator: {e}")
            raise

    def generate_grid_map(
        self,
        map_center: Tuple[float, float, float] = (0, 0, 0),
        lower_bound: Tuple[float, float, float] = (0, 0, 0),
        upper_bound: Tuple[float, float, float] = (0, 0, 0)
    ) -> Union[th.Tensor, np.ndarray]:
        """Generate occupancy grid map.

        Args:
            map_center: Center coordinates (x, y, z) of the map
            lower_bound: Lower bounds (x, y, z) of the map
            upper_bound: Upper bounds (x, y, z) of the map

        Returns:
            map_tensor (th.Tensor, shape: (h, w)): Occupancy grid map as tensor
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
            
            # Flip horizontally for alignment with OmniGibson
            return self._bgr_to_tensor(cv2.flip(cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR), 1))

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
            # Record robot info for post-processing
            robot_info = None
            if env.robots:
                robot = env.robots[0]
                pos, ori = robot.get_position_orientation()
                print(f"Robot position: {pos}, orientation: {ori}")
                # Get robot's AABB extent (consistent with navigation.py)
                
                robot_info = {
                    'position': pos,
                    'orientation': ori,
                    'aabb_extent': robot._reset_joint_pos_aabb_extent,
                    'yaw': T.quat2euler(ori)[2]  # Get yaw angle
                }
            
            # Generate map (robot stays in place)
            self.map_tensor = self.generate_grid_map(
                map_center=(0, 0, 0),
                lower_bound=(-self.world_size/2, -self.world_size/2, 0.1),
                upper_bound=(self.world_size/2, self.world_size/2, 0.5),
            )
            
            # Unavailable area
            unavailable_map_tensor = self.generate_grid_map(
                map_center=(0, 0, 0),
                lower_bound=(-self.world_size/2, -self.world_size/2, -0.1),
                upper_bound=(self.world_size/2, self.world_size/2, 5),
            )
            
            # Apply unavailable areas (value 0) from unavailable_map_tensor to map_tensor
            self.map_tensor[unavailable_map_tensor == 128] = 0
            
            # set unknown area to 255
            self.map_tensor[self.map_tensor != 0] = 255
            
            # Store reference to traversable map for coordinate conversion (before clearing robot area)
            self.trav_map = env.scene.trav_map
            # Update environment's traversability map
            self.trav_map.floor_map[0] = self.map_tensor
            self.trav_map.map_size = self.map_tensor.shape[0]
            self.trav_map.floor_heights = (0.0,)
            
            # Post-process: clear robot area and save visualization
            if robot_info:
                self.clear_robot_area(robot_info)
            
            # Update environment's traversability map again
            self.trav_map.floor_map[0] = self.map_tensor
            self.trav_map.map_size = self.map_tensor.shape[0]
            self.trav_map.floor_heights = (0.0,)

        except Exception as e:
            print(f"[Error] Error updating environment traversability map: {e}")
            raise

    def clear_robot_area(self, robot_info: dict, save_img: bool = False, save_path: str = "map_with_robot_bbox.png") -> None:
        """Clear robot occupied area using mask-based approach.
        
        This method creates a collision volume mask for the robot and applies it to the map tensor
        to ensure the robot's area is marked as traversable.
        
        Args:
            robot_info: Dictionary containing robot position, orientation, and AABB extent
            save_img: Whether to save visualization image
            save_path: Path to save the visualization image
        """
        # Step 1: Create robot collision volume mask
        robot_mask = self._create_robot_collision_mask(robot_info)
        
        # Step 2: Apply mask to map_tensor - set robot area as traversable (255)
        self.map_tensor[robot_mask == 1] = 255
    
    def _create_robot_collision_mask(self, robot_info: dict) -> np.ndarray:
        """Create a binary mask representing the robot's collision volume.
        
        Args:
            robot_info: Dictionary containing robot information
            
        Returns:
            Binary mask where 1 indicates robot collision area, 0 indicates free space
        """
        # Extract robot pose
        robot_x = robot_info['position'][0].item()
        robot_y = robot_info['position'][1].item()
        robot_yaw = robot_info['yaw']
        print(f"Robot x: {robot_x}, y: {robot_y}, yaw: {robot_yaw}")
        # Extract robot extents (half-sizes)
        half_extent_x = robot_info['aabb_extent'][0].item() / 2
        half_extent_y = robot_info['aabb_extent'][1].item() / 2
        
        # Calculate rotated rectangle corners in world coordinates
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        
        # Define rectangle corners in robot's local frame
        corners_local = [
            (-half_extent_x, -half_extent_y),
            (half_extent_x, -half_extent_y),
            (half_extent_x, half_extent_y),
            (-half_extent_x, half_extent_y)
        ]
        
        # Transform to world coordinates
        corners_world = []
        for lx, ly in corners_local:
            # Rotate and translate
            wx = robot_x + cos_yaw * lx - sin_yaw * ly
            wy = robot_y + sin_yaw * lx + cos_yaw * ly
            corners_world.append((wx, wy))
        
        # Convert to pixel coordinates
        corners_pixel = [self.trav_map.world_to_map(th.tensor([wx, wy])) for wx, wy in corners_world]
        corners_pixel = [th.flip(corner, dims=(0,)) for corner in corners_pixel]
        # Create binary mask
        mask = np.zeros((self.map_tensor.shape[0], self.map_tensor.shape[1]), dtype=np.uint8)
        corners_np = np.array(corners_pixel, dtype=np.int32)
        cv2.fillPoly(mask, [corners_np], 1)
        
        return mask
    
    def _bgr_to_tensor(self, map_img: np.ndarray) -> th.Tensor:
        """Convert BGR image to grayscale tensor.

        Args:
            map_img: BGR image array

        Returns:
            Grayscale tensor
        """
        map_tensor = th.from_numpy(map_img)[:, :, 0]
        return map_tensor

    def save_grayscale_tensor_to_image(self, map_tensor: th.Tensor, save_path: str) -> None:
        """Save grayscale tensor map as image.

        Args:
            map_tensor: Grayscale tensor map (shape: (h, w), values: 0-255)
            save_path: Path to save the image
        """
        # Convert tensor to numpy array with correct dtype
        map_np = map_tensor.cpu().numpy().astype(np.uint8)
        cv2.imwrite(save_path, map_np)
        print(f"Saved grayscale map to {save_path}")
        