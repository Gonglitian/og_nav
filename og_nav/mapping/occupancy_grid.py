"""Occupancy grid map generation utilities."""

from typing import Tuple, Union, Optional

import cv2
import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
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
            # Record robot info for post-processing
            robot_info = None
            if env.robots:
                robot = env.robots[0]
                pos, ori = robot.get_position_orientation()
                
                # Get robot's AABB extent (consistent with navigation.py)
                
                robot_info = {
                    'position': pos,
                    'orientation': ori,
                    'aabb_extent': robot._reset_joint_pos_aabb_extent,
                    'yaw': T.quat2euler(ori)[2]  # Get yaw angle
                }
            
            # Generate map (robot stays in place)
            map_tensor = self.generate_grid_map(
                map_center=(0, 0, 0),
                lower_bound=(-15, -15, 0.1),
                upper_bound=(15, 15, 0.5),
                return_img=False,
            )
            
            # Post-process: clear robot area and save visualization
            if robot_info:
                self._clear_robot_area_and_save(map_tensor, robot_info)
            
            # Update environment's traversability map
            env.scene.trav_map.floor_map[0] = map_tensor
            env.scene.trav_map.map_size = map_tensor.shape[0]

            return map_tensor

        except Exception as e:
            print(f"[Error] Error updating environment traversability map: {e}")
            raise

    def _clear_robot_area_and_save(self, map_tensor: th.Tensor, robot_info: dict, save_img: bool = False,save_path: str = "map_with_robot_bbox.png") -> None:
        """Clear robot occupied area and save map image with bounding box.
        
        Args:
            map_tensor: The occupancy grid map tensor
            robot_info: Dictionary containing robot position, orientation, and AABB extent
        """
        # Get robot information
        robot_x = robot_info['position'][0].item()
        robot_y = robot_info['position'][1].item()
        robot_yaw = robot_info['yaw']
        
        # AABB half extents (xy plane)
        half_extent_x = robot_info['aabb_extent'][0].item() / 2
        half_extent_y = robot_info['aabb_extent'][1].item() / 2
        
        # Map parameters
        map_size = map_tensor.shape[0]
        world_size = 30.0  # From (-15, -15) to (15, 15)
        
        # World to pixel coordinate conversion
        def world_to_pixel(x, y):
            px = int((x + 15) / world_size * map_size)
            py = int((y + 15) / world_size * map_size)
            return px, py
        
        # Calculate rotated rectangle corners (world coordinates)
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        
        # Local coordinate corners
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
        corners_pixel = [world_to_pixel(wx, wy) for wx, wy in corners_world]
        
        # Clear robot occupied area (fill rotated rectangle)
        mask = np.zeros((map_size, map_size), dtype=np.uint8)
        corners_np = np.array(corners_pixel, dtype=np.int32)
        cv2.fillPoly(mask, [corners_np], 1)
        
        # Set mask area as traversable (255)
        map_tensor[mask == 1] = 255
        
        # Save image with red bounding box
        if save_img and hasattr(self, 'bgr_map') and self.bgr_map is not None:
            # Copy for drawing
            img_with_box = self.bgr_map.copy()
            
            # Draw red rectangle (BGR format: B=0, G=0, R=255)
            cv2.polylines(img_with_box, [corners_np], isClosed=True, 
                         color=(0, 0, 255), thickness=2)
            
            # Mark robot center with green dot
            center_pixel = world_to_pixel(robot_x, robot_y)
            cv2.circle(img_with_box, center_pixel, 3, (0, 255, 0), -1)
            
            # Add text info
            text = f"Robot AABB: {half_extent_x*2:.2f}m x {half_extent_y*2:.2f}m"
            cv2.putText(img_with_box, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
            
            # Save image
            cv2.imwrite(save_path, img_with_box)
            print(f"Saved map with robot bounding box to {save_path}")
    
    def _bgr_to_tensor(self, map_img: np.ndarray) -> th.Tensor:
        """Convert BGR image to grayscale tensor.

        Args:
            map_img: BGR image array

        Returns:
            Grayscale tensor
        """
        map_tensor = th.from_numpy(map_img)
        return map_tensor[:, :, 0]  # Use blue channel for grayscale
