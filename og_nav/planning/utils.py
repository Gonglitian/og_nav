"""Navigation utility functions."""

from typing import List, Optional, Tuple

import numpy as np
import torch as th
import omnigibson as og
from omnigibson.robots.tiago import Tiago
from omnigibson.scenes import InteractiveTraversableScene


def is_point_available(
    x: float, y: float, env: og.Environment, robot: Optional[Tiago] = None
) -> bool:
    """Check if a point is traversable in the environment.
    
    Args:
        x: X coordinate in world frame.
        y: Y coordinate in world frame.
        env: OmniGibson environment.
        robot: Robot instance for size-aware collision checking.
        
    Returns:
        True if point is traversable, False otherwise.
    """
    try:
        scene: InteractiveTraversableScene = env.scene
        
        # Get traversable map with robot-aware erosion
        map_tensor = th.clone(scene.trav_map.floor_map[0])
        trav_map = scene.trav_map._erode_trav_map(map_tensor, robot=robot)
        
        # Convert world coordinates to map coordinates
        world_point = th.tensor([x, y])
        map_coords = scene.trav_map.world_to_map(world_point)
        
        # Check bounds
        map_height, map_width = trav_map.shape
        map_x, map_y = int(map_coords[0].item()), int(map_coords[1].item())
        
        if not (0 <= map_x < map_height and 0 <= map_y < map_width):
            print(
                f"Point ({x:.3f}, {y:.3f}) -> map({map_x}, {map_y}) "
                f"out of bounds. Map size: {map_height}x{map_width}"
            )
            return False
        
        # Check traversability (255 = available space)
        pixel_value = trav_map[map_x, map_y].item()
        return pixel_value == 255
        
    except (AttributeError, IndexError, RuntimeError) as e:
        print(f"[Error] Error checking point availability: {e}")
        return False


def get_available_points_in_radius(
    center_x: float,
    center_y: float,
    radius: float,
    env: og.Environment,
    robot: Optional[Tiago] = None,
    num_samples: int = 100,
) -> List[Tuple[float, float]]:
    """Get available points within a radius around a center point.
    
    Args:
        center_x: Center X coordinate
        center_y: Center Y coordinate
        radius: Search radius
        env: OmniGibson environment
        robot: Robot instance for collision checking
        num_samples: Number of random points to sample
        
    Returns:
        List of available (x, y) coordinate tuples
    """
    available_points = []
    
    try:
        for _ in range(num_samples):
            # Generate random point within radius
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, radius)
            
            x = center_x + distance * np.cos(angle)
            y = center_y + distance * np.sin(angle)
            
            if is_point_available(x, y, env, robot):
                available_points.append((x, y))
                
    except Exception as e:
        print(f"[Error] Error sampling available points: {e}")
    
    return available_points


def find_nearest_available_point(
    target_x: float,
    target_y: float,
    env: og.Environment,
    robot: Optional[Tiago] = None,
    search_radius: float = 2.0,
    num_samples: int = 50,
) -> Optional[Tuple[float, float, float]]:
    """Find the nearest available point to a target location.
    
    Args:
        target_x: Target X coordinate
        target_y: Target Y coordinate
        env: OmniGibson environment
        robot: Robot instance for collision checking
        search_radius: Maximum search radius
        num_samples: Number of points to sample per radius
        
    Returns:
        Tuple of (x, y, distance) of nearest available point, or None if not found
    """
    try:
        # First check if target point itself is available
        if is_point_available(target_x, target_y, env, robot):
            return (target_x, target_y, 0.0)
        
        # Search in expanding circles
        for radius in np.linspace(0.1, search_radius, 10):
            available_points = get_available_points_in_radius(
                target_x, target_y, radius, env, robot, num_samples
            )
            
            if available_points:
                # Find closest point
                distances = [
                    np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
                    for x, y in available_points
                ]
                min_idx = np.argmin(distances)
                closest_point = available_points[min_idx]
                
                return (closest_point[0], closest_point[1], distances[min_idx])
        
        print(
            f"[Warning] No available point found within radius {search_radius} "
            f"of ({target_x:.3f}, {target_y:.3f})"
        )
        return None
        
    except Exception as e:
        print(f"[Error] Error finding nearest available point: {e}")
        return None
