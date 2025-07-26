"""Path planning utilities for robot navigation."""

from typing import List, Optional, Tuple

import torch as th

import omnigibson as og
from omnigibson.robots.tiago import Tiago
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene

from og_nav.planning.utils import is_point_available, find_nearest_available_point


class PathPlanner:
    """Path planning for robot navigation.

    This class handles path planning algorithms and coordinate storage.
    All visualization is managed by NavigationInterface.
    """
    
    @staticmethod
    def get_default_cfg() -> dict:
        """Get default configuration for PathPlanner."""
        return {
            # Path planning algorithm parameters will go here in future
            # For now, path planning uses OmniGibson's built-in algorithms
        }

    def __init__(self, env: og.Environment, robot: Optional[Tiago] = None, config: Optional[dict] = None):
        """Initialize the path planner.

        Args:
            env: OmniGibson environment instance
            robot: Robot instance for collision checking (optional)
            config: Planning configuration dict (optional)
        """
        self.env = env
        self.scene: InteractiveTraversableScene = env.scene
        self.robot = robot
        
        # Merge config with defaults
        default_config = self.get_default_cfg()
        if config is not None:
            # Deep merge user config with defaults
            self.config = self._deep_merge(default_config, config)
        else:
            self.config = default_config

        # Coordinate storage
        self.start_point_coords: Optional[Tuple[float, float]] = None
        self.goal_point_coords: Optional[Tuple[float, float]] = None
        self.waypoints_coords: List[Tuple[float, float]] = []

        print(f"PathPlanner initialized with robot: {robot is not None}")
    
    def _deep_merge(self, default: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def set_start_point(self, position: Tuple[float, float]) -> None:
        """Set start point for path planning.

        Args:
            position: (x, y) coordinates
        """
        # Check if point is available
        if not is_point_available(position[0], position[1], self.env, self.robot):
            print("[Warning] Start point not available, finding nearest valid point...")
            nearest = find_nearest_available_point(position[0], position[1], self.env, self.robot)
            if nearest is None:
                print("[Error] No valid start point found nearby")
                return
            position = (nearest[0], nearest[1])

        self.start_point_coords = position
        print(f"Start point set to: {position}")

    def set_goal_point(self, position: Tuple[float, float]) -> None:
        """Set the goal point for path planning.

        Args:
            position: (x, y) coordinates for goal point
        """
        # Check point availability and find nearest if needed
        if not is_point_available(position[0], position[1], self.env, self.robot):
            print(f"[Warning] Goal point {position} is not available")
            nearest_point = find_nearest_available_point(position[0], position[1], self.env, self.robot)
            if nearest_point:
                position = (nearest_point[0], nearest_point[1])
                print(f"Using nearest available point: {position}")
            else:
                print("[Error] No available point found near goal position")
                return

        self.goal_point_coords = position
        print(f"Goal point set to: {position}")

    def clear_coordinates(self) -> None:
        """Clear all stored coordinates."""
        self.start_point_coords = None
        self.goal_point_coords = None
        self.waypoints_coords = []

    def plan_path(
        self,
        start_pos: Optional[Tuple[float, float]] = None,
        end_pos: Optional[Tuple[float, float]] = None,
    ) -> Optional[List[Tuple[float, float]]]:
        """Plan a path between two points.

        Args:
            start_pos: Start position (x, y). Uses current start point if None.
            end_pos: End position (x, y). Uses current goal point if None.

        Returns:
            List of waypoints as (x, y) tuples, or None if planning failed
        """
        # Use provided positions or fall back to stored coordinates
        if start_pos is None:
            start_pos = self.start_point_coords
        if end_pos is None:
            end_pos = self.goal_point_coords

        # Validate positions
        if start_pos is None or end_pos is None:
            print("[Error] Start or goal position not set")
            return None

        print(f"Planning path from {start_pos} to {end_pos}")

        # Use OmniGibson's path planning
        self.waypoints_coords, _ = self.scene.trav_map.get_shortest_path(
            0, start_pos, end_pos, entire_path=True, robot=self.robot
        )
        print(f"Path planned with {len(self.waypoints_coords)} waypoints")
        return self.waypoints_coords

    def get_start_point_coords(self) -> Optional[Tuple[float, float]]:
        """Get current start point coordinates.

        Returns:
            Start point (x, y) or None if not set
        """
        return self.start_point_coords

    def get_goal_point_coords(self) -> Optional[Tuple[float, float]]:
        """Get current goal point coordinates.

        Returns:
            Goal point (x, y) or None if not set
        """
        return self.goal_point_coords

    def get_waypoint_coords(self) -> List[Tuple[float, float]]:
        """Get current waypoint coordinates.

        Returns:
            List of waypoint coordinates
        """
        return self.waypoints_coords
