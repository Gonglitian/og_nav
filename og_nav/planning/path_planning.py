"""Path planning utilities with optional visualization support."""

from typing import List, Optional, Tuple

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.robots.tiago import Tiago
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.utils.ui_utils import KeyboardEventHandler

from ..core.navigation import NavigationUtils


class PathPlanner:
    """Path planning with optional visualization support.
    
    This class handles path planning and visualization for testing.
    When visualization is enabled, it manages visual markers for start/goal
    points and waypoints. When disabled, it only stores coordinates internally.
    """
    
    def __init__(self, env: og.Environment, robot: Optional[Tiago] = None, visualize: bool = True, n_waypoints: int = 50):
        """Initialize the path planner.
        
        Args:
            env: OmniGibson environment instance
            robot: Robot instance for collision checking (optional)
            visualize: Whether to enable visual markers
            n_waypoints: Number of waypoints to visualize
        """
        self.env = env
        self.robot = robot
        self.visualize = visualize
        self.n_waypoints = n_waypoints
        
        # Marker coordinate storage (always available)
        self.start_point_coords: Optional[Tuple[float, float]] = None
        self.goal_point_coords: Optional[Tuple[float, float]] = None
        self.waypoints_coords: List[Tuple[float, float]] = []
        
        # Visual marker objects (only if visualization enabled)
        # Also you should make sure markers with following names are created in the scene
        if self.visualize:
            self.start_point:PrimitiveObject = env.scene.object_registry("name", "start_point")
            self.end_point:PrimitiveObject = env.scene.object_registry("name", "end_point")
            
            # Get waypoint markers
            self.waypoints: List[PrimitiveObject] = []
            for i in range(self.n_waypoints):  # N_WAYPOINTS from original config
                waypoint = env.scene.object_registry("name", f"waypoint_{i}")
                self.waypoints.append(waypoint)
        
        print(f"PathPlanner initialized with visualization: {visualize}, robot: {robot is not None}")
    
    def set_start_point(self, position: Tuple[float, float]) -> None:
        """Set start point for path planning.
        
        Args:
            position: (x, y) coordinates
        """
        # Check if point is available
        if not NavigationUtils.is_point_available(position[0], position[1], self.env, self.robot):
            print("[Warning] Start point not available, finding nearest valid point...")
            nearest = NavigationUtils.find_nearest_available_point(
                position[0], position[1], self.env, self.robot
            )
            if nearest is None:
                print("[Error] No valid start point found nearby")
                return
            position = (nearest[0], nearest[1])
        
        # Update marker position
        if self.start_point is not None:
            self.start_point.set_position_orientation(
                th.tensor([position[0], position[1], 0.1])
            )
            self.start_point_coords = position
            print(f"Start point marker set to: {position}")
    
    def set_goal_point(self, position: Tuple[float, float]) -> None:
        """Set the goal point for path planning.
        
        Args:
            position: (x, y) coordinates for goal point
        """
        # Check point availability and find nearest if needed
        if not NavigationUtils.is_point_available(position[0], position[1], self.env, self.robot):
            print(f"[Warning] Goal point {position} is not available")
            nearest_point = NavigationUtils.find_nearest_available_point(
                position[0], position[1], self.env, self.robot
            )
            if nearest_point:
                position = (nearest_point[0], nearest_point[1])
                print(f"Using nearest available point: {position}")
            else:
                print("[Error] No available point found near goal position")
                return
                
        # Update marker position
        if self.end_point is not None:
            self.end_point.set_position_orientation(
                th.tensor([position[0], position[1], 0.1])
            )
            self.goal_point_coords = position
            print(f"Goal point marker set to: {position}")

    def clear_all_markers(self) -> None:
        """Clear all visual markers by moving them to hidden position."""
        hidden_pos = th.tensor([0, 0, 100])  # Move far away
        
        if self.start_point is not None:
            self.start_point.set_position_orientation(hidden_pos)
            
        if self.end_point is not None:
            self.end_point.set_position_orientation(hidden_pos)
            
        # Clear waypoints
        for waypoint in self.waypoints:
            if waypoint is not None:
                waypoint.set_position_orientation(hidden_pos)
        
        # Clear coordinate storage
        self.start_point_coords = None
        self.goal_point_coords = None
        self.waypoints_coords = []
        
        print("All markers cleared")

    def plan_path(
        self, 
        start_pos: Optional[Tuple[float, float]] = None, 
        end_pos: Optional[Tuple[float, float]] = None,
    ) -> Optional[List[th.Tensor]]:
        """Plan a path between two points.
        
        Args:
            start_pos: Start position (x, y). Uses current start point if None.
            end_pos: End position (x, y). Uses current goal point if None.
            
        Returns:
            List of waypoints as tensors, or None if planning failed
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
        
        # Simple straight line path for demonstration
        # In a real implementation, this would use a path planning algorithm
        waypoints = []
        num_waypoints = self.n_waypoints if self.visualize else 10
        
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1) if num_waypoints > 1 else 0
            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            waypoint = th.tensor([x, y])
            waypoints.append(waypoint)
            self.waypoints_coords.append((x, y))
            
        # Update visual markers if enabled
        if self.visualize and self.waypoints:
            for i, waypoint in enumerate(waypoints):
                if i < len(self.waypoints) and self.waypoints[i] is not None:
                    self.waypoints[i].set_position_orientation(
                        th.tensor([waypoint[0], waypoint[1], 0.05])
                    )
        
        print(f"Path planned with {len(waypoints)} waypoints")
        return waypoints

    def get_start_point(self) -> Optional[Tuple[float, float]]:
        """Get current start point coordinates.
        
        Returns:
            Start point (x, y) or None if not set
        """
        return self.start_point_coords

    def get_goal_point(self) -> Optional[Tuple[float, float]]:
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

    def setup_keyboard_callbacks(self) -> None:
        """Setup keyboard callbacks for interactive path planning."""
        if not self.visualize:
            print("[Warning] Keyboard callbacks only work with visualization enabled")
            return
            
        KeyboardEventHandler.initialize()
        
        # Z: Set start point
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.Z,
            callback_fn=lambda: self._set_start_from_camera()
        )
        
        # X: Set goal point
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.X,
            callback_fn=lambda: self._set_goal_from_camera()
        )
        
        # C: Clear all markers
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.C,
            callback_fn=self.clear_all_markers
        )
        
        # V: Plan current path
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.V,
            callback_fn=self._plan_current_path
        )
        
        self._print_controls()

    def _set_start_from_camera(self) -> None:
        """Set start point from current camera position."""
        try:
            camera_pos = og.sim.viewer_camera.get_position_orientation()[0]
            self.set_start_point((camera_pos[0].item(), camera_pos[1].item()))
        except Exception as e:
            print(f"[Error] Failed to set start point from camera: {e}")

    def _set_goal_from_camera(self) -> None:
        """Set goal point from current camera position."""
        try:
            camera_pos = og.sim.viewer_camera.get_position_orientation()[0]
            self.set_goal_point((camera_pos[0].item(), camera_pos[1].item()))
        except Exception as e:
            print(f"[Error] Failed to set goal point from camera: {e}")

    def _plan_current_path(self) -> None:
        """Plan path using current start and goal points."""
        if self.start_point_coords is None or self.goal_point_coords is None:
            print("[Warning] Start and goal points must be set before planning")
            return
            
        path = self.plan_path()
        if path:
            print(f"[Info] Path planned with {len(path)} waypoints")
        else:
            print("[Error] Failed to plan path")

    def _print_controls(self) -> None:
        """Print available keyboard controls."""
        print("\n=== Path Planning Controls ===")
        print("Z: Set start point (green sphere)")
        print("X: Set goal point (red sphere)")
        print("C: Clear all markers and reset state")
        print("V: Plan path (blue sphere waypoints)")
        print("===============================\n") 