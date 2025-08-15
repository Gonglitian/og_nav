"""Navigation interface for environment setup and navigation control."""

from typing import Tuple

import torch as th
import random

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.robots.tiago import Tiago
from omnigibson.utils.ui_utils import KeyboardEventHandler

from og_nav.mapping.occupancy_grid import OccupancyGridMap
from og_nav.planning.path_planning import PathPlanner
from og_nav.control.controllers import PathTrackingController
from og_nav.core.config_loader import NavigationConfig


class NavigationInterface:
    """
    High-level interface for robot navigation in OmniGibson environments.
    Manages both path planning and visualization.
    """

    def __init__(self, env: og.Environment, robot: Tiago, og_nav_config: dict = None, visualize: bool = True, keep_arm_pose: bool = True):
        """
        Initialize the navigation interface.

        Args:
            env: OmniGibson environment
            robot: Tiago robot instance
            og_nav_config: Dictionary containing og_nav configuration parameters
            visualize: Whether to enable path visualization
            keep_arm_pose: Whether to maintain arm pose during navigation
        """
        self.env = env
        self.robot = robot
        self.visualize = visualize
        self.keep_arm_pose = keep_arm_pose
        
        # Initialize unified configuration
        self.config = NavigationConfig(og_nav_config=og_nav_config)

        # Check if visualization is enabled in config
        self.visualization_enabled = self.config.get('visualization.enable', True) and visualize
        
        # Set robot arm pose
        if self.keep_arm_pose:
            nav_arm_pose = self.config.get('robot.nav_arm_pose')
            if nav_arm_pose:
                for arm_name in ["arm_left", "arm_right"]:
                    controller = self.robot.controllers[arm_name]
                    assert isinstance(controller, og.controllers.JointController)
        # Unlock robot arm command input limits
        self.robot.controllers['arm_left']._command_input_limits = None
        self.robot.controllers['arm_right']._command_input_limits = None
        
        # Update robot reset pose AABB extent
        self.original_reset_joint_pos_aabb_extent = self.robot._reset_joint_pos_aabb_extent
        self.robot._reset_joint_pos_aabb_extent *= 1.3
        
        # Initialize modules with their respective configurations
        ogm_config = self.config.get_ogm_config()
        planning_config = self.config.get_planning_config()
        controller_config = self.config.get_controller_config()
        
        # Initialize path planner with configuration
        self.planner = PathPlanner(self.env, self.robot, config=planning_config)

        # Initialize path tracking controller with configuration
        self.controller = PathTrackingController(robot=self.robot, config=controller_config)
        
        # Initialize OGM with configuration
        self.ogm = OccupancyGridMap(config=ogm_config)
        
        # Initialize state variables
        self.step_count = 0
        
        # Update environment traversability map
        self.ogm.update_env_trav_map(env)
        
        # Navigation state
        self.current_path = None
        self.goal_position = None
        self.scene = env.scene

        # Initialize visualization if enabled
        self._init_visualization()
        
    def _init_visualization(self):
        """Initialize visualization markers and keyboard callbacks."""
        if not self.visualization_enabled:
            self.start_point_marker = None
            self.end_point_marker = None
            self.waypoint_markers = []
            return

        # Get marker objects from scene
        self.start_point_marker = self.env.scene.object_registry("name", "start_point")
        self.end_point_marker = self.env.scene.object_registry("name", "end_point")

        # Get waypoint markers
        self.waypoint_markers = []
        n_waypoints = self.config.get('visualization.n_waypoints', 50)
        for i in range(n_waypoints):
            waypoint = self.env.scene.object_registry("name", f"waypoint_{i}")
            self.waypoint_markers.append(waypoint)

        # Setup keyboard callbacks
        self._setup_keyboard_callbacks()

        print(f"✓ Visualization initialized with {len(self.waypoint_markers)} waypoint markers")

    def _setup_keyboard_callbacks(self):
        """Setup keyboard callbacks for interactive path planning."""
        if not self.visualization_enabled:
            return

        KeyboardEventHandler.initialize()

        # Z: Set start point
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.Z,
            callback_fn=lambda: self._set_start_from_camera(),
        )

        # X: Set goal point
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.X,
            callback_fn=lambda: self._set_goal_from_camera(),
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

        # R: Reset robot pose
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.R, 
            callback_fn=self._reset_robot_pose
        )

        self._print_controls()

    def _set_start_from_camera(self):
        """Set start point from current camera position."""
        try:
            camera_pos = og.sim._viewer_camera.get_position_orientation()[0]
            self.set_goal((camera_pos[0].item(), camera_pos[1].item()), is_start=True)
            print(f"Start point set to: {camera_pos[:2]}")
        except Exception as e:
            print(f"[Error] Failed to set start point from camera: {e}")

    def _set_goal_from_camera(self):
        """Set goal point from current camera position."""
        try:
            camera_pos = og.sim._viewer_camera.get_position_orientation()[0]
            self.set_goal((camera_pos[0].item(), camera_pos[1].item()))
            print(f"Goal point set to: {camera_pos[:2]}")
        except Exception as e:
            print(f"[Error] Failed to set goal point from camera: {e}")

    def _plan_current_path(self):
        """Plan path using current start and goal points."""
        if self.planner.start_point_coords and self.planner.goal_point_coords:
            self.current_path = self.planner.plan_path()
            if self.current_path is not None:
                self.controller.set_path(self.current_path)
                self._update_waypoint_markers()

    def _reset_robot_pose(self):
        """Reset robot pose to original position."""
        self.robot.set_position_orientation(position=self.robot.get_position_orientation()[0], orientation=th.tensor([0, 0, 0, 1]))
    
    def _print_controls(self):
        """Print available keyboard controls."""
        print("\n=== Navigation Controls ===")
        print("Z: Set start point (green sphere)")
        print("X: Set goal point (red sphere)")
        print("C: Clear all markers and reset state")
        print("V: Plan path (blue sphere waypoints)")
        print("===========================\n")

    def _update_start_marker(self):
        """Update start point marker position."""
        if not self.visualization_enabled or not self.start_point_marker:
            return

        start_coords = self.planner.get_start_point_coords()
        if start_coords is not None:
            marker_height = self.config.get('visualization.marker_height', 0.1)
            self.start_point_marker.set_position_orientation(
                th.as_tensor([start_coords[0], start_coords[1], marker_height], dtype=th.float32)
            )

    def _update_goal_marker(self):
        """Update goal point marker position."""
        if not self.visualization_enabled or not self.end_point_marker:
            return

        goal_coords = self.planner.get_goal_point_coords()
        if goal_coords is not None:
            marker_height = self.config.get('visualization.marker_height', 0.1)
            self.end_point_marker.set_position_orientation(
                th.as_tensor([goal_coords[0], goal_coords[1], marker_height], dtype=th.float32)
            )

    def _update_waypoint_markers(self):
        """Update waypoint markers based on planned path."""
        if not self.visualization_enabled or not self.waypoint_markers:
            return

        waypoints = self.planner.get_waypoint_coords()
        waypoint_height = self.config.get('visualization.waypoint_radius', 0.05)

        # Update visible waypoints
        for i, waypoint_coords in enumerate(waypoints):
            if i < len(self.waypoint_markers) and self.waypoint_markers[i] is not None:
                self.waypoint_markers[i].set_position_orientation(
                    th.as_tensor([waypoint_coords[0], waypoint_coords[1], waypoint_height], dtype=th.float32)
                )

        # Hide unused waypoint markers
        hidden_position = self.config.get('visualization.hidden_position', [0, 0, 100])
        hidden_pos = th.as_tensor(hidden_position, dtype=th.float32)
        for i in range(len(waypoints), len(self.waypoint_markers)):
            if self.waypoint_markers[i] is not None:
                self.waypoint_markers[i].set_position_orientation(hidden_pos)

    def clear_all_markers(self):
        """Clear all visual markers by moving them to hidden position."""
        if not self.visualization_enabled:
            return

        hidden_position = self.config.get('visualization.hidden_position', [0, 0, 100])
        hidden_pos = th.as_tensor(hidden_position, dtype=th.float32)

        if self.start_point_marker is not None:
            self.start_point_marker.set_position_orientation(hidden_pos)

        if self.end_point_marker is not None:
            self.end_point_marker.set_position_orientation(hidden_pos)

        # Clear waypoints
        for waypoint in self.waypoint_markers:
            if waypoint is not None:
                waypoint.set_position_orientation(hidden_pos)

        # Clear planner coordinates
        self.planner.clear_coordinates()

    def set_goal(self, position: Tuple[float, float], is_start: bool = False):
        """Set the goal position for navigation.

        Args:
            position: Goal position as (x, y) tuple coordinates
            is_start: Whether this is setting start point instead of goal
        """
        if is_start:
            self.planner.set_start_point(position)
            self._update_start_marker()
            return

        goal_position = position
        self.goal_position = goal_position
        
        # Set start point as current robot position
        self.planner.set_start_point(self.robot.get_position_orientation()[0][:2], must_be_available=False)
        self.planner.set_goal_point(goal_position)

        # Update markers
        self._update_start_marker()
        self._update_goal_marker()

        # Plan path to the new goal
        self.current_path = self.planner.plan_path()
        if self.current_path is not None:
            # Skip first waypoint if there are multiple waypoints
            path_to_use = self.current_path
            if len(self.current_path) >= 2:
                path_to_use = self.current_path[1:]  # Skip the first waypoint
                # print(f"Skipped first waypoint, using {len(path_to_use)} waypoints for tracking")
            
            self.controller.set_path(path_to_use)
            self._update_waypoint_markers()
            
            # Log navigation status after successful planning
            status = self.get_navigation_status()
            return True
        else:
            # Log navigation status after planning failure
            status = self.get_navigation_status()
            print(f"Failed to plan path to goal: {goal_position} (status: {status})")
            return False

    def update(self) -> th.Tensor:
        """Update the navigation controller and return action.

        Returns:
            Action tensor for the robot
        """
        self.step_count += 1
        
        # Get control action from the controller
        # If no valid path is available, use no-op action
        if self.current_path is None or len(self.current_path) == 0:
            action = self.controller.no_op()
        else:
            action = self.controller.control()
        
        # If we've arrived at the goal, clear the path markers
        if self.is_arrived():
            self.clear_all_markers()
            
        # Set arm positions to navigation pose
        if self.keep_arm_pose:
            if isinstance(self.robot, Tiago):
                # Get arm pose from configuration
                nav_arm_pose = self.config.get('robot.nav_arm_pose')
                if nav_arm_pose:
                    nav_arm_pose_tensor = th.as_tensor(nav_arm_pose, dtype=th.float32)
                    # left arm
                    action[6:13] = nav_arm_pose_tensor
                    # right arm
                    action[14:21] = nav_arm_pose_tensor
                # head
                action[3:5] = th.zeros(2)
            else:
                raise NotImplementedError("Only Tiago robot is supported")
        return action

    def is_arrived(self) -> bool:
        """Check if the robot has arrived at the goal.
        
        Returns:
            True if robot has arrived at the goal, False otherwise
        """
        return self.controller.is_arrived()
    
    def has_valid_path(self) -> bool:
        """Check if there is a valid navigation path.
        
        Returns:
            True if a valid path is available for navigation, False otherwise
        """
        return (self.current_path is not None and 
                len(self.current_path) > 0 and 
                self.controller.has_valid_path())
    
    def get_navigation_status(self) -> dict:
        """Get comprehensive navigation status information.
        
        Returns:
            Dictionary containing navigation status details
        """
        return {
            'has_path': self.has_valid_path(),
            'is_arrived': self.is_arrived() if self.has_valid_path() else False,
            'current_path_length': len(self.current_path) if self.current_path is not None else 0,
            'goal_position': self.goal_position,
            'step_count': self.step_count
        }
    
    def is_position_valid(self, world_x: float, world_y: float) -> bool:
        """Check if a world position is in a valid (traversable) area on the map.
        
        Args:
            world_x: X coordinate in world space
            world_y: Y coordinate in world space
            
        Returns:
            True if position is valid (map value is 255), False otherwise
        """
        if self.ogm.map_tensor is None:
            return False
            
        try:
            pixel_x, pixel_y = self.ogm.trav_map.world_to_map(th.tensor([world_x, world_y]))
            map_size = self.ogm.trav_map.map_size
            
            # Check if coordinates are within map bounds
            if 0 <= pixel_x < map_size and 0 <= pixel_y < map_size:
                # Check if the map value at this position is 255 (traversable)
                map_value = self.ogm.map_tensor[pixel_y, pixel_x].item()
                return map_value == 255
            else:
                return False
        except Exception as e:
            print(f"Error checking position validity: {e}")
            return False
    
    def set_random_available_goal(self):
        """Set a random goal position that is in a valid (traversable) area on the map."""
        _, goal_position = self.ogm.trav_map.get_random_point(floor=0, robot=self.robot)
        
        self.set_goal(goal_position[:2])
        
        return goal_position[:2]
    
    def env_no_op(self):
        action = th.zeros(self.robot.action_dim)
        self.env.step(action)