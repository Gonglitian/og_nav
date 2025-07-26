"""Navigation interface for environment setup and navigation control."""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch as th
import yaml

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.robots.tiago import Tiago
from omnigibson.controllers import JointController
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.utils.ui_utils import KeyboardEventHandler

from og_nav.mapping.occupancy_grid import OGMGenerator
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
        self.robot._reset_joint_pos_aabb_extent *= 1.1
        
        # Initialize modules with their respective configurations
        ogm_config = self.config.get_ogm_config()
        planning_config = self.config.get_planning_config()
        controller_config = self.config.get_controller_config()
        
        # Initialize path planner with configuration
        self.planner = PathPlanner(self.env, self.robot, config=planning_config)

        # Initialize path tracking controller with configuration
        self.controller = PathTrackingController(robot=self.robot, config=controller_config)
        
        # Initialize OGM with configuration
        self.ogm = OGMGenerator(config=ogm_config)
        
        # Initialize state variables
        self.step_count = 0

        # Adjust robot reset pose AABB extent
        self.robot._reset_joint_pos_aabb_extent *= 1.2

        # Update environment traversability map
        self.ogm.update_env_trav_map(env)

        # Navigation state
        self.current_path = None
        self.goal_position = None
        self.scene = env.scene

        # Initialize visualization if enabled
        self._init_visualization()
        
        print(f"NavigationInterface initialized (visualization: {self.visualization_enabled})")

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

        self._print_controls()

    def _set_start_from_camera(self):
        """Set start point from current camera position."""
        try:
            camera_pos = og.sim._viewer_camera.get_position_orientation()[0]
            self.set_goal((camera_pos[0].item(), camera_pos[1].item()), is_start=True)
        except Exception as e:
            print(f"[Error] Failed to set start point from camera: {e}")

    def _set_goal_from_camera(self):
        """Set goal point from current camera position."""
        try:
            camera_pos = og.sim._viewer_camera.get_position_orientation()[0]
            self.set_goal((camera_pos[0].item(), camera_pos[1].item()))
        except Exception as e:
            print(f"[Error] Failed to set goal point from camera: {e}")

    def _plan_current_path(self):
        """Plan path using current start and goal points."""
        if self.planner.start_point_coords and self.planner.goal_point_coords:
            self.current_path = self.planner.plan_path()
            if self.current_path:
                self.controller.set_path(self.current_path)
                self._update_waypoint_markers()

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
        if goal_coords:
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
        self.planner.set_start_point(self.robot.get_position_orientation()[0][:2])
        self.planner.set_goal_point(goal_position)

        # Update markers
        self._update_start_marker()
        self._update_goal_marker()

        # Plan path to the new goal
        self.current_path = self.planner.plan_path()
        if self.current_path is not None:
            self.controller.set_path(self.current_path)
            self._update_waypoint_markers()
            print(f"Path planned to goal: {goal_position}")
        else:
            print("Failed to plan path to goal")

    def update(self) -> th.Tensor:
        """Update the navigation controller and return action.

        Returns:
            Action tensor for the robot
        """
        self.step_count += 1
        
        # Get control action from the controller
        action = self.controller.control()
        
        # If we've arrived at the goal, clear the path markers
        if self.controller.is_arrived():
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
        """Check if the robot has arrived at the goal."""
        return self.controller.is_arrived()