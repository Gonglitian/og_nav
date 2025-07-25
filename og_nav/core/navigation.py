"""Navigation utilities for environment setup and point availability checking."""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch as th
import yaml

import omnigibson as og
from omnigibson.robots.tiago import Tiago
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene

from ..planning.path_planning import PathPlanner
from ..control.controllers import PathTrackingController
from .constants import NavConstants


class NavigationUtils:
    """Utility class for navigation environment setup and management."""
    
    @staticmethod
    def is_point_available(x: float, y: float, env: og.Environment, robot: Optional[Tiago] = None) -> bool:
        """Check if a point is available for navigation.
        
        Args:
            x: X coordinate
            y: Y coordinate
            env: OmniGibson environment
            robot: Robot instance (optional)
            
        Returns:
            True if point is available, False otherwise
        """
        # Simple implementation - in a real scenario, this would check against the traversability map
        return True
    
    @staticmethod
    def find_nearest_available_point(x: float, y: float, env: og.Environment, robot: Optional[Tiago] = None) -> Optional[Tuple[float, float]]:
        """Find the nearest available point to the given coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            env: OmniGibson environment
            robot: Robot instance (optional)
            
        Returns:
            Nearest available point (x, y) or None if not found
        """
        # Simple implementation - in a real scenario, this would search the traversability map
        return (x, y)


class NavigationInterface:
    """Interface class for navigation."""
    
    def __init__(self, env: og.Environment, robot: Optional[Tiago] = None):
        """Initialize the navigation interface.
        
        Args:
            env: OmniGibson environment instance
            robot: Robot instance (optional, will use env.robots[0] if not provided)
        """
        self.env = env
        self.robot = robot if robot is not None else env.robots[0]
        self.scene = env.scene
        
        # Initialize path planner and controller
        self.planner = PathPlanner(env, robot=self.robot, visualize=False)
        self.controller = PathTrackingController(robot=self.robot)
        
        # Navigation state
        self.current_path = None
        self.goal_position = None
        
        print("NavigationInterface initialized")
    
    def setup(self):
        """Setup the navigation interface."""
        # Setup keyboard callbacks for the planner
        self.planner.setup_keyboard_callbacks()
        
        # Set default start and goal positions from constants
        self.planner.set_start_point(NavConstants.START_POSITION[:2])
        self.planner.set_goal_point(NavConstants.GOAL_POSITION[:2])
        
        print("NavigationInterface setup completed")
    
    def set_goal(self, goal_position: Tuple[float, float]):
        """Set the goal position for navigation.
        
        Args:
            goal_position: (x, y) coordinates for goal point
        """
        self.goal_position = goal_position
        self.planner.set_goal_point(goal_position)
        
        # Plan path to the new goal
        self.current_path = self.planner.plan_path()
        if self.current_path is not None:
            self.controller.set_path(self.current_path)
            print(f"Path planned to goal: {goal_position}")
        else:
            print("Failed to plan path to goal")
    
    def update(self) -> th.Tensor:
        """Update the navigation controller and return action.
        
        Returns:
            Action tensor for the robot
        """
        # If we don't have a path yet, plan one
        if self.current_path is None and self.goal_position is not None:
            self.current_path = self.planner.plan_path()
            if self.current_path is not None:
                self.controller.set_path(self.current_path)
        
        # Get control action from the controller
        action, arrived = self.controller.control(return_action=True)
        
        # If we've arrived at the goal, clear the path
        if arrived:
            self.current_path = None
            print("Arrived at goal")
        
        # Set arm positions to navigation pose
        # left arm
        action[6:13] = NavConstants.NAV_ARM_POSE
        # right arm
        action[14:21] = NavConstants.NAV_ARM_POSE
        # head
        action[3:5] = th.tensor([0, 0])
        
        return action
    
    def goto_position(self, position: Tuple[float, float]) -> None:
        """Navigate to a specific position.
        
        Args:
            position: (x, y) coordinates to navigate to
        """
        self.set_goal(position)