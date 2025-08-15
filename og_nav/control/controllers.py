"""Control algorithms for robot navigation."""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.robots.tiago import Tiago

from og_nav.core.constants import TIAGO_BASE_ACTION_START_IDX, TIAGO_BASE_ACTION_END_IDX


class PathTrackingController:
    """Pure Pursuit path tracking controller.

    This controller implements the Pure Pursuit algorithm which tracks a path
    by continuously "chasing" a lookahead point on the path.

    Attributes:
        robot (Tiago): The robot instance to control.
        lookahead_distance (float): Distance to lookahead point on path.
        cruise_speed (float): Constant forward speed.
        max_angular_vel (float): Maximum angular velocity.
        waypoint_threshold (float): Distance threshold for waypoint arrival.
        path (List[Tuple[float, float]]): Current path waypoints.
        current_target_idx (int): Index of current target waypoint.
    """
    
    @staticmethod
    def get_default_cfg() -> dict:
        """Get default configuration for PathTrackingController."""
        return {
            'lookahead_distance': 0.5,
            'cruise_speed': 0.5,
            'max_angular_vel': 0.2,
            'waypoint_threshold': 0.2,
            'heading_threshold': 0.3  # Angle threshold in radians for initial rotation
        }

    def __init__(
        self,
        robot: Tiago,
        lookahead_distance: Optional[float] = None,
        cruise_speed: Optional[float] = None,
        max_angular_vel: Optional[float] = None,
        waypoint_threshold: Optional[float] = None,
        heading_threshold: Optional[float] = None,
        dt: Optional[float] = None,
        config: Optional[dict] = None,
    ) -> None:
        """Initialize Pure Pursuit controller.

        Priority order for parameters:
        1. Constructor arguments (highest priority)
        2. config dict values
        3. Default values (lowest priority)

        Args:
            robot: Robot instance to control
            lookahead_distance: Distance to lookahead point
            cruise_speed: Constant forward speed
            max_angular_vel: Maximum angular velocity
            waypoint_threshold: Distance threshold for waypoint arrival
            heading_threshold: Angle threshold for initial rotation
            dt: Time step for control
            config: Configuration dictionary
        """
        self.robot = robot
        
        # Merge configuration with defaults
        default_config = self.get_default_cfg()
        merged_config = default_config.copy()
        if config is not None:
            merged_config.update(config)
        
        # Set parameters with priority order
        self.lookahead_distance = lookahead_distance if lookahead_distance is not None else merged_config['lookahead_distance']
        self.cruise_speed = cruise_speed if cruise_speed is not None else merged_config['cruise_speed']
        self.max_angular_vel = max_angular_vel if max_angular_vel is not None else merged_config['max_angular_vel']
        self.waypoint_threshold = waypoint_threshold if waypoint_threshold is not None else merged_config['waypoint_threshold']
        self.heading_threshold = heading_threshold if heading_threshold is not None else merged_config['heading_threshold']
        
        # Time step
        self.dt = dt if dt is not None else og.sim.get_sim_step_dt()
        
        # Path and tracking state
        self.path: List[Tuple[float, float]] = []
        self.current_target_idx = 0
        
        # Initial rotation state management
        self._is_new_path = False
        self._initial_rotation_done = False
        self._needs_initial_rotation = False
        self._initial_rotation_target = None
        
        # Arrival state flag - set to True when robot arrives at final goal
        self._arrived_flag = False
        
        # Robot base action indices
        self.base_start_idx, self.base_end_idx = (
            TIAGO_BASE_ACTION_START_IDX,
            TIAGO_BASE_ACTION_END_IDX,
        )

        print(f"Pure Pursuit Controller initialized: lookahead={self.lookahead_distance}m, speed={self.cruise_speed}m/s, heading_threshold={self.heading_threshold}rad")

    def set_path(
        self, waypoints: Union[List[Tuple[float, float]], List[th.Tensor]]
    ) -> None:
        """Set new path for tracking.

        Args:
            waypoints: List of (x, y) waypoints
        """
        # Convert tensors to tuples if necessary
        if waypoints is not None and isinstance(waypoints[0], th.Tensor):
            waypoints = [tuple(waypoint.tolist()) for waypoint in waypoints]
        
        self.path = waypoints.copy()
        self.current_target_idx = 0
        
        # Reset arrival flag for new path
        self._arrived_flag = False
        
        # Set new path flags
        self._is_new_path = True
        self._initial_rotation_done = False
        
        # Calculate if initial rotation is needed
        if len(waypoints) > 0:
            current_pos, current_orientation = self.robot.get_position_orientation()
            current_yaw = T.quat2euler(current_orientation)[2]
            
            # Calculate desired heading to first waypoint
            first_waypoint = waypoints[0]
            dx = first_waypoint[0] - current_pos[0].item()
            dy = first_waypoint[1] - current_pos[1].item()
            desired_heading = np.arctan2(dy, dx)
            
            # Calculate heading error
            heading_error = desired_heading - current_yaw
            # Normalize to [-pi, pi]
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
            
            # Check if initial rotation is needed
            if abs(heading_error) > self.heading_threshold:
                self._needs_initial_rotation = True
                self._initial_rotation_target = desired_heading
                print(f"Initial rotation needed: {np.degrees(heading_error):.1f}° to target heading {np.degrees(desired_heading):.1f}°")
            else:
                self._needs_initial_rotation = False
                self._initial_rotation_done = True

    def update_target_waypoint(self, current_pos: th.Tensor) -> None:
        """Update current target waypoint based on robot position.
        
        Args:
            current_pos: Current robot position [x, y]
        """
        if not self.path or self.current_target_idx >= len(self.path):
            return

        # Check if we've reached the current waypoint
        while self.current_target_idx < len(self.path):
            target = self.path[self.current_target_idx]
            distance = np.sqrt(
                (current_pos[0].item() - target[0]) ** 2 + 
                (current_pos[1].item() - target[1]) ** 2
            )
            
            if distance < self.waypoint_threshold:
                self.current_target_idx += 1
                if self.current_target_idx < len(self.path):
                    next_target = self.path[self.current_target_idx]
                    print(f"Advanced to waypoint {self.current_target_idx}/{len(self.path)-1} ({next_target[0]}, {next_target[1]})")
            else:
                break

    def control(self) -> th.Tensor:
        """Compute control commands using Pure Pursuit algorithm.

        Returns:
            Robot action tensor with base control commands
        """
        if not self.path:
            print("[Warning] No path set for Pure Pursuit controller")
            return self.no_op()

        # Get current robot state
        current_pos, current_orientation = self.robot.get_position_orientation()
        current_yaw = T.quat2euler(current_orientation)[2]

        # Update target waypoint based on current position
        self.update_target_waypoint(current_pos)

        # Check arrival status and update flag
        self._check_and_update_arrival_flag(current_pos)

        # If arrived, return zero action
        if self._arrived_flag:
            return th.zeros(self.robot.action_dim)

        # Calculate control commands
        control_vx, control_vy, control_w = self._calculate_control_commands(
            current_pos, current_yaw
        )

        # Create action tensor
        action = th.zeros(self.robot.action_dim)
        action[self.base_start_idx : self.base_end_idx] = th.as_tensor(
            [control_vx, control_vy, control_w]
        )

        return action

    def _check_and_update_arrival_flag(self, current_pos: th.Tensor) -> None:
        """Check if robot has arrived and update arrival flag.
        
        Args:
            current_pos: Current robot position [x, y]
        """
        if not self.path:
            return
            
        # Check distance to final goal
        final_target = self.path[-1]
        dx = current_pos[0].item() - final_target[0]
        dy = current_pos[1].item() - final_target[1]
        distance_to_goal = (dx * dx + dy * dy) ** 0.5
        
        # Check arrival conditions
        distance_arrived = distance_to_goal < self.waypoint_threshold
        waypoints_completed = self.current_target_idx >= len(self.path)
        
        # Update arrival flag if any condition is met
        if distance_arrived or waypoints_completed:
            if not self._arrived_flag:
                # First time arrival - log it
                if waypoints_completed:
                    print(f"✓ Path completed: visited all {len(self.path)} waypoints")
                else:
                    print(f"✓ Arrived at goal: distance={distance_to_goal:.3f}m < threshold={self.waypoint_threshold}m")
                self._arrived_flag = True

    def no_op(self) -> th.Tensor:
        """Return no-operation (zero) action.
        
        This method is used when no path is available or when the controller
        should not move the robot (e.g., during path planning failures).
        
        Returns:
            Zero action tensor for the robot
        """
        return th.zeros(self.robot.action_dim)
    
    def has_valid_path(self) -> bool:
        """Check if the controller has a valid path set.
        
        Returns:
            True if a valid path is available, False otherwise
        """
        return self.path is not None and len(self.path) > 0

    def _calculate_control_commands(
        self, current_pos: th.Tensor, current_yaw: float
    ) -> Tuple[float, float, float]:
        """Calculate Pure Pursuit control commands.

        Args:
            current_pos: Current robot position [x, y].
            current_yaw: Current robot yaw angle in radians.

        Returns:
            Tuple of (vx, vy, angular_velocity) control commands.
        """
        if self.current_target_idx >= len(self.path):
            return 0.0, 0.0, 0.0

        # Find lookahead point
        lookahead_x, lookahead_y = self._find_lookahead_point(current_pos)

        # Transform lookahead point to robot frame
        dx_world = lookahead_x - current_pos[0].item()
        dy_world = lookahead_y - current_pos[1].item()
        
        # Priority check: Initial rotation for new paths
        if self._needs_initial_rotation and not self._initial_rotation_done:
            # Use the saved target heading instead of recalculating
            heading_error = self._initial_rotation_target - current_yaw
            
            # Normalize heading error to [-pi, pi]
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
            
            # Check if rotation is still needed
            if abs(heading_error) > self.heading_threshold:
                # In-place rotation mode
                control_vx = 0.0
                control_vy = 0.0
                # Use proportional control for angular velocity
                control_w = np.sign(heading_error) * min(self.max_angular_vel, abs(heading_error))
                
                return control_vx, control_vy, control_w
            else:
                # Rotation complete
                self._initial_rotation_done = True
                self._needs_initial_rotation = False

        # Normal Pure Pursuit mode
        # Transform to robot frame
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)
        dx_robot = cos_yaw * dx_world + sin_yaw * dy_world
        dy_robot = -sin_yaw * dx_world + cos_yaw * dy_world

        # Calculate curvature for Pure Pursuit
        lookahead_dist = np.sqrt(dx_robot**2 + dy_robot**2)
        if lookahead_dist < 1e-6:
            curvature = 0.0
        else:
            curvature = 2 * dy_robot / (lookahead_dist**2)

        # Calculate control commands
        control_vx = self.cruise_speed
        control_vy = 0.0
        control_w = self.cruise_speed * curvature

        # Apply angular velocity limits
        control_w = np.clip(control_w, -self.max_angular_vel, self.max_angular_vel)

        return control_vx, control_vy, control_w

    def _find_lookahead_point(self, current_pos: th.Tensor) -> Tuple[float, float]:
        """Find the lookahead point on the path.

        Args:
            current_pos: Current robot position [x, y]

        Returns:
            Lookahead point coordinates (x, y)
        """
        if self.current_target_idx >= len(self.path):
            return self.path[-1]

        # Start from current target waypoint
        for i in range(self.current_target_idx, len(self.path)):
            waypoint = self.path[i]
            distance = np.sqrt(
                (current_pos[0].item() - waypoint[0]) ** 2 + 
                (current_pos[1].item() - waypoint[1]) ** 2
            )
            
            if distance >= self.lookahead_distance:
                return waypoint

        # If no waypoint is far enough, use the last waypoint
        return self.path[-1]

    def reset_arrival_state(self) -> None:
        """Reset arrival state for reuse."""
        # Reset arrival flag
        self._arrived_flag = False
        
        # Reset initial rotation state
        self._is_new_path = False
        self._initial_rotation_done = False
        self._needs_initial_rotation = False
        self._initial_rotation_target = None
        
    def is_arrived(self) -> bool:
        """Check if the controller has reached the target."""
        return self._arrived_flag