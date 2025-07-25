"""Control algorithms for robot navigation."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.robots.tiago import Tiago

from ..core.constants import (
    DEFAULT_CRUISE_SPEED,
    DEFAULT_LOOKAHEAD_DISTANCE,
    DEFAULT_MAX_ANGULAR_VEL,
    DEFAULT_WAYPOINT_THRESHOLD,
)

# Configure matplotlib for non-interactive backend
matplotlib.use("Agg")


class PIDController:
    """Generic PID controller implementation.
    
    This controller implements proportional-integral-derivative control
    with optional output limits and debug information tracking.
    
    Attributes:
        kp (float): Proportional gain coefficient.
        ki (float): Integral gain coefficient.
        kd (float): Derivative gain coefficient.
        output_limits (Optional[Tuple[float, float]]): Min/max output limits.
        integral (float): Accumulated integral term.
        last_error (float): Previous error value for derivative calculation.
    """
    
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limits: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: Output limits (min, max), None for unlimited
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        # Control state
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        
        # Debug information
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0
        
        print(f"PID Controller initialized: Kp={kp}, Ki={ki}, Kd={kd}")
    
    def update(self, error: float, dt: float) -> float:
        """Update PID controller and compute control output.
        
        Args:
            error: Current error value
            dt: Time step
            
        Returns:
            Control output value
        """
        if dt <= 0:
            print("[Warning] Invalid time step: {dt}, using minimal value")
            dt = 1e-6
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        if self.last_error is not None:
            derivative = (error - self.last_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits if specified
        if self.output_limits is not None:
            min_limit, max_limit = self.output_limits
            output = max(min_limit, min(max_limit, output))
        
        # Store values for next iteration
        self.last_error = error
        self.last_p = p_term
        self.last_i = i_term
        self.last_d = d_term
        
        return output
    
    def get_last_components(self) -> Tuple[float, float, float]:
        """Get the last P, I, D component values.
        
        Returns:
            Tuple of (P, I, D) components
        """
        return (self.last_p, self.last_i, self.last_d)
    
    def reset(self) -> None:
        """Reset the controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0


class PathTrackingController:
    """Path tracking controller using Pure Pursuit algorithm with PID heading control.
    
    This controller implements a Pure Pursuit path following algorithm combined
    with PID control for heading adjustment. It tracks a sequence of waypoints
    and generates appropriate velocity commands for the robot.
    """
    
    def __init__(
        self,
        robot: Tiago,
        lookahead_distance: float = DEFAULT_LOOKAHEAD_DISTANCE,
        cruise_speed: float = DEFAULT_CRUISE_SPEED,
        max_angular_vel: float = DEFAULT_MAX_ANGULAR_VEL,
        waypoint_threshold: float = DEFAULT_WAYPOINT_THRESHOLD,
        dt: Optional[float] = None,
    ) -> None:
        """Initialize path tracking controller.
        
        Args:
            robot: Tiago robot instance
            lookahead_distance: Lookahead distance for Pure Pursuit
            cruise_speed: Desired forward speed
            max_angular_vel: Maximum angular velocity
            waypoint_threshold: Distance threshold to consider waypoint reached
            dt: Control loop time step (None to use robot control frequency)
        """
        self.robot = robot
        self.lookahead_distance = lookahead_distance
        self.cruise_speed = cruise_speed
        self.max_angular_vel = max_angular_vel
        self.waypoint_threshold = waypoint_threshold
        
        # Control timing
        self.dt = dt if dt is not None else (1.0 / robot.control_freq)
        
        # Waypoint tracking
        self.waypoints: List[th.Tensor] = []
        self.current_waypoint_idx = 0
        self.target_waypoint: Optional[Tuple[float, float]] = None
        
        # Control components
        self.heading_pid = PIDController(
            kp=2.0, ki=0.0, kd=0.0, output_limits=(-max_angular_vel, max_angular_vel)
        )
        
        # Debug and logging
        self.control_log: List[Dict] = []
        self.debug_mode = False
        
        # Initialize base action indices
        self.base_action_indices = self._get_base_action_indices()
        
        print(f"PathTrackingController initialized with dt={self.dt:.3f}s")
    
    def _get_base_action_indices(self) -> List[int]:
        """Get indices for base joints in the action space.
        
        Returns:
            List of base joint indices
        """
        # Get base joint names from robot
        base_joint_names = [
            "base_footprint_x_joint",
            "base_footprint_y_joint", 
            "base_footprint_rz_joint"
        ]
        
        # Map to dof indices
        joint_names_list = list(self.robot.joints.keys())
        try:
            base_dof_indices = [
                joint_names_list.index(name) for name in base_joint_names
            ]
            return base_dof_indices
        except ValueError as e:
            print(f"[Warning] Could not find base joints: {e}")
            # Fallback to default indices
            return [0, 1, 2]
    
    def set_path(self, waypoints: Union[List[Tuple[float, float]], List[th.Tensor]]) -> None:
        """Set the path to follow.
        
        Args:
            waypoints: List of (x, y) waypoints or tensor waypoints
        """
        # Convert waypoints to tensor format
        self.waypoints = []
        for waypoint in waypoints:
            if isinstance(waypoint, tuple):
                self.waypoints.append(th.tensor([waypoint[0], waypoint[1]]))
            else:
                self.waypoints.append(waypoint)
        
        self.current_waypoint_idx = 0
        if self.waypoints:
            self.target_waypoint = (
                self.waypoints[0][0].item(),
                self.waypoints[0][1].item(),
            )
        else:
            self.target_waypoint = None
            
        # Reset PID controller
        self.heading_pid.reset()
        
        print(f"Path set with {len(self.waypoints)} waypoints")
    
    def find_lookahead_point(self, current_pos: th.Tensor) -> Tuple[float, float]:
        """Find the lookahead point on the path.
        
        Args:
            current_pos: Current robot position (x, y)
            
        Returns:
            Lookahead point coordinates (x, y)
        """
        if not self.waypoints:
            return (current_pos[0].item(), current_pos[1].item())
        
        # Find the first waypoint that is at least lookahead_distance away
        for waypoint in self.waypoints[self.current_waypoint_idx:]:
            distance = th.norm(waypoint - current_pos[:2])
            if distance >= self.lookahead_distance:
                return (waypoint[0].item(), waypoint[1].item())
        
        # If no waypoint is far enough, use the last waypoint
        last_waypoint = self.waypoints[-1]
        return (last_waypoint[0].item(), last_waypoint[1].item())
    
    def update_target_waypoint(self, current_pos: th.Tensor) -> None:
        """Update the target waypoint based on current position.
        
        Args:
            current_pos: Current robot position (x, y)
        """
        if not self.waypoints:
            return
            
        # Check if we've reached the current target waypoint
        if self.target_waypoint is not None:
            target_tensor = th.tensor([self.target_waypoint[0], self.target_waypoint[1]])
            distance = th.norm(target_tensor - current_pos[:2])
            
            if distance < self.waypoint_threshold:
                # Move to next waypoint
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx < len(self.waypoints):
                    next_waypoint = self.waypoints[self.current_waypoint_idx]
                    self.target_waypoint = (
                        next_waypoint[0].item(),
                        next_waypoint[1].item(),
                    )
                    print(f"Reached waypoint {self.current_waypoint_idx-1}, moving to {self.current_waypoint_idx}")
                else:
                    # Reached end of path
                    self.target_waypoint = None
    
    def control(self, return_action: bool = True) -> Union[Tuple[th.Tensor, bool], bool]:
        """Compute control commands to follow the path.
        
        Args:
            return_action: If True, return action tensor and arrived flag. 
                          If False, only return arrived flag.
            
        Returns:
            If return_action is True: (action_tensor, arrived)
            If return_action is False: arrived
        """
        # Get current robot state
        current_pos, current_orn = self.robot.get_position_orientation()
        current_yaw = T.euler_from_quat(current_orn)[2]  # Extract yaw from quaternion
        
        # Update target waypoint
        self.update_target_waypoint(current_pos)
        
        # Check if we've arrived at the final destination
        arrived = self.target_waypoint is None
        
        if arrived:
            if return_action:
                # Create zero action
                action = th.zeros(self.robot.action_dim)
                return action, True
            else:
                return True
        
        # Find lookahead point
        lookahead_x, lookahead_y = self.find_lookahead_point(current_pos)
        
        # Calculate errors
        error_x = lookahead_x - current_pos[0].item()
        error_y = lookahead_y - current_pos[1].item()
        
        # Calculate desired heading
        desired_heading = np.arctan2(error_y, error_x)
        
        # Calculate heading error (normalize to [-π, π])
        heading_error = desired_heading - current_yaw
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Use PID controller for angular velocity
        angular_vel = self.heading_pid.update(heading_error, self.dt)
        
        # Calculate linear velocity (reduce when approaching waypoints)
        distance_to_target = np.sqrt(error_x**2 + error_y**2)
        linear_vel = self.cruise_speed
        if distance_to_target < 1.0:  # Slow down when close
            linear_vel *= max(0.1, distance_to_target)
        
        # Convert to robot action space
        if return_action:
            # Dynamically get current action dimension
            current_action_dim = self.robot.action_dim
            action = th.zeros(current_action_dim)
            
            # Set base velocities
            # Get current base action indices
            current_base_indices = self._get_base_action_indices()
            
            if len(current_base_indices) >= 3:
                action[current_base_indices[0]] = linear_vel  # x velocity
                action[current_base_indices[1]] = 0.0         # y velocity (0 for differential drive)
                action[current_base_indices[2]] = angular_vel # angular velocity
            
            # Log control data if in debug mode
            if self.debug_mode:
                self._log_control_data(
                    current_pos, current_yaw, lookahead_x, lookahead_y,
                    error_x, error_y, linear_vel, 0.0, angular_vel, 0.0
                )
            
            return action, False
        else:
            return False
    
    def _calculate_control_commands(
        self, current_pos: th.Tensor, current_yaw: float
    ) -> Tuple[float, float, float]:
        """Calculate control commands (for future extension).
        
        Args:
            current_pos: Current robot position
            current_yaw: Current robot yaw
            
        Returns:
            Tuple of (vx, vy, w) control commands
        """
        # This is a placeholder for more complex control algorithms
        return (0.0, 0.0, 0.0)
    
    def _check_arrival_condition(self, current_pos: th.Tensor) -> bool:
        """Check if robot has arrived at final destination.
        
        Args:
            current_pos: Current robot position
            
        Returns:
            True if arrived, False otherwise
        """
        if not self.waypoints:
            return True
            
        final_waypoint = self.waypoints[-1]
        distance = th.norm(final_waypoint - current_pos[:2])
        return distance < self.waypoint_threshold
    
    def _log_control_data(
        self,
        current_pos: th.Tensor,
        current_yaw: float,
        lookahead_x: float,
        lookahead_y: float,
        error_x: float,
        error_y: float,
        control_vx: float,
        control_vy: float,
        control_w: float,
        curvature: float,
    ) -> None:
        """Log control data for debugging and analysis.
        
        Args:
            current_pos: Current robot position
            current_yaw: Current robot yaw
            lookahead_x: Lookahead point x coordinate
            lookahead_y: Lookahead point y coordinate
            error_x: X position error
            error_y: Y position error
            control_vx: X velocity command
            control_vy: Y velocity command
            control_w: Angular velocity command
            curvature: Path curvature
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "position": current_pos.tolist(),
            "yaw": current_yaw,
            "lookahead": [lookahead_x, lookahead_y],
            "error": [error_x, error_y],
            "control": [control_vx, control_vy, control_w],
            "curvature": curvature,
        }
        self.control_log.append(log_entry)
    
    def _log_debug_info(
        self,
        current_pos: th.Tensor,
        lookahead_x: float,
        lookahead_y: float,
        error_x: float,
        error_y: float,
        current_yaw: float,
        control_vx: float,
        control_vy: float,
        control_w: float,
        curvature: float,
    ) -> None:
        """Log debug information.
        
        Args:
            current_pos: Current robot position
            lookahead_x: Lookahead point x coordinate
            lookahead_y: Lookahead point y coordinate
            error_x: X position error
            error_y: Y position error
            current_yaw: Current robot yaw
            control_vx: X velocity command
            control_vy: Y velocity command
            control_w: Angular velocity command
            curvature: Path curvature
        """
        if self.debug_mode:
            print(f"[DEBUG] Pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f}), "
                  f"Yaw: {current_yaw:.2f}, "
                  f"Lookahead: ({lookahead_x:.2f}, {lookahead_y:.2f}), "
                  f"Error: ({error_x:.2f}, {error_y:.2f}), "
                  f"Control: (vx={control_vx:.2f}, vy={control_vy:.2f}, w={control_w:.2f})")
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Plot control results for analysis.
        
        Args:
            save_path: Path to save plot, None to display
        """
        if not self.control_log:
            print("[Warning] No control data to plot")
            return
            
        self._create_tracking_plots(save_path)
    
    def _create_tracking_plots(self, save_path: str) -> None:
        """Create tracking performance plots.
        
        Args:
            save_path: Path to save plots
        """
        # Extract data
        timestamps = [entry["timestamp"] for entry in self.control_log]
        positions = [entry["position"] for entry in self.control_log]
        yaws = [entry["yaw"] for entry in self.control_log]
        controls = [entry["control"] for entry in self.control_log]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Path Tracking Control Results")
        
        # Plot trajectory
        self._plot_trajectory(axes[0, 0])
        
        # Plot position errors
        self._plot_position_errors(axes[0, 1])
        
        # Plot control commands
        self._plot_linear_control(axes[1, 0])
        self._plot_angular_control(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
    
    def _plot_trajectory(self, ax) -> None:
        """Plot robot trajectory.
        
        Args:
            ax: Matplotlib axis
        """
        if not self.control_log:
            return
            
        positions = [entry["position"] for entry in self.control_log]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        ax.plot(x_coords, y_coords, 'b-', linewidth=2, label='Trajectory')
        ax.scatter(x_coords[0], y_coords[0], color='green', s=100, label='Start')
        ax.scatter(x_coords[-1], y_coords[-1], color='red', s=100, label='End')
        
        # Plot waypoints if available
        if self.waypoints:
            wp_x = [wp[0].item() for wp in self.waypoints]
            wp_y = [wp[1].item() for wp in self.waypoints]
            ax.scatter(wp_x, wp_y, color='orange', s=50, label='Waypoints')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Robot Trajectory')
        ax.legend()
        ax.grid(True)
    
    def _plot_position_errors(self, ax) -> None:
        """Plot position errors.
        
        Args:
            ax: Matplotlib axis
        """
        if not self.control_log:
            return
            
        errors = [entry["error"] for entry in self.control_log]
        error_x = [err[0] for err in errors]
        error_y = [err[1] for err in errors]
        
        time_points = range(len(errors))
        ax.plot(time_points, error_x, 'r-', label='X Error')
        ax.plot(time_points, error_y, 'b-', label='Y Error')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Position Error (m)')
        ax.set_title('Position Tracking Errors')
        ax.legend()
        ax.grid(True)
    
    def _plot_curvature(self, ax) -> None:
        """Plot path curvature.
        
        Args:
            ax: Matplotlib axis
        """
        if not self.control_log:
            return
            
        curvatures = [entry["curvature"] for entry in self.control_log]
        time_points = range(len(curvatures))
        
        ax.plot(time_points, curvatures, 'g-', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Curvature (1/m)')
        ax.set_title('Path Curvature')
        ax.grid(True)
    
    def _plot_linear_control(self, ax) -> None:
        """Plot linear control commands.
        
        Args:
            ax: Matplotlib axis
        """
        if not self.control_log:
            return
            
        controls = [entry["control"] for entry in self.control_log]
        vx = [ctrl[0] for ctrl in controls]
        vy = [ctrl[1] for ctrl in controls]
        
        time_points = range(len(controls))
        ax.plot(time_points, vx, 'r-', label='Vx')
        ax.plot(time_points, vy, 'b-', label='Vy')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Linear Velocity (m/s)')
        ax.set_title('Linear Control Commands')
        ax.legend()
        ax.grid(True)
    
    def _plot_angular_control(self, ax) -> None:
        """Plot angular control commands.
        
        Args:
            ax: Matplotlib axis
        """
        if not self.control_log:
            return
            
        controls = [entry["control"] for entry in self.control_log]
        w = [ctrl[2] for ctrl in controls]
        
        time_points = range(len(controls))
        ax.plot(time_points, w, 'g-', linewidth=2)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('Angular Control Commands')
        ax.grid(True)
    
    def _plot_lookahead_points(self, ax) -> None:
        """Plot lookahead points.
        
        Args:
            ax: Matplotlib axis
        """
        if not self.control_log:
            return
            
        lookaheads = [entry["lookahead"] for entry in self.control_log]
        lx = [point[0] for point in lookaheads]
        ly = [point[1] for point in lookaheads]
        
        ax.plot(lx, ly, 'm--', linewidth=1, label='Lookahead Points')
        ax.set_title('Lookahead Points')
        ax.legend()
        ax.grid(True) 