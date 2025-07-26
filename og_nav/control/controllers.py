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

from og_nav.core.constants import TIAGO_BASE_ACTION_START_IDX, TIAGO_BASE_ACTION_END_IDX
from og_nav.control.arrival_state import ArrivalState

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

        # Store components for debugging
        self.last_p = p_term
        self.last_i = i_term
        self.last_d = d_term

        # Calculate output
        output = p_term + i_term + d_term

        # Apply output limits
        if self.output_limits is not None:
            output = max(self.output_limits[0], min(self.output_limits[1], output))

        self.last_error = error
        return output

    def get_last_components(self) -> Tuple[float, float, float]:
        """Get the last computed P, I, D components.

        Returns:
            Tuple of (P, I, D) component values
        """
        return self.last_p, self.last_i, self.last_d

    def reset(self) -> None:
        """Reset PID controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0
        print("PID Controller reset")


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
            'waypoint_threshold': 0.2
        }

    def __init__(
        self,
        robot: Tiago,
        lookahead_distance: Optional[float] = None,
        cruise_speed: Optional[float] = None,
        max_angular_vel: Optional[float] = None,
        waypoint_threshold: Optional[float] = None,
        dt: Optional[float] = None,
        config: Optional[dict] = None,
    ) -> None:
        """Initialize Pure Pursuit controller.

        Priority order for parameters:
        1. Constructor arguments (highest priority)
        2. config dict values
        3. Default values (lowest priority)

        Args:
            robot: Tiago robot instance
            lookahead_distance: Distance to lookahead point on path
            cruise_speed: Constant forward speed
            max_angular_vel: Maximum angular velocity
            waypoint_threshold: Distance threshold for waypoint arrival
            dt: Control time step
            config: Controller configuration dict
        """
        self.robot = robot
        
        # Merge config with defaults following priority order
        default_config = self.get_default_cfg()
        merged_config = default_config.copy()
        if config is not None:
            merged_config.update(config)
        
        # Apply constructor arguments (highest priority)
        self.lookahead_distance = lookahead_distance if lookahead_distance is not None else merged_config['lookahead_distance']
        self.cruise_speed = cruise_speed if cruise_speed is not None else merged_config['cruise_speed']
        self.max_angular_vel = max_angular_vel if max_angular_vel is not None else merged_config['max_angular_vel']
        self.waypoint_threshold = waypoint_threshold if waypoint_threshold is not None else merged_config['waypoint_threshold']
        
        self.dt = dt if dt is not None else og.sim.get_sim_step_dt()
        
        # Store final config for reference
        self.config = {
            'lookahead_distance': self.lookahead_distance,
            'cruise_speed': self.cruise_speed,
            'max_angular_vel': self.max_angular_vel,
            'waypoint_threshold': self.waypoint_threshold
        }
        
        # Path and tracking state
        self.path: List[Tuple[float, float]] = []
        self.current_target_idx = 0
        
        # Centralized arrival state management
        self._arrival_state = ArrivalState()
        
        # Robot base action indices
        self.base_start_idx, self.base_end_idx = (
            TIAGO_BASE_ACTION_START_IDX,
            TIAGO_BASE_ACTION_END_IDX,
        )

        # Data logging for analysis and visualization
        self.data_logger: Dict[str, List] = {
            key: []
            for key in [
                "time",
                "x",
                "y",
                "theta",
                "target_x",
                "target_y",
                "lookahead_x",
                "lookahead_y",
                "error_x",
                "error_y",
                "control_vx",
                "control_vy",
                "control_w",
                "curvature",
            ]
        }

        print(
            f"Pure Pursuit Controller initialized: lookahead={lookahead_distance}m, speed={cruise_speed}m/s"
        )

    def set_path(
        self, waypoints: Union[List[Tuple[float, float]], List[th.Tensor]]
    ) -> None:
        """Set new path for tracking.

        Args:
            waypoints: List of (x, y) waypoints
        """
        # if waypoints is a list of tensors, convert to list of tuples
        if isinstance(waypoints[0], th.Tensor):
            waypoints = [tuple(waypoint.tolist()) for waypoint in waypoints]
        self.path = waypoints.copy()
        self.current_target_idx = 0
        
        # Reset arrival state for new path
        self._arrival_state.reset()

        # Clear previous logging data
        for key in self.data_logger:
            self.data_logger[key].clear()

        print(f"New path set with {len(waypoints)} waypoints")

    def find_lookahead_point(self, current_pos: th.Tensor) -> Tuple[float, float]:
        """Find lookahead point on the path.

        Args:
            current_pos: Current robot position [x, y].

        Returns:
            Lookahead point (x, y) coordinates.
        """
        if not self.path:
            return (current_pos[0].item(), current_pos[1].item())

        robot_pos = np.array([current_pos[0].item(), current_pos[1].item()])

        # Search for the lookahead point starting from current target
        for i in range(self.current_target_idx, len(self.path) - 1):
            # Get path segment
            segment_start = np.array(self.path[i])
            segment_end = np.array(self.path[i + 1])
            segment_vec = segment_end - segment_start

            # Project robot position onto path segment
            segment_length_sq = np.dot(segment_vec, segment_vec)
            if segment_length_sq > 1e-12:  # Avoid division by zero
                robot_to_start = robot_pos - segment_start
                t = np.clip(
                    np.dot(robot_to_start, segment_vec) / segment_length_sq, 0, 1
                )
                closest_point = segment_start + t * segment_vec

                # Check if we've found a point at lookahead distance
                dist_to_closest = np.linalg.norm(closest_point - robot_pos)

                # If this segment contains the lookahead point
                if dist_to_closest <= self.lookahead_distance:
                    remaining_dist = self.lookahead_distance - dist_to_closest

                    # Search forward from closest point
                    for j in range(i, len(self.path) - 1):
                        seg_start = np.array(self.path[j])
                        seg_end = np.array(self.path[j + 1])
                        seg_vec = seg_end - seg_start
                        seg_length = np.linalg.norm(seg_vec)

                        if seg_length > 1e-6:
                            if remaining_dist <= seg_length:
                                # Found lookahead point on this segment
                                direction = seg_vec / seg_length
                                lookahead_point = seg_start + remaining_dist * direction
                                return tuple(lookahead_point)
                            remaining_dist -= seg_length

        # Fallback: return last waypoint
        return self.path[-1]

    def update_target_waypoint(self, current_pos: th.Tensor) -> None:
        """Update current target waypoint index based on robot position.

        Args:
            current_pos: Current robot position [x, y]
        """
        if not self.path or self.current_target_idx >= len(self.path):
            return

        robot_x, robot_y = current_pos[0].item(), current_pos[1].item()

        # Check if we're close enough to current target to advance
        while self.current_target_idx < len(self.path):
            target_x, target_y = self.path[self.current_target_idx]
            distance_to_target = np.sqrt(
                (robot_x - target_x) ** 2 + (robot_y - target_y) ** 2
            )

            if distance_to_target < self.waypoint_threshold:
                print(
                    f"Advanced to waypoint {self.current_target_idx + 1}/{len(self.path)} {self.path[self.current_target_idx]}"
                )
                self.current_target_idx += 1
            else:
                break

    def control(self) -> th.Tensor:
        """Compute control commands using Pure Pursuit algorithm.

        Returns:
            Robot action tensor with base control commands
        """
        if not self.path:
            print("[Warning] No path set for Pure Pursuit controller")
            return th.zeros(self.robot.action_dim)

        # Get current robot state
        current_pos, current_orientation = self.robot.get_position_orientation()
        current_yaw = T.quat2euler(current_orientation)[2]

        # Update target waypoint based on current position
        self.update_target_waypoint(current_pos)

        # Update arrival state
        self._arrival_state.update(
            current_pos, self.path, self.waypoint_threshold, self.current_target_idx
        )
        
        # If arrived, return zero action
        if self._arrival_state.is_arrived():
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
        # Find lookahead point
        lookahead_x, lookahead_y = self.find_lookahead_point(current_pos)

        # Calculate relative position to lookahead point in world frame
        dx_world = lookahead_x - current_pos[0].item()
        dy_world = lookahead_y - current_pos[1].item()

        # Transform to robot body frame
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)

        x_local = cos_yaw * dx_world + sin_yaw * dy_world
        y_local = -sin_yaw * dx_world + cos_yaw * dy_world

        # Pure Pursuit algorithm: calculate curvature
        L_squared = self.lookahead_distance**2
        curvature = 2 * y_local / L_squared if L_squared > 1e-6 else 0.0

        # Calculate control outputs
        control_vx = self.cruise_speed
        control_vy = 0.0  # Pure pursuit typically doesn't use lateral velocity
        control_w = self.cruise_speed * curvature

        # Apply angular velocity limits
        control_w = max(-self.max_angular_vel, min(self.max_angular_vel, control_w))

        # Log data for analysis
        self._log_control_data(
            current_pos,
            current_yaw,
            lookahead_x,
            lookahead_y,
            dx_world,
            dy_world,
            control_vx,
            control_vy,
            control_w,
            curvature,
        )

        # Periodic debug output
        # if len(self.data_logger["time"]) % 50 == 0:
        #     self._log_debug_info(
        #         current_pos, lookahead_x, lookahead_y, dx_world, dy_world,
        #         current_yaw, control_vx, control_vy, control_w, curvature
        #     )

        return control_vx, control_vy, control_w


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
        """Log control data for analysis and visualization."""
        self.data_logger["time"].append(len(self.data_logger["time"]) * self.dt)
        self.data_logger["x"].append(current_pos[0].item())
        self.data_logger["y"].append(current_pos[1].item())
        self.data_logger["theta"].append(current_yaw)
        self.data_logger["lookahead_x"].append(lookahead_x)
        self.data_logger["lookahead_y"].append(lookahead_y)
        self.data_logger["error_x"].append(error_x)
        self.data_logger["error_y"].append(error_y)
        self.data_logger["control_vx"].append(control_vx)
        self.data_logger["control_vy"].append(control_vy)
        self.data_logger["control_w"].append(control_w)
        self.data_logger["curvature"].append(curvature)

        # For compatibility, add target position (using lookahead point)
        self.data_logger["target_x"].append(lookahead_x)
        self.data_logger["target_y"].append(lookahead_y)

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
        """Log detailed debug information."""
        step_count = len(self.data_logger["time"])
        print(f"Pure Pursuit Control Details (Step {step_count}):")
        print(f"  Position: [{current_pos[0]:.4f}, {current_pos[1]:.4f}]")
        print(f"  Lookahead Point: [{lookahead_x:.4f}, {lookahead_y:.4f}]")
        print(f"  Error: x={error_x:.4f}, y={error_y:.4f}")
        print(f"  Current orientation: {np.degrees(current_yaw):.2f}°")
        print(f"  Control: vx={control_vx:.4f}, vy={control_vy:.4f}, w={control_w:.4f}")
        print(f"  Curvature: {curvature:.4f}")
        print(f"  Target waypoint: {self.current_target_idx}/{len(self.path)}")

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Generate and save visualization plots of Pure Pursuit tracking results.

        Args:
            save_path: Path to save the plot. If None, generates automatic filename.
        """
        if not self.data_logger["time"]:
            print("[Warning] No tracking data available for plotting")
            return

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"pure_pursuit_results_{timestamp}.png"

        try:
            self._create_tracking_plots(save_path)
            print(f"Pure Pursuit tracking results saved to: {save_path}")
        except Exception as e:
            print(f"[Error] Error creating tracking plots: {e}")

    def _create_tracking_plots(self, save_path: str) -> None:
        """Create comprehensive Pure Pursuit tracking visualization plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Pure Pursuit Path Tracking Results", fontsize=16)

        # Trajectory plot
        self._plot_trajectory(axes[0, 0])

        # Position errors
        self._plot_position_errors(axes[0, 1])

        # Curvature plot
        self._plot_curvature(axes[0, 2])

        # Linear control outputs
        self._plot_linear_control(axes[1, 0])

        # Angular control output
        self._plot_angular_control(axes[1, 1])

        # Lookahead points
        self._plot_lookahead_points(axes[1, 2])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_trajectory(self, ax) -> None:
        """Plot robot trajectory and path."""
        ax.plot(
            self.data_logger["x"],
            self.data_logger["y"],
            "b-",
            label="Robot Path",
            linewidth=2,
        )

        # Plot original path waypoints
        if self.path:
            path_x = [p[0] for p in self.path]
            path_y = [p[1] for p in self.path]
            ax.plot(
                path_x, path_y, "r--", label="Reference Path", linewidth=2, alpha=0.7
            )
            ax.scatter(path_x, path_y, color="red", s=30, alpha=0.7)

        ax.scatter(
            self.data_logger["x"][0],
            self.data_logger["y"][0],
            color="green",
            s=100,
            label="Start",
        )
        ax.scatter(
            self.data_logger["x"][-1],
            self.data_logger["y"][-1],
            color="red",
            s=100,
            label="End",
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Robot Trajectory vs Reference Path")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")

    def _plot_position_errors(self, ax) -> None:
        """Plot position tracking errors."""
        ax.plot(
            self.data_logger["time"], self.data_logger["error_x"], "r-", label="Error X"
        )
        ax.plot(
            self.data_logger["time"], self.data_logger["error_y"], "g-", label="Error Y"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position Error (m)")
        ax.set_title("Position Tracking Error")
        ax.legend()
        ax.grid(True)

    def _plot_curvature(self, ax) -> None:
        """Plot curvature over time."""
        ax.plot(
            self.data_logger["time"],
            self.data_logger["curvature"],
            "m-",
            label="Curvature",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Curvature (1/m)")
        ax.set_title("Path Curvature")
        ax.legend()
        ax.grid(True)

    def _plot_linear_control(self, ax) -> None:
        """Plot linear control outputs."""
        ax.plot(
            self.data_logger["time"],
            self.data_logger["control_vx"],
            "r-",
            label="Control Vx",
        )
        ax.plot(
            self.data_logger["time"],
            self.data_logger["control_vy"],
            "g-",
            label="Control Vy",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Linear Velocity (m/s)")
        ax.set_title("Linear Control Output")
        ax.legend()
        ax.grid(True)

    def _plot_angular_control(self, ax) -> None:
        """Plot angular control output."""
        ax.plot(
            self.data_logger["time"],
            self.data_logger["control_w"],
            "b-",
            label="Control W",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angular Velocity (rad/s)")
        ax.set_title("Angular Control Output")
        ax.legend()
        ax.grid(True)

    def _plot_lookahead_points(self, ax) -> None:
        """Plot lookahead points trajectory."""
        ax.plot(
            self.data_logger["lookahead_x"],
            self.data_logger["lookahead_y"],
            "c-",
            label="Lookahead Points",
            linewidth=2,
            alpha=0.7,
        )
        ax.plot(
            self.data_logger["x"],
            self.data_logger["y"],
            "b-",
            label="Robot Path",
            linewidth=1,
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Lookahead Points")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")

    def reset_arrival_state(self) -> None:
        """Reset arrival state for reuse."""
        self._arrival_state.reset()
        
    def is_arrived(self) -> bool:
        """Check if the controller has reached the target."""
        return self._arrival_state.is_arrived()
