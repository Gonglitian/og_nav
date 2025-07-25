"""
OmniGibson Path Planning Test Module

This module provides interactive path planning functionality for the Tiago robot
in OmniGibson environment using keyboard controls and visual markers.

Key Features:
- Interactive start/goal point setting
- Visual path planning with waypoint markers
- Keyboard-based controls for testing
- Real-time visualization

Keyboard Controls:
- Z: Set start point (green sphere)
- X: Set goal point (red sphere)
- C: Clear all markers and reset state
- V: Plan path (blue sphere waypoints)
"""

from typing import Dict, Any, Tuple

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.robots.tiago import Tiago
from ..planning import PathPlanner
from ..control import PathTrackingController
from ..mapping import OGMGenerator
from ..control.robot_control import (
    set_arm_jointcontroller,
    get_original_arm_controllers,
)
from omnigibson.utils.ui_utils import KeyboardEventHandler
from ..core.constants import NavConstants
import omnigibson.lazy as lazy
import torch as th

# Configuration constants


config = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Pomaria_1_int",
            "load_object_categories": ["floors", "walls"],
            # "default_erosion_radius": Config.DEFAULT_EROSION_RADIUS,
        },
        "robots": [
            {
                "type": "Tiago",
                "position": NavConstants.START_POSITION,
                "reset_joint_pos": NavConstants.RESET_POSE,
                # "rigid_trunk": True,
                "controller_config":{
            "arm_left":{
                "name": "JointController",
                "motor_type": "position", 
                "use_delta_commands": False,
                "use_impedances": True,
                "command_input_limits": None,
                # "command_output_limits": None,
            },
            "arm_right":{
                "name": "JointController", 
                "motor_type": "position",
                "use_delta_commands": False,
                "use_impedances": True,
                "command_input_limits": None,
                # "command_output_limits": None,
            }
        },
            }
        ],
        "objects": [
            {
                "type": "PrimitiveObject",
                "name": "start_point",
                "position": NavConstants.HIDDEN_POSITION,
                "primitive_type": "Sphere",
                "visual_only": True,
                "radius": NavConstants.MARKER_RADIUS,
                "rgba": NavConstants.START_MARKER_COLOR,
            },
            {
                "type": "PrimitiveObject",
                "name": "end_point",
                "position": NavConstants.HIDDEN_POSITION,
                "primitive_type": "Sphere",
                "visual_only": True,
                "radius": NavConstants.MARKER_RADIUS,
                "rgba": NavConstants.GOAL_MARKER_COLOR,
            },
        ],
    }

# Add waypoint markers
for i in range(NavConstants.N_WAYPOINTS):
    config["objects"].append(
        {
            "type": "PrimitiveObject",
            "name": f"waypoint_{i}",
            "position": NavConstants.HIDDEN_POSITION,
            "primitive_type": "Sphere",
            "radius": NavConstants.WAYPOINT_RADIUS,
            "visual_only": True,
            "rgba": NavConstants.WAYPOINT_COLOR,
        }
    )


def main() -> None:
    """Main execution function for path planning test."""
    # Apply GUI settings
    gm.GUI_VIEWPORT_ONLY = NavConstants.GUI_VIEWPORT_ONLY

    # Initialize environment
    print("Initializing OmniGibson environment...")
    env = og.Environment(configs=config)
    camera = og.sim.enable_viewer_camera_teleoperation()
    # Get robot instance
    robot: Tiago = env.robots[0]
    # OGM
    gen = OGMGenerator(resolution=NavConstants.OG_MAP_RESOLUTION)
    gen.update_env_trav_map(env)
    # Create path planner with visualization enabled and optional robot
    path_planner = PathPlanner(env, visualize=True, robot=robot)
    path_planner.setup_keyboard_callbacks()
    # Plan path
    path = path_planner.plan_path(NavConstants.START_POSITION[:2], NavConstants.GOAL_POSITION[:2])
    # Create path tracking controller
    controller = PathTrackingController(robot=robot)
    controller.set_path(path)

    def goto(
        path_planner: PathPlanner,
        controller: PathTrackingController,
        goal_position: Tuple[float, float],
    ):
        # let robotgo to the given goal
        # 1. get path from current robotposition to goal
        path_planner.set_start_point(controller.robot.get_position_orientation()[0][:2])
        path_planner.set_goal_point(goal_position)
        path = path_planner.plan_path()
        # 2. set path to controller
        controller.set_path(path)

    # setup keybind of goto function
    KeyboardEventHandler.initialize()
    # Get camera for position detection
    # P: Set goal point
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.P,
        callback_fn=lambda: goto(
            path_planner, controller, camera.cam.get_position_orientation()[0][:2]
        ),
    )
    # Main loop
    # ik_controller_obj = get_original_arm_controllers(robot)
    # set_arm_jointcontroller(robot)
    
    # remove robot arm controller limits
    robot.controllers["arm_left"]._command_input_limits = None
    robot.controllers["arm_right"]._command_input_limits = None
    step = 0
    while True:
        # update aabb extent
        if step % 100 == 0:
            robot._reset_joint_pos_aabb_extent = robot.get_base_aligned_bbox(xy_aligned=False)[2]
        action, arrived = controller.control(return_action=True)
        # head
        action[3:5] = th.tensor([0, 0])
        # left arm
        action[6:13] = NavConstants.NAV_ARM_POSE
        # right arm
        action[14:21] = NavConstants.NAV_ARM_POSE
        env.step(action)
        step += 1
        
if __name__ == "__main__":
    main()
