"""
Simple Environment Demo

This demo shows how to use NavigationInterface with a simple environment
for basic robot navigation testing.
"""

import omnigibson as og
from omnigibson.macros import gm
import torch as th
from omnigibson.robots.tiago import Tiago
from ..control.robot_control import print_joint_info
from ..core import NavigationInterface

cfg = dict()

# Define scene
cfg["scene"] = {
    "type": "Scene",
    "floor_plane_visible": True,
}

# Define objects
cfg["objects"] = [
    {
        "type": "LightObject",
        "name": "brilliant_light",
        "light_type": "Sphere",
        "intensity": 50000,
        "radius": 0.1,
        "position": [3.0, 3.0, 4.0],
    },
]

# Define robots
cfg["robots"] = [
    {
        "type": "Tiago",
        # "default_arm_pose": "diagonal45",
        # "obs_modalities": ["scan", "rgb", "depth"]
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
        # "reset_joint_pos": NavigationUtils.RESET_POSE,  # This will be handled by NavigationInterface
    },
]

# Create the environment
env = og.Environment(cfg)
robot:Tiago = env.robots[0]
robot.keep_still()
# print_joint_info(robot)
# robot.control_enabled = False

# Allow camera teleoperation
camera = og.sim.enable_viewer_camera_teleoperation()
camera.cam.set_position_orientation(position=[ 1.3629, -2.1080,  1.3431], orientation=[0.5479, 0.1376, 0.2009, 0.8003])

# Initialize navigation interface
navigator = NavigationInterface(env, robot)
navigator.setup()

TARGET_POS_LEFT = th.tensor([1.5, 1.5, 0, 2.3, 0, -1.4, 0])
TARGET_POS_RIGHT = th.tensor([1.5, 1.5, 0, 2.3, 0, -1.4, 0])
print("Left Arm Target:", TARGET_POS_LEFT)
print("Right Arm Target:", TARGET_POS_RIGHT)


# unlock robot controller limits
robot.controllers["arm_left"]._command_input_limits = None
robot.controllers["arm_right"]._command_input_limits = None
# Main control loop
while True:
    # Create an empty action tensor.
    # action = th.zeros(robot.action_dim)
    # action[0] = 1
    # action[2] = 0.3
    # # left arm
    # action[6:13] = TARGET_POS_LEFT
    # # right arm
    # action[14:21] = TARGET_POS_RIGHT
    # print(f"robot.reset_joint_pos_aabb_extent[:2]: {robot.reset_joint_pos_aabb_extent[:2]}")
    # print(f"robot.get_base_aligned_bbox(xy_aligned=False): {robot.get_base_aligned_bbox(xy_aligned=False)}")
    
    action = navigator.update()
    env.step(action)