"""
Simple Navigation Demo

This module demonstrates the NavigationInterface for basic robot navigation
using a clean, minimal setup.
"""

import omnigibson as og
from omnigibson.macros import gm
import torch as th
from omnigibson.robots.tiago import Tiago

from ..core import NavigationInterface


def create_simple_environment():
    """Create a simple environment for navigation testing."""
    config = {
        "scene": {
            "type": "Scene",
            "floor_plane_visible": True,
        },
        "objects": [
            {
                "type": "LightObject",
                "name": "brilliant_light",
                "light_type": "Sphere",
                "intensity": 50000,
                "radius": 0.1,
                "position": [3.0, 3.0, 4.0],
            },
        ],
        "robots": [
            {
                "type": "Tiago",
                "controller_config": {
                    "arm_left": {
                        "name": "JointController",
                        "motor_type": "position", 
                        "use_delta_commands": False,
                        "use_impedances": True,
                        "command_input_limits": None,
                    },
                    "arm_right": {
                        "name": "JointController", 
                        "motor_type": "position",
                        "use_delta_commands": False,
                        "use_impedances": True,
                        "command_input_limits": None,
                    }
                },
            },
        ],
    }
    return config


def main() -> None:
    """Main execution function for simple navigation demo."""
    print("Starting Simple Navigation Demo...")
    
    # Create environment
    config = create_simple_environment()
    env = og.Environment(config)
    
    # Get robot
    robot: Tiago = env.robots[0]
    robot.keep_still()
    
    # Enable camera control
    camera = og.sim.enable_viewer_camera_teleoperation()
    camera.cam.set_position_orientation(
        position=[1.3629, -2.1080, 1.3431], 
        orientation=[0.5479, 0.1376, 0.2009, 0.8003]
    )
    
    # Initialize navigation interface
    navigator = NavigationInterface(env, robot)
    navigator.setup()
    
    # Set a simple goal
    navigator.set_goal((2.0, 2.0))
    
    # Main control loop
    print("Starting navigation... Press Ctrl+C to stop")
    try:
        while True:
            action = navigator.update()
            env.step(action)
    except KeyboardInterrupt:
        print("Navigation demo stopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    main() 