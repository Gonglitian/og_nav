"""
Simple Navigation Demo

This module demonstrates the NavigationInterface for basic robot navigation
using a clean, minimal setup.
"""

import omnigibson as og
from omnigibson import gm
from og_nav.core import NavigationInterface
from og_nav.core.config_loader import NavigationConfig
import os
import torch as th

gm.GUI_VIEWPORT_ONLY = True

# Configurable goal points list - you can modify this to test different routes
DEMO_GOAL_POINTS = [
    [-1.0, -0.3],    # Goal 1: Original goal position
    [0.40, 2.76],     # Goal 2: Upper left area
]

def main():
    """Main demo function."""
    # Use unified configuration manager to process config file
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "navigation_config.yaml")
    nav_config = NavigationConfig(config_path=config_path)
    
    print("Creating environment...")
    
    # Create environment using processed configuration
    env = og.Environment(configs=nav_config.omnigibson_config)
    
    og.sim.enable_viewer_camera_teleoperation()
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([-11.6915,   0.2339,  22.3074]),
        orientation=th.tensor([-0.0860,  0.0869,  0.7055, -0.6981]),
    )
    
    robot = env.robots[0]
    
    # Create navigation interface with og_nav configuration
    navigator = NavigationInterface(env, robot, nav_config.og_nav_config)
    
    print("Environment created. Starting navigation demo...")
    print(f"Will visit {len(DEMO_GOAL_POINTS)} goal points sequentially")
    
    # Visit goal points sequentially
    current_goal_idx = 0
    
    # Set the first goal point
    if DEMO_GOAL_POINTS:
        goal = DEMO_GOAL_POINTS[current_goal_idx]
        navigator.set_goal(goal)
        print(f"Goal {current_goal_idx + 1}/{len(DEMO_GOAL_POINTS)}: Moving to [{goal[0]:.2f}, {goal[1]:.2f}]")
    
    # Main loop
    while True:
        # Update environment and navigation
        action = navigator.update()
        env.step(action)
        
        # Check if current goal is reached
        if navigator.is_arrived():
            print(f"✓ Reached goal {current_goal_idx + 1}/{len(DEMO_GOAL_POINTS)}")
            # Move to next goal point
            current_goal_idx += 1
            if current_goal_idx < len(DEMO_GOAL_POINTS):
                goal = DEMO_GOAL_POINTS[current_goal_idx]
                navigator.set_goal(goal)
                print(f"Goal {current_goal_idx + 1}/{len(DEMO_GOAL_POINTS)}: Moving to [{goal[0]:.2f}, {goal[1]:.2f}]")
            else:
                print("🎉 All goal points visited! Demo completed.")
                break

if __name__ == "__main__":
    main()
