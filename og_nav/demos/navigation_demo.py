"""
Simple Navigation Demo

This module demonstrates the NavigationInterface for basic robot navigation
using a clean, minimal setup.
"""

import omnigibson as og
from omnigibson.robots.tiago import Tiago
from omnigibson import gm
from og_nav.core import NavigationInterface
from og_nav.core.config_loader import NavigationConfig
import os

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
    omnigibson_config = nav_config.get_omnigibson_config()
    if omnigibson_config is None:
        raise ValueError("No OmniGibson configuration found")
    env = og.Environment(configs=omnigibson_config)
    
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
        navigator.set_goal((goal[0], goal[1]))
        print(f"Goal {current_goal_idx + 1}/{len(DEMO_GOAL_POINTS)}: Moving to [{goal[0]:.2f}, {goal[1]:.2f}]")
    
    # Main loop
    while True:
        # Update environment and navigation
        navigator.update()
        env.step([])
        
        # Check if current goal is reached
        if navigator.controller.is_arrived():
            print(f"✓ Reached goal {current_goal_idx + 1}/{len(DEMO_GOAL_POINTS)}")
            
            # Move to next goal point
            current_goal_idx += 1
            if current_goal_idx < len(DEMO_GOAL_POINTS):
                goal = DEMO_GOAL_POINTS[current_goal_idx]
                navigator.set_goal((goal[0], goal[1]))
                print(f"Goal {current_goal_idx + 1}/{len(DEMO_GOAL_POINTS)}: Moving to [{goal[0]:.2f}, {goal[1]:.2f}]")
            else:
                print("🎉 All goal points visited! Demo completed.")
                break

if __name__ == "__main__":
    main()
