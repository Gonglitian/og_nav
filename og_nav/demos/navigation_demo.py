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
import time

gm.GUI_VIEWPORT_ONLY = True

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

    # Visit goal points sequentially
    navigator.set_random_available_goal()
    # Main loop
    step = 0
    while True:
        # Update map
        # cal time for each update
        if step % 100 == 0:
            # calc time for each update
            start_time = time.time()
            navigator.ogm.update_env_trav_map(env)
            # save map tensor to image
            # navigator.ogm.save_grayscale_tensor_to_image(navigator.ogm.map_tensor, "map_tensor.png")
            end_time = time.time()
            print(f"Time for map update: {end_time - start_time} seconds")
        step += 1
        
        # Update environment and navigation
        action = navigator.update()
        env.step(action)
        
        # Check if current goal is reached
        if navigator.is_arrived():
            status = navigator.get_navigation_status()
            print(f"✓ Reached goal! Navigation status: {status}")
            # Move to next goal point
            goal_pos = navigator.set_random_available_goal()
            print(f"→ New random goal set: {goal_pos}")
        
        # Print detailed status every 500 steps for debugging
        if step % 500 == 0 and step > 0:
            status = navigator.get_navigation_status()
            print(f"[Step {step}] Navigation status: {status}")

if __name__ == "__main__":
    main()
