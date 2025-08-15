import omnigibson as og
from omnigibson import gm
from og_nav.core import NavigationInterface
from og_nav.core.config_loader import NavigationConfig
import os
import torch as th
from omnigibson.robots.tiago import Tiago

gm.GUI_VIEWPORT_ONLY = True

def main():
    """Main demo function."""
    # Use unified configuration manager to process config file
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "test_90_degree_angle_tracking.yaml")
    nav_config = NavigationConfig(config_path=config_path)
    
    print("Creating environment...")
    
    # Create environment using processed configuration
    env = og.Environment(configs=nav_config.omnigibson_config)
    
    og.sim.enable_viewer_camera_teleoperation()
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([-11.6915,   0.2339,  22.3074]),
        orientation=th.tensor([-0.0860,  0.0869,  0.7055, -0.6981]),
    )
    
    robot:Tiago = env.robots[0]
    
    # Create navigation interface with og_nav configuration
    navigator = NavigationInterface(env, robot, nav_config.og_nav_config)
    
    print("Environment created. Starting navigation demo...")

    # Visit goal points sequentially
    navigator.set_goal([-4.0500,  9.6500])
    step = 0
    # Main loop
    while True:
        # Update environment and navigation
        action = navigator.update()
        env.step(action)
        # if step % 100 == 0:
            # navigator.ogm.update_env_trav_map(env)
            # navigator.ogm.save_grayscale_tensor_to_image(navigator.ogm.map_tensor, "map_tensor.png")
        step += 1

if __name__ == "__main__":
    main()
