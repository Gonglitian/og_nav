import omnigibson as og
import torch as th
import cv2
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.robots.tiago import Tiago
from omnigibson.macros import gm
from ..mapping import OGMGenerator
from .ui import OGMScrollUI

# 设置仅显示相机视角，隐藏其他 UI
gm.GUI_VIEWPORT_ONLY = True

# 初始机器人位置
START_POSITION = [-9.5, 1.3, 0]
MAP_RESOLUTION = 0.1
# 默认导航配置
DEFAULT_NAVIGATION_CONFIG = {
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "Pomaria_1_int",
        # "not_load_object_categories": ["door", "ceilings"],
        "load_object_categories": ["floors", "walls"],
    },
    "robots": [
        {
            "type": "Tiago",
            "default_arm_pose": "vertical",
            "position": START_POSITION,
            "controller_config": {
                "base": {"name": "JointController", "motor_type": "velocity"}
            },
        }
    ],
    "objects": [],
}

# 初始化环境和组件
env = og.Environment(configs=DEFAULT_NAVIGATION_CONFIG)
robot: Tiago = env.robots[0]
scene: InteractiveTraversableScene = env.scene
camera = og.sim.enable_viewer_camera_teleoperation()
generator = OGMGenerator(resolution=MAP_RESOLUTION)

# 初始化滚动条界面
ui = OGMScrollUI(generator)
ui.init_sliders()

print("[INFO] 使用滑动条调整参数，按空格生成地图，按 'q' 退出")

while True:
    action = th.zeros(robot.action_dim)
    env.step(action)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("[INFO] 程序退出")
        break
    elif key == ord(" "):
        param_set = ui.get_slider_values()
        ui.generate_map(param_set)

cv2.destroyAllWindows()
env.close()
