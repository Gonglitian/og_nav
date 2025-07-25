"""Robot control utilities for joint management."""

from typing import Dict, List, Tuple, Union

import torch as th
from omnigibson.robots.tiago import Tiago
from omnigibson.controllers import JointController, InverseKinematicsController, MultiFingerGripperController, create_controller

from .constants import BASE_JOINT_NAMES

# 导航模式时的关节位置配置
NAVIGATION_JOINT_POSITIONS = [
    # Base joints (6 DOF: x, y, z, rx, ry, rz)
    0,      # base_footprint_x_joint
    0,      # base_footprint_y_joint
    0,      # base_footprint_z_joint
    0,      # base_footprint_rx_joint
    0,      # base_footprint_ry_joint
    0,      # base_footprint_rz_joint
    
    # Trunk
    0,      # torso_lift_joint
    
    # Arms (交替排列: left_1, right_1, left_2, right_2, ...)
    1.570,  # arm_left_1_joint
    1.570,  # arm_right_1_joint
    0,      # head_1_joint
    2.356,  # arm_left_2_joint
    2.356,  # arm_right_2_joint
    0,      # head_2_joint
    0,      # arm_left_3_joint
    0,      # arm_right_3_joint
    2.356,  # arm_left_4_joint
    2.356,  # arm_right_4_joint
    1,      # arm_left_5_joint
    1,      # arm_right_5_joint
    -1.570, # arm_left_6_joint
    -1.570, # arm_right_6_joint
    0,      # arm_left_7_joint
    0,      # arm_right_7_joint
    
    # Grippers
    0.045,  # gripper_left_left_finger_joint
    0.045,  # gripper_left_right_finger_joint
    0.045,  # gripper_right_left_finger_joint
    0.045   # gripper_right_right_finger_joint
]

def get_original_arm_controllers(robot: Tiago):
    """获取原始控制器配置"""
    return {
        'arm_left': robot.controllers['arm_left'],
        'arm_right': robot.controllers['arm_right']
    }

def set_arm_jointcontroller(robot: Tiago) -> None:
    """
    设置手臂控制器
    
    Args:
        robot: Tiago机器人实例
        controller: 控制器类型，可选"JointController"或"InverseKinematicsController"
    """
    
    # 获取手臂关节的dof_idx
    arm_left_joints = ["arm_left_1_joint", "arm_left_2_joint", "arm_left_3_joint", 
                      "arm_left_4_joint", "arm_left_5_joint", "arm_left_6_joint", "arm_left_7_joint"]
    arm_right_joints = ["arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint",
                       "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint", "arm_right_7_joint"]
    joint_names_list = list(robot.joints.keys())
    arm_left_dof_idx = th.tensor([joint_names_list.index(joint_name) for joint_name in arm_left_joints], dtype=th.int32)
    arm_right_dof_idx = th.tensor([joint_names_list.index(joint_name) for joint_name in arm_right_joints], dtype=th.int32)
    
    arm_left_controller = JointController(
        control_freq=robot._control_freq,
        motor_type="position", 
        dof_idx=arm_left_dof_idx,
        use_delta_commands=False,
        control_limits=None,
        command_input_limits=None,
        command_output_limits=None,
    )
    arm_right_controller = JointController(
        control_freq=robot._control_freq,
        motor_type="position", 
        dof_idx=arm_right_dof_idx,
        use_delta_commands=False,
        command_input_limits=None,
        command_output_limits=None,
    )
    
    robot.controllers['arm_left'] = arm_left_controller
    robot.controllers['arm_right'] = arm_right_controller
    robot.controllers["arm_left"]._command_input_limits = None
    robot.controllers["arm_right"]._command_input_limits = None
    print("🔧 已切换手臂控制器为JointController")
    
def set_navigation_joint_positions(robot: Tiago, joint_positions: List[float] = None) -> None:
    """
    设置导航模式下的关节位置
    
    Args:
        robot: Tiago机器人实例
        joint_positions: 可选的关节位置列表，如果为None则使用默认导航姿态
    """
    if joint_positions is None:
        joint_positions = NAVIGATION_JOINT_POSITIONS
    
    if len(joint_positions) != len(robot.joints):
        raise ValueError(f"关节位置数量({len(joint_positions)})与机器人关节数量({len(robot.joints)})不匹配")
    
    robot.set_joint_positions(th.tensor(joint_positions, dtype=th.float32))
    print("📐 已设置关节位置")

def get_joint_velocity_summary(robot: Tiago) -> Dict[str, Union[int, float]]:
    """获取机器人关节速度摘要信息.
    
    Args:
        robot: 机器人实例.
        
    Returns:
        包含关节速度统计信息的字典，包含以下键：
        - total_joints: 总关节数
        - base_velocity_sum: 底盘关节速度总和
        - non_base_velocity_sum: 非底盘关节速度总和
        - max_velocity: 最大关节速度
        - mean_velocity: 平均关节速度
    """
    joint_vels = robot.get_joint_velocities()
    
    # 分别统计底盘关节和非底盘关节的速度
    base_vel_sum = 0.0
    non_base_vel_sum = 0.0
    
    for i, (joint_name, _) in enumerate(robot.joints.items()):
        if i < len(joint_vels):
            vel_abs = abs(joint_vels[i].item())
            if joint_name in BASE_JOINT_NAMES:
                base_vel_sum += vel_abs
            else:
                non_base_vel_sum += vel_abs
    
    return {
        "total_joints": len(robot.joints),
        "base_velocity_sum": base_vel_sum,
        "non_base_velocity_sum": non_base_vel_sum,
        "max_velocity": float(th.max(th.abs(joint_vels))),
        "mean_velocity": float(th.mean(th.abs(joint_vels)))
    }


def get_joint_info(robot: Tiago) -> Dict[str, Dict[str, Union[int, str, float]]]:
    """获取机器人关节信息，包括关节名称、action索引和当前位置.
    
    Args:
        robot: Tiago机器人实例
        
    Returns:
        字典，键为关节名称，值包含关节信息：
        - action_idx: 关节在action tensor中的索引
        - current_pos: 当前关节位置
        - joint_type: 关节类型
    """
    joint_info = {}
    joint_positions = robot.get_joint_positions()
    
    for i, (joint_name, joint) in enumerate(robot.joints.items()):
        joint_info[joint_name] = {
            "action_idx": i,
            "current_pos": joint_positions[i].item() if i < len(joint_positions) else 0.0,
            "joint_type": "base" if joint_name in BASE_JOINT_NAMES else "non_base"
        }
    
    return joint_info


def create_arm_control_action(
    robot: Tiago, 
    joint_targets: Dict[str, float], 
    keep_base_zero: bool = True
) -> th.Tensor:
    """创建用于控制特定手臂关节的action tensor.
    
    Args:
        robot: Tiago机器人实例
        joint_targets: 字典，键为关节名称，值为目标位置（弧度）
        keep_base_zero: 是否保持底盘关节为零
        
    Returns:
        完整的action tensor，可直接用于env.step()
        
    Example:
        # 控制左臂第一个关节到90度
        action = create_arm_control_action(robot, {
            "arm_left_1_joint": 1.57  # 90度 = π/2弧度
        })
        env.step(action)
    """
    # 创建零action
    action = th.zeros(robot.action_dim)
    
    if keep_base_zero:
        # 保持底盘关节为零（适用于位置控制）
        action[0:3] = 0.0
    
    # 设置指定关节的目标位置
    for joint_name, target_pos in joint_targets.items():
        if joint_name in robot.joints:
            # 找到关节在action tensor中的索引
            joint_idx = list(robot.joints.keys()).index(joint_name)
            if joint_idx < len(action):
                action[joint_idx] = target_pos
    
    return action


def detect_controller_mode(robot: Tiago, test_joint: str = "arm_left_1_joint") -> str:
    """检测关节控制器模式（绝对位置 vs 增量控制）.
    
    Args:
        robot: Tiago机器人实例
        test_joint: 用于测试的关节名称
        
    Returns:
        控制器模式："absolute"（绝对位置）或 "delta"（增量控制）
    """
    if test_joint not in robot.joints:
        print(f"警告：测试关节 {test_joint} 不存在，使用默认关节")
        test_joint = list(robot.joints.keys())[7]  # 使用arm_left_1_joint
    
    # 获取初始位置
    initial_positions = robot.get_joint_positions()
    joint_idx = list(robot.joints.keys()).index(test_joint)
    initial_pos = initial_positions[joint_idx].item()
    
    # 发送一个小的非零action
    test_action = th.zeros(robot.action_dim)
    test_action[joint_idx] = 0.1  # 发送0.1弧度
    
    # 执行几步
    for _ in range(10):
        # 这里需要一个环境实例，但我们没有，所以先返回提示
        pass
    
    print(f"需要在环境中测试才能确定控制器模式")
    print(f"请运行 python controller_test.py 进行完整测试")
    
    return "unknown"


def create_delta_control_action(
    robot: Tiago,
    joint_increments: Dict[str, float],
    keep_base_zero: bool = True
) -> th.Tensor:
    """创建用于增量控制的action tensor.
    
    适用于delta控制模式，action表示相对于当前位置的增量。
    
    Args:
        robot: Tiago机器人实例
        joint_increments: 字典，键为关节名称，值为增量（弧度）
        keep_base_zero: 是否保持底盘关节为零
        
    Returns:
        完整的action tensor，用于增量控制
        
    Example:
        # 将左臂第一个关节增加0.1弧度
        action = create_delta_control_action(robot, {
            "arm_left_1_joint": 0.1  # 增加0.1弧度
        })
        env.step(action)
    """
    # 创建零action（在delta模式下，0表示保持当前位置）
    action = th.zeros(robot.action_dim)
    
    if keep_base_zero:
        # 保持底盘关节为零
        action[0:3] = 0.0
    
    # 设置指定关节的增量
    for joint_name, increment in joint_increments.items():
        if joint_name in robot.joints:
            joint_idx = list(robot.joints.keys()).index(joint_name)
            if joint_idx < len(action):
                action[joint_idx] = increment
    
    return action


def create_absolute_control_action(
    robot: Tiago,
    joint_targets: Dict[str, float],
    keep_base_zero: bool = True
) -> th.Tensor:
    """创建用于绝对位置控制的action tensor.
    
    适用于绝对位置控制模式，action表示目标关节位置。
    
    Args:
        robot: Tiago机器人实例
        joint_targets: 字典，键为关节名称，值为目标位置（弧度）
        keep_base_zero: 是否保持底盘关节为零
        
    Returns:
        完整的action tensor，用于绝对位置控制
        
    Example:
        # 将左臂第一个关节移动到90度
        action = create_absolute_control_action(robot, {
            "arm_left_1_joint": 1.57  # 90度 = π/2弧度
        })
        env.step(action)
    """
    # 获取当前关节位置
    current_positions = robot.get_joint_positions()
    action = current_positions.clone()
    
    if keep_base_zero:
        # 保持底盘关节为零
        action[0:3] = 0.0
    
    # 设置指定关节的目标位置
    for joint_name, target_pos in joint_targets.items():
        if joint_name in robot.joints:
            joint_idx = list(robot.joints.keys()).index(joint_name)
            if joint_idx < len(action):
                action[joint_idx] = target_pos
    
    return action


def move_joint_to_position(
    robot: Tiago,
    joint_name: str,
    target_position: float,
    max_steps: int = 100,
    tolerance: float = 0.01,
    controller_mode: str = "auto"
) -> Tuple[bool, float]:
    """将指定关节移动到目标位置.
    
    Args:
        robot: Tiago机器人实例
        joint_name: 关节名称
        target_position: 目标位置（弧度）
        max_steps: 最大步数
        tolerance: 位置容差
        controller_mode: 控制器模式 ("auto", "absolute", "delta")
        
    Returns:
        Tuple[成功标志, 最终位置]
        
    Note:
        这个函数需要环境实例才能工作，仅作为参考实现
    """
    if joint_name not in robot.joints:
        print(f"错误：关节 {joint_name} 不存在")
        return False, 0.0
    
    joint_idx = list(robot.joints.keys()).index(joint_name)
    
    print(f"移动关节 {joint_name} 到位置 {target_position:.3f}")
    print(f"注意：此函数需要环境实例才能实际执行")
    
    # 这里需要实际的环境来执行action
    # 以下是伪代码示例：
    
    # for step in range(max_steps):
    #     current_pos = robot.get_joint_positions()[joint_idx].item()
    #     
    #     if abs(current_pos - target_position) < tolerance:
    #         print(f"成功：在步骤{step}达到目标位置")
    #         return True, current_pos
    #     
    #     if controller_mode == "delta":
    #         increment = min(0.1, target_position - current_pos)
    #         action = create_delta_control_action(robot, {joint_name: increment})
    #     else:  # absolute
    #         action = create_absolute_control_action(robot, {joint_name: target_position})
    #     
    #     env.step(action)
    
    return False, 0.0


def print_joint_info(robot: Tiago, show_positions: bool = True) -> None:
    """打印机器人关节信息，便于调试.
    
    Args:
        robot: Tiago机器人实例
        show_positions: 是否显示当前关节位置
    """
    print(f"\n=== Tiago Robot Joint Information ===")
    print(f"Total joints: {len(robot.joints)}")
    print(f"Action dimension: {robot.action_dim}")
    
    joint_info = get_joint_info(robot)
    
    print("\nJoint List:")
    for i, (joint_name, info) in enumerate(joint_info.items()):
        pos_str = f", pos={info['current_pos']:.3f}" if show_positions else ""
        print(f"  [{i:2d}] {joint_name:<25} ({info['joint_type']}){pos_str}")
    
    print(f"\nBase joints (velocity control): {BASE_JOINT_NAMES}")
    print(f"Non-base joints (position control): {len(joint_info) - len(BASE_JOINT_NAMES)} joints")
    
    # 添加控制器模式提示
    print("\n🔍 控制器模式诊断:")
    print("   如果传入action=0但关节移动到0位置 -> 绝对位置控制")
    print("   如果传入action=0但关节保持当前位置 -> 增量控制（delta）")
    print("   运行 'python controller_test.py' 进行详细测试")
    
    print("=" * 50)
    """返回手臂控制的示例代码字符串.
    
    Returns:
        包含示例代码的字符串
    """
    example_code = '''
# 手臂关节控制示例（支持两种控制模式）

# 1. 获取机器人关节信息
print_joint_info(robot)

# 2. 检测控制器模式
controller_mode = detect_controller_mode(robot)

# 3. 绝对位置控制示例
if controller_mode == "absolute":
    # 直接指定目标位置
    action = create_absolute_control_action(robot, {
        "arm_left_1_joint": 1.57  # 90度
    })
    env.step(action)

# 4. 增量控制示例
elif controller_mode == "delta":
    # 指定位置增量
    action = create_delta_control_action(robot, {
        "arm_left_1_joint": 0.1  # 增加0.1弧度
    })
    env.step(action)

# 5. 通用函数（自动检测模式）
action = create_arm_control_action(robot, {
    "arm_left_1_joint": 1.57  # 会根据控制器模式自动处理
})
env.step(action)

# 6. 分步移动到目标位置（适用于增量控制）
if controller_mode == "delta":
    current_pos = robot.get_joint_positions()[7].item()  # arm_left_1_joint
    target_pos = 1.57
    steps = 50
    
    for i in range(steps):
        increment = (target_pos - current_pos) / steps
        action = create_delta_control_action(robot, {
            "arm_left_1_joint": increment
        })
        env.step(action)

# 7. 检查当前关节位置
joint_positions = robot.get_joint_positions()
for i, (joint_name, _) in enumerate(robot.joints.items()):
    print(f"{joint_name}: {joint_positions[i].item():.3f} rad")
'''
    return example_code 