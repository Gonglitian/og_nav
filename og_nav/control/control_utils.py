"""Robot control utilities for joint management."""

from typing import Dict, List, Tuple, Union

import torch as th
from omnigibson.robots.tiago import Tiago
from omnigibson.controllers import (
    JointController,
    InverseKinematicsController,
    MultiFingerGripperController,
    create_controller,
)

from og_nav.core.constants import BASE_JOINT_NAMES


def get_original_arm_controllers(robot: Tiago):
    """Get original controller configuration"""
    return {
        "arm_left": robot.controllers["arm_left"],
        "arm_right": robot.controllers["arm_right"],
    }


def set_arm_jointcontroller(robot: Tiago) -> None:
    """
    Set up arm controllers

    Args:
        robot: Tiago robot instance
        controller: Controller type, options "JointController" or "InverseKinematicsController"
    """

    # Get DOF indices for arm joints
    arm_left_joints = [
        "arm_left_1_joint",
        "arm_left_2_joint",
        "arm_left_3_joint",
        "arm_left_4_joint",
        "arm_left_5_joint",
        "arm_left_6_joint",
        "arm_left_7_joint",
    ]
    arm_right_joints = [
        "arm_right_1_joint",
        "arm_right_2_joint",
        "arm_right_3_joint",
        "arm_right_4_joint",
        "arm_right_5_joint",
        "arm_right_6_joint",
        "arm_right_7_joint",
    ]
    joint_names_list = list(robot.joints.keys())
    arm_left_dof_idx = th.as_tensor(
        [joint_names_list.index(joint_name) for joint_name in arm_left_joints],
        dtype=th.int32,
    )
    arm_right_dof_idx = th.as_tensor(
        [joint_names_list.index(joint_name) for joint_name in arm_right_joints],
        dtype=th.int32,
    )

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

    robot.controllers["arm_left"] = arm_left_controller
    robot.controllers["arm_right"] = arm_right_controller
    robot.controllers["arm_left"]._command_input_limits = None
    robot.controllers["arm_right"]._command_input_limits = None
    print("🔧 Switched arm controllers to JointController")


def set_navigation_joint_positions(
    robot: Tiago, joint_positions: List[float] = None
) -> None:
    """
    Set joint positions for navigation mode

    Args:
        robot: Tiago robot instance
        joint_positions: Optional joint position list, uses default navigation pose if None
    """
    if joint_positions is None:
        joint_positions = NAVIGATION_JOINT_POSITIONS

    if len(joint_positions) != len(robot.joints):
        raise ValueError(
            f"Joint position count ({len(joint_positions)}) does not match robot joint count ({len(robot.joints)})"
        )

    robot.set_joint_positions(th.as_tensor(joint_positions, dtype=th.float32))
    print("📐 Joint positions set")


def get_joint_velocity_summary(robot: Tiago) -> Dict[str, Union[int, float]]:
    """Get robot joint velocity summary information.

    Args:
        robot: Robot instance.

    Returns:
        Dictionary containing joint velocity statistics with keys:
        - total_joints: Total joint count
        - base_velocity_sum: Sum of base joint velocities
        - non_base_velocity_sum: Sum of non-base joint velocities
        - max_velocity: Maximum joint velocity
        - mean_velocity: Average joint velocity
    """
    joint_vels = robot.get_joint_velocities()

    # Calculate velocities for base and non-base joints separately
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
        "mean_velocity": float(th.mean(th.abs(joint_vels))),
    }


def get_joint_info(robot: Tiago) -> Dict[str, Dict[str, Union[int, str, float]]]:
    """Get robot joint information including joint names, action indices and current positions.

    Args:
        robot: Tiago robot instance

    Returns:
        Dictionary with joint names as keys, values contain joint info:
        - action_idx: Joint index in action tensor
        - current_pos: Current joint position
        - joint_type: Joint type
    """
    joint_info = {}
    joint_positions = robot.get_joint_positions()

    for i, (joint_name, joint) in enumerate(robot.joints.items()):
        joint_info[joint_name] = {
            "action_idx": i,
            "current_pos": (
                joint_positions[i].item() if i < len(joint_positions) else 0.0
            ),
            "joint_type": "base" if joint_name in BASE_JOINT_NAMES else "non_base",
        }

    return joint_info


def create_arm_control_action(
    robot: Tiago, joint_targets: Dict[str, float], keep_base_zero: bool = True
) -> th.Tensor:
    """Create action tensor for controlling specific arm joints.

    Args:
        robot: Tiago robot instance
        joint_targets: Dictionary with joint names as keys, target positions (radians) as values
        keep_base_zero: Whether to keep base joints at zero

    Returns:
        Complete action tensor, can be used directly with env.step()

    Example:
        # Control left arm first joint to 90 degrees
        action = create_arm_control_action(robot, {
            "arm_left_1_joint": 1.57  # 90 degrees = π/2 radians
        })
        env.step(action)
    """
    # Create zero action
    action = th.zeros(robot.action_dim)

    if keep_base_zero:
        # Keep base joints at zero (suitable for position control)
        action[0:3] = 0.0

    # Set target positions for specified joints
    for joint_name, target_pos in joint_targets.items():
        if joint_name in robot.joints:
            # Find joint index in action tensor
            joint_idx = list(robot.joints.keys()).index(joint_name)
            if joint_idx < len(action):
                action[joint_idx] = target_pos

    return action


def detect_controller_mode(robot: Tiago, test_joint: str = "arm_left_1_joint") -> str:
    """Detect joint controller mode (absolute position vs incremental control).

    Args:
        robot: Tiago robot instance
        test_joint: Joint name for testing

    Returns:
        Controller mode: "absolute" (absolute position) or "delta" (incremental control)
    """
    if test_joint not in robot.joints:
        print(f"Warning: Test joint {test_joint} does not exist, using default joint")
        test_joint = list(robot.joints.keys())[7]  # Use arm_left_1_joint

    # Get initial position
    initial_positions = robot.get_joint_positions()
    joint_idx = list(robot.joints.keys()).index(test_joint)
    initial_pos = initial_positions[joint_idx].item()

    # Send a small non-zero action
    test_action = th.zeros(robot.action_dim)
    test_action[joint_idx] = 0.1  # Send 0.1 radians

    # Execute several steps
    for _ in range(10):
        # Need an environment instance here, but we don't have one, so return hint
        pass

    print(f"Need to test in environment to determine controller mode")
    print(f"Please run python controller_test.py for complete testing")

    return "unknown"


def create_delta_control_action(
    robot: Tiago, joint_increments: Dict[str, float], keep_base_zero: bool = True
) -> th.Tensor:
    """Create action tensor for incremental control.

    Suitable for delta control mode, where action represents increments relative to current position.

    Args:
        robot: Tiago robot instance
        joint_increments: Dictionary with joint names as keys, increments (radians) as values
        keep_base_zero: Whether to keep base joints at zero

    Returns:
        Complete action tensor for incremental control

    Example:
        # Increase left arm first joint by 0.1 radians
        action = create_delta_control_action(robot, {
            "arm_left_1_joint": 0.1  # Increase by 0.1 radians
        })
        env.step(action)
    """
    # Create zero action (in delta mode, 0 means maintain current position)
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
    robot: Tiago, joint_targets: Dict[str, float], keep_base_zero: bool = True
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
    controller_mode: str = "auto",
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
    """Print robot joint information for debugging.

    Args:
        robot: Tiago robot instance
        show_positions: Whether to display current joint positions
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
    print(
        f"Non-base joints (position control): {len(joint_info) - len(BASE_JOINT_NAMES)} joints"
    )

    # Add controller mode hints
    print("\n🔍 Controller Mode Diagnosis:")
    print("   If action=0 moves joint to 0 position -> Absolute position control")
    print("   If action=0 keeps joint at current position -> Incremental control (delta)")
    print("   Run 'python controller_test.py' for detailed testing")

    print("=" * 50)
    """返回手臂控制的示例代码字符串.
    
    Returns:
        包含示例代码的字符串
    """
    example_code = """
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
"""
    return example_code
