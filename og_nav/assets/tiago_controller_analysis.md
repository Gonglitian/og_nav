# Tiago机器人完整控制系统分析报告

本报告综合分析了Tiago机器人的关节结构、控制系统和导航模式，为机器人控制提供完整的技术参考。

---

## 📋 **目录**

1. [关节结构分析](#1-关节结构分析)
2. [控制系统总览](#2-控制系统总览)
3. [正常模式Action Space分析](#3-正常模式action-space分析)
4. [导航模式Action Space分析](#4-导航模式action-space分析)
5. [模式对比分析](#5-模式对比分析)
6. [实用指导](#6-实用指导)

---

## 1. 关节结构分析

### 1.1 完整关节列表
Tiago机器人总共有**27个关节**，按以下顺序排列：

```python
# 总共27个关节，按以下顺序排列：
[ 0] base_footprint_x_joint      # 底盘X轴平移
[ 1] base_footprint_y_joint      # 底盘Y轴平移
[ 2] base_footprint_z_joint      # 底盘Z轴平移（不受控）
[ 3] base_footprint_rx_joint     # 底盘X轴旋转（不受控）
[ 4] base_footprint_ry_joint     # 底盘Y轴旋转（不受控）
[ 5] base_footprint_rz_joint     # 底盘Z轴旋转
[ 6] torso_lift_joint           # 躯干升降关节
[ 7] arm_left_1_joint           # 左臂关节1
[ 8] arm_right_1_joint          # 右臂关节1
[ 9] head_1_joint               # 头部关节1（俯仰）
[10] arm_left_2_joint           # 左臂关节2
[11] arm_right_2_joint          # 右臂关节2
[12] head_2_joint               # 头部关节2（偏航）
[13] arm_left_3_joint           # 左臂关节3
[14] arm_right_3_joint          # 右臂关节3
[15] arm_left_4_joint           # 左臂关节4
[16] arm_right_4_joint          # 右臂关节4
[17] arm_left_5_joint           # 左臂关节5
[18] arm_right_5_joint          # 右臂关节5
[19] arm_left_6_joint           # 左臂关节6
[20] arm_right_6_joint          # 右臂关节6
[21] arm_left_7_joint           # 左臂关节7
[22] arm_right_7_joint          # 右臂关节7
[23] gripper_left_left_finger_joint   # 左手左手指
[24] gripper_left_right_finger_joint  # 左手右手指
[25] gripper_right_left_finger_joint  # 右手左手指
[26] gripper_right_right_finger_joint # 右手右手指
```

### 1.2 关节分组统计

| 分类 | 关节数量 | 关节索引范围 | 说明 |
|------|----------|-------------|------|
| **Base joints** | 6个 | [0-5] | 底盘6DOF，仅3个受控 |
| **Trunk** | 1个 | [6] | 躯干升降关节 |
| **Arms** | 14个 | [7-22] | 左右臂各7个关节 |
| **Head** | 2个 | [9,12] | 头部俯仰和偏航 |
| **Grippers** | 4个 | [23-26] | 左右手各2个手指 |
| **总计** | **27个** | [0-26] | 完整机器人关节 |

---

## 2. 控制系统总览

### 2.1 控制器架构

Tiago机器人采用分层控制器架构，不同类型的关节使用不同的控制器：

| 关节类型 | 控制模式 | 控制器类型 | 说明 |
|----------|----------|------------|------|
| **底盘关节** | 速度控制 | `JointController` | 显式配置 `motor_type: "velocity"` |
| **手臂关节** | 位置控制 | `InverseKinematicsController` | 底层使用位置控制的JointController |
| **抓取器关节** | 位置控制 | `MultiFingerGripperController` | 使用位置控制 |
| **头部关节** | 位置控制 | `JointController` | 使用默认位置控制 |
| **躯干关节** | 位置控制 | `JointController` | 使用默认位置控制 |

### 2.2 控制器顺序
```python
controller_order = ['base', 'camera', 'arm_left', 'gripper_left', 'arm_right', 'gripper_right']
```

---

## 3. 正常模式Action Space分析

### 3.1 Action Space结构（总维度：19）

| 控制器 | Action索引 | 维度 | 控制器类型 | 控制内容 |
|--------|-----------|------|------------|----------|
| **base** | [0:3] | 3维 | `JointController` | [vx, vy, w] |
| **camera** | [3:5] | 2维 | `JointController` | [head_1, head_2] |
| **arm_left** | [5:11] | 6维 | `InverseKinematicsController` | [dx,dy,dz,dax,day,daz] |
| **gripper_left** | [11:12] | 1维 | `MultiFingerGripperController` | 双指同步控制 |
| **arm_right** | [12:18] | 6维 | `InverseKinematicsController` | [dx,dy,dz,dax,day,daz] |
| **gripper_right** | [18:19] | 1维 | `MultiFingerGripperController` | 双指同步控制 |

### 3.2 正常模式详细映射

```python
# 正常模式Action结构 (总维度19)
action = torch.zeros(19)

# 底盘控制 [0:3) - JointController
action[0] = vx      # → base_footprint_x_joint
action[1] = vy      # → base_footprint_y_joint  
action[2] = w       # → base_footprint_rz_joint

# 头部控制 [3:5) - JointController
action[3] = head_1_pos  # → head_1_joint
action[4] = head_2_pos  # → head_2_joint

# 左臂IK控制 [5:11) - InverseKinematicsController
action[5:11] = [dx, dy, dz, dax, day, daz]  # 末端执行器增量控制

# 左手夹爪 [11:12) - MultiFingerGripperController
action[11] = left_gripper_pos

# 右臂IK控制 [12:18) - InverseKinematicsController  
action[12:18] = [dx, dy, dz, dax, day, daz]  # 末端执行器增量控制

# 右手夹爪 [18:19) - MultiFingerGripperController
action[18] = right_gripper_pos
```

---

## 4. 导航模式Action Space分析

### 4.1 导航模式Action Space结构（总维度：21）

在导航模式下，手臂控制器从IK控制切换为关节控制，Action维度从19增加到21：

| 控制器 | Action索引 | 维度 | 控制器类型 | 控制内容 | 控制关节 |
|--------|-----------|------|------------|----------|----------|
| **base** | [0:3) | 3维 | `JointController` | [vx, vy, w] | base_footprint_x/y/rz_joint |
| **camera** | [3:5) | 2维 | `JointController` | [θ1, θ2] | head_1/2_joint |
| **arm_left** | [5:12) | 7维 | `JointController` | 关节位置控制 | arm_left_1~7_joint |
| **gripper_left** | [12:13) | 1维 | `MultiFingerGripperController` | 夹爪控制 | gripper_left_*_finger |
| **arm_right** | [13:20) | 7维 | `JointController` | 关节位置控制 | arm_right_1~7_joint |
| **gripper_right** | [20:21) | 1维 | `MultiFingerGripperController` | 夹爪控制 | gripper_right_*_finger |

### 4.2 导航模式详细映射

| Action索引 | 控制器 | 关节索引 | 关节名称 | 说明 |
|-----------|-------|---------|----------|------|
| **Action[0:3)** | **base** | | | **底盘控制（3维）** |
| Action[0] | base | Joint[0] | `base_footprint_x_joint` | X方向平移 |
| Action[1] | base | Joint[1] | `base_footprint_y_joint` | Y方向平移 |
| Action[2] | base | Joint[5] | `base_footprint_rz_joint` | Z轴旋转 |
| **Action[3:5)** | **camera** | | | **头部控制（2维）** |
| Action[3] | camera | Joint[9] | `head_1_joint` | 头部俯仰 |
| Action[4] | camera | Joint[12] | `head_2_joint` | 头部偏航 |
| **Action[5:12)** | **arm_left** | | | **左臂关节控制（7维）** |
| Action[5] | arm_left | Joint[7] | `arm_left_1_joint` | 左臂关节1 |
| Action[6] | arm_left | Joint[10] | `arm_left_2_joint` | 左臂关节2 |
| Action[7] | arm_left | Joint[13] | `arm_left_3_joint` | 左臂关节3 |
| Action[8] | arm_left | Joint[15] | `arm_left_4_joint` | 左臂关节4 |
| Action[9] | arm_left | Joint[17] | `arm_left_5_joint` | 左臂关节5 |
| Action[10] | arm_left | Joint[19] | `arm_left_6_joint` | 左臂关节6 |
| Action[11] | arm_left | Joint[21] | `arm_left_7_joint` | 左臂关节7 |
| **Action[12:13)** | **gripper_left** | | | **左手夹爪（1维）** |
| Action[12] | gripper_left | Joint[24,23] | `gripper_left_*_finger` | 左手夹爪同步 |
| **Action[13:20)** | **arm_right** | | | **右臂关节控制（7维）** |
| Action[13] | arm_right | Joint[8] | `arm_right_1_joint` | 右臂关节1 |
| Action[14] | arm_right | Joint[11] | `arm_right_2_joint` | 右臂关节2 |
| Action[15] | arm_right | Joint[14] | `arm_right_3_joint` | 右臂关节3 |
| Action[16] | arm_right | Joint[16] | `arm_right_4_joint` | 右臂关节4 |
| Action[17] | arm_right | Joint[18] | `arm_right_5_joint` | 右臂关节5 |
| Action[18] | arm_right | Joint[20] | `arm_right_6_joint` | 右臂关节6 |
| Action[19] | arm_right | Joint[22] | `arm_right_7_joint` | 右臂关节7 |
| **Action[20:21)** | **gripper_right** | | | **右手夹爪（1维）** |
| Action[20] | gripper_right | Joint[26,25] | `gripper_right_*_finger` | 右手夹爪同步 |

### 4.3 导航模式代码示例

```python
# 导航模式Action结构 (总维度21)
action = torch.zeros(21)

# 底盘控制 [0:3)
action[0] = vx      # 前后速度
action[1] = vy      # 左右速度  
action[2] = w       # 旋转角速度

# 头部控制 [3:5)  
action[3] = head_1_pos  # 头部俯仰
action[4] = head_2_pos  # 头部偏航

# 左臂关节控制 [5:12) - 7个关节直接位置控制
action[5:12] = left_arm_joint_positions

# 左手夹爪 [12:13)
action[12] = left_gripper_pos

# 右臂关节控制 [13:20) - 7个关节直接位置控制  
action[13:20] = right_arm_joint_positions

# 右手夹爪 [20:21)
action[20] = right_gripper_pos
```

---

## 5. 模式对比分析

### 5.1 🔄 正常模式 vs 导航模式对比

| 项目 | 正常模式 | 导航模式 | 差异 |
|------|----------|----------|------|
| **Action总维度** | 19 | 21 | +2 |
| **手臂控制方式** | IK控制 | 关节控制 | 控制方式变化 |
| **手臂输入维度** | 6维/臂 | 7维/臂 | +1维/臂 |
| **Torso关节** | 包含在arm_left IK中 | 保持静止 | 控制方式变化 |

### 5.2 控制器变化详情

| 控制器 | 正常模式 | 导航模式 | 变化说明 |
|--------|----------|----------|----------|
| **base** | Action[0:3) JointController | Action[0:3) JointController | ✅ 无变化 |
| **camera** | Action[3:5) JointController | Action[3:5) JointController | ✅ 无变化 |
| **arm_left** | Action[5:11) IK控制(6维) | Action[5:12) 关节控制(7维) | 🔄 +1维，控制方式变化 |
| **gripper_left** | Action[11:12) | Action[12:13) | 🔄 索引偏移 |
| **arm_right** | Action[12:18) IK控制(6维) | Action[13:20) 关节控制(7维) | 🔄 +1维，控制方式变化 |
| **gripper_right** | Action[18:19) | Action[20:21) | 🔄 索引偏移 |

### 5.3 关键差异分析

#### **手臂控制方式变化**
- **正常模式**: IK控制器，输入6DOF末端执行器增量 `[dx,dy,dz,dax,day,daz]`
- **导航模式**: 关节控制器，输入7个关节的直接位置控制

#### **Torso关节处理**
- **正常模式**: `torso_lift_joint` 包含在 `arm_left` IK控制器中
- **导航模式**: `torso_lift_joint` 不在任何控制器中（保持静止）

#### **Action索引偏移**
- 由于左臂维度增加，所有后续控制器的Action索引都发生偏移
- 这就是为什么需要动态获取action索引的原因

### 5.4 优势对比

| 特性 | 正常模式 | 导航模式 |
|------|----------|----------|
| **控制精度** | 末端执行器精确控制 | 关节级精确控制 |
| **稳定性** | 依赖IK求解稳定性 | 直接关节控制，更稳定 |
| **导航性能** | IK求解可能影响导航 | 手臂姿态固定，导航更精确 |
| **灵活性** | 高（末端执行器任意姿态） | 低（预设固定姿态） |
| **计算开销** | 高（IK求解） | 低（直接关节控制） |

---

## 6. 实用指导

### 6.1 导航模式切换API

```python
from utils import enter_navigation_mode, exit_navigation_mode, is_in_navigation_mode

# 进入导航模式
enter_navigation_mode(robot)
print(f"当前Action维度: {robot.action_dim}")  # 输出: 21

# 退出导航模式  
exit_navigation_mode(robot)
print(f"当前Action维度: {robot.action_dim}")  # 输出: 19

# 检查导航模式状态
if is_in_navigation_mode():
    print("当前处于导航模式")
```

### 6.2 动态Action处理

```python
def create_action(robot, base_control):
    """创建动态适配的action tensor"""
    # 动态获取当前action维度
    current_action_dim = robot.action_dim
    action = torch.zeros(current_action_dim)
    
    # 底盘控制始终在前3个维度
    action[0:3] = base_control
    
    # 其他控制器根据当前模式动态处理
    return action
```

### 6.3 控制器检测

```python
def detect_current_mode(robot):
    """检测当前控制模式"""
    arm_left_controller = robot.controllers['arm_left']
    
    if isinstance(arm_left_controller, InverseKinematicsController):
        return "normal_mode", 19
    elif isinstance(arm_left_controller, JointController):
        return "navigation_mode", 21
    else:
        return "unknown_mode", robot.action_dim
```

### 6.4 最佳实践建议

1. **模式切换时机**
   - 导航任务开始前切换到导航模式
   - 精细操作任务前切换回正常模式

2. **Action维度处理**
   - 始终使用 `robot.action_dim` 动态获取维度
   - 避免硬编码action索引

3. **控制器状态管理**
   - 程序退出前确保恢复原始控制器配置
   - 异常处理中包含控制器恢复逻辑

4. **性能优化**
   - 导航模式下可以减少控制频率
   - 利用关节控制的稳定性优势

---

## 📚 **参考信息**

### 关键源码位置

| 文件 | 关键方法/属性 | 功能 |
|------|---------------|------|
| `omnigibson/robots/tiago.py` | `controller_order` | 定义控制器顺序 |
| `omnigibson/robots/tiago.py` | `_default_controllers` | 定义默认控制器类型 |
| `omnigibson/controllers/ik_controller.py` | `__init__` | IK控制器默认参数 |
| `omnigibson/controllers/joint_controller.py` | `command_dim` | 关节控制器维度 |
| `og_nav/utils/robot_control.py` | `enter_navigation_mode` | 进入导航模式 |
| `og_nav/utils/robot_control.py` | `exit_navigation_mode` | 退出导航模式 |

### 控制命令格式

| 控制器 | 命令格式 | 单位 | 坐标系 |
|--------|----------|------|-------|
| base | [vx, vy, w] | m/s, rad/s | 机器人基础坐标系 |
| camera | [θ1, θ2] | rad | 关节坐标系 |
| arm (正常) | [dx,dy,dz,dax,day,daz] | m, rad | 机器人基础坐标系 |
| arm (导航) | [q1,q2,q3,q4,q5,q6,q7] | rad | 关节坐标系 |
| gripper | [position] | m | 关节坐标系 |

---

**文档版本**: v1.0  
**最后更新**: 2024-07-24  
**作者**: AI Assistant  
**适用**: OmniGibson 4.1.0 + Tiago机器人 