"""Constants for navigation utilities."""

import torch as th

# Base joint names for Tiago robot
BASE_JOINT_NAMES = [
    "base_footprint_x_joint",  # x方向移动
    "base_footprint_y_joint",  # y方向移动  
    "base_footprint_rz_joint"  # z轴旋转
]

# Pure Pursuit algorithm default parameters
DEFAULT_LOOKAHEAD_DISTANCE = 0.5
DEFAULT_CRUISE_SPEED = 0.5
DEFAULT_MAX_ANGULAR_VEL = 0.2
DEFAULT_WAYPOINT_THRESHOLD = 0.2

# Navigation constants from NavConstants class
class NavConstants:
    """Configuration constants for the path planning test."""

    # Rendering settings
    # Block the GUI for better debugging
    GUI_VIEWPORT_ONLY = True

    # Robot position
    START_POSITION = [-9.5, 1.3, 0]
    GOAL_POSITION = [-1, -0.3, 0]

    # Robot Pose
    RESET_POSE = th.tensor([
        # Base joints (6 DOF: x, y, z, rx, ry, rz)
        0,  # base_footprint_x_joint
        0,  # base_footprint_y_joint
        0,  # base_footprint_z_joint
        0,  # base_footprint_rx_joint
        0,  # base_footprint_ry_joint
        0,  # base_footprint_rz_joint

        # Trunk
        0,   # torso_lift_joint

        # Arms (7 DOF each)
        1.5,   # arm_left_1_joint
        1.5,   # arm_right_1_joint
        0,  # head_1_joint
        1.5,   # arm_left_2_joint
        1.5,   # arm_right_2_joint
        0,  # head_2_joint
        0.0,   # arm_left_3_joint
        0.0,   # arm_right_3_joint
        2.3,   # arm_left_4_joint
        2.3,   # arm_right_4_joint
        0,   # arm_left_5_joint
        0,   # arm_right_5_joint
        -1.4,   # arm_left_6_joint
        -1.4,   # arm_right_6_joint
        0,   # arm_left_7_joint
        0,   # arm_right_7_joint

        # Grippers
        0.045,   # gripper_left_left_finger_joint
        0.045,   # gripper_left_right_finger_joint
        0.045,   # gripper_right_left_finger_joint
        0.045    # gripper_right_right_finger_joint
    ])
    
    # For both arms of Tiago
    NAV_ARM_POSE = th.tensor([1.5, 1.5, 0, 2.3, 0, -1.4, 0])

    # OGM Map settings
    OG_MAP_RESOLUTION = 0.1  # Default map resolution for OmniGibson maps

    # Erosion radius
    DEFAULT_EROSION_RADIUS = 0.2
    
    # Visualization settings
    N_WAYPOINTS = 50
    MARKER_HEIGHT = 0.1
    HIDDEN_POSITION = [0, 0, 100]  # Position to hide markers

    # Marker appearance
    START_MARKER_COLOR = (0, 1, 0, 1)  # Green
    GOAL_MARKER_COLOR = (1, 0, 0, 1)  # Red
    WAYPOINT_COLOR = (0, 0, 1, 1)  # Blue
    MARKER_RADIUS = 0.1
    WAYPOINT_RADIUS = 0.05 