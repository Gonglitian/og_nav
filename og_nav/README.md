# OmniGibson Navigation Package

A modular navigation system for robot navigation in OmniGibson environments.

## 📦 Installation

You can install this package and its dependencies with pip:

```bash
pip install .
# or from source repo
git clone https://github.com/Gonglitian/og_nav.git
cd og_nav/og_nav
pip install .
```

### Requirements

- Python >= 3.8
- OmniGibson
- PyTorch
- NumPy
- OpenCV (opencv-python)
- Matplotlib

You can also install all dependencies with:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
og_nav/
├── __init__.py              # Main package interface
├── core/                    # Core navigation components
│   ├── __init__.py
│   ├── constants.py         # Configuration constants
│   └── navigation.py        # Navigation interface and utilities
├── planning/                # Path planning algorithms
│   ├── __init__.py
│   └── path_planning.py     # Path planning with visualization
├── control/                 # Robot control systems
│   ├── __init__.py
│   ├── controllers.py       # Path tracking controllers (PID, Pure Pursuit)
│   └── robot_control.py     # Robot joint control utilities
├── mapping/                 # Mapping functionality
│   ├── __init__.py
│   └── occupancy_grid.py    # Occupancy grid map generation
├── demos/                   # Example demonstrations
│   ├── __init__.py
│   ├── navigation_demo.py            # Simple navigation demo
│   ├── planning_tracking_demo.py     # Full planning & tracking demo
│   └── simple_environment_demo.py    # Basic environment demo
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── helpers.py           # Helper functions
├── ogm_cv2_window/         # OpenCV visualization tools
│   ├── __init__.py
│   ├── ui.py               # Interactive UI components
│   ├── ogm_test.py         # OGM testing interface
│   └── ui_test.py          # UI testing utilities
└── configs/                # Configuration files
    ├── config.example.yaml
    └── test_config.yaml
```

## 🚀 Quick Start

### Basic Usage

```python
import omnigibson as og
from og_nav import NavigationInterface

# Create environment
env = og.Environment(your_config)
robot = env.robots[0]

# Initialize navigation
navigator = NavigationInterface(env, robot)
navigator.setup()

# Set goal and navigate
navigator.set_goal((2.0, 2.0))

# Main loop
while True:
    action = navigator.update()
    env.step(action)
```

### Advanced Usage

```python
from og_nav import PathPlanner, PathTrackingController, OGMGenerator

# Create components separately
planner = PathPlanner(env, robot=robot, visualize=True)
controller = PathTrackingController(robot=robot)
mapper = OGMGenerator(resolution=0.1)

# Use components individually
path = planner.plan_path(start_pos, goal_pos)
controller.set_path(path)
action, arrived = controller.control()
```

## 🎯 Key Features

### 🧭 Navigation Interface
- **Unified API**: Simple, clean interface for robot navigation
- **Flexible Configuration**: Easy setup with customizable parameters
- **State Management**: Automatic path planning and goal tracking

### 🗺️ Path Planning
- **Visual Markers**: Interactive waypoint visualization
- **Keyboard Controls**: Real-time path planning controls
- **Collision Checking**: Point availability validation

### 🎮 Control Systems
- **Pure Pursuit**: Advanced path following algorithm
- **PID Control**: Precise heading and velocity control
- **Dynamic Action**: Adaptive to changing robot configurations

### 🗺️ Mapping
- **Occupancy Grids**: Real-time map generation
- **Interactive UI**: Parameter tuning with visual feedback
- **Multi-format Export**: Support for various map formats

## 📋 Demos

Run the included demonstrations:

```bash
# Simple navigation demo
python -m og_nav.demos.navigation_demo

# Full planning and tracking demo
python -m og_nav.demos.planning_tracking_demo

# Basic environment demo
python -m og_nav.demos.simple_environment_demo
```

## 🛠️ Development

### Module Organization

- **core/**: Essential components used throughout the package
- **planning/**: Path planning algorithms and utilities
- **control/**: Robot control and path following systems
- **mapping/**: Occupancy grid and map generation
- **demos/**: Complete working examples
- **utils/**: Helper functions and utilities

### Code Standards

- **Type Hints**: All functions include proper type annotations
- **Documentation**: Comprehensive docstrings for all public APIs
- **Error Handling**: Robust error handling and user feedback
- **Modular Design**: Clean separation of concerns

## 📦 Dependencies

- OmniGibson
- PyTorch
- NumPy
- OpenCV (for visualization)
- Matplotlib (for plotting)

## 🔄 Version History

- **v1.0.0**: Initial release with modular architecture
  - Reorganized code structure
  - Unified navigation interface
  - Comprehensive demos
  - Improved documentation
