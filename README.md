# OG Nav - Modular Navigation System

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/og_nav.svg)](https://pypi.org/project/og_nav/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OmniGibson](https://img.shields.io/badge/OmniGibson-Compatible-green.svg)](https://github.com/StanfordVL/OmniGibson)

A comprehensive, modular navigation system designed for robot navigation in OmniGibson environments. This package provides a clean, well-structured interface for path planning, robot control, occupancy grid mapping, and visualization.

## 🚀 Features

### 🧭 **Unified Navigation Interface**
- Clean, intuitive API for seamless robot navigation
- Automatic path planning and goal tracking
- Real-time navigation state management
- Interactive keyboard controls for manual path planning

### 🗺️ **Advanced Path Planning**
- Integration with OmniGibson's built-in path planning
- Point availability validation and collision checking
- Visual waypoint markers with interactive controls
- Automatic nearest available point finding

### 🎮 **Sophisticated Control Systems**
- **Pure Pursuit Algorithm**: Advanced path following with configurable lookahead
- **PID Controllers**: Precise velocity and heading control
- **Dynamic Action Management**: Automatic action tensor generation for Tiago robot
- Support for both base movement and arm pose management

### 🗺️ **Occupancy Grid Mapping**
- Real-time occupancy grid generation using OmniGibson
- Configurable map resolution and update strategies
- Automatic robot occlusion handling for accurate mapping
- Export capabilities for various map formats

### 🎨 **Rich Visualization**
- Interactive 3D markers for start/goal/waypoints
- Real-time path visualization in OmniGibson viewer
- Keyboard shortcuts for intuitive navigation control
- Customizable marker colors and sizes

### ⚙️ **Flexible Configuration System**
- YAML-based configuration with intelligent defaults
- Module-specific configurations with override hierarchy
- Constructor arguments > YAML config > Module defaults
- Automatic visualization marker generation

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- OmniGibson (latest version)
- NVIDIA GPU with CUDA support (recommended)

### Install from PyPI (Recommended)

```bash
pip install og_nav
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/Gonglitian/og_nav.git
cd og_nav

# Install in development mode
pip install -e .
```

### Dependencies
The package automatically installs the following dependencies:
- `omnigibson` - Physics simulation and robotics framework
- `torch` - Deep learning framework (used for tensor operations)
- `numpy` - Numerical computing
- `opencv-python` - Computer vision and image processing
- `matplotlib` - Plotting and visualization
- `pyyaml` - YAML configuration file support

## 📖 Quick Start

### Basic Navigation Example

```python
import omnigibson as og
from og_nav import NavigationInterface
from og_nav.core.config_loader import NavigationConfig

# Create environment
config_path = "og_nav/configs/navigation_config.yaml"
nav_config = NavigationConfig(config_path=config_path)
env = og.Environment(configs=nav_config.get_omnigibson_config())
robot = env.robots[0]

# Initialize navigation system
navigator = NavigationInterface(env, robot, nav_config.og_nav_config)

# Set navigation goal
navigator.set_goal((2.0, 3.0))  # Move to position (2.0, 3.0)

# Main navigation loop
while not navigator.is_arrived():
    navigator.update()
    env.step([])

print("🎉 Navigation completed!")
```

### Advanced Usage with Custom Configuration

```python
from og_nav import NavigationInterface, NavigationConfig
from og_nav.planning import PathPlanner
from og_nav.control import PathTrackingController
from og_nav.mapping import OGMGenerator

# Custom configuration
custom_config = {
    'controller': {
        'lookahead_distance': 0.8,
        'cruise_speed': 0.7,
        'max_angular_vel': 0.3
    },
    'ogm': {
        'resolution': 0.05
    },
    'visualization': {
        'enable': True,
        'n_waypoints': 100
    }
}

# Initialize with custom config
navigator = NavigationInterface(env, robot, custom_config)

# Use individual components
planner = PathPlanner(env, robot, config=custom_config.get('planning', {}))
controller = PathTrackingController(robot, config=custom_config['controller'])
mapper = OGMGenerator(config=custom_config['ogm'])
```

## 🎯 Interactive Controls

When running with visualization enabled, use these keyboard shortcuts in the OmniGibson viewer:

- **Z** - Set start point at camera position
- **X** - Set goal point at camera position  
- **V** - Plan path between start and goal
- **C** - Clear all waypoints and markers

## 📁 Project Structure

```
og_nav/
├── __init__.py                    # Main package interface
├── core/                          # Core navigation components
│   ├── navigation.py             # Main NavigationInterface class
│   ├── config_loader.py          # Unified configuration management
│   └── constants.py              # System constants
├── planning/                      # Path planning algorithms
│   ├── path_planning.py          # PathPlanner with OmniGibson integration
│   └── utils.py                  # Planning utilities
├── control/                       # Robot control systems
│   ├── controllers.py            # Pure Pursuit and PID controllers
│   └── control_utils.py          # Joint and action management utilities
├── mapping/                       # Occupancy grid mapping
│   └── occupancy_grid.py         # OGMGenerator for real-time mapping
├── demos/                         # Example demonstrations
│   └── navigation_demo.py        # Complete navigation demo
├── configs/                       # Configuration files
│   ├── navigation_config.yaml    # Main navigation configuration
│   └── config.example.yaml       # Example configuration template
├── ogm_cv2_window/               # OpenCV visualization tools
│   └── ui.py                     # Interactive UI components
└── assets/                        # Documentation and resources
    ├── *.png                     # Example images
    └── *.md                      # Technical documentation
```

## ⚙️ Configuration

### YAML Configuration Structure

```yaml
og_nav:
  # Occupancy Grid Mapping
  ogm:
    resolution: 0.1               # Map resolution in meters
    
  # Path Planning  
  planning:
    # Algorithm-specific parameters
    
  # Path Tracking Controller
  controller:
    lookahead_distance: 0.5       # Pure pursuit lookahead (m)
    cruise_speed: 0.5             # Forward speed (m/s)
    max_angular_vel: 0.2          # Max rotation speed (rad/s)
    waypoint_threshold: 0.2       # Arrival threshold (m)
    
  # Robot Configuration
  robot:
    nav_arm_pose: [1.5, 1.5, 0, 2.3, 0, -1.4, 0]  # Arm pose during navigation
    
  # Visualization
  visualization:
    enable: true
    n_waypoints: 50               # Number of waypoint markers
    start_marker_color: [0, 1, 0, 1]  # Green
    goal_marker_color: [1, 0, 0, 1]   # Red
    waypoint_color: [0, 0, 1, 1]      # Blue

# Standard OmniGibson configuration
scene:
  type: InteractiveTraversableScene
  scene_model: Pomaria_1_int
  
robots:
  - type: Tiago
    position: [-2, -0.3, 0]
    # ... robot configuration
```

### Configuration Priority

The system follows a clear configuration priority hierarchy:

1. **Constructor Arguments** - Highest priority
2. **YAML Configuration** - Medium priority  
3. **Module Defaults** - Fallback defaults

## 🤖 Robot Support

Currently optimized for **Tiago robot** with:
- Differential drive base control
- Dual-arm manipulation capabilities
- Automatic arm pose management during navigation
- Base action tensor management (indices 0-2 for x, y, rotation)

Support for additional robots can be added by extending the base controller classes.

## 🧪 Running Demos

Explore the package capabilities with included demonstrations:

```bash
# Basic navigation demo with sequential goal visiting
python -m og_nav.demos.navigation_demo

# Test configuration system
python -c "from og_nav.core.config_loader import NavigationConfig; print('Config system working!')"

# Test with custom YAML config  
python -c "from og_nav.core.config_loader import NavigationConfig; config = NavigationConfig(config_path='og_nav/configs/navigation_config.yaml'); print('YAML config loaded successfully!')"
```

## 🏗️ Development

### Code Architecture Philosophy

- **Modular Design**: Clean separation between planning, control, mapping, and visualization
- **Configuration-Driven**: All behavior configurable through YAML files or code
- **Extensible**: Easy to add new planners, controllers, or robot types
- **Type-Safe**: Comprehensive type hints throughout the codebase
- **Well-Documented**: Detailed docstrings and inline documentation

### Adding New Components

To add a new path planner:

```python
from og_nav.planning.path_planning import PathPlanner

class MyCustomPlanner(PathPlanner):
    @staticmethod
    def get_default_cfg():
        return {
            'algorithm': 'my_algorithm',
            'param1': 1.0,
            'param2': True
        }
    
    def __init__(self, env, robot=None, config=None):
        super().__init__(env, robot, config)
        # Your custom initialization
        
    def plan_path(self):
        # Your custom planning logic
        pass
```

### Testing

```bash
# Run configuration tests
python -c "from og_nav.core.config_loader import NavigationConfig; NavigationConfig().test_config_system()"

# Run demo to test complete pipeline
python -m og_nav.demos.navigation_demo
```

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)** - Comprehensive architecture documentation for AI development
- **[og_nav/README.md](og_nav/README.md)** - Package-specific technical documentation
- **[og_nav/assets/](og_nav/assets/)** - Technical analysis and robot documentation

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the existing code style
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Standards

- **Type Hints**: All public functions must include type annotations
- **Docstrings**: Comprehensive documentation for all public APIs
- **Configuration**: All behavior should be configurable
- **Error Handling**: Robust error handling with informative messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[OmniGibson](https://github.com/StanfordVL/OmniGibson)** - Physics simulation framework
- **[Stanford Vision and Learning Lab](https://svl.stanford.edu/)** - OmniGibson development team
- **[PyTorch](https://pytorch.org/)** - Deep learning framework used for tensor operations

## 🔗 Links

- **Homepage**: [https://github.com/Gonglitian/og_nav](https://github.com/Gonglitian/og_nav)
- **Documentation**: [og_nav/README.md](og_nav/README.md)
- **Issues**: [https://github.com/Gonglitian/og_nav/issues](https://github.com/Gonglitian/og_nav/issues)
- **OmniGibson**: [https://github.com/StanfordVL/OmniGibson](https://github.com/StanfordVL/OmniGibson)

---

**Built with ❤️ for the robotics community**