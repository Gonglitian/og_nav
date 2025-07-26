# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Installation and Setup
```bash
# Install the package in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt

# Run demos
python -m og_nav.demos.navigation_demo
python -m og_nav.demos.planning_tracking_demo
```

### Configuration Testing
```bash
# Test configuration system
python -c "from og_nav.core.config_loader import NavigationConfig; print('Config system working')"

# Test with YAML config
python -c "from og_nav.core.config_loader import NavigationConfig; nav_config = NavigationConfig(config_path='og_nav/configs/navigation_config.yaml'); print('YAML config loaded successfully')"
```

## Code Architecture

### Core Architecture Overview
The codebase implements a modular navigation system for robot navigation in OmniGibson environments. The architecture follows a layered design with clear separation of concerns:

**Configuration System (og_nav/core/config_loader.py)**
- Unified configuration management through `NavigationConfig` class
- Module-specific default configurations via `get_default_cfg()` methods
- Configuration priority: Constructor arguments > YAML config > Module defaults
- Supports both YAML file loading and direct config dict input
- Automatic visualization marker generation for OmniGibson environments

**Navigation Interface (og_nav/core/navigation.py)**
- `NavigationInterface` serves as the main entry point for navigation functionality
- Orchestrates path planning, control, mapping, and visualization components
- Handles OmniGibson environment integration and robot arm pose management
- Provides keyboard-based interactive path planning (Z=start, X=goal, V=plan, C=clear)

**Module Architecture**
Each major component follows the same initialization pattern:
```python
class ModuleClass:
    @staticmethod
    def get_default_cfg() -> Dict[str, Any]:
        return {...}  # Module-specific defaults
    
    def __init__(self, ..., config: Optional[dict] = None):
        # Merge config with defaults following priority order
```

### Key Components

**Path Planning (og_nav/planning/)**
- `PathPlanner`: Uses OmniGibson's built-in path planning with point availability checking
- Stores waypoint coordinates and provides coordinate getters for visualization
- Point validation through `is_point_available()` and `find_nearest_available_point()`

**Control Systems (og_nav/control/)**
- `PathTrackingController`: Pure Pursuit algorithm implementation
- `PIDController`: Generic PID controller with configurable gains and limits
- Action tensor management for Tiago robot base control (indices 0-2)

**Mapping (og_nav/mapping/)**
- `OGMGenerator`: Wraps OmniGibson's occupancy grid generation
- Updates environment traversability maps by temporarily moving robot to avoid occlusion
- Converts between BGR images and grayscale tensors for OmniGibson compatibility

**Configuration Flow**
1. YAML files contain both `og_nav` section and standard OmniGibson config
2. `NavigationConfig` processes YAML, splits configs, and merges defaults
3. If visualization enabled, generates marker objects and injects into OmniGibson config
4. Each module receives its specific config section during initialization

### YAML Configuration Structure
```yaml
og_nav:
  ogm:
    resolution: 0.1
  planning: {}
  controller:
    lookahead_distance: 0.5
    cruise_speed: 0.5
  robot:
    nav_arm_pose: [...]
  visualization:
    enable: true
    n_waypoints: 50

# Standard OmniGibson configuration
scene: {...}
robots: [...]
objects: []  # Visualization markers auto-injected here
```

### Robot Integration Details
- Designed specifically for Tiago robot with differential drive base
- Base control uses action tensor indices 0-2 (x, y, rotation velocities)
- Arm pose management during navigation via `nav_arm_pose` configuration
- Joint controller limits are disabled for arm controllers during initialization

### Visualization System
- Interactive keyboard controls for path planning in OmniGibson viewer
- Marker objects (start/goal/waypoints) automatically generated based on config
- Markers positioned at `hidden_position` when not in use
- Caching system for frequently accessed configuration values

### Error Handling Philosophy
- Invalid configurations raise exceptions rather than using fallback defaults
- Point availability checking with automatic nearest point finding
- Comprehensive error messages for configuration and initialization failures

### Testing and Validation
- Configuration system includes comprehensive self-tests
- Module initialization validates constructor argument priority
- Demo files serve as integration tests for the complete navigation pipeline