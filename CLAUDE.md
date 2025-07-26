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
- `PathTrackingController`: Pure Pursuit algorithm implementation with simplified control() method
- `PIDController`: Generic PID controller with configurable gains and limits
- `ArrivalState`: Centralized arrival status management for consistent state tracking
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

## Recent Architecture Improvements

### Arrival State Management Refactoring (2025)
The navigation system underwent a major refactoring to eliminate redundant arrival detection logic and improve code clarity:

**Problem Solved:**
- Multiple conflicting arrival state sources (NavigationInterface.arrived, PathTrackingController.control() return value, controller.is_arrived())
- Inconsistent state synchronization between components
- Complex and unclear arrival detection logic scattered across multiple classes

**Solution Implemented:**

**1. ArrivalState Class (og_nav/control/arrival_state.py)**
```python
class ArrivalState:
    """Centralized arrival state management"""
    def update(self, current_pos, path, threshold, current_target_idx=None):
        # Unified arrival detection logic
    def is_arrived(self) -> bool:
        # Single source of truth for arrival status
    def reset(self):
        # Clean state reset for new paths
```

**2. Simplified PathTrackingController**
- `control()` method now returns only `th.Tensor` (was `Tuple[th.Tensor, bool]`)
- Removed internal `_arrived_logged` and `_check_arrival_condition()` 
- Uses ArrivalState for all arrival detection
- Cleaner separation of control logic and state management

**3. Streamlined NavigationInterface**
- Removed redundant `self.arrived` attribute
- `update()` method simplified: `action = self.controller.control()`
- `is_arrived()` delegates to controller for single data source
- Elimination of state synchronization issues

**4. Improved Demo Code**
- Uses `navigator.is_arrived()` instead of `navigator.controller.is_arrived()`
- Follows proper encapsulation principles
- Consistent interface usage throughout

**Benefits Achieved:**
- **Single Source of Truth**: All arrival status managed by ArrivalState
- **Cleaner APIs**: Simplified method signatures and return values  
- **Better Encapsulation**: External code uses unified NavigationInterface
- **Reduced Complexity**: Centralized arrival logic, easier debugging
- **Eliminated Race Conditions**: No more state synchronization issues

**Usage Pattern:**
```python
# NavigationInterface usage (recommended)
if navigator.is_arrived():
    # Handle arrival

# PathTrackingController usage (internal)
action = controller.control()  # Only returns action tensor
if controller.is_arrived():    # Check arrival separately
    # Handle arrival
```

This refactoring significantly improved code maintainability and eliminated a major source of bugs in the navigation system.

## Development Guidelines

### API Design Principles
1. **Single Responsibility**: Each class should have one clear purpose
2. **Single Source of Truth**: Avoid duplicating state across multiple components
3. **Clean Interfaces**: Methods should have simple, predictable signatures
4. **Proper Encapsulation**: External code should use public interfaces, not access internals directly

### Code Quality Standards
- **Type Hints**: All public methods must include proper type annotations
- **Docstrings**: Comprehensive documentation for all public APIs  
- **Error Handling**: Prefer exceptions over silent failures for invalid configurations
- **Modular Design**: Clean separation between configuration, control, planning, and visualization

### Common Patterns to Follow

**Module Configuration Pattern:**
```python
class ModuleClass:
    @staticmethod
    def get_default_cfg() -> Dict[str, Any]:
        return {
            'param1': default_value,
            'param2': default_value
        }
    
    def __init__(self, ..., config: Optional[dict] = None):
        default_config = self.get_default_cfg()
        merged_config = default_config.copy()
        if config is not None:
            merged_config.update(config)
        # Use merged_config for initialization
```

**State Management Pattern:**
- Use dedicated state classes for complex state logic (like ArrivalState)
- Avoid state duplication across multiple classes
- Provide clear reset mechanisms for state objects

**Interface Design Pattern:**
- High-level interfaces (like NavigationInterface) should hide implementation details
- Delegate to specialized components rather than implementing everything internally
- Provide simple, consistent method signatures

### Testing and Validation
- Test configuration loading: `python -c "from og_nav.core.config_loader import NavigationConfig; print('Config system working')"`
- Test component initialization: Ensure all modules can be imported and initialized
- Run demos as integration tests: `python -m og_nav.demos.navigation_demo`

### Common Pitfalls to Avoid
1. **Don't access internal components directly**: Use `navigator.is_arrived()` not `navigator.controller.is_arrived()`
2. **Don't duplicate state**: If multiple classes need the same information, create a shared state manager
3. **Don't mix responsibilities**: Keep control logic separate from state management
4. **Don't ignore error handling**: Invalid configs should raise exceptions, not use fallback values silently