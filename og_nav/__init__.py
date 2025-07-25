"""OmniGibson Navigation Package.

This package provides navigation utilities for robot navigation in OmniGibson environments.

The package is organized into the following modules:
- core: Core navigation components and interfaces
- planning: Path planning algorithms
- control: Robot control and path tracking
- mapping: Occupancy grid mapping
- demos: Example usage demonstrations
"""

# Import main navigation interface
from .core import NavigationInterface

# Import other commonly used components
from .core import NavigationUtils, NavConstants
from .planning import PathPlanner
from .control import PathTrackingController, PIDController
from .mapping import OGMGenerator

__all__ = [
    # Main interface
    "NavigationInterface",
    # Core components
    "NavigationUtils", 
    "NavConstants",
    # Planning
    "PathPlanner",
    # Control
    "PathTrackingController",
    "PIDController", 
    # Mapping
    "OGMGenerator",
]

__version__ = "1.0.0" 