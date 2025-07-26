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
from og_nav.core import NavigationInterface

# Import other commonly used components
from og_nav.core.config_loader import NavigationConfig
from og_nav.planning import PathPlanner, is_point_available, find_nearest_available_point
from og_nav.control import PathTrackingController, PIDController
from og_nav.mapping import OGMGenerator

__all__ = [
    # Main interface
    "NavigationInterface",
    # Core components
    "NavigationConfig",
    # Planning
    "PathPlanner",
    "is_point_available",
    "find_nearest_available_point",
    # Control
    "PathTrackingController",
    "PIDController", 
    # Mapping
    "OGMGenerator",
]

# Import version from setuptools_scm generated file
try:
    from og_nav._version import __version__
except ImportError:
    # Fallback for development installs without tags
    __version__ = "0.0.0+unknown" 