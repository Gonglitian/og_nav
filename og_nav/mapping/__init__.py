"""Mapping module for OmniGibson Navigation.

This module provides mapping functionality including:
- Occupancy grid map generation
"""

from .occupancy_grid import OGMGenerator

__all__ = [
    "OGMGenerator",
] 