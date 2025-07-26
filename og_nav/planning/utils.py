"""Navigation utility functions."""

from typing import Optional, Tuple

import omnigibson as og
from omnigibson.robots.tiago import Tiago


def is_point_available(
    x: float, y: float, env: og.Environment, robot: Optional[Tiago] = None
) -> bool:
    """Check if a point is available for navigation.

    Args:
        x: X coordinate
        y: Y coordinate
        env: OmniGibson environment
        robot: Robot instance (optional)

    Returns:
        True if point is available, False otherwise
    """
    # Simple implementation - in a real scenario, this would check against the traversability map
    return True


def find_nearest_available_point(
    x: float, y: float, env: og.Environment, robot: Optional[Tiago] = None
) -> Optional[Tuple[float, float]]:
    """Find the nearest available point to the given coordinates.

    Args:
        x: X coordinate
        y: Y coordinate
        env: OmniGibson environment
        robot: Robot instance (optional)

    Returns:
        Nearest available point (x, y) or None if not found
    """
    # Simple implementation - in a real scenario, this would search the traversability map
    return (x, y)
