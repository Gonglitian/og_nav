"""Centralized arrival state management for navigation controllers."""

from typing import List, Tuple
import torch as th


class ArrivalState:
    """
    Centralized arrival state management class.
    
    This class encapsulates all logic related to determining whether a robot
    has arrived at its destination, providing a single source of truth for
    arrival status throughout the navigation system.
    """
    
    def __init__(self):
        """Initialize arrival state manager."""
        self._arrived = False
        self._arrival_logged = False
        
    def update(self, current_pos: th.Tensor, path: List[Tuple[float, float]], 
               threshold: float, current_target_idx: int = None) -> None:
        """
        Update arrival status based on current robot position and path.
        
        Args:
            current_pos: Current robot position [x, y]
            path: List of waypoint coordinates [(x, y), ...]
            threshold: Distance threshold for arrival detection
            current_target_idx: Optional current waypoint index for additional checks
        """
        if not path:
            self._arrived = False
            return
            
        # Check if we've reached the final destination
        final_target = path[-1]
        distance_to_final = self._calculate_distance(current_pos, final_target)
        
        # Additional check: if we have current_target_idx, check if all waypoints visited
        all_waypoints_visited = (current_target_idx is not None and 
                               current_target_idx >= len(path))
        
        # Arrival condition: either close to final target OR all waypoints visited
        if distance_to_final < threshold or all_waypoints_visited:
            if not self._arrival_logged:
                if all_waypoints_visited:
                    print(f"✓ Path completed: visited all {len(path)} waypoints")
                else:
                    print(f"✓ Arrived at final target: distance={distance_to_final:.3f}m < threshold={threshold}m")
                self._arrival_logged = True
            self._arrived = True
        else:
            self._arrived = False
    
    def is_arrived(self) -> bool:
        """
        Check if robot has arrived at destination.
        
        Returns:
            True if robot has arrived, False otherwise
        """
        return self._arrived
    
    def reset(self) -> None:
        """Reset arrival state for new path or target."""
        self._arrived = False
        self._arrival_logged = False
        
    def _calculate_distance(self, pos1: th.Tensor, pos2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between robot position and target.
        
        Args:
            pos1: Robot position tensor [x, y]
            pos2: Target position tuple (x, y)
            
        Returns:
            Distance in meters
        """
        dx = pos1[0].item() - pos2[0]
        dy = pos1[1].item() - pos2[1]
        return (dx * dx + dy * dy) ** 0.5