"""Unified configuration loader and processor for og_nav navigation system."""

import os
import copy
from typing import Dict, Any, Optional, Tuple
import yaml
import torch as th


class NavigationConfig:
    """Unified navigation configuration manager.
    
    Handles both YAML loading and configuration processing with module-specific
    default configurations and proper override priority.
    """

    def __init__(self, config_path: Optional[str] = None, og_nav_config: Optional[Dict[str, Any]] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
            og_nav_config: Pre-loaded og_nav configuration dict (takes priority over file)
        """
        if og_nav_config is not None:
            # Use provided og_nav config directly
            self.og_nav_config = self._merge_with_defaults(og_nav_config)
            self.omnigibson_config = None
        elif config_path is not None:
            # Process YAML file
            self.omnigibson_config, self.og_nav_config = self._process_config_file(config_path)
        else:
            # Use defaults only
            self.og_nav_config = self._get_default_config()
            self.omnigibson_config = None
            
        # Cache frequently accessed config sections
        self._config_cache = {}

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get the complete default configuration for og_nav."""
        return {
            'ogm': OGMConfig.get_default_cfg(),
            'planning': PlanningConfig.get_default_cfg(), 
            'controller': ControllerConfig.get_default_cfg(),
            'robot': RobotConfig.get_default_cfg(),
            'visualization': VisualizationConfig.get_default_cfg()
        }
    
    def _merge_with_defaults(self, og_nav_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user config with defaults following priority order."""
        default_config = self._get_default_config()
        return self._deep_merge(default_config, og_nav_config)
    
    def _process_config_file(self, config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Process YAML config file and split omnigibson/og_nav configs.
        
        Returns:
            Tuple[Dict, Dict]: (omnigibson_config, og_nav_config)
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")
        
        # Extract og_nav config if present
        if 'og_nav' not in cfg:
            raise KeyError("No 'og_nav' section found in configuration file")
            
        og_nav_config = cfg.pop('og_nav')
        omnigibson_config = copy.deepcopy(cfg)
        
        # Merge og_nav config with defaults
        merged_og_nav_config = self._merge_with_defaults(og_nav_config)
        
        # Handle visualization marker generation
        visualization = merged_og_nav_config.get('visualization', {})
        if visualization.get('enable', True):
            marker_objects = self._generate_marker_objects(merged_og_nav_config)
            self._merge_objects(omnigibson_config, marker_objects)
            print("✓ Visualization enabled: Generated marker objects")
        else:
            print("✓ Visualization disabled: No marker objects generated")
        
        print(f"Loaded configuration from: {config_path}")
        return omnigibson_config, merged_og_nav_config

    def _deep_merge(self, default: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key_path: str, default=None):
        """Get og_nav configuration value using dot notation with caching.
        
        Args:
            key_path: Dot-separated path like 'controller.cruise_speed'
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Check cache first
        if key_path in self._config_cache:
            return self._config_cache[key_path]
            
        keys = key_path.split('.')
        value = self.og_nav_config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = default
                break
                
        # Cache the result
        self._config_cache[key_path] = value
        return value

    def get_tensor(self, key_path: str, dtype=th.float32):
        """Get configuration value as PyTorch tensor.
        
        Args:
            key_path: Dot-separated path
            dtype: PyTorch data type
            
        Returns:
            PyTorch tensor
        """
        value = self.get(key_path)
        if value is None:
            return None
        return th.as_tensor(value, dtype=dtype)

    # Module configuration getters
    def get_ogm_config(self) -> Dict[str, Any]:
        """Get OGM module configuration."""
        return self.get('ogm', {})
    
    def get_planning_config(self) -> Dict[str, Any]:
        """Get planning module configuration."""
        return self.get('planning', {})
    
    def get_controller_config(self) -> Dict[str, Any]:
        """Get controller module configuration."""
        return self.get('controller', {})
    
    def get_robot_config(self) -> Dict[str, Any]:
        """Get robot module configuration."""
        return self.get('robot', {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization module configuration."""
        return self.get('visualization', {})
    
    def get_omnigibson_config(self) -> Optional[Dict[str, Any]]:
        """Get the processed OmniGibson configuration."""
        return self.omnigibson_config
    
    # Convenience methods for common values
    def get_nav_arm_pose(self) -> th.Tensor:
        return self.get_tensor('robot.nav_arm_pose')
    
    def get_reset_pose(self) -> th.Tensor:
        return self.get_tensor('robot.reset_pose')
    
    @staticmethod
    def _generate_marker_objects(og_nav_config: Dict[str, Any]) -> list:
        """Generate marker objects for visualization."""
        marker_objects = []
        visualization = og_nav_config.get('visualization', {})
        
        hidden_position = visualization.get('hidden_position', [0, 0, 100])
        n_waypoints = visualization.get('n_waypoints', 50)
        
        # Generate start marker
        marker_objects.append({
            "type": "PrimitiveObject",
            "name": "start_point",
            "position": hidden_position,
            "primitive_type": "Sphere",
            "visual_only": True,
            "radius": visualization.get('marker_radius', 0.1),
            "rgba": visualization.get('start_marker_color', [0, 1, 0, 1])
        })
        
        # Generate goal marker  
        marker_objects.append({
            "type": "PrimitiveObject",
            "name": "end_point",
            "position": hidden_position,
            "primitive_type": "Sphere",
            "visual_only": True,
            "radius": visualization.get('marker_radius', 0.1),
            "rgba": visualization.get('goal_marker_color', [1, 0, 0, 1])
        })
        
        # Generate waypoint markers
        for i in range(n_waypoints):
            marker_objects.append({
                "type": "PrimitiveObject",
                "name": f"waypoint_{i}",
                "position": hidden_position,
                "primitive_type": "Sphere",
                "radius": visualization.get('waypoint_radius', 0.05),
                "visual_only": True,
                "rgba": visualization.get('waypoint_color', [0, 0, 1, 1])
            })
            
        return marker_objects
    
    @staticmethod
    def _merge_objects(cfg: Dict[str, Any], marker_objects: list) -> None:
        """Merge marker objects into configuration."""
        if 'objects' not in cfg:
            cfg['objects'] = []
        cfg['objects'].extend(marker_objects)


# Module-specific configuration classes
class OGMConfig:
    """OGM module configuration."""
    
    @staticmethod
    def get_default_cfg() -> Dict[str, Any]:
        return {
            'resolution': 0.1
        }


class PlanningConfig:
    """Path planning module configuration."""
    
    @staticmethod
    def get_default_cfg() -> Dict[str, Any]:
        return {
            # Path planning algorithm parameters will go here
        }


class ControllerConfig:
    """Path tracking controller configuration."""
    
    @staticmethod
    def get_default_cfg() -> Dict[str, Any]:
        return {
            'lookahead_distance': 0.5,
            'cruise_speed': 0.5,
            'max_angular_vel': 0.2,
            'waypoint_threshold': 0.2
        }


class RobotConfig:
    """Robot configuration."""
    
    @staticmethod
    def get_default_cfg() -> Dict[str, Any]:
        return {
            'reset_pose': [
                0, 0, 0, 0, 0, 0,  # Base joints
                0,  # Trunk
                1.5, 1.5, 0, 1.5, 1.5, 0, 0.0, 0.0, 2.3, 2.3, 0, 0, -1.4, -1.4, 0, 0,  # Arms
                0.045, 0.045, 0.045, 0.045  # Grippers
            ],
            'nav_arm_pose': [1.5, 1.5, 0, 2.3, 0, -1.4, 0]
        }


class VisualizationConfig:
    """Visualization configuration."""
    
    @staticmethod
    def get_default_cfg() -> Dict[str, Any]:
        return {
            'enable': True,
            'n_waypoints': 50,
            'hidden_position': [0, 0, 100],
            'start_marker_color': [0, 1, 0, 1],
            'goal_marker_color': [1, 0, 0, 1],
            'waypoint_color': [0, 0, 1, 1],
            'marker_radius': 0.1,
            'waypoint_radius': 0.05,
            'marker_height': 0.1
        }