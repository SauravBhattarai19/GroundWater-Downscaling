#!/usr/bin/env python3
"""
Centralized Configuration Manager for GRACE Downscaling Pipeline

This module provides a single point of configuration management for all pipeline components.
All modules (features.py, model_manager.py, data_loader.py, utils.py) should use this 
to load configuration settings consistently.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Centralized configuration manager for the GRACE downscaling pipeline.
    
    Features:
    - Loads configuration from YAML file
    - Provides default values for all settings
    - Validates configuration parameters  
    - Supports environment variable overrides
    - Caches loaded configuration for performance
    """
    
    _instance = None
    _config = None
    
    def __new__(cls, config_path: str = "src/config.yaml"):
        """Singleton pattern to ensure consistent config across modules."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str = "src/config.yaml"):
        """Initialize configuration manager."""
        if self._config is None:  # Only load once
            self.config_path = Path(config_path)
            self._load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with all required settings."""
        return {
            # Feature processing settings
            'feature_processing': {
                'correlation_threshold': 0.9,
                'remove_correlated_features': True,
                'max_features': None,
                'feature_selection_method': 'correlation',
                'scaling_method': 'robust',  # standard, robust, minmax, none
                'add_temporal_lags': True,
                'lag_months': [1, 3, 6],
                'add_seasonal_features': True,
                'pca_components': None,
                'chunk_size': 50,
                'validate_units': True,
                'models_needing_scaling': ['nn', 'svr'],
                'models_tree_based': ['rf', 'xgb', 'lgb', 'catb', 'gbr'],
                'enable_categorical_encoding': False,
                'scientific_resampling': True
            },
            
            # Model configuration
            'models': {
                'enabled': ['nn', 'rf', 'xgb'],
                'test_size': 0.2,
                'cross_validation': True,
                'ensemble': True,
                'selection_metric': 'test_r2',
                'hyperparameters': {
                    'rf': {
                        'n_estimators': 50,
                        'max_depth': 15,
                        'min_samples_split': 50,
                        'min_samples_leaf': 20,
                        'max_features': 0.3,
                        'max_samples': 0.8
                    },
                    'xgb': {
                        'n_estimators': 250,
                        'max_depth': 10,
                        'learning_rate': 0.08,
                        'subsample': 0.85
                    },
                    'lgb': {
                        'n_estimators': 250,
                        'max_depth': 10,
                        'learning_rate': 0.08,
                        'subsample': 0.85
                    },
                    'nn': {
                        'hidden_layer_sizes': [512, 256, 128, 64],
                        'max_iter': 1000,
                        'early_stopping': True
                    }
                },
                'advanced': {
                    'enable_tuning': False,
                    'tuning_trials': 50,
                    'feature_selection': False,
                    'save_feature_importance': True,
                    'save_predictions': True
                }
            },
            
            # Scientific processing options
            'scientific': {
                'resampling': True,
                'grace_method': 'gaussian',  # gaussian, conservative, bilinear
                'categorical_encoding': False,
                'use_weight_mask': True,
                'grace_data_type': 'mascon',  # mascon, raw_grace
                'mascon_resolution_km': 55.66
            },
            
            # Pipeline settings
            'pipeline': {
                'groundwater_model': 'best',
                'fallback_models': ['nn', 'xgb', 'ensemble'],
                'skip_training_if_exists': False
            },
            
            # Download settings
            'download': {
                'default_datasets': ['grace', 'gldas', 'chirps', 'terraclimate', 'modis', 'dem', 'openlandmap', 'landscan'],
                'default_region': 'mississippi',
                'regions': {
                    'mississippi': {
                        'name': 'Mississippi River Basin',
                        'lat_min': 28.84,
                        'lat_max': 49.74,
                        'lon_min': -113.94,
                        'lon_max': -77.84
                    },
                    'kansas': {
                        'name': 'Kansas Test Area',
                        'lat_min': 37.0,
                        'lat_max': 39.0,
                        'lon_min': -99.0,
                        'lon_max': -96.0
                    }
                }
            },
            
            # Path configuration
            'paths': {
                'raw_data': 'data/raw',
                'processed_data': 'data/processed',
                'models': 'models',
                'results': 'results',
                'figures': 'figures',
                'grace_dir': 'data/raw/grace',
                'feature_stack': 'data/processed/feature_stack.nc',
                'base_dir': 'data/raw',
                'input_dirs': [
                    'data/raw/gldas/SoilMoi0_10cm_inst',
                    'data/raw/gldas/SoilMoi10_40cm_inst', 
                    'data/raw/gldas/SoilMoi40_100cm_inst',
                    'data/raw/gldas/SoilMoi100_200cm_inst',
                    'data/raw/gldas/Evap_tavg',
                    'data/raw/gldas/SWE_inst',
                    'data/raw/chirps',
                    'data/raw/terraclimate/pr',
                    'data/raw/terraclimate/tmmn', 
                    'data/raw/terraclimate/tmmx',
                    'data/raw/terraclimate/aet',
                    'data/raw/terraclimate/def',
                    'data/raw/modis_land_cover',
                    'data/raw/openlandmap',
                    'data/raw/usgs_dem'
                ]
            },
            
            # Regional settings
            'region': {
                'name': 'Mississippi River Basin',
                'lat_min': 28.84,
                'lat_max': 49.74,
                'lon_min': -113.94,
                'lon_max': -77.84
            },
            
            # Target resolution and CRS
            'resolution': 0.1,
            'target_crs': 'EPSG:4326',
            'bbox': {
                'left': -100.05,
                'right': -81.63,
                'top': 49.58,
                'bottom': 27.56
            },
            
            # GRACE configuration
            'grace_native_resolution_km': 55.66,
            'grace_data_type': 'mascon'
        }
    
    def _load_config(self):
        """Load configuration from YAML file with defaults."""
        logger.info(f"Loading configuration from {self.config_path}")
        
        # Start with defaults
        self._config = self._get_default_config()
        
        # Try to load from YAML file
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                
                if yaml_config:
                    # Deep merge YAML config with defaults
                    self._config = self._deep_merge(self._config, yaml_config)
                    logger.info("✅ Configuration loaded successfully")
                else:
                    logger.warning("⚠️ Config file is empty, using defaults")
                    
            except Exception as e:
                logger.error(f"❌ Error loading config file: {e}")
                logger.info("Using default configuration")
        else:
            logger.warning(f"⚠️ Config file not found at {self.config_path}, using defaults")
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate configuration
        self._validate_config()
    
    def _deep_merge(self, default: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries, with override taking precedence."""
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Examples: GRACE_MODEL_RF_N_ESTIMATORS, GRACE_CORRELATION_THRESHOLD
        env_prefix = "GRACE_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_path = key[len(env_prefix):].lower().split('_')
                self._set_nested_value(self._config, config_path, self._parse_env_value(value))
    
    def _set_nested_value(self, config: Dict, path: List[str], value: Any):
        """Set a nested dictionary value using a path."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        if path:
            current[path[-1]] = value
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool, List]:
        """Parse environment variable value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # List (comma-separated)
        if ',' in value:
            return [self._parse_env_value(v.strip()) for v in value.split(',')]
        
        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value  # String
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate correlation threshold
        corr_threshold = self.get('feature_processing.correlation_threshold')
        if not 0 <= corr_threshold <= 1:
            raise ValueError(f"correlation_threshold must be between 0 and 1, got {corr_threshold}")
        
        # Validate models
        available_models = ['rf', 'xgb', 'lgb', 'catb', 'nn', 'svr', 'gbr']
        enabled_models = self.get('models.enabled')
        invalid_models = [m for m in enabled_models if m not in available_models]
        if invalid_models:
            logger.warning(f"⚠️ Invalid models in config: {invalid_models}")
        
        # Validate paths exist
        for path_name in ['raw_data', 'processed_data']:
            path = Path(self.get(f'paths.{path_name}'))
            if not path.exists():
                logger.info(f"Creating directory: {path}")
                path.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Configuration validation complete")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'models.rf.n_estimators')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key '{key}' not found")
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        current = self._config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        base_config = {
            'enabled': model_name in self.get('models.enabled', []),
            'needs_scaling': model_name in self.get('feature_processing.models_needing_scaling', []),
            'is_tree_based': model_name in self.get('feature_processing.models_tree_based', [])
        }
        
        # Add hyperparameters if available
        hyperparams = self.get(f'models.hyperparameters.{model_name}', {})
        base_config.update(hyperparams)
        
        return base_config
    
    def get_paths(self) -> Dict[str, str]:
        """Get all path configurations."""
        return self.get('paths', {})
    
    def get_region_config(self, region_name: str = None) -> Dict[str, Any]:
        """Get region configuration."""
        if region_name is None:
            region_name = self.get('download.default_region', 'mississippi')
        
        return self.get(f'download.regions.{region_name}', self.get('region', {}))
    
    def get_datasets(self) -> List[str]:
        """Get list of datasets to process."""
        return self.get('download.default_datasets', [])
    
    def save_config(self, output_path: str = None):
        """Save current configuration to YAML file."""
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✅ Configuration saved to {output_path}")
    
    def __getitem__(self, key: str):
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting."""
        self.set(key, value)


# Global configuration instance
config = ConfigManager()

# Convenience functions for common operations
def get_config(key: str = None, default: Any = None) -> Any:
    """Get configuration value (global convenience function)."""
    if key is None:
        return config._config
    return config.get(key, default)

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model configuration (global convenience function)."""
    return config.get_model_config(model_name)

def get_paths() -> Dict[str, str]:
    """Get path configurations (global convenience function)."""
    return config.get_paths()

def reload_config(config_path: str = "src/config.yaml"):
    """Reload configuration from file."""
    global config
    ConfigManager._instance = None
    ConfigManager._config = None
    config = ConfigManager(config_path)


if __name__ == "__main__":
    # Test the configuration manager
    print("🔧 Testing Configuration Manager")
    print(f"Models enabled: {get_config('models.enabled')}")
    print(f"Correlation threshold: {get_config('feature_processing.correlation_threshold')}")
    print(f"RF hyperparameters: {get_model_config('rf')}")
    print(f"Paths: {get_paths()}")
    print("✅ Configuration manager test complete")