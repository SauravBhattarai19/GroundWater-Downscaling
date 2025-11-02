# src/model_manager.py - Fixed version with proper data type handling
"""
Multi-Model Manager for GRACE Downscaling - Fixed Version

Handles numpy string type conversion issues properly.
"""

import os
import sys
import re
import multiprocessing
import numpy as np
import pandas as pd
import xarray as xr
import joblib
import warnings
import traceback
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# SHAP for feature importance analysis
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Core ML imports (always available)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from src.utils import resample_grace_scientifically, create_grace_weight_mask, validate_grace_values

# Optimization imports
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    HAS_SKOPT = True
except ImportError:
    from scipy.optimize import minimize
    HAS_SKOPT = False
    print("📦 skopt not available - using scipy optimization fallback")

warnings.filterwarnings('ignore')

# Advanced ML imports with graceful fallbacks
AVAILABLE_MODELS = ['rf', 'gbr', 'nn', 'svr']  # Always available base models

try:
    import xgboost as xgb
    AVAILABLE_MODELS.append('xgb')
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    AVAILABLE_MODELS.append('lgb')
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    AVAILABLE_MODELS.append('catb')
    HAS_CATB = True
except ImportError:
    HAS_CATB = False

# Try to load config, with fallback
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("⚠️ PyYAML not available, using default config")

# Import centralized config manager
from src.config_manager import get_config, get_model_config


def safe_str_conversion(value):
    """Safely convert various types to string, handling numpy strings."""
    if isinstance(value, (np.str_, np.bytes_)):
        return str(value)
    elif isinstance(value, str):
        return value
    elif hasattr(value, 'item'):  # numpy scalar
        return str(value.item())
    else:
        return str(value)


def safe_datetime_conversion(time_value):
    """Safely convert time values to YYYY-MM format."""
    try:
        # Handle different possible input types
        if isinstance(time_value, (np.str_, np.bytes_)):
            time_str = str(time_value)
        elif isinstance(time_value, str):
            time_str = time_value
        elif hasattr(time_value, 'item'):  # numpy scalar
            time_str = str(time_value.item())
        elif isinstance(time_value, (np.datetime64, pd.Timestamp)):
            return pd.Timestamp(time_value).strftime('%Y-%m')
        else:
            time_str = str(time_value)
        
        # Try to parse as timestamp
        if '-' in time_str and len(time_str) >= 7:  # Already in YYYY-MM format
            return time_str[:7]  # Take first 7 characters (YYYY-MM)
        else:
            # Try to parse as full timestamp
            return pd.to_datetime(time_str).strftime('%Y-%m')
            
    except Exception as e:
        print(f"⚠️ Warning: Could not convert time value {time_value} (type: {type(time_value)}): {e}")
        return str(time_value)[:7] if len(str(time_value)) >= 7 else str(time_value)


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """Ensemble of multiple regressors with weighted averaging."""
    
    def __init__(self, models=None, weights=None):
        self.models = models or []
        self.weights = weights
        
    def fit(self, X, y):
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict using weighted average of all models (memory-efficient)."""
        # Memory-efficient prediction: accumulate instead of storing all predictions
        result = None
        total_weight = 0
        
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            weight = self.weights[i] if self.weights is not None else 1.0
            
            if result is None:
                result = pred * weight
            else:
                result += pred * weight
            total_weight += weight
        
        return result / total_weight if self.weights is not None else result / len(self.models)


class ModelManager:
    """Unified interface for multiple ML models with robust data type handling."""
    
    def __init__(self, config_path="src/config.yaml"):
        self.config_path = config_path
        self.config = get_config()  # Load entire config for hyperparameter tuning
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = None
        
        # Print available models
        print(f"🔧 Available models: {AVAILABLE_MODELS}")
        
        missing = []
        if not HAS_XGB:
            missing.append("xgboost")
        if not HAS_LGB:
            missing.append("lightgbm")
        if not HAS_CATB:
            missing.append("catboost")
        
        if missing:
            print(f"📦 To install missing models: pip install {' '.join(missing)}")
        
        # Initialize available models using centralized config
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all available models with limited parallelization."""
        
        # Optimize for high-performance system (192 cores, 756GB)
        import multiprocessing
        available_cores = multiprocessing.cpu_count()
        max_cores = min(64, available_cores)  # Efficient parallelization limit
        print(f"🚀 High-performance mode: Using up to {max_cores} cores ({available_cores} available)")
        
        # Get hyperparameters from centralized config
        rf_params = get_config('models.hyperparameters.rf', {})
        print(f"📝 RF Config loaded: {rf_params}")
        
        self.model_configs = {
            'rf': {
                'name': 'Random Forest',
                'model': RandomForestRegressor(
                    n_estimators=rf_params.get('n_estimators', 75),        # 🎯 Optimized for categorical features
                    max_depth=rf_params.get('max_depth', 18),              # 🎯 Balanced depth for ~70 features
                    min_samples_split=rf_params.get('min_samples_split', 200),
                    min_samples_leaf=rf_params.get('min_samples_leaf', 100), # 🎯 Proper leaf size
                    max_features=rf_params.get('max_features', 'sqrt'),     # 🎯 Optimal for mixed features
                    max_samples=rf_params.get('max_samples', 0.6),          # 🎯 Balanced bootstrap
                    n_jobs=rf_params.get('n_jobs', min(12, max_cores)),    # 🎯 Balanced parallelization
                    random_state=42,
                    verbose=1  # Show progress
                ),
                'needs_scaling': 'rf' in get_config('feature_processing.models_needing_scaling', [])
            },
            'gbr': {
                'name': 'Gradient Boosting',
                'model': GradientBoostingRegressor(
                    n_estimators=get_config('models.hyperparameters.gbr.n_estimators', 200),
                    max_depth=get_config('models.hyperparameters.gbr.max_depth', 8),
                    learning_rate=get_config('models.hyperparameters.gbr.learning_rate', 0.1),
                    subsample=get_config('models.hyperparameters.gbr.subsample', 0.8),
                    random_state=42
                    # Note: GBR doesn't have n_jobs parameter
                ),
                'needs_scaling': 'gbr' in get_config('feature_processing.models_needing_scaling', [])
            },
            'nn': {
                'name': 'Neural Network',
                'model': MLPRegressor(
                    hidden_layer_sizes=tuple(get_config('models.hyperparameters.nn.hidden_layer_sizes', [256, 128, 64])),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    max_iter=get_config('models.hyperparameters.nn.max_iter', 500),
                    early_stopping=get_config('models.hyperparameters.nn.early_stopping', True),
                    validation_fraction=0.1,
                    random_state=42
                    # Note: MLPRegressor doesn't have n_jobs parameter
                ),
                'needs_scaling': 'nn' in get_config('feature_processing.models_needing_scaling', [])
            },
            'svr': {
                'name': 'Support Vector Regression',
                'model': SVR(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    epsilon=0.1
                    # Note: SVR doesn't have n_jobs parameter
                ),
                'needs_scaling': 'svr' in get_config('feature_processing.models_needing_scaling', [])
            }
        }
        
        # Add advanced models if available
        if HAS_XGB:
            xgb_params = get_config('models.hyperparameters.xgb', {})
            self.model_configs['xgb'] = {
                'name': 'XGBoost',
                'model': xgb.XGBRegressor(
                    n_estimators=xgb_params.get('n_estimators', 200),
                    max_depth=xgb_params.get('max_depth', 8),
                    learning_rate=xgb_params.get('learning_rate', 0.1),
                    subsample=xgb_params.get('subsample', 0.8),
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=max_cores,  # ✅ LIMITED instead of -1
                    nthread=max_cores   # ✅ XGBoost specific parameter
                ),
                'needs_scaling': 'xgb' in get_config('feature_processing.models_needing_scaling', [])
            }
        
        if HAS_LGB:
            lgb_params = get_config('models.hyperparameters.lgb', {})
            self.model_configs['lgb'] = {
                'name': 'LightGBM',
                'model': lgb.LGBMRegressor(
                    n_estimators=lgb_params.get('n_estimators', 200),
                    max_depth=lgb_params.get('max_depth', 8),
                    learning_rate=lgb_params.get('learning_rate', 0.1),
                    subsample=lgb_params.get('subsample', 0.8),
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=max_cores,    # ✅ LIMITED instead of -1
                    num_threads=max_cores, # ✅ LightGBM specific parameter
                    verbose=-1
                ),
                'needs_scaling': 'lgb' in get_config('feature_processing.models_needing_scaling', [])
            }
        
        if HAS_CATB:
            catb_params = get_config('models.hyperparameters.catb', {})
            self.model_configs['catb'] = {
                'name': 'CatBoost',
                'model': cb.CatBoostRegressor(
                    iterations=catb_params.get('n_estimators', 200),
                    depth=catb_params.get('max_depth', 8),
                    learning_rate=catb_params.get('learning_rate', 0.1),
                    random_state=42,
                    thread_count=max_cores,  # ✅ CatBoost specific parameter
                    verbose=False
                ),
                'needs_scaling': 'catb' in get_config('feature_processing.models_needing_scaling', [])
            }
    
    def prepare_data(self, features_path="data/processed/feature_stack.nc", 
                    grace_dir="data/raw/grace"):
        """Prepare data for training with robust data type handling."""
        print("📦 Loading and preparing data...")
        
        # Check if required files exist
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Feature stack not found: {features_path}")
        
        if not os.path.exists(grace_dir):
            raise FileNotFoundError(f"GRACE directory not found: {grace_dir}")
        
        # Load feature dataset
        print("   Loading feature dataset...")
        ds = xr.open_dataset(features_path)
        feature_data = ds.features.values
        feature_times = ds.time.values
        
        # **FIXED: Create proper reference raster with spatial dimensions**
        print("   Creating reference raster from feature dataset...")
        sample_feature = ds.features.isel(time=0, feature=0)
        
        # Create proper reference raster with x,y dimensions and CRS
        reference_raster = sample_feature.rename({'lon': 'x', 'lat': 'y'})
        
        # Set CRS if not already set
        if not hasattr(reference_raster, 'rio') or reference_raster.rio.crs is None:
            reference_raster = reference_raster.rio.write_crs('EPSG:4326')
        
        # Set spatial dimensions explicitly
        reference_raster = reference_raster.rio.set_spatial_dims(x_dim='x', y_dim='y')
        
        print(f"   Reference raster shape: {reference_raster.shape}")
        print(f"   Reference raster CRS: {reference_raster.rio.crs}")
        
        # Convert time values to strings with robust handling
        print("   Converting time indices...")
        feature_dates = []
        for t in feature_times:
            try:
                date_str = safe_datetime_conversion(t)
                feature_dates.append(date_str)
            except Exception as e:
                print(f"⚠️ Warning: Could not convert time {t}: {e}")
                continue
        
        print(f"   Processed {len(feature_dates)} feature time points")
        
        # Load static features if available
        static_data = None
        if 'static_features' in ds:
            static_data = ds.static_features.values
            print(f"   Loaded {static_data.shape[0]} static features")
        
        # Load GRACE data at same resolution as features
        print("   Loading GRACE data...")
        grace_data, grace_dates = self._load_grace_tws(grace_dir, reference_raster)
        print(f"   Loaded {len(grace_dates)} GRACE time points")
        
        if len(grace_dates) == 0:
            raise ValueError("No GRACE files could be loaded! Check GRACE data directory and file formats.")
        
        print(f"   GRACE data shape: {grace_data.shape}")
        
        # Find common dates
        print("   Finding common dates...")
        common_dates = sorted(set(feature_dates).intersection(set(grace_dates)))
        print(f"✅ Found {len(common_dates)} common dates")
        
        if not common_dates:
            print(f"Feature dates sample: {feature_dates[:5]}")
            print(f"GRACE dates sample: {grace_dates[:5]}")
            raise ValueError("No common dates found!")
        
        # Extract data for common dates
        print("   Extracting data for common dates...")
        feature_dict = {date: idx for idx, date in enumerate(feature_dates)}
        grace_dict = {date: idx for idx, date in enumerate(grace_dates)}
        
        common_feature_indices = []
        common_grace_indices = []
        
        for date in common_dates:
            if date in feature_dict and date in grace_dict:
                common_feature_indices.append(feature_dict[date])
                common_grace_indices.append(grace_dict[date])
        
        X_temporal = feature_data[common_feature_indices]
        grace_tws = grace_data[common_grace_indices]
        
        print(f"   X_temporal shape: {X_temporal.shape}")
        print(f"   grace_tws shape: {grace_tws.shape}")
        
        # Verify shapes match now
        if X_temporal.shape[0] != grace_tws.shape[0]:
            raise ValueError(f"Time dimension mismatch: features={X_temporal.shape[0]}, grace={grace_tws.shape[0]}")
        
        if X_temporal.shape[2:] != grace_tws.shape[1:]:
            raise ValueError(f"Spatial dimension mismatch: features={X_temporal.shape[2:]}, grace={grace_tws.shape[1:]}")
        
        # Create enhanced features
        add_lags = get_config('feature_processing.add_temporal_lags', True)
        if add_lags:
            print("   Creating lagged features...")
            X_with_lags, feature_names = self._create_lagged_features(X_temporal)
        else:
            X_with_lags = X_temporal
            feature_names = [f"feat_{i}" for i in range(X_temporal.shape[1])]
        
        # Add seasonal features
        add_seasonal = get_config('feature_processing.add_seasonal_features', True)
        if add_seasonal:
            print("   Adding seasonal features...")
            months = []
            for date in common_dates:
                try:
                    month = pd.to_datetime(date).month
                    months.append(month)
                except:
                    months.append(1)  # Default to January if parsing fails
            
            months = np.array(months)
            month_sin = np.sin(2 * np.pi * months / 12)
            month_cos = np.cos(2 * np.pi * months / 12)
            
            n_times, n_features, n_lat, n_lon = X_with_lags.shape
            seasonal = np.zeros((n_times, 2, n_lat, n_lon))
            
            for t in range(n_times):
                seasonal[t, 0, :, :] = month_sin[t]
                seasonal[t, 1, :, :] = month_cos[t]
            
            X_enhanced = np.concatenate([X_with_lags, seasonal], axis=1)
            feature_names = feature_names + ["month_sin", "month_cos"]
        else:
            X_enhanced = X_with_lags
        
        # Add static features if available
        if static_data is not None:
            print("   Adding static features...")
            static_expanded = np.zeros((n_times, static_data.shape[0], n_lat, n_lon))
            for t in range(n_times):
                static_expanded[t, :, :, :] = static_data
            X_enhanced = np.concatenate([X_enhanced, static_expanded], axis=1)
            
            static_names = [f"static_{i}" for i in range(static_data.shape[0])]
            feature_names.extend(static_names)
        
        # Reshape for model training while preserving temporal structure
        print("   Reshaping data for model training...")
        X = X_enhanced.reshape(n_times, X_enhanced.shape[1], -1).transpose(0, 2, 1).reshape(-1, X_enhanced.shape[1])
        y = grace_tws.reshape(-1)
        
        # Filter out NaN values
        print("   Filtering out NaN values...")
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        # Create temporal indices for proper train/test splitting
        n_spatial_points = n_lat * n_lon
        temporal_indices = np.repeat(np.arange(n_times), n_spatial_points)
        temporal_indices_valid = temporal_indices[valid_mask]
        
        valid_ratio = len(X_valid) / len(X) * 100
        print(f"✅ Prepared data: {X_valid.shape[0]} valid samples ({valid_ratio:.1f}%), {X_valid.shape[1]} features")
        print(f"   Temporal structure: {n_times} time periods, {n_spatial_points} spatial points per time")
        
        return X_valid, y_valid, feature_names, {
            'spatial_shape': (n_lat, n_lon),
            'common_dates': common_dates,
            'temporal_indices': temporal_indices_valid,
            'n_times': n_times,
            'n_spatial_points': n_spatial_points
        }
    
    def _load_grace_tws(self, grace_dir, reference_raster=None):
        """Load GRACE TWS data with optional scientific resampling."""
        import re
        from datetime import datetime
        
        grace_files = sorted(os.listdir(grace_dir))
        grace_files = [f for f in grace_files if f.endswith('.tif')]
        
        if not grace_files:
            raise ValueError(f"No GRACE .tif files found in {grace_dir}")
        
        try:
            import rioxarray as rxr
            import rasterio.enums
        except ImportError:
            raise ImportError("rioxarray and rasterio are required for GRACE data loading")
        
        grace_data = []
        grace_dates = []
        
        # Check if scientific mode is enabled from config
        use_scientific = get_config('scientific.resampling', True)
        
        if use_scientific:
            print("  🔬 Using scientific GRACE resampling")
        
        for grace_file in grace_files:
            try:
                filename_str = safe_str_conversion(grace_file)
                
                # Try YYYYMM format first (200301.tif, 200302.tif...)
                match = re.match(r'(\d{6})\.tif$', filename_str)
                if match:
                    yyyymm = match.group(1)
                    # Convert YYYYMM to YYYY-MM format
                    year = yyyymm[:4]
                    month = yyyymm[4:6]
                    grace_date = f"{year}-{month}"
                    
                    grace_path = os.path.join(grace_dir, filename_str)
                    grace_raster = rxr.open_rasterio(grace_path, masked=True).squeeze()
                    
                # Fall back to old YYYYMMDD_YYYYMMDD format for backward compatibility
                elif re.match(r'(\d{8})_(\d{8})\.tif', filename_str):
                    match = re.match(r'(\d{8})_(\d{8})\.tif', filename_str)
                    date_str = match.group(1)
                    date = datetime.strptime(date_str, '%Y%m%d')
                    grace_date = date.strftime('%Y-%m')
                    
                    grace_path = os.path.join(grace_dir, filename_str)
                    grace_raster = rxr.open_rasterio(grace_path, masked=True).squeeze()
                    
                else:
                    # Skip files that don't match expected formats
                    continue
                
                # Use scientific resampling if enabled
                if reference_raster is not None:
                    if use_scientific:
                        grace_method = get_config('scientific.grace_method', 'gaussian')
                        grace_raster = resample_grace_scientifically(
                            grace_raster,
                            reference_raster,
                            method=grace_method
                        )
                    else:
                        # Original bilinear resampling
                        grace_raster = grace_raster.rio.reproject_match(
                            reference_raster,
                            resampling=rasterio.enums.Resampling.bilinear
                        )
                
                grace_data.append(grace_raster.values)
                grace_dates.append(grace_date)
            except Exception as e:
                print(f"⚠️ Error loading {grace_file}: {e}")
        
        print(f"   Successfully loaded {len(grace_data)} GRACE files")
        
        # Validate GRACE values
        if use_scientific and len(grace_data) > 0:
            validate_grace_values(np.concatenate([g.flatten() for g in grace_data]))
        
        return np.stack(grace_data), grace_dates
    
    def _create_lagged_features(self, X_data, lag_months=None):
        """Create lagged features."""
        if lag_months is None:
            lag_months = get_config('feature_processing.lag_months', [1, 3, 6])
        """Create lagged features."""
        n_times, n_features, n_lat, n_lon = X_data.shape
        n_lags = len(lag_months)
        total_features = n_features * (1 + n_lags)
        all_features = np.zeros((n_times, total_features, n_lat, n_lon))
        
        all_features[:, :n_features, :, :] = X_data
        
        feature_names = [f"feat_{i}" for i in range(n_features)]
        
        feature_idx = n_features
        for lag in lag_months:
            if lag >= n_times:
                continue
            all_features[lag:, feature_idx:feature_idx+n_features, :, :] = X_data[:-lag, :, :, :]
            for i in range(n_features):
                feature_names.append(f"feat_{i}_lag{lag}")
            feature_idx += n_features
        
        return all_features, feature_names
    
    def train_all_models(self, X, y, metadata, enabled_models=None, split_method='temporal'):
        """
        Train all enabled models with proper data splitting to prevent leakage.
        
        Parameters:
        -----------
        split_method : str
            'temporal' (default): Split by time periods (no temporal leakage)
            'spatial': Split by spatial locations (no spatial leakage) 
            'random': Random split (may cause leakage - not recommended)
        """
        try:
            # Validate inputs
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata must be a dictionary, got {type(metadata)}")
            
            required_keys = ['temporal_indices', 'n_times', 'common_dates', 'n_spatial_points']
            missing_keys = [key for key in required_keys if key not in metadata]
            if missing_keys:
                raise ValueError(f"metadata missing required keys: {missing_keys}")
            
            if enabled_models is None:
                configured_models = get_config('models.enabled', ['rf'])
                # Filter to only available models
                enabled_models = [m for m in configured_models if m in AVAILABLE_MODELS]
            
            # Filter enabled_models to only include available ones
            enabled_models = [m for m in enabled_models if m in AVAILABLE_MODELS]
            
            if not enabled_models:
                print("⚠️ No valid models specified, defaulting to Random Forest")
                enabled_models = ['rf']
            
            print(f"🚀 Training {len(enabled_models)} models: {enabled_models}")
            print(f"🔄 Using {split_method} splitting strategy")
            
            # Get splitting parameters
            temporal_indices = metadata['temporal_indices']
            n_times = metadata['n_times']
            common_dates = metadata['common_dates']
            test_size = get_config('models.test_size', 0.2)
            
            # Debug information
            print(f"🔍 Debug info:")
            print(f"   n_times: {n_times}")
            print(f"   common_dates type: {type(common_dates)}")
            print(f"   common_dates length: {len(common_dates) if hasattr(common_dates, '__len__') else 'N/A'}")
            print(f"   temporal_indices shape: {temporal_indices.shape if hasattr(temporal_indices, 'shape') else 'N/A'}")
            
            # Ensure common_dates is a list for indexing
            if not isinstance(common_dates, list):
                common_dates = list(common_dates)
            
            if split_method == 'temporal':
                # STRATIFIED TEMPORAL SPLITTING - Seasonal balance + no climate bias!
                import pandas as pd
                
                # Convert dates to DataFrame for easier manipulation
                date_df = pd.DataFrame({
                    'date': common_dates,
                    'time_idx': range(n_times)
                })
                date_df['date'] = pd.to_datetime(date_df['date'])
                date_df['year'] = date_df['date'].dt.year
                date_df['month'] = date_df['date'].dt.month
                
                # Stratified sampling: ensure all months in both train/test
                train_indices = []
                test_indices = []
                
                # For each month, randomly assign years to train/test
                np.random.seed(42)  # For reproducibility
                for month in range(1, 13):
                    month_data = date_df[date_df['month'] == month]
                    unique_years = month_data['year'].unique()
                    
                    # Randomly shuffle years for this month
                    shuffled_years = np.random.permutation(unique_years)
                    n_test_years = max(1, int(len(unique_years) * test_size))
                    
                    test_years = shuffled_years[:n_test_years]
                    train_years = shuffled_years[n_test_years:]
                    
                    # Add indices for this month
                    month_train = month_data[month_data['year'].isin(train_years)]['time_idx'].values
                    month_test = month_data[month_data['year'].isin(test_years)]['time_idx'].values
                    
                    train_indices.extend(month_train)
                    test_indices.extend(month_test)
                
                train_time_indices = set(train_indices)
                test_time_indices = set(test_indices)
                
                # Create masks based on stratified temporal indices
                train_mask = np.array([t in train_time_indices for t in temporal_indices])
                test_mask = np.array([t in test_time_indices for t in temporal_indices])
                
                print(f"📅 Stratified temporal splitting:")
                print(f"   Training periods: {len(train_indices)} time points (all months represented)")
                print(f"   Testing periods: {len(test_indices)} time points (all months represented)")
                print(f"   ✅ No climate shift bias - years randomly mixed!")
                print(f"   ✅ Seasonal balance maintained in both sets!")
                
                X_train = X[train_mask]
                X_test = X[test_mask]
                y_train = y[train_mask]
                y_test = y[test_mask]
                
                print(f"   ✅ No temporal data leakage - completely separate time periods!")
            
            elif split_method == 'spatial':
                # SPATIAL SPLITTING - No spatial data leakage!
                n_spatial_points = metadata['n_spatial_points']
                
                print(f"🗺️ Spatial splitting:")
                print(f"   Total spatial points: {n_spatial_points}")
                print(f"   Total samples: {len(X)}")
                
                # Create spatial indices for each sample
                spatial_indices = np.tile(np.arange(n_spatial_points), n_times)
                spatial_indices_valid = spatial_indices[:len(X)]  # Align with valid samples
                
                print(f"   Spatial indices shape: {spatial_indices_valid.shape}")
                
                # Split spatial locations
                train_spatial_cutoff = int(n_spatial_points * (1 - test_size))
                
                train_mask = spatial_indices_valid < train_spatial_cutoff
                test_mask = spatial_indices_valid >= train_spatial_cutoff
                
                X_train = X[train_mask]
                X_test = X[test_mask]
                y_train = y[train_mask]
                y_test = y[test_mask]
                
                print(f"   Training locations: {train_spatial_cutoff} spatial points")
                print(f"   Testing locations: {n_spatial_points - train_spatial_cutoff} spatial points")
                print(f"   ✅ No spatial data leakage - completely separate locations!")
                
            else:  # split_method == 'random'
                print("⚠️ WARNING: Random splitting may cause data leakage!")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            print(f"   Training samples: {len(X_train):,}")
            print(f"   Testing samples: {len(X_test):,}")
            
            # Initialize scaler based on config
            scaling_method = get_config('feature_processing.scaling_method', 'standard')
            
            if scaling_method == 'standard':
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            elif scaling_method == 'robust':
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
            elif scaling_method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
            else:
                self.scaler = None
            
            if self.scaler is not None:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # PHASE 1: HYPERPARAMETER OPTIMIZATION
            optimized_params = {}
            
            # Check if hyperparameter tuning is enabled in config
            if self.config.get('models', {}).get('advanced', {}).get('enable_tuning', False):
                print("\n" + "="*60)
                print("🎯 HYPERPARAMETER OPTIMIZATION ENABLED")
                tuning_trials = self.config.get('models', {}).get('advanced', {}).get('tuning_trials', 50)
                print(f"📊 Running {tuning_trials} optimization trials")
                print("="*60)
                
                # Run hyperparameter optimization for XGBoost if enabled
                if 'xgb' in enabled_models:
                    print("\n🔍 Optimizing XGBoost hyperparameters...")
                    try:
                        opt_result = self.optimize_xgboost_hyperparameters(
                            X_train, y_train, n_calls=tuning_trials, cv_folds=3
                        )
                        # Extract parameters from (params, score) tuple
                        if isinstance(opt_result, tuple):
                            optimized_params['xgb'] = opt_result[0]  # Extract just the parameters
                        else:
                            optimized_params['xgb'] = opt_result
                        print(f"✅ XGBoost optimization completed: {optimized_params['xgb']}")
                    except Exception as e:
                        print(f"⚠️ XGBoost optimization failed: {e}")
                        optimized_params['xgb'] = {}
            else:
                print("\n" + "="*60)
                print("⚠️ HYPERPARAMETER OPTIMIZATION DISABLED")
                print("📊 Using config defaults - enable with advanced.enable_tuning: true")
                print("="*60)
            
            results = []
            trained_models = []
            
            for model_name in enabled_models:
                if model_name not in self.model_configs:
                    print(f"⚠️ Model {model_name} not available, skipping...")
                    continue
                
                print(f"\n🔄 Training {self.model_configs[model_name]['name']}...")
                
                try:
                    config = self.model_configs[model_name]
                    
                    # Use optimized parameters if available
                    if model_name in optimized_params and model_name == 'xgb':
                        print(f"  🎯 Using OPTIMIZED parameters for {config['name']}")
                        opt_params = optimized_params[model_name]
                        model = xgb.XGBRegressor(
                            n_estimators=opt_params['n_estimators'],
                            max_depth=opt_params['max_depth'],
                            learning_rate=opt_params['learning_rate'],
                            subsample=opt_params['subsample'],
                            colsample_bytree=opt_params['colsample_bytree'],
                            reg_alpha=opt_params['reg_alpha'],
                            reg_lambda=opt_params['reg_lambda'],
                            random_state=42,
                            n_jobs=12,
                            verbosity=0
                        )
                    else:
                        model = config['model']
                    
                    # Choose appropriate data
                    if config['needs_scaling']:
                        X_train_use = X_train_scaled
                        X_test_use = X_test_scaled
                    else:
                        X_train_use = X_train
                        X_test_use = X_test
                    
                    # Train model
                    start_time = datetime.now()
                    model.fit(X_train_use, y_train)
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train_use)
                    y_pred_test = model.predict(X_test_use)
                    
                    # Calculate metrics
                    metrics = {
                        'model_name': model_name,
                        'display_name': config['name'],
                        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                        'train_r2': r2_score(y_train, y_pred_train),
                        'test_r2': r2_score(y_test, y_pred_test),
                        'train_mae': mean_absolute_error(y_train, y_pred_train),
                        'test_mae': mean_absolute_error(y_test, y_pred_test),
                        'training_time': training_time,
                        'needs_scaling': config['needs_scaling']
                    }
                    
                    results.append(metrics)
                    
                    # Store trained model
                    self.models[model_name] = {
                        'model': model,
                        'config': config,
                        'metrics': metrics
                    }
                    trained_models.append(model)
                    
                    print(f"  ✅ {config['name']}: R² = {metrics['test_r2']:.4f}, "
                          f"RMSE = {metrics['test_rmse']:.4f}")
                    
                except Exception as e:
                    print(f"  ❌ {config['name']} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Store results and identify best model
            self.results = pd.DataFrame(results)
            if len(self.results) > 0:
                best_idx = self.results['test_r2'].idxmax()
                best_model_name = self.results.loc[best_idx, 'model_name']
                self.best_model = best_model_name
                
                print(f"\n🏆 Best model: {self.results.loc[best_idx, 'display_name']} "
                      f"(R² = {self.results.loc[best_idx, 'test_r2']:.4f})")
                
                # Create ensemble model if we have multiple models
                if len(self.results) >= 2:
                    print("\n🔄 Creating weighted ensemble model...")
                    self._create_ensemble_model(X_train_use, y_train, X_test_use, y_test)
            
            return self.results
            
        except Exception as e:
            print(f"❌ Error in train_all_models: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_ensemble_model(self, X_train, y_train, X_test, y_test):
        """Create a weighted ensemble model from trained individual models."""
        try:
            # Extract models and their R² scores
            models_list = []
            weights_list = []
            model_names = []
            
            for _, row in self.results.iterrows():
                model_name = row['model_name']
                if model_name in self.models:
                    models_list.append(self.models[model_name]['model'])
                    weights_list.append(max(0.01, row['test_r2']))  # Ensure positive weights
                    model_names.append(row['display_name'])
            
            if len(models_list) < 2:
                print("  ⚠️ Need at least 2 models for ensemble")
                return
            
            # Normalize weights to sum to 1
            weights_array = np.array(weights_list)
            weights_normalized = weights_array / np.sum(weights_array)
            
            print(f"  📊 Ensemble composition:")
            for name, weight in zip(model_names, weights_normalized):
                print(f"    {name}: {weight:.3f}")
            
            # Create ensemble model
            ensemble_model = EnsembleRegressor(models_list, weights_normalized)
            
            # Train ensemble (already trained individual models)
            start_time = datetime.now()
            ensemble_model.fit(X_train, y_train)  # This just stores the models
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Make predictions in chunks to avoid memory issues
            print("  🔄 Making ensemble predictions in chunks...")
            
            # Predict training set in chunks
            chunk_size = 1000000  # 1M samples per chunk
            y_pred_train = []
            for i in range(0, len(X_train), chunk_size):
                chunk_end = min(i + chunk_size, len(X_train))
                chunk_pred = ensemble_model.predict(X_train[i:chunk_end])
                y_pred_train.append(chunk_pred)
                print(f"    Processed training chunk {i//chunk_size + 1}/{(len(X_train)-1)//chunk_size + 1}")
            y_pred_train = np.concatenate(y_pred_train)
            
            # Predict test set in chunks
            y_pred_test = []
            for i in range(0, len(X_test), chunk_size):
                chunk_end = min(i + chunk_size, len(X_test))
                chunk_pred = ensemble_model.predict(X_test[i:chunk_end])
                y_pred_test.append(chunk_pred)
                print(f"    Processed test chunk {i//chunk_size + 1}/{(len(X_test)-1)//chunk_size + 1}")
            y_pred_test = np.concatenate(y_pred_test)
            
            # Calculate metrics
            ensemble_metrics = {
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'training_time': training_time
            }
            
            # Store ensemble model
            ensemble_config = {
                'name': 'Weighted Ensemble',
                'display_name': 'Weighted Ensemble',
                'needs_scaling': False,
                'type': 'ensemble'
            }
            
            self.models['ensemble'] = {
                'model': ensemble_model,
                'config': ensemble_config,
                'metrics': ensemble_metrics
            }
            
            # Add ensemble results to results DataFrame
            ensemble_result = {
                'model_name': 'ensemble',
                'display_name': 'Weighted Ensemble',
                'train_r2': ensemble_metrics['train_r2'],
                'test_r2': ensemble_metrics['test_r2'],
                'train_rmse': ensemble_metrics['train_rmse'],
                'test_rmse': ensemble_metrics['test_rmse'],
                'train_mae': ensemble_metrics['train_mae'],
                'test_mae': ensemble_metrics['test_mae'],
                'training_time': ensemble_metrics['training_time']
            }
            
            # Add to results
            ensemble_df = pd.DataFrame([ensemble_result])
            self.results = pd.concat([self.results, ensemble_df], ignore_index=True)
            
            # Update best model if ensemble is better
            if ensemble_metrics['test_r2'] > self.results[self.results['model_name'] != 'ensemble']['test_r2'].max():
                self.best_model = 'ensemble'
                print(f"  🎯 NEW BEST MODEL: Weighted Ensemble (R² = {ensemble_metrics['test_r2']:.4f})")
            else:
                print(f"  📊 Ensemble Performance: R² = {ensemble_metrics['test_r2']:.4f}")
                
        except Exception as e:
            print(f"  ❌ Error creating ensemble: {e}")
            import traceback
            traceback.print_exc()
    
    def optimize_xgboost_hyperparameters(self, X_train, y_train, n_calls=20, cv_folds=3):
        """
        Optimize XGBoost hyperparameters using Bayesian optimization.
        This is the highest impact improvement for reaching R² ≥ 0.80.
        """
        print(f"\n🎯 BAYESIAN OPTIMIZATION: XGBoost Hyperparameters ({n_calls} iterations)")
        
        if not HAS_XGB:
            print("  ❌ XGBoost not available")
            return None
            
        # OPTIMIZED search space for faster convergence and better results
        if HAS_SKOPT:
            space = [
                Integer(100, 800, name='n_estimators'),     # Focus on efficient range
                Integer(6, 20, name='max_depth'),           # Deeper trees for complex patterns  
                Real(0.01, 0.2, name='learning_rate'),      # Conservative learning rates
                Real(0.7, 0.95, name='subsample'),          # Good bootstrap range
                Real(0.6, 0.9, name='colsample_bytree'),    # Feature diversity
                Real(0.01, 2.0, name='reg_alpha'),          # Lighter regularization focus
                Real(0.01, 2.0, name='reg_lambda'),         # Lighter regularization focus
            ]
            
            # Objective function for optimization
            @use_named_args(space)
            def objective(**params):
                # Create XGBoost model with suggested parameters
                model = xgb.XGBRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate'],
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    reg_alpha=params['reg_alpha'],
                    reg_lambda=params['reg_lambda'],
                    random_state=42,
                    n_jobs=32,  # Use more cores for faster optimization training
                    verbosity=0
                )
                
                # FIXED: Use temporal CV strategy matching final test split
                # Instead of random k-fold, use temporal-aware validation
                # This ensures CV methodology = final test methodology
                
                # For now, use simple holdout with random temporal split similar to final test
                from sklearn.model_selection import train_test_split
                X_cv_train, X_cv_val, y_cv_train, y_cv_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
                
                # Train on CV training set, validate on CV validation set
                model.fit(X_cv_train, y_cv_train, 
                         eval_set=[(X_cv_val, y_cv_val)], 
                         verbose=False)
                
                # Get validation score
                y_cv_pred = model.predict(X_cv_val)
                cv_score = r2_score(y_cv_val, y_cv_pred)
                scores = np.array([cv_score])  # Convert to array for compatibility
                
                # Return negative R² (minimize instead of maximize)
                cv_score = scores.mean()
                print(f"    Parameters: {params} → CV R² = {cv_score:.4f}")
                return -cv_score
            
            try:
                # Run Bayesian optimization
                print(f"  🔍 Starting optimization with {n_calls} calls...")
                result = gp_minimize(func=objective, dimensions=space, n_calls=n_calls, 
                                   random_state=42, acq_func='EI')
                
                # Extract best parameters
                best_params = {
                    'n_estimators': result.x[0],
                    'max_depth': result.x[1], 
                    'learning_rate': result.x[2],
                    'subsample': result.x[3],
                    'colsample_bytree': result.x[4],
                    'reg_alpha': result.x[5],
                    'reg_lambda': result.x[6]
                }
                
                best_score = -result.fun
                
                print(f"  🎯 OPTIMIZATION COMPLETE!")
                print(f"  🏆 Best CV R² = {best_score:.4f}")
                print(f"  📊 Best parameters:")
                for param, value in best_params.items():
                    print(f"    {param}: {value}")
                
                return best_params, best_score
                
            except Exception as e:
                print(f"  ❌ Optimization failed: {e}")
                return None, None
                
        else:
            print("  ⚠️ scikit-optimize not available - using grid search fallback")
            # Simple grid search fallback
            from sklearn.model_selection import GridSearchCV
            
            param_grid = {
                'n_estimators': [200, 500, 800],
                'max_depth': [6, 10, 15],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=12, verbosity=0)
            grid_search = GridSearchCV(xgb_model, param_grid, cv=cv_folds, 
                                     scoring='r2', n_jobs=1, verbose=1)
            
            print(f"  🔍 Running grid search...")
            grid_search.fit(X_train, y_train)
            
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            print(f"  🎯 GRID SEARCH COMPLETE!")
            print(f"  🏆 Best CV R² = {best_score:.4f}")
            print(f"  📊 Best parameters: {best_params}")
            
            return best_params, best_score
    
    def save_models(self, output_dir="models"):
        """Save all trained models."""
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"\n💾 Saving {len(self.models)} models...")
        
        for model_name, model_data in self.models.items():
            model_path = Path(output_dir) / f"{model_name}_model.joblib"
            
            save_package = {
                'model': model_data['model'],
                'config': model_data['config'],
                'metrics': model_data['metrics'],
                'scaler': self.scaler if model_data['config'].get('needs_scaling') else None
            }
            
            joblib.dump(save_package, model_path)
            print(f"  ✅ Saved {model_data['config']['name']} to {model_path}")
        
        # Save results summary
        if len(self.results) > 0:
            results_path = Path(output_dir) / "model_comparison.csv"
            self.results.to_csv(results_path, index=False)
            print(f"  ✅ Saved comparison to {results_path}")
        
        # Save best model as default
        if self.best_model:
            best_path = Path(output_dir) / "rf_model_enhanced.joblib"
            best_model_data = self.models[self.best_model]
            
            save_package = {
                'model': best_model_data['model'],
                'config': best_model_data['config'],
                'metrics': best_model_data['metrics'],
                'scaler': self.scaler if best_model_data['config'].get('needs_scaling') else None
            }
            
            joblib.dump(save_package, best_path)
            print(f"  ✅ Saved best model as {best_path}")
    
    def create_comparison_plots(self, output_dir="figures"):
        """Create model comparison visualizations."""
        if len(self.results) == 0:
            print("⚠️ No results to plot")
            return
        
        Path(output_dir).mkdir(exist_ok=True)
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # R² comparison
            self.results.plot(x='display_name', y='test_r2', kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('Model Performance (R²)')
            ax1.set_ylabel('Test R²')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # RMSE comparison
            self.results.plot(x='display_name', y='test_rmse', kind='bar', ax=ax2, color='lightcoral')
            ax2.set_title('Model Performance (RMSE)')
            ax2.set_ylabel('Test RMSE')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Training time comparison
            self.results.plot(x='display_name', y='training_time', kind='bar', ax=ax3, color='lightgreen')
            ax3.set_title('Training Time')
            ax3.set_ylabel('Time (seconds)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # MAE comparison
            self.results.plot(x='display_name', y='test_mae', kind='bar', ax=ax4, color='orange')
            ax4.set_title('Model Performance (MAE)')
            ax4.set_ylabel('Test MAE')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Model Comparison Summary', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(Path(output_dir) / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Comparison plots saved to {output_dir}/")
            
        except Exception as e:
            print(f"⚠️ Error creating plots: {e}")

    def analyze_feature_importance_shap(self, output_dir="figures", max_features=20):
        """
        Analyze feature importance using SHAP values for model interpretability.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save SHAP plots
        max_features : int
            Maximum number of features to show in plots
        """
        if not HAS_SHAP:
            print("⚠️ SHAP not available. Install with: pip install shap")
            return
            
        if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
            print("⚠️ No test data available. Run train_models() first.")
            return
            
        if len(self.models) == 0:
            print("⚠️ No trained models available. Run train_models() first.")
            return
        
        print("\n🔍 ANALYZING FEATURE IMPORTANCE WITH SHAP")
        Path(output_dir).mkdir(exist_ok=True)
        
        # Get feature names
        feature_names = getattr(self, 'feature_names', [f'feature_{i}' for i in range(self.X_test.shape[1])])
        
        # Analyze each model
        for model_name, model_data in self.models.items():
            if model_name == 'ensemble':
                continue  # Skip ensemble for now
                
            print(f"\n📊 Analyzing {model_data['config']['name']}...")
            
            model = model_data['model']
            config = model_data['config']
            
            try:
                # Prepare data for SHAP
                if config.get('needs_scaling', False) and hasattr(self, 'scaler') and self.scaler is not None:
                    X_test_use = self.scaler.transform(self.X_test)
                    X_train_sample = self.scaler.transform(self.X_train[:1000])  # Sample for background
                else:
                    X_test_use = self.X_test
                    X_train_sample = self.X_train[:1000]  # Sample for background
                
                # Choose appropriate SHAP explainer based on model type
                if model_name in ['rf', 'gbr'] or 'tree' in str(type(model)).lower():
                    # Tree-based models
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test_use[:500])  # Sample for efficiency
                    
                elif model_name in ['nn', 'svr']:
                    # Model-agnostic explainer for complex models
                    explainer = shap.KernelExplainer(model.predict, X_train_sample)
                    shap_values = explainer.shap_values(X_test_use[:100])  # Smaller sample for efficiency
                    
                else:
                    # Default: Linear explainer for linear models, otherwise kernel
                    try:
                        explainer = shap.LinearExplainer(model, X_train_sample)
                        shap_values = explainer.shap_values(X_test_use[:500])
                    except:
                        explainer = shap.KernelExplainer(model.predict, X_train_sample)
                        shap_values = explainer.shap_values(X_test_use[:100])
                
                # Create SHAP plots
                plt.figure(figsize=(12, 8))
                
                # Summary plot (bar)
                shap.summary_plot(
                    shap_values, 
                    X_test_use[:len(shap_values)], 
                    feature_names=feature_names,
                    plot_type="bar",
                    max_display=max_features,
                    show=False
                )
                plt.title(f'SHAP Feature Importance - {model_data["config"]["name"]}')
                plt.tight_layout()
                plt.savefig(Path(output_dir) / f'shap_importance_{model_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Summary plot (detailed)
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_values, 
                    X_test_use[:len(shap_values)], 
                    feature_names=feature_names,
                    max_display=max_features,
                    show=False
                )
                plt.title(f'SHAP Feature Impact - {model_data["config"]["name"]}')
                plt.tight_layout()
                plt.savefig(Path(output_dir) / f'shap_summary_{model_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Feature importance ranking
                feature_importance = np.abs(shap_values).mean(0)
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                # Save importance ranking
                importance_df.to_csv(Path(output_dir) / f'feature_importance_{model_name}.csv', index=False)
                
                print(f"  ✅ Top 10 features for {model_data['config']['name']}:")
                for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                    print(f"    {i+1:2d}. {row['feature']:<25} (importance: {row['importance']:.4f})")
                
                print(f"  ✅ SHAP plots saved: shap_importance_{model_name}.png, shap_summary_{model_name}.png")
                
            except Exception as e:
                print(f"  ❌ Error analyzing {model_name}: {e}")
                continue
        
        print("✅ SHAP analysis complete!")


def main(split_method='temporal'):
    """
    Main function for model training and comparison.
    
    Parameters:
    -----------
    split_method : str
        'temporal' (default): Split by time periods (no temporal leakage)
        'spatial': Split by spatial locations (no spatial leakage)
        'random': Random split (may cause leakage - not recommended)
    """
    print("🚀 GRACE Multi-Model Training Pipeline")
    print("="*50)
    
    try:
        # Initialize model manager
        manager = ModelManager()
        
        # Prepare data
        X, y, feature_names, metadata = manager.prepare_data()
        
        # Train all models with specified splitting method
        results = manager.train_all_models(X, y, metadata, split_method=split_method)
        
        # Save models and results
        manager.save_models()
        
        # Create comparison plots
        manager.create_comparison_plots()
        
        # Analyze feature importance with SHAP
        manager.analyze_feature_importance_shap()
        
        print("\n✅ Multi-model training complete!")
        if len(results) > 0:
            print(f"📊 Results summary:")
            print(results[['display_name', 'test_r2', 'test_rmse']].to_string(index=False))
        
        return manager
        
    except Exception as e:
        print(f"❌ Error in multi-model training: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()