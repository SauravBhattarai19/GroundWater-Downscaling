# src/model_manager.py - Fixed version with proper data type handling
"""
Multi-Model Manager for GRACE Downscaling - Fixed Version

Handles numpy string type conversion issues properly.
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import joblib
import warnings
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# Core ML imports (always available)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from src.utils import resample_grace_scientifically, create_grace_weight_mask, validate_grace_values

warnings.filterwarnings('ignore')

# Advanced ML imports with graceful fallbacks
AVAILABLE_MODELS = ['nn', 'xgb']  # Always available

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
        """Predict using weighted average of all models."""
        predictions = np.array([model.predict(X) for model in self.models])
        
        if self.weights is None:
            return np.mean(predictions, axis=0)
        else:
            return np.average(predictions, weights=self.weights, axis=0)


class ModelManager:
    """Unified interface for multiple ML models with robust data type handling."""
    
    def __init__(self, config_path="src/config.yaml"):
        self.config_path = config_path
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
        
        # Load configuration
        self._load_config()
        
        # Initialize available models
        self._initialize_models()
        
    def _load_config(self):
        """Load model configurations with fallback."""
        default_config = {
            'models': {
                'enabled': [m for m in ['rf', 'xgb', 'lgb'] if m in AVAILABLE_MODELS],
                'ensemble': True,
                'cross_validation': True,
                'test_size': 0.2
            }
        }
        
        if HAS_YAML and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                
                # Merge with defaults
                if 'models' in loaded_config:
                    default_config['models'].update(loaded_config['models'])
                
                self.config = default_config
            except Exception as e:
                print(f"⚠️ Error loading config: {e}, using defaults")
                self.config = default_config
        else:
            self.config = default_config
    
    def _initialize_models(self):
        """Initialize all available models with limited parallelization."""
        
        # Calculate 50% of available cores
        import multiprocessing
        max_cores = max(1, multiprocessing.cpu_count() // 2)  # 50% of cores
        print(f"🔧 Limiting parallelization to {max_cores} cores (50% of available)")
        
        # Get hyperparameters from config
        rf_params = self.config.get('models', {}).get('hyperparameters', {}).get('rf', {})
        
        self.model_configs = {
            'rf': {
                'name': 'Random Forest',
                'model': RandomForestRegressor(
                    n_estimators=rf_params.get('n_estimators', 50),
                    max_depth=rf_params.get('max_depth', 15),
                    min_samples_split=rf_params.get('min_samples_split', 50),
                    min_samples_leaf=rf_params.get('min_samples_leaf', 20),
                    max_features=rf_params.get('max_features', 0.3),
                    max_samples=rf_params.get('max_samples', 0.8),
                    n_jobs=min(32, max_cores//2),  # ✅ VERY LIMITED for large datasets
                    random_state=42,
                    verbose=1  # Show progress
                ),
                'needs_scaling': False
            },
            'gbr': {
                'name': 'Gradient Boosting',
                'model': GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                    # Note: GBR doesn't have n_jobs parameter
                ),
                'needs_scaling': False
            },
            'nn': {
                'name': 'Neural Network',
                'model': MLPRegressor(
                    hidden_layer_sizes=(256, 128, 64),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42
                    # Note: MLPRegressor doesn't have n_jobs parameter
                ),
                'needs_scaling': True
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
                'needs_scaling': True
            }
        }
        
        # Add advanced models if available
        if HAS_XGB:
            self.model_configs['xgb'] = {
                'name': 'XGBoost',
                'model': xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=max_cores,  # ✅ LIMITED instead of -1
                    nthread=max_cores   # ✅ XGBoost specific parameter
                ),
                'needs_scaling': False
            }
        
        if HAS_LGB:
            self.model_configs['lgb'] = {
                'name': 'LightGBM',
                'model': lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=max_cores,    # ✅ LIMITED instead of -1
                    num_threads=max_cores, # ✅ LightGBM specific parameter
                    verbose=-1
                ),
                'needs_scaling': False
            }
        
        if HAS_CATB:
            self.model_configs['catb'] = {
                'name': 'CatBoost',
                'model': cb.CatBoostRegressor(
                    iterations=200,
                    depth=8,
                    learning_rate=0.1,
                    random_state=42,
                    thread_count=max_cores,  # ✅ CatBoost specific parameter
                    verbose=False
                ),
                'needs_scaling': False
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
        print("   Creating lagged features...")
        X_with_lags, feature_names = self._create_lagged_features(X_temporal, lag_months=[1, 3, 6])
        
        # Add seasonal features
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
        
        # Add static features if available
        if static_data is not None:
            print("   Adding static features...")
            static_expanded = np.zeros((n_times, static_data.shape[0], n_lat, n_lon))
            for t in range(n_times):
                static_expanded[t, :, :, :] = static_data
            X_enhanced = np.concatenate([X_enhanced, static_expanded], axis=1)
            
            static_names = [f"static_{i}" for i in range(static_data.shape[0])]
            feature_names.extend(static_names)
        
        # Reshape for model training
        print("   Reshaping data for model training...")
        X = X_enhanced.reshape(n_times, X_enhanced.shape[1], -1).transpose(0, 2, 1).reshape(-1, X_enhanced.shape[1])
        y = grace_tws.reshape(-1)
        
        # Filter out NaN values
        print("   Filtering out NaN values...")
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        valid_ratio = len(X_valid) / len(X) * 100
        print(f"✅ Prepared data: {X_valid.shape[0]} valid samples ({valid_ratio:.1f}%), {X_valid.shape[1]} features")
        
        return X_valid, y_valid, feature_names, {
            'spatial_shape': (n_lat, n_lon),
            'common_dates': common_dates
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
            raise ImportError("rioxarray is required for GRACE data loading")
        
        grace_data = []
        grace_dates = []
        
        # Check if scientific mode is enabled
        use_scientific = getattr(self, 'use_scientific_grace', True)  # Default to True
        
        if use_scientific:
            print("  🔬 Using scientific GRACE resampling")
        
        for grace_file in grace_files:
            try:
                filename_str = safe_str_conversion(grace_file)
                
                match = re.match(r'(\d{8})_(\d{8})\.tif', filename_str)
                if match:
                    date_str = match.group(1)
                    date = datetime.strptime(date_str, '%Y%m%d')
                    grace_date = date.strftime('%Y-%m')
                    
                    grace_path = os.path.join(grace_dir, filename_str)
                    grace_raster = rxr.open_rasterio(grace_path, masked=True).squeeze()
                    
                    # Use scientific resampling if enabled
                    if reference_raster is not None:
                        if use_scientific:
                            grace_raster = resample_grace_scientifically(
                                grace_raster,
                                reference_raster,
                                method='gaussian'  # or 'conservative'
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
    
    def _create_lagged_features(self, X_data, lag_months=[1, 3, 6]):
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
    
    def train_all_models(self, X, y, enabled_models=None):
        """Train all enabled models."""
        if enabled_models is None:
            configured_models = self.config['models'].get('enabled', ['rf'])
            # Filter to only available models
            enabled_models = [m for m in configured_models if m in AVAILABLE_MODELS]
        
        # Filter enabled_models to only include available ones
        enabled_models = [m for m in enabled_models if m in AVAILABLE_MODELS]
        
        if not enabled_models:
            print("⚠️ No valid models specified, defaulting to Random Forest")
            enabled_models = ['rf']
        
        print(f"🚀 Training {len(enabled_models)} models: {enabled_models}")
        
        # Split data
        test_size = self.config['models'].get('test_size', 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Initialize scaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = []
        trained_models = []
        
        for model_name in enabled_models:
            if model_name not in self.model_configs:
                print(f"⚠️ Model {model_name} not available, skipping...")
                continue
            
            print(f"\n🔄 Training {self.model_configs[model_name]['name']}...")
            
            try:
                config = self.model_configs[model_name]
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
        
        return self.results
    
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


def main():
    """Main function for model training and comparison."""
    print("🚀 GRACE Multi-Model Training Pipeline")
    print("="*50)
    
    try:
        # Initialize model manager
        manager = ModelManager()
        
        # Prepare data
        X, y, feature_names, metadata = manager.prepare_data()
        
        # Train all models
        results = manager.train_all_models(X, y)
        
        # Save models and results
        manager.save_models()
        
        # Create comparison plots
        manager.create_comparison_plots()
        
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