"""
Coarse-Scale Model Trainer

Trains ML models at GRACE native resolution (55km) using blocked spatiotemporal CV.
This ensures models learn relationships at the correct measurement scale.
"""

import numpy as np
import pandas as pd
import xarray as xr
import joblib
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Advanced ML (with fallbacks)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CATB = True
except ImportError:
    HAS_CATB = False

# Scale-consistent neural network (with fallback)
try:
    import torch
    from src_new_approach.scale_consistent_nn import (
        ScaleConsistentNNWrapper,
        ScaleConsistentTrainer,
        create_coarse_to_fine_mapping
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Spatiotemporal CV
from src_new_approach.spatiotemporal_cv import (
    BlockedSpatiotemporalCV,
    prepare_metadata_for_cv,
    evaluate_with_blocked_cv
)

from src_new_approach.utils_downscaling import (
    load_config,
    get_config_value,
    print_statistics
)


class CoarseModelTrainer:
    """
    Train ML models at coarse (55km) resolution.
    
    Uses blocked spatiotemporal cross-validation to prevent data leakage.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Parameters:
        -----------
        config : Dict
            Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.cv_results = {}
        self.scalers = {}
        self.tuned_params = {}  # Store tuned hyperparameters
        
        # Get enabled models
        self.enabled_models = get_config_value(config, 'models.enabled', ['rf', 'xgb'])
        
        # Filter by availability
        available_models = ['rf', 'gbr', 'nn', 'svr']
        if HAS_XGB:
            available_models.append('xgb')
        if HAS_LGB:
            available_models.append('lgb')
        if HAS_CATB:
            available_models.append('catb')
        if HAS_TORCH:
            available_models.append('nn_scale_consistent')
        
        self.enabled_models = [m for m in self.enabled_models if m in available_models]
        
        # Store fine data for scale-consistent training (set later if needed)
        self.fine_features_ds = None
        self.coarse_to_fine_mapping = None
        self.fine_valid_mask = None
        
        # Get actual CV method from config
        cv_method = get_config_value(config, 'models.cross_validation.method', 'blocked_spatiotemporal')
        cv_display = {
            'blocked_spatiotemporal': 'Blocked Spatiotemporal',
            'simple_split': 'Traditional 70-30 Split'
        }.get(cv_method, cv_method.title())
        
        print(f"🔧 Coarse Model Trainer initialized:")
        print(f"   Models to train: {self.enabled_models}")
        print(f"   CV method: {cv_display}")
        
    def prepare_training_data(self,
                             features_ds: xr.Dataset,
                             grace_ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """
        Prepare training data from coarse features and GRACE.
        
        Parameters:
        -----------
        features_ds : xr.Dataset
            Coarse-resolution features (55km)
        grace_ds : xr.Dataset
            Gap-filled GRACE data (55km)
        
        Returns:
        --------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target values (n_samples,)
        feature_names : List[str]
            Names of features
        metadata : Dict
            Metadata for CV (spatial_coords, temporal_indices, etc.)
        """
        print("\n" + "="*70)
        print("📦 PREPARING TRAINING DATA AT COARSE SCALE")
        print("="*70)
        
        # Extract GRACE targets
        grace_var = 'tws_anomaly'
        if grace_var not in grace_ds:
            # Try alternative names
            for alt_name in ['lwe_thickness', 'grace_anomaly', 'TWS']:
                if alt_name in grace_ds:
                    grace_var = alt_name
                    break
        
        grace_data = grace_ds[grace_var].values  # (time, lat, lon)
        
        # Extract features
        if 'features' in features_ds and 'static_features' in features_ds:
            # New aggregated format with separate temporal and static features
            temporal_features = features_ds['features'].values  # (time, feature, lat, lon)
            static_features = features_ds['static_features'].values  # (static_feature, lat, lon)
            
            # Transpose temporal features to (time, lat, lon, feature)
            temporal_features = temporal_features.transpose(0, 2, 3, 1)
            
            # Broadcast static features across time
            n_times = len(features_ds.time)
            static_broadcasted = np.broadcast_to(
                static_features[np.newaxis, :, :, :],  # (1, static_feature, lat, lon)
                (n_times,) + static_features.shape    # (time, static_feature, lat, lon)
            ).transpose(0, 2, 3, 1)  # (time, lat, lon, static_feature)
            
            # Combine temporal and static features
            feature_data = np.concatenate([temporal_features, static_broadcasted], axis=-1)  # (time, lat, lon, total_features)
            
            # Get feature names
            temporal_names = [str(f) if hasattr(f, 'item') else f 
                            for f in features_ds['feature'].values]
            static_names = [str(f) if hasattr(f, 'item') else f 
                          for f in features_ds['static_feature'].values]
            feature_names = temporal_names + static_names
            
        elif 'features' in features_ds:
            # Old stacked format
            feature_data = features_ds['features'].values  # (time, lat, lon, feature)
            feature_names = [str(f) if hasattr(f, 'item') else f 
                           for f in features_ds['feature'].values]
        else:
            # Variable format - stack all variables
            var_names = [v for v in features_ds.data_vars.keys()]
            feature_arrays = []
            feature_names = []
            
            for var_name in var_names:
                var_data = features_ds[var_name].values
                if var_data.ndim == 3:  # (time, lat, lon)
                    feature_arrays.append(var_data)
                    feature_names.append(var_name)
                elif var_data.ndim == 2:  # (lat, lon) - static
                    # Broadcast across time
                    n_times = len(features_ds.time)
                    var_data_3d = np.broadcast_to(
                        var_data[np.newaxis, :, :],
                        (n_times,) + var_data.shape
                    )
                    feature_arrays.append(var_data_3d)
                    feature_names.append(var_name)
            
            # Stack features
            feature_data = np.stack(feature_arrays, axis=-1)  # (time, lat, lon, feature)
        
        n_times, n_lat, n_lon, n_features = feature_data.shape
        
        print(f"📊 Data shape:")
        print(f"   GRACE: {grace_data.shape} (time, lat, lon)")
        print(f"   Features: {feature_data.shape} (time, lat, lon, features)")
        print(f"   Number of features: {n_features}")
        
        # Flatten spatial dimensions
        # Shape: (time * lat * lon, features)
        X_full = feature_data.reshape(-1, n_features)
        y_full = grace_data.reshape(-1)
        
        # Create metadata for CV
        times = pd.DatetimeIndex(features_ds.time.values)
        lat_values = features_ds.lat.values
        lon_values = features_ds.lon.values
        
        # Create spatial coordinates for each sample
        lat_grid, lon_grid = np.meshgrid(lat_values, lon_values, indexing='ij')
        spatial_coords_2d = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        
        # Repeat for each time step
        spatial_coords_full = np.tile(spatial_coords_2d, (n_times, 1))
        
        # Create temporal indices
        temporal_indices_full = np.repeat(np.arange(n_times), n_lat * n_lon)
        
        # Remove samples with NaN in target or features
        valid_mask = ~np.isnan(y_full) & ~np.any(np.isnan(X_full), axis=1)
        
        X = X_full[valid_mask]
        y = y_full[valid_mask]
        spatial_coords = spatial_coords_full[valid_mask]
        temporal_indices = temporal_indices_full[valid_mask]
        
        print(f"\n📊 Sample filtering:")
        print(f"   Total samples: {len(y_full):,}")
        print(f"   Valid samples: {len(y):,} ({100*len(y)/len(y_full):.1f}%)")
        print(f"   Removed: {len(y_full) - len(y):,} ({100*(len(y_full)-len(y))/len(y_full):.1f}%)")
        
        # Print statistics
        print_statistics(y, "GRACE targets")
        print(f"\n📋 Feature names ({len(feature_names)} total):")
        for i, name in enumerate(feature_names[:10]):
            print(f"   {i+1}. {name}")
        if len(feature_names) > 10:
            print(f"   ... and {len(feature_names) - 10} more")
        
        # Prepare metadata dictionary
        metadata = {
            'spatial_coords': spatial_coords,
            'temporal_indices': temporal_indices,
            'spatial_shape': (n_lat, n_lon),
            'n_times': n_times,
            'n_spatial': n_lat * n_lon,
            'common_dates': [t.strftime('%Y-%m') for t in times],
            'feature_names': feature_names
        }
        
        return X, y, feature_names, metadata
    
    def load_tuned_parameters(self, tuned_params_path: Path) -> None:
        """
        Load tuned hyperparameters from file.
        
        Parameters:
        -----------
        tuned_params_path : Path
            Path to tuned hyperparameters YAML file
        """
        try:
            with open(tuned_params_path, 'r') as f:
                content = f.read()
                
            # Try JSON first (since the file is actually JSON despite .yaml extension)
            try:
                import json
                self.tuned_params = json.loads(content)
            except json.JSONDecodeError:
                # Fallback to YAML
                self.tuned_params = yaml.safe_load(content)
            
            print(f"🎯 Loaded tuned hyperparameters:")
            for model_name in self.tuned_params.get('best_parameters', {}):
                if model_name in self.enabled_models:
                    print(f"   ✓ {model_name}: {len(self.tuned_params['best_parameters'][model_name])} parameters")
                    
        except Exception as e:
            print(f"⚠️ Warning: Could not load tuned parameters: {e}")
            print("   Falling back to default hyperparameters")
            self.tuned_params = {}
    
    def get_model_class_and_params(self, model_name: str) -> Tuple:
        """Get model class and hyperparameters from config or tuned parameters."""
        # Check if we have tuned parameters for this model
        if (self.tuned_params and 
            'best_parameters' in self.tuned_params and 
            model_name in self.tuned_params['best_parameters']):
            
            hyper_config = self.tuned_params['best_parameters'][model_name].copy()
            print(f"🎯 Using tuned parameters for {model_name}")
        else:
            # Fallback to config defaults
            hyper_config = get_config_value(self.config, f'models.hyperparameters.{model_name}', {})
        
        # Handle threading issues with NUMEXPR_MAX_THREADS
        import os
        max_threads = int(os.environ.get('NUMEXPR_MAX_THREADS', 64))
        
        # For simple split, we can use more threads since no concurrent CV training
        cv_method = get_config_value(self.config, 'models.cross_validation.method', 'blocked_spatiotemporal')
        if cv_method == 'simple_split':
            # Use more threads for simple split (no concurrent training)
            safe_threads = min(32, max(1, max_threads // 2))  # Use up to 32 threads (64÷2=32)
        else:
            # Conservative for blocked CV (concurrent training across folds)
            safe_threads = min(8, max(1, max_threads // 8))  # Use up to 8 threads per model (64÷8=8)
        
        if model_name == 'rf':
            # Remove any existing n_jobs and set safe value
            hyper_config.pop('n_jobs', None)
            hyper_config['n_jobs'] = safe_threads
            
            # Fix max_features_type issue - convert to proper max_features
            if 'max_features_type' in hyper_config:
                max_features_type = hyper_config.pop('max_features_type')
                if max_features_type == 'sqrt':
                    hyper_config['max_features'] = 'sqrt'
                elif max_features_type == 'log2':
                    hyper_config['max_features'] = 'log2'
                elif max_features_type == 'float':
                    # Use max_features_float if available
                    if 'max_features_float' in hyper_config:
                        hyper_config['max_features'] = hyper_config.pop('max_features_float')
                    else:
                        hyper_config['max_features'] = 'sqrt'  # Default fallback
                        
            # Remove any other non-sklearn parameters
            hyper_config.pop('max_features_float', None)
            
            return RandomForestRegressor, hyper_config
        elif model_name == 'gbr':
            return GradientBoostingRegressor, hyper_config
        elif model_name == 'nn':
            return MLPRegressor, hyper_config
        elif model_name == 'svr':
            return SVR, hyper_config
        elif model_name == 'xgb' and HAS_XGB:
            # Remove any existing threading parameters and set safe value
            hyper_config.pop('nthread', None)
            hyper_config.pop('n_jobs', None)
            hyper_config['nthread'] = safe_threads
            return xgb.XGBRegressor, hyper_config
        elif model_name == 'lgb' and HAS_LGB:
            # Remove any existing n_jobs and set safe value
            hyper_config.pop('n_jobs', None)
            hyper_config['n_jobs'] = safe_threads
            
            # Clean up conflicting LightGBM parameters
            # Remove duplicate parameters that cause warnings
            hyper_config.pop('colsample_bytree', None)  # Use feature_fraction instead
            hyper_config.pop('subsample', None)         # Use bagging_fraction instead  
            hyper_config.pop('subsample_freq', None)    # Use bagging_freq instead
            
            return lgb.LGBMRegressor, hyper_config
        elif model_name == 'catb' and HAS_CATB:
            return cb.CatBoostRegressor, hyper_config
        elif model_name == 'nn_scale_consistent' and HAS_TORCH:
            # Scale-consistent neural network with custom loss
            # Returns wrapper class that provides sklearn-compatible interface
            return ScaleConsistentNNWrapper, hyper_config
        else:
            raise ValueError(f"Model {model_name} not available")
    
    def needs_scaling(self, model_name: str) -> bool:
        """Check if model needs feature scaling."""
        # nn_scale_consistent handles its own scaling internally
        if model_name == 'nn_scale_consistent':
            return False
        
        models_needing_scaling = get_config_value(
            self.config,
            'models.scaling.models_needing_scaling',
            ['nn', 'svr']
        )
        return model_name in models_needing_scaling
    
    def set_fine_data_for_scale_consistent(self, 
                                           fine_features_ds: xr.Dataset,
                                           coarse_features_ds: xr.Dataset,
                                           grace_ds: xr.Dataset):
        """
        Set fine-resolution data required for scale-consistent NN training.
        
        This method prepares the coarse-to-fine mapping and fine features
        that are needed for the consistency loss computation.
        
        IMPORTANT: Fine features may have fewer features than coarse (which includes
        derived features like anomalies, lags, etc.). This method identifies the
        common features and stores indices for subsetting coarse features during training.
        
        Parameters:
        -----------
        fine_features_ds : xr.Dataset
            Fine resolution (5km) feature dataset
        coarse_features_ds : xr.Dataset
            Coarse resolution (55km) feature dataset
        grace_ds : xr.Dataset
            GRACE dataset (for coordinate reference)
        """
        if not HAS_TORCH:
            print("⚠️ PyTorch not available - cannot use scale-consistent NN")
            return
        
        print("\n" + "="*70)
        print("📦 PREPARING FINE DATA FOR SCALE-CONSISTENT TRAINING")
        print("="*70)
        
        # Get coordinates
        coarse_lat = coarse_features_ds.lat.values
        coarse_lon = coarse_features_ds.lon.values
        fine_lat = fine_features_ds.lat.values
        fine_lon = fine_features_ds.lon.values
        n_times = len(coarse_features_ds.time)
        
        aggregation_factor = get_config_value(self.config, 'resolution.aggregation_factor', 11)
        
        print(f"   Coarse grid: {len(coarse_lat)} × {len(coarse_lon)}")
        print(f"   Fine grid: {len(fine_lat)} × {len(fine_lon)}")
        print(f"   Time steps: {n_times}")
        print(f"   Aggregation factor: {aggregation_factor}")
        
        # Get feature names from both datasets
        fine_temporal_names = list(fine_features_ds.feature.values) if 'feature' in fine_features_ds.coords else []
        fine_static_names = list(fine_features_ds.static_feature.values) if 'static_feature' in fine_features_ds.coords else []
        coarse_temporal_names = list(coarse_features_ds.feature.values) if 'feature' in coarse_features_ds.coords else []
        coarse_static_names = list(coarse_features_ds.static_feature.values) if 'static_feature' in coarse_features_ds.coords else []
        
        # Combine feature names (temporal + static)
        fine_all_names = [str(n) for n in fine_temporal_names] + [str(n) for n in fine_static_names]
        coarse_all_names = [str(n) for n in coarse_temporal_names] + [str(n) for n in coarse_static_names]
        
        print(f"\n📊 Feature comparison:")
        print(f"   Coarse features: {len(coarse_all_names)} ({len(coarse_temporal_names)} temporal + {len(coarse_static_names)} static)")
        print(f"   Fine features: {len(fine_all_names)} ({len(fine_temporal_names)} temporal + {len(fine_static_names)} static)")
        
        # Find common features (features that exist in both datasets)
        # For scale-consistent training, we need features available at both scales
        common_features = []
        coarse_indices = []
        fine_indices = []
        
        for i, fname in enumerate(coarse_all_names):
            if fname in fine_all_names:
                common_features.append(fname)
                coarse_indices.append(i)
                fine_indices.append(fine_all_names.index(fname))
        
        print(f"   Common features: {len(common_features)}")
        
        if len(common_features) < len(fine_all_names):
            print(f"   ⚠️ Some fine features not in coarse: {set(fine_all_names) - set(common_features)}")
        
        if len(common_features) < len(coarse_all_names):
            print(f"   ⚠️ Some coarse features not in fine: {set(coarse_all_names) - set(common_features)}")
        
        # Store indices for subsetting features during training
        self.common_feature_indices = np.array(coarse_indices)
        self.fine_feature_indices = np.array(fine_indices)  # Also store fine indices!
        self.common_feature_names = common_features
        
        # Create coarse-to-fine mapping
        self.coarse_to_fine_mapping, fine_cell_counts = create_coarse_to_fine_mapping(
            coarse_lat, coarse_lon,
            fine_lat, fine_lon,
            n_times,
            aggregation_factor
        )
        
        # Prepare fine features (same format as coarse)
        print("\n🔄 Preparing fine features...")
        
        # Extract features from fine dataset
        if 'features' in fine_features_ds and 'static_features' in fine_features_ds:
            # Stacked format with separate temporal and static features
            temporal_features = fine_features_ds['features'].values  # (time, feature, lat, lon)
            static_features = fine_features_ds['static_features'].values  # (static_feature, lat, lon)
            
            # Transpose temporal features to (time, lat, lon, feature)
            temporal_features = temporal_features.transpose(0, 2, 3, 1)
            
            # Broadcast static features across time
            static_broadcasted = np.broadcast_to(
                static_features[np.newaxis, :, :, :],
                (n_times,) + static_features.shape
            ).transpose(0, 2, 3, 1)  # (time, lat, lon, static_feature)
            
            # Combine
            fine_feature_data = np.concatenate([temporal_features, static_broadcasted], axis=-1)
            
        elif 'features' in fine_features_ds:
            # Old stacked format
            fine_feature_data = fine_features_ds['features'].values
            if fine_feature_data.shape[1] != len(fine_lat):
                # Shape is (time, feature, lat, lon), transpose to (time, lat, lon, feature)
                fine_feature_data = fine_feature_data.transpose(0, 2, 3, 1)
        else:
            raise ValueError("Fine features dataset format not recognized")
        
        n_times_fine, n_lat_fine, n_lon_fine, n_features = fine_feature_data.shape
        
        # Flatten to (n_samples, n_features)
        X_fine_full = fine_feature_data.reshape(-1, n_features)
        
        # IMPORTANT: Subset fine features to ONLY the common features
        # This ensures fine and coarse have the same feature set
        X_fine_common = X_fine_full[:, self.fine_feature_indices]
        
        print(f"   Fine features subsetted: {n_features} → {X_fine_common.shape[1]} (common)")
        
        # Track valid samples - use a threshold instead of requiring all features valid
        # Lag features have NaN at the beginning, which is expected
        nan_counts = np.sum(np.isnan(X_fine_common), axis=1)
        max_nan_allowed = X_fine_common.shape[1] * 0.1  # Allow up to 10% NaN
        self.fine_valid_mask = nan_counts <= max_nan_allowed
        
        # For samples with some NaN, replace with 0 (the mask tracks validity)
        X_fine_common = np.nan_to_num(X_fine_common, nan=0.0)
        
        self.X_fine = X_fine_common.astype(np.float32)
        self.fine_features_ds = fine_features_ds
        
        n_fine_valid = np.sum(self.fine_valid_mask)
        print(f"   Fine samples: {len(self.X_fine):,} total, {n_fine_valid:,} valid ({100*n_fine_valid/len(self.X_fine):.1f}%)")
        print(f"   Fine features shape: {self.X_fine.shape}")
        print(f"\n✅ Fine data prepared for scale-consistent training")
        print(f"   🎯 Model will use {len(common_features)} common features at both scales")
    
    def train_with_blocked_cv(self,
                             X: np.ndarray,
                             y: np.ndarray,
                             metadata: Dict) -> pd.DataFrame:
        """
        Train all models with blocked spatiotemporal CV.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        metadata : Dict
            Metadata for CV
        
        Returns:
        --------
        pd.DataFrame
            CV results summary
        """
        print("\n" + "="*70)
        print("🔬 TRAINING MODELS WITH BLOCKED SPATIOTEMPORAL CV")
        print("="*70)
        
        # CV settings
        n_spatial_blocks = get_config_value(self.config, 'models.cross_validation.n_spatial_blocks', 5)
        n_temporal_blocks = get_config_value(self.config, 'models.cross_validation.n_temporal_blocks', 4)
        n_jobs = get_config_value(self.config, 'parallelization.cv_n_jobs', 2)
        
        print(f"   Spatial blocks: {n_spatial_blocks}")
        print(f"   Temporal blocks: {n_temporal_blocks}")
        print(f"   Expected folds: {n_spatial_blocks * n_temporal_blocks}")
        print(f"   Parallel jobs: {n_jobs}")
        
        all_results = []
        
        for model_name in self.enabled_models:
            print(f"\n{'='*70}")
            print(f"🎯 Training: {model_name.upper()}")
            print(f"{'='*70}")
            
            try:
                # Get model class and parameters
                model_class, model_params = self.get_model_class_and_params(model_name)
                
                # Check if needs scaling
                needs_scale = self.needs_scaling(model_name)
                
                print(f"   Model class: {model_class.__name__}")
                print(f"   Scaling: {'Yes' if needs_scale else 'No'}")
                print(f"   Hyperparameters: {len(model_params)} configured")
                
                # Run blocked CV
                cv_results_df = evaluate_with_blocked_cv(
                    model_class=model_class,
                    X=X,
                    y=y,
                    metadata=metadata,
                    model_params=model_params,
                    n_spatial_blocks=n_spatial_blocks,
                    n_temporal_blocks=n_temporal_blocks,
                    needs_scaling=needs_scale,
                    n_jobs=n_jobs
                )
                
                # Add model info
                cv_results_df['model_name'] = model_name
                
                # Store results
                self.cv_results[model_name] = cv_results_df
                
                # Calculate summary statistics
                summary = {
                    'model_name': model_name,
                    'mean_r2': cv_results_df['r2'].mean(),
                    'std_r2': cv_results_df['r2'].std(),
                    'mean_rmse': cv_results_df['rmse'].mean(),
                    'std_rmse': cv_results_df['rmse'].std(),
                    'mean_mae': cv_results_df['mae'].mean(),
                    'std_mae': cv_results_df['mae'].std(),
                    'n_folds': len(cv_results_df)
                }
                all_results.append(summary)
                
                print(f"\n✅ {model_name.upper()} CV complete!")
                print(f"   R² = {summary['mean_r2']:.4f} ± {summary['std_r2']:.4f}")
                print(f"   RMSE = {summary['mean_rmse']:.2f} ± {summary['std_rmse']:.2f}")
                
            except Exception as e:
                print(f"\n❌ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(all_results)
        
        # Print final summary
        print("\n" + "="*70)
        print("📊 BLOCKED CV RESULTS SUMMARY")
        print("="*70)
        print(summary_df.to_string(index=False))
        print("="*70)
        
        return summary_df
    
    def train_with_simple_split(self,
                               X: np.ndarray,
                               y: np.ndarray,
                               metadata: Dict) -> pd.DataFrame:
        """
        Train all models with traditional 70-30 split.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        metadata : Dict
            Metadata (kept for compatibility)
        
        Returns:
        --------
        pd.DataFrame
            Training results summary
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        print("\n" + "="*70)
        print("🔬 TRAINING MODELS WITH TRADITIONAL 70-30 SPLIT")
        print("="*70)
        
        # Get split configuration
        split_config = get_config_value(self.config, 'models.simple_split', {})
        train_fraction = split_config.get('train_fraction', 0.7)
        random_state = split_config.get('random_state', 42)
        shuffle = split_config.get('shuffle', True)
        
        print(f"   Train fraction: {train_fraction:.1%}")
        print(f"   Test fraction: {1-train_fraction:.1%}")
        print(f"   Random state: {random_state}")
        print(f"   Shuffle data: {shuffle}")
        print(f"   Total samples: {len(X)}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=train_fraction,
            random_state=random_state,
            shuffle=shuffle
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        all_results = []
        
        for model_name in self.enabled_models:
            print(f"\n{'='*70}")
            print(f"🎯 Training: {model_name.upper()}")
            print(f"{'='*70}")
            
            try:
                # Get model class and parameters
                model_class, model_params = self.get_model_class_and_params(model_name)
                
                # Check if needs scaling
                needs_scale = self.needs_scaling(model_name)
                
                print(f"   Model class: {model_class.__name__}")
                print(f"   Scaling: {'Yes' if needs_scale else 'No (handled internally)' if model_name == 'nn_scale_consistent' else 'No'}")
                print(f"   Hyperparameters: {len(model_params)} configured")
                
                # Special handling for scale-consistent neural network
                if model_name == 'nn_scale_consistent':
                    # Check if fine data is available
                    if self.X_fine is None or self.coarse_to_fine_mapping is None:
                        print("   ⚠️ Fine data not set - training without consistency loss")
                        print("   💡 Call set_fine_data_for_scale_consistent() before training for full benefits")
                    
                    # Subset coarse features to only use common features (available at both scales)
                    if hasattr(self, 'common_feature_indices') and self.common_feature_indices is not None:
                        X_common = X[:, self.common_feature_indices]
                        print(f"   📐 Using {len(self.common_feature_indices)} common features (subset of {X.shape[1]})")
                        print(f"      Features: {self.common_feature_names[:5]}... (and {len(self.common_feature_names)-5} more)")
                    else:
                        X_common = X
                        print(f"   📐 Using all {X.shape[1]} features (no common feature subset defined)")
                    
                    # Initialize model with config
                    model = model_class(**model_params)
                    model.config = self.config
                    
                    # Set fine data if available
                    if self.X_fine is not None and self.coarse_to_fine_mapping is not None:
                        model.set_fine_data(
                            self.X_fine,
                            self.coarse_to_fine_mapping,
                            self.fine_valid_mask
                        )
                    
                    # Train the model (it handles train/val split internally)
                    print(f"   Training {model_name.upper()} with scale-consistent loss...")
                    model.fit(X_common, y)  # Use common features - model handles its own split
                    
                    # Store feature indices for prediction
                    self.scale_consistent_feature_indices = self.common_feature_indices if hasattr(self, 'common_feature_indices') else None
                    
                    # Make predictions for metrics
                    y_pred = model.predict(X_common)
                    
                    # Calculate metrics on full data (model already validated internally)
                    train_r2 = r2_score(y, y_pred)
                    test_r2 = train_r2  # For scale-consistent, use internal validation
                    test_rmse = np.sqrt(mean_squared_error(y, y_pred))
                    test_mae = mean_absolute_error(y, y_pred)
                    
                    # Get internal validation metrics if available
                    if hasattr(model, 'trainer') and model.trainer is not None:
                        if model.trainer.history['val_loss']:
                            # Approximate R² from final validation loss
                            # MSE = variance * (1 - R²) => R² ≈ 1 - MSE/variance
                            val_mse = model.trainer.history['val_loss_pred'][-1]
                            y_var = np.var(y)
                            if y_var > 0:
                                test_r2 = max(0, 1 - val_mse / y_var)
                    
                    # Store the trained model
                    self.models[model_name] = model
                    self.scalers[model_name] = None  # Scaling handled internally
                    
                else:
                    # Standard model training
                    # Initialize model
                    model = model_class(**model_params)
                    
                    # Prepare training data (with scaling if needed)
                    X_train_scaled = X_train.copy()
                    X_test_scaled = X_test.copy()
                    scaler = None
                    
                    if needs_scale:
                        print(f"   Applying feature scaling...")
                        scaling_method = get_config_value(
                            self.config, 'models.scaling.method', 'robust'
                        )
                        
                        if scaling_method == 'robust':
                            scaler = RobustScaler()
                        else:
                            scaler = StandardScaler()
                        
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Store scaler
                        self.scalers[model_name] = scaler
                    
                    # Train the model
                    print(f"   Training {model_name.upper()}...")
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_train_pred = model.predict(X_train_scaled)
                    y_test_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    
                    # Store the trained model
                    self.models[model_name] = model
                
                # Store results (similar to CV format for compatibility)
                result_row = {
                    'fold': 1,  # Single fold for simple split
                    'r2': test_r2,
                    'rmse': test_rmse,
                    'mae': test_mae,
                    'train_r2': train_r2,
                    'model_name': model_name
                }
                
                # Store CV results for compatibility
                cv_results_df = pd.DataFrame([result_row])
                self.cv_results[model_name] = cv_results_df
                
                # Create summary
                summary = {
                    'model_name': model_name,
                    'mean_r2': test_r2,
                    'std_r2': 0.0,  # No std for single split
                    'mean_rmse': test_rmse,
                    'std_rmse': 0.0,
                    'mean_mae': test_mae,
                    'std_mae': 0.0,
                    'n_folds': 1,
                    'train_r2': train_r2
                }
                all_results.append(summary)
                
                print(f"\n✅ {model_name.upper()} training complete!")
                print(f"   Train R² = {train_r2:.4f}")
                print(f"   Test R² = {test_r2:.4f}")
                print(f"   Test RMSE = {test_rmse:.2f}")
                print(f"   Test MAE = {test_mae:.2f}")
                
            except Exception as e:
                print(f"\n❌ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(all_results)
        
        # Print final summary
        print("\n" + "="*70)
        print("📊 SIMPLE SPLIT RESULTS SUMMARY")
        print("="*70)
        print(summary_df.to_string(index=False))
        print("="*70)
        
        return summary_df
    
    def train_final_models(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          feature_names: List[str]):
        """
        Train final models on all data (for deployment).
        
        Parameters:
        -----------
        X : np.ndarray
            Full feature matrix
        y : np.ndarray
            Full target array
        feature_names : List[str]
            Feature names
        """
        print("\n" + "="*70)
        print("🎓 TRAINING FINAL MODELS ON FULL DATASET")
        print("="*70)
        
        for model_name in self.enabled_models:
            print(f"\n🔨 Training final {model_name.upper()} model...")
            
            try:
                # Get model and params
                model_class, model_params = self.get_model_class_and_params(model_name)
                
                # Initialize model
                model = model_class(**model_params)
                
                # Scale if needed
                if self.needs_scaling(model_name):
                    print(f"   Scaling features...")
                    scaling_method = get_config_value(
                        self.config, 'models.scaling.method', 'robust'
                    )
                    
                    if scaling_method == 'robust':
                        scaler = RobustScaler()
                    else:
                        scaler = StandardScaler()
                    
                    X_scaled = scaler.fit_transform(X)
                    self.scalers[model_name] = scaler
                else:
                    X_scaled = X
                    self.scalers[model_name] = None
                
                # Train
                print(f"   Fitting model on {len(X):,} samples...")
                model.fit(X_scaled, y)
                
                # Store
                self.models[model_name] = model
                
                # Quick validation
                y_pred = model.predict(X_scaled)
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                print(f"   ✅ Final model trained!")
                print(f"      Training R² = {r2:.4f}")
                print(f"      Training RMSE = {rmse:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error training final {model_name}: {e}")
                import traceback
                traceback.print_exc()
    
    def create_ensemble(self, summary_df: pd.DataFrame) -> Dict:
        """
        Create weighted ensemble from trained models.
        
        Parameters:
        -----------
        summary_df : pd.DataFrame
            CV results summary
        
        Returns:
        --------
        Dict with ensemble weights
        """
        print("\n" + "="*70)
        print("🎭 CREATING ENSEMBLE MODEL")
        print("="*70)
        
        # Handle case where no models were trained successfully
        if summary_df.empty or len(self.models) == 0:
            print("❌ No models available for ensemble creation")
            return {'method': 'none', 'weights': {}, 'models': []}
        
        # Handle single model case
        if len(self.models) == 1:
            model_name = list(self.models.keys())[0]
            print(f"📝 Single model detected: {model_name}")
            print(f"   Creating single-model 'ensemble' with weight 1.0")
            
            ensemble_info = {
                'method': 'single',
                'weights': {model_name: 1.0},
                'models': [model_name]
            }
            
            print(f"\n📊 Ensemble weights:")
            print(f"   {model_name}: 1.0000 (100.0%)")
            
            return ensemble_info
        
        # Multiple models case
        ensemble_config = get_config_value(self.config, 'models.ensemble', {})
        method = ensemble_config.get('method', 'weighted')
        weight_metric = ensemble_config.get('weight_metric', 'cv_r2')
        min_weight = ensemble_config.get('min_weight', 0.05)
        
        print(f"   Method: {method}")
        print(f"   Weight metric: {weight_metric}")
        
        if method == 'simple':
            # Equal weights
            weights = {model: 1.0 / len(self.models) for model in self.models.keys()}
        
        elif method == 'weighted':
            # Weight by CV performance
            if weight_metric == 'cv_r2':
                metric_values = summary_df.set_index('model_name')['mean_r2']
            elif weight_metric == 'cv_rmse':
                # Lower is better, so invert
                metric_values = 1.0 / summary_df.set_index('model_name')['mean_rmse']
            else:
                raise ValueError(f"Unknown weight metric: {weight_metric}")
            
            # Normalize to sum to 1
            total = metric_values.sum()
            weights = {model: max(val / total, min_weight) 
                      for model, val in metric_values.items()}
            
            # Renormalize after applying min_weight
            total = sum(weights.values())
            weights = {model: w / total for model, w in weights.items()}
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        print(f"\n📊 Ensemble weights:")
        for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"   {model}: {weight:.4f} ({100*weight:.1f}%)")
        
        ensemble_info = {
            'method': method,
            'weights': weights,
            'models': list(self.models.keys())
        }
        
        return ensemble_info
    
    def save_models(self, output_dir: str):
        """
        Save all trained models, scalers, and results.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("💾 SAVING TRAINED MODELS")
        print("="*70)
        
        # Save each model
        for model_name, model in self.models.items():
            # Handle scale-consistent NN separately (PyTorch model)
            if model_name == 'nn_scale_consistent' and HAS_TORCH:
                model_path = output_dir / f"{model_name}_coarse_model.pt"
                if hasattr(model, 'save'):
                    model.save(str(model_path))
                else:
                    # Fallback: save with joblib
                    joblib.dump(model, output_dir / f"{model_name}_coarse_model.joblib")
                print(f"✓ Saved {model_name}: {model_path}")
                
                # Save feature indices for scale-consistent model
                if hasattr(self, 'common_feature_indices') and self.common_feature_indices is not None:
                    feature_info = {
                        'common_feature_indices': self.common_feature_indices,
                        'common_feature_names': self.common_feature_names if hasattr(self, 'common_feature_names') else None
                    }
                    feature_info_path = output_dir / f"{model_name}_feature_info.joblib"
                    joblib.dump(feature_info, feature_info_path)
                    print(f"✓ Saved {model_name} feature info: {feature_info_path}")
            else:
                # Standard sklearn model
                model_path = output_dir / f"{model_name}_coarse_model.joblib"
                joblib.dump(model, model_path)
                print(f"✓ Saved {model_name}: {model_path}")
            
            # Save scaler if exists (not for scale-consistent NN which handles internally)
            if model_name != 'nn_scale_consistent' and self.scalers.get(model_name) is not None:
                scaler_path = output_dir / f"{model_name}_scaler.joblib"
                joblib.dump(self.scalers[model_name], scaler_path)
                print(f"✓ Saved {model_name} scaler: {scaler_path}")
        
        # Save CV results
        for model_name, cv_results in self.cv_results.items():
            cv_path = output_dir / f"{model_name}_cv_results.csv"
            cv_results.to_csv(cv_path, index=False)
            print(f"✓ Saved {model_name} CV results: {cv_path}")
        
        print(f"\n✅ All models saved to: {output_dir}")


def main():
    """Command-line interface for model training."""
    import argparse
    from src_new_approach.utils_downscaling import load_config, create_output_directories
    
    parser = argparse.ArgumentParser(
        description="Train ML models at coarse (55km) resolution"
    )
    parser.add_argument('--config',
                       default='src_new_approach/config_coarse_to_fine.yaml',
                       help='Configuration file')
    parser.add_argument('--features-coarse',
                       help='Coarse features (overrides config)')
    parser.add_argument('--grace-filled',
                       help='Gap-filled GRACE (overrides config)')
    parser.add_argument('--output-dir',
                       help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    create_output_directories(config)
    
    # Get paths
    features_path = args.features_coarse or config['paths']['feature_stack_coarse']
    grace_path = args.grace_filled or config['paths']['grace_filled']
    output_dir = args.output_dir or config['paths']['models']
    
    # Load data
    print("📂 Loading data...")
    features_ds = xr.open_dataset(features_path)
    grace_ds = xr.open_dataset(grace_path)
    
    # Initialize trainer
    trainer = CoarseModelTrainer(config)
    
    # Prepare data
    X, y, feature_names, metadata = trainer.prepare_training_data(features_ds, grace_ds)
    
    # Train with CV
    summary_df = trainer.train_with_blocked_cv(X, y, metadata)
    
    # Train final models
    trainer.train_final_models(X, y, feature_names)
    
    # Create ensemble
    ensemble_info = trainer.create_ensemble(summary_df)
    
    # Save everything
    trainer.save_models(output_dir)
    
    # Save summary and ensemble info
    summary_path = Path(output_dir) / "model_comparison_coarse.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved model comparison: {summary_path}")
    
    ensemble_path = Path(output_dir) / "ensemble_info.joblib"
    joblib.dump(ensemble_info, ensemble_path)
    print(f"✓ Saved ensemble info: {ensemble_path}")
    
    print("\n✅ Coarse model training complete!")


if __name__ == "__main__":
    main()

