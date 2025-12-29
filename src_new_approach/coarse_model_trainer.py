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
        
        self.enabled_models = [m for m in self.enabled_models if m in available_models]
        
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
        else:
            raise ValueError(f"Model {model_name} not available")
    
    def needs_scaling(self, model_name: str) -> bool:
        """Check if model needs feature scaling."""
        models_needing_scaling = get_config_value(
            self.config,
            'models.scaling.models_needing_scaling',
            ['nn', 'svr']
        )
        return model_name in models_needing_scaling
    
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
                print(f"   Scaling: {'Yes' if needs_scale else 'No'}")
                print(f"   Hyperparameters: {len(model_params)} configured")
                
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
                
                # Store the trained model
                self.models[model_name] = model
                
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
            model_path = output_dir / f"{model_name}_coarse_model.joblib"
            joblib.dump(model, model_path)
            print(f"✓ Saved {model_name}: {model_path}")
            
            # Save scaler if exists
            if self.scalers.get(model_name) is not None:
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

