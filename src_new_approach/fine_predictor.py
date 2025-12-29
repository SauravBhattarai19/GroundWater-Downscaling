"""
Fine-Scale Prediction

Apply trained coarse models to fine-resolution (5km) features.
This generates the initial high-resolution predictions before residual correction.
"""

import numpy as np
import xarray as xr
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src_new_approach.utils_downscaling import (
    load_config,
    get_config_value,
    add_metadata_to_dataset,
    print_statistics
)


class FinePredictor:
    """
    Apply coarse-trained models to fine-resolution features.
    """
    
    def __init__(self, config: Dict, models_dir: str):
        """
        Initialize predictor.
        
        Parameters:
        -----------
        config : Dict
            Configuration dictionary
        models_dir : str
            Directory containing trained models
        """
        self.config = config
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        self.ensemble_info = None
        
        print(f"🔧 Fine Predictor initialized:")
        print(f"   Models directory: {self.models_dir}")
    
    def load_models(self, model_names: Optional[List[str]] = None):
        """
        Load trained models and scalers.
        
        Parameters:
        -----------
        model_names : List[str], optional
            Models to load (default: all available)
        """
        print("\n📂 Loading trained models...")
        
        if model_names is None:
            # Find all model files
            model_files = list(self.models_dir.glob("*_coarse_model.joblib"))
            model_names = [f.stem.replace('_coarse_model', '') for f in model_files]
        
        if not model_names:
            raise FileNotFoundError(f"No models found in {self.models_dir}")
        
        print(f"   Found {len(model_names)} models: {model_names}")
        
        for model_name in model_names:
            # Load model
            model_path = self.models_dir / f"{model_name}_coarse_model.joblib"
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                print(f"   ✓ Loaded {model_name}")
                
                # Load scaler if exists
                scaler_path = self.models_dir / f"{model_name}_scaler.joblib"
                if scaler_path.exists():
                    self.scalers[model_name] = joblib.load(scaler_path)
                    print(f"     ✓ Loaded {model_name} scaler")
                else:
                    self.scalers[model_name] = None
            else:
                print(f"   ⚠️ Model file not found: {model_path}")
        
        # Load ensemble info if available
        ensemble_path = self.models_dir / "ensemble_info.joblib"
        if ensemble_path.exists():
            self.ensemble_info = joblib.load(ensemble_path)
            print(f"\n   ✓ Loaded ensemble configuration")
            print(f"     Method: {self.ensemble_info['method']}")
            print(f"     Weights: {self.ensemble_info['weights']}")
        
        print(f"\n✅ Loaded {len(self.models)} models successfully")
    
    def prepare_fine_features(self, features_ds: xr.Dataset) -> Tuple[np.ndarray, Dict]:
        """
        Prepare fine-resolution features for prediction.
        
        This now includes applying the same advanced feature engineering
        that was used during training to ensure compatibility.
        
        Parameters:
        -----------
        features_ds : xr.Dataset
            Fine-resolution feature dataset
        
        Returns:
        --------
        X_fine : np.ndarray
            Feature matrix (n_samples, n_features)
        metadata : Dict
            Metadata for reconstruction
        """
        print("\n📦 Preparing fine-resolution features...")
        
        # Apply advanced feature engineering if needed
        enable_advanced = self.config.get('feature_aggregation', {}).get('enable_advanced_features', False)
        if enable_advanced and 'features' in features_ds.data_vars:
            print("🚀 Applying advanced feature engineering to fine-resolution data...")
            features_ds = self._apply_advanced_features_to_fine(features_ds)
        
        # Extract features
        if 'features' in features_ds:
            # Stacked format
            temporal_data = features_ds['features'].values  # (time, feature, lat, lon)
            temporal_names = [str(f) if hasattr(f, 'item') else f 
                            for f in features_ds['feature'].values]
            
            # Handle static features if present - ensure exact match with training
            if 'static_features' in features_ds.data_vars:
                fine_static_data = features_ds['static_features'].values  # (feature, lat, lon)
                fine_static_names = [str(f) if hasattr(f, 'item') else f 
                                   for f in features_ds['static_feature'].values]
                
                # Load training static feature names to ensure exact match
                try:
                    coarse_path = self.config['paths']['feature_stack_coarse']
                    import xarray as xr
                    coarse_ds = xr.open_dataset(coarse_path)
                    training_static_names = [str(f) for f in coarse_ds.static_feature.values]
                    
                    print(f"   🎯 Training expects {len(training_static_names)} static features")
                    print(f"   📋 Fine data has {len(fine_static_names)} static features")
                    
                    # Find matching static features in correct order
                    static_indices = []
                    matched_static_names = []
                    
                    for train_static in training_static_names:
                        if train_static in fine_static_names:
                            idx = fine_static_names.index(train_static)
                            static_indices.append(idx)
                            matched_static_names.append(train_static)
                            print(f"   ✓ Found static feature {train_static} at index {idx}")
                        else:
                            # Handle soil feature naming discrepancy: soil_sand_0cm vs sand_0cm
                            if train_static.startswith('soil_'):
                                alt_name = train_static.replace('soil_', '')
                                if alt_name in fine_static_names:
                                    idx = fine_static_names.index(alt_name)
                                    static_indices.append(idx)
                                    matched_static_names.append(train_static)
                                    print(f"   ✓ Found static feature {train_static} (as {alt_name}) at index {idx}")
                                else:
                                    print(f"   ❌ Missing static feature {train_static} (tried {alt_name})")
                            else:
                                print(f"   ❌ Missing static feature {train_static}")
                    
                    if len(matched_static_names) != len(training_static_names):
                        print(f"   ⚠️ Static feature mismatch: expected {len(training_static_names)}, found {len(matched_static_names)}")
                    
                    # Extract matching static features
                    static_data = fine_static_data[static_indices, :, :]
                    static_names = matched_static_names
                    
                except Exception as e:
                    print(f"   ⚠️ Could not match static features ({e}), using all fine static features")
                    static_data = fine_static_data
                    static_names = fine_static_names
                
                # Broadcast static features across time
                n_times, n_temp_features, n_lat, n_lon = temporal_data.shape
                n_static_features = static_data.shape[0]
                
                # Expand static features to match temporal dimensions
                static_expanded = np.broadcast_to(
                    static_data[np.newaxis, :, :, :],  # Add time dimension
                    (n_times, n_static_features, n_lat, n_lon)
                )
                
                # Combine temporal and static features
                feature_data = np.concatenate([temporal_data, static_expanded], axis=1)
                feature_names = temporal_names + static_names
                
                print(f"   Combined {n_temp_features} temporal + {n_static_features} static = {len(feature_names)} total features")
            else:
                feature_data = temporal_data
                feature_names = temporal_names
            
            # Convert from (time, feature, lat, lon) to (time, lat, lon, feature) for processing
            feature_data = feature_data.transpose(0, 2, 3, 1)
        else:
            # Variable format
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
            
            feature_data = np.stack(feature_arrays, axis=-1)
        
        n_times, n_lat, n_lon, n_features = feature_data.shape
        
        print(f"   Fine features shape: {feature_data.shape}")
        print(f"   Number of features: {n_features}")
        print(f"   Spatial resolution: {n_lat} × {n_lon}")
        print(f"   Temporal resolution: {n_times} time steps")
        
        # Flatten spatial dimensions
        X_fine = feature_data.reshape(-1, n_features)
        
        # Track valid samples
        valid_mask = ~np.any(np.isnan(X_fine), axis=1)
        n_valid = np.sum(valid_mask)
        n_total = len(valid_mask)
        
        print(f"   Valid samples: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%)")
        
        metadata = {
            'shape': (n_times, n_lat, n_lon),
            'valid_mask': valid_mask,
            'feature_names': feature_names,
            'times': features_ds.time.values,
            'lat': features_ds.lat.values,
            'lon': features_ds.lon.values
        }
        
        return X_fine, metadata
    
    def predict_single_model(self,
                            X: np.ndarray,
                            model_name: str,
                            valid_mask: np.ndarray) -> np.ndarray:
        """
        Generate predictions using a single model.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (all samples, including invalid)
        model_name : str
            Name of model to use
        valid_mask : np.ndarray
            Boolean mask of valid samples
        
        Returns:
        --------
        np.ndarray
            Predictions (full length, NaN for invalid samples)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)
        
        # Initialize output
        predictions = np.full(len(X), np.nan)
        
        # Get valid samples
        X_valid = X[valid_mask]
        
        # Scale if needed
        if scaler is not None:
            X_valid = scaler.transform(X_valid)
        
        # Predict
        predictions[valid_mask] = model.predict(X_valid)
        
        return predictions
    
    def predict_ensemble(self,
                        X: np.ndarray,
                        valid_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate predictions using ensemble of models.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        valid_mask : np.ndarray
            Boolean mask of valid samples
        
        Returns:
        --------
        Dict with individual predictions and weighted ensemble
        """
        print(f"\n🔮 Generating ensemble predictions...")
        
        all_predictions = {}
        
        # Get predictions from each model
        for model_name in self.models.keys():
            print(f"   Predicting with {model_name}...")
            preds = self.predict_single_model(X, model_name, valid_mask)
            all_predictions[model_name] = preds
        
        # Create weighted ensemble if info available
        if self.ensemble_info is not None:
            weights = self.ensemble_info['weights']
            
            # Weighted average
            ensemble_preds = np.zeros(len(X))
            total_weight = 0.0
            
            for model_name, pred in all_predictions.items():
                if model_name in weights:
                    weight = weights[model_name]
                    ensemble_preds += weight * np.nan_to_num(pred, nan=0.0)
                    total_weight += weight
            
            ensemble_preds /= total_weight
            ensemble_preds[~valid_mask] = np.nan
            
            all_predictions['ensemble'] = ensemble_preds
            print(f"   ✓ Ensemble created (weighted average)")
        else:
            # Simple average
            valid_preds = np.array([p for p in all_predictions.values()])
            ensemble_preds = np.nanmean(valid_preds, axis=0)
            all_predictions['ensemble'] = ensemble_preds
            print(f"   ✓ Ensemble created (simple average)")
        
        return all_predictions
    
    def predict_fine_resolution(self,
                               features_ds: xr.Dataset,
                               use_ensemble: bool = True) -> xr.Dataset:
        """
        Generate fine-resolution predictions.
        
        Parameters:
        -----------
        features_ds : xr.Dataset
            Fine-resolution features
        use_ensemble : bool
            Whether to use ensemble or best single model
        
        Returns:
        --------
        xr.Dataset
            Predictions at fine resolution
        """
        print("\n" + "="*70)
        print("🔮 GENERATING FINE-RESOLUTION PREDICTIONS")
        print("="*70)
        
        # Prepare features
        X_fine, metadata = self.prepare_fine_features(features_ds)
        valid_mask = metadata['valid_mask']
        shape = metadata['shape']
        
        # Generate predictions
        if use_ensemble and len(self.models) > 1:
            print(f"\n🎭 Using ensemble of {len(self.models)} models")
            all_predictions = self.predict_ensemble(X_fine, valid_mask)
        else:
            # Use best single model (based on CV results)
            print(f"\n🎯 Using single model")
            
            # Load CV results to find best model
            best_model = None
            best_r2 = -np.inf
            
            for model_name in self.models.keys():
                cv_path = self.models_dir / f"{model_name}_cv_results.csv"
                if cv_path.exists():
                    import pandas as pd
                    cv_results = pd.read_csv(cv_path)
                    mean_r2 = cv_results['r2'].mean()
                    if mean_r2 > best_r2:
                        best_r2 = mean_r2
                        best_model = model_name
            
            if best_model is None:
                best_model = list(self.models.keys())[0]
            
            print(f"   Selected model: {best_model}")
            preds = self.predict_single_model(X_fine, best_model, valid_mask)
            all_predictions = {best_model: preds}
        
        # Reshape predictions to spatial grid
        n_times, n_lat, n_lon = shape
        
        predictions_ds = xr.Dataset(
            coords={
                'time': metadata['times'],
                'lat': metadata['lat'],
                'lon': metadata['lon']
            }
        )
        
        for pred_name, pred_values in all_predictions.items():
            # Reshape
            pred_grid = pred_values.reshape(n_times, n_lat, n_lon)
            
            # Add to dataset
            predictions_ds[f'prediction_{pred_name}'] = (
                ['time', 'lat', 'lon'],
                pred_grid
            )
            
            # Print statistics
            print(f"\n📊 {pred_name} predictions:")
            print_statistics(pred_values[valid_mask], pred_name)
        
        # Add metadata
        predictions_ds = add_metadata_to_dataset(
            predictions_ds, 
            self.config, 
            'fine_prediction'
        )
        predictions_ds.attrs['models_used'] = list(self.models.keys())
        predictions_ds.attrs['ensemble_method'] = (
            self.ensemble_info['method'] if self.ensemble_info else 'none'
        )
        
        print("\n✅ FINE-RESOLUTION PREDICTIONS COMPLETE")
        print(f"   Output shape: {n_times} × {n_lat} × {n_lon}")
        print(f"   Predictions: {list(all_predictions.keys())}")
        
        return predictions_ds
    
    def save_predictions(self, predictions_ds: xr.Dataset, output_path: str):
        """Save fine predictions to NetCDF."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Compress
        encoding = {var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
                   for var in predictions_ds.data_vars}
        
        print(f"\n💾 Saving predictions to: {output_path}")
        predictions_ds.to_netcdf(output_path, encoding=encoding)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   File size: {file_size_mb:.1f} MB")
        print("✅ Predictions saved successfully")
    
    def _apply_advanced_features_to_fine(self, features_ds: xr.Dataset) -> xr.Dataset:
        """
        Apply the same advanced feature engineering to fine-resolution data
        that was applied during training, using only the features that match training.
        """
        # Import the FeatureAggregator to use its advanced feature methods
        from src_new_approach.feature_aggregator import FeatureAggregator
        
        # Load the training feature names from the coarse dataset to ensure exact match
        coarse_ds = None
        try:
            coarse_path = self.config['paths']['feature_stack_coarse']
            import xarray as xr
            coarse_ds = xr.open_dataset(coarse_path)
            training_feature_names = list(coarse_ds.feature.values)
            training_static_names = list(coarse_ds.static_feature.values)
            
            print(f"   🎯 Matching training features: {len(training_feature_names)} temporal + {len(training_static_names)} static")
            
            # Extract only the base features that were used in training
            # Get the first 10 features from training (these are the base features before enhancement)
            training_base_features = list(training_feature_names[:10])
            
            print(f"   🎯 Training base features: {training_base_features}")
            
            # Find these exact features in the fine dataset
            fine_feature_names = list(features_ds['feature'].values)
            fine_indices = []
            matched_names = []
            
            for train_feat in training_base_features:
                if train_feat in fine_feature_names:
                    idx = fine_feature_names.index(train_feat)
                    fine_indices.append(idx)
                    matched_names.append(train_feat)
                    print(f"   ✓ Found {train_feat} at index {idx}")
                else:
                    print(f"   ❌ Missing {train_feat} in fine features")
            
            if len(matched_names) != 10:
                raise ValueError(f"Could not find all 10 training features. Found only: {matched_names}")
            
            # Extract the matching features in the correct order
            fine_features_subset = features_ds['features'][:, fine_indices, :, :].data
            fine_feature_names_subset = matched_names
            
            # Create a new dataset with matching base features
            subset_ds = xr.Dataset()
            subset_ds = subset_ds.assign_coords({
                'time': features_ds.time,
                'lat': features_ds.lat,
                'lon': features_ds.lon,
                'feature': fine_feature_names_subset,
                'static_feature': features_ds.static_feature
            })
            
            subset_ds['features'] = (['time', 'feature', 'lat', 'lon'], fine_features_subset)
            subset_ds['static_features'] = (['static_feature', 'lat', 'lon'], features_ds['static_features'].data)
            subset_ds.attrs = features_ds.attrs.copy()
            
            print(f"   📋 Using base features: {fine_feature_names_subset}")
            
        except Exception as e:
            print(f"   ⚠️ Could not load training features ({e}), using original fine features")
            subset_ds = features_ds
        
        # Create a temporary aggregator to use its methods
        aggregator = FeatureAggregator(self.config)
        
        # Apply advanced feature engineering to the subset
        enhanced_ds = aggregator.add_advanced_features_to_stacked(subset_ds)
        
        return enhanced_ds


def main():
    """Command-line interface for fine prediction."""
    import argparse
    from src_new_approach.utils_downscaling import load_config, create_output_directories
    
    parser = argparse.ArgumentParser(
        description="Apply coarse models to fine-resolution features"
    )
    parser.add_argument('--config',
                       default='src_new_approach/config_coarse_to_fine.yaml',
                       help='Configuration file')
    parser.add_argument('--features-fine',
                       help='Fine features (overrides config)')
    parser.add_argument('--models-dir',
                       help='Models directory (overrides config)')
    parser.add_argument('--output',
                       help='Output path (overrides config)')
    parser.add_argument('--no-ensemble',
                       action='store_true',
                       help='Use best single model instead of ensemble')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    create_output_directories(config)
    
    # Get paths
    features_path = args.features_fine or config['paths']['feature_stack_fine']
    models_dir = args.models_dir or config['paths']['models']
    output_path = args.output or config['paths']['predictions_fine']
    
    # Load fine features
    print("📂 Loading fine-resolution features...")
    features_ds = xr.open_dataset(features_path)
    
    # Initialize predictor
    predictor = FinePredictor(config, models_dir)
    
    # Load models
    predictor.load_models()
    
    # Generate predictions
    use_ensemble = not args.no_ensemble
    predictions_ds = predictor.predict_fine_resolution(features_ds, use_ensemble)
    
    # Save
    predictor.save_predictions(predictions_ds, output_path)
    
    print("\n✅ Fine-resolution prediction complete!")


if __name__ == "__main__":
    main()

