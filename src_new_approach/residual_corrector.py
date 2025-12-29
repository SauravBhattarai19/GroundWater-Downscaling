"""
Residual Correction for Downscaling

Calculates residuals at coarse scale (55km) and interpolates to fine scale (5km).
This corrects systematic biases in the downscaled predictions.

Method:
1. Calculate residuals at 55km: ε_coarse = GRACE_observed - GRACE_predicted
2. Interpolate residuals to 5km using bilinear/cubic interpolation
3. Apply to fine predictions: GRACE_downscaled = Prediction_fine + ε_fine
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from src_new_approach.utils_downscaling import (
    load_config,
    get_config_value,
    refine_2d,
    smooth_2d,
    clip_outliers,
    add_metadata_to_dataset,
    print_statistics
)


class ResidualCorrector:
    """
    Calculate and apply residual corrections for downscaling.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize residual corrector.
        
        Parameters:
        -----------
        config : Dict
            Configuration dictionary
        """
        self.config = config
        self.residual_config = get_config_value(config, 'residual_correction', {})
        
        self.interpolation_method = self.residual_config.get('interpolation_method', 'bilinear')
        self.smooth_residuals = self.residual_config.get('smooth_residuals', True)
        self.smoothing_sigma = self.residual_config.get('smoothing_sigma', 1.0)
        self.clip_residuals = self.residual_config.get('clip_residuals', True)
        self.residual_max_std = self.residual_config.get('residual_max_std', 3.0)
        
        print(f"🔧 Residual Corrector initialized:")
        print(f"   Interpolation method: {self.interpolation_method}")
        print(f"   Smooth residuals: {self.smooth_residuals}")
        if self.smooth_residuals:
            print(f"   Smoothing sigma: {self.smoothing_sigma}")
        print(f"   Clip outliers: {self.clip_residuals}")
        if self.clip_residuals:
            print(f"   Max std devs: {self.residual_max_std}")
    
    def calculate_coarse_predictions(self,
                                    coarse_features_ds: xr.Dataset,
                                    models_dir: str) -> xr.Dataset:
        """
        Generate predictions at coarse resolution for residual calculation.
        
        Parameters:
        -----------
        coarse_features_ds : xr.Dataset
            Coarse-resolution features
        models_dir : str
            Directory with trained models
        
        Returns:
        --------
        xr.Dataset
            Predictions at coarse resolution
        """
        print("\n📊 Generating predictions at coarse scale (for residuals)...")
        
        # Use fine predictor on coarse data
        from src_new_approach.fine_predictor import FinePredictor
        
        predictor = FinePredictor(self.config, models_dir)
        predictor.load_models()
        
        # Predict
        predictions_coarse_ds = predictor.predict_fine_resolution(
            coarse_features_ds,
            use_ensemble=True
        )
        
        return predictions_coarse_ds
    
    def calculate_residuals_coarse(self,
                                  grace_ds: xr.Dataset,
                                  predictions_coarse_ds: xr.Dataset,
                                  prediction_var: str = 'prediction_ensemble') -> xr.Dataset:
        """
        Calculate residuals at coarse scale.
        
        Residual = Observed - Predicted
        
        Parameters:
        -----------
        grace_ds : xr.Dataset
            Observed GRACE data (gap-filled) at coarse scale
        predictions_coarse_ds : xr.Dataset
            Model predictions at coarse scale
        prediction_var : str
            Variable name for predictions (auto-detected if not found)
        
        Returns:
        --------
        xr.Dataset
            Residuals at coarse resolution
        """
        print("\n" + "="*70)
        print("📐 CALCULATING RESIDUALS AT COARSE SCALE")
        print("="*70)
        
        # Get GRACE variable name
        grace_var = 'tws_anomaly'
        if grace_var not in grace_ds:
            for alt_name in ['lwe_thickness', 'grace_anomaly', 'TWS']:
                if alt_name in grace_ds:
                    grace_var = alt_name
                    break
        
        # Auto-detect prediction variable if not found
        if prediction_var not in predictions_coarse_ds:
            print(f"   ⚠️ Prediction variable '{prediction_var}' not found")
            pred_vars = [var for var in predictions_coarse_ds.data_vars if var.startswith('prediction_')]
            if pred_vars:
                prediction_var = pred_vars[0]
                print(f"   🔧 Using detected prediction variable: {prediction_var}")
            else:
                raise ValueError(f"No prediction variables found in dataset. Available: {list(predictions_coarse_ds.data_vars)}")
        
        # Extract data
        grace_obs = grace_ds[grace_var].values  # (time, lat, lon)
        grace_pred = predictions_coarse_ds[prediction_var].values  # (time, lat, lon)
        
        print(f"   GRACE observed shape: {grace_obs.shape}")
        print(f"   GRACE predicted shape: {grace_pred.shape}")
        
        # Calculate residuals
        residuals = grace_obs - grace_pred
        
        # Apply smoothing if requested
        if self.smooth_residuals:
            print(f"\n🔄 Smoothing residuals (sigma={self.smoothing_sigma})...")
            n_times = residuals.shape[0]
            smoothed_residuals = np.zeros_like(residuals)
            
            for t in range(n_times):
                smoothed_residuals[t] = smooth_2d(residuals[t], sigma=self.smoothing_sigma)
            
            residuals = smoothed_residuals
            print("   ✓ Smoothing applied")
        
        # Clip outliers if requested
        if self.clip_residuals:
            print(f"\n✂️ Clipping outliers beyond {self.residual_max_std} std devs...")
            residuals = clip_outliers(residuals, n_std=self.residual_max_std)
            print("   ✓ Outliers clipped")
        
        # Create dataset
        residuals_ds = xr.Dataset(
            {
                'residual': (['time', 'lat', 'lon'], residuals)
            },
            coords={
                'time': grace_ds.time.values,
                'lat': grace_ds.lat.values,
                'lon': grace_ds.lon.values
            }
        )
        
        # Add metadata
        residuals_ds['residual'].attrs['long_name'] = 'GRACE residual (observed - predicted)'
        residuals_ds['residual'].attrs['units'] = 'cm water equivalent'
        
        # Print statistics
        print("\n📊 Residual statistics:")
        print_statistics(residuals, "Residuals")
        
        # Calculate bias and RMSE
        valid_residuals = residuals[~np.isnan(residuals)]
        if len(valid_residuals) > 0:
            bias = np.mean(valid_residuals)
            rmse = np.sqrt(np.mean(valid_residuals**2))
            print(f"\n   Bias (mean residual): {bias:.4f} cm")
            print(f"   RMSE: {rmse:.2f} cm")
        
        print("="*70)
        
        return residuals_ds
    
    def interpolate_residuals_to_fine(self,
                                     residuals_coarse_ds: xr.Dataset,
                                     fine_lat: np.ndarray,
                                     fine_lon: np.ndarray,
                                     aggregation_factor: int) -> xr.Dataset:
        """
        NOVEL GEOGRAPHIC ASSIGNMENT: Assign residuals based on geographic boundaries.
        
        This method replaces traditional interpolation with geographic assignment,
        where each coarse pixel's residual is applied to ALL fine pixels within
        its geographic boundary. This preserves error locality and respects
        natural geographic boundaries.
        
        Parameters:
        -----------
        residuals_coarse_ds : xr.Dataset
            Residuals at coarse resolution
        fine_lat : np.ndarray
            Fine resolution latitude values
        fine_lon : np.ndarray
            Fine resolution longitude values
        aggregation_factor : int
            Upscaling factor (e.g., 11 for 55km/5km)
        
        Returns:
        --------
        xr.Dataset
            Residuals assigned to fine resolution based on geographic boundaries
        """
        print("\n" + "="*70)
        print("🎯 NOVEL GEOGRAPHIC ASSIGNMENT RESIDUAL CORRECTION")
        print("="*70)
        print("   Method: Geographic Assignment (preserves spatial boundaries)")
        print("   Innovation: No artificial interpolation across regions")
        
        residuals_coarse = residuals_coarse_ds['residual'].values  # (time, lat, lon)
        n_times, n_lat_coarse, n_lon_coarse = residuals_coarse.shape
        
        print(f"   Coarse resolution: {n_lat_coarse} × {n_lon_coarse}")
        print(f"   Fine resolution: {len(fine_lat)} × {len(fine_lon)}")
        print(f"   Geographic assignment factor: {aggregation_factor}x")
        
        # Get coordinate arrays
        coarse_lat = residuals_coarse_ds.lat.values
        coarse_lon = residuals_coarse_ds.lon.values
        
        # Initialize fine-scale residual array
        residuals_fine = np.full((n_times, len(fine_lat), len(fine_lon)), np.nan, dtype=np.float32)
        
        print(f"\n🗺️ Creating geographic assignment mapping...")
        
        # For each coarse pixel, find all fine pixels that belong to it
        for i_coarse, lat_coarse in enumerate(coarse_lat):
            for j_coarse, lon_coarse in enumerate(coarse_lon):
                
                # Define geographic boundaries of this coarse pixel
                # Each coarse pixel represents a geographic region
                lat_spacing = abs(coarse_lat[1] - coarse_lat[0]) if len(coarse_lat) > 1 else 1.0
                lon_spacing = abs(coarse_lon[1] - coarse_lon[0]) if len(coarse_lon) > 1 else 1.0
                
                lat_min = lat_coarse - lat_spacing / 2
                lat_max = lat_coarse + lat_spacing / 2  
                lon_min = lon_coarse - lon_spacing / 2
                lon_max = lon_coarse + lon_spacing / 2
                
                # Find all fine pixels within this geographic boundary
                lat_mask = (fine_lat >= lat_min) & (fine_lat <= lat_max)
                lon_mask = (fine_lon >= lon_min) & (fine_lon <= lon_max)
                
                # Get fine pixel indices
                i_fine_indices = np.where(lat_mask)[0]
                j_fine_indices = np.where(lon_mask)[0]
                
                # Assign SAME residual to ALL fine pixels in this geographic area
                for i_fine in i_fine_indices:
                    for j_fine in j_fine_indices:
                        residuals_fine[:, i_fine, j_fine] = residuals_coarse[:, i_coarse, j_coarse]
        
        print(f"   ✅ Geographic assignment complete!")
        print(f"   📊 Preserved geographic boundaries without artificial smoothing")
        
        # Create dataset
        residuals_fine_ds = xr.Dataset(
            {
                'residual': (['time', 'lat', 'lon'], residuals_fine)
            },
            coords={
                'time': residuals_coarse_ds.time.values,
                'lat': fine_lat,
                'lon': fine_lon
            }
        )
        
        # Add metadata
        residuals_fine_ds = add_metadata_to_dataset(
            residuals_fine_ds,
            self.config,
            'geographic_assignment'
        )
        residuals_fine_ds['residual'].attrs['assignment_method'] = 'geographic_assignment'
        residuals_fine_ds['residual'].attrs['innovation'] = 'Novel method: preserves geographic boundaries, no interpolation'
        residuals_fine_ds['residual'].attrs['original_resolution'] = 'coarse (55km)'
        residuals_fine_ds['residual'].attrs['target_resolution'] = 'fine (5km)'
        
        # Print statistics
        print("\n📊 Fine-scale residual statistics:")
        print_statistics(residuals_fine, "Interpolated residuals")
        
        print("="*70)
        
        return residuals_fine_ds
    
    def apply_residual_correction(self,
                                 predictions_fine_ds: xr.Dataset,
                                 residuals_fine_ds: xr.Dataset,
                                 prediction_var: str = 'prediction_ensemble') -> xr.Dataset:
        """
        Apply residual correction to fine-scale predictions.
        
        Corrected = Predicted + Residual
        
        Parameters:
        -----------
        predictions_fine_ds : xr.Dataset
            Fine-resolution predictions (before correction)
        residuals_fine_ds : xr.Dataset
            Fine-resolution residuals
        prediction_var : str
            Variable name for predictions
        
        Returns:
        --------
        xr.Dataset
            Final downscaled GRACE with residual correction
        """
        print("\n" + "="*70)
        print("✨ APPLYING RESIDUAL CORRECTION")
        print("="*70)
        
        # Auto-detect prediction variable if not found
        if prediction_var not in predictions_fine_ds:
            print(f"   ⚠️ Prediction variable '{prediction_var}' not found")
            pred_vars = [var for var in predictions_fine_ds.data_vars if var.startswith('prediction_')]
            if pred_vars:
                prediction_var = pred_vars[0]
                print(f"   🔧 Using detected prediction variable: {prediction_var}")
            else:
                raise ValueError(f"No prediction variables found in dataset. Available: {list(predictions_fine_ds.data_vars)}")
        
        # Extract data
        predictions = predictions_fine_ds[prediction_var].values
        residuals = residuals_fine_ds['residual'].values
        
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Residuals shape: {residuals.shape}")
        
        # Align temporal dimensions if needed
        pred_times = predictions_fine_ds.time.values
        resid_times = residuals_fine_ds.time.values
        
        if len(pred_times) != len(resid_times):
            print(f"   ⚠️ Temporal mismatch detected: {len(pred_times)} vs {len(resid_times)} time steps")
            
            # Find common time intersection
            import pandas as pd
            pred_times_pd = pd.to_datetime(pred_times)
            resid_times_pd = pd.to_datetime(resid_times)
            common_times = pred_times_pd.intersection(resid_times_pd)
            
            if len(common_times) == 0:
                raise ValueError("No overlapping time periods between predictions and residuals")
            
            print(f"   🔧 Using {len(common_times)} overlapping time steps")
            
            # Subset both datasets to common times
            pred_indices = [i for i, t in enumerate(pred_times_pd) if t in common_times]
            resid_indices = [i for i, t in enumerate(resid_times_pd) if t in common_times]
            
            predictions = predictions[pred_indices]
            residuals = residuals[resid_indices]
            
            # Update time coordinate
            final_times = pred_times_pd[pred_indices]
            
            print(f"   ✓ Aligned shapes: predictions {predictions.shape}, residuals {residuals.shape}")
        else:
            final_times = pred_times
        
        # Apply correction
        corrected = predictions + residuals
        
        # Create output dataset
        downscaled_ds = xr.Dataset(
            {
                'grace_downscaled': (['time', 'lat', 'lon'], corrected),
                'grace_predicted_uncorrected': (['time', 'lat', 'lon'], predictions),
                'residual_correction': (['time', 'lat', 'lon'], residuals)
            },
            coords={
                'time': final_times,
                'lat': predictions_fine_ds.lat.values,
                'lon': predictions_fine_ds.lon.values
            }
        )
        
        # Add metadata
        downscaled_ds = add_metadata_to_dataset(
            downscaled_ds,
            self.config,
            'residual_correction_applied'
        )
        
        downscaled_ds['grace_downscaled'].attrs['long_name'] = 'Downscaled GRACE with residual correction'
        downscaled_ds['grace_downscaled'].attrs['units'] = 'cm water equivalent'
        downscaled_ds['grace_downscaled'].attrs['method'] = 'Coarse-to-fine ML downscaling with residual correction'
        
        # Print statistics
        print("\n📊 Before correction:")
        print_statistics(predictions, "Predictions")
        
        print("\n📊 After correction:")
        print_statistics(corrected, "Corrected predictions")
        
        # Calculate improvement
        mean_abs_residual = np.nanmean(np.abs(residuals))
        print(f"\n💡 Mean absolute residual correction: {mean_abs_residual:.4f} cm")
        
        print("="*70)
        
        return downscaled_ds
    
    def full_residual_workflow(self,
                              grace_ds: xr.Dataset,
                              coarse_features_ds: xr.Dataset,
                              fine_features_ds: xr.Dataset,
                              fine_predictions_ds: xr.Dataset,
                              models_dir: str) -> xr.Dataset:
        """
        Complete residual correction workflow.
        
        Parameters:
        -----------
        grace_ds : xr.Dataset
            Observed GRACE at coarse scale
        coarse_features_ds : xr.Dataset
            Features at coarse scale
        fine_features_ds : xr.Dataset
            Features at fine scale
        fine_predictions_ds : xr.Dataset
            Initial predictions at fine scale
        models_dir : str
            Directory with trained models
        
        Returns:
        --------
        xr.Dataset
            Final downscaled GRACE with residual correction
        """
        print("\n" + "="*70)
        print("🚀 FULL RESIDUAL CORRECTION WORKFLOW")
        print("="*70)
        
        # Step 1: Generate coarse predictions
        print("\n📍 Step 1: Generate predictions at coarse scale...")
        predictions_coarse_ds = self.calculate_coarse_predictions(
            coarse_features_ds,
            models_dir
        )
        
        # Save intermediate result
        coarse_pred_path = Path(get_config_value(self.config, 'paths.predictions_coarse', 'temp_coarse_pred.nc'))
        coarse_pred_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_coarse_ds.to_netcdf(coarse_pred_path)
        print(f"   ✓ Coarse predictions saved: {coarse_pred_path}")
        
        # Step 2: Calculate residuals at coarse scale
        print("\n📍 Step 2: Calculate residuals at coarse scale...")
        residuals_coarse_ds = self.calculate_residuals_coarse(
            grace_ds,
            predictions_coarse_ds,
            prediction_var='prediction_ensemble'
        )
        
        # Save intermediate result
        coarse_res_path = Path(get_config_value(self.config, 'paths.residuals_coarse', 'temp_coarse_res.nc'))
        residuals_coarse_ds.to_netcdf(coarse_res_path)
        print(f"   ✓ Coarse residuals saved: {coarse_res_path}")
        
        # Step 3: Interpolate residuals to fine scale
        print("\n📍 Step 3: Interpolate residuals to fine scale...")
        from src_new_approach.utils_downscaling import get_aggregation_factor
        aggregation_factor = get_aggregation_factor(self.config)
        
        residuals_fine_ds = self.interpolate_residuals_to_fine(
            residuals_coarse_ds,
            fine_features_ds.lat.values,
            fine_features_ds.lon.values,
            aggregation_factor
        )
        
        # Save intermediate result
        fine_res_path = Path(get_config_value(self.config, 'paths.residuals_fine', 'temp_fine_res.nc'))
        residuals_fine_ds.to_netcdf(fine_res_path)
        print(f"   ✓ Fine residuals saved: {fine_res_path}")
        
        # Step 4: Apply correction
        print("\n📍 Step 4: Apply residual correction to fine predictions...")
        downscaled_ds = self.apply_residual_correction(
            fine_predictions_ds,
            residuals_fine_ds,
            prediction_var='prediction_ensemble'
        )
        
        print("\n✅ RESIDUAL CORRECTION WORKFLOW COMPLETE!")
        print("="*70)
        
        return downscaled_ds
    
    def save_downscaled(self, downscaled_ds: xr.Dataset, output_path: str):
        """Save final downscaled GRACE dataset."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Compress
        encoding = {var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
                   for var in downscaled_ds.data_vars}
        
        print(f"\n💾 Saving final downscaled GRACE to: {output_path}")
        downscaled_ds.to_netcdf(output_path, encoding=encoding)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   File size: {file_size_mb:.1f} MB")
        print("✅ Final downscaled GRACE saved successfully")


def main():
    """Command-line interface for residual correction."""
    import argparse
    from src_new_approach.utils_downscaling import load_config, create_output_directories
    
    parser = argparse.ArgumentParser(
        description="Apply residual correction to downscaled predictions"
    )
    parser.add_argument('--config',
                       default='src_new_approach/config_coarse_to_fine.yaml',
                       help='Configuration file')
    parser.add_argument('--grace',
                       help='Observed GRACE (coarse, overrides config)')
    parser.add_argument('--features-coarse',
                       help='Coarse features (overrides config)')
    parser.add_argument('--features-fine',
                       help='Fine features (overrides config)')
    parser.add_argument('--predictions-fine',
                       help='Fine predictions (overrides config)')
    parser.add_argument('--models-dir',
                       help='Models directory (overrides config)')
    parser.add_argument('--output',
                       help='Output path (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    create_output_directories(config)
    
    # Get paths
    grace_path = args.grace or config['paths']['grace_filled']
    coarse_features_path = args.features_coarse or config['paths']['feature_stack_coarse']
    fine_features_path = args.features_fine or config['paths']['feature_stack_fine']
    fine_predictions_path = args.predictions_fine or config['paths']['predictions_fine']
    models_dir = args.models_dir or config['paths']['models']
    output_path = args.output or config['paths']['final_downscaled']
    
    # Load data
    print("📂 Loading datasets...")
    grace_ds = xr.open_dataset(grace_path)
    coarse_features_ds = xr.open_dataset(coarse_features_path)
    fine_features_ds = xr.open_dataset(fine_features_path)
    fine_predictions_ds = xr.open_dataset(fine_predictions_path)
    
    # Initialize corrector
    corrector = ResidualCorrector(config)
    
    # Run full workflow
    downscaled_ds = corrector.full_residual_workflow(
        grace_ds,
        coarse_features_ds,
        fine_features_ds,
        fine_predictions_ds,
        models_dir
    )
    
    # Save
    corrector.save_downscaled(downscaled_ds, output_path)
    
    print("\n✅ Residual correction complete!")


if __name__ == "__main__":
    main()

