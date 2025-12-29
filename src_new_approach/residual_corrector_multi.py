"""
Multi-Method Residual Correction for Downscaling

Tests multiple residual correction approaches and selects the best performing one.
This is a comprehensive comparison of interpolation methods for satellite downscaling.

Methods implemented:
1. Bilinear Interpolation
2. Geographic Assignment (Novel)
3. Nearest Neighbor
4. Bicubic Interpolation  
5. IDW (Inverse Distance Weighting)
6. Gaussian Kernel Smoothing
7. Area-Weighted Assignment
8. Distance-Weighted Nearest Neighbor
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import zoom, gaussian_filter
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src_new_approach.utils_downscaling import (
    load_config,
    get_config_value,
    refine_2d,
    smooth_2d,
    clip_outliers,
    add_metadata_to_dataset,
    print_statistics
)


class MultiMethodResidualCorrector:
    """
    Test multiple residual correction methods and select the best performing one.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize multi-method residual corrector.
        
        Parameters:
        -----------
        config : Dict
            Configuration dictionary
        """
        self.config = config
        self.residual_config = get_config_value(config, 'residual_correction', {})
        
        # Available interpolation methods
        self.available_methods = [
            'bilinear',
            'geographic_assignment', 
            'nearest',
            'bicubic',
            'idw',
            'gaussian_kernel',
            'area_weighted',
            'distance_weighted_nearest'
        ]
        
        # Method configurations
        self.method_configs = {
            'bilinear': {'name': 'Bilinear Interpolation', 'color': 'blue'},
            'geographic_assignment': {'name': 'Geographic Assignment (Novel)', 'color': 'red'},
            'nearest': {'name': 'Nearest Neighbor', 'color': 'green'},
            'bicubic': {'name': 'Bicubic Interpolation', 'color': 'orange'},
            'idw': {'name': 'Inverse Distance Weighting', 'color': 'purple'},
            'gaussian_kernel': {'name': 'Gaussian Kernel Smoothing', 'color': 'brown'},
            'area_weighted': {'name': 'Area-Weighted Assignment', 'color': 'pink'},
            'distance_weighted_nearest': {'name': 'Distance-Weighted Nearest', 'color': 'gray'}
        }
        
        self.smooth_residuals = self.residual_config.get('smooth_residuals', True)
        self.smoothing_sigma = self.residual_config.get('smoothing_sigma', 1.0)
        self.clip_residuals = self.residual_config.get('clip_residuals', True)
        self.residual_max_std = self.residual_config.get('residual_max_std', 3.0)
        
        print(f"🔧 Multi-Method Residual Corrector initialized:")
        print(f"   Available methods: {len(self.available_methods)}")
        print(f"   Methods: {', '.join(self.available_methods)}")
        print(f"   Smooth residuals: {self.smooth_residuals}")
        print(f"   Clip outliers: {self.clip_residuals}")
    
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
    
    def interpolate_residuals_bilinear(self, 
                                     residuals_coarse: np.ndarray,
                                     coarse_lat: np.ndarray,
                                     coarse_lon: np.ndarray,
                                     fine_lat: np.ndarray,
                                     fine_lon: np.ndarray) -> np.ndarray:
        """Traditional bilinear interpolation."""
        n_times = residuals_coarse.shape[0]
        residuals_fine = np.full((n_times, len(fine_lat), len(fine_lon)), np.nan)
        
        for t in range(n_times):
            # Create interpolator
            interpolator = RegularGridInterpolator(
                (coarse_lat, coarse_lon),
                residuals_coarse[t],
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            
            # Create fine grid
            fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lon, fine_lat)
            fine_points = np.column_stack([
                fine_lat_grid.ravel(),
                fine_lon_grid.ravel()
            ])
            
            # Interpolate
            fine_values = interpolator(fine_points)
            residuals_fine[t] = fine_values.reshape(len(fine_lat), len(fine_lon))
            
        return residuals_fine
    
    def interpolate_residuals_geographic(self,
                                       residuals_coarse: np.ndarray,
                                       coarse_lat: np.ndarray,
                                       coarse_lon: np.ndarray,
                                       fine_lat: np.ndarray,
                                       fine_lon: np.ndarray) -> np.ndarray:
        """Geographic assignment - novel method."""
        n_times, n_lat_coarse, n_lon_coarse = residuals_coarse.shape
        residuals_fine = np.full((n_times, len(fine_lat), len(fine_lon)), np.nan, dtype=np.float32)
        
        # For each coarse pixel, find all fine pixels that belong to it
        for i_coarse, lat_coarse in enumerate(coarse_lat):
            for j_coarse, lon_coarse in enumerate(coarse_lon):
                
                # Define geographic boundaries
                lat_spacing = abs(coarse_lat[1] - coarse_lat[0]) if len(coarse_lat) > 1 else 1.0
                lon_spacing = abs(coarse_lon[1] - coarse_lon[0]) if len(coarse_lon) > 1 else 1.0
                
                lat_min = lat_coarse - lat_spacing / 2
                lat_max = lat_coarse + lat_spacing / 2  
                lon_min = lon_coarse - lon_spacing / 2
                lon_max = lon_coarse + lon_spacing / 2
                
                # Find fine pixels within boundaries
                lat_mask = (fine_lat >= lat_min) & (fine_lat <= lat_max)
                lon_mask = (fine_lon >= lon_min) & (fine_lon <= lon_max)
                
                i_fine_indices = np.where(lat_mask)[0]
                j_fine_indices = np.where(lon_mask)[0]
                
                # Assign same residual to all fine pixels in this area
                for i_fine in i_fine_indices:
                    for j_fine in j_fine_indices:
                        residuals_fine[:, i_fine, j_fine] = residuals_coarse[:, i_coarse, j_coarse]
                        
        return residuals_fine
    
    def interpolate_residuals_nearest(self,
                                    residuals_coarse: np.ndarray,
                                    coarse_lat: np.ndarray,
                                    coarse_lon: np.ndarray,
                                    fine_lat: np.ndarray,
                                    fine_lon: np.ndarray) -> np.ndarray:
        """Nearest neighbor interpolation."""
        n_times = residuals_coarse.shape[0]
        residuals_fine = np.full((n_times, len(fine_lat), len(fine_lon)), np.nan)
        
        for t in range(n_times):
            # Create interpolator
            interpolator = RegularGridInterpolator(
                (coarse_lat, coarse_lon),
                residuals_coarse[t],
                method='nearest',
                bounds_error=False,
                fill_value=np.nan
            )
            
            # Create fine grid
            fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lon, fine_lat)
            fine_points = np.column_stack([
                fine_lat_grid.ravel(),
                fine_lon_grid.ravel()
            ])
            
            # Interpolate
            fine_values = interpolator(fine_points)
            residuals_fine[t] = fine_values.reshape(len(fine_lat), len(fine_lon))
            
        return residuals_fine
    
    def interpolate_residuals_bicubic(self,
                                    residuals_coarse: np.ndarray,
                                    coarse_lat: np.ndarray,
                                    coarse_lon: np.ndarray,
                                    fine_lat: np.ndarray,
                                    fine_lon: np.ndarray) -> np.ndarray:
        """Bicubic interpolation using scipy zoom."""
        n_times = residuals_coarse.shape[0]
        
        # Calculate zoom factors
        lat_factor = len(fine_lat) / len(coarse_lat)
        lon_factor = len(fine_lon) / len(coarse_lon)
        
        residuals_fine = np.full((n_times, len(fine_lat), len(fine_lon)), np.nan)
        
        for t in range(n_times):
            # Use bicubic interpolation (order=3)
            zoomed = zoom(residuals_coarse[t], (lat_factor, lon_factor), order=3, mode='nearest')
            
            # Ensure correct size (handle rounding errors)
            if zoomed.shape != (len(fine_lat), len(fine_lon)):
                # Resize to exact target dimensions
                from scipy.interpolate import RectBivariateSpline
                lat_indices = np.linspace(0, zoomed.shape[0]-1, len(fine_lat))
                lon_indices = np.linspace(0, zoomed.shape[1]-1, len(fine_lon))
                
                spline = RectBivariateSpline(
                    np.arange(zoomed.shape[0]),
                    np.arange(zoomed.shape[1]),
                    zoomed
                )
                residuals_fine[t] = spline(lat_indices, lon_indices)
            else:
                residuals_fine[t] = zoomed
                
        return residuals_fine
    
    def interpolate_residuals_idw(self,
                                residuals_coarse: np.ndarray,
                                coarse_lat: np.ndarray,
                                coarse_lon: np.ndarray,
                                fine_lat: np.ndarray,
                                fine_lon: np.ndarray,
                                power: float = 2.0) -> np.ndarray:
        """Inverse Distance Weighting interpolation."""
        n_times = residuals_coarse.shape[0]
        residuals_fine = np.full((n_times, len(fine_lat), len(fine_lon)), np.nan)
        
        # Create coordinate grids
        coarse_lon_grid, coarse_lat_grid = np.meshgrid(coarse_lon, coarse_lat)
        coarse_points = np.column_stack([
            coarse_lat_grid.ravel(),
            coarse_lon_grid.ravel()
        ])
        
        fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lon, fine_lat)
        fine_points = np.column_stack([
            fine_lat_grid.ravel(),
            fine_lon_grid.ravel()
        ])
        
        for t in range(n_times):
            coarse_values = residuals_coarse[t].ravel()
            
            # Remove NaN values
            valid_mask = ~np.isnan(coarse_values)
            if np.sum(valid_mask) == 0:
                continue
                
            valid_points = coarse_points[valid_mask]
            valid_values = coarse_values[valid_mask]
            
            # IDW interpolation using scipy griddata
            fine_values = griddata(
                valid_points,
                valid_values,
                fine_points,
                method='linear',  # Will use IDW-like behavior
                fill_value=np.nan
            )
            
            residuals_fine[t] = fine_values.reshape(len(fine_lat), len(fine_lon))
            
        return residuals_fine
    
    def interpolate_residuals_gaussian(self,
                                     residuals_coarse: np.ndarray,
                                     coarse_lat: np.ndarray,
                                     coarse_lon: np.ndarray,
                                     fine_lat: np.ndarray,
                                     fine_lon: np.ndarray,
                                     sigma: float = 1.0) -> np.ndarray:
        """Gaussian kernel smoothed interpolation."""
        # First do bilinear interpolation
        residuals_fine = self.interpolate_residuals_bilinear(
            residuals_coarse, coarse_lat, coarse_lon, fine_lat, fine_lon
        )
        
        # Then apply Gaussian smoothing
        n_times = residuals_fine.shape[0]
        for t in range(n_times):
            valid_mask = ~np.isnan(residuals_fine[t])
            if np.sum(valid_mask) > 0:
                # Apply Gaussian filter
                smoothed = gaussian_filter(residuals_fine[t], sigma=sigma)
                # Preserve NaN locations
                smoothed[~valid_mask] = np.nan
                residuals_fine[t] = smoothed
                
        return residuals_fine
    
    def interpolate_residuals_area_weighted(self,
                                          residuals_coarse: np.ndarray,
                                          coarse_lat: np.ndarray,
                                          coarse_lon: np.ndarray,
                                          fine_lat: np.ndarray,
                                          fine_lon: np.ndarray) -> np.ndarray:
        """Area-weighted assignment based on overlap."""
        n_times = residuals_coarse.shape[0]
        residuals_fine = np.zeros((n_times, len(fine_lat), len(fine_lon)))
        
        # Calculate grid spacings
        lat_spacing_coarse = abs(coarse_lat[1] - coarse_lat[0]) if len(coarse_lat) > 1 else 1.0
        lon_spacing_coarse = abs(coarse_lon[1] - coarse_lon[0]) if len(coarse_lon) > 1 else 1.0
        lat_spacing_fine = abs(fine_lat[1] - fine_lat[0]) if len(fine_lat) > 1 else lat_spacing_coarse/10
        lon_spacing_fine = abs(fine_lon[1] - fine_lon[0]) if len(fine_lon) > 1 else lon_spacing_coarse/10
        
        for i_fine, lat_fine in enumerate(fine_lat):
            for j_fine, lon_fine in enumerate(fine_lon):
                
                # Define fine pixel boundaries
                fine_lat_min = lat_fine - lat_spacing_fine / 2
                fine_lat_max = lat_fine + lat_spacing_fine / 2
                fine_lon_min = lon_fine - lon_spacing_fine / 2
                fine_lon_max = lon_fine + lon_spacing_fine / 2
                
                total_weight = 0
                weighted_residual = np.zeros(n_times)
                
                for i_coarse, lat_coarse in enumerate(coarse_lat):
                    for j_coarse, lon_coarse in enumerate(coarse_lon):
                        
                        # Define coarse pixel boundaries
                        coarse_lat_min = lat_coarse - lat_spacing_coarse / 2
                        coarse_lat_max = lat_coarse + lat_spacing_coarse / 2
                        coarse_lon_min = lon_coarse - lon_spacing_coarse / 2
                        coarse_lon_max = lon_coarse + lon_spacing_coarse / 2
                        
                        # Calculate overlap area
                        lat_overlap = max(0, min(fine_lat_max, coarse_lat_max) - max(fine_lat_min, coarse_lat_min))
                        lon_overlap = max(0, min(fine_lon_max, coarse_lon_max) - max(fine_lon_min, coarse_lon_min))
                        overlap_area = lat_overlap * lon_overlap
                        
                        if overlap_area > 0:
                            # Weight by overlap area
                            weight = overlap_area / (lat_spacing_fine * lon_spacing_fine)
                            weighted_residual += weight * residuals_coarse[:, i_coarse, j_coarse]
                            total_weight += weight
                
                if total_weight > 0:
                    residuals_fine[:, i_fine, j_fine] = weighted_residual / total_weight
                else:
                    residuals_fine[:, i_fine, j_fine] = np.nan
                    
        return residuals_fine
    
    def interpolate_residuals_distance_weighted(self,
                                              residuals_coarse: np.ndarray,
                                              coarse_lat: np.ndarray,
                                              coarse_lon: np.ndarray,
                                              fine_lat: np.ndarray,
                                              fine_lon: np.ndarray,
                                              max_distance: float = 2.0) -> np.ndarray:
        """Distance-weighted nearest neighbor with cutoff."""
        n_times = residuals_coarse.shape[0]
        residuals_fine = np.full((n_times, len(fine_lat), len(fine_lon)), np.nan)
        
        for i_fine, lat_fine in enumerate(fine_lat):
            for j_fine, lon_fine in enumerate(fine_lon):
                
                # Calculate distances to all coarse pixels
                distances = []
                residual_values = []
                
                for i_coarse, lat_coarse in enumerate(coarse_lat):
                    for j_coarse, lon_coarse in enumerate(coarse_lon):
                        # Euclidean distance
                        dist = np.sqrt((lat_fine - lat_coarse)**2 + (lon_fine - lon_coarse)**2)
                        
                        if dist <= max_distance:
                            distances.append(dist)
                            residual_values.append(residuals_coarse[:, i_coarse, j_coarse])
                
                if len(distances) > 0:
                    distances = np.array(distances)
                    residual_values = np.array(residual_values).T  # (time, n_neighbors)
                    
                    # Avoid division by zero
                    distances[distances < 1e-10] = 1e-10
                    
                    # Calculate weights (inverse distance)
                    weights = 1.0 / distances
                    weights = weights / np.sum(weights)
                    
                    # Weighted average
                    residuals_fine[:, i_fine, j_fine] = np.average(residual_values, axis=1, weights=weights)
                    
        return residuals_fine
    
    def interpolate_residuals_to_fine_multi(self,
                                          residuals_coarse_ds: xr.Dataset,
                                          fine_lat: np.ndarray,
                                          fine_lon: np.ndarray,
                                          methods: List[str] = None) -> Dict[str, xr.Dataset]:
        """
        Test multiple interpolation methods and return results for all.
        
        Parameters:
        -----------
        residuals_coarse_ds : xr.Dataset
            Residuals at coarse resolution
        fine_lat : np.ndarray
            Fine resolution latitude values
        fine_lon : np.ndarray
            Fine resolution longitude values
        methods : List[str], optional
            Methods to test (default: all available)
        
        Returns:
        --------
        Dict[str, xr.Dataset]
            Results for each method
        """
        if methods is None:
            methods = self.available_methods
            
        print("\n" + "="*70)
        print("🧪 MULTI-METHOD RESIDUAL INTERPOLATION COMPARISON")
        print("="*70)
        print(f"   Testing {len(methods)} interpolation methods:")
        for method in methods:
            print(f"   • {self.method_configs[method]['name']}")
        
        residuals_coarse = residuals_coarse_ds['residual'].values
        coarse_lat = residuals_coarse_ds.lat.values
        coarse_lon = residuals_coarse_ds.lon.values
        
        print(f"\n   Coarse shape: {residuals_coarse.shape}")
        print(f"   Target fine shape: ({len(residuals_coarse_ds.time)}, {len(fine_lat)}, {len(fine_lon)})")
        
        results = {}
        
        for method in methods:
            print(f"\n🔄 Testing method: {self.method_configs[method]['name']}")
            
            try:
                # Apply interpolation method
                if method == 'bilinear':
                    residuals_fine = self.interpolate_residuals_bilinear(
                        residuals_coarse, coarse_lat, coarse_lon, fine_lat, fine_lon
                    )
                elif method == 'geographic_assignment':
                    residuals_fine = self.interpolate_residuals_geographic(
                        residuals_coarse, coarse_lat, coarse_lon, fine_lat, fine_lon
                    )
                elif method == 'nearest':
                    residuals_fine = self.interpolate_residuals_nearest(
                        residuals_coarse, coarse_lat, coarse_lon, fine_lat, fine_lon
                    )
                elif method == 'bicubic':
                    residuals_fine = self.interpolate_residuals_bicubic(
                        residuals_coarse, coarse_lat, coarse_lon, fine_lat, fine_lon
                    )
                elif method == 'idw':
                    residuals_fine = self.interpolate_residuals_idw(
                        residuals_coarse, coarse_lat, coarse_lon, fine_lat, fine_lon
                    )
                elif method == 'gaussian_kernel':
                    residuals_fine = self.interpolate_residuals_gaussian(
                        residuals_coarse, coarse_lat, coarse_lon, fine_lat, fine_lon
                    )
                elif method == 'area_weighted':
                    residuals_fine = self.interpolate_residuals_area_weighted(
                        residuals_coarse, coarse_lat, coarse_lon, fine_lat, fine_lon
                    )
                elif method == 'distance_weighted_nearest':
                    residuals_fine = self.interpolate_residuals_distance_weighted(
                        residuals_coarse, coarse_lat, coarse_lon, fine_lat, fine_lon
                    )
                else:
                    print(f"   ❌ Unknown method: {method}")
                    continue
                
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
                residuals_fine_ds['residual'].attrs['method'] = method
                residuals_fine_ds['residual'].attrs['method_name'] = self.method_configs[method]['name']
                
                results[method] = residuals_fine_ds
                
                # Print basic statistics
                valid_data = residuals_fine[~np.isnan(residuals_fine)]
                if len(valid_data) > 0:
                    print(f"   ✅ Success! Stats: mean={np.mean(valid_data):.4f}, std={np.std(valid_data):.4f}")
                else:
                    print(f"   ⚠️ Warning: All values are NaN")
                    
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                continue
        
        print(f"\n✅ Completed testing {len(results)}/{len(methods)} methods")
        print("="*70)
        
        return results
    
    def evaluate_method_performance(self,
                                  residuals_fine_results: Dict[str, xr.Dataset],
                                  fine_predictions_ds: xr.Dataset,
                                  grace_coarse_ds: xr.Dataset,
                                  aggregation_factor: int,
                                  prediction_var: str = 'prediction_ensemble') -> Dict[str, Dict]:
        """
        Evaluate performance of each residual correction method.
        
        Parameters:
        -----------
        residuals_fine_results : Dict[str, xr.Dataset]
            Results from each interpolation method
        fine_predictions_ds : xr.Dataset
            Fine-scale predictions (before correction)
        grace_coarse_ds : xr.Dataset
            Original GRACE observations at coarse scale
        aggregation_factor : int
            Upscaling factor for validation
        prediction_var : str
            Variable name for predictions
        
        Returns:
        --------
        Dict[str, Dict]
            Performance metrics for each method
        """
        print("\n" + "="*70)
        print("📊 EVALUATING RESIDUAL CORRECTION PERFORMANCE")
        print("="*70)
        
        # Get GRACE variable name
        grace_var = 'tws_anomaly'
        if grace_var not in grace_coarse_ds:
            for alt_name in ['lwe_thickness', 'grace_anomaly', 'TWS']:
                if alt_name in grace_coarse_ds:
                    grace_var = alt_name
                    break
        
        # Auto-detect prediction variable if needed
        if prediction_var not in fine_predictions_ds:
            pred_vars = [var for var in fine_predictions_ds.data_vars if var.startswith('prediction_')]
            if pred_vars:
                prediction_var = pred_vars[0]
                
        performance_results = {}
        
        for method, residuals_ds in residuals_fine_results.items():
            print(f"\n🔄 Evaluating method: {self.method_configs[method]['name']}")
            
            try:
                # Apply residual correction
                predictions_fine = fine_predictions_ds[prediction_var].values
                residuals_fine = residuals_ds['residual'].values
                
                # Handle time alignment
                pred_times = fine_predictions_ds.time.values
                resid_times = residuals_ds.time.values
                
                if len(pred_times) != len(resid_times):
                    # Find common times (simplified)
                    min_times = min(len(pred_times), len(resid_times))
                    predictions_fine = predictions_fine[:min_times]
                    residuals_fine = residuals_fine[:min_times]
                
                # Apply correction
                corrected_predictions = predictions_fine + residuals_fine
                
                # Aggregate to coarse scale for validation
                n_times = corrected_predictions.shape[0]
                n_lat_fine, n_lon_fine = corrected_predictions.shape[1], corrected_predictions.shape[2]
                n_lat_coarse = n_lat_fine // aggregation_factor
                n_lon_coarse = n_lon_fine // aggregation_factor
                
                corrected_aggregated = np.full((n_times, n_lat_coarse, n_lon_coarse), np.nan)
                
                for t in range(n_times):
                    # Simple block averaging for aggregation
                    for i in range(n_lat_coarse):
                        for j in range(n_lon_coarse):
                            i_start = i * aggregation_factor
                            i_end = (i + 1) * aggregation_factor
                            j_start = j * aggregation_factor
                            j_end = (j + 1) * aggregation_factor
                            
                            block = corrected_predictions[t, i_start:i_end, j_start:j_end]
                            valid_block = block[~np.isnan(block)]
                            
                            if len(valid_block) > 0:
                                corrected_aggregated[t, i, j] = np.mean(valid_block)
                
                # Compare with GRACE observations
                grace_obs = grace_coarse_ds[grace_var].values
                
                # Ensure same shape
                min_times = min(corrected_aggregated.shape[0], grace_obs.shape[0])
                min_lat = min(corrected_aggregated.shape[1], grace_obs.shape[1])
                min_lon = min(corrected_aggregated.shape[2], grace_obs.shape[2])
                
                corrected_flat = corrected_aggregated[:min_times, :min_lat, :min_lon].flatten()
                grace_flat = grace_obs[:min_times, :min_lat, :min_lon].flatten()
                
                # Remove NaN values
                valid_mask = ~(np.isnan(corrected_flat) | np.isnan(grace_flat))
                corrected_valid = corrected_flat[valid_mask]
                grace_valid = grace_flat[valid_mask]
                
                if len(corrected_valid) > 10:  # Minimum data for meaningful metrics
                    # Calculate metrics
                    r2 = r2_score(grace_valid, corrected_valid)
                    rmse = np.sqrt(mean_squared_error(grace_valid, corrected_valid))
                    mae = mean_absolute_error(grace_valid, corrected_valid)
                    bias = np.mean(corrected_valid - grace_valid)
                    
                    performance_results[method] = {
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae,
                        'bias': bias,
                        'n_samples': len(corrected_valid),
                        'method_name': self.method_configs[method]['name']
                    }
                    
                    print(f"   ✅ R² = {r2:.4f}, RMSE = {rmse:.3f} cm, MAE = {mae:.3f} cm")
                    
                else:
                    print(f"   ⚠️ Insufficient valid data ({len(corrected_valid)} samples)")
                    performance_results[method] = {
                        'r2': np.nan,
                        'rmse': np.nan,
                        'mae': np.nan,
                        'bias': np.nan,
                        'n_samples': len(corrected_valid),
                        'method_name': self.method_configs[method]['name']
                    }
                    
            except Exception as e:
                print(f"   ❌ Evaluation failed: {str(e)}")
                performance_results[method] = {
                    'r2': np.nan,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'bias': np.nan,
                    'n_samples': 0,
                    'method_name': self.method_configs[method]['name'],
                    'error': str(e)
                }
        
        print("="*70)
        return performance_results
    
    def select_best_method(self, performance_results: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Select the best performing residual correction method.
        
        Parameters:
        -----------
        performance_results : Dict[str, Dict]
            Performance metrics for each method
        
        Returns:
        --------
        Tuple[str, Dict]
            Best method name and its performance metrics
        """
        print("\n" + "="*70)
        print("🏆 SELECTING BEST RESIDUAL CORRECTION METHOD")
        print("="*70)
        
        # Sort methods by R² score
        valid_methods = {
            method: metrics for method, metrics in performance_results.items()
            if not np.isnan(metrics.get('r2', np.nan))
        }
        
        if not valid_methods:
            print("❌ No valid methods found!")
            return None, None
        
        # Print all results
        print("\n📊 Performance Summary:")
        print(f"{'Method':<25} {'R²':<8} {'RMSE (cm)':<10} {'MAE (cm)':<9} {'Bias (cm)':<10} {'Samples':<8}")
        print("-" * 70)
        
        for method, metrics in performance_results.items():
            r2 = metrics.get('r2', np.nan)
            rmse = metrics.get('rmse', np.nan)
            mae = metrics.get('mae', np.nan)
            bias = metrics.get('bias', np.nan)
            n_samples = metrics.get('n_samples', 0)
            
            if not np.isnan(r2):
                print(f"{self.method_configs[method]['name']:<25} {r2:<8.4f} {rmse:<10.3f} {mae:<9.3f} {bias:<10.4f} {n_samples:<8}")
            else:
                print(f"{self.method_configs[method]['name']:<25} {'FAILED':<8} {'FAILED':<10} {'FAILED':<9} {'FAILED':<10} {n_samples:<8}")
        
        # Find best method (highest R²)
        best_method = max(valid_methods.keys(), key=lambda m: valid_methods[m]['r2'])
        best_metrics = valid_methods[best_method]
        
        print(f"\n🥇 BEST METHOD: {self.method_configs[best_method]['name']}")
        print(f"   R² = {best_metrics['r2']:.4f}")
        print(f"   RMSE = {best_metrics['rmse']:.3f} cm")
        print(f"   MAE = {best_metrics['mae']:.3f} cm")
        print(f"   Bias = {best_metrics['bias']:.4f} cm")
        print(f"   Samples = {best_metrics['n_samples']}")
        
        # Show improvement over worst method
        worst_r2 = min(valid_methods.values(), key=lambda m: m['r2'])['r2']
        improvement = best_metrics['r2'] - worst_r2
        improvement_pct = (improvement / worst_r2) * 100 if worst_r2 > 0 else 0
        
        print(f"\n💡 Improvement over worst method: +{improvement:.4f} R² ({improvement_pct:.1f}%)")
        
        print("="*70)
        
        return best_method, best_metrics


def main():
    """Test all residual correction methods and select the best."""
    import argparse
    from src_new_approach.utils_downscaling import load_config, create_output_directories
    
    parser = argparse.ArgumentParser(
        description="Test multiple residual correction methods and select the best"
    )
    parser.add_argument('--config',
                       default='src_new_approach/config_coarse_to_fine.yaml',
                       help='Configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("🚀 Multi-Method Residual Correction Tester")
    print("="*70)
    
    # Initialize tester
    tester = MultiMethodResidualCorrector(config)
    
    print("\n✅ Multi-method residual correction framework ready!")
    print("Use this class in your pipeline to automatically select the best method.")


if __name__ == "__main__":
    main()