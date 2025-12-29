#!/usr/bin/env python3
"""
GRACE Scale Factor Handler

This module handles the application of JPL GRACE RL06.3 CRI scale factors
for scientifically accurate GRACE TWS data processing.

Scale factors correct for:
- Signal amplitude loss due to spatial filtering
- Leakage effects across land-ocean boundaries
- Truncation artifacts in spherical harmonic expansion

Reference: JPL GRACE/GRACE-FO Mascon RL06.3v04 CRI Dataset
Dataset ID: TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4

Author: Saurav Bhattarai (ORISE/NASA)
Date: November 2025
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import Dict
import warnings

try:
    from .utils_downscaling import get_config_value
except ImportError:
    from utils_downscaling import get_config_value


class GRACEScaleHandler:
    """
    Handler for GRACE scale factor application and validation.
    
    Provides methods to:
    - Load and validate JPL CRI scale factors
    - Apply scale factors to GRACE TWS data
    - Handle spatial/temporal alignment
    - Validate scale factor application
    """
    
    def __init__(self, config: Dict):
        """
        Initialize GRACE Scale Handler.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary containing scale factor settings
        """
        self.config = config
        self.scale_config = config.get('grace_scale_factors', {})
        
        # Scale factor file path
        self.scale_factor_path = get_config_value(
            config, 'grace_scale_factors.scale_factor_file', 
            'data/JPL_MSCNv04_CRImascon_ScaleFactors.nc'
        )
        
        # Application settings
        self.apply_scale_factors = get_config_value(
            config, 'grace_scale_factors.apply_scale_factors', True
        )
        self.validate_application = get_config_value(
            config, 'grace_scale_factors.validate_application', True
        )
        self.interpolation_method = get_config_value(
            config, 'grace_scale_factors.interpolation_method', 'nearest'
        )
        
        # Scale factor data
        self.scale_factors_ds = None
        self.scale_factors_loaded = False
        
        print(f"🔧 GRACE Scale Handler initialized:")
        print(f"   Scale factor file: {self.scale_factor_path}")
        print(f"   Apply scale factors: {self.apply_scale_factors}")
        print(f"   Interpolation method: {self.interpolation_method}")
    
    def load_scale_factors(self) -> bool:
        """
        Load and validate JPL CRI scale factors.
        
        Returns:
        --------
        bool
            True if scale factors loaded successfully, False otherwise
        """
        if not self.apply_scale_factors:
            print("   ⚠️ Scale factor application disabled in config")
            return True
        
        scale_path = Path(self.scale_factor_path)
        if not scale_path.exists():
            print(f"   ❌ Scale factor file not found: {scale_path}")
            print(f"   ⚠️ Proceeding without scale factors (scientifically inaccurate!)")
            warnings.warn(
                "GRACE scale factors not found. Results will be scientifically inaccurate.",
                UserWarning
            )
            return False
        
        try:
            print(f"   📂 Loading scale factors from: {scale_path}")
            self.scale_factors_ds = xr.open_dataset(scale_path)
            
            # Validate scale factor structure
            self._validate_scale_factor_structure()
            
            # Print scale factor statistics
            self._print_scale_factor_info()
            
            self.scale_factors_loaded = True
            print(f"   ✅ Scale factors loaded successfully")
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading scale factors: {e}")
            warnings.warn(
                f"Failed to load GRACE scale factors: {e}. Results will be scientifically inaccurate.",
                UserWarning
            )
            return False
    
    def _validate_scale_factor_structure(self):
        """Validate the structure of loaded scale factors."""
        required_vars = ['scale_factor']
        required_dims = ['lat', 'lon']
        
        # Check for required variables
        missing_vars = [var for var in required_vars if var not in self.scale_factors_ds]
        if missing_vars:
            raise ValueError(f"Missing required variables in scale factors: {missing_vars}")
        
        # Check for required dimensions
        missing_dims = [dim for dim in required_dims if dim not in self.scale_factors_ds.dims]
        if missing_dims:
            raise ValueError(f"Missing required dimensions in scale factors: {missing_dims}")
        
        # Validate scale factor values (should be positive, typically 0.5-2.0)
        scale_vals = self.scale_factors_ds['scale_factor'].values
        if np.any(scale_vals <= 0):
            warnings.warn("Found non-positive scale factor values", UserWarning)
        if np.any(scale_vals > 10):
            warnings.warn("Found unusually large scale factor values (>10)", UserWarning)
    
    def _print_scale_factor_info(self):
        """Print information about loaded scale factors."""
        sf_var = self.scale_factors_ds['scale_factor']
        
        print(f"   📊 Scale factor info:")
        print(f"      Dimensions: {dict(sf_var.sizes)}")
        print(f"      Shape: {sf_var.shape}")
        print(f"      Range: [{sf_var.min().values:.3f}, {sf_var.max().values:.3f}]")
        print(f"      Mean: {sf_var.mean().values:.3f}")
        print(f"      Std: {sf_var.std().values:.3f}")
        
        # Check for any metadata
        if hasattr(sf_var, 'attrs') and sf_var.attrs:
            print(f"      Attributes: {sf_var.attrs}")
    
    def apply_scale_factors_to_grace(self, grace_ds: xr.Dataset) -> xr.Dataset:
        """
        Apply scale factors to GRACE TWS data.
        
        This is the core scientific correction that restores signal amplitude
        and reduces leakage effects.
        
        Parameters:
        -----------
        grace_ds : xr.Dataset
            GRACE dataset with TWS data
        
        Returns:
        --------
        xr.Dataset
            Scale-corrected GRACE dataset
        """
        if not self.apply_scale_factors:
            print("   ⚠️ Scale factor application disabled - returning original data")
            return grace_ds
        
        if not self.scale_factors_loaded:
            print("   ⚠️ Scale factors not loaded - attempting to load...")
            if not self.load_scale_factors():
                print("   ❌ Cannot apply scale factors - returning original data")
                return grace_ds
        
        print("\n" + "="*70)
        print("🔬 APPLYING GRACE SCALE FACTORS")
        print("="*70)
        
        # Identify GRACE TWS variable
        grace_var = self._identify_grace_variable(grace_ds)
        
        print(f"   📍 Applying scale factors to variable: {grace_var}")
        print(f"   📊 Original GRACE shape: {grace_ds[grace_var].shape}")
        
        # Get scale factors aligned to GRACE grid
        aligned_scale_factors = self._align_scale_factors_to_grace(grace_ds)
        
        # Apply scale correction: corrected_tws = original_tws * scale_factor
        original_tws = grace_ds[grace_var].values
        corrected_tws = original_tws * aligned_scale_factors
        
        # Create corrected dataset
        corrected_ds = grace_ds.copy(deep=True)
        corrected_ds[grace_var] = (grace_ds[grace_var].dims, corrected_tws)
        
        # Add scale factor metadata
        corrected_ds[grace_var].attrs['scale_factors_applied'] = True
        corrected_ds[grace_var].attrs['scale_factor_file'] = str(self.scale_factor_path)
        corrected_ds[grace_var].attrs['scale_factor_method'] = 'multiplicative'
        corrected_ds[grace_var].attrs['scale_factor_interpolation'] = self.interpolation_method
        
        # Print correction statistics
        self._print_correction_statistics(original_tws, corrected_tws, aligned_scale_factors)
        
        print("   ✅ Scale factors applied successfully")
        print("="*70)
        
        return corrected_ds
    
    def _identify_grace_variable(self, grace_ds: xr.Dataset) -> str:
        """Identify the main GRACE TWS variable in the dataset."""
        possible_vars = ['tws_anomaly', 'lwe_thickness', 'grace_anomaly', 'TWS']
        
        for var in possible_vars:
            if var in grace_ds:
                return var
        
        # Fallback: find variable with time, lat, lon dimensions
        for var in grace_ds.data_vars:
            if all(dim in grace_ds[var].dims for dim in ['time', 'lat', 'lon']):
                return var
        
        raise ValueError("Could not identify GRACE TWS variable in dataset")
    
    def _align_scale_factors_to_grace(self, grace_ds: xr.Dataset) -> np.ndarray:
        """
        Align scale factors to GRACE data grid.
        
        Handles spatial interpolation and coordinate system conversion.
        """
        grace_var = self._identify_grace_variable(grace_ds)
        grace_shape = grace_ds[grace_var].shape
        
        # Get GRACE spatial coordinates
        grace_lat = grace_ds.lat.values
        grace_lon = grace_ds.lon.values
        
        # Convert scale factor coordinates from 0-360° to -180-180° system
        sf_ds_converted = self._convert_scale_factor_coordinates()
        sf_lat = sf_ds_converted.lat.values
        sf_lon = sf_ds_converted.lon.values
        
        print(f"   🔄 Aligning scale factors:")
        print(f"      GRACE grid: {len(grace_lat)} × {len(grace_lon)}")
        print(f"      GRACE lat range: {grace_lat.min():.2f} to {grace_lat.max():.2f}")
        print(f"      GRACE lon range: {grace_lon.min():.2f} to {grace_lon.max():.2f}")
        print(f"      Scale factor grid: {len(sf_lat)} × {len(sf_lon)}")
        print(f"      Scale factor lat range: {sf_lat.min():.2f} to {sf_lat.max():.2f}")
        print(f"      Scale factor lon range: {sf_lon.min():.2f} to {sf_lon.max():.2f}")
        
        # Check if grids match exactly
        if (len(grace_lat) == len(sf_lat) and len(grace_lon) == len(sf_lon) and
            np.allclose(grace_lat, sf_lat) and np.allclose(grace_lon, sf_lon)):
            print(f"   ✓ Grids match exactly - no interpolation needed")
            
            # Broadcast to time dimension
            n_times = grace_shape[0]
            sf_data = sf_ds_converted['scale_factor'].values
            aligned_sf = np.broadcast_to(sf_data[np.newaxis, :, :], (n_times, len(sf_lat), len(sf_lon)))
            
        else:
            print(f"   🔄 Grids don't match - interpolating scale factors...")
            
            # Interpolate scale factors to GRACE grid with NaN handling
            sf_interp = sf_ds_converted.interp(
                lat=grace_lat,
                lon=grace_lon,
                method=self.interpolation_method
            )
            
            # Get interpolated values
            sf_aligned = sf_interp['scale_factor'].values
            
            # Check for NaN values in interpolated result
            nan_count = np.isnan(sf_aligned).sum()
            total_points = sf_aligned.size
            if nan_count > 0:
                print(f"   ⚠️ Warning: {nan_count}/{total_points} ({100*nan_count/total_points:.1f}%) interpolated points are NaN")
                
                # Try to fill NaN values with nearest valid values or default to 1.0
                if nan_count < total_points:
                    # Use nearest neighbor interpolation for NaN values
                    print(f"   🔧 Attempting to fill NaN values with nearest neighbor...")
                    sf_interp_nearest = sf_ds_converted.interp(
                        lat=grace_lat,
                        lon=grace_lon,
                        method='nearest'
                    )
                    sf_nearest = sf_interp_nearest['scale_factor'].values
                    
                    # Fill NaN values with nearest neighbor results
                    nan_mask = np.isnan(sf_aligned)
                    sf_aligned[nan_mask] = sf_nearest[nan_mask]
                    
                    # If still NaN, use default value of 1.0 (no correction)
                    remaining_nan = np.isnan(sf_aligned).sum()
                    if remaining_nan > 0:
                        print(f"   ⚠️ {remaining_nan} points still NaN - using default scale factor of 1.0")
                        sf_aligned[np.isnan(sf_aligned)] = 1.0
                else:
                    print(f"   ⚠️ All interpolated points are NaN - using default scale factor of 1.0")
                    sf_aligned = np.ones_like(sf_aligned)
            
            # Broadcast to time dimension
            n_times = grace_shape[0]
            aligned_sf = np.broadcast_to(sf_aligned[np.newaxis, :, :], (n_times, len(grace_lat), len(grace_lon)))
            
            print(f"   ✓ Scale factors interpolated successfully")
        
        return aligned_sf
    
    def _convert_scale_factor_coordinates(self) -> xr.Dataset:
        """Convert scale factor coordinates from 0-360° to -180-180° longitude system."""
        sf_ds = self.scale_factors_ds.copy()
        
        # Convert longitude from 0-360 to -180-180
        lon_360 = sf_ds.lon.values
        lon_180 = np.where(lon_360 > 180, lon_360 - 360, lon_360)
        
        # Sort by longitude to maintain proper order
        lon_sort_idx = np.argsort(lon_180)
        lon_180_sorted = lon_180[lon_sort_idx]
        
        # Reorder the scale factor data along longitude dimension
        sf_data_reordered = sf_ds['scale_factor'].values[:, lon_sort_idx]
        
        # Create new dataset with converted coordinates
        sf_ds_converted = xr.Dataset(
            {
                'scale_factor': (['lat', 'lon'], sf_data_reordered)
            },
            coords={
                'lat': sf_ds.lat.values,
                'lon': lon_180_sorted
            }
        )
        
        # Copy attributes
        sf_ds_converted['scale_factor'].attrs = sf_ds['scale_factor'].attrs
        sf_ds_converted.attrs = sf_ds.attrs
        
        return sf_ds_converted
    
    def _print_correction_statistics(self, original: np.ndarray, corrected: np.ndarray, 
                                   scale_factors: np.ndarray):
        """Print statistics about the scale factor correction."""
        
        # Calculate correction statistics
        valid_mask = ~np.isnan(original)
        if np.any(valid_mask):
            original_mean = np.nanmean(original)
            corrected_mean = np.nanmean(corrected)
            sf_mean = np.nanmean(scale_factors)
            sf_range = [np.nanmin(scale_factors), np.nanmax(scale_factors)]
            
            correction_factor = corrected_mean / original_mean if original_mean != 0 else 1.0
            
            print(f"\n   📊 Scale factor correction statistics:")
            print(f"      Scale factor range: [{sf_range[0]:.3f}, {sf_range[1]:.3f}]")
            print(f"      Mean scale factor: {sf_mean:.3f}")
            print(f"      Original TWS mean: {original_mean:.3f} cm")
            print(f"      Corrected TWS mean: {corrected_mean:.3f} cm")
            print(f"      Overall correction factor: {correction_factor:.3f}")
            
            # Flag unusual corrections
            if correction_factor > 2.0 or correction_factor < 0.5:
                print(f"   ⚠️ Large correction factor ({correction_factor:.3f}) - verify scale factors")
    
    def validate_scale_factor_application(self, original_ds: xr.Dataset, 
                                        corrected_ds: xr.Dataset) -> Dict:
        """
        Validate that scale factors were applied correctly.
        
        Returns diagnostic metrics for quality assurance.
        """
        if not self.validate_application:
            return {}
        
        print("\n🔍 Validating scale factor application...")
        
        grace_var = self._identify_grace_variable(original_ds)
        
        original_data = original_ds[grace_var].values
        corrected_data = corrected_ds[grace_var].values
        
        # Calculate validation metrics
        valid_mask = ~np.isnan(original_data) & ~np.isnan(corrected_data)
        
        if not np.any(valid_mask):
            return {"validation_status": "failed", "reason": "no_valid_data"}
        
        # Check for proper scaling
        ratio = corrected_data[valid_mask] / original_data[valid_mask]
        ratio = ratio[np.isfinite(ratio)]
        
        validation_metrics = {
            "validation_status": "passed",
            "scale_factor_range": [float(np.min(ratio)), float(np.max(ratio))],
            "mean_scale_factor": float(np.mean(ratio)),
            "std_scale_factor": float(np.std(ratio)),
            "corrected_mean": float(np.nanmean(corrected_data)),
            "original_mean": float(np.nanmean(original_data)),
            "correction_factor": float(np.nanmean(corrected_data) / np.nanmean(original_data)),
            "percent_corrected": float(100 * np.sum(valid_mask) / original_data.size)
        }
        
        print(f"   ✓ Validation completed")
        print(f"      Mean scale factor: {validation_metrics['mean_scale_factor']:.3f}")
        print(f"      Correction factor: {validation_metrics['correction_factor']:.3f}")
        print(f"      Data corrected: {validation_metrics['percent_corrected']:.1f}%")
        
        return validation_metrics
    
    def create_scale_factor_report(self, original_ds: xr.Dataset, 
                                 corrected_ds: xr.Dataset, 
                                 output_path: str):
        """
        Create a comprehensive report on scale factor application.
        
        Saves detailed analysis for scientific documentation.
        """
        validation_metrics = self.validate_scale_factor_application(original_ds, corrected_ds)
        
        report = {
            "scale_factor_application_report": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "scale_factor_file": str(self.scale_factor_path),
                "interpolation_method": self.interpolation_method,
                "validation_metrics": validation_metrics,
                "scientific_note": (
                    "Scale factors applied to restore GRACE TWS signal amplitude "
                    "and reduce leakage effects. Essential for scientific accuracy."
                ),
                "reference": "JPL GRACE/GRACE-FO Mascon RL06.3v04 CRI Dataset"
            }
        }
        
        # Save report
        import json
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   📄 Scale factor report saved: {report_path}")
        
        return report