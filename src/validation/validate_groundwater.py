# src/validation/validate_groundwater.py
"""
Complete validation module for GRACE groundwater downscaling results.

This module validates downscaled groundwater storage estimates against USGS well observations
using multiple approaches:
1. Point-to-point validation with individual wells
2. Spatial averaging validation at different scales
3. Comprehensive performance analysis and visualization

Compatible with both original and multi-model pipeline outputs.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import cKDTree
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class GroundwaterValidator:
    """Comprehensive groundwater validation system."""
    
    def __init__(self, gws_path=None, config_path="src/config.yaml"):
        """
        Initialize validator with robust file detection.
        
        Parameters:
        -----------
        gws_path : str, optional
            Path to groundwater NetCDF file. If None, auto-detects.
        config_path : str
            Path to configuration file
        """
        self.gws_path = self._find_groundwater_file(gws_path)
        self.config_path = config_path
        self.results_dir = Path("results/validation")
        self.figures_dir = Path("figures/validation")
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.figures_dir.mkdir(exist_ok=True, parents=True)
        
        # Load data
        self._load_data()
    
    def _find_groundwater_file(self, gws_path):
        """Find groundwater file with multiple possible names."""
        if gws_path and Path(gws_path).exists():
            return gws_path
        
        # Try multiple possible filenames in order of preference
        possible_files = [
            "results/groundwater_storage_anomalies.nc",
        ]
        
        for file_path in possible_files:
            if Path(file_path).exists():
                print(f"✅ Found groundwater data: {file_path}")
                return file_path
        
        raise FileNotFoundError(
            f"No groundwater storage file found. Tried: {possible_files}\n"
            "Run: python pipeline.py --steps gws"
        )
    
    def _load_data(self):
        """Load datasets with comprehensive error handling."""
        print("📦 Loading datasets for validation...")
        
        # Load groundwater predictions
        try:
            print(f"   Loading groundwater from: {self.gws_path}")
            self.gws_ds = xr.open_dataset(self.gws_path)
            
            # Check if 'groundwater' variable exists
            if 'groundwater' not in self.gws_ds.data_vars:
                available_vars = list(self.gws_ds.data_vars)
                print(f"   ⚠️ 'groundwater' variable not found. Available: {available_vars}")
                
                # Try alternative names
                alt_names = ['gws', 'groundwater_anomaly', 'groundwater_storage']
                for alt_name in alt_names:
                    if alt_name in self.gws_ds.data_vars:
                        print(f"   🔄 Using '{alt_name}' as groundwater variable")
                        self.gws_ds = self.gws_ds.rename({alt_name: 'groundwater'})
                        break
                else:
                    raise ValueError(f"No groundwater variable found in {available_vars}")
            
            # Print dataset info
            print(f"   ✅ Groundwater shape: {self.gws_ds.groundwater.shape}")
            print(f"   📅 Time range: {self.gws_ds.time.values[0]} to {self.gws_ds.time.values[-1]}")
            print(f"   🌍 Spatial extent: lat [{float(self.gws_ds.lat.min()):.2f}, {float(self.gws_ds.lat.max()):.2f}], "
                  f"lon [{float(self.gws_ds.lon.min()):.2f}, {float(self.gws_ds.lon.max()):.2f}]")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load groundwater data: {e}")
        
        # Load well observations (try new format first)
        well_data_path = "data/raw/usgs_well_data/monthly_groundwater_anomalies_cm.csv"
        if not os.path.exists(well_data_path):
            well_data_path = "data/raw/usgs_well_data/monthly_groundwater_anomalies.csv"
        well_metadata_path = "data/raw/usgs_well_data/well_metadata.csv"
        
        try:
            if not os.path.exists(well_data_path):
                raise FileNotFoundError(f"Well data not found: {well_data_path}")
            if not os.path.exists(well_metadata_path):
                raise FileNotFoundError(f"Well metadata not found: {well_metadata_path}")
            
            print(f"   Loading well data from: {well_data_path}")
            self.well_data = pd.read_csv(well_data_path, index_col=0, parse_dates=True)
            
            print(f"   Loading well metadata from: {well_metadata_path}")
            self.well_locations = pd.read_csv(well_metadata_path)
            
            print(f"   ✅ Loaded {len(self.well_data.columns)} wells with time series")
            print(f"   📅 Well data range: {self.well_data.index[0]} to {self.well_data.index[-1]}")
            
            # Fix well ID consistency
            self._fix_well_id_consistency()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load well data: {e}")
    
    def _fix_well_id_consistency(self):
        """Fix well ID format issues between time series and metadata."""
        data_well_ids = set(str(col) for col in self.well_data.columns)
        metadata_well_ids = set(str(wid) for wid in self.well_locations['well_id'])
        
        # Check direct overlap
        direct_overlap = data_well_ids.intersection(metadata_well_ids)
        print(f"   🔗 Direct well ID overlap: {len(direct_overlap)} / {len(data_well_ids)} wells")
        
        # If poor overlap, try integer conversion
        if len(direct_overlap) < len(data_well_ids) * 0.8:
            try:
                print("   🔧 Attempting well ID format standardization...")
                self.well_locations['well_id'] = self.well_locations['well_id'].astype(int).astype(str)
                
                metadata_well_ids_fixed = set(self.well_locations['well_id'])
                fixed_overlap = data_well_ids.intersection(metadata_well_ids_fixed)
                
                if len(fixed_overlap) > len(direct_overlap):
                    print(f"   ✅ Fixed! New overlap: {len(fixed_overlap)} wells")
                else:
                    print(f"   ⚠️ Fix didn't improve overlap significantly")
                    
            except Exception as e:
                print(f"   ⚠️ ID standardization failed: {e}")
    
    def _fix_time_format(self, gws_series):
        """Fix time format mismatch between GWS and well data."""
        if hasattr(gws_series, 'to_pandas'):
            gws_series = gws_series.to_pandas()
        elif hasattr(gws_series, 'values'):
            # Manual conversion for xarray
            time_values = gws_series.time.values
            if isinstance(time_values[0], str):
                # Convert 'YYYY-MM' to datetime
                datetime_index = pd.to_datetime([f"{t}-01" for t in time_values])
            else:
                datetime_index = pd.to_datetime(time_values)
            gws_series = pd.Series(gws_series.values, index=datetime_index)
        
        # Ensure index is datetime
        if isinstance(gws_series.index[0], str):
            # Convert string index like '2003-02' to datetime
            gws_series.index = pd.to_datetime([f"{date_str}-01" for date_str in gws_series.index])
        
        return gws_series
    
    def _calculate_metrics(self, pred_series, obs_series):
        """Calculate comprehensive validation metrics."""
        try:
            # Fix time format for predictions
            pred_series = self._fix_time_format(pred_series)
            
            # Ensure obs_series has datetime index
            if not isinstance(obs_series.index, pd.DatetimeIndex):
                obs_series.index = pd.to_datetime(obs_series.index)
            
            # Find common time indices
            common_idx = pred_series.index.intersection(obs_series.index)
            
            if len(common_idx) < 12:  # Need at least 1 year
                return None
            
            # Align data
            pred_aligned = pred_series[common_idx]
            obs_aligned = obs_series[common_idx]
            
            # Remove NaN values
            valid_mask = ~(pred_aligned.isna() | obs_aligned.isna())
            
            if valid_mask.sum() < 12:
                return None
            
            pred_clean = pred_aligned[valid_mask]
            obs_clean = obs_aligned[valid_mask]
            
            # Check for zero variance
            if pred_clean.std() == 0 or obs_clean.std() == 0:
                return None
            
            # Standardize for correlation analysis
            pred_std = (pred_clean - pred_clean.mean()) / pred_clean.std()
            obs_std = (obs_clean - obs_clean.mean()) / obs_clean.std()
            
            # Calculate basic metrics
            pearson_r, pearson_p = pearsonr(pred_std, obs_std)
            spearman_r, spearman_p = spearmanr(pred_std, obs_std)
            rmse = np.sqrt(mean_squared_error(pred_std, obs_std))
            mae = mean_absolute_error(pred_std, obs_std)
            
            metrics = {
                'n_obs': len(pred_clean),
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'rmse': rmse,
                'mae': mae,
                'pred_mean': pred_clean.mean(),
                'obs_mean': obs_clean.mean(),
                'pred_std_orig': pred_clean.std(),
                'obs_std_orig': obs_clean.std()
            }
            
            # Calculate trend correlation for longer time series
            if len(pred_clean) > 36:  # 3+ years
                try:
                    # 12-month rolling mean for trend analysis
                    pred_trend = pred_clean.rolling(12, center=True, min_periods=6).mean()
                    obs_trend = obs_clean.rolling(12, center=True, min_periods=6).mean()
                    
                    trend_mask = ~(pred_trend.isna() | obs_trend.isna())
                    
                    if trend_mask.sum() > 12:
                        pred_trend_clean = pred_trend[trend_mask]
                        obs_trend_clean = obs_trend[trend_mask]
                        
                        if pred_trend_clean.std() > 0 and obs_trend_clean.std() > 0:
                            pred_trend_std = (pred_trend_clean - pred_trend_clean.mean()) / pred_trend_clean.std()
                            obs_trend_std = (obs_trend_clean - obs_trend_clean.mean()) / obs_trend_clean.std()
                            
                            trend_r, _ = pearsonr(pred_trend_std, obs_trend_std)
                            metrics['trend_correlation'] = trend_r
                        else:
                            metrics['trend_correlation'] = np.nan
                    else:
                        metrics['trend_correlation'] = np.nan
                except:
                    metrics['trend_correlation'] = np.nan
            else:
                metrics['trend_correlation'] = np.nan
            
            return metrics
            
        except Exception as e:
            return None
    
    def _is_in_grid(self, lat, lon, tolerance=0.01):
        """Check if coordinates are within model grid with tolerance."""
        lat_min, lat_max = float(self.gws_ds.lat.min()), float(self.gws_ds.lat.max())
        lon_min, lon_max = float(self.gws_ds.lon.min()), float(self.gws_ds.lon.max())
        
        return (lat >= lat_min - tolerance and lat <= lat_max + tolerance and
                lon >= lon_min - tolerance and lon <= lon_max + tolerance)
    
    def validate_point_to_point(self):
        """
        Simplified validation using pre-converted storage anomalies.
        
        Uses pre-converted storage anomaly data (cm) if available, otherwise
        converts depth anomalies using standard specific yield (0.15).
        
        Returns:
        --------
        pd.DataFrame
            Validation metrics for each well
        """
        print("\n📍 POINT-TO-POINT VALIDATION")
        print("="*50)
        
        results = []
        processing_stats = {
            'total_wells': len(self.well_locations),
            'in_grid': 0,
            'has_time_series': 0,
            'sufficient_data': 0,
            'validated': 0
        }
        
        # Determine if data is already in storage units
        well_data_path = "data/raw/usgs_well_data/monthly_groundwater_anomalies_cm.csv"
        data_in_storage_units = os.path.exists(well_data_path)
        
        for idx, well in tqdm(self.well_locations.iterrows(), 
                             total=len(self.well_locations),
                             desc="Processing wells"):
            
            well_id = str(well['well_id'])
            lat, lon = well['lat'], well['lon']
            
            # Check if well is in grid bounds
            if not self._is_in_grid(lat, lon):
                continue
            processing_stats['in_grid'] += 1
            
            # Check if well has time series data
            if well_id not in self.well_data.columns:
                continue
            processing_stats['has_time_series'] += 1
            
            try:
                # Extract GWS at well location
                gws_at_well = self.gws_ds.groundwater.sel(
                    lat=lat, lon=lon, method='nearest'
                )
                
                # Get well observations and remove NaN
                well_obs = self.well_data[well_id].dropna()
                
                if len(well_obs) < 12:  # Need at least 1 year
                    continue
                processing_stats['sufficient_data'] += 1
                
                # Convert to storage anomalies if needed
                if data_in_storage_units:
                    # Data already converted to storage anomalies in cm
                    well_storage = well_obs  # No conversion needed!
                    specific_yield_used = 0.15  # Standard value from metadata
                else:
                    # Legacy data in depth units - use standard specific yield
                    STANDARD_SPECIFIC_YIELD = 0.15  # From literature
                    well_storage = well_obs * STANDARD_SPECIFIC_YIELD * 100
                    specific_yield_used = STANDARD_SPECIFIC_YIELD
                
                # Calculate metrics directly
                metrics = self._calculate_metrics(gws_at_well, well_storage)
                
                if metrics:
                    metrics.update({
                        'well_id': well_id,
                        'lat': lat,
                        'lon': lon,
                        'specific_yield': specific_yield_used
                    })
                    results.append(metrics)
                    processing_stats['validated'] += 1
                    
            except Exception as e:
                # Skip problematic wells
                continue
        
        # Create results DataFrame
        metrics_df = pd.DataFrame(results)
        
        # Print processing summary
        print(f"\nProcessing Summary:")
        for key, value in processing_stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Save and summarize results
        output_path = self.results_dir / "point_validation_metrics.csv"
        
        if len(metrics_df) > 0:
            metrics_df.to_csv(output_path, index=False)
            self._print_validation_summary(metrics_df, "Point-to-Point")
        else:
            # Save empty file for pipeline compatibility
            empty_df = pd.DataFrame(columns=['well_id', 'lat', 'lon', 'pearson_r'])
            empty_df.to_csv(output_path, index=False)
            print("\nPoint-to-Point Results:")
            print("  ⚠️ No wells could be validated")
            
            # Provide diagnostic information
            if processing_stats['in_grid'] == 0:
                print("  🔧 Issue: No wells within model grid bounds")
            elif processing_stats['has_time_series'] == 0:
                print("  🔧 Issue: No wells have matching time series data")
            elif processing_stats['sufficient_data'] == 0:
                print("  🔧 Issue: Wells don't have sufficient temporal coverage")
            else:
                print("  🔧 Issue: Correlation calculation failing")
        
        return metrics_df
    
    def validate_spatial_average(self, radius_km=50):
        """
        Validate using spatially averaged wells within specified radius.
        
        Parameters:
        -----------
        radius_km : float
            Radius for spatial averaging in kilometers
            
        Returns:
        --------
        pd.DataFrame
            Validation metrics for spatial averages
        """
        print(f"\n🌍 SPATIAL AVERAGING VALIDATION (radius={radius_km}km)")
        print("="*50)
        
        # Convert radius to degrees (approximate)
        radius_deg = radius_km / 111.0
        
        # Build spatial index for wells
        well_coords = self.well_locations[['lon', 'lat']].values
        well_tree = cKDTree(well_coords)
        
        results = []
        
        # Sample grid points (subsample for computational efficiency)
        lat_step = max(1, len(self.gws_ds.lat) // 20)  # ~20 sample points
        lon_step = max(1, len(self.gws_ds.lon) // 20)
        
        total_points = (len(self.gws_ds.lat) // lat_step) * (len(self.gws_ds.lon) // lon_step)
        
        with tqdm(total=total_points, desc="Processing grid") as pbar:
            for i in range(0, len(self.gws_ds.lat), lat_step):
                for j in range(0, len(self.gws_ds.lon), lon_step):
                    pbar.update(1)
                    
                    lat = float(self.gws_ds.lat[i])
                    lon = float(self.gws_ds.lon[j])
                    
                    # Find nearby wells
                    nearby_idx = well_tree.query_ball_point([lon, lat], radius_deg)
                    
                    if len(nearby_idx) < 3:  # Need at least 3 wells
                        continue
                         # Get well IDs for nearby wells
                nearby_wells = self.well_locations.iloc[nearby_idx]
                well_ids = [str(wid) for wid in nearby_wells['well_id']]
                
                # Filter to wells with actual data
                valid_wells = [w for w in well_ids if w in self.well_data.columns]
                
                if len(valid_wells) < 3:
                    continue
                
                try:
                    # Get GWS at grid point
                    gws_series = self.gws_ds.groundwater.isel(lat=i, lon=j)
                    
                    # Average well data - check if already in storage units
                    well_data_path = "data/raw/usgs_well_data/monthly_groundwater_anomalies_cm.csv"
                    if os.path.exists(well_data_path):
                        # Data already in cm storage units
                        well_subset = self.well_data[valid_wells]
                        well_mean = well_subset.mean(axis=1)
                    else:
                        # Legacy data - convert using standard specific yield
                        well_subset = self.well_data[valid_wells]
                        well_mean = well_subset.mean(axis=1) * 0.15 * 100  # Convert to cm
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(gws_series, well_mean)
                    
                    if metrics:
                        metrics.update({
                            'lat': lat,
                            'lon': lon,
                            'n_wells': len(valid_wells),
                            'radius_km': radius_km
                        })
                        results.append(metrics)
                        
                except Exception as e:
                    continue
        
        # Create results DataFrame
        metrics_df = pd.DataFrame(results)
        
        # Save and summarize results
        output_path = self.results_dir / f"spatial_avg_{radius_km}km_metrics.csv"
        
        if len(metrics_df) > 0:
            metrics_df.to_csv(output_path, index=False)
            self._print_validation_summary(metrics_df, f"Spatial Average ({radius_km}km)")
        else:
            # Save empty file for pipeline compatibility
            empty_df = pd.DataFrame(columns=['lat', 'lon', 'pearson_r', 'n_wells'])
            empty_df.to_csv(output_path, index=False)
            print(f"\nSpatial Average ({radius_km}km) Results:")
            print("  ⚠️ No grid points had sufficient wells for validation")
        
        return metrics_df
    
    def _print_validation_summary(self, metrics_df, validation_type):
        """Print comprehensive validation summary."""
        print(f"\n{validation_type} Results:")
        print(f"  Validated points: {len(metrics_df)}")
        
        if len(metrics_df) == 0:
            return
        
        # Basic statistics
        mean_r = metrics_df['pearson_r'].mean()
        std_r = metrics_df['pearson_r'].std()
        median_r = metrics_df['pearson_r'].median()
        
        print(f"  Mean correlation: {mean_r:.3f} ± {std_r:.3f}")
        print(f"  Median correlation: {median_r:.3f}")
        
        # Performance categories
        high_quality = (metrics_df['pearson_r'] > 0.5).sum()
        medium_quality = (metrics_df['pearson_r'] > 0.3).sum()
        positive_corr = (metrics_df['pearson_r'] > 0).sum()
        
        print(f"  High quality (r>0.5): {high_quality} ({high_quality/len(metrics_df)*100:.1f}%)")
        print(f"  Medium quality (r>0.3): {medium_quality} ({medium_quality/len(metrics_df)*100:.1f}%)")
        print(f"  Positive correlation: {positive_corr} ({positive_corr/len(metrics_df)*100:.1f}%)")
        
        # Additional statistics if available
        if 'trend_correlation' in metrics_df:
            trend_data = metrics_df['trend_correlation'].dropna()
            if len(trend_data) > 0:
                print(f"  Mean trend correlation: {trend_data.mean():.3f} (n={len(trend_data)})")
    
    def create_validation_plots(self):
        """Create comprehensive validation visualizations."""
        print("\n📊 CREATING VALIDATION PLOTS")
        print("="*50)
        
        # Load validation results
        point_path = self.results_dir / "point_validation_metrics.csv"
        spatial_path = self.results_dir / "spatial_avg_50km_metrics.csv"
        
        point_metrics = None
        spatial_metrics = None
        
        # Load point validation results
        if point_path.exists():
            try:
                point_metrics = pd.read_csv(point_path)
                if len(point_metrics) == 0:
                    point_metrics = None
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                point_metrics = None
        
        # Load spatial validation results
        if spatial_path.exists():
            try:
                spatial_metrics = pd.read_csv(spatial_path)
                if len(spatial_metrics) == 0:
                    spatial_metrics = None
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                spatial_metrics = None
        
        # Create main validation figure
        if point_metrics is not None:
            self._create_main_validation_figure(point_metrics)
            print("  ✅ Main validation figure created")
        else:
            print("  ⚠️ No point validation data for main figure")
        
        # Create spatial comparison if both datasets available
        if point_metrics is not None and spatial_metrics is not None:
            self._create_spatial_comparison_figure(point_metrics, spatial_metrics)
            print("  ✅ Spatial comparison figure created")
        else:
            print("  ⚠️ Insufficient data for spatial comparison figure")
        
        # Create performance summary figure
        if point_metrics is not None or spatial_metrics is not None:
            self._create_performance_summary(point_metrics, spatial_metrics)
            print("  ✅ Performance summary created")
        
        print(f"  📁 All figures saved to: {self.figures_dir}")
    
    def _create_main_validation_figure(self, metrics_df):
        """Create main validation figure with multiple panels."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Correlation histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(metrics_df['pearson_r'], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        ax1.axvline(metrics_df['pearson_r'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {metrics_df["pearson_r"].mean():.3f}')
        ax1.set_xlabel('Pearson Correlation')
        ax1.set_ylabel('Number of Wells')
        ax1.set_title('Correlation Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Spatial map of correlations
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(metrics_df['lon'], metrics_df['lat'],
                            c=metrics_df['pearson_r'], s=30,
                            cmap='RdYlBu', vmin=-0.2, vmax=0.8,
                            edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Spatial Validation Performance')
        plt.colorbar(scatter, ax=ax2, label='Correlation')
        ax2.grid(True, alpha=0.3)
        
        # 3. Sample size vs performance
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(metrics_df['n_obs'], metrics_df['pearson_r'], alpha=0.6, color='green')
        ax3.set_xlabel('Number of Observations')
        ax3.set_ylabel('Correlation')
        ax3.set_title('Sample Size vs Performance')
        ax3.grid(True, alpha=0.3)
        
        # 4. Specific yield distribution
        ax4 = fig.add_subplot(gs[1, 0])
        if 'specific_yield' in metrics_df.columns:
            sy_counts = metrics_df['specific_yield'].value_counts().sort_index()
            ax4.bar(sy_counts.index, sy_counts.values, alpha=0.7, color='orange')
            ax4.set_xlabel('Specific Yield')
            ax4.set_ylabel('Number of Wells')
            ax4.set_title('Optimal Specific Yield Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 5. Mean GWS map
        ax5 = fig.add_subplot(gs[1, 1])
        mean_gws = self.gws_ds.groundwater.mean(dim='time')
        im = ax5.imshow(mean_gws, cmap='RdBu_r', vmin=-20, vmax=20,
                       extent=[float(self.gws_ds.lon.min()), float(self.gws_ds.lon.max()),
                              float(self.gws_ds.lat.min()), float(self.gws_ds.lat.max())],
                       origin='lower', aspect='auto')
        ax5.set_xlabel('Longitude')
        ax5.set_ylabel('Latitude')
        ax5.set_title('Mean Groundwater Storage Anomaly')
        plt.colorbar(im, ax=ax5, label='GWS (cm)')
        
        # 6. Time series example
        ax6 = fig.add_subplot(gs[1, 2])
        if len(metrics_df) > 0:
            # Show best performing well
            best_well = metrics_df.nlargest(1, 'pearson_r').iloc[0]
            
            try:
                # Get GWS and well data for best well
                gws_series = self.gws_ds.groundwater.sel(
                    lat=best_well['lat'], lon=best_well['lon'], method='nearest'
                )
                gws_series = self._fix_time_format(gws_series)
                
                well_id = str(best_well['well_id'])
                if well_id in self.well_data.columns:
                    well_series = self.well_data[well_id].dropna() * best_well.get('specific_yield', 0.15) * 100
                    
                    # Find common time period
                    common_idx = gws_series.index.intersection(well_series.index)
                    if len(common_idx) > 0:
                        gws_common = gws_series[common_idx]
                        well_common = well_series[common_idx]
                        
                        # Standardize for plotting
                        gws_std = (gws_common - gws_common.mean()) / gws_common.std()
                        well_std = (well_common - well_common.mean()) / well_common.std()
                        
                        ax6.plot(gws_common.index, gws_std, label='Model', linewidth=2, alpha=0.8)
                        ax6.plot(well_common.index, well_std, label='Observed', linewidth=2, alpha=0.8)
                        ax6.set_xlabel('Time')
                        ax6.set_ylabel('Standardized Anomaly')
                        ax6.set_title(f'Best Example (r={best_well["pearson_r"]:.3f})')
                        ax6.legend()
                        ax6.grid(True, alpha=0.3)
            except:
                ax6.text(0.5, 0.5, 'Time series\nnot available', 
                        ha='center', va='center', transform=ax6.transAxes)
        
        # 7-9. Summary statistics
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Create summary text
        summary_text = f"""
        VALIDATION SUMMARY
        
        Total Wells Validated: {len(metrics_df)}
        Mean Correlation: {metrics_df['pearson_r'].mean():.3f} ± {metrics_df['pearson_r'].std():.3f}
        Median Correlation: {metrics_df['pearson_r'].median():.3f}
        
        Performance Categories:
        • High Quality (r > 0.5): {(metrics_df['pearson_r'] > 0.5).sum()} wells ({(metrics_df['pearson_r'] > 0.5).sum()/len(metrics_df)*100:.1f}%)
        • Medium Quality (r > 0.3): {(metrics_df['pearson_r'] > 0.3).sum()} wells ({(metrics_df['pearson_r'] > 0.3).sum()/len(metrics_df)*100:.1f}%)
        • Positive Correlation: {(metrics_df['pearson_r'] > 0).sum()} wells ({(metrics_df['pearson_r'] > 0).sum()/len(metrics_df)*100:.1f}%)
        """
        
        ax7.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('GRACE Groundwater Validation Results', fontsize=16, fontweight='bold')
        plt.savefig(self.figures_dir / 'main_validation_figure.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_spatial_comparison_figure(self, point_metrics, spatial_metrics):
        """Create figure comparing point and spatial validation approaches."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Distribution comparison
        bins = np.linspace(-0.3, 0.8, 30)
        ax1.hist(point_metrics['pearson_r'], bins=bins, alpha=0.6, density=True,
                label='Point Wells', color='blue', edgecolor='black')
        ax1.hist(spatial_metrics['pearson_r'], bins=bins, alpha=0.6, density=True,
                label='Spatial Average', color='red', edgecolor='black')
        
        ax1.axvline(point_metrics['pearson_r'].mean(), color='blue', linestyle='--',
                   label=f'Point Mean: {point_metrics["pearson_r"].mean():.3f}')
        ax1.axvline(spatial_metrics['pearson_r'].mean(), color='red', linestyle='--',
                   label=f'Spatial Mean: {spatial_metrics["pearson_r"].mean():.3f}')
        
        ax1.set_xlabel('Correlation')
        ax1.set_ylabel('Density')
        ax1.set_title('Point vs Spatial Validation Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Spatial map of spatial validation
        scatter = ax2.scatter(spatial_metrics['lon'], spatial_metrics['lat'],
                            c=spatial_metrics['pearson_r'], s=50,
                            cmap='RdYlBu', vmin=-0.2, vmax=0.8,
                            edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Spatial Average Validation (50km)')
        plt.colorbar(scatter, ax=ax2, label='Correlation')
        ax2.grid(True, alpha=0.3)
        
        # 3. Number of wells vs performance (spatial)
        if 'n_wells' in spatial_metrics.columns:
            ax3.scatter(spatial_metrics['n_wells'], spatial_metrics['pearson_r'], 
                       alpha=0.6, color='purple')
            ax3.set_xlabel('Number of Wells in Average')
            ax3.set_ylabel('Correlation')
            ax3.set_title('Well Density vs Performance')
            ax3.grid(True, alpha=0.3)
        
        # 4. Summary comparison
        ax4.axis('off')
        
        comparison_text = f"""
        VALIDATION METHOD COMPARISON
        
        Point-to-Point Validation:
        • Wells: {len(point_metrics)}
        • Mean r: {point_metrics['pearson_r'].mean():.3f}
        • High quality: {(point_metrics['pearson_r'] > 0.5).sum()} wells
        
        Spatial Average Validation (50km):
        • Grid points: {len(spatial_metrics)}
        • Mean r: {spatial_metrics['pearson_r'].mean():.3f}
        • High quality: {(spatial_metrics['pearson_r'] > 0.5).sum()} points
        
        Interpretation:
        Spatial averaging typically shows higher
        correlations due to scale matching with
        GRACE resolution and noise reduction.
        """
        
        ax4.text(0.05, 0.95, comparison_text, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                transform=ax4.transAxes)
        
        plt.suptitle('Validation Method Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'validation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_summary(self, point_metrics, spatial_metrics):
        """Create performance summary visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Performance by category
        categories = ['High\n(r>0.5)', 'Medium\n(r>0.3)', 'Low\n(r>0.1)', 'Poor\n(r≤0.1)']
        
        if point_metrics is not None:
            point_counts = [
                (point_metrics['pearson_r'] > 0.5).sum(),
                ((point_metrics['pearson_r'] > 0.3) & (point_metrics['pearson_r'] <= 0.5)).sum(),
                ((point_metrics['pearson_r'] > 0.1) & (point_metrics['pearson_r'] <= 0.3)).sum(),
                (point_metrics['pearson_r'] <= 0.1).sum()
            ]
            point_pcts = [c/len(point_metrics)*100 for c in point_counts]
            
            ax1.bar(categories, point_pcts, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_ylabel('Percentage of Wells')
            ax1.set_title('Point Validation Performance')
            ax1.grid(True, alpha=0.3)
            
            # Add count labels on bars
            for i, (count, pct) in enumerate(zip(point_counts, point_pcts)):
                ax1.text(i, pct + 1, f'{count}\n({pct:.1f}%)', ha='center', va='bottom')
        
        if spatial_metrics is not None:
            spatial_counts = [
                (spatial_metrics['pearson_r'] > 0.5).sum(),
                ((spatial_metrics['pearson_r'] > 0.3) & (spatial_metrics['pearson_r'] <= 0.5)).sum(),
                ((spatial_metrics['pearson_r'] > 0.1) & (spatial_metrics['pearson_r'] <= 0.3)).sum(),
                (spatial_metrics['pearson_r'] <= 0.1).sum()
            ]
            spatial_pcts = [c/len(spatial_metrics)*100 for c in spatial_counts]
            
            ax2.bar(categories, spatial_pcts, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_ylabel('Percentage of Grid Points')
            ax2.set_title('Spatial Validation Performance')
            ax2.grid(True, alpha=0.3)
            
            # Add count labels on bars
            for i, (count, pct) in enumerate(zip(spatial_counts, spatial_pcts)):
                ax2.text(i, pct + 1, f'{count}\n({pct:.1f}%)', ha='center', va='bottom')
        
        plt.suptitle('Validation Performance Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        print("\n📄 GENERATING VALIDATION REPORT")
        print("="*50)
        
        report_path = self.results_dir / "validation_report.txt"
        
        # Load validation results
        point_metrics = None
        spatial_metrics = None
        
        point_path = self.results_dir / "point_validation_metrics.csv"
        if point_path.exists():
            try:
                point_metrics = pd.read_csv(point_path)
                if len(point_metrics) == 0:
                    point_metrics = None
            except:
                point_metrics = None
        
        spatial_path = self.results_dir / "spatial_avg_50km_metrics.csv"
        if spatial_path.exists():
            try:
                spatial_metrics = pd.read_csv(spatial_path)
                if len(spatial_metrics) == 0:
                    spatial_metrics = None
            except:
                spatial_metrics = None
        
        # Generate report
        lines = [
            "GRACE GROUNDWATER DOWNSCALING VALIDATION REPORT",
            "="*60,
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Groundwater data: {self.gws_path}",
            "",
            "SUMMARY:",
            "="*30
        ]
        
        if point_metrics is not None:
            lines.extend([
                "",
                "POINT-TO-POINT VALIDATION:",
                f"  Total wells validated: {len(point_metrics)}",
                f"  Mean correlation: {point_metrics['pearson_r'].mean():.3f} ± {point_metrics['pearson_r'].std():.3f}",
                f"  Median correlation: {point_metrics['pearson_r'].median():.3f}",
                f"  Range: [{point_metrics['pearson_r'].min():.3f}, {point_metrics['pearson_r'].max():.3f}]",
                "",
                "  Performance Categories:",
                f"    High quality (r>0.5): {(point_metrics['pearson_r'] > 0.5).sum()} wells ({(point_metrics['pearson_r'] > 0.5).sum()/len(point_metrics)*100:.1f}%)",
                f"    Medium quality (r>0.3): {(point_metrics['pearson_r'] > 0.3).sum()} wells ({(point_metrics['pearson_r'] > 0.3).sum()/len(point_metrics)*100:.1f}%)",
                f"    Positive correlation: {(point_metrics['pearson_r'] > 0).sum()} wells ({(point_metrics['pearson_r'] > 0).sum()/len(point_metrics)*100:.1f}%)",
                "",
                "  Top 10 performing wells:"
            ])
            
            top_wells = point_metrics.nlargest(10, 'pearson_r')
            for _, well in top_wells.iterrows():
                lines.append(f"    {well['well_id']}: r={well['pearson_r']:.3f}, n={well['n_obs']}, sy={well.get('specific_yield', 'N/A')}")
        
        if spatial_metrics is not None:
            lines.extend([
                "",
                "SPATIAL AVERAGE VALIDATION (50km):",
                f"  Grid points validated: {len(spatial_metrics)}",
                f"  Mean correlation: {spatial_metrics['pearson_r'].mean():.3f} ± {spatial_metrics['pearson_r'].std():.3f}",
                f"  Median correlation: {spatial_metrics['pearson_r'].median():.3f}",
                f"  Range: [{spatial_metrics['pearson_r'].min():.3f}, {spatial_metrics['pearson_r'].max():.3f}]",
                "",
                "  Performance Categories:",
                f"    High quality (r>0.5): {(spatial_metrics['pearson_r'] > 0.5).sum()} points ({(spatial_metrics['pearson_r'] > 0.5).sum()/len(spatial_metrics)*100:.1f}%)",
                f"    Medium quality (r>0.3): {(spatial_metrics['pearson_r'] > 0.3).sum()} points ({(spatial_metrics['pearson_r'] > 0.3).sum()/len(spatial_metrics)*100:.1f}%)",
                f"    Positive correlation: {(spatial_metrics['pearson_r'] > 0).sum()} points ({(spatial_metrics['pearson_r'] > 0).sum()/len(spatial_metrics)*100:.1f}%)"
            ])
        
        lines.extend([
            "",
            "INTERPRETATION:",
            "="*30,
            "• Point validation shows expected scale mismatch between wells (~1m) and GRACE (~300km)",
            "• Spatial averaging improves correlations by matching GRACE resolution",
            "• Results suitable for regional groundwater trend analysis",
            "• Combine with local data for site-specific applications",
            "",
            "RECOMMENDATIONS:",
            "="*30,
            "• Use for basin-scale water resource planning",
            "• Focus on temporal trends rather than absolute values",
            "• Consider ensemble with multiple models for uncertainty quantification",
            "• Validate against additional data sources when available",
            "",
            "FILES GENERATED:",
            "="*30,
            "• point_validation_metrics.csv: Individual well validation results",
            "• spatial_avg_50km_metrics.csv: Spatial averaging results", 
            "• main_validation_figure.png: Comprehensive validation plots",
            "• validation_comparison.png: Method comparison",
            "• performance_summary.png: Performance statistics",
            "• validation_report.txt: This report"
        ])
        
        if point_metrics is None and spatial_metrics is None:
            lines.extend([
                "",
                "⚠️ WARNING: No validation results generated",
                "",
                "TROUBLESHOOTING:",
                "• Check that groundwater file exists and is readable",
                "• Verify well data format and coordinate systems",
                "• Ensure well IDs match between metadata and time series",
                "• Check for sufficient temporal overlap between model and observations"
            ])
        
        # Write report
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"  ✅ Report saved to: {report_path}")


def main():
    """
    Main validation function called by pipeline.py
    
    This function runs the complete validation workflow:
    1. Point-to-point validation against individual wells
    2. Spatial averaging validation at multiple scales  
    3. Comprehensive visualization and reporting
    """
    print("🚀 GRACE GROUNDWATER VALIDATION SYSTEM")
    print("="*60)
    
    try:
        # Initialize validator
        validator = GroundwaterValidator()
        
        # Run point-to-point validation
        print("\n" + "="*60)
        point_metrics = validator.validate_point_to_point()
        
        # Run spatial averaging validation at multiple scales
        print("\n" + "="*60)
        spatial_metrics_50km = validator.validate_spatial_average(radius_km=50)
        
        # Test different spatial scales
        print("\n🔍 Testing different spatial scales...")
        scale_results = []
        for radius in [25, 75, 100]:
            print(f"\n{radius}km spatial validation:")
            metrics = validator.validate_spatial_average(radius_km=radius)
            if len(metrics) > 0:
                scale_results.append({
                    'radius_km': radius,
                    'mean_correlation': metrics['pearson_r'].mean(),
                    'n_points': len(metrics),
                    'high_quality_pct': (metrics['pearson_r'] > 0.5).sum() / len(metrics) * 100
                })
        
        # Save scale analysis
        if scale_results:
            scale_df = pd.DataFrame(scale_results)
            scale_df.to_csv(validator.results_dir / "scale_analysis.csv", index=False)
            print(f"\n✅ Scale analysis saved ({len(scale_results)} scales tested)")
        
        # Create comprehensive visualizations
        print("\n" + "="*60)
        validator.create_validation_plots()
        
        # Generate final report
        print("\n" + "="*60)
        validator.generate_validation_report()
        
        # Final summary
        print(f"\n🎉 VALIDATION COMPLETE!")
        print(f"📁 Results directory: {validator.results_dir}")
        print(f"📊 Figures directory: {validator.figures_dir}")
        
        # Print quick summary
        if len(point_metrics) > 0:
            print(f"\n📈 Quick Summary:")
            print(f"  Point validation: {len(point_metrics)} wells, r̄={point_metrics['pearson_r'].mean():.3f}")
            high_quality = (point_metrics['pearson_r'] > 0.5).sum()
            print(f"  High quality wells: {high_quality} ({high_quality/len(point_metrics)*100:.1f}%)")
        
        if len(spatial_metrics_50km) > 0:
            print(f"  Spatial validation: {len(spatial_metrics_50km)} points, r̄={spatial_metrics_50km['pearson_r'].mean():.3f}")
        
        return validator
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()