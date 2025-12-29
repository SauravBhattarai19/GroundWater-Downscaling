"""
GRACE Gap Filling using STL (Seasonal-Trend Decomposition using Loess)

This module fills missing GRACE data using statistical decomposition:
1. Decompose existing data into Trend + Seasonal + Residual components
2. Train STL on 80% of existing data
3. Reconstruct missing months using trend + long-term seasonal/residual patterns
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from statsmodels.tsa.seasonal import STL
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

from .grace_scale_handler import GRACEScaleHandler


class GRACEGapFiller:
    """
    Fill missing GRACE data using STL decomposition.
    
    Method:
    -------
    1. For each spatial pixel, extract time series
    2. Identify valid (non-NaN) months
    3. Split valid data: 80% for training STL, 20% for validation
    4. Decompose training data: Trend (T_t) + Seasonal (S_t) + Residual (R_t)
    5. For missing months:
       - Extrapolate/interpolate trend
       - Use mean seasonal component
       - Use mean residual component
    6. Reconstruct: GRACE_filled = T_t + S_t + R_t
    """
    
    def __init__(self, config: Dict):
        """
        Initialize gap filler.
        
        Parameters:
        -----------
        config : Dict
            Configuration dictionary with grace_gap_filling settings
        """
        self.config = config
        self.gap_config = config.get('grace_gap_filling', {})
        
        # STL parameters
        self.seasonal_period = self.gap_config.get('seasonal_period', 12)
        self.train_fraction = self.gap_config.get('train_fraction', 0.8)
        self.seasonal_deg = self.gap_config.get('seasonal_deg', 1)
        self.trend_deg = self.gap_config.get('trend_deg', 1)
        self.robust = self.gap_config.get('robust', True)
        
        # Quality control
        self.fill_threshold = self.gap_config.get('fill_threshold', 0.3)
        self.max_gap_length = self.gap_config.get('max_gap_length', 3)
        self.interpolate_short = self.gap_config.get('interpolate_short_gaps', True)
        
        # Initialize scale factor handler
        self.scale_handler = GRACEScaleHandler(config)
        
        print(f"🔧 GRACE Gap Filler initialized:")
        print(f"   Seasonal period: {self.seasonal_period} months")
        print(f"   Training fraction: {self.train_fraction:.0%}")
    
    def apply_spatial_clipping(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply spatial clipping to eliminate global processing inefficiency.
        
        Clips global GRACE data (720×360 = 259,200 points) to study area 
        (46×73 = 3,358 points) for 77× efficiency improvement.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Global GRACE dataset
            
        Returns:
        --------
        xr.Dataset
            Spatially clipped dataset
        """
        spatial_config = self.config.get('spatial_bounds', {})
        
        if not spatial_config.get('apply_spatial_clipping', False):
            print("   ⚠️ WARNING: Processing GLOBAL dataset - inefficient!")
            return ds
        
        # Get bounds from config
        north = spatial_config.get('north', 51.5)
        south = spatial_config.get('south', 28.5) 
        east = spatial_config.get('east', -77.5)
        west = spatial_config.get('west', -114.0)
        
        print(f"   🔪 Applying spatial clipping:")
        print(f"      Global shape: {ds.dims}")
        print(f"      Config bounds: {south}°N-{north}°N, {west}°E-{east}°E")
        
        # Check coordinate system and convert if necessary
        lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
        
        # If dataset uses 0-360° system, convert bounds
        if lon_min >= 0 and lon_max >= 180:
            print(f"      Dataset uses 0-360° longitude system")
            # Convert negative longitudes to 0-360° system
            west_360 = west + 360 if west < 0 else west
            east_360 = east + 360 if east < 0 else east
            print(f"      Converted bounds: {south}°N-{north}°N, {west_360}°E-{east_360}°E")
            lon_slice = slice(west_360, east_360)
        else:
            print(f"      Dataset uses -180-180° longitude system")
            lon_slice = slice(west, east)
        
        # Apply spatial clipping
        ds_clipped = ds.sel(
            lat=slice(south, north),
            lon=lon_slice
        )
        
        print(f"      Clipped shape: {ds_clipped.dims}")
        
        # Validate dimensions
        expected_lat = spatial_config.get('expected_lat_points', 46)
        expected_lon = spatial_config.get('expected_lon_points', 73)
        expected_total = spatial_config.get('total_points', 3358)
        
        actual_lat = len(ds_clipped.lat)
        actual_lon = len(ds_clipped.lon)
        actual_total = actual_lat * actual_lon
        
        if spatial_config.get('validate_bounds', True):
            print(f"      Validation:")
            print(f"         Lat points: {actual_lat} (expected: {expected_lat})")
            print(f"         Lon points: {actual_lon} (expected: {expected_lon})")
            print(f"         Total points: {actual_total} (expected: {expected_total})")
            
            if abs(actual_total - expected_total) > 100:  # Allow small tolerance
                print(f"      ⚠️ WARNING: Spatial dimensions don't match expected bounds")
        
        # Calculate efficiency improvement
        global_points = ds.sizes.get('lat', 1) * ds.sizes.get('lon', 1)
        if actual_total > 0 and global_points > actual_total:
            efficiency_factor = global_points / actual_total
            print(f"      ✅ Efficiency improvement: {efficiency_factor:.1f}× fewer points to process")
        elif actual_total == 0:
            print(f"      ❌ ERROR: No points found after clipping - check coordinate system/bounds")
        
        return ds_clipped
    
    def validate_spatial_bounds(self, ds: xr.Dataset, mode: str = "unknown") -> None:
        """
        Validate that dataset spatial bounds match configuration.
        
        Ensures both GEE and CRI modes use consistent spatial coverage.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Dataset to validate
        mode : str
            Data source mode ("GEE" or "CRI") for logging
        """
        spatial_config = self.config.get('spatial_bounds', {})
        
        if not spatial_config.get('validate_bounds', True):
            return
        
        # Get actual bounds
        lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())
        lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
        
        # Get expected bounds
        expected_north = spatial_config.get('north', 51.5)
        expected_south = spatial_config.get('south', 28.5)
        expected_east = spatial_config.get('east', -77.5)
        expected_west = spatial_config.get('west', -114.0)
        
        print(f"   📏 Spatial bounds validation ({mode} mode):")
        print(f"      Actual bounds: {lat_min:.1f}°N-{lat_max:.1f}°N, {lon_min:.1f}°E-{lon_max:.1f}°E")
        print(f"      Expected bounds: {expected_south:.1f}°N-{expected_north:.1f}°N, {expected_west:.1f}°E-{expected_east:.1f}°E")
        
        # Check dimensions
        actual_lat_points = len(ds.lat)
        actual_lon_points = len(ds.lon)
        expected_lat_points = spatial_config.get('expected_lat_points', 46)
        expected_lon_points = spatial_config.get('expected_lon_points', 73)
        
        print(f"      Actual dimensions: {actual_lat_points}×{actual_lon_points} = {actual_lat_points * actual_lon_points} points")
        print(f"      Expected dimensions: {expected_lat_points}×{expected_lon_points} = {expected_lat_points * expected_lon_points} points")
        
        # Tolerance for bounds checking (degrees)
        bounds_tolerance = 0.5
        
        # Check bounds alignment
        bounds_match = (
            abs(lat_min - expected_south) < bounds_tolerance and
            abs(lat_max - expected_north) < bounds_tolerance and  
            abs(lon_min - expected_west) < bounds_tolerance and
            abs(lon_max - expected_east) < bounds_tolerance
        )
        
        # Check dimensions alignment  
        dims_match = (
            abs(actual_lat_points - expected_lat_points) <= 2 and  # Small tolerance
            abs(actual_lon_points - expected_lon_points) <= 2
        )
        
        if bounds_match and dims_match:
            print(f"      ✅ {mode} spatial bounds are consistent with configuration")
        else:
            if not bounds_match:
                print(f"      ⚠️ WARNING: {mode} bounds don't match configuration (tolerance: {bounds_tolerance}°)")
            if not dims_match:
                print(f"      ⚠️ WARNING: {mode} dimensions don't match configuration")
    
    def map_irregular_to_regular_grid(self, irregular_data: np.ndarray, 
                                    irregular_times: pd.DatetimeIndex, 
                                    regular_grid: pd.DatetimeIndex) -> np.ndarray:
        """
        Map irregular time series data to a regular monthly grid.
        
        This function handles the core issue with JPL GRACE NetCDF files:
        they have irregular timestamps that need to be mapped to a regular
        monthly grid for proper STL decomposition.
        
        Parameters:
        -----------
        irregular_data : np.ndarray
            Data values at irregular timestamps (shape: time, lat, lon)
        irregular_times : pd.DatetimeIndex  
            Irregular timestamps from JPL file
        regular_grid : pd.DatetimeIndex
            Regular monthly grid (e.g., 2003-01 to 2024-11)
            
        Returns:
        --------
        np.ndarray
            Data mapped to regular grid with NaN for missing months
        """
        print(f"   🔄 Mapping irregular → regular temporal grid...")
        print(f"      Irregular: {len(irregular_times)} time steps")
        print(f"      Regular: {len(regular_grid)} months")
        
        # Initialize output array with NaN
        if irregular_data.ndim == 3:  # (time, lat, lon)
            regular_data = np.full((len(regular_grid), irregular_data.shape[1], irregular_data.shape[2]), 
                                 np.nan, dtype=irregular_data.dtype)
        else:  # 1D time series
            regular_data = np.full(len(regular_grid), np.nan, dtype=irregular_data.dtype)
        
        # Create mapping from irregular timestamps to regular grid indices
        # This is much faster than the loop approach
        month_starts = pd.DatetimeIndex([pd.Timestamp(t.year, t.month, 1) for t in irregular_times])
        
        # Find which irregular times map to regular grid
        valid_mask = month_starts.isin(regular_grid)
        valid_month_starts = month_starts[valid_mask]
        valid_irregular_indices = np.where(valid_mask)[0]
        
        mapped_count = 0
        duplicate_months = 0
        
        # Vectorized mapping for valid timestamps
        for i, month_start in enumerate(valid_month_starts):
            irr_idx = valid_irregular_indices[i]
            reg_idx = regular_grid.get_loc(month_start)
            
            # Simple assignment (assume no duplicates for now to avoid complexity)
            if irregular_data.ndim == 3:
                regular_data[reg_idx] = irregular_data[irr_idx]
            else:  # 1D
                regular_data[reg_idx] = irregular_data[irr_idx]
            
            mapped_count += 1
        
        print(f"      ✅ Mapped {mapped_count}/{len(irregular_times)} time steps")
        if duplicate_months > 0:
            print(f"      📊 Averaged {duplicate_months} duplicate months")
        
        # Calculate final statistics
        n_filled = len(regular_grid) - np.sum(np.isnan(regular_data) if irregular_data.ndim == 1 
                                            else np.all(np.isnan(regular_data), axis=(1,2)))
        print(f"      📈 Regular grid: {n_filled}/{len(regular_grid)} months have data")
        print(f"      🔍 Missing months: {len(regular_grid) - n_filled} (for STL to fill)")
        
        return regular_data
    
    def load_grace_data(self, grace_path: str) -> xr.Dataset:
        """
        Load GRACE data from NetCDF or directory of TIFFs.
        
        Parameters:
        -----------
        grace_path : str
            Path to GRACE data (NetCDF file or directory)
        
        Returns:
        --------
        xr.Dataset
            GRACE data with time, lat, lon dimensions
        """
        grace_path = Path(grace_path)
        
        if grace_path.is_file() and grace_path.suffix == '.nc':
            # Load NetCDF directly
            ds = xr.open_dataset(grace_path)
            print(f"✓ Loaded GRACE from NetCDF: {grace_path}")
            print(f"   Original shape: {ds.dims}")
            print(f"   Original time range: {pd.to_datetime(ds.time.values[0]).strftime('%Y-%m')} to {pd.to_datetime(ds.time.values[-1]).strftime('%Y-%m')}")
            
            # Apply temporal constraints based on config
            start_date = self.config.get('data_processing', {}).get('start_date', '2003-01-01')
            end_date = self.config.get('data_processing', {}).get('end_date', '2024-11-30')
            
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Filter to desired time range
            ds_filtered = ds.sel(time=slice(start_date, end_date))
            print(f"   Filtered time range: {pd.to_datetime(ds_filtered.time.values[0]).strftime('%Y-%m')} to {pd.to_datetime(ds_filtered.time.values[-1]).strftime('%Y-%m')}")
            print(f"   Filtered shape: {ds_filtered.dims}")
            
            # Apply spatial clipping to eliminate global processing inefficiency  
            ds_filtered = self.apply_spatial_clipping(ds_filtered)
            
            # Create COMPLETE monthly time range to match feature data
            complete_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
            irregular_times = pd.to_datetime(ds_filtered.time.values)
            
            print(f"   📅 Temporal Analysis:")
            print(f"      JPL irregular timestamps: {len(irregular_times)}")
            print(f"      Regular monthly grid: {len(complete_dates)} months")
            print(f"      Temporal mapping required: YES (irregular → regular)")
            
            # Get the TWS variable name
            tws_var = None
            for var in ds_filtered.data_vars:
                if 'time' in ds_filtered[var].dims:
                    tws_var = var
                    break
            
            if tws_var is None:
                raise ValueError("Could not find time-dependent variable in dataset")
            
            print(f"      TWS variable: '{tws_var}'")
            
            # Apply temporal mapping: irregular JPL → regular monthly grid
            irregular_data = ds_filtered[tws_var].values
            regular_data = self.map_irregular_to_regular_grid(
                irregular_data, irregular_times, complete_dates
            )
            
            # Create new dataset with regular monthly grid
            coords_dict = {
                'time': complete_dates,
                'lat': ds_filtered.lat,
                'lon': ds_filtered.lon
            }
            
            data_vars_dict = {
                tws_var: (['time', 'lat', 'lon'], regular_data)
            }
            
            # Copy other non-time-dependent variables
            for var_name, var_data in ds_filtered.data_vars.items():
                if var_name != tws_var and 'time' not in var_data.dims:
                    data_vars_dict[var_name] = var_data
                    
            ds_to_return = xr.Dataset(data_vars_dict, coords=coords_dict)
            
            # Copy attributes
            ds_to_return.attrs = ds_filtered.attrs
            for var_name in ds_to_return.data_vars:
                if var_name in ds_filtered.data_vars:
                    ds_to_return[var_name].attrs = ds_filtered[var_name].attrs
            
            print(f"   ✅ Regular monthly dataset created: {len(ds_to_return.time)} months")
            
            # Ensure monthly frequency
            time_diffs = pd.to_datetime(ds_to_return.time.values).to_series().diff().dt.days
            median_diff = time_diffs.median()
            print(f"   Median time step: {median_diff} days (should be ~30 for monthly)")
            
            if median_diff < 25 or median_diff > 35:
                print(f"   ⚠️ WARNING: Time steps don't appear to be monthly (median: {median_diff} days)")
            
            # Apply scale factors for scientific accuracy
            ds_to_return = self.scale_handler.apply_scale_factors_to_grace(ds_to_return)
            
            # Validate spatial bounds consistency
            self.validate_spatial_bounds(ds_to_return, mode="CRI")
            
            return ds_to_return
        
        elif grace_path.is_dir():
            # Load from directory of TIFFs
            print(f"📂 Loading GRACE TIFFs from: {grace_path}")
            
            import rioxarray
            
            tiff_files = sorted(grace_path.glob("*.tif"))
            if not tiff_files:
                raise FileNotFoundError(f"No TIFF files found in {grace_path}")
            
            print(f"   Found {len(tiff_files)} GRACE files")
            
            # Parse all available dates and load data
            file_data = {}
            dates_available = []
            
            # Get temporal constraints from config
            config_start = self.config.get('data_processing', {}).get('start_date', '2003-01-01')
            config_end = self.config.get('data_processing', {}).get('end_date', '2024-11-30')
            config_start_date = pd.to_datetime(config_start)
            config_end_date = pd.to_datetime(config_end)
            
            for tiff_file in tiff_files:
                try:
                    # Parse date from filename (YYYYMM format)
                    filename = tiff_file.stem
                    if len(filename) == 6 and filename.isdigit():
                        year = filename[:4]
                        month = filename[4:6]
                        date_str = f"{year}-{month}-01"
                        date = pd.to_datetime(date_str)
                        
                        # Check if date is within desired range
                        if config_start_date <= date <= config_end_date:
                            # Load raster
                            da = rioxarray.open_rasterio(tiff_file, masked=True).squeeze()
                            
                            file_data[date] = da.values
                            dates_available.append(date)
                        else:
                            print(f"   📅 Skipping {tiff_file.name}: outside date range ({config_start} to {config_end})")
                    else:
                        print(f"   ⚠️ Skipping {tiff_file.name}: unexpected filename format")
                except Exception as e:
                    print(f"   ⚠️ Failed to load {tiff_file.name}: {e}")
                    continue
            
            if not file_data:
                raise ValueError("No valid GRACE files loaded")
            
            # Create COMPLETE monthly time range based on config constraints
            dates_available = sorted(dates_available)
            
            # Use config dates, not file dates, to ensure alignment with features
            start_date = config_start_date
            end_date = config_end_date
            
            # Generate complete monthly range
            complete_dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Month Start
            
            print(f"   📅 Temporal Alignment:")
            print(f"      Config range: {config_start} to {config_end}")
            print(f"      Complete months: {len(complete_dates)} (should be 262 for 2003-01 to 2024-11)")
            print(f"      Available files: {len(dates_available)}")
            print(f"      Missing months: {len(complete_dates) - len(dates_available)}")
            
            # Get coordinates from first file
            first_date = dates_available[0]
            da_ref = rioxarray.open_rasterio(
                grace_path / f"{first_date.strftime('%Y%m')}.tif", 
                masked=True
            ).squeeze()
            
            # Extract 1D coordinate arrays
            if da_ref.y.ndim == 1:
                lat_coords = da_ref.y.values
            else:
                lat_coords = da_ref.y.values[:, 0]
            
            if da_ref.x.ndim == 1:
                lon_coords = da_ref.x.values
            else:
                lon_coords = da_ref.x.values[0, :]
            
            # Create data array with NaN for missing dates
            n_times = len(complete_dates)
            n_lat = len(lat_coords)
            n_lon = len(lon_coords)
            grace_stack = np.full((n_times, n_lat, n_lon), np.nan)
            
            # Fill in existing data
            for i, date in enumerate(complete_dates):
                if date in file_data:
                    grace_stack[i, :, :] = file_data[date]
            
            # Create dataset
            ds = xr.Dataset(
                {
                    'tws_anomaly': (['time', 'lat', 'lon'], grace_stack)
                },
                coords={
                    'time': complete_dates,
                    'lat': lat_coords,
                    'lon': lon_coords
                }
            )
            
            # Print missing time steps
            missing_dates = [date for date in complete_dates if date not in file_data]
            if missing_dates:
                print(f"\n   🔴 Missing time steps to be filled:")
                for date in missing_dates[:10]:  # Show first 10
                    print(f"      - {date.strftime('%Y-%m')} ({date.strftime('%Y%m')}.tif)")
                if len(missing_dates) > 10:
                    print(f"      ... and {len(missing_dates) - 10} more")
            
            print(f"\n✓ Created complete time series")
            print(f"   Shape: {grace_stack.shape}")
            print(f"   Lat: {n_lat}, Lon: {n_lon}")
            
            # Apply scale factors for scientific accuracy
            ds = self.scale_handler.apply_scale_factors_to_grace(ds)
            
            # Validate spatial bounds consistency
            self.validate_spatial_bounds(ds, mode="GEE")
            
            return ds
        
        else:
            raise ValueError(f"Invalid GRACE path: {grace_path}")
    
    def fill_short_gaps_linear(self, time_series: np.ndarray) -> np.ndarray:
        """
        Fill short gaps (<= max_gap_length) with linear interpolation.
        
        Parameters:
        -----------
        time_series : np.ndarray
            1D time series with potential gaps (NaNs)
        
        Returns:
        --------
        np.ndarray
            Time series with short gaps filled
        """
        if not self.interpolate_short:
            return time_series
        
        filled = time_series.copy()
        n = len(filled)
        
        # Find gaps
        is_valid = ~np.isnan(filled)
        
        i = 0
        while i < n:
            if not is_valid[i]:
                # Start of gap
                gap_start = i
                while i < n and not is_valid[i]:
                    i += 1
                gap_end = i
                gap_length = gap_end - gap_start
                
                # Fill if short enough and has boundaries
                if (gap_length <= self.max_gap_length and 
                    gap_start > 0 and gap_end < n):
                    
                    # Linear interpolation
                    x = np.array([gap_start - 1, gap_end])
                    y = np.array([filled[gap_start - 1], filled[gap_end]])
                    
                    interp_func = interp1d(x, y, kind='linear')
                    filled[gap_start:gap_end] = interp_func(np.arange(gap_start, gap_end))
            else:
                i += 1
        
        return filled
    
    def decompose_stl(self, time_series: np.ndarray, times: pd.DatetimeIndex) -> Dict:
        """
        Decompose time series using STL following scientific methodology.
        
        The decomposition follows: TWSA_t = T_t + S_t + R_t
        where T_t = Trend, S_t = Seasonality, R_t = Residual
        
        Parameters:
        -----------
        time_series : np.ndarray
            1D time series (observed data only, no NaNs)
        times : pd.DatetimeIndex
            Corresponding time stamps for observed data
        
        Returns:
        --------
        Dict with trend, seasonal, residual components and metadata
        """
        # Create pandas Series with monthly frequency
        ts = pd.Series(time_series, index=times)
        
        # Ensure we have a proper monthly frequency
        # STL requires regular spacing, so resample to monthly if needed
        if len(ts) > 1:
            # Check if already monthly
            freq = pd.infer_freq(times)
            if freq != 'MS' and freq != 'M':
                # Resample to month start if irregular
                ts = ts.resample('MS').mean()
                times = ts.index
        
        # Apply STL decomposition with 1-degree polynomial (as per specification)
        seasonal_window = 13 if self.seasonal_period == 12 else (
            self.seasonal_period + 1 if self.seasonal_period % 2 == 0 else self.seasonal_period
        )
        
        stl = STL(
            ts,
            seasonal=seasonal_window,
            seasonal_deg=1,  # 1-degree polynomial as specified
            trend_deg=1,     # 1-degree polynomial as specified
            robust=self.robust,
            period=self.seasonal_period
        )
        
        result = stl.fit()
        
        return {
            'trend': result.trend.values,
            'seasonal': result.seasonal.values, 
            'residual': result.resid.values,
            'observed': ts.values,
            'times': times,
            'stl_model': result  # Store the fitted model for trend extrapolation
        }
    
    def reconstruct_missing(self, 
                           stl_train_components: Dict,
                           all_observed_data: np.ndarray,
                           all_observed_times: pd.DatetimeIndex,
                           target_times: pd.DatetimeIndex) -> np.ndarray:
        """
        Reconstruct missing values using scientific STL methodology.
        
        Formula: TWSA_t(Missing) = y(T_t) + mean(S_t + R_t)
        
        Where:
        - y(T_t) = estimated trend for target times using trained STL trend
        - mean(S_t + R_t) = long-term mean of seasonal + residual from ALL observed data
        
        Parameters:
        -----------
        stl_train_components : Dict
            STL components from training data
        all_observed_data : np.ndarray
            ALL observed TWSA data (for getting long-term seasonal+residual mean)
        all_observed_times : pd.DatetimeIndex
            Times corresponding to all observed data
        target_times : pd.DatetimeIndex
            Times for data to reconstruct
        
        Returns:
        --------
        np.ndarray
            Reconstructed values for target times
        """
        try:
            # Step 1: Get components from ALL observed data for long-term mean
            # Use a subset if data is too long to avoid STL issues
            if len(all_observed_data) > 120:  # More than 10 years, subsample
                # Take recent complete years for seasonal pattern
                n_recent = min(60, len(all_observed_data))  # Last 5 years maximum
                recent_data = all_observed_data[-n_recent:]
                recent_times = all_observed_times[-n_recent:]
                full_stl = self.decompose_stl(recent_data, recent_times)
            else:
                full_stl = self.decompose_stl(all_observed_data, all_observed_times)
            
            full_seasonal = full_stl['seasonal']
            full_residual = full_stl['residual']
            
            # Calculate long-term mean of seasonal + residual components
            # This follows the scientific specification: mean(S_t + R_t)
            seasonal_plus_residual = full_seasonal + full_residual
            mean_seasonal_residual = np.mean(seasonal_plus_residual)
            
            # Alternative: Use monthly climatology of seasonal + residual
            # This might be more appropriate for realistic reconstruction
            monthly_seasonal_residual = {}
            for month in range(1, 13):
                month_mask = all_observed_times.month == month
                if np.any(month_mask):
                    monthly_vals = seasonal_plus_residual[month_mask]
                    monthly_seasonal_residual[month] = np.mean(monthly_vals)
                else:
                    monthly_seasonal_residual[month] = mean_seasonal_residual
            
            # Check for valid seasonal+residual mean
            if not np.isfinite(mean_seasonal_residual):
                mean_seasonal_residual = 0.0  # Fallback
            
            # Step 2: Use training STL trend to estimate trend for target times
            train_trend = stl_train_components['trend']
            train_times = stl_train_components['times']
            
            # Extrapolate/interpolate trend to target times
            if len(train_trend) > 1 and len(target_times) > 0:
                # Ensure both arrays are finite
                valid_trend_mask = np.isfinite(train_trend)
                if np.any(valid_trend_mask):
                    valid_trend = train_trend[valid_trend_mask]
                    valid_times = train_times[valid_trend_mask]
                    
                    if len(valid_trend) > 1:
                        trend_func = interp1d(
                            valid_times.astype(np.int64),
                            valid_trend,
                            kind='linear',
                            fill_value='extrapolate',
                            bounds_error=False
                        )
                        estimated_trend = trend_func(target_times.astype(np.int64))
                    else:
                        # Only one valid point
                        estimated_trend = np.full(len(target_times), valid_trend[0])
                else:
                    # No valid trend data
                    estimated_trend = np.zeros(len(target_times))
            else:
                # Fallback
                if len(train_trend) == 1:
                    estimated_trend = np.full(len(target_times), train_trend[0])
                else:
                    estimated_trend = np.zeros(len(target_times))
            
            # Ensure finite trend values
            estimated_trend = np.where(np.isfinite(estimated_trend), estimated_trend, 0.0)
            
            # Apply reconstruction formula following scientific specification
            # Two interpretations possible:
            # 1. TWSA_missing = y(T_t) + overall_mean(S_t + R_t) [current implementation]
            # 2. TWSA_missing = y(T_t) + monthly_mean(S_t + R_t) [more realistic]
            
            # Use the monthly approach for better reconstruction
            monthly_components = np.array([monthly_seasonal_residual[month] for month in target_times.month])
            reconstructed = estimated_trend + monthly_components
            
            return reconstructed
            
        except Exception as e:
            # Fallback: return zeros if reconstruction fails
            if not hasattr(self, '_reconstruct_error_printed'):
                self._reconstruct_error_printed = True
                print(f"⚠️ Reconstruction error (using fallback): {type(e).__name__}: {str(e)}")
            return np.zeros(len(target_times))
    
    def fill_pixel_timeseries_with_validation(self,
                                              time_series: np.ndarray,
                                              times: pd.DatetimeIndex) -> Tuple[np.ndarray, Dict]:
        """
        STL gap filling with proper train/test validation on known data.
        
        Scientific approach:
        1. For pixels with sufficient data (>36 months), randomly holdout 20% of known values
        2. Train STL on 80% of known data only
        3. Predict on 20% holdout data and calculate validation metrics (R², RMSE)
        4. Retrain on 100% known data to fill actual gaps
        5. Store validation metrics for manuscript accuracy assessment
        """
        valid_mask = ~np.isnan(time_series)
        missing_mask = np.isnan(time_series)
        
        n_valid = np.sum(valid_mask)
        n_missing = np.sum(missing_mask)
        
        info = {
            'n_valid': n_valid,
            'n_missing': n_missing,
            'pct_missing': n_missing / len(time_series),
            'method': 'none',
            'success': False,
            'values_filled': 0,
            'validation_performed': False,
            'test_r2': np.nan,
            'test_rmse': np.nan,
            'test_mae': np.nan,
            'validation_points': 0
        }
        
        # Need enough data for STL
        if n_valid < 24:  # At least 2 years
            info['method'] = 'insufficient_data'
            return time_series, info
        
        if n_missing == 0:
            info['method'] = 'no_gaps'
            info['success'] = True
            return time_series, info
        
        try:
            # === VALIDATION PHASE (if sufficient data) ===
            if n_valid >= 36:  # Need at least 3 years for reliable train/test split
                valid_indices = np.where(valid_mask)[0]
                valid_values = time_series[valid_mask]
                valid_times_subset = times[valid_indices]
                
                # Randomly holdout 20% of known values for validation
                np.random.seed(42)  # Reproducible results
                n_test = max(int(n_valid * 0.2), 6)  # At least 6 months for test
                test_indices = np.random.choice(len(valid_indices), n_test, replace=False)
                train_indices = np.setdiff1d(np.arange(len(valid_indices)), test_indices)
                
                # Create training dataset (80% of known data)
                train_global_indices = valid_indices[train_indices]
                test_global_indices = valid_indices[test_indices]
                
                # Build STL model on training data only
                train_ts = np.full_like(time_series, np.nan)
                train_ts[train_global_indices] = time_series[train_global_indices]
                
                # Linear interpolation for STL training (only on training data)
                train_ts_filled = train_ts.copy()
                train_valid_indices = train_global_indices
                train_missing_indices = np.setdiff1d(np.arange(len(time_series)), train_global_indices)
                
                if len(train_missing_indices) > 0 and len(train_valid_indices) > 1:
                    interp_func = interp1d(
                        train_valid_indices,
                        time_series[train_valid_indices],
                        kind='linear',
                        bounds_error=False,
                        fill_value='extrapolate'
                    )
                    train_ts_filled[train_missing_indices] = interp_func(train_missing_indices)
                
                # Apply STL to training data
                stl_result = self.decompose_stl(train_ts_filled, times)
                trend = stl_result['trend']
                seasonal = stl_result['seasonal']
                
                # Predict on test data using STL components
                test_predictions = []
                test_actual = []
                
                for test_idx in test_global_indices:
                    predicted = trend[test_idx] + seasonal[test_idx] + 0.0  # Zero-mean residual
                    actual = time_series[test_idx]
                    
                    if np.isfinite(predicted) and np.isfinite(actual):
                        test_predictions.append(predicted)
                        test_actual.append(actual)
                
                # Calculate validation metrics
                if len(test_predictions) >= 3:  # Need at least 3 points for meaningful metrics
                    test_predictions = np.array(test_predictions)
                    test_actual = np.array(test_actual)
                    
                    # R²
                    ss_res = np.sum((test_actual - test_predictions) ** 2)
                    ss_tot = np.sum((test_actual - np.mean(test_actual)) ** 2)
                    test_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                    
                    # RMSE and MAE
                    test_rmse = np.sqrt(np.mean((test_actual - test_predictions) ** 2))
                    test_mae = np.mean(np.abs(test_actual - test_predictions))
                    
                    info['validation_performed'] = True
                    info['test_r2'] = test_r2
                    info['test_rmse'] = test_rmse
                    info['test_mae'] = test_mae
                    info['validation_points'] = len(test_predictions)
            
            # === GAP FILLING PHASE (retrain on all available data) ===
            # Now use ALL known data to fill actual gaps
            filled_ts = time_series.copy()
            
            # Linear interpolation to fill gaps for STL processing
            if n_missing > 0:
                valid_indices = np.where(valid_mask)[0]
                missing_indices = np.where(missing_mask)[0]
                
                # Interpolate missing values
                interp_func = interp1d(
                    valid_indices, 
                    time_series[valid_mask],
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                filled_ts[missing_indices] = interp_func(missing_indices)
            
            # Apply STL to complete (interpolated) time series
            stl_result = self.decompose_stl(filled_ts, times)
            trend = stl_result['trend']
            seasonal = stl_result['seasonal']
            
            # Fill original missing values with STL reconstruction
            final_filled = time_series.copy()
            
            for idx in np.where(missing_mask)[0]:
                # Reconstruct: T + S + R (where R = 0 as per methodology)
                reconstructed_value = trend[idx] + seasonal[idx] + 0.0  # Zero-mean residual
                final_filled[idx] = reconstructed_value
            
            info['method'] = 'stl_with_validation' if info['validation_performed'] else 'stl_linear_interp'
            info['success'] = True
            info['values_filled'] = n_missing
            
            return final_filled, info
            
        except Exception as e:
            info['method'] = f'failed_{type(e).__name__}'
            info['error'] = str(e)
            if not hasattr(self, '_fill_error_printed'):
                self._fill_error_printed = True
                print(f"⚠️ STL Fill Error: {type(e).__name__}: {str(e)}")
            return time_series, info

    def fill_pixel_timeseries_simple(self, 
                                     time_series: np.ndarray,
                                     times: pd.DatetimeIndex) -> Tuple[np.ndarray, Dict]:
        """
        Simple STL gap filling without validation (legacy method).
        
        Approach:
        1. Use linear interpolation before STL to handle irregular gaps
        2. Use 100% of valid data to build STL model (no train/test split)
        3. Assume zero-mean residual for reconstruction
        4. Focus on gap filling, not validation
        """
        valid_mask = ~np.isnan(time_series)
        missing_mask = np.isnan(time_series)
        
        n_valid = np.sum(valid_mask)
        n_missing = np.sum(missing_mask)
        
        info = {
            'n_valid': n_valid,
            'n_missing': n_missing,
            'pct_missing': n_missing / len(time_series),
            'method': 'none',
            'success': False,
            'values_filled': 0
        }
        
        # Need enough data for STL
        if n_valid < 24:  # At least 2 years
            info['method'] = 'insufficient_data'
            return time_series, info
        
        if n_missing == 0:
            info['method'] = 'no_gaps'
            info['success'] = True
            return time_series, info
        
        try:
            # Step 1: Create complete time series with linear interpolation
            filled_ts = time_series.copy()
            
            # Linear interpolation to fill gaps for STL processing
            if n_missing > 0:
                # Find indices of valid data
                valid_indices = np.where(valid_mask)[0]
                missing_indices = np.where(missing_mask)[0]
                
                # Interpolate missing values
                interp_func = interp1d(
                    valid_indices, 
                    time_series[valid_mask],
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                filled_ts[missing_indices] = interp_func(missing_indices)
            
            # Step 2: Apply STL to complete (interpolated) time series
            stl_result = self.decompose_stl(filled_ts, times)
            
            # Step 3: Reconstruct missing values using STL components
            # Use trend + seasonal + zero-mean residual (as suggested)
            trend = stl_result['trend']
            seasonal = stl_result['seasonal']
            
            # Step 4: Fill original missing values with STL reconstruction
            final_filled = time_series.copy()
            
            for idx in np.where(missing_mask)[0]:
                # Reconstruct: T + S + R (where R = 0 as suggested)
                reconstructed_value = trend[idx] + seasonal[idx] + 0.0  # Zero-mean residual
                final_filled[idx] = reconstructed_value
            
            info['method'] = 'stl_linear_interp'
            info['success'] = True
            info['values_filled'] = n_missing
            
            return final_filled, info
            
        except Exception as e:
            info['method'] = f'failed_{type(e).__name__}'
            info['error'] = str(e)
            if not hasattr(self, '_fill_error_printed'):
                self._fill_error_printed = True
                print(f"⚠️ STL Fill Error: {type(e).__name__}: {str(e)}")
            return time_series, info
    
    def simple_reconstruct(self, stl_components: Dict, target_times: pd.DatetimeIndex) -> np.ndarray:
        """
        Simple reconstruction: trend extrapolation + seasonal climatology + mean residual.
        """
        trend = stl_components['trend']
        seasonal = stl_components['seasonal'] 
        residual = stl_components['residual']
        source_times = stl_components['times']
        
        # Extrapolate trend
        if len(trend) > 1:
            trend_func = interp1d(
                source_times.astype(np.int64),
                trend,
                kind='linear',
                fill_value='extrapolate'
            )
            target_trend = trend_func(target_times.astype(np.int64))
        else:
            target_trend = np.full(len(target_times), trend[0])
        
        # Use seasonal climatology (average by month)
        monthly_seasonal = {}
        for month in range(1, 13):
            month_mask = source_times.month == month
            if np.any(month_mask):
                monthly_seasonal[month] = np.mean(seasonal[month_mask])
            else:
                monthly_seasonal[month] = 0.0
        
        target_seasonal = np.array([monthly_seasonal[month] for month in target_times.month])
        
        # Use mean residual
        target_residual = np.full(len(target_times), np.mean(residual))
        
        # Reconstruct: TWSA = T + S + R
        reconstructed = target_trend + target_seasonal + target_residual
        
        return reconstructed

    def fill_pixel_timeseries(self, 
                              time_series: np.ndarray,
                              times: pd.DatetimeIndex) -> Tuple[np.ndarray, Dict]:
        """
        Use the simplified, correct STL approach.
        """
        return self.fill_pixel_timeseries_simple(time_series, times)
    
    def fill_grace_dataset_vectorized(self, grace_ds: xr.Dataset, 
                                    variable: str = 'tws_anomaly') -> Tuple[xr.Dataset, Dict]:
        """
        Fast vectorized gap filling using direct NumPy operations.
        
        This approach processes all pixels in chunks for better performance.
        """
        print("\n" + "="*70)
        print("🔬 GRACE GAP FILLING WITH STL DECOMPOSITION (VECTORIZED)")
        print("="*70)
        
        # Get data info
        data_array = grace_ds[variable]
        times = pd.DatetimeIndex(grace_ds.time.values)
        nt, nlat, nlon = data_array.shape
        n_pixels = nlat * nlon
        
        print(f"📊 Dataset shape: {nt} times × {nlat} lat × {nlon} lon = {n_pixels:,} pixels")
        print(f"   Time range: {times[0].strftime('%Y-%m')} to {times[-1].strftime('%Y-%m')}")
        
        # Calculate missing data statistics
        data = data_array.values
        n_missing_total = np.sum(np.isnan(data))
        pct_missing = 100 * n_missing_total / data.size
        print(f"   Missing data: {n_missing_total:,} / {data.size:,} ({pct_missing:.2f}%)")
        
        print(f"\n🚀 Processing {n_pixels:,} pixels with vectorized chunks...")
        
        # Process in chunks for memory efficiency
        chunk_size = min(1000, n_pixels)  # Process up to 1000 pixels at a time
        filled_data = np.copy(data)
        
        pixels_processed = 0
        pixels_filled = 0
        
        # Store results for validation analysis
        pixel_results = []
        
        # Reshape to (time, pixels) for easier processing
        data_reshaped = data.reshape(nt, -1)
        filled_reshaped = filled_data.reshape(nt, -1)
        
        for chunk_start in range(0, n_pixels, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_pixels)
            chunk_pixels = chunk_end - chunk_start
            
            # Process each pixel in the chunk
            for pixel_idx in range(chunk_start, chunk_end):
                pixel_ts = data_reshaped[:, pixel_idx]
                
                if np.sum(~np.isnan(pixel_ts)) >= 24:  # Enough data for STL
                    try:
                        # Use validation method for scientifically accurate assessment
                        filled_ts, info = self.fill_pixel_timeseries_with_validation(pixel_ts, times)
                        if info['success']:
                            filled_reshaped[:, pixel_idx] = filled_ts
                            pixels_filled += 1
                        
                        # Store results for validation analysis
                        pixel_results.append(info)
                    except:
                        pass  # Keep original data if filling fails
                
                pixels_processed += 1
            
            # Progress update
            if pixels_processed % (chunk_size * 5) == 0 or pixels_processed == n_pixels:
                print(f"   Processed: {pixels_processed:,}/{n_pixels:,} pixels ({100*pixels_processed/n_pixels:.1f}%)")
        
        # Reshape back to original dimensions
        filled_data = filled_reshaped.reshape(nt, nlat, nlon)
        
        # Create output dataset
        filled_ds = grace_ds.copy()
        filled_ds[variable] = (['time', 'lat', 'lon'], filled_data)
        
        # Add metadata
        filled_ds[variable].attrs['gap_filling_method'] = 'STL_vectorized'
        filled_ds[variable].attrs['gap_filling_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        # Calculate final statistics
        n_missing_after = np.sum(np.isnan(filled_data))
        reduction = 100 * (n_missing_total - n_missing_after) / n_missing_total if n_missing_total > 0 else 0
        pct_missing_after = 100 * n_missing_after / filled_data.size
        
        # Calculate validation metrics from pixel results
        stl_r2_list = []
        stl_rmse_list = []
        validation_points_list = []
        
        for pixel_info in pixel_results:
            # Check for STL method with validation metrics
            method = pixel_info.get('method', '')
            if (method.startswith('stl') and 
                not np.isnan(pixel_info.get('test_r2', np.nan)) and
                pixel_info.get('validation_points', 0) > 0):
                stl_r2_list.append(pixel_info['test_r2'])
                stl_rmse_list.append(pixel_info['test_rmse'])
                validation_points_list.append(pixel_info['validation_points'])
        
        # Summary statistics
        stats = {
            'n_pixels_total': n_pixels,
            'n_pixels_filled': pixels_filled,
            'n_pixels_skipped': n_pixels - pixels_filled,
            'n_values_original': data.size,
            'n_missing_original': n_missing_total,
            'n_missing_final': n_missing_after,
            'pct_missing_original': pct_missing,
            'pct_missing_final': pct_missing_after,
            'reduction_pct': reduction,
            'method': 'stl_vectorized'
        }
        
        # Add validation metrics if available
        if len(stl_r2_list) > 0:
            stats['stl_mean_test_r2'] = float(np.mean(stl_r2_list))
            stats['stl_std_test_r2'] = float(np.std(stl_r2_list))
            stats['stl_mean_test_rmse'] = float(np.mean(stl_rmse_list))
            stats['stl_std_test_rmse'] = float(np.std(stl_rmse_list))
            stats['pixels_with_validation'] = len(stl_r2_list)
            stats['total_validation_points'] = sum(validation_points_list)
            
            # Add to dataset attributes for manuscript use
            filled_ds[variable].attrs['stl_mean_test_r2'] = stats['stl_mean_test_r2']
            filled_ds[variable].attrs['stl_std_test_r2'] = stats['stl_std_test_r2']
            filled_ds[variable].attrs['stl_mean_test_rmse'] = stats['stl_mean_test_rmse']
            filled_ds[variable].attrs['stl_std_test_rmse'] = stats['stl_std_test_rmse']
            filled_ds[variable].attrs['pixels_with_validation'] = stats['pixels_with_validation']
            filled_ds[variable].attrs['total_validation_points'] = stats['total_validation_points']
        
        print("\n" + "="*70)
        print("✅ VECTORIZED GAP FILLING COMPLETE")
        print("="*70)
        print(f"📊 Pixels processed: {n_pixels:,}")
        print(f"   Successfully filled: {pixels_filled:,} ({100*pixels_filled/n_pixels:.1f}%)")
        print(f"   Missing data reduced: {n_missing_total:,} → {n_missing_after:,} ({reduction:.1f}% reduction)")
        print(f"   Final missing: {pct_missing_after:.2f}%")
        
        # Print validation metrics if available
        if len(stl_r2_list) > 0:
            print(f"\n🎯 STL VALIDATION METRICS (on 20% test data):")
            print(f"   Mean R²:   {np.mean(stl_r2_list):.4f} ± {np.std(stl_r2_list):.4f}")
            print(f"   Mean RMSE: {np.mean(stl_rmse_list):.4f} ± {np.std(stl_rmse_list):.4f} cm")
            print(f"   Pixels with validation: {len(stl_r2_list):,}")
            print(f"   Total validation points: {sum(validation_points_list):,}")
        else:
            print(f"\n⚠️ No validation metrics available (insufficient data for train/test split)")
        
        print("="*70)
        
        return filled_ds, stats

    def fill_grace_dataset(self, grace_ds: xr.Dataset, 
                          variable: str = None,
                          use_vectorized: bool = True) -> Tuple[xr.Dataset, Dict]:
        """
        Fill gaps in entire GRACE dataset.
        
        Parameters:
        -----------
        grace_ds : xr.Dataset
            GRACE dataset with time, lat, lon dimensions
        variable : str
            Variable name to fill. If None, auto-detect TWS variable.
        use_vectorized : bool
            If True, use fast vectorized approach. If False, use pixel-by-pixel loop.
        
        Returns:
        --------
        filled_ds : xr.Dataset
            Gap-filled dataset
        stats : Dict
            Summary statistics about filling
        """
        # Auto-detect TWS variable if not specified
        if variable is None:
            # Common TWS variable names in different GRACE datasets
            tws_candidates = ['tws_anomaly', 'lwe_thickness', 'GRACE_TWS', 'water_thickness']
            available_vars = list(grace_ds.data_vars.keys())
            
            for candidate in tws_candidates:
                if candidate in available_vars:
                    variable = candidate
                    break
            
            if variable is None:
                # Fallback: use first data variable with time dimension
                for var_name in available_vars:
                    if 'time' in grace_ds[var_name].dims:
                        variable = var_name
                        break
            
            if variable is None:
                raise ValueError(f"Could not auto-detect TWS variable. Available variables: {available_vars}")
            
            print(f"🔍 Auto-detected TWS variable: '{variable}'")
        
        if use_vectorized:
            return self.fill_grace_dataset_vectorized(grace_ds, variable)
        else:
            return self.fill_grace_dataset_loop(grace_ds, variable)
    
    def fill_grace_dataset_loop(self, grace_ds: xr.Dataset, 
                               variable: str = 'tws_anomaly') -> Tuple[xr.Dataset, Dict]:
        """
        Original pixel-by-pixel loop approach (slower but more detailed reporting).
        
        Parameters:
        -----------
        grace_ds : xr.Dataset
            GRACE dataset with time, lat, lon dimensions
        variable : str
            Variable name to fill
        
        Returns:
        --------
        filled_ds : xr.Dataset
            Gap-filled dataset
        stats : Dict
            Summary statistics about filling
        """
        print("\n" + "="*70)
        print("🔬 GRACE GAP FILLING WITH STL DECOMPOSITION")
        print("="*70)
        
        # Get data
        data = grace_ds[variable].values  # (time, lat, lon)
        times = pd.DatetimeIndex(grace_ds.time.values)
        
        nt, nlat, nlon = data.shape
        n_pixels = nlat * nlon
        
        print(f"📊 Dataset shape: {nt} times × {nlat} lat × {nlon} lon = {n_pixels:,} pixels")
        print(f"   Time range: {times[0].strftime('%Y-%m')} to {times[-1].strftime('%Y-%m')}")
        
        # Statistics
        n_missing_total = np.sum(np.isnan(data))
        pct_missing = 100 * n_missing_total / data.size
        print(f"   Missing data: {n_missing_total:,} / {data.size:,} ({pct_missing:.2f}%)")
        
        # Initialize output
        filled_data = np.zeros_like(data)
        
        # Track statistics
        stats = {
            'n_pixels_total': n_pixels,
            'n_pixels_filled': 0,
            'n_pixels_skipped': 0,
            'methods': {},
            'n_values_filled': 0
        }
        
        # Store individual pixel results for validation metrics
        pixel_results = []
        
        # Process each pixel
        from tqdm import tqdm
        
        print(f"\n🔄 Processing {n_pixels:,} pixels...")
        
        pixel_count = 0
        for i in range(nlat):
            for j in range(nlon):
                pixel_count += 1
                
                # Extract time series
                ts = data[:, i, j]
                
                # Fill gaps
                filled_ts, info = self.fill_pixel_timeseries(ts, times)
                
                # Store result
                filled_data[:, i, j] = filled_ts
                
                # Store info for validation
                pixel_results.append(info)
                
                # Update statistics
                method = info['method']
                stats['methods'][method] = stats['methods'].get(method, 0) + 1
                
                if info['success']:
                    stats['n_pixels_filled'] += 1
                    # Use values_filled if available, otherwise n_missing for backward compatibility
                    values_filled = info.get('values_filled', info.get('n_missing', 0))
                    stats['n_values_filled'] += values_filled
                else:
                    stats['n_pixels_skipped'] += 1
        
        # Create output dataset
        filled_ds = grace_ds.copy()
        filled_ds[variable].values = filled_data
        
        # Add metadata
        filled_ds[variable].attrs['gap_filling_method'] = 'STL_decomposition'
        filled_ds[variable].attrs['gap_filling_train_fraction'] = self.train_fraction
        filled_ds[variable].attrs['gap_filling_seasonal_period'] = self.seasonal_period
        
        # Calculate average validation metrics for STL-filled pixels
        stl_r2_list = []
        stl_rmse_list = []
        validation_points_list = []
        
        for pixel_info in pixel_results:
            # Check for any STL method with validation metrics
            method = pixel_info.get('method', '')
            if (method.startswith('stl') and 
                not np.isnan(pixel_info.get('test_r2', np.nan)) and
                pixel_info.get('validation_points', 0) > 0):
                stl_r2_list.append(pixel_info['test_r2'])
                stl_rmse_list.append(pixel_info['test_rmse'])
                validation_points_list.append(pixel_info['validation_points'])
        
        # Print summary
        print("\n" + "="*70)
        print("✅ GAP FILLING COMPLETE")
        print("="*70)
        print(f"📊 Pixels processed: {stats['n_pixels_total']:,}")
        print(f"   Successfully filled: {stats['n_pixels_filled']:,} ({100*stats['n_pixels_filled']/stats['n_pixels_total']:.1f}%)")
        print(f"   Skipped: {stats['n_pixels_skipped']:,}")
        print(f"   Values filled: {stats['n_values_filled']:,}")
        
        print(f"\n📋 Methods used:")
        for method, count in sorted(stats['methods'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {method}: {count:,} pixels")
        
        # Validation metrics for STL (on 20% test data as per scientific method)
        if len(stl_r2_list) > 0:
            print(f"\n🎯 STL VALIDATION METRICS (on 20% test data):")
            print(f"   Mean R²:   {np.mean(stl_r2_list):.4f} ± {np.std(stl_r2_list):.4f}")
            print(f"   Mean RMSE: {np.mean(stl_rmse_list):.4f} ± {np.std(stl_rmse_list):.4f} cm")
            print(f"   Pixels with validation: {len(stl_r2_list):,}")
            print(f"   Total validation points: {sum(validation_points_list):,}")
            print(f"   Avg validation points per pixel: {np.mean(validation_points_list):.1f}")
            
            # Check if we achieved the target metrics (R² ≥ 0.97, RMSE ~13mm)
            high_quality_pixels = sum(1 for r2 in stl_r2_list if r2 >= 0.97)
            low_rmse_pixels = sum(1 for rmse in stl_rmse_list if rmse <= 1.3)  # 13mm = 1.3cm
            
            print(f"\n📊 Quality Assessment:")
            print(f"   Pixels with R² ≥ 0.97: {high_quality_pixels}/{len(stl_r2_list)} ({100*high_quality_pixels/len(stl_r2_list):.1f}%)")
            print(f"   Pixels with RMSE ≤ 1.3cm: {low_rmse_pixels}/{len(stl_rmse_list)} ({100*low_rmse_pixels/len(stl_rmse_list):.1f}%)")
            
            # Store in stats
            stats['stl_mean_test_r2'] = float(np.mean(stl_r2_list))
            stats['stl_std_test_r2'] = float(np.std(stl_r2_list))
            stats['stl_mean_test_rmse'] = float(np.mean(stl_rmse_list))
            stats['stl_std_test_rmse'] = float(np.std(stl_rmse_list))
            stats['pixels_with_validation'] = len(stl_r2_list)
            stats['total_validation_points'] = sum(validation_points_list)
        else:
            print(f"\n⚠️ No validation metrics available (insufficient data for train/test split)")
        
        # Final statistics
        n_missing_after = np.sum(np.isnan(filled_data))
        pct_missing_after = 100 * n_missing_after / filled_data.size
        reduction = 100 * (n_missing_total - n_missing_after) / n_missing_total if n_missing_total > 0 else 0
        
        print(f"\n📈 Missing data reduced: {n_missing_total:,} → {n_missing_after:,} ({reduction:.1f}% reduction)")
        print(f"   Final missing: {pct_missing_after:.2f}%")
        print("="*70)
        
        return filled_ds, stats
    
    def save_filled_grace(self, filled_ds: xr.Dataset, output_path: str):
        """Save gap-filled GRACE dataset."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Compress for storage
        encoding = {var: {'zlib': True, 'complevel': 4} for var in filled_ds.data_vars}
        
        filled_ds.to_netcdf(output_path, encoding=encoding)
        print(f"💾 Gap-filled GRACE saved: {output_path}")


def main():
    """Command-line interface for gap filling."""
    import argparse
    from src_new_approach.utils_downscaling import load_config
    
    parser = argparse.ArgumentParser(description="Fill gaps in GRACE data using STL")
    parser.add_argument('--config', default='src_new_approach/config_coarse_to_fine.yaml',
                       help='Configuration file')
    parser.add_argument('--grace-path', help='Path to GRACE data (overrides config)')
    parser.add_argument('--output', help='Output path (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get paths
    grace_path = args.grace_path or config['paths']['grace_raw']
    output_path = args.output or config['paths']['grace_filled']
    
    # Initialize gap filler
    filler = GRACEGapFiller(config)
    
    # Load GRACE data
    grace_ds = filler.load_grace_data(grace_path)
    
    # Fill gaps
    filled_ds, stats = filler.fill_grace_dataset(grace_ds)
    
    # Save
    filler.save_filled_grace(filled_ds, output_path)
    
    print("\n✅ Gap filling complete!")


if __name__ == "__main__":
    main()

