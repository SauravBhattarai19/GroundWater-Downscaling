#!/usr/bin/env python3
"""
Annual Trend Validation for GRACE Groundwater Downscaling

This module compares annual trends between USGS well observations and model predictions.
Handles the difference between absolute annual values (USGS) and monthly anomalies (model).

Key Features:
- Converts USGS annual values to annual trends/anomalies
- Aggregates monthly model anomalies to annual anomalies
- Focuses on trend correlation rather than absolute value comparison
- Designed for the new USGS well trend dataset
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr, linregress, kendalltau, mannwhitneyu
import scipy.stats as stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AnnualTrendValidator:
    """Validates annual trends between USGS wells and model predictions."""
    
    def __init__(self, 
                 overlap_start_year=2003,     # Start of overlapping period
                 overlap_end_year=2020,       # End of overlapping period
                 min_years_in_period=15,      # Minimum years within overlap period
                 model_reference_start=2004,  # Model's reference period start (from NC metadata)
                 model_reference_end=2009,    # Model's reference period end (from NC metadata)
                 detrend_method='mean_reference', # How to calculate anomalies from USGS data
                 outlier_threshold=3.0):      # Standard deviations for outlier detection
        """
        Initialize annual trend validator.
        
        Parameters:
        -----------
        overlap_start_year : int
            Start year of the overlapping period (GRACE data available)
        overlap_end_year : int
            End year of the overlapping period
        min_years_in_period : int
            Minimum years of data needed within the overlap period
        model_reference_start : int
            Start year of model's reference period for anomaly calculation
        model_reference_end : int
            End year of model's reference period for anomaly calculation
        detrend_method : str
            Method to convert USGS values to anomalies ('mean_reference', 'linear', 'mean', 'median')
            'mean_reference' uses same reference period as model for consistency
        outlier_threshold : float
            Number of standard deviations to identify outliers
        """
        self.overlap_start = overlap_start_year
        self.overlap_end = overlap_end_year
        self.min_years_in_period = min_years_in_period
        self.overlap_years = list(range(overlap_start_year, overlap_end_year + 1))
        self.model_ref_start = model_reference_start
        self.model_ref_end = model_reference_end
        self.model_ref_years = list(range(model_reference_start, model_reference_end + 1))
        self.detrend_method = detrend_method
        self.outlier_threshold = outlier_threshold
        
        # Set up directories
        self.results_dir = Path("results/validation")
        self.figures_dir = Path("figures/validation")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.figures_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"📋 Annual Trend Validator initialized:")
        print(f"   Overlap period: {overlap_start_year}-{overlap_end_year} ({len(self.overlap_years)} years)")
        print(f"   Model reference period: {model_reference_start}-{model_reference_end} ({len(self.model_ref_years)} years)")
        print(f"   Min years in period: {min_years_in_period}")
        print(f"   Detrend method: {detrend_method}")
        print(f"   🔧 Using SAME reference period as model for fair comparison!")
        
        # Load data
        self._load_data()
        
        # Auto-detect and verify model reference period
        self._auto_detect_model_reference_period()
    
    def _load_data(self):
        """Load model predictions and USGS well data."""
        print("\n📦 Loading data for annual trend validation...")
        
        # Load model predictions (monthly anomalies)
        model_path = self._find_model_output()
        self.model_data = xr.open_dataset(model_path)
        print(f"✅ Model data: {self.model_data.groundwater.shape}")
        print(f"   Time range: {self.model_data.time.values[0]} to {self.model_data.time.values[-1]}")
        
        # Load USGS well annual data
        annual_data_path = "data/usgs_well_trend/siteyear_unconfined_annual_mean_all.csv"
        metadata_path = "data/usgs_well_trend/gwl_sites_region.csv"
        
        # Load annual data
        print("📊 Loading USGS annual data...")
        self.usgs_annual = pd.read_csv(annual_data_path, low_memory=False)
        print(f"✅ USGS annual data: {len(self.usgs_annual)} records")
        
        # Basic data validation
        required_cols = ['year_value', 'site_id', 'value']
        missing_cols = [col for col in required_cols if col not in self.usgs_annual.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in USGS data: {missing_cols}")
        
        # Remove rows with missing essential data
        before_cleanup = len(self.usgs_annual)
        self.usgs_annual = self.usgs_annual.dropna(subset=required_cols)
        after_cleanup = len(self.usgs_annual)
        if before_cleanup != after_cleanup:
            print(f"   Removed {before_cleanup - after_cleanup} rows with missing data")
        
        # Load metadata
        self.usgs_metadata = pd.read_csv(metadata_path)
        print(f"✅ USGS metadata: {len(self.usgs_metadata)} wells")
        
        # Process USGS data
        self._process_usgs_data()
        
        # Convert model data to annual
        self._convert_model_to_annual()
        
        # Basic data quality check
        self._check_data_quality()
        
        # Verify reference period consistency
        self._verify_reference_period_consistency()
    
    def _auto_detect_model_reference_period(self):
        """Auto-detect model reference period from NC file metadata."""
        print(f"\n🔍 AUTO-DETECTING MODEL REFERENCE PERIOD")
        print("="*50)
        
        try:
            # Read reference period from model metadata
            model_ref_attr = self.model_data.attrs.get('reference_period', None)
            
            if model_ref_attr:
                print(f"  Found model reference period: {model_ref_attr}")
                
                # Parse reference period (expected format: "2004-01 to 2009-12")
                import re
                match = re.match(r'(\d{4})-\d{2} to (\d{4})-\d{2}', str(model_ref_attr))
                
                if match:
                    detected_start = int(match.group(1))
                    detected_end = int(match.group(2))
                    
                    # Check if user-provided parameters match
                    if detected_start != self.model_ref_start or detected_end != self.model_ref_end:
                        print(f"  ⚠️  UPDATING reference period to match model:")
                        print(f"     Original: {self.model_ref_start}-{self.model_ref_end}")
                        print(f"     Detected: {detected_start}-{detected_end}")
                        
                        # Update parameters
                        self.model_ref_start = detected_start
                        self.model_ref_end = detected_end
                        self.model_ref_years = list(range(detected_start, detected_end + 1))
                        
                        print(f"  ✅ Updated to use model's actual reference period")
                    else:
                        print(f"  ✅ User parameters match model reference period")
                else:
                    print(f"  ⚠️  Could not parse reference period format: {model_ref_attr}")
            else:
                print(f"  ⚠️  No reference_period attribute found in model metadata")
                print(f"  Using user-provided reference period: {self.model_ref_start}-{self.model_ref_end}")
                
        except Exception as e:
            print(f"  ❌ Error detecting model reference period: {e}")
            print(f"  Using user-provided reference period: {self.model_ref_start}-{self.model_ref_end}")
        
        print(f"  Final reference period: {self.model_ref_start}-{self.model_ref_end} ({len(self.model_ref_years)} years)")
    
    def _find_model_output(self):
        """Find the model output file."""
        possible_files = [
            "results/groundwater_complete.nc",
            "results/groundwater_storage_anomalies.nc",
            "results/groundwater_storage_anomalies_enhanced.nc"
        ]
        
        for f in possible_files:
            if Path(f).exists():
                return f
        
        raise FileNotFoundError(f"No model output found. Tried: {possible_files}")
    
    def _process_usgs_data(self):
        """Process USGS data into annual anomalies by well."""
        print("🔄 Processing USGS data into annual anomalies...")
        
        # Handle duplicates - same well+year can have multiple POI_type_ID entries
        print("   Handling duplicate entries...")
        n_before = len(self.usgs_annual)
        
        # Use pivot_table with mean aggregation to handle duplicates
        usgs_pivot = self.usgs_annual.pivot_table(
            index='year_value', 
            columns='site_id', 
            values='value',
            aggfunc='mean'  # Take mean if multiple entries per well-year
        )
        
        n_unique_combinations = len(self.usgs_annual.drop_duplicates(['year_value', 'site_id']))
        print(f"   Original records: {n_before:,}")
        print(f"   Unique well-year combinations: {n_unique_combinations:,}")
        print(f"   Duplicates handled: {n_before - n_unique_combinations:,}")
        
        # Convert years to datetime for consistency
        usgs_pivot.index = pd.to_datetime(usgs_pivot.index, format='%Y')
        
        print(f"   USGS pivot shape: {usgs_pivot.shape}")
        print(f"   Year range: {usgs_pivot.index.min().year} to {usgs_pivot.index.max().year}")
        
        # Convert absolute values to anomalies
        self.usgs_anomalies = pd.DataFrame(index=usgs_pivot.index)
        
        for well_id in tqdm(usgs_pivot.columns, desc="Converting to anomalies"):
            well_series = usgs_pivot[well_id].dropna()
            
            # Filter to overlap period only
            overlap_mask = well_series.index.year.isin(self.overlap_years)
            well_overlap_series = well_series[overlap_mask]
            
            if len(well_overlap_series) < self.min_years_in_period:
                continue  # Skip wells with insufficient years in overlap period
            
            # Convert to anomalies using specified method
            if self.detrend_method == 'mean_reference':
                # Use SAME reference period as model (2004-2009) for fair comparison
                reference_mask = well_series.index.year.isin(self.model_ref_years)
                reference_data = well_series[reference_mask]
                
                if len(reference_data) < 3:  # Need at least 3 years of reference data
                    continue  # Skip wells without sufficient reference period data
                
                reference_mean = reference_data.mean()
                
                # Calculate anomalies for overlap period using reference mean
                anomalies = well_overlap_series.values - reference_mean
                
            elif self.detrend_method == 'linear':
                # Remove linear trend over overlap period
                years = np.arange(len(well_overlap_series))
                slope, intercept, _, _, _ = linregress(years, well_overlap_series.values)
                trend = slope * years + intercept
                anomalies = well_overlap_series.values - trend
                
            elif self.detrend_method == 'mean':
                # Simple mean removal over overlap period
                anomalies = well_overlap_series.values - well_overlap_series.mean()
                
            elif self.detrend_method == 'median':
                # Median removal over overlap period (more robust to outliers)
                anomalies = well_overlap_series.values - well_overlap_series.median()
                
            else:
                raise ValueError(f"Unknown detrend method: {self.detrend_method}")
            
            self.usgs_anomalies[well_id] = pd.Series(
                anomalies, 
                index=well_overlap_series.index
            )
        
        print(f"✅ Processed {len(self.usgs_anomalies.columns)} wells with sufficient data")
    
    def _convert_model_to_annual(self):
        """Convert monthly model anomalies to annual anomalies for overlap period."""
        print(f"🔄 Converting model data to annual anomalies ({self.overlap_start}-{self.overlap_end})...")
        
        # Convert time coordinate to pandas datetime
        model_time = pd.to_datetime(self.model_data.time.values)
        
        # Group by year and calculate annual means (only for overlap period)
        annual_data = []
        years = []
        
        for year in self.overlap_years:
            year_mask = model_time.year == year
            
            if year_mask.sum() >= 8:  # Need at least 8 months of data for reliable annual mean
                year_data = self.model_data.groundwater[year_mask].mean(dim='time')
                annual_data.append(year_data)
                years.append(pd.Timestamp(f'{year}-01-01'))
        
        if annual_data:
            # Create annual dataset
            self.model_annual = xr.concat(annual_data, dim='time')
            self.model_annual = self.model_annual.assign_coords(time=years)
            
            print(f"✅ Model annual data: {self.model_annual.shape}")
            print(f"   Year range: {min(years).year} to {max(years).year}")
            print(f"   Years included: {[y.year for y in years]}")
        else:
            raise ValueError(f"No valid annual data could be created from model for {self.overlap_start}-{self.overlap_end}")
    
    def _check_data_quality(self):
        """Check data quality and print summary."""
        print("\n📊 Data Quality Summary:")
        
        # Model data coverage
        model_years = len(self.model_annual.time)
        print(f"  Model annual data: {model_years} years")
        
        # USGS data coverage
        usgs_years = len(self.usgs_anomalies.index)
        usgs_wells = len(self.usgs_anomalies.columns)
        print(f"  USGS annual data: {usgs_wells} wells, {usgs_years} years")
        
        # Overlap analysis (should be perfect since we filtered to overlap period)
        model_years_set = set(pd.to_datetime(self.model_annual.time.values).year)
        usgs_years_set = set(self.usgs_anomalies.index.year)
        actual_overlap = model_years_set.intersection(usgs_years_set)
        
        print(f"  Target overlap period: {self.overlap_start}-{self.overlap_end} ({len(self.overlap_years)} years)")
        print(f"  Actual overlap achieved: {len(actual_overlap)} years")
        if len(actual_overlap) < len(self.overlap_years):
            missing_years = set(self.overlap_years) - actual_overlap
            print(f"  Missing years: {sorted(missing_years)}")
        
        # Wells meeting minimum requirement in overlap period
        wells_sufficient = usgs_wells  # All wells already filtered to meet requirement
        
        print(f"  Wells with ≥{self.min_years_in_period} years in overlap period: {wells_sufficient}/{usgs_wells}")
        
        if wells_sufficient == 0:
            print(f"  ⚠️  No wells meet the {self.min_years_in_period}-year requirement in overlap period")
        elif wells_sufficient < 50:
            print(f"  ⚠️  Only {wells_sufficient} wells meet the requirements - this may limit analysis")
        else:
            print(f"  ✅ {wells_sufficient} wells available for robust trend analysis")
    
    def _verify_reference_period_consistency(self):
        """Verify that model and USGS data use consistent reference periods."""
        print(f"\n🔍 REFERENCE PERIOD CONSISTENCY CHECK")
        print("="*50)
        
        # Check model reference period from metadata
        try:
            model_ref_attr = self.model_data.attrs.get('reference_period', 'Unknown')
            print(f"  Model reference period: {model_ref_attr}")
            print(f"  Validator reference period: {self.model_ref_start}-{self.model_ref_end}")
            
            # Verify they match
            expected_ref = f"{self.model_ref_start}-01 to {self.model_ref_end}-12"
            if model_ref_attr == expected_ref:
                print(f"  ✅ Reference periods MATCH - consistent anomaly calculation")
            else:
                print(f"  ⚠️  Reference periods may not match exactly")
                print(f"      Expected: {expected_ref}")
                print(f"      Found: {model_ref_attr}")
        except Exception as e:
            print(f"  ⚠️  Could not verify model reference period: {e}")
        
        # Check USGS reference period coverage
        if self.detrend_method == 'mean_reference':
            wells_with_ref_data = 0
            total_ref_years_available = 0
            
            for well_id in self.usgs_anomalies.columns:
                well_data_years = set(self.usgs_anomalies[well_id].dropna().index.year)
                ref_years_available = len(set(self.model_ref_years).intersection(well_data_years))
                if ref_years_available >= 3:
                    wells_with_ref_data += 1
                total_ref_years_available += ref_years_available
            
            avg_ref_years = total_ref_years_available / len(self.usgs_anomalies.columns) if len(self.usgs_anomalies.columns) > 0 else 0
            
            print(f"\n  USGS Reference Period Coverage:")
            print(f"    Wells with ≥3 years in reference period: {wells_with_ref_data}/{len(self.usgs_anomalies.columns)}")
            print(f"    Average reference years per well: {avg_ref_years:.1f}/{len(self.model_ref_years)}")
            
            if wells_with_ref_data < len(self.usgs_anomalies.columns) * 0.8:
                print(f"    ⚠️  Many wells lack sufficient reference period data")
            else:
                print(f"    ✅ Good reference period coverage")
        
        else:
            print(f"  Using {self.detrend_method} method - not using model reference period")
            print(f"  ⚠️  This may lead to inconsistent anomaly calculations!")
    
    def validate_annual_trends(self):
        """
        Main validation: compare annual trends between USGS wells and model.
        
        Returns:
        --------
        pd.DataFrame
            Validation metrics for each well
        """
        print(f"\n🎯 ANNUAL TREND VALIDATION")
        print("="*50)
        print(f"Focus period: {self.overlap_start}-{self.overlap_end} ({len(self.overlap_years)} years)")
        print(f"Requirement: ≥{self.min_years_in_period} years within this period")
        print("This ensures robust trend analysis using the complete GRACE-USGS overlap period.")
        
        results = []
        processing_stats = {
            'total_wells': len(self.usgs_anomalies.columns),
            'has_metadata': 0,
            'in_bounds': 0,
            'sufficient_data': 0,
            'validated': 0
        }
        
        # Process each well
        for well_id in tqdm(self.usgs_anomalies.columns, desc="Validating wells"):
            
            # Get well metadata
            well_meta = self.usgs_metadata[self.usgs_metadata['site_id'] == well_id]
            if len(well_meta) == 0:
                continue
            processing_stats['has_metadata'] += 1
            
            lat = well_meta.iloc[0]['latitude']
            lon = well_meta.iloc[0]['longitude']
            
            # Check if within model bounds
            if not self._is_in_bounds(lat, lon):
                continue
            processing_stats['in_bounds'] += 1
            
            # Get USGS annual anomaly time series (already filtered to overlap period)
            usgs_series = self.usgs_anomalies[well_id].dropna()
            
            # Data is already filtered - all wells should meet requirements
            processing_stats['sufficient_data'] += 1
            
            try:
                # Extract model annual anomaly at well location
                model_series = self.model_annual.sel(
                    lat=lat, lon=lon, method='nearest'
                ).to_pandas()
                
                # Align time series (both should be annual now within overlap period)
                aligned = self._align_annual_series(model_series, usgs_series)
                
                if aligned is None or len(aligned[0]) < 5:  # Need at least 5 years for basic correlation
                    continue
                
                model_aligned, usgs_aligned = aligned
                
                # Remove outliers
                model_clean, usgs_clean = self._remove_outliers(model_aligned, usgs_aligned)
                
                if len(model_clean) < 5:  # Need at least 5 years after outlier removal
                    continue
                
                # Calculate metrics
                metrics = self._calculate_annual_metrics(model_clean, usgs_clean)
                
                if metrics:
                    # Add well information
                    metrics.update({
                        'well_id': well_id,
                        'lat': lat,
                        'lon': lon,
                        'state': well_meta.iloc[0].get('state', 'Unknown'),
                        'region': well_meta.iloc[0].get('Region_nam', 'Unknown'),
                        'n_years_total': len(usgs_series),
                        'n_years_used': len(model_clean),
                        'outliers_removed': len(model_aligned) - len(model_clean),
                        'detrend_method': self.detrend_method
                    })
                    results.append(metrics)
                    processing_stats['validated'] += 1
                    
            except Exception as e:
                continue
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Print processing summary
        print(f"\nProcessing Summary:")
        for key, value in processing_stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        if len(results_df) > 0:
            # Save results
            output_path = self.results_dir / "annual_trend_validation.csv"
            results_df.to_csv(output_path, index=False)
            
            # Print summary
            self._print_summary(results_df)
            
            # Create visualizations
            self._create_validation_plots(results_df)
            
            print(f"\n✅ Results saved to: {output_path}")
        else:
            print("❌ No wells could be validated with current requirements")
        
        return results_df
    
    def _is_in_bounds(self, lat, lon):
        """Check if coordinates are within model bounds."""
        return (lat >= self.model_annual.lat.min() and 
                lat <= self.model_annual.lat.max() and
                lon >= self.model_annual.lon.min() and 
                lon <= self.model_annual.lon.max())
    
    def _align_annual_series(self, model_series, usgs_series):
        """Align annual model and USGS time series within overlap period."""
        # Ensure both have datetime index with yearly frequency
        if not isinstance(model_series.index, pd.DatetimeIndex):
            model_series.index = pd.to_datetime(model_series.index)
        if not isinstance(usgs_series.index, pd.DatetimeIndex):
            usgs_series.index = pd.to_datetime(usgs_series.index)
        
        # Extract years and find common years (should be within overlap period)
        model_years = set(model_series.index.year)
        usgs_years = set(usgs_series.index.year)
        common_years = model_years.intersection(usgs_years)
        
        if len(common_years) < 3:  # Need at least 3 years for basic analysis
            return None
        
        # Create common time indices (use January 1st for each year)
        common_times = [pd.Timestamp(f'{year}-01-01') for year in sorted(common_years)]
        
        # Extract data for common years
        model_aligned = pd.Series(index=common_times, dtype=float)
        usgs_aligned = pd.Series(index=common_times, dtype=float)
        
        for time in common_times:
            year = time.year
            
            # Get model data for this year
            model_year_data = model_series[model_series.index.year == year]
            if len(model_year_data) > 0:
                model_aligned[time] = model_year_data.iloc[0]
            
            # Get USGS data for this year
            usgs_year_data = usgs_series[usgs_series.index.year == year]
            if len(usgs_year_data) > 0:
                usgs_aligned[time] = usgs_year_data.iloc[0]
        
        # Remove any NaN values
        valid_mask = ~(model_aligned.isna() | usgs_aligned.isna())
        
        return model_aligned[valid_mask], usgs_aligned[valid_mask]
    
    def _remove_outliers(self, model_series, usgs_series):
        """Remove outliers using z-score method."""
        # Calculate z-scores
        model_z = np.abs((model_series - model_series.mean()) / model_series.std())
        usgs_z = np.abs((usgs_series - usgs_series.mean()) / usgs_series.std())
        
        # Keep only points where both are within threshold
        mask = (model_z < self.outlier_threshold) & (usgs_z < self.outlier_threshold)
        
        return model_series[mask], usgs_series[mask]
    
    def _calculate_annual_metrics(self, model_series, usgs_series):
        """Calculate comprehensive annual trend validation metrics optimized for trend analysis."""
        try:
            # Primary correlation metrics - Spearman is better for trends
            spearman_r, spearman_p = spearmanr(model_series, usgs_series)
            pearson_r, pearson_p = pearsonr(model_series, usgs_series)  # For comparison
            kendall_tau, kendall_p = kendalltau(model_series, usgs_series)
            
            # Mann-Kendall trend test for both series
            model_mk_trend, model_mk_p = self._mann_kendall_test(model_series)
            usgs_mk_trend, usgs_mk_p = self._mann_kendall_test(usgs_series)
            
            # Sen's slope estimation (robust trend slope)
            model_sens_slope = self._sens_slope(model_series)
            usgs_sens_slope = self._sens_slope(usgs_series)
            
            # Linear trend analysis (for comparison)
            model_trend_slope, _, model_trend_r, model_trend_p, _ = linregress(
                range(len(model_series)), model_series
            )
            usgs_trend_slope, _, usgs_trend_r, usgs_trend_p, _ = linregress(
                range(len(usgs_series)), usgs_series
            )
            
            # Trend characteristics
            trend_direction_agreement = np.sign(model_sens_slope) == np.sign(usgs_sens_slope)
            both_trends_significant = (model_mk_p < 0.05) and (usgs_mk_p < 0.05)
            
            # Nash-Sutcliffe Efficiency
            nse = self._nash_sutcliffe_efficiency(usgs_series, model_series)
            
            # Kling-Gupta Efficiency  
            kge, kge_components = self._kling_gupta_efficiency(usgs_series, model_series)
            
            # Error metrics (on original scales for interpretability)
            rmse = np.sqrt(mean_squared_error(usgs_series, model_series))
            mae = np.mean(np.abs(usgs_series - model_series))
            
            # Relative metrics
            relative_rmse = rmse / np.mean(np.abs(usgs_series)) if np.mean(np.abs(usgs_series)) > 0 else np.nan
            bias = np.mean(model_series - usgs_series)
            relative_bias = bias / np.mean(np.abs(usgs_series)) if np.mean(np.abs(usgs_series)) > 0 else np.nan
            
            # Variability comparison
            variance_ratio = model_series.std() / usgs_series.std()
            
            # Interannual variability correlation (year-to-year changes)
            if len(model_series) > 3:
                model_diff = model_series.diff().dropna()
                usgs_diff = usgs_series.diff().dropna()
                if len(model_diff) >= 3 and model_diff.std() > 0 and usgs_diff.std() > 0:
                    variability_spearman, variability_p = spearmanr(model_diff, usgs_diff)
                else:
                    variability_spearman, variability_p = np.nan, np.nan
            else:
                variability_spearman, variability_p = np.nan, np.nan
            
            return {
                # Primary trend metrics (Spearman is most important for trends)
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'kendall_tau': kendall_tau,
                'kendall_p': kendall_p,
                
                # Traditional correlation (for comparison)
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                
                # Mann-Kendall trend tests
                'model_mk_trend': model_mk_trend,
                'model_mk_p': model_mk_p,
                'usgs_mk_trend': usgs_mk_trend,
                'usgs_mk_p': usgs_mk_p,
                
                # Robust trend slopes (Sen's slope - preferred for trends)
                'model_sens_slope': model_sens_slope,
                'usgs_sens_slope': usgs_sens_slope,
                
                # Linear trend slopes (for comparison)
                'model_linear_slope': model_trend_slope,
                'usgs_linear_slope': usgs_trend_slope,
                
                # Trend agreement metrics
                'trend_direction_agreement': trend_direction_agreement,
                'both_trends_significant': both_trends_significant,
                'slope_ratio': model_sens_slope / usgs_sens_slope if usgs_sens_slope != 0 else np.nan,
                
                # Efficiency metrics (widely used in hydrology)
                'nash_sutcliffe_efficiency': nse,
                'kling_gupta_efficiency': kge,
                'kge_correlation': kge_components[0],
                'kge_bias': kge_components[1], 
                'kge_variability': kge_components[2],
                
                # Error metrics
                'rmse': rmse,
                'mae': mae,
                'relative_rmse': relative_rmse,
                'bias': bias,
                'relative_bias': relative_bias,
                
                # Variability metrics
                'variance_ratio': variance_ratio,
                'interannual_variability_spearman': variability_spearman,
                'interannual_variability_p': variability_p,
                
                # Data characteristics
                'model_std': model_series.std(),
                'usgs_std': usgs_series.std(),
                'model_mean': model_series.mean(),
                'usgs_mean': usgs_series.mean()
            }
            
        except Exception as e:
            return None
    
    def _mann_kendall_test(self, data):
        """
        Mann-Kendall trend test for time series.
        Returns: (trend_direction, p_value)
        trend_direction: 1 (increasing), -1 (decreasing), 0 (no trend)
        """
        n = len(data)
        if n < 3:
            return 0, 1.0
        
        # Calculate S statistic
        S = 0
        for i in range(n-1):
            for j in range(i+1, n):
                S += np.sign(data.iloc[j] - data.iloc[i])
        
        # Calculate variance
        var_S = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate Z statistic
        if S > 0:
            Z = (S - 1) / np.sqrt(var_S)
        elif S < 0:
            Z = (S + 1) / np.sqrt(var_S)
        else:
            Z = 0
        
        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
        
        # Determine trend direction
        if p_value < 0.05:
            trend_direction = 1 if S > 0 else -1
        else:
            trend_direction = 0
        
        return trend_direction, p_value
    
    def _sens_slope(self, data):
        """
        Calculate Sen's slope estimator (robust trend slope).
        """
        n = len(data)
        if n < 2:
            return np.nan
        
        slopes = []
        for i in range(n-1):
            for j in range(i+1, n):
                if j != i:
                    slope = (data.iloc[j] - data.iloc[i]) / (j - i)
                    slopes.append(slope)
        
        return np.median(slopes) if slopes else np.nan
    
    def _nash_sutcliffe_efficiency(self, observed, modeled):
        """
        Calculate Nash-Sutcliffe Efficiency.
        NSE = 1 - (sum of squared residuals) / (sum of squared deviations from mean)
        Range: -∞ to 1, where 1 is perfect fit
        """
        if len(observed) != len(modeled):
            return np.nan
        
        numerator = np.sum((observed - modeled) ** 2)
        denominator = np.sum((observed - np.mean(observed)) ** 2)
        
        if denominator == 0:
            return np.nan
        
        return 1 - (numerator / denominator)
    
    def _kling_gupta_efficiency(self, observed, modeled):
        """
        Calculate Kling-Gupta Efficiency and its components.
        KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
        where r=correlation, α=variability ratio, β=bias ratio
        """
        if len(observed) != len(modeled):
            return np.nan, (np.nan, np.nan, np.nan)
        
        # Correlation coefficient
        r = np.corrcoef(observed, modeled)[0, 1] if np.std(observed) > 0 and np.std(modeled) > 0 else 0
        
        # Variability ratio (α)
        alpha = np.std(modeled) / np.std(observed) if np.std(observed) > 0 else np.nan
        
        # Bias ratio (β) 
        beta = np.mean(modeled) / np.mean(observed) if np.mean(observed) != 0 else np.nan
        
        # KGE calculation
        if not (np.isnan(r) or np.isnan(alpha) or np.isnan(beta)):
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        else:
            kge = np.nan
        
        return kge, (r, alpha, beta)
    
    def _print_summary(self, results_df):
        """Print comprehensive validation summary."""
        print(f"\n📊 ANNUAL TREND VALIDATION SUMMARY")
        print("="*50)
        print(f"Wells validated: {len(results_df)}")
        print(f"🔧 Reference period consistency: Model anomalies use {self.model_ref_start}-{self.model_ref_end}")
        print(f"🔧 USGS anomalies calculated using {'SAME reference period' if self.detrend_method == 'mean_reference' else self.detrend_method + ' method'}")
        
        # Primary correlation summary (Spearman is better for trends)
        print(f"\nTrend Correlation (Spearman ρ - PRIMARY METRIC):")
        print(f"  Mean: {results_df['spearman_r'].mean():.3f} ± {results_df['spearman_r'].std():.3f}")
        print(f"  Median: {results_df['spearman_r'].median():.3f}")
        print(f"  Range: [{results_df['spearman_r'].min():.3f}, {results_df['spearman_r'].max():.3f}]")
        
        # Performance categories based on Spearman correlation
        excellent = (results_df['spearman_r'] >= 0.7).sum()
        good = ((results_df['spearman_r'] >= 0.5) & (results_df['spearman_r'] < 0.7)).sum()
        fair = ((results_df['spearman_r'] >= 0.3) & (results_df['spearman_r'] < 0.5)).sum()
        poor = (results_df['spearman_r'] < 0.3).sum()
        
        print(f"\nPerformance Distribution:")
        print(f"  Excellent (r≥0.7): {excellent} wells ({excellent/len(results_df)*100:.1f}%)")
        print(f"  Good (0.5≤r<0.7): {good} wells ({good/len(results_df)*100:.1f}%)")
        print(f"  Fair (0.3≤r<0.5): {fair} wells ({fair/len(results_df)*100:.1f}%)")
        print(f"  Poor (r<0.3): {poor} wells ({poor/len(results_df)*100:.1f}%)")
        
        # Hydrological efficiency metrics
        print(f"\nHydrological Efficiency Metrics:")
        nse_mean = results_df['nash_sutcliffe_efficiency'].mean()
        kge_mean = results_df['kling_gupta_efficiency'].mean()
        print(f"  Nash-Sutcliffe Efficiency (NSE): {nse_mean:.3f} (>0.5 = good, >0.7 = very good)")
        print(f"  Kling-Gupta Efficiency (KGE): {kge_mean:.3f} (>0.5 = good, >0.7 = very good)")
        
        good_nse = (results_df['nash_sutcliffe_efficiency'] > 0.5).sum()
        good_kge = (results_df['kling_gupta_efficiency'] > 0.5).sum()
        print(f"  Wells with good NSE (>0.5): {good_nse}/{len(results_df)} ({good_nse/len(results_df)*100:.1f}%)")
        print(f"  Wells with good KGE (>0.5): {good_kge}/{len(results_df)} ({good_kge/len(results_df)*100:.1f}%)")
        
        # Mann-Kendall trend analysis
        print(f"\nMann-Kendall Trend Analysis:")
        model_trends_sig = (results_df['model_mk_p'] < 0.05).sum()
        usgs_trends_sig = (results_df['usgs_mk_p'] < 0.05).sum()
        both_trends_sig = results_df['both_trends_significant'].sum()
        trend_agree = results_df['trend_direction_agreement'].sum()
        
        print(f"  Wells with significant model trends (p<0.05): {model_trends_sig} ({model_trends_sig/len(results_df)*100:.1f}%)")
        print(f"  Wells with significant USGS trends (p<0.05): {usgs_trends_sig} ({usgs_trends_sig/len(results_df)*100:.1f}%)")
        print(f"  Wells with both trends significant: {both_trends_sig} ({both_trends_sig/len(results_df)*100:.1f}%)")
        print(f"  Wells with same trend direction: {trend_agree}/{len(results_df)} ({trend_agree/len(results_df)*100:.1f}%)")
        
        # Sen's slope comparison
        print(f"\nSen's Slope Analysis (Robust Trend Slopes):")
        slope_ratio_data = results_df['slope_ratio'].dropna()
        if len(slope_ratio_data) > 0:
            print(f"  Mean slope ratio (model/USGS): {slope_ratio_data.mean():.3f}")
            print(f"  Median slope ratio: {slope_ratio_data.median():.3f}")
            print(f"  Ideal is ~1.0 (similar trend magnitudes)")
        
        # Error metrics
        print(f"\nError Metrics:")
        print(f"  Mean RMSE: {results_df['rmse'].mean():.3f}")
        print(f"  Mean MAE: {results_df['mae'].mean():.3f}")
        print(f"  Mean relative bias: {results_df['relative_bias'].mean():.1%}")
        
        # Variability analysis
        print(f"\nVariability Analysis:")
        print(f"  Mean variance ratio (model/USGS): {results_df['variance_ratio'].mean():.3f}")
        print(f"  Ideal is ~1.0 (model captures similar variability)")
        
        variability_data = results_df['interannual_variability_spearman'].dropna()
        if len(variability_data) > 0:
            print(f"  Interannual variability correlation (Spearman): {variability_data.mean():.3f} (n={len(variability_data)})")
        
        # Statistical significance
        significant_spearman = (results_df['spearman_p'] < 0.05).sum()
        significant_kendall = (results_df['kendall_p'] < 0.05).sum()
        print(f"\nStatistical Significance:")
        print(f"  Significant Spearman correlations (p<0.05): {significant_spearman}/{len(results_df)} ({significant_spearman/len(results_df)*100:.1f}%)")
        print(f"  Significant Kendall tau (p<0.05): {significant_kendall}/{len(results_df)} ({significant_kendall/len(results_df)*100:.1f}%)")
        
        # Regional breakdown if available
        if 'region' in results_df.columns:
            print(f"\nRegional Performance (Spearman ρ):")
            regional_stats = results_df.groupby('region')['spearman_r'].agg(['count', 'mean', 'std']).round(3)
            for region, stats in regional_stats.iterrows():
                print(f"  {region}: {stats['count']} wells, ρ̄={stats['mean']:.3f}±{stats['std']:.3f}")
    
    def _create_validation_plots(self, results_df):
        """Create comprehensive validation visualizations."""
        # Set up figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # 1. Spearman correlation histogram (primary metric)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(results_df['spearman_r'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.axvline(results_df['spearman_r'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {results_df["spearman_r"].mean():.3f}')
        ax1.set_xlabel('Spearman Correlation (ρ)')
        ax1.set_ylabel('Number of Wells')
        ax1.set_title('Trend Correlation Distribution (Primary Metric)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Spatial map of Spearman correlations
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(results_df['lon'], results_df['lat'], 
                            c=results_df['spearman_r'], s=60,
                            cmap='RdYlBu', vmin=-0.2, vmax=1,
                            edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Spatial Distribution of Trend Correlations')
        plt.colorbar(scatter, ax=ax2, label='Spearman ρ')
        ax2.grid(True, alpha=0.3)
        
        # 3. Nash-Sutcliffe Efficiency distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(results_df['nash_sutcliffe_efficiency'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        ax3.axvline(0.5, color='orange', linestyle='--', label='Good (0.5)')
        ax3.axvline(0.7, color='red', linestyle='--', label='Very Good (0.7)')
        ax3.set_xlabel('Nash-Sutcliffe Efficiency')
        ax3.set_ylabel('Number of Wells')
        ax3.set_title('NSE Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Kling-Gupta Efficiency distribution
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.hist(results_df['kling_gupta_efficiency'], bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
        ax4.axvline(0.5, color='orange', linestyle='--', label='Good (0.5)')
        ax4.axvline(0.7, color='red', linestyle='--', label='Very Good (0.7)')
        ax4.set_xlabel('Kling-Gupta Efficiency')
        ax4.set_ylabel('Number of Wells')
        ax4.set_title('KGE Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Sen's slope comparison (robust trend slopes)
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.scatter(results_df['usgs_sens_slope'], results_df['model_sens_slope'], alpha=0.6, color='orange')
        # Add diagonal line
        slope_data = pd.concat([results_df['usgs_sens_slope'], results_df['model_sens_slope']]).dropna()
        if len(slope_data) > 0:
            trend_min, trend_max = slope_data.min(), slope_data.max()
            ax5.plot([trend_min, trend_max], [trend_min, trend_max], 'r--', alpha=0.7, label='Perfect Agreement')
        ax5.set_xlabel('USGS Sen\'s Slope')
        ax5.set_ylabel('Model Sen\'s Slope')
        ax5.set_title('Robust Trend Slope Comparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Trend direction agreement pie chart
        ax6 = fig.add_subplot(gs[1, 1])
        trend_counts = results_df['trend_direction_agreement'].value_counts()
        labels = ['Disagree', 'Agree'] if False in trend_counts.index else ['Agree']
        ax6.pie(trend_counts.values, labels=labels, autopct='%1.1f%%', 
               colors=['lightcoral', 'lightblue'][:len(labels)])
        ax6.set_title('Trend Direction Agreement\n(Sen\'s Slope)')
        
        # 7. NSE vs KGE comparison
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.scatter(results_df['nash_sutcliffe_efficiency'], results_df['kling_gupta_efficiency'], 
                   alpha=0.6, color='green')
        ax7.axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='Good Threshold')
        ax7.axvline(0.5, color='orange', linestyle='--', alpha=0.7)
        ax7.set_xlabel('Nash-Sutcliffe Efficiency')
        ax7.set_ylabel('Kling-Gupta Efficiency')
        ax7.set_title('NSE vs KGE Comparison')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Regional performance (if available)
        if 'region' in results_df.columns and results_df['region'].nunique() > 1:
            ax8 = fig.add_subplot(gs[1, 3])
            region_stats = results_df.groupby('region')['spearman_r'].mean().sort_values(ascending=True)
            ax8.barh(range(len(region_stats)), region_stats.values)
            ax8.set_yticks(range(len(region_stats)))
            ax8.set_yticklabels([r[:20] for r in region_stats.index], fontsize=10)
            ax8.set_xlabel('Mean Spearman ρ')
            ax8.set_title('Performance by Region')
            ax8.grid(True, alpha=0.3)
        else:
            # If no regional data, show Mann-Kendall significance
            ax8 = fig.add_subplot(gs[1, 3])
            mk_counts = [
                results_df['both_trends_significant'].sum(),
                (results_df['model_mk_p'] < 0.05).sum() - results_df['both_trends_significant'].sum(),
                (results_df['usgs_mk_p'] < 0.05).sum() - results_df['both_trends_significant'].sum(),
                len(results_df) - (results_df['model_mk_p'] < 0.05).sum() - (results_df['usgs_mk_p'] < 0.05).sum() + results_df['both_trends_significant'].sum()
            ]
            labels = ['Both Significant', 'Model Only', 'USGS Only', 'Neither']
            ax8.pie([c for c in mk_counts if c > 0], 
                   labels=[l for i, l in enumerate(labels) if mk_counts[i] > 0],
                   autopct='%1.1f%%')
            ax8.set_title('Mann-Kendall Trend\nSignificance')
        
        # 9-12. Example time series plots (best performing wells by Spearman correlation)
        top_wells = results_df.nlargest(4, 'spearman_r')
        
        for i, (_, well) in enumerate(top_wells.iterrows()):
            ax = fig.add_subplot(gs[2 + i//2, i%2])
            
            well_id = well['well_id']
            lat, lon = well['lat'], well['lon']
            
            # Get the time series for this well
            usgs_series = self.usgs_anomalies[well_id].dropna()
            model_series = self.model_annual.sel(lat=lat, lon=lon, method='nearest').to_pandas()
            
            # Align them
            aligned = self._align_annual_series(model_series, usgs_series)
            if aligned is not None:
                model_aligned, usgs_aligned = aligned
                
                years = [t.year for t in model_aligned.index]
                ax.plot(years, model_aligned.values, 'b-o', label='Model', markersize=4, linewidth=2)
                ax.plot(years, usgs_aligned.values, 'r-s', label='USGS', markersize=4, linewidth=2)
                
                ax.set_xlabel('Year')
                ax.set_ylabel('Anomaly')
                ax.set_title(f'{well_id[:15]}... (ρ={well["spearman_r"]:.3f}, NSE={well["nash_sutcliffe_efficiency"]:.2f})')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Annual Trend Validation Results (2003-2020)\n'
                    f'{len(results_df)} wells, mean Spearman ρ={results_df["spearman_r"].mean():.3f}, mean NSE={results_df["nash_sutcliffe_efficiency"].mean():.3f}', 
                    fontsize=16, y=0.98)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'annual_trend_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Validation plots saved to: {self.figures_dir / 'annual_trend_validation.png'}")
    
    def create_detailed_comparison(self, n_wells=20):
        """Create detailed comparison plots for top performing wells."""
        # Load results
        results_path = self.results_dir / "annual_trend_validation.csv"
        if not results_path.exists():
            print("❌ Run validation first to generate detailed comparison")
            return
        
        results_df = pd.read_csv(results_path)
        top_wells = results_df.nlargest(n_wells, 'spearman_r')
        
        # Create subplot grid
        n_cols = 4
        n_rows = (n_wells + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_wells > 1 else [axes]
        
        for i, (_, well) in enumerate(top_wells.iterrows()):
            if i >= n_wells:
                break
                
            ax = axes[i]
            
            well_id = well['well_id']
            lat, lon = well['lat'], well['lon']
            
            # Get the time series for this well
            usgs_series = self.usgs_anomalies[well_id].dropna()
            model_series = self.model_annual.sel(lat=lat, lon=lon, method='nearest').to_pandas()
            
            # Align them
            aligned = self._align_annual_series(model_series, usgs_series)
            if aligned is not None:
                model_aligned, usgs_aligned = aligned
                
                years = [t.year for t in model_aligned.index]
                ax.plot(years, model_aligned.values, 'b-o', label='Model', markersize=6, linewidth=2)
                ax.plot(years, usgs_aligned.values, 'r-s', label='USGS', markersize=6, linewidth=2)
                
                # Add trend lines
                from scipy.stats import linregress
                model_slope, model_intercept, _, _, _ = linregress(range(len(model_aligned)), model_aligned.values)
                usgs_slope, usgs_intercept, _, _, _ = linregress(range(len(usgs_aligned)), usgs_aligned.values)
                
                model_trend = [model_intercept + model_slope * j for j in range(len(years))]
                usgs_trend = [usgs_intercept + usgs_slope * j for j in range(len(years))]
                
                ax.plot(years, model_trend, 'b--', alpha=0.7, linewidth=1)
                ax.plot(years, usgs_trend, 'r--', alpha=0.7, linewidth=1)
                
                ax.set_xlabel('Year')
                ax.set_ylabel('Anomaly')
                ax.set_title(f'{well_id}\nρ={well["spearman_r"]:.3f}, NSE={well["nash_sutcliffe_efficiency"]:.3f}, trend_agree={well["trend_direction_agreement"]}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_wells, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Top {n_wells} Performing Wells - Annual Trend Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'top_{n_wells}_wells_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Detailed comparison saved to: {self.figures_dir / f'top_{n_wells}_wells_comparison.png'}")


def main():
    """Main function for annual trend validation."""
    print("🚀 ANNUAL TREND VALIDATION FOR GRACE GROUNDWATER")
    print("="*60)
    
    # Initialize validator with model's reference period for consistency
    validator = AnnualTrendValidator(
        overlap_start_year=2003,         # Start of GRACE data availability
        overlap_end_year=2020,           # End of reliable overlap period
        min_years_in_period=15,          # Need at least 15 years within 2003-2020 period
        model_reference_start=2004,      # Model's reference period start (from NC metadata)
        model_reference_end=2009,        # Model's reference period end (from NC metadata)
        detrend_method='mean_reference', # Use SAME reference period as model
        outlier_threshold=3.0            # Remove extreme outliers
    )
    
    # Run validation
    results = validator.validate_annual_trends()
    
    # Create detailed comparison for top wells
    if len(results) > 0:
        validator.create_detailed_comparison(n_wells=min(20, len(results)))
    
    print("\n✅ Annual trend validation complete!")
    print(f"📁 Results saved to: {validator.results_dir}")
    print(f"📊 Figures saved to: {validator.figures_dir}")
    
    return results


if __name__ == "__main__":
    main() 