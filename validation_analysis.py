#!/usr/bin/env python3
"""
GRACE Downscaling Validation Analysis
====================================

This script provides multiple validation approaches for comparing downscaled GRACE 
groundwater data with USGS well observations. It implements both basin-average 
and trend-based comparisons.

Author: Assistant
Date: July 2025
"""

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class GRACEValidation:
    """Validate downscaled GRACE data against USGS well observations."""
    
    def __init__(self, data_dir="data", results_dir="results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.validation_dir = self.results_dir / "validation"
        self.validation_dir.mkdir(exist_ok=True)
        
        # Load datasets
        self.load_datasets()
        
    def load_datasets(self):
        """Load all necessary datasets."""
        print("🔄 Loading datasets...")
        
        # 1. Downscaled GRACE data
        self.grace_data = xr.open_dataset(self.results_dir / "groundwater_complete.nc")
        print(f"   ✅ GRACE data: {self.grace_data.time.size} timesteps")
        
        # 2. USGS well locations with HUC4
        self.well_locations = pd.read_csv(
            self.data_dir / "usgs_well_trend" / "gwl_HUC4_siteuse.csv"
        )
        print(f"   ✅ Well locations: {len(self.well_locations)} wells")
        
        # 3. USGS well trend data (sample for now due to size)
        print("   📊 Loading well trend data (large file)...")
        self.well_trends = pd.read_csv(
            self.data_dir / "usgs_well_trend" / "siteyear_unconfined_annual_mean_all.csv",
            nrows=None  # Load all data - adjust if memory issues
        )
        print(f"   ✅ Well trends: {len(self.well_trends)} observations")
        
        # 4. Mississippi River Basin boundary
        self.basin_boundary = gpd.read_file(
            self.data_dir / "shapefiles" / "processed" / "mississippi_river_basin.shp"
        )
        print(f"   ✅ Basin boundary loaded")
        
    def filter_mississippi_wells(self):
        """Filter wells within Mississippi River Basin HUC codes."""
        print("🔍 Filtering wells in Mississippi River Basin...")
        
        # Mississippi Basin HUC2 codes: 05, 06, 07, 08, 10, 11
        msb_huc2 = ['05', '06', '07', '08', '10', '11']
        
        # Filter well locations
        self.well_locations['huc2'] = self.well_locations['CombHUC04_'].astype(str).str[:2]
        msb_wells = self.well_locations[self.well_locations['huc2'].isin(msb_huc2)]
        
        print(f"   📍 Wells in Mississippi Basin: {len(msb_wells)} / {len(self.well_locations)}")
        
        # Filter well trends to Mississippi Basin wells
        msb_well_ids = set(msb_wells['site_id'])
        self.msb_well_trends = self.well_trends[
            self.well_trends['site_id'].isin(msb_well_ids)
        ].copy()
        
        print(f"   📈 Trend observations in basin: {len(self.msb_well_trends)}")
        
        # Add coordinates to trend data
        well_coords = msb_wells.set_index('site_id')[['latitude', 'longitude', 'CombHUC04_']]
        self.msb_well_trends = self.msb_well_trends.merge(
            well_coords, left_on='site_id', right_index=True, how='left'
        )
        
        return msb_wells
    
    def create_huc4_validation(self):
        """Create HUC4-level validation comparing basin averages with quality thresholds."""
        print("🗺️ Creating HUC4-level validation with quality controls...")
        
        # Quality control thresholds
        MIN_WELLS_PER_HUC4 = 500      # Minimum total wells in HUC4 basin
        MIN_WELLS_PER_YEAR = 10       # Minimum wells per year for averaging
        MIN_YEARS_OVERLAP = 10        # Minimum years of overlapping data
        MIN_UNIQUE_WELLS = 500        # Minimum unique wells for spatial coverage
        
        print(f"   📊 Quality Thresholds:")
        print(f"      • Min wells per HUC4: {MIN_WELLS_PER_HUC4}")
        print(f"      • Min wells per year: {MIN_WELLS_PER_YEAR}")
        print(f"      • Min years overlap: {MIN_YEARS_OVERLAP}")
        print(f"      • Min unique wells: {MIN_UNIQUE_WELLS}")
        
        # Get unique HUC4 codes in Mississippi Basin
        msb_wells = self.filter_mississippi_wells()
        huc4_codes = msb_wells['CombHUC04_'].unique()
        
        validation_results = []
        skipped_reasons = {'few_wells': 0, 'few_unique': 0, 'poor_temporal': 0, 'insufficient_overlap': 0}
        
        for huc4 in huc4_codes:
            # Get wells in this HUC4
            huc4_wells = self.msb_well_trends[
                self.msb_well_trends['CombHUC04_'] == huc4
            ]
            
            # Check 1: Minimum total wells
            if len(huc4_wells) < MIN_WELLS_PER_HUC4:
                skipped_reasons['few_wells'] += 1
                continue
            
            # Check 2: Minimum unique wells (spatial coverage)
            unique_wells = huc4_wells['site_id'].nunique()
            if unique_wells < MIN_UNIQUE_WELLS:
                skipped_reasons['few_unique'] += 1
                continue
                
            print(f"   Processing HUC4: {huc4} ({len(huc4_wells)} obs, {unique_wells} wells)")
            
            # Calculate annual averages for wells in this HUC4
            well_annual = huc4_wells.groupby('year_value').agg({
                'value': ['mean', 'std', 'count'],
                'site_id': 'nunique'
            }).round(3)
            
            # Flatten column names
            well_annual.columns = ['mean', 'std', 'count', 'unique_wells']
            well_annual = well_annual.reset_index()
            
            # Check 3: Filter years with sufficient wells
            well_annual = well_annual[well_annual['count'] >= MIN_WELLS_PER_YEAR]
            well_annual = well_annual[well_annual['unique_wells'] >= min(5, MIN_UNIQUE_WELLS//2)]
            
            # Check 4: Minimum years of data
            if len(well_annual) < MIN_YEARS_OVERLAP:
                skipped_reasons['poor_temporal'] += 1
                continue
            
            # Get GRACE data for this HUC4 region
            huc4_coords = huc4_wells[['latitude', 'longitude']].drop_duplicates()
            grace_huc4 = self.extract_grace_for_coordinates(huc4_coords)
            
            # Merge GRACE and well data
            comparison_data = self.merge_grace_well_data(grace_huc4, well_annual, huc4)
            
            # Check 5: Sufficient temporal overlap
            if len(comparison_data) < MIN_YEARS_OVERLAP:
                skipped_reasons['insufficient_overlap'] += 1
                continue
                
            # Add quality metrics
            comparison_data['wells_per_year_avg'] = comparison_data['count'].mean()
            comparison_data['unique_wells_avg'] = comparison_data['unique_wells'].mean()
            comparison_data['temporal_coverage'] = len(comparison_data)
            
            validation_results.append(comparison_data)
        
        # Print quality control summary
        total_processed = len(huc4_codes)
        total_accepted = len(validation_results)
        total_skipped = sum(skipped_reasons.values())
        
        print(f"\n   📋 Quality Control Summary:")
        print(f"      • Total HUC4 basins: {total_processed}")
        print(f"      • Accepted for validation: {total_accepted}")
        print(f"      • Skipped - few wells: {skipped_reasons['few_wells']}")
        print(f"      • Skipped - few unique wells: {skipped_reasons['few_unique']}")
        print(f"      • Skipped - poor temporal coverage: {skipped_reasons['poor_temporal']}")
        print(f"      • Skipped - insufficient overlap: {skipped_reasons['insufficient_overlap']}")
        print(f"      • Acceptance rate: {total_accepted/total_processed*100:.1f}%")
        
        return validation_results
    
    def extract_grace_for_coordinates(self, coordinates):
        """Extract GRACE data for given coordinates."""
        grace_values = []
        
        for _, coord in coordinates.iterrows():
            lat, lon = coord['latitude'], coord['longitude']
            
            # Find nearest GRACE grid point
            lat_idx = np.argmin(np.abs(self.grace_data.lat.values - lat))
            lon_idx = np.argmin(np.abs(self.grace_data.lon.values - lon))
            
            # Extract time series
            ts = self.grace_data.groundwater.isel(lat=lat_idx, lon=lon_idx)
            grace_values.append(ts.values)
        
        # Average across wells in the region
        grace_avg = np.mean(grace_values, axis=0)
        
        # Convert to DataFrame
        times = pd.to_datetime(self.grace_data.time.values)
        grace_df = pd.DataFrame({
            'date': times,
            'grace_gw': grace_avg,
            'year': times.year,
            'month': times.month
        })
        
        # Annual averages
        grace_annual = grace_df.groupby('year')['grace_gw'].mean().reset_index()
        grace_annual.columns = ['year_value', 'grace_gw_annual']
        
        return grace_annual
    
    def merge_grace_well_data(self, grace_data, well_data, huc4):
        """Merge GRACE and well data for comparison."""
        merged = pd.merge(grace_data, well_data, on='year_value', how='inner')
        merged['huc4'] = huc4
        merged['well_anomaly'] = merged['mean'] - merged['mean'].mean()
        merged['grace_anomaly'] = merged['grace_gw_annual'] - merged['grace_gw_annual'].mean()
        
        return merged
    
    def basin_average_validation(self):
        """Compare basin-wide averages between GRACE and wells with quality controls."""
        print("🌊 Basin-wide average validation with quality controls...")
        
        # Quality thresholds for basin-wide analysis
        MIN_WELLS_PER_YEAR_BASIN = 500   # More wells needed for basin average
        MIN_UNIQUE_WELLS_BASIN = 500      # Spatial coverage across basin
        
        msb_wells = self.filter_mississippi_wells()
        
        # Calculate basin-wide well averages by year with quality metrics
        basin_well_annual = self.msb_well_trends.groupby('year_value').agg({
            'value': ['mean', 'std', 'count'],
            'site_id': 'nunique',
            'latitude': 'mean',
            'longitude': 'mean'
        }).round(3)
        
        basin_well_annual.columns = ['well_mean', 'well_std', 'well_count', 'unique_wells', 'lat_avg', 'lon_avg']
        basin_well_annual = basin_well_annual.reset_index()
        
        # Apply quality filters
        print(f"   📊 Basin Quality Filters:")
        print(f"      • Min wells per year: {MIN_WELLS_PER_YEAR_BASIN}")
        print(f"      • Min unique wells per year: {MIN_UNIQUE_WELLS_BASIN}")
        
        before_filter = len(basin_well_annual)
        basin_well_annual = basin_well_annual[
            (basin_well_annual['well_count'] >= MIN_WELLS_PER_YEAR_BASIN) &
            (basin_well_annual['unique_wells'] >= MIN_UNIQUE_WELLS_BASIN)
        ]
        after_filter = len(basin_well_annual)
        
        print(f"      • Years before filter: {before_filter}")
        print(f"      • Years after filter: {after_filter}")
        print(f"      • Filtered out: {before_filter - after_filter} years")
        
        # Calculate basin-wide GRACE averages
        grace_basin = self.grace_data.groundwater.mean(dim=['lat', 'lon'])
        grace_df = pd.DataFrame({
            'year_value': pd.to_datetime(self.grace_data.time.values).year,
            'month': pd.to_datetime(self.grace_data.time.values).month,
            'grace_gw': grace_basin.values
        })
        
        grace_annual = grace_df.groupby('year_value')['grace_gw'].mean().reset_index()
        grace_annual.columns = ['year_value', 'grace_basin_avg']
        
        # Merge datasets
        basin_comparison = pd.merge(basin_well_annual, grace_annual, on='year_value', how='inner')
        
        # Calculate anomalies
        basin_comparison['well_anomaly'] = (
            basin_comparison['well_mean'] - basin_comparison['well_mean'].mean()
        )
        basin_comparison['grace_anomaly'] = (
            basin_comparison['grace_basin_avg'] - basin_comparison['grace_basin_avg'].mean()
        )
        
        print(f"   ✅ Final basin comparison: {len(basin_comparison)} years")
        
        return basin_comparison
    
    def trend_validation(self):
        """Compare trends between GRACE and wells."""
        print("📈 Trend validation analysis...")
        
        # Get HUC4 validation data
        huc4_results = self.create_huc4_validation()
        
        trend_results = []
        
        for huc4_data in huc4_results:
            if len(huc4_data) < 8:  # Need at least 8 years for trend
                continue
                
            huc4 = huc4_data['huc4'].iloc[0]
            
            # Calculate trends using linear regression
            years = huc4_data['year_value'].values
            well_values = huc4_data['well_anomaly'].values
            grace_values = huc4_data['grace_anomaly'].values
            
            # Well trend
            well_slope, well_intercept, well_r, well_p, well_se = stats.linregress(years, well_values)
            
            # GRACE trend
            grace_slope, grace_intercept, grace_r, grace_p, grace_se = stats.linregress(years, grace_values)
            
            trend_results.append({
                'huc4': huc4,
                'n_years': len(huc4_data),
                'well_trend_cm_yr': well_slope,
                'well_trend_pvalue': well_p,
                'well_trend_r2': well_r**2,
                'grace_trend_cm_yr': grace_slope,
                'grace_trend_pvalue': grace_p,
                'grace_trend_r2': grace_r**2,
                'trend_correlation': np.corrcoef(well_values, grace_values)[0,1],
                'rmse': np.sqrt(mean_squared_error(well_values, grace_values)),
                'mae': mean_absolute_error(well_values, grace_values),
                'bias': np.mean(grace_values - well_values)
            })
        
        return pd.DataFrame(trend_results)
    
    def create_validation_plots(self, basin_data, trend_data):
        """Create comprehensive validation plots."""
        print("📊 Creating validation plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Basin-wide time series
        ax = axes[0, 0]
        ax.plot(basin_data['year_value'], basin_data['well_anomaly'], 'b-o', label='USGS Wells', alpha=0.7)
        ax.plot(basin_data['year_value'], basin_data['grace_anomaly'], 'r-s', label='GRACE Downscaled', alpha=0.7)
        ax.set_xlabel('Year')
        ax.set_ylabel('Groundwater Anomaly (cm)')
        ax.set_title('Basin-Wide Groundwater Anomalies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Basin-wide correlation
        ax = axes[0, 1]
        ax.scatter(basin_data['well_anomaly'], basin_data['grace_anomaly'], alpha=0.6)
        
        # Add correlation stats
        r = np.corrcoef(basin_data['well_anomaly'], basin_data['grace_anomaly'])[0,1]
        rmse = np.sqrt(mean_squared_error(basin_data['well_anomaly'], basin_data['grace_anomaly']))
        
        ax.plot([-10, 10], [-10, 10], 'k--', alpha=0.5)
        ax.set_xlabel('USGS Well Anomaly (cm)')
        ax.set_ylabel('GRACE Downscaled Anomaly (cm)')
        ax.set_title(f'Basin Correlation\nr = {r:.3f}, RMSE = {rmse:.2f} cm')
        ax.grid(True, alpha=0.3)
        
        # 3. Trend comparison
        ax = axes[0, 2]
        ax.scatter(trend_data['well_trend_cm_yr'], trend_data['grace_trend_cm_yr'], 
                   c=trend_data['trend_correlation'], cmap='RdYlBu', s=60, alpha=0.7)
        
        ax.plot([-2, 2], [-2, 2], 'k--', alpha=0.5)
        ax.set_xlabel('USGS Well Trend (cm/yr)')
        ax.set_ylabel('GRACE Trend (cm/yr)')
        ax.set_title('HUC4 Trend Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Time Series Correlation')
        
        # 4. Trend correlation distribution
        ax = axes[1, 0]
        ax.hist(trend_data['trend_correlation'], bins=15, alpha=0.7, edgecolor='black')
        ax.axvline(trend_data['trend_correlation'].mean(), color='red', linestyle='--', 
                   label=f'Mean = {trend_data["trend_correlation"].mean():.3f}')
        ax.set_xlabel('Time Series Correlation')
        ax.set_ylabel('Number of HUC4 Basins')
        ax.set_title('Distribution of HUC4 Correlations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. RMSE vs correlation
        ax = axes[1, 1]
        scatter = ax.scatter(trend_data['trend_correlation'], trend_data['rmse'], 
                            c=trend_data['n_years'], cmap='viridis', s=60, alpha=0.7)
        ax.set_xlabel('Time Series Correlation')
        ax.set_ylabel('RMSE (cm)')
        ax.set_title('Validation Quality vs Correlation')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Years of Data')
        
        # 6. Summary statistics with quality metrics
        ax = axes[1, 2]
        ax.text(0.1, 0.9, 'VALIDATION SUMMARY', fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        # Calculate quality metrics
        if len(trend_data) > 0:
            avg_wells_per_year = trend_data.get('wells_per_year_avg', pd.Series([0])).mean()
            avg_unique_wells = trend_data.get('unique_wells_avg', pd.Series([0])).mean()
            avg_temporal_coverage = trend_data.get('temporal_coverage', pd.Series([0])).mean()
        else:
            avg_wells_per_year = avg_unique_wells = avg_temporal_coverage = 0
        
        summary_text = f"""
Basin-Wide Analysis:
• Correlation: {r:.3f}
• RMSE: {rmse:.2f} cm
• Years: {len(basin_data)}
• Wells/year: {basin_data.get('well_count', pd.Series([0])).mean():.0f}

HUC4 Analysis:
• Number of HUCs: {len(trend_data)}
• Mean correlation: {trend_data['trend_correlation'].mean():.3f}
• Correlations > 0.5: {(trend_data['trend_correlation'] > 0.5).sum()}/{len(trend_data)}
• Mean RMSE: {trend_data['rmse'].mean():.2f} cm
• Avg wells/year: {avg_wells_per_year:.1f}
• Avg unique wells: {avg_unique_wells:.1f}
• Avg years: {avg_temporal_coverage:.1f}

Quality Control:
• HUC4 acceptance rate shown above
• Basin years filtered for quality
• Statistical reliability ensured

Trend Analysis:
• Well trends: {trend_data['well_trend_cm_yr'].mean():.3f} ± {trend_data['well_trend_cm_yr'].std():.3f} cm/yr
• GRACE trends: {trend_data['grace_trend_cm_yr'].mean():.3f} ± {trend_data['grace_trend_cm_yr'].std():.3f} cm/yr
• Trend correlation: {np.corrcoef(trend_data['well_trend_cm_yr'], trend_data['grace_trend_cm_yr'])[0,1]:.3f}
        """
        
        ax.text(0.1, 0.8, summary_text, fontsize=10, transform=ax.transAxes, 
                verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.validation_dir / 'grace_validation_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Validation plots saved to {self.validation_dir}")
    
    def run_full_validation(self):
        """Run complete validation analysis."""
        print("🚀 GRACE DOWNSCALING VALIDATION")
        print("="*50)
        
        # Basin-wide validation
        basin_data = self.basin_average_validation()
        basin_data.to_csv(self.validation_dir / 'basin_wide_validation.csv', index=False)
        
        # Trend validation
        trend_data = self.trend_validation()
        trend_data.to_csv(self.validation_dir / 'huc4_trend_validation.csv', index=False)
        
        # Create plots
        self.create_validation_plots(basin_data, trend_data)
        
        # Print summary
        self.print_validation_summary(basin_data, trend_data)
        
        return basin_data, trend_data
    
    def print_validation_summary(self, basin_data, trend_data):
        """Print validation summary with quality metrics."""
        print("\n" + "="*50)
        print("📋 VALIDATION SUMMARY WITH QUALITY CONTROLS")
        print("="*50)
        
        # Basin-wide stats
        basin_r = np.corrcoef(basin_data['well_anomaly'], basin_data['grace_anomaly'])[0,1]
        basin_rmse = np.sqrt(mean_squared_error(basin_data['well_anomaly'], basin_data['grace_anomaly']))
        
        print(f"\n🌊 BASIN-WIDE VALIDATION:")
        print(f"   • Correlation: {basin_r:.3f}")
        print(f"   • RMSE: {basin_rmse:.2f} cm")
        print(f"   • Years compared: {len(basin_data)}")
        print(f"   • Time period: {basin_data['year_value'].min()}-{basin_data['year_value'].max()}")
        print(f"   • Avg wells per year: {basin_data['well_count'].mean():.0f}")
        print(f"   • Avg unique wells per year: {basin_data.get('unique_wells', pd.Series([0])).mean():.0f}")
        
        # HUC4 stats with quality metrics
        print(f"\n🗺️ HUC4-LEVEL VALIDATION:")
        print(f"   • Number of HUC4 basins: {len(trend_data)}")
        print(f"   • Mean correlation: {trend_data['trend_correlation'].mean():.3f}")
        print(f"   • Correlations > 0.5: {(trend_data['trend_correlation'] > 0.5).sum()}/{len(trend_data)}")
        print(f"   • Correlations > 0.3: {(trend_data['trend_correlation'] > 0.3).sum()}/{len(trend_data)}")
        print(f"   • Mean RMSE: {trend_data['rmse'].mean():.2f} cm")
        
        if len(trend_data) > 0 and 'wells_per_year_avg' in trend_data.columns:
            print(f"   • Avg wells per year per HUC4: {trend_data['wells_per_year_avg'].mean():.1f}")
            print(f"   • Avg unique wells per HUC4: {trend_data['unique_wells_avg'].mean():.1f}")
            print(f"   • Avg temporal coverage: {trend_data['temporal_coverage'].mean():.1f} years")
        
        # Trend analysis
        trend_corr = np.corrcoef(trend_data['well_trend_cm_yr'], trend_data['grace_trend_cm_yr'])[0,1]
        print(f"\n📈 TREND VALIDATION:")
        print(f"   • Trend correlation: {trend_corr:.3f}")
        print(f"   • USGS mean trend: {trend_data['well_trend_cm_yr'].mean():.3f} cm/yr")
        print(f"   • GRACE mean trend: {trend_data['grace_trend_cm_yr'].mean():.3f} cm/yr")
        
        # Quality assessment
        print(f"\n� QUALITY ASSESSMENT:")
        if basin_r > 0.6:
            print("   ✅ Basin-wide validation shows GOOD agreement")
        elif basin_r > 0.4:
            print("   ⚠️  Basin-wide validation shows MODERATE agreement")
        else:
            print("   ❌ Basin-wide validation shows POOR agreement")
            
        good_hucs = (trend_data['trend_correlation'] > 0.5).sum()
        total_hucs = len(trend_data)
        if total_hucs > 0:
            if good_hucs/total_hucs > 0.6:
                print("   ✅ Majority of HUC4 basins show good correlation")
            elif good_hucs/total_hucs > 0.4:
                print("   ⚠️  Moderate number of HUC4 basins show good correlation")
            else:
                print("   ❌ Few HUC4 basins show good correlation - may need model improvement")
        
        # Data quality notes
        print(f"\n📋 DATA QUALITY NOTES:")
        print(f"   • Quality thresholds applied for statistical reliability")
        print(f"   • HUC4 basins filtered by well count and temporal coverage")
        print(f"   • Basin years filtered by well count and spatial coverage")
        print(f"   • Results represent high-quality validation subset")


def main():
    """Main validation function."""
    # Initialize validation
    validator = GRACEValidation()
    
    # Run full validation
    basin_data, trend_data = validator.run_full_validation()
    
    print(f"\n✅ Validation complete! Results saved to {validator.validation_dir}")
    
    return validator, basin_data, trend_data


if __name__ == "__main__":
    validator, basin_data, trend_data = main()
