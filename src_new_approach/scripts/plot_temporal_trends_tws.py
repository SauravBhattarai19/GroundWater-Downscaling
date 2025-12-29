#!/usr/bin/env python3
"""
Temporal Trends TWS Visualization Script for Downscaled GRACE Data

Creates a publication-quality 3x4 subplot layout showing temporal trends of TWS 
for each month (JAN-DEC) with median and quartile ranges across the MRB.

Usage:
    python plot_temporal_trends_tws.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.1)
sns.set_palette("husl")

def load_mrb_boundary():
    """Load Mississippi River Basin boundary."""
    mrb_file = Path("/home/sauravbhattarai/Documents/ORISE/GroundWater-Downscaling/data/shapefiles/MRB.geojson")
    
    if mrb_file.exists():
        print(f"✅ Loading MRB boundary from: {mrb_file}")
        gdf = gpd.read_file(mrb_file)
        print(f"   Features: {len(gdf)}")
        
        # If multiple features, combine into single geometry
        if len(gdf) > 1:
            combined_geometry = unary_union(gdf.geometry)
            print(f"   Combined {len(gdf)} features into single MRB boundary")
            return combined_geometry
        else:
            return gdf.geometry.iloc[0]
    else:
        print("⚠️ MRB boundary file not found, creating approximate boundary")
        mrb_coords = [
            (-89.0, 47.8), (-95.2, 48.9), (-103.0, 45.8), (-106.8, 40.5),
            (-108.0, 37.0), (-106.0, 32.0), (-100.0, 29.5), (-94.0, 28.8),
            (-89.2, 29.1), (-85.0, 30.2), (-82.5, 32.0), (-81.0, 35.5),
            (-80.5, 39.0), (-79.8, 40.8), (-80.2, 42.0), (-82.8, 43.5),
            (-84.5, 45.2), (-87.0, 46.8), (-89.0, 47.8)
        ]
        return Polygon(mrb_coords)

def create_precise_mask(lats, lons, mrb_polygon):
    """Create a precise mask for MRB boundary."""
    print("🎯 Creating precise MRB mask...")
    
    # Create coordinate arrays
    lon_2d, lat_2d = np.meshgrid(lons, lats)
    
    # Vectorized point-in-polygon test
    mask = np.zeros_like(lon_2d, dtype=bool)
    
    # Process in chunks for better performance
    chunk_size = 1000
    total_points = lon_2d.size
    processed = 0
    
    for i in range(0, len(lats), chunk_size):
        for j in range(0, len(lons), chunk_size):
            i_end = min(i + chunk_size, len(lats))
            j_end = min(j + chunk_size, len(lons))
            
            for ii in range(i, i_end):
                for jj in range(j, j_end):
                    point = Point(lon_2d[ii, jj], lat_2d[ii, jj])
                    mask[ii, jj] = mrb_polygon.contains(point) or mrb_polygon.touches(point)
                    processed += 1
        
        if processed % 5000 == 0:
            print(f"   Progress: {processed:,}/{total_points:,} ({100*processed/total_points:.1f}%)")
    
    print(f"   Points inside MRB: {mask.sum():,} / {mask.size:,} ({100*mask.sum()/mask.size:.1f}%)")
    return mask

def extract_temporal_trends(ds_downscaled, ds_original, mrb_mask_fine, mrb_mask_coarse, tws_var_down, tws_var_orig):
    """
    Extract temporal trends for both downscaled and original GRACE data.
    
    Returns:
    --------
    dict: Monthly trends with median, Q25, Q75 for each year for both datasets
    """
    print("📈 Extracting temporal trends for downscaled and original GRACE...")
    
    # Get downscaled data and apply MRB mask
    tws_data_down = ds_downscaled[tws_var_down].values.copy()  # (time, lat, lon)
    for t in range(tws_data_down.shape[0]):
        tws_data_down[t, ~mrb_mask_fine] = np.nan
    
    # Get original GRACE data and apply MRB mask  
    tws_data_orig = ds_original[tws_var_orig].values.copy()  # (time, lat, lon)
    for t in range(tws_data_orig.shape[0]):
        tws_data_orig[t, ~mrb_mask_coarse] = np.nan
    
    # Convert to DataFrame with time index
    times_down = pd.to_datetime(ds_downscaled.time.values)
    times_orig = pd.to_datetime(ds_original.time.values)
    
    monthly_trends = {}
    
    for month in range(1, 13):  # 1=Jan, 2=Feb, ..., 12=Dec
        month_name = pd.Timestamp(2000, month, 1).strftime('%B')  # Full month name
        print(f"   Processing {month_name}...")
        
        # Find indices for this month in both datasets
        month_indices_down = [i for i, t in enumerate(times_down) if t.month == month]
        month_indices_orig = [i for i, t in enumerate(times_orig) if t.month == month]
        
        if len(month_indices_down) == 0 or len(month_indices_orig) == 0:
            continue
        
        # Extract data for this month across years
        month_data_down = tws_data_down[month_indices_down, :, :]
        month_times_down = [times_down[i] for i in month_indices_down]
        
        month_data_orig = tws_data_orig[month_indices_orig, :, :]
        month_times_orig = [times_orig[i] for i in month_indices_orig]
        
        # Match years between datasets
        years_down = [t.year for t in month_times_down]
        years_orig = [t.year for t in month_times_orig]
        common_years = sorted(set(years_down) & set(years_orig))
        
        # Calculate statistics for each common year
        yearly_stats = []
        for year in common_years:
            # Get data for this year from both datasets
            down_idx = next(i for i, t in enumerate(month_times_down) if t.year == year)
            orig_idx = next(i for i, t in enumerate(month_times_orig) if t.year == year)
            
            down_slice = month_data_down[down_idx, :, :]
            orig_slice = month_data_orig[orig_idx, :, :]
            
            # Get valid values
            valid_down = down_slice[~np.isnan(down_slice)]
            valid_orig = orig_slice[~np.isnan(orig_slice)]
            
            if len(valid_down) > 0 and len(valid_orig) > 0:
                # Calculate validation metrics (using aggregated values)
                median_down = np.median(valid_down)
                median_orig = np.median(valid_orig)
                
                stats = {
                    'year': year,
                    'date': month_times_down[down_idx],
                    # Downscaled stats
                    'median_down': median_down,
                    'q25_down': np.percentile(valid_down, 25),
                    'q75_down': np.percentile(valid_down, 75),
                    'mean_down': np.mean(valid_down),
                    # Original GRACE stats
                    'median_orig': median_orig,
                    'q25_orig': np.percentile(valid_orig, 25),
                    'q75_orig': np.percentile(valid_orig, 75),
                    'mean_orig': np.mean(valid_orig),
                    # Validation metrics
                    'diff_median': median_down - median_orig,
                    'count_down': len(valid_down),
                    'count_orig': len(valid_orig)
                }
                yearly_stats.append(stats)
        
        if yearly_stats:
            trends_df = pd.DataFrame(yearly_stats)
            
            # Calculate R² and RMSE for this month
            if len(trends_df) > 1:
                from sklearn.metrics import r2_score, mean_squared_error
                r2 = r2_score(trends_df['median_orig'], trends_df['median_down'])
                rmse = np.sqrt(mean_squared_error(trends_df['median_orig'], trends_df['median_down']))
                
                # Add metrics to dataframe
                trends_df['r2'] = r2
                trends_df['rmse'] = rmse
                
                print(f"      {month_name}: {len(trends_df)} years, R²={r2:.3f}, RMSE={rmse:.2f} cm")
            else:
                trends_df['r2'] = np.nan
                trends_df['rmse'] = np.nan
        else:
            trends_df = pd.DataFrame()
            
        monthly_trends[month] = trends_df
    
    return monthly_trends

def create_temporal_trends_plot():
    """Create temporal trends visualization comparing downscaled vs original GRACE."""
    
    print("📊 Creating temporal trends visualization with validation...")
    
    # Load downscaled data
    grace_downscaled_path = "results_coarse_to_fine/grace_downscaled_5km.nc"
    print(f"📂 Loading downscaled data from: {grace_downscaled_path}")
    ds_downscaled = xr.open_dataset(grace_downscaled_path)
    
    # Load original GRACE data
    grace_original_path = "/home/sauravbhattarai/Documents/ORISE/GroundWater-Downscaling/data/processed_coarse_to_fine/grace_filled_stl.nc"
    print(f"📂 Loading original GRACE from: {grace_original_path}")
    ds_original = xr.open_dataset(grace_original_path)
    
    # Get TWS variables
    # Downscaled
    tws_var_down = None
    for var_name in ['grace_downscaled', 'tws_anomaly', 'downscaled_grace', 'tws', 'grace']:
        if var_name in ds_downscaled:
            tws_var_down = var_name
            break
    
    if tws_var_down is None:
        tws_var_down = list(ds_downscaled.data_vars)[0]
        print(f"⚠️ Using downscaled variable: {tws_var_down}")
    else:
        print(f"✅ Using downscaled variable: {tws_var_down}")
    
    # Original  
    tws_var_orig = None
    for var_name in ['tws_anomaly', 'grace', 'lwe_thickness', 'TWS']:
        if var_name in ds_original:
            tws_var_orig = var_name
            break
    
    if tws_var_orig is None:
        tws_var_orig = list(ds_original.data_vars)[0]
        print(f"⚠️ Using original variable: {tws_var_orig}")
    else:
        print(f"✅ Using original variable: {tws_var_orig}")
    
    # Load MRB boundary and create masks for both resolutions
    mrb_polygon = load_mrb_boundary()
    
    # Fine resolution mask (for downscaled data)
    lats_fine = ds_downscaled.lat.values
    lons_fine = ds_downscaled.lon.values
    print("🔍 Creating MRB mask for downscaled data (5km)...")
    mrb_mask_fine = create_precise_mask(lats_fine, lons_fine, mrb_polygon)
    
    # Coarse resolution mask (for original GRACE data)
    lats_coarse = ds_original.lat.values
    lons_coarse = ds_original.lon.values
    print("🔍 Creating MRB mask for original GRACE data...")
    mrb_mask_coarse = create_precise_mask(lats_coarse, lons_coarse, mrb_polygon)
    
    # Extract temporal trends with validation
    monthly_trends = extract_temporal_trends(
        ds_downscaled, ds_original, mrb_mask_fine, mrb_mask_coarse, 
        tws_var_down, tws_var_orig
    )
    
    # Get time info
    times_down = pd.to_datetime(ds_downscaled.time.values)
    times_orig = pd.to_datetime(ds_original.time.values)
    print(f"📅 Downscaled range: {times_down[0].strftime('%Y-%m')} to {times_down[-1].strftime('%Y-%m')}")
    print(f"📅 Original range: {times_orig[0].strftime('%Y-%m')} to {times_orig[-1].strftime('%Y-%m')}")
    
    # Set up publication-quality figure
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    
    # Month names
    month_names = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                   'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    
    # Calculate global y-limits for consistency from both datasets
    all_values = []
    
    for month_num, trends_df in monthly_trends.items():
        if not trends_df.empty:
            all_values.extend(trends_df['q25_down'].values)
            all_values.extend(trends_df['q75_down'].values)
            all_values.extend(trends_df['q25_orig'].values)
            all_values.extend(trends_df['q75_orig'].values)
            all_values.extend(trends_df['median_down'].values)
            all_values.extend(trends_df['median_orig'].values)
    
    y_min = min(all_values) * 1.1
    y_max = max(all_values) * 1.1
    
    print(f"🎨 Y-axis range: {y_min:.1f} to {y_max:.1f} cm")
    
    # Create subplots (3 rows, 4 columns)
    for i in range(12):
        month_num = i + 1
        ax = plt.subplot(3, 4, i + 1)
        
        if month_num in monthly_trends and not monthly_trends[month_num].empty:
            trends_df = monthly_trends[month_num].copy()
            trends_df = trends_df.sort_values('year')
            
            # Extract data for plotting
            years = trends_df['year'].values
            
            # Downscaled data
            medians_down = trends_df['median_down'].values
            q25_down = trends_df['q25_down'].values
            q75_down = trends_df['q75_down'].values
            
            # Original GRACE data
            medians_orig = trends_df['median_orig'].values
            q25_orig = trends_df['q25_orig'].values
            q75_orig = trends_df['q75_orig'].values
            
            # Plot original GRACE (background)
            ax.fill_between(years, q25_orig, q75_orig, alpha=0.2, color='red', 
                           label='Original GRACE (25th-75th %ile)')
            ax.plot(years, medians_orig, color='darkred', linewidth=2, 
                   marker='s', markersize=3, label='Original GRACE (median)')
            
            # Plot downscaled (foreground)
            ax.fill_between(years, q25_down, q75_down, alpha=0.3, color='blue', 
                           label='Downscaled (25th-75th %ile)')
            ax.plot(years, medians_down, color='darkblue', linewidth=2.5, 
                   marker='o', markersize=4, label='Downscaled (median)')
            
            # Formatting
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
            
            # Format x-axis - only show labels on bottom row
            ax.set_xticks(np.arange(years.min(), years.max() + 1, 5))
            if i >= 8:  # Bottom row only
                ax.set_xticklabels(np.arange(years.min(), years.max() + 1, 5), fontsize=10)
                ax.set_xlabel('Year', fontsize=11)
            else:
                ax.set_xticklabels([])  # Remove x-axis labels for top/middle rows
            
            # Format y-axis
            ax.tick_params(axis='y', labelsize=10)
            if i % 4 == 0:  # Only leftmost column
                ax.set_ylabel('TWS Anomaly (cm)', fontsize=11)
            else:
                ax.set_yticklabels([])
            
            # Add validation metrics text
            r2 = trends_df['r2'].iloc[0] if 'r2' in trends_df.columns else np.nan
            rmse = trends_df['rmse'].iloc[0] if 'rmse' in trends_df.columns else np.nan
            
            if not np.isnan(r2) and not np.isnan(rmse):
                stats_text = f'R² = {r2:.3f}\nRMSE = {rmse:.1f} cm\nn = {len(years)} years'
            else:
                stats_text = f'n = {len(years)} years'
                
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
            
        else:
            # No data for this month
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                   fontsize=12, ha='center', va='center')
            ax.set_ylim(y_min, y_max)
            # Still need to remove x-labels for non-bottom rows
            if i < 8:
                ax.set_xticks([])
        
        # Add month label
        ax.text(0.98, 0.02, month_names[i], 
               transform=ax.transAxes, 
               fontsize=12, fontweight='bold', ha='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
                        alpha=0.9, edgecolor='none'))
        
        # Grid
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Add legend (only once, in top-right)
    if 1 in monthly_trends and not monthly_trends[1].empty:
        # Get handles and labels from the first subplot
        handles, labels = plt.subplot(3, 4, 1).get_legend_handles_labels()
        # Only show the median lines in legend for clarity
        selected_handles = [handles[1], handles[3]]  # Original median, Downscaled median
        selected_labels = ['Original GRACE', 'Downscaled GRACE']
        fig.legend(selected_handles, selected_labels, loc='upper right', 
                  bbox_to_anchor=(0.98, 0.98), frameon=True, 
                  fancybox=True, shadow=True, fontsize=11)
    
    # Add main title
    plt.suptitle('Temporal Trends and Validation: Downscaled vs Original GRACE\n' +
                'Mississippi River Basin - Monthly Comparisons (2003-2024)', 
                fontsize=16, fontweight='bold', y=0.96)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.96, 
                       hspace=0.25, wspace=0.15)
    
    # Save figure
    output_path = Path("figures_coarse_to_fine/temporal_trends_tws_mrb.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', transparent=False, format='png')
    print(f"💾 Saved temporal trends figure: {output_path}")
    
    # Save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white',
                edgecolor='none', transparent=False, format='pdf')
    print(f"💾 Saved temporal trends PDF: {pdf_path}")
    
    # Print validation summary
    print("\n📊 VALIDATION SUMMARY (Downscaled vs Original GRACE):")
    print("Month    | R²     | RMSE (cm) | Years | Downscaled μ | Original μ")
    print("-" * 65)
    
    overall_r2 = []
    overall_rmse = []
    
    for month_num, trends_df in monthly_trends.items():
        if not trends_df.empty and 'r2' in trends_df.columns:
            month_name = month_names[month_num - 1]
            r2 = trends_df['r2'].iloc[0]
            rmse = trends_df['rmse'].iloc[0]
            n_years = len(trends_df)
            mean_down = trends_df['median_down'].mean()
            mean_orig = trends_df['median_orig'].mean()
            
            if not np.isnan(r2) and not np.isnan(rmse):
                overall_r2.append(r2)
                overall_rmse.append(rmse)
                
            print(f"{month_name:8s} | {r2:5.3f}  | {rmse:8.2f}  | {n_years:4d}  | {mean_down:+9.1f}   | {mean_orig:+8.1f}")
    
    if overall_r2:
        mean_r2 = np.mean(overall_r2)
        mean_rmse = np.mean(overall_rmse)
        print("-" * 65)
        print(f"{'OVERALL':8s} | {mean_r2:5.3f}  | {mean_rmse:8.2f}  | {'':4s}  | {'':9s}   | {'':8s}")
        print(f"\n🎯 Overall Performance: R² = {mean_r2:.3f}, RMSE = {mean_rmse:.2f} cm")
    
    return output_path

def main():
    """Main function."""
    print("📈 Creating Temporal Trends TWS Visualization...")
    output_path = create_temporal_trends_plot()
    print(f"✅ Temporal trends visualization complete!")
    print(f"📄 Ready for research paper: {output_path}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())