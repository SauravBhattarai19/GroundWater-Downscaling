#!/usr/bin/env python3
"""
STL Gap Filling Results Visualization

This script creates a comprehensive figure showing:
1. Raw GRACE data with gaps clearly visible
2. STL gap-filled result (complete time series)
3. Comparison showing what STL accomplished

Designed for A4 manuscript quality with large, readable text.

Usage:
    python src_new_approach/scripts/01_see_stl_result.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project path
sys.path.insert(0, 'src_new_approach')

# Set up manuscript-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# A4 Manuscript-quality settings - LARGE TEXT for publication
plt.rcParams.update({
    'figure.figsize': (16, 20),      # Large figure for A4
    'font.size': 20,                 # Much larger base font
    'axes.labelsize': 26,            # Large axis labels
    'axes.titlesize': 30,            # Large titles
    'xtick.labelsize': 22,           # Large tick labels
    'ytick.labelsize': 22,           # Large tick labels
    'legend.fontsize': 24,           # Large legend
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'lines.linewidth': 4,            # Thick lines for visibility
    'axes.linewidth': 2,             # Thick axes
    'grid.linewidth': 1.5,           # Thick grid
})


def load_data():
    """Load both original raw GRACE data and STL gap-filled results."""
    print("📊 Loading original raw GRACE data...")
    
    # Import the gap filler to load original data
    from grace_gap_handler import GRACEGapFiller
    from utils_downscaling import load_config
    
    config = load_config('src_new_approach/config_coarse_to_fine.yaml')
    filler = GRACEGapFiller(config)
    
    # Load original data with gaps
    grace_path = "data/raw/grace"
    raw_ds = filler.load_grace_data(grace_path)
    
    # Load STL filled results
    filled_path = "data/processed_coarse_to_fine/grace_filled_stl.nc"
    if not Path(filled_path).exists():
        raise FileNotFoundError(f"STL results not found at {filled_path}")
    
    print(f"📊 Loading STL results from: {filled_path}")
    filled_ds = xr.open_dataset(filled_path)
    
    # Get basic info
    print(f"   Raw data shape: {raw_ds.tws_anomaly.shape}")
    print(f"   Filled data shape: {filled_ds.tws_anomaly.shape}")
    print(f"   Time range: {filled_ds.time.values[0]} to {filled_ds.time.values[-1]}")
    
    return raw_ds, filled_ds


def calculate_spatial_statistics(ds_raw, ds_filled):
    """Calculate spatial statistics for raw and filled data."""
    print("📈 Calculating spatial statistics...")
    
    times = pd.DatetimeIndex(ds_filled.time.values)
    
    # Raw data stats (with gaps)
    raw_stats = {'time': times, 'mean': [], 'n_valid': [], 'n_missing': []}
    
    # Filled data stats (complete)
    filled_stats = {'time': times, 'mean': [], 'std': [], 'median': [], 'q25': [], 'q75': []}
    
    for t in range(len(times)):
        # Raw data (with gaps)
        raw_data_t = ds_raw.tws_anomaly.isel(time=t).values.flatten()
        raw_valid = raw_data_t[~np.isnan(raw_data_t)]
        raw_missing = np.sum(np.isnan(raw_data_t))
        
        raw_stats['mean'].append(np.mean(raw_valid) if len(raw_valid) > 0 else np.nan)
        raw_stats['n_valid'].append(len(raw_valid))
        raw_stats['n_missing'].append(raw_missing)
        
        # Filled data (complete)
        filled_data_t = ds_filled.tws_anomaly.isel(time=t).values.flatten()
        filled_valid = filled_data_t[~np.isnan(filled_data_t)]
        
        if len(filled_valid) > 0:
            filled_stats['mean'].append(np.mean(filled_valid))
            filled_stats['std'].append(np.std(filled_valid))
            filled_stats['median'].append(np.median(filled_valid))
            filled_stats['q25'].append(np.percentile(filled_valid, 25))
            filled_stats['q75'].append(np.percentile(filled_valid, 75))
        else:
            for key in ['mean', 'std', 'median', 'q25', 'q75']:
                filled_stats[key].append(np.nan)
    
    raw_df = pd.DataFrame(raw_stats)
    filled_df = pd.DataFrame(filled_stats)
    
    print(f"   Calculated statistics for {len(filled_df)} time steps")
    
    return raw_df, filled_df


def create_comprehensive_stl_figure(raw_df, filled_df, output_dir):
    """Create single comprehensive figure showing raw data, STL result, and comparison."""
    print("🎨 Creating comprehensive STL comparison figure...")
    
    # Create figure with 3 subplots - manuscript quality A4 size
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 24), sharex=True)
    
    times = filled_df['time']
    
    # Panel A: Raw GRACE data with gaps clearly shown
    ax1.plot(times, raw_df['mean'], color='darkred', linewidth=5, 
             label='Raw GRACE (with gaps)', alpha=0.8)
    
    # Highlight missing data periods with vertical spans
    missing_periods = raw_df['n_missing'] > 0
    if np.any(missing_periods):
        # Find continuous missing periods for better visualization
        missing_starts = []
        missing_ends = []
        in_gap = False
        
        for i, is_missing in enumerate(missing_periods):
            if is_missing and not in_gap:
                missing_starts.append(i)
                in_gap = True
            elif not is_missing and in_gap:
                missing_ends.append(i-1)
                in_gap = False
        if in_gap:  # Handle case where data ends with missing values
            missing_ends.append(len(missing_periods)-1)
        
        # Add gap indicators
        for start, end in zip(missing_starts, missing_ends):
            ax1.axvspan(times[start], times[end], alpha=0.4, color='red', 
                       label='Data gaps' if start == missing_starts[0] else "")
    
    ax1.set_ylabel('TWS Anomaly (cm)', fontsize=26)
    ax1.set_title('(A) Raw GRACE Data with Gaps', fontsize=32, fontweight='bold', pad=30)
    ax1.legend(loc='upper left', fontsize=24)
    ax1.grid(True, alpha=0.4)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=3)
    
    # Panel B: STL gap-filled result
    mean_vals = filled_df['mean']
    q25_vals = filled_df['q25']
    q75_vals = filled_df['q75']
    
    ax2.plot(times, mean_vals, color='darkblue', linewidth=5, label='STL Gap-filled')
    ax2.fill_between(times, q25_vals, q75_vals, alpha=0.3, color='lightblue', 
                     label='Interquartile Range')
    
    ax2.set_ylabel('TWS Anomaly (cm)', fontsize=26)
    ax2.set_title('(B) STL Gap-Filled Result (Complete Time Series)', fontsize=32, fontweight='bold', pad=30)
    ax2.legend(loc='upper left', fontsize=24)
    ax2.grid(True, alpha=0.4)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=3)
    
    # Panel C: Comparison - what STL accomplished
    # Show both raw and filled on same plot
    ax3.plot(times, raw_df['mean'], color='darkred', linewidth=4, 
             label='Original (with gaps)', alpha=0.7, linestyle='-')
    ax3.plot(times, filled_df['mean'], color='darkblue', linewidth=4, 
             label='STL Filled', alpha=0.9, linestyle='-')
    
    # Highlight filled portions
    filled_mask = raw_df['n_missing'] > 0
    if np.any(filled_mask):
        filled_times = times[filled_mask]
        filled_values = np.array(filled_df['mean'])[filled_mask]
        ax3.scatter(filled_times, filled_values, color='green', s=120, 
                   label='STL-filled values', alpha=0.8, zorder=5)
    
    ax3.set_ylabel('TWS Anomaly (cm)', fontsize=26)
    ax3.set_xlabel('Time', fontsize=26)
    ax3.set_title('(C) Comparison: Original vs. STL Gap-Filled', fontsize=32, fontweight='bold', pad=30)
    ax3.legend(loc='upper left', fontsize=24)
    ax3.grid(True, alpha=0.4)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=3)
    
    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', rotation=45, labelsize=22)
        ax.tick_params(axis='y', labelsize=22)
        # Increase spine thickness
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    
    plt.tight_layout()
    
    # Save the comprehensive figure
    output_path = output_dir / "stl_comprehensive_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved comprehensive figure: {output_path}")
    
    plt.close()
    
    return output_path


def create_summary_statistics(raw_df, filled_df, output_dir):
    """Create summary statistics and quality assessment."""
    print("📋 Creating summary statistics...")
    
    # Calculate key metrics
    total_missing_original = raw_df['n_missing'].sum()
    total_missing_filled = 0  # Should be zero after STL
    times = filled_df['time']
    
    # Quality metrics
    mean_abs_tws = np.abs(filled_df['mean']).mean()
    seasonal_range = filled_df['mean'].max() - filled_df['mean'].min()
    temporal_corr = filled_df['mean'].autocorr(lag=1) if len(filled_df['mean']) > 1 else np.nan
    
    summary_text = f"""GRACE TWS Anomaly - STL Gap Filling Results
=====================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Time Period              : {times.iloc[0].strftime('%Y-%m')} to {times.iloc[-1].strftime('%Y-%m')}
Total Months             : {len(times)}
Original Missing Values  : {total_missing_original:,}
Final Missing Values     : {total_missing_filled}
Gap Filling Success      : {100 * (total_missing_original - total_missing_filled) / total_missing_original:.1f}%

Quality Assessment:
- Mean Absolute TWS      : {mean_abs_tws:.2f} cm
- Seasonal Range         : {seasonal_range:.2f} cm
- Temporal Correlation   : {temporal_corr:.3f}
- Spatial Std (mean)     : {filled_df['std'].mean():.2f} cm

Method: STL (Seasonal-Trend-Loess) Decomposition
- Linear interpolation before STL decomposition
- Zero-mean residual assumption for reconstruction
- Complete gap filling using trend + seasonal components
"""
    
    # Save summary
    summary_path = output_dir / "stl_summary_statistics.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(f"   ✅ Saved: {summary_path}")
    
    return summary_path


def main():
    """Main execution function."""
    print("🔬 STL Gap Filling Results Analysis")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path("figures_coarse_to_fine/01_stl_result")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load both raw and filled data
        raw_ds, filled_ds = load_data()
        
        # Calculate spatial statistics for both datasets
        raw_df, filled_df = calculate_spatial_statistics(raw_ds, filled_ds)
        
        # Create the comprehensive comparison figure
        comprehensive_fig = create_comprehensive_stl_figure(raw_df, filled_df, output_dir)
        
        # Create summary statistics
        summary_file = create_summary_statistics(raw_df, filled_df, output_dir)
        
        print("\n" + "=" * 60)
        print("✅ STL Results Analysis Complete!")
        print(f"📁 Outputs saved to: {output_dir}")
        print("\nGenerated files:")
        for file in output_dir.glob("*"):
            print(f"   - {file.name}")
        
        # Quick quality assessment
        mean_abs_tws = np.abs(filled_df['mean']).mean()
        seasonal_range = filled_df['mean'].max() - filled_df['mean'].min()
        
        print(f"\n🎯 Quick Quality Assessment:")
        print(f"   Mean absolute TWS: {mean_abs_tws:.2f} cm")
        print(f"   Seasonal range: {seasonal_range:.2f} cm") 
        print(f"   Spatial variability: {filled_df['std'].mean():.2f} ± {filled_df['std'].std():.2f} cm")
        
        if mean_abs_tws < 10 and seasonal_range > 1:
            print("   ✅ Results look reasonable for GRACE data!")
        else:
            print("   ⚠️ Results may need review - check for anomalies")
        
        print(f"\n📖 Main output for manuscript: {comprehensive_fig.name}")
        print("   This figure shows the complete STL gap filling story!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())