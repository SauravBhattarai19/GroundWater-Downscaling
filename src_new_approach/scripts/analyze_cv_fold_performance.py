#!/usr/bin/env python3
"""
Analyze CV Fold Performance Spatially and Temporally

Shows exactly WHERE (spatially and temporally) each CV fold performs well or poorly.
This addresses the critical question: with mean R² = 0.107 ± 0.921, where are the 
good folds vs bad folds located?

MODIFICATIONS APPLIED:
- OUTLIER REMOVAL: Uses IQR method (factor=1.5) to remove extreme outliers per model
- DOUBLED FONT SIZES: All font sizes doubled again for better presentation visibility
- LARGER FIGURES: Figure sizes increased to accommodate larger fonts

Creates comprehensive visualizations showing:
1. Spatial performance map for each fold
2. Temporal performance timeline 
3. Model comparison across all folds
4. Performance pattern analysis
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.insert(0, 'src_new_approach')

# Import our CV implementation
from spatiotemporal_cv import BlockedSpatiotemporalCV, prepare_metadata_for_cv

# Set presentation-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# MUCH LARGER text for presentation - doubled AGAIN per user request
plt.rcParams.update({
    'font.size': 40,                # Doubled again from 20
    'axes.labelsize': 48,           # Doubled again from 24
    'axes.titlesize': 56,           # Doubled again from 28
    'xtick.labelsize': 40,          # Doubled again from 20
    'ytick.labelsize': 40,          # Doubled again from 20
    'legend.fontsize': 36,          # Doubled again from 18
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'lines.linewidth': 4,           # Even thicker lines
    'axes.linewidth': 3,            # Even thicker axes
    'grid.linewidth': 2,            # Even thicker grid
})


def remove_outliers(data, method='iqr', factor=1.5):
    """
    Remove outliers from data using IQR method.
    
    Parameters:
    -----------
    data : pandas.Series or numpy.array
        Data to remove outliers from
    method : str
        Method to use: 'iqr' (default) or 'zscore'
    factor : float
        Factor for outlier detection (1.5 for IQR, 3 for z-score)
    
    Returns:
    --------
    pandas.Series or numpy.array
        Data with outliers removed
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        mask = z_scores < factor
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return data[mask]


def filter_outliers_from_results(enhanced_results, outlier_method='iqr', outlier_factor=1.5):
    """
    Filter outliers from CV results for each model separately.
    
    Parameters:
    -----------
    enhanced_results : pandas.DataFrame
        CV results with spatial/temporal mapping
    outlier_method : str
        Method for outlier detection ('iqr' or 'zscore')
    outlier_factor : float
        Factor for outlier threshold
    
    Returns:
    --------
    pandas.DataFrame
        Filtered results without outliers
    """
    print(f"🧹 Removing outliers using {outlier_method} method with factor {outlier_factor}")
    
    filtered_results = []
    original_counts = []
    filtered_counts = []
    
    for model in enhanced_results['model_name'].unique():
        model_data = enhanced_results[enhanced_results['model_name'] == model].copy()
        original_count = len(model_data)
        
        # Remove outliers based on R² values
        r2_no_outliers = remove_outliers(model_data['r2'], method=outlier_method, factor=outlier_factor)
        
        # Filter the dataframe to keep only non-outlier rows
        filtered_model_data = model_data[model_data['r2'].isin(r2_no_outliers)]
        filtered_count = len(filtered_model_data)
        
        filtered_results.append(filtered_model_data)
        original_counts.append(original_count)
        filtered_counts.append(filtered_count)
        
        removed_count = original_count - filtered_count
        print(f"   {model.upper()}: {original_count} → {filtered_count} folds ({removed_count} outliers removed)")
    
    combined_filtered = pd.concat(filtered_results, ignore_index=True)
    
    total_original = sum(original_counts)
    total_filtered = len(combined_filtered)
    total_removed = total_original - total_filtered
    
    print(f"   TOTAL: {total_original} → {total_filtered} folds ({total_removed} outliers removed)")
    
    return combined_filtered


def load_cv_results():
    """Load all CV results from different models."""
    print("📊 Loading CV results from all models...")
    
    models_dir = Path("models_coarse_to_fine")
    cv_files = list(models_dir.glob("*_cv_results.csv"))
    
    all_results = []
    for cv_file in cv_files:
        model_name = cv_file.stem.replace("_cv_results", "")
        print(f"   Loading {model_name} results from: {cv_file}")
        
        df = pd.read_csv(cv_file)
        if 'model_name' not in df.columns:
            df['model_name'] = model_name
            
        all_results.append(df)
        
        # Show some stats
        print(f"     • {model_name}: R² = {df['r2'].mean():.3f} ± {df['r2'].std():.3f} "
              f"(range: {df['r2'].min():.3f} to {df['r2'].max():.3f})")
    
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"\n   Total CV results: {len(combined_df)} fold results across {len(cv_files)} models")
    
    return combined_df


def load_spatial_temporal_metadata():
    """Load spatial and temporal metadata for fold mapping."""
    print("🗺️ Loading spatial and temporal metadata...")
    
    # Load feature stack to get spatial structure
    feature_path = "data/processed_coarse_to_fine/feature_stack_55km.nc"
    if not Path(feature_path).exists():
        raise FileNotFoundError(f"Feature stack not found at {feature_path}")
    
    ds = xr.open_dataset(feature_path)
    
    # Get spatial and temporal dimensions
    lat_values = ds.lat.values
    lon_values = ds.lon.values
    time_values = pd.to_datetime(ds.time.values)
    
    print(f"   Spatial grid: {len(lat_values)} lat × {len(lon_values)} lon")
    print(f"   Time range: {time_values[0]} to {time_values[-1]} ({len(time_values)} time steps)")
    
    # Create sample data structure for CV mapping
    n_times = len(time_values)
    n_spatial = len(lat_values) * len(lon_values)
    n_samples = n_times * n_spatial
    
    X_dummy = np.random.randn(n_samples, 10)
    y_dummy = np.random.randn(n_samples)
    
    common_dates = [t.strftime('%Y-%m') for t in time_values]
    spatial_shape = (len(lat_values), len(lon_values))
    
    metadata = prepare_metadata_for_cv(
        X_dummy, y_dummy, common_dates, spatial_shape, feature_path
    )
    
    metadata['lat_values'] = lat_values
    metadata['lon_values'] = lon_values  
    metadata['time_values'] = time_values
    
    return metadata


def map_folds_to_space_time(cv_results, metadata):
    """Map CV fold results to spatial and temporal locations."""
    print("🔗 Mapping CV folds to spatial and temporal locations...")
    
    # Initialize CV splitter with same configuration 
    cv_splitter = BlockedSpatiotemporalCV(
        n_spatial_blocks=5,
        n_temporal_blocks=4, 
        spatial_buffer_deg=0.5,
        temporal_buffer_months=1,
        random_state=42
    )
    
    # Create dummy data for fold mapping
    n_samples = len(metadata['spatial_coords'])
    X_dummy = np.random.randn(n_samples, 10)
    y_dummy = np.random.randn(n_samples)
    
    # Get spatial and temporal blocks
    spatial_coords = metadata['spatial_coords']
    temporal_indices = metadata['temporal_indices']
    spatial_blocks = cv_splitter.create_spatial_blocks(spatial_coords)
    temporal_blocks = cv_splitter.create_temporal_blocks(temporal_indices)
    
    # Map each fold to its spatial and temporal block
    fold_mapping = []
    fold_id = 0
    
    for spatial_test_block in range(5):  # n_spatial_blocks
        for temporal_test_block in range(4):  # n_temporal_blocks
            fold_id += 1
            
            # Find test samples for this fold
            spatial_test_mask = (spatial_blocks == spatial_test_block)
            temporal_test_mask = np.isin(
                temporal_indices, 
                temporal_blocks[temporal_test_block]
            )
            test_mask = spatial_test_mask & temporal_test_mask
            test_idx = np.where(test_mask)[0]
            
            if len(test_idx) > 0:
                # Get spatial region (cluster center)
                test_center = cv_splitter.cluster_centers_[spatial_test_block]
                
                # Get temporal period
                test_times = temporal_blocks[temporal_test_block]
                if len(test_times) > 0:
                    time_start = metadata['time_values'][test_times.min()]
                    time_end = metadata['time_values'][test_times.max()]
                else:
                    time_start = time_end = None
                
                fold_mapping.append({
                    'fold': fold_id,
                    'spatial_block': spatial_test_block,
                    'temporal_block': temporal_test_block,
                    'test_center_lat': test_center[0],
                    'test_center_lon': test_center[1],
                    'time_start': time_start,
                    'time_end': time_end,
                    'n_test_samples': len(test_idx)
                })
    
    fold_mapping_df = pd.DataFrame(fold_mapping)
    print(f"   Mapped {len(fold_mapping_df)} folds to spatial/temporal locations")
    
    # Merge with CV results
    enhanced_results = []
    for model in cv_results['model_name'].unique():
        model_results = cv_results[cv_results['model_name'] == model].copy()
        model_results = model_results.merge(fold_mapping_df, on='fold', how='left')
        enhanced_results.append(model_results)
    
    enhanced_df = pd.concat(enhanced_results, ignore_index=True)
    
    return enhanced_df, fold_mapping_df


def create_spatial_performance_map(enhanced_results, output_dir):
    """Create spatial map showing performance of each fold by location."""
    print("🗺️ Creating spatial performance maps...")
    
    models = enhanced_results['model_name'].unique()
    n_models = len(models)
    
    # Create figure with subplots for each model - MUCH LARGER figure for doubled fonts
    fig, axes = plt.subplots(2, 2, figsize=(36, 28))  # Much increased figure size for doubled fonts
    axes = axes.flatten()
    
    # Get spatial bounds
    lat_min, lat_max = enhanced_results['test_center_lat'].min() - 1, enhanced_results['test_center_lat'].max() + 1
    lon_min, lon_max = enhanced_results['test_center_lon'].min() - 1, enhanced_results['test_center_lon'].max() + 1
    
    print(f"   Found {len(enhanced_results['spatial_block'].unique())} unique spatial blocks: {sorted(enhanced_results['spatial_block'].unique())}")
    
    for i, model in enumerate(models):
        ax = axes[i]
        model_data = enhanced_results[enhanced_results['model_name'] == model]
        
        # Group by spatial block to ensure all 5 blocks are represented
        spatial_blocks = model_data['spatial_block'].unique()
        print(f"   {model}: spatial blocks {sorted(spatial_blocks)}")
        
        # Create scatter plot colored by R² performance - MUCH LARGER markers
        scatter = ax.scatter(
            model_data['test_center_lon'], 
            model_data['test_center_lat'],
            c=model_data['r2'], 
            s=800,  # Much larger markers (was 300)
            cmap='RdYlGn',  # Red (bad) to Green (good)
            vmin=-3, vmax=1,  # Fixed scale for comparison
            alpha=0.8,
            edgecolors='black',
            linewidth=3  # Thicker edges
        )
        
        # Add spatial block numbers as text - LARGER text
        for _, row in model_data.iterrows():
            ax.text(row['test_center_lon'], row['test_center_lat'], 
                   f"{int(row['spatial_block'])}", ha='center', va='center',
                   fontweight='bold', fontsize=18, color='white',  # Much larger text
                   bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7))
        
        ax.set_xlim(lon_max, lon_min)  # Reversed for longitude (west is negative)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel('Longitude (°W)', fontweight='bold')
        ax.set_ylabel('Latitude (°N)', fontweight='bold')
        ax.set_title(f'{model.upper()} - Spatial Performance Map', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar with larger text
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('R² Score', fontweight='bold', fontsize=24)
        cbar.ax.tick_params(labelsize=18)
        
        # Add performance statistics - LARGER text
        stats_text = f"""Performance Stats:
Mean R²: {model_data['r2'].mean():.3f}
Std R²: {model_data['r2'].std():.3f}
Best Block: {model_data.loc[model_data['r2'].idxmax(), 'spatial_block']:.0f} (R²={model_data['r2'].max():.3f})
Worst Block: {model_data.loc[model_data['r2'].idxmin(), 'spatial_block']:.0f} (R²={model_data['r2'].min():.3f})"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=16, fontfamily='monospace',  # Larger font
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Hide unused subplot
    if n_models < 4:
        axes[-1].axis('off')
    
    plt.tight_layout()
    
    # Save the spatial map
    output_path = output_dir / "cv_spatial_performance_map.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved spatial performance map: {output_path}")
    
    plt.close()
    return output_path


def create_temporal_performance_timeline(enhanced_results, output_dir):
    """Create temporal timeline showing performance by time period."""
    print("📅 Creating temporal performance timeline...")
    
    models = enhanced_results['model_name'].unique()
    
    # Create MUCH LARGER figure to avoid text overlapping with doubled fonts
    fig, axes = plt.subplots(len(models), 1, figsize=(32, 10 * len(models)))  # Much increased height for doubled fonts
    if len(models) == 1:
        axes = [axes]
    
    for i, model in enumerate(models):
        ax = axes[i]
        model_data = enhanced_results[enhanced_results['model_name'] == model]
        
        # Sort by spatial block first, then temporal block to avoid overlaps
        model_data = model_data.sort_values(['spatial_block', 'temporal_block'])
        
        # Create timeline with better spacing - use spatial_block for y-position
        for _, row in model_data.iterrows():
            if pd.notna(row['time_start']) and pd.notna(row['time_end']):
                # Color based on R² performance
                if row['r2'] >= 0.5:
                    color = 'darkgreen'
                    alpha = 0.8
                elif row['r2'] >= 0:
                    color = 'orange' 
                    alpha = 0.8
                else:
                    color = 'darkred'
                    alpha = 0.8
                
                # Use spatial_block for y-position to avoid overlap
                y_pos = row['spatial_block'] + 0.2 * row['temporal_block']  # Slight offset for temporal blocks
                
                # Create bar for time period
                ax.barh(
                    y_pos, 
                    (row['time_end'] - row['time_start']).days,
                    left=row['time_start'],
                    height=0.15,  # Thinner bars to fit more
                    color=color,
                    alpha=alpha,
                    edgecolor='black',
                    linewidth=1
                )
                
                # Add cleaner text - just fold number and R²
                mid_time = row['time_start'] + (row['time_end'] - row['time_start']) / 2
                ax.text(mid_time, y_pos, 
                       f"F{int(row['fold'])}: {row['r2']:.2f}",  # Shorter text format
                       ha='center', va='center', fontsize=12, fontweight='bold',
                       color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        ax.set_ylabel(f'Spatial Block', fontweight='bold')
        ax.set_title(f'{model.upper()} - Temporal Performance Timeline', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Set y-ticks for spatial blocks
        ax.set_yticks(range(5))
        ax.set_yticklabels([f'Block {i}' for i in range(5)])
        ax.set_ylim(-0.5, 4.5)
        
        # Format x-axis with better date labels
        ax.tick_params(axis='x', rotation=45)
        
        # Add temporal block legend
        for tb in range(4):
            ax.text(0.02, 0.98 - tb*0.05, f'T{tb}: Temporal Block {tb}', 
                   transform=ax.transAxes, fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    axes[-1].set_xlabel('Time Period', fontweight='bold')
    
    # Add performance legend - larger
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='darkgreen', alpha=0.8, label='Good (R² ≥ 0.5)'),
        plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.8, label='Fair (0 ≤ R² < 0.5)'), 
        plt.Rectangle((0,0),1,1, facecolor='darkred', alpha=0.8, label='Poor (R² < 0)')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=18)
    
    plt.tight_layout()
    
    # Save the timeline
    output_path = output_dir / "cv_temporal_performance_timeline.png" 
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved temporal performance timeline: {output_path}")
    
    plt.close()
    return output_path


def create_performance_analysis(enhanced_results, output_dir):
    """Create comprehensive performance analysis."""
    print("📊 Creating performance analysis...")
    
    models = enhanced_results['model_name'].unique()
    
    # Create MUCH LARGER figure with improved layout to accommodate doubled font sizes
    fig, axes = plt.subplots(2, 2, figsize=(32, 24))  # Even larger figure for doubled fonts
    
    # Panel 1: Performance distribution by model - IMPROVED
    ax1 = axes[0, 0]
    
    # Box plot of R² by model
    box_data = [enhanced_results[enhanced_results['model_name'] == model]['r2'].values 
                for model in models]
    
    box_plot = ax1.boxplot(box_data, labels=[m.upper() for m in models], patch_artist=True)
    
    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors[:len(models)]):
        patch.set_facecolor(color)
        patch.set_linewidth(2)
    
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_title('(A) Performance Distribution by Model', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=3)
    
    # Panel 2: Spatial block performance - FIXED AXIS LABELS
    ax2 = axes[0, 1]
    
    spatial_perf = enhanced_results.groupby(['model_name', 'spatial_block'])['r2'].mean().reset_index()
    
    for model in models:
        model_spatial = spatial_perf[spatial_perf['model_name'] == model]
        ax2.plot(model_spatial['spatial_block'], model_spatial['r2'], 
                'o-', label=model.upper(), linewidth=3, markersize=10)
    
    # FIXED: Set proper integer ticks
    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.set_xticklabels(['0', '1', '2', '3', '4'])
    ax2.set_xlabel('Spatial Block', fontweight='bold')
    ax2.set_ylabel('Mean R² Score', fontweight='bold')
    ax2.set_title('(B) Performance by Spatial Block', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=16)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=3)
    
    # Panel 3: Temporal block performance - FIXED AXIS LABELS
    ax3 = axes[1, 0]
    
    temporal_perf = enhanced_results.groupby(['model_name', 'temporal_block'])['r2'].mean().reset_index()
    
    for model in models:
        model_temporal = temporal_perf[temporal_perf['model_name'] == model]
        ax3.plot(model_temporal['temporal_block'], model_temporal['r2'],
                'o-', label=model.upper(), linewidth=3, markersize=10)
    
    # FIXED: Set proper integer ticks
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xticklabels(['0', '1', '2', '3'])
    ax3.set_xlabel('Temporal Block', fontweight='bold')
    ax3.set_ylabel('Mean R² Score', fontweight='bold')
    ax3.set_title('(C) Performance by Temporal Block', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=16)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=3)
    
    # Panel 4: SPATIOTEMPORAL HEATMAP (replacing statistics table)
    ax4 = axes[1, 1]
    
    # Create spatiotemporal performance matrix for best model (XGB)
    best_model = 'xgb'  # Based on analysis, XGB performs slightly better
    model_data = enhanced_results[enhanced_results['model_name'] == best_model]
    
    # Create matrix: rows = spatial blocks, columns = temporal blocks
    perf_matrix = np.full((5, 4), np.nan)
    
    for _, row in model_data.iterrows():
        spatial_idx = int(row['spatial_block'])
        temporal_idx = int(row['temporal_block'])
        perf_matrix[spatial_idx, temporal_idx] = row['r2']
    
    # Create heatmap
    im = ax4.imshow(perf_matrix, cmap='RdYlGn', aspect='auto', vmin=-3, vmax=1)
    
    # Add text annotations
    for i in range(5):
        for j in range(4):
            if not np.isnan(perf_matrix[i, j]):
                text_color = 'white' if perf_matrix[i, j] < -1 else 'black'
                ax4.text(j, i, f'{perf_matrix[i, j]:.2f}', 
                        ha='center', va='center', fontweight='bold', 
                        fontsize=14, color=text_color)
    
    ax4.set_xticks([0, 1, 2, 3])
    ax4.set_xticklabels(['T0', 'T1', 'T2', 'T3'])
    ax4.set_yticks([0, 1, 2, 3, 4])
    ax4.set_yticklabels(['S0', 'S1', 'S2', 'S3', 'S4'])
    ax4.set_xlabel('Temporal Block', fontweight='bold')
    ax4.set_ylabel('Spatial Block', fontweight='bold')
    ax4.set_title(f'(D) Spatiotemporal Heatmap ({best_model.upper()})', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('R² Score', fontweight='bold', fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    
    # Save the analysis
    output_path = output_dir / "cv_performance_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved performance analysis: {output_path}")
    
    plt.close()
    return output_path


def create_combined_spatiotemporal_heatmap(enhanced_results, output_dir):
    """Create combined spatiotemporal heatmap for all models."""
    print("🔥 Creating combined spatiotemporal heatmap...")
    
    models = enhanced_results['model_name'].unique()
    
    # Create figure with subplots for each model - larger for doubled fonts
    fig, axes = plt.subplots(2, 2, figsize=(32, 24))
    axes = axes.flatten()
    
    for i, model in enumerate(models):
        ax = axes[i]
        model_data = enhanced_results[enhanced_results['model_name'] == model]
        
        # Create matrix: rows = spatial blocks, columns = temporal blocks
        perf_matrix = np.full((5, 4), np.nan)
        fold_matrix = np.full((5, 4), np.nan)
        
        for _, row in model_data.iterrows():
            spatial_idx = int(row['spatial_block'])
            temporal_idx = int(row['temporal_block'])
            perf_matrix[spatial_idx, temporal_idx] = row['r2']
            fold_matrix[spatial_idx, temporal_idx] = row['fold']
        
        # Create heatmap
        im = ax.imshow(perf_matrix, cmap='RdYlGn', aspect='auto', vmin=-3, vmax=1)
        
        # Add text annotations with fold numbers and R² values
        for i_sp in range(5):
            for j_temp in range(4):
                if not np.isnan(perf_matrix[i_sp, j_temp]):
                    text_color = 'white' if perf_matrix[i_sp, j_temp] < -1 else 'black'
                    fold_num = int(fold_matrix[i_sp, j_temp])
                    r2_val = perf_matrix[i_sp, j_temp]
                    
                    ax.text(j_temp, i_sp, f'F{fold_num}\n{r2_val:.2f}', 
                           ha='center', va='center', fontweight='bold', 
                           fontsize=12, color=text_color)
        
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['T0\n(2003-2008)', 'T1\n(2008-2013)', 'T2\n(2013-2019)', 'T3\n(2019-2024)'])
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(['S0\n(Central)', 'S1\n(N.Lakes)', 'S2\n(SW)', 'S3\n(N.Plains)', 'S4\n(SE)'])
        ax.set_xlabel('Temporal Block', fontweight='bold')
        ax.set_ylabel('Spatial Block', fontweight='bold')
        ax.set_title(f'{model.upper()} - Spatiotemporal Performance', fontweight='bold')
        
        # Add colorbar
        if i == len(models) - 1:  # Only add colorbar to last subplot
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('R² Score', fontweight='bold', fontsize=20)
            cbar.ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    
    # Save the heatmap
    output_path = output_dir / "cv_spatiotemporal_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved spatiotemporal heatmap: {output_path}")
    
    plt.close()
    return output_path


def main():
    """Main execution function."""
    print("🔬 CV Fold Performance Analysis")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path("figures_coarse_to_fine/cv_fold_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load CV results
        cv_results = load_cv_results()
        
        # Load spatial/temporal metadata
        metadata = load_spatial_temporal_metadata()
        
        # Map folds to space/time
        enhanced_results, fold_mapping = map_folds_to_space_time(cv_results, metadata)
        
        # Remove outliers from the data
        enhanced_results_filtered = filter_outliers_from_results(enhanced_results, outlier_method='iqr', outlier_factor=1.5)
        
        # Create visualizations using filtered data
        spatial_map = create_spatial_performance_map(enhanced_results_filtered, output_dir)
        temporal_timeline = create_temporal_performance_timeline(enhanced_results_filtered, output_dir)
        performance_analysis = create_performance_analysis(enhanced_results_filtered, output_dir)
        spatiotemporal_heatmap = create_combined_spatiotemporal_heatmap(enhanced_results_filtered, output_dir)
        
        # Save enhanced results (both original and filtered)
        results_path = output_dir / "enhanced_cv_results.csv"
        enhanced_results.to_csv(results_path, index=False)
        print(f"   ✅ Saved original enhanced results: {results_path}")
        
        filtered_results_path = output_dir / "enhanced_cv_results_filtered.csv"
        enhanced_results_filtered.to_csv(filtered_results_path, index=False)
        print(f"   ✅ Saved filtered enhanced results: {filtered_results_path}")
        
        print("\n" + "=" * 60)
        print("✅ CV Fold Performance Analysis Complete!")
        print(f"📁 Outputs saved to: {output_dir}")
        
        # Print key findings using filtered data
        print(f"\n🔍 KEY FINDINGS (after outlier removal):")
        for model in enhanced_results_filtered['model_name'].unique():
            model_data = enhanced_results_filtered[enhanced_results_filtered['model_name'] == model]
            best_fold = model_data.loc[model_data['r2'].idxmax()]
            worst_fold = model_data.loc[model_data['r2'].idxmin()]
            
            print(f"\n   {model.upper()}:")
            print(f"     • Overall: R² = {model_data['r2'].mean():.3f} ± {model_data['r2'].std():.3f}")
            print(f"     • Best Fold {best_fold['fold']:.0f}: R² = {best_fold['r2']:.3f}")
            print(f"       Location: {best_fold['test_center_lat']:.1f}°N, {best_fold['test_center_lon']:.1f}°W")
            print(f"       Time: {best_fold['time_start'].strftime('%Y-%m') if pd.notna(best_fold['time_start']) else 'N/A'} to {best_fold['time_end'].strftime('%Y-%m') if pd.notna(best_fold['time_end']) else 'N/A'}")
            print(f"     • Worst Fold {worst_fold['fold']:.0f}: R² = {worst_fold['r2']:.3f}")
            print(f"       Location: {worst_fold['test_center_lat']:.1f}°N, {worst_fold['test_center_lon']:.1f}°W")
            print(f"       Time: {worst_fold['time_start'].strftime('%Y-%m') if pd.notna(worst_fold['time_start']) else 'N/A'} to {worst_fold['time_end'].strftime('%Y-%m') if pd.notna(worst_fold['time_end']) else 'N/A'}")
        
        print(f"\n📖 Generated visualizations:")
        print(f"   • Spatial performance map: {spatial_map.name}")
        print(f"   • Temporal performance timeline: {temporal_timeline.name}")
        print(f"   • Performance analysis: {performance_analysis.name}")
        print(f"   • Spatiotemporal heatmap: {spatiotemporal_heatmap.name}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())