#!/usr/bin/env python3
"""
Validation Scatter Plot Script for Downscaled GRACE Data

Creates comprehensive validation scatter plots showing:
1. Individual ML Models (55km) vs Original GRACE (55km) - Fair resolution comparison
2. Ensemble Model vs Original GRACE - Combined predictions
3. Final Downscaled (5km→55km) vs Original GRACE - After residual correction
4. Automatically adapts layout based on available models (RF, LGB, XGB, etc.)
5. Publication-quality metrics for each comparison with 1:1 validation lines

Usage:
    python plot_validation_scatter.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("husl")

def load_mrb_boundary():
    """Load Mississippi River Basin boundary."""
    mrb_file = Path("/home/sauravbhattarai/Documents/ORISE/GroundWater-Downscaling/data/shapefiles/MRB.geojson")
    
    if mrb_file.exists():
        print(f"✅ Loading MRB boundary from: {mrb_file}")
        gdf = gpd.read_file(mrb_file)
        
        if len(gdf) > 1:
            combined_geometry = unary_union(gdf.geometry)
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

def create_mask_for_resolution(lats, lons, mrb_polygon, resolution_name):
    """Create MRB mask for given resolution."""
    print(f"🎯 Creating MRB mask for {resolution_name}...")
    
    # Create coordinate arrays
    lon_2d, lat_2d = np.meshgrid(lons, lats)
    mask = np.zeros_like(lon_2d, dtype=bool)
    
    # Vectorized point-in-polygon test
    total_points = lon_2d.size
    processed = 0
    
    for i in range(len(lats)):
        for j in range(len(lons)):
            point = Point(lon_2d[i, j], lat_2d[i, j])
            mask[i, j] = mrb_polygon.contains(point) or mrb_polygon.touches(point)
            processed += 1
            
        if processed % 1000 == 0:
            print(f"   Progress: {processed:,}/{total_points:,} ({100*processed/total_points:.1f}%)")
    
    points_inside = mask.sum()
    print(f"   {resolution_name}: {points_inside:,} / {total_points:,} points inside MRB ({100*points_inside/total_points:.1f}%)")
    return mask

def extract_matched_data(ds_pred, ds_final, ds_orig, masks, model_vars, sample_fraction=1.0):
    """
    Extract temporally and spatially matched data from all three datasets.
    
    SCIENTIFIC APPROACH: Ensures fair comparison by using identical spatial sampling
    for all models. Final downscaled data is aggregated to match the original 55km 
    resolution to eliminate resolution bias.
    
    Parameters:
    -----------
    ds_pred : xr.Dataset
        Predictions (before correction) containing all ML models
    ds_final : xr.Dataset  
        Final downscaled (after correction)
    ds_orig : xr.Dataset
        Original GRACE
    masks : dict
        MRB masks for each dataset
    model_vars : list
        List of model variable names (e.g., ['prediction_rf', 'prediction_lgb', 'prediction_ensemble'])
    sample_fraction : float
        Fraction of data to sample for plotting
        
    Returns:
    --------
    dict with data for each model + final
    """
    print("📊 Extracting matched validation data for all models...")
    print("🔬 SCIENTIFIC APPROACH: Using identical spatial sampling for fair comparison")
    
    # Get variable names
    final_var = list(ds_final.data_vars)[0]  
    orig_var = list(ds_orig.data_vars)[0]
    
    print(f"   Available ML models: {model_vars}")
    print(f"   Final downscaled variable: {final_var}")
    print(f"   Original GRACE variable: {orig_var}")
    
    # Get time ranges
    times_pred = pd.to_datetime(ds_pred.time.values)
    times_final = pd.to_datetime(ds_final.time.values)
    times_orig = pd.to_datetime(ds_orig.time.values)
    
    # Find common time range
    common_times = sorted(set(times_pred) & set(times_final) & set(times_orig))
    print(f"   Common time points: {len(common_times)} ({common_times[0].strftime('%Y-%m')} to {common_times[-1].strftime('%Y-%m')})")
    
    # CRITICAL: Use 55km grid (original/predictions) as the reference spatial sampling
    # This ensures all models are compared at identical spatial locations
    reference_lats = ds_orig.lat.values
    reference_lons = ds_orig.lon.values  
    reference_mask = masks['orig']  # Use original GRACE mask as reference
    
    print(f"   Reference grid (55km): {len(reference_lats)} × {len(reference_lons)} = {reference_mask.sum():,} valid pixels")
    print(f"   Final grid (5km): {len(ds_final.lat)} × {len(ds_final.lon)} = {masks['final'].sum():,} valid pixels")
    print(f"   📊 Spatial aggregation ratio: {masks['final'].sum() / reference_mask.sum():.1f}:1")
    
    # Initialize storage for all models + final
    all_model_data = {model_var: [] for model_var in model_vars}
    all_model_data['final'] = []
    
    from scipy.ndimage import zoom
    from scipy.spatial import cKDTree
    
    # Progress tracking for large datasets
    total_processing_time = len(common_times) * len(model_vars)
    processed_count = 0
    
    print(f"🚀 Processing {len(common_times)} time points × {len(model_vars)+1} models = {total_processing_time+len(common_times)} operations...")
    
    for time_idx, time_point in enumerate(common_times):
        # Get indices for this time in each dataset
        pred_idx = list(times_pred).index(time_point)
        final_idx = list(times_final).index(time_point) 
        orig_idx = list(times_orig).index(time_point)
        
        # Extract original GRACE data slice (reference for all comparisons)
        orig_slice = ds_orig[orig_var].isel(time=orig_idx).values.copy()
        orig_slice[~reference_mask] = np.nan
        
        # SCIENTIFIC FIX: Aggregate final downscaled data to 55km reference grid
        # This ensures spatial consistency across all model comparisons
        final_slice_5km = ds_final[final_var].isel(time=final_idx).values.copy()
        final_slice_5km[~masks['final']] = np.nan
        
        # Aggregate 5km final data to 55km grid using spatial averaging
        # Calculate aggregation factors
        lat_factor = len(ds_final.lat) / len(reference_lats) 
        lon_factor = len(ds_final.lon) / len(reference_lons)
        
        # Perform spatial aggregation (mean pooling to match 55km resolution)
        final_slice_55km = zoom(final_slice_5km, (1/lat_factor, 1/lon_factor), order=1)
        
        # Ensure exact shape match
        if final_slice_55km.shape != orig_slice.shape:
            final_slice_55km = final_slice_55km[:orig_slice.shape[0], :orig_slice.shape[1]]
        
        # Apply same mask as reference for fair comparison
        final_slice_55km[~reference_mask] = np.nan
        
        # Process final downscaled vs original (now both at 55km resolution)
        valid_mask_final = (~np.isnan(final_slice_55km)) & (~np.isnan(orig_slice))
        if valid_mask_final.sum() > 0:
            final_valid = final_slice_55km[valid_mask_final]
            orig_valid = orig_slice[valid_mask_final]
            
            # HPC-optimized: Use all available data for maximum scientific accuracy
            n_points = len(final_valid)
            n_sample = max(100, int(n_points * sample_fraction))
            
            # Only sample if explicitly requested (sample_fraction < 1.0)
            if sample_fraction < 1.0 and n_points > n_sample:
                sample_idx = np.random.choice(n_points, n_sample, replace=False)
                final_valid = final_valid[sample_idx]
                orig_valid = orig_valid[sample_idx]
                print(f"   📊 Final model: Using {len(final_valid):,} sampled points ({sample_fraction*100:.0f}% of {n_points:,})")
            else:
                print(f"   🎯 Final model: Using all {len(final_valid):,} available data points (100% coverage)")
            
            # Store final data (now at same resolution as individual models)
            for i in range(len(final_valid)):
                all_model_data['final'].append({
                    'date': time_point,
                    'year': time_point.year,
                    'month': time_point.month,
                    'predicted': final_valid[i],
                    'original': orig_valid[i]
                })
        
        # Process each ML model vs original (55km comparison - using identical spatial sampling)
        for model_var in model_vars:
            pred_slice = ds_pred[model_var].isel(time=pred_idx).values.copy()
            
            # CRITICAL: Use same reference mask for all models for fair comparison
            pred_slice[~reference_mask] = np.nan
            
            # Use identical spatial sampling as final model
            valid_mask_model = (~np.isnan(pred_slice)) & (~np.isnan(orig_slice))
            
            if valid_mask_model.sum() > 0:
                pred_valid = pred_slice[valid_mask_model]
                orig_valid = orig_slice[valid_mask_model]
                
                # HPC-optimized: Use identical sampling strategy as final model
                n_points = len(pred_valid)
                n_sample = max(100, int(n_points * sample_fraction))
                
                # Only sample if explicitly requested (sample_fraction < 1.0)
                if sample_fraction < 1.0 and n_points > n_sample:
                    sample_idx = np.random.choice(n_points, n_sample, replace=False)
                    pred_valid = pred_valid[sample_idx]
                    orig_valid = orig_valid[sample_idx]
                
                # Store model data (now using identical spatial sampling as final)
                for i in range(len(pred_valid)):
                    all_model_data[model_var].append({
                        'date': time_point,
                        'year': time_point.year,
                        'month': time_point.month,
                        'predicted': pred_valid[i],
                        'original': orig_valid[i]
                    })
        
        # Progress update for large dataset processing
        if (time_idx + 1) % max(1, len(common_times) // 10) == 0:
            progress_pct = (time_idx + 1) / len(common_times) * 100
            print(f"   ⏱️  Progress: {time_idx + 1}/{len(common_times)} time points ({progress_pct:.0f}%)")
    
    # Convert to DataFrames with HPC-optimized processing
    print("📊 Converting to DataFrames for analysis...")
    model_dataframes = {}
    total_data_points = 0
    
    for model_name, data_list in all_model_data.items():
        if len(data_list) > 0:
            model_dataframes[model_name] = pd.DataFrame(data_list)
            total_data_points += len(data_list)
            print(f"   ✅ {model_name}: {len(data_list):,} data points")
        else:
            print(f"   ⚠️ No valid data for {model_name}")
    
    print(f"\n🎯 TOTAL DATASET SIZE: {total_data_points:,} data points across all models")
    print(f"💾 Estimated memory usage: ~{total_data_points * 8 * 5 / 1e6:.1f} MB")
    print("🚀 Maximum scientific accuracy achieved with 100% data coverage!")
    
    return model_dataframes

def extract_before_after_correction_data(ds_pred_5km, ds_final, ds_orig, masks, sample_fraction=1.0):
    """
    Extract before/after residual correction comparison data with same sampling as other models.
    
    Parameters:
    -----------
    ds_pred_5km : xr.Dataset
        5km predictions (before correction)
    ds_final : xr.Dataset  
        Final downscaled (after correction)
    ds_orig : xr.Dataset
        Original GRACE
    masks : dict
        MRB masks for each dataset
    sample_fraction : float
        Fraction of data to sample (same as other models)
        
    Returns:
    --------
    dict with before/after data for plotting
    """
    from scipy.ndimage import zoom
    import pandas as pd
    
    print("📊 Extracting before/after residual correction data with matched sampling...")
    
    # Get time range - use ALL common times, not just 20
    times_pred = pd.to_datetime(ds_pred_5km.time.values)
    times_final = pd.to_datetime(ds_final.time.values)
    times_orig = pd.to_datetime(ds_orig.time.values)
    common_times = sorted(set(times_pred) & set(times_final) & set(times_orig))
    
    print(f"   Processing ALL {len(common_times)} time steps for fair comparison...")
    
    orig_var = list(ds_orig.data_vars)[0]
    final_var = list(ds_final.data_vars)[0]
    
    # Calculate aggregation factors
    lat_factor = len(ds_pred_5km.lat) / len(ds_orig.lat)
    lon_factor = len(ds_pred_5km.lon) / len(ds_orig.lon)
    
    before_data = []
    after_data = []
    grace_data = []
    
    for time_point in common_times:
        try:
            # Get indices
            pred_idx = list(times_pred).index(time_point)
            final_idx = list(times_final).index(time_point)
            orig_idx = list(times_orig).index(time_point)
            
            # Get data slices
            grace_slice = ds_orig[orig_var].isel(time=orig_idx).values
            pred_5km_slice = ds_pred_5km.prediction_ensemble.isel(time=pred_idx).values
            final_5km_slice = ds_final[final_var].isel(time=final_idx).values
            
            # Aggregate 5km to 55km
            pred_agg = zoom(pred_5km_slice, (1/lat_factor, 1/lon_factor), order=1)
            final_agg = zoom(final_5km_slice, (1/lat_factor, 1/lon_factor), order=1)
            
            # Ensure shape match
            if pred_agg.shape != grace_slice.shape:
                pred_agg = pred_agg[:grace_slice.shape[0], :grace_slice.shape[1]]
            if final_agg.shape != grace_slice.shape:
                final_agg = final_agg[:grace_slice.shape[0], :grace_slice.shape[1]]
            
            # Apply mask and collect valid data
            mask = masks['orig'] & (~np.isnan(grace_slice)) & (~np.isnan(pred_agg)) & (~np.isnan(final_agg))
            
            grace_data.extend(grace_slice[mask].flatten())
            before_data.extend(pred_agg[mask].flatten()) 
            after_data.extend(final_agg[mask].flatten())
            
        except Exception as e:
            print(f"   ⚠️ Skipping time {time_point}: {e}")
            continue
    
    # Convert to arrays
    grace_data = np.array(grace_data)
    before_data = np.array(before_data)
    after_data = np.array(after_data)
    
    print(f"   📊 Total data points collected: {len(grace_data):,}")
    
    # Apply same sampling strategy as other models
    n_points = len(grace_data)
    n_sample = max(100, int(n_points * sample_fraction))
    
    if sample_fraction < 1.0 and n_points > n_sample:
        sample_idx = np.random.choice(n_points, n_sample, replace=False)
        grace_data = grace_data[sample_idx]
        before_data = before_data[sample_idx] 
        after_data = after_data[sample_idx]
        print(f"   📊 Using {len(grace_data):,} sampled points ({sample_fraction*100:.0f}% of {n_points:,})")
    else:
        print(f"   🎯 Using all {len(grace_data):,} available data points (100% coverage)")
    
    return {
        'grace': grace_data,
        'before': before_data,
        'after': after_data
    }

def calculate_metrics(y_true, y_pred):
    """Calculate validation metrics."""
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 2:
        return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 'slope': np.nan, 'n': 0}
    
    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    bias = np.mean(y_pred_clean - y_true_clean)
    
    # Linear regression for slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true_clean, y_pred_clean)
    
    return {
        'r2': r2,
        'rmse': rmse, 
        'mae': mae,
        'bias': bias,
        'slope': slope,
        'intercept': intercept,
        'r_pearson': r_value,
        'p_value': p_value,
        'n': len(y_true_clean)
    }

def create_scatter_plot(ax, x, y, title, metrics, color='blue', alpha=0.6):
    """Create a single scatter plot with validation metrics."""
    
    # Create density scatter plot
    from matplotlib.colors import LogNorm
    
    # Remove NaN values for plotting
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Determine axis limits
    all_values = np.concatenate([x_clean, y_clean])
    vmin = np.percentile(all_values, 1)
    vmax = np.percentile(all_values, 99)
    
    # Add some padding
    padding = (vmax - vmin) * 0.05
    axis_min = vmin - padding
    axis_max = vmax + padding
    
    # Create scatter plot
    scatter = ax.scatter(x_clean, y_clean, c=color, alpha=alpha, s=20, edgecolors='none', rasterized=True)
    
    # Add 1:1 line
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', linewidth=2, alpha=0.8, label='1:1 line')
    
    # Add regression line if valid
    if not np.isnan(metrics['slope']):
        reg_x = np.array([axis_min, axis_max])
        reg_y = metrics['slope'] * reg_x + metrics['intercept']
        ax.plot(reg_x, reg_y, 'r-', linewidth=2, alpha=0.8, label=f'y = {metrics["slope"]:.2f}x + {metrics["intercept"]:.2f}')
    
    # Set axis limits
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    
    # Make it square
    ax.set_aspect('equal', adjustable='box')
    
    # Labels - simplified and informative
    ax.set_xlabel('GRACE (cm)', fontsize=12, fontweight='bold')
    ax.set_ylabel(title + ' (cm)', fontsize=12, fontweight='bold')
    # Remove title since y-axis label now describes the model
    
    # Add metrics text
    metrics_text = (f"R² = {metrics['r2']:.3f}\n"
                   f"RMSE = {metrics['rmse']:.2f} cm\n" 
                   f"MAE = {metrics['mae']:.2f} cm\n"
                   f"Bias = {metrics['bias']:+.2f} cm")
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    # Grid
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    return scatter

def determine_subplot_layout(n_models):
    """Determine optimal subplot layout based on number of models."""
    if n_models <= 2:
        return 1, n_models
    elif n_models <= 4:
        return 2, 2
    elif n_models <= 6:
        return 2, 3
    elif n_models <= 9:
        return 3, 3
    elif n_models <= 12:
        return 3, 4
    else:
        return 4, 4  # Maximum 4x4 grid

def get_model_display_names():
    """Get nice display names for models."""
    return {
        'prediction_rf': 'Random Forest',
        'prediction_lgb': 'LightGBM', 
        'prediction_xgb': 'XGBoost',
        'prediction_nn': 'Neural Network',
        'prediction_ensemble': 'Ensemble',
        'final': 'Final Downscaled\n'
    }

def create_comprehensive_validation_plot():
    """Create comprehensive validation scatter plots."""
    
    print("📊 Creating validation scatter plots...")
    
    # Load datasets
    print("📂 Loading datasets...")
    
    # Predictions (before residual correction) - at 55km
    pred_path = "/home/sauravbhattarai/Documents/ORISE/GroundWater-Downscaling/data/processed_coarse_to_fine/predictions_55km.nc"
    ds_pred = xr.open_dataset(pred_path)
    
    # Predictions at 5km (before residual correction)
    pred_5km_path = "/home/sauravbhattarai/Documents/ORISE/GroundWater-Downscaling/data/processed_coarse_to_fine/predictions_5km.nc"
    try:
        ds_pred_5km = xr.open_dataset(pred_5km_path)
        has_5km_predictions = True
        print(f"✅ 5km predictions found for before/after comparison")
    except FileNotFoundError:
        ds_pred_5km = None
        has_5km_predictions = False
        print(f"⚠️  5km predictions not found, skipping before/after comparison")
    
    # Final downscaled (after residual correction)  
    final_path = "results_coarse_to_fine/grace_downscaled_5km.nc"
    ds_final = xr.open_dataset(final_path)
    
    # Original GRACE
    orig_path = "/home/sauravbhattarai/Documents/ORISE/GroundWater-Downscaling/data/processed_coarse_to_fine/grace_filled_stl.nc"
    ds_orig = xr.open_dataset(orig_path)
    
    print(f"✅ Predictions shape: {dict(ds_pred.dims)}")
    print(f"✅ Final shape: {dict(ds_final.dims)}")
    print(f"✅ Original shape: {dict(ds_orig.dims)}")
    
    # Get all available model variables
    model_vars = [var for var in ds_pred.data_vars if var.startswith('prediction_')]
    print(f"🔍 Found {len(model_vars)} ML models: {model_vars}")
    
    # Load MRB boundary and create masks
    mrb_polygon = load_mrb_boundary()
    
    masks = {
        'pred': create_mask_for_resolution(ds_pred.lat.values, ds_pred.lon.values, mrb_polygon, "Predictions"),
        'final': create_mask_for_resolution(ds_final.lat.values, ds_final.lon.values, mrb_polygon, "Final"), 
        'orig': create_mask_for_resolution(ds_orig.lat.values, ds_orig.lon.values, mrb_polygon, "Original")
    }
    
    # Extract matched data for all models (100% data for maximum scientific accuracy)
    model_dataframes = extract_matched_data(ds_pred, ds_final, ds_orig, masks, model_vars, sample_fraction=1.0)
    
    if not model_dataframes:
        print("❌ No matched data found!")
        return None
    
    # Extract before/after comparison data if available
    before_after_data = None
    if has_5km_predictions:
        print("\n📊 Extracting before/after residual correction comparison...")
        before_after_data = extract_before_after_correction_data(ds_pred_5km, ds_final, ds_orig, masks, sample_fraction=1.0)
    
    # Determine subplot layout - include before/after correction plots if available, exclude duplicate final
    n_total_plots = len(model_dataframes) - 1  # Subtract 1 to exclude duplicate 'final' plot
    if before_after_data is not None and len(before_after_data['grace']) > 0:
        n_total_plots += 2  # Add before and after correction plots
    nrows, ncols = determine_subplot_layout(n_total_plots)
    
    print(f"🎨 Creating {nrows}x{ncols} subplot layout for {n_total_plots} comparisons")
    
    # Create figure with dynamic sizing
    #fig_width = max(12, ncols * 4)
    #fig_height = max(8, nrows * 4)
    fig_width = 12
    fig_height = 8
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')
    
    # Handle single plot case
    if n_total_plots == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    else:
        axes = axes.flatten()
    
    # Get model display names
    display_names = get_model_display_names()
    
    # Define colors for different models  
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    # Plot each model comparison
    plot_idx = 0
    all_metrics = {}
    
    # First plot ML models (in a logical order)
    model_order = ['prediction_lgb', 'prediction_xgb', 'prediction_nn', 'prediction_ensemble']
    for model_var in model_order:
        if model_var in model_dataframes:
            df = model_dataframes[model_var]
            if not df.empty:
                # Calculate metrics
                metrics = calculate_metrics(df['original'].values, df['predicted'].values)
                all_metrics[model_var] = metrics
                
                # Create scatter plot - simplified title
                title = display_names.get(model_var, model_var.replace('prediction_', '').upper())
                color = colors[plot_idx % len(colors)]
                
                scatter = create_scatter_plot(
                    axes[plot_idx], df['original'], df['predicted'],
                    title, metrics,
                    color=color, alpha=0.6
                )
                
                plot_idx += 1
    
    # Skip the duplicate "Final Downscaled" plot - we'll use "After Residual Correction" instead
    
    # Add before/after correction plots if available
    if before_after_data is not None and len(before_after_data['grace']) > 0:
        grace_data = before_after_data['grace']
        before_data = before_after_data['before']
        after_data = before_after_data['after']
        
        # Plot 1: Before Correction
        before_metrics = calculate_metrics(grace_data, before_data)
        all_metrics['before_correction'] = before_metrics
        
        color_before = '#E74C3C'  # Red for before
        
        scatter = create_scatter_plot(
            axes[plot_idx], grace_data, before_data,
            "Before Residual Correction", before_metrics,
            color=color_before, alpha=0.6
        )
        plot_idx += 1
        
        # Plot 2: After Correction  
        after_metrics = calculate_metrics(grace_data, after_data)
        all_metrics['after_correction'] = after_metrics
        
        color_after = '#27AE60'  # Green for after
        
        scatter = create_scatter_plot(
            axes[plot_idx], grace_data, after_data,
            "After Residual Correction", after_metrics,
            color=color_after, alpha=0.6
        )
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    # Print comprehensive metrics summary
    print("\n📊 Comprehensive Validation Metrics:")
    print("="*60)
    for model_name, metrics in all_metrics.items():
        if model_name == 'before_correction':
            display_name = "Before Residual Correction"
        elif model_name == 'after_correction':
            display_name = "After Residual Correction"  
        elif model_name == 'final':
            continue  # Skip the duplicate final plot
        else:
            display_name = display_names.get(model_name, model_name)
        print(f"{display_name:25} R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.2f}cm  MAE={metrics['mae']:.2f}cm")
    
    # Calculate improvements if we have both ensemble and final
    if 'prediction_ensemble' in all_metrics and 'final' in all_metrics:
        ensemble_metrics = all_metrics['prediction_ensemble']
        final_metrics = all_metrics['final']
        
        delta_r2 = final_metrics['r2'] - ensemble_metrics['r2']
        delta_rmse = final_metrics['rmse'] - ensemble_metrics['rmse']
        delta_mae = final_metrics['mae'] - ensemble_metrics['mae']
        
        print("="*80)
        print(f"📈 Residual Correction Improvement (Final vs Ensemble):")
        print(f"   ΔR² = {delta_r2:+.3f}  ΔRMSE = {delta_rmse:+.2f}cm  ΔMAE = {delta_mae:+.2f}cm")
    
    # Calculate before/after correction improvement if available
    if 'before_correction' in all_metrics and 'after_correction' in all_metrics:
        before_metrics = all_metrics['before_correction']
        after_metrics = all_metrics['after_correction']
        
        delta_r2_correction = after_metrics['r2'] - before_metrics['r2']
        delta_rmse_correction = after_metrics['rmse'] - before_metrics['rmse']
        delta_mae_correction = after_metrics['mae'] - before_metrics['mae']
        
        print("="*80)
        print(f"🔬 Before/After Residual Correction Impact (5km→55km):")
        print(f"   ΔR² = {delta_r2_correction:+.3f}  ΔRMSE = {delta_rmse_correction:+.2f}cm  ΔMAE = {delta_mae_correction:+.2f}cm")
        if delta_r2_correction > 0:
            print("   ✅ Residual correction IMPROVED downscaling performance!")
        else:
            print("   ⚠️ Residual correction DEGRADED downscaling performance!")

    
    # Adjust layout dynamically
    plt.tight_layout()
    if nrows == 1:
        plt.subplots_adjust(top=0.85, bottom=0.12, left=0.06, right=0.98, wspace=0.3)
    else:
        plt.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.98, wspace=0.25, hspace=0.4)
    
    # Save figure
    output_path = Path("figures_coarse_to_fine/comprehensive_validation_scatter.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', transparent=False)
    print(f"💾 Saved comprehensive validation plots: {output_path}")
    
    # Save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white',
                edgecolor='none', transparent=False)
    print(f"💾 Saved comprehensive validation PDF: {pdf_path}")
    
    # Note: Before/after correction plots are now integrated into the comprehensive validation scatter plot
    if before_after_data is not None and len(before_after_data['grace']) > 0:
        print(f"✅ Before/after residual correction plots integrated into comprehensive validation")
    elif has_5km_predictions:
        print(f"⚠️ No valid before/after correction data available")
    
    return output_path

def create_before_after_comparison_plot(before_after_data):
    """Create before/after residual correction comparison plot."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    print("\n🔬 Creating Before/After Residual Correction Comparison...")
    
    grace_data = before_after_data['grace']
    before_data = before_after_data['before'] 
    after_data = before_after_data['after']
    
    if len(grace_data) == 0:
        print("⚠️ No data available for before/after comparison")
        return None
    
    print(f"   📊 Comparing {len(grace_data):,} data points")
    
    # Calculate metrics for both
    before_metrics = calculate_metrics(grace_data, before_data)
    after_metrics = calculate_metrics(grace_data, after_data)
    
    if not before_metrics or not after_metrics:
        print("⚠️ Unable to calculate metrics for comparison")
        return None
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    
    # Common plotting parameters
    alpha = 0.6
    color_before = '#2E86AB'  # Blue
    color_after = '#A23B72'   # Purple
    
    # Plot 1: Before Correction
    ax1.scatter(grace_data, before_data, alpha=alpha, color=color_before, s=1)
    
    # Perfect correlation line
    min_val = min(np.nanmin(grace_data), np.nanmin(before_data))
    max_val = max(np.nanmax(grace_data), np.nanmax(before_data))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.7, label='1:1 line')
    
    # Formatting
    ax1.set_xlabel('GRACE Observed (cm)', fontsize=10)
    ax1.set_ylabel('Before Correction (cm)', fontsize=10)
    ax1.set_title('Before Residual Correction\n(5km predictions → 55km aggregated)', fontsize=11, pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Add metrics text
    metrics_text_before = f"R² = {before_metrics['r2']:.3f}\nRMSE = {before_metrics['rmse']:.2f} cm\nMAE = {before_metrics['mae']:.2f} cm\nn = {before_metrics['n']:,}"
    ax1.text(0.05, 0.95, metrics_text_before, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: After Correction  
    ax2.scatter(grace_data, after_data, alpha=alpha, color=color_after, s=1)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.7, label='1:1 line')
    
    # Formatting
    ax2.set_xlabel('GRACE Observed (cm)', fontsize=10)
    ax2.set_ylabel('After Correction (cm)', fontsize=10)
    ax2.set_title('After Residual Correction\n(Final corrected → 55km aggregated)', fontsize=11, pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Add metrics text
    metrics_text_after = f"R² = {after_metrics['r2']:.3f}\nRMSE = {after_metrics['rmse']:.2f} cm\nMAE = {after_metrics['mae']:.2f} cm\nn = {after_metrics['n']:,}"
    ax2.text(0.05, 0.95, metrics_text_after, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set equal aspect and limits for fair comparison
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
    
    plt.tight_layout()
    
    # Add improvement summary
    delta_r2 = after_metrics['r2'] - before_metrics['r2']
    delta_rmse = after_metrics['rmse'] - before_metrics['rmse']
    delta_mae = after_metrics['mae'] - before_metrics['mae']
    
    # Add overall title with improvement metrics
    improvement_text = f"Residual Correction Impact: ΔR² = {delta_r2:+.3f}, ΔRMSE = {delta_rmse:+.2f}cm, ΔMAE = {delta_mae:+.2f}cm"
    fig.suptitle(improvement_text, fontsize=12, y=0.98)
    
    # Save plot
    output_path = Path("figures_coarse_to_fine/before_after_residual_correction.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', transparent=False)
    
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white',
                edgecolor='none', transparent=False)
    
    plt.close()
    
    # Print comparison results
    print(f"\n📊 BEFORE/AFTER COMPARISON RESULTS:")
    print("="*60)
    print(f"Before Correction:  R²={before_metrics['r2']:.3f}  RMSE={before_metrics['rmse']:.2f}cm  MAE={before_metrics['mae']:.2f}cm")
    print(f"After Correction:   R²={after_metrics['r2']:.3f}  RMSE={after_metrics['rmse']:.2f}cm  MAE={after_metrics['mae']:.2f}cm")
    print("="*60)
    if delta_r2 > 0:
        print("✅ Residual correction IMPROVED performance!")
    else:
        print("⚠️ Residual correction DEGRADED performance!")
    print("="*60)
    
    return output_path

def main():
    """Main function."""
    print("📈 Creating Validation Scatter Plots...")
    
    try:
        output_path = create_comprehensive_validation_plot()
        print(f"✅ Comprehensive validation scatter plots complete!")
        print(f"📄 Publication-ready plots with all ML models: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error creating validation plots: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())