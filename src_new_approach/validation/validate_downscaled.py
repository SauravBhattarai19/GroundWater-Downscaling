"""
Validation Module for Downscaled GRACE

Validates downscaled GRACE against:
1. Original GRACE (coarse scale) - consistency check
2. USGS well observations - groundwater validation
3. Spatial patterns and temporal trends
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def validate_downscaled_grace(downscaled_path: str,
                              config: Dict,
                              output_dir: str = "results_coarse_to_fine",
                              figures_dir: str = "figures_coarse_to_fine",
                              logger=None):
    """
    Comprehensive validation of downscaled GRACE.
    
    Parameters:
    -----------
    downscaled_path : str
        Path to downscaled GRACE NetCDF
    config : Dict
        Configuration dictionary
    output_dir : str
        Output directory for validation results
    figures_dir : str
        Output directory for figures
    logger : logging.Logger
        Logger instance
    """
    if logger:
        logger.info("="*70)
        logger.info("🔍 VALIDATING DOWNSCALED GRACE")
        logger.info("="*70)
    else:
        print("="*70)
        print("🔍 VALIDATING DOWNSCALED GRACE")
        print("="*70)
    
    # Create output directories
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load downscaled data
    ds = xr.open_dataset(downscaled_path)
    
    # 1. Basic statistics
    stats = compute_statistics(ds, logger)
    save_statistics(stats, output_dir / "validation_statistics.csv", logger)
    
    # 2. Spatial validation
    spatial_metrics = validate_spatial_patterns(ds, figures_dir, logger)
    save_metrics(spatial_metrics, output_dir / "spatial_metrics.csv", logger)
    
    # 3. Temporal validation
    temporal_metrics = validate_temporal_patterns(ds, figures_dir, logger)
    save_metrics(temporal_metrics, output_dir / "temporal_metrics.csv", logger)
    
    # 4. Compare with original GRACE (if available)
    grace_filled_path = config.get('paths', {}).get('grace_filled')
    if grace_filled_path and Path(grace_filled_path).exists():
        comparison_metrics = compare_with_grace(
            ds, 
            grace_filled_path, 
            figures_dir, 
            logger
        )
        save_metrics(comparison_metrics, output_dir / "grace_comparison.csv", logger)
    
    # 5. USGS well validation (if available)
    if config.get('validation', {}).get('enable_well_validation', True):
        try:
            well_metrics = validate_against_wells(ds, config, figures_dir, logger)
            save_metrics(well_metrics, output_dir / "well_validation.csv", logger)
        except Exception as e:
            if logger:
                logger.warning(f"Well validation skipped: {e}")
            else:
                print(f"⚠️ Well validation skipped: {e}")
    
    # 6. Generate summary report
    generate_validation_report(
        stats,
        spatial_metrics,
        temporal_metrics,
        output_dir / "validation_report.txt",
        logger
    )
    
    if logger:
        logger.info("✅ Validation complete!")
        logger.info(f"   Results: {output_dir}")
        logger.info(f"   Figures: {figures_dir}")
    else:
        print("✅ Validation complete!")
        print(f"   Results: {output_dir}")
        print(f"   Figures: {figures_dir}")


def compute_statistics(ds: xr.Dataset, logger=None) -> Dict:
    """Compute comprehensive statistics of downscaled data."""
    if logger:
        logger.info("\n📊 Computing statistics...")
    else:
        print("\n📊 Computing statistics...")
    
    downscaled = ds['grace_downscaled'].values
    uncorrected = ds['grace_predicted_uncorrected'].values
    residuals = ds['residual_correction'].values
    
    stats = {}
    
    # Overall statistics
    for name, data in [('downscaled', downscaled), 
                       ('uncorrected', uncorrected),
                       ('residuals', residuals)]:
        valid_data = data[~np.isnan(data)]
        
        stats[f'{name}_mean'] = float(np.mean(valid_data))
        stats[f'{name}_std'] = float(np.std(valid_data))
        stats[f'{name}_min'] = float(np.min(valid_data))
        stats[f'{name}_max'] = float(np.max(valid_data))
        stats[f'{name}_median'] = float(np.median(valid_data))
        stats[f'{name}_q25'] = float(np.percentile(valid_data, 25))
        stats[f'{name}_q75'] = float(np.percentile(valid_data, 75))
    
    # Impact of residual correction
    mean_abs_correction = np.nanmean(np.abs(residuals))
    stats['mean_absolute_correction'] = float(mean_abs_correction)
    
    # Spatial coverage
    n_total = downscaled.size
    n_valid = np.sum(~np.isnan(downscaled))
    stats['spatial_coverage_pct'] = float(100 * n_valid / n_total)
    
    if logger:
        logger.info(f"   Downscaled mean: {stats['downscaled_mean']:.4f} ± {stats['downscaled_std']:.4f} cm")
        logger.info(f"   Downscaled range: [{stats['downscaled_min']:.2f}, {stats['downscaled_max']:.2f}] cm")
        logger.info(f"   Mean correction: {mean_abs_correction:.4f} cm")
        logger.info(f"   Spatial coverage: {stats['spatial_coverage_pct']:.1f}%")
    
    return stats


def validate_spatial_patterns(ds: xr.Dataset, figures_dir: Path, logger=None) -> Dict:
    """Validate spatial patterns in downscaled data."""
    if logger:
        logger.info("\n🗺️  Validating spatial patterns...")
    else:
        print("\n🗺️  Validating spatial patterns...")
    
    downscaled = ds['grace_downscaled']
    
    # Calculate spatial statistics
    spatial_mean = downscaled.mean(dim='time')
    spatial_std = downscaled.std(dim='time')
    
    # Metrics
    metrics = {
        'spatial_mean_avg': float(spatial_mean.mean()),
        'spatial_mean_std': float(spatial_mean.std()),
        'spatial_std_avg': float(spatial_std.mean()),
        'spatial_std_range': float(spatial_std.max() - spatial_std.min())
    }
    
    # Create spatial maps
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean map
    spatial_mean.plot(ax=axes[0, 0], cmap='RdBu_r', robust=True)
    axes[0, 0].set_title('Temporal Mean')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    
    # Std map
    spatial_std.plot(ax=axes[0, 1], cmap='YlOrRd', robust=True)
    axes[0, 1].set_title('Temporal Std Dev')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')
    
    # Example time slice
    mid_time = len(ds.time) // 2
    ds['grace_downscaled'].isel(time=mid_time).plot(
        ax=axes[1, 0], cmap='RdBu_r', robust=True
    )
    axes[1, 0].set_title(f'Example: {ds.time.values[mid_time]}')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    
    # Residual correction map
    ds['residual_correction'].isel(time=mid_time).plot(
        ax=axes[1, 1], cmap='PuOr', robust=True
    )
    axes[1, 1].set_title('Residual Correction (same time)')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    
    plt.tight_layout()
    spatial_fig_path = figures_dir / "spatial_patterns.png"
    plt.savefig(spatial_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"   Spatial figure saved: {spatial_fig_path}")
    
    return metrics


def validate_temporal_patterns(ds: xr.Dataset, figures_dir: Path, logger=None) -> Dict:
    """Validate temporal patterns in downscaled data."""
    if logger:
        logger.info("\n📈 Validating temporal patterns...")
    else:
        print("\n📈 Validating temporal patterns...")
    
    # Calculate temporal means
    temporal_mean = ds['grace_downscaled'].mean(dim=['lat', 'lon'])
    temporal_std = ds['grace_downscaled'].std(dim=['lat', 'lon'])
    
    # Metrics
    metrics = {
        'temporal_mean_avg': float(temporal_mean.mean()),
        'temporal_std_avg': float(temporal_std.mean()),
        'temporal_trend': float(np.polyfit(np.arange(len(temporal_mean)), 
                                          temporal_mean.values, 1)[0])
    }
    
    # Create temporal plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Time series
    temporal_mean.plot(ax=axes[0], color='blue', linewidth=1.5)
    axes[0].set_title('Spatial Mean Time Series')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('GRACE Anomaly (cm)')
    axes[0].grid(True, alpha=0.3)
    
    # With uncertainty band
    temporal_mean_np = temporal_mean.values
    temporal_std_np = temporal_std.values
    times = ds.time.values
    
    axes[1].fill_between(
        range(len(times)),
        temporal_mean_np - temporal_std_np,
        temporal_mean_np + temporal_std_np,
        alpha=0.3,
        color='blue',
        label='±1 Std Dev'
    )
    axes[1].plot(temporal_mean_np, color='blue', linewidth=1.5, label='Mean')
    axes[1].set_title('Time Series with Spatial Uncertainty')
    axes[1].set_xlabel('Time Index')
    axes[1].set_ylabel('GRACE Anomaly (cm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal pattern
    if len(times) >= 12:
        # Group by month
        months = pd.to_datetime(times).month
        monthly_means = [temporal_mean_np[months == m].mean() for m in range(1, 13)]
        monthly_stds = [temporal_mean_np[months == m].std() for m in range(1, 13)]
        
        axes[2].errorbar(range(1, 13), monthly_means, yerr=monthly_stds, 
                        marker='o', capsize=5, linewidth=2)
        axes[2].set_title('Seasonal Pattern (Monthly Averages)')
        axes[2].set_xlabel('Month')
        axes[2].set_ylabel('GRACE Anomaly (cm)')
        axes[2].set_xticks(range(1, 13))
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    temporal_fig_path = figures_dir / "temporal_patterns.png"
    plt.savefig(temporal_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"   Temporal figure saved: {temporal_fig_path}")
    
    return metrics


def compare_with_grace(ds: xr.Dataset, 
                      grace_path: str, 
                      figures_dir: Path, 
                      logger=None) -> Dict:
    """Compare downscaled with original GRACE at coarse scale."""
    if logger:
        logger.info("\n🔄 Comparing with original GRACE...")
    else:
        print("\n🔄 Comparing with original GRACE...")
    
    # Load GRACE
    grace_ds = xr.open_dataset(grace_path)
    grace_var = 'tws_anomaly' if 'tws_anomaly' in grace_ds else list(grace_ds.data_vars)[0]
    
    # Coarsen downscaled to GRACE resolution
    from src_new_approach.utils_downscaling import get_aggregation_factor, coarsen_2d
    
    downscaled = ds['grace_downscaled'].values
    grace_obs = grace_ds[grace_var].values
    
    # For simplicity, interpolate to same grid
    downscaled_coarse = []
    for t in range(min(len(downscaled), len(grace_obs))):
        # Interpolate downscaled to GRACE grid
        ds_at_t = ds['grace_downscaled'].isel(time=t).interp(
            lat=grace_ds.lat,
            lon=grace_ds.lon,
            method='linear'
        )
        downscaled_coarse.append(ds_at_t.values)
    
    downscaled_coarse = np.array(downscaled_coarse)
    grace_obs = grace_obs[:len(downscaled_coarse)]
    
    # Calculate metrics
    valid_mask = ~np.isnan(downscaled_coarse) & ~np.isnan(grace_obs)
    
    if np.sum(valid_mask) > 0:
        r2 = r2_score(grace_obs[valid_mask], downscaled_coarse[valid_mask])
        rmse = np.sqrt(mean_squared_error(grace_obs[valid_mask], downscaled_coarse[valid_mask]))
        mae = mean_absolute_error(grace_obs[valid_mask], downscaled_coarse[valid_mask])
        correlation = pearsonr(grace_obs[valid_mask], downscaled_coarse[valid_mask])[0]
        bias = np.mean(downscaled_coarse[valid_mask] - grace_obs[valid_mask])
    else:
        r2 = rmse = mae = correlation = bias = np.nan
    
    metrics = {
        'grace_comparison_r2': float(r2),
        'grace_comparison_rmse': float(rmse),
        'grace_comparison_mae': float(mae),
        'grace_comparison_correlation': float(correlation),
        'grace_comparison_bias': float(bias)
    }
    
    if logger:
        logger.info(f"   R² = {r2:.4f}")
        logger.info(f"   RMSE = {rmse:.2f} cm")
        logger.info(f"   Correlation = {correlation:.4f}")
    
    # Create scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter
    axes[0].scatter(grace_obs[valid_mask], downscaled_coarse[valid_mask], 
                   alpha=0.1, s=1)
    axes[0].plot([grace_obs[valid_mask].min(), grace_obs[valid_mask].max()],
                [grace_obs[valid_mask].min(), grace_obs[valid_mask].max()],
                'r--', label='1:1 line')
    axes[0].set_xlabel('Original GRACE (cm)')
    axes[0].set_ylabel('Downscaled (aggregated to GRACE resolution) (cm)')
    axes[0].set_title(f'GRACE Comparison\nR² = {r2:.3f}, RMSE = {rmse:.2f} cm')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Time series comparison
    grace_mean = np.nanmean(grace_obs, axis=(1, 2))
    downscaled_mean = np.nanmean(downscaled_coarse, axis=(1, 2))
    
    axes[1].plot(grace_mean, label='Original GRACE', linewidth=2)
    axes[1].plot(downscaled_mean, label='Downscaled (coarsened)', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Time Index')
    axes[1].set_ylabel('Spatial Mean (cm)')
    axes[1].set_title('Time Series Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_fig_path = figures_dir / "grace_comparison.png"
    plt.savefig(comparison_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"   Comparison figure saved: {comparison_fig_path}")
    
    return metrics


def validate_against_wells(ds: xr.Dataset, 
                          config: Dict, 
                          figures_dir: Path, 
                          logger=None) -> Dict:
    """Validate against USGS well observations."""
    if logger:
        logger.info("\n🚰 Validating against wells...")
    else:
        print("\n🚰 Validating against wells...")
    
    # This is a placeholder - actual implementation would require well data
    metrics = {
        'well_validation_note': 'Not implemented - requires USGS well data'
    }
    
    if logger:
        logger.info("   Well validation not yet implemented")
    
    return metrics


def save_statistics(stats: Dict, output_path: Path, logger=None):
    """Save statistics to CSV."""
    df = pd.DataFrame([stats])
    df.to_csv(output_path, index=False)
    
    if logger:
        logger.info(f"📄 Statistics saved: {output_path}")


def save_metrics(metrics: Dict, output_path: Path, logger=None):
    """Save metrics to CSV."""
    df = pd.DataFrame([metrics])
    df.to_csv(output_path, index=False)
    
    if logger:
        logger.info(f"📄 Metrics saved: {output_path}")


def generate_validation_report(stats: Dict,
                              spatial_metrics: Dict,
                              temporal_metrics: Dict,
                              output_path: Path,
                              logger=None):
    """Generate text validation report."""
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GRACE DOWNSCALING VALIDATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Downscaled Mean: {stats['downscaled_mean']:.4f} cm\n")
        f.write(f"Downscaled Std:  {stats['downscaled_std']:.4f} cm\n")
        f.write(f"Range: [{stats['downscaled_min']:.2f}, {stats['downscaled_max']:.2f}] cm\n")
        f.write(f"Median: {stats['downscaled_median']:.4f} cm\n")
        f.write(f"Mean Absolute Correction: {stats['mean_absolute_correction']:.4f} cm\n")
        f.write(f"Spatial Coverage: {stats['spatial_coverage_pct']:.1f}%\n\n")
        
        f.write("SPATIAL PATTERNS\n")
        f.write("-"*70 + "\n")
        for key, val in spatial_metrics.items():
            f.write(f"{key}: {val:.4f}\n")
        f.write("\n")
        
        f.write("TEMPORAL PATTERNS\n")
        f.write("-"*70 + "\n")
        for key, val in temporal_metrics.items():
            f.write(f"{key}: {val:.4f}\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    if logger:
        logger.info(f"📄 Validation report saved: {output_path}")


if __name__ == "__main__":
    import argparse
    from src_new_approach.utils_downscaling import load_config
    
    parser = argparse.ArgumentParser(description="Validate downscaled GRACE")
    parser.add_argument('--downscaled', required=True, help='Downscaled GRACE NetCDF')
    parser.add_argument('--config', default='src_new_approach/config_coarse_to_fine.yaml')
    parser.add_argument('--output-dir', default='results_coarse_to_fine')
    parser.add_argument('--figures-dir', default='figures_coarse_to_fine')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    validate_downscaled_grace(
        args.downscaled,
        config,
        args.output_dir,
        args.figures_dir
    )

