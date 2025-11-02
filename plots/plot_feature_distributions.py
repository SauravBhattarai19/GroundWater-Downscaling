#!/usr/bin/env python3
"""
Feature Distribution Analysis and Visualization

Shows distribution and scaling comparison plots between original and enhanced features.
Validates unit conversions and scaling transformations.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up plotting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

def setup_output_directory():
    """Create output directory for this script's plots."""
    script_name = Path(__file__).stem
    output_dir = Path("figures") / script_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_datasets_and_scaler():
    """Load datasets and scaler."""
    print("📦 Loading datasets and scaler...")
    
    try:
        original_ds = xr.open_dataset("data/processed/feature_stack.nc")
        enhanced_ds = xr.open_dataset("data/processed/feature_stack_enhanced.nc")
        
        # Load scaler if available
        scaler = None
        try:
            scaler = joblib.load("data/processed/feature_stack_enhanced_scaler.joblib")
            print("✅ Loaded scaler")
        except:
            print("⚠️ No scaler found")
        
        print("✅ Loaded feature stacks")
        return original_ds, enhanced_ds, scaler
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return None, None, None

def plot_feature_value_ranges(original_ds, enhanced_ds, output_path):
    """Compare value ranges between original and enhanced features."""
    print("   Creating feature value ranges comparison")
    
    # Get common features
    original_features = [str(f) for f in original_ds.feature.values]
    enhanced_features = [str(f) for f in enhanced_ds.feature.values]
    
    # Find features that exist in both (original features in enhanced)
    common_features = [f for f in original_features if f in enhanced_features]
    
    if not common_features:
        print("   ⚠️ No common features found")
        return
    
    # Calculate ranges for each feature
    ranges_data = []
    
    for feature in common_features:
        # Original
        orig_data = original_ds.features.sel(feature=feature).values
        orig_valid = orig_data[~np.isnan(orig_data)]
        
        # Enhanced
        enh_data = enhanced_ds.features.sel(feature=feature).values
        enh_valid = enh_data[~np.isnan(enh_data)]
        
        if len(orig_valid) > 0 and len(enh_valid) > 0:
            ranges_data.append({
                'feature': feature,
                'original_min': orig_valid.min(),
                'original_max': orig_valid.max(),
                'original_range': orig_valid.max() - orig_valid.min(),
                'enhanced_min': enh_valid.min(),
                'enhanced_max': enh_valid.max(),
                'enhanced_range': enh_valid.max() - enh_valid.min()
            })
    
    ranges_df = pd.DataFrame(ranges_data)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Min/Max values
    x_pos = np.arange(len(ranges_df))
    width = 0.35
    
    ax1.bar(x_pos - width/2, ranges_df['original_min'], width, 
            label='Original Min', alpha=0.7, color='lightcoral')
    ax1.bar(x_pos - width/2, ranges_df['original_max'] - ranges_df['original_min'], width,
            bottom=ranges_df['original_min'], label='Original Range', alpha=0.7, color='lightblue')
    
    ax1.bar(x_pos + width/2, ranges_df['enhanced_min'], width,
            label='Enhanced Min', alpha=0.7, color='salmon')
    ax1.bar(x_pos + width/2, ranges_df['enhanced_max'] - ranges_df['enhanced_min'], width,
            bottom=ranges_df['enhanced_min'], label='Enhanced Range', alpha=0.7, color='skyblue')
    
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Value Range')
    ax1.set_title('Feature Value Ranges: Original vs Enhanced')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ranges_df['feature'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Range ratios (log scale)
    range_ratios = ranges_df['enhanced_range'] / ranges_df['original_range']
    colors = ['green' if abs(r - 1) < 0.1 else 'orange' if abs(r - 1) < 0.5 else 'red' for r in range_ratios]
    
    bars = ax2.bar(x_pos, range_ratios, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No change')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Range Ratio (Enhanced/Original)')
    ax2.set_title('Range Change Ratio')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ranges_df['feature'], rotation=45, ha='right')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add ratio values on bars
    for bar, ratio in zip(bars, range_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{ratio:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_distributions_comparison(original_ds, enhanced_ds, output_path):
    """Compare distributions of common features."""
    print("   Creating feature distribution comparisons")
    
    # Get common features
    original_features = [str(f) for f in original_ds.feature.values]
    enhanced_features = [str(f) for f in enhanced_ds.feature.values]
    common_features = [f for f in original_features if f in enhanced_features]
    
    # Select a subset for visualization (max 9 features)
    if len(common_features) > 9:
        common_features = common_features[:9]
    
    n_features = len(common_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(common_features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Get data
        orig_data = original_ds.features.sel(feature=feature).values.flatten()
        enh_data = enhanced_ds.features.sel(feature=feature).values.flatten()
        
        # Remove NaN values
        orig_valid = orig_data[~np.isnan(orig_data)]
        enh_valid = enh_data[~np.isnan(enh_data)]
        
        if len(orig_valid) > 0 and len(enh_valid) > 0:
            # Sample data if too large
            if len(orig_valid) > 10000:
                orig_sample = np.random.choice(orig_valid, 10000, replace=False)
            else:
                orig_sample = orig_valid
                
            if len(enh_valid) > 10000:
                enh_sample = np.random.choice(enh_valid, 10000, replace=False)
            else:
                enh_sample = enh_valid
            
            # Plot histograms
            ax.hist(orig_sample, bins=50, alpha=0.5, label='Original', density=True, color='lightcoral')
            ax.hist(enh_sample, bins=50, alpha=0.5, label='Enhanced', density=True, color='lightblue')
            
            ax.set_title(f'{feature}', fontsize=10)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{feature}', fontsize=10)
    
    # Hide extra subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Feature Distributions: Original vs Enhanced', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_scaling_effects(enhanced_ds, scaler, output_path):
    """Show effects of scaling on feature distributions."""
    if scaler is None:
        print("   ⚠️ No scaler available, skipping scaling effects plot")
        return
    
    print("   Creating scaling effects visualization")
    
    # Get sample data from enhanced features
    features_array = enhanced_ds.features.values
    feature_names = [str(f) for f in enhanced_ds.feature.values]
    
    # Sample data for scaling demonstration
    n_times, n_features, n_lat, n_lon = features_array.shape
    
    # Flatten and sample
    flat_features = features_array.reshape(n_times, n_features, -1)
    flat_features = flat_features.transpose(0, 2, 1).reshape(-1, n_features)
    
    # Remove NaN rows
    valid_mask = ~np.isnan(flat_features).any(axis=1)
    valid_features = flat_features[valid_mask]
    
    if len(valid_features) == 0:
        print("   ⚠️ No valid data for scaling demonstration")
        return
    
    # Sample for visualization
    sample_size = min(10000, len(valid_features))
    sample_indices = np.random.choice(len(valid_features), sample_size, replace=False)
    sample_data = valid_features[sample_indices]
    
    # Apply scaling
    scaled_data = scaler.transform(sample_data)
    
    # Select features to show (max 9)
    features_to_show = min(9, n_features)
    feature_indices = np.linspace(0, n_features-1, features_to_show, dtype=int)
    
    n_cols = 3
    n_rows = (features_to_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, feat_idx in enumerate(feature_indices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot before and after scaling
        ax.hist(sample_data[:, feat_idx], bins=30, alpha=0.5, label='Before scaling', 
                density=True, color='lightcoral')
        ax.hist(scaled_data[:, feat_idx], bins=30, alpha=0.5, label='After scaling', 
                density=True, color='lightblue')
        
        ax.set_title(f'{feature_names[feat_idx][:20]}...', fontsize=10)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for i in range(features_to_show, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(f'Scaling Effects ({type(scaler).__name__})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_scale_comparison(original_ds, enhanced_ds, output_path):
    """Compare feature scales between original and enhanced datasets."""
    print("   Creating feature scale comparison")
    
    # Calculate statistics for original features
    orig_stats = []
    for feature in original_ds.feature.values:
        data = original_ds.features.sel(feature=feature).values.flatten()
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            orig_stats.append({
                'feature': str(feature),
                'mean': valid_data.mean(),
                'std': valid_data.std(),
                'min': valid_data.min(),
                'max': valid_data.max(),
                'range': valid_data.max() - valid_data.min()
            })
    
    # Calculate statistics for enhanced features (only original features)
    enh_stats = []
    for feature in enhanced_ds.feature.values:
        feature_str = str(feature)
        # Only include original features (no lag or seasonal)
        if not any(x in feature_str for x in ['lag', 'month_sin', 'month_cos']):
            data = enhanced_ds.features.sel(feature=feature).values.flatten()
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                enh_stats.append({
                    'feature': feature_str,
                    'mean': valid_data.mean(),
                    'std': valid_data.std(),
                    'min': valid_data.min(),
                    'max': valid_data.max(),
                    'range': valid_data.max() - valid_data.min()
                })
    
    orig_df = pd.DataFrame(orig_stats)
    enh_df = pd.DataFrame(enh_stats)
    
    # Create scale comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Standard deviations
    if len(orig_df) > 0 and len(enh_df) > 0:
        common_features = set(orig_df['feature']) & set(enh_df['feature'])
        
        if common_features:
            orig_subset = orig_df[orig_df['feature'].isin(common_features)].sort_values('feature')
            enh_subset = enh_df[enh_df['feature'].isin(common_features)].sort_values('feature')
            
            x_pos = np.arange(len(common_features))
            width = 0.35
            
            ax1.bar(x_pos - width/2, orig_subset['std'], width, 
                   label='Original', alpha=0.7, color='lightcoral')
            ax1.bar(x_pos + width/2, enh_subset['std'], width,
                   label='Enhanced', alpha=0.7, color='lightblue')
            
            ax1.set_xlabel('Features')
            ax1.set_ylabel('Standard Deviation')
            ax1.set_title('Standard Deviations Comparison')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(orig_subset['feature'], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # Plot 2: Ranges
            ax2.bar(x_pos - width/2, orig_subset['range'], width,
                   label='Original', alpha=0.7, color='lightcoral')
            ax2.bar(x_pos + width/2, enh_subset['range'], width,
                   label='Enhanced', alpha=0.7, color='lightblue')
            
            ax2.set_xlabel('Features')
            ax2.set_ylabel('Range (Max - Min)')
            ax2.set_title('Value Ranges Comparison')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(orig_subset['feature'], rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            
            # Plot 3: Scale ratios
            std_ratios = enh_subset['std'].values / orig_subset['std'].values
            range_ratios = enh_subset['range'].values / orig_subset['range'].values
            
            ax3.plot(x_pos, std_ratios, 'o-', label='Std Ratio', linewidth=2, markersize=8)
            ax3.plot(x_pos, range_ratios, 's-', label='Range Ratio', linewidth=2, markersize=8)
            ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No change')
            
            ax3.set_xlabel('Features')
            ax3.set_ylabel('Ratio (Enhanced/Original)')
            ax3.set_title('Scale Change Ratios')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(orig_subset['feature'], rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
            
            # Plot 4: Feature scale distribution
            all_orig_ranges = orig_df['range'].values
            all_enh_ranges = enh_df['range'].values
            
            ax4.hist(all_orig_ranges, bins=20, alpha=0.5, label='Original', 
                    density=True, color='lightcoral')
            ax4.hist(all_enh_ranges, bins=20, alpha=0.5, label='Enhanced', 
                    density=True, color='lightblue')
            
            ax4.set_xlabel('Feature Range')
            ax4.set_ylabel('Density')
            ax4.set_title('Distribution of Feature Ranges')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_distribution_summary_report(original_ds, enhanced_ds, scaler, output_dir):
    """Create summary report of distribution analysis."""
    print("📋 Creating distribution summary report...")
    
    report_path = output_dir / "feature_distributions_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("FEATURE DISTRIBUTIONS ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        # Dataset comparison
        f.write("DATASET COMPARISON:\n")
        f.write(f"Original features: {len(original_ds.feature)}\n")
        f.write(f"Enhanced features: {len(enhanced_ds.feature)}\n")
        f.write(f"Features added: {len(enhanced_ds.feature) - len(original_ds.feature)}\n\n")
        
        # Memory usage
        orig_size = original_ds.features.nbytes / 1024**3
        enh_size = enhanced_ds.features.nbytes / 1024**3
        f.write(f"Memory usage:\n")
        f.write(f"  Original: {orig_size:.2f} GB\n")
        f.write(f"  Enhanced: {enh_size:.2f} GB\n")
        f.write(f"  Increase: {enh_size/orig_size:.1f}x\n\n")
        
        # Scaling information
        f.write("SCALING ANALYSIS:\n")
        if scaler is not None:
            f.write(f"Scaler type: {type(scaler).__name__}\n")
            if hasattr(scaler, 'scale_'):
                f.write(f"Number of features scaled: {len(scaler.scale_)}\n")
                f.write(f"Scale factors range: [{scaler.scale_.min():.3f}, {scaler.scale_.max():.3f}]\n")
            if hasattr(scaler, 'center_'):
                f.write(f"Center values range: [{scaler.center_.min():.3f}, {scaler.center_.max():.3f}]\n")
        else:
            f.write("No scaler applied\n")
        
        # Unit fixes applied
        f.write(f"\nUNIT FIXES IDENTIFIED:\n")
        f.write("Temperature features likely fixed (divided by 10)\n")
        f.write("Check temperature ranges in plots for validation\n")
        
        f.write(f"\nRECOMMENDations:\n")
        f.write("1. Verify temperature unit fixes are correct\n")
        f.write("2. Ensure scaling is applied consistently during model training\n")
        f.write("3. Monitor model performance improvements\n")
        f.write("4. Consider additional feature engineering if needed\n")
    
    print(f"✅ Report saved to {report_path}")

def main():
    """Main function for distribution analysis and visualization."""
    
    print("📊 FEATURE DISTRIBUTIONS ANALYSIS")
    print("=" * 40)
    
    # Setup output directory
    output_dir = setup_output_directory()
    print(f"📁 Output directory: {output_dir}")
    
    # Load datasets and scaler
    original_ds, enhanced_ds, scaler = load_datasets_and_scaler()
    if original_ds is None or enhanced_ds is None:
        return
    
    print("\n🎨 Creating visualizations...")
    
    # 1. Feature value ranges comparison
    plot_feature_value_ranges(
        original_ds, enhanced_ds,
        output_dir / "01_feature_value_ranges_comparison.png"
    )
    
    # 2. Feature distributions comparison
    plot_feature_distributions_comparison(
        original_ds, enhanced_ds,
        output_dir / "02_feature_distributions_comparison.png"
    )
    
    # 3. Scaling effects
    plot_scaling_effects(
        enhanced_ds, scaler,
        output_dir / "03_scaling_effects.png"
    )
    
    # 4. Feature scale comparison
    plot_feature_scale_comparison(
        original_ds, enhanced_ds,
        output_dir / "04_feature_scale_comparison.png"
    )
    
    # 5. Create summary report
    create_distribution_summary_report(original_ds, enhanced_ds, scaler, output_dir)
    
    print(f"\n✅ Distribution analysis complete!")
    print(f"📁 All plots saved to: {output_dir}")
    print(f"📋 Summary report: {output_dir}/feature_distributions_report.txt")
    
    # Print key findings
    print(f"\n🔍 KEY FINDINGS:")
    print(f"   Features: {len(original_ds.feature)} → {len(enhanced_ds.feature)}")
    print(f"   Memory: {original_ds.features.nbytes/1024**3:.2f} GB → {enhanced_ds.features.nbytes/1024**3:.2f} GB")
    print(f"   Scaler: {type(scaler).__name__ if scaler else 'None'}")

if __name__ == "__main__":
    main()