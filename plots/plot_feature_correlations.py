#!/usr/bin/env python3
"""
Feature Correlation Analysis and Visualization

Creates correlation matrices, heatmaps, and identifies highly correlated feature pairs 
to validate feature selection decisions and understand feature relationships.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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

def load_datasets():
    """Load both original and enhanced feature datasets."""
    print("📦 Loading datasets...")
    
    try:
        original_ds = xr.open_dataset("data/processed/feature_stack.nc")
        enhanced_ds = xr.open_dataset("data/processed/feature_stack_enhanced.nc")
        print("✅ Loaded both feature stacks")
        return original_ds, enhanced_ds
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return None, None

def calculate_correlation_matrix(features_array, feature_names, sample_size=50000):
    """Calculate correlation matrix efficiently."""
    print(f"🔗 Calculating correlation matrix for {len(feature_names)} features...")
    
    # Sample data for correlation calculation
    n_times, n_features, n_lat, n_lon = features_array.shape
    total_samples = n_times * n_lat * n_lon
    
    if total_samples > sample_size:
        # Random sampling
        indices = np.random.choice(total_samples, sample_size, replace=False)
        flat_features = features_array.reshape(n_times, n_features, -1)
        flat_features = flat_features.transpose(0, 2, 1).reshape(-1, n_features)
        sampled_features = flat_features[indices]
    else:
        # Use all data
        sampled_features = features_array.reshape(n_times, n_features, -1)
        sampled_features = sampled_features.transpose(0, 2, 1).reshape(-1, n_features)
    
    # Remove NaN rows
    valid_mask = ~np.isnan(sampled_features).any(axis=1)
    valid_features = sampled_features[valid_mask]
    
    print(f"   Using {len(valid_features):,} valid samples")
    
    if len(valid_features) < 1000:
        print("   ⚠️ Too few valid samples for correlation analysis")
        return None
    
    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(valid_features.T)
    
    return pd.DataFrame(correlation_matrix, index=feature_names, columns=feature_names)

def plot_correlation_heatmap(corr_df, title, output_path, figsize=(12, 10)):
    """Create correlation heatmap."""
    print(f"   Creating heatmap: {title}")
    
    plt.figure(figsize=figsize)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_df.values, dtype=bool))
    
    # Create heatmap
    ax = sns.heatmap(
        corr_df,
        mask=mask,
        annot=True if len(corr_df) <= 15 else False,  # Only annotate if not too many features
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_distribution(corr_df, title, output_path):
    """Plot distribution of correlation values."""
    print(f"   Creating correlation distribution: {title}")
    
    # Get upper triangle correlation values (excluding diagonal)
    mask = np.triu(np.ones_like(corr_df.values, dtype=bool), k=1)
    correlations = corr_df.values[mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(correlations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Correlation Coefficient')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Correlations')
    ax1.axvline(0.9, color='red', linestyle='--', label='High correlation threshold (0.9)')
    ax1.axvline(-0.9, color='red', linestyle='--')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(correlations, vert=True)
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title('Correlation Statistics')
    ax2.axhline(0.9, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(-0.9, color='red', linestyle='--', alpha=0.7)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def find_high_correlations(corr_df, threshold=0.9):
    """Find highly correlated pairs."""
    high_corr_pairs = []
    
    for i in range(len(corr_df.columns)):
        for j in range(i+1, len(corr_df.columns)):
            corr_val = corr_df.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr_pairs.append({
                    'feature1': corr_df.columns[i],
                    'feature2': corr_df.columns[j],
                    'correlation': corr_val
                })
    
    return pd.DataFrame(high_corr_pairs)

def plot_high_correlation_pairs(high_corr_df, title, output_path):
    """Plot high correlation pairs."""
    if len(high_corr_df) == 0:
        print("   No high correlation pairs found")
        return
    
    print(f"   Creating high correlation pairs plot: {title}")
    
    plt.figure(figsize=(10, max(6, len(high_corr_df) * 0.5)))
    
    # Create labels for pairs
    pair_labels = [f"{row['feature1']}\n↔\n{row['feature2']}" for _, row in high_corr_df.iterrows()]
    correlations = high_corr_df['correlation'].values
    
    # Color code by correlation strength
    colors = ['red' if abs(c) > 0.95 else 'orange' if abs(c) > 0.9 else 'yellow' for c in correlations]
    
    bars = plt.barh(range(len(correlations)), correlations, color=colors, alpha=0.7, edgecolor='black')
    
    plt.yticks(range(len(correlations)), pair_labels)
    plt.xlabel('Correlation Coefficient')
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add correlation values on bars
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        plt.text(corr + 0.01 if corr > 0 else corr - 0.01, i, f'{corr:.3f}', 
                va='center', ha='left' if corr > 0 else 'right', fontweight='bold')
    
    plt.axvline(0.9, color='red', linestyle='--', alpha=0.5, label='High correlation threshold')
    plt.axvline(-0.9, color='red', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_categories_correlation(enhanced_corr_df, output_path):
    """Plot correlation between different feature categories."""
    print("   Creating feature category correlation analysis")
    
    feature_names = enhanced_corr_df.columns
    
    # Categorize features
    categories = {
        'Original': [],
        'Lag 1': [],
        'Lag 3': [],
        'Lag 6': [],
        'Seasonal': []
    }
    
    for name in feature_names:
        name_str = str(name)
        if 'lag6' in name_str:
            categories['Lag 6'].append(name)
        elif 'lag3' in name_str:
            categories['Lag 3'].append(name)
        elif 'lag1' in name_str:
            categories['Lag 1'].append(name)
        elif any(x in name_str for x in ['month_sin', 'month_cos']):
            categories['Seasonal'].append(name)
        else:
            categories['Original'].append(name)
    
    # Calculate average correlations between categories
    category_corr = {}
    for cat1, features1 in categories.items():
        if not features1:
            continue
        for cat2, features2 in categories.items():
            if not features2 or cat1 == cat2:
                continue
            
            # Get correlations between all pairs
            corrs = []
            for f1 in features1:
                for f2 in features2:
                    if f1 in enhanced_corr_df.index and f2 in enhanced_corr_df.columns:
                        corrs.append(abs(enhanced_corr_df.loc[f1, f2]))
            
            if corrs:
                category_corr[f"{cat1} ↔ {cat2}"] = np.mean(corrs)
    
    if category_corr:
        plt.figure(figsize=(10, 6))
        
        pairs = list(category_corr.keys())
        correlations = list(category_corr.values())
        
        bars = plt.bar(range(len(pairs)), correlations, color='lightblue', alpha=0.7, edgecolor='black')
        
        plt.xticks(range(len(pairs)), pairs, rotation=45, ha='right')
        plt.ylabel('Average Absolute Correlation')
        plt.title('Average Correlations Between Feature Categories', fontsize=14, fontweight='bold')
        
        # Add values on bars
        for bar, corr in zip(bars, correlations):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def create_correlation_summary_report(original_corr_df, enhanced_corr_df, output_dir):
    """Create a summary report of correlation analysis."""
    print("📋 Creating correlation summary report...")
    
    report_path = output_dir / "correlation_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("FEATURE CORRELATION ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        # Original features analysis
        f.write("ORIGINAL FEATURES ANALYSIS:\n")
        f.write(f"Number of features: {len(original_corr_df)}\n")
        
        original_high = find_high_correlations(original_corr_df, 0.9)
        f.write(f"High correlations (|r| > 0.9): {len(original_high)}\n")
        
        if len(original_high) > 0:
            f.write("\nHighly correlated pairs:\n")
            for _, row in original_high.iterrows():
                f.write(f"  {row['feature1']} ↔ {row['feature2']}: {row['correlation']:.3f}\n")
        
        # Enhanced features analysis
        f.write(f"\nENHANCED FEATURES ANALYSIS:\n")
        f.write(f"Number of features: {len(enhanced_corr_df)}\n")
        
        enhanced_high = find_high_correlations(enhanced_corr_df, 0.9)
        f.write(f"High correlations (|r| > 0.9): {len(enhanced_high)}\n")
        
        # Correlation statistics
        orig_mask = np.triu(np.ones_like(original_corr_df.values, dtype=bool), k=1)
        orig_corrs = original_corr_df.values[orig_mask]
        
        enh_mask = np.triu(np.ones_like(enhanced_corr_df.values, dtype=bool), k=1)
        enh_corrs = enhanced_corr_df.values[enh_mask]
        
        f.write(f"\nCORRELATION STATISTICS:\n")
        f.write(f"Original features:\n")
        f.write(f"  Mean |correlation|: {np.mean(np.abs(orig_corrs)):.3f}\n")
        f.write(f"  Max |correlation|: {np.max(np.abs(orig_corrs)):.3f}\n")
        f.write(f"  Std correlation: {np.std(orig_corrs):.3f}\n")
        
        f.write(f"\nEnhanced features:\n")
        f.write(f"  Mean |correlation|: {np.mean(np.abs(enh_corrs)):.3f}\n")
        f.write(f"  Max |correlation|: {np.max(np.abs(enh_corrs)):.3f}\n")
        f.write(f"  Std correlation: {np.std(enh_corrs):.3f}\n")
        
        f.write(f"\nIMPROVEMENTS:\n")
        f.write(f"High correlations removed: {len(original_high) - len(find_high_correlations(enhanced_corr_df, 0.9))}\n")
        f.write(f"Features added: {len(enhanced_corr_df) - len(original_corr_df)}\n")
    
    print(f"✅ Report saved to {report_path}")

def main():
    """Main function for correlation analysis and visualization."""
    
    print("🔗 FEATURE CORRELATION ANALYSIS")
    print("=" * 40)
    
    # Setup output directory
    output_dir = setup_output_directory()
    print(f"📁 Output directory: {output_dir}")
    
    # Load datasets
    original_ds, enhanced_ds = load_datasets()
    if original_ds is None or enhanced_ds is None:
        return
    
    # Calculate correlation matrices
    print("\n📊 Calculating correlation matrices...")
    
    # Original features correlation
    original_corr_df = calculate_correlation_matrix(
        original_ds.features.values,
        [str(name) for name in original_ds.feature.values]
    )
    
    # Enhanced features correlation
    enhanced_corr_df = calculate_correlation_matrix(
        enhanced_ds.features.values,
        [str(name) for name in enhanced_ds.feature.values]
    )
    
    if original_corr_df is None or enhanced_corr_df is None:
        print("❌ Failed to calculate correlation matrices")
        return
    
    print("\n🎨 Creating visualizations...")
    
    # 1. Original features correlation heatmap
    plot_correlation_heatmap(
        original_corr_df,
        "Original Features Correlation Matrix",
        output_dir / "01_original_correlation_heatmap.png",
        figsize=(10, 8)
    )
    
    # 2. Enhanced features correlation heatmap (sample if too large)
    if len(enhanced_corr_df) > 20:
        # Show only original features + a few lag features for visualization
        sample_features = list(enhanced_corr_df.columns[:10]) + list(enhanced_corr_df.columns[-5:])
        enhanced_sample = enhanced_corr_df.loc[sample_features, sample_features]
        plot_correlation_heatmap(
            enhanced_sample,
            "Enhanced Features Correlation Matrix (Sample)",
            output_dir / "02_enhanced_correlation_heatmap_sample.png",
            figsize=(12, 10)
        )
    else:
        plot_correlation_heatmap(
            enhanced_corr_df,
            "Enhanced Features Correlation Matrix",
            output_dir / "02_enhanced_correlation_heatmap.png"
        )
    
    # 3. Correlation distributions
    plot_correlation_distribution(
        original_corr_df,
        "Original Features - Correlation Distribution",
        output_dir / "03_original_correlation_distribution.png"
    )
    
    plot_correlation_distribution(
        enhanced_corr_df,
        "Enhanced Features - Correlation Distribution",
        output_dir / "04_enhanced_correlation_distribution.png"
    )
    
    # 4. High correlation pairs
    original_high = find_high_correlations(original_corr_df, 0.9)
    enhanced_high = find_high_correlations(enhanced_corr_df, 0.9)
    
    if len(original_high) > 0:
        plot_high_correlation_pairs(
            original_high,
            "High Correlation Pairs - Original Features",
            output_dir / "05_original_high_correlations.png"
        )
    
    if len(enhanced_high) > 0:
        plot_high_correlation_pairs(
            enhanced_high,
            "High Correlation Pairs - Enhanced Features",
            output_dir / "06_enhanced_high_correlations.png"
        )
    
    # 5. Feature category correlations
    plot_feature_categories_correlation(
        enhanced_corr_df,
        output_dir / "07_feature_categories_correlation.png"
    )
    
    # 6. Create summary report
    create_correlation_summary_report(original_corr_df, enhanced_corr_df, output_dir)
    
    print(f"\n✅ Correlation analysis complete!")
    print(f"📁 All plots saved to: {output_dir}")
    print(f"📋 Summary report: {output_dir}/correlation_analysis_report.txt")
    
    # Print key findings
    print(f"\n🔍 KEY FINDINGS:")
    print(f"   Original high correlations (|r| > 0.9): {len(original_high)}")
    print(f"   Enhanced high correlations (|r| > 0.9): {len(enhanced_high)}")
    print(f"   Correlation pairs removed: {len(original_high) - len(enhanced_high)}")
    print(f"   Features: {len(original_corr_df)} → {len(enhanced_corr_df)} (+{len(enhanced_corr_df) - len(original_corr_df)})")

if __name__ == "__main__":
    main()