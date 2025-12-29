#!/usr/bin/env python3
"""
Diagnose Spatial Buffer Conflicts in CV

Shows what happens when spatial buffer zones overlap with other blocks,
potentially excluding entire spatial blocks from training data.

This addresses the critical issue: if one spatial block is surrounded by others
within the buffer distance, those blocks get excluded from training, 
potentially causing severe data loss or fold failures.
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

plt.rcParams.update({
    'font.size': 14,                
    'axes.labelsize': 16,           
    'axes.titlesize': 18,           
    'xtick.labelsize': 14,          
    'ytick.labelsize': 14,          
    'legend.fontsize': 12,          
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'lines.linewidth': 2,           
    'axes.linewidth': 1.5,            
    'grid.linewidth': 1,          
})


def load_example_data():
    """Load actual data for CV analysis."""
    print("📊 Loading GRACE data for buffer conflict analysis...")
    
    # Load feature stack to get spatial structure
    feature_path = "data/processed_coarse_to_fine/feature_stack_55km.nc"
    if not Path(feature_path).exists():
        raise FileNotFoundError(f"Feature stack not found at {feature_path}")
    
    ds = xr.open_dataset(feature_path)
    
    # Get spatial and temporal dimensions
    lat_values = ds.lat.values
    lon_values = ds.lon.values
    time_values = pd.to_datetime(ds.time.values)
    
    print(f"   Spatial grid: {len(lat_values)} lat × {len(lon_values)} lon = {len(lat_values) * len(lon_values)} pixels")
    print(f"   Time range: {time_values[0]} to {time_values[-1]} ({len(time_values)} time steps)")
    
    # Create sample data structure for CV
    n_times = len(time_values)
    n_spatial = len(lat_values) * len(lon_values)
    n_samples = n_times * n_spatial
    
    # Create dummy X, y for CV structure
    X_dummy = np.random.randn(n_samples, 10)  # 10 features
    y_dummy = np.random.randn(n_samples)
    
    # Create metadata for CV
    common_dates = [t.strftime('%Y-%m') for t in time_values]
    spatial_shape = (len(lat_values), len(lon_values))
    
    metadata = prepare_metadata_for_cv(
        X_dummy, y_dummy, common_dates, spatial_shape, feature_path
    )
    
    # Add actual lat/lon values to metadata
    metadata['lat_values'] = lat_values
    metadata['lon_values'] = lon_values  
    metadata['time_values'] = time_values
    
    return X_dummy, y_dummy, metadata


def analyze_all_folds_buffer_conflicts(X, y, metadata):
    """Analyze buffer conflicts across all CV folds."""
    print("🔍 Analyzing buffer conflicts across all CV folds...")
    
    # Test different buffer sizes to see the impact
    buffer_sizes = [0.25, 0.5, 1.0, 1.5]  # degrees
    results = []
    
    for buffer_size in buffer_sizes:
        print(f"\n--- Analyzing buffer size: {buffer_size}° ({buffer_size * 111:.0f} km) ---")
        
        # Initialize CV splitter
        cv_splitter = BlockedSpatiotemporalCV(
            n_spatial_blocks=5,
            n_temporal_blocks=4, 
            spatial_buffer_deg=buffer_size,
            temporal_buffer_months=1,
            random_state=42
        )
        
        # Get spatial blocks
        spatial_coords = metadata['spatial_coords']
        temporal_indices = metadata['temporal_indices']
        spatial_blocks = cv_splitter.create_spatial_blocks(spatial_coords)
        temporal_blocks = cv_splitter.create_temporal_blocks(temporal_indices)
        
        # Analyze each fold
        fold_id = 0
        buffer_results = []
        
        for spatial_test_block in range(5):  # n_spatial_blocks
            for temporal_test_block in range(4):  # n_temporal_blocks
                fold_id += 1
                
                # Create test mask
                spatial_test_mask = (spatial_blocks == spatial_test_block)
                temporal_test_mask = np.isin(
                    temporal_indices, 
                    temporal_blocks[temporal_test_block]
                )
                test_mask = spatial_test_mask & temporal_test_mask
                
                # Create spatial buffer mask
                spatial_buffer_mask = cv_splitter.create_spatial_buffer_mask(
                    spatial_coords, spatial_test_block, spatial_blocks
                )
                
                # Create temporal buffer mask
                test_times = temporal_blocks[temporal_test_block]
                if len(test_times) > 0:
                    min_test_time = test_times.min()
                    max_test_time = test_times.max()
                    
                    temporal_buffer_mask = (
                        (temporal_indices >= min_test_time - cv_splitter.temporal_buffer) &
                        (temporal_indices <= max_test_time + cv_splitter.temporal_buffer) &
                        ~temporal_test_mask
                    )
                else:
                    temporal_buffer_mask = np.zeros(len(temporal_indices), dtype=bool)
                
                # Combine: test + all buffers are excluded from training
                exclude_mask = test_mask | spatial_buffer_mask | temporal_buffer_mask
                train_mask = ~exclude_mask
                
                # Count samples
                total_samples = len(spatial_coords)
                train_samples = np.sum(train_mask)
                test_samples = np.sum(test_mask)
                spatial_buffer_samples = np.sum(spatial_buffer_mask)
                temporal_buffer_samples = np.sum(temporal_buffer_mask)
                
                # Analyze which spatial blocks are affected by buffer
                affected_blocks = []
                for other_block in range(5):
                    if other_block != spatial_test_block:
                        other_block_mask = (spatial_blocks == other_block)
                        overlap = np.sum(other_block_mask & spatial_buffer_mask)
                        total_in_block = np.sum(other_block_mask)
                        
                        if overlap > 0:
                            affected_blocks.append({
                                'block_id': other_block,
                                'overlap_samples': overlap,
                                'total_samples': total_in_block,
                                'overlap_pct': 100 * overlap / total_in_block if total_in_block > 0 else 0
                            })
                
                buffer_results.append({
                    'fold_id': fold_id,
                    'buffer_size_deg': buffer_size,
                    'buffer_size_km': buffer_size * 111,
                    'test_spatial_block': spatial_test_block,
                    'test_temporal_block': temporal_test_block,
                    'total_samples': total_samples,
                    'train_samples': train_samples,
                    'test_samples': test_samples,
                    'spatial_buffer_samples': spatial_buffer_samples,
                    'temporal_buffer_samples': temporal_buffer_samples,
                    'train_pct': 100 * train_samples / total_samples,
                    'test_pct': 100 * test_samples / total_samples,
                    'spatial_buffer_pct': 100 * spatial_buffer_samples / total_samples,
                    'temporal_buffer_pct': 100 * temporal_buffer_samples / total_samples,
                    'affected_blocks': affected_blocks,
                    'n_affected_blocks': len(affected_blocks)
                })
                
                if fold_id <= 5:  # Show first few folds in detail
                    print(f"  Fold {fold_id}: Train={train_samples:,} ({100*train_samples/total_samples:.1f}%), "
                          f"Test={test_samples:,} ({100*test_samples/total_samples:.1f}%), "
                          f"Spatial Buffer={spatial_buffer_samples:,} ({100*spatial_buffer_samples/total_samples:.1f}%)")
                    
                    if len(affected_blocks) > 0:
                        print(f"    Affected blocks: {[b['block_id'] for b in affected_blocks]}")
                        for block in affected_blocks:
                            print(f"      Block {block['block_id']}: {block['overlap_samples']:,}/{block['total_samples']:,} "
                                  f"samples ({block['overlap_pct']:.1f}%) in buffer")
        
        results.extend(buffer_results)
        
        # Summary for this buffer size
        df = pd.DataFrame(buffer_results)
        print(f"\n  Summary for buffer size {buffer_size}°:")
        print(f"    Mean training %: {df['train_pct'].mean():.1f}% ± {df['train_pct'].std():.1f}%")
        print(f"    Mean spatial buffer %: {df['spatial_buffer_pct'].mean():.1f}% ± {df['spatial_buffer_pct'].std():.1f}%")
        print(f"    Folds with affected blocks: {df['n_affected_blocks'].sum()}/{len(df)} total block overlaps")
        print(f"    Max spatial buffer %: {df['spatial_buffer_pct'].max():.1f}%")
    
    return pd.DataFrame(results)


def create_buffer_conflict_visualization(results_df, output_dir):
    """Create visualization showing buffer conflicts."""
    print("🎨 Creating buffer conflict visualization...")
    
    # Create figure with multiple panels
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Training data percentage vs buffer size
    ax1 = axes[0, 0]
    buffer_sizes = results_df['buffer_size_deg'].unique()
    
    for buffer_size in buffer_sizes:
        df_buffer = results_df[results_df['buffer_size_deg'] == buffer_size]
        ax1.scatter([buffer_size] * len(df_buffer), df_buffer['train_pct'], 
                   alpha=0.6, s=30, label=f'{buffer_size}° buffer')
    
    # Add mean line
    means = [results_df[results_df['buffer_size_deg'] == bs]['train_pct'].mean() 
             for bs in buffer_sizes]
    ax1.plot(buffer_sizes, means, 'ro-', linewidth=2, markersize=8, label='Mean')
    
    ax1.set_xlabel('Buffer Size (degrees)', fontweight='bold')
    ax1.set_ylabel('Training Data %', fontweight='bold')
    ax1.set_title('(A) Training Data Loss vs Buffer Size', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Spatial buffer percentage vs buffer size
    ax2 = axes[0, 1]
    
    for buffer_size in buffer_sizes:
        df_buffer = results_df[results_df['buffer_size_deg'] == buffer_size]
        ax2.scatter([buffer_size] * len(df_buffer), df_buffer['spatial_buffer_pct'], 
                   alpha=0.6, s=30, label=f'{buffer_size}° buffer')
    
    # Add mean line
    means = [results_df[results_df['buffer_size_deg'] == bs]['spatial_buffer_pct'].mean() 
             for bs in buffer_sizes]
    ax2.plot(buffer_sizes, means, 'ro-', linewidth=2, markersize=8, label='Mean')
    
    ax2.set_xlabel('Buffer Size (degrees)', fontweight='bold')
    ax2.set_ylabel('Spatial Buffer Data %', fontweight='bold')
    ax2.set_title('(B) Spatial Buffer Size vs Data Loss', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Panel 3: Number of affected blocks vs buffer size
    ax3 = axes[1, 0]
    
    affected_counts = []
    for buffer_size in buffer_sizes:
        df_buffer = results_df[results_df['buffer_size_deg'] == buffer_size]
        total_affected = df_buffer['n_affected_blocks'].sum()
        affected_counts.append(total_affected)
    
    ax3.bar(buffer_sizes, affected_counts, alpha=0.7, color='orange', edgecolor='black')
    
    for i, count in enumerate(affected_counts):
        ax3.text(buffer_sizes[i], count + 1, str(count), ha='center', fontweight='bold')
    
    ax3.set_xlabel('Buffer Size (degrees)', fontweight='bold')
    ax3.set_ylabel('Total Block Overlaps (All Folds)', fontweight='bold')
    ax3.set_title('(C) Spatial Block Conflicts', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    summary_data = []
    for buffer_size in buffer_sizes:
        df_buffer = results_df[results_df['buffer_size_deg'] == buffer_size]
        
        summary_data.append([
            f"{buffer_size}° ({buffer_size * 111:.0f} km)",
            f"{df_buffer['train_pct'].mean():.1f}% ± {df_buffer['train_pct'].std():.1f}%",
            f"{df_buffer['spatial_buffer_pct'].mean():.1f}% ± {df_buffer['spatial_buffer_pct'].std():.1f}%", 
            f"{df_buffer['spatial_buffer_pct'].max():.1f}%",
            f"{df_buffer['n_affected_blocks'].sum()}"
        ])
    
    table_data = [
        ['Buffer Size', 'Mean Training %', 'Mean Spatial Buffer %', 'Max Spatial Buffer %', 'Block Conflicts']
    ] + summary_data
    
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('(D) Buffer Conflict Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = output_dir / "cv_buffer_conflict_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved buffer conflict analysis: {output_path}")
    
    plt.close()
    
    return output_path


def main():
    """Main execution function."""
    print("🔬 CV Buffer Conflict Analysis")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path("figures_coarse_to_fine/cv_buffer_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load example data
        X, y, metadata = load_example_data()
        
        # Analyze buffer conflicts
        results_df = analyze_all_folds_buffer_conflicts(X, y, metadata)
        
        # Create visualization
        visualization_fig = create_buffer_conflict_visualization(results_df, output_dir)
        
        # Save detailed results
        results_path = output_dir / "buffer_conflict_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"   ✅ Saved detailed results: {results_path}")
        
        print("\n" + "=" * 60)
        print("✅ Buffer Conflict Analysis Complete!")
        print(f"📁 Output saved to: {output_dir}")
        
        # Print key findings
        print(f"\n🔍 KEY FINDINGS:")
        for buffer_size in results_df['buffer_size_deg'].unique():
            df_buffer = results_df[results_df['buffer_size_deg'] == buffer_size]
            worst_case = df_buffer.loc[df_buffer['spatial_buffer_pct'].idxmax()]
            
            print(f"   Buffer {buffer_size}° ({buffer_size * 111:.0f} km):")
            print(f"     • Mean training data: {df_buffer['train_pct'].mean():.1f}%")
            print(f"     • Worst case: {worst_case['spatial_buffer_pct']:.1f}% lost to spatial buffer (Fold {worst_case['fold_id']:.0f})")
            print(f"     • Total block conflicts: {df_buffer['n_affected_blocks'].sum()}")
        
        print(f"\n⚠️ CRITICAL ISSUE IDENTIFIED:")
        print(f"   When spatial blocks are close together, buffer zones can exclude")
        print(f"   entire spatial blocks from training, causing severe data loss!")
        print(f"   This violates the assumption of sufficient training data.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())