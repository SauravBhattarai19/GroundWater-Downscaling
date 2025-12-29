#!/usr/bin/env python3
"""
Visualize Spatiotemporal Cross-Validation Fold Structure

Creates a comprehensive presentation figure showing:
1. Spatial blocks with test region and buffer zones
2. Temporal blocks with test period and buffer zones  
3. Combined spatiotemporal fold visualization
4. Statistics and details

Usage:
    python src_new_approach/scripts/visualize_cv_fold.py
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

# Presentation-quality settings - LARGE TEXT for visibility
plt.rcParams.update({
    'font.size': 16,                
    'axes.labelsize': 18,           
    'axes.titlesize': 22,           
    'xtick.labelsize': 16,          
    'ytick.labelsize': 16,          
    'legend.fontsize': 14,          
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'lines.linewidth': 3,           
    'axes.linewidth': 2,            
    'grid.linewidth': 1.2,          
})


def load_example_data():
    """Load actual data for CV visualization."""
    print("📊 Loading actual GRACE data for CV demonstration...")
    
    # Load feature stack to get spatial structure
    feature_path = "data/processed_coarse_to_fine/feature_stack_55km.nc"
    if not Path(feature_path).exists():
        raise FileNotFoundError(f"Feature stack not found at {feature_path}")
    
    print(f"   Loading feature stack from: {feature_path}")
    ds = xr.open_dataset(feature_path)
    
    # Get spatial and temporal dimensions
    lat_values = ds.lat.values
    lon_values = ds.lon.values
    time_values = pd.to_datetime(ds.time.values)
    
    print(f"   Spatial grid: {len(lat_values)} lat × {len(lon_values)} lon = {len(lat_values) * len(lon_values)} pixels")
    print(f"   Time range: {time_values[0]} to {time_values[-1]} ({len(time_values)} time steps)")
    print(f"   Geographic bounds: {lat_values.min():.1f}°N to {lat_values.max():.1f}°N, {lon_values.min():.1f}°W to {lon_values.max():.1f}°W")
    
    # Create sample data structure for CV
    n_times = len(time_values)
    n_spatial = len(lat_values) * len(lon_values)
    n_samples = n_times * n_spatial
    
    # Create dummy X, y for CV structure (we just need the spatial/temporal organization)
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
    
    print(f"   Metadata prepared: {len(metadata['spatial_coords'])} samples")
    
    return X_dummy, y_dummy, metadata


def demonstrate_one_fold(X, y, metadata, fold_to_show=3):
    """Demonstrate one specific CV fold in detail."""
    print(f"🔍 Demonstrating CV fold #{fold_to_show}...")
    
    # Initialize CV splitter with configuration from your config
    cv_splitter = BlockedSpatiotemporalCV(
        n_spatial_blocks=5,
        n_temporal_blocks=4, 
        spatial_buffer_deg=0.5,  # ~55km buffer
        temporal_buffer_months=1,
        random_state=42
    )
    
    # Get the specific fold
    fold_count = 0
    target_fold = None
    
    for train_idx, test_idx in cv_splitter.split(X, y, metadata):
        fold_count += 1
        if fold_count == fold_to_show:
            target_fold = (train_idx, test_idx)
            break
    
    if target_fold is None:
        raise ValueError(f"Could not find fold #{fold_to_show}. Total folds available: {fold_count}")
    
    train_idx, test_idx = target_fold
    
    print(f"   Fold #{fold_to_show}: {len(train_idx)} training samples, {len(test_idx)} test samples")
    
    # Get spatial and temporal information
    spatial_coords = metadata['spatial_coords']
    temporal_indices = metadata['temporal_indices']
    lat_values = metadata['lat_values']
    lon_values = metadata['lon_values']
    time_values = metadata['time_values']
    
    # Create spatial blocks for visualization
    spatial_blocks = cv_splitter.create_spatial_blocks(spatial_coords)
    temporal_blocks = cv_splitter.create_temporal_blocks(temporal_indices)
    
    # Calculate which spatial and temporal blocks are being tested
    test_spatial_coords = spatial_coords[test_idx]
    test_temporal_indices = temporal_indices[test_idx]
    
    if len(test_idx) > 0:
        # Find dominant spatial block in test set
        test_spatial_blocks = spatial_blocks[test_idx]
        test_spatial_block_id = np.argmax(np.bincount(test_spatial_blocks))
        
        # Find temporal block
        test_temporal_block_id = None
        for block_id, times in temporal_blocks.items():
            if len(times) > 0 and np.any(np.isin(test_temporal_indices, times)):
                test_temporal_block_id = block_id
                break
        
        print(f"   Testing spatial block: {test_spatial_block_id}")
        print(f"   Testing temporal block: {test_temporal_block_id}")
        
        # Create buffer masks for visualization
        spatial_buffer_mask = cv_splitter.create_spatial_buffer_mask(
            spatial_coords, test_spatial_block_id, spatial_blocks
        )
        
        # Temporal buffer mask
        test_times = temporal_blocks[test_temporal_block_id] if test_temporal_block_id is not None else np.array([])
        if len(test_times) > 0:
            min_test_time = test_times.min()
            max_test_time = test_times.max()
            
            temporal_buffer_mask = (
                (temporal_indices >= min_test_time - cv_splitter.temporal_buffer) &
                (temporal_indices <= max_test_time + cv_splitter.temporal_buffer) &
                ~np.isin(temporal_indices, test_times)
            )
        else:
            temporal_buffer_mask = np.zeros(len(temporal_indices), dtype=bool)
        
        return {
            'fold_id': fold_to_show,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'spatial_blocks': spatial_blocks,
            'temporal_blocks': temporal_blocks,
            'test_spatial_block_id': test_spatial_block_id,
            'test_temporal_block_id': test_temporal_block_id,
            'spatial_buffer_mask': spatial_buffer_mask,
            'temporal_buffer_mask': temporal_buffer_mask,
            'spatial_coords': spatial_coords,
            'temporal_indices': temporal_indices,
            'lat_values': lat_values,
            'lon_values': lon_values,
            'time_values': time_values,
            'cv_splitter': cv_splitter
        }
    else:
        raise ValueError(f"No test samples found in fold #{fold_to_show}")


def create_cv_visualization(fold_data, output_dir):
    """Create comprehensive CV fold visualization."""
    print("🎨 Creating CV fold visualization...")
    
    # Create figure with 4 panels: 2x2 layout
    fig = plt.figure(figsize=(20, 16))
    
    # Panel layout
    ax1 = plt.subplot(2, 2, 1)  # Spatial blocks
    ax2 = plt.subplot(2, 2, 2)  # Temporal blocks  
    ax3 = plt.subplot(2, 2, 3)  # Spatiotemporal combined
    ax4 = plt.subplot(2, 2, 4)  # Statistics and details
    
    # Extract data
    spatial_coords = fold_data['spatial_coords']
    temporal_indices = fold_data['temporal_indices']
    spatial_blocks = fold_data['spatial_blocks']
    test_spatial_block_id = fold_data['test_spatial_block_id']
    test_temporal_block_id = fold_data['test_temporal_block_id']
    spatial_buffer_mask = fold_data['spatial_buffer_mask']
    temporal_buffer_mask = fold_data['temporal_buffer_mask']
    lat_values = fold_data['lat_values']
    lon_values = fold_data['lon_values']
    time_values = fold_data['time_values']
    train_idx = fold_data['train_idx']
    test_idx = fold_data['test_idx']
    fold_id = fold_data['fold_id']
    
    # === Panel 1: Spatial Blocks ===
    ax1.set_title(f'(A) Spatial Blocks - Fold #{fold_id}', fontsize=22, fontweight='bold', pad=20)
    
    # Create spatial grid for visualization
    lat_grid, lon_grid = np.meshgrid(lat_values, lon_values, indexing='ij')
    
    # Plot all spatial blocks
    unique_spatial_blocks = np.unique(spatial_blocks)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_spatial_blocks)))
    
    for i, block_id in enumerate(unique_spatial_blocks):
        mask_2d = np.zeros_like(lat_grid, dtype=bool)
        
        # Find samples in this spatial block (using first time step)
        block_samples = np.where((spatial_blocks == block_id) & (temporal_indices == 0))[0]
        
        if len(block_samples) > 0:
            for idx in block_samples:
                coord = spatial_coords[idx]
                lat_idx = np.argmin(np.abs(lat_values - coord[0]))
                lon_idx = np.argmin(np.abs(lon_values - coord[1]))
                mask_2d[lat_idx, lon_idx] = True
        
        if block_id == test_spatial_block_id:
            # Test block in red
            ax1.contourf(lon_grid, lat_grid, mask_2d, levels=[0.5, 1.5], colors=['red'], alpha=0.8)
            ax1.contour(lon_grid, lat_grid, mask_2d, levels=[0.5], colors=['darkred'], linewidths=3)
        else:
            # Other blocks in different colors
            ax1.contourf(lon_grid, lat_grid, mask_2d, levels=[0.5, 1.5], colors=[colors[i]], alpha=0.5)
    
    # Plot spatial buffer zone
    buffer_mask_2d = np.zeros_like(lat_grid, dtype=bool)
    buffer_samples = np.where(spatial_buffer_mask & (temporal_indices == 0))[0]
    
    for idx in buffer_samples:
        coord = spatial_coords[idx] 
        lat_idx = np.argmin(np.abs(lat_values - coord[0]))
        lon_idx = np.argmin(np.abs(lon_values - coord[1]))
        buffer_mask_2d[lat_idx, lon_idx] = True
    
    if np.any(buffer_mask_2d):
        ax1.contourf(lon_grid, lat_grid, buffer_mask_2d, levels=[0.5, 1.5], colors=['orange'], alpha=0.6)
        ax1.contour(lon_grid, lat_grid, buffer_mask_2d, levels=[0.5], colors=['darkorange'], linewidths=2, linestyles='--')
    
    ax1.set_xlabel('Longitude (°W)', fontweight='bold')
    ax1.set_ylabel('Latitude (°N)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.8, label='Test Block'),
        plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.6, label='Spatial Buffer'),
        plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.5, label='Other Blocks')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=14)
    
    # === Panel 2: Temporal Blocks ===
    ax2.set_title(f'(B) Temporal Blocks - Fold #{fold_id}', fontsize=22, fontweight='bold', pad=20)
    
    # Create timeline
    n_times = len(time_values)
    time_positions = np.arange(n_times)
    
    # Plot all temporal blocks
    temporal_blocks = fold_data['temporal_blocks']
    block_colors = plt.cm.Set2(np.linspace(0, 1, len(temporal_blocks)))
    
    for block_id, times in temporal_blocks.items():
        if len(times) > 0:
            y_pos = block_id
            
            if block_id == test_temporal_block_id:
                # Test block in red
                ax2.barh(y_pos, len(times), left=times.min(), height=0.8, 
                        color='red', alpha=0.8, edgecolor='darkred', linewidth=2)
            else:
                # Other blocks
                ax2.barh(y_pos, len(times), left=times.min(), height=0.8,
                        color=block_colors[block_id], alpha=0.6, edgecolor='black', linewidth=1)
    
    # Plot temporal buffer
    if test_temporal_block_id is not None:
        test_times = temporal_blocks[test_temporal_block_id]
        if len(test_times) > 0:
            buffer_start = test_times.min() - fold_data['cv_splitter'].temporal_buffer
            buffer_end = test_times.max() + fold_data['cv_splitter'].temporal_buffer
            
            # Left buffer
            if buffer_start >= 0:
                ax2.barh(test_temporal_block_id, fold_data['cv_splitter'].temporal_buffer, 
                        left=buffer_start, height=0.6, color='orange', alpha=0.6,
                        edgecolor='darkorange', linewidth=2, linestyle='--')
            
            # Right buffer  
            if buffer_end < n_times:
                ax2.barh(test_temporal_block_id, fold_data['cv_splitter'].temporal_buffer,
                        left=test_times.max() + 1, height=0.6, color='orange', alpha=0.6,
                        edgecolor='darkorange', linewidth=2, linestyle='--')
    
    ax2.set_xlabel('Time Index', fontweight='bold')
    ax2.set_ylabel('Temporal Block ID', fontweight='bold')
    ax2.set_yticks(range(len(temporal_blocks)))
    ax2.grid(True, alpha=0.3)
    
    # Add time labels
    time_labels = [f"{time_values[i].strftime('%Y-%m')}" for i in range(0, n_times, max(1, n_times//8))]
    time_positions_labels = list(range(0, n_times, max(1, n_times//8)))
    ax2.set_xticks(time_positions_labels)
    ax2.set_xticklabels(time_labels, rotation=45)
    
    # Legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.8, label='Test Period'),
        plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.6, label='Temporal Buffer'), 
        plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.6, label='Other Periods')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=14)
    
    # === Panel 3: Combined Spatiotemporal ===
    ax3.set_title(f'(C) Spatiotemporal Fold #{fold_id} Combined', fontsize=22, fontweight='bold', pad=20)
    
    # Create sample classification for visualization
    n_samples = len(spatial_coords)
    sample_types = np.full(n_samples, 0, dtype=int)  # 0 = training
    sample_types[test_idx] = 3  # 3 = test
    sample_types[spatial_buffer_mask] = 1  # 1 = spatial buffer
    sample_types[temporal_buffer_mask] = 2  # 2 = temporal buffer
    
    # Count samples by type
    type_counts = np.bincount(sample_types, minlength=4)
    type_labels = ['Training', 'Spatial Buffer', 'Temporal Buffer', 'Test']
    type_colors = ['lightblue', 'orange', 'yellow', 'red']
    
    # Create pie chart
    wedges, texts, autotexts = ax3.pie(type_counts, labels=type_labels, colors=type_colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 14})
    
    # Add sample counts to labels
    for i, (wedge, text, autotext) in enumerate(zip(wedges, texts, autotexts)):
        text.set_text(f'{type_labels[i]}\n({type_counts[i]:,} samples)')
        autotext.set_fontweight('bold')
    
    ax3.axis('equal')
    
    # === Panel 4: Statistics and Details ===
    ax4.set_title(f'(D) Fold #{fold_id} Details', fontsize=22, fontweight='bold', pad=20)
    ax4.axis('off')
    
    # Calculate statistics
    total_samples = len(spatial_coords)
    train_samples = len(train_idx)
    test_samples = len(test_idx)
    spatial_buffer_samples = np.sum(spatial_buffer_mask)
    temporal_buffer_samples = np.sum(temporal_buffer_mask)
    
    # Spatial info
    n_spatial_blocks = len(np.unique(spatial_blocks))
    spatial_buffer_deg = fold_data['cv_splitter'].spatial_buffer
    spatial_buffer_km = spatial_buffer_deg * 111  # Approximate conversion
    
    # Temporal info
    n_temporal_blocks = len(temporal_blocks)
    temporal_buffer_months = fold_data['cv_splitter'].temporal_buffer
    
    if test_temporal_block_id is not None:
        test_times = temporal_blocks[test_temporal_block_id]
        test_period_start = time_values[test_times.min()] if len(test_times) > 0 else "N/A"
        test_period_end = time_values[test_times.max()] if len(test_times) > 0 else "N/A"
    else:
        test_period_start = test_period_end = "N/A"
    
    # Create detailed text
    details_text = f"""
CROSS-VALIDATION CONFIGURATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Spatial Blocks: {n_spatial_blocks}
Temporal Blocks: {n_temporal_blocks} 
Spatial Buffer: {spatial_buffer_deg}° ({spatial_buffer_km:.0f} km)
Temporal Buffer: {temporal_buffer_months} months

FOLD #{fold_id} DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test Spatial Block: {test_spatial_block_id}
Test Temporal Block: {test_temporal_block_id}

Test Period: 
  {test_period_start.strftime('%Y-%m') if test_period_start != 'N/A' else 'N/A'} to {test_period_end.strftime('%Y-%m') if test_period_end != 'N/A' else 'N/A'}

SAMPLE DISTRIBUTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Samples: {total_samples:,}
Training: {train_samples:,} ({100*train_samples/total_samples:.1f}%)
Test: {test_samples:,} ({100*test_samples/total_samples:.1f}%)
Spatial Buffer: {spatial_buffer_samples:,} ({100*spatial_buffer_samples/total_samples:.1f}%)
Temporal Buffer: {temporal_buffer_samples:,} ({100*temporal_buffer_samples/total_samples:.1f}%)

ANTI-LEAKAGE MEASURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Spatial blocking prevents nearby pixels in train/test
✓ Temporal blocking prevents adjacent time periods
✓ Buffer zones create gaps around test regions
✓ No data leakage between training and testing
    """
    
    ax4.text(0.05, 0.95, details_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = output_dir / f"cv_fold_{fold_id}_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved CV visualization: {output_path}")
    
    # Also save PDF for presentations
    pdf_path = output_dir / f"cv_fold_{fold_id}_visualization.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved PDF version: {pdf_path}")
    
    plt.close()
    
    return output_path


def main():
    """Main execution function."""
    print("🔬 Spatiotemporal CV Fold Visualization")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path("figures_coarse_to_fine/cv_demonstration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load example data
        X, y, metadata = load_example_data()
        
        # Demonstrate one fold in detail
        fold_data = demonstrate_one_fold(X, y, metadata, fold_to_show=3)
        
        # Create visualization
        visualization_fig = create_cv_visualization(fold_data, output_dir)
        
        print("\n" + "=" * 60)
        print("✅ CV Fold Visualization Complete!")
        print(f"📁 Output saved to: {output_dir}")
        print(f"\n📖 Main presentation figure: {visualization_fig.name}")
        print("   This figure shows the complete spatiotemporal CV structure!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())