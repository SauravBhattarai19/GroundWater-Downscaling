"""
Visualization script to check coarse-to-fine mapping quality.
Verifies that fine cells are correctly mapped to coarse cells.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Add project path
import sys
sys.path.insert(0, str(Path(__file__).parent))

def load_datasets():
    """Load coarse and fine datasets."""
    print("📂 Loading datasets...")
    
    coarse_ds = xr.open_dataset('processed_coarse_to_fine/feature_stack_55km.nc')
    fine_ds = xr.open_dataset('processed_coarse_to_fine/feature_stack_all_5km.nc')
    grace_ds = xr.open_dataset('processed_coarse_to_fine/grace_filled_stl.nc')
    
    print(f"   Coarse: {len(coarse_ds.lat)} × {len(coarse_ds.lon)} = {len(coarse_ds.lat) * len(coarse_ds.lon)} cells")
    print(f"   Fine: {len(fine_ds.lat)} × {len(fine_ds.lon)} = {len(fine_ds.lat) * len(fine_ds.lon)} cells")
    
    return coarse_ds, fine_ds, grace_ds


def create_mapping(coarse_lat, coarse_lon, fine_lat, fine_lon):
    """Create coarse-to-fine mapping using nearest neighbor."""
    print("\n🗺️ Creating coarse-to-fine mapping...")
    
    n_lat_coarse = len(coarse_lat)
    n_lon_coarse = len(coarse_lon)
    n_lat_fine = len(fine_lat)
    n_lon_fine = len(fine_lon)
    
    n_coarse_spatial = n_lat_coarse * n_lon_coarse
    n_fine_spatial = n_lat_fine * n_lon_fine
    
    # For each fine cell, find nearest coarse cell
    fine_to_coarse_lat = np.zeros(n_fine_spatial, dtype=np.int32)
    fine_to_coarse_lon = np.zeros(n_fine_spatial, dtype=np.int32)
    
    for i_fine in range(n_lat_fine):
        lat_diffs = np.abs(coarse_lat - fine_lat[i_fine])
        nearest_lat_idx = np.argmin(lat_diffs)
        for j_fine in range(n_lon_fine):
            fine_idx = i_fine * n_lon_fine + j_fine
            fine_to_coarse_lat[fine_idx] = nearest_lat_idx
    
    for j_fine in range(n_lon_fine):
        lon_diffs = np.abs(coarse_lon - fine_lon[j_fine])
        nearest_lon_idx = np.argmin(lon_diffs)
        for i_fine in range(n_lat_fine):
            fine_idx = i_fine * n_lon_fine + j_fine
            fine_to_coarse_lon[fine_idx] = nearest_lon_idx
    
    # Convert to flattened coarse index
    fine_to_coarse_spatial = fine_to_coarse_lat * n_lon_coarse + fine_to_coarse_lon
    
    # Invert: for each coarse cell, get list of fine cells
    spatial_mapping = [[] for _ in range(n_coarse_spatial)]
    for fine_idx in range(n_fine_spatial):
        coarse_idx = fine_to_coarse_spatial[fine_idx]
        spatial_mapping[coarse_idx].append(fine_idx)
    
    # Count fine cells per coarse cell
    counts = np.array([len(m) for m in spatial_mapping])
    counts_2d = counts.reshape(n_lat_coarse, n_lon_coarse)
    
    return spatial_mapping, counts_2d, fine_to_coarse_lat.reshape(n_lat_fine, n_lon_fine), fine_to_coarse_lon.reshape(n_lat_fine, n_lon_fine)


def plot_fine_cells_per_coarse(counts_2d, coarse_lat, coarse_lon, save_path='figures_coarse_to_fine/mapping_fine_per_coarse.png'):
    """Plot heatmap of fine cells per coarse cell."""
    print("\n📊 Creating fine cells per coarse cell heatmap...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create meshgrid for plotting
    lon_edges = np.concatenate([coarse_lon - 0.25, [coarse_lon[-1] + 0.25]])
    lat_edges = np.concatenate([coarse_lat + 0.25, [coarse_lat[-1] - 0.25]])
    
    # Plot heatmap
    im = ax.pcolormesh(lon_edges, lat_edges, counts_2d, cmap='RdYlGn', vmin=0, vmax=144)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Number of fine (5km) cells per coarse (55km) cell')
    
    # Highlight cells with low counts
    low_count_mask = counts_2d < 100
    if np.any(low_count_mask):
        low_i, low_j = np.where(low_count_mask)
        for i, j in zip(low_i, low_j):
            rect = patches.Rectangle(
                (coarse_lon[j] - 0.25, coarse_lat[i] - 0.25),
                0.5, 0.5,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Fine Cells per Coarse Cell\nMin={counts_2d.min()}, Max={counts_2d.max()}, Mean={counts_2d.mean():.1f}\nRed boxes: <100 fine cells (edge/boundary)')
    
    # Add grid
    ax.set_xlim(coarse_lon.min() - 0.5, coarse_lon.max() + 0.5)
    ax.set_ylim(coarse_lat.min() - 0.5, coarse_lat.max() + 0.5)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    plt.close()


def plot_example_coarse_cells(spatial_mapping, coarse_lat, coarse_lon, fine_lat, fine_lon, save_path='figures_coarse_to_fine/mapping_example_cells.png'):
    """Plot example coarse cells with their fine cells."""
    print("\n📊 Creating example coarse cell visualizations...")
    
    n_lat_coarse = len(coarse_lat)
    n_lon_coarse = len(coarse_lon)
    n_lat_fine = len(fine_lat)
    n_lon_fine = len(fine_lon)
    
    # Select example cells: corners, center, and one with low count
    counts = np.array([len(m) for m in spatial_mapping])
    
    examples = {
        'Top-Left (0,0)': 0,
        'Top-Right (0,72)': n_lon_coarse - 1,
        'Bottom-Left (45,0)': (n_lat_coarse - 1) * n_lon_coarse,
        'Bottom-Right (45,72)': len(spatial_mapping) - 1,
        'Center': (n_lat_coarse // 2) * n_lon_coarse + (n_lon_coarse // 2),
        f'Min Count ({counts.min()} cells)': np.argmin(counts),
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (title, coarse_idx) in enumerate(examples.items()):
        ax = axes[idx]
        
        # Get coarse cell coordinates
        coarse_lat_idx = coarse_idx // n_lon_coarse
        coarse_lon_idx = coarse_idx % n_lon_coarse
        
        c_lat = coarse_lat[coarse_lat_idx]
        c_lon = coarse_lon[coarse_lon_idx]
        
        # Get fine cell indices
        fine_indices = spatial_mapping[coarse_idx]
        n_fine = len(fine_indices)
        
        # Convert fine indices to lat/lon
        fine_lats = []
        fine_lons = []
        for f_idx in fine_indices:
            f_lat_idx = f_idx // n_lon_fine
            f_lon_idx = f_idx % n_lon_fine
            fine_lats.append(fine_lat[f_lat_idx])
            fine_lons.append(fine_lon[f_lon_idx])
        
        # Plot coarse cell boundary
        coarse_half = 0.25  # Half of 0.5° spacing
        rect = patches.Rectangle(
            (c_lon - coarse_half, c_lat - coarse_half),
            0.5, 0.5,
            linewidth=3, edgecolor='blue', facecolor='lightblue', alpha=0.3, label='Coarse cell'
        )
        ax.add_patch(rect)
        
        # Plot fine cells
        if fine_lons:
            ax.scatter(fine_lons, fine_lats, s=10, c='red', alpha=0.7, label=f'Fine cells ({n_fine})')
        
        # Plot coarse cell center
        ax.scatter([c_lon], [c_lat], s=100, c='blue', marker='x', linewidths=3, label='Coarse center')
        
        ax.set_xlim(c_lon - 0.4, c_lon + 0.4)
        ax.set_ylim(c_lat - 0.4, c_lat + 0.4)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{title}\nCoarse: ({c_lat:.2f}°, {c_lon:.2f}°)\n{n_fine} fine cells')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    plt.close()


def plot_edge_analysis(counts_2d, coarse_lat, coarse_lon, grace_ds, save_path='figures_coarse_to_fine/mapping_edge_analysis.png'):
    """Analyze edge cells and compare with GRACE valid data."""
    print("\n📊 Creating edge analysis visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Fine cells per coarse
    ax1 = axes[0]
    im1 = ax1.imshow(counts_2d, cmap='RdYlGn', vmin=0, vmax=144, 
                     extent=[coarse_lon.min()-0.25, coarse_lon.max()+0.25, 
                             coarse_lat.min()-0.25, coarse_lat.max()+0.25],
                     origin='upper' if coarse_lat[0] > coarse_lat[-1] else 'lower')
    plt.colorbar(im1, ax=ax1, label='Fine cells')
    ax1.set_title('Fine Cells per Coarse Cell')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # 2. GRACE valid data mask (first timestep)
    ax2 = axes[1]
    grace_var = 'tws_anomaly' if 'tws_anomaly' in grace_ds else list(grace_ds.data_vars)[0]
    grace_data = grace_ds[grace_var].values[0]  # First timestep
    grace_valid = ~np.isnan(grace_data)
    
    im2 = ax2.imshow(grace_valid.astype(float), cmap='RdYlGn', vmin=0, vmax=1,
                     extent=[coarse_lon.min()-0.25, coarse_lon.max()+0.25, 
                             coarse_lat.min()-0.25, coarse_lat.max()+0.25],
                     origin='upper' if coarse_lat[0] > coarse_lat[-1] else 'lower')
    plt.colorbar(im2, ax=ax2, label='Valid (1) / NaN (0)')
    ax2.set_title('GRACE Valid Data Mask (t=0)')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    
    # 3. Low count cells overlaid on GRACE
    ax3 = axes[2]
    low_count = counts_2d < 100
    
    im3 = ax3.imshow(grace_valid.astype(float), cmap='Blues', vmin=0, vmax=1, alpha=0.5,
                     extent=[coarse_lon.min()-0.25, coarse_lon.max()+0.25, 
                             coarse_lat.min()-0.25, coarse_lat.max()+0.25],
                     origin='upper' if coarse_lat[0] > coarse_lat[-1] else 'lower')
    
    # Overlay low count cells
    im3b = ax3.imshow(np.where(low_count, 1, np.nan), cmap='Reds', vmin=0, vmax=1, alpha=0.7,
                      extent=[coarse_lon.min()-0.25, coarse_lon.max()+0.25, 
                              coarse_lat.min()-0.25, coarse_lat.max()+0.25],
                      origin='upper' if coarse_lat[0] > coarse_lat[-1] else 'lower')
    
    ax3.set_title(f'Low Count Cells (<100) Overlay\n{np.sum(low_count)} cells with partial coverage')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    plt.close()


def plot_distribution_histogram(counts_2d, save_path='figures_coarse_to_fine/mapping_distribution.png'):
    """Plot histogram of fine cells per coarse cell."""
    print("\n📊 Creating distribution histogram...")
    
    counts_flat = counts_2d.flatten()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bins = np.arange(0, 160, 5)
    n, bins_out, patches_out = ax.hist(counts_flat, bins=bins, edgecolor='black', alpha=0.7)
    
    # Color bars by value
    for patch, left_edge in zip(patches_out, bins_out[:-1]):
        if left_edge < 100:
            patch.set_facecolor('red')
        elif left_edge < 115:
            patch.set_facecolor('orange')
        else:
            patch.set_facecolor('green')
    
    ax.axvline(121, color='blue', linestyle='--', linewidth=2, label='Expected (11×11=121)')
    ax.axvline(counts_flat.mean(), color='purple', linestyle='-', linewidth=2, label=f'Mean ({counts_flat.mean():.1f})')
    
    ax.set_xlabel('Number of Fine Cells per Coarse Cell')
    ax.set_ylabel('Count of Coarse Cells')
    ax.set_title(f'Distribution of Fine Cells per Coarse Cell\nMin={counts_flat.min()}, Max={counts_flat.max()}, Mean={counts_flat.mean():.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotations
    low_count = np.sum(counts_flat < 100)
    ax.text(0.02, 0.98, f'Low coverage (<100): {low_count} cells ({100*low_count/len(counts_flat):.1f}%)',
            transform=ax.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    plt.close()


def print_statistics(counts_2d, coarse_lat, coarse_lon):
    """Print detailed statistics about the mapping."""
    print("\n" + "="*70)
    print("📊 MAPPING STATISTICS")
    print("="*70)
    
    counts_flat = counts_2d.flatten()
    
    print(f"\nOverall Statistics:")
    print(f"   Total coarse cells: {len(counts_flat):,}")
    print(f"   Min fine cells: {counts_flat.min()}")
    print(f"   Max fine cells: {counts_flat.max()}")
    print(f"   Mean fine cells: {counts_flat.mean():.2f}")
    print(f"   Median fine cells: {np.median(counts_flat):.0f}")
    print(f"   Expected (11×11): 121")
    
    print(f"\nCoverage Analysis:")
    for threshold in [50, 80, 100, 110, 120]:
        n_below = np.sum(counts_flat < threshold)
        print(f"   Cells with <{threshold} fine: {n_below} ({100*n_below/len(counts_flat):.1f}%)")
    
    print(f"\nEdge Analysis:")
    # Check edges
    top_edge = counts_2d[0, :]
    bottom_edge = counts_2d[-1, :]
    left_edge = counts_2d[:, 0]
    right_edge = counts_2d[:, -1]
    
    print(f"   Top edge: min={top_edge.min()}, mean={top_edge.mean():.1f}")
    print(f"   Bottom edge: min={bottom_edge.min()}, mean={bottom_edge.mean():.1f}")
    print(f"   Left edge: min={left_edge.min()}, mean={left_edge.mean():.1f}")
    print(f"   Right edge: min={right_edge.min()}, mean={right_edge.mean():.1f}")
    
    # Interior
    interior = counts_2d[1:-1, 1:-1]
    print(f"   Interior: min={interior.min()}, mean={interior.mean():.1f}")
    
    print(f"\nCells with minimum count ({counts_flat.min()}):")
    min_indices = np.where(counts_2d == counts_flat.min())
    for i, j in zip(min_indices[0][:5], min_indices[1][:5]):
        print(f"   ({coarse_lat[i]:.2f}°, {coarse_lon[j]:.2f}°)")


def main():
    print("="*70)
    print("🔍 COARSE-TO-FINE MAPPING VERIFICATION")
    print("="*70)
    
    # Load data
    coarse_ds, fine_ds, grace_ds = load_datasets()
    
    coarse_lat = coarse_ds.lat.values
    coarse_lon = coarse_ds.lon.values
    fine_lat = fine_ds.lat.values
    fine_lon = fine_ds.lon.values
    
    # Create mapping
    spatial_mapping, counts_2d, _, _ = create_mapping(coarse_lat, coarse_lon, fine_lat, fine_lon)
    
    # Print statistics
    print_statistics(counts_2d, coarse_lat, coarse_lon)
    
    # Create visualizations
    plot_fine_cells_per_coarse(counts_2d, coarse_lat, coarse_lon)
    plot_example_coarse_cells(spatial_mapping, coarse_lat, coarse_lon, fine_lat, fine_lon)
    plot_edge_analysis(counts_2d, coarse_lat, coarse_lon, grace_ds)
    plot_distribution_histogram(counts_2d)
    
    print("\n" + "="*70)
    print("✅ VERIFICATION COMPLETE")
    print("="*70)
    print("\nFigures saved to: figures_coarse_to_fine/")
    print("   - mapping_fine_per_coarse.png")
    print("   - mapping_example_cells.png")
    print("   - mapping_edge_analysis.png")
    print("   - mapping_distribution.png")


if __name__ == "__main__":
    main()

