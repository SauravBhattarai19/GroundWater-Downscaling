"""
Real geographic visualization of coarse-to-fine mapping.
Shows actual 55km and 5km grids on maps.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from pathlib import Path

def load_datasets():
    """Load coarse and fine datasets."""
    print("📂 Loading datasets...")
    
    coarse_ds = xr.open_dataset('processed_coarse_to_fine/feature_stack_55km.nc')
    fine_ds = xr.open_dataset('processed_coarse_to_fine/feature_stack_all_5km.nc')
    grace_ds = xr.open_dataset('processed_coarse_to_fine/grace_filled_stl.nc')
    
    return coarse_ds, fine_ds, grace_ds


def create_mapping_counts(coarse_lat, coarse_lon, fine_lat, fine_lon):
    """Create mapping and return counts."""
    n_lat_coarse = len(coarse_lat)
    n_lon_coarse = len(coarse_lon)
    n_lat_fine = len(fine_lat)
    n_lon_fine = len(fine_lon)
    
    n_coarse_spatial = n_lat_coarse * n_lon_coarse
    n_fine_spatial = n_lat_fine * n_lon_fine
    
    # For each fine cell, find nearest coarse cell
    fine_to_coarse = np.zeros(n_fine_spatial, dtype=np.int32)
    
    for i_fine in range(n_lat_fine):
        lat_diffs = np.abs(coarse_lat - fine_lat[i_fine])
        nearest_lat_idx = np.argmin(lat_diffs)
        for j_fine in range(n_lon_fine):
            lon_diffs = np.abs(coarse_lon - fine_lon[j_fine])
            nearest_lon_idx = np.argmin(lon_diffs)
            fine_idx = i_fine * n_lon_fine + j_fine
            fine_to_coarse[fine_idx] = nearest_lat_idx * n_lon_coarse + nearest_lon_idx
    
    # Count fine cells per coarse
    spatial_mapping = [[] for _ in range(n_coarse_spatial)]
    for fine_idx in range(n_fine_spatial):
        coarse_idx = fine_to_coarse[fine_idx]
        spatial_mapping[coarse_idx].append(fine_idx)
    
    counts = np.array([len(m) for m in spatial_mapping])
    counts_2d = counts.reshape(n_lat_coarse, n_lon_coarse)
    
    return spatial_mapping, counts_2d


def plot_real_grids_overview(coarse_ds, fine_ds, grace_ds, counts_2d, save_path='figures_coarse_to_fine/real_grid_overview.png'):
    """Plot overview showing both grids with geographic context."""
    print("\n📊 Creating real grid overview...")
    
    coarse_lat = coarse_ds.lat.values
    coarse_lon = coarse_ds.lon.values
    fine_lat = fine_ds.lat.values
    fine_lon = fine_ds.lon.values
    
    # Get GRACE data for background
    grace_var = 'tws_anomaly' if 'tws_anomaly' in grace_ds else list(grace_ds.data_vars)[0]
    grace_mean = np.nanmean(grace_ds[grace_var].values, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: Coarse grid (55km) with fine cell counts
    ax1 = axes[0]
    
    # Plot fine cell counts as background
    im1 = ax1.imshow(counts_2d, cmap='RdYlGn', 
                     extent=[coarse_lon.min()-0.25, coarse_lon.max()+0.25,
                             coarse_lat.min()-0.25, coarse_lat.max()+0.25],
                     origin='upper' if coarse_lat[0] > coarse_lat[-1] else 'lower',
                     vmin=0, vmax=150)
    
    # Draw coarse grid lines
    for lat in coarse_lat:
        ax1.axhline(lat - 0.25, color='black', linewidth=0.5, alpha=0.3)
        ax1.axhline(lat + 0.25, color='black', linewidth=0.5, alpha=0.3)
    for lon in coarse_lon:
        ax1.axvline(lon - 0.25, color='black', linewidth=0.5, alpha=0.3)
        ax1.axvline(lon + 0.25, color='black', linewidth=0.5, alpha=0.3)
    
    plt.colorbar(im1, ax=ax1, label='Fine cells per coarse cell', shrink=0.8)
    
    ax1.set_xlabel('Longitude (°)', fontsize=12)
    ax1.set_ylabel('Latitude (°)', fontsize=12)
    ax1.set_title(f'55km Coarse Grid ({len(coarse_lat)}×{len(coarse_lon)} = {len(coarse_lat)*len(coarse_lon)} cells)\n'
                  f'Color = Number of 5km cells per 55km cell', fontsize=14)
    
    # Right: Fine grid extent
    ax2 = axes[1]
    
    # Show GRACE valid mask
    grace_valid = ~np.isnan(grace_mean)
    ax2.imshow(grace_valid.astype(float), cmap='Blues', alpha=0.3,
               extent=[coarse_lon.min()-0.25, coarse_lon.max()+0.25,
                       coarse_lat.min()-0.25, coarse_lat.max()+0.25],
               origin='upper' if coarse_lat[0] > coarse_lat[-1] else 'lower')
    
    # Draw fine grid extent
    rect = patches.Rectangle(
        (fine_lon.min(), fine_lat.min()),
        fine_lon.max() - fine_lon.min(),
        fine_lat.max() - fine_lat.min(),
        linewidth=3, edgecolor='red', facecolor='none', label='Fine grid extent'
    )
    ax2.add_patch(rect)
    
    # Draw coarse grid extent
    rect2 = patches.Rectangle(
        (coarse_lon.min()-0.25, coarse_lat.min()-0.25),
        (coarse_lon.max() - coarse_lon.min()) + 0.5,
        (coarse_lat.max() - coarse_lat.min()) + 0.5,
        linewidth=3, edgecolor='blue', facecolor='none', linestyle='--', label='Coarse grid extent'
    )
    ax2.add_patch(rect2)
    
    ax2.set_xlabel('Longitude (°)', fontsize=12)
    ax2.set_ylabel('Latitude (°)', fontsize=12)
    ax2.set_title(f'Grid Extent Comparison\n'
                  f'Coarse: [{coarse_lat.min():.2f}°, {coarse_lat.max():.2f}°] × [{coarse_lon.min():.2f}°, {coarse_lon.max():.2f}°]\n'
                  f'Fine: [{fine_lat.min():.2f}°, {fine_lat.max():.2f}°] × [{fine_lon.min():.2f}°, {fine_lon.max():.2f}°]', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(coarse_lon.min()-1, coarse_lon.max()+1)
    ax2.set_ylim(coarse_lat.min()-1, coarse_lat.max()+1)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    plt.close()


def plot_min_max_cells(coarse_ds, fine_ds, counts_2d, spatial_mapping, save_path='figures_coarse_to_fine/real_min_max_cells.png'):
    """Plot actual locations with minimum and maximum fine cells."""
    print("\n📊 Creating min/max cell visualization...")
    
    coarse_lat = coarse_ds.lat.values
    coarse_lon = coarse_ds.lon.values
    fine_lat = fine_ds.lat.values
    fine_lon = fine_ds.lon.values
    
    n_lon_coarse = len(coarse_lon)
    n_lon_fine = len(fine_lon)
    
    counts_flat = counts_2d.flatten()
    min_count = counts_flat.min()
    max_count = counts_flat.max()
    
    # Find all cells with min and max counts
    min_indices = np.where(counts_flat == min_count)[0]
    max_indices = np.where(counts_flat == max_count)[0]
    
    fig = plt.figure(figsize=(24, 16))
    
    # Top row: Overall map with min/max highlighted
    ax_main = fig.add_subplot(2, 2, (1, 2))
    
    im = ax_main.imshow(counts_2d, cmap='RdYlGn',
                        extent=[coarse_lon.min()-0.25, coarse_lon.max()+0.25,
                                coarse_lat.min()-0.25, coarse_lat.max()+0.25],
                        origin='upper' if coarse_lat[0] > coarse_lat[-1] else 'lower',
                        vmin=0, vmax=150)
    
    # Mark minimum cells (red X)
    for idx in min_indices:
        lat_idx = idx // n_lon_coarse
        lon_idx = idx % n_lon_coarse
        ax_main.scatter(coarse_lon[lon_idx], coarse_lat[lat_idx], 
                       s=200, c='red', marker='X', linewidths=2, edgecolors='black',
                       label='Min' if idx == min_indices[0] else '')
    
    # Mark maximum cells (blue star)
    for idx in max_indices[:5]:  # Show first 5 max cells
        lat_idx = idx // n_lon_coarse
        lon_idx = idx % n_lon_coarse
        ax_main.scatter(coarse_lon[lon_idx], coarse_lat[lat_idx], 
                       s=200, c='blue', marker='*', linewidths=1, edgecolors='black',
                       label='Max' if idx == max_indices[0] else '')
    
    plt.colorbar(im, ax=ax_main, label='Fine cells per coarse cell', shrink=0.6)
    ax_main.set_xlabel('Longitude (°)', fontsize=12)
    ax_main.set_ylabel('Latitude (°)', fontsize=12)
    ax_main.set_title(f'Fine Cells per Coarse Cell\n'
                      f'Red X = Min ({min_count} cells, n={len(min_indices)}), '
                      f'Blue ★ = Max ({max_count} cells, n={len(max_indices)})', fontsize=14)
    ax_main.legend(loc='upper right')
    
    # Bottom left: Zoom on a MIN cell
    ax_min = fig.add_subplot(2, 2, 3)
    
    min_idx = min_indices[0]
    min_lat_idx = min_idx // n_lon_coarse
    min_lon_idx = min_idx % n_lon_coarse
    min_c_lat = coarse_lat[min_lat_idx]
    min_c_lon = coarse_lon[min_lon_idx]
    
    # Get fine cells for this coarse cell
    fine_indices_min = spatial_mapping[min_idx]
    fine_lats_min = [fine_lat[f // n_lon_fine] for f in fine_indices_min]
    fine_lons_min = [fine_lon[f % n_lon_fine] for f in fine_indices_min]
    
    # Draw coarse cell
    rect_min = patches.Rectangle(
        (min_c_lon - 0.25, min_c_lat - 0.25), 0.5, 0.5,
        linewidth=3, edgecolor='blue', facecolor='lightblue', alpha=0.3
    )
    ax_min.add_patch(rect_min)
    
    # Draw fine cells as points
    if fine_lons_min:
        ax_min.scatter(fine_lons_min, fine_lats_min, s=20, c='red', alpha=0.8, label=f'{len(fine_indices_min)} fine cells')
    
    # Draw coarse cell center
    ax_min.scatter([min_c_lon], [min_c_lat], s=150, c='blue', marker='X', linewidths=2)
    
    ax_min.set_xlim(min_c_lon - 0.35, min_c_lon + 0.35)
    ax_min.set_ylim(min_c_lat - 0.35, min_c_lat + 0.35)
    ax_min.set_xlabel('Longitude (°)', fontsize=10)
    ax_min.set_ylabel('Latitude (°)', fontsize=10)
    ax_min.set_title(f'MINIMUM: {min_count} fine cells\n'
                     f'Location: ({min_c_lat:.2f}°N, {min_c_lon:.2f}°E)\n'
                     f'This is at the EDGE of the fine grid!', fontsize=12)
    ax_min.legend()
    ax_min.set_aspect('equal')
    ax_min.grid(True, alpha=0.3)
    
    # Bottom right: Zoom on a MAX cell  
    ax_max = fig.add_subplot(2, 2, 4)
    
    max_idx = max_indices[0]
    max_lat_idx = max_idx // n_lon_coarse
    max_lon_idx = max_idx % n_lon_coarse
    max_c_lat = coarse_lat[max_lat_idx]
    max_c_lon = coarse_lon[max_lon_idx]
    
    # Get fine cells for this coarse cell
    fine_indices_max = spatial_mapping[max_idx]
    fine_lats_max = [fine_lat[f // n_lon_fine] for f in fine_indices_max]
    fine_lons_max = [fine_lon[f % n_lon_fine] for f in fine_indices_max]
    
    # Draw coarse cell
    rect_max = patches.Rectangle(
        (max_c_lon - 0.25, max_c_lat - 0.25), 0.5, 0.5,
        linewidth=3, edgecolor='blue', facecolor='lightblue', alpha=0.3
    )
    ax_max.add_patch(rect_max)
    
    # Draw fine cells as points
    ax_max.scatter(fine_lons_max, fine_lats_max, s=20, c='green', alpha=0.8, label=f'{len(fine_indices_max)} fine cells')
    
    # Draw coarse cell center
    ax_max.scatter([max_c_lon], [max_c_lat], s=150, c='blue', marker='*', linewidths=2)
    
    ax_max.set_xlim(max_c_lon - 0.35, max_c_lon + 0.35)
    ax_max.set_ylim(max_c_lat - 0.35, max_c_lat + 0.35)
    ax_max.set_xlabel('Longitude (°)', fontsize=10)
    ax_max.set_ylabel('Latitude (°)', fontsize=10)
    ax_max.set_title(f'MAXIMUM: {max_count} fine cells\n'
                     f'Location: ({max_c_lat:.2f}°N, {max_c_lon:.2f}°E)\n'
                     f'This is in the INTERIOR of the grid', fontsize=12)
    ax_max.legend()
    ax_max.set_aspect('equal')
    ax_max.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    plt.close()
    
    # Print locations
    print(f"\n   MIN locations ({min_count} fine cells):")
    for idx in min_indices:
        lat_idx = idx // n_lon_coarse
        lon_idx = idx % n_lon_coarse
        print(f"      ({coarse_lat[lat_idx]:.2f}°, {coarse_lon[lon_idx]:.2f}°)")
    
    print(f"\n   MAX locations ({max_count} fine cells):")
    for idx in max_indices[:5]:
        lat_idx = idx // n_lon_coarse
        lon_idx = idx % n_lon_coarse
        print(f"      ({coarse_lat[lat_idx]:.2f}°, {coarse_lon[lon_idx]:.2f}°)")


def plot_edge_boundary_analysis(coarse_ds, fine_ds, counts_2d, save_path='figures_coarse_to_fine/real_edge_boundary.png'):
    """Analyze why edge cells have fewer fine cells."""
    print("\n📊 Creating edge/boundary analysis...")
    
    coarse_lat = coarse_ds.lat.values
    coarse_lon = coarse_ds.lon.values
    fine_lat = fine_ds.lat.values
    fine_lon = fine_ds.lon.values
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Calculate fine grid extent vs coarse
    coarse_lat_min, coarse_lat_max = coarse_lat.min() - 0.25, coarse_lat.max() + 0.25
    coarse_lon_min, coarse_lon_max = coarse_lon.min() - 0.25, coarse_lon.max() + 0.25
    fine_lat_min, fine_lat_max = fine_lat.min(), fine_lat.max()
    fine_lon_min, fine_lon_max = fine_lon.min(), fine_lon.max()
    
    # 1. Top edge
    ax1 = axes[0, 0]
    zoom_lat = coarse_lat[0]  # Top row
    zoom_lon_start, zoom_lon_end = coarse_lon[30], coarse_lon[40]  # Middle section
    
    # Draw coarse cells in this zoom
    for i, lon in enumerate(coarse_lon[30:41]):
        rect = patches.Rectangle(
            (lon - 0.25, zoom_lat - 0.25), 0.5, 0.5,
            linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.3
        )
        ax1.add_patch(rect)
        # Label with count
        lat_idx = 0
        lon_idx = 30 + i
        count = counts_2d[lat_idx, lon_idx]
        ax1.text(lon, zoom_lat, str(count), ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw fine grid boundary
    ax1.axhline(fine_lat_max, color='red', linewidth=2, linestyle='--', label=f'Fine grid edge ({fine_lat_max:.2f}°)')
    ax1.axhline(coarse_lat_max + 0.25, color='blue', linewidth=2, linestyle=':', label=f'Coarse grid edge ({coarse_lat_max + 0.25:.2f}°)')
    
    ax1.set_xlim(zoom_lon_start - 0.5, zoom_lon_end + 0.5)
    ax1.set_ylim(zoom_lat - 0.5, zoom_lat + 0.7)
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.set_title(f'TOP EDGE: Fine grid ends at {fine_lat_max:.2f}°\nCoarse grid extends to {coarse_lat_max + 0.25:.2f}°')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_aspect('equal')
    
    # 2. Right edge
    ax2 = axes[0, 1]
    zoom_lon = coarse_lon[-1]  # Right column
    zoom_lat_start, zoom_lat_end = coarse_lat[20], coarse_lat[30]
    
    for i, lat in enumerate(coarse_lat[20:31]):
        rect = patches.Rectangle(
            (zoom_lon - 0.25, lat - 0.25), 0.5, 0.5,
            linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.3
        )
        ax2.add_patch(rect)
        lat_idx = 20 + i
        lon_idx = len(coarse_lon) - 1
        count = counts_2d[lat_idx, lon_idx]
        ax2.text(zoom_lon, lat, str(count), ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax2.axvline(fine_lon_max, color='red', linewidth=2, linestyle='--', label=f'Fine grid edge ({fine_lon_max:.2f}°)')
    ax2.axvline(coarse_lon_max + 0.25, color='blue', linewidth=2, linestyle=':', label=f'Coarse grid edge ({coarse_lon_max + 0.25:.2f}°)')
    
    ax2.set_xlim(zoom_lon - 0.5, zoom_lon + 0.7)
    ax2.set_ylim(zoom_lat_end - 0.5, zoom_lat_start + 0.5)
    ax2.set_xlabel('Longitude (°)')
    ax2.set_ylabel('Latitude (°)')
    ax2.set_title(f'RIGHT EDGE: Fine grid ends at {fine_lon_max:.2f}°\nCoarse grid extends to {coarse_lon_max + 0.25:.2f}°')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_aspect('equal')
    
    # 3. Summary statistics by position
    ax3 = axes[1, 0]
    
    # Calculate mean counts by position
    edge_types = {
        'Top edge': counts_2d[0, :].mean(),
        'Bottom edge': counts_2d[-1, :].mean(),
        'Left edge': counts_2d[:, 0].mean(),
        'Right edge': counts_2d[:, -1].mean(),
        'Corners': np.mean([counts_2d[0,0], counts_2d[0,-1], counts_2d[-1,0], counts_2d[-1,-1]]),
        'Interior': counts_2d[1:-1, 1:-1].mean(),
    }
    
    colors = ['red', 'red', 'red', 'red', 'darkred', 'green']
    bars = ax3.bar(edge_types.keys(), edge_types.values(), color=colors, edgecolor='black')
    ax3.axhline(121, color='blue', linestyle='--', linewidth=2, label='Expected (11×11=121)')
    ax3.set_ylabel('Mean fine cells per coarse cell')
    ax3.set_title('Average Fine Cells by Grid Position')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, edge_types.values()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Text explanation
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    explanation = f"""
    WHY EDGE CELLS HAVE FEWER FINE CELLS:
    
    The fine (5km) grid has a SMALLER extent than the coarse (55km) grid:
    
    LATITUDE:
      • Coarse: {coarse_lat_min:.2f}° to {coarse_lat_max:.2f}°
      • Fine:   {fine_lat_min:.2f}° to {fine_lat_max:.2f}°
      • Difference: ~{(coarse_lat_max - fine_lat_max):.2f}° at top, ~{(fine_lat_min - coarse_lat_min):.2f}° at bottom
    
    LONGITUDE:
      • Coarse: {coarse_lon_min:.2f}° to {coarse_lon_max:.2f}°
      • Fine:   {fine_lon_min:.2f}° to {fine_lon_max:.2f}°
      • Difference: ~{(fine_lon_min - coarse_lon_min):.2f}° at left, ~{(coarse_lon_max - fine_lon_max):.2f}° at right
    
    IMPACT:
      • Edge coarse cells only partially overlap with fine grid
      • Corner cells have the least overlap (min = {counts_2d.min()} cells)
      • Interior cells have full overlap (~{counts_2d[1:-1, 1:-1].mean():.0f} cells)
    
    THIS IS EXPECTED for land-only analysis where:
      • Fine data may be clipped to land boundaries
      • Coarse GRACE data extends slightly over ocean
    """
    
    ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    plt.close()


def main():
    print("="*70)
    print("🔍 REAL GRID VISUALIZATION")
    print("="*70)
    
    # Load data
    coarse_ds, fine_ds, grace_ds = load_datasets()
    
    coarse_lat = coarse_ds.lat.values
    coarse_lon = coarse_ds.lon.values
    fine_lat = fine_ds.lat.values
    fine_lon = fine_ds.lon.values
    
    # Create mapping
    print("\n🗺️ Creating mapping...")
    spatial_mapping, counts_2d = create_mapping_counts(coarse_lat, coarse_lon, fine_lat, fine_lon)
    
    # Statistics
    print(f"\n📊 Statistics:")
    print(f"   Coarse grid: {len(coarse_lat)} × {len(coarse_lon)} = {len(coarse_lat)*len(coarse_lon)} cells")
    print(f"   Fine grid: {len(fine_lat)} × {len(fine_lon)} = {len(fine_lat)*len(fine_lon)} cells")
    print(f"   Fine cells per coarse: min={counts_2d.min()}, max={counts_2d.max()}, mean={counts_2d.mean():.1f}")
    
    # Create visualizations
    plot_real_grids_overview(coarse_ds, fine_ds, grace_ds, counts_2d)
    plot_min_max_cells(coarse_ds, fine_ds, counts_2d, spatial_mapping)
    plot_edge_boundary_analysis(coarse_ds, fine_ds, counts_2d)
    
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE")
    print("="*70)
    print("\nFigures saved to: figures_coarse_to_fine/")
    print("   - real_grid_overview.png")
    print("   - real_min_max_cells.png")  
    print("   - real_edge_boundary.png")


if __name__ == "__main__":
    main()

