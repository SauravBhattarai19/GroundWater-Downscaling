#!/usr/bin/env python3
"""
GRACE Original vs Downscaled Comparison Analysis
================================================

This script compares the downloaded original GRACE Land data with the 
downscaled data to assess how close they are.

Author: Assistant
Date: July 2025
"""

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GRACEComparisonAnalysis:
    """Compare original GRACE data with downscaled results using downloaded files."""
    
    def __init__(self, results_dir="results", data_dir="data"):
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        self.grace_original_dir = Path("data/raw/grace_original")
        self.figures_dir = Path("figures") / "grace_comparison"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
    def load_original_grace_data(self):
        """Load the downloaded original GRACE data."""
        print("📂 Loading original GRACE data...")
        
        # Find all GRACE files
        grace_files = sorted(glob.glob(str(self.grace_original_dir / "*.tif")))
        
        if not grace_files:
            print(f"   ❌ No GRACE files found in {self.grace_original_dir}")
            print("   Run download_grace_original.py first!")
            return None
        
        print(f"   📊 Found {len(grace_files)} GRACE files")
        
        # Load and combine all files
        grace_datasets = []
        dates = []
        
        for file_path in grace_files:
            try:
                # Extract date from filename
                filename = Path(file_path).stem
                # Expected format: YYYYMMDD_YYYYMMDD.tif (start_date_end_date)
                
                if '_' in filename and len(filename.split('_')) == 2:
                    start_date_str, end_date_str = filename.split('_')
                    
                    # Check if both parts are 8-digit dates
                    if (len(start_date_str) == 8 and start_date_str.isdigit() and 
                        len(end_date_str) == 8 and end_date_str.isdigit()):
                        
                        # Parse start and end dates
                        start_year = int(start_date_str[:4])
                        start_month = int(start_date_str[4:6])
                        start_day = int(start_date_str[6:8])
                        
                        end_year = int(end_date_str[:4])
                        end_month = int(end_date_str[4:6])
                        end_day = int(end_date_str[6:8])
                        
                        # Use the center date of the period
                        start_date = datetime(start_year, start_month, start_day)
                        end_date = datetime(end_year, end_month, end_day)
                        center_date = start_date + (end_date - start_date) / 2
                        
                        # Round to the nearest month (first day of month)
                        center_date = datetime(center_date.year, center_date.month, 1)
                        
                        dates.append(center_date)
                        
                        # Load the dataset
                        ds = xr.open_dataset(file_path, engine='rasterio')
                        ds = ds.squeeze()  # Remove any singleton dimensions
                        grace_datasets.append(ds)
                        
                        if len(dates) <= 5:  # Show first few for debugging
                            print(f"   📅 File: {filename} -> Date: {center_date.strftime('%Y-%m-%d')}")
                    else:
                        print(f"   ⚠️ Invalid date format in filename: {filename}")
                else:
                    print(f"   ⚠️ Could not extract date from {filename}")
                    
            except Exception as e:
                print(f"   ⚠️ Error loading {file_path}: {e}")
                continue
        
        if not grace_datasets:
            print("   ❌ No valid GRACE datasets loaded")
            return None
        
        # Combine into a single dataset with time dimension
        print("   🔄 Combining GRACE datasets...")
        
        # Rename coordinate names to match common conventions
        for i, ds in enumerate(grace_datasets):
            # Common coordinate name variations
            coord_mapping = {}
            for coord in ds.coords:
                if coord.lower() in ['x', 'longitude', 'lon']:
                    coord_mapping[coord] = 'lon'
                elif coord.lower() in ['y', 'latitude', 'lat']:
                    coord_mapping[coord] = 'lat'
            
            if coord_mapping:
                grace_datasets[i] = ds.rename(coord_mapping)
        
        # Stack along time dimension
        combined_grace = xr.concat(grace_datasets, dim='time')
        combined_grace = combined_grace.assign_coords(time=dates)
        
        # Ensure we have the right variable name
        var_names = list(combined_grace.data_vars)
        if len(var_names) == 1:
            grace_var = var_names[0]
        else:
            # Look for LWE variable
            grace_var = None
            for var in var_names:
                if 'lwe' in var.lower() or 'thickness' in var.lower() or 'mean' in var.lower():
                    grace_var = var
                    break
            if grace_var is None:
                grace_var = var_names[0]  # Use first variable as fallback
        
        self.grace_original = combined_grace[grace_var]
        print(f"   ✅ Loaded {len(dates)} GRACE timesteps")
        print(f"   📊 Using variable: {grace_var}")
        print(f"   🗓️ Time range: {min(dates).strftime('%Y-%m')} to {max(dates).strftime('%Y-%m')}")
        
        return self.grace_original
    
    def load_downscaled_data(self):
        """Load the downscaled data from results."""
        print("📂 Loading downscaled data...")
        
        # Load TWS complete data
        try:
            self.tws_downscaled = xr.open_dataset(self.results_dir / "tws_complete.nc")
            print(f"   ✅ TWS data: {self.tws_downscaled.time.size} timesteps")
            print(f"   📊 TWS spatial: {self.tws_downscaled.lat.size} x {self.tws_downscaled.lon.size}")
        except FileNotFoundError:
            print("   ⚠️ tws_complete.nc not found, trying groundwater_complete.nc")
            self.tws_downscaled = None
        
        # Load groundwater data as backup
        try:
            self.gw_downscaled = xr.open_dataset(self.results_dir / "groundwater_complete.nc")
            print(f"   ✅ Groundwater data: {self.gw_downscaled.time.size} timesteps")
            print(f"   📊 GW spatial: {self.gw_downscaled.lat.size} x {self.gw_downscaled.lon.size}")
        except FileNotFoundError:
            print("   ❌ No downscaled data found!")
            self.gw_downscaled = None
        
        # Use TWS if available, otherwise groundwater
        if self.tws_downscaled is not None:
            self.downscaled_data = self.tws_downscaled
            self.data_var = 'tws' if 'tws' in self.downscaled_data.data_vars else list(self.downscaled_data.data_vars)[0]
            print(f"   🎯 Using TWS data variable: {self.data_var}")
        elif self.gw_downscaled is not None:
            self.downscaled_data = self.gw_downscaled
            self.data_var = 'groundwater' if 'groundwater' in self.downscaled_data.data_vars else list(self.downscaled_data.data_vars)[0]
            print(f"   🎯 Using groundwater data variable: {self.data_var}")
        else:
            raise FileNotFoundError("No downscaled data files found!")
        
        return self.downscaled_data
    
    def process_timeseries_data(self):
        """Process both datasets to create comparable time series."""
        print("🔄 Processing time series data...")
        
        # Calculate spatial means
        grace_spatial_mean = self.grace_original.mean(dim=['lat', 'lon'])
        downscaled_spatial_mean = self.downscaled_data[self.data_var].mean(dim=['lat', 'lon'])
        
        # Convert to DataFrames
        grace_df = pd.DataFrame({
            'date': pd.to_datetime(grace_spatial_mean.time.values),
            'grace_original': grace_spatial_mean.values
        })
        
        downscaled_df = pd.DataFrame({
            'date': pd.to_datetime(downscaled_spatial_mean.time.values),
            'downscaled': downscaled_spatial_mean.values
        })
        
        # Add year and month for grouping
        grace_df['year'] = grace_df['date'].dt.year
        grace_df['month'] = grace_df['date'].dt.month
        downscaled_df['year'] = downscaled_df['date'].dt.year
        downscaled_df['month'] = downscaled_df['date'].dt.month
        
        print(f"   ✅ GRACE: {len(grace_df)} observations")
        print(f"   ✅ Downscaled: {len(downscaled_df)} observations")
        
        return grace_df, downscaled_df
    
    def create_comparison_plots(self, grace_df, downscaled_df):
        """Create comprehensive comparison plots."""
        print("📊 Creating comparison plots...")
        
        # Merge datasets on date first
        merged_daily = pd.merge(grace_df, downscaled_df, on='date', how='inner')
        
        if len(merged_daily) == 0:
            print("   ⚠️ No overlapping dates found - trying monthly aggregation")
            
            # Try monthly aggregation
            grace_monthly = grace_df.groupby(['year', 'month'])['grace_original'].mean().reset_index()
            downscaled_monthly = downscaled_df.groupby(['year', 'month'])['downscaled'].mean().reset_index()
            merged_data = pd.merge(grace_monthly, downscaled_monthly, on=['year', 'month'], how='inner')
            merged_data['date'] = pd.to_datetime(merged_data[['year', 'month']].assign(day=1))
        else:
            merged_data = merged_daily.copy()
        
        print(f"   📅 Overlapping observations: {len(merged_data)}")
        
        if len(merged_data) == 0:
            print("   ❌ No overlapping data found for comparison!")
            return None, None, None, None
        
        # Calculate statistics
        correlation = np.corrcoef(merged_data['grace_original'], merged_data['downscaled'])[0, 1]
        rmse = np.sqrt(np.mean((merged_data['grace_original'] - merged_data['downscaled'])**2))
        bias = np.mean(merged_data['downscaled'] - merged_data['grace_original'])
        
        print(f"   📈 Correlation: {correlation:.3f}")
        print(f"   📊 RMSE: {rmse:.2f} cm")
        print(f"   ⚖️ Bias: {bias:.2f} cm")
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Time series comparison
        ax = axes[0, 0]
        ax.plot(merged_data['date'], merged_data['grace_original'], 'b-o', 
                label='GRACE Original', alpha=0.7, markersize=3)
        ax.plot(merged_data['date'], merged_data['downscaled'], 'r-s', 
                label='Downscaled', alpha=0.7, markersize=3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Liquid Water Equivalent (cm)')
        ax.set_title('GRACE Original vs Downscaled Time Series')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Scatter plot
        ax = axes[0, 1]
        ax.scatter(merged_data['grace_original'], merged_data['downscaled'], 
                   alpha=0.6, c=range(len(merged_data)), cmap='viridis')
        
        # Add 1:1 line
        min_val = min(merged_data['grace_original'].min(), merged_data['downscaled'].min())
        max_val = max(merged_data['grace_original'].max(), merged_data['downscaled'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='1:1 Line')
        
        ax.set_xlabel('GRACE Original (cm)')
        ax.set_ylabel('Downscaled (cm)')
        ax.set_title(f'Correlation Plot\nr = {correlation:.3f}, RMSE = {rmse:.2f} cm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Difference time series
        ax = axes[1, 0]
        difference = merged_data['downscaled'] - merged_data['grace_original']
        ax.plot(merged_data['date'], difference, 'g-o', alpha=0.7, markersize=3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(y=bias, color='r', linestyle='-', alpha=0.7, label=f'Bias = {bias:.2f} cm')
        ax.set_xlabel('Date')
        ax.set_ylabel('Difference (Downscaled - Original) (cm)')
        ax.set_title('Difference Time Series')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Summary statistics and distribution
        ax = axes[1, 1]
        
        # Create histogram of differences
        ax.hist(difference, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Zero Difference')
        ax.axvline(x=bias, color='r', linestyle='-', alpha=0.7, label=f'Mean Bias = {bias:.2f}')
        ax.set_xlabel('Difference (cm)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Differences')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text summary
        stats_text = f"""
Comparison Statistics:
• Observations: {len(merged_data)}
• Time Range: {merged_data['date'].min().strftime('%Y-%m')} to {merged_data['date'].max().strftime('%Y-%m')}
• Correlation: {correlation:.3f}
• RMSE: {rmse:.2f} cm
• Bias: {bias:.2f} cm
• Std Difference: {difference.std():.2f} cm

Original GRACE Stats:
• Mean: {merged_data['grace_original'].mean():.2f} cm
• Std: {merged_data['grace_original'].std():.2f} cm
• Range: {merged_data['grace_original'].min():.2f} to {merged_data['grace_original'].max():.2f} cm

Downscaled Stats:
• Mean: {merged_data['downscaled'].mean():.2f} cm
• Std: {merged_data['downscaled'].std():.2f} cm
• Range: {merged_data['downscaled'].min():.2f} to {merged_data['downscaled'].max():.2f} cm
        """
        
        # Add text box
        ax.text(1.05, 0.5, stats_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.figures_dir / 'grace_original_vs_downscaled_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Comparison plot saved to {plot_path}")
        
        # Save comparison data
        merged_data.to_csv(self.figures_dir / 'grace_comparison_data.csv', index=False)
        print(f"   💾 Comparison data saved to {self.figures_dir / 'grace_comparison_data.csv'}")
        
        return merged_data, correlation, rmse, bias
    
    def create_spatial_comparison(self):
        """Create spatial comparison plots."""
        print("🗺️ Creating spatial comparison...")
        
        try:
            # Calculate temporal means
            grace_mean = self.grace_original.mean(dim='time')
            downscaled_mean = self.downscaled_data[self.data_var].mean(dim='time')
            
            # Interpolate GRACE to downscaled grid for fair comparison
            grace_interp = grace_mean.interp(lat=downscaled_mean.lat, lon=downscaled_mean.lon)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original GRACE (interpolated)
            im1 = grace_interp.plot(ax=axes[0], add_colorbar=False, cmap='RdBu_r')
            axes[0].set_title('Original GRACE (Mean)')
            axes[0].set_xlabel('Longitude')
            axes[0].set_ylabel('Latitude')
            
            # Downscaled
            im2 = downscaled_mean.plot(ax=axes[1], add_colorbar=False, cmap='RdBu_r')
            axes[1].set_title('Downscaled (Mean)')
            axes[1].set_xlabel('Longitude')
            axes[1].set_ylabel('')
            
            # Difference
            difference = downscaled_mean - grace_interp
            im3 = difference.plot(ax=axes[2], add_colorbar=False, cmap='RdBu_r')
            axes[2].set_title('Difference (Downscaled - Original)')
            axes[2].set_xlabel('Longitude')
            axes[2].set_ylabel('')
            
            # Add colorbars
            plt.colorbar(im1, ax=axes[0], label='LWE (cm)')
            plt.colorbar(im2, ax=axes[1], label='LWE (cm)')
            plt.colorbar(im3, ax=axes[2], label='Difference (cm)')
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'grace_spatial_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Spatial comparison saved")
            
        except Exception as e:
            print(f"   ⚠️ Could not create spatial comparison: {e}")
    
    def run_full_comparison(self):
        """Run the complete comparison analysis."""
        print("🚀 GRACE ORIGINAL vs DOWNSCALED COMPARISON")
        print("=" * 60)
        
        # Load original GRACE data
        grace_data = self.load_original_grace_data()
        if grace_data is None:
            print("❌ Could not load original GRACE data")
            print("Run download_grace_original.py first!")
            return None, None, None, None
        
        # Load downscaled data
        self.load_downscaled_data()
        
        # Process time series
        grace_df, downscaled_df = self.process_timeseries_data()
        
        # Create comparison plots
        comparison_data, correlation, rmse, bias = self.create_comparison_plots(grace_df, downscaled_df)
        
        if comparison_data is not None:
            # Create spatial comparison
            self.create_spatial_comparison()
            
            print("\n" + "=" * 60)
            print("📋 FINAL COMPARISON SUMMARY")
            print("=" * 60)
            print(f"✅ Comparison completed successfully!")
            print(f"📊 Correlation: {correlation:.3f}")
            print(f"📈 RMSE: {rmse:.2f} cm")
            print(f"⚖️ Bias: {bias:.2f} cm")
            print(f"📅 Overlapping observations: {len(comparison_data)}")
            print(f"🗂️ Results saved to: {self.figures_dir}")
            
            # Assessment
            if correlation > 0.8:
                print("🎉 EXCELLENT agreement between original and downscaled data!")
            elif correlation > 0.6:
                print("✅ GOOD agreement between original and downscaled data!")
            elif correlation > 0.4:
                print("⚠️ MODERATE agreement - room for improvement")
            else:
                print("⚠️ POOR agreement - significant differences detected")
        
        return grace_df, downscaled_df, comparison_data, {'correlation': correlation, 'rmse': rmse, 'bias': bias} if comparison_data is not None else None


def main():
    """Main comparison function."""
    # Initialize comparison
    comparator = GRACEComparisonAnalysis()
    
    # Run full comparison
    grace_df, downscaled_df, comparison_data, stats = comparator.run_full_comparison()
    
    return comparator, grace_df, downscaled_df, comparison_data, stats


if __name__ == "__main__":
    comparator, grace_df, downscaled_df, comparison_data, stats = main()
