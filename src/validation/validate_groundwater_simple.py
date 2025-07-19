# src/validation/validate_groundwater_simple.py
"""
Simplified validation for GRACE groundwater downscaling.

This module directly compares model anomalies with well anomalies - no complex conversions.
Focuses on robust statistical comparison with clear data requirements.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SimpleGroundwaterValidator:
    """Simple, direct validation of groundwater anomalies."""
    
    def __init__(self, 
                 min_months_required=24,      # Minimum months for reliable correlation
                 min_overlap_months=12,       # Minimum overlapping months
                 outlier_threshold=3.0):      # Standard deviations for outlier detection
        """
        Initialize validator with data requirements.
        
        Parameters:
        -----------
        min_months_required : int
            Minimum months of data needed for a well to be included
        min_overlap_months : int
            Minimum overlapping months between model and observations
        outlier_threshold : float
            Number of standard deviations to identify outliers
        """
        self.min_months = min_months_required
        self.min_overlap = min_overlap_months
        self.outlier_threshold = outlier_threshold
        
        # Set up directories
        self.results_dir = Path("results/validation")
        self.figures_dir = Path("figures/validation")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.figures_dir.mkdir(exist_ok=True, parents=True)
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load model predictions and well observations."""
        print("📦 Loading data for validation...")
        
        # Load model predictions
        gws_path = self._find_model_output()
        self.model_data = xr.open_dataset(gws_path)
        print(f"✅ Model data: {self.model_data.groundwater.shape}")
        
        # Load well anomalies (already calculated!)
        well_path = "data/raw/usgs_well_data/monthly_groundwater_anomalies.csv"
        self.well_anomalies = pd.read_csv(well_path, index_col=0, parse_dates=True)
        print(f"✅ Well anomalies: {self.well_anomalies.shape}")
        
        # Load well locations
        meta_path = "data/raw/usgs_well_data/well_metadata.csv"
        self.well_metadata = pd.read_csv(meta_path)
        print(f"✅ Well metadata: {len(self.well_metadata)} wells")
        
        # Basic data quality check
        self._check_data_quality()
    
    def _find_model_output(self):
        """Find the model output file."""
        possible_files = [
            "results/groundwater_storage_anomalies.nc",
            "results/groundwater_storage_anomalies_enhanced.nc"
        ]
        
        for f in possible_files:
            if Path(f).exists():
                return f
        
        raise FileNotFoundError(f"No model output found. Tried: {possible_files}")
    
    def _check_data_quality(self):
        """Check data quality and print summary."""
        print("\n📊 Data Quality Summary:")
        
        # Check model data
        model_nans = np.isnan(self.model_data.groundwater.values).sum()
        model_total = self.model_data.groundwater.size
        print(f"  Model: {model_nans}/{model_total} NaN values ({model_nans/model_total*100:.1f}%)")
        
        # Check well data
        well_nans = self.well_anomalies.isna().sum().sum()
        well_total = self.well_anomalies.size
        print(f"  Wells: {well_nans}/{well_total} NaN values ({well_nans/well_total*100:.1f}%)")
        
        # Check temporal coverage
        print(f"\n  Model time range: {self.model_data.time.values[0]} to {self.model_data.time.values[-1]}")
        print(f"  Well time range: {self.well_anomalies.index[0]} to {self.well_anomalies.index[-1]}")
        
        # Count wells with sufficient data
        wells_sufficient = (self.well_anomalies.count() >= self.min_months).sum()
        print(f"\n  Wells with ≥{self.min_months} months: {wells_sufficient}/{len(self.well_anomalies.columns)}")
    
    def validate_direct_comparison(self):
        """
        Direct comparison of model and well anomalies.
        
        Returns:
        --------
        pd.DataFrame
            Validation metrics for each well
        """
        print(f"\n🎯 DIRECT ANOMALY COMPARISON")
        print("="*50)
        print(f"Requirements: ≥{self.min_months} total months, ≥{self.min_overlap} overlapping months")
        
        results = []
        
        # Process each well
        for well_id in tqdm(self.well_anomalies.columns, desc="Validating wells"):
            
            # Get well location
            well_info = self.well_metadata[self.well_metadata['well_id'].astype(str) == str(well_id)]
            if len(well_info) == 0:
                continue
            
            lat = well_info.iloc[0]['lat']
            lon = well_info.iloc[0]['lon']
            
            # Check if within model bounds
            if not self._is_in_bounds(lat, lon):
                continue
            
            # Get well anomaly time series
            well_series = self.well_anomalies[well_id].dropna()
            
            # Check minimum data requirement
            if len(well_series) < self.min_months:
                continue
            
            try:
                # Extract model anomaly at well location
                model_series = self.model_data.groundwater.sel(
                    lat=lat, lon=lon, method='nearest'
                ).to_pandas()
                
                # Align time series
                aligned = self._align_time_series(model_series, well_series)
                
                if aligned is None:
                    continue
                
                model_aligned, well_aligned = aligned
                
                # Remove outliers
                model_clean, well_clean = self._remove_outliers(model_aligned, well_aligned)
                
                if len(model_clean) < self.min_overlap:
                    continue
                
                # Calculate metrics
                metrics = self._calculate_metrics(model_clean, well_clean)
                
                if metrics:
                    metrics.update({
                        'well_id': well_id,
                        'lat': lat,
                        'lon': lon,
                        'n_months_total': len(well_series),
                        'n_months_used': len(model_clean),
                        'outliers_removed': len(model_aligned) - len(model_clean)
                    })
                    results.append(metrics)
                    
            except Exception as e:
                continue
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # Save results
            results_df.to_csv(self.results_dir / "validation_metrics.csv", index=False)
            
            # Print summary
            self._print_summary(results_df)
            
            # Create visualizations
            self._create_validation_plots(results_df)
        else:
            print("❌ No wells could be validated with current requirements")
        
        return results_df
    
    def _is_in_bounds(self, lat, lon):
        """Check if coordinates are within model bounds."""
        return (lat >= self.model_data.lat.min() and 
                lat <= self.model_data.lat.max() and
                lon >= self.model_data.lon.min() and 
                lon <= self.model_data.lon.max())
    
    def _align_time_series(self, model_series, well_series):
        """Align model and well time series."""
        # Ensure both have datetime index
        if not isinstance(model_series.index, pd.DatetimeIndex):
            model_series.index = pd.to_datetime(model_series.index)
        if not isinstance(well_series.index, pd.DatetimeIndex):
            well_series.index = pd.to_datetime(well_series.index)
        
        # Find common time period
        common_idx = model_series.index.intersection(well_series.index)
        
        if len(common_idx) < self.min_overlap:
            return None
        
        return model_series[common_idx], well_series[common_idx]
    
    def _remove_outliers(self, model_series, well_series):
        """Remove outliers using z-score method."""
        # Calculate z-scores
        model_z = np.abs((model_series - model_series.mean()) / model_series.std())
        well_z = np.abs((well_series - well_series.mean()) / well_series.std())
        
        # Keep only points where both are within threshold
        mask = (model_z < self.outlier_threshold) & (well_z < self.outlier_threshold)
        
        return model_series[mask], well_series[mask]
    
    def _calculate_metrics(self, model_series, well_series):
        """Calculate validation metrics."""
        # Basic correlation metrics
        pearson_r, pearson_p = pearsonr(model_series, well_series)
        spearman_r, spearman_p = spearmanr(model_series, well_series)
        
        # Error metrics (on normalized data to be fair)
        model_norm = (model_series - model_series.mean()) / model_series.std()
        well_norm = (well_series - well_series.mean()) / well_series.std()
        
        rmse = np.sqrt(mean_squared_error(model_norm, well_norm))
        mae = mean_absolute_error(model_norm, well_norm)
        
        # Bias (systematic over/under estimation)
        bias = (model_series.mean() - well_series.mean()) / well_series.std()
        
        # Variance ratio (does model capture variability?)
        var_ratio = model_series.std() / well_series.std()
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'rmse_normalized': rmse,
            'mae_normalized': mae,
            'bias_normalized': bias,
            'variance_ratio': var_ratio
        }
    
    def _print_summary(self, results_df):
        """Print validation summary."""
        print(f"\n📊 VALIDATION SUMMARY")
        print("="*50)
        print(f"Wells validated: {len(results_df)}")
        
        # Correlation summary
        print(f"\nCorrelation (Pearson r):")
        print(f"  Mean: {results_df['pearson_r'].mean():.3f} ± {results_df['pearson_r'].std():.3f}")
        print(f"  Median: {results_df['pearson_r'].median():.3f}")
        print(f"  Range: [{results_df['pearson_r'].min():.3f}, {results_df['pearson_r'].max():.3f}]")
        
        # Performance categories
        excellent = (results_df['pearson_r'] >= 0.7).sum()
        good = ((results_df['pearson_r'] >= 0.5) & (results_df['pearson_r'] < 0.7)).sum()
        fair = ((results_df['pearson_r'] >= 0.3) & (results_df['pearson_r'] < 0.5)).sum()
        poor = (results_df['pearson_r'] < 0.3).sum()
        
        print(f"\nPerformance Distribution:")
        print(f"  Excellent (r≥0.7): {excellent} wells ({excellent/len(results_df)*100:.1f}%)")
        print(f"  Good (0.5≤r<0.7): {good} wells ({good/len(results_df)*100:.1f}%)")
        print(f"  Fair (0.3≤r<0.5): {fair} wells ({fair/len(results_df)*100:.1f}%)")
        print(f"  Poor (r<0.3): {poor} wells ({poor/len(results_df)*100:.1f}%)")
        
        # Other metrics
        print(f"\nVariance Ratio (model_std/well_std):")
        print(f"  Mean: {results_df['variance_ratio'].mean():.3f}")
        print(f"  Ideal is ~1.0 (model captures similar variability)")
        
        print(f"\nSignificant correlations (p<0.05): {(results_df['pearson_p'] < 0.05).sum()} wells")
        
        # Data usage
        print(f"\nData Usage:")
        print(f"  Mean months used: {results_df['n_months_used'].mean():.0f}")
        print(f"  Mean outliers removed: {results_df['outliers_removed'].mean():.1f}")
    
    def _create_validation_plots(self, results_df):
        """Create simple, informative plots."""
        # Set up figure
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Correlation histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(results_df['pearson_r'], bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(results_df['pearson_r'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {results_df["pearson_r"].mean():.3f}')
        ax1.set_xlabel('Pearson Correlation')
        ax1.set_ylabel('Number of Wells')
        ax1.set_title('Correlation Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Spatial map
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(results_df['lon'], results_df['lat'], 
                            c=results_df['pearson_r'], s=50,
                            cmap='RdYlBu', vmin=0, vmax=1,
                            edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Spatial Distribution of Correlations')
        plt.colorbar(scatter, ax=ax2, label='Correlation')
        
        # 3. Variance ratio
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(results_df['variance_ratio'], bins=30, edgecolor='black', alpha=0.7)
        ax3.axvline(1.0, color='green', linestyle='--', label='Ideal (1.0)')
        ax3.set_xlabel('Variance Ratio (Model/Well)')
        ax3.set_ylabel('Number of Wells')
        ax3.set_title('Variance Ratio Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Sample size vs correlation
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(results_df['n_months_used'], results_df['pearson_r'], alpha=0.6)
        ax4.set_xlabel('Months of Data Used')
        ax4.set_ylabel('Correlation')
        ax4.set_title('Data Quantity vs Performance')
        ax4.grid(True, alpha=0.3)
        
        # 5. Best example time series
        ax5 = fig.add_subplot(gs[1, 1:])
        best_well = results_df.nlargest(1, 'pearson_r').iloc[0]
        
        # Get time series for best well
        well_series = self.well_anomalies[best_well['well_id']].dropna()
        model_series = self.model_data.groundwater.sel(
            lat=best_well['lat'], lon=best_well['lon'], method='nearest'
        ).to_pandas()
        
        # Align and plot
        common_idx = model_series.index.intersection(well_series.index)
        ax5.plot(pd.to_datetime(common_idx), model_series[common_idx], 
                label='Model', linewidth=2, alpha=0.8)
        ax5.plot(pd.to_datetime(common_idx), well_series[common_idx], 
                label='Observed', linewidth=2, alpha=0.8, linestyle='--')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Groundwater Anomaly')
        ax5.set_title(f'Best Example - Well {best_well["well_id"]} (r={best_well["pearson_r"]:.3f})')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance summary
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Summary statistics text
        summary_text = f"""
VALIDATION SUMMARY

Total Wells Validated: {len(results_df)}
Mean Correlation: {results_df['pearson_r'].mean():.3f} ± {results_df['pearson_r'].std():.3f}

Performance Categories:
• Excellent (r≥0.7): {(results_df['pearson_r'] >= 0.7).sum()} wells
• Good (0.5≤r<0.7): {((results_df['pearson_r'] >= 0.5) & (results_df['pearson_r'] < 0.7)).sum()} wells  
• Fair (0.3≤r<0.5): {((results_df['pearson_r'] >= 0.3) & (results_df['pearson_r'] < 0.5)).sum()} wells
• Poor (r<0.3): {(results_df['pearson_r'] < 0.3).sum()} wells

Data Quality:
• Mean months used: {results_df['n_months_used'].mean():.0f}
• Wells with p<0.05: {(results_df['pearson_p'] < 0.05).sum()}
• Mean outliers removed: {results_df['outliers_removed'].mean():.1f}
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('GRACE Groundwater Validation Results', fontsize=16, fontweight='bold')
        plt.savefig(self.figures_dir / 'validation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Validation plot saved to: {self.figures_dir / 'validation_results.png'}")
    
    def sensitivity_analysis(self):
        """Test sensitivity to data requirements."""
        print(f"\n🔍 SENSITIVITY ANALYSIS")
        print("="*50)
        
        min_months_range = [12, 18, 24, 36, 48]
        overlap_range = [6, 12, 18, 24]
        
        results = []
        
        for min_months in min_months_range:
            for min_overlap in overlap_range:
                if min_overlap > min_months:
                    continue
                
                # Temporarily change requirements
                original_months = self.min_months
                original_overlap = self.min_overlap
                
                self.min_months = min_months
                self.min_overlap = min_overlap
                
                # Run validation
                metrics_df = self.validate_direct_comparison()
                
                if len(metrics_df) > 0:
                    results.append({
                        'min_months': min_months,
                        'min_overlap': min_overlap,
                        'n_wells': len(metrics_df),
                        'mean_r': metrics_df['pearson_r'].mean(),
                        'std_r': metrics_df['pearson_r'].std(),
                        'pct_good': (metrics_df['pearson_r'] >= 0.5).sum() / len(metrics_df) * 100
                    })
                
                # Restore original settings
                self.min_months = original_months
                self.min_overlap = original_overlap
        
        # Create sensitivity plot
        if results:
            sens_df = pd.DataFrame(results)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Number of wells vs requirements
            pivot1 = sens_df.pivot(index='min_months', columns='min_overlap', values='n_wells')
            sns.heatmap(pivot1, annot=True, fmt='g', cmap='YlOrRd', ax=ax1)
            ax1.set_title('Number of Wells Validated')
            ax1.set_xlabel('Minimum Overlap (months)')
            ax1.set_ylabel('Minimum Total Months')
            
            # Plot 2: Mean correlation vs requirements
            pivot2 = sens_df.pivot(index='min_months', columns='min_overlap', values='mean_r')
            sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='RdYlBu', ax=ax2, vmin=0, vmax=1)
            ax2.set_title('Mean Correlation')
            ax2.set_xlabel('Minimum Overlap (months)')
            ax2.set_ylabel('Minimum Total Months')
            
            plt.suptitle('Sensitivity to Data Requirements', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'sensitivity_analysis.png', dpi=300)
            plt.close()
            
            # Save results
            sens_df.to_csv(self.results_dir / 'sensitivity_analysis.csv', index=False)
            
            print("✅ Sensitivity analysis complete")
            print(f"   Results saved to: {self.results_dir / 'sensitivity_analysis.csv'}")


def main():
    """Main validation function - simple and effective."""
    print("🚀 SIMPLE GROUNDWATER VALIDATION")
    print("="*60)
    
    # Initialize validator with clear requirements
    validator = SimpleGroundwaterValidator(
        min_months_required=24,  # 2 years minimum
        min_overlap_months=12,   # 1 year overlap minimum
        outlier_threshold=3.0    # 3 standard deviations
    )
    
    # Run direct validation
    results = validator.validate_direct_comparison()
    
    # Optional: Run sensitivity analysis
    validator.sensitivity_analysis()
    
    print("\n✅ Validation complete!")
    print(f"📁 Results saved to: {validator.results_dir}")
    print(f"📊 Figures saved to: {validator.figures_dir}")
    
    return results


if __name__ == "__main__":
    main()