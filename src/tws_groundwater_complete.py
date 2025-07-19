# src/tws_groundwater_complete.py
"""
Complete TWS and Groundwater calculation with gap-filling.

This module produces COMPLETE time series for:
- Total Water Storage (TWS) - the direct model output
- Groundwater Storage (GWS) - derived from TWS
- All water storage components with proper gap-filling

Creates a continuous GRACE-like dataset for 2003-2022 without gaps!
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import joblib
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class CompleteWaterStorageCalculator:
    """Calculate complete TWS and groundwater with gap-filling."""
    
    def __init__(self, model_path=None, reference_period=("2004-01", "2009-12")):
        """
        Initialize calculator.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to trained model. If None, searches for best available.
        reference_period : tuple
            Start and end dates for reference period (for anomaly calculation)
        """
        self.model_path = self._find_model(model_path)
        self.model = None
        self.scaler = None
        self.imputer = None
        self.model_info = {}
        self.model_type = None
        self.reference_start, self.reference_end = reference_period
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def _find_model(self, model_path):
        """Find the trained model."""
        if model_path and Path(model_path).exists():
            return Path(model_path)
        
        # Search for models in order of preference
        search_paths = [
            "models/best_model.joblib",
            "models/nn_model.joblib",  # Neural networks often perform best
            "models/rf_model_enhanced.joblib",
            "models/rf_model.joblib",
            "models/xgb_model.joblib"
        ]
        
        for path in search_paths:
            if Path(path).exists():
                return Path(path)
        
        # Check model comparison file
        comparison_file = Path("models/model_comparison.csv")
        if comparison_file.exists():
            comp_df = pd.read_csv(comparison_file)
            best_model = comp_df.loc[comp_df['test_r2'].idxmax(), 'model_name']
            best_path = Path(f"models/{best_model}_model.joblib")
            if best_path.exists():
                return best_path
        
        raise FileNotFoundError("No trained model found! Run training first.")
    
    def calculate_complete_storage(self, output_mode='complete'):
        """
        Calculate complete TWS and groundwater storage.
        
        Parameters:
        -----------
        output_mode : str
            'complete' - Output all months with gap-filling
            'grace_only' - Output only months with GRACE data
            'both' - Output both versions
        
        Returns:
        --------
        dict
            Dictionary with 'tws' and 'groundwater' xarray Datasets
        """
        print("🌊 COMPLETE WATER STORAGE CALCULATION")
        print("="*70)
        print(f"Mode: {output_mode}")
        print(f"Reference period: {self.reference_start} to {self.reference_end}")
        
        # Load model
        print(f"\n📦 Loading model from {self.model_path}...")
        model_data = joblib.load(self.model_path)
        
        # Check if it's a dictionary (new format) or direct model (old format)
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.model_info = model_data.get('config', {})
            self.scaler = model_data.get('scaler', None)
            print(f"✅ Model loaded from package format")
            if 'metrics' in model_data:
                print(f"   Model performance: R² = {model_data['metrics'].get('test_r2', 'N/A'):.4f}")
            if self.scaler is not None:
                print(f"   ✅ Scaler loaded (model requires feature scaling)")
        else:
            # Old format - direct model object
            self.model = model_data
            print(f"✅ Model loaded (legacy format)")
        
        # Determine model type and preprocessing needs
        model_class = type(self.model).__name__
        print(f"   Model type: {model_class}")
        
        # Check if model needs NaN handling
        if 'MLP' in model_class or 'Neural' in model_class or 'SVR' in model_class:
            self.model_type = 'needs_preprocessing'
            self.imputer = SimpleImputer(strategy='mean')
            print(f"   ⚠️ Model requires NaN handling - imputer initialized")
            
            # Create scaler if not provided
            if self.scaler is None and 'SVR' in model_class:
                self.scaler = StandardScaler()
                print(f"   ⚠️ Created StandardScaler for SVR model")
        else:
            self.model_type = 'handles_nan'
            print(f"   ✅ Model can handle NaN values natively")
        
        # Load feature data
        print("\n📦 Loading feature data...")
        feature_ds = xr.open_dataset("data/processed/feature_stack.nc")
        
        # Identify GRACE coverage
        grace_months, all_months, gaps = self._identify_grace_coverage()
        
        # Determine which months to process
        feature_times = pd.to_datetime(feature_ds.time.values)
        feature_months = [t.strftime('%Y-%m') for t in feature_times]
        
        if output_mode == 'complete':
            months_to_process = feature_months
        elif output_mode == 'grace_only':
            months_to_process = [m for m in feature_months if m in grace_months]
        else:  # both
            return self._calculate_both_modes(feature_ds, grace_months)
        
        print(f"\n🔄 Processing {len(months_to_process)} months...")
        
        # Calculate TWS for all months
        tws_results = self._calculate_tws(feature_ds, months_to_process, grace_months)
        
        # Check if we got enough results
        if len(tws_results['months']) == 0:
            raise RuntimeError("No months could be processed successfully!")
        
        print(f"\n✅ Successfully processed {len(tws_results['months'])}/{len(months_to_process)} months")
        if len(tws_results['months']) < len(months_to_process):
            print(f"   ⚠️ {len(months_to_process) - len(tws_results['months'])} months failed (likely due to missing data)")
        
        # Calculate water balance components
        print("\n💧 Calculating water balance components...")
        components = self._calculate_components(feature_ds, months_to_process)
        
        # Calculate reference means
        ref_means = self._calculate_reference_means(components, months_to_process)
        
        # Calculate anomalies and groundwater
        results = self._calculate_anomalies_and_groundwater(
            tws_results, components, ref_means, months_to_process
        )
        
        # Create output datasets
        datasets = self._create_output_datasets(
            results, feature_ds, months_to_process, grace_months, output_mode
        )
        
        # Save outputs
        self._save_outputs(datasets, output_mode)
        
        # Create visualizations
        self._create_visualizations(datasets, output_mode)
        
        return datasets
    
    def _identify_grace_coverage(self):
        """Identify GRACE data coverage and gaps."""
        import re
        
        grace_dir = Path("data/raw/grace")
        grace_files = sorted([f for f in os.listdir(grace_dir) if f.endswith('.tif')])
        
        grace_months = set()
        for f in grace_files:
            match = re.match(r'(\d{8})_(\d{8})\.tif', f)
            if match:
                date = datetime.strptime(match.group(1), '%Y%m%d')
                grace_months.add(date.strftime('%Y-%m'))
        
        # Complete time range
        start = pd.to_datetime('2003-01')
        end = pd.to_datetime('2022-12')
        all_months = pd.date_range(start, end, freq='MS').strftime('%Y-%m').tolist()
        
        gaps = set(all_months) - grace_months
        
        print(f"\n📊 GRACE Coverage Analysis:")
        print(f"   Total possible months: {len(all_months)}")
        print(f"   GRACE observations: {len(grace_months)}")
        print(f"   Missing months: {len(gaps)}")
        print(f"   Coverage: {len(grace_months)/len(all_months)*100:.1f}%")
        
        # Identify major gaps
        gap_list = sorted(list(gaps))
        if gap_list:
            print("\n   Major gaps:")
            current_gap = [gap_list[0]]
            
            for i in range(1, len(gap_list)):
                if (pd.to_datetime(gap_list[i]) - pd.to_datetime(gap_list[i-1])).days < 35:
                    current_gap.append(gap_list[i])
                else:
                    if len(current_gap) >= 3:
                        print(f"     • {current_gap[0]} to {current_gap[-1]} ({len(current_gap)} months)")
                    current_gap = [gap_list[i]]
            
            if len(current_gap) >= 3:
                print(f"     • {current_gap[0]} to {current_gap[-1]} ({len(current_gap)} months)")
        
        return grace_months, all_months, gaps
    
    def _calculate_tws(self, feature_ds, months_to_process, grace_months):
        """Calculate TWS for all specified months."""
        tws_data = []
        data_sources = []
        valid_months = []
        
        # Fit imputer if needed
        if self.model_type == 'needs_preprocessing' and self.imputer is not None:
            print("   Fitting imputer on available data...")
            # Collect some samples to fit the imputer
            sample_features = []
            for i, month in enumerate(months_to_process[:50]):  # Use first 50 months
                X_sample = self._prepare_features(feature_ds, month)
                if X_sample is not None:
                    # Remove completely NaN samples
                    mask = ~np.isnan(X_sample).all(axis=1)
                    if mask.any():
                        sample_features.append(X_sample[mask])
            
            if sample_features:
                X_fit = np.vstack(sample_features)
                self.imputer.fit(X_fit)
                print(f"   ✅ Imputer fitted on {len(X_fit)} samples")
                
                # Fit scaler if needed
                if self.scaler is not None and not hasattr(self.scaler, 'mean_'):
                    X_imputed = self.imputer.transform(X_fit)
                    self.scaler.fit(X_imputed)
                    print(f"   ✅ Scaler fitted")
        
        for month in tqdm(months_to_process, desc="Calculating TWS"):
            try:
                # Prepare features
                X = self._prepare_features(feature_ds, month)
                
                if X is None:
                    continue
                
                # Handle NaN values if needed
                if self.model_type == 'needs_preprocessing':
                    # Check for NaN values
                    if np.isnan(X).any():
                        # Impute missing values
                        X = self.imputer.transform(X)
                    
                    # Apply scaling if needed
                    if self.scaler is not None:
                        X = self.scaler.transform(X)
                
                # Predict TWS
                tws_pred = self.model.predict(X)
                
                # Reshape to spatial grid
                n_lat = feature_ds.lat.shape[0]
                n_lon = feature_ds.lon.shape[0]
                tws_spatial = tws_pred.reshape(n_lat, n_lon)
                
                # Store results
                tws_data.append(tws_spatial)
                valid_months.append(month)
                
                # Track data source
                if month in grace_months:
                    data_sources.append('grace_period')
                else:
                    data_sources.append('gap_filled')
                    
            except Exception as e:
                print(f"\n⚠️ Error processing {month}: {e}")
                continue
        
        return {
            'tws': np.stack(tws_data),
            'months': valid_months,
            'sources': data_sources
        }
    
    def _prepare_features(self, ds, month, lag_months=[1, 3, 6]):
        """Prepare features for prediction."""
        try:
            # Find time index
            time_idx = None
            for i, t in enumerate(ds.time.values):
                if pd.to_datetime(t).strftime('%Y-%m') == month:
                    time_idx = i
                    break
            
            if time_idx is None:
                return None
            
            # Get current features
            current = ds.features.isel(time=time_idx).values
            
            # Create feature array
            features = [current]
            
            # Add lagged features
            target_date = pd.to_datetime(f"{month}-01")
            
            for lag in lag_months:
                lag_date = target_date - pd.DateOffset(months=lag)
                lag_month = lag_date.strftime('%Y-%m')
                
                # Find lag index
                lag_idx = None
                for i, t in enumerate(ds.time.values):
                    if pd.to_datetime(t).strftime('%Y-%m') == lag_month:
                        lag_idx = i
                        break
                
                if lag_idx is not None:
                    features.append(ds.features.isel(time=lag_idx).values)
                else:
                    features.append(np.zeros_like(current))
            
            # Add seasonal features
            month_num = target_date.month
            spatial_shape = current.shape[1:]
            
            month_sin = np.sin(2 * np.pi * month_num / 12) * np.ones(spatial_shape)
            month_cos = np.cos(2 * np.pi * month_num / 12) * np.ones(spatial_shape)
            
            features.append(month_sin[np.newaxis, :, :])
            features.append(month_cos[np.newaxis, :, :])
            
            # Add static features
            if 'static_features' in ds:
                features.append(ds.static_features.values)
            
            # Stack and reshape
            X = np.vstack(features)
            X_flat = X.reshape(X.shape[0], -1).T
            
            return X_flat
            
        except Exception as e:
            print(f"Error preparing features for {month}: {e}")
            return None
    
    def _calculate_components(self, ds, months):
        """Calculate water storage components."""
        soil_vars = [v for v in ds.feature.values if 'SoilMoi' in str(v)]
        swe_vars = [v for v in ds.feature.values if 'SWE' in str(v)]
        
        soil_data = []
        swe_data = []
        
        for month in months:
            # Find time index
            time_idx = None
            for i, t in enumerate(ds.time.values):
                if pd.to_datetime(t).strftime('%Y-%m') == month:
                    time_idx = i
                    break
            
            if time_idx is None:
                soil_data.append(np.zeros((ds.lat.shape[0], ds.lon.shape[0])))
                swe_data.append(np.zeros((ds.lat.shape[0], ds.lon.shape[0])))
                continue
            
            # Extract components
            soil = np.zeros((ds.lat.shape[0], ds.lon.shape[0]))
            for var in soil_vars:
                var_idx = np.where(ds.feature.values == var)[0][0]
                soil += ds.features.isel(time=time_idx, feature=var_idx).values
            
            swe = np.zeros((ds.lat.shape[0], ds.lon.shape[0]))
            for var in swe_vars:
                var_idx = np.where(ds.feature.values == var)[0][0]
                swe += ds.features.isel(time=time_idx, feature=var_idx).values
            
            # Convert units (kg/m² to cm)
            soil_data.append(soil * 0.1)
            swe_data.append(swe * 0.1)
        
        return {
            'soil_moisture': np.stack(soil_data),
            'swe': np.stack(swe_data)
        }
    
    def _calculate_reference_means(self, components, months):
        """Calculate reference period means."""
        ref_start = pd.to_datetime(self.reference_start)
        ref_end = pd.to_datetime(self.reference_end)
        
        ref_indices = []
        for i, month in enumerate(months):
            month_date = pd.to_datetime(f"{month}-01")
            if ref_start <= month_date <= ref_end:
                ref_indices.append(i)
        
        if not ref_indices:
            print("⚠️ No data in reference period, using full period mean")
            ref_indices = list(range(len(months)))
        
        print(f"   Using {len(ref_indices)} months for reference period")
        
        return {
            'soil_moisture': np.mean(components['soil_moisture'][ref_indices], axis=0),
            'swe': np.mean(components['swe'][ref_indices], axis=0)
        }
    
    def _calculate_anomalies_and_groundwater(self, tws_results, components, ref_means, months):
        """Calculate anomalies and derive groundwater."""
        n_times = len(months)
        
        # Calculate anomalies
        soil_anomalies = components['soil_moisture'] - ref_means['soil_moisture']
        swe_anomalies = components['swe'] - ref_means['swe']
        
        # Calculate groundwater as residual
        groundwater = tws_results['tws'] - soil_anomalies - swe_anomalies
        
        # Apply reasonable bounds
        MAX_VALUE = 100  # cm
        tws_results['tws'] = np.clip(tws_results['tws'], -MAX_VALUE, MAX_VALUE)
        groundwater = np.clip(groundwater, -MAX_VALUE, MAX_VALUE)
        
        return {
            'tws': tws_results['tws'],
            'groundwater': groundwater,
            'soil_moisture_anomaly': soil_anomalies,
            'swe_anomaly': swe_anomalies,
            'data_sources': tws_results['sources']
        }
    
    def _create_output_datasets(self, results, feature_ds, months, grace_months, mode):
        """Create xarray datasets for output."""
        # Convert months to datetime
        times = pd.to_datetime([f"{m}-01" for m in months])
        
        # Count data sources
        sources = results['data_sources']
        n_grace_period = sources.count('grace_period')
        n_gap_filled = sources.count('gap_filled')
        
        # Create TWS dataset
        tws_ds = xr.Dataset(
            data_vars={
                "tws": (["time", "lat", "lon"], results['tws']),
                "data_source": (["time"], sources)
            },
            coords={
                "time": times,
                "lat": feature_ds.lat,
                "lon": feature_ds.lon
            },
            attrs={
                "title": "Complete Total Water Storage (TWS) with Gap-Filling",
                "description": "Machine learning gap-filled TWS providing continuous GRACE-like data",
                "units": "cm water equivalent",
                "reference_period": f"{self.reference_start} to {self.reference_end}",
                "model_used": str(self.model_path.name),
                "n_grace_period_months": n_grace_period,
                "n_gap_filled_months": n_gap_filled,
                "coverage_mode": mode,
                "creation_date": datetime.now().isoformat()
            }
        )
        
        # Add variable attributes
        tws_ds.tws.attrs = {
            "long_name": "Total Water Storage Anomaly",
            "standard_name": "total_water_storage_anomaly",
            "units": "cm",
            "description": "TWS anomaly relative to reference period, including gap-filled estimates"
        }
        
        tws_ds.data_source.attrs = {
            "long_name": "Data Source Flag",
            "flag_values": "grace_period, gap_filled",
            "flag_meanings": "Month within GRACE operational period, Gap-filled using ML model"
        }
        
        # Create groundwater dataset
        gws_ds = xr.Dataset(
            data_vars={
                "groundwater": (["time", "lat", "lon"], results['groundwater']),
                "tws": (["time", "lat", "lon"], results['tws']),
                "soil_moisture_anomaly": (["time", "lat", "lon"], results['soil_moisture_anomaly']),
                "swe_anomaly": (["time", "lat", "lon"], results['swe_anomaly']),
                "data_source": (["time"], sources)
            },
            coords={
                "time": times,
                "lat": feature_ds.lat,
                "lon": feature_ds.lon
            },
            attrs={
                "title": "Complete Groundwater Storage with Gap-Filling",
                "description": "Groundwater derived from gap-filled TWS minus soil moisture and snow",
                "units": "cm water equivalent",
                "reference_period": f"{self.reference_start} to {self.reference_end}",
                "model_used": str(self.model_path.name),
                "n_grace_period_months": n_grace_period,
                "n_gap_filled_months": n_gap_filled,
                "coverage_mode": mode,
                "creation_date": datetime.now().isoformat()
            }
        )
        
        return {'tws': tws_ds, 'groundwater': gws_ds}
    
    def _save_outputs(self, datasets, mode):
        """Save output datasets."""
        if mode == 'complete':
            tws_path = self.results_dir / "tws_complete.nc"
            gws_path = self.results_dir / "groundwater_complete.nc"
        else:
            tws_path = self.results_dir / "tws_grace_only.nc"
            gws_path = self.results_dir / "groundwater_grace_only.nc"
        
        datasets['tws'].to_netcdf(tws_path)
        datasets['groundwater'].to_netcdf(gws_path)
        
        print(f"\n✅ Outputs saved:")
        print(f"   TWS: {tws_path}")
        print(f"   Groundwater: {gws_path}")
    
    def _create_visualizations(self, datasets, mode):
        """Create comprehensive visualizations."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)
        
        tws_ds = datasets['tws']
        gws_ds = datasets['groundwater']
        
        # 1. Data availability timeline
        ax1 = fig.add_subplot(gs[0, :])
        times = pd.to_datetime(tws_ds.time.values)
        sources = tws_ds.data_source.values
        
        colors = ['blue' if s == 'grace_period' else 'red' for s in sources]
        ax1.scatter(times, np.ones_like(times), c=colors, s=50, alpha=0.7)
        ax1.set_ylim(0.5, 1.5)
        ax1.set_yticks([])
        ax1.set_xlabel('Time')
        ax1.set_title('Data Coverage Timeline', fontsize=14, fontweight='bold')
        ax1.grid(True, axis='x', alpha=0.3)
        
        # Legend
        grace_patch = mpatches.Patch(color='blue', label='GRACE operational period')
        filled_patch = mpatches.Patch(color='red', label='Gap-filled')
        ax1.legend(handles=[grace_patch, filled_patch], loc='upper right')
        
        # 2. TWS time series
        ax2 = fig.add_subplot(gs[1, :])
        mean_tws = tws_ds.tws.mean(dim=['lat', 'lon'])
        
        grace_mask = sources == 'grace_period'
        gap_mask = sources == 'gap_filled'
        
        # Plot with different styles
        if grace_mask.any():
            ax2.plot(times[grace_mask], mean_tws[grace_mask], 'b-', 
                    linewidth=2, label='GRACE period', alpha=0.8)
        if gap_mask.any():
            ax2.plot(times[gap_mask], mean_tws[gap_mask], 'r--', 
                    linewidth=2, label='Gap-filled', alpha=0.8)
        
        # Connect lines
        ax2.plot(times, mean_tws, 'k-', linewidth=0.5, alpha=0.2)
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('TWS Anomaly (cm)')
        ax2.set_title('Complete Total Water Storage Time Series', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Groundwater time series
        ax3 = fig.add_subplot(gs[2, 0])
        mean_gws = gws_ds.groundwater.mean(dim=['lat', 'lon'])
        
        if grace_mask.any():
            ax3.plot(times[grace_mask], mean_gws[grace_mask], 'b-', 
                    linewidth=2, alpha=0.8)
        if gap_mask.any():
            ax3.plot(times[gap_mask], mean_gws[gap_mask], 'r--', 
                    linewidth=2, alpha=0.8)
        
        ax3.plot(times, mean_gws, 'k-', linewidth=0.5, alpha=0.2)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Groundwater Anomaly (cm)')
        ax3.set_title('Complete Groundwater Storage', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis('off')
        
        n_total = len(times)
        n_grace = np.sum(grace_mask)
        n_filled = np.sum(gap_mask)
        coverage_original = n_grace / n_total * 100
        
        summary_text = f"""
DATA SUMMARY ({mode.upper()})

Total months: {n_total}
GRACE period: {n_grace} months
Gap-filled: {n_filled} months
Original coverage: {coverage_original:.1f}%

Key Statistics:
• TWS range: [{float(tws_ds.tws.min()):.1f}, {float(tws_ds.tws.max()):.1f}] cm
• TWS mean: {float(tws_ds.tws.mean()):.2f} cm
• GWS range: [{float(gws_ds.groundwater.min()):.1f}, {float(gws_ds.groundwater.max()):.1f}] cm
• GWS mean: {float(gws_ds.groundwater.mean()):.2f} cm

Benefits:
✓ Continuous time series
✓ No gaps for trend analysis
✓ Complete seasonal cycles
✓ Better climate studies
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(f'Complete Water Storage Analysis - {mode.title()} Mode', 
                    fontsize=16, fontweight='bold')
        
        # Save figure
        fig_path = Path("figures") / f"water_storage_complete_{mode}.png"
        fig_path.parent.mkdir(exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved to {fig_path}")
    
    def _calculate_both_modes(self, feature_ds, grace_months):
        """Calculate outputs for both complete and grace_only modes."""
        print("\n📊 Calculating both complete and GRACE-only outputs...")
        
        # Complete mode
        print("\n[1/2] Complete mode...")
        self.calculate_complete_storage('complete')
        
        # GRACE-only mode
        print("\n[2/2] GRACE-only mode...")
        self.calculate_complete_storage('grace_only')
        
        return {
            'complete': {
                'tws': xr.open_dataset(self.results_dir / "tws_complete.nc"),
                'groundwater': xr.open_dataset(self.results_dir / "groundwater_complete.nc")
            },
            'grace_only': {
                'tws': xr.open_dataset(self.results_dir / "tws_grace_only.nc"),
                'groundwater': xr.open_dataset(self.results_dir / "groundwater_grace_only.nc")
            }
        }
    
    def compare_outputs(self):
        """Compare complete vs GRACE-only outputs."""
        print("\n📊 COMPARING OUTPUTS")
        print("="*50)
        
        # Check if both versions exist
        complete_tws = self.results_dir / "tws_complete.nc"
        grace_tws = self.results_dir / "tws_grace_only.nc"
        
        if not (complete_tws.exists() and grace_tws.exists()):
            print("❌ Both versions need to be generated first!")
            print("   Run with --mode both")
            return
        
        # Load datasets
        tws_complete = xr.open_dataset(complete_tws)
        tws_grace = xr.open_dataset(grace_tws)
        
        print(f"\nComplete dataset: {len(tws_complete.time)} months")
        print(f"GRACE-only dataset: {len(tws_grace.time)} months")
        print(f"Additional months: {len(tws_complete.time) - len(tws_grace.time)}")
        
        # Calculate improvement
        total_possible = 12 * 20  # 20 years
        coverage_complete = len(tws_complete.time) / total_possible * 100
        coverage_grace = len(tws_grace.time) / total_possible * 100
        
        print(f"\nCoverage improvement:")
        print(f"  GRACE-only: {coverage_grace:.1f}%")
        print(f"  Complete: {coverage_complete:.1f}%")
        print(f"  Improvement: +{coverage_complete - coverage_grace:.1f} percentage points")
        
        # Compare statistics
        print(f"\nTWS Statistics:")
        print(f"  Complete - Mean: {float(tws_complete.tws.mean()):.2f} cm, "
              f"Std: {float(tws_complete.tws.std()):.2f} cm")
        print(f"  GRACE-only - Mean: {float(tws_grace.tws.mean()):.2f} cm, "
              f"Std: {float(tws_grace.tws.std()):.2f} cm")


def main():
    """Main function for complete water storage calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calculate complete TWS and groundwater with gap-filling"
    )
    parser.add_argument(
        '--mode',
        choices=['complete', 'grace_only', 'both'],
        default='complete',
        help='Output mode (default: complete)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare complete vs GRACE-only outputs'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to specific model file'
    )
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = CompleteWaterStorageCalculator(model_path=args.model)
    
    if args.compare:
        calculator.compare_outputs()
    else:
        calculator.calculate_complete_storage(args.mode)


if __name__ == "__main__":
    main()