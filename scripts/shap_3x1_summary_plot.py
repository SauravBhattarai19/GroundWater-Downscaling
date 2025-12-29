#!/usr/bin/env python3
"""
Custom SHAP Feature Importance Summary Plot Generator
====================================================

This script creates a 3-column, 1-row SHAP feature importance summary plot 
for better visualization of multiple models side-by-side.

Features:
- 3x1 subplot layout for better horizontal comparison
- Loads existing SHAP data or computes from models
- Customizable plot parameters
- High-quality output for publications

Usage:
    python scripts/shap_3x1_summary_plot.py
    python scripts/shap_3x1_summary_plot.py --data-source csv
    python scripts/shap_3x1_summary_plot.py --models models_coarse_to_fine_simple
"""

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

# SHAP imports
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ SHAP not available - limited functionality")

# ML imports
import joblib
from sklearn.ensemble import RandomForestRegressor
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from src_new_approach.utils_downscaling import load_config

warnings.filterwarnings('ignore')


class SHAP3x1PlotGenerator:
    """
    Custom SHAP plot generator for 3x1 layout.
    """
    
    def __init__(self, output_dir: str = "figures_shap"):
        """
        Initialize the plot generator.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.shap_values = {}
        self.feature_names = []
        self.X_data = None
        self.models = {}
        
        # Plot settings
        self.setup_plotting_style()
        
        print(f"🎨 SHAP 3x1 Plot Generator initialized")
        print(f"   Output directory: {self.output_dir}")
    
    def setup_plotting_style(self):
        """Setup matplotlib and seaborn styles for publication-quality plots."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("deep")
        
        # Custom plotting parameters
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white',
            'lines.linewidth': 1.5,
        })
    
    def load_from_csv(self, csv_path: str = "figures_shap/feature_importance_values.csv") -> bool:
        """
        Load feature importance data from CSV file.
        
        Parameters:
        -----------
        csv_path : str
            Path to feature importance CSV file
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            # Load feature importance CSV
            importance_df = pd.read_csv(csv_path, index_col=0)
            
            # Extract model columns (exclude 'mean' if present)
            model_columns = [col for col in importance_df.columns if col != 'mean']
            
            # Get feature names
            self.feature_names = importance_df.index.tolist()
            
            # Create mock SHAP values from importance scores
            # Note: This is a simplified representation for plotting
            print(f"📊 Loading from CSV: {csv_path}")
            print(f"   Models found: {model_columns}")
            print(f"   Features: {len(self.feature_names)}")
            
            # Store the importance data
            self.importance_data = importance_df[model_columns]
            self.csv_mode = True
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading CSV data: {e}")
            return False
    
    def load_from_pickle(self, pickle_path: str = "figures_shap/shap_data/shap_analysis_complete.pkl") -> bool:
        """
        Load SHAP data from pickle file.
        
        Parameters:
        -----------
        pickle_path : str
            Path to pickle file containing SHAP data
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            self.shap_values = data.get('shap_values', {})
            self.feature_names = data.get('feature_names', [])
            self.X_data = data.get('X_data', None)
            
            self.csv_mode = False
            
            print(f"📦 Loaded SHAP pickle data:")
            print(f"   Models: {list(self.shap_values.keys())}")
            print(f"   Features: {len(self.feature_names)}")
            print(f"   Samples: {self.X_data.shape[0] if self.X_data is not None else 'N/A'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading pickle data: {e}")
            return False
    
    def load_from_npy_files(self, shap_dir: str = "figures_shap/shap_data") -> bool:
        """
        Load SHAP data from individual .npy files.
        
        Parameters:
        -----------
        shap_dir : str
            Directory containing SHAP .npy files
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            shap_path = Path(shap_dir)
            
            # Load SHAP values for each model
            shap_files = list(shap_path.glob("*_shap_values.npy"))
            
            for shap_file in shap_files:
                model_name = shap_file.stem.replace('_shap_values', '')
                self.shap_values[model_name] = np.load(shap_file)
            
            # Load feature data
            X_file = shap_path / "X_data.npy"
            if X_file.exists():
                self.X_data = np.load(X_file)
            
            # Load feature names
            feature_names_file = shap_path / "feature_names.txt"
            if feature_names_file.exists():
                with open(feature_names_file, 'r') as f:
                    self.feature_names = [line.strip() for line in f]
            
            self.csv_mode = False
            
            print(f"📁 Loaded SHAP .npy files:")
            print(f"   Models: {list(self.shap_values.keys())}")
            print(f"   Features: {len(self.feature_names)}")
            print(f"   Samples: {self.X_data.shape[0] if self.X_data is not None else 'N/A'}")
            
            return len(self.shap_values) > 0
            
        except Exception as e:
            print(f"❌ Error loading .npy files: {e}")
            return False
    
    def compute_fresh_shap(self, 
                          models_dir: str = "models_coarse_to_fine_simple",
                          features_path: str = "processed_coarse_to_fine/feature_stack_55km.nc",
                          config_path: str = "src_new_approach/config_coarse_to_fine.yaml",
                          sample_size: int = 1000) -> bool:
        """
        Compute fresh SHAP values from models and data.
        
        Parameters:
        -----------
        models_dir : str
            Directory containing trained models
        features_path : str
            Path to features NetCDF file
        config_path : str
            Path to configuration file
        sample_size : int
            Number of samples for SHAP computation
            
        Returns:
        --------
        bool
            Success status
        """
        if not HAS_SHAP:
            print("❌ SHAP not available - cannot compute fresh values")
            return False
        
        try:
            # Load configuration
            config = load_config(config_path)
            enabled_models = config.get('models', {}).get('enabled', ['rf', 'xgb', 'lgb'])
            
            # Load models
            models_path = Path(models_dir)
            for model_name in enabled_models:
                model_file = models_path / f"{model_name}_coarse_model.joblib"
                if model_file.exists():
                    self.models[model_name] = joblib.load(model_file)
                    print(f"✅ Loaded {model_name} model")
            
            if not self.models:
                print("❌ No models loaded")
                return False
            
            # Load and prepare feature data (simplified version)
            ds = xr.open_dataset(features_path)
            
            if 'features' in ds and 'static_features' in ds:
                # Aggregated format with separate temporal and static features
                temporal_features = ds['features'].values.transpose(0, 2, 3, 1)
                static_features = ds['static_features'].values
                
                n_times = len(ds.time)
                static_broadcasted = np.broadcast_to(
                    static_features[np.newaxis, :, :, :],
                    (n_times,) + static_features.shape
                ).transpose(0, 2, 3, 1)
                
                feature_data = np.concatenate([temporal_features, static_broadcasted], axis=-1)
                
                temporal_names = [str(f) for f in ds['feature'].values]
                static_names = [str(f) for f in ds['static_feature'].values]
                self.feature_names = temporal_names + static_names
                
                n_times, n_lat, n_lon, n_features = feature_data.shape
                X_full = feature_data.reshape(-1, n_features)
                
            else:
                print("❌ Unsupported data format")
                return False
            
            # Remove NaN rows and sample
            valid_mask = ~np.isnan(X_full).any(axis=1)
            X_valid = X_full[valid_mask]
            
            if len(X_valid) > sample_size:
                indices = np.random.choice(len(X_valid), sample_size, replace=False)
                self.X_data = X_valid[indices]
            else:
                self.X_data = X_valid
            
            print(f"📊 Prepared data shape: {self.X_data.shape}")
            
            # Compute SHAP values
            self._compute_shap_values()
            
            self.csv_mode = False
            return len(self.shap_values) > 0
            
        except Exception as e:
            print(f"❌ Error computing fresh SHAP: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _compute_shap_values(self):
        """Compute SHAP values for loaded models."""
        if not HAS_SHAP or not self.models or self.X_data is None:
            return
        
        print("\n🧠 Computing SHAP values for 3x1 plot...")
        
        for model_name, model in self.models.items():
            try:
                print(f"   Computing for {model_name}...")
                
                # Choose appropriate explainer
                if model_name in ['rf', 'lgb'] or isinstance(model, RandomForestRegressor):
                    explainer = shap.TreeExplainer(model)
                elif model_name == 'xgb' and HAS_XGB:
                    explainer = shap.TreeExplainer(model)
                else:
                    background = shap.sample(self.X_data, min(100, len(self.X_data)))
                    explainer = shap.KernelExplainer(model.predict, background)
                
                # Compute SHAP values
                shap_values = explainer.shap_values(self.X_data)
                self.shap_values[model_name] = shap_values
                
                print(f"   ✅ {model_name}: {shap_values.shape}")
                
            except Exception as e:
                print(f"   ❌ Error computing SHAP for {model_name}: {e}")
                continue
    
    def create_3x1_summary_plot(self, 
                               max_features: int = 15,
                               figsize: Tuple[int, int] = (20, 8),
                               output_name: str = "shap_3x1_feature_importance_summary.png") -> str:
        """
        Create 3x1 SHAP feature importance summary plot.
        
        Parameters:
        -----------
        max_features : int
            Maximum features to display
        figsize : tuple
            Figure size (width, height)
        output_name : str
            Output filename
            
        Returns:
        --------
        str
            Path to saved plot
        """
        if hasattr(self, 'csv_mode') and self.csv_mode:
            return self._create_3x1_from_csv(max_features, figsize, output_name)
        elif self.shap_values and HAS_SHAP:
            return self._create_3x1_from_shap(max_features, figsize, output_name)
        else:
            print("❌ No data available for plotting")
            return ""
    
    def _create_3x1_from_shap(self, max_features: int, figsize: tuple, output_name: str) -> str:
        """Create 3x1 plot from SHAP values."""
        model_names = list(self.shap_values.keys())
        n_models = len(model_names)
        
        if n_models == 0:
            print("❌ No SHAP values available")
            return ""
        
        # Create figure with 3 columns, 1 row
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif n_models == 2:
            axes = axes[:2]
        
        for idx, model_name in enumerate(model_names[:3]):  # Limit to 3 models
            plt.sca(axes[idx])
            
            try:
                shap_vals = self.shap_values[model_name]
                
                # Create SHAP summary plot
                shap.summary_plot(
                    shap_vals, 
                    self.X_data,
                    feature_names=self.feature_names,
                    show=False,
                    max_display=max_features,
                    plot_size=None  # Let matplotlib handle sizing
                )
                
                # Customize title and formatting
                plt.title(f'{model_name.upper()}\nFeature Importance', 
                         fontsize=16, fontweight='bold', pad=15)
                plt.xlabel('SHAP Value', fontsize=14)
                
                # Adjust tick label sizes
                plt.xticks(fontsize=11)
                plt.yticks(fontsize=10)
                
            except Exception as e:
                # Fallback error display
                axes[idx].text(0.5, 0.5, f'Error plotting {model_name}:\n{str(e)}', 
                             ha='center', va='center', fontsize=12,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
                axes[idx].set_title(f'{model_name.upper()} - Error', fontweight='bold')
                axes[idx].set_xlim([0, 1])
                axes[idx].set_ylim([0, 1])
        
        # Hide unused subplots if less than 3 models
        for idx in range(n_models, 3):
            if idx < len(axes):
                axes[idx].set_visible(False)
        
        # Adjust layout and save
        #plt.suptitle('SHAP Feature Importance Summary (All Models)', 
        #            fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        # Save plot
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 3x1 SHAP summary plot saved: {output_path}")
        return str(output_path)
    
    def _create_3x1_from_csv(self, max_features: int, figsize: tuple, output_name: str) -> str:
        """Create 3x1 plot from CSV importance data."""
        if not hasattr(self, 'importance_data'):
            print("❌ No importance data loaded")
            return ""
        
        model_names = list(self.importance_data.columns)
        n_models = len(model_names)
        
        # Create figure with 3 columns, 1 row
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif n_models == 2:
            axes = axes[:2]
        
        for idx, model_name in enumerate(model_names[:3]):  # Limit to 3 models
            # Get top features for this model
            model_importance = self.importance_data[model_name].sort_values(ascending=False)
            top_features = model_importance.head(max_features)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(top_features))
            axes[idx].barh(y_pos, top_features.values, alpha=0.8, color=f'C{idx}')
            
            # Customize plot
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(top_features.index, fontsize=10)
            axes[idx].invert_yaxis()  # Top feature at top
            axes[idx].set_xlabel('Mean |SHAP Value|', fontsize=12)
            axes[idx].set_title(f'{model_name.upper()}\nFeature Importance', 
                              fontsize=16, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, v in enumerate(top_features.values):
                axes[idx].text(v + max(top_features.values) * 0.01, i, f'{v:.3f}', 
                             va='center', fontsize=9)
        
        # Hide unused subplots if less than 3 models
        for idx in range(n_models, 3):
            if idx < len(axes):
                axes[idx].set_visible(False)
        
        # Adjust layout and save
        plt.suptitle('Feature Importance Summary (All Models)', 
                    fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        # Save plot
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 3x1 importance plot saved: {output_path}")
        return str(output_path)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate 3x1 SHAP feature importance summary plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--data-source',
        choices=['auto', 'csv', 'pickle', 'npy', 'fresh'],
        default='auto',
        help='Data source for SHAP values (default: auto-detect)'
    )
    
    parser.add_argument(
        '--csv-path',
        default='figures_shap/feature_importance_values.csv',
        help='Path to feature importance CSV file'
    )
    
    parser.add_argument(
        '--pickle-path',
        default='figures_shap/shap_data/shap_analysis_complete.pkl',
        help='Path to SHAP pickle file'
    )
    
    parser.add_argument(
        '--shap-dir',
        default='figures_shap/shap_data',
        help='Directory containing SHAP .npy files'
    )
    
    parser.add_argument(
        '--models-dir',
        default='models_coarse_to_fine_simple',
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--features',
        default='processed_coarse_to_fine/feature_stack_55km.nc',
        help='Path to features NetCDF file'
    )
    
    parser.add_argument(
        '--config',
        default='src_new_approach/config_coarse_to_fine.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--output-dir',
        default='figures_shap',
        help='Output directory for plots'
    )
    
    parser.add_argument(
        '--max-features',
        type=int,
        default=15,
        help='Maximum number of features to display'
    )
    
    parser.add_argument(
        '--figsize',
        nargs=2,
        type=int,
        default=[20, 8],
        help='Figure size (width height)'
    )
    
    parser.add_argument(
        '--output-name',
        default='shap_3x1_feature_importance_summary.png',
        help='Output filename'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SHAP3x1PlotGenerator(output_dir=args.output_dir)
    
    # Load data based on source preference
    data_loaded = False
    
    if args.data_source == 'auto':
        # Auto-detect available data sources
        print("🔍 Auto-detecting available data sources...")
        
        # Try pickle first (most complete)
        if Path(args.pickle_path).exists():
            print("   Found pickle file - loading...")
            data_loaded = generator.load_from_pickle(args.pickle_path)
        
        # Try .npy files
        elif Path(args.shap_dir).exists() and list(Path(args.shap_dir).glob("*_shap_values.npy")):
            print("   Found .npy files - loading...")
            data_loaded = generator.load_from_npy_files(args.shap_dir)
        
        # Fall back to CSV
        elif Path(args.csv_path).exists():
            print("   Found CSV file - loading...")
            data_loaded = generator.load_from_csv(args.csv_path)
        
        # Compute fresh if nothing available
        else:
            print("   No existing data found - computing fresh...")
            data_loaded = generator.compute_fresh_shap(
                models_dir=args.models_dir,
                features_path=args.features,
                config_path=args.config
            )
    
    elif args.data_source == 'csv':
        data_loaded = generator.load_from_csv(args.csv_path)
    elif args.data_source == 'pickle':
        data_loaded = generator.load_from_pickle(args.pickle_path)
    elif args.data_source == 'npy':
        data_loaded = generator.load_from_npy_files(args.shap_dir)
    elif args.data_source == 'fresh':
        data_loaded = generator.compute_fresh_shap(
            models_dir=args.models_dir,
            features_path=args.features,
            config_path=args.config
        )
    
    if not data_loaded:
        print("❌ Failed to load any data - cannot generate plot")
        return 1
    
    # Generate the 3x1 plot
    output_path = generator.create_3x1_summary_plot(
        max_features=args.max_features,
        figsize=tuple(args.figsize),
        output_name=args.output_name
    )
    
    if output_path:
        print(f"\n✅ 3x1 SHAP plot generated successfully!")
        print(f"📁 Saved to: {output_path}")
        return 0
    else:
        print("❌ Failed to generate plot")
        return 1


if __name__ == "__main__":
    sys.exit(main())