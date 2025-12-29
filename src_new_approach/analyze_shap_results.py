#!/usr/bin/env python3
"""
SHAP Analysis and Visualization Tool

This script provides comprehensive SHAP (SHapley Additive exPlanations) analysis 
and visualization for machine learning models, focusing on:
1. Feature importance analysis
2. Model interpretability visualization  
3. Feature interaction discovery
4. Scientific insight generation

CRITICAL BUG FIX (v2.0):
========================
Fixed scaling issue for Neural Network SHAP computation that was causing extreme 
values (-100,000+). The issue was:
- NN models were trained with scaled features (RobustScaler)
- SHAP computation was using raw unscaled features  
- This caused impossible predictions and extreme SHAP values

The fix:
- Automatically loads feature scalers for NN models
- Applies proper feature scaling before SHAP computation
- Validates SHAP ranges to catch scaling issues
- Tree models (XGB, LGB) unaffected as they don't need scaling

Usage:
    python analyze_shap_results.py --shap-data shap_values.pkl --output-dir figures/
    python analyze_shap_results.py --models models/ --features features.nc --config config.yaml
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

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

from utils_downscaling import load_config


class SHAPAnalysisVisualizer:
    """
    Comprehensive SHAP analysis and visualization tool.
    """
    
    def __init__(self, output_dir: str = "shap_analysis"):
        """
        Initialize SHAP analyzer.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for plots and results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.models = {}
        self.scalers = {}  # For feature scaling (critical for NN models)
        self.shap_values = {}
        self.feature_names = []
        self.X_data = None
        self.y_data = None
        
        # Setup plotting
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        print(f"🧠 SHAP Analysis Visualizer initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   SHAP available: {HAS_SHAP}")
    
    def load_shap_data(self, shap_data_path: str) -> None:
        """
        Load pre-computed SHAP values from pickle file.
        
        Parameters:
        -----------
        shap_data_path : str
            Path to SHAP data pickle file
        """
        try:
            with open(shap_data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.shap_values = data.get('shap_values', {})
            self.feature_names = data.get('feature_names', [])
            self.X_data = data.get('X_data', None)
            self.y_data = data.get('y_data', None)
            
            print(f"✅ Loaded SHAP data:")
            print(f"   Models: {list(self.shap_values.keys())}")
            print(f"   Features: {len(self.feature_names)}")
            print(f"   Samples: {self.X_data.shape[0] if self.X_data is not None else 'N/A'}")
            
        except Exception as e:
            print(f"❌ Could not load SHAP data: {e}")
    
    def load_models_and_compute_shap(self, 
                                   models_dir: str,
                                   features_path: str,
                                   config_path: str,
                                   sample_size: int = 1000) -> None:
        """
        Load models and compute SHAP values.
        
        Parameters:
        -----------
        models_dir : str
            Directory containing trained models
        features_path : str
            Path to features NetCDF file
        config_path : str
            Path to configuration file
        sample_size : int
            Number of samples for SHAP analysis
        """
        if not HAS_SHAP:
            print("❌ SHAP not available - cannot compute SHAP values")
            return
        
        try:
            # Load configuration
            config = load_config(config_path)
            enabled_models = config.get('models', {}).get('enabled', ['rf', 'xgb', 'lgb'])
            
            # Load models and scalers
            models_path = Path(models_dir)
            self.scalers = {}  # Initialize scalers dictionary
            
            for model_name in enabled_models:
                model_file = models_path / f"{model_name}_coarse_model.joblib"
                if model_file.exists():
                    self.models[model_name] = joblib.load(model_file)
                    print(f"✅ Loaded {model_name} model")
                else:
                    print(f"⚠️ Model file not found: {model_file}")
                
                # Load scaler if it exists (needed for NN model)
                scaler_file = models_path / f"{model_name}_scaler.joblib"
                if scaler_file.exists():
                    self.scalers[model_name] = joblib.load(scaler_file)
                    print(f"✅ Loaded {model_name} scaler")
                else:
                    print(f"ℹ️ No scaler found for {model_name}")
            
            # Load features data
            ds = xr.open_dataset(features_path)
            
            # Check if we have separate temporal and static features (like training data)
            if 'features' in ds and 'static_features' in ds:
                # Aggregated format with separate temporal and static features
                temporal_features = ds['features'].values  # (time, feature, lat, lon)
                static_features = ds['static_features'].values  # (static_feature, lat, lon)
                
                # Transpose temporal features to (time, lat, lon, feature)
                temporal_features = temporal_features.transpose(0, 2, 3, 1)
                
                # Broadcast static features across time
                n_times = len(ds.time)
                static_broadcasted = np.broadcast_to(
                    static_features[np.newaxis, :, :, :],  # (1, static_feature, lat, lon)
                    (n_times,) + static_features.shape    # (time, static_feature, lat, lon)
                ).transpose(0, 2, 3, 1)  # (time, lat, lon, static_feature)
                
                # Combine temporal and static features
                feature_data = np.concatenate([temporal_features, static_broadcasted], axis=-1)  # (time, lat, lon, total_features)
                
                # Get feature names
                temporal_names = [str(f) if hasattr(f, 'item') else f 
                                for f in ds['feature'].values]
                static_names = [str(f) if hasattr(f, 'item') else f 
                              for f in ds['static_feature'].values]
                feature_names = temporal_names + static_names
                
                # Reshape to samples x features
                n_times, n_lat, n_lon, n_features = feature_data.shape
                X_full = feature_data.reshape(-1, n_features)  # (time*lat*lon, features)
                
            elif 'features' in ds:
                # Simple aggregated format: (time, lat, lon, feature)
                feature_data = ds['features'].values  # (time, lat, lon, feature)
                n_times, n_lat, n_lon, n_features = feature_data.shape
                
                # Get feature names
                if 'feature' in ds:
                    feature_names = [str(f) for f in ds['feature'].values]
                else:
                    feature_names = [f'feature_{i}' for i in range(n_features)]
                
                # Reshape to samples x features
                X_full = feature_data.reshape(-1, n_features)  # (time*lat*lon, features)
                
            else:
                # Variable format - extract each variable
                feature_vars = [var for var in ds.data_vars if var != 'tws_anomaly']
                feature_names = feature_vars
                
                X_full = []
                for var in feature_vars:
                    data = ds[var].values
                    if data.ndim == 3:  # (time, lat, lon)
                        data = data.reshape(-1)  # flatten
                    X_full.append(data)
                
                X_full = np.column_stack(X_full)  # (samples, features)
            
            # Remove NaN rows
            valid_mask = ~np.isnan(X_full).any(axis=1)
            self.X_data = X_full[valid_mask]
            
            # Sample data for efficiency
            if len(self.X_data) > sample_size:
                indices = np.random.choice(len(self.X_data), sample_size, replace=False)
                self.X_data = self.X_data[indices]
            
            self.feature_names = feature_names
            
            print(f"✅ Loaded features:")
            print(f"   Shape: {self.X_data.shape}")
            print(f"   Features: {len(self.feature_names)}")
            
            # Compute SHAP values for each model
            self._compute_shap_values()
            
        except Exception as e:
            print(f"❌ Error loading models and computing SHAP: {e}")
            import traceback
            traceback.print_exc()
    
    def _compute_shap_values(self) -> None:
        """Compute SHAP values for all loaded models."""
        if not HAS_SHAP or not self.models or self.X_data is None:
            print("⚠️ Cannot compute SHAP values - missing requirements")
            return
        
        print("\n🧠 Computing SHAP values...")
        
        for model_name, model in self.models.items():
            try:
                print(f"   Computing for {model_name}...")
                
                # Choose appropriate explainer
                if model_name in ['rf', 'lgb'] or isinstance(model, (RandomForestRegressor)):
                    # TreeExplainer for tree-based models
                    explainer = shap.TreeExplainer(model)
                elif model_name == 'xgb' and HAS_XGB:
                    # TreeExplainer for XGBoost
                    explainer = shap.TreeExplainer(model)
                else:
                    # KernelExplainer for NN and other models
                    background = shap.sample(self.X_data, min(100, len(self.X_data)))
                    
                    # CRITICAL FIX: Apply feature scaling for NN model
                    if model_name == 'nn' and hasattr(self, 'scalers') and model_name in self.scalers:
                        print(f"   🔧 Applying feature scaling for {model_name} SHAP computation...")
                        scaler = self.scalers[model_name]
                        
                        def scaled_predict(X):
                            """Prediction wrapper that applies feature scaling for NN."""
                            X_scaled = scaler.transform(X)
                            return model.predict(X_scaled)
                        
                        explainer = shap.KernelExplainer(scaled_predict, background)
                        print(f"   ✅ Using scaled prediction wrapper for {model_name}")
                        
                    else:
                        # Standard approach for other models
                        explainer = shap.KernelExplainer(model.predict, background)
                        if model_name == 'nn':
                            print(f"   ⚠️ Warning: No scaler found for NN - may produce extreme SHAP values")
                
                # Compute SHAP values
                shap_values = explainer.shap_values(self.X_data)
                self.shap_values[model_name] = shap_values
                
                # Validate SHAP values (catch scaling issues)
                shap_range = [np.min(shap_values), np.max(shap_values)]
                max_abs_shap = np.max(np.abs(shap_values))
                
                print(f"   ✅ {model_name}: {shap_values.shape}")
                print(f"      SHAP range: [{shap_range[0]:.1f}, {shap_range[1]:.1f}]")
                
                # Warning for extreme values (may indicate scaling issues)
                if max_abs_shap > 1000:
                    print(f"   ⚠️ WARNING: Extreme SHAP values detected for {model_name}")
                    print(f"      Max |SHAP|: {max_abs_shap:.1f} (may indicate scaling issues)")
                else:
                    print(f"      Max |SHAP|: {max_abs_shap:.1f} ✅")
                
            except Exception as e:
                print(f"   ❌ Error computing SHAP for {model_name}: {e}")
                continue
    
    def plot_feature_importance_summary(self) -> None:
        """Plot SHAP summary plots for all models."""
        if not self.shap_values or not HAS_SHAP:
            print("⚠️ No SHAP values available")
            return
        
        n_models = len(self.shap_values)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for idx, (model_name, shap_vals) in enumerate(self.shap_values.items()):
            if idx >= 4:  # Limit to 4 models
                break
                
            plt.sca(axes[idx])
            
            try:
                # Create summary plot
                shap.summary_plot(
                    shap_vals, 
                    self.X_data,
                    feature_names=self.feature_names,
                    show=False,
                    max_display=15
                )
                
                plt.title(f'{model_name.upper()} - Feature Importance Summary', fontsize=14)
                
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[idx].set_title(f'{model_name.upper()} - Error')
        
        # Hide unused subplots
        for idx in range(len(self.shap_values), 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "shap_feature_importance_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Feature importance summary saved: {plot_path}")
    
    def plot_feature_importance_bar(self) -> None:
        """Plot bar charts of mean absolute SHAP values."""
        if not self.shap_values:
            print("⚠️ No SHAP values available")
            return
        
        # Calculate mean absolute SHAP values for each model
        importance_data = {}
        
        for model_name, shap_vals in self.shap_values.items():
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            importance_data[model_name] = mean_abs_shap
        
        # Create DataFrame
        importance_df = pd.DataFrame(importance_data, index=self.feature_names)
        importance_df = importance_df.fillna(0)
        
        # Sort by mean importance across all models
        importance_df['mean'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('mean', ascending=True)
        
        # Take top 20 features
        top_features = importance_df.tail(20)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot horizontal bar chart
        x_pos = np.arange(len(top_features))
        width = 0.8 / len(self.shap_values)
        
        for idx, model_name in enumerate(self.shap_values.keys()):
            values = top_features[model_name].values
            ax.barh(x_pos + idx * width, values, width, 
                   label=model_name.upper(), alpha=0.8)
        
        ax.set_yticks(x_pos + width * (len(self.shap_values) - 1) / 2)
        ax.set_yticklabels(top_features.index, fontsize=10)
        ax.set_xlabel('Mean |SHAP value|', fontsize=12)
        ax.set_title('Top 20 Most Important Features (Mean |SHAP|)', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "shap_feature_importance_bar.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Feature importance bar chart saved: {plot_path}")
        
        # Save importance data
        csv_path = self.output_dir / "feature_importance_values.csv"
        importance_df.to_csv(csv_path)
        print(f"📄 Feature importance data saved: {csv_path}")
    
    def plot_waterfall_examples(self, n_examples: int = 4) -> None:
        """Plot waterfall plots for example predictions."""
        if not self.shap_values or not HAS_SHAP:
            print("⚠️ No SHAP values available")
            return
        
        # Select random examples
        n_samples = min(n_examples, len(self.X_data))
        example_indices = np.random.choice(len(self.X_data), n_samples, replace=False)
        
        for model_name, shap_vals in self.shap_values.items():
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for idx, example_idx in enumerate(example_indices):
                if idx >= 4:
                    break
                
                plt.sca(axes[idx])
                
                try:
                    # Create waterfall plot
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_vals[example_idx],
                            base_values=np.mean(shap_vals),
                            data=self.X_data[example_idx],
                            feature_names=self.feature_names
                        ),
                        max_display=15,
                        show=False
                    )
                    
                    plt.title(f'Example {idx+1}', fontsize=12)
                    
                except Exception as e:
                    axes[idx].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                    axes[idx].set_title(f'Example {idx+1} - Error')
            
            # Hide unused subplots
            for idx in range(n_samples, 4):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'{model_name.upper()} - Prediction Explanations', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / f"shap_waterfall_{model_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🌊 Waterfall plots for {model_name} saved: {plot_path}")
    
    def analyze_feature_groups(self) -> None:
        """Analyze SHAP values by feature groups (e.g., climate, topography)."""
        if not self.shap_values:
            print("⚠️ No SHAP values available")
            return
        
        # Define feature groups based on naming patterns
        feature_groups = {
            'Precipitation': [f for f in self.feature_names if 'pr' in f.lower() or 'precip' in f.lower()],
            'Temperature': [f for f in self.feature_names if any(x in f.lower() for x in ['temp', 'tmmn', 'tmmx'])],
            'Soil Moisture': [f for f in self.feature_names if 'soilmoi' in f.lower()],
            'Evapotranspiration': [f for f in self.feature_names if any(x in f.lower() for x in ['evap', 'aet', 'def'])],
            'Topography': [f for f in self.feature_names if any(x in f.lower() for x in ['elevation', 'slope', 'aspect', 'topo'])],
            'Snow': [f for f in self.feature_names if 'swe' in f.lower()],
            'Land Cover': [f for f in self.feature_names if any(x in f.lower() for x in ['land', 'cover', 'modis'])],
            'Other': []
        }
        
        # Assign ungrouped features to 'Other'
        grouped_features = set()
        for group_features in feature_groups.values():
            grouped_features.update(group_features)
        
        feature_groups['Other'] = [f for f in self.feature_names if f not in grouped_features]
        
        # Calculate group importance for each model
        group_importance = {}
        
        for model_name, shap_vals in self.shap_values.items():
            model_groups = {}
            
            for group_name, group_features in feature_groups.items():
                if not group_features:
                    continue
                
                # Find indices of group features
                group_indices = [i for i, fname in enumerate(self.feature_names) if fname in group_features]
                
                if group_indices:
                    # Sum absolute SHAP values for this group
                    group_shap = np.abs(shap_vals[:, group_indices]).sum(axis=1).mean()
                    model_groups[group_name] = group_shap
            
            group_importance[model_name] = model_groups
        
        # Create grouped importance plot
        group_df = pd.DataFrame(group_importance).fillna(0)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar chart
        x_pos = np.arange(len(group_df))
        width = 0.8 / len(self.shap_values)
        
        for idx, model_name in enumerate(self.shap_values.keys()):
            values = group_df[model_name].values
            ax.bar(x_pos + idx * width, values, width, 
                  label=model_name.upper(), alpha=0.8)
        
        ax.set_xticks(x_pos + width * (len(self.shap_values) - 1) / 2)
        ax.set_xticklabels(group_df.index, rotation=45, ha='right')
        ax.set_ylabel('Mean |SHAP value|', fontsize=12)
        ax.set_title('Feature Group Importance Analysis', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "shap_feature_groups.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Feature group analysis saved: {plot_path}")
        
        # Save group data
        csv_path = self.output_dir / "feature_group_importance.csv"
        group_df.to_csv(csv_path)
        print(f"📄 Feature group data saved: {csv_path}")
    
    def generate_shap_report(self) -> None:
        """Generate comprehensive SHAP analysis report."""
        if not self.shap_values:
            print("⚠️ No SHAP values available for report")
            return
        
        report_path = self.output_dir / "shap_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SHAP INTERPRETABILITY ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Dataset summary
            f.write("DATASET SUMMARY:\n")
            f.write("-"*40 + "\n")
            f.write(f"• Number of features: {len(self.feature_names)}\n")
            f.write(f"• Number of samples: {len(self.X_data) if self.X_data is not None else 'N/A'}\n")
            f.write(f"• Models analyzed: {list(self.shap_values.keys())}\n\n")
            
            # Top features summary
            f.write("TOP IMPORTANT FEATURES (Global):\n")
            f.write("-"*40 + "\n")
            
            # Calculate overall importance
            if self.shap_values:
                overall_importance = {}
                for model_name, shap_vals in self.shap_values.items():
                    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                    for i, importance in enumerate(mean_abs_shap):
                        feature_name = self.feature_names[i]
                        if feature_name not in overall_importance:
                            overall_importance[feature_name] = []
                        overall_importance[feature_name].append(importance)
                
                # Average across models
                avg_importance = {
                    feature: np.mean(importances) 
                    for feature, importances in overall_importance.items()
                }
                
                # Sort and show top 15
                sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
                
                for i, (feature, importance) in enumerate(sorted_features[:15]):
                    f.write(f"{i+1:2d}. {feature}: {importance:.6f}\n")
            
            f.write("\n")
            
            # Model-specific insights
            f.write("MODEL-SPECIFIC INSIGHTS:\n")
            f.write("-"*40 + "\n")
            
            for model_name, shap_vals in self.shap_values.items():
                f.write(f"\n{model_name.upper()}:\n")
                
                # Calculate statistics
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                top_feature_idx = np.argmax(mean_abs_shap)
                top_feature = self.feature_names[top_feature_idx]
                
                f.write(f"  • Most important feature: {top_feature}\n")
                f.write(f"  • Max importance: {mean_abs_shap[top_feature_idx]:.6f}\n")
                f.write(f"  • Mean importance: {mean_abs_shap.mean():.6f}\n")
                f.write(f"  • Importance std: {mean_abs_shap.std():.6f}\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-"*40 + "\n")
            f.write("1. Focus on top 10-15 most important features for model optimization\n")
            f.write("2. Consider feature engineering for consistently important variables\n")
            f.write("3. Investigate domain knowledge for unexpectedly important features\n")
            f.write("4. Use SHAP values for feature selection in future model iterations\n")
            f.write("5. Monitor feature importance changes with new data\n")
            f.write("6. Consider interaction effects for top features\n")
        
        print(f"📋 SHAP analysis report saved: {report_path}")
    
    def save_shap_data(self) -> None:
        """Save raw SHAP values and related data for future use."""
        if not self.shap_values:
            print("⚠️ No SHAP values to save")
            return
        
        # Create shap_data directory
        shap_data_dir = self.output_dir / "shap_data"
        shap_data_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n💾 Saving raw SHAP data...")
        
        for model_name, shap_vals in self.shap_values.items():
            # Save SHAP values as .npy
            shap_path = shap_data_dir / f"{model_name}_shap_values.npy"
            np.save(shap_path, shap_vals)
            print(f"   ✅ {model_name} SHAP values: {shap_path}")
        
        # Save feature data if available
        if self.X_data is not None:
            X_path = shap_data_dir / "X_data.npy"
            np.save(X_path, self.X_data)
            print(f"   ✅ Feature data: {X_path}")
        
        if self.y_data is not None:
            y_path = shap_data_dir / "y_data.npy"
            np.save(y_path, self.y_data)
            print(f"   ✅ Target data: {y_path}")
        
        # Save feature names
        feature_names_path = shap_data_dir / "feature_names.txt"
        with open(feature_names_path, 'w') as f:
            for name in self.feature_names:
                f.write(f"{name}\n")
        print(f"   ✅ Feature names: {feature_names_path}")
        
        # Save scalers if available
        if hasattr(self, 'scalers') and self.scalers:
            for scaler_name, scaler in self.scalers.items():
                scaler_path = shap_data_dir / f"{scaler_name}_scaler.joblib"
                joblib.dump(scaler, scaler_path)
                print(f"   ✅ {scaler_name} scaler: {scaler_path}")
        
        # Save all data as pickle for easy loading
        pickle_path = shap_data_dir / "shap_analysis_complete.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'shap_values': self.shap_values,
                'feature_names': self.feature_names,
                'X_data': self.X_data,
                'y_data': self.y_data,
                'scalers': getattr(self, 'scalers', {}),
                'version': '2.0',
                'scaling_fix_applied': True,
                'description': 'SHAP values computed with proper feature scaling for NN models'
            }, f)
        print(f"   ✅ Complete data (pickle): {pickle_path}")
        print(f"   🔧 Scaling fix metadata included")

    def run_full_analysis(self) -> None:
        """Run complete SHAP analysis pipeline."""
        print(f"\n{'='*60}")
        print("RUNNING COMPREHENSIVE SHAP ANALYSIS")
        print(f"{'='*60}")
        
        if not self.shap_values:
            print("❌ No SHAP values loaded - cannot run analysis")
            return
        
        # Save raw SHAP data first
        print("\n0. Saving raw SHAP data...")
        self.save_shap_data()
        
        # Feature importance summary
        print("\n1. Generating feature importance summary...")
        self.plot_feature_importance_summary()
        
        # Feature importance bar charts
        print("\n2. Creating feature importance bar charts...")
        self.plot_feature_importance_bar()
        
        # Waterfall examples
        print("\n3. Creating waterfall plot examples...")
        self.plot_waterfall_examples()
        
        # Feature group analysis
        print("\n4. Analyzing feature groups...")
        self.analyze_feature_groups()
        
        # Generate report
        print("\n5. Generating analysis report...")
        self.generate_shap_report()
        
        print(f"\n✅ SHAP analysis complete! Results saved to: {self.output_dir}")
        print(f"   • Raw SHAP data: shap_data/")
        print(f"   • Feature importance: shap_feature_importance_*.png")
        print(f"   • Waterfall plots: shap_waterfall_*.png") 
        print(f"   • Feature groups: shap_feature_groups.png")
        print(f"   • Analysis report: shap_analysis_report.txt")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize SHAP values for model interpretability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--shap-data',
        type=str,
        help='Path to pre-computed SHAP values pickle file'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='models_coarse_to_fine_simple',
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--features', 
        type=str,
        default='processed_coarse_to_fine/feature_stack_55km.nc',
        help='Path to features NetCDF file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='src_new_approach/config_coarse_to_fine.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures_shap',
        help='Output directory for analysis results'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=20000,
        help='Number of samples for SHAP computation'
    )
    
    args = parser.parse_args()
    
    # With default values, we always have models, features, and config
    # Only check if shap_data is provided for pre-computed values
    pass
    
    # Initialize analyzer
    analyzer = SHAPAnalysisVisualizer(output_dir=args.output_dir)
    
    # Load data
    if args.shap_data:
        print("📊 Loading pre-computed SHAP data...")
        analyzer.load_shap_data(args.shap_data)
    else:
        print("🤖 Loading models and computing SHAP values...")
        analyzer.load_models_and_compute_shap(
            models_dir=args.models,
            features_path=args.features,
            config_path=args.config,
            sample_size=args.sample_size
        )
    
    # Run analysis
    analyzer.run_full_analysis()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())