#!/usr/bin/env python3
"""
SHAP (SHapley Additive exPlanations) Analysis for GRACE Downscaling Models

This module provides comprehensive interpretability analysis for tree-based models
using SHAP values. Designed to work with 96 environmental features and provide
both global and local explanations for model predictions.

Features:
- Tree-based SHAP explainer optimization
- Global feature importance analysis
- Local prediction explanations (waterfall plots)
- Feature interaction analysis
- Model comparison and ensemble interpretation
- High-performance computation for large datasets
- Scientific visualization and reporting
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# SHAP imports
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ML model imports
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

from sklearn.ensemble import RandomForestRegressor
import joblib

# Local imports
from utils_downscaling import (
    get_config_value, print_statistics, add_metadata_to_dataset
)

warnings.filterwarnings('ignore')

# Set plotting style for scientific publications
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Enhanced plotting parameters for SHAP visualizations
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'lines.linewidth': 1.5,
})


class SHAPAnalyzer:
    """
    Comprehensive SHAP analysis for interpretable machine learning.
    
    Provides global and local explanations for tree-based models with
    optimized computation for large datasets and many features.
    """
    
    def __init__(self, 
                 models: Dict[str, Any],
                 feature_names: List[str],
                 config: Dict,
                 max_samples: int = 1000):
        """
        Initialize SHAP analyzer.
        
        Parameters:
        -----------
        models : Dict[str, Any]
            Dictionary of trained models {model_name: model_object}
        feature_names : List[str]
            List of feature names (96 features)
        config : Dict
            Configuration dictionary
        max_samples : int
            Maximum samples for SHAP computation (performance optimization)
        """
        if not HAS_SHAP:
            raise ImportError("SHAP is required for interpretability analysis. Install with: pip install shap")
        
        self.models = models
        self.feature_names = feature_names
        self.config = config
        self.max_samples = max_samples
        
        # SHAP explainers and values storage
        self.explainers = {}
        self.shap_values = {}
        self.base_values = {}
        
        # Analysis settings
        shap_config = get_config_value(config, 'shap_analysis', {})
        self.enable_interactions = shap_config.get('compute_interactions', True)
        self.top_features = shap_config.get('top_features_display', 20)
        self.sample_explanations = shap_config.get('sample_explanations', 5)
        
        print(f"🔧 SHAP Analyzer initialized:")
        print(f"   Models: {list(models.keys())}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Max samples: {max_samples}")
        print(f"   Top features display: {self.top_features}")
    
    def _create_explainer(self, model_name: str, model: Any, X_background: np.ndarray) -> Any:
        """
        Create appropriate SHAP explainer for the model type.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model : Any
            Trained model object
        X_background : np.ndarray
            Background dataset for SHAP explainer
        
        Returns:
        --------
        Any
            SHAP explainer object
        """
        print(f"   Creating {model_name} explainer...")
        
        if model_name == 'xgb' and HAS_XGB and isinstance(model, xgb.XGBRegressor):
            # Use Tree explainer for XGBoost (most efficient)
            explainer = shap.TreeExplainer(model)
            
        elif model_name == 'lgb' and HAS_LGB and isinstance(model, lgb.LGBMRegressor):
            # Use Tree explainer for LightGBM
            explainer = shap.TreeExplainer(model)
            
        elif model_name == 'rf' and isinstance(model, RandomForestRegressor):
            # Use Tree explainer for Random Forest
            explainer = shap.TreeExplainer(model)
            
        else:
            # Fallback to Kernel explainer (slower but works for any model)
            print(f"   Warning: Using Kernel explainer for {model_name} (slower)")
            # Use smaller background for kernel explainer (performance)
            background_size = min(100, len(X_background))
            background_sample = shap.sample(X_background, background_size)
            explainer = shap.KernelExplainer(model.predict, background_sample)
        
        return explainer
    
    def compute_shap_values(self, 
                           X: np.ndarray, 
                           compute_interactions: bool = None) -> Dict[str, Dict[str, Any]]:
        """
        Compute SHAP values for all models.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features for SHAP computation
        compute_interactions : bool, optional
            Whether to compute interaction values (expensive)
        
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            SHAP values and related data for each model
        """
        print(f"\n{'='*70}")
        print(f"🔍 COMPUTING SHAP VALUES")
        print(f"{'='*70}")
        print(f"   Input shape: {X.shape}")
        
        if compute_interactions is None:
            compute_interactions = self.enable_interactions
        
        # Sample data for efficiency if too large
        if len(X) > self.max_samples:
            print(f"   Sampling {self.max_samples} from {len(X)} samples for efficiency")
            sample_indices = np.random.choice(len(X), self.max_samples, replace=False)
            X_sample = X[sample_indices]
        else:
            X_sample = X.copy()
            sample_indices = np.arange(len(X))
        
        # Use smaller background for tree explainers
        background_size = min(100, len(X_sample))
        X_background = shap.sample(X_sample, background_size)
        
        all_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n🎯 Processing {model_name.upper()}")
            start_time = time.time()
            
            try:
                # Create explainer
                explainer = self._create_explainer(model_name, model, X_background)
                self.explainers[model_name] = explainer
                
                # Compute SHAP values
                print(f"   Computing SHAP values...")
                shap_values = explainer.shap_values(X_sample)
                
                # Handle different explainer output formats
                if isinstance(shap_values, list):
                    # Multi-output case (shouldn't happen for regression)
                    shap_values = shap_values[0]
                
                # Get base value
                if hasattr(explainer, 'expected_value'):
                    if isinstance(explainer.expected_value, (list, np.ndarray)):
                        base_value = explainer.expected_value[0]
                    else:
                        base_value = explainer.expected_value
                else:
                    base_value = 0.0
                
                # Compute feature importance (mean absolute SHAP values)
                feature_importance = np.abs(shap_values).mean(axis=0)
                
                # Compute interaction values if requested and supported
                interaction_values = None
                if compute_interactions:
                    try:
                        print(f"   Computing interaction values...")
                        if hasattr(explainer, 'shap_interaction_values'):
                            # Limit to top features for interactions (expensive)
                            top_indices = np.argsort(feature_importance)[-10:]
                            X_top = X_sample[:, top_indices]
                            
                            interaction_values = explainer.shap_interaction_values(X_top)
                            
                            if isinstance(interaction_values, list):
                                interaction_values = interaction_values[0]
                        
                    except Exception as e:
                        print(f"   Warning: Could not compute interactions for {model_name}: {e}")
                        interaction_values = None
                
                # Store results
                model_results = {
                    'shap_values': shap_values,
                    'base_value': base_value,
                    'feature_importance': feature_importance,
                    'interaction_values': interaction_values,
                    'sample_indices': sample_indices,
                    'explainer': explainer
                }
                
                all_results[model_name] = model_results
                self.shap_values[model_name] = shap_values
                self.base_values[model_name] = base_value
                
                computation_time = time.time() - start_time
                print(f"   ✅ {model_name} completed in {computation_time:.1f}s")
                print(f"   SHAP values shape: {shap_values.shape}")
                print(f"   Base value: {base_value:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error computing SHAP for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n✅ SHAP computation completed for {len(all_results)} models")
        return all_results
    
    def create_summary_plots(self, 
                           shap_results: Dict[str, Dict[str, Any]], 
                           X_data: np.ndarray,
                           output_dir: str = "figures_coarse_to_fine/shap_analysis"):
        """
        Create SHAP summary plots for all models.
        
        Parameters:
        -----------
        shap_results : Dict[str, Dict[str, Any]]
            SHAP computation results
        X_data : np.ndarray
            Feature data for plotting
        output_dir : str
            Directory to save plots
        """
        output_dir = Path(output_dir)
        summary_dir = output_dir / "summary_plots"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Creating SHAP summary plots...")
        
        for model_name, results in shap_results.items():
            print(f"   Creating plots for {model_name.upper()}...")
            
            shap_values = results['shap_values']
            sample_indices = results['sample_indices']
            X_sample = X_data[sample_indices] if len(sample_indices) < len(X_data) else X_data
            
            # 1. Summary plot (beeswarm)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, 
                X_sample, 
                feature_names=self.feature_names,
                max_display=self.top_features,
                show=False
            )
            plt.title(f'{model_name.upper()} - Feature Importance (SHAP Summary)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            summary_path = summary_dir / f"{model_name}_shap_summary.png"
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"     ✅ Summary plot: {summary_path.name}")
            
            # 2. Bar plot (mean absolute SHAP values)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_names,
                plot_type="bar",
                max_display=self.top_features,
                show=False
            )
            plt.title(f'{model_name.upper()} - Mean |SHAP Value| (Feature Importance)', 
                     fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            bar_path = summary_dir / f"{model_name}_shap_bar.png"
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"     ✅ Bar plot: {bar_path.name}")
            
            # 3. Feature importance ranking table
            feature_importance = results['feature_importance']
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Mean_Abs_SHAP': feature_importance,
                'Rank': range(1, len(feature_importance) + 1)
            }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
            importance_df['Rank'] = importance_df.index + 1
            
            # Save top features table
            top_features_path = summary_dir / f"{model_name}_top_features.csv"
            importance_df.head(self.top_features).to_csv(top_features_path, index=False)
            print(f"     ✅ Top features table: {top_features_path.name}")
    
    def create_waterfall_plots(self,
                              shap_results: Dict[str, Dict[str, Any]],
                              X_data: np.ndarray,
                              y_data: np.ndarray = None,
                              n_samples: int = 5,
                              output_dir: str = "figures_coarse_to_fine/shap_analysis"):
        """
        Create waterfall plots for individual predictions.
        
        Parameters:
        -----------
        shap_results : Dict[str, Dict[str, Any]]
            SHAP computation results
        X_data : np.ndarray
            Feature data
        y_data : np.ndarray, optional
            True target values
        n_samples : int
            Number of sample predictions to explain
        output_dir : str
            Directory to save plots
        """
        output_dir = Path(output_dir)
        waterfall_dir = output_dir / "waterfall_plots"
        waterfall_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🌊 Creating SHAP waterfall plots...")
        
        for model_name, results in shap_results.items():
            print(f"   Creating waterfall plots for {model_name.upper()}...")
            
            shap_values = results['shap_values']
            base_value = results['base_value']
            sample_indices = results['sample_indices']
            
            # Select diverse samples for explanation
            n_available = len(shap_values)
            if n_samples > n_available:
                n_samples = n_available
            
            # Choose samples with diverse predictions
            sample_selection = np.linspace(0, n_available - 1, n_samples, dtype=int)
            
            for i, sample_idx in enumerate(sample_selection):
                # Get sample data
                sample_shap = shap_values[sample_idx]
                original_idx = sample_indices[sample_idx] if len(sample_indices) < len(X_data) else sample_idx
                sample_features = X_data[original_idx]
                
                # Create waterfall plot
                plt.figure(figsize=(12, 8))
                
                # Create SHAP explanation object
                explanation = shap.Explanation(
                    values=sample_shap,
                    base_values=base_value,
                    data=sample_features,
                    feature_names=self.feature_names
                )
                
                shap.waterfall_plot(explanation, max_display=15, show=False)
                
                # Add title with prediction info
                prediction = base_value + np.sum(sample_shap)
                title = f'{model_name.upper()} - Sample {i+1} Prediction Explanation\n'
                title += f'Prediction: {prediction:.3f}'
                
                if y_data is not None:
                    true_value = y_data[original_idx]
                    title += f' | True: {true_value:.3f} | Error: {abs(prediction - true_value):.3f}'
                
                plt.suptitle(title, fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                waterfall_path = waterfall_dir / f"{model_name}_waterfall_sample_{i+1}.png"
                plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"     ✅ Created {n_samples} waterfall plots")
    
    def create_interaction_plots(self,
                               shap_results: Dict[str, Dict[str, Any]],
                               X_data: np.ndarray,
                               top_k: int = 10,
                               output_dir: str = "figures_coarse_to_fine/shap_analysis"):
        """
        Create feature interaction plots (if interaction values computed).
        
        Parameters:
        -----------
        shap_results : Dict[str, Dict[str, Any]]
            SHAP computation results
        X_data : np.ndarray
            Feature data
        top_k : int
            Number of top interactions to plot
        output_dir : str
            Directory to save plots
        """
        output_dir = Path(output_dir)
        interaction_dir = output_dir / "interaction_plots"
        interaction_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔄 Creating SHAP interaction plots...")
        
        for model_name, results in shap_results.items():
            interaction_values = results.get('interaction_values')
            
            if interaction_values is None:
                print(f"   No interaction values for {model_name} - skipping")
                continue
                
            print(f"   Creating interaction plots for {model_name.upper()}...")
            
            # Get feature importance for top features
            feature_importance = results['feature_importance']
            top_indices = np.argsort(feature_importance)[-top_k:]
            top_features = [self.feature_names[i] for i in top_indices]
            
            # Create interaction summary plot
            plt.figure(figsize=(12, 10))
            
            # Average interaction strengths
            interaction_summary = np.abs(interaction_values).mean(axis=0)
            
            # Create heatmap
            sns.heatmap(
                interaction_summary[:top_k, :top_k],
                xticklabels=top_features,
                yticklabels=top_features,
                cmap='viridis',
                annot=True,
                fmt='.3f',
                square=True
            )
            
            plt.title(f'{model_name.upper()} - Feature Interactions (Top {top_k})',
                     fontsize=16, fontweight='bold')
            plt.xlabel('Features', fontweight='bold')
            plt.ylabel('Features', fontweight='bold')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            interaction_path = interaction_dir / f"{model_name}_interactions.png"
            plt.savefig(interaction_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"     ✅ Interaction plot: {interaction_path.name}")
    
    def create_feature_dependence_plots(self,
                                      shap_results: Dict[str, Dict[str, Any]],
                                      X_data: np.ndarray,
                                      top_features: int = 5,
                                      output_dir: str = "figures_coarse_to_fine/shap_analysis"):
        """
        Create partial dependence plots for top features.
        
        Parameters:
        -----------
        shap_results : Dict[str, Dict[str, Any]]
            SHAP computation results
        X_data : np.ndarray
            Feature data
        top_features : int
            Number of top features to plot
        output_dir : str
            Directory to save plots
        """
        output_dir = Path(output_dir)
        dependence_dir = output_dir / "dependence_plots"
        dependence_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📈 Creating SHAP dependence plots...")
        
        for model_name, results in shap_results.items():
            print(f"   Creating dependence plots for {model_name.upper()}...")
            
            shap_values = results['shap_values']
            sample_indices = results['sample_indices']
            X_sample = X_data[sample_indices] if len(sample_indices) < len(X_data) else X_data
            
            # Get top features by importance
            feature_importance = results['feature_importance']
            top_indices = np.argsort(feature_importance)[-top_features:][::-1]
            
            for i, feat_idx in enumerate(top_indices):
                plt.figure(figsize=(10, 6))
                
                shap.dependence_plot(
                    feat_idx,
                    shap_values,
                    X_sample,
                    feature_names=self.feature_names,
                    show=False
                )
                
                plt.title(f'{model_name.upper()} - {self.feature_names[feat_idx]} Dependence',
                         fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                feat_name_clean = self.feature_names[feat_idx].replace('/', '_').replace(' ', '_')
                dependence_path = dependence_dir / f"{model_name}_dependence_{feat_name_clean}.png"
                plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"     ✅ Created {top_features} dependence plots")
    
    def create_model_comparison(self,
                              shap_results: Dict[str, Dict[str, Any]],
                              output_dir: str = "figures_coarse_to_fine/shap_analysis"):
        """
        Create comparison plots across models.
        
        Parameters:
        -----------
        shap_results : Dict[str, Dict[str, Any]]
            SHAP computation results for all models
        output_dir : str
            Directory to save plots
        """
        output_dir = Path(output_dir)
        comparison_dir = output_dir / "model_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔍 Creating model comparison plots...")
        
        # Collect feature importance from all models
        all_importance = {}
        for model_name, results in shap_results.items():
            all_importance[model_name] = results['feature_importance']
        
        # Create comparison DataFrame
        importance_df = pd.DataFrame(all_importance, index=self.feature_names)
        
        # 1. Feature importance correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation = importance_df.corr()
        sns.heatmap(
            correlation,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.3f'
        )
        plt.title('Model Feature Importance Correlations', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        correlation_path = comparison_dir / "feature_importance_correlation.png"
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Top features comparison bar plot
        # Get union of top features across models
        top_features_sets = []
        for model_name, results in shap_results.items():
            feature_importance = results['feature_importance']
            top_indices = np.argsort(feature_importance)[-15:]
            top_features_sets.extend([self.feature_names[i] for i in top_indices])
        
        unique_top_features = list(set(top_features_sets))[:20]  # Limit to 20 for visualization
        
        # Create comparison plot
        comparison_data = importance_df.loc[unique_top_features].copy()
        
        plt.figure(figsize=(14, 8))
        comparison_data.plot(kind='bar', width=0.8)
        plt.title('Feature Importance Comparison Across Models', fontsize=16, fontweight='bold')
        plt.xlabel('Features', fontweight='bold')
        plt.ylabel('Mean |SHAP Value|', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Model', title_fontsize=12, fontsize=10)
        plt.tight_layout()
        
        comparison_path = comparison_dir / "top_features_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Save feature importance rankings
        rankings_path = comparison_dir / "feature_importance_rankings.csv"
        importance_df.to_csv(rankings_path)
        
        print(f"   ✅ Model comparison plots created")
        print(f"     • Correlation heatmap: {correlation_path.name}")
        print(f"     • Top features comparison: {comparison_path.name}")
        print(f"     • Feature rankings: {rankings_path.name}")
    
    def generate_comprehensive_report(self,
                                    X_data: np.ndarray,
                                    y_data: np.ndarray = None,
                                    compute_interactions: bool = True,
                                    output_dir: str = "figures_coarse_to_fine/shap_analysis"):
        """
        Generate complete SHAP analysis report with all visualizations.
        
        Parameters:
        -----------
        X_data : np.ndarray
            Feature data
        y_data : np.ndarray, optional
            Target values for prediction error analysis
        compute_interactions : bool
            Whether to compute expensive interaction values
        output_dir : str
            Directory to save all outputs
        """
        print(f"\n{'='*70}")
        print(f"🎯 GENERATING COMPREHENSIVE SHAP REPORT")
        print(f"{'='*70}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Compute SHAP values for all models
        shap_results = self.compute_shap_values(X_data, compute_interactions)
        
        if not shap_results:
            print("❌ No SHAP results computed - aborting report generation")
            return
        
        # 2. Create all visualizations
        self.create_summary_plots(shap_results, X_data, output_dir)
        self.create_waterfall_plots(shap_results, X_data, y_data, 
                                   self.sample_explanations, output_dir)
        self.create_feature_dependence_plots(shap_results, X_data, 5, output_dir)
        self.create_model_comparison(shap_results, output_dir)
        
        if compute_interactions:
            self.create_interaction_plots(shap_results, X_data, 10, output_dir)
        
        # 3. Save SHAP values for future use
        shap_data_dir = output_dir / "shap_data"
        shap_data_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, results in shap_results.items():
            # Save SHAP values
            shap_path = shap_data_dir / f"{model_name}_shap_values.npy"
            np.save(shap_path, results['shap_values'])
            
            # Save feature importance
            importance_path = shap_data_dir / f"{model_name}_feature_importance.csv"
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': results['feature_importance']
            }).sort_values('Importance', ascending=False)
            importance_df.to_csv(importance_path, index=False)
        
        # 4. Create summary report
        self._create_summary_report(shap_results, output_dir)
        
        print(f"\n✅ Comprehensive SHAP report generated!")
        print(f"📁 All outputs saved to: {output_dir}")
    
    def _create_summary_report(self, 
                              shap_results: Dict[str, Dict[str, Any]], 
                              output_dir: Path):
        """Create a text summary report of SHAP analysis."""
        report_path = output_dir / "shap_analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("SHAP ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models analyzed: {len(shap_results)}\n")
            f.write(f"Features: {len(self.feature_names)}\n\n")
            
            for model_name, results in shap_results.items():
                f.write(f"{model_name.upper()} MODEL\n")
                f.write("-" * 20 + "\n")
                
                feature_importance = results['feature_importance']
                top_indices = np.argsort(feature_importance)[-10:][::-1]
                
                f.write("Top 10 Most Important Features:\n")
                for i, idx in enumerate(top_indices, 1):
                    f.write(f"  {i:2d}. {self.feature_names[idx]}: {feature_importance[idx]:.4f}\n")
                
                f.write(f"\nBase value: {results['base_value']:.4f}\n")
                f.write(f"Mean |SHAP|: {np.mean(feature_importance):.4f}\n")
                f.write(f"Std |SHAP|: {np.std(feature_importance):.4f}\n\n")
        
        print(f"   ✅ Summary report: {report_path.name}")


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SHAP Analysis for GRACE Downscaling")
    parser.add_argument('--models-dir', required=True,
                       help='Directory containing trained models')
    parser.add_argument('--features-coarse', required=True,
                       help='Coarse features dataset path')
    parser.add_argument('--grace-filled', required=True,
                       help='Gap-filled GRACE dataset path')
    parser.add_argument('--config', default='src_new_approach/config_coarse_to_fine.yaml',
                       help='Configuration file path')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum samples for SHAP computation')
    parser.add_argument('--interactions', action='store_true',
                       help='Compute feature interactions (expensive)')
    parser.add_argument('--output', default='figures_coarse_to_fine/shap_analysis',
                       help='Output directory for plots and data')
    
    args = parser.parse_args()
    
    # Load configuration
    from utils_downscaling import load_config
    config = load_config(args.config)
    
    # Load models
    models_dir = Path(args.models_dir)
    models = {}
    
    for model_file in models_dir.glob("*_coarse_model.joblib"):
        model_name = model_file.stem.replace('_coarse_model', '')
        model = joblib.load(model_file)
        models[model_name] = model
    
    if not models:
        raise FileNotFoundError(f"No models found in {models_dir}")
    
    print(f"📂 Loaded {len(models)} models: {list(models.keys())}")
    
    # Load data
    print("📂 Loading datasets...")
    import xarray as xr
    
    features_ds = xr.open_dataset(args.features_coarse)
    grace_ds = xr.open_dataset(args.grace_filled)
    
    # Prepare training data
    from coarse_model_trainer import CoarseModelTrainer
    trainer = CoarseModelTrainer(config)
    X, y, feature_names, metadata = trainer.prepare_training_data(features_ds, grace_ds)
    
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    print(f"   Features: {len(feature_names)}")
    
    # Initialize SHAP analyzer
    analyzer = SHAPAnalyzer(
        models=models,
        feature_names=feature_names,
        config=config,
        max_samples=args.max_samples
    )
    
    # Generate comprehensive report
    analyzer.generate_comprehensive_report(
        X_data=X,
        y_data=y,
        compute_interactions=args.interactions,
        output_dir=args.output
    )
    
    print("\n🎉 SHAP analysis completed successfully!")


if __name__ == "__main__":
    main()