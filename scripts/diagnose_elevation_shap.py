#!/usr/bin/env python3
"""
Elevation SHAP Value Diagnostic Tool
===================================

This script investigates the extreme SHAP value (-100000) for topographic elevation
in the NN model while other models show normal ranges.

Key Investigations:
1. Feature value distributions and scaling
2. SHAP value comparisons across models  
3. Model-specific behavior with elevation
4. Potential normalization/scaling issues
"""

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')


class ElevationSHAPDiagnostic:
    """Diagnostic tool for elevation SHAP analysis."""
    
    def __init__(self, 
                 shap_pickle_path: str = "figures_shap/shap_data/shap_analysis_complete.pkl",
                 models_dir: str = "models_coarse_to_fine_simple",
                 output_dir: str = "figures_shap/diagnostics"):
        """
        Initialize diagnostic tool.
        
        Parameters:
        -----------
        shap_pickle_path : str
            Path to SHAP pickle file
        models_dir : str  
            Directory containing models
        output_dir : str
            Output directory for diagnostic plots
        """
        self.shap_pickle_path = shap_pickle_path
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.shap_data = {}
        self.models = {}
        self.elevation_idx = None
        
        print(f"🔍 Elevation SHAP Diagnostic Tool")
        print(f"   SHAP data: {shap_pickle_path}")
        print(f"   Models dir: {models_dir}")
        print(f"   Output: {output_dir}")
    
    def load_data(self) -> bool:
        """Load SHAP data and models."""
        try:
            # Load SHAP data
            with open(self.shap_pickle_path, 'rb') as f:
                self.shap_data = pickle.load(f)
            
            print(f"✅ Loaded SHAP data:")
            print(f"   Models: {list(self.shap_data['shap_values'].keys())}")
            print(f"   Features: {len(self.shap_data['feature_names'])}")
            print(f"   Samples: {self.shap_data['X_data'].shape[0]}")
            
            # Find elevation feature index
            feature_names = self.shap_data['feature_names']
            elevation_features = [i for i, name in enumerate(feature_names) 
                                if 'elevation' in name.lower() or 'topo' in name.lower()]
            
            if elevation_features:
                self.elevation_idx = elevation_features[0]
                elevation_name = feature_names[self.elevation_idx]
                print(f"✅ Found elevation feature: '{elevation_name}' at index {self.elevation_idx}")
            else:
                print(f"❌ No elevation feature found in: {feature_names[:10]}...")
                return False
            
            # Load models
            for model_name in self.shap_data['shap_values'].keys():
                model_file = self.models_dir / f"{model_name}_coarse_model.joblib"
                if model_file.exists():
                    self.models[model_name] = joblib.load(model_file)
                    print(f"✅ Loaded {model_name} model")
                else:
                    print(f"⚠️ Model file not found: {model_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def analyze_elevation_feature(self) -> Dict[str, Any]:
        """Analyze elevation feature values and distribution."""
        if self.elevation_idx is None:
            print("❌ Elevation feature not found")
            return {}
        
        X_data = self.shap_data['X_data']
        elevation_values = X_data[:, self.elevation_idx]
        feature_name = self.shap_data['feature_names'][self.elevation_idx]
        
        stats = {
            'feature_name': feature_name,
            'min': np.min(elevation_values),
            'max': np.max(elevation_values),
            'mean': np.mean(elevation_values),
            'std': np.std(elevation_values),
            'median': np.median(elevation_values),
            'q25': np.percentile(elevation_values, 25),
            'q75': np.percentile(elevation_values, 75),
            'unique_values': len(np.unique(elevation_values)),
            'has_negatives': np.any(elevation_values < 0),
            'has_zeros': np.any(elevation_values == 0),
            'values': elevation_values
        }
        
        print(f"\n📊 Elevation Feature Analysis: {feature_name}")
        print(f"   Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"   Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"   Median: {stats['median']:.3f}")
        print(f"   Q25-Q75: [{stats['q25']:.3f}, {stats['q75']:.3f}]")
        print(f"   Unique values: {stats['unique_values']}")
        print(f"   Has negatives: {stats['has_negatives']}")
        print(f"   Has zeros: {stats['has_zeros']}")
        
        # Create elevation distribution plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(elevation_values, bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{feature_name}\nDistribution')
        plt.xlabel('Elevation Value')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        plt.boxplot(elevation_values)
        plt.title('Box Plot')
        plt.ylabel('Elevation Value')
        
        plt.subplot(1, 3, 3)
        from scipy import stats as scipy_stats
        scipy_stats.probplot(elevation_values, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normality)')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "elevation_feature_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Elevation distribution plot: {plot_path}")
        
        return stats
    
    def analyze_shap_values(self) -> Dict[str, Any]:
        """Analyze SHAP values for elevation across all models."""
        if self.elevation_idx is None:
            print("❌ Elevation feature not found")
            return {}
        
        shap_values = self.shap_data['shap_values']
        feature_name = self.shap_data['feature_names'][self.elevation_idx]
        
        elevation_shap_analysis = {}
        
        print(f"\n🧠 SHAP Analysis for {feature_name}:")
        
        for model_name, model_shap in shap_values.items():
            elevation_shap = model_shap[:, self.elevation_idx]
            
            analysis = {
                'min': np.min(elevation_shap),
                'max': np.max(elevation_shap),
                'mean': np.mean(elevation_shap),
                'std': np.std(elevation_shap),
                'median': np.median(elevation_shap),
                'abs_mean': np.mean(np.abs(elevation_shap)),
                'extreme_count': np.sum(np.abs(elevation_shap) > 1000),
                'very_extreme_count': np.sum(np.abs(elevation_shap) > 10000),
                'values': elevation_shap
            }
            
            elevation_shap_analysis[model_name] = analysis
            
            print(f"   {model_name.upper()}:")
            print(f"     Range: [{analysis['min']:.1f}, {analysis['max']:.1f}]")
            print(f"     Mean: {analysis['mean']:.3f} ± {analysis['std']:.3f}")
            print(f"     Mean |SHAP|: {analysis['abs_mean']:.3f}")
            print(f"     Extreme values (>1000): {analysis['extreme_count']}")
            print(f"     Very extreme (>10000): {analysis['very_extreme_count']}")
        
        # Create SHAP comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. SHAP value distributions
        axes[0, 0].set_title('SHAP Value Distributions')
        for model_name, analysis in elevation_shap_analysis.items():
            axes[0, 0].hist(analysis['values'], alpha=0.6, label=model_name, bins=50)
        axes[0, 0].set_xlabel('SHAP Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')  # Log scale to see extreme values
        
        # 2. Box plots
        axes[0, 1].set_title('SHAP Value Box Plots')
        shap_data_for_box = [analysis['values'] for analysis in elevation_shap_analysis.values()]
        axes[0, 1].boxplot(shap_data_for_box, labels=list(elevation_shap_analysis.keys()))
        axes[0, 1].set_ylabel('SHAP Value')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Mean absolute SHAP comparison
        axes[1, 0].set_title('Mean |SHAP| Comparison')
        models = list(elevation_shap_analysis.keys())
        mean_abs_shap = [analysis['abs_mean'] for analysis in elevation_shap_analysis.values()]
        bars = axes[1, 0].bar(models, mean_abs_shap)
        axes[1, 0].set_ylabel('Mean |SHAP Value|')
        
        # Add value labels on bars
        for bar, value in zip(bars, mean_abs_shap):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                          f'{value:.1f}', ha='center', va='bottom')
        
        # 4. Extreme value counts
        axes[1, 1].set_title('Extreme SHAP Value Counts')
        extreme_counts = [analysis['extreme_count'] for analysis in elevation_shap_analysis.values()]
        very_extreme_counts = [analysis['very_extreme_count'] for analysis in elevation_shap_analysis.values()]
        
        x = np.arange(len(models))
        width = 0.35
        axes[1, 1].bar(x - width/2, extreme_counts, width, label='> 1000', alpha=0.8)
        axes[1, 1].bar(x + width/2, very_extreme_counts, width, label='> 10000', alpha=0.8)
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models)
        axes[1, 1].legend()
        
        plt.suptitle(f'Elevation SHAP Analysis: {feature_name}', fontsize=16)
        plt.tight_layout()
        
        plot_path = self.output_dir / "elevation_shap_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 SHAP analysis plot: {plot_path}")
        
        return elevation_shap_analysis
    
    def investigate_nn_model(self) -> Dict[str, Any]:
        """Investigate NN model's relationship with elevation."""
        if 'nn' not in self.models:
            print("❌ NN model not available for investigation")
            return {}
        
        nn_model = self.models['nn']
        X_data = self.shap_data['X_data']
        elevation_values = X_data[:, self.elevation_idx]
        
        print(f"\n🤖 NN Model Investigation:")
        print(f"   Model type: {type(nn_model)}")
        
        # Get NN predictions
        try:
            predictions = nn_model.predict(X_data)
            
            # Analyze relationship between elevation and predictions
            correlation = np.corrcoef(elevation_values, predictions)[0, 1]
            
            print(f"   Elevation-Prediction correlation: {correlation:.4f}")
            
            # Create scatter plot
            plt.figure(figsize=(12, 4))
            
            # Sample for visualization (too many points)
            n_plot = min(2000, len(elevation_values))
            idx = np.random.choice(len(elevation_values), n_plot, replace=False)
            
            plt.subplot(1, 3, 1)
            plt.scatter(elevation_values[idx], predictions[idx], alpha=0.5, s=1)
            plt.xlabel('Elevation Value')
            plt.ylabel('NN Prediction')
            plt.title(f'Elevation vs Prediction\n(r={correlation:.3f})')
            
            # SHAP values vs elevation
            nn_shap = self.shap_data['shap_values']['nn'][:, self.elevation_idx]
            
            plt.subplot(1, 3, 2)
            plt.scatter(elevation_values[idx], nn_shap[idx], alpha=0.5, s=1, color='red')
            plt.xlabel('Elevation Value')
            plt.ylabel('Elevation SHAP Value')
            plt.title('Elevation vs SHAP Value')
            
            # SHAP values vs predictions
            plt.subplot(1, 3, 3)
            plt.scatter(predictions[idx], nn_shap[idx], alpha=0.5, s=1, color='green')
            plt.xlabel('NN Prediction')
            plt.ylabel('Elevation SHAP Value')
            plt.title('Prediction vs Elevation SHAP')
            
            plt.tight_layout()
            
            plot_path = self.output_dir / "nn_elevation_relationships.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 NN relationship plot: {plot_path}")
            
            # Check for model weights (if accessible)
            nn_analysis = {
                'correlation': correlation,
                'predictions_range': [np.min(predictions), np.max(predictions)],
                'predictions_mean': np.mean(predictions),
                'predictions_std': np.std(predictions)
            }
            
            # Try to access NN weights if possible
            if hasattr(nn_model, 'coefs_'):  # sklearn MLPRegressor
                print(f"   NN has {len(nn_model.coefs_)} layers")
                first_layer_weights = nn_model.coefs_[0]
                elevation_weights = first_layer_weights[self.elevation_idx, :]
                
                nn_analysis['elevation_weights'] = elevation_weights
                nn_analysis['elevation_weight_stats'] = {
                    'min': np.min(elevation_weights),
                    'max': np.max(elevation_weights),
                    'mean': np.mean(elevation_weights),
                    'std': np.std(elevation_weights)
                }
                
                print(f"   Elevation weights in first layer:")
                print(f"     Range: [{np.min(elevation_weights):.3f}, {np.max(elevation_weights):.3f}]")
                print(f"     Mean: {np.mean(elevation_weights):.3f} ± {np.std(elevation_weights):.3f}")
            
            return nn_analysis
            
        except Exception as e:
            print(f"   ❌ Error investigating NN: {e}")
            return {}
    
    def compare_feature_scaling(self) -> Dict[str, Any]:
        """Compare how elevation is scaled compared to other features."""
        X_data = self.shap_data['X_data']
        feature_names = self.shap_data['feature_names']
        
        print(f"\n⚖️ Feature Scaling Analysis:")
        
        # Calculate statistics for all features
        feature_stats = {}
        for i, name in enumerate(feature_names):
            values = X_data[:, i]
            feature_stats[name] = {
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'range': np.max(values) - np.min(values),
                'abs_mean': np.mean(np.abs(values))
            }
        
        # Focus on elevation
        elevation_name = feature_names[self.elevation_idx]
        elev_stats = feature_stats[elevation_name]
        
        # Compare with other features
        other_ranges = [stats['range'] for name, stats in feature_stats.items() 
                       if name != elevation_name]
        other_stds = [stats['std'] for name, stats in feature_stats.items() 
                     if name != elevation_name]
        
        print(f"   Elevation range: {elev_stats['range']:.3f}")
        print(f"   Other features range - mean: {np.mean(other_ranges):.3f}")
        print(f"   Other features range - median: {np.median(other_ranges):.3f}")
        print(f"   Elevation std: {elev_stats['std']:.3f}")
        print(f"   Other features std - mean: {np.mean(other_stds):.3f}")
        
        # Check if elevation is an outlier in scaling
        elev_range_zscore = (elev_stats['range'] - np.mean(other_ranges)) / np.std(other_ranges)
        elev_std_zscore = (elev_stats['std'] - np.mean(other_stds)) / np.std(other_stds)
        
        print(f"   Elevation range z-score: {elev_range_zscore:.3f}")
        print(f"   Elevation std z-score: {elev_std_zscore:.3f}")
        
        scaling_analysis = {
            'elevation_stats': elev_stats,
            'other_ranges_mean': np.mean(other_ranges),
            'other_stds_mean': np.mean(other_stds),
            'elevation_range_zscore': elev_range_zscore,
            'elevation_std_zscore': elev_std_zscore,
            'is_scaling_outlier': abs(elev_range_zscore) > 2 or abs(elev_std_zscore) > 2
        }
        
        return scaling_analysis
    
    def generate_diagnostic_report(self, 
                                 elevation_stats: Dict,
                                 shap_analysis: Dict,
                                 nn_analysis: Dict,
                                 scaling_analysis: Dict) -> None:
        """Generate comprehensive diagnostic report."""
        report_path = self.output_dir / "elevation_shap_diagnostic_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ELEVATION SHAP DIAGNOSTIC REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("ISSUE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write("NN model shows extreme SHAP value (~-100000) for topographic elevation\n")
            f.write("while XGB and LGB models show normal SHAP ranges.\n")
            f.write("Model predictions are reportedly good overall.\n\n")
            
            # Feature analysis
            f.write("ELEVATION FEATURE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Feature name: {elevation_stats['feature_name']}\n")
            f.write(f"Value range: [{elevation_stats['min']:.3f}, {elevation_stats['max']:.3f}]\n")
            f.write(f"Mean ± Std: {elevation_stats['mean']:.3f} ± {elevation_stats['std']:.3f}\n")
            f.write(f"Unique values: {elevation_stats['unique_values']}\n")
            f.write(f"Has negatives: {elevation_stats['has_negatives']}\n")
            f.write(f"Scaling outlier: {scaling_analysis.get('is_scaling_outlier', 'Unknown')}\n\n")
            
            # SHAP analysis
            f.write("SHAP VALUE COMPARISON:\n")
            f.write("-" * 40 + "\n")
            for model_name, analysis in shap_analysis.items():
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Range: [{analysis['min']:.1f}, {analysis['max']:.1f}]\n")
                f.write(f"  Mean |SHAP|: {analysis['abs_mean']:.3f}\n")
                f.write(f"  Extreme values (>10k): {analysis['very_extreme_count']}\n")
            f.write("\n")
            
            # NN analysis
            if nn_analysis:
                f.write("NN MODEL ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Elevation-Prediction correlation: {nn_analysis.get('correlation', 'N/A'):.4f}\n")
                if 'elevation_weight_stats' in nn_analysis:
                    weight_stats = nn_analysis['elevation_weight_stats']
                    f.write(f"Elevation weights range: [{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]\n")
                    f.write(f"Elevation weights mean: {weight_stats['mean']:.3f}\n")
                f.write("\n")
            
            # Diagnosis
            f.write("PROBABLE CAUSES:\n")
            f.write("-" * 40 + "\n")
            
            if scaling_analysis.get('is_scaling_outlier', False):
                f.write("1. SCALING ISSUE: Elevation feature has unusual scaling compared to others\n")
            
            if shap_analysis.get('nn', {}).get('very_extreme_count', 0) > 0:
                f.write("2. NN LEARNING ISSUE: Model learned extreme sensitivity to elevation\n")
            
            f.write("3. POSSIBLE SHAP COMPUTATION ARTIFACT: Numerical instability in SHAP calculation\n")
            f.write("4. FEATURE PREPROCESSING MISMATCH: Different scaling between models\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Check feature preprocessing pipeline for NN vs tree models\n")
            f.write("2. Verify elevation feature normalization/scaling\n")
            f.write("3. Try different SHAP explainer (KernelExplainer vs TreeExplainer)\n")
            f.write("4. Retrain NN with proper feature scaling\n")
            f.write("5. Check if extreme SHAP reflects real model behavior\n")
            f.write("6. Consider feature importance from model weights directly\n")
        
        print(f"📋 Diagnostic report: {report_path}")
    
    def run_full_diagnostic(self) -> bool:
        """Run complete diagnostic pipeline."""
        print(f"\n{'='*60}")
        print("ELEVATION SHAP DIAGNOSTIC ANALYSIS")
        print(f"{'='*60}")
        
        if not self.load_data():
            return False
        
        # 1. Analyze elevation feature
        elevation_stats = self.analyze_elevation_feature()
        
        # 2. Analyze SHAP values
        shap_analysis = self.analyze_shap_values()
        
        # 3. Investigate NN model
        nn_analysis = self.investigate_nn_model()
        
        # 4. Compare feature scaling
        scaling_analysis = self.compare_feature_scaling()
        
        # 5. Generate report
        self.generate_diagnostic_report(
            elevation_stats, shap_analysis, nn_analysis, scaling_analysis
        )
        
        print(f"\n✅ Diagnostic complete! Results in: {self.output_dir}")
        return True


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose elevation SHAP issue")
    parser.add_argument('--shap-data', default='figures_shap/shap_data/shap_analysis_complete.pkl')
    parser.add_argument('--models-dir', default='models_coarse_to_fine_simple')
    parser.add_argument('--output-dir', default='figures_shap/diagnostics')
    
    args = parser.parse_args()
    
    diagnostic = ElevationSHAPDiagnostic(
        shap_pickle_path=args.shap_data,
        models_dir=args.models_dir,
        output_dir=args.output_dir
    )
    
    success = diagnostic.run_full_diagnostic()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())