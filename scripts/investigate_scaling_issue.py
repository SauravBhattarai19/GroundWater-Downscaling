#!/usr/bin/env python3
"""
Scaling Issue Investigation for SHAP
===================================

This script investigates the scaling mismatch causing extreme SHAP values
for the NN model. The hypothesis is that:

1. NN was trained with scaled target values
2. A target scaler was saved (nn_scaler.joblib)  
3. During normal prediction, outputs are inverse-scaled
4. During SHAP computation, this inverse scaling is NOT applied
5. Result: SHAP shows raw scaled predictions (3000-15000) instead of real TWS (-50 to +50)

This script will:
1. Load and examine the NN scaler
2. Check current SHAP predictions vs properly scaled predictions
3. Create corrected SHAP computation
"""

import joblib
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

warnings.filterwarnings('ignore')


class ScalingIssueInvestigator:
    """Investigate and fix scaling issues in SHAP computation."""
    
    def __init__(self, 
                 models_dir: str = "models_coarse_to_fine_simple",
                 shap_pickle: str = "figures_shap/shap_data/shap_analysis_complete.pkl",
                 output_dir: str = "figures_shap/scaling_investigation"):
        """
        Initialize scaling investigator.
        
        Parameters:
        -----------
        models_dir : str
            Directory containing models and scalers
        shap_pickle : str
            Path to SHAP data pickle
        output_dir : str
            Output directory for analysis
        """
        self.models_dir = Path(models_dir)
        self.shap_pickle = shap_pickle
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.models = {}
        self.scalers = {}
        self.shap_data = {}
        
        print(f"🔍 Scaling Issue Investigator")
        print(f"   Models dir: {models_dir}")
        print(f"   SHAP data: {shap_pickle}")
        print(f"   Output: {output_dir}")
    
    def load_models_and_scalers(self) -> bool:
        """Load all models and associated scalers."""
        try:
            # Load models
            model_files = {
                'xgb': self.models_dir / 'xgb_coarse_model.joblib',
                'lgb': self.models_dir / 'lgb_coarse_model.joblib', 
                'nn': self.models_dir / 'nn_coarse_model.joblib'
            }
            
            for model_name, model_file in model_files.items():
                if model_file.exists():
                    self.models[model_name] = joblib.load(model_file)
                    print(f"✅ Loaded {model_name} model")
                else:
                    print(f"⚠️ Model not found: {model_file}")
            
            # Look for scalers
            scaler_files = list(self.models_dir.glob("*scaler*.joblib"))
            
            for scaler_file in scaler_files:
                scaler_name = scaler_file.stem
                self.scalers[scaler_name] = joblib.load(scaler_file)
                print(f"✅ Loaded scaler: {scaler_name}")
            
            # Load SHAP data
            with open(self.shap_pickle, 'rb') as f:
                self.shap_data = pickle.load(f)
            
            print(f"✅ Loaded SHAP data with {len(self.shap_data['shap_values'])} models")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def examine_scalers(self) -> Dict[str, Any]:
        """Examine what scaling was applied."""
        scaler_info = {}
        
        print(f"\n📏 Scaler Analysis:")
        
        for scaler_name, scaler in self.scalers.items():
            print(f"\n   {scaler_name}:")
            print(f"     Type: {type(scaler)}")
            
            info = {
                'type': str(type(scaler)),
                'scaler_name': scaler_name
            }
            
            # Check if it's a StandardScaler, MinMaxScaler, etc.
            if hasattr(scaler, 'scale_'):
                info['scale'] = scaler.scale_
                info['mean'] = getattr(scaler, 'mean_', None)
                info['var'] = getattr(scaler, 'var_', None)
                print(f"     Scale: {scaler.scale_}")
                if hasattr(scaler, 'mean_'):
                    print(f"     Mean: {scaler.mean_}")
                if hasattr(scaler, 'var_'):
                    print(f"     Variance: {scaler.var_}")
            
            if hasattr(scaler, 'data_min_'):
                info['data_min'] = scaler.data_min_
                info['data_max'] = scaler.data_max_
                print(f"     Data min: {scaler.data_min_}")
                print(f"     Data max: {scaler.data_max_}")
            
            scaler_info[scaler_name] = info
        
        return scaler_info
    
    def test_scaling_hypothesis(self) -> Dict[str, Any]:
        """Test if scaling explains the extreme SHAP values."""
        if 'nn' not in self.models or 'nn_scaler' not in self.scalers:
            print("❌ NN model or scaler not available")
            return {}
        
        nn_model = self.models['nn']
        nn_scaler = self.scalers['nn_scaler']
        X_data = self.shap_data['X_data']
        
        print(f"\n🧪 Testing Scaling Hypothesis:")
        
        # Get raw NN predictions (what SHAP sees)
        raw_predictions = nn_model.predict(X_data)
        
        # Get properly scaled predictions (what should be used)
        if hasattr(nn_scaler, 'inverse_transform'):
            # If it's a target scaler
            scaled_predictions = nn_scaler.inverse_transform(raw_predictions.reshape(-1, 1)).flatten()
        else:
            # If it's feature scaler, predictions might already be scaled
            scaled_predictions = raw_predictions
        
        # Compare ranges
        raw_range = [np.min(raw_predictions), np.max(raw_predictions)]
        scaled_range = [np.min(scaled_predictions), np.max(scaled_predictions)]
        
        print(f"   Raw NN predictions range: [{raw_range[0]:.1f}, {raw_range[1]:.1f}]")
        print(f"   Scaled predictions range: [{scaled_range[0]:.3f}, {scaled_range[1]:.3f}]")
        
        # Check if raw predictions match SHAP f(x) values
        nn_shap = self.shap_data['shap_values']['nn']
        
        # Calculate what SHAP thinks the base prediction is
        if hasattr(nn_model, 'predict'):
            # Sample a few predictions and compare with SHAP base values
            sample_indices = np.random.choice(len(X_data), 5, replace=False)
            
            print(f"\n   Sample prediction comparison:")
            for i, idx in enumerate(sample_indices):
                raw_pred = raw_predictions[idx]
                scaled_pred = scaled_predictions[idx]
                
                # Calculate SHAP prediction (base + sum of SHAP values)
                shap_pred = np.sum(nn_shap[idx])  # This should equal raw_pred if no base value issues
                
                print(f"     Sample {i+1}:")
                print(f"       Raw NN prediction: {raw_pred:.1f}")
                print(f"       Scaled prediction: {scaled_pred:.3f}")
                print(f"       SHAP sum: {shap_pred:.1f}")
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Sample for plotting
        n_plot = min(2000, len(raw_predictions))
        plot_indices = np.random.choice(len(raw_predictions), n_plot, replace=False)
        
        plt.subplot(1, 3, 1)
        plt.hist(raw_predictions[plot_indices], bins=50, alpha=0.7, label='Raw NN Output')
        plt.xlabel('Raw Prediction Value')
        plt.ylabel('Frequency')
        plt.title('Raw NN Predictions\n(What SHAP sees)')
        plt.yscale('log')
        
        plt.subplot(1, 3, 2)
        plt.hist(scaled_predictions[plot_indices], bins=50, alpha=0.7, label='Scaled Prediction', color='orange')
        plt.xlabel('Scaled Prediction Value')
        plt.ylabel('Frequency')
        plt.title('Properly Scaled Predictions\n(What should be used)')
        plt.yscale('log')
        
        plt.subplot(1, 3, 3)
        plt.scatter(raw_predictions[plot_indices], scaled_predictions[plot_indices], alpha=0.5, s=1)
        plt.xlabel('Raw Prediction')
        plt.ylabel('Scaled Prediction')
        plt.title('Raw vs Scaled\nPrediction Relationship')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "scaling_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Scaling comparison plot: {plot_path}")
        
        scaling_test = {
            'raw_range': raw_range,
            'scaled_range': scaled_range,
            'scaling_factor_estimated': raw_range[1] / scaled_range[1] if scaled_range[1] != 0 else None,
            'raw_predictions': raw_predictions,
            'scaled_predictions': scaled_predictions
        }
        
        return scaling_test
    
    def compare_with_other_models(self) -> Dict[str, Any]:
        """Compare NN predictions with XGB/LGB predictions."""
        X_data = self.shap_data['X_data']
        
        print(f"\n🔍 Model Prediction Comparison:")
        
        predictions = {}
        
        # Get predictions from all models
        for model_name, model in self.models.items():
            preds = model.predict(X_data)
            predictions[model_name] = preds
            print(f"   {model_name.upper()} range: [{np.min(preds):.3f}, {np.max(preds):.3f}]")
        
        # If we have NN scaler, try to scale NN predictions
        if 'nn_scaler' in self.scalers and 'nn' in predictions:
            nn_scaler = self.scalers['nn_scaler']
            raw_nn_preds = predictions['nn']
            
            try:
                if hasattr(nn_scaler, 'inverse_transform'):
                    scaled_nn_preds = nn_scaler.inverse_transform(raw_nn_preds.reshape(-1, 1)).flatten()
                    predictions['nn_scaled'] = scaled_nn_preds
                    print(f"   NN (scaled) range: [{np.min(scaled_nn_preds):.3f}, {np.max(scaled_nn_preds):.3f}]")
            except Exception as e:
                print(f"   ⚠️ Could not scale NN predictions: {e}")
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Prediction distributions
        plt.subplot(2, 2, 1)
        for model_name, preds in predictions.items():
            if len(preds) > 0:
                plt.hist(preds, bins=50, alpha=0.6, label=model_name, density=True)
        plt.xlabel('Prediction Value')
        plt.ylabel('Density')
        plt.title('Prediction Distributions')
        plt.legend()
        plt.yscale('log')
        
        # Box plot comparison
        plt.subplot(2, 2, 2)
        pred_data = [preds for preds in predictions.values() if len(preds) > 0]
        pred_labels = [name for name, preds in predictions.items() if len(preds) > 0]
        plt.boxplot(pred_data, labels=pred_labels)
        plt.ylabel('Prediction Value')
        plt.title('Prediction Box Plots')
        plt.xticks(rotation=45)
        
        # Correlation matrix if we have multiple models
        if len(predictions) > 1:
            plt.subplot(2, 2, 3)
            pred_df = pd.DataFrame(predictions)
            corr_matrix = pred_df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, square=True)
            plt.title('Prediction Correlations')
        
        # SHAP value ranges comparison
        plt.subplot(2, 2, 4)
        shap_ranges = {}
        for model_name, shap_vals in self.shap_data['shap_values'].items():
            shap_ranges[model_name] = [np.min(shap_vals), np.max(shap_vals)]
        
        models = list(shap_ranges.keys())
        min_vals = [shap_ranges[model][0] for model in models]
        max_vals = [shap_ranges[model][1] for model in models]
        
        x = np.arange(len(models))
        plt.bar(x, max_vals, alpha=0.7, label='Max SHAP')
        plt.bar(x, min_vals, alpha=0.7, label='Min SHAP')
        plt.xticks(x, models)
        plt.ylabel('SHAP Value Range')
        plt.title('SHAP Value Ranges')
        plt.legend()
        plt.yscale('symlog')  # Handle negative values
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Model comparison plot: {plot_path}")
        
        return predictions
    
    def generate_investigation_report(self, 
                                    scaler_info: Dict,
                                    scaling_test: Dict,
                                    predictions: Dict) -> None:
        """Generate comprehensive investigation report."""
        report_path = self.output_dir / "scaling_investigation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SCALING ISSUE INVESTIGATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("HYPOTHESIS:\n")
            f.write("-" * 40 + "\n")
            f.write("The extreme SHAP values for NN are caused by a scaling mismatch:\n")
            f.write("1. NN was trained with scaled target values\n")
            f.write("2. During normal prediction, targets are inverse-scaled\n") 
            f.write("3. During SHAP computation, this inverse scaling is NOT applied\n")
            f.write("4. Result: SHAP shows scaled predictions (3000-15000) not real TWS (-50 to +50)\n\n")
            
            f.write("SCALER ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            for scaler_name, info in scaler_info.items():
                f.write(f"{scaler_name}:\n")
                f.write(f"  Type: {info['type']}\n")
                if 'scale' in info and info['scale'] is not None:
                    f.write(f"  Scale factor: {info['scale']}\n")
                if 'mean' in info and info['mean'] is not None:
                    f.write(f"  Mean: {info['mean']}\n")
                if 'data_min' in info:
                    f.write(f"  Data range: [{info['data_min']}, {info['data_max']}]\n")
                f.write("\n")
            
            f.write("PREDICTION ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            if scaling_test:
                f.write(f"Raw NN predictions range: {scaling_test['raw_range']}\n")
                f.write(f"Scaled NN predictions range: {scaling_test['scaled_range']}\n")
                if scaling_test['scaling_factor_estimated']:
                    f.write(f"Estimated scaling factor: {scaling_test['scaling_factor_estimated']:.1f}\n")
                f.write("\n")
            
            f.write("MODEL COMPARISON:\n")
            f.write("-" * 40 + "\n")
            for model_name, preds in predictions.items():
                if len(preds) > 0:
                    f.write(f"{model_name}: [{np.min(preds):.3f}, {np.max(preds):.3f}]\n")
            f.write("\n")
            
            f.write("CONCLUSION:\n")
            f.write("-" * 40 + "\n")
            if scaling_test and scaling_test['raw_range'][1] > 1000:
                f.write("✅ HYPOTHESIS CONFIRMED: NN predictions are in scaled space\n")
                f.write("The extreme SHAP values are caused by missing inverse scaling\n")
                f.write("during SHAP computation.\n\n")
                
                f.write("RECOMMENDED FIX:\n")
                f.write("1. Modify SHAP computation to apply nn_scaler.inverse_transform()\n")
                f.write("2. Wrap NN model to return properly scaled predictions\n")
                f.write("3. Recompute SHAP values with corrected predictions\n")
            else:
                f.write("❓ HYPOTHESIS UNCLEAR: Further investigation needed\n")
        
        print(f"📋 Investigation report: {report_path}")
    
    def run_investigation(self) -> bool:
        """Run complete scaling investigation."""
        print(f"\n{'='*60}")
        print("SCALING ISSUE INVESTIGATION")
        print(f"{'='*60}")
        
        if not self.load_models_and_scalers():
            return False
        
        # 1. Examine scalers
        scaler_info = self.examine_scalers()
        
        # 2. Test scaling hypothesis
        scaling_test = self.test_scaling_hypothesis()
        
        # 3. Compare with other models
        predictions = self.compare_with_other_models()
        
        # 4. Generate report
        self.generate_investigation_report(scaler_info, scaling_test, predictions)
        
        print(f"\n✅ Investigation complete! Results in: {self.output_dir}")
        return True


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Investigate scaling issues in SHAP")
    parser.add_argument('--models-dir', default='models_coarse_to_fine_simple')
    parser.add_argument('--shap-data', default='figures_shap/shap_data/shap_analysis_complete.pkl')
    parser.add_argument('--output-dir', default='figures_shap/scaling_investigation')
    
    args = parser.parse_args()
    
    investigator = ScalingIssueInvestigator(
        models_dir=args.models_dir,
        shap_pickle=args.shap_data,
        output_dir=args.output_dir
    )
    
    success = investigator.run_investigation()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())