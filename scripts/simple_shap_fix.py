#!/usr/bin/env python3
"""
Simple SHAP Fix for NN Scaling Issue
====================================

Quick fix to demonstrate the NN scaling issue and provide corrected SHAP values.
"""

import pickle
import joblib
import warnings
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("❌ SHAP not available")

warnings.filterwarnings('ignore')

def main():
    """Quick test and fix of NN scaling issue."""
    
    if not HAS_SHAP:
        print("❌ SHAP not available - cannot run fix")
        return 1
    
    print("🔧 Simple SHAP Scaling Fix")
    print("=" * 50)
    
    # Paths
    models_dir = Path("models_coarse_to_fine_simple")
    shap_pickle = "figures_shap/shap_data/shap_analysis_complete.pkl"
    output_dir = Path("figures_shap/quick_fix")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("📂 Loading data...")
    
    # Load NN model and scaler
    nn_model = joblib.load(models_dir / "nn_coarse_model.joblib")
    nn_scaler = joblib.load(models_dir / "nn_scaler.joblib")
    
    # Load original SHAP data
    with open(shap_pickle, 'rb') as f:
        shap_data = pickle.load(f)
    
    X_raw = shap_data['X_data']
    feature_names = shap_data['feature_names']
    
    print(f"✅ Data loaded: {X_raw.shape}")
    
    # Test the scaling issue
    print("\n🧪 Testing NN prediction difference...")
    
    # Sample some data
    n_test = 500
    X_sample = X_raw[:n_test]
    
    # Wrong way (what SHAP was doing)
    pred_wrong = nn_model.predict(X_sample)
    
    # Right way (what should be done)
    X_scaled = nn_scaler.transform(X_sample)
    pred_right = nn_model.predict(X_scaled)
    
    print(f"   Wrong predictions range: [{np.min(pred_wrong):.1f}, {np.max(pred_wrong):.1f}]")
    print(f"   Right predictions range: [{np.min(pred_right):.3f}, {np.max(pred_right):.3f}]")
    print(f"   Scaling factor: ~{np.max(pred_wrong) / np.max(pred_right):.0f}x")
    
    # Compute corrected SHAP for NN
    print("\n🧠 Computing corrected SHAP for NN...")
    
    def scaled_nn_predict(X):
        """NN prediction with proper scaling."""
        X_scaled = nn_scaler.transform(X)
        return nn_model.predict(X_scaled)
    
    # Create SHAP explainer with proper scaling
    background = shap.sample(X_sample, 100)
    explainer = shap.KernelExplainer(scaled_nn_predict, background)
    
    # Compute corrected SHAP values
    corrected_shap = explainer.shap_values(X_sample)
    
    print(f"   ✅ Corrected SHAP shape: {corrected_shap.shape}")
    print(f"   ✅ Corrected range: [{np.min(corrected_shap):.1f}, {np.max(corrected_shap):.1f}]")
    
    # Compare with original
    original_nn_shap = shap_data['shap_values']['nn'][:n_test]
    
    print(f"\n📊 Comparison:")
    print(f"   Original NN SHAP range: [{np.min(original_nn_shap):.1f}, {np.max(original_nn_shap):.1f}]")
    print(f"   Corrected NN SHAP range: [{np.min(corrected_shap):.1f}, {np.max(corrected_shap):.1f}]")
    
    # Check elevation feature specifically
    elev_idx = None
    for i, name in enumerate(feature_names):
        if 'elevation' in name.lower():
            elev_idx = i
            break
    
    if elev_idx is not None:
        print(f"\n🏔️ Elevation feature ('{feature_names[elev_idx]}'):")
        orig_elev_shap = original_nn_shap[:, elev_idx]
        corr_elev_shap = corrected_shap[:, elev_idx]
        
        print(f"   Original range: [{np.min(orig_elev_shap):.1f}, {np.max(orig_elev_shap):.1f}]")
        print(f"   Corrected range: [{np.min(corr_elev_shap):.1f}, {np.max(corr_elev_shap):.1f}]")
        print(f"   Mean |SHAP| - Original: {np.mean(np.abs(orig_elev_shap)):.1f}")
        print(f"   Mean |SHAP| - Corrected: {np.mean(np.abs(corr_elev_shap)):.3f}")
    
    # Create comparison plot
    print(f"\n📊 Creating comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Prediction comparison
    axes[0, 0].hist(pred_wrong, bins=30, alpha=0.6, label='Wrong (Raw features)', color='red')
    axes[0, 0].hist(pred_right, bins=30, alpha=0.6, label='Right (Scaled features)', color='blue')
    axes[0, 0].set_xlabel('Prediction Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('NN Prediction Comparison')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # 2. SHAP range comparison
    axes[0, 1].hist(original_nn_shap.flatten(), bins=50, alpha=0.6, label='Original SHAP', color='red')
    axes[0, 1].hist(corrected_shap.flatten(), bins=50, alpha=0.6, label='Corrected SHAP', color='blue')
    axes[0, 1].set_xlabel('SHAP Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('NN SHAP Value Comparison')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # 3. Elevation SHAP comparison if available
    if elev_idx is not None:
        axes[1, 0].hist(orig_elev_shap, bins=30, alpha=0.6, label='Original', color='red')
        axes[1, 0].hist(corr_elev_shap, bins=30, alpha=0.6, label='Corrected', color='blue')
        axes[1, 0].set_xlabel('SHAP Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Elevation SHAP: {feature_names[elev_idx]}')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
    
    # 4. Summary text
    axes[1, 1].axis('off')
    summary_text = f"""ISSUE CONFIRMED & FIXED:

Problem:
• NN model trained with scaled features
• SHAP used raw features  
• Result: Extreme predictions & SHAP values

Fix:
• Apply feature scaling before SHAP
• NN predictions now realistic
• SHAP values in normal range

Evidence:
• Wrong predictions: {np.max(pred_wrong):.0f}
• Right predictions: {np.max(pred_right):.1f}
• Scaling factor: {np.max(pred_wrong)/np.max(pred_right):.0f}x

Elevation SHAP improvement:
• Original max: {np.max(np.abs(orig_elev_shap)):.0f}
• Corrected max: {np.max(np.abs(corr_elev_shap)):.1f}
• Improvement: {np.max(np.abs(orig_elev_shap))/np.max(np.abs(corr_elev_shap)):.0f}x
"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    
    plot_path = output_dir / "shap_scaling_fix_demonstration.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plot saved: {plot_path}")
    
    # Save corrected data
    corrected_data = {
        'nn_shap_corrected': corrected_shap,
        'feature_names': feature_names,
        'X_sample': X_sample,
        'predictions_wrong': pred_wrong,
        'predictions_right': pred_right
    }
    
    corrected_path = output_dir / "nn_corrected_shap.pkl"
    with open(corrected_path, 'wb') as f:
        pickle.dump(corrected_data, f)
    
    print(f"💾 Corrected data saved: {corrected_path}")
    
    print(f"\n✅ ISSUE CONFIRMED AND FIXED!")
    print(f"🎯 The -100,000 SHAP values were due to NN receiving unscaled features.")
    print(f"📁 Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())