#!/usr/bin/env python
"""
Comprehensive Residual Correction Method Testing

This script tests all available residual correction methods and automatically
selects the best performing one for the GRACE downscaling pipeline.

Tests the following methods:
1. Bilinear Interpolation (original)
2. Geographic Assignment (novel)
3. Nearest Neighbor
4. Bicubic Interpolation
5. IDW (Inverse Distance Weighting)
6. Gaussian Kernel Smoothing
7. Area-Weighted Assignment
8. Distance-Weighted Nearest Neighbor

The script will:
1. Load the coarse-to-fine pipeline data
2. Test all residual correction methods
3. Compare performance using validation metrics
4. Select the best method
5. Generate comparison plots
6. Update the pipeline configuration
7. Run the full pipeline with the best method
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src_new_approach.residual_corrector_multi import MultiMethodResidualCorrector
from src_new_approach.utils_downscaling import load_config, get_config_value, get_aggregation_factor
from src_new_approach.fine_predictor import FinePredictor
from scipy.ndimage import zoom
import yaml


def load_pipeline_data(config):
    """Load all required data for the residual correction comparison."""
    print("\n📂 Loading pipeline data...")
    
    # Get paths
    grace_path = config['paths']['grace_filled']
    coarse_features_path = config['paths']['feature_stack_coarse']
    fine_features_path = config['paths']['feature_stack_fine']
    fine_predictions_path = config['paths']['predictions_fine']
    models_dir = config['paths']['models']
    
    # Load datasets
    grace_ds = xr.open_dataset(grace_path)
    coarse_features_ds = xr.open_dataset(coarse_features_path)
    fine_features_ds = xr.open_dataset(fine_features_path)
    fine_predictions_ds = xr.open_dataset(fine_predictions_path)
    
    print(f"   ✓ GRACE shape: {grace_ds.dims}")
    print(f"   ✓ Coarse features shape: {coarse_features_ds.dims}")
    print(f"   ✓ Fine features shape: {fine_features_ds.dims}")
    print(f"   ✓ Fine predictions shape: {fine_predictions_ds.dims}")
    
    return {
        'grace_ds': grace_ds,
        'coarse_features_ds': coarse_features_ds,
        'fine_features_ds': fine_features_ds,
        'fine_predictions_ds': fine_predictions_ds,
        'models_dir': models_dir
    }


def generate_coarse_predictions(config, data, tester):
    """Generate predictions at coarse scale for residual calculation."""
    print("\n📊 Generating coarse-scale predictions...")
    
    # Use fine predictor on coarse data
    predictor = FinePredictor(config, data['models_dir'])
    predictor.load_models()
    
    # Predict
    predictions_coarse_ds = predictor.predict_fine_resolution(
        data['coarse_features_ds'],
        use_ensemble=True
    )
    
    print(f"   ✓ Generated coarse predictions: {predictions_coarse_ds.dims}")
    return predictions_coarse_ds


def calculate_coarse_residuals(grace_ds, predictions_coarse_ds, tester):
    """Calculate residuals at coarse scale."""
    print("\n📐 Calculating coarse-scale residuals...")
    
    residuals_coarse_ds = tester.calculate_residuals_coarse(
        grace_ds,
        predictions_coarse_ds,
        prediction_var='prediction_ensemble'
    )
    
    print(f"   ✓ Calculated residuals: {residuals_coarse_ds.dims}")
    return residuals_coarse_ds


def test_all_interpolation_methods(residuals_coarse_ds, fine_features_ds, tester):
    """Test all available interpolation methods."""
    print("\n🧪 Testing all interpolation methods...")
    
    fine_lat = fine_features_ds.lat.values
    fine_lon = fine_features_ds.lon.values
    
    # Test all methods
    residuals_fine_results = tester.interpolate_residuals_to_fine_multi(
        residuals_coarse_ds,
        fine_lat,
        fine_lon,
        methods=tester.available_methods
    )
    
    print(f"   ✅ Successfully tested {len(residuals_fine_results)} methods")
    return residuals_fine_results


def evaluate_all_methods(residuals_fine_results, data, config, tester):
    """Evaluate performance of all residual correction methods."""
    print("\n📊 Evaluating all methods...")
    
    aggregation_factor = get_aggregation_factor(config)
    
    performance_results = tester.evaluate_method_performance(
        residuals_fine_results,
        data['fine_predictions_ds'],
        data['grace_ds'],
        aggregation_factor,
        prediction_var='prediction_ensemble'
    )
    
    print(f"   ✅ Evaluated {len(performance_results)} methods")
    return performance_results


def create_comparison_plots(performance_results, residuals_fine_results, output_dir):
    """Create comprehensive comparison plots."""
    print("\n📈 Creating comparison plots...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Performance comparison bar chart
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: R² comparison
    plt.subplot(2, 2, 1)
    methods = []
    r2_scores = []
    colors = []
    
    for method, metrics in performance_results.items():
        if not np.isnan(metrics.get('r2', np.nan)):
            methods.append(metrics['method_name'])
            r2_scores.append(metrics['r2'])
            colors.append(residuals_fine_results.get(method, {}).get('color', 'blue'))
    
    bars = plt.bar(range(len(methods)), r2_scores, color=['red' if i == np.argmax(r2_scores) else 'lightblue' for i in range(len(r2_scores))])
    plt.title('R² Score Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Highlight best method
    best_idx = np.argmax(r2_scores)
    bars[best_idx].set_color('red')
    bars[best_idx].set_label('Best Method')
    plt.legend()
    
    # Add values on bars
    for i, v in enumerate(r2_scores):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 2: RMSE comparison
    plt.subplot(2, 2, 2)
    rmse_scores = [performance_results[method]['rmse'] for method in [m for m in performance_results.keys() if not np.isnan(performance_results[m].get('rmse', np.nan))]]
    
    bars = plt.bar(range(len(methods)), rmse_scores, color=['red' if i == np.argmin(rmse_scores) else 'lightcoral' for i in range(len(rmse_scores))])
    plt.title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE (cm)')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(rmse_scores):
        plt.text(i, v + max(rmse_scores)*0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 3: MAE comparison
    plt.subplot(2, 2, 3)
    mae_scores = [performance_results[method]['mae'] for method in [m for m in performance_results.keys() if not np.isnan(performance_results[m].get('mae', np.nan))]]
    
    bars = plt.bar(range(len(methods)), mae_scores, color=['red' if i == np.argmin(mae_scores) else 'lightsalmon' for i in range(len(mae_scores))])
    plt.title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    plt.ylabel('MAE (cm)')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(mae_scores):
        plt.text(i, v + max(mae_scores)*0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 4: Bias comparison
    plt.subplot(2, 2, 4)
    bias_scores = [performance_results[method]['bias'] for method in [m for m in performance_results.keys() if not np.isnan(performance_results[m].get('bias', np.nan))]]
    
    bars = plt.bar(range(len(methods)), bias_scores, color=['red' if abs(bias_scores[i]) == min([abs(b) for b in bias_scores]) else 'lightpink' for i in range(len(bias_scores))])
    plt.title('Bias Comparison (Closer to 0 is Better)', fontsize=14, fontweight='bold')
    plt.ylabel('Bias (cm)')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add values on bars
    for i, v in enumerate(bias_scores):
        plt.text(i, v + max(bias_scores)*0.05 if v >= 0 else v - max(bias_scores)*0.05, f'{v:.4f}', ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residual_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Summary table plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create summary table
    table_data = []
    for method, metrics in performance_results.items():
        if not np.isnan(metrics.get('r2', np.nan)):
            table_data.append([
                metrics['method_name'],
                f"{metrics['r2']:.4f}",
                f"{metrics['rmse']:.3f}",
                f"{metrics['mae']:.3f}",
                f"{metrics['bias']:.4f}",
                str(metrics['n_samples'])
            ])
    
    # Sort by R²
    table_data.sort(key=lambda x: float(x[1]), reverse=True)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Method', 'R²', 'RMSE (cm)', 'MAE (cm)', 'Bias (cm)', 'Samples'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Highlight best method (first row after sorting)
    for i in range(len(table_data[0])):
        table[(1, i)].set_facecolor('#ffcccc')
        table[(1, i)].set_text_props(weight='bold')
    
    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#cccccc')
        table[(0, i)].set_text_props(weight='bold')
    
    ax.set_title('Residual Correction Method Performance Summary', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.savefig(output_dir / 'residual_method_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved plots to {output_dir}")


def update_config_with_best_method(config, best_method, config_path):
    """Update configuration file with the best residual correction method."""
    print(f"\n⚙️ Updating configuration with best method: {best_method}")
    
    # Update config
    if 'residual_correction' not in config:
        config['residual_correction'] = {}
    
    config['residual_correction']['interpolation_method'] = best_method
    config['residual_correction']['auto_selected'] = True
    config['residual_correction']['selection_timestamp'] = pd.Timestamp.now().isoformat()
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"   ✅ Configuration updated in {config_path}")


def run_full_pipeline_with_best_method(config, best_method, best_metrics):
    """Run the complete pipeline with the best residual correction method."""
    print(f"\n🚀 Running full pipeline with best method: {best_method}")
    print(f"   Expected performance: R² = {best_metrics['r2']:.4f}")
    
    # Import and run the main pipeline
    import subprocess
    import sys
    
    # Run the coarse-to-fine pipeline
    try:
        result = subprocess.run([
            sys.executable, 'pipeline_coarse_to_fine.py',
            '--config', 'src_new_approach/config_coarse_to_fine.yaml'
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print("   ✅ Pipeline completed successfully!")
            print("   📊 Check validation plots for final results")
        else:
            print(f"   ❌ Pipeline failed: {result.stderr}")
            
    except Exception as e:
        print(f"   ⚠️ Could not automatically run pipeline: {e}")
        print(f"   💡 Manually run: python pipeline_coarse_to_fine.py")


def main():
    """Main function to test all residual correction methods."""
    print("="*80)
    print("🧪 COMPREHENSIVE RESIDUAL CORRECTION METHOD TESTING")
    print("="*80)
    print("Testing 8 different interpolation methods to find the optimal approach")
    print("for GRACE satellite data downscaling residual correction.")
    print("="*80)
    
    # Load configuration
    config_path = Path('src_new_approach/config_coarse_to_fine.yaml')
    config = load_config(str(config_path))
    
    # Initialize tester
    tester = MultiMethodResidualCorrector(config)
    
    # Load all pipeline data
    data = load_pipeline_data(config)
    
    # Generate coarse predictions
    predictions_coarse_ds = generate_coarse_predictions(config, data, tester)
    
    # Calculate residuals at coarse scale
    residuals_coarse_ds = calculate_coarse_residuals(
        data['grace_ds'], 
        predictions_coarse_ds, 
        tester
    )
    
    # Test all interpolation methods
    residuals_fine_results = test_all_interpolation_methods(
        residuals_coarse_ds,
        data['fine_features_ds'],
        tester
    )
    
    # Evaluate performance of all methods
    performance_results = evaluate_all_methods(
        residuals_fine_results,
        data,
        config,
        tester
    )
    
    # Select best method
    best_method, best_metrics = tester.select_best_method(performance_results)
    
    if best_method is None:
        print("❌ No valid methods found! Check your data and configuration.")
        return
    
    # Create comparison plots
    figures_dir = Path('figures_coarse_to_fine')
    create_comparison_plots(performance_results, residuals_fine_results, figures_dir)
    
    # Update configuration
    update_config_with_best_method(config, best_method, config_path)
    
    # Ask user if they want to run the full pipeline
    print(f"\n🎯 OPTIMIZATION COMPLETE!")
    print(f"   Best method: {tester.method_configs[best_method]['name']}")
    print(f"   Performance: R² = {best_metrics['r2']:.4f}")
    print(f"   Configuration updated with optimal method")
    
    response = input("\n💡 Would you like to run the full pipeline with the best method? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        run_full_pipeline_with_best_method(config, best_method, best_metrics)
    else:
        print("✅ Optimization complete! Run the pipeline manually when ready.")
    
    print("\n" + "="*80)
    print("🏆 RESIDUAL CORRECTION METHOD OPTIMIZATION FINISHED")
    print("="*80)
    print(f"Results saved to: {figures_dir}")
    print(f"Configuration updated: {config_path}")
    print("="*80)


if __name__ == "__main__":
    main()