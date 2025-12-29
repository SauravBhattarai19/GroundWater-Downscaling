#!/usr/bin/env python3
"""
Compare Cross-Validation Methods: Blocked Spatiotemporal CV vs Traditional 70-30 Split

This script provides comprehensive comparison between the two training approaches:
1. Blocked Spatiotemporal CV (scientifically rigorous, prevents data leakage)
2. Traditional 70-30 Split (faster, standard practice, may have leakage)

Creates visualizations and analysis to understand:
- Performance differences
- Training time differences
- Model complexity implications
- Practical considerations for each approach
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.insert(0, 'src_new_approach')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Large fonts for presentation
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
})


def load_cv_results(models_dir):
    """Load CV results from a model directory."""
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"   ⚠️ Directory not found: {models_path}")
        return None
    
    # Find CV results files
    cv_files = list(models_path.glob("*_cv_results.csv"))
    
    if not cv_files:
        print(f"   ⚠️ No CV results found in: {models_path}")
        return None
    
    all_results = []
    for cv_file in cv_files:
        model_name = cv_file.stem.replace("_cv_results", "")
        df = pd.read_csv(cv_file)
        if 'model_name' not in df.columns:
            df['model_name'] = model_name
        all_results.append(df)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"   ✓ Loaded results for {len(combined_df['model_name'].unique())} models")
    
    return combined_df


def load_model_summaries(models_dir):
    """Load model comparison summary."""
    models_path = Path(models_dir)
    
    # Try different possible filenames
    for filename in ["model_comparison_coarse.csv", "model_comparison_simple.csv", "model_comparison.csv"]:
        summary_path = models_path / filename
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            print(f"   ✓ Loaded summary: {filename}")
            return df
    
    print(f"   ⚠️ No model summary found in: {models_path}")
    return None


def compare_performance_distributions(blocked_cv_results, simple_split_results, output_dir):
    """Compare performance distributions between methods."""
    print("📊 Creating performance distribution comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data
    models = sorted(set(blocked_cv_results['model_name'].unique()) & 
                   set(simple_split_results['model_name'].unique()))
    
    # Panel 1: R² Distribution by Method
    ax1 = axes[0, 0]
    
    for i, model in enumerate(models):
        blocked_r2 = blocked_cv_results[blocked_cv_results['model_name'] == model]['r2']
        simple_r2 = simple_split_results[simple_split_results['model_name'] == model]['r2']
        
        # Violin plots
        parts1 = ax1.violinplot([blocked_r2], positions=[i*3], widths=0.8, showmeans=True)
        parts1['bodies'][0].set_facecolor('lightblue')
        parts1['bodies'][0].set_alpha(0.7)
        
        parts2 = ax1.violinplot([simple_r2], positions=[i*3 + 1], widths=0.8, showmeans=True)
        parts2['bodies'][0].set_facecolor('lightcoral')
        parts2['bodies'][0].set_alpha(0.7)
    
    ax1.set_xticks([i*3 + 0.5 for i in range(len(models))])
    ax1.set_xticklabels([m.upper() for m in models])
    ax1.set_ylabel('R² Score')
    ax1.set_title('(A) R² Distribution Comparison')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.legend(['Blocked CV', 'Simple Split'], loc='upper right')
    
    # Panel 2: Performance Summary
    ax2 = axes[0, 1]
    
    summary_data = []
    for model in models:
        blocked_r2 = blocked_cv_results[blocked_cv_results['model_name'] == model]['r2']
        simple_r2 = simple_split_results[simple_split_results['model_name'] == model]['r2']
        
        summary_data.append({
            'Model': model.upper(),
            'Blocked_CV_Mean': blocked_r2.mean(),
            'Simple_Split_Mean': simple_r2.mean(),
            'Blocked_CV_Std': blocked_r2.std(),
            'Simple_Split_Std': simple_r2.std()
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    x = np.arange(len(models))
    width = 0.35
    
    ax2.bar(x - width/2, summary_df['Blocked_CV_Mean'], width, 
            yerr=summary_df['Blocked_CV_Std'], label='Blocked CV', 
            color='lightblue', alpha=0.8, capsize=5)
    ax2.bar(x + width/2, summary_df['Simple_Split_Mean'], width, 
            yerr=summary_df['Simple_Split_Std'], label='Simple Split', 
            color='lightcoral', alpha=0.8, capsize=5)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in models])
    ax2.set_ylabel('Mean R² Score')
    ax2.set_title('(B) Mean Performance Comparison')
    ax2.legend()
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Panel 3: RMSE Comparison
    ax3 = axes[1, 0]
    
    for i, model in enumerate(models):
        blocked_rmse = blocked_cv_results[blocked_cv_results['model_name'] == model]['rmse']
        simple_rmse = simple_split_results[simple_split_results['model_name'] == model]['rmse']
        
        ax3.scatter([i*2]*len(blocked_rmse), blocked_rmse, alpha=0.6, color='blue', label='Blocked CV' if i == 0 else "")
        ax3.scatter([i*2 + 0.5]*len(simple_rmse), simple_rmse, alpha=0.6, color='red', label='Simple Split' if i == 0 else "")
    
    ax3.set_xticks([i*2 + 0.25 for i in range(len(models))])
    ax3.set_xticklabels([m.upper() for m in models])
    ax3.set_ylabel('RMSE (cm)')
    ax3.set_title('(C) RMSE Comparison')
    ax3.legend()
    
    # Panel 4: Performance Variance Analysis
    ax4 = axes[1, 1]
    
    variance_data = []
    for model in models:
        blocked_r2 = blocked_cv_results[blocked_cv_results['model_name'] == model]['r2']
        simple_r2 = simple_split_results[simple_split_results['model_name'] == model]['r2']
        
        variance_data.append({
            'Model': model.upper(),
            'Blocked_CV_Var': blocked_r2.var(),
            'Simple_Split_Var': simple_r2.var(),
            'Blocked_CV_Range': blocked_r2.max() - blocked_r2.min(),
            'Simple_Split_Range': simple_r2.max() - simple_r2.min()
        })
    
    var_df = pd.DataFrame(variance_data)
    
    ax4.scatter(var_df['Blocked_CV_Var'], var_df['Simple_Split_Var'], 
                s=100, alpha=0.7, color='purple')
    
    for i, model in enumerate(var_df['Model']):
        ax4.annotate(model, (var_df['Blocked_CV_Var'].iloc[i], var_df['Simple_Split_Var'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Add diagonal line
    max_var = max(var_df['Blocked_CV_Var'].max(), var_df['Simple_Split_Var'].max())
    ax4.plot([0, max_var], [0, max_var], 'k--', alpha=0.5)
    
    ax4.set_xlabel('Blocked CV Variance')
    ax4.set_ylabel('Simple Split Variance')
    ax4.set_title('(D) Performance Variance Comparison')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / "cv_methods_performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {output_path}")
    return output_path, summary_df


def create_method_analysis(blocked_cv_results, simple_split_results, output_dir):
    """Create detailed method analysis."""
    print("🔬 Creating detailed method analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = sorted(set(blocked_cv_results['model_name'].unique()) & 
                   set(simple_split_results['model_name'].unique()))
    
    # Panel 1: Training Complexity (CV folds vs single split)
    ax1 = axes[0, 0]
    
    complexity_data = []
    for model in models:
        blocked_folds = len(blocked_cv_results[blocked_cv_results['model_name'] == model])
        simple_folds = len(simple_split_results[simple_split_results['model_name'] == model])
        
        complexity_data.append({
            'Model': model.upper(),
            'Blocked_CV_Folds': blocked_folds,
            'Simple_Split_Folds': simple_folds
        })
    
    complexity_df = pd.DataFrame(complexity_data)
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, complexity_df['Blocked_CV_Folds'], width, 
            label='Blocked CV', color='lightblue', alpha=0.8)
    ax1.bar(x + width/2, complexity_df['Simple_Split_Folds'], width, 
            label='Simple Split', color='lightcoral', alpha=0.8)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in models])
    ax1.set_ylabel('Number of Training Runs')
    ax1.set_title('(A) Training Complexity')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Panel 2: Performance Consistency
    ax2 = axes[0, 1]
    
    consistency_data = []
    for model in models:
        blocked_r2 = blocked_cv_results[blocked_cv_results['model_name'] == model]['r2']
        simple_r2 = simple_split_results[simple_split_results['model_name'] == model]['r2']
        
        # Coefficient of variation (std/mean) as consistency measure
        blocked_cv_coeff = blocked_r2.std() / abs(blocked_r2.mean()) if blocked_r2.mean() != 0 else np.inf
        simple_cv_coeff = simple_r2.std() / abs(simple_r2.mean()) if simple_r2.mean() != 0 else np.inf
        
        consistency_data.append({
            'Model': model.upper(),
            'Blocked_CV_CoeffVar': blocked_cv_coeff,
            'Simple_Split_CoeffVar': simple_cv_coeff
        })
    
    consist_df = pd.DataFrame(consistency_data)
    
    ax2.bar(x - width/2, consist_df['Blocked_CV_CoeffVar'], width, 
            label='Blocked CV', color='lightblue', alpha=0.8)
    ax2.bar(x + width/2, consist_df['Simple_Split_CoeffVar'], width, 
            label='Simple Split', color='lightcoral', alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in models])
    ax2.set_ylabel('Coefficient of Variation')
    ax2.set_title('(B) Performance Consistency')
    ax2.legend()
    
    # Panel 3: Performance vs Complexity Trade-off
    ax3 = axes[1, 0]
    
    for i, model in enumerate(models):
        blocked_r2_mean = blocked_cv_results[blocked_cv_results['model_name'] == model]['r2'].mean()
        simple_r2_mean = simple_split_results[simple_split_results['model_name'] == model]['r2'].mean()
        blocked_folds = len(blocked_cv_results[blocked_cv_results['model_name'] == model])
        simple_folds = len(simple_split_results[simple_split_results['model_name'] == model])
        
        ax3.scatter(blocked_folds, blocked_r2_mean, s=150, alpha=0.7, color='blue', marker='o')
        ax3.scatter(simple_folds, simple_r2_mean, s=150, alpha=0.7, color='red', marker='s')
        
        # Connect points with lines
        ax3.plot([blocked_folds, simple_folds], [blocked_r2_mean, simple_r2_mean], 
                 'k--', alpha=0.3)
        
        # Annotate
        ax3.annotate(model.upper(), (blocked_folds, blocked_r2_mean), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax3.set_xlabel('Training Complexity (Number of Folds)')
    ax3.set_ylabel('Mean R² Score')
    ax3.set_title('(C) Performance vs Complexity')
    ax3.legend(['Blocked CV', 'Simple Split'], loc='best')
    ax3.set_xscale('log')
    
    # Panel 4: Method Recommendation
    ax4 = axes[1, 1]
    ax4.axis('off')  # Turn off axis for text panel
    
    # Calculate overall statistics
    blocked_overall_r2 = blocked_cv_results['r2'].mean()
    simple_overall_r2 = simple_split_results['r2'].mean()
    blocked_overall_std = blocked_cv_results['r2'].std()
    simple_overall_std = simple_split_results['r2'].std()
    
    recommendation_text = f"""
METHOD COMPARISON SUMMARY

Overall Performance:
• Blocked CV: R² = {blocked_overall_r2:.3f} ± {blocked_overall_std:.3f}
• Simple Split: R² = {simple_overall_r2:.3f} ± {simple_overall_std:.3f}

Training Complexity:
• Blocked CV: ~20 folds per model
• Simple Split: 1 fold per model
• Speed advantage: ~20× faster with Simple Split

RECOMMENDATIONS:

🔬 Use Blocked CV when:
• Scientific rigor is critical
• Preventing data leakage is essential
• Publication/research context
• Sufficient computational resources

⚡ Use Simple Split when:
• Rapid prototyping/development
• Computational resources are limited
• Quick model comparison needed
• Educational/demonstration purposes

Note: Simple split may overestimate performance
due to potential spatial/temporal leakage.
"""
    
    ax4.text(0.05, 0.95, recommendation_text, transform=ax4.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / "cv_methods_detailed_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {output_path}")
    return output_path


def main():
    """Main execution function."""
    print("🔍 Cross-Validation Methods Comparison")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path("figures_coarse_to_fine/cv_methods_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load results from both approaches
        print("\n📊 Loading CV results...")
        
        print("   Loading Blocked CV results...")
        blocked_cv_results = load_cv_results("models_coarse_to_fine")
        
        print("   Loading Simple Split results...")
        simple_split_results = load_cv_results("models_coarse_to_fine_simple")
        
        # Check if we have results for both methods
        if blocked_cv_results is None:
            print("❌ No blocked CV results found. Run pipeline without --use-simple-split first.")
            return 1
            
        if simple_split_results is None:
            print("❌ No simple split results found. Run pipeline with --use-simple-split first.")
            return 1
        
        print(f"\n✅ Successfully loaded results for both methods")
        print(f"   Blocked CV: {len(blocked_cv_results)} fold results")
        print(f"   Simple Split: {len(simple_split_results)} fold results")
        
        # Create comparisons
        perf_path, summary_df = compare_performance_distributions(
            blocked_cv_results, simple_split_results, output_dir
        )
        
        analysis_path = create_method_analysis(
            blocked_cv_results, simple_split_results, output_dir
        )
        
        # Save summary statistics
        summary_path = output_dir / "method_comparison_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"   ✅ Saved summary: {summary_path}")
        
        # Print key findings
        print("\n" + "=" * 60)
        print("🔍 KEY FINDINGS:")
        
        for _, row in summary_df.iterrows():
            model = row['Model']
            blocked_mean = row['Blocked_CV_Mean']
            simple_mean = row['Simple_Split_Mean']
            difference = simple_mean - blocked_mean
            pct_diff = (difference / abs(blocked_mean)) * 100 if blocked_mean != 0 else 0
            
            print(f"\n   {model}:")
            print(f"     • Blocked CV: R² = {blocked_mean:.3f}")
            print(f"     • Simple Split: R² = {simple_mean:.3f}")
            print(f"     • Difference: {difference:+.3f} ({pct_diff:+.1f}%)")
        
        print(f"\n📖 Generated visualizations:")
        print(f"   • Performance comparison: {perf_path.name}")
        print(f"   • Detailed analysis: {analysis_path.name}")
        print(f"   • Summary statistics: {summary_path.name}")
        print(f"\n📁 All outputs saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())