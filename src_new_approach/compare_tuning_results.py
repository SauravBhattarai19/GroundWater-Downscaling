#!/usr/bin/env python3
"""
Tuning Results Comparison and Analysis Tool

This script provides comprehensive analysis and visualization of hyperparameter 
tuning results, allowing users to:
1. Compare tuned vs default hyperparameters 
2. Analyze optimization convergence
3. Visualize parameter importance
4. Generate comparison reports

Usage:
    python compare_tuning_results.py --tuned-params tuned_hyperparameters.json --config config.yaml
    python compare_tuning_results.py --study-db optuna_studies.db --output-dir results/
"""

import argparse
import json
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optuna imports
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# YAML import
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from utils_downscaling import load_config, get_config_value


class TuningResultsAnalyzer:
    """
    Comprehensive analysis of hyperparameter tuning results.
    """
    
    def __init__(self, 
                 tuned_params_path: Optional[str] = None,
                 study_db_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 output_dir: str = "tuning_analysis"):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        tuned_params_path : str, optional
            Path to tuned hyperparameters JSON file
        study_db_path : str, optional  
            Path to Optuna study database
        config_path : str, optional
            Path to original configuration file
        output_dir : str
            Output directory for analysis results
        """
        self.tuned_params_path = tuned_params_path
        self.study_db_path = study_db_path
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.tuned_params = {}
        self.default_params = {}
        self.studies = {}
        self.config = {}
        
        print(f"📊 Tuning Results Analyzer initialized")
        print(f"   Output directory: {self.output_dir}")
        
    def load_data(self) -> None:
        """Load all available data sources."""
        
        # Load tuned parameters
        if self.tuned_params_path and Path(self.tuned_params_path).exists():
            self._load_tuned_parameters()
        
        # Load Optuna studies
        if self.study_db_path and Path(self.study_db_path).exists():
            self._load_optuna_studies()
        
        # Load configuration
        if self.config_path and Path(self.config_path).exists():
            self._load_configuration()
            
    def _load_tuned_parameters(self) -> None:
        """Load tuned hyperparameters from JSON/pickle file."""
        tuned_path = Path(self.tuned_params_path)
        
        try:
            if tuned_path.suffix == '.json':
                with open(tuned_path, 'r') as f:
                    data = json.load(f)
                self.tuned_params = data.get('best_parameters', {})
                
            elif tuned_path.suffix == '.pkl':
                with open(tuned_path, 'rb') as f:
                    data = pickle.load(f)
                self.tuned_params = data.get('best_parameters', {})
                
            print(f"✅ Loaded tuned parameters for {len(self.tuned_params)} models")
            
        except Exception as e:
            print(f"⚠️ Could not load tuned parameters: {e}")
    
    def _load_optuna_studies(self) -> None:
        """Load Optuna studies from SQLite database."""
        if not HAS_OPTUNA:
            print("⚠️ Optuna not available - cannot load study database")
            return
            
        try:
            # Connect to SQLite database
            storage_url = f"sqlite:///{self.study_db_path}"
            
            # Get all study names
            conn = sqlite3.connect(self.study_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT study_name FROM studies")
            study_names = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            # Load each study
            for study_name in study_names:
                try:
                    study = optuna.load_study(study_name=study_name, storage=storage_url)
                    self.studies[study_name] = study
                except Exception as e:
                    print(f"⚠️ Could not load study {study_name}: {e}")
                    
            print(f"✅ Loaded {len(self.studies)} Optuna studies")
            
        except Exception as e:
            print(f"⚠️ Could not load Optuna studies: {e}")
    
    def _load_configuration(self) -> None:
        """Load original configuration file."""
        if not HAS_YAML:
            print("⚠️ YAML not available - cannot load configuration")
            return
            
        try:
            self.config = load_config(self.config_path)
            
            # Extract default hyperparameters
            for model_name in ['rf', 'xgb', 'lgb', 'nn']:
                default_params = get_config_value(self.config, f'models.hyperparameters.{model_name}', {})
                if default_params:
                    self.default_params[model_name] = default_params
                    
            print(f"✅ Loaded default parameters for {len(self.default_params)} models")
            
        except Exception as e:
            print(f"⚠️ Could not load configuration: {e}")
    
    def compare_hyperparameters(self) -> pd.DataFrame:
        """
        Compare tuned vs default hyperparameters.
        
        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        if not self.tuned_params or not self.default_params:
            print("⚠️ Need both tuned and default parameters for comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name in self.tuned_params.keys():
            if model_name not in self.default_params:
                continue
                
            tuned = self.tuned_params[model_name]
            default = self.default_params[model_name]
            
            # Find all parameters (union of both sets)
            all_params = set(tuned.keys()) | set(default.keys())
            
            for param_name in all_params:
                tuned_val = tuned.get(param_name, "N/A")
                default_val = default.get(param_name, "N/A")
                
                # Calculate improvement if both are numeric
                improvement = None
                if (isinstance(tuned_val, (int, float)) and 
                    isinstance(default_val, (int, float)) and 
                    default_val != 0):
                    improvement = ((tuned_val - default_val) / default_val) * 100
                
                comparison_data.append({
                    'model': model_name,
                    'parameter': param_name,
                    'default_value': default_val,
                    'tuned_value': tuned_val,
                    'improvement_percent': improvement,
                    'changed': tuned_val != default_val
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_path = self.output_dir / "hyperparameter_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"📊 Hyperparameter comparison saved: {comparison_path}")
        
        return comparison_df
    
    def plot_optimization_history(self) -> None:
        """Plot optimization history for each study."""
        if not self.studies:
            print("⚠️ No Optuna studies loaded")
            return
        
        n_studies = len(self.studies)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (study_name, study) in enumerate(self.studies.items()):
            if idx >= 4:  # Limit to 4 plots
                break
                
            ax = axes[idx]
            
            # Extract trials data
            trials = study.trials
            values = [t.value for t in trials if t.value is not None]
            trial_numbers = range(1, len(values) + 1)
            
            if not values:
                ax.text(0.5, 0.5, 'No completed trials', ha='center', va='center')
                ax.set_title(study_name)
                continue
            
            # Plot optimization history
            ax.plot(trial_numbers, values, 'b-', alpha=0.6, linewidth=1)
            
            # Plot best value line
            best_values = []
            best_so_far = float('-inf')
            for val in values:
                if val > best_so_far:
                    best_so_far = val
                best_values.append(best_so_far)
            
            ax.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best so far')
            
            ax.set_title(f'{study_name}\nBest: {max(values):.4f}')
            ax.set_xlabel('Trial')
            ax.set_ylabel('Objective Value (R²)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(self.studies), 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "optimization_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Optimization history saved: {plot_path}")
    
    def plot_parameter_importance(self) -> None:
        """Plot parameter importance from Optuna studies."""
        if not self.studies or not HAS_OPTUNA:
            print("⚠️ No Optuna studies loaded or Optuna not available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (study_name, study) in enumerate(self.studies.items()):
            if idx >= 4:  # Limit to 4 plots
                break
                
            ax = axes[idx]
            
            try:
                # Calculate parameter importance
                importance = optuna.importance.get_param_importances(study)
                
                if not importance:
                    ax.text(0.5, 0.5, 'No parameter importance data', ha='center', va='center')
                    ax.set_title(study_name)
                    continue
                
                # Plot horizontal bar chart
                params = list(importance.keys())
                values = list(importance.values())
                
                # Sort by importance
                sorted_data = sorted(zip(params, values), key=lambda x: x[1], reverse=True)
                params, values = zip(*sorted_data)
                
                # Take top 10
                params = params[:10]
                values = values[:10]
                
                bars = ax.barh(range(len(params)), values, color='skyblue', alpha=0.7)
                ax.set_yticks(range(len(params)))
                ax.set_yticklabels(params)
                ax.set_xlabel('Importance')
                ax.set_title(f'{study_name}\nParameter Importance')
                
                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, values)):
                    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                           f'{val:.3f}', va='center', fontsize=8)
                
                ax.grid(True, alpha=0.3, axis='x')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                ax.set_title(study_name)
        
        # Hide unused subplots
        for idx in range(len(self.studies), 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "parameter_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Parameter importance saved: {plot_path}")
    
    def generate_summary_report(self) -> None:
        """Generate comprehensive text summary report."""
        report_path = self.output_dir / "tuning_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HYPERPARAMETER TUNING ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Tuned parameters summary
            if self.tuned_params:
                f.write("TUNED HYPERPARAMETERS SUMMARY:\n")
                f.write("-"*40 + "\n")
                for model_name, params in self.tuned_params.items():
                    f.write(f"\n{model_name.upper()}:\n")
                    for param, value in params.items():
                        f.write(f"  • {param}: {value}\n")
                f.write("\n")
            
            # Studies summary
            if self.studies:
                f.write("OPTIMIZATION STUDIES SUMMARY:\n")
                f.write("-"*40 + "\n")
                for study_name, study in self.studies.items():
                    trials = study.trials
                    completed_trials = [t for t in trials if t.value is not None]
                    
                    f.write(f"\n{study_name}:\n")
                    f.write(f"  • Total trials: {len(trials)}\n")
                    f.write(f"  • Completed trials: {len(completed_trials)}\n")
                    
                    if completed_trials:
                        values = [t.value for t in completed_trials]
                        f.write(f"  • Best value: {max(values):.6f}\n")
                        f.write(f"  • Mean value: {np.mean(values):.6f}\n")
                        f.write(f"  • Std value: {np.std(values):.6f}\n")
                f.write("\n")
            
            # Comparison summary
            if self.tuned_params and self.default_params:
                f.write("TUNED VS DEFAULT COMPARISON:\n")
                f.write("-"*40 + "\n")
                for model_name in self.tuned_params.keys():
                    if model_name in self.default_params:
                        f.write(f"\n{model_name.upper()} parameter changes:\n")
                        
                        tuned = self.tuned_params[model_name]
                        default = self.default_params[model_name]
                        
                        changes = 0
                        for param in set(tuned.keys()) | set(default.keys()):
                            tuned_val = tuned.get(param, "N/A")
                            default_val = default.get(param, "N/A")
                            
                            if tuned_val != default_val:
                                changes += 1
                                f.write(f"  • {param}: {default_val} → {tuned_val}\n")
                        
                        if changes == 0:
                            f.write("  • No parameter changes\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-"*40 + "\n")
            f.write("1. Use the tuned hyperparameters in production training\n")
            f.write("2. Monitor model performance with tuned parameters\n")
            f.write("3. Consider re-tuning periodically with new data\n")
            f.write("4. Document parameter changes for reproducibility\n")
            f.write("5. Validate improvements on held-out test data\n")
        
        print(f"📋 Summary report saved: {report_path}")
    
    def run_full_analysis(self) -> None:
        """Run complete analysis pipeline."""
        print(f"\n{'='*60}")
        print("RUNNING COMPREHENSIVE TUNING ANALYSIS")
        print(f"{'='*60}")
        
        # Load all data
        print("\n1. Loading data...")
        self.load_data()
        
        # Compare hyperparameters
        print("\n2. Comparing hyperparameters...")
        comparison_df = self.compare_hyperparameters()
        
        # Plot optimization history
        print("\n3. Plotting optimization history...")
        self.plot_optimization_history()
        
        # Plot parameter importance
        print("\n4. Plotting parameter importance...")
        self.plot_parameter_importance()
        
        # Generate summary report
        print("\n5. Generating summary report...")
        self.generate_summary_report()
        
        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
        print(f"   • Hyperparameter comparison: hyperparameter_comparison.csv")
        print(f"   • Optimization history: optimization_history.png") 
        print(f"   • Parameter importance: parameter_importance.png")
        print(f"   • Summary report: tuning_summary_report.txt")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze and compare hyperparameter tuning results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--tuned-params',
        type=str,
        help='Path to tuned hyperparameters JSON/pickle file'
    )
    
    parser.add_argument(
        '--study-db',
        type=str,
        help='Path to Optuna study SQLite database'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to original configuration YAML file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='tuning_analysis',
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    if not any([args.tuned_params, args.study_db, args.config]):
        print("❌ Error: Must provide at least one data source")
        print("   Use --tuned-params, --study-db, or --config")
        parser.print_help()
        return 1
    
    # Initialize analyzer
    analyzer = TuningResultsAnalyzer(
        tuned_params_path=args.tuned_params,
        study_db_path=args.study_db,
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    # Run analysis
    analyzer.run_full_analysis()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())