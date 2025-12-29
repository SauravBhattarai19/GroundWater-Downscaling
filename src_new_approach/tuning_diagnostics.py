#!/usr/bin/env python3
"""
Hyperparameter Tuning Diagnostics Tool

This script provides comprehensive diagnostics and monitoring for hyperparameter
optimization processes, including:
1. Optimization convergence analysis
2. Parameter search space coverage
3. Trial efficiency and failure analysis
4. Resource utilization monitoring
5. Recommendation generation

Usage:
    python tuning_diagnostics.py --study-db optuna_studies.db --output-dir diagnostics/
    python tuning_diagnostics.py --study-name xgb_tuning --storage sqlite:///studies.db
"""

import argparse
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
from datetime import datetime, timedelta

# Optuna imports
try:
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances, 
        plot_slice,
        plot_parallel_coordinate,
        plot_contour
    )
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class TuningDiagnosticAnalyzer:
    """
    Comprehensive diagnostics for hyperparameter tuning optimization.
    """
    
    def __init__(self, output_dir: str = "tuning_diagnostics"):
        """
        Initialize diagnostic analyzer.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for diagnostic reports and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.studies = {}
        self.study_metadata = {}
        
        # Setup plotting
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        print(f"🔍 Tuning Diagnostics Analyzer initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Optuna available: {HAS_OPTUNA}")
        print(f"   Plotly available: {HAS_PLOTLY}")
    
    def load_studies_from_db(self, study_db_path: str) -> None:
        """
        Load all studies from Optuna SQLite database.
        
        Parameters:
        -----------
        study_db_path : str
            Path to Optuna study database
        """
        if not HAS_OPTUNA:
            print("❌ Optuna not available")
            return
        
        try:
            storage_url = f"sqlite:///{study_db_path}"
            
            # Get study names from database
            conn = sqlite3.connect(study_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT study_name FROM studies")
            study_names = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            # Load each study
            for study_name in study_names:
                try:
                    study = optuna.load_study(study_name=study_name, storage=storage_url)
                    self.studies[study_name] = study
                    
                    # Extract metadata
                    self.study_metadata[study_name] = self._extract_study_metadata(study)
                    
                    print(f"✅ Loaded study: {study_name}")
                    
                except Exception as e:
                    print(f"⚠️ Could not load study {study_name}: {e}")
                    
            print(f"\n📊 Loaded {len(self.studies)} studies for analysis")
            
        except Exception as e:
            print(f"❌ Error loading studies: {e}")
    
    def load_single_study(self, study_name: str, storage_url: str) -> None:
        """
        Load a single study from storage.
        
        Parameters:
        -----------
        study_name : str
            Name of the study to load
        storage_url : str
            Storage URL (e.g., sqlite:///studies.db)
        """
        if not HAS_OPTUNA:
            print("❌ Optuna not available")
            return
        
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            self.studies[study_name] = study
            self.study_metadata[study_name] = self._extract_study_metadata(study)
            
            print(f"✅ Loaded study: {study_name}")
            
        except Exception as e:
            print(f"❌ Error loading study {study_name}: {e}")
    
    def _extract_study_metadata(self, study: 'optuna.Study') -> Dict[str, Any]:
        """Extract metadata from an Optuna study."""
        trials = study.trials
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed_trials = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
        pruned_trials = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
        
        metadata = {
            'direction': study.direction.name,
            'n_trials': len(trials),
            'n_completed': len(completed_trials),
            'n_failed': len(failed_trials), 
            'n_pruned': len(pruned_trials),
            'best_value': study.best_value if completed_trials else None,
            'best_params': study.best_params if completed_trials else {},
            'optimization_time': None,
            'avg_trial_time': None,
            'success_rate': len(completed_trials) / len(trials) if trials else 0
        }
        
        # Calculate timing statistics
        if completed_trials:
            trial_times = []
            start_times = []
            
            for trial in completed_trials:
                if trial.datetime_start and trial.datetime_complete:
                    duration = trial.datetime_complete - trial.datetime_start
                    trial_times.append(duration.total_seconds())
                    start_times.append(trial.datetime_start)
            
            if trial_times:
                metadata['avg_trial_time'] = np.mean(trial_times)
                metadata['min_trial_time'] = np.min(trial_times)
                metadata['max_trial_time'] = np.max(trial_times)
                
            if start_times:
                total_time = max(start_times) - min(start_times)
                metadata['optimization_time'] = total_time.total_seconds()
        
        return metadata
    
    def analyze_convergence(self) -> None:
        """Analyze optimization convergence for all studies."""
        if not self.studies:
            print("⚠️ No studies loaded")
            return
        
        print("\n🔍 Analyzing convergence patterns...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (study_name, study) in enumerate(self.studies.items()):
            if idx >= 4:  # Limit to 4 studies
                break
                
            ax = axes[idx]
            
            trials = [t for t in study.trials if t.value is not None]
            if not trials:
                ax.text(0.5, 0.5, 'No completed trials', ha='center', va='center')
                ax.set_title(study_name)
                continue
            
            # Extract trial data
            trial_numbers = list(range(1, len(trials) + 1))
            objective_values = [t.value for t in trials]
            
            # Plot objective values
            ax.plot(trial_numbers, objective_values, 'b-', alpha=0.6, linewidth=1, label='Trial values')
            
            # Plot best values (cumulative best)
            best_values = []
            if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                best_so_far = float('-inf')
                for val in objective_values:
                    if val > best_so_far:
                        best_so_far = val
                    best_values.append(best_so_far)
            else:
                best_so_far = float('inf')
                for val in objective_values:
                    if val < best_so_far:
                        best_so_far = val
                    best_values.append(best_so_far)
            
            ax.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best so far')
            
            # Add convergence indicators
            if len(best_values) > 50:
                # Check for convergence (no improvement in last 25% of trials)
                improvement_threshold = 0.001  # 0.1% improvement threshold
                last_quarter = int(len(best_values) * 0.25)
                recent_best = best_values[-1]
                quarter_ago_best = best_values[-last_quarter]
                
                improvement = abs(recent_best - quarter_ago_best) / abs(quarter_ago_best) if quarter_ago_best != 0 else 0
                
                if improvement < improvement_threshold:
                    ax.axvline(len(best_values) - last_quarter, color='orange', 
                              linestyle='--', alpha=0.7, label='Potential convergence')
            
            ax.set_title(f'{study_name}\nBest: {best_values[-1]:.6f}')
            ax.set_xlabel('Trial')
            ax.set_ylabel('Objective Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(self.studies), 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "convergence_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Convergence analysis saved: {plot_path}")
    
    def analyze_parameter_space_coverage(self) -> None:
        """Analyze how well the parameter space was explored."""
        if not self.studies:
            print("⚠️ No studies loaded")
            return
        
        print("\n🗺️  Analyzing parameter space coverage...")
        
        for study_name, study in self.studies.items():
            trials = [t for t in study.trials if t.value is not None]
            if not trials or len(trials) < 10:
                continue
            
            # Get all parameters
            all_params = set()
            for trial in trials:
                all_params.update(trial.params.keys())
            
            all_params = list(all_params)
            
            if len(all_params) < 2:
                continue
            
            # Create parameter coverage visualization
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            # Plot distributions for top 4 parameters (by variance)
            param_variances = {}
            for param in all_params:
                values = [trial.params.get(param, np.nan) for trial in trials]
                values = [v for v in values if not pd.isna(v)]
                if values and len(set(values)) > 1:
                    param_variances[param] = np.var(values) if isinstance(values[0], (int, float)) else len(set(values))
            
            top_params = sorted(param_variances.items(), key=lambda x: x[1], reverse=True)[:4]
            
            for idx, (param_name, variance) in enumerate(top_params):
                ax = axes[idx]
                
                # Get parameter values
                param_values = [trial.params.get(param_name) for trial in trials]
                param_values = [v for v in param_values if v is not None]
                
                if not param_values:
                    continue
                
                # Plot distribution
                if isinstance(param_values[0], (int, float)):
                    # Numerical parameter
                    ax.hist(param_values, bins=20, alpha=0.7, edgecolor='black')
                    ax.set_title(f'{param_name}\nVariance: {variance:.4f}')
                    ax.set_xlabel('Parameter Value')
                    ax.set_ylabel('Frequency')
                else:
                    # Categorical parameter
                    value_counts = pd.Series(param_values).value_counts()
                    ax.bar(range(len(value_counts)), value_counts.values, alpha=0.7)
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45)
                    ax.set_title(f'{param_name}\nUnique values: {len(value_counts)}')
                    ax.set_ylabel('Frequency')
                
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(top_params), 4):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Parameter Space Coverage - {study_name}', fontsize=14)
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / f"parameter_coverage_{study_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🗺️  Parameter coverage for {study_name} saved: {plot_path}")
    
    def analyze_trial_failures(self) -> None:
        """Analyze trial failures and their patterns."""
        if not self.studies:
            print("⚠️ No studies loaded")
            return
        
        print("\n❌ Analyzing trial failures...")
        
        failure_data = []
        
        for study_name, study in self.studies.items():
            failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            total_trials = len(study.trials)
            
            failure_data.append({
                'study': study_name,
                'total_trials': total_trials,
                'failed_trials': len(failed_trials),
                'pruned_trials': len(pruned_trials),
                'failed_rate': len(failed_trials) / total_trials if total_trials > 0 else 0,
                'pruned_rate': len(pruned_trials) / total_trials if total_trials > 0 else 0,
                'success_rate': 1 - (len(failed_trials) + len(pruned_trials)) / total_trials if total_trials > 0 else 0
            })
        
        failure_df = pd.DataFrame(failure_data)
        
        # Create failure analysis plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Trial outcomes stacked bar chart
        studies = failure_df['study'].values
        x_pos = np.arange(len(studies))
        
        completed = failure_df['success_rate'].values
        failed = failure_df['failed_rate'].values
        pruned = failure_df['pruned_rate'].values
        
        ax1.bar(x_pos, completed, label='Completed', alpha=0.8)
        ax1.bar(x_pos, failed, bottom=completed, label='Failed', alpha=0.8) 
        ax1.bar(x_pos, pruned, bottom=completed+failed, label='Pruned', alpha=0.8)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(studies, rotation=45)
        ax1.set_ylabel('Proportion of Trials')
        ax1.set_title('Trial Outcome Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Failure rate comparison
        ax2.bar(x_pos, failure_df['failed_rate'].values, alpha=0.7, color='red', label='Failed')
        ax2.bar(x_pos, failure_df['pruned_rate'].values, alpha=0.7, color='orange', label='Pruned', bottom=failure_df['failed_rate'].values)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(studies, rotation=45)
        ax2.set_ylabel('Failure Rate')
        ax2.set_title('Trial Failure Rates')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "trial_failure_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save data
        csv_path = self.output_dir / "trial_failure_data.csv"
        failure_df.to_csv(csv_path, index=False)
        
        print(f"❌ Trial failure analysis saved: {plot_path}")
        print(f"📄 Trial failure data saved: {csv_path}")
    
    def analyze_optimization_efficiency(self) -> None:
        """Analyze optimization efficiency and resource utilization."""
        if not self.studies:
            print("⚠️ No studies loaded")
            return
        
        print("\n⚡ Analyzing optimization efficiency...")
        
        efficiency_data = []
        
        for study_name, study in self.studies.items():
            metadata = self.study_metadata[study_name]
            
            completed_trials = [t for t in study.trials if t.value is not None]
            if not completed_trials:
                continue
            
            # Calculate efficiency metrics
            values = [t.value for t in completed_trials]
            
            # Improvement rate (how quickly best value is reached)
            best_values = []
            if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                best_so_far = float('-inf')
                for val in values:
                    best_so_far = max(best_so_far, val)
                    best_values.append(best_so_far)
            else:
                best_so_far = float('inf')
                for val in values:
                    best_so_far = min(best_so_far, val)
                    best_values.append(best_so_far)
            
            # Find when 90% of final improvement was reached
            final_improvement = abs(best_values[-1] - best_values[0])
            target_improvement = 0.9 * final_improvement
            
            trials_to_90pct = len(best_values)  # Default to all trials
            for i, val in enumerate(best_values):
                current_improvement = abs(val - best_values[0])
                if current_improvement >= target_improvement:
                    trials_to_90pct = i + 1
                    break
            
            efficiency_data.append({
                'study': study_name,
                'total_trials': metadata['n_trials'],
                'completed_trials': metadata['n_completed'],
                'trials_to_90pct': trials_to_90pct,
                'efficiency_ratio': trials_to_90pct / metadata['n_completed'] if metadata['n_completed'] > 0 else 1,
                'avg_trial_time': metadata.get('avg_trial_time', 0),
                'total_time': metadata.get('optimization_time', 0),
                'best_value': metadata['best_value'],
                'success_rate': metadata['success_rate']
            })
        
        efficiency_df = pd.DataFrame(efficiency_data)
        
        # Create efficiency visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        studies = efficiency_df['study'].values
        x_pos = np.arange(len(studies))
        
        # Efficiency ratio (lower is better)
        ax1.bar(x_pos, efficiency_df['efficiency_ratio'].values, alpha=0.7)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(studies, rotation=45)
        ax1.set_ylabel('Efficiency Ratio (Trials to 90%)')
        ax1.set_title('Optimization Efficiency\n(Lower is Better)')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Average trial time
        ax2.bar(x_pos, efficiency_df['avg_trial_time'].values, alpha=0.7, color='orange')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(studies, rotation=45)
        ax2.set_ylabel('Avg Trial Time (seconds)')
        ax2.set_title('Average Trial Duration')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Success rate vs efficiency
        ax3.scatter(efficiency_df['success_rate'].values, efficiency_df['efficiency_ratio'].values, 
                   s=100, alpha=0.7)
        for i, study in enumerate(studies):
            ax3.annotate(study, (efficiency_df['success_rate'].iloc[i], efficiency_df['efficiency_ratio'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax3.set_xlabel('Success Rate')
        ax3.set_ylabel('Efficiency Ratio')
        ax3.set_title('Success Rate vs Efficiency')
        ax3.grid(True, alpha=0.3)
        
        # Total optimization time
        ax4.bar(x_pos, efficiency_df['total_time'].values / 3600, alpha=0.7, color='green')  # Convert to hours
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(studies, rotation=45)
        ax4.set_ylabel('Total Time (hours)')
        ax4.set_title('Total Optimization Time')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "optimization_efficiency.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save data
        csv_path = self.output_dir / "efficiency_analysis.csv"
        efficiency_df.to_csv(csv_path, index=False)
        
        print(f"⚡ Optimization efficiency saved: {plot_path}")
        print(f"📄 Efficiency data saved: {csv_path}")
    
    def generate_diagnostic_report(self) -> None:
        """Generate comprehensive diagnostic report."""
        if not self.studies:
            print("⚠️ No studies loaded for report")
            return
        
        report_path = self.output_dir / "tuning_diagnostics_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HYPERPARAMETER TUNING DIAGNOSTICS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive summary
            f.write("EXECUTIVE SUMMARY:\n")
            f.write("-"*40 + "\n")
            f.write(f"• Studies analyzed: {len(self.studies)}\n")
            
            total_trials = sum(meta['n_trials'] for meta in self.study_metadata.values())
            total_completed = sum(meta['n_completed'] for meta in self.study_metadata.values())
            
            f.write(f"• Total trials: {total_trials:,}\n")
            f.write(f"• Completed trials: {total_completed:,}\n")
            f.write(f"• Overall success rate: {total_completed/total_trials*100:.1f}%\n\n")
            
            # Study-by-study analysis
            f.write("STUDY-BY-STUDY ANALYSIS:\n")
            f.write("-"*40 + "\n")
            
            for study_name, metadata in self.study_metadata.items():
                f.write(f"\n{study_name.upper()}:\n")
                f.write(f"  • Total trials: {metadata['n_trials']:,}\n")
                f.write(f"  • Completed: {metadata['n_completed']:,} ({metadata['success_rate']*100:.1f}%)\n")
                f.write(f"  • Failed: {metadata['n_failed']:,}\n")
                f.write(f"  • Pruned: {metadata['n_pruned']:,}\n")
                
                if metadata['best_value']:
                    f.write(f"  • Best value: {metadata['best_value']:.6f}\n")
                
                if metadata['avg_trial_time']:
                    f.write(f"  • Avg trial time: {metadata['avg_trial_time']:.1f}s\n")
                
                if metadata['optimization_time']:
                    f.write(f"  • Total time: {metadata['optimization_time']/3600:.1f}h\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-"*40 + "\n")
            
            # Analyze patterns for recommendations
            avg_success_rate = np.mean([meta['success_rate'] for meta in self.study_metadata.values()])
            
            if avg_success_rate < 0.8:
                f.write("1. SUCCESS RATE OPTIMIZATION:\n")
                f.write(f"   • Current success rate ({avg_success_rate*100:.1f}%) is below optimal (80%+)\n")
                f.write("   • Consider adjusting search space bounds\n")
                f.write("   • Review memory/timeout limits\n")
                f.write("   • Check for data-related failures\n\n")
            
            # Check for convergence issues
            long_studies = [name for name, meta in self.study_metadata.items() if meta['n_completed'] > 200]
            if long_studies:
                f.write("2. CONVERGENCE OPTIMIZATION:\n")
                f.write("   • Some studies ran many trials (>200)\n")
                f.write("   • Consider implementing early stopping\n")
                f.write("   • Use pruning for faster convergence\n")
                f.write("   • Review convergence criteria\n\n")
            
            f.write("3. EFFICIENCY IMPROVEMENTS:\n")
            f.write("   • Monitor parameter importance to focus search\n")
            f.write("   • Use warm start from previous successful runs\n") 
            f.write("   • Consider multi-objective optimization if needed\n")
            f.write("   • Implement better resource management\n\n")
            
            f.write("4. VALIDATION:\n")
            f.write("   • Test tuned parameters on independent dataset\n")
            f.write("   • Monitor for overfitting to validation set\n")
            f.write("   • Document parameter changes and improvements\n")
            f.write("   • Plan periodic re-tuning schedule\n")
        
        print(f"📋 Diagnostic report saved: {report_path}")
    
    def run_full_diagnostics(self) -> None:
        """Run complete diagnostic analysis pipeline."""
        print(f"\n{'='*60}")
        print("RUNNING COMPREHENSIVE TUNING DIAGNOSTICS")
        print(f"{'='*60}")
        
        if not self.studies:
            print("❌ No studies loaded - cannot run diagnostics")
            return
        
        # Convergence analysis
        print("\n1. Analyzing convergence patterns...")
        self.analyze_convergence()
        
        # Parameter space coverage
        print("\n2. Analyzing parameter space coverage...")
        self.analyze_parameter_space_coverage()
        
        # Trial failures
        print("\n3. Analyzing trial failures...")
        self.analyze_trial_failures()
        
        # Optimization efficiency
        print("\n4. Analyzing optimization efficiency...")
        self.analyze_optimization_efficiency()
        
        # Generate report
        print("\n5. Generating diagnostic report...")
        self.generate_diagnostic_report()
        
        print(f"\n✅ Diagnostic analysis complete! Results saved to: {self.output_dir}")
        print(f"   • Convergence: convergence_analysis.png")
        print(f"   • Parameter coverage: parameter_coverage_*.png")
        print(f"   • Trial failures: trial_failure_analysis.png")
        print(f"   • Efficiency: optimization_efficiency.png")
        print(f"   • Report: tuning_diagnostics_report.txt")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive diagnostics for hyperparameter tuning optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--study-db',
        type=str,
        help='Path to Optuna study SQLite database'
    )
    
    parser.add_argument(
        '--study-name',
        type=str,
        help='Name of specific study to analyze'
    )
    
    parser.add_argument(
        '--storage',
        type=str,
        help='Optuna storage URL (e.g., sqlite:///studies.db)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='tuning_diagnostics',
        help='Output directory for diagnostic results'
    )
    
    args = parser.parse_args()
    
    if not any([args.study_db, args.study_name]):
        print("❌ Error: Must provide either --study-db or --study-name")
        parser.print_help()
        return 1
    
    # Initialize analyzer
    analyzer = TuningDiagnosticAnalyzer(output_dir=args.output_dir)
    
    # Load studies
    if args.study_db:
        print(f"📊 Loading studies from database: {args.study_db}")
        analyzer.load_studies_from_db(args.study_db)
    elif args.study_name and args.storage:
        print(f"📊 Loading study: {args.study_name}")
        analyzer.load_single_study(args.study_name, args.storage)
    else:
        print("❌ Error: Need --storage with --study-name")
        return 1
    
    # Run diagnostics
    analyzer.run_full_diagnostics()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())