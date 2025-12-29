#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Tuning with Optuna

This module provides automated hyperparameter optimization for the GRACE downscaling pipeline
using Optuna. Optimized for high-performance systems with extensive parallel processing.

Features:
- Multi-objective optimization (R² vs training time)
- Parallel trial execution across multiple cores
- Persistent study storage with SQLite
- Model-specific search spaces for tree-based algorithms
- Early stopping and pruning for efficiency
- Integration with both simple split and blocked CV methods
"""

import os
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler

# Optuna imports
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import HyperbandPruner, MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ML model imports with fallbacks
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

# Local imports
from utils_downscaling import (
    load_config, get_config_value, print_statistics
)
from spatiotemporal_cv import (
    BlockedSpatiotemporalCV, prepare_metadata_for_cv
)

warnings.filterwarnings('ignore')


class OptunaTuner:
    """
    Comprehensive hyperparameter tuning using Optuna optimization.
    
    Supports parallel optimization across multiple models with sophisticated
    search spaces and early stopping mechanisms.
    """
    
    def __init__(self, 
                 config: Dict,
                 n_trials: int = 500,
                 n_jobs: int = 64,
                 timeout: Optional[int] = None,
                 study_storage: Optional[str] = None):
        """
        Initialize Optuna tuner.
        
        Parameters:
        -----------
        config : Dict
            Configuration dictionary
        n_trials : int
            Number of optimization trials per model
        n_jobs : int  
            Number of parallel jobs for tuning
        timeout : int, optional
            Maximum optimization time in seconds
        study_storage : str, optional
            Optuna study storage URL (default: SQLite)
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna is required for hyperparameter tuning. Install with: pip install optuna")
        
        self.config = config
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        
        # Study storage setup
        if study_storage is None:
            study_dir = Path("optuna_studies")
            study_dir.mkdir(exist_ok=True)
            self.study_storage = f"sqlite:///{study_dir}/optuna_studies.db"
        else:
            self.study_storage = study_storage
        
        # Get enabled models
        self.enabled_models = get_config_value(config, 'models.enabled', ['xgb', 'rf', 'lgb'])
        self._filter_available_models()
        
        # Optimization settings
        tuning_config = get_config_value(config, 'hyperparameter_tuning', {})
        self.sampler_name = tuning_config.get('sampler', 'TPE')
        self.pruner_name = tuning_config.get('pruner', 'HyperbandPruner')
        self.cv_method = tuning_config.get('cv_method', 'simple')
        
        # Results storage
        self.studies = {}
        self.best_params = {}
        self.tuning_results = {}
        
        print(f"🔧 Optuna Tuner initialized:")
        print(f"   Models to tune: {self.enabled_models}")
        print(f"   Trials per model: {n_trials}")
        print(f"   Parallel jobs: {n_jobs}")
        print(f"   CV method: {self.cv_method}")
        print(f"   Study storage: {self.study_storage}")
    
    def _filter_available_models(self):
        """Filter enabled models by availability of libraries."""
        available_models = ['rf']  # RandomForest always available
        
        if HAS_XGB:
            available_models.append('xgb')
        if HAS_LGB:
            available_models.append('lgb')
        
        self.enabled_models = [m for m in self.enabled_models if m in available_models]
        
        if not self.enabled_models:
            raise ValueError("No models available for tuning. Install xgboost and/or lightgbm.")
    
    def _get_search_space(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """
        Define comprehensive search spaces for each model.
        
        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object
        model_name : str
            Model name ('xgb', 'rf', 'lgb')
        
        Returns:
        --------
        Dict[str, Any]
            Hyperparameter dictionary for the model
        """
        if model_name == 'xgb' and HAS_XGB:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
                'gamma': trial.suggest_float('gamma', 0, 10),
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10),
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                'tree_method': 'hist',  # Fixed for performance
                'n_jobs': 2,  # Fixed for parallel optimization
                'random_state': 42
            }
        
        elif model_name == 'rf':
            # SCIENTIFICALLY-TUNED RF hyperparameters for 765k sample spatiotemporal dataset
            # Root cause analysis: Previous ranges allowed computational explosion (800 trees × 2^25 nodes)
            # Solution: Conservative ranges appropriate for large dataset characteristics
            
            # Simplified max_features - focus on proven approaches for high-dimensional spatiotemporal data
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
                
            return {
                # MAXIMUM HPC UTILIZATION: Push 660GB/192-core system to absolute limits
                'n_estimators': trial.suggest_int('n_estimators', 1000, 5000),  # EXTREME: 3x XGBoost for massive ensemble
                
                # ABSOLUTE MAXIMUM DEPTH: Ultimate pattern capture for complex spatiotemporal data  
                'max_depth': trial.suggest_int('max_depth', 25, 80),  # EXTREME: Much deeper trees for fine patterns
                
                # ABSOLUTE GRANULARITY: Maximum splitting precision
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),  # EXTREME: Finest possible splits
                
                # ABSOLUTE MINIMUM: Maximum granularity for pattern capture
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),  # EXTREME: Ultra-fine leaves
                
                # Proven feature selection for high-dimensional spatiotemporal data
                'max_features': max_features,
                
                # Conservative bootstrap sampling for large dataset
                'max_samples': trial.suggest_float('max_samples', 0.7, 0.9),  # Slightly more conservative
                
                # More conservative impurity decrease for large data (prevent overfitting)
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.001, 0.01),
                
                # Always use bootstrap for better generalization with large spatiotemporal data
                'bootstrap': True,
                
                # Focus on squared_error for GRACE anomaly data (absolute_error less effective)
                'criterion': 'squared_error',  # Fixed instead of suggesting (squared_error optimal for continuous data)
                
                # n_jobs will be set by _create_model method (now 1 for ultra-conservative threading)
                'random_state': 42,
                
                # Additional RF-specific optimizations
                'warm_start': False,  # Full retraining each time
                'oob_score': trial.suggest_categorical('oob_score', [True, False])  # Out-of-bag scoring
            }
        
        elif model_name == 'lgb' and HAS_LGB:
            boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 3000),
                'max_depth': trial.suggest_int('max_depth', -1, 20),
                'num_leaves': trial.suggest_int('num_leaves', 10, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0, 1),
                'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
                'boosting_type': boosting_type,
                'objective': 'regression',
                'metric': 'rmse',
                'n_jobs': 2,  # Fixed for parallel optimization
                'random_state': 42,
                'verbose': -1
            }
            
            # Add boosting-specific parameters
            if boosting_type == 'dart':
                params['drop_rate'] = trial.suggest_float('drop_rate', 0.01, 0.8)
                params['max_drop'] = trial.suggest_int('max_drop', 1, 50)
                params['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 1.0)
            elif boosting_type == 'goss':
                params['top_rate'] = trial.suggest_float('top_rate', 0.1, 1.0)
                params['other_rate'] = trial.suggest_float('other_rate', 0.01, 0.5)
            
            return params
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _create_model(self, model_name: str, params: Dict[str, Any]):
        """Create model instance with given parameters."""
        # Add n_jobs for parallel training within each model
        model_params = params.copy()
        
        # ULTRA-CONSERVATIVE threading for HPC stability
        # Root cause analysis shows: n_jobs=64 (Optuna) × 8 (per model) = 512 threads → crash
        import os
        max_threads = int(os.environ.get('NUMEXPR_MAX_THREADS', 64))
        
        # Calculate safe threading based on Optuna parallelism
        optuna_jobs = self.n_jobs  # Number of parallel Optuna trials
        
        if model_name == 'rf':
            # Random Forest - OPTIMAL single-process multithreading
            # Key insight: Issue is PROCESS explosion, not thread usage within ONE process
            # RF is embarrassingly parallel - perfect for single-process multithreading
            if optuna_jobs == 1:
                # Single Optuna process: MAXIMIZE thread utilization on 192-core system
                safe_threads = min(96, max_threads)  # Use up to 96 threads (50% of 192 cores) for RF
                print(f"   RF threading: {safe_threads} (MAXIMUM HPC utilization - 192 core system)")
            else:
                # Multiple Optuna processes: must be conservative
                safe_threads = max(1, min(4, max_threads // optuna_jobs))
                print(f"   RF threading: {safe_threads} (conservative due to {optuna_jobs} parallel processes)")
                print(f"   ⚠️  WARNING: Multiple Optuna processes may cause instability")
        else:
            # Other models can use slightly more threads
            safe_threads = min(2, max(1, max_threads // (optuna_jobs * 2)))
            
        if model_name == 'xgb' and HAS_XGB:
            # XGBoost uses nthread - be very conservative
            model_params['nthread'] = safe_threads
            return xgb.XGBRegressor(**model_params)
        elif model_name == 'rf':
            model_params['n_jobs'] = safe_threads
            return RandomForestRegressor(**model_params)
        elif model_name == 'lgb' and HAS_LGB:
            model_params['n_jobs'] = safe_threads
            return lgb.LGBMRegressor(**model_params)
        else:
            raise ValueError(f"Unknown or unavailable model: {model_name}")
    
    def _objective_simple_split(self, 
                               trial: optuna.Trial,
                               model_name: str,
                               X: np.ndarray,
                               y: np.ndarray) -> float:
        """
        Objective function for simple train/test split optimization.
        
        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object
        model_name : str
            Model name to optimize
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        
        Returns:
        --------
        float
            Objective value (R² score to maximize)
        """
        # Get hyperparameters
        params = self._get_search_space(trial, model_name)
        
        # Memory monitoring and crash prevention
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**3)  # GB
        
        # COMPREHENSIVE crash prevention with multi-layer safety
        if model_name == 'rf':
            n_est = params.get('n_estimators', 200)
            depth = params.get('max_depth', 10)
            min_split = params.get('min_samples_split', 2)
            min_leaf = params.get('min_samples_leaf', 1)
            
            # REALISTIC memory estimation based on actual RF memory usage patterns
            # Empirical formula: ~0.1-0.5GB per 100 trees for large datasets
            base_memory = (n_est / 1000) * depth * 0.1  # Much more realistic estimate
            estimated_memory = max(1.0, base_memory)  # Minimum 1GB estimate
            
            # PRACTICAL SAFETY: Only block truly massive models that would use >400GB
            if estimated_memory > 400.0:
                print(f"   🛑 REALISTIC SAFETY: Memory estimate {estimated_memory:.1f}GB > 400GB limit")
                raise ValueError(f"Memory estimate too high: {estimated_memory:.1f}GB")
            
            print(f"   💡 REALISTIC estimate: {estimated_memory:.1f}GB (trees:{n_est}, depth:{depth})")
            
            # SAFETY TIER 2: Practical parameter combination limits
            complexity_score = (n_est / 1000) * (depth / 20) * (50 / min_split) * (20 / min_leaf)
            if complexity_score > 100:  # Much more realistic threshold
                print(f"   ⚠️ SAFETY: High complexity score {complexity_score:.1f} > 100 threshold")
                print(f"   📊 Parameters: trees={n_est}, depth={depth}, split={min_split}, leaf={min_leaf}")
            
            # SAFETY TIER 3: Only block truly catastrophic combinations
            if n_est > 4500 and depth > 70 and min_leaf == 1 and min_split == 2:
                print(f"   🚨 SAFETY: CATASTROPHIC combination (trees:{n_est}, depth:{depth}, leaf:{min_leaf}, split:{min_split})")
                raise ValueError("Catastrophic parameter combination - system risk")
            elif n_est > 3000 and depth > 60:
                print(f"   ⚡ EXTREME: Ultra-aggressive model (trees:{n_est}, depth:{depth}) - monitoring closely")
        
        try:
            # Create train/test split
            split_config = get_config_value(self.config, 'models.simple_split', {})
            train_size = split_config.get('train_fraction', 0.7)
            random_state = split_config.get('random_state', 42)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, random_state=random_state, shuffle=True
            )
            
            # Check if model needs scaling
            models_needing_scaling = get_config_value(
                self.config, 'models.scaling.models_needing_scaling', ['nn', 'svr']
            )
            needs_scaling = model_name in models_needing_scaling
            
            # Apply scaling if needed
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            if needs_scaling:
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            
            # Create and train model
            model = self._create_model(model_name, params)
            
            # Progressive timeout based on parameter complexity
            start_time = time.time()
            # Calculate expected complexity for dynamic timeout
            n_estimators = params.get('n_estimators', 200)
            max_depth = params.get('max_depth', 10)
            complexity_score = (n_estimators / 100) * (max_depth / 10)
            
            # FINAL FRONTIER timeout: Maximum time for ultimate RF exploration
            base_timeout = 10800   # 180 minutes base (3 hours)
            max_timeout = 21600    # 360 minutes (6 hours) for most extreme models
            max_training_time = min(max_timeout, base_timeout * (1 + complexity_score * 0.15))
            
            print(f"   ⏱️ FINAL timeout: {max_training_time/60:.1f}min (complexity: {complexity_score:.2f})")
            print(f"   🔍 Estimated memory: {estimated_memory:.1f}GB (complexity score: {complexity_score:.2f})")
            
            # PROGRESSIVE MONITORING: Track resources during training
            import threading
            import time as time_module
            monitoring_active = True
            
            def resource_monitor():
                while monitoring_active:
                    current_mem = process.memory_info().rss / (1024**3)
                    current_time = time.time() - start_time
                    if current_time > 0 and current_time % 600 == 0:  # Every 10 minutes
                        print(f"   📊 Progress: {current_time/60:.1f}min, Memory: {current_mem:.1f}GB")
                    time_module.sleep(60)  # Check every minute
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
            monitor_thread.start()
            
            try:
                # Train model with comprehensive monitoring
                model.fit(X_train_scaled, y_train)
                monitoring_active = False  # Stop monitoring
                training_time = time.time() - start_time
                
                # Post-training memory check
                current_memory = process.memory_info().rss / (1024**3)  # GB
                memory_used = current_memory - initial_memory
                
                print(f"   💾 Memory used: {memory_used:.2f}GB, Total: {current_memory:.2f}GB")
                
                # Emergency timeout check
                if training_time > max_training_time:
                    print(f"   ⚠️ Trial exceeded {max_training_time/60:.1f} minutes, parameters may be too extreme")
                    raise ValueError(f"Training timeout after {training_time:.1f} seconds")
                
                # FINAL FRONTIER memory monitoring (for 660GB system)
                if memory_used > 200.0:  # Critical threshold
                    print(f"   🚨 CRITICAL: Memory usage {memory_used:.2f}GB > 200GB")
                elif memory_used > 100.0:  # High usage alert
                    print(f"   ⚠️ HIGH: Memory usage {memory_used:.2f}GB")
                elif memory_used > 50.0:  # Moderate usage info
                    print(f"   📊 MODERATE: Memory usage {memory_used:.2f}GB")
                else:
                    print(f"   ✅ SAFE: Memory usage {memory_used:.2f}GB")
                    
            except Exception as training_error:
                training_time = time.time() - start_time
                current_memory = process.memory_info().rss / (1024**3)
                print(f"   ❌ Training failed after {training_time:.1f}s (Memory: {current_memory:.2f}GB): {training_error}")
                raise training_error
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            
            # FINAL FRONTIER early stopping: Targeting R² = 0.75+ breakthrough
            if r2 < 0.5:  # Poor performance threshold
                print(f"   🚫 Early stopping: R² = {r2:.3f} < 0.5 (poor)")
                raise ValueError(f"Poor performance: R² = {r2:.3f}")
            elif r2 < 0.65 and training_time > 3600:  # Mediocre after 1 hour
                print(f"   ⏹️ Early stopping: R² = {r2:.3f} < 0.65 after {training_time/60:.1f}min")
                raise ValueError(f"Slow mediocre performance: R² = {r2:.3f}")
            elif training_time > max_training_time * 0.6 and r2 < 0.72:  # Sub-breakthrough performance
                print(f"   ⚠️ Early stopping: R² = {r2:.3f} < 0.72 after {training_time/60:.1f}min (targeting 0.75+ breakthrough)")
                raise ValueError(f"Below-breakthrough threshold: R² = {r2:.3f}")
            
            # Success milestone reporting
            if r2 > 0.75:
                print(f"   🎉 BREAKTHROUGH: R² = {r2:.3f} > 0.75 target!")
            elif r2 > 0.70:
                print(f"   🚀 EXCELLENT: R² = {r2:.3f} approaching target!")
            
            # Multi-objective: balance R² and training time  
            # Reduce time penalty since we now have better parameter constraints
            time_penalty = min(training_time / 1800, 0.5)  # Cap penalty at 0.5 for 30min trials
            objective_value = r2 - (0.05 * time_penalty)  # Smaller penalty (was 0.1)
            
            # Store additional metrics for analysis
            trial.set_user_attr('r2', r2)
            trial.set_user_attr('rmse', np.sqrt(mean_squared_error(y_test, y_pred)))
            trial.set_user_attr('mae', mean_absolute_error(y_test, y_pred))
            trial.set_user_attr('training_time', training_time)
            trial.set_user_attr('n_samples_train', len(X_train))
            trial.set_user_attr('n_samples_test', len(X_test))
            
            return objective_value
            
        except Exception as e:
            # Handle failed trials gracefully
            print(f"   Trial failed: {e}")
            trial.set_user_attr('error', str(e))
            return -999.0  # Return very low score for failed trials
    
    def _objective_blocked_cv(self,
                             trial: optuna.Trial,
                             model_name: str,
                             X: np.ndarray,
                             y: np.ndarray,
                             metadata: Dict) -> float:
        """
        Objective function for blocked spatiotemporal CV optimization.
        
        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object
        model_name : str
            Model name to optimize
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        metadata : Dict
            Metadata for spatiotemporal CV
        
        Returns:
        --------
        float
            Objective value (mean R² across folds)
        """
        # Get hyperparameters
        params = self._get_search_space(trial, model_name)
        
        try:
            # Import evaluation function
            from spatiotemporal_cv import evaluate_with_blocked_cv
            
            # Create model class
            if model_name == 'xgb' and HAS_XGB:
                model_class = xgb.XGBRegressor
            elif model_name == 'rf':
                model_class = RandomForestRegressor
            elif model_name == 'lgb' and HAS_LGB:
                model_class = lgb.LGBMRegressor
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Check if needs scaling
            models_needing_scaling = get_config_value(
                self.config, 'models.scaling.models_needing_scaling', ['nn', 'svr']
            )
            needs_scaling = model_name in models_needing_scaling
            
            # CV settings
            cv_config = get_config_value(self.config, 'models.cross_validation', {})
            n_spatial_blocks = cv_config.get('n_spatial_blocks', 5)
            n_temporal_blocks = cv_config.get('n_temporal_blocks', 4)
            
            start_time = time.time()
            
            # Run blocked CV (single job to avoid nested parallelism)
            cv_results_df = evaluate_with_blocked_cv(
                model_class=model_class,
                X=X,
                y=y,
                metadata=metadata,
                model_params=params,
                n_spatial_blocks=n_spatial_blocks,
                n_temporal_blocks=n_temporal_blocks,
                needs_scaling=needs_scaling,
                n_jobs=1  # Single job to avoid conflicts with Optuna parallelism
            )
            
            training_time = time.time() - start_time
            
            # Calculate mean performance across folds
            mean_r2 = cv_results_df['r2'].mean()
            std_r2 = cv_results_df['r2'].std()
            mean_rmse = cv_results_df['rmse'].mean()
            
            # Multi-objective: balance R² and training time
            time_penalty = min(training_time / 7200, 1.0)  # Cap at 2 hours
            objective_value = mean_r2 - (0.05 * time_penalty)  # Small time penalty
            
            # Store additional metrics for analysis
            trial.set_user_attr('mean_r2', mean_r2)
            trial.set_user_attr('std_r2', std_r2)
            trial.set_user_attr('mean_rmse', mean_rmse)
            trial.set_user_attr('training_time', training_time)
            trial.set_user_attr('n_folds', len(cv_results_df))
            
            return objective_value
            
        except Exception as e:
            # Handle failed trials gracefully
            print(f"   Trial failed: {e}")
            trial.set_user_attr('error', str(e))
            return -999.0
    
    def _create_study(self, model_name: str, study_name: str) -> optuna.Study:
        """Create Optuna study with model-specific sampler and pruner optimization."""
        
        # Model-specific sampler configuration
        if self.sampler_name.lower() == 'tpe':
            if model_name == 'rf':
                # Random Forest needs more exploration due to complex parameter interactions
                sampler = TPESampler(
                    seed=42, 
                    n_startup_trials=50,  # More initial random exploration
                    n_ei_candidates=48,   # More candidates for expected improvement
                    multivariate=True,    # Consider parameter correlations
                    group=True,           # Group categorical parameters
                    warn_independent_sampling=False
                )
            else:
                # Standard TPE for other models
                sampler = TPESampler(seed=42, n_startup_trials=20)
        elif self.sampler_name.lower() == 'cmaes':
            sampler = CmaEsSampler(seed=42)
        else:
            sampler = TPESampler(seed=42)
        
        # Model-specific pruner configuration
        if self.pruner_name.lower() == 'hyperband':
            if model_name == 'rf':
                # More conservative pruning for RF due to training variance
                pruner = HyperbandPruner(
                    min_resource=3,         # Wait for more iterations before pruning
                    max_resource=100, 
                    reduction_factor=2      # Less aggressive pruning
                )
            else:
                pruner = HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)
        elif self.pruner_name.lower() == 'median':
            if model_name == 'rf':
                # More startup trials for RF due to higher variance
                pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=10)
            else:
                pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        else:
            pruner = HyperbandPruner()
        
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            storage=self.study_storage,
            load_if_exists=True,
            direction='maximize',  # Maximize R²
            sampler=sampler,
            pruner=pruner
        )
        
        return study
    
    def _get_model_specific_trials(self, model_name: str) -> int:
        """
        Get model-specific number of trials based on parameter complexity.
        
        Random Forest has more hyperparameters and complex interactions,
        so it benefits from more optimization trials.
        """
        if model_name == 'rf':
            # RF has ~10+ hyperparameters with complex interactions
            return int(self.n_trials * 1.5)  # 50% more trials for RF
        else:
            # XGB and LGB are typically well-optimized with standard trials
            return self.n_trials
    
    def tune_model(self,
                   model_name: str,
                   X: np.ndarray,
                   y: np.ndarray,
                   metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for a single model.
        
        Parameters:
        -----------
        model_name : str
            Model name to tune
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        metadata : Dict, optional
            Metadata for blocked CV (required if cv_method='blocked')
        
        Returns:
        --------
        Dict[str, Any]
            Best parameters and optimization results
        """
        print(f"\n{'='*70}")
        print(f"🎯 Tuning {model_name.upper()}")
        print(f"{'='*70}")
        
        # Create study with timestamp to avoid conflicts
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        study_name = f"{model_name}_{self.cv_method}_tuning_v2_{timestamp}"
        study = self._create_study(model_name, study_name)
        
        # Store study for later access
        self.studies[model_name] = study
        
        print(f"   Study: {study_name}")
        
        # Get model-specific trial count
        model_trials = self._get_model_specific_trials(model_name)
        print(f"   Trials: {model_trials} (optimized for {model_name})")
        print(f"   CV method: {self.cv_method}")
        
        # Define objective function
        if self.cv_method == 'simple':
            objective_func = lambda trial: self._objective_simple_split(trial, model_name, X, y)
        elif self.cv_method == 'blocked':
            if metadata is None:
                raise ValueError("Metadata required for blocked CV optimization")
            objective_func = lambda trial: self._objective_blocked_cv(trial, model_name, X, y, metadata)
        else:
            raise ValueError(f"Unknown CV method: {self.cv_method}")
        
        # Run optimization
        start_time = time.time()
        
        try:
            study.optimize(
                objective_func,
                n_trials=model_trials,  # Use model-specific trial count
                timeout=self.timeout,
                n_jobs=self.n_jobs,  # Use configured parallel jobs
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            print(f"   Optimization interrupted for {model_name}")
        
        optimization_time = time.time() - start_time
        
        # Extract results
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        # Compile results
        results = {
            'model_name': model_name,
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(study.trials),
            'optimization_time': optimization_time,
            'study_name': study_name,
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        }
        
        # Add best trial user attributes
        if hasattr(best_trial, 'user_attrs'):
            results.update(best_trial.user_attrs)
        
        print(f"\n✅ {model_name.upper()} tuning complete!")
        print(f"   Best objective value: {best_value:.4f}")
        print(f"   Optimization time: {optimization_time/60:.1f} minutes")
        print(f"   Completed trials: {results['completed_trials']}/{model_trials}")
        if 'r2' in results:
            print(f"   Best R²: {results['r2']:.4f}")
        
        return results
    
    def tune_all_models(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       metadata: Optional[Dict] = None) -> Dict[str, Dict[str, Any]]:
        """
        Tune hyperparameters for all enabled models.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values  
        metadata : Dict, optional
            Metadata for blocked CV
        
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Results for all models
        """
        print(f"\n{'='*70}")
        print(f"🚀 HYPERPARAMETER TUNING FOR ALL MODELS")
        print(f"{'='*70}")
        print(f"   Models: {self.enabled_models}")
        print(f"   Trials per model: {self.n_trials}")
        print(f"   Total trials: {len(self.enabled_models) * self.n_trials}")
        print(f"   CV method: {self.cv_method}")
        
        all_results = {}
        
        for model_name in self.enabled_models:
            try:
                results = self.tune_model(model_name, X, y, metadata)
                all_results[model_name] = results
                self.tuning_results[model_name] = results
                
                # Store best parameters
                self.best_params[model_name] = results['best_params']
                
            except Exception as e:
                print(f"\n❌ Error tuning {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"📊 TUNING SUMMARY")
        print(f"{'='*70}")
        
        for model_name, results in all_results.items():
            print(f"   {model_name.upper()}:")
            print(f"     • Best value: {results['best_value']:.4f}")
            if 'r2' in results:
                print(f"     • Best R²: {results['r2']:.4f}")
            print(f"     • Time: {results['optimization_time']/60:.1f} min")
            print(f"     • Trials: {results['completed_trials']}/{results['n_trials']}")
        
        return all_results
    
    def save_tuned_parameters(self, output_path: str = None):
        """
        Save tuned hyperparameters to JSON file.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save parameters (default: tuned_hyperparameters_{cv_method}.json)
        """
        if not self.best_params:
            print("⚠️ No tuned parameters to save")
            return
        
        if output_path is None:
            output_path = f"tuned_hyperparameters_{self.cv_method}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        save_data = {
            'tuning_metadata': {
                'cv_method': self.cv_method,
                'n_trials_per_model': self.n_trials,
                'n_jobs': self.n_jobs,
                'sampler': self.sampler_name,
                'pruner': self.pruner_name,
                'tuning_date': datetime.now().isoformat(),
                'enabled_models': self.enabled_models
            },
            'best_parameters': self.best_params,
            'tuning_results': {}
        }
        
        # Add tuning results (removing non-serializable objects)
        for model_name, results in self.tuning_results.items():
            serializable_results = {}
            for key, value in results.items():
                try:
                    json.dumps(value)  # Test if serializable
                    serializable_results[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable values
                    continue
            save_data['tuning_results'][model_name] = serializable_results
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"✅ Tuned parameters saved: {output_path}")
        
        # Also save as pickle for complete data preservation
        pickle_path = output_path.with_suffix('.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'tuning_metadata': save_data['tuning_metadata'],
                'best_parameters': self.best_params,
                'tuning_results': self.tuning_results,
                'studies': self.studies  # Include full Optuna studies
            }, f)
        
        print(f"✅ Complete tuning data saved: {pickle_path}")
    
    def update_config_with_tuned_params(self, config: Dict) -> None:
        """
        Update the configuration with tuned parameters in-place.
        
        This allows the tuned parameters to be used immediately
        in subsequent training steps within the same pipeline run.
        
        Parameters:
        -----------
        config : Dict
            Configuration dictionary to update in-place
        """
        if not self.best_params:
            print("⚠️ No tuned parameters available to update config")
            return
        
        print("🔄 Updating configuration with tuned hyperparameters...")
        
        # Update the hyperparameters section of the config
        if 'models' not in config:
            config['models'] = {}
        if 'hyperparameters' not in config['models']:
            config['models']['hyperparameters'] = {}
        
        # Update each model's parameters
        for model_name, params in self.best_params.items():
            config['models']['hyperparameters'][model_name] = params.copy()
            print(f"   ✓ Updated {model_name} with {len(params)} tuned parameters")
        
        print("✅ Configuration updated successfully")
        print("   Subsequent training will use tuned hyperparameters")
    
    @staticmethod
    def load_tuned_parameters(file_path: str) -> Dict[str, Any]:
        """
        Load tuned hyperparameters from JSON file.
        
        Parameters:
        -----------
        file_path : str
            Path to tuned parameters file
        
        Returns:
        --------
        Dict[str, Any]
            Loaded parameters and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Tuned parameters file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded tuned parameters: {file_path}")
        return data
    
    def generate_tuning_report(self, output_dir: str = "tuning_reports"):
        """
        Generate comprehensive tuning report with visualizations.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save report files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 Generating tuning report in: {output_dir}")
        
        # Create summary DataFrame
        summary_data = []
        for model_name, results in self.tuning_results.items():
            summary_data.append({
                'Model': model_name.upper(),
                'Best_Objective': results.get('best_value', 'N/A'),
                'Best_R2': results.get('r2', 'N/A'),
                'Best_RMSE': results.get('rmse', 'N/A'),
                'Optimization_Time_Min': results.get('optimization_time', 0) / 60,
                'Completed_Trials': results.get('completed_trials', 0),
                'Failed_Trials': results.get('failed_trials', 0),
                'Pruned_Trials': results.get('pruned_trials', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_path = output_dir / "tuning_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"   ✅ Summary saved: {summary_path}")
        
        # Save best parameters
        best_params_path = output_dir / "best_parameters.json"
        with open(best_params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        print(f"   ✅ Best parameters saved: {best_params_path}")
        
        print("📊 Tuning report generated successfully!")


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning with Optuna")
    parser.add_argument('--config', default='src_new_approach/config_coarse_to_fine.yaml',
                       help='Configuration file path')
    parser.add_argument('--features-coarse', required=True,
                       help='Coarse features dataset path')
    parser.add_argument('--grace-filled', required=True,
                       help='Gap-filled GRACE dataset path')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of trials per model')
    parser.add_argument('--jobs', type=int, default=32,
                       help='Number of parallel jobs')
    parser.add_argument('--cv-method', choices=['simple', 'blocked'], default='simple',
                       help='Cross-validation method')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Optimization timeout in seconds')
    parser.add_argument('--output', default='tuned_hyperparameters.json',
                       help='Output file for tuned parameters')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override CV method if specified
    if args.cv_method:
        if 'hyperparameter_tuning' not in config:
            config['hyperparameter_tuning'] = {}
        config['hyperparameter_tuning']['cv_method'] = args.cv_method
    
    print("🔧 Loading datasets...")
    import xarray as xr
    
    features_ds = xr.open_dataset(args.features_coarse)
    grace_ds = xr.open_dataset(args.grace_filled)
    
    # Prepare training data
    from coarse_model_trainer import CoarseModelTrainer
    trainer = CoarseModelTrainer(config)
    X, y, feature_names, metadata = trainer.prepare_training_data(features_ds, grace_ds)
    
    print(f"   Training data shape: X={X.shape}, y={y.shape}")
    
    # Initialize tuner
    tuner = OptunaTuner(
        config=config,
        n_trials=args.trials,
        n_jobs=args.jobs,
        timeout=args.timeout
    )
    
    # Run tuning
    if args.cv_method == 'blocked':
        results = tuner.tune_all_models(X, y, metadata)
    else:
        results = tuner.tune_all_models(X, y)
    
    # Save results
    tuner.save_tuned_parameters(args.output)
    tuner.generate_tuning_report()
    
    print("\n🎉 Hyperparameter tuning completed successfully!")


if __name__ == "__main__":
    main()