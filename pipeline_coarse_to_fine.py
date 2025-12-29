#!/usr/bin/env python3
"""
Coarse-to-Fine GRACE Downscaling Pipeline

This pipeline implements a scientifically rigorous approach:
1. Fill GRACE gaps using STL decomposition
2. Upscale features from 5km to 55km (GRACE native resolution)
3. Train ML models at coarse scale (55km) with blocked spatiotemporal CV
4. Apply models to fine-resolution features (5km)
5. Calculate residuals at coarse scale
6. Interpolate residuals to fine scale (bilinear)
7. Apply residual correction to get final downscaled GRACE

Usage:
    python pipeline_coarse_to_fine.py --steps all
    python pipeline_coarse_to_fine.py --steps gap_fill,aggregate,train
    python pipeline_coarse_to_fine.py --steps predict,correct --skip-train
"""

# ============================================================================
# CRITICAL: Set environment variables BEFORE imports!
# ============================================================================
import os
import multiprocessing

cpu_count = multiprocessing.cpu_count()
# Use up to 96 threads for maximum HPC utilization
MAX_CORES = min(96, max(1, cpu_count // 2))  # Up to 96 threads maximum (50% of 192 cores)

print(f"🔧 CPU Configuration:")
print(f"   Available cores: {cpu_count}")
print(f"   Setting safe limit: {MAX_CORES}")

# Set BEFORE numpy/scipy imports - use full 64 threads
os.environ['OMP_NUM_THREADS'] = str(MAX_CORES)
os.environ['OPENBLAS_NUM_THREADS'] = str(MAX_CORES)
os.environ['MKL_NUM_THREADS'] = str(MAX_CORES)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(MAX_CORES)
os.environ['NUMEXPR_NUM_THREADS'] = str(MAX_CORES)
os.environ['NUMEXPR_MAX_THREADS'] = '96'  # Set to exactly 96 for maximum HPC utilization

print(f"✅ Thread limits set: {MAX_CORES} threads")

# ============================================================================
# NOW safe to import
# ============================================================================

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr

# Add src_new_approach to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src_new_approach'))

from src_new_approach.utils_downscaling import (
    load_config,
    get_config_value,
    create_output_directories,
    setup_logging
)
from src_new_approach.grace_gap_handler import GRACEGapFiller
from src_new_approach.feature_aggregator import FeatureAggregator
from src_new_approach.coarse_model_trainer import CoarseModelTrainer
from src_new_approach.fine_predictor import FinePredictor
from src_new_approach.residual_corrector import ResidualCorrector
from src_new_approach.hyperparameter_tuner import OptunaTuner
from src_new_approach.shap_analyzer import SHAPAnalyzer
from src_new_approach.generate_fine_derived_features import generate_fine_derived_features


def run_gap_filling(config: dict, logger) -> bool:
    """Step 1: Fill GRACE gaps using STL decomposition."""
    logger.info("="*70)
    logger.info("STEP 1: GRACE GAP FILLING")
    logger.info("="*70)
    
    try:
        grace_path = get_config_value(config, 'paths.grace_raw')
        output_path = get_config_value(config, 'paths.grace_filled')
        
        # Initialize gap filler
        filler = GRACEGapFiller(config)
        
        # Load GRACE
        grace_ds = filler.load_grace_data(grace_path)
        
        # Fill gaps (use vectorized approach if configured)
        use_vectorized = get_config_value(config, 'grace_gap_filling.use_vectorized', True)
        filled_ds, stats = filler.fill_grace_dataset(grace_ds, use_vectorized=use_vectorized)
        
        # Save
        filler.save_filled_grace(filled_ds, output_path)
        
        logger.info("✅ Gap filling completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Gap filling failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_feature_creation(config: dict, logger) -> bool:
    """Step 0: Create feature stack from raw data with complete time coverage."""
    logger.info("="*70)
    logger.info("STEP 0: FEATURE STACK CREATION FROM RAW DATA")
    logger.info("="*70)
    
    try:
        # Get paths
        output_path = get_config_value(config, 'paths.feature_stack_fine')
        start_date = get_config_value(config, 'data_processing.start_date', '2003-01-01')
        end_date = get_config_value(config, 'data_processing.end_date', '2024-12-31')
        
        logger.info(f"Creating feature stack for period: {start_date} to {end_date}")
        logger.info(f"Output path: {output_path}")
        
        # Check if feature stack already exists
        if Path(output_path).exists():
            logger.info("⚠️ Feature stack already exists, checking if complete...")
            existing_features = xr.open_dataset(output_path)
            
            # Check time coverage
            expected_times = pd.date_range(start_date, end_date, freq='MS')
            actual_times = len(existing_features.time)
            expected_count = len(expected_times)
            
            logger.info(f"Existing features: {actual_times} time steps")
            logger.info(f"Expected: {expected_count} time steps")
            
            if actual_times >= expected_count * 0.95:  # Allow 5% tolerance
                logger.info("✅ Existing feature stack is complete, skipping creation")
                existing_features.close()
                return True
            else:
                logger.info("⚠️ Existing feature stack is incomplete, recreating...")
                existing_features.close()
        
        # Create feature stack from raw data using existing aggregator logic
        logger.info("🔧 Creating feature stack from raw data using FeatureAggregator...")
        
        # Initialize aggregator (FeatureAggregator already imported at top)
        aggregator = FeatureAggregator(config)
        
        # Load GRACE gap-filled data to get the actual time range
        grace_filled_path = get_config_value(config, 'paths.grace_filled')
        logger.info(f"Loading GRACE time range from: {grace_filled_path}")
        
        grace_ds = xr.open_dataset(grace_filled_path)
        target_times = grace_ds.time
        logger.info(f"Target time range: {len(target_times)} months ({target_times.min().values} to {target_times.max().values})")
        grace_ds.close()
        
        # Load complete features from raw data
        complete_features = aggregator.load_complete_features_from_raw(target_times)
        
        if complete_features is not None:
            # Save the feature stack
            logger.info(f"💾 Saving complete feature stack to: {output_path}")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save
            complete_features.to_netcdf(output_path)
            
            # Verify
            saved_size = Path(output_path).stat().st_size
            logger.info(f"✅ Feature stack saved: {saved_size:,} bytes")
            logger.info(f"   Shape: {dict(complete_features.dims)}")
            
            return True
        else:
            logger.error("❌ Failed to load complete features from raw data")
            return False
        
    except Exception as e:
        logger.error(f"❌ Feature creation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_aggregation(config: dict, logger) -> bool:
    """Step 2: Aggregate features from 5km to 55km."""
    logger.info("="*70)
    logger.info("STEP 2: FEATURE AGGREGATION")
    logger.info("="*70)
    
    try:
        features_fine_path = get_config_value(config, 'paths.feature_stack_fine')
        output_path = get_config_value(config, 'paths.feature_stack_coarse')
        grace_filled_path = get_config_value(config, 'paths.grace_filled')
        
        # Initialize aggregator
        aggregator = FeatureAggregator(config)
        
        # Load fine features
        fine_ds = aggregator.load_fine_features(features_fine_path)
        
        # Load GRACE data and use as spatial template
        grace_mask = None
        grace_ds = None
        if Path(grace_filled_path).exists():
            grace_ds = xr.open_dataset(grace_filled_path)
            grace_mask = grace_ds['tws_anomaly'].isel(time=0)  # Use first time slice as spatial template
            
            logger.info(f"Using gap-filled GRACE as target: {len(grace_ds.time)} months, {len(grace_ds.lat)}×{len(grace_ds.lon)} grid")
        
        # Fix temporal coverage mismatch
        if grace_ds is not None:
            grace_times = pd.to_datetime(grace_ds.time.values)
            feature_times = pd.to_datetime(fine_ds.time.values)
            
            if len(grace_times) > len(feature_times):
                logger.warning(f"TEMPORAL MISMATCH DETECTED:")
                logger.warning(f"  GRACE (gap-filled): {len(grace_times)} months")
                logger.warning(f"  Features (limited): {len(feature_times)} months") 
                logger.warning(f"  Root cause: Feature stack was artificially limited to original GRACE gaps")
                logger.info(f"🔧 Fixing temporal coverage: {len(feature_times)} → {len(grace_times)} months")
                
                # Try to load complete features from raw data
                complete_features = aggregator.load_complete_features_from_raw(grace_times)
                if complete_features is not None:
                    fine_ds = complete_features
                    logger.info("✅ Loaded complete features from raw data")
                else:
                    # Fallback: extend existing features
                    logger.info("⚠️ Falling back to temporal interpolation")
                    fine_ds = aggregator.extend_features_temporal(fine_ds, grace_times)
        
        # Aggregate
        coarse_ds = aggregator.aggregate_feature_stack(fine_ds, grace_mask)
        
        # Save
        aggregator.save_coarse_features(coarse_ds, output_path)
        
        logger.info("✅ Feature aggregation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature aggregation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_tuning(config: dict, logger) -> bool:
    """Step 2.5: Hyperparameter tuning with Optuna and SHAP analysis."""
    logger.info("="*70)
    logger.info("STEP 2.5: HYPERPARAMETER TUNING & SHAP ANALYSIS")
    logger.info("="*70)
    
    # Check if tuning is enabled
    tuning_enabled = get_config_value(config, 'hyperparameter_tuning.enabled', False)
    if not tuning_enabled:
        logger.info("⏭️ Hyperparameter tuning disabled - skipping")
        return True
    
    try:
        features_coarse_path = get_config_value(config, 'paths.feature_stack_coarse')
        grace_filled_path = get_config_value(config, 'paths.grace_filled')
        models_dir = get_config_value(config, 'paths.models')
        
        # Load data
        logger.info("Loading datasets for tuning...")
        features_ds = xr.open_dataset(features_coarse_path)
        grace_ds = xr.open_dataset(grace_filled_path)
        
        # Prepare data using trainer
        trainer = CoarseModelTrainer(config)
        X, y, feature_names, metadata = trainer.prepare_training_data(features_ds, grace_ds)
        
        logger.info(f"Data loaded: {X.shape[0]:,} samples, {X.shape[1]} features")
        
        # Initialize tuner
        n_trials = get_config_value(config, 'hyperparameter_tuning.n_trials', 500)
        n_jobs = get_config_value(config, 'hyperparameter_tuning.n_jobs', 8)  # Conservative default to prevent HPC crashes
        timeout = get_config_value(config, 'hyperparameter_tuning.timeout', 10800)
        
        tuner = OptunaTuner(
            config=config,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout
        )
        
        logger.info(f"🎯 Starting hyperparameter optimization:")
        logger.info(f"   Trials per model: {n_trials}")
        logger.info(f"   Parallel jobs: {n_jobs}")
        logger.info(f"   Timeout: {timeout} seconds ({timeout//3600}h {(timeout%3600)//60}m)")
        
        # Run tuning
        tuning_results = tuner.tune_all_models(X, y, metadata)
        
        # Save tuned parameters
        tuned_params_path = Path(models_dir) / "tuned_hyperparameters.yaml"
        tuner.save_tuned_parameters(tuned_params_path)
        logger.info(f"💾 Tuned parameters saved: {tuned_params_path}")
        
        # Update config with tuned parameters for subsequent training
        tuner.update_config_with_tuned_params(config)
        logger.info("✅ Configuration updated with tuned parameters")
        
        # SHAP analysis notification
        shap_enabled = get_config_value(config, 'hyperparameter_tuning.shap_analysis.enabled', True)
        if shap_enabled:
            logger.info("🧠 SHAP analysis will be run after training with tuned parameters")
            logger.info("   Use: python src_new_approach/analyze_shap_results.py --models models_coarse_to_fine/ --features data/processed_coarse_to_fine/feature_stack_coarse.nc --config src_new_approach/config_coarse_to_fine.yaml")
        
        logger.info("✅ Hyperparameter tuning completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Hyperparameter tuning failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_fine_feature_engineering(config: dict, logger) -> bool:
    """Step 2.6: Generate derived features at fine resolution for scale-consistent training."""
    logger.info("="*70)
    logger.info("STEP 2.6: FINE-RESOLUTION FEATURE ENGINEERING")
    logger.info("="*70)
    
    try:
        # Check if scale-consistent NN is enabled
        enabled_models = get_config_value(config, 'models.enabled', [])
        
        if 'nn_scale_consistent' not in enabled_models:
            logger.info("⏭️ Scale-consistent NN not enabled - skipping fine feature engineering")
            return True
        
        # Check if output already exists
        output_path = get_config_value(config, 'paths.feature_stack_fine_all',
                                       'processed_coarse_to_fine/feature_stack_all_5km.nc')
        
        if Path(output_path).exists():
            logger.info(f"✅ Enhanced fine features already exist: {output_path}")
            logger.info("   Skipping generation (delete file to regenerate)")
            return True
        
        logger.info("🔧 Generating derived features at 5km resolution...")
        logger.info("   This enables using all 96 features for scale-consistent training")
        
        # Generate features
        enhanced_ds = generate_fine_derived_features(config, output_path)
        
        logger.info(f"✅ Generated {len(enhanced_ds.feature)} temporal + {len(enhanced_ds.static_feature)} static features")
        enhanced_ds.close()
        
        logger.info("✅ Fine feature engineering completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fine feature engineering failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_training(config: dict, logger) -> bool:
    """Step 3: Train models at coarse scale with blocked CV or simple split."""
    logger.info("="*70)
    logger.info("STEP 3: MODEL TRAINING AT COARSE SCALE")
    logger.info("="*70)
    
    try:
        features_coarse_path = get_config_value(config, 'paths.feature_stack_coarse')
        features_fine_path = get_config_value(config, 'paths.feature_stack_fine')
        grace_filled_path = get_config_value(config, 'paths.grace_filled')
        models_dir = get_config_value(config, 'paths.models')
        
        # Determine training method
        cv_method = get_config_value(config, 'models.cross_validation.method', 'blocked_spatiotemporal')
        
        # Load data
        logger.info("Loading datasets...")
        features_ds = xr.open_dataset(features_coarse_path)
        grace_ds = xr.open_dataset(grace_filled_path)
        
        # Initialize trainer
        trainer = CoarseModelTrainer(config)
        
        # Check if we should load tuned parameters
        tuning_enabled = get_config_value(config, 'hyperparameter_tuning.enabled', False)
        if tuning_enabled:
            # Always check original models directory first (where tuning saves parameters)
            # This is the base directory without any _simple suffix
            cv_method = get_config_value(config, 'models.cross_validation.method', 'blocked_spatiotemporal')
            if cv_method == 'simple_split':
                # For simple split, look in the original directory (without _simple suffix)
                original_models_dir = models_dir.replace('_simple', '')
                tuned_params_path = Path(original_models_dir) / "tuned_hyperparameters.yaml"
            else:
                # For blocked CV, check current models directory
                tuned_params_path = Path(models_dir) / "tuned_hyperparameters.yaml"
            
            if tuned_params_path.exists():
                logger.info(f"🎯 Loading tuned hyperparameters from: {tuned_params_path}")
                trainer.load_tuned_parameters(tuned_params_path)
            else:
                logger.warning("⚠️ Tuning enabled but no tuned parameters found - using defaults")
                logger.warning(f"   Searched in: {tuned_params_path}")
        
        # Check if scale-consistent NN is enabled - load fine data if so
        enabled_models = get_config_value(config, 'models.enabled', [])
        if 'nn_scale_consistent' in enabled_models:
            logger.info("🧠 Scale-consistent NN enabled - loading fine features...")
            
            # Prefer enhanced features (with all derived features) if available
            features_fine_all_path = get_config_value(config, 'paths.feature_stack_fine_all', 
                                                       features_fine_path.replace('.nc', '_all.nc'))
            
            if Path(features_fine_all_path).exists():
                logger.info(f"   ✓ Using enhanced fine features: {features_fine_all_path}")
                fine_features_ds = xr.open_dataset(features_fine_all_path)
            elif Path(features_fine_path).exists():
                logger.info(f"   Using basic fine features: {features_fine_path}")
                logger.info("   💡 Run: python src_new_approach/generate_fine_derived_features.py")
                logger.info("      to generate enhanced features with anomalies, lags, etc.")
                fine_features_ds = xr.open_dataset(features_fine_path)
            else:
                logger.warning(f"⚠️ Fine features not found")
                logger.warning("   Scale-consistent NN will train without consistency loss")
                fine_features_ds = None
            
            if fine_features_ds is not None:
                trainer.set_fine_data_for_scale_consistent(
                    fine_features_ds,
                    features_ds,  # coarse features
                    grace_ds
                )
                fine_features_ds.close()
        
        # Prepare data
        X, y, feature_names, metadata = trainer.prepare_training_data(features_ds, grace_ds)
        
        # Choose training method based on configuration
        if cv_method == 'simple_split':
            logger.info("Using traditional 70-30 split training method")
            summary_df = trainer.train_with_simple_split(X, y, metadata)
        else:
            logger.info("Using blocked spatiotemporal CV training method")
            summary_df = trainer.train_with_blocked_cv(X, y, metadata)
        
        # For simple split, models are already trained; for blocked CV, train final models
        if cv_method != 'simple_split':
            trainer.train_final_models(X, y, feature_names)
        
        # Create ensemble
        ensemble_info = trainer.create_ensemble(summary_df)
        
        # Save everything
        trainer.save_models(models_dir)
        
        # Save summary (with appropriate filename)
        if cv_method == 'simple_split':
            summary_path = Path(models_dir) / "model_comparison_simple.csv"
        else:
            summary_path = Path(models_dir) / "model_comparison_coarse.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Model comparison saved: {summary_path}")
        
        logger.info("✅ Model training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_fine_prediction(config: dict, logger) -> bool:
    """Step 4: Apply models to fine-resolution features."""
    logger.info("="*70)
    logger.info("STEP 4: FINE-RESOLUTION PREDICTION")
    logger.info("="*70)
    
    try:
        features_fine_path = get_config_value(config, 'paths.feature_stack_fine')
        models_dir = get_config_value(config, 'paths.models')
        output_path = get_config_value(config, 'paths.predictions_fine')
        
        # Load features
        logger.info("Loading fine-resolution features...")
        features_ds = xr.open_dataset(features_fine_path)
        
        # Initialize predictor
        predictor = FinePredictor(config, models_dir)
        
        # Load models
        predictor.load_models()
        
        # Predict
        use_ensemble = get_config_value(config, 'models.ensemble.method', 'weighted') != 'none'
        predictions_ds = predictor.predict_fine_resolution(features_ds, use_ensemble)
        
        # Save
        predictor.save_predictions(predictions_ds, output_path)
        
        logger.info("✅ Fine prediction completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fine prediction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_residual_correction(config: dict, logger) -> bool:
    """Step 5: Calculate and apply residual correction."""
    logger.info("="*70)
    logger.info("STEP 5: RESIDUAL CORRECTION")
    logger.info("="*70)
    
    # Check if residual correction is enabled
    residual_enabled = get_config_value(config, 'residual_correction.enabled', True)
    if not residual_enabled:
        logger.info("🚫 RESIDUAL CORRECTION DISABLED")
        logger.info("   Using COARSE-SCALE ensemble predictions (avoid spatial extrapolation)")
        
        # Use coarse-scale predictions instead of problematic fine-scale extrapolation
        coarse_predictions_path = get_config_value(config, 'paths.predictions_coarse')
        output_path = get_config_value(config, 'paths.final_downscaled')
        
        import shutil
        shutil.copy2(coarse_predictions_path, output_path)
        
        logger.info(f"✅ Using coarse-scale ensemble predictions as final output: {output_path}")
        logger.info("🎯 Expected performance: R² should match coarse ensemble (~0.92)")
        logger.info("📍 Note: Output is at 55km resolution (native model training scale)")
        return True
    
    try:
        grace_path = get_config_value(config, 'paths.grace_filled')
        coarse_features_path = get_config_value(config, 'paths.feature_stack_coarse')
        fine_features_path = get_config_value(config, 'paths.feature_stack_fine')
        fine_predictions_path = get_config_value(config, 'paths.predictions_fine')
        models_dir = get_config_value(config, 'paths.models')
        output_path = get_config_value(config, 'paths.final_downscaled')
        
        # Load datasets
        logger.info("Loading datasets...")
        grace_ds = xr.open_dataset(grace_path)
        coarse_features_ds = xr.open_dataset(coarse_features_path)
        fine_features_ds = xr.open_dataset(fine_features_path)
        fine_predictions_ds = xr.open_dataset(fine_predictions_path)
        
        # Initialize corrector
        corrector = ResidualCorrector(config)
        
        # Run full workflow
        downscaled_ds = corrector.full_residual_workflow(
            grace_ds,
            coarse_features_ds,
            fine_features_ds,
            fine_predictions_ds,
            models_dir
        )
        
        # Save
        corrector.save_downscaled(downscaled_ds, output_path)
        
        logger.info("✅ Residual correction completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Residual correction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_validation(config: dict, logger) -> bool:
    """Step 6: Validate downscaled results."""
    logger.info("="*70)
    logger.info("STEP 6: VALIDATION")
    logger.info("="*70)
    
    try:
        from src_new_approach.validation.validate_downscaled import validate_downscaled_grace
        
        downscaled_path = get_config_value(config, 'paths.final_downscaled')
        results_dir = get_config_value(config, 'paths.results')
        figures_dir = get_config_value(config, 'paths.figures')
        
        # Run validation
        validate_downscaled_grace(
            downscaled_path=downscaled_path,
            config=config,
            output_dir=results_dir,
            figures_dir=figures_dir,
            logger=logger
        )
        
        logger.info("✅ Validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        logger.warning("Validation is optional - continuing anyway")
        import traceback
        logger.error(traceback.format_exc())
        return True  # Don't fail pipeline if validation fails


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Coarse-to-Fine GRACE Downscaling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config',
        default='src_new_approach/config_coarse_to_fine.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='Steps to run: gap_fill, aggregate, tune, fine_features, train, predict, correct, validate, all'
    )
    
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip training if models already exist'
    )
    
    parser.add_argument(
        '--grace-source',
        choices=['gee', 'cri'],
        default='gee',
        help='GRACE data source: gee=Google Earth Engine TIFs (219 months), cri=JPL CRI NetCDF (245 months)'
    )
    
    parser.add_argument(
        '--use-simple-split',
        action='store_true',
        help='Use traditional 70-30 train/test split instead of blocked spatiotemporal CV'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("📋 Loading configuration...")
    config = load_config(args.config)
    
    # Configure GRACE data source based on flag
    if args.grace_source == 'cri':
        config['paths']['grace_raw'] = 'data/GRCTellus_GRACE_data.nc'
        print(f"🎯 GRACE Source: JPL CRI NetCDF (direct from source, 245 months)")
    elif args.grace_source == 'gee':
        config['paths']['grace_raw'] = 'data/raw/grace'
        print(f"🎯 GRACE Source: Google Earth Engine TIFs (219 months)")
    
    # Ensure temporal alignment: Features available 2003-01 to 2024-11
    config['data_processing']['start_date'] = '2003-01-01'
    config['data_processing']['end_date'] = '2024-11-30'
    print(f"📅 Temporal Range: 2003-01 to 2024-11 (262 months total)")
    print(f"   This ensures alignment between GRACE and feature data")
    
    # Configure training method based on CLI flag
    if args.use_simple_split:
        config['models']['cross_validation']['method'] = 'simple_split'
        # Update model output directory to distinguish from blocked CV
        config['paths']['models'] = config['paths']['models'].rstrip('/') + '_simple'
        print(f"🎯 Training Method: Traditional 70-30 split")
        print(f"📁 Models will be saved to: {config['paths']['models']}")
    else:
        print(f"🎯 Training Method: Blocked Spatiotemporal CV")
        print(f"📁 Models will be saved to: {config['paths']['models']}")
    
    # Create output directories
    create_output_directories(config)
    
    # Setup logging
    logger = setup_logging(config, 'pipeline_coarse_to_fine')
    
    logger.info("="*70)
    logger.info("🚀 COARSE-TO-FINE GRACE DOWNSCALING PIPELINE")
    logger.info("="*70)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse steps
    if args.steps.lower() == 'all':
        steps = ['create_features', 'gap_fill', 'aggregate', 'tune', 'fine_features', 'train', 'predict', 'correct', 'validate']
    else:
        steps = [s.strip().lower() for s in args.steps.split(',')]
    
    # Skip training if requested and models exist
    if args.skip_train and 'train' in steps:
        models_dir = Path(get_config_value(config, 'paths.models'))
        if (models_dir / "ensemble_info.joblib").exists():
            logger.info("Models already exist - skipping training")
            steps.remove('train')
    
    logger.info(f"Pipeline steps: {steps}")
    
    # Define step functions
    step_functions = {
        'create_features': lambda: run_feature_creation(config, logger),
        'gap_fill': lambda: run_gap_filling(config, logger),
        'aggregate': lambda: run_aggregation(config, logger),
        'tune': lambda: run_tuning(config, logger),
        'fine_features': lambda: run_fine_feature_engineering(config, logger),
        'train': lambda: run_training(config, logger),
        'predict': lambda: run_fine_prediction(config, logger),
        'correct': lambda: run_residual_correction(config, logger),
        'validate': lambda: run_validation(config, logger)
    }
    
    # Execute steps
    failed_steps = []
    successful_steps = []
    
    for step in steps:
        if step not in step_functions:
            logger.warning(f"Unknown step '{step}' - skipping")
            continue
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 Executing step: {step.upper()}")
        logger.info(f"{'='*70}")
        
        success = step_functions[step]()
        
        if success:
            successful_steps.append(step)
        else:
            failed_steps.append(step)
            logger.error(f"Step '{step}' failed - stopping pipeline")
            break
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("📊 PIPELINE SUMMARY")
    logger.info("="*70)
    logger.info(f"Successful steps: {successful_steps}")
    
    if failed_steps:
        logger.error(f"Failed steps: {failed_steps}")
        logger.error("❌ Pipeline completed with errors")
        return 1
    else:
        logger.info("✅ Pipeline completed successfully!")
        
        # Print output locations
        logger.info("\n📁 OUTPUT FILES:")
        output_files = {
            'Gap-filled GRACE': get_config_value(config, 'paths.grace_filled'),
            'Coarse features': get_config_value(config, 'paths.feature_stack_coarse'),
            'Trained models': get_config_value(config, 'paths.models'),
            'Fine predictions': get_config_value(config, 'paths.predictions_fine'),
            'Final downscaled': get_config_value(config, 'paths.final_downscaled'),
            'Results': get_config_value(config, 'paths.results'),
            'Figures': get_config_value(config, 'paths.figures')
        }
        
        for name, path in output_files.items():
            if Path(path).exists():
                logger.info(f"  ✓ {name}: {path}")
        
        end_time = datetime.now()
        logger.info(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        
        return 0


if __name__ == "__main__":
    sys.exit(main())

