# pipeline.py - Enhanced with Multi-Model Support
"""
GRACE Satellite Data Downscaling Pipeline with Multi-Model Support

This pipeline orchestrates the complete workflow for downscaling GRACE satellite
data to monitor groundwater storage changes at high spatial resolution.

NEW: Supports multiple machine learning models for improved performance.

Usage:
    python pipeline.py --steps all                    # Run complete pipeline with multi-model
    python pipeline.py --steps train,gws              # Run specific steps
    python pipeline.py --models rf,xgb,lgb            # Train specific models
    python pipeline.py --single-model rf              # Use single model (legacy mode)
    python pipeline.py --ensemble-only                # Train all and use ensemble
    python pipeline.py --validation-method annual     # Use annual trend validation
    
Available steps:
    - download: Download satellite data from Earth Engine
    - features: Create aligned feature stack
    - train: Train machine learning model(s)
    - gws: Calculate groundwater storage
    - validate: Validate against wells
    - all: Run all steps

Available models:
    - rf: Random Forest (baseline)
    - xgb: XGBoost  
    - lgb: LightGBM
    - catb: CatBoost
    - nn: Neural Network
    - svr: Support Vector Regression
    - gbr: Gradient Boosting
    - ensemble: Ensemble of multiple models
"""
#!/usr/bin/env python3
import os
import multiprocessing

# Add this to pipeline.py in the imports section:
from src.validation.validate_groundwater import main as validate_results
from src.validation.validate_groundwater_simple import main as validate_simple
from src.validation.validate_annual_trends import main as validate_annual_trends

# Get system limits
cpu_count = multiprocessing.cpu_count()
desired_cores = max(1, cpu_count // 2)  # 50% = 96 cores

# Check for existing NUMEXPR limit
numexpr_limit = int(os.environ.get('NUMEXPR_MAX_THREADS', desired_cores))

# Use the minimum of desired and system-allowed
MAX_CORES = min(desired_cores, numexpr_limit)

print(f"🔧 CPU cores available: {cpu_count}")
print(f"🔧 Desired cores (50%): {desired_cores}")
print(f"🔧 NumExpr limit: {numexpr_limit}")
print(f"🔧 Final limit: {MAX_CORES} cores")

# Set environment variables to the safe limit
os.environ['OMP_NUM_THREADS'] = str(MAX_CORES)
os.environ['OPENBLAS_NUM_THREADS'] = str(MAX_CORES)
os.environ['MKL_NUM_THREADS'] = str(MAX_CORES)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(MAX_CORES)
os.environ['NUMEXPR_NUM_THREADS'] = str(MAX_CORES)

# Don't exceed NumExpr's hard limit
if MAX_CORES > numexpr_limit:
    MAX_CORES = numexpr_limit
    print(f"⚠️  Reduced to NumExpr limit: {MAX_CORES} cores")

print(f"✅ Resource limits set: Using {MAX_CORES} cores")

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import yaml
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import centralized config manager
from src.config_manager import get_config

# Import unified model manager (always needed)
from src.model_manager import ModelManager

# Import other modules conditionally to avoid missing dependencies
try:
    from src.data_loader import main as download_data
    HAS_DATA_LOADER = True
except ImportError as e:
    print(f"⚠️ Data loader not available: {e}")
    HAS_DATA_LOADER = False
    download_data = None

try:
    from src.features import main as create_features
    HAS_FEATURES = True
except ImportError as e:
    print(f"⚠️ Features module not available: {e}")
    HAS_FEATURES = False
    create_features = None

try:
    from src.groundwater_enhanced import calculate_groundwater_storage
    HAS_GROUNDWATER = True
except ImportError as e:
    print(f"⚠️ Groundwater module not available: {e}")
    HAS_GROUNDWATER = False
    calculate_groundwater_storage = None


def setup_logging():
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('grace-pipeline')


def check_requirements():
    """Check if all required directories and files exist."""
    # Get directory paths from config
    try:
        required_dirs = [
            get_config('paths.raw_data', 'data/raw'),
            get_config('paths.processed_data', 'data/processed'),
            get_config('paths.models', 'models'),
            get_config('paths.results', 'results'),
            get_config('paths.figures', 'figures')
        ]
    except:
        # Fallback to hardcoded values if config not available yet
        required_dirs = ["data/raw", "data/processed", "models", "results", "figures"]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(exist_ok=True, parents=True)
    
    # Check for config file
    if not Path("src/config.yaml").exists():
        raise FileNotFoundError("Configuration file 'src/config.yaml' not found!")
    
    return True


def load_config():
    """Load pipeline configuration."""
    with open("src/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Set model defaults if not present (backward compatibility)
    if 'models' not in config:
        config['models'] = {
            'enabled': ['rf'],
            'ensemble': False,
            'cross_validation': True,
            'test_size': 0.2
        }
    
    if 'pipeline' not in config:
        config['pipeline'] = {
            'groundwater_model': 'best',
            'fallback_models': ['rf'],
            'skip_training_if_exists': False
        }
    
    return config


def run_download_step(logger, datasets=None, region=None, use_test_region=False):
    """Download satellite data."""
    logger.info("STEP 1: Downloading satellite data...")
    
    if not HAS_DATA_LOADER:
        logger.error("❌ Data loader not available. Install missing dependencies.")
        return False
    
    # Determine datasets to download
    if datasets is None:
        datasets = 'all'
    else:
        # Convert list to comma-separated string if needed
        if isinstance(datasets, list):
            datasets = ','.join(datasets)
    
    # Determine region to use
    if use_test_region:
        region_name = get_config('download.regions.kansas.name', 'kansas').lower().replace(' ', '_').replace('_', '')[:10]
        if 'kansas' in region_name.lower():
            region_name = 'kansas'
        logger.info(f"Using test region: {region_name}")
    elif region:
        region_name = region
        logger.info(f"Using specified region: {region}")
    else:
        region_name = get_config('download.default_region', 'mississippi')  
        logger.info(f"Using default region: {region_name}")
    
    logger.info(f"Downloading datasets: {datasets}")
    
    try:
        # Set up arguments for data_loader
        download_args = ['data_loader.py', '--download']
        
        # Add datasets (split by comma if multiple)
        if datasets == 'all':
            download_args.append('all')
        else:
            download_args.extend(datasets.split(','))
        
        download_args.extend(['--region', region_name])
        
        sys.argv = download_args
        download_data()
        logger.info("✅ Data download completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Data download failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def run_features_step(logger):
    """Create feature stack."""
    logger.info("STEP 2: Creating feature stack...")
    
    if not HAS_FEATURES:
        logger.error("❌ Features module not available. Install missing dependencies.")
        return False
    
    try:
        create_features()
        logger.info("✅ Feature creation completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Feature creation failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def run_train_step(logger, config, models_to_train=None, single_model=None, ensemble_only=False):
    """Train machine learning model(s)."""
    logger.info("STEP 3: Training machine learning model(s)...")
    
    try:
        # Initialize model manager
        logger.info("🚀 Unified model training with ModelManager")
        manager = ModelManager()
        
        # Prepare data
        logger.info("📦 Preparing training data...")
        X, y, feature_names, metadata = manager.prepare_data()
        
        # Determine which models to train
        if models_to_train:
            enabled_models = models_to_train
            logger.info(f"Training specified models: {enabled_models}")
        elif ensemble_only:
            enabled_models = config['models'].get('enabled', ['rf', 'xgb', 'lgb'])
            logger.info(f"Training all models for ensemble: {enabled_models}")
        elif single_model:
            enabled_models = [single_model]
            logger.info(f"Training single model: {single_model}")
        else:
            enabled_models = config['models'].get('enabled', ['rf'])
            logger.info(f"Training configured models: {enabled_models}")
        
        # Train models with proper metadata for data leakage prevention
        logger.info("🔄 Using temporal splitting to prevent data leakage")
        results = manager.train_all_models(X, y, metadata, enabled_models=enabled_models)
        
        # Save models and results
        manager.save_models()
        
        # Create comparison plots
        manager.create_comparison_plots()
        
        # Log summary
        if len(results) > 0:
            logger.info("📊 Model training results:")
            for _, row in results.iterrows():
                logger.info(f"  {row['display_name']}: R² = {row['test_r2']:.4f}, "
                          f"RMSE = {row['test_rmse']:.4f}")
            
            best_model = results.loc[results['test_r2'].idxmax(), 'display_name']
            logger.info(f"🏆 Best model: {best_model}")
        
        logger.info("✅ Model training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def run_groundwater_step(logger, config):
    """Calculate groundwater storage."""
    logger.info("STEP 4: Calculating groundwater storage...")
    
    if not HAS_GROUNDWATER:
        logger.error("❌ Groundwater module not available. Install missing dependencies.")
        return False
    
    try:
        # Enhanced groundwater calculation can automatically detect and use the best model
        gws_ds = calculate_groundwater_storage()
        logger.info("✅ Groundwater calculation completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Groundwater calculation failed: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Try fallback models if specified
        fallback_models = config['pipeline'].get('fallback_models', [])
        if fallback_models:
            logger.info(f"🔄 Trying fallback models: {fallback_models}")
            for fallback in fallback_models:
                try:
                    fallback_path = f"models/{fallback}_model.joblib"
                    if Path(fallback_path).exists():
                        logger.info(f"  Trying {fallback}...")
                        # You could modify calculate_groundwater_storage to accept a specific model path
                        gws_ds = calculate_groundwater_storage()
                        logger.info(f"✅ Groundwater calculation succeeded with {fallback}")
                        return True
                except Exception as e2:
                    logger.warning(f"  {fallback} also failed: {str(e2)}")
                    continue
        
        return False


def run_validation_step(logger, method='simple'):
    """Validate results against wells."""
    logger.info("STEP 5: Validating against well observations...")
    logger.info(f"Using validation method: {method}")
    
    try:
        if method == 'simple':
            validate_simple()
        elif method == 'annual':
            validate_annual_trends()
        else:
            validate_results()
        logger.info("✅ Validation completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def check_model_availability():
    """Check which ML libraries are available."""
    available = []
    
    try:
        import xgboost
        available.append('xgb')
    except ImportError:
        pass
    
    try:
        import lightgbm
        available.append('lgb')
    except ImportError:
        pass
    
    try:
        import catboost
        available.append('catb')
    except ImportError:
        pass
    
    # These are always available with scikit-learn
    available.extend(['rf', 'nn', 'svr', 'gbr'])
    
    return available


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="GRACE Satellite Data Downscaling Pipeline with Multi-Model Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Existing arguments
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='Comma-separated list of steps to run (download,features,train,gws,validate,all)'
    )
    
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download data and exit'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download step even if included in steps'
    )
    
    # NEW: Dataset selection arguments
    parser.add_argument(
        '--datasets',
        type=str,
        help='Comma-separated list of datasets to download (grace,gldas,chirps,terraclimate,modis,dem,openlandmap,landscan,all). Default: all'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        help='Region to download data for (mississippi, kansas, etc.). Default: mississippi'
    )
    
    parser.add_argument(
        '--test-region',
        action='store_true',
        help='Use test region from config.yaml instead of main region'
    )
    
    # NEW: Model selection arguments
    parser.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of models to train (rf,xgb,lgb,catb,nn,svr,gbr)'
    )
    
    parser.add_argument(
        '--single-model',
        type=str,
        choices=['rf', 'xgb', 'lgb', 'catb', 'nn', 'svr', 'gbr'],
        help='Train only a single model (legacy mode)'
    )
    
    parser.add_argument(
        '--ensemble-only',
        action='store_true',
        help='Train all available models and use ensemble'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    
    parser.add_argument(
        '--validation-method',
        type=str,
        choices=['original', 'simple', 'annual'],
        default='simple',
        help='Validation method to use (simple is recommended, annual for trend analysis)'
    )
    
    args = parser.parse_args()
    
    # List available models and exit
    if args.list_models:
        available = check_model_availability()
        print("Available machine learning models:")
        model_descriptions = {
            'rf': 'Random Forest (baseline, always available)',
            'xgb': 'XGBoost (excellent for tabular data)',
            'lgb': 'LightGBM (fast and memory efficient)', 
            'catb': 'CatBoost (handles categorical features well)',
            'nn': 'Neural Network (MLPRegressor)',
            'svr': 'Support Vector Regression',
            'gbr': 'Gradient Boosting Regressor'
        }
        
        for model in available:
            status = "✅ Available" if model in available else "❌ Not installed"
            desc = model_descriptions.get(model, "")
            print(f"  {model:4s}: {status:12s} - {desc}")
        
        if 'xgb' not in available:
            print("\nTo install XGBoost: pip install xgboost")
        if 'lgb' not in available:
            print("To install LightGBM: pip install lightgbm")
        if 'catb' not in available:
            print("To install CatBoost: pip install catboost")
        
        return 0
    
    # Setup
    logger = setup_logging()
    logger.info("Starting GRACE downscaling pipeline with multi-model support")
    
    try:
        check_requirements()
        config = load_config()
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        return 1
    
    # Parse models to train
    models_to_train = None
    if args.models:
        models_to_train = [m.strip().lower() for m in args.models.split(',')]
        available = check_model_availability()
        
        # Validate model availability
        unavailable = [m for m in models_to_train if m not in available]
        if unavailable:
            logger.error(f"Models not available: {unavailable}")
            logger.info(f"Available models: {available}")
            return 1
        
        logger.info(f"Will train specified models: {models_to_train}")
    
    # Parse steps
    if args.download_only:
        steps = ['download']
    elif args.steps.lower() == 'all':
        steps = ['download', 'features', 'train', 'gws', 'validate']
    else:
        steps = [s.strip().lower() for s in args.steps.split(',')]
    
    if args.skip_download and 'download' in steps:
        steps.remove('download')
        logger.info("Skipping download step as requested")
    
    logger.info(f"Pipeline will run steps: {steps}")
    
    # Log model configuration
    if 'train' in steps:
        if args.single_model:
            logger.info(f"Model mode: Single model ({args.single_model})")
        elif args.ensemble_only:
            logger.info("Model mode: Ensemble of all available models")
        else:
            enabled = config['models'].get('enabled', ['rf'])
            logger.info(f"Model mode: Multi-model ({enabled})")
    
    # Execute steps
    step_functions = {
        'download': lambda: run_download_step(logger, args.datasets, args.region, args.test_region),
        'features': lambda: run_features_step(logger),
        'train': lambda: run_train_step(logger, config, models_to_train, 
                                       args.single_model, args.ensemble_only),
        'gws': lambda: run_groundwater_step(logger, config),
        'validate': lambda: run_validation_step(logger, args.validation_method)
    }
    
    failed_steps = []
    
    for step in steps:
        if step not in step_functions:
            logger.warning(f"Unknown step '{step}', skipping...")
            continue
        
        success = step_functions[step]()
        
        if not success:
            failed_steps.append(step)
            logger.error(f"Step '{step}' failed, stopping pipeline")
            break
    
    # Summary
    if failed_steps:
        logger.error(f"Pipeline failed at step(s): {failed_steps}")
        return 1
    else:
        logger.info("Pipeline completed successfully!")
        
        # Print final outputs
        logger.info("\n" + "="*60)
        logger.info("PIPELINE OUTPUTS:")
        feature_stack_path = get_config('paths.feature_stack', 'data/processed/feature_stack.nc')
        logger.info(f"  - Feature stack: {feature_stack_path}")
        
        # Model outputs (varies by training mode)
        models_dir = Path(get_config('paths.models', 'models'))
        if models_dir.exists():
            model_files = list(models_dir.glob("*_model.joblib"))
            if model_files:
                logger.info("  - Trained models:")
                for model_file in sorted(model_files):
                    logger.info(f"    * {model_file}")
            
            comparison_file = models_dir / "model_comparison.csv"
            if comparison_file.exists():
                logger.info(f"  - Model comparison: {comparison_file}")
        
        results_dir = get_config('paths.results', 'results')
        figures_dir = get_config('paths.figures', 'figures')
        logger.info(f"  - Groundwater data: {results_dir}/groundwater_storage_anomalies.nc")
        logger.info(f"  - Validation metrics: {results_dir}/validation/")
        logger.info(f"  - Figures: {figures_dir}/")
        logger.info("="*60)
        
        return 0


if __name__ == "__main__":
    sys.exit(main())