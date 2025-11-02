# GRACE Groundwater Downscaling Pipeline

A comprehensive machine learning pipeline for downscaling GRACE satellite data to monitor groundwater storage changes at high spatial resolution.

## Project Overview

This project uses multiple satellite datasets and advanced machine learning models to predict high-resolution Total Water Storage (TWS) from GRACE satellites and calculate groundwater storage anomalies by removing soil moisture and snow water equivalent components.

**Key Innovation**: Combines GRACE, GLDAS, CHIRPS, TerraClimate, and static geographic data to achieve groundwater monitoring at 0.1° resolution (vs GRACE's native ~3° resolution).

## Repository Structure

### 🎯 Main Pipeline
- **`pipeline.py`** - Main orchestrator that runs the complete workflow from data download to validation
  - Supports all pipeline steps: download → features → train → groundwater → validate
  - Command-line interface with flexible model selection
  - Handles multi-model training and ensemble approaches

### 📁 Core Modules (`src/`)

#### Data Processing
- **`data_loader.py`** - Downloads satellite data from Google Earth Engine
  - GRACE TWS data from JPL Mascon RL06.3
  - GLDAS land surface model data (soil moisture, evapotranspiration, snow)
  - CHIRPS precipitation data
  - TerraClimate meteorological data
  - Static geographic features (DEM, land cover, soil properties)
  - **🆕 5km native processing**: All data exported at 5km with scientifically appropriate reducers

- **`features.py`** - Creates aligned multi-temporal feature stack
  - Temporal alignment of all satellite datasets
  - Spatial resampling to common 0.1° grid
  - Data quality validation and gap filling
  - Outputs standardized NetCDF feature stack

#### Machine Learning
- **`model_manager.py`** - 🎯 **UNIFIED MODEL TRAINING HUB**
  - **Multi-model support**: Random Forest, XGBoost, LightGBM, CatBoost, Neural Networks, SVR
  - **Advanced feature engineering**: Temporal lags (1,3,6 months), seasonal encoding, static features
  - **Intelligent preprocessing**: Correlation analysis, feature scaling, NaN handling
  - **Robust validation**: Temporal splitting to prevent data leakage
  - **Model comparison**: Automated performance comparison and best model selection
  - **Ensemble methods**: Weighted ensemble of multiple models
  - **Scientific scaling**: Model-specific preprocessing (tree-based vs neural networks)

#### Groundwater Calculation
- **`groundwater_enhanced.py`** - Calculates groundwater storage anomalies
  - **Equation**: `GWS = TWS - Soil_Moisture - Snow_Water_Equivalent`
  - Robust model loading with automatic fallback mechanisms
  - Handles different model types (neural networks need scaling, tree models don't)
  - Reference period normalization (2004-2009 baseline)
  - Scientific unit validation and conversion

#### Utilities & Validation
- **`utils.py`** - Scientific processing utilities
  - GRACE data type detection (Mascon vs raw spherical harmonics)
  - Appropriate resampling methods for different data types
  - Scientific validation functions
  - Coordinate system handling

- **`validation/`** - Results validation against ground truth
  - `validate_groundwater.py` - Comprehensive validation against USGS well observations
  - `validate_groundwater_simple.py` - Quick validation metrics
  - `validate_annual_trends.py` - Long-term trend analysis

### 📊 Visualization System (`plots/`)
- **`plot_feature_correlations.py`** - Correlation analysis and heatmaps
- **`plot_feature_distributions.py`** - Distribution analysis and scaling validation
- Organized output structure: `figures/{script_name}/` for easy identification

### ⚙️ Configuration
- **`src/config.yaml`** - Central configuration file
  - Region definition and processing parameters
  - Model selection and hyperparameters
  - Feature engineering settings (correlation thresholds, lag periods)
  - Pipeline behavior configuration

## Key Features

### 🔬 Scientific Accuracy
- **Proper GRACE handling**: Automatic detection of Mascon vs raw GRACE data with appropriate processing
- **Temporal integrity**: Prevents data leakage with proper temporal train/test splitting
- **Unit validation**: Automatic detection and correction of unit issues (e.g., temperature scaling)
- **Reference period consistency**: Standardized anomaly calculation relative to 2004-2009

### 🚀 Advanced ML Pipeline
- **Multi-model ensemble**: Combines strengths of different algorithm families
- **Intelligent feature engineering**: Automated lag feature creation and correlation filtering
- **Robust preprocessing**: Model-specific scaling and NaN handling
- **Comprehensive validation**: Multiple validation approaches including temporal trends

### 🛠️ Production Ready
- **Modular design**: Each component has clear responsibilities
- **Error handling**: Robust fallback mechanisms and error recovery
- **Logging**: Comprehensive logging for debugging and monitoring
- **Configurable**: Easy to adapt for different regions or model configurations
- **🆕 Flexible data selection**: Choose specific datasets and regions for faster development/testing

## Usage

### Quick Start
```bash
# Run complete pipeline
python pipeline.py --steps all

# Train specific models
python pipeline.py --steps train --models rf,xgb,lgb

# Single model training
python pipeline.py --single-model rf

# Ensemble approach
python pipeline.py --ensemble-only
```

### Individual Steps
```bash
# Download satellite data
python pipeline.py --steps download

# Create feature stack
python pipeline.py --steps features

# Train models only
python pipeline.py --steps train

# Calculate groundwater storage
python pipeline.py --steps gws

# Validate results
python pipeline.py --steps validate
```

### **NEW: Dataset Selection & Region Control**
```bash
# Select specific datasets (much faster for testing)
python pipeline.py --steps download --datasets grace,gldas,chirps

# Use test region (Kansas - small area for development)
python pipeline.py --steps download --test-region

# Combine dataset selection with test region
python pipeline.py --steps download --datasets grace,gldas --test-region

# Use different region
python pipeline.py --steps download --region kansas

# Run full pipeline with test region (recommended for development)
python pipeline.py --steps all --test-region

# Available datasets: grace, gldas, chirps, terraclimate, modis, dem, openlandmap, landscan, all
# Available regions: mississippi (default), kansas (test region)
```

## Data Flow

```
Raw Satellite Data → Feature Engineering → ML Training → Groundwater Calculation → Validation
      ↓                    ↓                  ↓              ↓                   ↓
  data_loader.py    →   features.py   →  model_manager.py → groundwater_enhanced.py → validation/
```

## Model Performance

The pipeline supports multiple ML approaches:
- **Random Forest**: Baseline model, handles mixed data types well
- **XGBoost/LightGBM**: Gradient boosting for complex patterns
- **Neural Networks**: Deep learning for non-linear relationships
- **Ensemble**: Weighted combination of multiple models

Typical performance: R² > 0.85, RMSE < 4 cm for groundwater anomalies

## Dependencies

See `requirements.txt` for full dependencies. Key requirements:
- Scientific computing: numpy, pandas, xarray, scipy, scikit-learn
- Geospatial: rasterio, rioxarray, cartopy
- ML libraries: xgboost, lightgbm, catboost
- Earth observation: earthengine-api, geemap

## Output Structure

```
data/
├── raw/           # Downloaded satellite data
└── processed/     # Processed feature stacks

models/            # Trained ML models
├── rf_model.joblib
├── xgb_model.joblib
└── model_comparison.csv

results/           # Groundwater storage outputs
└── groundwater_storage_anomalies.nc

figures/           # Generated plots and analysis
└── {script_name}/ # Organized by generating script
```

## Contributing

The codebase is designed for scientific reproducibility and extensibility:
1. All processing steps are logged and configurable
2. Modular design allows easy addition of new models or datasets
3. Comprehensive validation ensures scientific accuracy
4. Clean separation between data processing, ML training, and analysis

## Citation

If you use this pipeline in your research, please cite the relevant GRACE and satellite data sources:
- GRACE JPL Mascon RL06.3
- GLDAS Noah Land Surface Model
- CHIRPS Precipitation Dataset
- TerraClimate Global Dataset

---

**Maintained by**: ORISE Groundwater Research Team  
**Last Updated**: September 2024  
**Python Version**: 3.8+