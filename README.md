# GRACE Coarse-to-Fine Downscaling: New Approach

This directory contains the complete **new coarse-to-fine approach** for GRACE downscaling with optimized residual correction.

## 📁 Directory Structure

```
New Approach/
├── pipeline_coarse_to_fine.py           # Main pipeline script
├── src_new_approach/                    # Source code
│   ├── config_coarse_to_fine.yaml      # Configuration file
│   ├── feature_creator.py               # Feature creation
│   ├── gap_filler.py                   # GRACE gap filling
│   ├── feature_aggregator.py           # Feature aggregation
│   ├── model_trainer.py                # ML model training
│   ├── fine_predictor.py               # Fine-scale prediction
│   ├── residual_corrector.py           # Residual correction
│   ├── residual_corrector_multi.py     # Multi-method testing
│   ├── utils_downscaling.py            # Utilities
│   └── scripts/                        # Analysis scripts
│       ├── plot_validation_scatter.py
│       ├── test_all_residual_methods.py
│       └── test_residual_methods_corrected.py
├── models_coarse_to_fine_simple/       # Trained ML models (XGB, LGB, NN)
├── processed_coarse_to_fine/           # Processed data
│   ├── feature_stack_5km.nc           # Fine-resolution features
│   ├── feature_stack_55km.nc          # Coarse-resolution features
│   ├── grace_filled_stl.nc            # Gap-filled GRACE
│   ├── predictions_55km.nc            # Coarse predictions
│   └── predictions_5km.nc             # Fine predictions
├── results_coarse_to_fine/             # Final results
│   └── grace_downscaled_5km.nc        # Final downscaled GRACE
├── figures_coarse_to_fine/             # Plots and validation
│   ├── comprehensive_validation_scatter.png
│   └── residual_method_comparison.png
└── logs_coarse_to_fine/                # Pipeline logs
```

## 🚀 How to Run

### 1. Full Pipeline
```bash
cd "../New Approach"
python pipeline_coarse_to_fine.py --config src_new_approach/config_coarse_to_fine.yaml
```

### 2. Specific Steps Only
```bash
# Run only residual correction step
python pipeline_coarse_to_fine.py --steps correct --use-simple-split

# Run training + prediction + correction
python pipeline_coarse_to_fine.py --steps train,predict,correct
```

### 3. Test Multiple Residual Methods
```bash
python src_new_approach/scripts/test_all_residual_methods.py
```

### 4. Generate Validation Plots
```bash
python src_new_approach/scripts/plot_validation_scatter.py
```

## 📊 Key Results

### Model Performance (from comprehensive_validation_scatter.png):
- **LightGBM**: R² = 0.972, RMSE = 1.82 cm
- **XGBoost**: R² = 0.971, RMSE = 1.19 cm  
- **Neural Network**: R² = 0.932, RMSE = 2.01 cm
- **Ensemble**: R² = 0.974, RMSE = 1.77 cm
- **Final Downscaled**: R² = 0.753, RMSE = 5.37 cm

### Residual Correction Methods:
8 different interpolation methods tested:
1. Bilinear Interpolation
2. Geographic Assignment (Novel)
3. Nearest Neighbor
4. Bicubic Interpolation
5. IDW (Inverse Distance Weighting)
6. Gaussian Kernel Smoothing
7. Area-Weighted Assignment
8. **Distance-Weighted Nearest** (Best: R² = 0.4396)

## 🔧 Configuration

The main configuration is in `src_new_approach/config_coarse_to_fine.yaml`:

```yaml
# Key settings
resolution:
  grace_native_km: 55.66    # GRACE native resolution
  fine_resolution_km: 5     # Target resolution
  aggregation_factor: 11    # 55km / 5km ≈ 11

residual_correction:
  interpolation_method: distance_weighted_nearest  # Optimized method
  smooth_residuals: true
  clip_outliers: true
```

## 🎯 Pipeline Steps

1. **CREATE_FEATURES**: Generate 5km features from satellite data
2. **GAP_FILL**: Fill GRACE gaps using STL decomposition
3. **AGGREGATE**: Create 55km features for model training
4. **TUNE**: Hyperparameter optimization (if needed)
5. **TRAIN**: Train ensemble models (XGB, LGB, NN)
6. **PREDICT**: Generate predictions at fine scale
7. **CORRECT**: Apply optimized residual correction
8. **VALIDATE**: Create validation plots and metrics

## 💡 Key Innovations

### 1. Coarse-to-Fine Approach
- Train models at GRACE native resolution (55km)
- Apply to fine-resolution features (5km)
- Maintains spatial extrapolation integrity

### 2. Optimized Residual Correction
- Tested 8 different interpolation methods
- Distance-Weighted Nearest performs best
- Preserves spatial error patterns

### 3. Comprehensive Validation
- Before/after residual correction comparison
- Spatially matched validation approach
- Publication-ready scatter plots

## 🔍 Validation Methodology

The validation follows a rigorous approach:
1. **Before**: 5km predictions → aggregate to 55km → compare with GRACE 55km
2. **After**: 5km corrected → aggregate to 55km → compare with GRACE 55km
3. Uses identical spatial sampling for fair comparison
4. 100% data coverage for maximum statistical power

This ensures the R² = 0.753 represents true downscaling performance with residual correction applied.

## 📈 Next Steps

1. **Test with different study regions**
2. **Experiment with advanced interpolation methods**
3. **Optimize hyperparameters for residual correction**
4. **Compare with other downscaling approaches**

---

**Note**: This approach represents a complete, standalone implementation of the coarse-to-fine GRACE downscaling methodology with optimized residual correction.