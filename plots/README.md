# Plots Directory

This directory contains all visualization and plotting scripts for the GRACE Groundwater Downscaling project.

## Organization Structure

```
plots/
├── README.md                           # This file
├── plot_feature_correlations.py        # Visualizes feature correlation matrices and analysis
├── plot_feature_distributions.py       # Shows distribution and scaling comparison plots  
├── plot_temporal_analysis.py           # Temporal patterns and lag feature analysis
└── plot_data_quality.py               # Data quality, NaN patterns, and validation plots

figures/
├── plot_feature_correlations/          # Correlation heatmaps and pair plots
├── plot_feature_distributions/         # Histograms, box plots, scaling comparisons
├── plot_temporal_analysis/             # Time series, lag correlations, seasonal patterns
└── plot_data_quality/                 # Data coverage, NaN maps, quality metrics
```

## Script Descriptions

### `plot_feature_correlations.py`
Creates correlation matrices, heatmaps, and identifies highly correlated feature pairs to validate feature selection decisions.

### `plot_feature_distributions.py` 
Visualizes feature value distributions, scaling effects, and comparison between original vs enhanced features.

### `plot_temporal_analysis.py`
Analyzes temporal patterns, lag feature effectiveness, and seasonal component validation.

### `plot_data_quality.py`
Examines data completeness, spatial coverage, NaN patterns, and validates unit conversions.

## Usage

Run any script from the project root directory:
```bash
python plots/plot_feature_correlations.py
python plots/plot_feature_distributions.py
# etc.
```

Each script automatically creates its own subdirectory in `figures/` and saves all generated plots there.