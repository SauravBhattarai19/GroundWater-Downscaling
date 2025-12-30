# Prompt for Claude Sonnet: Anomaly-Based Feature Engineering

## Context
We are building a machine learning model to downscale GRACE Terrestrial Water Storage (TWS) from 55km to 5km resolution. GRACE data are provided as **anomalies relative to a 2004-2009 baseline period**. Currently, our predictor variables are in absolute values, which creates a mismatch. We need to convert water-balance predictor variables to **baseline period anomalies** to match GRACE's anomaly representation.

## Objective
Convert temporal predictor variables to baseline anomalies (2004-2009 mean subtracted) for water-balance variables, ensuring consistency with GRACE anomaly data and improving model performance.

## Key Requirements

### 1. Baseline Period
- **Reference period**: 2004-01 to 2009-12 (72 months)
- This matches GRACE JPL mascon RL06.3 baseline period
- Calculate: `anomaly = value - mean(2004-2009)`
- Store baseline statistics for each variable (mean, std) for validation

### 2. Variables to Convert to Baseline Anomalies

**Water-balance variables (primary focus):**
- `chirps` - Precipitation (CHIRPS, 5km resolution)
- `aet` - Actual evapotranspiration (TerraClimate, 4km resolution)
- `SWE_inst` - Snow Water Equivalent (currently GLDAS, 27km)
- `SoilMoi0_10cm_inst` - Surface soil moisture (currently GLDAS, 27km)
- `SoilMoi10_40cm_inst` - Shallow soil moisture (currently GLDAS, 27km)
- `SoilMoi100_200cm_inst` - Deep soil moisture (currently GLDAS, 27km)

**Optional (for consistency):**
- `tmean` - Temperature (TerraClimate, 4km)
- `def` - Climate water deficit (TerraClimate, 4km)

### 3. Data Source Improvements

**Current issue**: GLDAS provides soil moisture and SWE at ~27km resolution, but our target is 5km downscaling.

**Recommended upgrade**: Switch to **ERA5-Land** (available on Google Earth Engine):
- **Resolution**: ~9km (0.1°) - 3x better than GLDAS
- **Coverage**: 1950-present (covers our 2003-2024 period)
- **Available variables**:
  - `volumetric_soil_water_layer_1` (0-7cm)
  - `volumetric_soil_water_layer_2` (7-28cm)
  - `volumetric_soil_water_layer_3` (28-100cm)
  - `volumetric_soil_water_layer_4` (100-289cm)
  - `snow_depth_water_equivalent` (SWE)
  - `total_evaporation` (ET alternative)

**Action**: Add ERA5-Land download functionality to `data_loader.py` and replace GLDAS soil moisture/SWE with ERA5-Land equivalents.

### 4. Remove Redundant Variables

**Current redundancies:**
- `pr` (TerraClimate precipitation) - **REMOVE** (redundant with `chirps`, which is higher quality)
- `Evap_tavg` (GLDAS evapotranspiration) - **REMOVE** (redundant with TerraClimate `aet`, and coarser resolution)

**Rationale**: 
- CHIRPS precipitation is higher quality than TerraClimate `pr`
- TerraClimate `aet` is at 4km vs GLDAS `Evap_tavg` at 27km
- Keep the higher resolution, better quality sources

### 5. Handle Desert/Zero Precipitation Regions

**Problem**: In arid regions, mean precipitation during 2004-2009 may be near-zero, causing division issues or meaningless anomalies.

**Solution options:**
- **Option A (Recommended)**: Use absolute anomalies (value - mean) for all variables, not relative anomalies
- **Option B**: Add small epsilon floor (e.g., 0.1 mm) before division for relative anomalies
- **Option C**: Mask out desert regions (mean precip < 1 mm/month) from training

**Recommendation**: Use Option A (absolute anomalies) - simpler and scientifically sound.

### 6. Implementation Details

#### Files to Modify:

1. **`src_new_approach/data_loader.py`**:
   - Add `export_era5_land()` function to download ERA5-Land soil moisture, SWE, and ET
   - Remove or deprecate GLDAS soil moisture/SWE downloads
   - Keep GLDAS only if ERA5-Land is unavailable

2. **`src_new_approach/feature_aggregator.py`**:
   - Modify `load_complete_features_from_raw()` to:
     - Load ERA5-Land instead of GLDAS for soil moisture/SWE
     - Calculate baseline anomalies (2004-2009) for water-balance variables
     - Remove `pr` and `Evap_tavg` from feature list
   - Add function `calculate_baseline_anomalies()` that:
     - Computes 2004-2009 mean for each variable at each grid cell
     - Subtracts baseline mean from all time steps
     - Handles missing data in baseline period (use available months)
     - Validates anomaly calculation (check for reasonable ranges)

3. **`src_new_approach/generate_fine_derived_features.py`**:
   - Update to work with baseline anomalies instead of absolute values
   - Ensure derived features (lags, spatial, accumulations) are computed from anomalies
   - Remove seasonal anomaly features (redundant with baseline anomalies)

4. **`src_new_approach/config_coarse_to_fine.yaml`**:
   - Add `baseline_anomaly` configuration section:
     ```yaml
     baseline_anomaly:
       enabled: true
       reference_period:
         start: 2004-01
         end: 2009-12
       variables:
         water_balance: ['chirps', 'aet', 'SWE_inst', 'SoilMoi0_10cm_inst', 'SoilMoi10_40cm_inst', 'SoilMoi100_200cm_inst']
         optional: ['tmean', 'def']
       method: 'absolute'  # value - mean (not relative)
     ```
   - Add ERA5-Land data source configuration
   - Remove `pr` and `Evap_tavg` from feature lists

#### Processing Pipeline:

```
Raw Data (data/raw/)
    ↓
1. Download ERA5-Land (soil moisture, SWE, ET)
    ↓
2. Load all temporal variables for 2003-2024
    ↓
3. Calculate 2004-2009 baseline means (per grid cell, per variable)
    ↓
4. Compute baseline anomalies: anomaly = value - baseline_mean
    ↓
5. Create feature stack with anomalies (not absolute values)
    ↓
6. Generate derived features (lags, spatial, accumulations) from anomalies
    ↓
7. Save enhanced feature stack
```

### 7. Validation Steps

1. **Baseline statistics validation**:
   - Check that baseline means are reasonable (e.g., precip means should be positive in non-desert regions)
   - Verify baseline period has sufficient data (at least 60/72 months)
   - Compare baseline means with climatology from literature

2. **Anomaly validation**:
   - Anomalies should have mean ≈ 0 over baseline period (by construction)
   - Anomalies should show temporal variability (not all zeros)
   - Check for extreme outliers (flag if |anomaly| > 5×std)

3. **Spatial consistency**:
   - Anomalies should be spatially smooth (no abrupt jumps)
   - Desert regions should have near-zero anomalies (expected)

4. **Comparison with GRACE**:
   - GRACE anomalies should correlate better with predictor anomalies than with absolute values
   - Check correlation improvement in validation

### 8. Backward Compatibility

- Keep existing feature stack files for comparison
- Add versioning to feature stacks (e.g., `feature_stack_55km_v2_anomaly.nc`)
- Document changes in feature names/units

### 9. Expected Outcomes

1. **Better alignment with GRACE**: Predictor anomalies should correlate better with GRACE anomalies
2. **Improved model performance**: Model should learn relationships in anomaly space more effectively
3. **Reduced redundancy**: Fewer features, less multicollinearity
4. **Higher resolution**: ERA5-Land provides better spatial detail than GLDAS

### 10. Testing Checklist

- [ ] ERA5-Land data downloads successfully
- [ ] Baseline means calculated correctly (2004-2009)
- [ ] Anomalies computed correctly (value - mean)
- [ ] Desert regions handled appropriately (no division by zero)
- [ ] Redundant variables removed (`pr`, `Evap_tavg`)
- [ ] Feature stack created with anomalies
- [ ] Derived features computed from anomalies
- [ ] Model training works with new features
- [ ] Validation shows improved correlation with GRACE

## Code Structure Reference

Current codebase structure:
- `src_new_approach/data_loader.py` - Downloads raw data from GEE
- `src_new_approach/feature_aggregator.py` - Aggregates features from raw data
- `src_new_approach/generate_fine_derived_features.py` - Adds derived features (lags, spatial, etc.)
- `src_new_approach/config_coarse_to_fine.yaml` - Configuration file
- `processed_coarse_to_fine/feature_stack_55km.nc` - Current feature stack
- `processed_coarse_to_fine/feature_stack_5km.nc` - Current fine feature stack

## Questions to Resolve

1. **ERA5-Land vs GLDAS**: Should we switch to ERA5-Land now, or keep GLDAS and add ERA5-Land as option?
   - **Recommendation**: Switch to ERA5-Land (better resolution, complete layers)

2. **Anomaly method**: Absolute (value - mean) or relative ((value - mean) / mean)?
   - **Recommendation**: Absolute anomalies (simpler, handles zero means)

3. **Temperature and deficit**: Convert to anomalies too, or keep absolute?
   - **Recommendation**: Convert for consistency, but lower priority than water-balance variables

4. **Missing soil layer**: ERA5-Land has 4 layers (0-7, 7-28, 28-100, 100-289cm). GLDAS has 0-10, 10-40, 100-200cm. Should we map them or use all 4 ERA5-Land layers?
   - **Recommendation**: Use all 4 ERA5-Land layers (more complete vertical profile)

## Implementation Priority

**Phase 1 (Critical)**:
1. Add ERA5-Land download to `data_loader.py`
2. Implement baseline anomaly calculation in `feature_aggregator.py`
3. Remove redundant variables (`pr`, `Evap_tavg`)
4. Update config file

**Phase 2 (Important)**:
1. Update derived feature generation to work with anomalies
2. Add validation functions
3. Test with existing pipeline

**Phase 3 (Optional)**:
1. Convert temperature/deficit to anomalies
2. Add relative anomaly option
3. Enhanced desert region handling

---

**Note**: This is an experimental branch (`anomaly-experiments`). Keep the baseline implementation intact for comparison.

