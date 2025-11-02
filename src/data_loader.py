import argparse
import os
import ee
import geemap
from datetime import datetime
try:
    from dataretrieval import nwis
    HAS_NWIS = True
except ImportError:
    HAS_NWIS = False
    print("⚠️ dataretrieval not available - USGS well data functionality disabled")
import pandas as pd
from .config_manager import get_config

# Earth Engine initialization state
_ee_initialized = False

def initialize_earth_engine():
    """Initialize Earth Engine with lazy loading."""
    global _ee_initialized
    if _ee_initialized:
        return
    
    try:
        ee.Initialize(project='ee-sauravbhattarai1999')
        _ee_initialized = True
    except Exception as e:
        try:
            print("⚠️ Attempting Earth Engine authentication...")
            ee.Authenticate()
            ee.Initialize()
            _ee_initialized = True
        except Exception as auth_error:
            raise RuntimeError(
                f"Failed to initialize Earth Engine. Please ensure:\n"
                f"1. You have authenticated with 'earthengine authenticate'\n"
                f"2. Your project is registered for Earth Engine access\n"
                f"Original error: {auth_error}"
            )

def get_region(region_name):
    """Get region geometry, initializing EE if needed."""
    initialize_earth_engine()
    regions = {
        "mississippi": ee.Geometry.Rectangle([-113.94, 28.84, -77.84, 49.74]),
        "kansas": ee.Geometry.Rectangle([-99.0, 37.0, -96.0, 39.0])  # Test region from config
    }
    return regions.get(region_name)

# Output directories
RAW_DIR = "data/raw"
ALL_DATASETS = ["grace", "gldas", "chirps", "modis", "terraclimate", "dem", "usgs", "openlandmap", "landscan"]

def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    for sub in ["grace", "gldas", "chirps", "modis_land_cover", "terraclimate", "usgs_dem", "usgs_well_data", "openlandmap", "landscan"]:
        os.makedirs(os.path.join(RAW_DIR, sub), exist_ok=True)

def export_grace(region):
    """Export GRACE using task-based approach with monthly aggregation for temporal alignment"""
    initialize_earth_engine()
    from src.data_load.grace import submit_grace_monthly_export_tasks
    
    print("Submitting GRACE monthly export tasks to Google Earth Engine...")
    print("This creates proper calendar month composites for temporal alignment.")
    
    try:
        task_info = submit_grace_monthly_export_tasks(region, description_suffix="pipeline")
        print(f"✅ GRACE monthly export task submitted successfully!")
        print(f"Task will process ~240 monthly files asynchronously in Google Earth Engine.")
        return task_info
    except Exception as e:
        print(f"❌ Failed to submit GRACE monthly export task: {e}")
        raise

def export_gldas(region):
    initialize_earth_engine()
    print("Aggregating and exporting GLDAS monthly means...")
    variables = [
        "SoilMoi0_10cm_inst",
        "SoilMoi10_40cm_inst",
        "SoilMoi40_100cm_inst",
        "SoilMoi100_200cm_inst",
        "Evap_tavg",
        "SWE_inst"
    ]
    for var in variables:
        # Get date range and scale from config
        start_date = get_config('data_processing.start_date', '2003-01-01')
        end_date = get_config('data_processing.end_date', '2022-12-31')
        scale = get_config('data_processing.export_scale', 5000)
        
        monthly = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H") \
            .filterDate(start_date, end_date) \
            .filterBounds(region) \
            .select(var)

        def monthly_composite(date):
            start = ee.Date(date)
            end = start.advance(1, 'month')
            return monthly.filterDate(start, end).mean().set('system:time_start', start.millis())

        # Calculate months sequence from config dates
        import datetime
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        n_years = end_dt.year - start_dt.year + 1
        months = ee.List.sequence(0, 12 * n_years - 1).map(lambda i: ee.Date(start_date).advance(i, 'month'))
        monthly_collection = ee.ImageCollection(months.map(monthly_composite))

        raw_dir = get_config('paths.raw_data', 'data/raw')
        geemap.ee_export_image_collection(
            monthly_collection,
            out_dir=os.path.join(raw_dir, "gldas", var),
            scale=scale,
            region=region,
            file_per_band=False
        )

def export_chirps(region):
    initialize_earth_engine()
    print("Aggregating and exporting CHIRPS monthly totals...")
    # Get config values
    start_date = get_config('data_processing.start_date', '2003-01-01')
    end_date = get_config('data_processing.end_date', '2022-12-31')
    scale = get_config('data_processing.export_scale', 5000)
    raw_dir = get_config('paths.raw_data', 'data/raw')
    
    collection = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate(start_date, end_date) \
        .filterBounds(region)

    def monthly_sum(date):
        start = ee.Date(date)
        end = start.advance(1, 'month')
        return collection.filterDate(start, end).sum().set('system:time_start', start.millis())

    # Calculate months sequence from config dates
    import datetime
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    n_years = end_dt.year - start_dt.year + 1
    months = ee.List.sequence(0, 12 * n_years - 1).map(lambda i: ee.Date(start_date).advance(i, 'month'))
    monthly_collection = ee.ImageCollection(months.map(monthly_sum))

    geemap.ee_export_image_collection(
        monthly_collection,
        out_dir=os.path.join(raw_dir, "chirps"),
        scale=scale,
        region=region,
        file_per_band=False
    )

def export_modis_landcover(region):
    """Export MODIS Land Cover using task-based approach for asynchronous processing"""
    initialize_earth_engine()
    from src.data_load.modis import submit_modis_export_task
    
    print("Submitting MODIS Land Cover export task to Google Earth Engine...")
    print("This will create an asynchronous task for processing.")
    
    try:
        scale = get_config('data_processing.export_scale', 5000)
        task_info = submit_modis_export_task(region, scale=scale, description_suffix="pipeline")
        print(f"✅ MODIS Land Cover export task submitted successfully!")
        print(f"Task will be processed asynchronously in Google Earth Engine.")
        return task_info
    except Exception as e:
        print(f"❌ Failed to submit MODIS Land Cover export task: {e}")
        raise

def export_terraclimate(region):
    initialize_earth_engine()
    # Get config values
    start_date = get_config('data_processing.start_date', '2003-01-01')
    end_date = get_config('data_processing.end_date', '2022-12-31')
    scale = get_config('data_processing.export_scale', 5000)
    raw_dir = get_config('paths.raw_data', 'data/raw')
    
    collection = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE") \
        .filterDate(start_date, end_date) \
        .filterBounds(region)

    bands = ["tmmx", "tmmn", "pr", "aet", "def"]

    for band in bands:
        print(f"Exporting TerraClimate: {band}")
        subset = collection.select(band)
        geemap.ee_export_image_collection(
            subset,
            out_dir=os.path.join(raw_dir, "terraclimate", band),
            scale=scale,
            region=region,
            file_per_band=False
        )

def export_usgs_dem(region):
    """Export DEM using task-based approach for asynchronous processing"""
    initialize_earth_engine()
    from src.data_load.dem import submit_dem_export_task
    
    print("Submitting DEM export task to Google Earth Engine...")
    print("This will create an asynchronous task for processing.")
    
    try:
        scale = get_config('data_processing.export_scale', 5000)
        task_info = submit_dem_export_task(region, scale=scale, description_suffix="pipeline")
        print(f"✅ DEM export task submitted successfully!")
        print(f"Task will be processed asynchronously in Google Earth Engine.")
        return task_info
    except Exception as e:
        print(f"❌ Failed to submit DEM export task: {e}")
        raise

def download_usgs_well_data():
    """Download ALL available USGS well data with comprehensive coverage"""
    import time
    import signal
    from tqdm import tqdm
    print("🚰 COMPREHENSIVE USGS WELL DATA DOWNLOAD")
    print("=" * 60)
    print("Downloading ALL available groundwater wells (no arbitrary limits)")
    print("Using 2004-2009 reference period to match GRACE data processing")
    print("🔧 Realistic quality control for irregular groundwater sampling")
    print("⏱️  This may take 30-60 minutes for comprehensive coverage...")
    
    # Get reference period from config
    REFERENCE_START = get_config('data_processing.reference_period.start', '2004-01')
    REFERENCE_END = get_config('data_processing.reference_period.end', '2009-12')
    raw_dir = get_config('paths.raw_data', 'data/raw')
    
    states = ['MS', 'AR', 'LA', 'TN', 'MO', 'KY', 'IL', 'IN', 'OH', 'AL', 
              'WV', 'PA', 'MN', 'WI', 'IA', 'ND', 'SD', 'NE', 'KS', 'OK', 
              'TX', 'NM', 'CO', 'WY', 'MT']
    all_data = []
    well_metadata = []
    
    # Comprehensive debug counters
    debug_stats = {
        'states_processed': 0,
        'total_wells_found': 0,
        'total_wells_tested': 0,
        'no_data_returned': 0,
        'missing_columns': 0,
        'insufficient_total_data': 0,
        'insufficient_reference_data': 0,
        'successful_wells': 0,
        'api_errors': 0,
        'timeout_errors': 0
    }
    
    # Progress tracking
    start_time = time.time()
    state_results = {}

    for state_idx, state in enumerate(states):
        state_start_time = time.time()
        print(f"\n{'='*20} STATE {state_idx+1}/10: {state} {'='*20}")
        
        try:
            # Get info for ALL groundwater sites in the state
            print(f"🔍 Fetching all groundwater sites in {state}...")
            info, _ = nwis.get_info(stateCd=state, siteType="GW", siteStatus="active")
            site_ids = info['site_no'].unique().tolist()
            
            debug_stats['total_wells_found'] += len(site_ids)
            print(f"   ✅ Found {len(site_ids)} potential wells")
            
            if len(site_ids) == 0:
                print(f"   ⚠️ No wells found in {state}")
                continue
                
        except Exception as e:
            print(f"   ❌ Failed to fetch sites for {state}: {e}")
            debug_stats['api_errors'] += 1
            continue

        # Process ALL wells in this state (no limit!)
        state_successful = 0
        state_tested = 0
        
        print(f"🔄 Processing ALL {len(site_ids)} wells in {state}...")
        
        # Use progress bar for each state
        for i, site in enumerate(tqdm(site_ids, desc=f"{state} wells", 
                                     unit="well", leave=False)):
            state_tested += 1
            debug_stats['total_wells_tested'] += 1
            
            # Show detailed progress every 50 wells
            show_details = (i % 50 == 0) or (i < 3)  
            
            try:
                # Add timeout for individual wells to prevent hanging
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Well processing timeout")
                
                # Set 30 second timeout per well
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                
                try:
                    # Get groundwater level data
                    df, _ = nwis.get_gwlevels(site, start='2003-01-01', end='2022-12-31', 
                                            datetime_index=False)
                    
                    if df.empty:
                        if show_details: print(f"      No data: {site}")
                        debug_stats['no_data_returned'] += 1
                        continue
                    
                    # Check for required columns
                    if 'lev_dt' not in df.columns or 'lev_va' not in df.columns:
                        if show_details: print(f"      Missing columns: {site}")
                        debug_stats['missing_columns'] += 1
                        continue
                    
                    if show_details: print(f"      Processing {site}: {len(df)} raw measurements")
                    
                    # Process the data
                    df['datetime'] = pd.to_datetime(df['lev_dt'])
                    df['depth_m'] = df['lev_va'] * 0.3048  # Convert feet to meters
                    
                    # Aggregate to monthly (any measurement per month)
                    monthly = df.set_index('datetime')['depth_m'].resample('MS').mean().dropna()
                    
                    # Quality control: need at least 36 months total (improved from 12)
                    if len(monthly) < 36:
                        if show_details: print(f"      Insufficient data: {len(monthly)} months")
                        debug_stats['insufficient_total_data'] += 1
                        continue
                    
                    # Check reference period data
                    ref_start = pd.to_datetime(REFERENCE_START)
                    ref_end = pd.to_datetime(REFERENCE_END)
                    reference_data = monthly[(monthly.index >= ref_start) & (monthly.index <= ref_end)]
                    
                    # Need at least 24 months in reference period (improved from 6)
                    if len(reference_data) < 24:
                        if show_details: print(f"      Insufficient reference: {len(reference_data)} months")
                        debug_stats['insufficient_reference_data'] += 1
                        continue
                    
                    # SUCCESS! Calculate anomalies
                    reference_mean = reference_data.mean()
                    depth_anomaly = monthly - reference_mean  # Positive = deeper water table
                    
                    # Step 2: Convert to storage anomaly for GRACE comparison
                    # Use standard specific yield of 0.15 based on literature
                    SPECIFIC_YIELD = 0.15  # Standard value from USGS studies
                    storage_anomaly = -depth_anomaly * SPECIFIC_YIELD * 100  # Convert to cm
                    
                    # Save storage anomaly instead of depth anomaly
                    all_data.append(storage_anomaly.rename(site))
                    
                    # Save well metadata
                    site_info = info[info['site_no'] == site].iloc[0]
                    well_metadata.append({
                        'well_id': site,
                        'lat': float(site_info['dec_lat_va']),
                        'lon': float(site_info['dec_long_va']),
                        'state': state,
                        'station_nm': site_info.get('station_nm', ''),
                        'n_months_total': len(monthly),
                        'n_months_reference': len(reference_data),
                        'reference_period': f"{REFERENCE_START}_to_{REFERENCE_END}",
                        'first_measurement': str(monthly.index[0].date()),
                        'last_measurement': str(monthly.index[-1].date()),
                        'specific_yield_used': SPECIFIC_YIELD,  # Add this
                        'units': 'cm_water_equivalent'  # Add this
                    })
                    
                    state_successful += 1
                    debug_stats['successful_wells'] += 1
                    
                    if show_details: print(f"      ✅ SUCCESS: {site}")
                    
                finally:
                    # Clear the timeout
                    signal.alarm(0)
                    
            except TimeoutError:
                debug_stats['timeout_errors'] += 1
                if show_details: print(f"      ⏱️ Timeout: {site}")
                continue
            except Exception as e:
                debug_stats['api_errors'] += 1
                if show_details: print(f"      ❌ Error {site}: {e}")
                continue
        
        # State summary
        state_elapsed = time.time() - state_start_time
        success_rate = (state_successful / state_tested * 100) if state_tested > 0 else 0
        
        state_results[state] = {
            'tested': state_tested,
            'successful': state_successful,
            'success_rate': success_rate,
            'time_minutes': state_elapsed / 60
        }
        
        print(f"   🏁 {state} COMPLETE: {state_successful}/{state_tested} wells ({success_rate:.1f}%) in {state_elapsed/60:.1f} minutes")
        debug_stats['states_processed'] += 1

    # Final comprehensive summary
    total_elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎉 COMPREHENSIVE DOWNLOAD COMPLETE!")
    print(f"{'='*60}")
    print(f"⏱️  Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    
    print(f"\n📊 FINAL STATISTICS:")
    print(f"   States processed: {debug_stats['states_processed']}/10")
    print(f"   Total wells found: {debug_stats['total_wells_found']:,}")
    print(f"   Total wells tested: {debug_stats['total_wells_tested']:,}")
    print(f"   ✅ SUCCESSFUL wells: {debug_stats['successful_wells']:,}")
    print(f"   Overall success rate: {debug_stats['successful_wells']/debug_stats['total_wells_tested']*100:.1f}%")
    
    print(f"\n📋 REJECTION REASONS:")
    print(f"   No data returned: {debug_stats['no_data_returned']:,}")
    print(f"   Missing columns: {debug_stats['missing_columns']:,}")
    print(f"   Insufficient total data (<36 months): {debug_stats['insufficient_total_data']:,}")
    print(f"   Insufficient reference data (<24 months): {debug_stats['insufficient_reference_data']:,}")
    print(f"   API/processing errors: {debug_stats['api_errors']:,}")
    print(f"   Timeout errors: {debug_stats['timeout_errors']:,}")

    if not all_data:
        print("\n❌ No valid well data found after comprehensive search.")
        return

    print(f"\n🎯 COMPREHENSIVE SUCCESS! Processed {len(all_data):,} wells")
    
    # Save time series data
    print("💾 Saving comprehensive well time series data...")
    combined_df = pd.concat(all_data, axis=1)
    combined_df.index.name = 'Date'
    combined_df.to_csv(os.path.join(raw_dir, "usgs_well_data", "monthly_groundwater_anomalies_cm.csv"))
    
    # Save comprehensive metadata
    print("💾 Saving comprehensive well metadata...")
    metadata_df = pd.DataFrame(well_metadata)
    metadata_df.to_csv(os.path.join(raw_dir, "usgs_well_data", "well_metadata.csv"), index=False)
    
    print(f"\n✅ SAVED COMPREHENSIVE DATASET:")
    print(f"   - Time series: monthly_groundwater_anomalies_cm.csv")
    print(f"   - Metadata: well_metadata.csv")
    print(f"   - Total wells: {len(all_data):,}")
    print(f"   - Total measurements: {combined_df.count().sum():,}")
    print(f"   - Units: cm water equivalent storage anomaly")
    
    # Detailed state breakdown
    print(f"\n📊 WELLS BY STATE:")
    state_counts = metadata_df['state'].value_counts()
    for state, count in state_counts.items():
        tested = state_results.get(state, {}).get('tested', 'N/A')
        rate = state_results.get(state, {}).get('success_rate', 0)
        time_min = state_results.get(state, {}).get('time_minutes', 0)
        print(f"   {state}: {count:3d} wells ({rate:5.1f}% success, {time_min:4.1f} min)")
    
    # Data quality metrics
    print(f"\n📊 COMPREHENSIVE DATA QUALITY:")
    total_possible = len(combined_df) * len(combined_df.columns)
    actual_data = combined_df.count().sum()
    coverage = actual_data / total_possible * 100
    
    print(f"   Time coverage: {len(combined_df)} months ({combined_df.index[0].date()} to {combined_df.index[-1].date()})")
    print(f"   Data completeness: {coverage:.1f}% ({actual_data:,}/{total_possible:,} values)")
    print(f"   Average measurements per well: {actual_data/len(combined_df.columns):.1f}")
    
    # Reference period statistics
    avg_ref_months = metadata_df['n_months_reference'].mean()
    min_ref_months = metadata_df['n_months_reference'].min()
    max_ref_months = metadata_df['n_months_reference'].max()
    
    print(f"\n📊 REFERENCE PERIOD COVERAGE ({REFERENCE_START} to {REFERENCE_END}):")
    print(f"   Average months: {avg_ref_months:.1f}")
    print(f"   Range: {min_ref_months} to {max_ref_months} months")
    print(f"   Wells with ≥36 months: {(metadata_df['n_months_reference'] >= 36).sum():,}")
    print(f"   Wells with ≥24 months: {(metadata_df['n_months_reference'] >= 24).sum():,}")
    
    print(f"\n🚀 READY FOR COMPREHENSIVE VALIDATION WITH {len(all_data):,} WELLS!")
    print(f"This is a substantial dataset for robust model validation.")
    
    return len(all_data)  # Return number of wells for pipeline tracking

def export_openlandmap_soil(region):
    initialize_earth_engine()
    datasets = {
        'clay': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
        'sand': 'OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02',
        'silt': 'OpenLandMap/SOL/SOL_SILT-WFRACTION_USDA-3A1A1A_M/v02'
    }
    depths = {
        '0cm': 'b0',
        '10cm': 'b10',
        '30cm': 'b30',  # Focus on most relevant depths for groundwater
        '60cm': 'b60',
        '100cm': 'b100',
        '200cm': 'b200'
    }
    
    for prop, asset_id in datasets.items():
        for depth_label, band_name in depths.items():
            print(f"Exporting {prop} at {depth_label} (5km resolution)")
            try:
                img = ee.Image(asset_id).select(band_name)
                
                # Resample to 5km using mean
                img_5km = img.reduceResolution(
                    reducer=ee.Reducer.mean(),
                    maxPixels=1024
                )
                
                raw_dir = get_config('paths.raw_data', 'data/raw')
                scale = get_config('data_processing.export_scale', 5000)
                geemap.ee_export_image(
                    img_5km,
                    filename=os.path.join(raw_dir, "openlandmap", f"{prop}_{depth_label}_5km.tif"),
                    scale=scale,
                    region=region
                )
            except Exception as e:
                print(f"Failed to export {prop} at {depth_label}: {e}")

def export_landscan(region):
    """Export LandScan Global Population Dataset for the study region"""
    initialize_earth_engine()
    print("Exporting LandScan Global Population Dataset...")
    
    # Get config values
    start_date = get_config('data_processing.start_date', '2003-01-01')
    end_date = get_config('data_processing.end_date', '2022-12-31')
    scale = get_config('data_processing.export_scale', 5000)
    raw_dir = get_config('paths.raw_data', 'data/raw')
    
    # Get the collection and filter to our study period
    collection = ee.ImageCollection("projects/sat-io/open-datasets/ORNL/LANDSCAN_GLOBAL") \
        .filterDate(start_date, end_date) \
        .filterBounds(region) \
        .select("b1")
    
    # Export each year's data based on config date range
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    for year in range(start_year, end_year + 1):
        print(f"Processing year {year}...")
        try:
            # Get image for this year
            img = collection.filter(ee.Filter.calendarRange(year, year, 'year')).first()
            
            # Export the image with sum reducer to preserve population counts
            filename = f"{year}.tif"
            geemap.ee_export_image(
                img.reduceResolution(
                    reducer=ee.Reducer.sum(),
                    maxPixels=1024
                ),
                filename=os.path.join(raw_dir, "landscan", filename),
                scale=scale,
                region=region
            )
            print(f"✅ Exported {filename}")
        except Exception as e:
            print(f"❌ Failed to export {year}: {e}")
    
    print("✅ LandScan export complete")

def main():
    # Get available datasets from config
    all_datasets = get_config('download.default_datasets', ["grace", "gldas", "chirps", "modis", "terraclimate", "dem", "usgs", "openlandmap", "landscan"])
    default_region = get_config('download.default_region', 'mississippi')
    
    parser = argparse.ArgumentParser(description="Download datasets for GRACE downscaling")
    parser.add_argument("--download", nargs="+", choices=all_datasets + ["all"], required=True, help="Datasets to download")
    parser.add_argument("--region", type=str, default=default_region, help=f"Region name (default: {default_region})")
    args = parser.parse_args()

    region = get_region(args.region.lower())
    
    ensure_dirs()

    datasets_to_download = all_datasets if "all" in args.download else args.download

    if "grace" in datasets_to_download:
        export_grace(region)
    if "gldas" in datasets_to_download:
        export_gldas(region)
    if "chirps" in datasets_to_download:
        export_chirps(region)
    if "modis" in datasets_to_download:
        export_modis_landcover(region)
    if "terraclimate" in datasets_to_download:
        export_terraclimate(region)
    if "dem" in datasets_to_download:
        export_usgs_dem(region)
    if "usgs" in datasets_to_download:
        download_usgs_well_data()
    if "openlandmap" in datasets_to_download:
        export_openlandmap_soil(region)
    if "landscan" in datasets_to_download:
        export_landscan(region)

if __name__ == "__main__":
    main()
