import argparse
import os
import ee
import geemap
from datetime import datetime
from dataretrieval import nwis
import pandas as pd

# Initialize Earth Engine
try:
    ee.Initialize(project = 'ee-jsuhydrolabenb')
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Study region: Mississippi River Basin (example bounding box)
REGIONS = {
    "mississippi": ee.Geometry.Rectangle([-113.94, 28.84, -77.84, 49.74])
}

# Output directories
RAW_DIR = "data/raw"
ALL_DATASETS = ["grace", "gldas", "chirps", "modis", "terraclimate", "dem", "usgs", "openlandmap", "landscan"]

def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    for sub in ["grace", "gldas", "chirps", "modis_land_cover", "terraclimate", "usgs_dem", "usgs_well_data", "openlandmap", "landscan"]:
        os.makedirs(os.path.join(RAW_DIR, sub), exist_ok=True)

def export_grace(region):
    collection = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI") \
        .select("lwe_thickness") \
        .filterDate("2003-01-01", "2022-12-31") \
        .filterBounds(region)

    print("Exporting GRACE...")
    geemap.ee_export_image_collection(
        collection,
        out_dir=os.path.join(RAW_DIR, "grace"),
        scale=50000,
        region=region,
        file_per_band=False
    )

def export_gldas(region):
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
        monthly = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H") \
            .filterDate("2003-01-01", "2022-12-31") \
            .filterBounds(region) \
            .select(var)

        def monthly_composite(date):
            start = ee.Date(date)
            end = start.advance(1, 'month')
            return monthly.filterDate(start, end).mean().set('system:time_start', start.millis())

        months = ee.List.sequence(0, 12 * 20 - 1).map(lambda i: ee.Date("2003-01-01").advance(i, 'month'))
        monthly_collection = ee.ImageCollection(months.map(monthly_composite))

        geemap.ee_export_image_collection(
            monthly_collection,
            out_dir=os.path.join(RAW_DIR, "gldas", var),
            scale=25000,
            region=region,
            file_per_band=False
        )

def export_chirps(region):
    print("Aggregating and exporting CHIRPS monthly totals...")
    collection = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate("2003-01-01", "2022-12-31") \
        .filterBounds(region)

    def monthly_sum(date):
        start = ee.Date(date)
        end = start.advance(1, 'month')
        return collection.filterDate(start, end).sum().set('system:time_start', start.millis())

    months = ee.List.sequence(0, 12 * 20 - 1).map(lambda i: ee.Date("2003-01-01").advance(i, 'month'))
    monthly_collection = ee.ImageCollection(months.map(monthly_sum))

    geemap.ee_export_image_collection(
        monthly_collection,
        out_dir=os.path.join(RAW_DIR, "chirps"),
        scale=5000,
        region=region,
        file_per_band=False
    )

def export_modis_landcover(region):
    collection = ee.ImageCollection("MODIS/061/MCD12Q1") \
        .select("LC_Type1") \
        .filterDate("2000-01-01", "2022-12-31") \
        .filterBounds(region)

    print("Exporting MODIS Land Cover...")
    geemap.ee_export_image_collection(
        collection,
        out_dir=os.path.join(RAW_DIR, "modis_land_cover"),
        scale=2000,
        region=region,
        file_per_band=False
    )

def export_terraclimate(region):
    collection = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE") \
        .filterDate("2003-01-01", "2022-12-31") \
        .filterBounds(region)

    bands = ["tmmx", "tmmn", "pr", "aet", "def"]

    for band in bands:
        print(f"Exporting TerraClimate: {band}")
        subset = collection.select(band)
        geemap.ee_export_image_collection(
            subset,
            out_dir=os.path.join(RAW_DIR, "terraclimate", band),
            scale=4000,
            region=region,
            file_per_band=False
        )

def export_usgs_dem(region):
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")

    print("Exporting USGS DEM...")
    geemap.ee_export_image(
        dem,
        filename=os.path.join(RAW_DIR, "usgs_dem", "srtm_dem.tif"),
        scale=750,
        region=region
    )

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
    
    # SAME reference period as GRACE
    REFERENCE_START = "2004-01"
    REFERENCE_END = "2009-12"
    
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
    combined_df.to_csv(os.path.join(RAW_DIR, "usgs_well_data", "monthly_groundwater_anomalies_cm.csv"))
    
    # Save comprehensive metadata
    print("💾 Saving comprehensive well metadata...")
    metadata_df = pd.DataFrame(well_metadata)
    metadata_df.to_csv(os.path.join(RAW_DIR, "usgs_well_data", "well_metadata.csv"), index=False)
    
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
    datasets = {
        'clay': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
        'sand': 'OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02',
        'silt': 'OpenLandMap/SOL/SOL_SILT-WFRACTION_USDA-3A1A1A_M/v02'
    }
    depths = {
        '0cm': 'b0',
        '10cm': 'b10',
        '30cm': 'b30',
        '60cm': 'b60',
        '100cm': 'b100',
        '200cm': 'b200'
    }
    for prop, asset_id in datasets.items():
        for depth_label, band_name in depths.items():
            print(f"Exporting {prop} at {depth_label}")
            try:
                img = ee.Image(asset_id).select(band_name).clip(region).reproject(crs='EPSG:4326', scale=250)
                geemap.ee_export_image(
                    img,
                    filename=os.path.join(RAW_DIR, "openlandmap", f"{prop}_{depth_label}.tif"),
                    scale=750,
                    region=region.bounds()
                )
            except Exception as e:
                print(f"Failed to export {prop} at {depth_label}: {e}")

def export_landscan(region):
    """Export LandScan Global Population Dataset for the study region"""
    print("Exporting LandScan Global Population Dataset...")
    
    # Get the collection and filter to our study period
    collection = ee.ImageCollection("projects/sat-io/open-datasets/ORNL/LANDSCAN_GLOBAL") \
        .filterDate("2003-01-01", "2022-12-31") \
        .filterBounds(region) \
        .select("b1")
    
    # Export each year's data
    for year in range(2003, 2023):
        print(f"Processing year {year}...")
        try:
            # Get image for this year
            img = collection.filter(ee.Filter.calendarRange(year, year, 'year')).first()
            
            # Export the image
            filename = f"{year}.tif"
            geemap.ee_export_image(
                img,
                filename=os.path.join(RAW_DIR, "landscan", filename),
                scale=1000,  # 1km resolution as specified
                region=region
            )
            print(f"✅ Exported {filename}")
        except Exception as e:
            print(f"❌ Failed to export {year}: {e}")
    
    print("✅ LandScan export complete")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for GRACE downscaling")
    parser.add_argument("--download", nargs="+", choices=ALL_DATASETS + ["all"], required=True, help="Datasets to download")
    parser.add_argument("--region", type=str, default="mississippi", help="Region name (default: mississippi)")
    args = parser.parse_args()

    region = REGIONS.get(args.region.lower())
    if not region:
        raise ValueError(f"Unknown region: {args.region}")

    ensure_dirs()

    datasets_to_download = ALL_DATASETS if "all" in args.download else args.download

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
