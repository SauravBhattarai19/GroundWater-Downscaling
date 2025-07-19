#!/usr/bin/env python3
"""
Download Original GRACE Land Data
=================================

This script downloads the original GRACE Land data using the same approach 
as data_loader.py for reliable comparison with downscaled data.

Author: Assistant
Date: July 2025
"""

import os
import ee
import geemap
from pathlib import Path

def initialize_earth_engine():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project='ee-jsuhydrolabenb')
        print("✅ Earth Engine initialized successfully")
    except Exception as e:
        print("🔐 Authenticating Earth Engine...")
        try:
            ee.Authenticate()
            ee.Initialize()
            print("✅ Earth Engine authenticated and initialized")
        except Exception as auth_error:
            print(f"❌ Earth Engine authentication failed: {auth_error}")
            raise

def download_grace_land_data():
    """Download original GRACE Land data for comparison."""
    print("🛰️ Downloading original GRACE Land data...")
    
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Create output directory
    output_dir = Path("data/raw/grace_original")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mississippi River Basin bounds (same as data_loader.py)
    region = ee.Geometry.Rectangle([-113.94, 28.84, -77.84, 49.74])
    
    # Create GRACE Land collection (matching time range of downscaled data)
    grace_collection = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/LAND") \
        .filterDate("2003-01-01", "2022-12-31") \
        .filterBounds(region)
    
    # Check available data
    collection_size = grace_collection.size().getInfo()
    print(f"   📊 Found {collection_size} GRACE Land images")
    
    if collection_size == 0:
        print("   ❌ No GRACE data found for the specified time range")
        return False
    
    # Get the first image to check available bands
    first_image = ee.Image(grace_collection.first())
    band_names = first_image.bandNames().getInfo()
    print(f"   📝 Available bands: {band_names}")
    
    # Select all liquid water equivalent thickness bands
    lwe_bands = [band for band in band_names if 'lwe_thickness' in band]
    print(f"   🎯 Using LWE bands: {lwe_bands}")
    
    if not lwe_bands:
        print("   ❌ No LWE thickness bands found!")
        return False
    
    # Select the bands
    grace_selected = grace_collection.select(lwe_bands)
    
    # Calculate ensemble mean if multiple bands available
    if len(lwe_bands) > 1:
        grace_mean = grace_selected.map(lambda img: img.reduce(ee.Reducer.mean()).rename('lwe_thickness_mean'))
        print(f"   🧮 Calculating ensemble mean of {len(lwe_bands)} solutions")
        final_collection = grace_mean
    else:
        grace_renamed = grace_selected.select(lwe_bands[0]).map(lambda img: img.rename('lwe_thickness_mean'))
        print(f"   📊 Using single solution: {lwe_bands[0]}")
        final_collection = grace_renamed
    
    # Download using geemap (same approach as data_loader.py)
    print("   ⬇️ Downloading GRACE data files...")
    try:
        geemap.ee_export_image_collection(
            final_collection,
            out_dir=str(output_dir),
            scale=25000,  # 25km resolution similar to original GRACE
            region=region,
            file_per_band=False
        )
        print(f"   ✅ GRACE data downloaded to {output_dir}")
        return True
        
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    success = download_grace_land_data()
    if success:
        print("\n🎉 Download completed successfully!")
        print("Next step: Run grace_comparison_analysis.py to compare with downscaled data")
    else:
        print("\n❌ Download failed!")
