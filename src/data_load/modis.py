"""
MODIS Land Cover Export Module for Google Earth Engine
This module submits MODIS land cover export tasks to Google Earth Engine for asynchronous processing
"""

import ee
import os
from datetime import datetime


def initialize_earth_engine():
    """Initialize Earth Engine with the project ID"""
    try:
        # Try to initialize directly first
        ee.Initialize(project='ee-sauravbhattarai1999')
        print("✅ Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Earth Engine initialization failed: {e}")
        try:
            print("🔄 Attempting to authenticate and initialize...")
            # First authenticate
            ee.Authenticate()
            # Then initialize with project
            ee.Initialize(project='ee-sauravbhattarai1999')
            print("✅ Earth Engine authenticated and initialized")
            return True
        except Exception as auth_error:
            print(f"❌ Authentication failed: {auth_error}")
            print("💡 Please run: earthengine authenticate")
            print("💡 Or visit: https://code.earthengine.google.com")
            return False


def submit_modis_export_task(region, scale=5000, description_suffix=""):
    """
    Submit a MODIS Land Cover export task to Google Earth Engine
    
    Args:
        region: ee.Geometry object defining the region of interest
        scale: Pixel scale in meters (default: 5000 for 5km)
        description_suffix: Optional suffix for task description
    
    Returns:
        dict: Task information including task_id and status
    """
    
    if not initialize_earth_engine():
        raise RuntimeError("Failed to initialize Earth Engine")
    
    # Get the MODIS Land Cover dataset
    collection = ee.ImageCollection("MODIS/061/MCD12Q1") \
        .select("LC_Type1") \
        .filterDate("2000-01-01", "2022-12-31") \
        .filterBounds(region)
    
    # Create mode composite across ALL years (most stable land cover)
    print("   🔄 Creating mode composite across all years (2000-2022)...")
    lc_mode_composite = collection.reduce(ee.Reducer.mode())
    
    # Set default projection before resampling (required for reduceResolution)
    lc_mode_composite = lc_mode_composite.setDefaultProjection(
        crs='EPSG:4326',
        scale=500  # Original MODIS resolution
    )
    
    # Resample to specified resolution using mode (preserve categorical nature)
    lc_resampled = lc_mode_composite.reduceResolution(
        reducer=ee.Reducer.mode(),
        maxPixels=65535  # Maximum allowed value
    )
    
    # Generate task description with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_description = f"MODIS_landcover_export_{timestamp}"
    if description_suffix:
        task_description += f"_{description_suffix}"
    
    # Create export task
    print(f"🚀 Submitting MODIS Land Cover export task: {task_description}")
    print(f"   📏 Scale: {scale}m ({scale/1000}km)")
    print(f"   📍 Region: {region.bounds().getInfo()}")
    print(f"   📅 Mode composite across all years (2000-2022) for stability")
    
    task = ee.batch.Export.image.toDrive(
        image=lc_resampled,
        description=task_description,
        folder='GRACE_Downscaling_MODIS',  # Google Drive folder
        fileNamePrefix=f'modis_landcover_mode_composite_{scale}m_{timestamp}',
        scale=scale,
        region=region,
        maxPixels=1e9,
        crs='EPSG:4326',
        fileFormat='GeoTIFF'
    )
    
    # Start the task
    task.start()
    
    # Get task information
    task_info = {
        'task_id': task.id,
        'description': task_description,
        'status': task.status()['state'],
        'scale_m': scale,
        'timestamp': timestamp,
        'google_drive_folder': 'GRACE_Downscaling_MODIS',
        'filename_prefix': f'modis_landcover_mode_composite_{scale}m_{timestamp}',
        'dataset': 'MODIS/061/MCD12Q1 LC_Type1'
    }
    
    print(f"✅ Task submitted successfully!")
    print(f"   🔧 Task ID: {task_info['task_id']}")
    print(f"   📊 Status: {task_info['status']}")
    print(f"   📁 Google Drive folder: {task_info['google_drive_folder']}")
    print(f"   📄 File prefix: {task_info['filename_prefix']}")
    print(f"   🗺️  Dataset: {task_info['dataset']}")
    print(f"\n💡 Instructions:")
    print(f"   1. Go to: https://code.earthengine.google.com/tasks")
    print(f"   2. Monitor task: {task_description}")
    print(f"   3. Download from Google Drive when complete")
    print(f"   4. Rename to: 2003_01_01.tif (features.py expects this name)")
    print(f"   5. Move file to: data/raw/modis_land_cover/2003_01_01.tif")
    
    return task_info


def submit_mississippi_modis_export():
    """Convenience function to export MODIS Land Cover for Mississippi Basin"""
    # Define Mississippi River Basin region (same as in data_loader.py)
    mississippi_region = ee.Geometry.Rectangle([-113.94, 28.84, -77.84, 49.74])
    
    return submit_modis_export_task(
        region=mississippi_region,
        scale=5000,
        description_suffix="mississippi"
    )


def check_task_status(task_id):
    """
    Check the status of a submitted task
    
    Args:
        task_id: The task ID returned from submit_modis_export_task
    
    Returns:
        dict: Current task status information
    """
    if not initialize_earth_engine():
        raise RuntimeError("Failed to initialize Earth Engine")
    
    try:
        task = ee.data.getTaskStatus(task_id)
        print(f"📊 Task Status for {task_id}:")
        print(f"   State: {task.get('state', 'UNKNOWN')}")
        print(f"   Description: {task.get('description', 'N/A')}")
        
        if task.get('state') == 'COMPLETED':
            print(f"   ✅ Task completed successfully!")
        elif task.get('state') == 'FAILED':
            print(f"   ❌ Task failed: {task.get('error_message', 'Unknown error')}")
        elif task.get('state') == 'RUNNING':
            print(f"   🔄 Task is running... Progress: {task.get('progress', 0)}%")
        elif task.get('state') == 'READY':
            print(f"   ⏳ Task is queued and ready to run")
        
        return task
    except Exception as e:
        print(f"❌ Failed to check task status: {e}")
        return None


def list_all_tasks():
    """List all Earth Engine tasks for this project"""
    if not initialize_earth_engine():
        raise RuntimeError("Failed to initialize Earth Engine")
    
    try:
        tasks = ee.data.getTaskList()
        print(f"📋 All Earth Engine Tasks (showing last 20):")
        print("-" * 80)
        
        for i, task in enumerate(tasks[:20]):
            status = task.get('state', 'UNKNOWN')
            desc = task.get('description', 'N/A')
            task_id = task.get('id', 'N/A')
            
            status_emoji = {
                'COMPLETED': '✅',
                'FAILED': '❌', 
                'RUNNING': '🔄',
                'READY': '⏳',
                'CANCELLED': '❌'
            }.get(status, '❓')
            
            print(f"{i+1:2d}. {status_emoji} {status:10s} | {desc[:40]:40s} | {task_id}")
        
        return tasks
    except Exception as e:
        print(f"❌ Failed to list tasks: {e}")
        return []


if __name__ == "__main__":
    """Example usage when run directly"""
    print("🗺️  MODIS Land Cover Export for GRACE Downscaling")
    print("=" * 60)
    
    try:
        # Submit export task for Mississippi Basin
        task_info = submit_mississippi_modis_export()
        
        # Show how to check status later
        print(f"\n🔍 To check task status later, use:")
        print(f"   from src.data_load.modis import check_task_status")
        print(f"   check_task_status('{task_info['task_id']}')")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")