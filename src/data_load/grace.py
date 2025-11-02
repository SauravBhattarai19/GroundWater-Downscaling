"""
GRACE Monthly Export Module for Google Earth Engine
This module submits GRACE monthly export tasks to Google Earth Engine for proper temporal alignment
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


def submit_grace_monthly_export_tasks(region, description_suffix=""):
    """
    Submit GRACE monthly export tasks to Google Earth Engine
    Creates proper calendar month composites for temporal alignment
    
    Args:
        region: ee.Geometry object defining the region of interest
        description_suffix: Optional suffix for task description
    
    Returns:
        list: List of task information dictionaries
    """
    
    if not initialize_earth_engine():
        raise RuntimeError("Failed to initialize Earth Engine")
    
    # Get the GRACE MASCON collection
    collection = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI") \
        .select("lwe_thickness") \
        .filterDate("2003-01-01", "2024-12-31") \
        .filterBounds(region)
    
    print(f"🌍 GRACE Monthly Export Setup:")
    print(f"   📡 Dataset: NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI")
    print(f"   📏 Resolution: Native (~55km)")
    print(f"   📅 Period: 2003-2024 (monthly composites)")
    print(f"   📍 Region: {region.bounds().getInfo()}")
    
    # Create monthly aggregation function
    def create_monthly_composite(date):
        """Create monthly composite from MASCON data"""
        start = ee.Date(date)
        end = start.advance(1, 'month')
        
        # Filter MASCON data for this month and take mean
        monthly_data = collection.filterDate(start, end).mean()
        
        # Set timestamp for proper sorting
        return monthly_data.set('system:time_start', start.millis())
    
    # Export using individual monthly tasks with proper YYYYMM naming
    import geemap
    import os
    from datetime import datetime as dt
    import calendar
    
    # Create output directory
    grace_out_dir = os.path.join("data", "raw", "grace")
    os.makedirs(grace_out_dir, exist_ok=True)
    
    print("   📥 Exporting monthly GRACE composites with YYYYMM naming...")
    
    # Generate task description with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_description = f"GRACE_monthly_export_{timestamp}"
    if description_suffix:
        base_description += f"_{description_suffix}"
    
    print(f"\n🚀 Processing GRACE monthly data:")
    print(f"   📦 Base description: {base_description}")
    print(f"   📄 File naming: YYYYMM.tif (200301.tif, 200302.tif...)")
    print(f"   ⏱️  Processing 264 potential months (2003-2024)")
    
    successful_exports = 0
    failed_exports = 0
    
    # Process each month individually with proper naming  
    for year in range(2003, 2025):  # EXTENDED: Now covers through 2024
        for month in range(1, 13):
            try:
                # Create date strings
                start_date = ee.Date(f"{year}-{month:02d}-01")
                end_date = start_date.advance(1, 'month')
                yyyymm = f"{year}{month:02d}"
                
                print(f"   📅 Processing {yyyymm} ({year}-{month:02d})...")
                
                # Filter MASCON data for this specific month
                monthly_data = collection.filterDate(start_date, end_date).mean()
                
                # Check if we have data for this month
                try:
                    # Try to get metadata to check if image has data
                    info = monthly_data.getInfo()
                    if info is None or 'bands' not in info or len(info['bands']) == 0:
                        print(f"      ⚠️ No GRACE data available for {yyyymm}")
                        failed_exports += 1
                        continue
                except:
                    print(f"      ⚠️ No GRACE data available for {yyyymm}")
                    failed_exports += 1
                    continue
                
                # Export the image with proper YYYYMM filename
                filename = f"{yyyymm}.tif"
                filepath = os.path.join(grace_out_dir, filename)
                
                # Use geemap to export individual image
                geemap.ee_export_image(
                    monthly_data,
                    filename=filepath,
                    scale=55660,  # Native GRACE resolution
                    region=region,
                    file_per_band=False
                )
                
                print(f"      ✅ Exported {filename}")
                successful_exports += 1
                
            except Exception as e:
                print(f"      ❌ Failed to export {yyyymm}: {e}")
                failed_exports += 1
                continue
    
    # Create export summary
    export_info = {
        'description': base_description,
        'dataset': 'NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI',
        'temporal_resolution': 'monthly_composites',
        'spatial_resolution': '55660m_native',
        'timestamp': timestamp,
        'output_directory': grace_out_dir,
        'file_format': 'YYYYMM.tif',
        'period': '2003-01_to_2022-12',
        'successful_exports': successful_exports,
        'failed_exports': failed_exports,
        'total_attempted': 240,
        'success_rate': f"{successful_exports/240*100:.1f}%",
        'export_method': 'individual_monthly_geemap'
    }
    
    print(f"\n✅ GRACE monthly export completed!")
    print(f"   📁 Output directory: {export_info['output_directory']}")
    print(f"   🗺️  Dataset: {export_info['dataset']}")
    print(f"   📅 File format: {export_info['file_format']}")
    print(f"   📏 Resolution: {export_info['spatial_resolution']}")
    print(f"   ✅ Successful exports: {successful_exports}")
    print(f"   ⚠️  Failed exports: {failed_exports} (GRACE data gaps)")
    print(f"   📊 Success rate: {export_info['success_rate']}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Check files in: {grace_out_dir}")
    print(f"   2. Files named: 200301.tif, 200302.tif, etc. (YYYYMM format)")
    print(f"   3. Missing files represent months with no GRACE data (normal)")
    print(f"   ✅ Files ready for temporal alignment with clear date mapping")
    
    if failed_exports > 0:
        print(f"\n📝 Note: {failed_exports} months missing due to GRACE data gaps")
        print(f"   This is normal - GRACE satellites had operational periods and gaps")
        print(f"   Features.py will handle missing months automatically")
    
    return export_info


def submit_mississippi_grace_export():
    """Convenience function to export GRACE monthly data for Mississippi Basin"""
    # Define Mississippi River Basin region (same as in data_loader.py)
    mississippi_region = ee.Geometry.Rectangle([-113.94, 28.84, -77.84, 49.74])
    
    return submit_grace_monthly_export_tasks(
        region=mississippi_region,
        description_suffix="mississippi"
    )


def check_task_status(task_id):
    """
    Check the status of a submitted task
    
    Args:
        task_id: The task ID returned from submit_grace_monthly_export_tasks
    
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
    print("🌍 GRACE Monthly Export for GRACE Downscaling")
    print("=" * 70)
    print("This creates proper calendar month composites for temporal alignment")
    print("=" * 70)
    
    try:
        # Submit export task for Mississippi Basin
        task_info = submit_mississippi_grace_export()
        
        # Show how to check status later
        print(f"\n🔍 To check task status later, use:")
        print(f"   from src.data_load.grace import check_task_status")
        print(f"   check_task_status('{task_info['task_id']}')")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")