"""
Blocked Spatiotemporal Cross-Validation for GRACE Downscaling
Prevents both spatial and temporal data leakage with buffer zones
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from typing import Tuple, List, Dict, Generator, Optional
import warnings
warnings.filterwarnings('ignore')


class BlockedSpatiotemporalCV:
    """
    Implements blocked cross-validation preventing spatial and temporal leakage.
    
    Key features:
    - Spatial blocking using KMeans clustering
    - Temporal blocking with buffer periods
    - Spatial buffer zones around test regions
    - Ensures no data leakage between train/test
    """
    
    def __init__(self, 
                 n_spatial_blocks: int = 5,
                 n_temporal_blocks: int = 5, 
                 spatial_buffer_deg: float = 0.5,  # ~55km at mid-latitudes
                 temporal_buffer_months: int = 1,
                 random_state: int = 42):
        """
        Parameters:
        -----------
        n_spatial_blocks : int
            Number of spatial blocks for cross-validation
        n_temporal_blocks : int
            Number of temporal blocks for cross-validation
        spatial_buffer_deg : float
            Spatial buffer in degrees around test blocks (~55km at mid-latitudes)
        temporal_buffer_months : int
            Temporal buffer in months around test periods
        random_state : int
            Random seed for reproducibility
        """
        self.n_spatial_blocks = n_spatial_blocks
        self.n_temporal_blocks = n_temporal_blocks
        self.spatial_buffer = spatial_buffer_deg
        self.temporal_buffer = temporal_buffer_months
        self.random_state = random_state
        self.cluster_centers_ = None
        
    def create_spatial_blocks(self, spatial_coords: np.ndarray) -> np.ndarray:
        """
        Create spatial blocks using KMeans clustering.
        
        Parameters:
        -----------
        spatial_coords : np.ndarray, shape (n_samples, 2)
            Latitude and longitude for each sample
            
        Returns:
        --------
        np.ndarray, shape (n_samples,)
            Spatial block assignment for each sample
        """
        # Validate input
        if len(spatial_coords) == 0:
            raise ValueError("Cannot create spatial blocks: spatial_coords is empty!")
        
        if spatial_coords.shape[1] != 2:
            raise ValueError(f"Expected spatial_coords with 2 columns (lat, lon), got {spatial_coords.shape[1]}")
        
        # Remove duplicates for clustering (same location appears multiple times)
        unique_coords, inverse_indices = np.unique(
            spatial_coords, axis=0, return_inverse=True
        )
        
        n_unique = len(unique_coords)
        
        # Adjust number of clusters if we have fewer unique locations than requested
        n_clusters = min(self.n_spatial_blocks, n_unique)
        
        if n_clusters < self.n_spatial_blocks:
            print(f"   ⚠️ Warning: Only {n_unique} unique spatial locations, reducing spatial blocks from {self.n_spatial_blocks} to {n_clusters}")
        
        if n_unique == 0:
            raise ValueError("No unique spatial coordinates found after removing duplicates!")
        
        if n_clusters < 2:
            raise ValueError(f"Need at least 2 unique spatial locations for CV, got {n_unique}")
        
        # Cluster unique spatial locations
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        unique_blocks = kmeans.fit_predict(unique_coords)
        
        # Map back to all samples
        spatial_blocks = unique_blocks[inverse_indices]
        
        # Store cluster centers for buffer calculation
        self.cluster_centers_ = kmeans.cluster_centers_
        
        return spatial_blocks
    
    def create_temporal_blocks(self, temporal_indices: np.ndarray) -> Dict:
        """
        Create temporal blocks with gaps to prevent leakage.
        
        Parameters:
        -----------
        temporal_indices : np.ndarray
            Temporal index for each sample
            
        Returns:
        --------
        Dict mapping block_id to time indices
        """
        unique_times = np.unique(temporal_indices)
        n_times = len(unique_times)
        
        # Calculate block size with buffer
        block_size = n_times // self.n_temporal_blocks
        buffer_size = self.temporal_buffer
        
        temporal_blocks = {}
        for i in range(self.n_temporal_blocks):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size if i < self.n_temporal_blocks - 1 else n_times
            
            # Add buffer gap between blocks
            if i > 0:
                start_idx += buffer_size
            if i < self.n_temporal_blocks - 1:
                end_idx -= buffer_size
                
            if start_idx < end_idx:
                temporal_blocks[i] = unique_times[start_idx:end_idx]
            else:
                temporal_blocks[i] = np.array([])
        
        return temporal_blocks
    
    def create_spatial_buffer_mask(self, 
                                  spatial_coords: np.ndarray,
                                  test_block_id: int,
                                  spatial_blocks: np.ndarray) -> np.ndarray:
        """
        Create buffer mask around test spatial block.
        
        Parameters:
        -----------
        spatial_coords : np.ndarray
            Lat/lon coordinates
        test_block_id : int
            ID of test spatial block
        spatial_blocks : np.ndarray
            Spatial block assignments
            
        Returns:
        --------
        np.ndarray
            Boolean mask for buffer zone
        """
        # Get test block center
        test_center = self.cluster_centers_[test_block_id]
        
        # Calculate distances from all points to test center
        distances = cdist(spatial_coords, test_center.reshape(1, -1), metric='euclidean')
        
        # Points within buffer distance but not in test block
        buffer_mask = (
            (distances.ravel() < self.spatial_buffer) & 
            (spatial_blocks != test_block_id)
        )
        
        return buffer_mask
    
    def split(self, 
              X: np.ndarray, 
              y: np.ndarray,
              metadata: Dict) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test splits with spatiotemporal blocking.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray, shape (n_samples,)
            Target values
        metadata : Dict
            Must contain:
            - 'spatial_coords': (n_samples, 2) array of lat/lon
            - 'temporal_indices': (n_samples,) array of time indices
            
        Yields:
        -------
        train_idx, test_idx : np.ndarray
            Indices for training and testing
        """
        # Extract metadata
        spatial_coords = metadata['spatial_coords']
        temporal_indices = metadata['temporal_indices']
        
        # Validate inputs
        if len(X) == 0 or len(y) == 0:
            raise ValueError(f"Cannot perform CV: X and y are empty (X: {len(X)}, y: {len(y)})")
        
        if len(spatial_coords) == 0:
            raise ValueError("Cannot perform CV: spatial_coords is empty! Check that samples weren't all filtered out.")
        
        if len(temporal_indices) == 0:
            raise ValueError("Cannot perform CV: temporal_indices is empty!")
        
        if not (len(X) == len(y) == len(spatial_coords) == len(temporal_indices)):
            raise ValueError(
                f"Input dimensions mismatch: X={len(X)}, y={len(y)}, "
                f"spatial_coords={len(spatial_coords)}, temporal_indices={len(temporal_indices)}"
            )
        
        # Create spatial and temporal blocks
        print(f"📍 Creating {self.n_spatial_blocks} spatial blocks...")
        spatial_blocks = self.create_spatial_blocks(spatial_coords)
        
        print(f"📅 Creating {self.n_temporal_blocks} temporal blocks...")
        temporal_blocks = self.create_temporal_blocks(temporal_indices)
        
        # Generate CV folds
        fold_id = 0
        total_folds = self.n_spatial_blocks * self.n_temporal_blocks
        
        for spatial_test_block in range(self.n_spatial_blocks):
            for temporal_test_block in range(self.n_temporal_blocks):
                fold_id += 1
                
                # Create test mask
                spatial_test_mask = (spatial_blocks == spatial_test_block)
                temporal_test_mask = np.isin(
                    temporal_indices, 
                    temporal_blocks[temporal_test_block]
                )
                test_mask = spatial_test_mask & temporal_test_mask
                
                # Create spatial buffer mask
                spatial_buffer_mask = self.create_spatial_buffer_mask(
                    spatial_coords, spatial_test_block, spatial_blocks
                )
                
                # Create temporal buffer mask
                test_times = temporal_blocks[temporal_test_block]
                if len(test_times) > 0:
                    min_test_time = test_times.min()
                    max_test_time = test_times.max()
                    
                    temporal_buffer_mask = (
                        (temporal_indices >= min_test_time - self.temporal_buffer) &
                        (temporal_indices <= max_test_time + self.temporal_buffer) &
                        ~temporal_test_mask
                    )
                else:
                    temporal_buffer_mask = np.zeros(len(temporal_indices), dtype=bool)
                
                # Combine: test + all buffers are excluded from training
                exclude_mask = test_mask | spatial_buffer_mask | temporal_buffer_mask
                train_mask = ~exclude_mask
                
                # Get indices
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
                
                # Only yield fold if both train and test have samples
                if len(train_idx) > 0 and len(test_idx) > 0:
                    yield train_idx, test_idx
                else:
                    print(f"  ⚠️ Fold {fold_id}/{total_folds} skipped - insufficient samples")


def prepare_metadata_for_cv(X: np.ndarray, 
                           y: np.ndarray,
                           common_dates: List[str],
                           spatial_shape: Tuple[int, int],
                           feature_stack_path: Optional[str] = None) -> Dict:
    """
    Prepare metadata required for spatiotemporal CV from your feature stack.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix from model_manager
    y : np.ndarray
        Target values from model_manager
    common_dates : List[str]
        List of YYYY-MM dates corresponding to temporal dimension
    spatial_shape : Tuple[int, int]
        (n_lat, n_lon) shape of spatial grid
    feature_stack_path : str, optional
        Path to feature stack NetCDF (used to extract lat/lon if provided)
        
    Returns:
    --------
    Dict with spatial_coords and temporal_indices
    """
    n_lat, n_lon = spatial_shape
    n_times = len(common_dates)
    n_spatial = n_lat * n_lon
    
    # Load spatial coordinates
    if feature_stack_path is not None:
        try:
            import xarray as xr
            ds = xr.open_dataset(feature_stack_path)
            lat_values = ds.lat.values
            lon_values = ds.lon.values
        except:
            # Fallback if file doesn't exist or can't be loaded
            lat_values = np.linspace(-90, 90, n_lat)
            lon_values = np.linspace(-180, 180, n_lon)
    else:
        # Create default lat/lon grid
        lat_values = np.linspace(-90, 90, n_lat)
        lon_values = np.linspace(-180, 180, n_lon)
    
    # Create spatial coordinates for each sample
    # Samples are arranged as [time0_space0, time0_space1, ..., time1_space0, ...]
    lat_grid, lon_grid = np.meshgrid(lat_values, lon_values, indexing='ij')
    spatial_coords_2d = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    
    # Repeat spatial coords for each time
    spatial_coords = np.tile(spatial_coords_2d, (n_times, 1))
    
    # Create temporal indices
    temporal_indices = np.repeat(np.arange(n_times), n_spatial)
    
    # Filter to match valid samples in X, y
    # (X and y have NaN values removed by model_manager)
    assert len(X) == len(y), "X and y must have same length"
    n_valid = len(X)
    
    # Assuming valid samples are the first n_valid samples
    # (this matches how model_manager filters NaN)
    spatial_coords = spatial_coords[:n_valid]
    temporal_indices = temporal_indices[:n_valid]
    
    metadata = {
        'spatial_coords': spatial_coords,
        'temporal_indices': temporal_indices,
        'n_times': n_times,
        'n_spatial': n_spatial,
        'spatial_shape': spatial_shape,
        'common_dates': common_dates
    }
    
    return metadata


def evaluate_with_blocked_cv(model_class,
                            X: np.ndarray,
                            y: np.ndarray,
                            metadata: Dict,
                            model_params: Dict = None,
                            n_spatial_blocks: int = 5,
                            n_temporal_blocks: int = 4,
                            needs_scaling: bool = True,
                            n_jobs: int = -1) -> pd.DataFrame:
    """
    Main function to evaluate model with blocked spatiotemporal CV.
    
    Parameters:
    -----------
    model_class : class
        Model class (e.g., XGBRegressor)
    X, y : np.ndarray
        Features and targets
    metadata : Dict
        Metadata with spatial and temporal info
    model_params : Dict
        Model hyperparameters
    n_spatial_blocks : int
        Number of spatial CV blocks
    n_temporal_blocks : int
        Number of temporal CV blocks
    needs_scaling : bool
        Whether to scale features
    n_jobs : int
        Number of parallel jobs
        
    Returns:
    --------
    pd.DataFrame with CV results
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from joblib import Parallel, delayed
    
    # Initialize CV splitter
    cv_splitter = BlockedSpatiotemporalCV(
        n_spatial_blocks=n_spatial_blocks,
        n_temporal_blocks=n_temporal_blocks,
        spatial_buffer_deg=0.5,  # ~55km
        temporal_buffer_months=1
    )
    
    def train_fold(fold_data):
        """Train and evaluate single fold"""
        fold_id, train_idx, test_idx = fold_data
        
        # Prepare data
        if needs_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
        else:
            X_train = X[train_idx]
            X_test = X[test_idx]
        
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Train model
        model = model_class(**(model_params or {}))
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        return {
            'fold': fold_id,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
    
    # Prepare fold data
    print(f"\n🔄 Preparing CV folds...")
    folds = []
    for fold_id, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, metadata)):
        folds.append((fold_id + 1, train_idx, test_idx))
    
    print(f"✅ Generated {len(folds)} valid CV folds")
    
    # Run CV in parallel
    print(f"\n🚀 Running {len(folds)} CV folds (n_jobs={n_jobs})...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_fold)(fold_data) for fold_data in folds
    )
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 BLOCKED SPATIOTEMPORAL CV RESULTS:")
    print("="*60)
    print(f"Mean R²:   {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}")
    print(f"Mean RMSE: {results_df['rmse'].mean():.2f} ± {results_df['rmse'].std():.2f}")
    print(f"Mean MAE:  {results_df['mae'].mean():.2f} ± {results_df['mae'].std():.2f}")
    print("="*60)
    
    return results_df
