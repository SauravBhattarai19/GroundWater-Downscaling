"""
Scale-Consistent Neural Network for GRACE Downscaling

This module implements a PyTorch-based MLP with a custom dual-scale loss function:
- Prediction loss (MSE at 55km coarse resolution)
- Consistency loss (aggregated 5km fine predictions vs GRACE observations)

The consistency loss ensures that when fine-scale predictions are aggregated back
to coarse scale, they match the original GRACE observations. This eliminates the
need for post-hoc residual correction.

Mathematical Formulation:
    L_total = L_pred + λ * L_cons
    
    L_pred = (1/M) * Σ (G_i - ŷ_i^coarse)²
    
    L_cons = (1/M) * Σ (G_i - (1/n_i) * Σ_{j∈C_i} ŷ_j^fine)²

where:
    G_i = GRACE observation for coarse cell i
    ŷ_i^coarse = model prediction at coarse resolution
    ŷ_j^fine = model prediction at fine resolution
    C_i = set of fine cells within coarse cell i
    n_i = number of fine cells in coarse cell i
    λ = hyperparameter controlling consistency strength
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ScaleConsistentMLP(nn.Module):
    """
    Multi-Layer Perceptron for scale-consistent TWS prediction.
    
    Architecture matches the original sklearn MLPRegressor but implemented
    in PyTorch to enable custom loss functions.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_layers: List[int] = [512, 256, 128],
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize the MLP.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_layers : List[int]
            Sizes of hidden layers (default: [512, 256, 128])
        dropout : float
            Dropout rate between layers (default: 0.1)
        activation : str
            Activation function ('relu', 'leaky_relu', 'elu')
        """
        super(ScaleConsistentMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_layers):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization for stability
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'elu':
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer (single output for TWS prediction)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim)
        
        Returns:
        --------
        torch.Tensor
            Predictions of shape (batch_size,)
        """
        output = self.network(x)
        return output.squeeze(-1)  # Remove last dimension
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict for numpy array input (sklearn-compatible interface).
        
        Parameters:
        -----------
        x : np.ndarray
            Input features
        
        Returns:
        --------
        np.ndarray
            Predictions
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            if next(self.parameters()).is_cuda:
                x_tensor = x_tensor.cuda()
            predictions = self.forward(x_tensor)
            return predictions.cpu().numpy()


class ScaleConsistentLoss(nn.Module):
    """
    Custom loss function for scale-consistent training.
    
    Combines:
    - Prediction loss: MSE between coarse predictions and GRACE
    - Consistency loss: MSE between aggregated fine predictions and GRACE
    
    L_total = L_pred + λ * L_cons
    """
    
    def __init__(self, 
                 lambda_consistency: float = 1.0,
                 reduction: str = 'mean'):
        """
        Initialize the loss function.
        
        Parameters:
        -----------
        lambda_consistency : float
            Weight for consistency loss (default: 1.0)
        reduction : str
            Reduction method ('mean', 'sum', 'none')
        """
        super(ScaleConsistentLoss, self).__init__()
        
        self.lambda_consistency = lambda_consistency
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self,
                y_pred_coarse: torch.Tensor,
                y_true: torch.Tensor,
                y_pred_fine: Optional[torch.Tensor] = None,
                coarse_to_fine_mapping: Optional[List[List[int]]] = None,
                fine_valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the scale-consistent loss.
        
        Parameters:
        -----------
        y_pred_coarse : torch.Tensor
            Predictions at coarse resolution (batch_size,)
        y_true : torch.Tensor
            GRACE observations at coarse resolution (batch_size,)
        y_pred_fine : torch.Tensor, optional
            Predictions at fine resolution (n_fine_samples,)
        coarse_to_fine_mapping : List[List[int]], optional
            Mapping from coarse cell index to list of fine cell indices
        fine_valid_mask : torch.Tensor, optional
            Boolean mask for valid fine predictions
        
        Returns:
        --------
        Tuple[torch.Tensor, Dict[str, float]]
            Total loss and dictionary of individual loss components
        """
        # Prediction loss (MSE at coarse scale)
        loss_pred = self.mse(y_pred_coarse, y_true)
        
        loss_components = {
            'loss_pred': loss_pred.item(),
            'loss_cons': 0.0,
            'loss_total': loss_pred.item()
        }
        
        # Consistency loss (if fine predictions provided)
        if y_pred_fine is not None and coarse_to_fine_mapping is not None:
            loss_cons = self._compute_consistency_loss(
                y_pred_fine, 
                y_true, 
                coarse_to_fine_mapping,
                fine_valid_mask
            )
            
            # Total loss
            loss_total = loss_pred + self.lambda_consistency * loss_cons
            
            loss_components['loss_cons'] = loss_cons.item()
            loss_components['loss_total'] = loss_total.item()
            
            return loss_total, loss_components
        
        return loss_pred, loss_components
    
    def _compute_consistency_loss(self,
                                  y_pred_fine: torch.Tensor,
                                  y_true_coarse: torch.Tensor,
                                  coarse_to_fine_mapping: List[List[int]],
                                  fine_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute consistency loss: MSE between aggregated fine predictions and GRACE.
        
        L_cons = (1/M) * Σ (G_i - (1/n_i) * Σ_{j∈C_i} ŷ_j^fine)²
        
        Parameters:
        -----------
        y_pred_fine : torch.Tensor
            Fine-scale predictions
        y_true_coarse : torch.Tensor
            Coarse-scale GRACE observations
        coarse_to_fine_mapping : List[List[int]]
            Mapping from coarse index to fine indices
        fine_valid_mask : torch.Tensor, optional
            Mask for valid fine cells
        
        Returns:
        --------
        torch.Tensor
            Consistency loss value
        """
        batch_size = len(y_true_coarse)
        aggregated_predictions = torch.zeros(batch_size, device=y_pred_fine.device)
        valid_coarse_count = 0
        
        for i in range(batch_size):
            fine_indices = coarse_to_fine_mapping[i]
            
            if len(fine_indices) > 0:
                # Get fine predictions for this coarse cell
                fine_preds = y_pred_fine[fine_indices]
                
                # Apply validity mask if provided
                if fine_valid_mask is not None:
                    fine_mask = fine_valid_mask[fine_indices]
                    if fine_mask.sum() > 0:
                        fine_preds = fine_preds[fine_mask]
                        aggregated_predictions[i] = fine_preds.mean()
                        valid_coarse_count += 1
                    else:
                        # No valid fine cells, use NaN (will be masked later)
                        aggregated_predictions[i] = float('nan')
                else:
                    # Average all fine predictions
                    aggregated_predictions[i] = fine_preds.mean()
                    valid_coarse_count += 1
            else:
                aggregated_predictions[i] = float('nan')
        
        # Mask out invalid aggregations
        valid_mask = ~torch.isnan(aggregated_predictions)
        
        if valid_mask.sum() > 0:
            loss_cons = self.mse(
                aggregated_predictions[valid_mask],
                y_true_coarse[valid_mask]
            )
        else:
            # No valid consistency comparisons
            loss_cons = torch.tensor(0.0, device=y_pred_fine.device)
        
        return loss_cons


class DualScaleDataset(Dataset):
    """
    PyTorch Dataset for dual-scale (coarse + fine) training.
    
    Provides both coarse and fine features for each sample, along with
    the mapping between coarse cells and their corresponding fine cells.
    """
    
    def __init__(self,
                 X_coarse: np.ndarray,
                 y_coarse: np.ndarray,
                 X_fine: np.ndarray,
                 coarse_to_fine_mapping: List[List[int]],
                 fine_valid_mask: Optional[np.ndarray] = None,
                 coarse_indices: Optional[np.ndarray] = None):
        """
        Initialize the dataset.
        
        Parameters:
        -----------
        X_coarse : np.ndarray
            Coarse features (n_coarse_samples, n_features)
        y_coarse : np.ndarray
            Coarse targets / GRACE (n_coarse_samples,)
        X_fine : np.ndarray
            Fine features (n_fine_samples, n_features)
        coarse_to_fine_mapping : List[List[int]]
            Maps each coarse index to list of fine indices
        fine_valid_mask : np.ndarray, optional
            Boolean mask for valid fine samples
        coarse_indices : np.ndarray, optional
            Subset of coarse indices to use (for train/test split)
        """
        self.X_coarse = torch.FloatTensor(X_coarse)
        self.y_coarse = torch.FloatTensor(y_coarse)
        self.X_fine = torch.FloatTensor(X_fine)
        self.coarse_to_fine_mapping = coarse_to_fine_mapping
        
        if fine_valid_mask is not None:
            self.fine_valid_mask = torch.BoolTensor(fine_valid_mask)
        else:
            self.fine_valid_mask = torch.ones(len(X_fine), dtype=torch.bool)
        
        # Use specified coarse indices or all
        if coarse_indices is not None:
            self.coarse_indices = coarse_indices
        else:
            self.coarse_indices = np.arange(len(X_coarse))
    
    def __len__(self) -> int:
        return len(self.coarse_indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns both coarse and corresponding fine data.
        """
        coarse_idx = self.coarse_indices[idx]
        
        return {
            'X_coarse': self.X_coarse[coarse_idx],
            'y_coarse': self.y_coarse[coarse_idx],
            'coarse_idx': coarse_idx
        }


def create_coarse_to_fine_mapping(
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
    fine_lat: np.ndarray,
    fine_lon: np.ndarray,
    n_times: int,
    aggregation_factor: int = 11
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Create mapping from coarse cell indices to fine cell indices.
    
    This uses a NEAREST-NEIGHBOR ASSIGNMENT approach:
    - For each fine cell, find the closest coarse cell
    - Invert this to get: for each coarse cell, which fine cells belong to it
    
    This is scientifically more accurate when grids are not perfectly aligned,
    as it assigns each fine cell to exactly one coarse cell based on proximity.
    
    Parameters:
    -----------
    coarse_lat : np.ndarray
        Coarse latitude values (cell centers)
    coarse_lon : np.ndarray
        Coarse longitude values (cell centers)
    fine_lat : np.ndarray
        Fine latitude values (cell centers)
    fine_lon : np.ndarray
        Fine longitude values (cell centers)
    n_times : int
        Number of time steps
    aggregation_factor : int
        Expected spatial aggregation factor (for reporting only)
    
    Returns:
    --------
    Tuple[List[List[int]], np.ndarray]
        - coarse_to_fine_mapping: List of fine indices for each coarse sample
        - fine_cell_counts: Number of fine cells per coarse cell
    """
    print("🗺️ Creating coarse-to-fine spatial mapping (nearest-neighbor)...")
    
    n_lat_coarse = len(coarse_lat)
    n_lon_coarse = len(coarse_lon)
    n_lat_fine = len(fine_lat)
    n_lon_fine = len(fine_lon)
    
    n_coarse_spatial = n_lat_coarse * n_lon_coarse
    n_fine_spatial = n_lat_fine * n_lon_fine
    n_coarse_total = n_coarse_spatial * n_times
    
    print(f"   Coarse grid: {n_lat_coarse} × {n_lon_coarse} = {n_coarse_spatial:,} cells")
    print(f"   Fine grid: {n_lat_fine} × {n_lon_fine} = {n_fine_spatial:,} cells")
    
    # Calculate grid spacings
    coarse_lat_spacing = abs(coarse_lat[1] - coarse_lat[0]) if len(coarse_lat) > 1 else 0.5
    coarse_lon_spacing = abs(coarse_lon[1] - coarse_lon[0]) if len(coarse_lon) > 1 else 0.5
    fine_lat_spacing = abs(fine_lat[1] - fine_lat[0]) if len(fine_lat) > 1 else 0.05
    fine_lon_spacing = abs(fine_lon[1] - fine_lon[0]) if len(fine_lon) > 1 else 0.05
    
    print(f"   Coarse spacing: {coarse_lat_spacing:.4f}° lat, {coarse_lon_spacing:.4f}° lon")
    print(f"   Fine spacing: {fine_lat_spacing:.5f}° lat, {fine_lon_spacing:.5f}° lon")
    print(f"   Effective ratio: {coarse_lat_spacing/fine_lat_spacing:.2f}× lat, {coarse_lon_spacing/fine_lon_spacing:.2f}× lon")
    
    # For each fine cell, find the nearest coarse cell
    # Use vectorized operations for efficiency
    print("   Computing fine-to-coarse assignments...")
    
    # Create meshgrid of fine coordinates
    fine_lat_grid, fine_lon_grid = np.meshgrid(fine_lat, fine_lon, indexing='ij')
    fine_lat_flat = fine_lat_grid.flatten()
    fine_lon_flat = fine_lon_grid.flatten()
    
    # For each fine cell, find nearest coarse lat and lon index
    # Using searchsorted for efficiency
    coarse_lat_sorted = np.sort(coarse_lat)[::-1] if coarse_lat[0] > coarse_lat[-1] else np.sort(coarse_lat)
    coarse_lon_sorted = np.sort(coarse_lon)
    
    # Handle descending lat (common in geospatial data)
    lat_descending = coarse_lat[0] > coarse_lat[-1]
    
    # Assign each fine cell to nearest coarse cell
    fine_to_coarse_lat = np.zeros(n_fine_spatial, dtype=np.int32)
    fine_to_coarse_lon = np.zeros(n_fine_spatial, dtype=np.int32)
    
    for i_fine in range(n_lat_fine):
        # Find nearest coarse lat index
        lat_diffs = np.abs(coarse_lat - fine_lat[i_fine])
        nearest_lat_idx = np.argmin(lat_diffs)
        
        for j_fine in range(n_lon_fine):
            fine_idx = i_fine * n_lon_fine + j_fine
            fine_to_coarse_lat[fine_idx] = nearest_lat_idx
    
    for j_fine in range(n_lon_fine):
        # Find nearest coarse lon index
        lon_diffs = np.abs(coarse_lon - fine_lon[j_fine])
        nearest_lon_idx = np.argmin(lon_diffs)
        
        for i_fine in range(n_lat_fine):
            fine_idx = i_fine * n_lon_fine + j_fine
            fine_to_coarse_lon[fine_idx] = nearest_lon_idx
    
    # Convert (lat_idx, lon_idx) to flattened coarse index
    fine_to_coarse_spatial = fine_to_coarse_lat * n_lon_coarse + fine_to_coarse_lon
    
    # Invert: for each coarse cell, get list of fine cells
    print("   Building coarse-to-fine mapping...")
    spatial_mapping = [[] for _ in range(n_coarse_spatial)]
    
    for fine_idx in range(n_fine_spatial):
        coarse_idx = fine_to_coarse_spatial[fine_idx]
        spatial_mapping[coarse_idx].append(fine_idx)
    
    # Expand spatial mapping to all time steps
    coarse_to_fine_mapping = []
    fine_cell_counts = np.zeros(n_coarse_total, dtype=np.int32)
    
    for t in range(n_times):
        coarse_time_offset = t * n_coarse_spatial
        fine_time_offset = t * n_fine_spatial
        
        for coarse_spatial_idx, fine_spatial_indices in enumerate(spatial_mapping):
            coarse_idx = coarse_time_offset + coarse_spatial_idx
            
            # Add time offset to fine indices
            fine_indices = [fine_time_offset + idx for idx in fine_spatial_indices]
            
            coarse_to_fine_mapping.append(fine_indices)
            fine_cell_counts[coarse_idx] = len(fine_indices)
    
    # Statistics
    spatial_counts = np.array([len(m) for m in spatial_mapping])
    avg_fine_per_coarse = np.mean(spatial_counts)
    min_fine_per_coarse = np.min(spatial_counts)
    max_fine_per_coarse = np.max(spatial_counts)
    
    print(f"\n   📊 Mapping Statistics:")
    print(f"   Fine cells per coarse: {avg_fine_per_coarse:.1f} avg (min={min_fine_per_coarse}, max={max_fine_per_coarse})")
    print(f"   Expected (if aligned): {aggregation_factor**2}")
    print(f"   Total coarse cells: {n_coarse_total:,}")
    print(f"   Total fine cells mapped: {sum(len(m) for m in coarse_to_fine_mapping):,}")
    
    # Check for unmapped fine cells (should be 0 with nearest-neighbor)
    all_mapped_fine = set()
    for m in spatial_mapping:
        all_mapped_fine.update(m)
    unmapped = n_fine_spatial - len(all_mapped_fine)
    if unmapped > 0:
        print(f"   ⚠️ Warning: {unmapped} fine cells not mapped to any coarse cell")
    else:
        print(f"   ✅ All {n_fine_spatial:,} fine cells mapped (each to exactly one coarse cell)")
    
    return coarse_to_fine_mapping, fine_cell_counts


class ScaleConsistentTrainer:
    """
    Trainer for the scale-consistent neural network.
    
    Handles the complete training loop including:
    - Data loading at both resolutions
    - Custom loss computation
    - Early stopping
    - Model checkpointing
    """
    
    def __init__(self,
                 config: Dict,
                 device: Optional[str] = None):
        """
        Initialize the trainer.
        
        Parameters:
        -----------
        config : Dict
            Configuration dictionary
        device : str, optional
            Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.config = config
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Scale-Consistent Trainer initialized")
        print(f"   Device: {self.device}")
        
        # Model and training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.scaler = None  # For feature scaling
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_loss_pred': [],
            'train_loss_cons': [],
            'val_loss_pred': [],
            'val_loss_cons': []
        }
    
    def _get_hyperparams(self) -> Dict:
        """Get hyperparameters from config."""
        default_params = {
            'hidden_layers': [512, 256, 128],
            'learning_rate': 0.001,
            'batch_size': 1024,
            'max_epochs': 2000,
            'early_stopping_patience': 50,
            'lambda_consistency': 1.0,
            'dropout': 0.1,
            'weight_decay': 0.0001,
            'lr_scheduler_patience': 20,
            'lr_scheduler_factor': 0.5
        }
        
        # Get from config if available
        config_params = self.config.get('models', {}).get('hyperparameters', {}).get('nn_scale_consistent', {})
        
        # Merge with defaults
        params = {**default_params, **config_params}
        
        return params
    
    def prepare_data(self,
                    X_coarse: np.ndarray,
                    y_coarse: np.ndarray,
                    X_fine: np.ndarray,
                    coarse_to_fine_mapping: List[List[int]],
                    fine_valid_mask: Optional[np.ndarray] = None,
                    train_fraction: float = 0.7,
                    random_state: int = 42) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders.
        
        Parameters:
        -----------
        X_coarse : np.ndarray
            Coarse features
        y_coarse : np.ndarray
            Coarse targets (GRACE)
        X_fine : np.ndarray
            Fine features
        coarse_to_fine_mapping : List[List[int]]
            Coarse to fine index mapping
        fine_valid_mask : np.ndarray, optional
            Validity mask for fine samples
        train_fraction : float
            Fraction of data for training
        random_state : int
            Random seed
        
        Returns:
        --------
        Tuple[DataLoader, DataLoader]
            Training and validation data loaders
        """
        params = self._get_hyperparams()
        
        # Split indices
        n_samples = len(X_coarse)
        np.random.seed(random_state)
        indices = np.random.permutation(n_samples)
        
        n_train = int(n_samples * train_fraction)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        print(f"📦 Preparing data loaders...")
        print(f"   Training samples: {len(train_indices):,}")
        print(f"   Validation samples: {len(val_indices):,}")
        
        # Apply feature scaling
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()
        X_coarse_scaled = self.scaler.fit_transform(X_coarse)
        X_fine_scaled = self.scaler.transform(X_fine)
        
        # Create datasets
        train_dataset = DualScaleDataset(
            X_coarse_scaled, y_coarse, X_fine_scaled,
            coarse_to_fine_mapping, fine_valid_mask, train_indices
        )
        
        val_dataset = DualScaleDataset(
            X_coarse_scaled, y_coarse, X_fine_scaled,
            coarse_to_fine_mapping, fine_valid_mask, val_indices
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Store data for consistency loss computation
        # IMPORTANT: Keep fine data on CPU to avoid GPU OOM - only move batches to GPU during training
        self.X_fine_tensor = torch.FloatTensor(X_fine_scaled)  # Keep on CPU!
        self.coarse_to_fine_mapping = coarse_to_fine_mapping
        
        if fine_valid_mask is not None:
            self.fine_valid_mask = torch.BoolTensor(fine_valid_mask)  # Keep on CPU!
        else:
            self.fine_valid_mask = torch.ones(len(X_fine), dtype=torch.bool)
        
        print(f"   Fine data kept on CPU ({self.X_fine_tensor.shape[0]:,} samples × {self.X_fine_tensor.shape[1]} features)")
        print(f"   Memory: ~{self.X_fine_tensor.numel() * 4 / 1e9:.2f} GB (will batch-transfer to GPU)")
        
        return train_loader, val_loader
    
    def build_model(self, input_dim: int):
        """
        Build the neural network model.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        """
        params = self._get_hyperparams()
        
        self.model = ScaleConsistentMLP(
            input_dim=input_dim,
            hidden_layers=params['hidden_layers'],
            dropout=params['dropout']
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=params['lr_scheduler_factor'],
            patience=params['lr_scheduler_patience'],
            verbose=True
        )
        
        self.loss_fn = ScaleConsistentLoss(
            lambda_consistency=params['lambda_consistency']
        )
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"🏗️ Model built:")
        print(f"   Architecture: {input_dim} → {params['hidden_layers']} → 1")
        print(f"   Parameters: {n_params:,}")
        print(f"   λ_consistency: {params['lambda_consistency']}")
    
    def train_epoch(self, train_loader: DataLoader, use_consistency: bool = True) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        use_consistency : bool
            Whether to compute consistency loss
        
        Returns:
        --------
        Dict[str, float]
            Loss metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_loss_pred = 0.0
        total_loss_cons = 0.0
        n_batches = 0
        
        for batch in train_loader:
            X_coarse = batch['X_coarse'].to(self.device)
            y_coarse = batch['y_coarse'].to(self.device)
            coarse_indices = batch['coarse_idx'].numpy()
            
            # Forward pass - coarse predictions
            self.optimizer.zero_grad()
            y_pred_coarse = self.model(X_coarse)
            
            # Compute fine predictions for consistency loss
            if use_consistency:
                # Get fine indices for this batch
                batch_fine_indices = []
                batch_mapping = []
                
                for i, coarse_idx in enumerate(coarse_indices):
                    fine_indices = self.coarse_to_fine_mapping[coarse_idx]
                    batch_mapping.append(list(range(len(batch_fine_indices), 
                                                    len(batch_fine_indices) + len(fine_indices))))
                    batch_fine_indices.extend(fine_indices)
                
                if len(batch_fine_indices) > 0:
                    # Get fine features for this batch and move to GPU
                    X_fine_batch = self.X_fine_tensor[batch_fine_indices].to(self.device)
                    fine_valid_batch = self.fine_valid_mask[batch_fine_indices].to(self.device)
                    
                    # Forward pass - fine predictions
                    y_pred_fine = self.model(X_fine_batch)
                    
                    # Compute loss
                    loss, loss_components = self.loss_fn(
                        y_pred_coarse, y_coarse,
                        y_pred_fine, batch_mapping, fine_valid_batch
                    )
                else:
                    # No fine cells, just use prediction loss
                    loss, loss_components = self.loss_fn(y_pred_coarse, y_coarse)
            else:
                # Only prediction loss
                loss, loss_components = self.loss_fn(y_pred_coarse, y_coarse)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss_components['loss_total']
            total_loss_pred += loss_components['loss_pred']
            total_loss_cons += loss_components['loss_cons']
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'loss_pred': total_loss_pred / n_batches,
            'loss_cons': total_loss_cons / n_batches
        }
    
    def validate(self, val_loader: DataLoader, use_consistency: bool = True) -> Dict[str, float]:
        """
        Validate the model.
        
        Parameters:
        -----------
        val_loader : DataLoader
            Validation data loader
        use_consistency : bool
            Whether to compute consistency loss
        
        Returns:
        --------
        Dict[str, float]
            Validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_loss_pred = 0.0
        total_loss_cons = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                X_coarse = batch['X_coarse'].to(self.device)
                y_coarse = batch['y_coarse'].to(self.device)
                coarse_indices = batch['coarse_idx'].numpy()
                
                # Forward pass - coarse predictions
                y_pred_coarse = self.model(X_coarse)
                
                # Compute fine predictions for consistency loss
                if use_consistency:
                    batch_fine_indices = []
                    batch_mapping = []
                    
                    for i, coarse_idx in enumerate(coarse_indices):
                        fine_indices = self.coarse_to_fine_mapping[coarse_idx]
                        batch_mapping.append(list(range(len(batch_fine_indices), 
                                                        len(batch_fine_indices) + len(fine_indices))))
                        batch_fine_indices.extend(fine_indices)
                    
                    if len(batch_fine_indices) > 0:
                        # Move batch to GPU (fine data stays on CPU)
                        X_fine_batch = self.X_fine_tensor[batch_fine_indices].to(self.device)
                        fine_valid_batch = self.fine_valid_mask[batch_fine_indices].to(self.device)
                        y_pred_fine = self.model(X_fine_batch)
                        
                        loss, loss_components = self.loss_fn(
                            y_pred_coarse, y_coarse,
                            y_pred_fine, batch_mapping, fine_valid_batch
                        )
                    else:
                        loss, loss_components = self.loss_fn(y_pred_coarse, y_coarse)
                else:
                    loss, loss_components = self.loss_fn(y_pred_coarse, y_coarse)
                
                total_loss += loss_components['loss_total']
                total_loss_pred += loss_components['loss_pred']
                total_loss_cons += loss_components['loss_cons']
                n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'loss_pred': total_loss_pred / n_batches,
            'loss_cons': total_loss_cons / n_batches
        }
    
    def fit(self,
            X_coarse: np.ndarray,
            y_coarse: np.ndarray,
            X_fine: np.ndarray,
            coarse_to_fine_mapping: List[List[int]],
            fine_valid_mask: Optional[np.ndarray] = None,
            train_fraction: float = 0.7,
            random_state: int = 42,
            verbose: bool = True) -> Dict:
        """
        Train the model.
        
        Parameters:
        -----------
        X_coarse : np.ndarray
            Coarse features (n_coarse_samples, n_features)
        y_coarse : np.ndarray
            Coarse targets (n_coarse_samples,)
        X_fine : np.ndarray
            Fine features (n_fine_samples, n_features)
        coarse_to_fine_mapping : List[List[int]]
            Mapping from coarse to fine indices
        fine_valid_mask : np.ndarray, optional
            Validity mask for fine samples
        train_fraction : float
            Training data fraction
        random_state : int
            Random seed
        verbose : bool
            Print progress
        
        Returns:
        --------
        Dict
            Training history
        """
        params = self._get_hyperparams()
        
        print("\n" + "="*70)
        print("🚀 TRAINING SCALE-CONSISTENT NEURAL NETWORK")
        print("="*70)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(
            X_coarse, y_coarse, X_fine, coarse_to_fine_mapping,
            fine_valid_mask, train_fraction, random_state
        )
        
        # Build model
        input_dim = X_coarse.shape[1]
        self.build_model(input_dim)
        
        # Training loop
        max_epochs = params['max_epochs']
        patience = params['early_stopping_patience']
        
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        epochs_without_improvement = 0
        
        print(f"\n🏃 Starting training for up to {max_epochs} epochs...")
        print(f"   Early stopping patience: {patience}")
        
        for epoch in range(max_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, use_consistency=True)
            
            # Validate
            val_metrics = self.validate(val_loader, use_consistency=True)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_loss_pred'].append(train_metrics['loss_pred'])
            self.history['train_loss_cons'].append(train_metrics['loss_cons'])
            self.history['val_loss_pred'].append(val_metrics['loss_pred'])
            self.history['val_loss_cons'].append(val_metrics['loss_cons'])
            
            # Check for improvement
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
                print(f"   Epoch {epoch+1:4d}: "
                      f"Train Loss={train_metrics['loss']:.4f} (pred={train_metrics['loss_pred']:.4f}, cons={train_metrics['loss_cons']:.4f}) | "
                      f"Val Loss={val_metrics['loss']:.4f} (pred={val_metrics['loss_pred']:.4f}, cons={val_metrics['loss_cons']:.4f})")
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✅ Training complete!")
            print(f"   Best epoch: {best_epoch + 1}")
            print(f"   Best validation loss: {best_val_loss:.4f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions (sklearn-compatible interface).
        
        Uses batched processing to avoid GPU OOM for large inputs.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        
        Returns:
        --------
        np.ndarray
            Predictions
        """
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        self.model.eval()
        
        # Batch predictions to avoid GPU OOM for large inputs
        batch_size = 65536  # Process 64K samples at a time
        n_samples = len(X)
        predictions = np.empty(n_samples, dtype=np.float32)
        
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = torch.FloatTensor(X[start:end]).to(self.device)
                pred_batch = self.model(X_batch)
                predictions[start:end] = pred_batch.cpu().numpy().flatten()
        
        return predictions
    
    def save(self, path: str):
        """
        Save model and training state.
        
        Parameters:
        -----------
        path : str
            Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler': self.scaler,
            'history': self.history,
            'config': self._get_hyperparams()
        }
        
        torch.save(save_dict, path)
        print(f"💾 Model saved to: {path}")
    
    def load(self, path: str, input_dim: int):
        """
        Load model from checkpoint.
        
        Parameters:
        -----------
        path : str
            Path to the saved model
        input_dim : int
            Number of input features
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Build model structure
        self.build_model(input_dim)
        
        # Load state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler = checkpoint['scaler']
        self.history = checkpoint['history']
        
        print(f"📂 Model loaded from: {path}")


class ScaleConsistentNNWrapper:
    """
    Wrapper class that provides sklearn-compatible interface for the 
    scale-consistent neural network.
    
    This allows integration with the existing CoarseModelTrainer workflow.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the wrapper.
        
        Parameters:
        -----------
        **kwargs : dict
            Hyperparameters passed to the trainer
        """
        self.config = {'models': {'hyperparameters': {'nn_scale_consistent': kwargs}}}
        self.trainer = None
        self._is_fitted = False
        
        # Store data needed for training
        self.X_fine = None
        self.coarse_to_fine_mapping = None
        self.fine_valid_mask = None
    
    def set_fine_data(self, X_fine: np.ndarray, 
                      coarse_to_fine_mapping: List[List[int]],
                      fine_valid_mask: Optional[np.ndarray] = None):
        """
        Set fine-resolution data for consistency loss.
        
        Must be called before fit() for scale-consistent training.
        
        Parameters:
        -----------
        X_fine : np.ndarray
            Fine features
        coarse_to_fine_mapping : List[List[int]]
            Coarse to fine index mapping
        fine_valid_mask : np.ndarray, optional
            Validity mask for fine samples
        """
        self.X_fine = X_fine
        self.coarse_to_fine_mapping = coarse_to_fine_mapping
        self.fine_valid_mask = fine_valid_mask
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ScaleConsistentNNWrapper':
        """
        Fit the model (sklearn-compatible interface).
        
        Parameters:
        -----------
        X : np.ndarray
            Coarse features (n_samples, n_features)
        y : np.ndarray
            Coarse targets (n_samples,)
        
        Returns:
        --------
        self
        """
        self.trainer = ScaleConsistentTrainer(self.config)
        
        if self.X_fine is not None and self.coarse_to_fine_mapping is not None:
            # Full scale-consistent training
            self.trainer.fit(
                X, y,
                self.X_fine,
                self.coarse_to_fine_mapping,
                self.fine_valid_mask
            )
        else:
            # Fallback: train without consistency loss
            print("⚠️ Fine data not set - training without consistency loss")
            # Create dummy fine data
            self.trainer.fit(
                X, y,
                X,  # Use coarse as fine
                [[i] for i in range(len(X))],  # Identity mapping
                None
            )
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions (sklearn-compatible interface).
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        
        Returns:
        --------
        np.ndarray
            Predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        return self.trainer.predict(X)
    
    def save(self, path: str):
        """Save the model."""
        if self.trainer is not None:
            self.trainer.save(path)
    
    def load(self, path: str, input_dim: int):
        """Load the model."""
        if self.trainer is None:
            self.trainer = ScaleConsistentTrainer(self.config)
        self.trainer.load(path, input_dim)
        self._is_fitted = True


def test_scale_consistent_nn():
    """
    Simple test to verify the implementation works.
    """
    print("\n" + "="*70)
    print("🧪 TESTING SCALE-CONSISTENT NEURAL NETWORK")
    print("="*70)
    
    # Create synthetic data
    np.random.seed(42)
    
    n_coarse = 1000
    n_fine = 11000  # 11x more fine samples
    n_features = 10
    
    X_coarse = np.random.randn(n_coarse, n_features).astype(np.float32)
    y_coarse = np.sin(X_coarse[:, 0]) + 0.1 * np.random.randn(n_coarse)
    
    X_fine = np.random.randn(n_fine, n_features).astype(np.float32)
    
    # Create mapping (each coarse cell has ~11 fine cells)
    coarse_to_fine_mapping = []
    for i in range(n_coarse):
        start_idx = i * 11
        end_idx = min(start_idx + 11, n_fine)
        coarse_to_fine_mapping.append(list(range(start_idx, end_idx)))
    
    # Create and train model
    config = {
        'models': {
            'hyperparameters': {
                'nn_scale_consistent': {
                    'hidden_layers': [64, 32],
                    'max_epochs': 50,
                    'batch_size': 128,
                    'lambda_consistency': 1.0
                }
            }
        }
    }
    
    trainer = ScaleConsistentTrainer(config)
    history = trainer.fit(
        X_coarse, y_coarse,
        X_fine, coarse_to_fine_mapping,
        verbose=True
    )
    
    # Make predictions
    y_pred = trainer.predict(X_coarse)
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_coarse, y_pred)
    
    print(f"\n✅ Test complete!")
    print(f"   R² score: {r2:.4f}")
    print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final val loss: {history['val_loss'][-1]:.4f}")
    
    return r2 > 0.5  # Basic sanity check


if __name__ == "__main__":
    test_scale_consistent_nn()

