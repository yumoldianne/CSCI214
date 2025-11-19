"""
Context-Aware Residual Block (CARB) Implementation for Fashion-MNIST
Novel Architecture: Multi-path block with dynamic context-based routing

This is an ARCHITECTURAL modification, not just an activation function.
Key innovation: Layer arrangement with context-dependent path selection.

Author: Graduate Student - CSCI 214
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, silhouette_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ==================== DATA CLASSES ====================

@dataclass
class TrainingConfig:
    """Configuration for training experiments"""
    batch_size: int = 128
    learning_rate: float = 0.001
    epochs: int = 30
    weight_decay: float = 1e-4
    dropout_rate: float = 0.3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    subset_size: Optional[int] = None  # Set to use subset (e.g., 10000)
    
@dataclass
class NetworkConfig:
    """Configuration for network architecture"""
    input_size: int = 784
    hidden_sizes: List[int] = None
    num_classes: int = 10
    latent_dim: int = 32
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 128]

@dataclass
class ExperimentResults:
    """Store experiment results"""
    method_name: str
    accuracy: float
    f1_score: float
    train_history: Dict[str, List[float]]
    parameters: int
    training_time: float

# ==================== CORE ARCHITECTURAL COMPONENTS ====================

class ContextNetwork(nn.Module):
    """
    Context Network: Learns to map context to routing weights
    
    Architecture: 4 → 16 → 8 → 3
    Input: [input_mean, input_std, training_progress, layer_depth]
    Output: 3 routing weights (softmax normalized)
    """
    def __init__(self, context_dim: int = 4, hidden_dim1: int = 16, hidden_dim2: int = 8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim2, 3)  # 3 routing weights
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            context: [batch_mean, batch_std, progress, depth]
        Returns:
            routing_weights: [w1, w2, w3] (softmax normalized)
        """
        logits = self.network(context)
        weights = F.softmax(logits, dim=0)
        return weights


class ContextAwareResidualBlock(nn.Module):
    """
    Context-Aware Residual Block (CARB) - Novel Architectural Component
    
    This is an ARCHITECTURAL modification because:
    1. Requires three parallel layer branches
    2. Different path depths (1-layer vs 2-layer)
    3. Dynamic routing mechanism
    4. Cannot be implemented by changing activations
    
    Architecture:
        Input → Path 1 (Identity: 0-1 layer)
             → Path 2 (Linear: 1 layer)
             → Path 3 (Non-linear: 2 layers)
             → Weighted combination based on learned context
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 layer_depth: int, total_layers: int,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_depth = layer_depth
        self.total_layers = total_layers
        
        # === PATH 1: Identity/Projection ===
        if in_features == out_features:
            self.path_identity = nn.Identity()
        else:
            self.path_identity = nn.Linear(in_features, out_features, bias=False)
        
        # === PATH 2: Linear Transformation ===
        self.path_linear = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        # === PATH 3: Non-linear Transformation (2-layer) ===
        hidden_dim = max(out_features // 2, 32)
        self.path_nonlinear = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        # === CONTEXT NETWORK ===
        self.context_network = ContextNetwork()
        
        # === FINAL ACTIVATION ===
        self.activation = nn.ReLU(inplace=True)
        
        # Buffer to track training progress
        self.register_buffer('training_progress', torch.tensor(0.0))
        
        # For analysis: store routing weights history
        self.routing_history = []
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _compute_context(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute context vector from input statistics and metadata
        
        Context = [batch_mean, batch_std, training_progress, layer_depth]
        """
        with torch.no_grad():
            batch_mean = x.mean().item()
            batch_std = (x.std().item() + 1e-8)  # Add epsilon for stability
            depth_ratio = self.layer_depth / self.total_layers
            progress = self.training_progress.item()
        
        context = torch.tensor([
            batch_mean,
            batch_std,
            depth_ratio,
            progress
        ], device=x.device, dtype=x.dtype)
        
        return context
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with context-dependent routing
        
        This is architectural because:
        1. Requires three parallel paths to be computed
        2. Routing weights determined by learned function of context
        3. Cannot be represented as single pointwise operation
        """
        # Compute context and routing weights
        context = self._compute_context(x)
        routing_weights = self.context_network(context)
        
        # Store routing weights for analysis
        if self.training:
            self.routing_history.append(routing_weights.detach().cpu().numpy())
        
        # Compute outputs from all three paths (PARALLEL ARCHITECTURE)
        out_identity = self.path_identity(x)
        out_linear = self.path_linear(x)
        out_nonlinear = self.path_nonlinear(x)
        
        # Weighted combination (DYNAMIC ROUTING)
        output = (routing_weights[0] * out_identity + 
                 routing_weights[1] * out_linear + 
                 routing_weights[2] * out_nonlinear)
        
        # Final activation
        output = self.activation(output)
        
        return output
    
    def update_progress(self, current_epoch: int, total_epochs: int):
        """Update training progress for context computation"""
        self.training_progress.fill_(current_epoch / total_epochs)
    
    def get_routing_weights(self, x: torch.Tensor) -> np.ndarray:
        """Get current routing weights for a given input"""
        context = self._compute_context(x)
        routing_weights = self.context_network(context)
        return routing_weights.detach().cpu().numpy()
    
    def get_routing_history(self) -> np.ndarray:
        """Get routing weight history during training"""
        if len(self.routing_history) > 0:
            return np.array(self.routing_history)
        return np.array([])
    
    def reset_routing_history(self):
        """Reset routing history"""
        self.routing_history = []


# ==================== NEURAL NETWORK ARCHITECTURES ====================

class BaselineFFNN(nn.Module):
    """Baseline feed-forward neural network (standard architecture)"""
    
    def __init__(self, config: NetworkConfig, training_config: TrainingConfig):
        super().__init__()
        self.config = config
        
        layers = []
        layers.append(nn.Flatten())
        
        # Hidden layers
        in_features = config.input_size
        for hidden_size in config.hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(training_config.dropout_rate)
            ])
            in_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_features, config.num_classes))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualFFNN(nn.Module):
    """Standard residual network with fixed skip connections"""
    
    def __init__(self, config: NetworkConfig, training_config: TrainingConfig):
        super().__init__()
        self.config = config
        
        self.flatten = nn.Flatten()
        
        # Input projection
        self.input_proj = nn.Linear(config.input_size, config.hidden_sizes[0])
        
        # Residual block 1
        self.block1 = nn.Sequential(
            nn.Linear(config.hidden_sizes[0], config.hidden_sizes[0]),
            nn.BatchNorm1d(config.hidden_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(training_config.dropout_rate)
        )
        
        # Residual block 2
        self.block2 = nn.Sequential(
            nn.Linear(config.hidden_sizes[0], config.hidden_sizes[1]),
            nn.BatchNorm1d(config.hidden_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(training_config.dropout_rate)
        )
        self.proj2 = nn.Linear(config.hidden_sizes[0], config.hidden_sizes[1])
        
        # Output layer
        self.fc_out = nn.Linear(config.hidden_sizes[1], config.num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.input_proj(x)
        
        # Residual connection 1 (fixed skip)
        identity = x
        x = self.block1(x)
        x = x + identity
        
        # Residual connection 2 (fixed skip with projection)
        identity = self.proj2(x)
        x = self.block2(x)
        x = x + identity
        
        x = self.fc_out(x)
        return x
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CARBNetwork(nn.Module):
    """
    Network using Context-Aware Residual Blocks (CARB)
    This is the ARCHITECTURAL MODIFICATION being proposed
    """
    
    def __init__(self, config: NetworkConfig, training_config: TrainingConfig):
        super().__init__()
        self.config = config
        self.total_carb_layers = len(config.hidden_sizes)
        
        self.flatten = nn.Flatten()
        
        # Input projection
        self.input_proj = nn.Linear(config.input_size, config.hidden_sizes[0])
        
        # CARB blocks (architectural components)
        self.carb_blocks = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(zip(
            [config.hidden_sizes[0]] + config.hidden_sizes[:-1],
            config.hidden_sizes
        )):
            carb = ContextAwareResidualBlock(
                in_features=in_size,
                out_features=out_size,
                layer_depth=i + 1,
                total_layers=self.total_carb_layers + 1,
                dropout_rate=training_config.dropout_rate
            )
            self.carb_blocks.append(carb)
        
        # Output layer
        self.fc_out = nn.Linear(config.hidden_sizes[-1], config.num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m.parent() if hasattr(m, 'parent') else None, ContextAwareResidualBlock):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.input_proj(x)
        
        # Pass through CARB blocks
        for carb in self.carb_blocks:
            x = carb(x)
        
        x = self.fc_out(x)
        return x
    
    def update_carb_progress(self, current_epoch: int, total_epochs: int):
        """Update training progress for all CARB blocks"""
        for carb in self.carb_blocks:
            carb.update_progress(current_epoch, total_epochs)
    
    def get_routing_weights(self, x: torch.Tensor) -> List[np.ndarray]:
        """Get routing weights from all CARB blocks"""
        x = self.flatten(x)
        x = self.input_proj(x)
        
        weights = []
        for carb in self.carb_blocks:
            w = carb.get_routing_weights(x)
            weights.append(w)
            x = carb(x)
        
        return weights
    
    def get_all_routing_histories(self) -> List[np.ndarray]:
        """Get routing histories from all CARB blocks"""
        return [carb.get_routing_history() for carb in self.carb_blocks]
    
    def reset_routing_histories(self):
        """Reset routing histories for all CARB blocks"""
        for carb in self.carb_blocks:
            carb.reset_routing_history()
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== AUTOENCODER ARCHITECTURES ====================

class VanillaAutoencoder(nn.Module):
    """Baseline autoencoder with standard architecture"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, config.latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, config.input_size),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CARBAutoencoder(nn.Module):
    """Autoencoder using CARB architecture"""
    
    def __init__(self, config: NetworkConfig, training_config: TrainingConfig):
        super().__init__()
        self.config = config
        
        self.flatten = nn.Flatten()
        
        # Encoder with CARB blocks
        self.enc_proj = nn.Linear(config.input_size, 256)
        self.enc_carb1 = ContextAwareResidualBlock(
            256, 256, layer_depth=1, total_layers=6,
            dropout_rate=training_config.dropout_rate
        )
        self.enc_carb2 = ContextAwareResidualBlock(
            256, 128, layer_depth=2, total_layers=6,
            dropout_rate=training_config.dropout_rate
        )
        self.enc_fc = nn.Linear(128, config.latent_dim)
        
        # Decoder with CARB blocks
        self.dec_fc = nn.Linear(config.latent_dim, 128)
        self.dec_carb1 = ContextAwareResidualBlock(
            128, 128, layer_depth=4, total_layers=6,
            dropout_rate=training_config.dropout_rate
        )
        self.dec_carb2 = ContextAwareResidualBlock(
            128, 256, layer_depth=5, total_layers=6,
            dropout_rate=training_config.dropout_rate
        )
        self.dec_out = nn.Linear(256, config.input_size)
        self.unflatten = nn.Unflatten(1, (1, 28, 28))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear) and not any(isinstance(p, ContextAwareResidualBlock) for p in m.modules()):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.enc_proj(x)
        x = self.enc_carb1(x)
        x = self.enc_carb2(x)
        x = self.enc_fc(x)
        return x
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dec_fc(x)
        x = self.dec_carb1(x)
        x = self.dec_carb2(x)
        x = torch.sigmoid(self.dec_out(x))
        x = self.unflatten(x)
        return x
    
    def update_carb_progress(self, current_epoch: int, total_epochs: int):
        """Update training progress for all CARB blocks"""
        for module in self.modules():
            if isinstance(module, ContextAwareResidualBlock):
                module.update_progress(current_epoch, total_epochs)
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== TRAINER CLASSES ====================

class ClassifierTrainer:
    """Trainer for classification models"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig, device: str):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, config.epochs
        )
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Update CARB progress if applicable
        if hasattr(self.model, 'update_carb_progress'):
            self.model.update_carb_progress(epoch, self.config.epochs)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        return avg_loss, accuracy, f1
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Full training loop"""
        print(f"\nTraining {self.model.__class__.__name__}")
        print(f"Parameters: {self.model.count_parameters():,}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, val_f1 = self.evaluate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            self.scheduler.step()
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, '
                  f'Val F1: {val_f1:.4f}')
        
        training_time = time.time() - start_time
        
        # Final evaluation
        final_loss, final_acc, final_f1 = self.evaluate(val_loader)
        
        return {
            'accuracy': final_acc,
            'f1_score': final_f1,
            'history': self.history,
            'training_time': training_time
        }


class AutoencoderTrainer:
    """Trainer for autoencoder models"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig, device: str):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, config.epochs
        )
        
        self.history = {'train_loss': []}
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        
        # Update CARB progress if applicable
        if hasattr(self.model, 'update_carb_progress'):
            self.model.update_carb_progress(epoch, self.config.epochs)
        
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
        for inputs, _ in pbar:
            inputs = inputs.to(self.device)
            
            self.optimizer.zero_grad()
            reconstructed, _ = self.model(inputs)
            loss = self.criterion(reconstructed, inputs)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def train(self, train_loader: DataLoader) -> Dict:
        """Full training loop"""
        print(f"\nTraining {self.model.__class__.__name__}")
        print(f"Parameters: {self.model.count_parameters():,}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            self.scheduler.step()
            
            print(f'Epoch {epoch+1}: Reconstruction Loss: {train_loss:.6f}')
        
        training_time = time.time() - start_time
        
        return {
            'history': self.history,
            'training_time': training_time
        }
    
    def extract_latent(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract latent representations"""
        self.model.eval()
        latent_codes = []
        labels = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                if hasattr(self.model, 'encode'):
                    latent = self.model.encode(inputs)
                else:
                    _, latent = self.model(inputs)
                latent_codes.append(latent.cpu().numpy())
                labels.append(targets.numpy())
        
        return np.vstack(latent_codes), np.hstack(labels)


# ==================== EXPERIMENT MANAGER ====================

class ExperimentManager:
    """Manages experiments and results"""
    
    def __init__(self, config: TrainingConfig, network_config: NetworkConfig):
        self.config = config
        self.network_config = network_config
        self.results = {}
        
    def run_supervised_experiments(self, train_loader: DataLoader, 
                                   test_loader: DataLoader) -> Dict:
        """Run all supervised learning experiments"""
        print("="*70)
        print("SUPERVISED LEARNING EXPERIMENTS")
        print("="*70)
        
        experiments = [
            ('Baseline', BaselineFFNN),
            ('ResNet', ResidualFFNN),
            ('CARB', CARBNetwork)
        ]
        
        for name, model_class in experiments:
            print(f"\n{'='*70}")
            print(f"Training {name}")
            print(f"{'='*70}")
            
            # Create model
            if model_class == BaselineFFNN:
                model = model_class(self.network_config, self.config)
            elif model_class == ResidualFFNN:
                model = model_class(self.network_config, self.config)
            else:  # CARBNetwork
                model = model_class(self.network_config, self.config)
            
            # Train
            trainer = ClassifierTrainer(model, self.config, self.config.device)
            results = trainer.train(train_loader, test_loader)
            
            # Store results
            self.results[name] = ExperimentResults(
                method_name=name,
                accuracy=results['accuracy'],
                f1_score=results['f1_score'],
                train_history=results['history'],
                parameters=model.count_parameters(),
                training_time=results['training_time']
            )
            
            # Save model
            torch.save(model.state_dict(), f'models/{name.lower()}_model.pth')
            
            print(f"\n{name} Results:")
            print(f"  Accuracy: {results['accuracy']:.2f}%")
            print(f"  F1-Score: {results['f1_score']:.4f}")
            print(f"  Parameters: {model.count_parameters():,}")
            print(f"  Training Time: {results['training_time']:.2f}s")
        
        return self.results
    
    def run_unsupervised_experiments(self, train_loader: DataLoader, 
                                    test_loader: DataLoader) -> Dict:
        """Run unsupervised learning experiments"""
        print("="*70)
        print("UNSUPERVISED LEARNING EXPERIMENTS")
        print("="*70)
        
        unsupervised_results = {}
        
        # Original data clustering
        print("\n[1/3] Clustering original data...")
        original_data, labels = self._get_original_data(test_loader)
        sil_original, _ = self._evaluate_clustering(original_data, labels)
        unsupervised_results['Original'] = {
            'silhouette': sil_original,
            'dimensions': 784
        }
        print(f"Original Data - Silhouette Score: {sil_original:.4f}")
        
        # Vanilla Autoencoder
        print("\n[2/3] Training Vanilla Autoencoder...")
        vanilla_ae = VanillaAutoencoder(self.network_config)
        trainer = AutoencoderTrainer(vanilla_ae, self.config, self.config.device)
        ae_results = trainer.train(train_loader)
        
        latent_vanilla, _ = trainer.extract_latent(test_loader)
        sil_vanilla, _ = self._evaluate_clustering(latent_vanilla, labels)
        unsupervised_results['Vanilla_AE'] = {
            'silhouette': sil_vanilla,
            'dimensions': self.network_config.latent_dim,
            'parameters': vanilla_ae.count_parameters(),
            'training_time': ae_results['training_time']
        }
        print(f"Vanilla AE - Silhouette Score: {sil_vanilla:.4f}")
        
        # CARB Autoencoder
        print("\n[3/3] Training CARB Autoencoder...")
        carb_ae = CARBAutoencoder(self.network_config, self.config)
        trainer = AutoencoderTrainer(carb_ae, self.config, self.config.device)
        ae_results = trainer.train(train_loader)
        
        latent_carb, _ = trainer.extract_latent(test_loader)
        sil_carb, _ = self._evaluate_clustering(latent_carb, labels)
        unsupervised_results['CARB_AE'] = {
            'silhouette': sil_carb,
            'dimensions': self.network_config.latent_dim,
            'parameters': carb_ae.count_parameters(),
            'training_time': ae_results['training_time']
        }
        print(f"CARB AE - Silhouette Score: {sil_carb:.4f}")
        
        return unsupervised_results
    
    def _get_original_data(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get original flattened data"""
        data = []
        labels = []
        for inputs, targets in data_loader:
            data.append(inputs.reshape(inputs.size(0), -1).numpy())
            labels.append(targets.numpy())
        return np.vstack(data), np.hstack(labels)
    
    def _evaluate_clustering(self, data: np.ndarray, 
                            labels: np.ndarray, n_clusters: int = 10) -> Tuple[float, np.ndarray]:
        """Evaluate clustering with K-Means"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        silhouette = silhouette_score(data, cluster_labels)
        return silhouette, cluster_labels
    
    def save_results(self, output_dir: str = 'results'):
        """Save all results to JSON"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Supervised results
        if self.results:
            supervised_data = {}
            for name, result in self.results.items():
                supervised_data[name] = {
                    'accuracy': result.accuracy,
                    'f1_score': result.f1_score,
                    'parameters': result.parameters,
                    'training_time': result.training_time
                }
            
            with open(f'{output_dir}/supervised_results.json', 'w') as f:
                json.dump(supervised_data, f, indent=2)
            
            print(f"\n✓ Saved supervised results to {output_dir}/supervised_results.json")
    
    def print_summary(self):
        """Print experiment summary"""
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        
        if self.results:
            print("\nSupervised Learning:")
            print(f"{'Method':<15} {'Accuracy':<12} {'F1-Score':<12} {'Parameters':<12} {'Time (s)':<10}")
            print("-"*70)
            for name, result in self.results.items():
                print(f"{name:<15} {result.accuracy:<12.2f} {result.f1_score:<12.4f} "
                      f"{result.parameters:<12,} {result.training_time:<10.2f}")


# ==================== DATA LOADER ====================

class FashionMNISTDataLoader:
    """Handles Fashion-MNIST data loading"""
    
    def __init__(self, batch_size: int = 128, data_dir: str = './data', 
                 subset_size: Optional[int] = None):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.subset_size = subset_size
        
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and test data loaders"""
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        train_dataset = datasets.FashionMNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.FashionMNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        # Create subset if specified
        if self.subset_size is not None:
            train_dataset = self._create_balanced_subset(
                train_dataset, self.subset_size
            )
            # Use proportional test set (e.g., if train is 1/6 of full, test is too)
            test_subset_size = int(self.subset_size * (10000 / 60000))
            test_dataset = self._create_balanced_subset(
                test_dataset, test_subset_size
            )
            
            print(f"\n✓ Using subset of data:")
            print(f"  Training samples: {len(train_dataset)}")
            print(f"  Test samples: {len(test_dataset)}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def _create_balanced_subset(self, dataset, subset_size: int):
        """
        Create a balanced subset with equal samples per class
        
        Args:
            dataset: Original dataset
            subset_size: Target number of samples
            
        Returns:
            Subset dataset with balanced classes
        """
        # Get all labels
        if hasattr(dataset, 'targets'):
            labels = dataset.targets.numpy() if torch.is_tensor(dataset.targets) else np.array(dataset.targets)
        else:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
        # Calculate samples per class for balanced subset
        num_classes = 10
        samples_per_class = subset_size // num_classes
        
        # Get indices for each class
        indices = []
        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            
            # Randomly sample from this class
            if len(class_indices) >= samples_per_class:
                selected = np.random.choice(
                    class_indices, samples_per_class, replace=False
                )
            else:
                # If not enough samples, take all available
                selected = class_indices
            
            indices.extend(selected.tolist())
        
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Create subset
        subset = torch.utils.data.Subset(dataset, indices)
        
        return subset


# ==================== VISUALIZATION ====================

class Visualizer:
    """Create visualizations for paper"""
    
    def __init__(self, output_dir: str = 'figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
    
    def plot_training_curves(self, results: Dict[str, ExperimentResults]):
        """Plot training curves for all methods"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = {'Baseline': '#1f77b4', 'ResNet': '#ff7f0e', 'CARB': '#d62728'}
        
        for method, result in results.items():
            history = result.train_history
            color = colors.get(method, '#2ca02c')
            
            # Train loss
            axes[0, 0].plot(history['train_loss'], label=method, 
                           linewidth=2, color=color, alpha=0.8)
            
            # Val loss
            axes[0, 1].plot(history['val_loss'], label=method,
                           linewidth=2, color=color, alpha=0.8)
            
            # Train accuracy
            axes[1, 0].plot(history['train_acc'], label=method,
                           linewidth=2, color=color, alpha=0.8)
            
            # Val accuracy
            axes[1, 1].plot(history['val_acc'], label=method,
                           linewidth=2, color=color, alpha=0.8)
        
        # Formatting
        axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Training Accuracy', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/training_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/training_curves.pdf', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training curves to {self.output_dir}/training_curves.png")
    
    def plot_routing_weights(self, model: CARBNetwork, sample_input: torch.Tensor):
        """Plot routing weight evolution during training"""
        histories = model.get_all_routing_histories()
        
        if len(histories) == 0 or len(histories[0]) == 0:
            print("⚠ No routing history available")
            return
        
        fig, axes = plt.subplots(len(histories), 1, figsize=(10, 4*len(histories)))
        if len(histories) == 1:
            axes = [axes]
        
        path_names = ['Identity', 'Linear', 'Non-linear']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, history in enumerate(histories):
            if len(history) > 0:
                history = np.array(history)
                
                for j, (name, color) in enumerate(zip(path_names, colors)):
                    axes[i].plot(history[:, j], label=name, 
                               linewidth=2, color=color, alpha=0.8)
                
                axes[i].set_xlabel('Training Step', fontsize=11)
                axes[i].set_ylabel('Routing Weight', fontsize=11)
                axes[i].set_title(f'CARB Block {i+1} - Routing Weight Evolution',
                                fontsize=12, fontweight='bold')
                axes[i].legend(loc='best')
                axes[i].grid(True, alpha=0.3)
                axes[i].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/routing_weights.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/routing_weights.pdf', bbox_inches='tight')
        plt.close()
        print(f"✓ Saved routing weights to {self.output_dir}/routing_weights.png")
    
    def plot_confusion_matrix(self, model: nn.Module, test_loader: DataLoader, 
                             device: str, method_name: str = 'CARB'):
        """Plot confusion matrix"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        cm = confusion_matrix(all_targets, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Frequency'}, ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {method_name}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        filename = method_name.lower().replace(' ', '_')
        plt.savefig(f'{self.output_dir}/confusion_matrix_{filename}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved confusion matrix to {self.output_dir}/confusion_matrix_{filename}.png")


# ==================== MAIN ====================

def main():
    """Main execution function"""
    print("="*70)
    print("CONTEXT-AWARE RESIDUAL BLOCKS (CARB) IMPLEMENTATION")
    print("Fashion-MNIST Experiments")
    print("="*70)
    
    # Setup configurations
    training_config = TrainingConfig()
    network_config = NetworkConfig()
    
    # Ask about subset usage
    print("\n" + "="*70)
    print("Dataset Configuration:")
    print("  Full Fashion-MNIST: 60,000 training + 10,000 test samples")
    print("  Subset options for faster experimentation:")
    print("    1. Use full dataset (60,000 train)")
    print("    2. Use 10,000 samples (1,000 per class)")
    print("    3. Use 20,000 samples (2,000 per class)")
    print("    4. Use 30,000 samples (3,000 per class)")
    print("    5. Custom size")
    print("="*70)
    
    subset_choice = input("\nEnter choice (1-5): ").strip()
    
    if subset_choice == '1':
        training_config.subset_size = None
        print("✓ Using full dataset")
    elif subset_choice == '2':
        training_config.subset_size = 10000
        print("✓ Using 10,000 training samples")
    elif subset_choice == '3':
        training_config.subset_size = 20000
        print("✓ Using 20,000 training samples")
    elif subset_choice == '4':
        training_config.subset_size = 30000
        print("✓ Using 30,000 training samples")
    elif subset_choice == '5':
        try:
            custom_size = int(input("Enter custom training size (multiple of 10): "))
            if custom_size > 0 and custom_size <= 60000 and custom_size % 10 == 0:
                training_config.subset_size = custom_size
                print(f"✓ Using {custom_size} training samples")
            else:
                print("⚠ Invalid size. Using full dataset.")
                training_config.subset_size = None
        except ValueError:
            print("⚠ Invalid input. Using full dataset.")
            training_config.subset_size = None
    else:
        print("⚠ Invalid choice. Using full dataset.")
        training_config.subset_size = None
    
    print(f"\nConfiguration:")
    print(f"  Device: {training_config.device}")
    print(f"  Batch Size: {training_config.batch_size}")
    print(f"  Learning Rate: {training_config.learning_rate}")
    print(f"  Epochs: {training_config.epochs}")
    print(f"  Hidden Sizes: {network_config.hidden_sizes}")
    if training_config.subset_size:
        print(f"  Subset Size: {training_config.subset_size}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Load data
    print("\nLoading Fashion-MNIST dataset...")
    data_loader = FashionMNISTDataLoader(
        batch_size=training_config.batch_size,
        subset_size=training_config.subset_size
    )
    train_loader, test_loader = data_loader.get_loaders()
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Create experiment manager
    experiment_manager = ExperimentManager(training_config, network_config)
    
    # User choice
    print("\n" + "="*70)
    print("Select Experiment Type:")
    print("  1. Supervised Learning (Classification)")
    print("  2. Unsupervised Learning (Autoencoder + Clustering)")
    print("  3. Run Both")
    print("="*70)
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    # Run experiments
    supervised_results = None
    unsupervised_results = None
    
    if choice in ['1', '3']:
        supervised_results = experiment_manager.run_supervised_experiments(
            train_loader, test_loader
        )
        experiment_manager.save_results()
    
    if choice in ['2', '3']:
        unsupervised_results = experiment_manager.run_unsupervised_experiments(
            train_loader, test_loader
        )
        
        # Save unsupervised results
        if unsupervised_results:
            with open('results/unsupervised_results.json', 'w') as f:
                json.dump(unsupervised_results, f, indent=2)
            print("\n✓ Saved unsupervised results to results/unsupervised_results.json")
    
    # Print summary
    experiment_manager.print_summary()
    
    # Create visualizations
    if supervised_results:
        print("\n" + "="*70)
        print("Creating Visualizations...")
        print("="*70)
        
        visualizer = Visualizer()
        
        # Training curves
        visualizer.plot_training_curves(supervised_results)
        
        # Load CARB model and plot routing weights
        carb_model = CARBNetwork(network_config, training_config)
        carb_model.load_state_dict(torch.load('models/carb_model.pth'))
        carb_model.to(training_config.device)
        
        # Get sample input
        sample_input, _ = next(iter(test_loader))
        sample_input = sample_input[:1].to(training_config.device)
        
        visualizer.plot_routing_weights(carb_model, sample_input)
        
        # Confusion matrix for CARB
        visualizer.plot_confusion_matrix(
            carb_model, test_loader, training_config.device, 'CARB'
        )
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  Models: models/")
    print("  Results: results/")
    print("  Figures: figures/")
    print("\nNext steps:")
    print("  1. Review results in results/ directory")
    print("  2. Check figures in figures/ directory")
    print("  3. Use results to fill in your paper template")
    
    if training_config.subset_size:
        print("\n⚠ Note: Results are based on subset of data.")
        print("   For final paper results, run with full dataset (option 1).")
    
    print("="*70)


if __name__ == "__main__":
    main()