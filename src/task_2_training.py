"""
COMPLETE EXECUTABLE TASK 2 SCRIPT
Training Models on Smaller Out-of-Sample Dataset with Bias Handling

This script provides a complete, runnable workflow for Task 2:
1. Loads and analyzes the holdout dataset for class imbalance
2. Implements multiple bias mitigation strategies
3. Trains models using different approaches
4. Evaluates and compares all strategies
5. Generates all visualizations for the report

Author: [Your Name]
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler, Subset

import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import copy
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Reference: Lin et al. (2017). Focal Loss for Dense Object Detection
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AugmentedMNISTDataset(torch.utils.data.Dataset):
    """
    MNIST Dataset with data augmentation applied on-the-fly
    """
    
    def __init__(self, images, labels, augment=True, augmentation_level='medium'):
        """
        Args:
            images: Tensor of images (N, 1, 28, 28) or (N, 28, 28)
            labels: Tensor of labels (N,)
            augment: Whether to apply augmentation
            augmentation_level: 'light', 'medium', or 'heavy'
        """
        self.images = images
        self.labels = labels
        self.augment = augment
        
        # Ensure correct shape
        if len(self.images.shape) == 3:
            self.images = self.images.unsqueeze(1)
        
        # Define augmentation transforms
        if augment:
            if augmentation_level == 'light':
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                ])
            elif augmentation_level == 'medium':
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                ])
            elif augmentation_level == 'heavy':
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomRotation(15),
                    transforms.RandomAffine(
                        degrees=0, 
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1),
                        shear=5
                    ),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = None
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            # Convert to PIL, apply transform, convert back
            image = self.transform(image.squeeze())
            image = image.unsqueeze(0) if len(image.shape) == 2 else image
        
        return image, label


class Task2Pipeline:
    """
    Complete pipeline for Task 2: Training with bias handling
    """
    
    def __init__(self, results_dir='./results/task2'):
        """
        Initialize Task 2 pipeline
        
        Args:
            results_dir: Directory to save results
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'figures').mkdir(exist_ok=True)
        (self.results_dir / 'models').mkdir(exist_ok=True)
        (self.results_dir / 'tables').mkdir(exist_ok=True)
        
        print(f"Device: {self.device}")
        print(f"Results directory: {self.results_dir}")
        
        self.training_results = {}
        self.class_distribution = None
    
    def load_holdout_data(self, data_dir='./data'):
        """
        Load holdout dataset and split into train/validation
        
        Args:
            data_dir: Directory containing holdout_dataset.pt
        
        Returns:
            train_loader, val_loader, train_labels
        """
        print("\n" + "="*80)
        print("STEP 1: LOADING HOLDOUT DATASET")
        print("="*80)
        
        data_path = Path(data_dir) / 'holdout_dataset.pt'
        
        if not data_path.exists():
            print(f"✗ Holdout dataset not found: {data_path}")
            return None, None, None
            
        with torch.serialization.safe_globals([Subset]):
            holdout_data = torch.load(
            data_path, 
            map_location=self.device,  # Map to CPU if needed
            weights_only=False  # Keep as False for now
        )
        
        # Load data
        #holdout_data = torch.load(data_path, map_location=self.device, weights_only=False)
        
        # Handle different possible formats
        if isinstance(holdout_data, dict):
            X_holdout = holdout_data['data']
            y_holdout = holdout_data['labels']
        elif isinstance(holdout_data, (list, tuple)) and len(holdout_data) == 2:
            X_holdout, y_holdout = holdout_data
        elif hasattr(holdout_data, '__getitem__'):
            X_holdout = torch.stack([holdout_data[i][0] for i in range(len(holdout_data))])
            y_holdout = torch.tensor([holdout_data[i][1] for i in range(len(holdout_data))])
        else:
            raise ValueError(f"Unknown holdout data format: {type(holdout_data)}")
        
        # Ensure correct shape
        if len(X_holdout.shape) == 3:
            X_holdout = X_holdout.unsqueeze(1)
        
        print(f"✓ Holdout dataset loaded")
        print(f"  Total samples: {len(X_holdout)}")
        print(f"  Shape: {X_holdout.shape}")
        
        # Analyze class distribution
        self.analyze_class_imbalance(y_holdout)
        
        # Split into train/val (80/20)
        train_size = int(0.8 * len(X_holdout))
        val_size = len(X_holdout) - train_size
        
        # Create indices for splitting
        indices = torch.randperm(len(X_holdout))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train = X_holdout[train_indices]
        y_train = y_holdout[train_indices]
        X_val = X_holdout[val_indices]
        y_val = y_holdout[val_indices]
        
        print(f"\n✓ Data split:")
        print(f"  Training:   {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        
        # Create datasets (without augmentation for now)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader, y_train.numpy()
    
    def analyze_class_imbalance(self, labels):
        """
        Analyze and visualize class imbalance
        
        Args:
            labels: Tensor of labels
        """
        print("\n" + "="*80)
        print("STEP 2: ANALYZING CLASS IMBALANCE")
        print("="*80)
        
        # Convert to numpy if needed
        if torch.is_tensor(labels):
            labels = labels.numpy()
        
        # Count classes
        self.class_distribution = Counter(labels)
        counts = [self.class_distribution[i] for i in range(10)]
        
        # Calculate imbalance metrics
        max_samples = max(counts)
        min_samples = min(counts)
        imbalance_ratio = max_samples / min_samples
        
        print(f"\nClass Distribution:")
        print(f"  Total Samples: {len(labels)}")
        print(f"  Max per class: {max_samples}")
        print(f"  Min per class: {min_samples}")
        print(f"  Imbalance Ratio: {imbalance_ratio:.2f}")
        print("\nPer-Class Breakdown:")
        
        for cls, count in sorted(self.class_distribution.items()):
            percentage = count / len(labels) * 100
            bar = '█' * int(percentage / 2)
            print(f"  Class {cls}: {count:4d} samples ({percentage:5.1f}%) {bar}")
        
        # Create visualization
        self.visualize_class_imbalance(counts)
    
    def visualize_class_imbalance(self, class_counts):
        """
        Create visualizations of class imbalance
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        classes = range(10)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.7, 10))
        
        # Bar chart
        bars = axes[0].bar(classes, class_counts, color=colors, edgecolor='black')
        axes[0].set_xlabel('Digit Class', fontsize=12)
        axes[0].set_ylabel('Number of Samples', fontsize=12)
        axes[0].set_title('Class Distribution in Holdout Dataset', fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, class_counts):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(count)}', ha='center', va='bottom', fontsize=9)
        
        # Pie chart
        axes[1].pie(class_counts, labels=classes, autopct='%1.1f%%',
                   startangle=90, colors=colors)
        axes[1].set_title('Class Distribution (%)', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'figures' / 'class_imbalance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Class imbalance visualization saved")
    
    def compute_class_weights(self, labels, method='inverse'):
        """
        Compute class weights for loss function
        
        Args:
            labels: Training labels
            method: 'inverse' or 'effective_num'
        
        Returns:
            Tensor of class weights
        """
        class_counts = Counter(labels)
        n_samples = len(labels)
        n_classes = len(class_counts)
        
        if method == 'inverse':
            # Inverse frequency
            weights = {cls: n_samples / count for cls, count in class_counts.items()}
        elif method == 'effective_num':
            # Effective number (Cui et al., 2019)
            beta = 0.9999
            weights = {}
            for cls, count in class_counts.items():
                effective_num = 1.0 - np.power(beta, count)
                weights[cls] = (1.0 - beta) / effective_num
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert to tensor and normalize
        weight_tensor = torch.FloatTensor([weights[i] for i in range(n_classes)])
        weight_tensor = weight_tensor / weight_tensor.sum() * n_classes
        
        return weight_tensor.to(self.device)
    
    def create_balanced_sampler(self, labels):
        """
        Create weighted sampler for balanced batches
        
        Args:
            labels: Training labels
        
        Returns:
            WeightedRandomSampler
        """
        class_weights = self.compute_class_weights(labels, method='inverse')
        sample_weights = [class_weights[label].item() for label in labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        return sampler
    
    def train_model(self, model, train_loader, val_loader, strategy_name,
                   epochs=50, lr=0.001, criterion=None, use_augmentation=False,
                   X_train=None, y_train=None):
        """
        Train a model with specified strategy
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            strategy_name: Name of the training strategy
            epochs: Number of epochs
            lr: Learning rate
            criterion: Loss function (optional)
            use_augmentation: Whether to use data augmentation
            X_train, y_train: Training data for augmentation
        
        Returns:
            Trained model and history
        """
        print(f"\n{'='*80}")
        print(f"TRAINING: {strategy_name}")
        print(f"{'='*80}")
        
        model.to(self.device)
        model.train()
        
        # Setup optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Setup criterion if not provided
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # If using augmentation, create augmented dataset
        if use_augmentation and X_train is not None and y_train is not None:
            print("  Using data augmentation: medium level")
            aug_dataset = AugmentedMNISTDataset(X_train, torch.tensor(y_train), 
                                               augment=True, augmentation_level='medium')
            train_loader = DataLoader(aug_dataset, batch_size=32, shuffle=True)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0
        patience = 10
        
        print(f"  Epochs: {epochs}")
        print(f"  Learning Rate: {lr}")
        print(f"  Device: {self.device}\n")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(data)
                
                # Handle different output types
                if isinstance(outputs, tuple):
                    logits = outputs[1]  # (reconstruction, logits)
                else:
                    logits = outputs
                
                #print(f"logits shape: {logits.shape}")  # Should be [batch_size, 10]
                #print(f"labels shape: {labels.shape}")  # Should be [batch_size]
                #print(f"labels values: {labels[:10]}")  # Should be 0-9
                
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * data.size(0)
                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            train_loss = running_loss / total
            train_acc = correct / total
            
            # Validation phase
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    
                    outputs = model(data)
                    
                    if isinstance(outputs, tuple):
                        logits = outputs[1]
                    else:
                        logits = outputs
                    
                    loss = criterion(logits, labels)
                    
                    val_running_loss += loss.item() * data.size(0)
                    _, preds = torch.max(logits, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total
            
            # Update scheduler
            scheduler.step(val_acc)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1:3d}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
        
        print(f"\n  ✓ Best Validation Accuracy: {best_val_acc:.4f}")
        
        # Load best weights
        model.load_state_dict(best_model_wts)
        
        # Save model
        torch.save(model.state_dict(), 
                  self.results_dir / 'models' / f'{strategy_name}_weights.pth')
        
        return model, history
    
    def evaluate_model(self, model, test_loader, strategy_name):
        """
        Evaluate trained model
        
        Args:
            model: Trained model
            test_loader: Test data loader
            strategy_name: Strategy name for labeling
        
        Returns:
            Dictionary of evaluation results
        """
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                outputs = model(data)
                
                if isinstance(outputs, tuple):
                    logits = outputs[1]
                else:
                    logits = outputs
                
                _, preds = torch.max(logits, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = \
            precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        results = {
            'strategy_name': strategy_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'per_class_f1': per_class_f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels
        }
        
        print(f"\n  Evaluation Results:")
        print(f"    Accuracy:  {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
        
        return results
    
    def run_all_strategies(self, data_dir='./data'):
        """
        Run all bias mitigation strategies
        
        Args:
            data_dir: Directory containing data
        """
        print("\n" + "="*80)
        print("TASK 2: TRAINING WITH BIAS HANDLING")
        print("="*80)
        
        # Load data
        train_loader, val_loader, train_labels = self.load_holdout_data(data_dir)
        
        if train_loader is None:
            print("✗ Could not load data")
            return
        
        # Get train data for augmentation
        X_train = train_loader.dataset.tensors[0]
        y_train = train_loader.dataset.tensors[1]
        
        # Import model architecture
        try:
            import sys
            sys.path.append(str(Path(data_dir).parent / 'src'))
            from model import VariationalAutoencoder  # Replace with actual model name
            print("\n✓ Model architecture imported")
## ===================================================================

        # Create a wrapper model that adds classification
            class VAEClassifier(nn.Module):
                def __init__(self, latent_dim=20):  # Adjust latent_dim based on your VAE
                    super().__init__()
                # Create VAE
                    self.vae = VariationalAutoencoder()
                
                # Extract latent dimension from VAE (you might need to adjust this)
                # Based on your attributes, it seems to have latent_dims
                    if hasattr(self.vae, 'latent_dims'):
                        latent_dim = self.vae.latent_dims
                    else:
            # Try to determine latent dimension from forward pass
                        with torch.no_grad():
                            test_input = torch.randn(1, 1, 28, 28)
                            vae_output = self.vae(test_input)
                            if isinstance(vae_output, tuple) and len(vae_output) >= 2:
                                mu = vae_output[1]
                                latent_dim = mu.shape[1]
        
                    print(f"  Detected latent dimension: {latent_dim}")
        
                # Add classifier head
                    self.classifier = nn.Sequential(
                        nn.Linear(latent_dim, 64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 10)
                    )
                
                # Freeze VAE layers (optional)
                    for param in self.vae.parameters():
                        param.requires_grad = False
            
                def forward(self, x):
                # Get latent representation from VAE
                # Your VAE likely returns: (reconstructed_x, mu, logvar)
                    vae_output = self.vae(x)
                
                    if isinstance(vae_output, tuple):
                        mu = vae_output[1]  # Latent mean (batch_size, latent_dim)
                        return self.classifier(mu)
                    else:
                    # If not tuple, use the output directly
                        return self.classifier(vae_output)
        
        # Now use VAEClassifier instead of plain VAE
            ModelClass = VAEClassifier

## ===================================================================
        except ImportError:
            print("\n⚠ Using placeholder model")
            # Placeholder model
            class VariationalAutoencoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(1, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Flatten(),
                        nn.Linear(64*7*7, 128),
                        nn.ReLU(),
                        nn.Dropout(0.5)
                    )
                    self.classifier = nn.Linear(128, 10)  # 10 classes for digits 0-9
                
                def forward(self, x):
                    x = self.features(x)
                    return self.classifier(x)
        
        # Strategy 1: Baseline (no bias handling)
        print("\n" + "="*80)
        print("STRATEGY 1: BASELINE (No Bias Handling)")
        print("="*80)
        
        ##model1 = VariationalAutoencoder()
        model1 = ModelClass()
        ##print(f"Model1 output dimension: {model1.classifier.out_features}")  # Should be 10

        criterion1 = nn.CrossEntropyLoss()
        model1, history1 = self.train_model(
            model1, train_loader, val_loader, 'baseline',
            epochs=50, lr=0.001, criterion=criterion1
        )
        results1 = self.evaluate_model(model1, val_loader, 'baseline')
        results1['history'] = history1
        self.training_results['baseline'] = results1
        
        # Strategy 2: Class-weighted loss
        print("\n" + "="*80)
        print("STRATEGY 2: CLASS-WEIGHTED LOSS")
        print("="*80)
        
        ##model2 = VariationalAutoencoder()
        model2 = ModelClass()
        ##print(f"Model2 output dimension: {model2.classifier.out_features}")  # Should be 10

        class_weights = self.compute_class_weights(train_labels, method='inverse')
        criterion2 = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Class weights: {class_weights.cpu().numpy()}")
        model2, history2 = self.train_model(
            model2, train_loader, val_loader, 'weighted_loss',
            epochs=50, lr=0.001, criterion=criterion2
        )
        results2 = self.evaluate_model(model2, val_loader, 'weighted_loss')
        results2['history'] = history2
        self.training_results['weighted_loss'] = results2
        
        # Strategy 3: Focal loss
        print("\n" + "="*80)
        print("STRATEGY 3: FOCAL LOSS")
        print("="*80)
        
        ##model3 = VariationalAutoencoder()
        model3 = ModelClass()
        ##print(f"Model3 output dimension: {model3.classifier.out_features}")  # Should be 10

        criterion3 = FocalLoss(alpha=class_weights, gamma=2.0)
        print(f"  Using Focal Loss with gamma=2.0")
        model3, history3 = self.train_model(
            model3, train_loader, val_loader, 'focal_loss',
            epochs=50, lr=0.001, criterion=criterion3
        )
        results3 = self.evaluate_model(model3, val_loader, 'focal_loss')
        results3['history'] = history3
        self.training_results['focal_loss'] = results3
        
        # Strategy 4: Data augmentation + weighted loss
        print("\n" + "="*80)
        print("STRATEGY 4: DATA AUGMENTATION + WEIGHTED LOSS")
        print("="*80)
        
        ##model4 = VariationalAutoencoder()
        model4 = ModelClass()        
        ##print(f"Model4 output dimension: {model4.classifier.out_features}")  # Should be 10

        criterion4 = nn.CrossEntropyLoss(weight=class_weights)
        model4, history4 = self.train_model(
            model4, train_loader, val_loader, 'augmented_weighted',
            epochs=50, lr=0.001, criterion=criterion4,
            use_augmentation=True, X_train=X_train, y_train=y_train
        )
        results4 = self.evaluate_model(model4, val_loader, 'augmented_weighted')
        results4['history'] = history4
        self.training_results['augmented_weighted'] = results4
        
        # Create visualizations
        self.create_all_visualizations()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*80)
        print("TASK 2 COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {self.results_dir}")
    
    def create_all_visualizations(self):
        """
        Generate all visualizations for Task 2
        """
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        fig_dir = self.results_dir / 'figures'
        
        # 1. Training curves for all strategies
        print("\n1. Creating training curves...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, (name, results) in enumerate(self.training_results.items()):
            history = results['history']
            
            ax = axes[idx]
            epochs = range(1, len(history['train_loss']) + 1)
            
            ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Loss', fontsize=11)
            ax.set_title(f'{name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            
            # Add text with best val accuracy
            best_acc = max(history['val_acc'])
            ax.text(0.02, 0.98, f'Best Val Acc: {best_acc:.4f}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'training_curves_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Accuracy curves
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, (name, results) in enumerate(self.training_results.items()):
            history = results['history']
            
            ax = axes[idx]
            epochs = range(1, len(history['train_acc']) + 1)
            
            ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
            ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_title(f'{name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'training_curves_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Strategy comparison
        print("2. Creating strategy comparison...")
        
        strategy_names = list(self.training_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']
        
        for idx, metric in enumerate(metrics):
            values = [self.training_results[s][metric] for s in strategy_names]
            
            bars = axes[idx].bar(range(len(strategy_names)), values, color=colors, edgecolor='black')
            axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            axes[idx].set_xticks(range(len(strategy_names)))
            axes[idx].set_xticklabels([s.replace('_', '\n') for s in strategy_names], fontsize=9)
            axes[idx].set_ylim([0, 1.0])
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison', 
                               fontsize=12, fontweight='bold')
            
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Strategy Comparison', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(fig_dir / 'strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Per-class performance improvement
        print("3. Creating per-class improvement analysis...")
        
        # Compare baseline vs best strategy
        baseline_f1 = self.training_results['baseline']['per_class_f1']
        
        # Find best strategy
        best_strategy = max(self.training_results.keys(), 
                           key=lambda x: self.training_results[x]['accuracy'])
        best_f1 = self.training_results[best_strategy]['per_class_f1']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        classes = range(10)
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_f1, width, label='Baseline', 
                      color='#FF6B6B', edgecolor='black')
        bars2 = ax.bar(x + width/2, best_f1, width, label=best_strategy.replace('_', ' ').title(),
                      color='#4ECDC4', edgecolor='black')
        
        ax.set_xlabel('Digit Class', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_title(f'Per-Class Performance: Baseline vs {best_strategy.replace("_", " ").title()}',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'per_class_improvement.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Summary table
        print("4. Creating summary table...")
        
        data = []
        for name, results in self.training_results.items():
            row = {
                'Strategy': name.replace('_', ' ').title(),
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'Best Epoch': len(results['history']['train_loss'])
            }
            data.append(row)
        
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv(self.results_dir / 'tables' / 'strategy_comparison.csv', index=False)
        
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colColours=['#E8E8E8']*len(df.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Strategy Comparison Summary', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(fig_dir / 'strategy_summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ All visualizations saved to {fig_dir}")
    
    def generate_report(self):
        """
        Generate text report
        """
        report_path = self.results_dir / 'task2_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TASK 2: TRAINING WITH BIAS HANDLING - REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Class imbalance analysis
            f.write("CLASS IMBALANCE ANALYSIS:\n")
            f.write("-"*40 + "\n")
            if self.class_distribution:
                counts = [self.class_distribution[i] for i in range(10)]
                f.write(f"Max samples: {max(counts)}\n")
                f.write(f"Min samples: {min(counts)}\n")
                f.write(f"Imbalance ratio: {max(counts)/min(counts):.2f}\n\n")
            
            # Strategy results
            f.write("STRATEGY RESULTS:\n")
            f.write("-"*40 + "\n\n")
            
            for name, results in self.training_results.items():
                f.write(f"{name.upper().replace('_', ' ')}:\n")
                f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
                f.write(f"  Precision: {results['precision']:.4f}\n")
                f.write(f"  Recall:    {results['recall']:.4f}\n")
                f.write(f"  F1-Score:  {results['f1_score']:.4f}\n")
                f.write(f"  Epochs:    {len(results['history']['train_loss'])}\n\n")
            
            # Best strategy
            best_strategy = max(self.training_results.keys(),
                              key=lambda x: self.training_results[x]['accuracy'])
            f.write("BEST STRATEGY:\n")
            f.write("-"*40 + "\n")
            f.write(f"Strategy: {best_strategy.upper().replace('_', ' ')}\n")
            f.write(f"Accuracy: {self.training_results[best_strategy]['accuracy']:.4f}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-"*40 + "\n")
            f.write("1. Use class-weighted loss or focal loss for imbalanced datasets\n")
            f.write("2. Data augmentation helps with limited training data\n")
            f.write("3. Monitor per-class performance, not just overall accuracy\n")
            f.write("4. Consider combining multiple strategies for best results\n")
        
        print(f"\n✓ Report saved to {report_path}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    """
    Run this script to execute Task 2
    
    USAGE:
    ------
    1. Ensure you have ./data/holdout_dataset.pt
    2. Update YourModelClass import with actual model name
    3. Run: python task2_training.py
    4. Results saved to ./results/task2/
    """
    
    # Create pipeline
    pipeline = Task2Pipeline(results_dir='./results/task2')
    
    # Run all strategies
    pipeline.run_all_strategies(data_dir='./data')
    
    print("\n" + "="*80)
    print("TASK 2 SUMMARY")
    print("="*80)
    print("\nGenerated Outputs:")
    print("  1. Class imbalance visualization")
    print("  2. Training curves (loss and accuracy)")
    print("  3. Strategy comparison charts")
    print("  4. Per-class improvement analysis")
    print("  5. Summary tables")
    print("  6. Trained model weights")
    print("\nNext Steps:")
    print("  1. Review figures in ./results/task2/figures/")
    print("  2. Analyze which strategy works best for your data")
    print("  3. Include visualizations in your report")
    print("  4. Explain WHY the best strategy worked")
    print("="*80)
