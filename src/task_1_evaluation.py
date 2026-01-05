"""
COMPLETE EXECUTABLE EVALUATION SCRIPT FOR TASK 1

This script provides a complete, runnable workflow for evaluating all three models.
Simply adjust the paths and run this script.

Author: [Your Name]
Date: December 2024
"""

import torch
import torch.nn as nn

# ================================================

# First, let's import ALL the classes you've encountered
import torchvision.datasets.mnist
import torchvision.datasets.vision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision.transforms import Compose
# ================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
from sklearn.manifold import TSNE
import pandas as pd
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class Task1Evaluator:
    """
    Complete evaluation pipeline for Task 1
    Evaluates all three pre-trained models and generates all required outputs
    """
    
    def __init__(self, results_dir='./results'):
        """
        Initialize evaluator
        
        Args:
            results_dir: Directory to save results
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / 'figures').mkdir(exist_ok=True)
        (self.results_dir / 'tables').mkdir(exist_ok=True)
        
        print(f"Device: {self.device}")
        print(f"Results will be saved to: {self.results_dir}")
        
        self.all_results = {}
    
    def load_models_and_data(self, data_dir='./data'):
        """
        Load all three pre-trained models and test dataset
        
        Args:
            data_dir: Directory containing model weights and data
        
        Returns:
            models (dict), test_loader (DataLoader)
        """
        print("\n" + "="*80)
        print("STEP 1: LOADING MODELS AND DATA")
        print("="*80)
        
        data_path = Path(data_dir)
        
        # Import model architecture from provided repository
        # You need to adjust this import based on the actual repository structure
        try:
            import sys
            sys.path.append(str(data_path.parent / 'src'))
            from model import VariationalAutoencoder  # Replace with actual model class name
            print("✓ Model architecture imported")
        except ImportError:
            print("⚠ Warning: Could not import model from src/model.py")
            print("   Using placeholder. You must replace YourModelClass with the actual model.")
            # Placeholder - replace with actual model
            class VariationalAutoencoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    # This is a placeholder - use the actual architecture
                    self.encoder = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(784, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64)
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(64, 128),
                        nn.ReLU(),
                        nn.Linear(128, 784),
                        nn.Sigmoid()
                    )
                    self.classifier = nn.Linear(64, 10)
                
                def forward(self, x):
                    x = x.view(x.size(0), -1)
                    z = self.encoder(x)
                    reconstruction = self.decoder(z).view(-1, 1, 28, 28)
                    logits = self.classifier(z)
                    return reconstruction, logits
        
        # Load the three models
        models = {}
        model_names = ['model0', 'model1', 'model2']
        # model_names = ['model0', 'model1']

        for model_name in model_names:
            print(f"\nLoading {model_name} using {self.device}...")
            
            # Initialize model
            model = VariationalAutoencoder()
            
            # Load weights
            weight_path = data_path / f'amml_{model_name}_weights.pth'
            
            if weight_path.exists():
                try:
                    # Added by P.N 2025.12.22.22:11
                    with torch.serialization.safe_globals([Subset]):
                        state_dict = torch.load(
                        weight_path, 
                        map_location=self.device,  # Map to CPU if needed
                        weights_only=False  # Keep as False for now
                    )
                    
                    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
                    model.to(self.device)
                    model.eval()
                    models[model_name] = model
                    print(f"  ✓ Loaded from {weight_path}")
                except Exception as e:
                    print(f"  ✗ Error loading weights: {e}")
            else:
                print(f"  ✗ Weight file not found: {weight_path}")
                print(f"     Please ensure the file exists at this location")
        
        # Load test dataset
        print(f"\nLoading test dataset...{data_path}")
        test_path = data_path / 'test_dataset.pt'
        
                    # Added by P.N 2025.12.22.22:39

        with torch.serialization.safe_globals([torchvision.datasets.mnist.MNIST, Subset, Compose, transforms.ToTensor, torchvision.datasets.vision.StandardTransform]):
            test_data = torch.load(test_path, weights_only=False)
        
        if test_path.exists():
            try:
                test_data = torch.load(test_path, weights_only=False)
                
                # Handle different possible data formats
                if isinstance(test_data, dict):
                    # Format: {'data': tensor, 'labels': tensor}
                    X_test = test_data['data']
                    y_test = test_data['labels']
                elif isinstance(test_data, (list, tuple)) and len(test_data) == 2:
                    # Format: (data_tensor, labels_tensor)
                    X_test, y_test = test_data
                elif hasattr(test_data, '__getitem__'):
                    # Format: Dataset object
                    X_test = torch.stack([test_data[i][0] for i in range(len(test_data))])
                    y_test = torch.tensor([test_data[i][1] for i in range(len(test_data))])
                else:
                    raise ValueError(f"Unknown test data format: {type(test_data)}")
                
                # Ensure correct shape
                if len(X_test.shape) == 3:  # (N, 28, 28)
                    X_test = X_test.unsqueeze(1)  # (N, 1, 28, 28)
                
                # Create DataLoader
                test_dataset = TensorDataset(X_test, y_test)
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
                
                print(f"  ✓ Test dataset loaded")
                print(f"     Samples: {len(X_test)}")
                print(f"     Shape: {X_test.shape}")
                
            except Exception as e:
                print(f"  ✗ Error loading test data: {e}")
                test_loader = None
        else:
            print(f"  ✗ Test data file not found: {test_path}")
            test_loader = None
        
        return models, test_loader
    
    def evaluate_single_model(self, model, model_name, test_loader):
        """
        Evaluate a single model on all metrics
        
        Args:
            model: PyTorch model
            model_name: Name identifier
            test_loader: DataLoader for test data
        
        Returns:
            Dictionary of results
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING: {model_name.upper()}")
        print(f"{'='*80}")
        
        model.eval()
        results = {
            'model_name': model_name,
            'classification': {},
            'reconstruction': {},
            'computational': {}
        }
        
        # ============================================================
        # 1. CLASSIFICATION EVALUATION
        # ============================================================
        print("\n1. Classification Performance...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                outputs = model(data)
                
                # Handle different model outputs
                if isinstance(outputs, tuple):
                    # Assume (reconstruction, logits)
                    logits = outputs[1]
                else:
                    logits = outputs
                
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # AUC-ROC
        try:
            auc_scores = []
            for i in range(10):
                binary_labels = (all_labels == i).astype(int)
                if len(np.unique(binary_labels)) > 1:
                    auc = roc_auc_score(binary_labels, all_probs[:, i])
                    auc_scores.append(auc)
            mean_auc = np.mean(auc_scores) if auc_scores else None
        except Exception as e:
            print(f"   Warning: Could not calculate AUC: {e}")
            mean_auc = None
            auc_scores = None
        
        results['classification'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'per_class_f1': per_class_f1,
            'support': per_class_support,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'auc_scores': auc_scores,
            'mean_auc': mean_auc
        }
        
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        if mean_auc:
            print(f"   Mean AUC:  {mean_auc:.4f}")
        
        # ============================================================
        # 2. RECONSTRUCTION EVALUATION
        # ============================================================
        print("\n2. Reconstruction Performance...")
        
        criterion = nn.MSELoss(reduction='none')
        total_loss = 0
        per_sample_losses = []
        reconstructions = []
        originals = []
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                
                outputs = model(data)
                
                # Handle different output types
                if isinstance(outputs, tuple):
                    recon = outputs[0]
                else:
                    # If model only outputs logits, skip reconstruction
                    print("   Model does not output reconstructions")
                    break
                
                # Calculate loss
                loss = criterion(recon, data)
                per_pixel_loss = loss.view(loss.size(0), -1).mean(dim=1)
                
                total_loss += loss.mean().item() * data.size(0)
                per_sample_losses.extend(per_pixel_loss.cpu().numpy())
                
                # Store samples for visualization
                if len(reconstructions) < 10:
                    reconstructions.append(recon.cpu())
                    originals.append(data.cpu())
        
        if reconstructions:
            avg_loss = total_loss / len(test_loader.dataset)
            per_sample_losses = np.array(per_sample_losses)
            
            # Calculate PSNR
            psnr_scores = []
            for orig, recon in zip(originals, reconstructions):
                mse = torch.mean((orig - recon) ** 2, dim=[1, 2, 3])
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-10))
                psnr_scores.extend(psnr.numpy())
            
            results['reconstruction'] = {
                'avg_mse_loss': avg_loss,
                'per_sample_losses': per_sample_losses,
                'mean_psnr': np.mean(psnr_scores),
                'std_psnr': np.std(psnr_scores),
                'reconstructions': torch.cat(reconstructions),
                'originals': torch.cat(originals)
            }
            
            print(f"   Avg MSE Loss: {avg_loss:.6f}")
            print(f"   Mean PSNR:    {np.mean(psnr_scores):.2f} dB")
        
        # ============================================================
        # 3. COMPUTATIONAL EFFICIENCY (for comparison)
        # ============================================================
        print("\n3. Computational Efficiency...")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Measure inference time
        times = []
        
        # Warmup
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= 5:
                    break
                data = data.to(self.device)
                _ = model(data)
        
        # Actual timing
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= 50:
                    break
                
                data = data.to(self.device)
                
                start = time.time()
                _ = model(data)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()
                
                times.append((end - start) / data.size(0))
        
        results['computational'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'avg_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'throughput': 1.0 / np.mean(times)
        }
        
        print(f"   Total Parameters:     {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        print(f"   Avg Inference Time:   {np.mean(times)*1000:.2f} ms/sample")
        print(f"   Throughput:           {1.0/np.mean(times):.2f} samples/sec")
        
        return results
    
    def create_all_visualizations(self):
        """
        Generate all required visualizations for the report
        """
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*80}")
        
        fig_dir = self.results_dir / 'figures'
        
        # Set style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
        model_names = list(self.all_results.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # ============================================================
        # 1. CONFUSION MATRICES
        # ============================================================
        print("\n1. Creating confusion matrices...")
        
        for model_name, results in self.all_results.items():
            fig, ax = plt.subplots(figsize=(10, 8))
            
            cm = results['classification']['confusion_matrix']
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       square=True, cbar_kws={'label': 'Normalized Count'},
                       xticklabels=range(10), yticklabels=range(10), ax=ax)
            
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(fig_dir / f'{model_name}_confusion.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # ============================================================
        # 2. MODEL COMPARISON
        # ============================================================
        print("2. Creating model comparison chart...")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            values = [self.all_results[m]['classification'][metric] for m in model_names]
            bars = axes[idx].bar(model_names, values, color=colors)
            axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            axes[idx].set_ylim([0, 1.0])
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison', 
                               fontsize=12, fontweight='bold')
            
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(fig_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ============================================================
        # 3. PER-CLASS PERFORMANCE
        # ============================================================
        print("3. Creating per-class performance charts...")
        
        for model_name, results in self.all_results.items():
            fig, ax = plt.subplots(figsize=(12, 6))
            
            precision = results['classification']['per_class_precision']
            recall = results['classification']['per_class_recall']
            f1 = results['classification']['per_class_f1']

# ========================================================================
    # Convert to numpy arrays if they aren't already

            precision = np.array(precision)
            recall = np.array(recall)
            f1 = np.array(f1)
    
    # Ensure all have same length (take minimum)
            min_length = min(len(precision), len(recall), len(f1))
            if len(precision) != len(recall) or len(precision) != len(f1):
                print(f"⚠️ Fixing shape mismatch for {model_name}: "
                      f"Precision({len(precision)}), Recall({len(recall)}), F1({len(f1)}) → Using {min_length}")

            precision = precision[:min_length]
            recall = recall[:min_length]
            f1 = f1[:min_length]
# ========================================================================

            # Create x-axis based on actual number of classes
            classes = range(min_length)
            x = np.arange(len(classes))
            width = 0.25
            
            ax.bar(x - width, precision, width, label='Precision', color='#FF6B6B')
            ax.bar(x, recall, width, label='Recall', color='#4ECDC4')
            ax.bar(x + width, f1, width, label='F1-Score', color='#45B7D1')
            
            ax.set_xlabel('Digit Class', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(f'Per-Class Performance - {model_name}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(classes)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.1])
            
            plt.tight_layout()
            plt.savefig(fig_dir / f'{model_name}_per_class.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # ============================================================
        # 4. RECONSTRUCTION QUALITY
        # ============================================================
        print("4. Creating reconstruction visualizations...")
        
        for model_name, results in self.all_results.items():
            if 'reconstructions' not in results['reconstruction']:
                continue
            
            originals = results['reconstruction']['originals']
            reconstructions = results['reconstruction']['reconstructions']
            
            fig, axes = plt.subplots(3, 10, figsize=(15, 5))
            
            for i in range(10):
                # Original
                axes[0, i].imshow(originals[i].squeeze(), cmap='gray')
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_title('Original', fontsize=10, fontweight='bold', loc='left')
                
                # Reconstruction
                axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_title('Reconstructed', fontsize=10, fontweight='bold', loc='left')
                
                # Difference
                diff = np.abs(originals[i].squeeze().numpy() - reconstructions[i].squeeze().numpy())
                axes[2, i].imshow(diff, cmap='hot')
                axes[2, i].axis('off')
                if i == 0:
                    axes[2, i].set_title('Difference', fontsize=10, fontweight='bold', loc='left')
            
            plt.suptitle(f'Reconstruction Quality - {model_name}', 
                        fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(fig_dir / f'{model_name}_reconstruction.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # ============================================================
        # 5. COMPUTATIONAL COMPARISON
        # ============================================================
        print("5. Creating computational comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        params = [self.all_results[m]['computational']['total_parameters']/1e6 
                 for m in model_names]
        times = [self.all_results[m]['computational']['avg_inference_time']*1000 
                for m in model_names]
        
        bars1 = axes[0].bar(model_names, params, color=colors, edgecolor='black')
        axes[0].set_ylabel('Parameters (Millions)', fontsize=12)
        axes[0].set_title('Model Size Comparison', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}M', ha='center', va='bottom', fontsize=9)
        
        bars2 = axes[1].bar(model_names, times, color=colors, edgecolor='black')
        axes[1].set_ylabel('Inference Time (ms/sample)', fontsize=12)
        axes[1].set_title('Inference Speed Comparison', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}ms', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'computational_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ============================================================
        # 6. SUMMARY TABLE
        # ============================================================
        print("6. Creating summary table...")
        
        data = []
        for model_name, results in self.all_results.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{results['classification']['accuracy']:.4f}",
                'Precision': f"{results['classification']['precision']:.4f}",
                'Recall': f"{results['classification']['recall']:.4f}",
                'F1-Score': f"{results['classification']['f1_score']:.4f}",
                'Parameters': f"{results['computational']['total_parameters']/1e6:.2f}M",
                'Inference (ms)': f"{results['computational']['avg_inference_time']*1000:.2f}"
            }
            if 'avg_mse_loss' in results['reconstruction']:
                row['MSE Loss'] = f"{results['reconstruction']['avg_mse_loss']:.6f}"
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(self.results_dir / 'tables' / 'summary_table.csv', index=False)
        
        fig, ax = plt.subplots(figsize=(14, 3))
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
        
        plt.title('Model Comparison Summary', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(fig_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ All visualizations saved to {fig_dir}")
        print(f"✓ Summary table saved to {self.results_dir / 'tables' / 'summary_table.csv'}")
    
    def run_complete_evaluation(self, data_dir='./data'):
        """
        Execute complete Task 1 evaluation workflow
        
        Args:
            data_dir: Directory containing model weights and data
        """
        print("\n" + "="*80)
        print("TASK 1: COMPLETE EVALUATION WORKFLOW")
        print("="*80)
        
        # Step 1: Load models and data
        models, test_loader = self.load_models_and_data(data_dir)
        
        if not models:
            print("\n✗ No models loaded. Please check your data directory and model files.")
            return
        
        if test_loader is None:
            print("\n✗ Test data not loaded. Please check your test_dataset.pt file.")
            return
        
        # Step 2: Evaluate each model
        for model_name, model in models.items():
            results = self.evaluate_single_model(model, model_name, test_loader)
            self.all_results[model_name] = results
        
        # Step 3: Create visualizations
        if self.all_results:
            self.create_all_visualizations()
            self.create_tsne_visualizations(models, test_loader)            
        
        # Step 4: Generate summary report
        self.generate_text_report()
        
        print("\n" + "="*80)
        print("TASK 1 EVALUATION COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {self.results_dir}")
        print("\nGenerated files:")
        print(f"  - Figures: {self.results_dir / 'figures'}")
        print(f"  - Tables: {self.results_dir / 'tables'}")
        print(f"  - Report: {self.results_dir / 'evaluation_report.txt'}")
    
    def generate_text_report(self):
        """
        Generate a text summary report
        """
        report_path = self.results_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TASK 1: MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            for model_name, results in self.all_results.items():
                f.write(f"\n{model_name.upper()}\n")
                f.write("-"*40 + "\n\n")
                
                f.write("Classification Performance:\n")
                f.write(f"  Accuracy:  {results['classification']['accuracy']:.4f}\n")
                f.write(f"  Precision: {results['classification']['precision']:.4f}\n")
                f.write(f"  Recall:    {results['classification']['recall']:.4f}\n")
                f.write(f"  F1-Score:  {results['classification']['f1_score']:.4f}\n")
                if results['classification']['mean_auc']:
                    f.write(f"  Mean AUC:  {results['classification']['mean_auc']:.4f}\n")
                
                if 'avg_mse_loss' in results['reconstruction']:
                    f.write("\nReconstruction Performance:\n")
                    f.write(f"  MSE Loss:  {results['reconstruction']['avg_mse_loss']:.6f}\n")
                    f.write(f"  Mean PSNR: {results['reconstruction']['mean_psnr']:.2f} dB\n")
                
                f.write("\nComputational Efficiency:\n")
                f.write(f"  Parameters: {results['computational']['total_parameters']:,}\n")
                f.write(f"  Inference:  {results['computational']['avg_inference_time']*1000:.2f} ms/sample\n")
                f.write("\n")
        
        print(f"\n✓ Text report saved to {report_path}")


##
    def create_tsne_visualizations(self, models, test_loader):
        """
        Create t-SNE visualizations for latent space representations
        """
        print("\n7. Creating t-SNE latent space visualizations...")
        
        #fig_dir = self.results_dir / 'figures'
        fig_dir = self.results_dir / 'figures'
        
        print(fig_dir)
        
        for model_name, model in models.items():
            print(f"  Generating t-SNE for {model_name}...")
            
            model.eval()
            latents = []
            labels = []
            
            # Collect latent representations (use first 500 samples for speed)
            with torch.no_grad():
                sample_count = 0
                for data, batch_labels in test_loader:
                    if sample_count >= 500:
                        break
                    
                    data = data.to(self.device)
                    outputs = model(data)
                    
                    # Try to extract latent representation
                    latent = None
                    
                    if hasattr(model, 'encoder'):
                        try:
                            encoder_output = model.encoder(data)
                            if isinstance(encoder_output, tuple):
                                latent = encoder_output[0]
                            else:
                                latent = encoder_output
                        except:
                            pass
                    
                    if latent is None and isinstance(outputs, tuple):
                        for out in outputs:
                            if isinstance(out, torch.Tensor) and len(out.shape) == 2:
                                latent = out
                                break
                    
                    if latent is None and hasattr(model, 'features'):
                        try:
                            latent = model.features(data)
                            if len(latent.shape) > 2:
                                latent = latent.view(latent.size(0), -1)
                        except:
                            pass
                    
                    if latent is not None:
                        latents.append(latent.cpu().numpy())
                        labels.append(batch_labels.cpu().numpy())
                        sample_count += len(batch_labels)
            
            if not latents:
                print(f"    ⚠ Could not extract latent features from {model_name}")
                continue
            
            latents_np = np.concatenate(latents, axis=0)
            labels_np = np.concatenate(labels, axis=0)
            
            print(f"    Extracted {latents_np.shape[0]} samples, latent dim: {latents_np.shape[1]}")
            
            try:
                # Try with max_iter (newer scikit-learn versions)
                tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
            except TypeError:
                # Fall back to n_iter (older scikit-learn versions)
                tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
                
                tsne_result = tsne.fit_transform(latents_np)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                   c=labels_np, cmap='tab10', s=30, alpha=0.7, 
                                   edgecolors='w', linewidth=0.5)
                
                cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
                cbar.set_label('Digit Class', fontsize=12)
                
                ax.set_xlabel('t-SNE Component 1', fontsize=12)
                ax.set_ylabel('t-SNE Component 2', fontsize=12)
                ax.set_title(f'Latent Space Visualization (t-SNE) - {model_name}', 
                           fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)
                
                ax.text(0.02, 0.98, f'n={len(tsne_result)} samples\nperplexity=30',
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                
                tsne_path = fig_dir / f'{model_name}_tsne_latent.png'
                plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"    ✓ Saved t-SNE visualization to {tsne_path}")
                
            except Exception as e:
                print(f"    ⚠ Error running t-SNE for {model_name}: {e}")
##


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    """
    Run this script to evaluate all three models for Task 1
    
    USAGE:
    ------
    1. Ensure you have the following directory structure:
       ./data/
         ├── amml_model0_weights.pth
         ├── amml_model1_weights.pth
         ├── amml_model2_weights.pth
         └── test_dataset.pt
    
    2. Update the YourModelClass import in load_models_and_data() method
       to match the actual model architecture from src/model.py
    
    3. Run: python task1_evaluation.py
    
    4. Results will be saved to ./results/
    """
    
    # Create evaluator
    evaluator = Task1Evaluator(results_dir='./results')
    
    # Run complete evaluation
    evaluator.run_complete_evaluation(data_dir='./data')
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review the generated visualizations in ./results/figures/")
    print("2. Check the summary table in ./results/tables/summary_table.csv")
    print("3. Read the text report in ./results/evaluation_report.txt")
    print("4. Include these figures in your report")
    print("5. Analyze why models perform differently (Task 1 requirement)")
    print("\nKey questions to answer in your report:")
    print("  - Which model performs best overall?")
    print("  - Which model excels at specific digit classes?")
    print("  - What's the trade-off between accuracy and efficiency?")
    print("  - How do reconstruction qualities differ?")
    print("  - What architectural differences explain performance gaps?")
    print("="*80)
