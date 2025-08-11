#!/usr/bin/env python3
"""
Script de fine-tuning pour le classificateur d'actions football
Entraîne TimeSformer sur un dataset custom d'actions football
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import json
import cv2
from typing import List, Tuple, Dict, Optional
import argparse
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.technical.action_classifier import (
    ActionConfig, TimeSformerBackbone, ActionDataAugmentation
)
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class FootballActionDataset(Dataset):
    """Dataset pour les actions football"""
    
    def __init__(
        self, 
        data_dir: Path,
        annotations_file: str,
        config: ActionConfig,
        split: str = 'train',
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.transform = transform
        
        # Charger les annotations
        with open(self.data_dir / annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Filtrer par split
        self.samples = [
            s for s in self.annotations['samples'] 
            if s.get('split', 'train') == split
        ]
        
        # Créer mapping des labels
        self.label_to_idx = {
            label: idx for idx, label in enumerate(config.action_classes)
        }
        
        logger.info(f"Loaded {len(self.samples)} {split} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Charger les frames
        video_path = self.data_dir / sample['video_path']
        frames = self._load_video_frames(
            video_path, 
            sample['start_frame'], 
            sample['end_frame']
        )
        
        # Échantillonner les frames
        frames = self._sample_frames(frames, self.config.num_frames)
        
        # Appliquer les transformations
        if self.transform and self.split == 'train':
            frames = self._apply_augmentations(frames)
        
        # Préprocessing standard
        frames = self._preprocess_frames(frames)
        
        # Label
        label = self.label_to_idx[sample['action']]
        
        return {
            'frames': torch.from_numpy(frames).float(),
            'label': torch.tensor(label, dtype=torch.long),
            'action': sample['action'],
            'video_id': sample.get('video_id', idx)
        }
    
    def _load_video_frames(self, video_path: Path, start: int, end: int) -> np.ndarray:
        """Charger les frames d'une vidéo"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        for i in range(start, min(end + 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        return np.array(frames)
    
    def _sample_frames(self, frames: np.ndarray, target_count: int) -> np.ndarray:
        """Échantillonner uniformément les frames"""
        num_frames = len(frames)
        
        if num_frames == target_count:
            return frames
        elif num_frames > target_count:
            indices = np.linspace(0, num_frames - 1, target_count, dtype=int)
            return frames[indices]
        else:
            indices = np.linspace(0, num_frames - 1, target_count, dtype=float)
            indices = np.round(indices).astype(int)
            return frames[indices]
    
    def _apply_augmentations(self, frames: np.ndarray) -> np.ndarray:
        """Appliquer les augmentations"""
        # Augmentations temporelles
        if np.random.rand() < 0.5:
            frames = ActionDataAugmentation.speed_variation(frames)
        
        # Augmentations spatiales (appliquées frame par frame)
        augmented = []
        for frame in frames:
            if self.transform:
                augmented_frame = self.transform(image=frame)['image']
            else:
                augmented_frame = frame
            augmented.append(augmented_frame)
        
        frames = np.array(augmented)
        
        # Autres augmentations
        if np.random.rand() < 0.3:
            frames = ActionDataAugmentation.add_noise(frames, noise_level=0.05)
        
        return frames
    
    def _preprocess_frames(self, frames: np.ndarray) -> np.ndarray:
        """Préprocessing standard"""
        processed = []
        
        for frame in frames:
            # Resize
            frame = cv2.resize(frame, self.config.frame_size)
            
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normaliser
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - self.config.mean) / self.config.std
            
            processed.append(frame)
        
        # T, H, W, C -> T, C, H, W
        processed = np.stack(processed)
        processed = processed.transpose(0, 3, 1, 2)
        
        return processed


def get_transforms(config: ActionConfig):
    """Obtenir les transformations d'augmentation"""
    train_transform = A.Compose([
        A.RandomResizedCrop(
            config.frame_size[0], 
            config.frame_size[1],
            scale=(0.8, 1.0)
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.RandomBrightnessContrast(p=0.3),
    ])
    
    val_transform = A.Compose([
        A.Resize(config.frame_size[0], config.frame_size[1])
    ])
    
    return train_transform, val_transform


class ActionClassifierTrainer:
    """Entraîneur pour le classificateur d'actions"""
    
    def __init__(
        self,
        config: ActionConfig,
        data_dir: Path,
        output_dir: Path,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        num_epochs: int = 50,
        use_wandb: bool = True
    ):
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.use_wandb = use_wandb
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Modèle
        self.model = TimeSformerBackbone(config).to(self.device)
        
        # Optimiseur
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.05
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Métriques
        self.best_val_acc = 0.0
        
        # WandB
        if self.use_wandb:
            wandb.init(
                project="football-action-classifier",
                config={
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "model": config.model_name,
                    "num_classes": config.num_classes
                }
            )
    
    def train(self, annotations_file: str = "annotations.json"):
        """Entraîner le modèle"""
        # Créer les datasets
        train_transform, val_transform = get_transforms(self.config)
        
        train_dataset = FootballActionDataset(
            self.data_dir,
            annotations_file,
            self.config,
            split='train',
            transform=train_transform
        )
        
        val_dataset = FootballActionDataset(
            self.data_dir,
            annotations_file,
            self.config,
            split='val',
            transform=val_transform
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Boucle d'entraînement
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc, val_metrics = self._validate(val_loader)
            
            # Scheduler
            self.scheduler.step()
            
            # Logging
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
            
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": self.scheduler.get_last_lr()[0]
                })
            
            # Sauvegarder le meilleur modèle
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc, is_best=True)
                logger.info(f"New best model! Val Acc: {val_acc:.3f}")
            
            # Sauvegarder checkpoint régulier
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_acc, is_best=False)
        
        # Sauvegarder le modèle final
        self._save_final_model()
        
        if self.use_wandb:
            wandb.finish()
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Entraîner une époque"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward avec mixed precision
            with autocast():
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
            
            # Backward
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Métriques
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
            
            # Log batch metrics
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_acc": 100. * correct / total
                })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """Valider le modèle"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            
            for batch in pbar:
                frames = batch['frames'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward
                with autocast():
                    outputs = self.model(frames)
                    loss = self.criterion(outputs, labels)
                
                # Métriques
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100. * correct / total:.2f}%"
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Calculer métriques par classe
        from sklearn.metrics import classification_report
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=self.config.action_classes,
            output_dict=True
        )
        
        return avg_loss, accuracy, report
    
    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool):
        """Sauvegarder un checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        if is_best:
            path = self.output_dir / 'best_model.pth'
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch + 1}.pth'
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def _save_final_model(self):
        """Sauvegarder le modèle final et l'exporter en ONNX"""
        # Sauvegarder PyTorch
        final_path = self.output_dir / 'final_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, final_path)
        
        # Exporter en ONNX
        self.model.eval()
        dummy_input = torch.randn(
            1, self.config.num_frames, 3,
            self.config.frame_size[0], self.config.frame_size[1]
        ).to(self.device)
        
        onnx_path = self.output_dir / 'timesformer_football.onnx'
        
        torch.onnx.export(
            self.model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Exported model to ONNX: {onnx_path}")
        
        # Quantization INT8 (optionnel)
        if self.device.type == 'cuda':
            self._export_int8_model(onnx_path)
    
    def _export_int8_model(self, onnx_path: Path):
        """Exporter modèle quantifié INT8"""
        try:
            import onnxruntime.quantization as ort_quant
            
            int8_path = self.output_dir / 'timesformer_football_int8.onnx'
            
            ort_quant.quantize_dynamic(
                str(onnx_path),
                str(int8_path),
                weight_type=ort_quant.QuantType.QInt8
            )
            
            logger.info(f"Exported INT8 model to {int8_path}")
            
        except ImportError:
            logger.warning("ONNX Runtime quantization not available")


def create_sample_annotations(output_path: Path):
    """Créer un fichier d'annotations exemple"""
    sample_annotations = {
        "dataset": "football_actions",
        "version": "1.0",
        "action_classes": [
            "pass", "shot", "dribble", "control", "tackle",
            "header", "cross", "clearance", "save", "throw_in",
            "goal_kick", "corner", "sprint", "jog", "stand"
        ],
        "samples": [
            {
                "video_id": "match_001_clip_001",
                "video_path": "videos/match_001.mp4",
                "start_frame": 100,
                "end_frame": 160,
                "action": "pass",
                "split": "train"
            },
            {
                "video_id": "match_001_clip_002",
                "video_path": "videos/match_001.mp4",
                "start_frame": 200,
                "end_frame": 260,
                "action": "shot",
                "split": "train"
            },
            # Ajouter plus d'exemples...
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(sample_annotations, f, indent=2)
    
    logger.info(f"Created sample annotations at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Football Action Classifier")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='models/action_classifier',
                        help='Output directory for models')
    parser.add_argument('--annotations', type=str, default='annotations.json',
                        help='Annotations file name')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--create_sample', action='store_true',
                        help='Create sample annotations file')
    
    args = parser.parse_args()
    
    # Créer annotations exemple si demandé
    if args.create_sample:
        create_sample_annotations(Path(args.data_dir) / 'annotations_sample.json')
        return
    
    # Configuration
    config = ActionConfig()
    
    # Créer l'entraîneur
    trainer = ActionClassifierTrainer(
        config=config,
        data_dir=args.data_dir,
        output_dir=Path(args.output_dir),
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        use_wandb=not args.no_wandb
    )
    
    # Entraîner
    trainer.train(annotations_file=args.annotations)
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {trainer.best_val_acc:.3f}")


if __name__ == "__main__":
    main()