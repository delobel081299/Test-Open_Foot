"""
Module de classification des actions techniques au football
Utilise TimeSformer pour analyser des séquences vidéo de 2 secondes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import cv2
from collections import deque
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None
from pathlib import Path
import json
import time
from dataclasses import dataclass
from functools import lru_cache

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ActionConfig:
    """Configuration pour le classificateur d'actions"""
    # Classes d'actions
    action_classes: List[str] = None
    num_classes: int = 15
    
    # Paramètres du modèle
    model_name: str = "timesformer_base"
    pretrained_path: Optional[str] = None
    use_onnx: bool = True
    
    # Paramètres vidéo
    window_duration: float = 2.0  # secondes
    fps: int = 30
    num_frames: int = 16  # frames à échantillonner
    frame_size: Tuple[int, int] = (224, 224)
    
    # Preprocessing
    mean: List[float] = None
    std: List[float] = None
    
    # Inference
    batch_size: int = 4
    confidence_threshold: float = 0.5
    smoothing_window: int = 5
    
    # Cache
    use_cache: bool = True
    cache_size: int = 100
    
    def __post_init__(self):
        if self.action_classes is None:
            self.action_classes = [
                "pass", "shot", "dribble", "control", "tackle",
                "header", "cross", "clearance", "save", "throw_in",
                "goal_kick", "corner", "sprint", "jog", "stand"
            ]
        
        if self.mean is None:
            self.mean = [0.485, 0.456, 0.406]
        
        if self.std is None:
            self.std = [0.229, 0.224, 0.225]


class TimeSformerBackbone(nn.Module):
    """
    Implémentation simplifiée de TimeSformer pour la classification d'actions
    """
    
    def __init__(self, config: ActionConfig):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.num_frames = config.num_frames
        self.num_patches = 14 * 14  # Pour 224x224 avec patch 16x16
        self.embed_dim = 768
        self.num_heads = 12
        self.num_layers = 12
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            3, self.embed_dim,
            kernel_size=(1, 16, 16),
            stride=(1, 16, 16)
        )
        
        # Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embed_dim)
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, self.num_frames, self.embed_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TimeSformerBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                num_frames=self.num_frames,
                num_patches=self.num_patches
            )
            for _ in range(self.num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, config.num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialisation des poids
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x.transpose(1, 2))  # B, C, T, H, W -> B, E, T, H', W'
        x = x.flatten(3).transpose(1, 2)  # B, T, E, N
        x = x.flatten(1, 2)  # B, T*N, E
        
        # Ajouter token CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Ajouter embeddings positionnels
        T = self.num_frames
        N = self.num_patches
        
        # Spatial position pour chaque frame
        x[:, 1:] = x[:, 1:] + self.pos_embed_spatial.repeat(1, T, 1)
        
        # Temporal position pour chaque patch
        temporal_pos = self.pos_embed_temporal.unsqueeze(2).repeat(1, 1, N, 1)
        temporal_pos = temporal_pos.reshape(1, T*N, -1)
        x[:, 1:] = x[:, 1:] + temporal_pos
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, T, N)
        
        # Classification
        x = self.norm(x[:, 0])  # Prendre seulement le token CLS
        x = self.head(x)
        
        return x


class TimeSformerBlock(nn.Module):
    """Block transformer avec attention spatiale et temporelle divisée"""
    
    def __init__(self, dim, num_heads, num_frames, num_patches):
        super().__init__()
        self.num_frames = num_frames
        self.num_patches = num_patches
        
        # Temporal attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn_temporal = nn.MultiheadAttention(dim, num_heads)
        
        # Spatial attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn_spatial = nn.MultiheadAttention(dim, num_heads)
        
        # MLP
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x, T, N):
        B = x.shape[0]
        
        # Temporal attention
        xt = x[:, 1:, :]  # Enlever CLS token
        xt = xt.reshape(B, T, N, -1)
        xt = xt.transpose(1, 2)  # B, N, T, C
        xt = xt.reshape(B * N, T, -1)
        
        xt = self.norm1(xt)
        xt, _ = self.attn_temporal(xt, xt, xt)
        
        # Reconstruire avec CLS token
        xt = xt.reshape(B, N, T, -1).transpose(1, 2).reshape(B, T*N, -1)
        x[:, 1:, :] = x[:, 1:, :] + xt
        
        # Spatial attention
        xs = x[:, 1:, :]
        xs = xs.reshape(B, T, N, -1)  # B, T, N, C
        xs = xs.reshape(B * T, N, -1)
        
        xs = self.norm2(xs)
        xs, _ = self.attn_spatial(xs, xs, xs)
        
        # Reconstruire
        xs = xs.reshape(B, T, N, -1).reshape(B, T*N, -1)
        x[:, 1:, :] = x[:, 1:, :] + xs
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x


class ActionClassifier:
    """
    Classificateur d'actions football utilisant TimeSformer
    """
    
    def __init__(self, config: Optional[ActionConfig] = None):
        self.config = config or ActionConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Modèle
        self.model = None
        self.onnx_session = None
        
        # Cache pour prédictions
        if self.config.use_cache:
            self.prediction_cache = {}
        
        # Buffer pour smoothing temporel
        self.prediction_buffer = deque(maxlen=self.config.smoothing_window)
        
        # Statistiques
        self.inference_times = deque(maxlen=100)
        
        # Charger le modèle
        self._load_model()
        
        logger.info(f"ActionClassifier initialized on {self.device}")
        logger.info(f"Classes: {', '.join(self.config.action_classes)}")
    
    def _load_model(self):
        """Charger le modèle (PyTorch ou ONNX)"""
        models_dir = Path("models/action_recognition")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.use_onnx:
            onnx_path = models_dir / f"{self.config.model_name}.onnx"
            if onnx_path.exists():
                self._load_onnx_model(onnx_path)
            else:
                logger.warning(f"ONNX model not found at {onnx_path}")
                self._load_pytorch_model()
                self._export_to_onnx(onnx_path)
        else:
            self._load_pytorch_model()
    
    def _load_pytorch_model(self):
        """Charger modèle PyTorch"""
        self.model = TimeSformerBackbone(self.config)
        
        if self.config.pretrained_path:
            pretrained_path = Path(self.config.pretrained_path)
            if pretrained_path.exists():
                checkpoint = torch.load(pretrained_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded pretrained weights from {pretrained_path}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def _load_onnx_model(self, onnx_path: Path):
        """Charger modèle ONNX"""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
        logger.info(f"Loaded ONNX model from {onnx_path}")
    
    def _export_to_onnx(self, onnx_path: Path):
        """Exporter le modèle PyTorch vers ONNX"""
        if self.model is None:
            return
        
        dummy_input = torch.randn(
            1, self.config.num_frames, 3,
            self.config.frame_size[0], self.config.frame_size[1]
        ).to(self.device)
        
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
        
        # Recharger en ONNX
        self._load_onnx_model(onnx_path)
    
    def classify(self, frames: Union[List[np.ndarray], np.ndarray]) -> Dict[str, any]:
        """
        Classifier une séquence de frames
        
        Args:
            frames: Liste de frames ou array numpy (T, H, W, C)
            
        Returns:
            Dict contenant:
                - action: Nom de l'action prédite
                - confidence: Score de confiance
                - probabilities: Probabilités pour toutes les classes
                - inference_time: Temps d'inférence en ms
        """
        start_time = time.time()
        
        # Préprocessing
        processed_frames = self._preprocess_frames(frames)
        
        # Vérifier le cache
        if self.config.use_cache:
            cache_key = self._compute_cache_key(processed_frames)
            if cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key].copy()
                cached_result['from_cache'] = True
                return cached_result
        
        # Inférence
        if self.onnx_session:
            predictions = self._inference_onnx(processed_frames)
        else:
            predictions = self._inference_pytorch(processed_frames)
        
        # Post-processing
        result = self._postprocess_predictions(predictions)
        
        # Smoothing temporel
        if self.config.smoothing_window > 1:
            result = self._apply_temporal_smoothing(result)
        
        # Temps d'inférence
        inference_time = (time.time() - start_time) * 1000
        result['inference_time'] = inference_time
        self.inference_times.append(inference_time)
        
        # Mettre en cache
        if self.config.use_cache and 'cache_key' in locals():
            self.prediction_cache[cache_key] = result.copy()
            
            # Limiter la taille du cache
            if len(self.prediction_cache) > self.config.cache_size:
                oldest_key = next(iter(self.prediction_cache))
                del self.prediction_cache[oldest_key]
        
        return result
    
    def classify_batch(self, frame_sequences: List[List[np.ndarray]]) -> List[Dict[str, any]]:
        """Classifier plusieurs séquences en batch"""
        # Préprocessing en batch
        all_processed = []
        for frames in frame_sequences:
            processed = self._preprocess_frames(frames)
            all_processed.append(processed)
        
        batch_tensor = np.concatenate(all_processed, axis=0)
        
        # Inférence batch
        if self.onnx_session:
            predictions = self._inference_onnx(batch_tensor)
        else:
            predictions = self._inference_pytorch(batch_tensor)
        
        # Post-processing pour chaque séquence
        results = []
        for i in range(len(frame_sequences)):
            pred = predictions[i:i+1]
            result = self._postprocess_predictions(pred)
            results.append(result)
        
        return results
    
    def _preprocess_frames(self, frames: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """Préprocesser les frames pour le modèle"""
        # Convertir en array si nécessaire
        if isinstance(frames, list):
            frames = np.array(frames)
        
        # Échantillonner le bon nombre de frames
        if len(frames) != self.config.num_frames:
            frames = self._sample_frames(frames, self.config.num_frames)
        
        # Resize et normalisation
        processed = []
        for frame in frames:
            # Resize
            frame = cv2.resize(frame, self.config.frame_size)
            
            # Convert BGR to RGB si nécessaire
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normaliser
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - self.config.mean) / self.config.std
            
            processed.append(frame)
        
        # Stack et réorganiser dimensions
        processed = np.stack(processed)  # T, H, W, C
        processed = processed.transpose(0, 3, 1, 2)  # T, C, H, W
        processed = np.expand_dims(processed, 0)  # 1, T, C, H, W
        
        return processed
    
    def _sample_frames(self, frames: np.ndarray, target_count: int) -> np.ndarray:
        """Échantillonner uniformément les frames"""
        num_frames = len(frames)
        
        if num_frames == target_count:
            return frames
        elif num_frames > target_count:
            # Sous-échantillonner
            indices = np.linspace(0, num_frames - 1, target_count, dtype=int)
            return frames[indices]
        else:
            # Sur-échantillonner avec répétition
            indices = np.linspace(0, num_frames - 1, target_count, dtype=float)
            indices = np.round(indices).astype(int)
            return frames[indices]
    
    def _inference_pytorch(self, frames: np.ndarray) -> np.ndarray:
        """Inférence avec PyTorch"""
        with torch.no_grad():
            input_tensor = torch.from_numpy(frames).float().to(self.device)
            output = self.model(input_tensor)
            predictions = F.softmax(output, dim=-1)
            return predictions.cpu().numpy()
    
    def _inference_onnx(self, frames: np.ndarray) -> np.ndarray:
        """Inférence avec ONNX Runtime"""
        input_name = self.onnx_session.get_inputs()[0].name
        output = self.onnx_session.run(None, {input_name: frames.astype(np.float32)})
        predictions = self._softmax(output[0])
        return predictions
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax numpy"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _postprocess_predictions(self, predictions: np.ndarray) -> Dict[str, any]:
        """Post-traiter les prédictions"""
        probs = predictions[0]  # Première (et seule) prédiction du batch
        
        # Top prediction
        top_idx = np.argmax(probs)
        top_prob = probs[top_idx]
        
        # Créer le résultat
        result = {
            'action': self.config.action_classes[top_idx],
            'confidence': float(top_prob),
            'probabilities': {
                cls: float(prob)
                for cls, prob in zip(self.config.action_classes, probs)
            }
        }
        
        # Ajouter top-k predictions
        top_k = 3
        top_indices = np.argsort(probs)[-top_k:][::-1]
        result['top_k'] = [
            {
                'action': self.config.action_classes[idx],
                'confidence': float(probs[idx])
            }
            for idx in top_indices
        ]
        
        return result
    
    def _apply_temporal_smoothing(self, result: Dict[str, any]) -> Dict[str, any]:
        """Appliquer un lissage temporel aux prédictions"""
        self.prediction_buffer.append(result['probabilities'])
        
        if len(self.prediction_buffer) < 2:
            return result
        
        # Moyenner les probabilités sur la fenêtre
        all_probs = []
        for probs_dict in self.prediction_buffer:
            probs_array = [probs_dict[cls] for cls in self.config.action_classes]
            all_probs.append(probs_array)
        
        avg_probs = np.mean(all_probs, axis=0)
        
        # Recréer le résultat avec les probabilités lissées
        top_idx = np.argmax(avg_probs)
        
        smoothed_result = {
            'action': self.config.action_classes[top_idx],
            'confidence': float(avg_probs[top_idx]),
            'probabilities': {
                cls: float(prob)
                for cls, prob in zip(self.config.action_classes, avg_probs)
            },
            'smoothed': True
        }
        
        # Conserver le temps d'inférence original
        if 'inference_time' in result:
            smoothed_result['inference_time'] = result['inference_time']
        
        return smoothed_result
    
    def _compute_cache_key(self, frames: np.ndarray) -> str:
        """Calculer une clé de cache pour les frames"""
        # Utiliser un hash simple des frames sous-échantillonnées
        downsampled = frames[:, ::4, ::8, ::8, ::8]  # Réduire fortement
        return str(hash(downsampled.tobytes()))
    
    @lru_cache(maxsize=1000)
    def get_action_embedding(self, action: str) -> np.ndarray:
        """Obtenir l'embedding d'une action (pour la similarité)"""
        if action not in self.config.action_classes:
            return np.zeros(self.config.num_classes)
        
        embedding = np.zeros(self.config.num_classes)
        embedding[self.config.action_classes.index(action)] = 1.0
        return embedding
    
    def get_stats(self) -> Dict[str, any]:
        """Obtenir les statistiques du classificateur"""
        stats = {
            'device': str(self.device),
            'model_type': 'ONNX' if self.onnx_session else 'PyTorch',
            'num_classes': self.config.num_classes,
            'classes': self.config.action_classes,
            'cache_size': len(self.prediction_cache) if self.config.use_cache else 0,
            'smoothing_window': self.config.smoothing_window
        }
        
        if self.inference_times:
            stats['inference_stats'] = {
                'mean_ms': np.mean(self.inference_times),
                'std_ms': np.std(self.inference_times),
                'min_ms': np.min(self.inference_times),
                'max_ms': np.max(self.inference_times),
                'last_ms': self.inference_times[-1]
            }
        
        return stats


class ActionDataAugmentation:
    """Augmentations spécifiques pour les actions football"""
    
    @staticmethod
    def random_crop(frames: np.ndarray, scale=(0.8, 1.0)) -> np.ndarray:
        """Crop aléatoire spatial"""
        h, w = frames.shape[2:4]
        scale_factor = np.random.uniform(scale[0], scale[1])
        
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        
        cropped = frames[:, :, top:top+new_h, left:left+new_w]
        
        # Resize back
        resized = []
        for frame in cropped:
            frame = frame.transpose(1, 2, 0)
            frame = cv2.resize(frame, (w, h))
            resized.append(frame.transpose(2, 0, 1))
        
        return np.array(resized)
    
    @staticmethod
    def random_flip(frames: np.ndarray, p=0.5) -> np.ndarray:
        """Flip horizontal aléatoire"""
        if np.random.random() < p:
            return frames[:, :, :, ::-1]
        return frames
    
    @staticmethod
    def speed_variation(frames: np.ndarray, speed_range=(0.8, 1.2)) -> np.ndarray:
        """Variation de vitesse par interpolation"""
        speed_factor = np.random.uniform(speed_range[0], speed_range[1])
        num_frames = len(frames)
        new_num_frames = int(num_frames * speed_factor)
        
        if new_num_frames == num_frames:
            return frames
        
        # Indices pour interpolation
        old_indices = np.linspace(0, num_frames - 1, num_frames)
        new_indices = np.linspace(0, num_frames - 1, new_num_frames)
        
        # Interpoler chaque channel
        interpolated = []
        for c in range(frames.shape[1]):
            channel_frames = []
            for y in range(frames.shape[2]):
                for x in range(frames.shape[3]):
                    values = frames[:, c, y, x]
                    interp = np.interp(new_indices, old_indices, values)
                    channel_frames.append(interp)
            
            channel_frames = np.array(channel_frames).reshape(
                frames.shape[2], frames.shape[3], new_num_frames
            ).transpose(2, 0, 1)
            interpolated.append(channel_frames)
        
        interpolated = np.array(interpolated).transpose(1, 0, 2, 3)
        
        # Échantillonner pour revenir au nombre original
        if len(interpolated) > num_frames:
            indices = np.linspace(0, len(interpolated) - 1, num_frames, dtype=int)
            return interpolated[indices]
        else:
            # Pad si nécessaire
            pad_size = num_frames - len(interpolated)
            padding = np.repeat(interpolated[-1:], pad_size, axis=0)
            return np.concatenate([interpolated, padding], axis=0)
    
    @staticmethod
    def add_noise(frames: np.ndarray, noise_level=0.1) -> np.ndarray:
        """Ajouter du bruit gaussien"""
        noise = np.random.normal(0, noise_level, frames.shape)
        return np.clip(frames + noise, -1, 1)
    
    @staticmethod
    def mixup(frames1: np.ndarray, frames2: np.ndarray, 
              label1: int, label2: int, alpha=0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Mixup entre deux séquences"""
        lam = np.random.beta(alpha, alpha)
        
        mixed_frames = lam * frames1 + (1 - lam) * frames2
        
        # Labels mixés (one-hot)
        num_classes = 15  # Ou depuis config
        mixed_label = np.zeros(num_classes)
        mixed_label[label1] = lam
        mixed_label[label2] = 1 - lam
        
        return mixed_frames, mixed_label


# Fonction utilitaire pour évaluation
def evaluate_classifier(
    classifier: ActionClassifier,
    test_data: List[Tuple[List[np.ndarray], str]],
    save_path: Optional[Path] = None
) -> Dict[str, any]:
    """
    Évaluer le classificateur sur un dataset de test
    
    Args:
        classifier: Instance du classificateur
        test_data: Liste de tuples (frames, action_label)
        save_path: Chemin pour sauvegarder les résultats
        
    Returns:
        Métriques d'évaluation
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    y_true = []
    y_pred = []
    confidences = []
    inference_times = []
    
    logger.info(f"Evaluating classifier on {len(test_data)} samples...")
    
    for i, (frames, true_action) in enumerate(test_data):
        result = classifier.classify(frames)
        
        y_true.append(true_action)
        y_pred.append(result['action'])
        confidences.append(result['confidence'])
        inference_times.append(result['inference_time'])
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(test_data)} samples")
    
    # Calculer métriques
    classes = classifier.config.action_classes
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True)
    
    # Accuracy par classe
    class_accuracies = {}
    for i, cls in enumerate(classes):
        if cm[i].sum() > 0:
            class_accuracies[cls] = cm[i, i] / cm[i].sum()
        else:
            class_accuracies[cls] = 0.0
    
    # Résultats
    results = {
        'overall_accuracy': report['accuracy'],
        'class_accuracies': class_accuracies,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'inference_stats': {
            'mean_ms': np.mean(inference_times),
            'std_ms': np.std(inference_times),
            'min_ms': np.min(inference_times),
            'max_ms': np.max(inference_times),
            'below_50ms_ratio': sum(t < 50 for t in inference_times) / len(inference_times)
        },
        'confidence_stats': {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }
    }
    
    # Sauvegarder si demandé
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {save_path}")
    
    # Log summary
    logger.info(f"Overall accuracy: {results['overall_accuracy']:.3f}")
    logger.info(f"Mean inference time: {results['inference_stats']['mean_ms']:.1f}ms")
    logger.info(f"Below 50ms: {results['inference_stats']['below_50ms_ratio']:.1%}")
    
    return results