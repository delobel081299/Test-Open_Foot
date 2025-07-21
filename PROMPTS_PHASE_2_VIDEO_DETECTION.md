# 🎥 PROMPTS VIBE CODING - PHASE 2 : VIDÉO & DÉTECTION

## 📅 Durée : 1.5 semaines

## 🎯 Objectifs
- Module de prétraitement vidéo robuste
- Détection et tracking des joueurs/ballon
- Classification des équipes
- Optimisation des performances

---

## 1️⃣ Prompt Prétraitement Vidéo Avancé

```
Implémente un module de prétraitement vidéo professionnel pour l'analyse football :

1. CLASSE VIDEOPROCESSOR COMPLÈTE :

```python
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np

@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str
    bitrate: int
    has_audio: bool

@dataclass
class QualityReport:
    resolution_score: float  # 0-1
    sharpness_score: float  # 0-1
    lighting_score: float   # 0-1
    stability_score: float  # 0-1
    overall_score: float    # 0-1
    issues: List[str]       # ["low_light", "motion_blur", etc.]

class VideoProcessor:
    def __init__(self, config: VideoConfig):
        self.config = config
        
    async def process_video(self, video_path: str) -> ProcessedVideo:
        """Pipeline complet de prétraitement"""
        # 1. Validation et métadonnées
        # 2. Extraction frames optimisée
        # 3. Stabilisation si nécessaire
        # 4. Enhancement qualité
        # 5. Segmentation temporelle
        # 6. Export optimisé
```

2. STABILISATION AVANCÉE :
   - Détection points SIFT/ORB
   - Estimation homographie
   - Compensation mouvement caméra
   - Smooth temporal filtering
   - Crop intelligent sans perte d'info

3. QUALITY ENHANCEMENT :
   - Super-résolution ESRGAN si < 720p
   - Denoising adaptatif
   - Correction exposition/contraste
   - Sharpening intelligent
   - Color correction pour gazon

4. SEGMENTATION ACTIONS :
   - Détection changements de scène
   - Clustering temporal des actions
   - Extraction clips pertinents
   - Métadonnées par segment

5. OPTIMISATIONS GPU :
   - Batch processing frames
   - Pipeline CUDA si disponible
   - Memory pooling
   - Async I/O

Implémente avec gestion erreurs robuste et progress tracking.
```

## 2️⃣ Prompt Détection YOLOv10 Football

```
Configure et fine-tune YOLOv10 spécifiquement pour le football :

1. CLASSES PERSONNALISÉES :
```yaml
classes:
  0: player_team_a
  1: player_team_b
  2: goalkeeper_team_a
  3: goalkeeper_team_b
  4: referee
  5: linesman
  6: ball
  7: goal_post
  8: corner_flag
  9: penalty_spot
```

2. AUGMENTATIONS SPÉCIFIQUES :
```python
augmentations = A.Compose([
    # Variations terrain
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    
    # Flou mouvement ballon
    A.MotionBlur(blur_limit=7, p=0.3),
    
    # Variations météo
    A.RandomRain(p=0.1),
    A.RandomFog(p=0.1),
    
    # Perspectives caméra
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    
    # Occlusions joueurs
    A.CoarseDropout(max_holes=3, max_height=50, max_width=50, p=0.2)
])
```

3. ARCHITECTURE OPTIMISÉE :
   - Anchor boxes adaptés taille ballon
   - FPN pour multi-échelles
   - Attention modules pour petits objets
   - NMS optimisé groupes joueurs

4. TRAINING STRATEGY :
   - Dataset : 50k images annotées
   - Learning rate scheduling
   - Early stopping intelligent
   - Validation par type de vue

5. POST-PROCESSING :
   - Filtrage détections par confiance
   - Merge boxes joueurs proches
   - Tracking prioritaire ballon
   - Association équipe par couleur

Génère pipeline complet avec métriques football.
```

## 3️⃣ Prompt Architecture Hybride Progressive

```
Implémente une architecture de détection hybride progressive pour le football :

1. PIPELINE PRINCIPAL :
```python
from ultralytics import YOLO, RTDETR
from transformers import AutoModelForObjectDetection
import torch
import numpy as np
from typing import List, Dict, Optional
import time

class HybridFootballDetector:
    """
    Détection progressive : YOLOv10 → RT-DETR → DINO-DETR
    """
    def __init__(self):
        # Phase 1 : YOLOv10 (toujours actif)
        self.yolo = YOLO('yolov10x.pt')
        self.yolo.overrides.update({
            'conf': 0.5,
            'iou': 0.5,
            'imgsz': 1280,
            'device': 'cuda:0'
        })
        
        # Phase 2 : RT-DETR (zones denses)
        self.rtdetr = None  # Lazy loading
        self.rtdetr_config = {
            'model': 'rtdetr-l.pt',
            'conf': 0.4,
            'imgsz': 1280
        }
        
        # Phase 3 : DINO-DETR (précision max)
        self.dino = None  # Lazy loading
        self.dino_config = {
            'model': 'IDEA-Research/dino-detr-r50',
            'threshold': 0.35
        }
        
        # Seuils de décision
        self.density_threshold = 5  # joueurs/zone
        self.occlusion_threshold = 0.3
        self.precision_threshold = 0.85
```

2. ANALYSE DE SCÈNE INTELLIGENTE :
```python
def analyze_scene_complexity(self, detections, frame):
    """
    Détermine si RT-DETR ou DINO-DETR sont nécessaires
    """
    # Grille de densité
    h, w = frame.shape[:2]
    grid_size = 100
    density_grid = np.zeros((h//grid_size+1, w//grid_size+1))
    
    # Calcul densité par zone
    for det in detections:
        if det['class'] in ['player', 'goalkeeper']:
            cx = (det['bbox'][0] + det['bbox'][2]) // 2
            cy = (det['bbox'][1] + det['bbox'][3]) // 2
            grid_x, grid_y = cx // grid_size, cy // grid_size
            density_grid[grid_y, grid_x] += 1
    
    # Identifier zones denses
    dense_regions = []
    for y in range(density_grid.shape[0]):
        for x in range(density_grid.shape[1]):
            if density_grid[y, x] >= self.density_threshold:
                dense_regions.append({
                    'bbox': [x*grid_size, y*grid_size, 
                            (x+1)*grid_size, (y+1)*grid_size],
                    'density': density_grid[y, x]
                })
    
    return {
        'has_dense_areas': len(dense_regions) > 0,
        'dense_regions': dense_regions,
        'max_density': np.max(density_grid),
        'occlusion_score': self._calculate_occlusion(detections)
    }
```

3. DÉTECTION PROGRESSIVE :
```python
def detect_progressive(self, frame):
    """
    Pipeline de détection adaptatif
    """
    results = {
        'detections': [],
        'methods_used': [],
        'timing': {}
    }
    
    # PHASE 1 : YOLOv10 (toujours)
    t0 = time.time()
    yolo_dets = self.yolo(frame, verbose=False)
    results['detections'] = self._parse_yolo(yolo_dets)
    results['timing']['yolo'] = time.time() - t0
    results['methods_used'].append('YOLOv10')
    
    # Analyse complexité
    complexity = self.analyze_scene_complexity(
        results['detections'], frame
    )
    
    # PHASE 2 : RT-DETR si zones denses
    if complexity['has_dense_areas']:
        if self.rtdetr is None:
            self._load_rtdetr()
        
        t0 = time.time()
        for region in complexity['dense_regions']:
            roi = self._extract_roi(frame, region['bbox'])
            rtdetr_dets = self.rtdetr(roi, verbose=False)
            
            # Fusionner détections
            new_dets = self._parse_rtdetr(rtdetr_dets, region)
            results['detections'] = self._merge_detections(
                results['detections'], new_dets
            )
        
        results['timing']['rtdetr'] = time.time() - t0
        results['methods_used'].append('RT-DETR')
    
    # PHASE 3 : DINO-DETR si précision insuffisante
    if self._needs_high_precision(results, complexity):
        if self.dino is None:
            self._load_dino()
        
        t0 = time.time()
        dino_dets = self._detect_dino(frame)
        results['detections'] = self._refine_with_dino(
            results['detections'], dino_dets
        )
        results['timing']['dino'] = time.time() - t0
        results['methods_used'].append('DINO-DETR')
    
    return results
```

4. FUSION INTELLIGENTE :
```python
def _merge_detections(self, primary, secondary):
    """
    Fusionne détections en évitant doublons
    """
    merged = primary.copy()
    
    for sec_det in secondary:
        is_duplicate = False
        
        for i, prim_det in enumerate(merged):
            iou = self._calculate_iou(
                prim_det['bbox'], sec_det['bbox']
            )
            
            if iou > 0.5 and sec_det['class'] == prim_det['class']:
                # Garder meilleure confiance
                if sec_det['confidence'] > prim_det['confidence']:
                    merged[i] = sec_det
                is_duplicate = True
                break
        
        if not is_duplicate:
            merged.append(sec_det)
    
    return merged
```

5. MÉTRIQUES ET MONITORING :
```python
class DetectionMonitor:
    def __init__(self):
        self.stats = {
            'yolo_only': 0,
            'yolo_rtdetr': 0,
            'full_pipeline': 0,
            'avg_fps': 0,
            'avg_precision': 0
        }
    
    def update(self, results):
        # Compteurs utilisation
        methods = len(results['methods_used'])
        if methods == 1:
            self.stats['yolo_only'] += 1
        elif methods == 2:
            self.stats['yolo_rtdetr'] += 1
        else:
            self.stats['full_pipeline'] += 1
        
        # FPS
        total_time = sum(results['timing'].values())
        fps = 1.0 / total_time
        self.stats['avg_fps'] = 0.9 * self.stats['avg_fps'] + 0.1 * fps
        
    def report(self):
        total = sum([
            self.stats['yolo_only'],
            self.stats['yolo_rtdetr'],
            self.stats['full_pipeline']
        ])
        
        print(f"YOLOv10 seul : {self.stats['yolo_only']/total*100:.1f}%")
        print(f"YOLOv10+RT-DETR : {self.stats['yolo_rtdetr']/total*100:.1f}%")
        print(f"Pipeline complet : {self.stats['full_pipeline']/total*100:.1f}%")
        print(f"FPS moyen : {self.stats['avg_fps']:.1f}")
```

6. OPTIMISATIONS PRODUCTION :
- Export TensorRT pour YOLOv10 et RT-DETR
- Batching intelligent pour throughput
- Cache GPU pour modèles
- Profiling et monitoring temps réel

Implémente cette architecture avec gestion erreurs robuste.
```

## 4️⃣ Prompt ByteTrack Multi-Objets

```
Implémente un système de tracking robuste avec ByteTrack pour le football :

1. TRACKER PERSONNALISÉ :
```python
class FootballTracker:
    def __init__(self):
        self.player_tracker = ByteTracker(
            track_thresh=0.6,
            match_thresh=0.8,
            frame_rate=30
        )
        self.ball_tracker = ByteTracker(
            track_thresh=0.4,  # Plus permissif pour le ballon
            match_thresh=0.7,
            frame_rate=30
        )
        
    def track_frame(self, detections: List[Detection]) -> TrackedObjects:
        # Sépare joueurs et ballon
        # Applique tracking spécialisé
        # Gère les occlusions
        # Prédit positions manquantes
```

2. GESTION OCCLUSIONS :
   - Historique positions par ID
   - Prédiction Kalman Filter
   - Re-identification par features
   - Interpolation trajectoires

3. FEATURES SPÉCIFIQUES :
   - Priorité tracking ballon
   - Association joueur-équipe stable
   - Détection sorties terrain
   - Gestion substitutions

4. OPTIMISATIONS :
   - ROI tracking (zone active)
   - Batch processing GPU
   - Cache features visuelles
   - Multi-threading

5. MÉTRIQUES TRACKING :
   - MOTA/MOTP scores
   - ID switches count
   - Fragment tracks
   - Précision par classe

Intègre avec visualisation temps réel des tracks.
```

## 5️⃣ Prompt Classification Équipes

```
Développe un système intelligent de classification des équipes :

1. EXTRACTION COULEURS :
```python
class TeamClassifier:
    def extract_jersey_colors(self, player_crop: np.ndarray) -> ColorProfile:
        # 1. Segmentation jersey (remove background)
        # 2. Clustering couleurs dominantes (K-means)
        # 3. Filtrage couleurs peau/cheveux
        # 4. Extraction couleur principale + secondaire
        
    def classify_teams(self, all_players: List[Player]) -> TeamAssignment:
        # 1. Extraction profils couleurs tous joueurs
        # 2. Clustering en 2-3 équipes (+ arbitres)
        # 3. Validation cohérence spatiale
        # 4. Gestion cas edge (maillots similaires)
```

2. APPRENTISSAGE DYNAMIQUE :
   - Mise à jour profils durant match
   - Adaptation changements luminosité
   - Gestion maillots domicile/extérieur
   - Détection gardiens différents

3. CAS PARTICULIERS :
   - Maillots bicolores
   - Sponsors similaires
   - Conditions météo (pluie, boue)
   - Éclairage nocturne

4. VALIDATION :
   - Cohérence formations tactiques
   - Proximité spatiale équipes
   - Règles football (11 par équipe max)
   - Confirmation manuelle si doute

5. PERSISTENCE :
   - Sauvegarde profils équipes
   - Réutilisation matchs suivants
   - Export pour stats équipe

Implémente avec UI de validation/correction.
```

## 6️⃣ Prompt Optimisation Pipeline

```
Optimise le pipeline complet de traitement vidéo pour performance maximale :

1. PROFILING DÉTAILLÉ :
```python
import cProfile
import torch.profiler

class PerformanceMonitor:
    def profile_pipeline(self, video_path: str):
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True
        ) as prof:
            # Profile chaque étape
            # Identifie bottlenecks
            # Génère rapport visuel
```

2. OPTIMISATIONS GPU :
   - Batch size dynamique
   - Mixed precision (FP16)
   - TensorRT pour inference
   - Multi-GPU si disponible
   - Memory pinning

3. OPTIMISATIONS CPU :
   - Multiprocessing pool
   - SIMD operations (NumPy)
   - Cython fonctions critiques
   - Cache LRU résultats

4. PIPELINE STREAMING :
   - Traitement par chunks
   - Queue producer/consumer
   - Overlap I/O et compute
   - Early termination

5. BENCHMARKS CIBLES :
   - 720p @ 30fps : temps réel (1x)
   - 1080p @ 30fps : 0.5x temps réel
   - 4K @ 30fps : 0.2x temps réel
   - Latence première frame < 2s

Implémente monitoring temps réel avec dashboard.
```

## 7️⃣ Prompt Intégration Données Football

```
Intègre des données contextuelles football pour enrichir l'analyse :

1. DÉTECTION ZONES TERRAIN :
```python
class PitchAnalyzer:
    def detect_pitch_lines(self, frame: np.ndarray) -> PitchMarkings:
        # Lignes blanches (Hough transform)
        # Surface réparation
        # Ligne médiane
        # Corners et buts
        
    def create_pitch_map(self) -> HomographyMatrix:
        # Mapping 2D vers vue top-down
        # Calibration caméra
        # Coordonnées métriques réelles
```

2. CONTEXTE TACTIQUE :
   - Formation détectée (4-4-2, 4-3-3, etc.)
   - Lignes défensives/offensives
   - Bloc équipe compact/étiré
   - Pressing intensity zones

3. ÉVÉNEMENTS MATCH :
   - Détection phases de jeu
   - Transitions def/att
   - Situations arrêtées
   - Célébrations (à filtrer)

4. STATISTIQUES TEMPS RÉEL :
   - Possession par zone
   - Heatmaps joueurs
   - Distances parcourues
   - Vitesses instantanées

5. EXPORT FORMAT STANDARD :
   - Compatible Opta/StatsBomb
   - JSON structuré
   - Timestamps précis
   - Coordonnées normalisées

Crée parsers pour formats données pro existants.
```

## 8️⃣ Prompt Tests et Validation

```
Développe une suite de tests complète pour le module vidéo :

1. TESTS UNITAIRES :
```python
class TestVideoProcessor:
    def test_stabilization_quality(self):
        # Vidéo synthétique avec shake connu
        # Vérifie réduction mouvement > 80%
        
    def test_enhancement_metrics(self):
        # PSNR/SSIM avant/après
        # Sharpness quantifiée
        # Préservation couleurs
        
    def test_memory_usage(self):
        # Process vidéo 1h
        # RAM < 8GB constant
        # Pas de memory leaks
```

2. TESTS INTÉGRATION :
   - Pipeline bout en bout
   - Formats vidéo variés
   - Résolutions multiples
   - Corrupted files handling

3. TESTS PERFORMANCE :
   - Benchmark par résolution
   - Scaling multi-GPU
   - Latence par frame
   - Throughput maximal

4. VALIDATION FOOTBALL :
   - Dataset matchs pro annotés
   - Métriques vs ground truth
   - Cas edge (pluie, nuit, etc.)
   - Feedback experts

5. TESTS REGRESSION :
   - Suite vidéos référence
   - Métriques automatisées
   - Alertes dégradation
   - Version comparison

Génère rapport HTML avec visualisations.
```

## 🎬 Datasets et Ressources

```yaml
datasets:
  - name: "Football Player Detection Dataset"
    size: "50k images"
    source: "Custom + SoccerNet"
    
  - name: "Ball Tracking Dataset"
    size: "500 videos"
    annotations: "Frame-level ball position"
    
  - name: "Team Classification Set"
    size: "100 matches"
    teams: "Premier League, La Liga, Ligue 1"

models:
  - yolov10: "THU/yolov10x"
  - bytetrack: "ifzhang/ByteTrack"
  - team_classifier: "custom/team-color-clf-v2"
```

## 📝 Checklist Intégration

- [ ] Module vidéo stable et testé
- [ ] Détection >95% précision joueurs
- [ ] Tracking <5% ID switches
- [ ] Classification équipes >98% accuracy
- [ ] Performance temps réel 720p
- [ ] Documentation API complète
- [ ] Tests couverture >85% 