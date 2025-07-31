# 📊 Stratégie Datasets - FootballAI Analyzer

## 🎯 Vue d'ensemble

Pour entraîner et valider nos modèles d'analyse football, nous avons besoin de datasets diversifiés et de haute qualité. Ce document détaille notre stratégie d'acquisition, création et gestion des données.

---

## 📦 Datasets Nécessaires

### 1. Détection d'Objets (YOLOv8)

#### Besoins
- **Images** : 10,000+ frames annotés
- **Classes** : joueur, ballon, but, arbitre, ligne terrain
- **Annotations** : Bounding boxes précises
- **Diversité** : Différents stades, conditions météo, angles caméra

#### Sources Publiques
```yaml
SoccerNet:
  url: "https://www.soccer-net.org/"
  contenu: "500+ matchs complets annotés"
  licence: "Recherche uniquement"
  
FootAndBall:
  url: "https://github.com/FootAndBall/dataset"
  contenu: "Détection joueurs/ballon"
  images: 5000+
  
ISSIA-CNR:
  url: "http://www.issia.cnr.it/wp/"
  contenu: "Tracking joueurs"
  format: "Annotations XML"
```

#### Création Dataset Custom
```python
# Stratégie d'annotation semi-automatique
1. Pré-annotation avec modèle existant
2. Correction manuelle via LabelImg
3. Augmentation données (flip, rotate, brightness)
4. Validation croisée 3 annotateurs
```

### 2. Reconnaissance d'Actions

#### Besoins
- **Vidéos** : 1000+ clips par action (2-3 secondes)
- **Actions** : passe, tir, contrôle, dribble, tacle, tête, etc.
- **Angles** : Multiple vues par action
- **Qualité** : Labels précis début/fin action

#### Dataset Structure
```
action_dataset/
├── pass/
│   ├── short_pass/
│   │   ├── video_001.mp4
│   │   ├── video_001.json  # metadata
│   │   └── ...
│   ├── long_pass/
│   └── through_pass/
├── shot/
│   ├── penalty/
│   ├── free_kick/
│   └── open_play/
├── control/
│   ├── chest_control/
│   ├── foot_control/
│   └── head_control/
└── metadata.json
```

#### Annotation Format
```json
{
  "video_id": "pass_001",
  "action": "short_pass",
  "start_frame": 30,
  "end_frame": 75,
  "quality_score": 8,
  "player_id": "player_5",
  "success": true,
  "tags": ["inside_foot", "ground_pass", "accurate"],
  "biomechanics_score": 7.5
}
```

### 3. Analyse Biomécanique

#### Besoins
- **Poses** : 5000+ exemples par geste technique
- **Annotations** : Angles articulaires de référence
- **Experts** : Validation par entraîneurs certifiés
- **Variations** : Différentes morphologies joueurs

#### Métriques à Annoter
```python
biomechanics_annotations = {
    "pose_id": "shot_001_frame_45",
    "keypoints_3d": [...],  # 33 points MediaPipe
    "joint_angles": {
        "knee_flexion": 145,
        "hip_rotation": 30,
        "ankle_extension": 120,
        "trunk_lean": 15,
        "shoulder_abduction": 45
    },
    "balance_score": 8.5,
    "technique_errors": ["late_plant_foot", "limited_follow_through"],
    "expert_validated": true
}
```

### 4. Analyse Tactique

#### Besoins
- **Formations** : 500+ exemples par système (4-4-2, 4-3-3, etc.)
- **Transitions** : 1000+ séquences attaque/défense
- **Mouvements** : Patterns pressing, contre-attaque, build-up
- **Contexte** : Score, temps, phase de jeu

#### Données Tactiques
```yaml
Formation Detection:
  - Positions moyennes joueurs (heatmaps)
  - Distances inter-lignes
  - Largeur/profondeur équipe
  - Adaptations dynamiques

Decision Making:
  - Contexte spatial complet
  - Options disponibles
  - Décision prise
  - Résultat action
  - Alternatives optimales
```

---

## 🔄 Pipeline de Création de Dataset

### Phase 1 : Collection Initiale

```python
# 1. Scraping matchs YouTube/publics
def collect_videos():
    sources = [
        "Matchs amateurs filmés",
        "Extraits entraînements",
        "Vidéos exercices techniques"
    ]
    
    for source in sources:
        # Download avec youtube-dl
        # Vérifier droits/permissions
        # Convertir format uniforme
        
# 2. Découpage en clips
def extract_clips(video, annotations):
    # Détection automatique actions
    # Découpage fenêtres temporelles
    # Sauvegarde clips + metadata
```

### Phase 2 : Annotation Semi-Automatique

```python
# Outil annotation custom
class FootballAnnotator:
    def __init__(self):
        self.detector = YOLOv8()
        self.action_classifier = PretrainedModel()
        self.pose_extractor = MediaPipe()
    
    def pre_annotate(self, video):
        # 1. Détection objets automatique
        detections = self.detector.detect(video)
        
        # 2. Classification actions probable
        actions = self.action_classifier.predict(video)
        
        # 3. Extraction poses clés
        poses = self.pose_extractor.extract(video)
        
        return {
            'detections': detections,
            'actions': actions,
            'poses': poses,
            'confidence': 0.7
        }
    
    def human_review(self, pre_annotations):
        # Interface web pour validation
        # Correction erreurs
        # Ajout labels manquants
        # Scoring qualité
```

### Phase 3 : Augmentation Données

```python
class DataAugmentation:
    def __init__(self):
        self.transforms = [
            RandomHorizontalFlip(p=0.5),
            RandomBrightness(limit=0.2),
            RandomContrast(limit=0.2),
            RandomRotate(limit=5),
            RandomCrop(size=(0.8, 1.0)),
            TimeShift(max_shift=5),
            SpeedChange(factor=(0.8, 1.2))
        ]
    
    def augment_video(self, video, annotations):
        augmented = []
        
        for transform in self.transforms:
            aug_video = transform(video)
            aug_annot = transform.adjust_annotations(annotations)
            
            augmented.append({
                'video': aug_video,
                'annotations': aug_annot
            })
        
        return augmented
```

### Phase 4 : Validation Qualité

```python
def validate_dataset(dataset):
    metrics = {
        'completeness': check_all_classes_covered(),
        'balance': check_class_distribution(),
        'quality': check_annotation_accuracy(),
        'diversity': check_visual_diversity()
    }
    
    # Critères minimum
    assert metrics['completeness'] > 0.95
    assert metrics['balance'] > 0.7
    assert metrics['quality'] > 0.9
    assert metrics['diversity'] > 0.8
    
    return metrics
```

---

## 🏭 Datasets Synthétiques

### Génération avec Unity/Unreal

```csharp
// Simulateur Football Unity
public class SyntheticDataGenerator : MonoBehaviour {
    
    public void GenerateTrainingData() {
        // 1. Spawn joueurs positions aléatoires
        SpawnPlayers(formation: "4-3-3");
        
        // 2. Simuler actions
        SimulatePass(player1, player2);
        SimulateShot(player3, goalPosition);
        
        // 3. Capturer depuis multiples caméras
        CaptureFromCameras(cameras);
        
        // 4. Export annotations parfaites
        ExportAnnotations(format: "COCO");
    }
    
    private void RandomizeEnvironment() {
        // Varier conditions
        weather.Randomize();
        lighting.RandomizeTimeOfDay();
        stadium.RandomizeType();
        crowd.RandomizeDensity();
    }
}
```

### Avantages Données Synthétiques
- Annotations 100% précises
- Variations infinies
- Cas rares générables
- Pas de problèmes droits

---

## 📈 Stratégie Progressive

### Mois 1-2 : Bootstrap
```yaml
Objectif: Dataset minimal fonctionnel
Actions:
  - Utiliser datasets publics existants
  - Annoter 100 vidéos manuellement
  - Fine-tuner modèles pré-entraînés
  
Résultat attendu:
  - 5,000 images annotées
  - 500 clips actions
  - Précision 75%+
```

### Mois 3-4 : Expansion
```yaml
Objectif: Dataset qualité production
Actions:
  - Crowdsourcing annotations
  - Partenariat clubs locaux
  - Génération données synthétiques
  
Résultat attendu:
  - 20,000 images annotées
  - 2,000 clips actions
  - Précision 85%+
```

### Mois 5-6 : Spécialisation
```yaml
Objectif: Dataset expert niveau pro
Actions:
  - Collaboration entraîneurs pros
  - Capture matchs haute qualité
  - Annotations biomécaniques fines
  
Résultat attendu:
  - 50,000+ images
  - 5,000+ clips
  - Précision 90%+
```

---

## 🛠️ Outils d'Annotation

### 1. LabelImg (Détection)
```bash
# Installation
pip install labelImg

# Utilisation
labelImg ./images ./annotations
```

### 2. CVAT (Vidéo + Tracking)
```yaml
Installation:
  - Docker recommandé
  - Support annotations vidéo
  - Collaboration multi-utilisateurs
  
Features:
  - Interpolation automatique
  - Tracking objets
  - Export COCO/YOLO
```

### 3. Outil Custom (Spécifique Football)
```python
# Interface annotation football
class FootballAnnotationTool:
    features = [
        "Détection automatique joueurs",
        "Suggestion actions via ML",
        "Validation poses biomécanique",
        "Scoring qualité technique",
        "Export format unifié"
    ]
    
    def annotate_match(self, video):
        # Pre-process avec IA
        # Interface review humain
        # Export annotations
        pass
```

---

## 💾 Stockage et Versioning

### Structure Stockage
```
datasets/
├── raw/              # Vidéos originales
├── processed/        # Vidéos normalisées
├── annotations/      # Labels tous formats
├── augmented/        # Données augmentées
├── synthetic/        # Données générées
├── splits/          # Train/val/test
└── versions/        # Historique datasets
```

### Versioning avec DVC
```bash
# Initialiser DVC
dvc init

# Tracker dataset
dvc add datasets/
git add datasets.dvc .gitignore
git commit -m "Add football dataset v1.0"

# Pusher vers stockage distant
dvc remote add -d storage s3://football-datasets
dvc push
```

### Metadata Tracking
```yaml
# datasets/metadata.yaml
version: "1.2.0"
created: "2024-01-15"
stats:
  total_videos: 1543
  total_frames: 487650
  total_actions: 8934
  
classes:
  detection:
    player: 45632
    ball: 12453
    goal: 3421
    
  actions:
    pass: 2341
    shot: 876
    dribble: 1234
    
quality_metrics:
  annotation_accuracy: 0.92
  inter_annotator_agreement: 0.87
  
changelog:
  - "Added 500 penalty kicks"
  - "Fixed mislabeled headers"
  - "Improved night match visibility"
```

---

## 🔐 Considérations Légales

### Droits et Permissions
```yaml
Public Datasets:
  - Vérifier licences (commercial/recherche)
  - Attribution correcte
  - Respect conditions utilisation

Données Propres:
  - Consentement filmés
  - Floutage visages mineurs
  - Stockage RGPD compliant
  
Données Clubs:
  - Contrats utilisation
  - Confidentialité tactique
  - Partage revenus éventuel
```

### Template Accord
```markdown
# Accord d'Utilisation Données Football

Entre : [Club/Joueur]
Et : FootballAI Analyzer

1. Utilisation limitée à l'analyse technique
2. Pas de diffusion publique sans accord
3. Anonymisation sur demande
4. Partage résultats analyse
5. Durée : 24 mois renouvelable
```

---

## 📊 Métriques Qualité Dataset

### KPIs Principaux
```python
def evaluate_dataset_quality(dataset):
    metrics = {
        # Couverture
        'class_balance': calculate_class_distribution(dataset),
        'action_diversity': count_unique_actions(dataset),
        'angle_coverage': analyze_camera_angles(dataset),
        
        # Qualité
        'annotation_precision': measure_bbox_accuracy(dataset),
        'label_consistency': check_label_coherence(dataset),
        'temporal_smoothness': verify_tracking_continuity(dataset),
        
        # Utilisabilité  
        'train_val_split': verify_split_distribution(dataset),
        'metadata_completeness': check_all_fields_present(dataset),
        'format_compatibility': test_loader_compatibility(dataset)
    }
    
    return metrics
```

### Benchmarks Cibles
| Métrique | Minimum | Optimal |
|----------|---------|---------|
| Images annotées | 10,000 | 50,000+ |
| Clips par action | 500 | 2,000+ |
| Précision annotation | 85% | 95%+ |
| Diversité angles | 3 | 5+ |
| Balance classes | 0.7 | 0.9+ |

---

## 🚀 Roadmap Données

### Q1 2024
- ✅ 10K images basiques annotées
- ✅ 500 clips actions principales
- ✅ Partenariat 2 clubs locaux

### Q2 2024
- 📋 25K images multi-angles
- 📋 2K clips haute qualité
- 📋 Dataset synthétique Unity

### Q3 2024
- 📋 50K+ images production
- 📋 5K clips avec biomécanique
- 📋 Benchmarks publics publiés

### Q4 2024
- 📋 Dataset propriétaire complet
- 📋 API accès chercheurs
- 📋 Compétition Kaggle organisée 