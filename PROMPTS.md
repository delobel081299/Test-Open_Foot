# 🤖 Prompts IA - FootballAI Analyzer

## 📋 Introduction

Ce document contient des prompts détaillés et optimisés pour utiliser Claude, ChatGPT ou Cursor dans le développement de chaque module. Chaque prompt est conçu pour produire du code production-ready.

---

## 🏗️ Prompts de Setup Initial

### Prompt 1 : Création de la structure du projet

```
Je développe une application d'analyse vidéo football 100% locale appelée FootballAI Analyzer. 

Crée la structure complète du projet Python avec :
1. Architecture backend modulaire (FastAPI)
2. Frontend React moderne
3. Configuration pour GPU (CUDA/PyTorch)
4. Scripts d'installation automatique
5. Gestion des dépendances avec versions fixes
6. Structure de dossiers claire et scalable
7. Configuration YAML pour tous les paramètres
8. Support Windows/Mac/Linux

Le projet doit inclure :
- backend/ (API, core modules, database)
- frontend/ (React app)
- models/ (IA models storage)
- scripts/ (installation, run, utils)
- config/ (YAML configs)
- tests/ (unit & integration)
- docs/ (documentation)

Génère tous les fichiers de base avec leur contenu initial.
```

### Prompt 2 : Configuration de l'environnement

```
Crée un script d'installation Python complet (scripts/install.py) qui :

1. Détecte l'OS (Windows/Mac/Linux)
2. Vérifie Python 3.10+ installé
3. Crée et active un environnement virtuel
4. Installe les dépendances depuis requirements.txt
5. Détecte si GPU NVIDIA disponible
6. Configure CUDA/cuDNN si GPU présent
7. Télécharge les modèles IA nécessaires
8. Initialise la base de données SQLite
9. Crée les dossiers de travail
10. Vérifie FFmpeg installé
11. Configure les variables d'environnement
12. Affiche un rapport de succès/erreurs

Le script doit être robuste avec gestion d'erreurs et logs détaillés.
```

---

## 📹 Prompts Module Prétraitement

### Prompt 3 : Video Loader

```
Implémente un module complet de chargement vidéo (backend/core/preprocessing/video_loader.py) avec :

Classes principales :
- VideoLoader : Charge et valide les vidéos
- VideoMetadata : Stocke les métadonnées
- ValidationResult : Résultat de validation

Fonctionnalités :
1. Support formats : MP4, AVI, MOV, MKV
2. Validation taille max (2GB par défaut)
3. Vérification corruption avec FFmpeg
4. Extraction métadonnées (durée, FPS, résolution, codec)
5. Détection orientation (portrait/paysage)
6. Estimation qualité vidéo
7. Gestion mémoire pour grandes vidéos
8. Support chemins Unicode

Utilise OpenCV et FFmpeg-python. Inclus gestion d'erreurs complète et logging.
```

### Prompt 4 : Frame Extractor

```
Crée un extracteur de frames optimisé (backend/core/preprocessing/frame_extractor.py) :

Fonctionnalités :
1. Extraction parallèle multi-thread
2. 3 modes : all_frames, keyframes, interval
3. Détection automatique frames floues
4. Normalisation FPS (interpolation si nécessaire)
5. Redimensionnement intelligent préservant ratio
6. Cache intelligent pour ré-utilisation
7. Extraction par batch pour mémoire
8. Sauvegarde optionnelle sur disque

Optimisations :
- Utilise ThreadPoolExecutor pour parallélisme
- Batch processing par chunks de 100 frames
- Libération mémoire automatique
- Progress bar avec tqdm

Code production-ready avec types, docstrings, et tests unitaires.
```

---

## 🎯 Prompts Module Détection

### Prompt 5 : YOLO Detector

```
Implémente un détecteur haute précision (backend/core/detection/advanced_detector.py) pour football avec support YOLOv10/RT-DETR/DINO-DETR :

Architecture :
- Classe AdvancedDetector avec support multi-modèles
- GPU obligatoire pour 60 FPS
- Batch inference optimisée TensorRT
- Post-processing haute précision

Modèles supportés :
1. YOLOv10 : NMS-free, plus rapide
2. RT-DETR : Real-time DETR, meilleure précision
3. DINO-DETR : State-of-the-art accuracy

Fonctionnalités spécifiques football :
1. Classes : player, ball, goal, referee, coach
2. Post-processing avancé sans NMS (DETR)
3. Confidence élevée (>0.7) pour précision max
4. ROI focusing sur terrain de jeu
5. Gestion occlusions avec attention mechanism
6. Tracking-aware detection (cohérence temporelle)
7. 60 FPS target avec précision maximale

Configuration YAML :
- Modèle : rtdetr-x ou yolov10x ou dino-detr
- Thresholds élevés pour précision
- Batch size optimisé VRAM
- FP16 precision pour speed

Inclus benchmarks performance et métriques mAP.
```

### Prompt 6 : Ball Detector Spécialisé

```
Crée un détecteur de ballon haute précision (backend/core/detection/ball_detector.py) :

Techniques combinées :
1. YOLO pour détection initiale
2. Hough Circle Transform pour validation
3. Optical Flow pour prédiction trajectoire
4. Template matching pour cas difficiles
5. Kalman Filter pour smooth tracking

Gestion cas spéciaux :
- Ballon partiellement visible
- Ballon en l'air (taille variable)
- Occlusion par joueurs
- Confusion avec têtes/objets ronds
- Ballon hors champ (prédiction)

Features avancées :
- Détection possession (joueur le plus proche)
- Estimation vitesse/direction
- Prédiction trajectoire parabolique
- Détection contacts pied/tête

Retourne : position, vitesse, possession, confiance
```

---

## 🏃 Prompts Module Tracking

### Prompt 7 : ByteTrack Implementation

```
Implémente ByteTrack pour tracking football (backend/core/tracking/byte_tracker.py) :

Architecture complète :
1. Classe ByteTracker avec configuration
2. Gestion multi-classes (joueurs, arbitres)
3. Association par IoU + features visuelles
4. Prédiction Kalman pour occlusions

Adaptations football :
- Gestion 22+ joueurs simultanés
- Tracks persistants même hors champ
- Ré-identification après occlusion longue
- Association équipe par couleur maillot
- Détection substitutions

Optimisations :
- Hungarian algorithm optimisé
- Cache features visuelles
- Batch processing GPU
- Mémoire tampons limitée

Métriques :
- MOTA, MOTP, IDF1
- Précision par équipe
- Taux de switch ID

Inclus visualisation debug et export formats standard.
```

### Prompt 8 : Team Classifier

```
Développe un classificateur d'équipes robuste (backend/core/tracking/team_classifier.py) :

Pipeline complet :
1. Extraction couleur dominante maillot
2. Clustering K-means (K=3 : équipe1, équipe2, arbitre)
3. Validation temporelle (cohérence sur N frames)
4. Mise à jour adaptative des clusters

Techniques :
- Masquage zone maillot via pose
- Histogramme couleur HSV
- Distance colorimétrique CIE Lab
- Voting majoritaire sur séquence
- Gestion maillots bicolores

Cas spéciaux :
- Gardiens couleur différente
- Arbitres détection automatique
- Changement maillot mi-temps
- Conditions lumière variables

Sortie : {player_id: team_id} avec confiance
Performance : <5ms par joueur
```

---

## 🦴 Prompts Module Biomécanique

### Prompt 9 : Pose Extraction avec MediaPipe

```
Crée un extracteur de pose 3D optimisé (backend/core/biomechanics/pose_extractor.py) :

Configuration MediaPipe :
- Model complexity: 2 (heavy)
- Enable segmentation
- Smooth landmarks
- 3D world coordinates

Pipeline complet :
1. Crop intelligent autour du joueur
2. Padding pour éviter coupures
3. Extraction 33 keypoints 3D
4. Filtrage points occludés
5. Interpolation points manquants
6. Lissage temporel (Savitzky-Golay)
7. Normalisation par taille joueur

Optimisations batch :
- Process 8 joueurs en parallèle
- Réutilisation modèle chargé
- GPU acceleration si disponible

Données extraites :
- 33 landmarks 3D + visibility
- Angles articulaires calculés
- Centre de masse
- Orientation corps

Gestion erreurs et fallback 2D si échec 3D.
```

### Prompt 10 : Analyse Biomécanique Avancée

```
Implémente analyseur biomécanique complet (backend/core/biomechanics/movement_analyzer.py) :

Métriques calculées :
1. Angles articulaires (15 articulations)
2. Symétrie gauche/droite
3. Stabilité via centre de masse
4. Fluidité mouvement (jerk)
5. Amplitude mouvement
6. Coordination inter-segments

Analyse spécifique football :
- Angle flexion genou (tir)
- Rotation hanches (passe)
- Inclinaison tronc (équilibre)
- Extension cheville (frappe)
- Position bras (équilibre)

Détection problèmes :
- Déséquilibres posturaux
- Asymétries dangereuses
- Mouvements à risque
- Fatigue via dégradation

Scoring 0-100 avec feedback :
- Points forts identifiés
- Corrections suggérées
- Exercices recommandés

Visualisation 3D skeleton optionnelle.
```

---

## ⚽ Prompts Module Technique

### Prompt 11 : Action Recognition

```
Développe un classificateur d'actions football (backend/core/technical/action_classifier.py) :

Architecture ML :
- TimeSformer ou VideoMAE backbone
- Fine-tuning sur actions football
- 15 classes : pass, shot, dribble, control, tackle, etc.

Pipeline :
1. Extraction fenêtre temporelle (2 sec)
2. Preprocessing : resize, normalize
3. Inference model transformer
4. Post-processing confidences
5. Smoothing temporel prédictions

Data augmentation :
- Random crop/flip
- Vitesse variable
- Ajout bruit
- Mixup entre classes

Optimisations :
- ONNX export pour vitesse
- Quantization INT8
- Batch processing
- Cache prédictions similaires

Métriques :
- Accuracy par classe >90%
- Confusion matrix
- Temps inference <50ms

Inclus script fine-tuning sur dataset custom.
```

### Prompt 12 : Évaluation Technique des Gestes

```
Crée un évaluateur technique expert (backend/core/technical/technique_scorer.py) :

Analyse par type de geste :

PASSE :
- Timing contact (anticipation)
- Surface utilisée (intérieur/extérieur)
- Direction regard avant passe
- Follow-through du geste
- Précision direction/force

TIR :
- Position pied d'appui
- Angle approche ballon
- Point d'impact ballon
- Extension jambe frappe
- Équilibre après tir

CONTRÔLE :
- Amorti première touche
- Orientation après contrôle
- Distance ballon-corps
- Rapidité enchaînement

DRIBBLE :
- Fréquence touches
- Changements direction
- Protection ballon
- Feintes corps

Scoring basé sur :
- Règles expertes pondérées
- Comparaison pros référence
- Cohérence biomécanique

Output : score, points amélioration, gif comparatif
```

---

## 📊 Prompts Module Tactique

### Prompt 13 : Analyse Formation et Positionnement

```
Implémente analyseur tactique formation (backend/core/tactical/formation_analyzer.py) :

Détection formation :
1. Clustering positions moyennes joueurs
2. Classification parmi : 442, 433, 352, 4231, etc.
3. Détection variations dynamiques
4. Calcul compacité horizontale/verticale

Métriques tactiques :
- Distance inter-lignes
- Largeur occupation
- Profondeur équipe  
- Surface convexe occupée
- Centre gravité équipe

Analyse avancée :
- Détection bloc haut/bas
- Asymétries formation
- Joueurs hors position
- Coordination pressing
- Transitions def/att

Visualisations :
- Heatmaps positions
- Lignes formation moyenne
- Animations transitions
- Réseaux de passes

Comparaisons :
- Vs formation théorique
- Vs adversaire
- Evolution temporelle

Export données pour tableau tactique.
```

### Prompt 14 : Analyse Décisionnelle

```
Développe analyseur décisions tactiques (backend/core/tactical/decision_analyzer.py) :

Contexte décisionnel :
1. Position tous joueurs
2. Espaces libres
3. Pression adversaire
4. Options disponibles
5. Risque/récompense

Évaluation décisions :

PASSE :
- Joueurs démarqués disponibles
- Lignes de passe ouvertes
- Progressivité option
- Danger créé
- Alternative conservatrice

DRIBBLE :
- Espace disponible
- Nombre adversaires
- Support coéquipiers
- Zone terrain
- Alternatives passe

TIR :
- Angle/distance but
- Pression défensive
- Position gardien
- Coéquipiers mieux placés

Scoring xDecision :
- Modèle ML entraîné sur pros
- Pondération contexte match
- Facteur risque personnalisé

Output : score décision, meilleures alternatives, visualisation options
```

---

## 🎯 Prompts Module Scoring

### Prompt 15 : Agrégateur de Scores

```
Crée système scoring unifié (backend/core/scoring/score_aggregator.py) :

Architecture scoring :
1. Collecte scores tous modules
2. Normalisation échelle 0-100
3. Pondération contextuelle
4. Intervalles confiance

Pondérations adaptatives :
```python
if video_type == "training":
    weights = {
        "biomechanics": 0.35,
        "technical": 0.45,
        "tactical": 0.10,
        "physical": 0.10
    }
elif video_type == "match":
    weights = {
        "biomechanics": 0.15,
        "technical": 0.30,
        "tactical": 0.35,
        "physical": 0.20
    }
```

Features avancées :
- Détection points forts/faibles
- Comparaison percentiles
- Évolution temporelle
- Clustering profils joueurs

Personnalisation :
- Par âge joueur
- Par poste
- Par niveau
- Par objectifs

API claire pour modules :
- register_score()
- get_final_score()
- get_breakdown()
- export_report()
```

### Prompt 16 : Générateur de Feedback

```
Implémente générateur feedback intelligent (backend/core/scoring/feedback_generator.py) :

Pipeline génération :
1. Analyse scores détaillés
2. Identification priorités
3. Formulation constructive
4. Personnalisation ton
5. Ajout exemples visuels

Templates feedback par catégorie :

TECHNIQUE :
- "Excellente qualité de passe, maintenir l'angle d'ouverture du pied"
- "Attention au timing de la frappe, anticiper 0.2s plus tôt"
- "Premier contrôle perfectible, orienter vers l'espace libre"

TACTIQUE :
- "Bon timing des appels, continuer à étirer la défense"
- "Positionnement défensif à améliorer, réduire distance avec #6"
- "Excellentes prises de décision sous pression"

PHYSIQUE :
- "Bonne intensité maintenue, pic à 28.5 km/h"
- "Améliorer fréquence des sprints courts"
- "Excellent volume de course : 8.2 km"

Utilise LLM local (Ollama) pour variations naturelles.
Limite 3-5 feedbacks prioritaires.
Inclus GIFs démonstratifs.
```

---

## 🎮 Prompts API & Frontend

### Prompt 17 : API FastAPI Complète

```
Développe API REST complète (backend/api/main.py) :

Endpoints principaux :

POST /api/upload
- Upload multipart vidéo
- Validation format/taille
- Génération UUID job
- Stockage sécurisé
- WebSocket notification progress

POST /api/analyze/{job_id}
- Lancement analyse async
- Configuration personnalisable
- Queue management
- Status tracking

GET /api/status/{job_id}
- SSE pour updates real-time
- Pourcentage progression
- Étape courante
- ETA restant

GET /api/results/{job_id}
- Résultats JSON complets
- Pagination si besoin
- Filtres par joueur/métrique

GET /api/report/{job_id}/pdf
- Génération PDF à la volée
- Streaming response
- Cache 24h

Sécurité :
- CORS configuré
- Rate limiting
- Validation inputs
- Gestion erreurs globale

Optimisations :
- Connection pooling DB
- Response caching
- Compression gzip
- Async partout

Docs auto Swagger UI.
```

### Prompt 18 : Interface React Moderne

```
Crée frontend React moderne (frontend/src) :

Stack technique :
- React 18 + TypeScript
- Vite pour build
- Tailwind CSS
- Framer Motion animations
- Recharts pour graphiques
- React Player vidéo

Composants principaux :

1. UploadZone
- Drag & drop
- Preview vidéo
- Validation client
- Progress bar chunked upload

2. AnalysisDashboard
- Tabs par catégorie
- Graphiques interactifs
- Filtres dynamiques
- Export PDF/CSV

3. VideoPlayer
- Annotations overlay
- Timeline événements
- Slow motion
- Frame par frame

4. PlayerCard
- Photo/avatar
- Radar chart perfs
- Stats clés
- Évolution

5. ReportViewer
- PDF embed
- Sections navigables
- Mode impression

État global :
- Zustand pour simplicité
- React Query pour API
- Persist localStorage

Responsive mobile-first.
Dark mode natif.
```

---

## 🧪 Prompts Tests & Documentation

### Prompt 19 : Suite de Tests Complète

```
Crée suite tests exhaustive (tests/) :

Structure tests :
tests/
├── unit/           # Tests unitaires par module
├── integration/    # Tests intégration pipeline
├── fixtures/       # Données test
├── conftest.py     # Config pytest
└── benchmarks/     # Tests performance

Tests unitaires exemple :
```python
# test_yolo_detector.py
def test_detect_players_in_frame():
    detector = YOLODetector()
    frame = load_test_frame("match_sample.jpg")
    detections = detector.detect(frame)
    
    assert len(detections.players) >= 10
    assert detections.ball is not None
    assert all(d.confidence > 0.5 for d in detections.players)

def test_batch_detection_performance():
    detector = YOLODetector()
    frames = load_test_video_frames("test_10s.mp4")
    
    start = time.time()
    results = detector.batch_detect(frames)
    duration = time.time() - start
    
    assert duration < 5.0  # Max 5s pour 10s vidéo
    assert len(results) == len(frames)
```

Coverage minimum 80%.
CI/CD avec GitHub Actions.
Tests performance GPU/CPU.
```

### Prompt 20 : Documentation Technique

```
Génère documentation complète (docs/) :

1. API Reference (auto depuis docstrings)
2. Guide architecture
3. Tutoriels pas-à-pas
4. Troubleshooting
5. Performances tuning

Format Markdown avec :
- Table des matières
- Exemples code
- Diagrammes mermaid
- Screenshots annotés
- Liens vidéos démo

Exemple section :

## Configuration Modèles IA

### YOLOv8 Optimization

Pour optimiser les performances de détection :

```yaml
# config/models.yaml
yolo:
  model_size: "yolov8x"  # ou yolov8l pour GPU < 8GB
  confidence: 0.45
  iou_threshold: 0.5
  max_detections: 30
  
  # Optimisations GPU
  fp16: true  # Half precision
  batch_size: 16  # Ajuster selon VRAM
  
  # Classes football
  classes:
    - person
    - sports ball
    - goal
```

Utilise Sphinx ou MkDocs pour site statique.
```

---

## 🚀 Prompts Avancés

### Prompt 21 : Optimisation GPU/Performance

```
Optimise l'application pour performance maximale :

1. Profiling GPU :
- Utilise NVIDIA Nsight
- Identifie bottlenecks
- Optimise mémoire VRAM

2. Optimisations PyTorch :
- torch.cuda.amp (mixed precision)
- torch.jit.script (compilation)
- cudnn.benchmark = True
- Batch size dynamique

3. Parallélisation :
- multiprocessing pour CPU tasks
- asyncio pour I/O
- ThreadPoolExecutor pour frames

4. Caching intelligent :
- Redis pour résultats
- LRU cache fonctions
- Memoization poses

5. Quantization modèles :
- INT8 pour inference
- ONNX export
- TensorRT si possible

Code exemple optimisation batch:
```python
@torch.cuda.amp.autocast()
def batch_inference_optimized(frames, model):
    # Dynamic batch size based on GPU memory
    batch_size = calculate_optimal_batch_size()
    
    results = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        batch_tensor = preprocess_batch_gpu(batch)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
        
        results.extend(postprocess_gpu(outputs))
        
        # Clear cache periodically
        if i % 100 == 0:
            torch.cuda.empty_cache()
    
    return results
```
```

### Prompt 22 : Déploiement Production

```
Prépare déploiement production complet :

1. Dockerisation :
- Dockerfile multi-stage
- docker-compose.yml
- Volumes pour data
- Networks isolation

2. Packaging :
- PyInstaller pour exe Windows
- .app bundle macOS  
- AppImage Linux
- Auto-updater intégré

3. Monitoring :
- Prometheus metrics
- Grafana dashboards
- Logging structuré
- Health checks

4. Sécurité :
- Sanitization inputs
- Sandboxing ffmpeg
- Permissions minimales
- No eval/exec

5. Installation user-friendly :
- Setup wizard GUI
- Auto-détection GPU
- Téléchargement modèles
- Validation système

Script build complet :
```bash
#!/bin/bash
# build.sh

# Version
VERSION=$(git describe --tags)

# Clean
rm -rf dist/

# Backend
cd backend
python -m PyInstaller \
    --onefile \
    --windowed \
    --icon=../assets/icon.ico \
    --add-data "models;models" \
    --hidden-import torch \
    main.py

# Frontend  
cd ../frontend
npm run build

# Package
cd ..
python scripts/package.py --version $VERSION
```
```

---

## 💡 Tips pour Utilisation des Prompts

### 1. Adaptation au Contexte
- Ajustez les paths selon votre structure
- Modifiez les configs selon vos besoins
- Adaptez le style de code à vos conventions

### 2. Utilisation Incrémentale
- Commencez par les prompts de base
- Testez chaque module avant le suivant
- Intégrez progressivement

### 3. Personnalisation
- Ajoutez vos propres métriques
- Modifiez les seuils selon votre niveau
- Étendez avec nouvelles features

### 4. Debugging avec l'IA
- Copiez les erreurs dans le prompt
- Demandez des explications détaillées
- Itérez jusqu'à résolution

### 5. Optimisation Continue
- Profilez régulièrement
- Demandez des optimisations spécifiques
- Benchmarkez avant/après 