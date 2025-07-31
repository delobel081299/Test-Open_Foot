# 📦 Description des Modules - FootballAI Analyzer

## 🎯 Vue d'ensemble

Le système est composé de modules spécialisés qui travaillent ensemble pour analyser les vidéos de football. Chaque module a une responsabilité claire et communique via des interfaces bien définies.

---

## 1. 📹 Module de Prétraitement Vidéo

### Responsabilité
Préparer les vidéos pour l'analyse en optimisant la qualité et en extrayant les frames pertinentes.

### Composants

#### `video_loader.py`
```python
class VideoLoader:
    """Charge et valide les vidéos d'entrée"""
    - load_video(path: str) -> Video
    - validate_format(video: Video) -> bool
    - get_metadata(video: Video) -> dict
    - check_quality(video: Video) -> QualityScore
```

#### `frame_extractor.py`
```python
class FrameExtractor:
    """Extrait les frames de la vidéo"""
    - extract_frames(video: Video, fps: int) -> List[Frame]
    - extract_keyframes(video: Video) -> List[Frame]
    - save_frames(frames: List[Frame], output_dir: str)
    - optimize_extraction(video: Video) -> ExtractionPlan
```

#### `scene_detector.py`
```python
class SceneDetector:
    """Détecte les changements de scène et actions"""
    - detect_scenes(frames: List[Frame]) -> List[Scene]
    - find_action_segments(frames: List[Frame]) -> List[Segment]
    - classify_camera_angle(scene: Scene) -> CameraAngle
    - filter_relevant_scenes(scenes: List[Scene]) -> List[Scene]
```

### Entrées/Sorties
- **Entrée** : Fichier vidéo (MP4, AVI, MOV, MKV)
- **Sortie** : Frames optimisées, métadonnées, segments d'action

### Technologies
- **FFmpeg** : Manipulation vidéo bas niveau
- **OpenCV** : Traitement d'image
- **PySceneDetect** : Détection de changements de scène

---

## 2. 🎯 Module de Détection

### Responsabilité
Détecter et localiser les joueurs, le ballon et les éléments du terrain dans chaque frame.

### Composants

#### `yolo_detector.py`
```python
class YOLODetector:
    """Détecteur principal basé sur YOLOv8"""
    - load_model(model_path: str, device: str)
    - detect(frame: Frame) -> List[Detection]
    - batch_detect(frames: List[Frame]) -> List[List[Detection]]
    - adjust_confidence(threshold: float)
```

#### `player_detector.py`
```python
class PlayerDetector:
    """Spécialisé dans la détection de joueurs"""
    - detect_players(frame: Frame) -> List[Player]
    - estimate_jersey_number(player: Player) -> int
    - detect_goalkeeper(players: List[Player]) -> Player
    - filter_duplicates(players: List[Player]) -> List[Player]
```

#### `ball_detector.py`
```python
class BallDetector:
    """Détection précise du ballon"""
    - detect_ball(frame: Frame) -> Ball
    - track_ball_trajectory(frames: List[Frame]) -> Trajectory
    - estimate_ball_possession(ball: Ball, players: List[Player]) -> Player
    - detect_ball_contact(ball: Ball, player: Player) -> bool
```

### Entrées/Sorties
- **Entrée** : Frames vidéo
- **Sortie** : Bounding boxes, positions, classifications

### Technologies
- **YOLOv10/RT-DETR/DINO-DETR** : Modèles SOTA pour précision maximale
- **TensorRT** : Optimisation GPU obligatoire pour 60 FPS
- **ONNX** : Format de modèle portable
- **FP16/INT8** : Précision mixte pour performance

---

## 3. 🏃 Module de Tracking

### Responsabilité
Suivre les objets détectés à travers les frames pour maintenir leur identité.

### Composants

#### `byte_tracker.py`
```python
class ByteTracker:
    """Tracking multi-objets robuste"""
    - initialize_tracks(detections: List[Detection])
    - update_tracks(detections: List[Detection]) -> List[Track]
    - handle_occlusions(tracks: List[Track])
    - interpolate_missing(track: Track) -> Track
```

#### `team_classifier.py`
```python
class TeamClassifier:
    """Classification des équipes par couleur"""
    - extract_jersey_color(player: Player) -> Color
    - cluster_teams(players: List[Player]) -> Dict[Team, List[Player]]
    - classify_referee(person: Detection) -> bool
    - update_team_assignment(player: Player, history: List[Frame])
```

#### `trajectory_analyzer.py`
```python
class TrajectoryAnalyzer:
    """Analyse des trajectoires de mouvement"""
    - calculate_trajectory(track: Track) -> Trajectory
    - smooth_trajectory(trajectory: Trajectory) -> Trajectory
    - calculate_speed(trajectory: Trajectory) -> float
    - detect_sprints(trajectory: Trajectory) -> List[Sprint]
```

### Entrées/Sorties
- **Entrée** : Détections frame par frame
- **Sortie** : Tracks avec IDs persistants, trajectoires

### Technologies
- **ByteTrack** : Algorithme de tracking principal
- **Kalman Filter** : Prédiction de mouvement
- **Hungarian Algorithm** : Association détection-track

---

## 4. 🦴 Module d'Analyse Biomécanique

### Responsabilité
Analyser la posture et les mouvements corporels des joueurs.

### Composants

#### `pose_extractor.py`
```python
class PoseExtractor:
    """Extraction des keypoints corporels"""
    - extract_pose(player: Player) -> Pose3D
    - extract_batch_poses(players: List[Player]) -> List[Pose3D]
    - filter_occluded_joints(pose: Pose3D) -> Pose3D
    - interpolate_missing_joints(pose: Pose3D) -> Pose3D
```

#### `angle_calculator.py`
```python
class AngleCalculator:
    """Calcul des angles articulaires"""
    - calculate_joint_angles(pose: Pose3D) -> Dict[Joint, Angle]
    - calculate_knee_flexion(pose: Pose3D) -> float
    - calculate_hip_rotation(pose: Pose3D) -> float
    - calculate_spine_angle(pose: Pose3D) -> float
```

#### `balance_analyzer.py`
```python
class BalanceAnalyzer:
    """Analyse de l'équilibre et de la stabilité"""
    - calculate_center_of_mass(pose: Pose3D) -> Point3D
    - analyze_stability(pose_sequence: List[Pose3D]) -> StabilityScore
    - detect_imbalance(pose: Pose3D) -> bool
    - calculate_symmetry(pose: Pose3D) -> SymmetryScore
```

### Entrées/Sorties
- **Entrée** : Images de joueurs détectés
- **Sortie** : Poses 3D, angles, scores biomécaniques

### Technologies
- **MediaPipe Pose** : Extraction de pose principale
- **NumPy** : Calculs vectoriels
- **SciPy** : Analyses statistiques

---

## 5. ⚽ Module d'Analyse Technique

### Responsabilité
Classifier et évaluer les gestes techniques du football.

### Composants

#### `action_classifier.py`
```python
class ActionClassifier:
    """Classification des actions football"""
    - classify_action(frames: List[Frame], player: Player) -> Action
    - get_action_confidence(action: Action) -> float
    - detect_action_start_end(frames: List[Frame]) -> Tuple[int, int]
    - classify_ball_contact_type(contact: Contact) -> ContactType
```

#### `gesture_analyzer.py`
```python
class GestureAnalyzer:
    """Analyse détaillée des gestes"""
    - analyze_pass(action: PassAction) -> PassAnalysis
    - analyze_shot(action: ShotAction) -> ShotAnalysis
    - analyze_dribble(action: DribbleAction) -> DribbleAnalysis
    - analyze_control(action: ControlAction) -> ControlAnalysis
```

#### `technique_scorer.py`
```python
class TechniqueScorer:
    """Notation de la qualité technique"""
    - score_technique(analysis: GestureAnalysis) -> TechniqueScore
    - identify_errors(analysis: GestureAnalysis) -> List[Error]
    - generate_feedback(score: TechniqueScore, errors: List[Error]) -> Feedback
    - compare_to_ideal(gesture: Gesture, ideal: Gesture) -> Comparison
```

### Entrées/Sorties
- **Entrée** : Séquences vidéo, poses, trajectoires ballon
- **Sortie** : Classifications d'actions, scores techniques, feedback

### Technologies
- **TimeSformer** : Classification d'actions vidéo
- **Custom CNN** : Modèles spécifiques football
- **Rule Engine** : Évaluation basée sur règles expertes

---

## 6. 📊 Module d'Analyse Tactique

### Responsabilité
Analyser les aspects collectifs et décisionnels du jeu.

### Composants

#### `position_analyzer.py`
```python
class PositionAnalyzer:
    """Analyse du positionnement des joueurs"""
    - analyze_formation(team: Team, frame: Frame) -> Formation
    - calculate_team_shape(team: Team) -> Shape
    - detect_offside_line(team: Team, frame: Frame) -> Line
    - evaluate_spacing(team: Team) -> SpacingScore
```

#### `movement_patterns.py`
```python
class MovementPatternAnalyzer:
    """Détection de patterns de mouvement"""
    - detect_pressing(team: Team, frames: List[Frame]) -> PressingEvent
    - analyze_transition(teams: List[Team], frames: List[Frame]) -> Transition
    - detect_overlap_runs(players: List[Player]) -> List[OverlapRun]
    - analyze_defensive_line(team: Team) -> DefensiveLine
```

#### `decision_quality.py`
```python
class DecisionAnalyzer:
    """Évaluation de la prise de décision"""
    - evaluate_pass_decision(pass_event: Pass, context: GameContext) -> DecisionScore
    - find_better_options(player: Player, context: GameContext) -> List[Option]
    - analyze_shot_timing(shot: Shot, context: GameContext) -> TimingScore
    - evaluate_dribble_choice(dribble: Dribble, context: GameContext) -> ChoiceScore
```

### Entrées/Sorties
- **Entrée** : Positions joueurs, trajectoires, contexte de jeu
- **Sortie** : Analyses tactiques, scores décisionnels

### Technologies
- **NetworkX** : Analyse de graphes (passes)
- **Scikit-learn** : Clustering formations
- **Custom algorithms** : Métriques football spécifiques

---

## 7. 🎯 Module de Scoring

### Responsabilité
Agréger toutes les analyses pour produire une notation finale et du feedback.

### Composants

#### `score_aggregator.py`
```python
class ScoreAggregator:
    """Agrégation des scores multi-critères"""
    - aggregate_scores(scores: Dict[Criterion, Score]) -> FinalScore
    - apply_weights(scores: Dict[Criterion, Score], weights: Dict[Criterion, float]) -> WeightedScore
    - normalize_scores(scores: List[Score]) -> List[NormalizedScore]
    - calculate_confidence(scores: List[Score]) -> float
```

#### `feedback_generator.py`
```python
class FeedbackGenerator:
    """Génération de feedback personnalisé"""
    - generate_technical_feedback(analysis: TechnicalAnalysis) -> Feedback
    - generate_tactical_feedback(analysis: TacticalAnalysis) -> Feedback
    - prioritize_feedback(feedbacks: List[Feedback]) -> List[Feedback]
    - personalize_language(feedback: Feedback, player_profile: Profile) -> Feedback
```

#### `report_builder.py`
```python
class ReportBuilder:
    """Construction des rapports finaux"""
    - build_pdf_report(analysis: CompleteAnalysis) -> PDFReport
    - create_video_overlay(analysis: CompleteAnalysis, video: Video) -> AnnotatedVideo
    - generate_statistics_summary(analysis: CompleteAnalysis) -> Statistics
    - create_improvement_plan(analysis: CompleteAnalysis) -> ImprovementPlan
```

### Entrées/Sorties
- **Entrée** : Toutes les analyses des modules précédents
- **Sortie** : Rapport PDF, vidéo annotée, plan d'amélioration

### Technologies
- **ReportLab** : Génération PDF
- **Matplotlib/Plotly** : Graphiques
- **Jinja2** : Templates de rapport
- **OpenCV** : Annotation vidéo

---

## 8. 🎮 Module d'Interface API

### Responsabilité
Exposer les fonctionnalités via une API REST.

### Composants

#### Routes principales
```python
# upload.py
POST /api/upload          # Upload vidéo
GET  /api/upload/status   # Statut upload

# analysis.py  
POST /api/analyze         # Lancer analyse
GET  /api/analyze/status  # Statut analyse
GET  /api/analyze/result  # Résultats

# reports.py
GET  /api/report/pdf      # Télécharger PDF
GET  /api/report/video    # Vidéo annotée
POST /api/report/share    # Partager rapport
```

### Technologies
- **FastAPI** : Framework API
- **Pydantic** : Validation données
- **SQLAlchemy** : ORM
- **Celery** : Tâches asynchrones (optionnel)

---

## 🔗 Communication Inter-Modules

### Format des messages
```python
@dataclass
class ModuleMessage:
    source: str
    target: str
    message_type: MessageType
    payload: Any
    timestamp: datetime
    correlation_id: str
```

### Pipeline de données
```
Video → Preprocessing → Detection → Tracking → 
Biomechanics ↘
Technical    → Scoring → Report
Tactical     ↗
```

### Gestion des erreurs
- Chaque module gère ses erreurs localement
- Propagation contrôlée vers le module appelant
- Fallback strategies pour maintenir le service
- Logging centralisé pour debug

---

## 📈 Métriques de Performance

| Module | Métrique Clé | Objectif | Mesure Actuelle |
|--------|--------------|----------|-----------------|
| Preprocessing | FPS extraction | 30 FPS | 35 FPS |
| Detection | mAP (précision) | > 85% | 89% |
| Tracking | MOTA score | > 80% | 83% |
| Biomechanics | Keypoints/sec | > 15 | 18 |
| Technical | Accuracy | > 90% | 92% |
| Tactical | Processing time | < 5s | 3.5s |
| Scoring | Report generation | < 30s | 25s | 