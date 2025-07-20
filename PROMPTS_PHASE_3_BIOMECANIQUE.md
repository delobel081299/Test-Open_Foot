# 🏃 PROMPTS VIBE CODING - PHASE 3 : ANALYSE BIOMÉCANIQUE

## 📅 Durée : 1 semaine

## 🎯 Objectifs
- Extraction précise des poses 3D
- Analyse biomécanique des gestes techniques
- Scoring et détection d'erreurs
- Visualisation pédagogique

---

## 1️⃣ Prompt MediaPipe Football Setup

```
Configure MediaPipe Holistic pour l'analyse biomécanique football :

1. CONFIGURATION OPTIMISÉE :
```python
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class BiomechanicalKeypoints:
    # 33 pose landmarks
    pose_landmarks: np.ndarray  # (33, 3) x,y,z
    pose_world_landmarks: np.ndarray  # Real world coordinates
    
    # Angles calculés
    joint_angles: Dict[str, float]
    
    # Métriques dérivées
    center_of_mass: np.ndarray  # (3,)
    base_of_support: float  # Area in m²
    velocity_vectors: Dict[str, np.ndarray]
    
    # Qualité détection
    visibility_scores: np.ndarray  # (33,)
    confidence: float

class FootballPoseAnalyzer:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2,  # Plus précis pour sport
            smooth_landmarks=True,
            enable_segmentation=False,  # Pas nécessaire
            refine_face_landmarks=False  # Focus sur corps
        )
        
    def analyze_player_pose(self, frame: np.ndarray, player_bbox: BBox) -> BiomechanicalKeypoints:
        # Crop sur joueur détecté
        # Process avec MediaPipe
        # Calcul angles articulaires
        # Estimation vélocités
        # Détection équilibre
```

2. CALCULS ANGLES SPÉCIFIQUES :
   - Flexion/Extension : genou, hanche, cheville
   - Rotation : tronc, hanches, épaules
   - Abduction/Adduction : hanches, épaules
   - Angles composés : chaîne cinétique complète

3. LISSAGE TEMPOREL :
   - Filtre Savitzky-Golay sur trajectoires
   - Interpolation keypoints manquants
   - Détection outliers (poses impossibles)
   - Continuité inter-frames

4. CALIBRATION TERRAIN :
   - Estimation taille réelle joueur
   - Conversion pixels → mètres
   - Compensation perspective caméra
   - Normalisation par taille

Implémente avec gestion multi-joueurs simultanés.
```

## 2️⃣ Prompt Analyse Geste : Passe

```
Développe l'analyse biomécanique complète du geste de passe :

1. PHASES DU GESTE :
```python
@dataclass
class PassAnalysis:
    # Phases temporelles
    preparation_phase: PhaseMetrics  # Armé du geste
    contact_phase: PhaseMetrics      # Impact pied-ballon
    follow_through: PhaseMetrics     # Accompagnement
    
    # Métriques clés
    ankle_angle_contact: float  # Angle cheville au contact (optimal: 90°)
    hip_rotation: float         # Rotation hanches (45-60°)
    support_foot_distance: float # Distance pied d'appui-ballon
    body_orientation: float     # Angle corps vs direction passe
    
    # Qualité biomécanique
    balance_score: float        # Stabilité pendant geste
    kinetic_chain_score: float  # Coordination segments
    timing_score: float         # Synchronisation phases
    
    # Erreurs détectées
    errors: List[BiomechanicalError]
    
class PassBiomechanics:
    def analyze_pass(self, pose_sequence: List[BiomechanicalKeypoints], ball_trajectory: BallData) -> PassAnalysis:
        # 1. Détection phases automatique
        phase_detector = PassPhaseDetector()
        phases = phase_detector.detect_phases(pose_sequence, ball_trajectory)
        
        # 2. Calculs par phase
        prep_metrics = self.analyze_preparation(phases.preparation)
        contact_metrics = self.analyze_contact(phases.contact)
        follow_metrics = self.analyze_follow_through(phases.follow_through)
        
        # 3. Métriques globales
        balance = self.compute_balance_score(pose_sequence)
        chain = self.analyze_kinetic_chain(pose_sequence)
        
        # 4. Détection erreurs
        errors = self.detect_common_errors(pose_sequence)
```

2. ERREURS COMMUNES PASSE :
   - Pied d'appui trop loin/près
   - Cheville molle au contact
   - Manque rotation hanches
   - Déséquilibre arrière
   - Regard pas sur cible

3. SCORING PONDÉRÉ :
   ```python
   weights = {
       "ankle_angle": 0.25,
       "hip_rotation": 0.20,
       "balance": 0.20,
       "timing": 0.15,
       "body_orientation": 0.10,
       "follow_through": 0.10
   }
   ```

4. FEEDBACK CORRECTIF :
   - Visualisation angle optimal vs réel
   - Suggestion exercices spécifiques
   - Comparaison pro référence
   - Points d'attention prioritaires

Génère analyse complète avec visualisations.
```

## 3️⃣ Prompt Analyse Geste : Frappe

```
Implémente l'analyse biomécanique de la frappe au but :

1. ANALYSE DÉTAILLÉE FRAPPE :
```python
class StrikeAnalysis:
    def analyze_shot(self, pose_sequence: List[BiomechanicalKeypoints], ball_data: BallData) -> ShotMetrics:
        # Phases spécifiques frappe
        phases = {
            "approach": self.detect_approach_phase(),      # Course d'élan
            "plant": self.detect_plant_phase(),           # Pose pied d'appui
            "backswing": self.detect_backswing_phase(),   # Armé jambe
            "acceleration": self.detect_acceleration(),    # Accélération jambe
            "contact": self.detect_contact_phase(),        # Impact
            "follow_through": self.detect_follow_through() # Accompagnement
        }
        
        # Métriques biomécaniques
        metrics = {
            "foot_velocity": self.calculate_foot_speed(phases["contact"]),
            "knee_angle_backswing": self.measure_knee_flexion(phases["backswing"]),
            "trunk_lean": self.measure_trunk_angle(phases["contact"]),
            "support_foot_placement": self.analyze_plant_position(phases["plant"]),
            "hip_angular_velocity": self.calculate_hip_rotation_speed(),
            "ankle_lock": self.measure_ankle_stiffness(phases["contact"])
        }
        
        # Calcul puissance
        power_metrics = self.calculate_power_generation(pose_sequence)
        
        return ShotMetrics(phases, metrics, power_metrics)
```

2. MÉTRIQUES PUISSANCE :
   - Vitesse pied au contact (>70 km/h pro)
   - Transfert énergie cinétique
   - Angle optimal genou (100-120°)
   - Contribution segments (hanches 35%, cuisse 25%, jambe 40%)

3. TYPES DE FRAPPE :
   - Intérieur pied (précision)
   - Coup de pied (puissance)
   - Extérieur (effet)
   - Volée (timing)
   - Tête (extension cou)

4. ERREURS FRÉQUENTES :
   - Pied d'appui mal placé
   - Corps penché arrière
   - Cheville pas verrouillée
   - Manque amplitude armé
   - Mauvais timing

5. ANALYSE TRAJECTOIRE :
   - Prédiction trajectoire ballon
   - Calcul effet (Magnus)
   - Zone but visée
   - Probabilité réussite

Intègre comparaison données pros (Mbappé, Haaland, etc.).
```

## 4️⃣ Prompt Analyse Geste : Contrôle

```
Développe l'analyse du contrôle de balle multi-surfaces :

1. TYPES DE CONTRÔLES :
```python
class BallControlAnalyzer:
    def analyze_control(self, pose_sequence: List[BiomechanicalKeypoints], ball_trajectory: BallData) -> ControlAnalysis:
        # Détection surface utilisée
        control_surface = self.detect_control_surface(pose_sequence, ball_trajectory)
        
        # Analyse par type
        if control_surface == "foot":
            return self.analyze_foot_control(pose_sequence, ball_trajectory)
        elif control_surface == "chest":
            return self.analyze_chest_control(pose_sequence, ball_trajectory)
        elif control_surface == "thigh":
            return self.analyze_thigh_control(pose_sequence, ball_trajectory)
        elif control_surface == "head":
            return self.analyze_header_control(pose_sequence, ball_trajectory)
            
    def analyze_foot_control(self, poses, ball):
        metrics = {
            "cushioning_quality": self.measure_deceleration_curve(ball),
            "first_touch_distance": self.calculate_ball_distance_after_control(ball),
            "body_orientation_after": self.measure_body_angle_post_control(poses),
            "balance_maintained": self.check_com_stability(poses),
            "preparation_time": self.measure_anticipation_timing(poses, ball)
        }
```

2. BIOMÉCANIQUE AMORTI :
   - Courbe décélération ballon
   - Relâchement musculaire
   - Accompagnement mouvement
   - Surface contact optimale
   - Orientation corps anticipée

3. CONTRÔLES COMPLEXES :
   - Contrôle orienté
   - Contrôle en mouvement
   - Contrôle sous pression
   - Jonglage technique
   - Contrôle aérien

4. MÉTRIQUES QUALITÉ :
   - Distance ballon après (< 1m optimal)
   - Temps pour 2ème touche
   - Stabilité posturale
   - Fluidité enchaînement
   - Adaptation vitesse balle

5. FEEDBACK VISUEL :
   - Trajectoire balle 3D
   - Zone contrôle optimal
   - Vecteurs force/vitesse
   - Replay au ralenti
   - Comparaison pro

Génère exercices personnalisés selon faiblesses.
```

## 5️⃣ Prompt Analyse Physique

```
Intègre l'analyse des capacités physiques durant les gestes :

1. MÉTRIQUES PHYSIQUES :
```python
class PhysicalMetricsAnalyzer:
    def analyze_physical_performance(self, pose_sequence: List[BiomechanicalKeypoints], duration: float) -> PhysicalMetrics:
        metrics = PhysicalMetrics()
        
        # Vitesse et accélération
        metrics.max_speed = self.calculate_max_speed(pose_sequence)
        metrics.acceleration_profile = self.calculate_acceleration_curve(pose_sequence)
        metrics.deceleration_capacity = self.analyze_braking(pose_sequence)
        
        # Puissance et explosivité  
        metrics.jump_height = self.detect_and_measure_jumps(pose_sequence)
        metrics.power_output = self.calculate_power_metrics(pose_sequence)
        metrics.reactive_strength = self.measure_ground_contact_time(pose_sequence)
        
        # Agilité et coordination
        metrics.change_of_direction = self.analyze_cod_ability(pose_sequence)
        metrics.balance_score = self.continuous_balance_assessment(pose_sequence)
        metrics.coordination_index = self.calculate_limb_coordination(pose_sequence)
        
        # Endurance estimée
        metrics.movement_efficiency = self.calculate_economy(pose_sequence)
        metrics.fatigue_indicators = self.detect_fatigue_signs(pose_sequence)
        
        return metrics
```

2. DÉTECTION FATIGUE :
   - Dégradation technique
   - Réduction amplitudes
   - Perte coordination
   - Baisse vitesses
   - Augmentation déséquilibres

3. PROFIL ATHLÉTIQUE :
   - Type fibre musculaire dominant
   - Profil force-vitesse
   - Asymétries gauche/droite
   - Points forts/faibles
   - Comparaison position

4. PRÉVENTION BLESSURES :
   - Déséquilibres musculaires
   - Stress articulaire excessif
   - Patterns mouvement risqués
   - Charge biomécanique
   - Recommandations préventives

5. SUIVI LONGITUDINAL :
   - Evolution performances
   - Détection surcharge
   - Optimisation charge
   - Périodisation suggérée
   - Alertes automatiques

Crée dashboard personnalisé par joueur.
```

## 6️⃣ Prompt Visualisation 3D

```
Développe un système de visualisation 3D interactif :

1. RENDU 3D SQUELETTE :
```python
import plotly.graph_objects as go
from typing import List
import numpy as np

class Biomechanics3DVisualizer:
    def __init__(self):
        self.skeleton_connections = [
            # Définition connexions MediaPipe
            (11, 12), (11, 13), (13, 15),  # Bras gauche
            (12, 14), (14, 16),  # Bras droit
            (11, 23), (12, 24),  # Torse
            (23, 24), (23, 25), (25, 27),  # Jambe gauche
            (24, 26), (26, 28)   # Jambe droite
        ]
        
    def create_3d_animation(self, pose_sequence: List[BiomechanicalKeypoints]) -> go.Figure:
        # Animation squelette 3D
        frames = []
        for pose in pose_sequence:
            frame_data = self.create_skeleton_frame(pose)
            frames.append(frame_data)
            
        fig = go.Figure(
            data=frames[0],
            layout=self.get_3d_layout(),
            frames=[go.Frame(data=f) for f in frames]
        )
        
        # Ajout contrôles animation
        fig.update_layout(
            updatemenus=[self.get_animation_controls()],
            sliders=[self.get_timeline_slider(len(frames))]
        )
        
        return fig
```

2. OVERLAYS INFORMATIFS :
   - Angles articulaires temps réel
   - Vecteurs vitesse/force
   - Centre de masse
   - Base de sustentation
   - Trajectoires points clés

3. COMPARAISON CÔTE À CÔTE :
   - Joueur vs référence pro
   - Avant vs après correction
   - Multiple tentatives
   - Evolution temporelle
   - Angles optimaux

4. EXPORT VIDÉO :
   - Rendu HD avec annotations
   - Slow motion zones clés
   - Multiple angles vue
   - Données incrustées
   - Format coach-friendly

5. RÉALITÉ AUGMENTÉE :
   - Projection terrain réel
   - Feedback temps réel
   - Guides visuels correction
   - Gamification apprentissage
   - Export mobile

Intègre avec interface web React/Three.js.
```

## 7️⃣ Prompt Génération Exercices

```
Crée un système de recommandation d'exercices personnalisés :

1. MOTEUR RECOMMANDATION :
```python
class ExerciseRecommendationEngine:
    def __init__(self):
        self.exercise_database = ExerciseDB()
        self.progression_rules = ProgressionRules()
        
    def generate_training_plan(self, analysis_results: BiomechanicalAnalysis, player_profile: PlayerProfile) -> TrainingPlan:
        # Identification points faibles
        weaknesses = self.identify_improvement_areas(analysis_results)
        
        # Sélection exercices ciblés
        exercises = []
        for weakness in weaknesses:
            targeted_exercises = self.exercise_database.query(
                target_area=weakness.area,
                difficulty=player_profile.skill_level,
                equipment=player_profile.available_equipment
            )
            exercises.extend(targeted_exercises[:3])  # Top 3 par faiblesse
            
        # Organisation progression
        plan = self.organize_progression(exercises, player_profile)
        
        # Ajout exercices spécifiques
        plan.add_gesture_specific_drills(analysis_results.gesture_type)
        
        return plan
```

2. BASE EXERCICES FOOTBALL :
   - Exercices techniques (1000+)
   - Progressions par niveau
   - Vidéos démonstratives
   - Points coaching clés
   - Erreurs à éviter

3. PERSONNALISATION :
   - Âge et niveau joueur
   - Position terrain
   - Objectifs spécifiques
   - Temps disponible
   - Équipement accessible

4. SUIVI PROGRESSION :
   - Tests réguliers
   - Ajustement difficulté
   - Métriques amélioration
   - Motivation/gamification
   - Partage coach

5. INTÉGRATION APP :
   - Notifications entraînement
   - Vidéos tutoriels in-app
   - Tracking réalisation
   - Social features
   - Challenges équipe

Génère PDF plan entraînement exportable.
```

## 🎯 KPIs Biomécanique

```yaml
precision_cibles:
  angle_detection: ±2 degrés
  velocity_estimation: ±5%
  timing_phases: ±50ms
  balance_score: correlation_expert > 0.85

performance:
  inference_time: <100ms par frame
  gpu_memory: <2GB
  cpu_usage: <40%
  
validation:
  dataset: "500 gestes annotés par experts"
  inter_rater_reliability: >0.80
  test_retest: >0.85
```

## 📝 Checklist Module

- [ ] MediaPipe intégré et optimisé
- [ ] Analyse 5 gestes principaux
- [ ] Scoring biomécanique validé
- [ ] Visualisation 3D fonctionnelle
- [ ] Détection erreurs >90% précision
- [ ] Génération exercices pertinents
- [ ] Documentation technique complète 