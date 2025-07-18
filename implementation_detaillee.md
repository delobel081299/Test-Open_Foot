# Implémentation Technique Détaillée

## 🔧 Spécifications Modules Critiques

### 1. **Détection d'Actions (Action Segmentation)**

#### Problème identifié dans votre pipeline :
Votre pipeline mentionne "découpage automatique des actions" sans préciser la méthode. Voici une solution robuste :

```python
class ActionSegmentationEngine:
    def __init__(self):
        self.temporal_detector = TemporalActionLocalization()
        self.ball_contact_detector = BallContactDetector()
        self.phase_classifier = GamePhaseClassifier()
    
    def segment_actions(self, video_frames, poses, ball_detections):
        """
        Segmentation intelligente basée sur:
        - Contact ballon (événement déclencheur)
        - Changement de pose significatif
        - Contexte temporel
        """
        # 1. Détection contact ballon (méthode améliorée)
        ball_contacts = self.detect_ball_contact_advanced(
            video_frames, poses, ball_detections
        )
        
        # 2. Fenêtrage adaptatif autour des contacts
        action_windows = self.create_adaptive_windows(ball_contacts)
        
        # 3. Classification du type d'action
        action_types = self.classify_action_types(action_windows)
        
        return action_windows, action_types
    
    def detect_ball_contact_advanced(self, frames, poses, balls):
        """
        Méthode multi-critères pour détecter le contact ballon
        """
        criteria = {
            'distance_foot_ball': self.foot_ball_proximity(poses, balls),
            'ball_velocity_change': self.ball_trajectory_analysis(balls),
            'pose_intention': self.pose_intention_analysis(poses),
            'audio_impact': self.audio_contact_detection(frames)  # Nouveau
        }
        
        # Fusion bayésienne des critères
        contact_probability = self.bayesian_fusion(criteria)
        return contact_probability > 0.7
```

### 2. **Estimation Frames Clés (Solution Recommandée)**

#### Votre question : "Estimation des frames clés (ex : contact pied/ballon)"

```python
class KeyFrameEstimator:
    def __init__(self):
        self.contact_detector = ContactMomentDetector()
        self.peak_detector = MovementPeakDetector()
        
    def estimate_key_frames(self, action_sequence):
        """
        Identification automatique des frames critiques
        """
        key_frames = {
            'preparation': self.find_preparation_phase(action_sequence),
            'contact': self.find_contact_moment(action_sequence),
            'follow_through': self.find_follow_through(action_sequence),
            'result': self.find_action_result(action_sequence)
        }
        
        return key_frames
    
    def find_contact_moment(self, sequence):
        """
        Détection précise du moment de contact
        Méthodes combinées :
        """
        # 1. Analyse trajectoire ballon (changement directionnel maximal)
        ball_trajectory_peaks = self.analyze_ball_trajectory_change(sequence)
        
        # 2. Analyse pose (distance minimale pied-ballon)
        pose_contact_moments = self.analyze_foot_ball_distance(sequence)
        
        # 3. Analyse visuelle (changement intensité pixels)
        visual_impact_moments = self.analyze_visual_impact(sequence)
        
        # 4. Fusion temporelle
        contact_frame = self.temporal_fusion([
            ball_trajectory_peaks,
            pose_contact_moments,
            visual_impact_moments
        ])
        
        return contact_frame
```

### 3. **Coordination Motrice (Méthode Proposée)**

#### Votre demande : "Coordination motrice (proposition de méthode attendue)"

```python
class MotorCoordinationAnalyzer:
    def __init__(self):
        self.symmetry_analyzer = BodySymmetryAnalyzer()
        self.timing_analyzer = MovementTimingAnalyzer()
        self.efficiency_calculator = MovementEfficiencyCalculator()
    
    def assess_motor_coordination(self, pose_sequence, action_type):
        """
        Évaluation complète de la coordination motrice
        """
        coordination_metrics = {
            'temporal_coordination': self.analyze_temporal_coordination(pose_sequence),
            'spatial_coordination': self.analyze_spatial_coordination(pose_sequence),
            'inter_limb_coordination': self.analyze_inter_limb_coordination(pose_sequence),
            'balance_control': self.analyze_balance_control(pose_sequence),
            'movement_fluidity': self.analyze_movement_fluidity(pose_sequence)
        }
        
        # Score global de coordination (0-100)
        coordination_score = self.calculate_coordination_score(coordination_metrics)
        
        # Recommandations d'amélioration spécifiques
        recommendations = self.generate_coordination_feedback(
            coordination_metrics, action_type
        )
        
        return {
            'score': coordination_score,
            'metrics': coordination_metrics,
            'recommendations': recommendations
        }
    
    def analyze_temporal_coordination(self, poses):
        """
        Analyse de la coordination temporelle des mouvements
        """
        # 1. Séquencement des mouvements corporels
        movement_sequence = self.extract_movement_sequence(poses)
        
        # 2. Évaluation de la synchronisation
        synchronization_score = self.evaluate_movement_synchronization(movement_sequence)
        
        # 3. Détection des décalages temporels anormaux
        timing_errors = self.detect_timing_errors(movement_sequence)
        
        return {
            'synchronization_score': synchronization_score,
            'timing_errors': timing_errors,
            'phase_transitions': self.analyze_phase_transitions(movement_sequence)
        }
    
    def analyze_inter_limb_coordination(self, poses):
        """
        Coordination entre les membres (bras-jambes, gauche-droite)
        """
        # Extraction des patterns de mouvement par membre
        limb_movements = {
            'left_arm': self.extract_limb_movement(poses, 'left_arm'),
            'right_arm': self.extract_limb_movement(poses, 'right_arm'),
            'left_leg': self.extract_limb_movement(poses, 'left_leg'),
            'right_leg': self.extract_limb_movement(poses, 'right_leg')
        }
        
        # Calcul de corrélations croisées
        coordination_patterns = self.calculate_cross_correlations(limb_movements)
        
        # Évaluation de la coordination optimale pour l'action
        optimal_coordination = self.get_optimal_coordination_pattern(action_type)
        coordination_quality = self.compare_to_optimal(
            coordination_patterns, optimal_coordination
        )
        
        return coordination_quality
```

## 🎯 Modèles d'Évaluation Spécialisés

### 1. **Système Expert pour Règles Biomécaniques**

```python
class BiomechanicalExpertSystem:
    def __init__(self):
        self.rules_database = self.load_expert_rules()
        self.technique_standards = self.load_technique_standards()
    
    def evaluate_technique(self, action_type, biomech_features):
        """
        Évaluation basée sur règles expertes biomécaniques
        """
        evaluation = {
            'technical_score': 0,
            'errors_detected': [],
            'improvement_points': [],
            'biomech_analysis': {}
        }
        
        # Règles spécifiques par technique
        if action_type == 'passe_courte':
            evaluation = self.evaluate_short_pass(biomech_features)
        elif action_type == 'frappe':
            evaluation = self.evaluate_shot(biomech_features)
        elif action_type == 'controle':
            evaluation = self.evaluate_ball_control(biomech_features)
        # ... autres techniques
        
        return evaluation
    
    def evaluate_short_pass(self, features):
        """
        Règles expertes pour l'évaluation de la passe courte
        """
        score = 100
        errors = []
        improvements = []
        
        # Règle 1 : Angle du pied au contact
        foot_angle = features['foot_angle_at_contact']
        if not (15 <= foot_angle <= 45):
            score -= 15
            errors.append("Angle du pied incorrect au contact")
            improvements.append("Orienter le pied entre 15° et 45° vers la cible")
        
        # Règle 2 : Position du corps
        body_position = features['body_position']
        if body_position['lean_angle'] > 20:
            score -= 10
            errors.append("Corps trop penché")
            improvements.append("Maintenir l'équilibre avec le buste plus droit")
        
        # Règle 3 : Suivi du geste
        follow_through = features['follow_through_distance']
        if follow_through < 0.3:  # 30cm minimum
            score -= 20
            errors.append("Suivi du geste insuffisant")
            improvements.append("Prolonger le mouvement vers la cible après le contact")
        
        # Règle 4 : Pied d'appui
        support_foot = features['support_foot_position']
        if support_foot['distance_to_ball'] > 0.25:  # 25cm max
            score -= 12
            errors.append("Pied d'appui trop éloigné")
            improvements.append("Placer le pied d'appui plus près du ballon")
        
        return {
            'technical_score': max(0, score),
            'errors_detected': errors,
            'improvement_points': improvements,
            'biomech_analysis': self.detailed_biomech_analysis(features)
        }
```

### 2. **Métriques Avancées de Performance**

```python
class AdvancedPerformanceMetrics:
    def __init__(self):
        self.efficiency_calculator = MovementEfficiencyCalculator()
        self.power_analyzer = PowerTransferAnalyzer()
        
    def calculate_advanced_metrics(self, action_data):
        """
        Calcul de métriques avancées de performance
        """
        metrics = {
            # Métriques biomécaniques
            'movement_efficiency': self.calculate_movement_efficiency(action_data),
            'power_transfer_coefficient': self.calculate_power_transfer(action_data),
            'energy_expenditure': self.calculate_energy_expenditure(action_data),
            
            # Métriques techniques
            'precision_index': self.calculate_precision_index(action_data),
            'consistency_score': self.calculate_consistency_score(action_data),
            'adaptability_rating': self.calculate_adaptability_rating(action_data),
            
            # Métriques cognitives
            'decision_timing': self.calculate_decision_timing(action_data),
            'situational_awareness': self.calculate_situational_awareness(action_data),
            'anticipation_quality': self.calculate_anticipation_quality(action_data)
        }
        
        return metrics
    
    def calculate_movement_efficiency(self, action_data):
        """
        Efficacité du mouvement : ratio résultat/effort
        """
        # Effort mesuré par la complexité du mouvement
        movement_complexity = self.measure_movement_complexity(action_data['poses'])
        
        # Résultat mesuré par la précision de l'action
        action_success = self.measure_action_success(action_data)
        
        # Efficacité = Succès / Complexité
        efficiency = action_success / (movement_complexity + 1e-6)
        
        return min(1.0, efficiency)  # Normalisé entre 0 et 1
```

## 🔍 Solutions aux Points Flous Identifiés

### 1. **Attribution d'Équipe (Amélioration)**

```python
class TeamClassificationEngine:
    def __init__(self):
        self.color_analyzer = ColorClusteringAnalyzer()
        self.clip_model = CLIPVisionClassifier()
        self.template_matcher = JerseyTemplateMatching()
    
    def classify_team_robust(self, player_detections, frame):
        """
        Classification robuste d'équipe multi-critères
        """
        classification_results = []
        
        for player_bbox in player_detections:
            player_region = self.extract_player_region(frame, player_bbox)
            
            # Méthode 1 : Analyse couleur dominante (rapide)
            color_prediction = self.color_analyzer.predict_team(player_region)
            
            # Méthode 2 : CLIP Vision (robuste)
            clip_prediction = self.clip_model.classify_team(player_region)
            
            # Méthode 3 : Template matching (précis)
            template_prediction = self.template_matcher.match_jersey(player_region)
            
            # Fusion bayésienne des prédictions
            team_prediction = self.bayesian_team_fusion([
                color_prediction,
                clip_prediction,
                template_prediction
            ])
            
            classification_results.append({
                'player_id': player_bbox['id'],
                'team': team_prediction['team'],
                'confidence': team_prediction['confidence']
            })
        
        return classification_results
```

### 2. **Calcul Vitesse/Direction Ballon (Optimisé)**

```python
class BallTrajectoryAnalyzer:
    def __init__(self):
        self.kalman_filter = KalmanFilterBall()
        self.optical_flow = OpticalFlowCalculator()
        
    def analyze_ball_trajectory_advanced(self, ball_detections):
        """
        Analyse avancée de trajectoire avec prédiction
        """
        # 1. Filtrage de Kalman pour trajectoire lisse
        filtered_positions = self.kalman_filter.filter_trajectory(ball_detections)
        
        # 2. Calcul vitesse instantanée et accélération
        velocities = self.calculate_instantaneous_velocity(filtered_positions)
        accelerations = self.calculate_acceleration(velocities)
        
        # 3. Détection des changements de direction significatifs
        direction_changes = self.detect_significant_direction_changes(
            filtered_positions, velocities
        )
        
        # 4. Prédiction trajectoire future (utile pour l'évaluation)
        predicted_trajectory = self.predict_future_trajectory(
            filtered_positions[-5:], velocities[-5:]
        )
        
        return {
            'positions': filtered_positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'direction_changes': direction_changes,
            'predicted_trajectory': predicted_trajectory,
            'trajectory_quality': self.assess_trajectory_quality(velocities)
        }
    
    def assess_trajectory_quality(self, velocities):
        """
        Évaluation de la qualité de la trajectoire (pour notation technique)
        """
        # Consistance de la vitesse
        velocity_consistency = 1 - np.std(velocities) / (np.mean(velocities) + 1e-6)
        
        # Smoothness de la trajectoire
        velocity_changes = np.diff(velocities)
        trajectory_smoothness = 1 - np.std(velocity_changes) / (np.mean(np.abs(velocity_changes)) + 1e-6)
        
        # Score global de qualité
        quality_score = (velocity_consistency + trajectory_smoothness) / 2
        
        return {
            'overall_quality': quality_score,
            'velocity_consistency': velocity_consistency,
            'trajectory_smoothness': trajectory_smoothness
        }
```

## 📈 Système de Notation Dual (Implémentation)

```python
class DualScoringSystem:
    def __init__(self):
        self.biomech_scorer = BiomechanicalScorer()
        self.performance_scorer = PerformanceScorer()
        
    def generate_dual_score(self, action_analysis):
        """
        Génération du système de double notation
        """
        # Note biomécanique (précision technique)
        biomech_score = self.biomech_scorer.calculate_score(
            action_analysis['biomech_features'],
            action_analysis['action_type']
        )
        
        # Note terrain (performance globale)
        performance_score = self.performance_scorer.calculate_score(
            action_analysis['context_features'],
            action_analysis['result_effectiveness']
        )
        
        return {
            'biomechanical_score': {
                'value': biomech_score['total'],
                'breakdown': biomech_score['details'],
                'areas_improvement': biomech_score['improvements']
            },
            'performance_score': {
                'value': performance_score['total'],
                'breakdown': performance_score['details'],
                'tactical_rating': performance_score['tactical_component']
            },
            'combined_insights': self.generate_combined_insights(
                biomech_score, performance_score
            )
        }
```

Cette implémentation détaillée résout les points flous de votre pipeline initial et propose des solutions concrètes pour chaque défi technique identifié. 