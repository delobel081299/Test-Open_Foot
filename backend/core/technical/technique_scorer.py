"""
Module d'évaluation technique des gestes footballistiques
Analyse biomécanique détaillée et scoring expert
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import cv2
from pathlib import Path
import json
from datetime import datetime

from backend.core.biomechanics.pose_extractor import PoseExtractor
from backend.core.biomechanics.movement_analyzer import MovementAnalyzer


class TechniqueType(Enum):
    """Types de gestes techniques à analyser"""
    PASS = "pass"
    SHOT = "shot"
    CONTROL = "control"
    DRIBBLE = "dribble"


@dataclass
class TechnicalScore:
    """Score technique détaillé"""
    gesture_type: TechniqueType
    overall_score: float  # 0-100
    sub_scores: Dict[str, float]  # Scores par critère
    improvements: List[str]  # Points d'amélioration
    strengths: List[str]  # Points forts
    comparison_gif: Optional[str] = None  # Chemin vers GIF comparatif
    frame_analysis: Optional[Dict] = None  # Analyse frame par frame


@dataclass
class BiomechanicalMetrics:
    """Métriques biomécaniques extraites"""
    joint_angles: Dict[str, float]
    velocities: Dict[str, float]
    accelerations: Dict[str, float]
    positions: Dict[str, np.ndarray]
    timing: Dict[str, float]


class TechniqueScorer:
    """Évaluateur technique expert pour les gestes footballistiques"""
    
    def __init__(self, reference_data_path: Optional[str] = None):
        self.pose_extractor = PoseExtractor()
        self.movement_analyzer = MovementAnalyzer()
        
        # Chargement des données de référence pros
        self.reference_data = self._load_reference_data(reference_data_path)
        
        # Poids pour chaque critère par type de geste
        self.criteria_weights = {
            TechniqueType.PASS: {
                "timing": 0.20,
                "surface": 0.15,
                "vision": 0.20,
                "follow_through": 0.20,
                "accuracy": 0.25
            },
            TechniqueType.SHOT: {
                "support_foot": 0.20,
                "approach_angle": 0.15,
                "impact_point": 0.25,
                "leg_extension": 0.20,
                "balance": 0.20
            },
            TechniqueType.CONTROL: {
                "first_touch": 0.30,
                "orientation": 0.25,
                "ball_distance": 0.20,
                "next_action": 0.25
            },
            TechniqueType.DRIBBLE: {
                "touch_frequency": 0.25,
                "direction_changes": 0.25,
                "ball_protection": 0.25,
                "body_feints": 0.25
            }
        }
        
        # Seuils experts pour chaque critère
        self.expert_thresholds = self._initialize_expert_thresholds()
    
    def analyze_technique(self, video_path: str, technique_type: TechniqueType,
                         start_frame: int, end_frame: int,
                         player_id: Optional[int] = None) -> TechnicalScore:
        """
        Analyse technique complète d'un geste
        
        Args:
            video_path: Chemin de la vidéo
            technique_type: Type de geste à analyser
            start_frame: Frame de début
            end_frame: Frame de fin
            player_id: ID du joueur à analyser
            
        Returns:
            Score technique détaillé
        """
        # Extraction des métriques biomécaniques
        metrics = self._extract_biomechanical_metrics(
            video_path, start_frame, end_frame, player_id
        )
        
        # Analyse selon le type de geste
        if technique_type == TechniqueType.PASS:
            score = self._analyze_pass(metrics)
        elif technique_type == TechniqueType.SHOT:
            score = self._analyze_shot(metrics)
        elif technique_type == TechniqueType.CONTROL:
            score = self._analyze_control(metrics)
        elif technique_type == TechniqueType.DRIBBLE:
            score = self._analyze_dribble(metrics)
        else:
            raise ValueError(f"Type de technique non supporté: {technique_type}")
        
        # Génération du GIF comparatif si référence disponible
        if self.reference_data and technique_type.value in self.reference_data:
            score.comparison_gif = self._generate_comparison_gif(
                video_path, start_frame, end_frame,
                technique_type, metrics
            )
        
        return score
    
    def _analyze_pass(self, metrics: BiomechanicalMetrics) -> TechnicalScore:
        """Analyse technique d'une passe"""
        sub_scores = {}
        improvements = []
        strengths = []
        
        # 1. Timing du contact (anticipation)
        timing_score = self._evaluate_pass_timing(metrics)
        sub_scores["timing"] = timing_score
        if timing_score < 70:
            improvements.append("Anticiper davantage le mouvement pour un meilleur timing")
        else:
            strengths.append("Excellent timing d'anticipation")
        
        # 2. Surface utilisée
        surface_score = self._evaluate_pass_surface(metrics)
        sub_scores["surface"] = surface_score
        if surface_score < 70:
            improvements.append("Varier les surfaces de contact (intérieur/extérieur)")
        else:
            strengths.append("Bonne utilisation de la surface de contact")
        
        # 3. Direction du regard
        vision_score = self._evaluate_pass_vision(metrics)
        sub_scores["vision"] = vision_score
        if vision_score < 70:
            improvements.append("Lever la tête avant la passe pour mieux visualiser")
        else:
            strengths.append("Excellente vision du jeu avant la passe")
        
        # 4. Follow-through
        follow_score = self._evaluate_pass_follow_through(metrics)
        sub_scores["follow_through"] = follow_score
        if follow_score < 70:
            improvements.append("Accompagner davantage le geste après l'impact")
        else:
            strengths.append("Follow-through bien exécuté")
        
        # 5. Précision direction/force
        accuracy_score = self._evaluate_pass_accuracy(metrics)
        sub_scores["accuracy"] = accuracy_score
        if accuracy_score < 70:
            improvements.append("Améliorer la précision de direction et de dosage")
        else:
            strengths.append("Excellente précision de passe")
        
        # Calcul du score global pondéré
        weights = self.criteria_weights[TechniqueType.PASS]
        overall_score = sum(sub_scores[key] * weights[key] for key in sub_scores)
        
        return TechnicalScore(
            gesture_type=TechniqueType.PASS,
            overall_score=overall_score,
            sub_scores=sub_scores,
            improvements=improvements,
            strengths=strengths,
            frame_analysis=self._get_frame_analysis(metrics)
        )
    
    def _analyze_shot(self, metrics: BiomechanicalMetrics) -> TechnicalScore:
        """Analyse technique d'un tir"""
        sub_scores = {}
        improvements = []
        strengths = []
        
        # 1. Position pied d'appui
        support_score = self._evaluate_shot_support_foot(metrics)
        sub_scores["support_foot"] = support_score
        if support_score < 70:
            improvements.append("Placer le pied d'appui à côté du ballon, pas trop loin")
        else:
            strengths.append("Excellent placement du pied d'appui")
        
        # 2. Angle d'approche
        angle_score = self._evaluate_shot_approach_angle(metrics)
        sub_scores["approach_angle"] = angle_score
        if angle_score < 70:
            improvements.append("Optimiser l'angle d'approche vers le ballon")
        else:
            strengths.append("Angle d'approche optimal")
        
        # 3. Point d'impact
        impact_score = self._evaluate_shot_impact_point(metrics)
        sub_scores["impact_point"] = impact_score
        if impact_score < 70:
            improvements.append("Frapper le ballon au centre pour plus de puissance")
        else:
            strengths.append("Point d'impact parfait sur le ballon")
        
        # 4. Extension de la jambe
        extension_score = self._evaluate_shot_leg_extension(metrics)
        sub_scores["leg_extension"] = extension_score
        if extension_score < 70:
            improvements.append("Étendre complètement la jambe lors de la frappe")
        else:
            strengths.append("Extension de jambe complète et puissante")
        
        # 5. Équilibre après tir
        balance_score = self._evaluate_shot_balance(metrics)
        sub_scores["balance"] = balance_score
        if balance_score < 70:
            improvements.append("Maintenir l'équilibre après la frappe")
        else:
            strengths.append("Excellent équilibre post-frappe")
        
        # Calcul du score global
        weights = self.criteria_weights[TechniqueType.SHOT]
        overall_score = sum(sub_scores[key] * weights[key] for key in sub_scores)
        
        return TechnicalScore(
            gesture_type=TechniqueType.SHOT,
            overall_score=overall_score,
            sub_scores=sub_scores,
            improvements=improvements,
            strengths=strengths,
            frame_analysis=self._get_frame_analysis(metrics)
        )
    
    def _analyze_control(self, metrics: BiomechanicalMetrics) -> TechnicalScore:
        """Analyse technique d'un contrôle"""
        sub_scores = {}
        improvements = []
        strengths = []
        
        # 1. Amorti première touche
        touch_score = self._evaluate_control_first_touch(metrics)
        sub_scores["first_touch"] = touch_score
        if touch_score < 70:
            improvements.append("Amortir davantage le ballon à la première touche")
        else:
            strengths.append("Excellent amorti du ballon")
        
        # 2. Orientation après contrôle
        orientation_score = self._evaluate_control_orientation(metrics)
        sub_scores["orientation"] = orientation_score
        if orientation_score < 70:
            improvements.append("Orienter le contrôle vers l'espace libre")
        else:
            strengths.append("Très bonne orientation du contrôle")
        
        # 3. Distance ballon-corps
        distance_score = self._evaluate_control_ball_distance(metrics)
        sub_scores["ball_distance"] = distance_score
        if distance_score < 70:
            improvements.append("Garder le ballon plus proche du corps")
        else:
            strengths.append("Distance ballon-corps optimale")
        
        # 4. Rapidité d'enchaînement
        chain_score = self._evaluate_control_next_action(metrics)
        sub_scores["next_action"] = chain_score
        if chain_score < 70:
            improvements.append("Enchaîner plus rapidement après le contrôle")
        else:
            strengths.append("Enchaînement rapide et fluide")
        
        # Calcul du score global
        weights = self.criteria_weights[TechniqueType.CONTROL]
        overall_score = sum(sub_scores[key] * weights[key] for key in sub_scores)
        
        return TechnicalScore(
            gesture_type=TechniqueType.CONTROL,
            overall_score=overall_score,
            sub_scores=sub_scores,
            improvements=improvements,
            strengths=strengths,
            frame_analysis=self._get_frame_analysis(metrics)
        )
    
    def _analyze_dribble(self, metrics: BiomechanicalMetrics) -> TechnicalScore:
        """Analyse technique d'un dribble"""
        sub_scores = {}
        improvements = []
        strengths = []
        
        # 1. Fréquence des touches
        frequency_score = self._evaluate_dribble_touch_frequency(metrics)
        sub_scores["touch_frequency"] = frequency_score
        if frequency_score < 70:
            improvements.append("Augmenter la fréquence des touches de balle")
        else:
            strengths.append("Excellente fréquence de touches")
        
        # 2. Changements de direction
        direction_score = self._evaluate_dribble_direction_changes(metrics)
        sub_scores["direction_changes"] = direction_score
        if direction_score < 70:
            improvements.append("Varier davantage les changements de direction")
        else:
            strengths.append("Changements de direction imprévisibles")
        
        # 3. Protection du ballon
        protection_score = self._evaluate_dribble_ball_protection(metrics)
        sub_scores["ball_protection"] = protection_score
        if protection_score < 70:
            improvements.append("Mieux protéger le ballon avec le corps")
        else:
            strengths.append("Excellente protection du ballon")
        
        # 4. Feintes de corps
        feint_score = self._evaluate_dribble_body_feints(metrics)
        sub_scores["body_feints"] = feint_score
        if feint_score < 70:
            improvements.append("Utiliser plus de feintes de corps")
        else:
            strengths.append("Feintes de corps très efficaces")
        
        # Calcul du score global
        weights = self.criteria_weights[TechniqueType.DRIBBLE]
        overall_score = sum(sub_scores[key] * weights[key] for key in sub_scores)
        
        return TechnicalScore(
            gesture_type=TechniqueType.DRIBBLE,
            overall_score=overall_score,
            sub_scores=sub_scores,
            improvements=improvements,
            strengths=strengths,
            frame_analysis=self._get_frame_analysis(metrics)
        )
    
    def _extract_biomechanical_metrics(self, video_path: str, 
                                      start_frame: int, end_frame: int,
                                      player_id: Optional[int]) -> BiomechanicalMetrics:
        """Extrait les métriques biomécaniques de la séquence"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        joint_angles_series = []
        velocities_series = []
        positions_series = []
        
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extraction de la pose
            pose_data = self.pose_extractor.extract_pose(frame, player_id)
            if pose_data:
                # Calcul des angles articulaires
                angles = self.movement_analyzer.calculate_joint_angles(pose_data)
                joint_angles_series.append(angles)
                
                # Positions des articulations
                positions_series.append(pose_data)
        
        cap.release()
        
        # Calcul des vitesses et accélérations
        velocities = self._calculate_velocities(positions_series)
        accelerations = self._calculate_accelerations(velocities)
        
        # Moyennes et timing
        avg_angles = self._average_dict_series(joint_angles_series)
        avg_velocities = self._average_dict_series(velocities)
        avg_accelerations = self._average_dict_series(accelerations)
        
        # Analyse du timing
        timing = self._analyze_timing(positions_series, velocities)
        
        return BiomechanicalMetrics(
            joint_angles=avg_angles,
            velocities=avg_velocities,
            accelerations=avg_accelerations,
            positions=positions_series[-1] if positions_series else {},
            timing=timing
        )
    
    def _evaluate_pass_timing(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue le timing d'anticipation de la passe"""
        # Analyse de la préparation du geste
        prep_time = metrics.timing.get("preparation_time", 0)
        
        # Temps de préparation optimal : 0.3-0.5 secondes
        if 0.3 <= prep_time <= 0.5:
            score = 100
        elif prep_time < 0.3:
            score = max(0, 100 - (0.3 - prep_time) * 200)  # Trop rapide
        else:
            score = max(0, 100 - (prep_time - 0.5) * 100)  # Trop lent
        
        return score
    
    def _evaluate_pass_surface(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue la surface de contact utilisée"""
        # Angle du pied au moment du contact
        foot_angle = metrics.joint_angles.get("ankle_angle", 90)
        
        # Angle optimal pour intérieur du pied : 80-100°
        if 80 <= foot_angle <= 100:
            score = 100
        else:
            deviation = min(abs(foot_angle - 80), abs(foot_angle - 100))
            score = max(0, 100 - deviation * 2)
        
        return score
    
    def _evaluate_pass_vision(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue la direction du regard avant la passe"""
        # Angle de la tête par rapport au tronc
        head_angle = metrics.joint_angles.get("head_trunk_angle", 0)
        
        # Tête levée : angle > 15°
        if head_angle > 15:
            score = 100
        else:
            score = max(0, (head_angle / 15) * 100)
        
        return score
    
    def _evaluate_pass_follow_through(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue le follow-through après la passe"""
        # Extension finale de la jambe
        final_extension = metrics.joint_angles.get("knee_extension_final", 0)
        
        # Extension optimale : 160-180°
        if 160 <= final_extension <= 180:
            score = 100
        else:
            score = max(0, 100 - abs(170 - final_extension))
        
        return score
    
    def _evaluate_pass_accuracy(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue la précision de la passe"""
        # Vitesse du pied au contact
        foot_velocity = metrics.velocities.get("foot_velocity", 0)
        
        # Cohérence du mouvement (faible variance = bon contrôle)
        velocity_variance = metrics.velocities.get("velocity_variance", 1)
        
        # Score basé sur la stabilité du mouvement
        stability_score = max(0, 100 - velocity_variance * 50)
        
        # Vitesse appropriée (ni trop fort ni trop faible)
        if 3 <= foot_velocity <= 8:  # m/s
            velocity_score = 100
        else:
            velocity_score = max(0, 100 - abs(5.5 - foot_velocity) * 20)
        
        return (stability_score + velocity_score) / 2
    
    def _evaluate_shot_support_foot(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue la position du pied d'appui"""
        # Distance latérale pied d'appui - ballon
        support_distance = metrics.positions.get("support_foot_ball_distance", 0.5)
        
        # Distance optimale : 0.2-0.3m
        if 0.2 <= support_distance <= 0.3:
            score = 100
        else:
            deviation = min(abs(support_distance - 0.2), abs(support_distance - 0.3))
            score = max(0, 100 - deviation * 200)
        
        return score
    
    def _evaluate_shot_approach_angle(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue l'angle d'approche du tir"""
        # Angle d'approche par rapport au ballon
        approach_angle = metrics.joint_angles.get("approach_angle", 45)
        
        # Angle optimal : 30-45°
        if 30 <= approach_angle <= 45:
            score = 100
        else:
            score = max(0, 100 - abs(37.5 - approach_angle) * 2)
        
        return score
    
    def _evaluate_shot_impact_point(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue le point d'impact sur le ballon"""
        # Hauteur du point d'impact (0 = bas, 1 = haut du ballon)
        impact_height = metrics.positions.get("impact_height_ratio", 0.5)
        
        # Impact optimal : centre du ballon (0.4-0.6)
        if 0.4 <= impact_height <= 0.6:
            score = 100
        else:
            score = max(0, 100 - abs(0.5 - impact_height) * 200)
        
        return score
    
    def _evaluate_shot_leg_extension(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue l'extension de la jambe lors du tir"""
        # Angle du genou au moment de l'impact
        knee_angle = metrics.joint_angles.get("knee_angle_impact", 120)
        
        # Extension optimale : 170-180°
        if 170 <= knee_angle <= 180:
            score = 100
        else:
            score = max(0, 100 - (170 - knee_angle) * 2)
        
        return score
    
    def _evaluate_shot_balance(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue l'équilibre après le tir"""
        # Variance de la position du centre de masse
        com_variance = metrics.positions.get("center_mass_variance", 0.1)
        
        # Faible variance = bon équilibre
        if com_variance < 0.05:
            score = 100
        else:
            score = max(0, 100 - com_variance * 200)
        
        return score
    
    def _evaluate_control_first_touch(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue l'amorti de la première touche"""
        # Réduction de vitesse du ballon après contact
        velocity_reduction = metrics.velocities.get("ball_velocity_reduction", 0.5)
        
        # Réduction optimale : 70-90%
        if 0.7 <= velocity_reduction <= 0.9:
            score = 100
        else:
            score = max(0, 100 - abs(0.8 - velocity_reduction) * 200)
        
        return score
    
    def _evaluate_control_orientation(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue l'orientation du contrôle"""
        # Angle de sortie du ballon
        exit_angle = metrics.joint_angles.get("ball_exit_angle", 0)
        
        # Orientation vers l'espace libre (score basé sur contexte)
        # Ici on suppose qu'une déviation de 30-60° est optimale
        if 30 <= abs(exit_angle) <= 60:
            score = 100
        else:
            score = max(0, 100 - min(abs(exit_angle - 45), 45))
        
        return score
    
    def _evaluate_control_ball_distance(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue la distance ballon-corps après contrôle"""
        # Distance finale ballon-joueur
        final_distance = metrics.positions.get("ball_player_distance_final", 0.5)
        
        # Distance optimale : 0.3-0.5m
        if 0.3 <= final_distance <= 0.5:
            score = 100
        else:
            deviation = min(abs(final_distance - 0.3), abs(final_distance - 0.5))
            score = max(0, 100 - deviation * 200)
        
        return score
    
    def _evaluate_control_next_action(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue la rapidité d'enchaînement"""
        # Temps avant la prochaine action
        next_action_time = metrics.timing.get("time_to_next_action", 1.0)
        
        # Temps optimal : < 0.5s
        if next_action_time < 0.5:
            score = 100
        elif next_action_time < 1.0:
            score = 100 - (next_action_time - 0.5) * 100
        else:
            score = max(0, 50 - (next_action_time - 1.0) * 50)
        
        return score
    
    def _evaluate_dribble_touch_frequency(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue la fréquence des touches"""
        # Nombre de touches par seconde
        touch_frequency = metrics.timing.get("touch_frequency", 2.0)
        
        # Fréquence optimale : 3-4 touches/seconde
        if 3 <= touch_frequency <= 4:
            score = 100
        elif touch_frequency < 3:
            score = max(0, 100 - (3 - touch_frequency) * 30)
        else:
            score = max(0, 100 - (touch_frequency - 4) * 20)
        
        return score
    
    def _evaluate_dribble_direction_changes(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue les changements de direction"""
        # Nombre de changements de direction significatifs
        direction_changes = metrics.timing.get("direction_changes", 0)
        
        # Au moins 2-3 changements pour un bon dribble
        if direction_changes >= 2:
            score = min(100, 50 + direction_changes * 15)
        else:
            score = direction_changes * 25
        
        return score
    
    def _evaluate_dribble_ball_protection(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue la protection du ballon"""
        # Angle corps-ballon-adversaire
        protection_angle = metrics.joint_angles.get("body_ball_angle", 45)
        
        # Corps entre ballon et adversaire : angle > 120°
        if protection_angle > 120:
            score = 100
        else:
            score = max(0, (protection_angle / 120) * 100)
        
        return score
    
    def _evaluate_dribble_body_feints(self, metrics: BiomechanicalMetrics) -> float:
        """Évalue les feintes de corps"""
        # Variance des mouvements du tronc
        trunk_variance = metrics.velocities.get("trunk_movement_variance", 0)
        
        # Plus de variance = plus de feintes
        if trunk_variance > 0.3:
            score = 100
        else:
            score = (trunk_variance / 0.3) * 100
        
        return score
    
    def _calculate_velocities(self, positions_series: List[Dict]) -> List[Dict[str, float]]:
        """Calcule les vitesses à partir des positions"""
        velocities = []
        
        for i in range(1, len(positions_series)):
            frame_velocities = {}
            for joint in positions_series[i]:
                if joint in positions_series[i-1]:
                    # Vitesse = distance / temps (assumant 30 FPS)
                    pos_current = np.array(positions_series[i][joint])
                    pos_prev = np.array(positions_series[i-1][joint])
                    velocity = np.linalg.norm(pos_current - pos_prev) * 30
                    frame_velocities[f"{joint}_velocity"] = velocity
            
            velocities.append(frame_velocities)
        
        return velocities
    
    def _calculate_accelerations(self, velocities: List[Dict]) -> List[Dict[str, float]]:
        """Calcule les accélérations à partir des vitesses"""
        accelerations = []
        
        for i in range(1, len(velocities)):
            frame_accelerations = {}
            for joint_vel in velocities[i]:
                if joint_vel in velocities[i-1]:
                    # Accélération = delta vitesse / temps
                    acc = (velocities[i][joint_vel] - velocities[i-1][joint_vel]) * 30
                    frame_accelerations[joint_vel.replace("_velocity", "_acceleration")] = acc
            
            accelerations.append(frame_accelerations)
        
        return accelerations
    
    def _average_dict_series(self, dict_series: List[Dict]) -> Dict:
        """Calcule la moyenne d'une série de dictionnaires"""
        if not dict_series:
            return {}
        
        averaged = {}
        all_keys = set()
        for d in dict_series:
            all_keys.update(d.keys())
        
        for key in all_keys:
            values = [d[key] for d in dict_series if key in d]
            if values:
                averaged[key] = np.mean(values)
        
        return averaged
    
    def _analyze_timing(self, positions_series: List[Dict], 
                       velocities: List[Dict]) -> Dict[str, float]:
        """Analyse les aspects temporels du mouvement"""
        timing = {}
        
        # Temps de préparation (frames avec vitesse faible avant pic)
        if velocities:
            max_vel_idx = np.argmax([
                max(v.values()) if v else 0 for v in velocities
            ])
            
            prep_frames = 0
            for i in range(max_vel_idx):
                if velocities[i] and max(velocities[i].values()) < 1.0:
                    prep_frames += 1
            
            timing["preparation_time"] = prep_frames / 30.0  # En secondes
        
        # Autres métriques temporelles
        timing["total_duration"] = len(positions_series) / 30.0
        timing["execution_time"] = timing["total_duration"] - timing.get("preparation_time", 0)
        
        return timing
    
    def _get_frame_analysis(self, metrics: BiomechanicalMetrics) -> Dict:
        """Retourne l'analyse détaillée frame par frame"""
        return {
            "key_angles": metrics.joint_angles,
            "peak_velocities": {
                k: v for k, v in metrics.velocities.items() 
                if v > np.percentile(list(metrics.velocities.values()), 90)
            },
            "timing_markers": metrics.timing
        }
    
    def _load_reference_data(self, path: Optional[str]) -> Optional[Dict]:
        """Charge les données de référence des professionnels"""
        if not path or not Path(path).exists():
            return None
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def _initialize_expert_thresholds(self) -> Dict:
        """Initialise les seuils experts pour chaque critère"""
        return {
            "pass": {
                "prep_time_min": 0.3,
                "prep_time_max": 0.5,
                "foot_angle_min": 80,
                "foot_angle_max": 100,
                "head_angle_min": 15
            },
            "shot": {
                "support_distance_min": 0.2,
                "support_distance_max": 0.3,
                "approach_angle_min": 30,
                "approach_angle_max": 45,
                "knee_extension_min": 170
            },
            "control": {
                "velocity_reduction_min": 0.7,
                "velocity_reduction_max": 0.9,
                "ball_distance_min": 0.3,
                "ball_distance_max": 0.5,
                "reaction_time_max": 0.5
            },
            "dribble": {
                "touch_frequency_min": 3,
                "touch_frequency_max": 4,
                "direction_changes_min": 2,
                "protection_angle_min": 120,
                "trunk_variance_min": 0.3
            }
        }
    
    def _generate_comparison_gif(self, video_path: str, start_frame: int, 
                                end_frame: int, technique_type: TechniqueType,
                                metrics: BiomechanicalMetrics) -> str:
        """Génère un GIF comparatif avec une référence pro"""
        # TODO: Implémenter la génération de GIF
        # 1. Extraire les frames de la vidéo analysée
        # 2. Charger la vidéo de référence correspondante
        # 3. Synchroniser les mouvements
        # 4. Créer un GIF côte à côte
        # 5. Ajouter des annotations (angles, vitesses, etc.)
        
        output_path = f"comparisons/{technique_type.value}_{datetime.now().timestamp()}.gif"
        return output_path
    
    def generate_improvement_plan(self, scores: List[TechnicalScore]) -> Dict:
        """
        Génère un plan d'amélioration personnalisé basé sur plusieurs analyses
        
        Args:
            scores: Liste des scores techniques
            
        Returns:
            Plan d'amélioration structuré
        """
        # Identifier les points faibles récurrents
        all_improvements = []
        for score in scores:
            all_improvements.extend(score.improvements)
        
        # Compter les occurrences
        improvement_counts = {}
        for imp in all_improvements:
            improvement_counts[imp] = improvement_counts.get(imp, 0) + 1
        
        # Trier par priorité
        priority_improvements = sorted(
            improvement_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Créer le plan
        plan = {
            "priority_areas": [imp[0] for imp in priority_improvements[:3]],
            "exercises": self._get_exercises_for_improvements(priority_improvements[:3]),
            "progression_timeline": self._create_progression_timeline(priority_improvements[:3])
        }
        
        return plan
    
    def _get_exercises_for_improvements(self, improvements: List[Tuple[str, int]]) -> List[Dict]:
        """Retourne des exercices spécifiques pour chaque point d'amélioration"""
        exercise_database = {
            "Anticiper davantage le mouvement": {
                "name": "Passes en mouvement avec cibles mobiles",
                "description": "Travail de l'anticipation et du timing",
                "duration": "15 minutes",
                "frequency": "3x par semaine"
            },
            "Lever la tête avant la passe": {
                "name": "Passes avec contraintes visuelles",
                "description": "Exercices forçant la prise d'information",
                "duration": "20 minutes",
                "frequency": "Quotidien"
            },
            "Placer le pied d'appui à côté du ballon": {
                "name": "Tirs statiques avec repères au sol",
                "description": "Travail du placement du pied d'appui",
                "duration": "10 minutes",
                "frequency": "Avant chaque séance de tir"
            }
            # Ajouter plus d'exercices...
        }
        
        exercises = []
        for imp, _ in improvements:
            if imp in exercise_database:
                exercises.append(exercise_database[imp])
        
        return exercises
    
    def _create_progression_timeline(self, improvements: List[Tuple[str, int]]) -> Dict:
        """Crée une timeline de progression pour l'amélioration"""
        return {
            "week_1_2": "Focus sur le premier point d'amélioration",
            "week_3_4": "Intégration du deuxième point",
            "week_5_6": "Combinaison des améliorations",
            "week_7_8": "Perfectionnement et automatisation"
        }