"""
Movement Analyzer - Analyseur biomécanique avancé pour football
Analyse complète des mouvements avec détection de problèmes et recommandations
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import math
import time
from collections import deque, defaultdict
from enum import Enum
import logging

# Try to import scipy for advanced signal processing
try:
    from scipy.signal import savgol_filter
    from scipy.interpolate import interp1d
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from backend.core.biomechanics.pose_extractor import Pose3D, PoseExtractor
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class MovementQuality(Enum):
    """Qualité du mouvement"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"  
    POOR = "poor"
    CRITICAL = "critical"


class RiskLevel(Enum):
    """Niveau de risque blessure"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ArticularAngle:
    """Angle articulaire avec métadonnées"""
    name: str
    value: float  # Degrés
    normal_range: Tuple[float, float]  # Min, max normal
    asymmetry_score: float = 0.0  # Score asymétrie vs côté opposé
    risk_level: RiskLevel = RiskLevel.LOW
    is_within_normal: bool = True


@dataclass
class MovementMetrics:
    """Métriques biomécaniqu es complètes"""
    # Angles articulaires
    joint_angles: Dict[str, ArticularAngle] = field(default_factory=dict)
    
    # Symétrie
    left_right_symmetry: float = 100.0  # % (100 = parfait)
    symmetry_details: Dict[str, float] = field(default_factory=dict)
    
    # Stabilité
    com_stability: float = 0.0  # Variance centre de masse
    balance_score: float = 50.0  # Score équilibre 0-100
    
    # Fluidité
    movement_jerk: float = 0.0  # Dérivée 3e de position (saccades)
    smoothness_score: float = 50.0  # Score fluidité 0-100
    
    # Amplitude
    range_of_motion: Dict[str, float] = field(default_factory=dict)
    movement_amplitude: float = 0.0
    
    # Coordination
    inter_segment_coordination: float = 50.0  # Coordination entre segments
    timing_synchronization: float = 50.0  # Synchronisation temporelle
    
    # Spécifique football
    kicking_mechanics: Dict[str, float] = field(default_factory=dict)
    running_efficiency: float = 50.0
    balance_control: float = 50.0


@dataclass
class BiomechanicalIssue:
    """Problème biomécanique détecté"""
    type: str  # Type de problème
    severity: RiskLevel  # Gravité
    description: str  # Description
    affected_joints: List[str]  # Articulations affectées  
    recommendations: List[str]  # Recommandations
    exercises: List[str]  # Exercices suggérés
    confidence: float = 0.0  # Confiance détection


@dataclass
class BiomechanicalReport:
    """Rapport biomécanique complet"""
    player_id: int
    analysis_timestamp: float
    overall_score: float  # Score global 0-100
    quality_rating: MovementQuality
    
    # Métriques détaillées
    metrics: MovementMetrics
    
    # Problèmes détectés
    issues: List[BiomechanicalIssue] = field(default_factory=list)
    
    # Points forts
    strengths: List[str] = field(default_factory=list)
    
    # Recommandations
    improvement_areas: List[str] = field(default_factory=list)
    exercise_program: List[str] = field(default_factory=list)
    
    # Progression
    fatigue_indicators: Dict[str, float] = field(default_factory=dict)
    performance_trends: Dict[str, List[float]] = field(default_factory=dict)


class MovementAnalyzer:
    """Analyseur biomécanique avancé pour football"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialiser l'analyseur biomécanique
        
        Args:
            config: Configuration de l'analyseur
        """
        self.config = config or self._get_default_config()
        
        # Normes biomécaniques
        self.joint_normal_ranges = self._init_joint_ranges()
        self.football_specific_ranges = self._init_football_ranges()
        
        # Historiques pour analyse temporelle
        self.player_histories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=300))
        self.baseline_metrics: Dict[int, MovementMetrics] = {}
        
        # Détecteurs de problèmes
        self.issue_detectors = self._init_issue_detectors()
        
        # Cache de calculs
        self.calculation_cache = {}
        
        logger.info("MovementAnalyzer initialisé avec analyse biomécanique complète")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut"""
        return {
            'symmetry_threshold': 15.0,  # % écart max pour symétrie acceptable
            'stability_window': 30,      # Frames pour calcul stabilité
            'jerk_threshold': 5.0,       # Seuil jerk pour mouvement fluide
            'fatigue_detection': True,   # Activer détection fatigue
            'risk_assessment': True,     # Évaluation des risques
            'min_confidence': 0.7,       # Confiance minimale poses
            'enable_3d_analysis': True,  # Utiliser coordonnées 3D si disponibles
            'temporal_smoothing': True,  # Lissage temporel
        }
    
    def _init_joint_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Initialiser plages normales des angles articulaires"""
        return {
            # Genou (extension complète = 180°)
            'left_knee': (160, 180),
            'right_knee': (160, 180),
            
            # Hanche (position neutre debout)
            'left_hip': (160, 200),
            'right_hip': (160, 200),
            
            # Cheville (position neutre)
            'left_ankle': (85, 95),
            'right_ankle': (85, 95),
            
            # Épaule
            'left_shoulder': (160, 200),
            'right_shoulder': (160, 200),
            
            # Coude  
            'left_elbow': (160, 180),
            'right_elbow': (160, 180),
            
            # Tronc
            'spine_lean': (0, 15),        # Inclinaison avant acceptable
            'trunk_rotation': (0, 20),    # Rotation tronc
            
            # Spécifique football
            'kicking_preparation': (90, 140),  # Flexion genou frappe
            'hip_rotation_pass': (10, 45),     # Rotation hanche passe
        }
    
    def _init_football_ranges(self) -> Dict[str, Dict[str, float]]:
        """Plages spécifiques aux mouvements football"""
        return {
            'shooting': {
                'knee_flexion_min': 90,    # Flexion minimale genou
                'hip_rotation_max': 45,    # Rotation maximale hanche
                'trunk_lean_optimal': 15,  # Inclinaison optimale tronc
                'ankle_extension': 120,    # Extension cheville frappe
            },
            'passing': {
                'hip_rotation_min': 15,    # Rotation minimale hanche
                'knee_flexion': 160,       # Flexion genou passe
                'balance_score_min': 70,   # Score équilibre minimal
            },
            'running': {
                'knee_lift_max': 110,      # Levée genou maximale
                'trunk_lean_optimal': 8,   # Inclinaison course optimale
                'arm_swing_range': 60,     # Amplitude balancier bras
                'cadence_optimal': 180,    # Cadence optimale (pas/min)
            },
            'defending': {
                'balance_score_min': 80,   # Équilibre crucial défense
                'center_of_gravity_low': True,  # Centre gravité bas
                'reaction_readiness': 70,  # Score préparation réaction
            }
        }
    
    def _init_issue_detectors(self) -> Dict[str, callable]:
        """Initialiser détecteurs de problèmes"""
        return {
            'asymmetry': self._detect_asymmetry,
            'postural_imbalance': self._detect_postural_imbalance,
            'injury_risk': self._detect_injury_risk,
            'fatigue': self._detect_fatigue,
            'movement_dysfunction': self._detect_movement_dysfunction,
            'coordination_issues': self._detect_coordination_issues,
        }

    def analyze_movement_sequence(self, poses: List[Pose3D], 
                                movement_type: str = "general") -> BiomechanicalReport:
        """
        Analyser séquence de mouvements
        
        Args:
            poses: Séquence de poses 3D
            movement_type: Type de mouvement (shooting, passing, running, etc.)
            
        Returns:
            Rapport biomécanique complet
        """
        if not poses:
            raise ValueError("Aucune pose fournie pour l'analyse")
        
        player_id = poses[0].track_id
        logger.info(f"Analyse biomécanique joueur {player_id}, {len(poses)} poses, type: {movement_type}")
        
        start_time = time.time()
        
        # Filtrer poses de qualité insuffisante
        valid_poses = [p for p in poses if p.confidence >= self.config['min_confidence']]
        if len(valid_poses) < 3:
            logger.warning(f"Poses de qualité insuffisante: {len(valid_poses)}")
        
        # Calculer métriques biomécaniques
        metrics = self._calculate_comprehensive_metrics(valid_poses, movement_type)
        
        # Détecter problèmes
        issues = self._detect_all_issues(valid_poses, metrics)
        
        # Identifier points forts
        strengths = self._identify_strengths(metrics)
        
        # Générer recommandations
        improvement_areas, exercises = self._generate_recommendations(metrics, issues)
        
        # Calculer score global
        overall_score = self._calculate_overall_score(metrics, issues)
        quality_rating = self._determine_quality_rating(overall_score)
        
        # Analyser fatigue et tendances
        fatigue_indicators = self._analyze_fatigue(valid_poses, player_id)
        
        # Créer rapport
        report = BiomechanicalReport(
            player_id=player_id,
            analysis_timestamp=time.time(),
            overall_score=overall_score,
            quality_rating=quality_rating,
            metrics=metrics,
            issues=issues,
            strengths=strengths,
            improvement_areas=improvement_areas,
            exercise_program=exercises,
            fatigue_indicators=fatigue_indicators
        )
        
        # Mettre à jour historique
        self._update_player_history(player_id, report)
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Analyse terminée en {processing_time:.1f}ms, score: {overall_score:.1f}")
        
        return report

    def _calculate_comprehensive_metrics(self, poses: List[Pose3D], 
                                       movement_type: str) -> MovementMetrics:
        """Calculer métriques biomécaniques complètes"""
        
        metrics = MovementMetrics()
        
        # 1. Angles articulaires détaillés (15 articulations)
        metrics.joint_angles = self._calculate_all_joint_angles(poses)
        
        # 2. Analyse de symétrie gauche/droite
        metrics.left_right_symmetry, metrics.symmetry_details = self._analyze_symmetry(poses)
        
        # 3. Stabilité via centre de masse
        metrics.com_stability, metrics.balance_score = self._analyze_stability(poses)
        
        # 4. Fluidité mouvement (analyse du jerk)
        metrics.movement_jerk, metrics.smoothness_score = self._analyze_movement_smoothness(poses)
        
        # 5. Amplitude mouvement
        metrics.range_of_motion, metrics.movement_amplitude = self._analyze_range_of_motion(poses)
        
        # 6. Coordination inter-segments
        metrics.inter_segment_coordination, metrics.timing_synchronization = self._analyze_coordination(poses)
        
        # 7. Métriques spécifiques football
        metrics.kicking_mechanics = self._analyze_football_mechanics(poses, movement_type)
        metrics.running_efficiency = self._calculate_running_efficiency(poses)
        metrics.balance_control = self._analyze_balance_control(poses)
        
        return metrics

    def _calculate_all_joint_angles(self, poses: List[Pose3D]) -> Dict[str, ArticularAngle]:
        """Calculer tous les angles articulaires avec évaluation"""
        
        joint_angles = {}
        
        # Moyenner sur toutes les poses valides
        angle_sequences = defaultdict(list)
        
        for pose in poses:
            if pose.joint_angles:
                for joint_name, angle_value in pose.joint_angles.items():
                    if isinstance(angle_value, (int, float)) and not math.isnan(angle_value):
                        angle_sequences[joint_name].append(angle_value)
        
        # Créer ArticularAngle pour chaque articulation
        for joint_name, angles in angle_sequences.items():
            if not angles:
                continue
                
            mean_angle = float(np.mean(angles))
            std_angle = float(np.std(angles))
            
            # Obtenir plage normale
            normal_range = self.joint_normal_ranges.get(joint_name, (0, 180))
            
            # Évaluer si dans plage normale
            is_within_normal = normal_range[0] <= mean_angle <= normal_range[1]
            
            # Calculer niveau de risque
            risk_level = self._assess_joint_risk(joint_name, mean_angle, std_angle, normal_range)
            
            joint_angles[joint_name] = ArticularAngle(
                name=joint_name,
                value=mean_angle,
                normal_range=normal_range,
                risk_level=risk_level,
                is_within_normal=is_within_normal
            )
        
        # Calculer asymétries gauche/droite
        self._calculate_joint_asymmetries(joint_angles)
        
        return joint_angles

    def _calculate_joint_asymmetries(self, joint_angles: Dict[str, ArticularAngle]):
        """Calculer scores d'asymétrie entre côtés"""
        
        # Paires articulaires gauche/droite
        bilateral_pairs = [
            ('left_knee', 'right_knee'),
            ('left_hip', 'right_hip'),
            ('left_ankle', 'right_ankle'),
            ('left_shoulder', 'right_shoulder'),
            ('left_elbow', 'right_elbow')
        ]
        
        for left_joint, right_joint in bilateral_pairs:
            if left_joint in joint_angles and right_joint in joint_angles:
                left_angle = joint_angles[left_joint].value
                right_angle = joint_angles[right_joint].value
                
                # Calculer asymétrie (%)
                asymmetry = abs(left_angle - right_angle) / max(left_angle, right_angle) * 100
                
                joint_angles[left_joint].asymmetry_score = asymmetry
                joint_angles[right_joint].asymmetry_score = asymmetry

    def _assess_joint_risk(self, joint_name: str, angle: float, std: float, 
                          normal_range: Tuple[float, float]) -> RiskLevel:
        """Évaluer niveau de risque d'une articulation"""
        
        # Distance depuis plage normale
        if normal_range[0] <= angle <= normal_range[1]:
            distance_from_normal = 0
        else:
            distance_from_normal = min(abs(angle - normal_range[0]), abs(angle - normal_range[1]))
        
        # Variabilité (instabilité)
        high_variability = std > 10.0
        
        # Critères de risque
        if distance_from_normal > 30 or (distance_from_normal > 15 and high_variability):
            return RiskLevel.CRITICAL
        elif distance_from_normal > 20 or high_variability:
            return RiskLevel.HIGH
        elif distance_from_normal > 10:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    def _analyze_symmetry(self, poses: List[Pose3D]) -> Tuple[float, Dict[str, float]]:
        """Analyser symétrie gauche/droite"""
        
        symmetry_details = {}
        
        # Paires à analyser
        pairs = [
            ('left_knee', 'right_knee'),
            ('left_hip', 'right_hip'), 
            ('left_ankle', 'right_ankle'),
            ('left_shoulder', 'right_shoulder')
        ]
        
        pair_symmetries = []
        
        for left_joint, right_joint in pairs:
            left_angles = []
            right_angles = []
            
            for pose in poses:
                if pose.joint_angles:
                    if left_joint in pose.joint_angles and right_joint in pose.joint_angles:
                        left_angles.append(pose.joint_angles[left_joint])
                        right_angles.append(pose.joint_angles[right_joint])
            
            if left_angles and right_angles:
                # Corrélation entre côtés
                if SCIPY_AVAILABLE and len(left_angles) > 3:
                    correlation, _ = pearsonr(left_angles, right_angles)
                    symmetry_score = max(0, correlation * 100)
                else:
                    # Méthode simple: inverse de la différence moyenne
                    mean_diff = np.mean([abs(l - r) for l, r in zip(left_angles, right_angles)])
                    symmetry_score = max(0, 100 - mean_diff)
                
                symmetry_details[f"{left_joint}_{right_joint}"] = float(symmetry_score)
                pair_symmetries.append(symmetry_score)
        
        # Symétrie globale
        overall_symmetry = float(np.mean(pair_symmetries)) if pair_symmetries else 50.0
        
        return overall_symmetry, symmetry_details

    def _analyze_stability(self, poses: List[Pose3D]) -> Tuple[float, float]:
        """Analyser stabilité via centre de masse"""
        
        com_positions = []
        balance_scores = []
        
        for pose in poses:
            if pose.center_of_mass is not None:
                com_positions.append(pose.center_of_mass)
            
            if pose.joint_angles and 'balance_score' in pose.joint_angles:
                balance_scores.append(pose.joint_angles['balance_score'])
        
        # Stabilité COM (variance)
        if len(com_positions) > 3:
            com_array = np.array(com_positions)
            x_var = np.var(com_array[:, 0])
            y_var = np.var(com_array[:, 1])
            com_stability = float(x_var + y_var)
        else:
            com_stability = 0.0
        
        # Score d'équilibre moyen
        balance_score = float(np.mean(balance_scores)) if balance_scores else 50.0
        
        return com_stability, balance_score

    def _analyze_movement_smoothness(self, poses: List[Pose3D]) -> Tuple[float, float]:
        """Analyser fluidité mouvement via jerk"""
        
        if len(poses) < 5:
            return 0.0, 50.0
        
        # Extraire trajectoires des points clés
        trajectories = {
            'com': [],
            'left_knee': [],
            'right_knee': [],
            'left_ankle': [],
            'right_ankle': []
        }
        
        for pose in poses:
            if pose.center_of_mass is not None:
                trajectories['com'].append(pose.center_of_mass[:2])
            
            # Points articulaires
            for joint in ['left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
                if hasattr(pose, 'key_landmarks') and joint in pose.key_landmarks:
                    joint_pos = pose.keypoints[pose.key_landmarks[joint], :2]
                    trajectories[joint].append(joint_pos)
        
        jerk_values = []
        
        for trajectory_name, positions in trajectories.items():
            if len(positions) >= 5:
                jerk = self._calculate_trajectory_jerk(positions)
                jerk_values.append(jerk)
        
        # Jerk moyen
        mean_jerk = float(np.mean(jerk_values)) if jerk_values else 0.0
        
        # Score de fluidité (inverse du jerk)
        smoothness_score = max(0, min(100, 100 - mean_jerk * 10))
        
        return mean_jerk, float(smoothness_score)

    def _calculate_trajectory_jerk(self, positions: List[np.ndarray]) -> float:
        """Calculer jerk d'une trajectoire (dérivée 3e)"""
        
        positions = np.array(positions)
        
        if len(positions) < 4:
            return 0.0
        
        # Vitesse (dérivée 1e)
        velocity = np.diff(positions, axis=0)
        
        # Accélération (dérivée 2e)
        acceleration = np.diff(velocity, axis=0)
        
        # Jerk (dérivée 3e)
        jerk = np.diff(acceleration, axis=0)
        
        # Magnitude moyenne du jerk
        jerk_magnitudes = [np.linalg.norm(j) for j in jerk]
        
        return float(np.mean(jerk_magnitudes))

    def _analyze_range_of_motion(self, poses: List[Pose3D]) -> Tuple[Dict[str, float], float]:
        """Analyser amplitude de mouvement"""
        
        range_of_motion = {}
        joint_ranges = []
        
        # Collecter toutes les valeurs d'angles par articulation
        joint_values = defaultdict(list)
        
        for pose in poses:
            if pose.joint_angles:
                for joint_name, angle_value in pose.joint_angles.items():
                    if isinstance(angle_value, (int, float)) and not math.isnan(angle_value):
                        joint_values[joint_name].append(angle_value)
        
        # Calculer amplitude pour chaque articulation
        for joint_name, angles in joint_values.items():
            if len(angles) > 1:
                joint_range = float(max(angles) - min(angles))
                range_of_motion[joint_name] = joint_range
                joint_ranges.append(joint_range)
        
        # Amplitude globale
        movement_amplitude = float(np.mean(joint_ranges)) if joint_ranges else 0.0
        
        return range_of_motion, movement_amplitude

    def _analyze_coordination(self, poses: List[Pose3D]) -> Tuple[float, float]:
        """Analyser coordination inter-segments"""
        
        if len(poses) < 10:
            return 50.0, 50.0
        
        # Extraire séquences d'angles pour segments liés
        coordination_pairs = [
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee'),
            ('left_knee', 'left_ankle'),
            ('right_knee', 'right_ankle'),
            ('left_shoulder', 'left_elbow'),
            ('right_shoulder', 'right_elbow')
        ]
        
        coordination_scores = []
        timing_scores = []
        
        for joint1, joint2 in coordination_pairs:
            angles1 = []
            angles2 = []
            
            for pose in poses:
                if pose.joint_angles:
                    if joint1 in pose.joint_angles and joint2 in pose.joint_angles:
                        angles1.append(pose.joint_angles[joint1])
                        angles2.append(pose.joint_angles[joint2])
            
            if len(angles1) >= 10 and SCIPY_AVAILABLE:
                # Corrélation (coordination)
                correlation, _ = pearsonr(angles1, angles2)
                coordination_scores.append(abs(correlation) * 100)
                
                # Analyse de phase (timing)
                phase_sync = self._analyze_phase_synchronization(angles1, angles2)
                timing_scores.append(phase_sync)
        
        # Scores moyens
        inter_segment_coordination = float(np.mean(coordination_scores)) if coordination_scores else 50.0
        timing_synchronization = float(np.mean(timing_scores)) if timing_scores else 50.0
        
        return inter_segment_coordination, timing_synchronization

    def _analyze_phase_synchronization(self, signal1: List[float], signal2: List[float]) -> float:
        """Analyser synchronisation de phase entre deux signaux"""
        
        # Version simplifiée sans traitement de signal avancé
        # Analyser corrélation croisée à différents décalages
        
        signal1 = np.array(signal1)
        signal2 = np.array(signal2)
        
        max_correlation = 0
        
        # Tester décalages de -5 à +5 frames
        for lag in range(-5, 6):
            if lag < 0:
                s1 = signal1[-lag:]
                s2 = signal2[:lag]
            elif lag > 0:
                s1 = signal1[:-lag]
                s2 = signal2[lag:]
            else:
                s1 = signal1
                s2 = signal2
            
            if len(s1) > 5 and len(s2) > 5:
                if SCIPY_AVAILABLE:
                    corr, _ = pearsonr(s1, s2)
                    max_correlation = max(max_correlation, abs(corr))
        
        return float(max_correlation * 100)

    def _analyze_football_mechanics(self, poses: List[Pose3D], movement_type: str) -> Dict[str, float]:
        """Analyser mécaniques spécifiques football"""
        
        mechanics = {}
        
        if movement_type == "shooting":
            mechanics.update(self._analyze_shooting_mechanics(poses))
        elif movement_type == "passing":
            mechanics.update(self._analyze_passing_mechanics(poses))
        elif movement_type == "running":
            mechanics.update(self._analyze_running_mechanics(poses))
        elif movement_type == "defending":
            mechanics.update(self._analyze_defending_mechanics(poses))
        else:
            # Analyse générale
            mechanics.update(self._analyze_general_mechanics(poses))
        
        return mechanics

    def _analyze_shooting_mechanics(self, poses: List[Pose3D]) -> Dict[str, float]:
        """Analyser mécaniques de tir"""
        
        mechanics = {
            'knee_flexion_optimal': 0.0,
            'hip_rotation_power': 0.0,
            'trunk_stability': 0.0,
            'ankle_extension_power': 0.0,
            'follow_through_quality': 0.0
        }
        
        knee_flexions = []
        hip_rotations = []
        trunk_angles = []
        ankle_extensions = []
        
        for pose in poses:
            if pose.joint_angles:
                # Flexion genou jambe de frappe (minimum des deux)
                left_knee = pose.joint_angles.get('left_knee', 180)
                right_knee = pose.joint_angles.get('right_knee', 180)
                kicking_knee = min(left_knee, right_knee)  # Plus fléchi = jambe de frappe
                knee_flexions.append(kicking_knee)
                
                # Rotation hanches
                hip_rot = pose.joint_angles.get('trunk_rotation', 0)
                hip_rotations.append(hip_rot)
                
                # Stabilité tronc
                trunk_lean = pose.joint_angles.get('spine_lean', 0)
                trunk_angles.append(trunk_lean)
                
                # Extension cheville
                left_ankle = pose.joint_angles.get('left_ankle', 90)
                right_ankle = pose.joint_angles.get('right_ankle', 90)
                ankle_ext = max(left_ankle, right_ankle)  # Plus étendu = pied de frappe
                ankle_extensions.append(ankle_ext)
        
        if knee_flexions:
            # Évaluer par rapport aux plages optimales
            optimal_ranges = self.football_specific_ranges['shooting']
            
            # Flexion genou (plus faible = meilleur)
            min_knee_flex = min(knee_flexions)
            mechanics['knee_flexion_optimal'] = max(0, min(100, 
                100 - abs(min_knee_flex - optimal_ranges['knee_flexion_min'])))
            
            # Rotation hanche (puissance)
            max_hip_rot = max(hip_rotations) if hip_rotations else 0
            mechanics['hip_rotation_power'] = min(100, max_hip_rot / optimal_ranges['hip_rotation_max'] * 100)
            
            # Stabilité tronc (proche de optimal)
            mean_trunk = np.mean(trunk_angles)
            mechanics['trunk_stability'] = max(0, min(100,
                100 - abs(mean_trunk - optimal_ranges['trunk_lean_optimal']) * 5))
            
            # Extension cheville (puissance)
            max_ankle_ext = max(ankle_extensions) if ankle_extensions else 90
            mechanics['ankle_extension_power'] = min(100, max_ankle_ext / optimal_ranges['ankle_extension'] * 100)
            
            # Qualité du suivi (amplitude mouvement)
            movement_range = max(knee_flexions) - min(knee_flexions)
            mechanics['follow_through_quality'] = min(100, movement_range / 50 * 100)
        
        return mechanics

    def _analyze_passing_mechanics(self, poses: List[Pose3D]) -> Dict[str, float]:
        """Analyser mécaniques de passe"""
        
        mechanics = {
            'hip_rotation_control': 0.0,
            'knee_stability': 0.0,
            'balance_maintenance': 0.0,
            'precision_setup': 0.0
        }
        
        # Analyser stabilité et contrôle pour la précision
        hip_rotations = []
        knee_angles = []
        balance_scores = []
        
        for pose in poses:
            if pose.joint_angles:
                hip_rot = pose.joint_angles.get('trunk_rotation', 0)
                hip_rotations.append(hip_rot)
                
                left_knee = pose.joint_angles.get('left_knee', 180)
                right_knee = pose.joint_angles.get('right_knee', 180)
                knee_angles.extend([left_knee, right_knee])
                
                balance = pose.joint_angles.get('balance_score', 50)
                balance_scores.append(balance)
        
        if hip_rotations:
            optimal = self.football_specific_ranges['passing']
            
            # Contrôle rotation hanche
            hip_control = 100 - np.std(hip_rotations)  # Moins de variation = meilleur contrôle
            mechanics['hip_rotation_control'] = max(0, min(100, hip_control))
            
            # Stabilité genoux
            knee_stability = 100 - np.std(knee_angles) / 2  # Stabilité articulaire
            mechanics['knee_stability'] = max(0, min(100, knee_stability))
            
            # Maintien équilibre
            mean_balance = np.mean(balance_scores)
            mechanics['balance_maintenance'] = mean_balance
            
            # Setup précision (combinaison facteurs)
            mechanics['precision_setup'] = (
                mechanics['hip_rotation_control'] * 0.3 +
                mechanics['knee_stability'] * 0.3 +
                mechanics['balance_maintenance'] * 0.4
            )
        
        return mechanics

    def _analyze_general_mechanics(self, poses: List[Pose3D]) -> Dict[str, float]:
        """Analyse mécanique générale"""
        
        mechanics = {
            'overall_stability': 0.0,
            'movement_efficiency': 0.0,
            'postural_control': 0.0
        }
        
        balance_scores = []
        movement_smoothness = []
        postural_angles = []
        
        for pose in poses:
            if pose.joint_angles:
                balance = pose.joint_angles.get('balance_score', 50)
                balance_scores.append(balance)
                
                smoothness = pose.joint_angles.get('running_efficiency', 50)
                movement_smoothness.append(smoothness)
                
                spine_lean = pose.joint_angles.get('spine_lean', 0)
                postural_angles.append(spine_lean)
        
        if balance_scores:
            mechanics['overall_stability'] = float(np.mean(balance_scores))
            mechanics['movement_efficiency'] = float(np.mean(movement_smoothness))
            
            # Contrôle postural (faible variation = bon contrôle)
            postural_control = max(0, 100 - np.std(postural_angles) * 5)
            mechanics['postural_control'] = float(postural_control)
        
        return mechanics

    def _detect_all_issues(self, poses: List[Pose3D], metrics: MovementMetrics) -> List[BiomechanicalIssue]:
        """Détecter tous les problèmes biomécaniques"""
        
        all_issues = []
        
        # Exécuter tous les détecteurs
        for detector_name, detector_func in self.issue_detectors.items():
            try:
                issues = detector_func(poses, metrics)
                if issues:
                    all_issues.extend(issues)
            except Exception as e:
                logger.warning(f"Erreur détecteur {detector_name}: {e}")
        
        # Trier par sévérité
        severity_order = {
            RiskLevel.CRITICAL: 0,
            RiskLevel.HIGH: 1, 
            RiskLevel.MODERATE: 2,
            RiskLevel.LOW: 3
        }
        
        all_issues.sort(key=lambda x: severity_order.get(x.severity, 3))
        
        return all_issues

    def _detect_asymmetry(self, poses: List[Pose3D], metrics: MovementMetrics) -> List[BiomechanicalIssue]:
        """Détecter asymétries dangereuses"""
        
        issues = []
        
        if metrics.left_right_symmetry < (100 - self.config['symmetry_threshold']):
            severity = RiskLevel.MODERATE
            if metrics.left_right_symmetry < 70:
                severity = RiskLevel.HIGH
            elif metrics.left_right_symmetry < 50:
                severity = RiskLevel.CRITICAL
            
            # Identifier articulations les plus asymétriques
            asymmetric_joints = []
            for detail_name, symmetry_score in metrics.symmetry_details.items():
                if symmetry_score < 80:
                    asymmetric_joints.append(detail_name)
            
            issue = BiomechanicalIssue(
                type="asymmetry",
                severity=severity,
                description=f"Asymétrie significative détectée ({metrics.left_right_symmetry:.1f}%). "
                           f"Peut indiquer déséquilibre musculaire ou compensation.",
                affected_joints=asymmetric_joints,
                recommendations=[
                    "Évaluation approfondie par kinésithérapeute",
                    "Exercices de rééquilibrage musculaire",
                    "Travail unilatéral du côté faible",
                    "Correction technique des mouvements"
                ],
                exercises=[
                    "Squats unilatéraux",
                    "Fentes latérales",
                    "Travail proprioceptif sur une jambe",
                    "Renforcement spécifique côté faible"
                ],
                confidence=min(1.0, (100 - metrics.left_right_symmetry) / 50)
            )
            issues.append(issue)
        
        return issues

    def _detect_postural_imbalance(self, poses: List[Pose3D], metrics: MovementMetrics) -> List[BiomechanicalIssue]:
        """Détecter déséquilibres posturaux"""
        
        issues = []
        
        # Analyser inclinaison tronc excessive
        spine_angles = []
        for pose in poses:
            if pose.joint_angles and 'spine_lean' in pose.joint_angles:
                spine_angles.append(pose.joint_angles['spine_lean'])
        
        if spine_angles:
            mean_spine_lean = np.mean(spine_angles)
            spine_variability = np.std(spine_angles)
            
            # Inclinaison excessive
            if mean_spine_lean > 25:
                severity = RiskLevel.MODERATE if mean_spine_lean < 35 else RiskLevel.HIGH
                
                issue = BiomechanicalIssue(
                    type="postural_imbalance",
                    severity=severity,
                    description=f"Inclinaison tronc excessive ({mean_spine_lean:.1f}°). "
                               f"Risque de surcharge lombaire et déséquilibre.",
                    affected_joints=["spine", "lower_back", "hips"],
                    recommendations=[
                        "Renforcement musculature profonde",
                        "Étirements chaîne postérieure", 
                        "Correction posturale",
                        "Évaluation ergonomique"
                    ],
                    exercises=[
                        "Planche et variantes",
                        "Dead bug",
                        "Étirements hip flexors",
                        "Renforcement erecteurs spinaux"
                    ],
                    confidence=min(1.0, (mean_spine_lean - 15) / 30)
                )
                issues.append(issue)
            
            # Instabilité posturale
            if spine_variability > 15:
                issue = BiomechanicalIssue(
                    type="postural_instability",
                    severity=RiskLevel.MODERATE,
                    description=f"Instabilité posturale détectée (variation {spine_variability:.1f}°). "
                               f"Contrôle postural insuffisant.",
                    affected_joints=["core", "spine"],
                    recommendations=[
                        "Travail proprioceptif",
                        "Renforcement core",
                        "Exercices d'équilibre",
                        "Correction neurommusculaire"
                    ],
                    exercises=[
                        "Exercices instabilité (Swiss ball)",
                        "Équilibre dynamique",
                        "Yoga/Pilates",
                        "Proprioception bipodal/unipodal"
                    ],
                    confidence=min(1.0, spine_variability / 20)
                )
                issues.append(issue)
        
        return issues

    def _detect_injury_risk(self, poses: List[Pose3D], metrics: MovementMetrics) -> List[BiomechanicalIssue]:
        """Détecter risques de blessure"""
        
        issues = []
        
        # Analyser angles articulaires dangereux
        for joint_name, joint_angle in metrics.joint_angles.items():
            if joint_angle.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                
                # Déterminer type de risque
                risk_type = "hyperextension" if joint_angle.value > joint_angle.normal_range[1] else "hyperflexion"
                
                issue = BiomechanicalIssue(
                    type="injury_risk",
                    severity=joint_angle.risk_level,
                    description=f"Angle dangereux détecté: {joint_name} à {joint_angle.value:.1f}° "
                               f"(normal: {joint_angle.normal_range[0]}-{joint_angle.normal_range[1]}°). "
                               f"Risque de {risk_type}.",
                    affected_joints=[joint_name],
                    recommendations=[
                        "Modification technique immédiate",
                        "Évaluation médicale si douleur",
                        "Renforcement musculature protectrice",
                        "Amélioration mobilité articulaire"
                    ],
                    exercises=[
                        f"Étirements spécifiques {joint_name}",
                        f"Renforcement antagonistes {joint_name}",
                        "Mobilisation articulaire",
                        "Proprioception articulaire"
                    ],
                    confidence=0.9
                )
                issues.append(issue)
        
        # Détecter patron de mouvement à risque
        if metrics.balance_score < 40:
            issue = BiomechanicalIssue(
                type="fall_risk",
                severity=RiskLevel.HIGH,
                description=f"Score d'équilibre critique ({metrics.balance_score:.1f}/100). "
                           f"Risque élevé de chute et blessure.",
                affected_joints=["ankles", "knees", "hips"],
                recommendations=[
                    "Arrêt activité si fatigue",
                    "Travail équilibre urgent",
                    "Évaluation vestibulaire",
                    "Renforcement stabilisateurs"
                ],
                exercises=[
                    "Équilibre statique/dynamique",
                    "Proprioception intensive",
                    "Renforcement chevilles",
                    "Core stability"
                ],
                confidence=0.8
            )
            issues.append(issue)
        
        return issues

    def _detect_fatigue(self, poses: List[Pose3D], metrics: MovementMetrics) -> List[BiomechanicalIssue]:
        """Détecter signes de fatigue"""
        
        issues = []
        
        if not self.config['fatigue_detection']:
            return issues
        
        # Analyser dégradation progressive
        if len(poses) >= 30:
            # Diviser en segments temporels
            segment_size = len(poses) // 3
            early_poses = poses[:segment_size]
            late_poses = poses[-segment_size:]
            
            # Comparer métriques début vs fin
            early_balance = np.mean([p.joint_angles.get('balance_score', 50) 
                                   for p in early_poses if p.joint_angles])
            late_balance = np.mean([p.joint_angles.get('balance_score', 50) 
                                  for p in late_poses if p.joint_angles])
            
            balance_degradation = early_balance - late_balance
            
            if balance_degradation > 15:
                severity = RiskLevel.MODERATE if balance_degradation < 25 else RiskLevel.HIGH
                
                issue = BiomechanicalIssue(
                    type="fatigue",
                    severity=severity,
                    description=f"Dégradation équilibre détectée ({balance_degradation:.1f} points). "
                               f"Signes de fatigue neuromusculaire.",
                    affected_joints=["general"],
                    recommendations=[
                        "Repos et récupération",
                        "Réduction intensité",
                        "Hydratation et nutrition",
                        "Évaluation charge d'entraînement"
                    ],
                    exercises=[
                        "Récupération active",
                        "Étirements doux",
                        "Relaxation musculaire",
                        "Techniques récupération"
                    ],
                    confidence=min(1.0, balance_degradation / 30)
                )
                issues.append(issue)
        
        return issues

    def _detect_movement_dysfunction(self, poses: List[Pose3D], metrics: MovementMetrics) -> List[BiomechanicalIssue]:
        """Détecter dysfonctions mouvement"""
        
        issues = []
        
        # Mouvement saccadé (jerk élevé)
        if metrics.movement_jerk > self.config['jerk_threshold']:
            issue = BiomechanicalIssue(
                type="movement_dysfunction",
                severity=RiskLevel.MODERATE,
                description=f"Mouvement saccadé détecté (jerk: {metrics.movement_jerk:.2f}). "
                           f"Manque de fluidité et contrôle moteur.",
                affected_joints=["general"],
                recommendations=[
                    "Travail contrôle moteur",
                    "Réduction vitesse mouvement",
                    "Exercices coordination",
                    "Renforcement stabilisateurs"
                ],
                exercises=[
                    "Mouvements lents contrôlés",
                    "Proprioception",
                    "Coordination oeil-main",
                    "Exercices rythmiques"
                ],
                confidence=min(1.0, metrics.movement_jerk / 10)
            )
            issues.append(issue)
        
        # Amplitude mouvement réduite
        if metrics.movement_amplitude < 20:
            issue = BiomechanicalIssue(
                type="reduced_mobility",
                severity=RiskLevel.MODERATE,
                description=f"Amplitude mouvement réduite ({metrics.movement_amplitude:.1f}°). "
                           f"Possible restriction mobilité.",
                affected_joints=list(metrics.range_of_motion.keys()),
                recommendations=[
                    "Évaluation mobilité articulaire",
                    "Étirements spécifiques",
                    "Mobilisation passive",
                    "Échauffement prolongé"
                ],
                exercises=[
                    "Étirements dynamiques",
                    "Mobilisation articulaire",
                    "Yoga/stretching",
                    "Amplitude active"
                ],
                confidence=0.7
            )
            issues.append(issue)
        
        return issues

    def _detect_coordination_issues(self, poses: List[Pose3D], metrics: MovementMetrics) -> List[BiomechanicalIssue]:
        """Détecter problèmes de coordination"""
        
        issues = []
        
        # Coordination inter-segments faible
        if metrics.inter_segment_coordination < 60:
            severity = RiskLevel.MODERATE if metrics.inter_segment_coordination > 40 else RiskLevel.HIGH
            
            issue = BiomechanicalIssue(
                type="coordination_deficit",
                severity=severity,
                description=f"Coordination inter-segments déficiente ({metrics.inter_segment_coordination:.1f}%). "
                           f"Chaîne cinétique non optimale.",
                affected_joints=["kinetic_chain"],
                recommendations=[
                    "Travail coordination spécifique",
                    "Exercices chaîne cinétique",
                    "Rééducation neuromoeurtrice",
                    "Pratique gestes techniques"
                ],
                exercises=[
                    "Exercices coordination complexes",
                    "Mouvements multi-articulaires",
                    "Entraînement réactivité",
                    "Patterns moteurs"
                ],
                confidence=min(1.0, (80 - metrics.inter_segment_coordination) / 40)
            )
            issues.append(issue)
        
        # Synchronisation temporelle mauvaise
        if metrics.timing_synchronization < 50:
            issue = BiomechanicalIssue(
                type="timing_dysfunction",
                severity=RiskLevel.MODERATE,
                description=f"Synchronisation temporelle déficiente ({metrics.timing_synchronization:.1f}%). "
                           f"Timing musculaire non optimal.",
                affected_joints=["neuromuscular"],
                recommendations=[
                    "Travail timing musculaire",
                    "Exercices plyométriques",
                    "Rééducation sensori-motrice",
                    "Feedback temps réel"
                ],
                exercises=[
                    "Exercices réactivité",
                    "Plyométrie contrôlée",
                    "Travail rythme",
                    "Coordination tempo"
                ],
                confidence=0.6
            )
            issues.append(issue)
        
        return issues

    def _identify_strengths(self, metrics: MovementMetrics) -> List[str]:
        """Identifier points forts"""
        
        strengths = []
        
        # Excellente symétrie
        if metrics.left_right_symmetry > 90:
            strengths.append("Excellente symétrie gauche/droite maintenue")
        
        # Bon équilibre
        if metrics.balance_score > 80:
            strengths.append("Contrôle d'équilibre remarquable")
        
        # Mouvement fluide
        if metrics.smoothness_score > 80:
            strengths.append("Fluidité de mouvement excellente")
        
        # Bonne coordination
        if metrics.inter_segment_coordination > 80:
            strengths.append("Coordination inter-segments optimale")
        
        # Stabilité centre de masse
        if metrics.com_stability < 0.5:
            strengths.append("Stabilité du centre de masse remarquable")
        
        # Amplitude mouvement appropriée
        if 30 <= metrics.movement_amplitude <= 80:
            strengths.append("Amplitude de mouvement dans plages optimales")
        
        # Angles articulaires normaux
        normal_joints = [name for name, angle in metrics.joint_angles.items() 
                        if angle.is_within_normal]
        if len(normal_joints) > len(metrics.joint_angles) * 0.8:
            strengths.append("Majorité des angles articulaires dans normes")
        
        return strengths

    def _generate_recommendations(self, metrics: MovementMetrics, 
                                issues: List[BiomechanicalIssue]) -> Tuple[List[str], List[str]]:
        """Générer recommandations et exercices"""
        
        improvement_areas = []
        exercises = []
        
        # Recommandations basées sur les problèmes
        for issue in issues:
            improvement_areas.extend(issue.recommendations)
            exercises.extend(issue.exercises)
        
        # Recommandations générales basées sur métriques
        if metrics.left_right_symmetry < 85:
            improvement_areas.append("Améliorer symétrie corporelle")
            exercises.extend(["Travail unilatéral", "Renforcement compensateur"])
        
        if metrics.balance_score < 70:
            improvement_areas.append("Renforcer contrôle d'équilibre")
            exercises.extend(["Proprioception", "Équilibre dynamique"])
        
        if metrics.smoothness_score < 70:
            improvement_areas.append("Améliorer fluidité mouvement")
            exercises.extend(["Mouvements lents contrôlés", "Coordination"])
        
        if metrics.inter_segment_coordination < 70:
            improvement_areas.append("Optimiser coordination inter-segments")
            exercises.extend(["Chaîne cinétique", "Mouvements fonctionnels"])
        
        # Supprimer doublons
        improvement_areas = list(set(improvement_areas))
        exercises = list(set(exercises))
        
        return improvement_areas, exercises

    def _calculate_overall_score(self, metrics: MovementMetrics, 
                               issues: List[BiomechanicalIssue]) -> float:
        """Calculer score global 0-100"""
        
        # Score de base depuis métriques
        base_score = (
            metrics.left_right_symmetry * 0.2 +
            metrics.balance_score * 0.2 +
            metrics.smoothness_score * 0.15 +
            metrics.inter_segment_coordination * 0.15 +
            metrics.running_efficiency * 0.1 +
            (100 - min(100, metrics.com_stability * 20)) * 0.1 +
            min(100, metrics.movement_amplitude) * 0.1
        )
        
        # Pénalités pour problèmes
        penalty = 0
        for issue in issues:
            if issue.severity == RiskLevel.CRITICAL:
                penalty += 25
            elif issue.severity == RiskLevel.HIGH:
                penalty += 15
            elif issue.severity == RiskLevel.MODERATE:
                penalty += 8
            elif issue.severity == RiskLevel.LOW:
                penalty += 3
        
        # Score final
        final_score = max(0, min(100, base_score - penalty))
        
        return float(final_score)

    def _determine_quality_rating(self, score: float) -> MovementQuality:
        """Déterminer évaluation qualitative"""
        
        if score >= 90:
            return MovementQuality.EXCELLENT
        elif score >= 75:
            return MovementQuality.GOOD
        elif score >= 60:
            return MovementQuality.AVERAGE
        elif score >= 40:
            return MovementQuality.POOR
        else:
            return MovementQuality.CRITICAL

    def _analyze_fatigue(self, poses: List[Pose3D], player_id: int) -> Dict[str, float]:
        """Analyser indicateurs de fatigue"""
        
        fatigue_indicators = {}
        
        if len(poses) < 20:
            return fatigue_indicators
        
        # Analyser dégradation temporelle
        thirds = len(poses) // 3
        segments = [
            poses[:thirds],           # Début
            poses[thirds:2*thirds],   # Milieu  
            poses[2*thirds:]          # Fin
        ]
        
        segment_scores = []
        
        for segment in segments:
            # Calculer score moyen du segment
            balance_scores = []
            smoothness_scores = []
            
            for pose in segment:
                if pose.joint_angles:
                    balance_scores.append(pose.joint_angles.get('balance_score', 50))
                    smoothness_scores.append(pose.joint_angles.get('running_efficiency', 50))
            
            if balance_scores:
                segment_score = (np.mean(balance_scores) + np.mean(smoothness_scores)) / 2
                segment_scores.append(segment_score)
        
        if len(segment_scores) == 3:
            # Tendance fatigue
            fatigue_trend = segment_scores[0] - segment_scores[2]
            fatigue_indicators['performance_decline'] = max(0, fatigue_trend)
            
            # Variabilité (instabilité)
            fatigue_indicators['movement_variability'] = float(np.std(segment_scores))
            
            # Score fatigue global
            fatigue_score = min(100, fatigue_trend + np.std(segment_scores) * 2)
            fatigue_indicators['fatigue_level'] = max(0, fatigue_score)
        
        return fatigue_indicators

    def _update_player_history(self, player_id: int, report: BiomechanicalReport):
        """Mettre à jour historique joueur"""
        
        self.player_histories[player_id].append({
            'timestamp': report.analysis_timestamp,
            'overall_score': report.overall_score,
            'quality_rating': report.quality_rating.value,
            'symmetry': report.metrics.left_right_symmetry,
            'balance': report.metrics.balance_score,
            'issues_count': len(report.issues),
            'critical_issues': len([i for i in report.issues if i.severity == RiskLevel.CRITICAL])
        })

    def get_player_progression(self, player_id: int) -> Dict[str, Any]:
        """Obtenir progression d'un joueur"""
        
        if player_id not in self.player_histories:
            return {"error": "Aucun historique pour ce joueur"}
        
        history = list(self.player_histories[player_id])
        
        if len(history) < 2:
            return {"error": "Historique insuffisant pour analyse progression"}
        
        # Tendances
        scores = [h['overall_score'] for h in history]
        symmetry = [h['symmetry'] for h in history]
        balance = [h['balance'] for h in history]
        
        return {
            'total_analyses': len(history),
            'score_trend': float(np.polyfit(range(len(scores)), scores, 1)[0]),
            'current_score': float(scores[-1]),
            'best_score': float(max(scores)),
            'worst_score': float(min(scores)),
            'symmetry_trend': float(np.polyfit(range(len(symmetry)), symmetry, 1)[0]),
            'balance_trend': float(np.polyfit(range(len(balance)), balance, 1)[0]),
            'recent_issues': history[-1]['issues_count'],
            'critical_issues_trend': [h['critical_issues'] for h in history[-5:]]  # 5 dernières
        }

    def export_analysis_report(self, report: BiomechanicalReport, 
                             output_path: str, format: str = "json") -> bool:
        """Exporter rapport d'analyse"""
        
        try:
            export_data = {
                'player_id': report.player_id,
                'analysis_timestamp': report.analysis_timestamp,
                'overall_score': report.overall_score,
                'quality_rating': report.quality_rating.value,
                'metrics': {
                    'joint_angles': {name: {
                        'value': angle.value,
                        'normal_range': angle.normal_range,
                        'asymmetry_score': angle.asymmetry_score,
                        'risk_level': angle.risk_level.value,
                        'is_within_normal': angle.is_within_normal
                    } for name, angle in report.metrics.joint_angles.items()},
                    'symmetry': {
                        'left_right_symmetry': report.metrics.left_right_symmetry,
                        'details': report.metrics.symmetry_details
                    },
                    'stability': {
                        'com_stability': report.metrics.com_stability,
                        'balance_score': report.metrics.balance_score
                    },
                    'smoothness': {
                        'movement_jerk': report.metrics.movement_jerk,
                        'smoothness_score': report.metrics.smoothness_score
                    },
                    'range_of_motion': report.metrics.range_of_motion,
                    'coordination': {
                        'inter_segment': report.metrics.inter_segment_coordination,
                        'timing_sync': report.metrics.timing_synchronization
                    },
                    'football_mechanics': report.metrics.kicking_mechanics
                },
                'issues': [{
                    'type': issue.type,
                    'severity': issue.severity.value,
                    'description': issue.description,
                    'affected_joints': issue.affected_joints,
                    'recommendations': issue.recommendations,
                    'exercises': issue.exercises,
                    'confidence': issue.confidence
                } for issue in report.issues],
                'strengths': report.strengths,
                'improvement_areas': report.improvement_areas,
                'exercise_program': report.exercise_program,
                'fatigue_indicators': report.fatigue_indicators
            }
            
            if format.lower() == "json":
                import json
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            logger.info(f"Rapport exporté vers {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur export rapport: {e}")
            return False

    def visualize_3d_skeleton(self, pose: Pose3D, output_path: str = None) -> Optional[np.ndarray]:
        """Visualisation 3D skeleton (optionnelle)"""
        
        if not pose.world_landmarks:
            logger.warning("Coordonnées 3D non disponibles pour visualisation")
            return None
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Créer figure 3D
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Points 3D
            landmarks = pose.world_landmarks
            x = landmarks[:, 0]
            y = landmarks[:, 1] 
            z = landmarks[:, 2]
            
            # Dessiner points
            ax.scatter(x, y, z, c=pose.keypoints[:, 3], cmap='viridis', s=50)
            
            # Connexions principales
            connections = [
                (11, 12), (11, 23), (12, 24), (23, 24),  # Torse
                (23, 25), (25, 27), (24, 26), (26, 28),  # Jambes
                (11, 13), (13, 15), (12, 14), (14, 16)   # Bras
            ]
            
            for start_idx, end_idx in connections:
                if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                    pose.keypoints[start_idx, 3] > 0.5 and pose.keypoints[end_idx, 3] > 0.5):
                    
                    ax.plot([x[start_idx], x[end_idx]], 
                           [y[start_idx], y[end_idx]], 
                           [z[start_idx], z[end_idx]], 'b-', linewidth=2)
            
            # Configuration axes
            ax.set_xlabel('X')
            ax.set_ylabel('Y') 
            ax.set_zlabel('Z')
            ax.set_title(f'Skeleton 3D - Track {pose.track_id} - Frame {pose.frame_number}')
            
            # Sauvegarder si demandé
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualisation 3D sauvée: {output_path}")
            
            # Convertir en array pour retour
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            
            return buf
            
        except ImportError:
            logger.warning("Matplotlib non disponible pour visualisation 3D")
            return None
        except Exception as e:
            logger.error(f"Erreur visualisation 3D: {e}")
            return None


# Fonctions utilitaires
def create_movement_analyzer(config: Optional[Dict[str, Any]] = None) -> MovementAnalyzer:
    """Créer instance d'analyseur biomécanique"""
    return MovementAnalyzer(config)


def analyze_player_movement(poses: List[Pose3D], movement_type: str = "general", 
                          config: Optional[Dict[str, Any]] = None) -> BiomechanicalReport:
    """Fonction utilitaire pour analyser mouvement joueur"""
    
    analyzer = MovementAnalyzer(config)
    return analyzer.analyze_movement_sequence(poses, movement_type)


if __name__ == "__main__":
    # Test de l'analyseur
    print("Test MovementAnalyzer")
    
    analyzer = MovementAnalyzer()
    
    # Créer poses de test
    test_poses = []
    for i in range(30):
        # Simuler pose avec angles aléatoires
        pose = Pose3D(
            keypoints=np.random.rand(33, 4),
            track_id=1,
            frame_number=i
        )
        
        # Ajouter angles articulaires simulés
        pose.joint_angles = {
            'left_knee': 160 + np.random.normal(0, 5),
            'right_knee': 165 + np.random.normal(0, 5),
            'left_hip': 180 + np.random.normal(0, 8),
            'right_hip': 175 + np.random.normal(0, 8),
            'spine_lean': 10 + np.random.normal(0, 3),
            'balance_score': 75 + np.random.normal(0, 10)
        }
        
        # Centre de masse simulé
        pose.center_of_mass = np.random.rand(3)
        
        test_poses.append(pose)
    
    # Analyser
    report = analyzer.analyze_movement_sequence(test_poses, "running")
    
    print(f"Score global: {report.overall_score:.1f}")
    print(f"Qualité: {report.quality_rating.value}")
    print(f"Problèmes détectés: {len(report.issues)}")
    print(f"Points forts: {len(report.strengths)}")
    
    # Afficher problèmes critiques
    critical_issues = [i for i in report.issues if i.severity == RiskLevel.CRITICAL]
    if critical_issues:
        print(f"\nProblèmes critiques:")
        for issue in critical_issues:
            print(f"- {issue.description}")
    
    print(f"\nPoints forts identifiés:")
    for strength in report.strengths:
        print(f"- {strength}")