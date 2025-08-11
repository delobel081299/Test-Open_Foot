"""
Module d'analyse décisionnelle tactique
Évaluation des décisions prises par les joueurs en contexte
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
import cv2
from scipy.spatial import distance, Voronoi
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import math


class DecisionType(Enum):
    """Types de décisions tactiques"""
    PASS = "pass"
    DRIBBLE = "dribble"
    SHOT = "shot"
    HOLD = "hold"
    CLEAR = "clear"
    CROSS = "cross"


class ZoneType(Enum):
    """Zones du terrain pour contextualisation"""
    DEFENSIVE_THIRD = "defensive_third"
    MIDDLE_THIRD = "middle_third"
    ATTACKING_THIRD = "attacking_third"
    PENALTY_AREA = "penalty_area"
    DANGER_ZONE = "danger_zone"


@dataclass
class DecisionContext:
    """Contexte complet d'une décision"""
    player_position: Tuple[float, float]
    teammate_positions: Dict[int, Tuple[float, float]]
    opponent_positions: Dict[int, Tuple[float, float]]
    ball_position: Tuple[float, float]
    goal_position: Tuple[float, float]
    pressure_level: float  # 0-1
    space_available: float  # mètres carrés
    zone: ZoneType
    time_on_ball: float  # secondes
    game_state: Dict[str, Any]  # score, temps, etc.


@dataclass
class DecisionOption:
    """Option de décision disponible"""
    decision_type: DecisionType
    target: Optional[Union[int, Tuple[float, float]]]  # ID joueur ou position
    success_probability: float
    danger_created: float  # xT (expected threat)
    risk_level: float
    reward_potential: float
    execution_difficulty: float
    is_progressive: bool
    alternative_rank: int


@dataclass
class DecisionAnalysis:
    """Analyse complète d'une décision"""
    actual_decision: DecisionType
    decision_quality: float  # 0-100
    best_option: DecisionOption
    all_options: List[DecisionOption]
    xDecision_score: float  # Expected decision value
    context_summary: str
    improvements: List[str]
    visualization_data: Dict[str, Any]


class DecisionAnalyzer:
    """Analyseur de décisions tactiques en temps réel"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 pitch_dimensions: Tuple[float, float] = (105, 68)):
        self.pitch_length, self.pitch_width = pitch_dimensions
        
        # Charger ou initialiser le modèle ML
        if model_path and Path(model_path).exists():
            self.ml_model = joblib.load(model_path)
            self.scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
        else:
            self.ml_model = self._initialize_default_model()
            self.scaler = StandardScaler()
        
        # Paramètres d'analyse
        self.pressure_threshold = 3.0  # mètres pour considérer la pression
        self.space_grid_size = 1.0  # mètres pour la grille spatiale
        
        # Poids pour l'évaluation
        self.decision_weights = {
            'success_probability': 0.25,
            'danger_created': 0.30,
            'risk_level': -0.15,
            'reward_potential': 0.20,
            'execution_difficulty': -0.10
        }
        
        # Cache pour optimisation
        self.voronoi_cache = {}
        
    def analyze_decision(self, context: DecisionContext, 
                        actual_decision: DecisionType,
                        decision_outcome: Optional[Dict] = None) -> DecisionAnalysis:
        """
        Analyse une décision prise par un joueur
        
        Args:
            context: Contexte complet de la décision
            actual_decision: Décision réellement prise
            decision_outcome: Résultat de la décision si disponible
            
        Returns:
            Analyse complète avec scoring et alternatives
        """
        # Analyser toutes les options disponibles
        all_options = self._evaluate_all_options(context)
        
        # Trier par score
        all_options.sort(key=lambda x: self._calculate_option_score(x), reverse=True)
        
        # Assigner les rangs
        for i, option in enumerate(all_options):
            option.alternative_rank = i + 1
        
        # Trouver la meilleure option
        best_option = all_options[0] if all_options else None
        
        # Calculer la qualité de la décision actuelle
        actual_option = next((opt for opt in all_options 
                            if opt.decision_type == actual_decision), None)
        
        if actual_option:
            decision_quality = self._calculate_decision_quality(
                actual_option, best_option, all_options
            )
        else:
            decision_quality = 30.0  # Décision non reconnue
        
        # Calculer le xDecision score
        xDecision_score = self._calculate_xDecision(context, all_options)
        
        # Générer le résumé contextuel
        context_summary = self._generate_context_summary(context)
        
        # Identifier les améliorations
        improvements = self._identify_improvements(
            actual_decision, actual_option, best_option, context
        )
        
        # Préparer les données de visualisation
        viz_data = self._prepare_visualization_data(
            context, all_options, actual_decision
        )
        
        return DecisionAnalysis(
            actual_decision=actual_decision,
            decision_quality=decision_quality,
            best_option=best_option,
            all_options=all_options,
            xDecision_score=xDecision_score,
            context_summary=context_summary,
            improvements=improvements,
            visualization_data=viz_data
        )
    
    def _evaluate_all_options(self, context: DecisionContext) -> List[DecisionOption]:
        """Évalue toutes les options de décision disponibles"""
        options = []
        
        # Évaluer les options de passe
        pass_options = self._evaluate_pass_options(context)
        options.extend(pass_options)
        
        # Évaluer les options de dribble
        dribble_options = self._evaluate_dribble_options(context)
        options.extend(dribble_options)
        
        # Évaluer les options de tir
        shot_options = self._evaluate_shot_options(context)
        options.extend(shot_options)
        
        # Autres options contextuelles
        if context.zone == ZoneType.DEFENSIVE_THIRD:
            clear_option = self._evaluate_clear_option(context)
            if clear_option:
                options.append(clear_option)
        
        if context.zone in [ZoneType.ATTACKING_THIRD, ZoneType.DANGER_ZONE]:
            cross_options = self._evaluate_cross_options(context)
            options.extend(cross_options)
        
        # Option de conservation
        hold_option = self._evaluate_hold_option(context)
        if hold_option:
            options.append(hold_option)
        
        return options
    
    def _evaluate_pass_options(self, context: DecisionContext) -> List[DecisionOption]:
        """Évalue toutes les options de passe disponibles"""
        pass_options = []
        
        for teammate_id, teammate_pos in context.teammate_positions.items():
            # Vérifier si la ligne de passe est ouverte
            pass_line_clear = self._is_pass_line_clear(
                context.player_position, teammate_pos, 
                context.opponent_positions
            )
            
            if not pass_line_clear:
                continue
            
            # Calculer les métriques de la passe
            pass_distance = distance.euclidean(context.player_position, teammate_pos)
            
            # Probabilité de succès
            success_prob = self._calculate_pass_success_probability(
                pass_distance, context.pressure_level, pass_line_clear
            )
            
            # Danger créé (xT - expected threat)
            danger_created = self._calculate_expected_threat(
                teammate_pos, context.goal_position
            )
            
            # Progressivité
            is_progressive = self._is_progressive_pass(
                context.player_position, teammate_pos, context.goal_position
            )
            
            # Niveau de risque
            risk_level = self._calculate_pass_risk(
                context, teammate_pos, pass_distance
            )
            
            # Potentiel de récompense
            reward_potential = self._calculate_pass_reward(
                teammate_pos, context, is_progressive
            )
            
            # Difficulté d'exécution
            execution_difficulty = self._calculate_pass_difficulty(
                pass_distance, context.pressure_level, 
                self._calculate_pass_angle(context.player_position, teammate_pos)
            )
            
            option = DecisionOption(
                decision_type=DecisionType.PASS,
                target=teammate_id,
                success_probability=success_prob,
                danger_created=danger_created,
                risk_level=risk_level,
                reward_potential=reward_potential,
                execution_difficulty=execution_difficulty,
                is_progressive=is_progressive,
                alternative_rank=0  # Sera assigné plus tard
            )
            
            pass_options.append(option)
        
        return pass_options
    
    def _evaluate_dribble_options(self, context: DecisionContext) -> List[DecisionOption]:
        """Évalue les options de dribble"""
        dribble_options = []
        
        # Directions de dribble possibles (8 directions)
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        dribble_distance = 5.0  # mètres
        
        for angle in angles:
            # Position cible du dribble
            target_x = context.player_position[0] + dribble_distance * np.cos(angle)
            target_y = context.player_position[1] + dribble_distance * np.sin(angle)
            target_pos = (target_x, target_y)
            
            # Vérifier si la position est sur le terrain
            if not self._is_position_on_pitch(target_pos):
                continue
            
            # Espace disponible dans cette direction
            space = self._calculate_space_in_direction(
                context.player_position, angle, context.opponent_positions
            )
            
            if space < 2.0:  # Pas assez d'espace
                continue
            
            # Nombre d'adversaires à battre
            opponents_to_beat = self._count_opponents_in_path(
                context.player_position, target_pos, context.opponent_positions
            )
            
            # Probabilité de succès
            success_prob = self._calculate_dribble_success_probability(
                space, opponents_to_beat, context.pressure_level
            )
            
            # Danger créé
            danger_created = self._calculate_expected_threat(
                target_pos, context.goal_position
            )
            
            # Progressivité
            is_progressive = self._is_progressive_move(
                context.player_position, target_pos, context.goal_position
            )
            
            # Support des coéquipiers
            teammate_support = self._calculate_teammate_support(
                target_pos, context.teammate_positions
            )
            
            # Risque et récompense
            risk_level = self._calculate_dribble_risk(
                opponents_to_beat, context.zone, teammate_support
            )
            
            reward_potential = self._calculate_dribble_reward(
                target_pos, context, danger_created
            )
            
            # Difficulté
            execution_difficulty = self._calculate_dribble_difficulty(
                opponents_to_beat, space, context.pressure_level
            )
            
            option = DecisionOption(
                decision_type=DecisionType.DRIBBLE,
                target=target_pos,
                success_probability=success_prob,
                danger_created=danger_created,
                risk_level=risk_level,
                reward_potential=reward_potential,
                execution_difficulty=execution_difficulty,
                is_progressive=is_progressive,
                alternative_rank=0
            )
            
            dribble_options.append(option)
        
        return dribble_options
    
    def _evaluate_shot_options(self, context: DecisionContext) -> List[DecisionOption]:
        """Évalue les options de tir"""
        shot_options = []
        
        # Distance et angle au but
        shot_distance = distance.euclidean(context.player_position, context.goal_position)
        shot_angle = self._calculate_shot_angle(context.player_position, context.goal_position)
        
        # Ne considérer le tir que si dans une zone raisonnable
        if shot_distance > 30:  # Au-delà de 30m, très peu probable
            return shot_options
        
        # Pression défensive sur le tireur
        shooting_pressure = self._calculate_shooting_pressure(
            context.player_position, context.opponent_positions
        )
        
        # Position du gardien (supposée)
        gk_position = self._estimate_goalkeeper_position(
            context.goal_position, context.ball_position
        )
        
        # Probabilité de marquer (xG)
        xG = self._calculate_xG(
            shot_distance, shot_angle, shooting_pressure, gk_position
        )
        
        # Vérifier si des coéquipiers sont mieux placés
        better_positioned = self._find_better_positioned_teammates(
            context, xG
        )
        
        # Risque de contre-attaque si échec
        counter_risk = self._calculate_counter_attack_risk(
            context.player_position, context.zone
        )
        
        # Créer l'option de tir
        option = DecisionOption(
            decision_type=DecisionType.SHOT,
            target=context.goal_position,
            success_probability=xG,
            danger_created=xG,  # Pour un tir, danger = xG
            risk_level=counter_risk,
            reward_potential=xG * 3,  # Un but vaut beaucoup
            execution_difficulty=self._calculate_shot_difficulty(
                shot_distance, shot_angle, shooting_pressure
            ),
            is_progressive=True,  # Un tir est toujours progressif
            alternative_rank=0
        )
        
        # Pénaliser si des coéquipiers sont mieux placés
        if better_positioned:
            option.decision_quality_modifier = -0.2
        
        shot_options.append(option)
        
        return shot_options
    
    def _evaluate_cross_options(self, context: DecisionContext) -> List[DecisionOption]:
        """Évalue les options de centre"""
        cross_options = []
        
        # Identifier la zone de centre (près des lignes de touche)
        if abs(context.player_position[1] - self.pitch_width/2) < self.pitch_width/3:
            return cross_options  # Pas assez près de la ligne
        
        # Trouver les cibles potentielles dans la surface
        targets_in_box = self._find_targets_in_penalty_area(
            context.teammate_positions, context.goal_position
        )
        
        for target_id, target_pos in targets_in_box.items():
            # Espace pour le receveur
            target_space = self._calculate_space_around_player(
                target_pos, context.opponent_positions
            )
            
            # Qualité de la position de centre
            cross_quality = self._evaluate_cross_position_quality(
                context.player_position, target_pos, context.goal_position
            )
            
            # Probabilité de succès
            success_prob = self._calculate_cross_success_probability(
                cross_quality, target_space, context.pressure_level
            )
            
            # Danger créé
            danger_created = self._calculate_cross_danger(
                target_pos, context.goal_position, target_space
            )
            
            option = DecisionOption(
                decision_type=DecisionType.CROSS,
                target=target_id,
                success_probability=success_prob,
                danger_created=danger_created,
                risk_level=0.3,  # Les centres sont relativement peu risqués
                reward_potential=danger_created * 2,
                execution_difficulty=0.5,
                is_progressive=True,
                alternative_rank=0
            )
            
            cross_options.append(option)
        
        return cross_options
    
    def _evaluate_clear_option(self, context: DecisionContext) -> Optional[DecisionOption]:
        """Évalue l'option de dégagement défensif"""
        if context.zone != ZoneType.DEFENSIVE_THIRD:
            return None
        
        # Pression immédiate
        immediate_pressure = self._calculate_immediate_pressure(
            context.player_position, context.opponent_positions
        )
        
        if immediate_pressure < 0.7:  # Pas assez de pression pour justifier
            return None
        
        return DecisionOption(
            decision_type=DecisionType.CLEAR,
            target=None,
            success_probability=0.95,  # Très probable de réussir
            danger_created=0.0,
            risk_level=0.1,  # Peu risqué
            reward_potential=0.2,  # Soulage la pression
            execution_difficulty=0.2,
            is_progressive=False,
            alternative_rank=0
        )
    
    def _evaluate_hold_option(self, context: DecisionContext) -> Optional[DecisionOption]:
        """Évalue l'option de conservation du ballon"""
        # Utile quand peu d'options et peu de pression
        if context.pressure_level > 0.6:
            return None
        
        return DecisionOption(
            decision_type=DecisionType.HOLD,
            target=None,
            success_probability=0.9 - context.pressure_level,
            danger_created=0.0,
            risk_level=context.pressure_level * 0.5,
            reward_potential=0.1,  # Permet de temporiser
            execution_difficulty=0.3,
            is_progressive=False,
            alternative_rank=0
        )
    
    def _calculate_option_score(self, option: DecisionOption) -> float:
        """Calcule le score global d'une option"""
        score = 0
        
        # Appliquer les poids
        score += self.decision_weights['success_probability'] * option.success_probability
        score += self.decision_weights['danger_created'] * option.danger_created
        score += self.decision_weights['risk_level'] * option.risk_level
        score += self.decision_weights['reward_potential'] * option.reward_potential
        score += self.decision_weights['execution_difficulty'] * option.execution_difficulty
        
        # Bonus pour progressivité
        if option.is_progressive:
            score *= 1.1
        
        # Normaliser entre 0 et 1
        return max(0, min(1, score))
    
    def _calculate_decision_quality(self, actual_option: DecisionOption,
                                  best_option: DecisionOption,
                                  all_options: List[DecisionOption]) -> float:
        """Calcule la qualité de la décision prise"""
        if not actual_option:
            return 30.0
        
        actual_score = self._calculate_option_score(actual_option)
        best_score = self._calculate_option_score(best_option)
        
        # Score de base
        base_score = (actual_score / best_score) * 100 if best_score > 0 else 50
        
        # Ajustements contextuels
        if actual_option.alternative_rank == 1:
            base_score = min(100, base_score * 1.1)  # Bonus pour meilleure décision
        elif actual_option.alternative_rank <= 3:
            base_score *= 0.95  # Légère pénalité
        else:
            base_score *= 0.85  # Pénalité plus importante
        
        # Pénalité si option très risquée choisie sans nécessité
        if actual_option.risk_level > 0.7 and len([o for o in all_options if o.risk_level < 0.4]) > 2:
            base_score *= 0.9
        
        return max(0, min(100, base_score))
    
    def _calculate_xDecision(self, context: DecisionContext,
                           options: List[DecisionOption]) -> float:
        """Calcule le score xDecision basé sur le modèle ML"""
        if not options:
            return 0.0
        
        # Extraire les features pour le modèle
        features = self._extract_ml_features(context, options)
        
        # Prédire avec le modèle
        if hasattr(self, 'ml_model') and self.ml_model:
            try:
                features_scaled = self.scaler.transform([features])
                xDecision = self.ml_model.predict(features_scaled)[0]
                return max(0, min(1, xDecision))
            except:
                pass
        
        # Fallback : moyenne pondérée des meilleures options
        top_options = sorted(options, key=self._calculate_option_score, reverse=True)[:3]
        if top_options:
            return np.mean([self._calculate_option_score(opt) for opt in top_options])
        
        return 0.5
    
    def _extract_ml_features(self, context: DecisionContext,
                           options: List[DecisionOption]) -> List[float]:
        """Extrait les features pour le modèle ML"""
        features = []
        
        # Features contextuelles
        features.append(context.pressure_level)
        features.append(context.space_available)
        features.append(float(context.zone.value == ZoneType.ATTACKING_THIRD.value))
        features.append(float(context.zone.value == ZoneType.DEFENSIVE_THIRD.value))
        features.append(context.time_on_ball)
        
        # Position relative au but
        dist_to_goal = distance.euclidean(context.player_position, context.goal_position)
        features.append(dist_to_goal / self.pitch_length)  # Normaliser
        
        # Nombre d'options par type
        features.append(len([o for o in options if o.decision_type == DecisionType.PASS]))
        features.append(len([o for o in options if o.decision_type == DecisionType.DRIBBLE]))
        features.append(len([o for o in options if o.decision_type == DecisionType.SHOT]))
        
        # Qualité des meilleures options
        for decision_type in [DecisionType.PASS, DecisionType.DRIBBLE, DecisionType.SHOT]:
            type_options = [o for o in options if o.decision_type == decision_type]
            if type_options:
                best = max(type_options, key=self._calculate_option_score)
                features.append(best.success_probability)
                features.append(best.danger_created)
            else:
                features.extend([0.0, 0.0])
        
        # Densité de joueurs autour
        nearby_teammates = len([p for p in context.teammate_positions.values() 
                              if distance.euclidean(p, context.player_position) < 10])
        nearby_opponents = len([p for p in context.opponent_positions.values() 
                              if distance.euclidean(p, context.player_position) < 10])
        features.append(nearby_teammates)
        features.append(nearby_opponents)
        
        return features
    
    def _is_pass_line_clear(self, from_pos: Tuple[float, float],
                           to_pos: Tuple[float, float],
                           opponents: Dict[int, Tuple[float, float]]) -> bool:
        """Vérifie si une ligne de passe est dégagée"""
        pass_vector = np.array(to_pos) - np.array(from_pos)
        pass_length = np.linalg.norm(pass_vector)
        
        if pass_length == 0:
            return False
        
        pass_direction = pass_vector / pass_length
        
        for opp_pos in opponents.values():
            # Projeter la position de l'adversaire sur la ligne de passe
            opp_vector = np.array(opp_pos) - np.array(from_pos)
            projection_length = np.dot(opp_vector, pass_direction)
            
            # Si la projection est en dehors du segment de passe
            if projection_length < 0 or projection_length > pass_length:
                continue
            
            # Point le plus proche sur la ligne de passe
            closest_point = np.array(from_pos) + projection_length * pass_direction
            
            # Distance de l'adversaire à la ligne
            dist_to_line = np.linalg.norm(np.array(opp_pos) - closest_point)
            
            # Si trop proche, la ligne n'est pas claire
            if dist_to_line < 2.0:  # 2 mètres de marge
                return False
        
        return True
    
    def _calculate_pass_success_probability(self, distance: float,
                                          pressure: float,
                                          line_clear: bool) -> float:
        """Calcule la probabilité de succès d'une passe"""
        # Probabilité de base selon la distance
        if distance < 5:
            base_prob = 0.95
        elif distance < 15:
            base_prob = 0.85
        elif distance < 30:
            base_prob = 0.70
        else:
            base_prob = 0.50
        
        # Ajuster pour la pression
        base_prob *= (1 - pressure * 0.3)
        
        # Ajuster si la ligne n'est pas claire
        if not line_clear:
            base_prob *= 0.7
        
        return max(0.1, min(0.99, base_prob))
    
    def _calculate_expected_threat(self, position: Tuple[float, float],
                                 goal_pos: Tuple[float, float]) -> float:
        """Calcule l'expected threat (xT) d'une position"""
        # Distance au but
        dist_to_goal = distance.euclidean(position, goal_pos)
        
        # xT basique basé sur la distance
        if dist_to_goal < 10:
            xT = 0.8
        elif dist_to_goal < 20:
            xT = 0.5
        elif dist_to_goal < 30:
            xT = 0.3
        elif dist_to_goal < 50:
            xT = 0.1
        else:
            xT = 0.05
        
        # Ajuster pour la centralité
        center_y = self.pitch_width / 2
        lateral_distance = abs(position[1] - center_y)
        centrality_factor = 1 - (lateral_distance / (self.pitch_width / 2)) * 0.3
        
        xT *= centrality_factor
        
        return max(0, min(1, xT))
    
    def _is_progressive_pass(self, from_pos: Tuple[float, float],
                           to_pos: Tuple[float, float],
                           goal_pos: Tuple[float, float]) -> bool:
        """Détermine si une passe est progressive"""
        # Distance au but avant et après
        dist_before = distance.euclidean(from_pos, goal_pos)
        dist_after = distance.euclidean(to_pos, goal_pos)
        
        # Progressive si réduit la distance d'au moins 10m ou 25%
        return (dist_before - dist_after) > 10 or (dist_after / dist_before) < 0.75
    
    def _calculate_space_in_direction(self, position: Tuple[float, float],
                                    angle: float,
                                    opponents: Dict[int, Tuple[float, float]]) -> float:
        """Calcule l'espace disponible dans une direction"""
        # Vecteur de direction
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        min_distance = float('inf')
        
        for opp_pos in opponents.values():
            # Vecteur vers l'adversaire
            to_opponent = np.array(opp_pos) - np.array(position)
            
            # Projection sur la direction
            projection = np.dot(to_opponent, direction)
            
            # Si l'adversaire est derrière, ignorer
            if projection < 0:
                continue
            
            # Distance perpendiculaire à la direction
            perp_distance = np.linalg.norm(to_opponent - projection * direction)
            
            # Si dans le couloir (largeur 3m)
            if perp_distance < 1.5:
                min_distance = min(min_distance, projection)
        
        # Limiter à la distance du bord du terrain
        edge_distance = self._distance_to_edge_in_direction(position, angle)
        
        return min(min_distance, edge_distance)
    
    def _calculate_xG(self, distance: float, angle: float,
                     pressure: float, gk_position: Tuple[float, float]) -> float:
        """Calcule les expected goals (xG) pour un tir"""
        # xG de base selon distance et angle
        if distance < 6:  # Dans les 6 mètres
            base_xG = 0.5
        elif distance < 12:
            base_xG = 0.3
        elif distance < 18:
            base_xG = 0.15
        elif distance < 25:
            base_xG = 0.08
        else:
            base_xG = 0.03
        
        # Ajuster pour l'angle
        angle_factor = 1 - (abs(angle - 90) / 90) * 0.5
        base_xG *= angle_factor
        
        # Ajuster pour la pression
        base_xG *= (1 - pressure * 0.4)
        
        return max(0.01, min(0.99, base_xG))
    
    def _identify_improvements(self, actual_decision: DecisionType,
                             actual_option: Optional[DecisionOption],
                             best_option: DecisionOption,
                             context: DecisionContext) -> List[str]:
        """Identifie les points d'amélioration"""
        improvements = []
        
        if not actual_option:
            improvements.append("Option non reconnue - considérer les alternatives disponibles")
            return improvements
        
        # Si pas la meilleure décision
        if actual_option.alternative_rank > 1:
            improvements.append(
                f"Meilleure option disponible : {best_option.decision_type.value} "
                f"(+{(best_option.success_probability - actual_option.success_probability)*100:.0f}% succès)"
            )
        
        # Si décision trop risquée
        if actual_option.risk_level > 0.7:
            safer_options = [o for o in context.all_options if o.risk_level < 0.4]
            if safer_options:
                improvements.append("Considérer des options moins risquées disponibles")
        
        # Si passe non progressive alors que possible
        if (actual_decision == DecisionType.PASS and 
            not actual_option.is_progressive):
            prog_passes = [o for o in context.all_options 
                          if o.decision_type == DecisionType.PASS and o.is_progressive]
            if prog_passes:
                improvements.append("Passes progressives disponibles non exploitées")
        
        # Si tir avec coéquipiers mieux placés
        if actual_decision == DecisionType.SHOT:
            better_shots = [o for o in context.all_options 
                           if o.decision_type == DecisionType.PASS and 
                           o.danger_created > actual_option.danger_created]
            if better_shots:
                improvements.append("Coéquipiers mieux placés pour finir l'action")
        
        return improvements
    
    def visualize_decision_analysis(self, analysis: DecisionAnalysis,
                                   save_path: Optional[str] = None) -> None:
        """Visualise l'analyse de décision"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Contexte spatial avec options
        ax1 = axes[0, 0]
        self._plot_spatial_context(ax1, analysis)
        ax1.set_title("Contexte spatial et options disponibles")
        
        # 2. Comparaison des options
        ax2 = axes[0, 1]
        self._plot_options_comparison(ax2, analysis)
        ax2.set_title("Comparaison des options")
        
        # 3. Radar de la décision
        ax3 = axes[1, 0]
        self._plot_decision_radar(ax3, analysis)
        ax3.set_title("Profil de la décision")
        
        # 4. Timeline et contexte
        ax4 = axes[1, 1]
        self._plot_decision_context(ax4, analysis)
        ax4.set_title("Contexte et recommandations")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def _plot_spatial_context(self, ax, analysis: DecisionAnalysis):
        """Trace le contexte spatial avec les options"""
        # Dessiner le terrain
        self._draw_pitch_section(ax, analysis.visualization_data)
        
        # Position du joueur
        player_pos = analysis.visualization_data['player_position']
        ax.scatter(player_pos[0], player_pos[1], s=300, c='yellow', 
                  edgecolors='black', linewidth=2, zorder=5)
        
        # Coéquipiers
        for teammate_pos in analysis.visualization_data['teammate_positions'].values():
            ax.scatter(teammate_pos[0], teammate_pos[1], s=200, c='blue',
                      edgecolors='white', linewidth=1, zorder=4)
        
        # Adversaires
        for opp_pos in analysis.visualization_data['opponent_positions'].values():
            ax.scatter(opp_pos[0], opp_pos[1], s=200, c='red',
                      edgecolors='white', linewidth=1, zorder=4)
        
        # Visualiser les options principales
        for i, option in enumerate(analysis.all_options[:5]):  # Top 5 options
            if option.decision_type == DecisionType.PASS:
                # Ligne de passe
                target_pos = analysis.visualization_data['teammate_positions'][option.target]
                ax.plot([player_pos[0], target_pos[0]], 
                       [player_pos[1], target_pos[1]],
                       'g--', alpha=0.5 + 0.1*i, linewidth=2-0.2*i)
                
            elif option.decision_type == DecisionType.DRIBBLE:
                # Flèche de dribble
                if isinstance(option.target, tuple):
                    ax.arrow(player_pos[0], player_pos[1],
                           option.target[0] - player_pos[0],
                           option.target[1] - player_pos[1],
                           head_width=1, head_length=0.5,
                           fc='orange', ec='orange', alpha=0.5 + 0.1*i)
                
            elif option.decision_type == DecisionType.SHOT:
                # Ligne de tir
                goal_pos = analysis.visualization_data['goal_position']
                ax.plot([player_pos[0], goal_pos[0]], 
                       [player_pos[1], goal_pos[1]],
                       'r-', alpha=0.5 + 0.1*i, linewidth=3-0.3*i)
        
        # Marquer la décision réelle
        if analysis.actual_decision == DecisionType.PASS:
            actual_target = next((o.target for o in analysis.all_options 
                                if o.decision_type == analysis.actual_decision), None)
            if actual_target and actual_target in analysis.visualization_data['teammate_positions']:
                target_pos = analysis.visualization_data['teammate_positions'][actual_target]
                ax.plot([player_pos[0], target_pos[0]], 
                       [player_pos[1], target_pos[1]],
                       'k-', linewidth=3, label='Décision réelle')
        
        ax.legend()
    
    def _plot_options_comparison(self, ax, analysis: DecisionAnalysis):
        """Compare les différentes options disponibles"""
        # Préparer les données
        options = analysis.all_options[:8]  # Top 8 options
        
        if not options:
            ax.text(0.5, 0.5, 'Aucune option disponible',
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Données pour le graphique
        labels = []
        success_probs = []
        danger_values = []
        risk_levels = []
        
        for i, opt in enumerate(options):
            label = f"{opt.decision_type.value}"
            if opt.decision_type == DecisionType.PASS and opt.target:
                label += f" #{opt.target}"
            labels.append(f"{i+1}. {label}")
            
            success_probs.append(opt.success_probability)
            danger_values.append(opt.danger_created)
            risk_levels.append(opt.risk_level)
        
        x = np.arange(len(labels))
        width = 0.25
        
        # Barres groupées
        bars1 = ax.bar(x - width, success_probs, width, label='Succès', color='green', alpha=0.7)
        bars2 = ax.bar(x, danger_values, width, label='Danger créé', color='orange', alpha=0.7)
        bars3 = ax.bar(x + width, risk_levels, width, label='Risque', color='red', alpha=0.7)
        
        # Marquer la décision réelle
        actual_idx = next((i for i, opt in enumerate(options) 
                          if opt.decision_type == analysis.actual_decision), None)
        if actual_idx is not None:
            ax.axvspan(actual_idx - 0.5, actual_idx + 0.5, alpha=0.2, color='yellow')
        
        ax.set_ylabel('Valeur')
        ax.set_xlabel('Options')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Score de qualité
        ax.text(0.02, 0.98, f"Qualité décision: {analysis.decision_quality:.0f}%",
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_decision_radar(self, ax, analysis: DecisionAnalysis):
        """Trace le radar de la décision"""
        # Trouver l'option actuelle
        actual_option = next((opt for opt in analysis.all_options 
                            if opt.decision_type == analysis.actual_decision), None)
        
        if not actual_option:
            ax.text(0.5, 0.5, 'Décision non analysée',
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Catégories
        categories = ['Succès', 'Danger', 'Risque\n(inversé)', 
                     'Récompense', 'Facilité\n(inversé)', 'Progressivité']
        
        # Valeurs pour la décision actuelle
        actual_values = [
            actual_option.success_probability,
            actual_option.danger_created,
            1 - actual_option.risk_level,  # Inverser pour que plus = mieux
            actual_option.reward_potential,
            1 - actual_option.execution_difficulty,  # Inverser
            1.0 if actual_option.is_progressive else 0.0
        ]
        
        # Valeurs pour la meilleure option
        best_values = [
            analysis.best_option.success_probability,
            analysis.best_option.danger_created,
            1 - analysis.best_option.risk_level,
            analysis.best_option.reward_potential,
            1 - analysis.best_option.execution_difficulty,
            1.0 if analysis.best_option.is_progressive else 0.0
        ]
        
        # Créer le radar
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        actual_values += actual_values[:1]
        best_values += best_values[:1]
        angles += angles[:1]
        
        ax.plot(angles, actual_values, 'o-', linewidth=2, label='Décision actuelle', color='blue')
        ax.fill(angles, actual_values, alpha=0.25, color='blue')
        
        ax.plot(angles, best_values, 'o--', linewidth=2, label='Meilleure option', color='green')
        ax.fill(angles, best_values, alpha=0.15, color='green')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc='upper right')
    
    def _plot_decision_context(self, ax, analysis: DecisionAnalysis):
        """Affiche le contexte et les recommandations"""
        ax.axis('off')
        
        # Titre
        ax.text(0.5, 0.95, 'Analyse de Décision', 
               transform=ax.transAxes, ha='center', va='top',
               fontsize=16, fontweight='bold')
        
        # Informations principales
        info_text = f"Décision: {analysis.actual_decision.value}\n"
        info_text += f"Qualité: {analysis.decision_quality:.0f}%\n"
        info_text += f"xDecision Score: {analysis.xDecision_score:.2f}\n\n"
        info_text += f"Contexte: {analysis.context_summary}\n\n"
        
        ax.text(0.05, 0.80, info_text, transform=ax.transAxes,
               va='top', fontsize=12)
        
        # Meilleure alternative
        if analysis.best_option and analysis.best_option.decision_type != analysis.actual_decision:
            alt_text = "Meilleure alternative:\n"
            alt_text += f"• {analysis.best_option.decision_type.value}\n"
            alt_text += f"• Succès: {analysis.best_option.success_probability:.0%}\n"
            alt_text += f"• Danger: {analysis.best_option.danger_created:.2f}\n"
            
            ax.text(0.05, 0.45, alt_text, transform=ax.transAxes,
                   va='top', fontsize=11, 
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Points d'amélioration
        if analysis.improvements:
            imp_text = "Points d'amélioration:\n"
            for i, imp in enumerate(analysis.improvements[:3]):
                imp_text += f"{i+1}. {imp}\n"
            
            ax.text(0.05, 0.20, imp_text, transform=ax.transAxes,
                   va='top', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    def _draw_pitch_section(self, ax, viz_data):
        """Dessine une section du terrain"""
        # Déterminer la zone à afficher
        player_pos = viz_data['player_position']
        
        # Zone de 40x40m autour du joueur
        x_min = max(0, player_pos[0] - 20)
        x_max = min(self.pitch_length, player_pos[0] + 20)
        y_min = max(0, player_pos[1] - 20)
        y_max = min(self.pitch_width, player_pos[1] + 20)
        
        # Terrain
        pitch = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                linewidth=2, edgecolor='black', facecolor='green', alpha=0.3)
        ax.add_patch(pitch)
        
        # Lignes du terrain si visibles
        if x_min <= self.pitch_length/2 <= x_max:
            ax.plot([self.pitch_length/2, self.pitch_length/2], [y_min, y_max],
                   'white', linewidth=2)
        
        # Surface de réparation si visible
        if x_max >= self.pitch_length - 16.5:
            penalty_area = patches.Rectangle(
                (self.pitch_length - 16.5, self.pitch_width/2 - 20.15),
                16.5, 40.3, linewidth=2, edgecolor='white', facecolor='none'
            )
            ax.add_patch(penalty_area)
        
        ax.set_xlim(x_min - 2, x_max + 2)
        ax.set_ylim(y_min - 2, y_max + 2)
        ax.set_aspect('equal')
    
    def _initialize_default_model(self):
        """Initialise un modèle par défaut si aucun n'est fourni"""
        # Modèle simple pour démonstration
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def _generate_context_summary(self, context: DecisionContext) -> str:
        """Génère un résumé textuel du contexte"""
        summary = f"Zone: {context.zone.value.replace('_', ' ')}, "
        summary += f"Pression: {'haute' if context.pressure_level > 0.7 else 'moyenne' if context.pressure_level > 0.4 else 'faible'}, "
        summary += f"Espace: {context.space_available:.0f}m², "
        summary += f"Temps sur ballon: {context.time_on_ball:.1f}s"
        
        return summary
    
    # Méthodes utilitaires supplémentaires
    
    def _is_position_on_pitch(self, pos: Tuple[float, float]) -> bool:
        """Vérifie si une position est sur le terrain"""
        return (0 <= pos[0] <= self.pitch_length and 
                0 <= pos[1] <= self.pitch_width)
    
    def _count_opponents_in_path(self, from_pos: Tuple[float, float],
                                to_pos: Tuple[float, float],
                                opponents: Dict[int, Tuple[float, float]]) -> int:
        """Compte les adversaires sur un chemin"""
        count = 0
        path_vector = np.array(to_pos) - np.array(from_pos)
        path_length = np.linalg.norm(path_vector)
        
        if path_length == 0:
            return 0
        
        path_direction = path_vector / path_length
        
        for opp_pos in opponents.values():
            # Distance au chemin
            to_opp = np.array(opp_pos) - np.array(from_pos)
            projection = np.dot(to_opp, path_direction)
            
            if 0 <= projection <= path_length:
                perp_dist = np.linalg.norm(to_opp - projection * path_direction)
                if perp_dist < 2.0:  # Dans le chemin
                    count += 1
        
        return count
    
    def _is_progressive_move(self, from_pos: Tuple[float, float],
                           to_pos: Tuple[float, float],
                           goal_pos: Tuple[float, float]) -> bool:
        """Vérifie si un mouvement est progressif"""
        dist_before = distance.euclidean(from_pos, goal_pos)
        dist_after = distance.euclidean(to_pos, goal_pos)
        
        return (dist_before - dist_after) > 5  # Au moins 5m de progression
    
    def _calculate_teammate_support(self, position: Tuple[float, float],
                                  teammates: Dict[int, Tuple[float, float]]) -> float:
        """Calcule le support des coéquipiers autour d'une position"""
        nearby_teammates = 0
        
        for teammate_pos in teammates.values():
            if distance.euclidean(position, teammate_pos) < 15:
                nearby_teammates += 1
        
        # Normaliser
        return min(1.0, nearby_teammates / 3)  # 3 coéquipiers = support max
    
    def _calculate_pass_angle(self, from_pos: Tuple[float, float],
                            to_pos: Tuple[float, float]) -> float:
        """Calcule l'angle d'une passe"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        return math.degrees(math.atan2(dy, dx))
    
    def _calculate_dribble_success_probability(self, space: float,
                                             opponents: int,
                                             pressure: float) -> float:
        """Calcule la probabilité de succès d'un dribble"""
        # Base selon l'espace
        if space > 10:
            base_prob = 0.85
        elif space > 5:
            base_prob = 0.70
        elif space > 3:
            base_prob = 0.50
        else:
            base_prob = 0.30
        
        # Réduire pour chaque adversaire
        base_prob *= (0.7 ** opponents)
        
        # Ajuster pour la pression
        base_prob *= (1 - pressure * 0.4)
        
        return max(0.1, min(0.9, base_prob))
    
    def _calculate_shot_angle(self, pos: Tuple[float, float],
                            goal_pos: Tuple[float, float]) -> float:
        """Calcule l'angle de tir au but"""
        # Positions des poteaux
        post1 = (goal_pos[0], goal_pos[1] - 3.66)  # Largeur but / 2
        post2 = (goal_pos[0], goal_pos[1] + 3.66)
        
        # Angles vers chaque poteau
        angle1 = math.atan2(post1[1] - pos[1], post1[0] - pos[0])
        angle2 = math.atan2(post2[1] - pos[1], post2[0] - pos[0])
        
        # Angle de tir en degrés
        shot_angle = abs(math.degrees(angle2 - angle1))
        
        return shot_angle
    
    def _calculate_space_around_player(self, position: Tuple[float, float],
                                     opponents: Dict[int, Tuple[float, float]]) -> float:
        """Calcule l'espace libre autour d'un joueur"""
        if not opponents:
            return 100.0
        
        # Distance au plus proche adversaire
        min_dist = min(distance.euclidean(position, opp_pos) 
                      for opp_pos in opponents.values())
        
        # Convertir en espace (approximation circulaire)
        space = math.pi * (min_dist ** 2)
        
        return min(100.0, space)
    
    def _distance_to_edge_in_direction(self, position: Tuple[float, float],
                                      angle: float) -> float:
        """Calcule la distance jusqu'au bord du terrain dans une direction"""
        # Vecteur direction
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Distances aux bords
        distances = []
        
        # Bord droit
        if dx > 0:
            t = (self.pitch_length - position[0]) / dx
            if t > 0:
                y = position[1] + t * dy
                if 0 <= y <= self.pitch_width:
                    distances.append(t)
        
        # Bord gauche
        elif dx < 0:
            t = -position[0] / dx
            if t > 0:
                y = position[1] + t * dy
                if 0 <= y <= self.pitch_width:
                    distances.append(t)
        
        # Bord haut
        if dy > 0:
            t = (self.pitch_width - position[1]) / dy
            if t > 0:
                x = position[0] + t * dx
                if 0 <= x <= self.pitch_length:
                    distances.append(t)
        
        # Bord bas
        elif dy < 0:
            t = -position[1] / dy
            if t > 0:
                x = position[0] + t * dx
                if 0 <= x <= self.pitch_length:
                    distances.append(t)
        
        return min(distances) if distances else 0
    
    def _prepare_visualization_data(self, context: DecisionContext,
                                  options: List[DecisionOption],
                                  actual_decision: DecisionType) -> Dict:
        """Prépare les données pour la visualisation"""
        return {
            'player_position': context.player_position,
            'teammate_positions': context.teammate_positions,
            'opponent_positions': context.opponent_positions,
            'ball_position': context.ball_position,
            'goal_position': context.goal_position,
            'options': options[:10],  # Top 10 pour visualisation
            'actual_decision': actual_decision,
            'pressure_level': context.pressure_level,
            'zone': context.zone
        }
    
    def _calculate_pass_risk(self, context: DecisionContext,
                           target_pos: Tuple[float, float],
                           distance: float) -> float:
        """Calcule le risque d'une passe"""
        # Risque de base selon la distance
        base_risk = min(1.0, distance / 40)
        
        # Augmenter si dans zone défensive
        if context.zone == ZoneType.DEFENSIVE_THIRD:
            base_risk *= 1.5
        
        # Augmenter selon la pression
        base_risk *= (1 + context.pressure_level * 0.5)
        
        return min(1.0, base_risk)
    
    def _calculate_pass_reward(self, target_pos: Tuple[float, float],
                             context: DecisionContext,
                             is_progressive: bool) -> float:
        """Calcule le potentiel de récompense d'une passe"""
        # xT de la position cible
        xT = self._calculate_expected_threat(target_pos, context.goal_position)
        
        # Bonus si progressive
        if is_progressive:
            xT *= 1.3
        
        # Bonus si crée une occasion
        if context.zone == ZoneType.DANGER_ZONE:
            xT *= 1.5
        
        return min(1.0, xT)
    
    def _calculate_pass_difficulty(self, distance: float, pressure: float,
                                 angle: float) -> float:
        """Calcule la difficulté d'exécution d'une passe"""
        # Difficulté selon distance
        dist_difficulty = min(1.0, distance / 40)
        
        # Difficulté selon angle (passes en retrait plus faciles)
        angle_difficulty = 0.3 if abs(angle) > 120 else 0.5
        
        # Combiner avec pression
        total_difficulty = (dist_difficulty + angle_difficulty) / 2 * (1 + pressure * 0.5)
        
        return min(1.0, total_difficulty)
    
    def _calculate_dribble_risk(self, opponents: int, zone: ZoneType,
                              support: float) -> float:
        """Calcule le risque d'un dribble"""
        # Risque de base selon adversaires
        base_risk = min(1.0, opponents * 0.3)
        
        # Augmenter en zone défensive
        if zone == ZoneType.DEFENSIVE_THIRD:
            base_risk *= 1.5
        
        # Réduire si bon support
        base_risk *= (1 - support * 0.3)
        
        return min(1.0, base_risk)
    
    def _calculate_dribble_reward(self, target_pos: Tuple[float, float],
                                context: DecisionContext,
                                danger: float) -> float:
        """Calcule la récompense potentielle d'un dribble"""
        # Base sur le danger créé
        reward = danger
        
        # Bonus si élimine des adversaires
        eliminated = self._count_opponents_in_path(
            context.player_position, target_pos, context.opponent_positions
        )
        reward *= (1 + eliminated * 0.2)
        
        # Bonus en zone offensive
        if context.zone == ZoneType.ATTACKING_THIRD:
            reward *= 1.3
        
        return min(1.0, reward)
    
    def _calculate_dribble_difficulty(self, opponents: int, space: float,
                                    pressure: float) -> float:
        """Calcule la difficulté d'un dribble"""
        # Base selon adversaires
        opp_difficulty = min(1.0, opponents * 0.4)
        
        # Selon l'espace
        space_difficulty = 1.0 - min(1.0, space / 10)
        
        # Combiner
        total = (opp_difficulty + space_difficulty) / 2 * (1 + pressure * 0.3)
        
        return min(1.0, total)
    
    def _calculate_shooting_pressure(self, pos: Tuple[float, float],
                                   opponents: Dict[int, Tuple[float, float]]) -> float:
        """Calcule la pression sur un tireur"""
        nearby_opponents = 0
        very_close_opponents = 0
        
        for opp_pos in opponents.values():
            dist = distance.euclidean(pos, opp_pos)
            if dist < 5:
                nearby_opponents += 1
                if dist < 2:
                    very_close_opponents += 1
        
        # Pression = proche + très proche avec poids double
        pressure = min(1.0, (nearby_opponents + very_close_opponents) / 4)
        
        return pressure
    
    def _estimate_goalkeeper_position(self, goal_pos: Tuple[float, float],
                                    ball_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Estime la position du gardien"""
        # Le gardien se positionne entre le but et le ballon
        # À environ 5m devant la ligne
        direction = np.array(ball_pos) - np.array(goal_pos)
        direction = direction / np.linalg.norm(direction)
        
        gk_pos = np.array(goal_pos) + direction * 5
        
        return tuple(gk_pos)
    
    def _find_better_positioned_teammates(self, context: DecisionContext,
                                        shooter_xG: float) -> List[int]:
        """Trouve les coéquipiers mieux placés pour tirer"""
        better_positioned = []
        
        for teammate_id, teammate_pos in context.teammate_positions.items():
            # Calculer le xG potentiel du coéquipier
            dist = distance.euclidean(teammate_pos, context.goal_position)
            angle = self._calculate_shot_angle(teammate_pos, context.goal_position)
            
            # xG simplifié
            teammate_xG = 0
            if dist < 12:
                teammate_xG = 0.4
            elif dist < 18:
                teammate_xG = 0.2
            
            # Si significativement mieux placé
            if teammate_xG > shooter_xG * 1.5:
                better_positioned.append(teammate_id)
        
        return better_positioned
    
    def _calculate_counter_attack_risk(self, pos: Tuple[float, float],
                                     zone: ZoneType) -> float:
        """Calcule le risque de contre-attaque si perte de balle"""
        # Plus risqué si on perd la balle haut sur le terrain
        if zone == ZoneType.ATTACKING_THIRD:
            base_risk = 0.7
        elif zone == ZoneType.MIDDLE_THIRD:
            base_risk = 0.5
        else:
            base_risk = 0.3
        
        # Ajuster selon la position latérale (centre plus risqué)
        center_y = self.pitch_width / 2
        lateral_factor = 1 - abs(pos[1] - center_y) / (self.pitch_width / 2) * 0.3
        
        return base_risk * lateral_factor
    
    def _calculate_shot_difficulty(self, distance: float, angle: float,
                                  pressure: float) -> float:
        """Calcule la difficulté d'un tir"""
        # Distance
        dist_difficulty = min(1.0, distance / 30)
        
        # Angle (plus difficile si angle fermé)
        angle_difficulty = 1.0 - (angle / 90)  # 90° = facile, 0° = difficile
        
        # Pression
        pressure_factor = 1 + pressure * 0.5
        
        total = (dist_difficulty + angle_difficulty) / 2 * pressure_factor
        
        return min(1.0, total)
    
    def _find_targets_in_penalty_area(self, teammates: Dict[int, Tuple[float, float]],
                                    goal_pos: Tuple[float, float]) -> Dict[int, Tuple[float, float]]:
        """Trouve les coéquipiers dans la surface de réparation"""
        targets = {}
        
        # Limites de la surface (16.5m x 40.3m)
        box_x_min = goal_pos[0] - 16.5
        box_x_max = goal_pos[0]
        box_y_min = goal_pos[1] - 20.15
        box_y_max = goal_pos[1] + 20.15
        
        for teammate_id, pos in teammates.items():
            if (box_x_min <= pos[0] <= box_x_max and
                box_y_min <= pos[1] <= box_y_max):
                targets[teammate_id] = pos
        
        return targets
    
    def _evaluate_cross_position_quality(self, from_pos: Tuple[float, float],
                                       to_pos: Tuple[float, float],
                                       goal_pos: Tuple[float, float]) -> float:
        """Évalue la qualité d'une position de centre"""
        # Distance de la cible au but
        target_dist = distance.euclidean(to_pos, goal_pos)
        
        # Qualité selon distance
        if target_dist < 6:
            quality = 0.9
        elif target_dist < 11:
            quality = 0.7
        else:
            quality = 0.5
        
        # Angle du centre
        cross_angle = abs(from_pos[1] - to_pos[1]) / distance.euclidean(from_pos, to_pos)
        quality *= (0.5 + cross_angle * 0.5)  # Préférer les centres croisés
        
        return quality
    
    def _calculate_cross_success_probability(self, quality: float, space: float,
                                           pressure: float) -> float:
        """Calcule la probabilité de succès d'un centre"""
        # Base sur la qualité de position
        base_prob = quality * 0.7
        
        # Ajuster pour l'espace du receveur
        space_factor = min(1.0, space / 10)
        base_prob *= (0.5 + space_factor * 0.5)
        
        # Ajuster pour la pression sur le centreur
        base_prob *= (1 - pressure * 0.3)
        
        return max(0.2, min(0.8, base_prob))
    
    def _calculate_cross_danger(self, target_pos: Tuple[float, float],
                              goal_pos: Tuple[float, float],
                              space: float) -> float:
        """Calcule le danger créé par un centre"""
        # xG potentiel de la position
        dist = distance.euclidean(target_pos, goal_pos)
        
        if dist < 6:
            base_danger = 0.6
        elif dist < 11:
            base_danger = 0.4
        else:
            base_danger = 0.2
        
        # Ajuster pour l'espace
        space_factor = min(1.0, space / 5)
        
        return base_danger * (0.5 + space_factor * 0.5)
    
    def _calculate_immediate_pressure(self, pos: Tuple[float, float],
                                    opponents: Dict[int, Tuple[float, float]]) -> float:
        """Calcule la pression immédiate sur un joueur"""
        # Compter les adversaires très proches
        very_close = sum(1 for opp_pos in opponents.values() 
                        if distance.euclidean(pos, opp_pos) < 2)
        
        close = sum(1 for opp_pos in opponents.values() 
                   if 2 <= distance.euclidean(pos, opp_pos) < 5)
        
        # Pression = très proches avec poids double + proches
        pressure = min(1.0, (very_close * 2 + close) / 5)
        
        return pressure