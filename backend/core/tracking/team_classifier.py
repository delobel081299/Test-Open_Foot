"""
Team Classifier for football players
Identification automatique des équipes par analyse couleur des maillots
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from skimage import color
import logging
import time

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TeamInfo:
    """Information sur une équipe"""
    team_id: int
    color_hsv: np.ndarray  # Couleur dominante en HSV
    color_lab: np.ndarray  # Couleur dominante en Lab
    color_rgb: np.ndarray  # Couleur dominante en RGB
    confidence: float
    player_count: int = 0
    is_goalkeeper: bool = False


@dataclass
class PlayerTeamAssignment:
    """Assignation d'un joueur à une équipe"""
    player_id: int
    team_id: int
    confidence: float
    jersey_color_hsv: np.ndarray
    jersey_color_rgb: np.ndarray
    is_goalkeeper: bool = False
    is_referee: bool = False


class TeamClassifier:
    """
    Classificateur d'équipes basé sur la couleur des maillots
    Gestion robuste des cas spéciaux (gardiens, arbitres, changements lumière)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialiser le classificateur
        
        Args:
            config: Configuration du classificateur
        """
        self.config = config or self._get_default_config()
        
        # Clusters des équipes
        self.team_clusters: Dict[int, TeamInfo] = {}
        self.player_assignments: Dict[int, PlayerTeamAssignment] = {}
        
        # Historique pour validation temporelle
        self.assignment_history: Dict[int, List[int]] = defaultdict(list)
        self.color_history: Dict[int, List[np.ndarray]] = defaultdict(list)
        
        # KMeans pour clustering initial
        self.kmeans = None
        self.n_clusters = 3  # Équipe 1, Équipe 2, Arbitres
        
        # Statistiques
        self.processing_times = []
        
        logger.info("TeamClassifier initialisé")
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut"""
        return {
            'min_jersey_area': 500,           # Aire minimale zone maillot
            'history_window': 30,             # Frames pour validation temporelle
            'confidence_threshold': 0.7,      # Seuil de confiance
            'color_distance_threshold': 30,   # Distance max en Lab
            'use_pose_mask': True,            # Utiliser pose pour masquer
            'adaptive_update': True,          # Mise à jour adaptative
            'referee_detection': True,        # Détection auto arbitres
            'goalkeeper_detection': True,     # Détection gardiens
        }
    
    def classify_teams(
        self,
        player_images: Dict[int, np.ndarray],
        player_poses: Optional[Dict[int, Any]] = None,
        frame_id: int = 0
    ) -> Dict[int, int]:
        """
        Classifier les joueurs en équipes
        
        Args:
            player_images: Images des joueurs {player_id: image}
            player_poses: Poses des joueurs pour masquage
            frame_id: ID de la frame courante
            
        Returns:
            Mapping {player_id: team_id}
        """
        start_time = time.time()
        
        # Extraire couleurs des maillots
        player_colors = self._extract_jersey_colors(player_images, player_poses)
        
        # Initialiser clusters si première frame
        if len(self.team_clusters) == 0:
            self._initialize_clusters(player_colors)
        
        # Assigner joueurs aux équipes
        assignments = {}
        
        for player_id, color_data in player_colors.items():
            # Assigner à l'équipe la plus proche
            team_id, confidence = self._assign_to_team(
                player_id, 
                color_data,
                frame_id
            )
            
            assignments[player_id] = team_id
            
            # Mettre à jour historique
            self.assignment_history[player_id].append(team_id)
            self.color_history[player_id].append(color_data['hsv'])
            
            # Limiter taille historique
            if len(self.assignment_history[player_id]) > self.config['history_window']:
                self.assignment_history[player_id].pop(0)
                self.color_history[player_id].pop(0)
        
        # Validation temporelle
        assignments = self._temporal_validation(assignments)
        
        # Mise à jour adaptative des clusters
        if self.config['adaptive_update']:
            self._update_clusters(player_colors, assignments)
        
        # Détection cas spéciaux
        if self.config['referee_detection']:
            self._detect_referees(player_colors, assignments)
        
        if self.config['goalkeeper_detection']:
            self._detect_goalkeepers(player_colors, assignments)
        
        # Mesurer performance
        processing_time = (time.time() - start_time) * 1000  # ms
        self.processing_times.append(processing_time)
        
        if len(player_images) > 0:
            avg_time_per_player = processing_time / len(player_images)
            if avg_time_per_player > 5:
                logger.warning(f"Performance dégradée: {avg_time_per_player:.1f}ms par joueur")
        
        return assignments
    
    def _extract_jersey_colors(
        self,
        player_images: Dict[int, np.ndarray],
        player_poses: Optional[Dict[int, Any]] = None
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Extraire la couleur dominante du maillot de chaque joueur
        
        Returns:
            {player_id: {'hsv': color_hsv, 'rgb': color_rgb, 'lab': color_lab}}
        """
        player_colors = {}
        
        for player_id, image in player_images.items():
            # Extraire région du maillot
            if self.config['use_pose_mask'] and player_poses and player_id in player_poses:
                jersey_region = self._extract_jersey_region_with_pose(
                    image, 
                    player_poses[player_id]
                )
            else:
                jersey_region = self._extract_jersey_region_simple(image)
            
            if jersey_region is None or jersey_region.size < self.config['min_jersey_area']:
                continue
            
            # Calculer couleur dominante
            dominant_color = self._get_dominant_color(jersey_region)
            
            # Convertir en différents espaces couleur
            color_rgb = dominant_color
            color_hsv = cv2.cvtColor(
                dominant_color.reshape(1, 1, 3), 
                cv2.COLOR_RGB2HSV
            ).reshape(3)
            color_lab = color.rgb2lab(dominant_color / 255.0).reshape(3)
            
            player_colors[player_id] = {
                'rgb': color_rgb,
                'hsv': color_hsv,
                'lab': color_lab
            }
        
        return player_colors
    
    def _extract_jersey_region_with_pose(
        self, 
        image: np.ndarray, 
        pose: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        Extraire région maillot en utilisant les keypoints de pose
        Focus sur le torse (shoulders -> hips)
        """
        try:
            keypoints = pose.get('keypoints', {})
            
            # Points clés pour le torse
            left_shoulder = keypoints.get('left_shoulder')
            right_shoulder = keypoints.get('right_shoulder')
            left_hip = keypoints.get('left_hip')
            right_hip = keypoints.get('right_hip')
            
            if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
                return self._extract_jersey_region_simple(image)
            
            # Créer masque pour la région du torse
            points = np.array([
                left_shoulder[:2],
                right_shoulder[:2],
                right_hip[:2],
                left_hip[:2]
            ], dtype=np.int32)
            
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            
            # Extraire région
            jersey_region = cv2.bitwise_and(image, image, mask=mask)
            
            # Recadrer sur la zone d'intérêt
            x, y, w, h = cv2.boundingRect(points)
            jersey_region = jersey_region[y:y+h, x:x+w]
            
            return jersey_region
            
        except Exception as e:
            logger.debug(f"Erreur extraction avec pose: {e}")
            return self._extract_jersey_region_simple(image)
    
    def _extract_jersey_region_simple(self, image: np.ndarray) -> np.ndarray:
        """
        Extraire région maillot simple (partie centrale haute)
        """
        h, w = image.shape[:2]
        
        # Prendre la partie centrale haute (torse probable)
        y_start = int(h * 0.2)
        y_end = int(h * 0.6)
        x_start = int(w * 0.2)
        x_end = int(w * 0.8)
        
        jersey_region = image[y_start:y_end, x_start:x_end]
        
        return jersey_region
    
    def _get_dominant_color(self, region: np.ndarray) -> np.ndarray:
        """
        Obtenir couleur dominante d'une région
        Utilise histogramme HSV pour robustesse
        """
        # Convertir en HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        
        # Masquer pixels noirs/blancs (éviter shorts, chaussettes)
        mask = cv2.inRange(hsv, 
                          np.array([0, 30, 30]),    # Min HSV
                          np.array([180, 255, 255])) # Max HSV
        
        # Calculer histogramme sur la teinte
        hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
        
        # Trouver teinte dominante
        dominant_hue = np.argmax(hist)
        
        # Obtenir pixels de cette teinte
        hue_range = 10
        hue_mask = cv2.inRange(hsv[:, :, 0], 
                              dominant_hue - hue_range,
                              dominant_hue + hue_range)
        
        combined_mask = cv2.bitwise_and(mask, hue_mask)
        
        # Moyenne des pixels dans cette gamme
        mean_color = cv2.mean(region, mask=combined_mask)[:3]
        
        return np.array(mean_color, dtype=np.uint8)
    
    def _initialize_clusters(self, player_colors: Dict[int, Dict[str, np.ndarray]]):
        """
        Initialiser clusters K-means avec les couleurs des joueurs
        """
        if len(player_colors) < self.n_clusters:
            logger.warning(f"Pas assez de joueurs pour {self.n_clusters} clusters")
            self.n_clusters = min(2, len(player_colors))
        
        # Préparer données pour clustering
        colors_lab = []
        player_ids = []
        
        for player_id, color_data in player_colors.items():
            colors_lab.append(color_data['lab'])
            player_ids.append(player_id)
        
        colors_lab = np.array(colors_lab)
        
        # K-means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(colors_lab)
        
        # Créer TeamInfo pour chaque cluster
        for i in range(self.n_clusters):
            cluster_mask = cluster_labels == i
            cluster_colors = colors_lab[cluster_mask]
            
            if len(cluster_colors) > 0:
                # Couleur moyenne du cluster
                mean_lab = np.mean(cluster_colors, axis=0)
                mean_rgb = (color.lab2rgb(mean_lab.reshape(1, 1, 3)) * 255).astype(np.uint8).reshape(3)
                mean_hsv = cv2.cvtColor(mean_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2HSV).reshape(3)
                
                self.team_clusters[i] = TeamInfo(
                    team_id=i,
                    color_hsv=mean_hsv,
                    color_lab=mean_lab,
                    color_rgb=mean_rgb,
                    confidence=1.0,
                    player_count=np.sum(cluster_mask)
                )
        
        logger.info(f"Initialisé {len(self.team_clusters)} clusters d'équipes")
    
    def _assign_to_team(
        self,
        player_id: int,
        color_data: Dict[str, np.ndarray],
        frame_id: int
    ) -> Tuple[int, float]:
        """
        Assigner un joueur à une équipe
        
        Returns:
            (team_id, confidence)
        """
        min_distance = float('inf')
        best_team_id = -1
        
        # Calculer distance à chaque cluster
        player_lab = color_data['lab']
        
        for team_id, team_info in self.team_clusters.items():
            # Distance colorimétrique CIE Lab
            distance = np.linalg.norm(player_lab - team_info.color_lab)
            
            if distance < min_distance:
                min_distance = distance
                best_team_id = team_id
        
        # Calculer confiance basée sur distance
        confidence = max(0, 1 - (min_distance / self.config['color_distance_threshold']))
        
        # Créer/mettre à jour assignment
        self.player_assignments[player_id] = PlayerTeamAssignment(
            player_id=player_id,
            team_id=best_team_id,
            confidence=confidence,
            jersey_color_hsv=color_data['hsv'],
            jersey_color_rgb=color_data['rgb']
        )
        
        return best_team_id, confidence
    
    def _temporal_validation(self, assignments: Dict[int, int]) -> Dict[int, int]:
        """
        Validation temporelle des assignments
        Utilise l'historique pour corriger les erreurs ponctuelles
        """
        validated_assignments = assignments.copy()
        
        for player_id, current_team in assignments.items():
            history = self.assignment_history[player_id]
            
            if len(history) >= 5:  # Besoin d'historique suffisant
                # Vote majoritaire sur les dernières frames
                team_counts = Counter(history[-10:])
                most_common_team, count = team_counts.most_common(1)[0]
                
                # Si le vote majoritaire diffère et est fort
                if most_common_team != current_team and count >= 7:
                    logger.debug(f"Correction temporelle: Joueur {player_id} de team {current_team} à {most_common_team}")
                    validated_assignments[player_id] = most_common_team
        
        return validated_assignments
    
    def _update_clusters(
        self,
        player_colors: Dict[int, Dict[str, np.ndarray]],
        assignments: Dict[int, int]
    ):
        """
        Mise à jour adaptative des clusters
        Ajuste les couleurs moyennes en fonction des nouvelles observations
        """
        if not self.config['adaptive_update']:
            return
        
        # Regrouper couleurs par équipe
        team_colors = defaultdict(list)
        
        for player_id, team_id in assignments.items():
            if player_id in player_colors:
                team_colors[team_id].append(player_colors[player_id]['lab'])
        
        # Mettre à jour couleurs moyennes
        alpha = 0.1  # Taux d'apprentissage
        
        for team_id, colors in team_colors.items():
            if team_id in self.team_clusters and len(colors) > 0:
                new_mean = np.mean(colors, axis=0)
                old_mean = self.team_clusters[team_id].color_lab
                
                # Moyenne pondérée
                updated_mean = (1 - alpha) * old_mean + alpha * new_mean
                
                # Mettre à jour toutes les représentations couleur
                self.team_clusters[team_id].color_lab = updated_mean
                self.team_clusters[team_id].color_rgb = (
                    color.lab2rgb(updated_mean.reshape(1, 1, 3)) * 255
                ).astype(np.uint8).reshape(3)
                self.team_clusters[team_id].color_hsv = cv2.cvtColor(
                    self.team_clusters[team_id].color_rgb.reshape(1, 1, 3),
                    cv2.COLOR_RGB2HSV
                ).reshape(3)
    
    def _detect_referees(
        self,
        player_colors: Dict[int, Dict[str, np.ndarray]],
        assignments: Dict[int, int]
    ):
        """
        Détecter automatiquement les arbitres
        Généralement en noir ou couleur très différente
        """
        if not self.config['referee_detection']:
            return
        
        # Chercher joueurs avec couleurs très sombres ou très différentes
        for player_id, color_data in player_colors.items():
            hsv = color_data['hsv']
            
            # Critères pour arbitre
            is_dark = hsv[2] < 50  # Valeur faible (noir)
            is_unique = True  # TODO: vérifier unicité couleur
            
            if is_dark or (assignments[player_id] == 2 and self.n_clusters == 3):
                if player_id in self.player_assignments:
                    self.player_assignments[player_id].is_referee = True
                    logger.debug(f"Joueur {player_id} identifié comme arbitre")
    
    def _detect_goalkeepers(
        self,
        player_colors: Dict[int, Dict[str, np.ndarray]],
        assignments: Dict[int, int]
    ):
        """
        Détecter les gardiens de but
        Couleur généralement différente de leur équipe
        """
        if not self.config['goalkeeper_detection']:
            return
        
        # Analyser outliers par équipe
        team_players = defaultdict(list)
        
        for player_id, team_id in assignments.items():
            if player_id in player_colors:
                team_players[team_id].append({
                    'id': player_id,
                    'color': player_colors[player_id]['lab']
                })
        
        # Pour chaque équipe, chercher outlier
        for team_id, players in team_players.items():
            if len(players) < 3:  # Pas assez de joueurs
                continue
            
            # Calculer distances intra-équipe
            colors = np.array([p['color'] for p in players])
            mean_color = np.mean(colors, axis=0)
            
            for i, player in enumerate(players):
                distance = np.linalg.norm(player['color'] - mean_color)
                
                # Si distance importante, probable gardien
                if distance > self.config['color_distance_threshold'] * 1.5:
                    player_id = player['id']
                    if player_id in self.player_assignments:
                        self.player_assignments[player_id].is_goalkeeper = True
                        logger.debug(f"Joueur {player_id} identifié comme gardien de l'équipe {team_id}")
    
    def handle_halftime_change(self):
        """
        Gérer changement de maillots à la mi-temps
        Réinitialise les clusters
        """
        logger.info("Réinitialisation pour changement mi-temps")
        
        # Sauvegarder statistiques
        first_half_stats = {
            'clusters': self.team_clusters.copy(),
            'assignments': self.player_assignments.copy()
        }
        
        # Réinitialiser
        self.team_clusters.clear()
        self.assignment_history.clear()
        self.color_history.clear()
        self.kmeans = None
        
        return first_half_stats
    
    def get_team_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Obtenir les couleurs RGB de chaque équipe
        
        Returns:
            {team_id: (r, g, b)}
        """
        team_colors = {}
        
        for team_id, team_info in self.team_clusters.items():
            team_colors[team_id] = tuple(team_info.color_rgb.tolist())
        
        return team_colors
    
    def get_assignments_with_confidence(self) -> Dict[int, Tuple[int, float]]:
        """
        Obtenir assignments avec confiance
        
        Returns:
            {player_id: (team_id, confidence)}
        """
        result = {}
        
        for player_id, assignment in self.player_assignments.items():
            result[player_id] = (assignment.team_id, assignment.confidence)
        
        return result
    
    def visualize_teams(self, image: np.ndarray) -> np.ndarray:
        """
        Visualiser les équipes identifiées
        """
        vis_image = image.copy()
        
        # Afficher couleurs des équipes
        y_offset = 30
        for team_id, team_info in self.team_clusters.items():
            color = tuple(int(c) for c in team_info.color_rgb)
            
            # Rectangle avec couleur équipe
            cv2.rectangle(vis_image, (10, y_offset), (100, y_offset + 20), color, -1)
            
            # Texte
            text = f"Team {team_id}"
            if team_info.is_goalkeeper:
                text += " (GK)"
            
            cv2.putText(vis_image, text, (110, y_offset + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 30
        
        return vis_image
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Obtenir statistiques de performance
        """
        if not self.processing_times:
            return {}
        
        return {
            'avg_time_ms': np.mean(self.processing_times),
            'max_time_ms': np.max(self.processing_times),
            'min_time_ms': np.min(self.processing_times),
            'total_classifications': len(self.player_assignments)
        }


if __name__ == "__main__":
    # Test du classificateur
    classifier = TeamClassifier()
    
    # Créer images de test
    player_images = {
        1: np.full((100, 50, 3), [255, 0, 0], dtype=np.uint8),    # Rouge
        2: np.full((100, 50, 3), [255, 0, 0], dtype=np.uint8),    # Rouge
        3: np.full((100, 50, 3), [0, 0, 255], dtype=np.uint8),    # Bleu
        4: np.full((100, 50, 3), [0, 0, 255], dtype=np.uint8),    # Bleu
        5: np.full((100, 50, 3), [0, 0, 0], dtype=np.uint8),      # Noir (arbitre)
    }
    
    # Classifier
    assignments = classifier.classify_teams(player_images)
    
    print("Assignments:", assignments)
    print("Couleurs équipes:", classifier.get_team_colors())