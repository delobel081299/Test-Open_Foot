"""
Module d'analyse tactique des formations
Détection, analyse et visualisation des schémas tactiques
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
import cv2
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, distance
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class Formation(Enum):
    """Types de formations tactiques"""
    F_442 = "4-4-2"
    F_433 = "4-3-3"
    F_352 = "3-5-2"
    F_4231 = "4-2-3-1"
    F_4141 = "4-1-4-1"
    F_343 = "3-4-3"
    F_532 = "5-3-2"
    F_4222 = "4-2-2-2"
    F_451 = "4-5-1"
    F_UNKNOWN = "Unknown"


class TacticalPhase(Enum):
    """Phases tactiques du jeu"""
    DEFENSIVE = "defensive"
    OFFENSIVE = "offensive"
    TRANSITION_DEF_TO_ATT = "transition_def_to_att"
    TRANSITION_ATT_TO_DEF = "transition_att_to_def"
    PRESSING = "pressing"
    BUILD_UP = "build_up"


@dataclass
class FormationMetrics:
    """Métriques tactiques d'une formation"""
    formation_type: Formation
    confidence: float
    compactness_horizontal: float
    compactness_vertical: float
    inter_line_distances: Dict[str, float]
    width: float
    depth: float
    surface_area: float
    center_of_gravity: Tuple[float, float]
    defensive_line_height: float
    offensive_line_height: float
    
    
@dataclass
class TacticalAnalysis:
    """Analyse tactique complète"""
    formation: FormationMetrics
    phase: TacticalPhase
    bloc_height: str  # "high", "medium", "low"
    asymmetries: Dict[str, float]
    out_of_position_players: List[int]
    pressing_coordination: float
    transition_speed: Optional[float]
    pass_network: Optional[Dict]
    heatmaps: Optional[Dict]
    

class FormationAnalyzer:
    """Analyseur tactique des formations et positionnements"""
    
    def __init__(self, pitch_dimensions: Tuple[float, float] = (105, 68)):
        self.pitch_length, self.pitch_width = pitch_dimensions
        
        # Définitions des formations de référence (positions normalisées 0-1)
        self.reference_formations = self._initialize_reference_formations()
        
        # Paramètres d'analyse
        self.clustering_params = {
            'eps': 0.15,  # Distance max pour clustering
            'min_samples': 2
        }
        
        # Historique pour analyses temporelles
        self.position_history = []
        self.formation_history = []
        
    def analyze_formation(self, player_positions: Dict[int, Tuple[float, float]], 
                         team_id: int, ball_position: Optional[Tuple[float, float]] = None,
                         opponent_positions: Optional[Dict] = None) -> TacticalAnalysis:
        """
        Analyse complète de la formation tactique
        
        Args:
            player_positions: Positions des joueurs {player_id: (x, y)}
            team_id: ID de l'équipe analysée
            ball_position: Position du ballon
            opponent_positions: Positions des adversaires
            
        Returns:
            Analyse tactique complète
        """
        # Normaliser les positions
        normalized_positions = self._normalize_positions(player_positions)
        
        # Détecter la formation
        formation_metrics = self._detect_formation(normalized_positions)
        
        # Déterminer la phase tactique
        phase = self._determine_tactical_phase(
            normalized_positions, ball_position, opponent_positions
        )
        
        # Analyser le bloc et la compacité
        bloc_height = self._analyze_bloc_height(normalized_positions)
        
        # Détecter les asymétries
        asymmetries = self._detect_asymmetries(normalized_positions, formation_metrics.formation_type)
        
        # Identifier les joueurs hors position
        out_of_position = self._find_out_of_position_players(
            normalized_positions, formation_metrics.formation_type
        )
        
        # Analyser la coordination du pressing
        pressing_coord = self._analyze_pressing_coordination(
            normalized_positions, opponent_positions, ball_position
        )
        
        # Calculer la vitesse de transition si applicable
        transition_speed = self._calculate_transition_speed(phase)
        
        # Générer les visualisations
        heatmaps = self._generate_heatmaps(self.position_history[-100:] if self.position_history else [])
        pass_network = self._analyze_pass_network() if hasattr(self, 'pass_data') else None
        
        # Mise à jour de l'historique
        self.position_history.append(normalized_positions)
        self.formation_history.append(formation_metrics.formation_type)
        
        return TacticalAnalysis(
            formation=formation_metrics,
            phase=phase,
            bloc_height=bloc_height,
            asymmetries=asymmetries,
            out_of_position_players=out_of_position,
            pressing_coordination=pressing_coord,
            transition_speed=transition_speed,
            pass_network=pass_network,
            heatmaps=heatmaps
        )
    
    def _detect_formation(self, positions: Dict[int, Tuple[float, float]]) -> FormationMetrics:
        """Détecte le type de formation tactique"""
        # Extraire les coordonnées
        coords = np.array(list(positions.values()))
        
        # Clustering pour identifier les lignes
        lines = self._cluster_into_lines(coords)
        
        # Classifier la formation
        formation_type, confidence = self._classify_formation(lines)
        
        # Calculer les métriques
        metrics = self._calculate_formation_metrics(coords, lines)
        
        return FormationMetrics(
            formation_type=formation_type,
            confidence=confidence,
            compactness_horizontal=metrics['compactness_h'],
            compactness_vertical=metrics['compactness_v'],
            inter_line_distances=metrics['inter_line_distances'],
            width=metrics['width'],
            depth=metrics['depth'],
            surface_area=metrics['surface_area'],
            center_of_gravity=metrics['center_of_gravity'],
            defensive_line_height=metrics['defensive_line_height'],
            offensive_line_height=metrics['offensive_line_height']
        )
    
    def _cluster_into_lines(self, coords: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """Regroupe les joueurs en lignes tactiques"""
        # Trier par position Y (défense vers attaque)
        sorted_indices = np.argsort(coords[:, 1])
        sorted_coords = coords[sorted_indices]
        
        # Clustering DBSCAN sur l'axe Y
        y_coords = sorted_coords[:, 1].reshape(-1, 1)
        clustering = DBSCAN(eps=0.1, min_samples=2).fit(y_coords)
        
        # Grouper par cluster
        lines = {'defense': [], 'midfield': [], 'attack': []}
        unique_labels = set(clustering.labels_) - {-1}
        
        line_positions = []
        for label in unique_labels:
            mask = clustering.labels_ == label
            line_coords = sorted_coords[mask]
            avg_y = np.mean(line_coords[:, 1])
            line_positions.append((avg_y, line_coords))
        
        # Trier les lignes par position Y
        line_positions.sort(key=lambda x: x[0])
        
        # Assigner aux catégories
        if len(line_positions) >= 3:
            lines['defense'] = line_positions[0][1]
            lines['midfield'] = line_positions[1][1]
            lines['attack'] = line_positions[2][1]
        elif len(line_positions) == 2:
            lines['defense'] = line_positions[0][1]
            lines['midfield'] = line_positions[1][1]
        elif len(line_positions) == 1:
            lines['defense'] = line_positions[0][1]
        
        return lines
    
    def _classify_formation(self, lines: Dict[str, List[np.ndarray]]) -> Tuple[Formation, float]:
        """Classifie la formation parmi les types connus"""
        # Compter les joueurs par ligne
        def_count = len(lines['defense'])
        mid_count = len(lines['midfield'])
        att_count = len(lines['attack'])
        
        # Analyser la distribution spatiale pour affiner
        mid_distribution = self._analyze_line_distribution(lines['midfield']) if mid_count > 0 else None
        
        # Mapping simple basé sur le nombre de joueurs
        formation_map = {
            (4, 4, 2): Formation.F_442,
            (4, 3, 3): Formation.F_433,
            (3, 5, 2): Formation.F_352,
            (4, 5, 1): Formation.F_451,
            (3, 4, 3): Formation.F_343,
            (5, 3, 2): Formation.F_532,
        }
        
        # Cas spéciaux nécessitant une analyse plus fine
        if def_count == 4 and mid_count in [4, 5] and att_count == 1:
            # Distinguer 4-2-3-1 vs 4-1-4-1 vs 4-5-1
            if mid_distribution and mid_distribution['layers'] == 2:
                if mid_distribution['bottom_layer_count'] == 2:
                    return Formation.F_4231, 0.85
                elif mid_distribution['bottom_layer_count'] == 1:
                    return Formation.F_4141, 0.85
            else:
                return Formation.F_451, 0.8
        
        if def_count == 4 and mid_count == 4 and att_count == 2:
            # Distinguer 4-4-2 vs 4-2-2-2
            if mid_distribution and mid_distribution['layers'] == 2:
                return Formation.F_4222, 0.85
            else:
                return Formation.F_442, 0.9
        
        # Recherche dans le mapping
        key = (def_count, mid_count, att_count)
        if key in formation_map:
            return formation_map[key], 0.9
        
        # Si pas de correspondance exacte, trouver la plus proche
        best_formation = Formation.F_UNKNOWN
        best_distance = float('inf')
        
        for form_key, formation in formation_map.items():
            distance = sum(abs(a - b) for a, b in zip(key, form_key))
            if distance < best_distance:
                best_distance = distance
                best_formation = formation
        
        confidence = max(0.5, 1 - (best_distance * 0.2))
        return best_formation, confidence
    
    def _analyze_line_distribution(self, line_coords: List[np.ndarray]) -> Dict:
        """Analyse la distribution spatiale d'une ligne"""
        if len(line_coords) == 0:
            return None
        
        coords = np.vstack(line_coords) if isinstance(line_coords, list) else line_coords
        
        # Clustering sur l'axe Y pour détecter les sous-lignes
        y_coords = coords[:, 1].reshape(-1, 1)
        if len(y_coords) > 1:
            clustering = DBSCAN(eps=0.05, min_samples=1).fit(y_coords)
            n_layers = len(set(clustering.labels_))
            
            # Compter les joueurs dans chaque sous-ligne
            layer_counts = []
            for label in set(clustering.labels_):
                count = np.sum(clustering.labels_ == label)
                layer_counts.append(count)
            
            layer_counts.sort(reverse=True)
            
            return {
                'layers': n_layers,
                'bottom_layer_count': layer_counts[-1] if n_layers > 1 else len(coords),
                'top_layer_count': layer_counts[0] if n_layers > 1 else 0
            }
        
        return {'layers': 1, 'bottom_layer_count': len(coords), 'top_layer_count': 0}
    
    def _calculate_formation_metrics(self, coords: np.ndarray, 
                                   lines: Dict[str, List[np.ndarray]]) -> Dict:
        """Calcule les métriques tactiques de la formation"""
        metrics = {}
        
        # Compacité horizontale et verticale
        if len(coords) > 1:
            x_std = np.std(coords[:, 0])
            y_std = np.std(coords[:, 1])
            metrics['compactness_h'] = 1 / (1 + x_std)
            metrics['compactness_v'] = 1 / (1 + y_std)
        else:
            metrics['compactness_h'] = 0
            metrics['compactness_v'] = 0
        
        # Largeur et profondeur
        metrics['width'] = np.max(coords[:, 0]) - np.min(coords[:, 0])
        metrics['depth'] = np.max(coords[:, 1]) - np.min(coords[:, 1])
        
        # Surface convexe
        if len(coords) >= 3:
            try:
                hull = ConvexHull(coords)
                metrics['surface_area'] = hull.volume  # En 2D, volume = aire
            except:
                metrics['surface_area'] = 0
        else:
            metrics['surface_area'] = 0
        
        # Centre de gravité
        metrics['center_of_gravity'] = tuple(np.mean(coords, axis=0))
        
        # Distances inter-lignes
        metrics['inter_line_distances'] = {}
        if lines['defense'].size > 0 and lines['midfield'].size > 0:
            def_y = np.mean([c[1] for c in lines['defense']])
            mid_y = np.mean([c[1] for c in lines['midfield']])
            metrics['inter_line_distances']['def_mid'] = abs(mid_y - def_y)
        
        if lines['midfield'].size > 0 and lines['attack'].size > 0:
            mid_y = np.mean([c[1] for c in lines['midfield']])
            att_y = np.mean([c[1] for c in lines['attack']])
            metrics['inter_line_distances']['mid_att'] = abs(att_y - mid_y)
        
        # Hauteur des lignes
        metrics['defensive_line_height'] = np.mean([c[1] for c in lines['defense']]) if lines['defense'].size > 0 else 0
        metrics['offensive_line_height'] = np.mean([c[1] for c in lines['attack']]) if lines['attack'].size > 0 else 1
        
        return metrics
    
    def _determine_tactical_phase(self, positions: Dict, ball_pos: Optional[Tuple], 
                                 opponents: Optional[Dict]) -> TacticalPhase:
        """Détermine la phase tactique actuelle"""
        if not ball_pos:
            return TacticalPhase.DEFENSIVE
        
        # Position moyenne de l'équipe
        team_coords = np.array(list(positions.values()))
        team_center_y = np.mean(team_coords[:, 1])
        
        # Position du ballon normalisée
        ball_y = ball_pos[1] / self.pitch_length
        
        # Analyser la vitesse de mouvement si historique disponible
        if len(self.position_history) > 5:
            recent_centers = [np.mean(list(p.values()), axis=0)[1] for p in self.position_history[-5:]]
            movement_speed = np.std(recent_centers)
            
            # Transition détectée si mouvement rapide
            if movement_speed > 0.1:
                if recent_centers[-1] > recent_centers[0]:
                    return TacticalPhase.TRANSITION_DEF_TO_ATT
                else:
                    return TacticalPhase.TRANSITION_ATT_TO_DEF
        
        # Phase selon position
        if ball_y < 0.25:  # Ballon dans le tiers défensif
            return TacticalPhase.DEFENSIVE
        elif ball_y > 0.75:  # Ballon dans le tiers offensif
            if team_center_y > 0.6:
                return TacticalPhase.PRESSING
            else:
                return TacticalPhase.OFFENSIVE
        else:  # Milieu de terrain
            if abs(team_center_y - 0.5) < 0.1:
                return TacticalPhase.BUILD_UP
            else:
                return TacticalPhase.OFFENSIVE if team_center_y > 0.5 else TacticalPhase.DEFENSIVE
    
    def _analyze_bloc_height(self, positions: Dict) -> str:
        """Analyse la hauteur du bloc défensif"""
        coords = np.array(list(positions.values()))
        avg_y = np.mean(coords[:, 1])
        
        if avg_y > 0.65:
            return "high"
        elif avg_y > 0.45:
            return "medium"
        else:
            return "low"
    
    def _detect_asymmetries(self, positions: Dict, formation: Formation) -> Dict[str, float]:
        """Détecte les asymétries dans la formation"""
        coords = np.array(list(positions.values()))
        asymmetries = {}
        
        # Asymétrie latérale
        left_players = np.sum(coords[:, 0] < 0.5)
        right_players = np.sum(coords[:, 0] > 0.5)
        asymmetries['lateral'] = abs(left_players - right_players) / len(coords)
        
        # Asymétrie par rapport à la formation théorique
        if formation != Formation.F_UNKNOWN:
            ref_positions = self.reference_formations.get(formation, {})
            if ref_positions:
                # Calculer la déviation moyenne
                deviations = []
                for player_id, pos in positions.items():
                    # Trouver la position de référence la plus proche
                    if player_id in ref_positions:
                        ref_pos = ref_positions[player_id]
                        deviation = np.linalg.norm(np.array(pos) - np.array(ref_pos))
                        deviations.append(deviation)
                
                asymmetries['formation_deviation'] = np.mean(deviations) if deviations else 0
        
        # Asymétrie de densité (entropie spatiale)
        grid_size = 5
        grid = np.zeros((grid_size, grid_size))
        for x, y in coords:
            grid_x = int(x * grid_size)
            grid_y = int(y * grid_size)
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                grid[grid_y, grid_x] += 1
        
        grid_flat = grid.flatten()
        grid_flat = grid_flat[grid_flat > 0]
        if len(grid_flat) > 0:
            asymmetries['spatial_entropy'] = entropy(grid_flat / grid_flat.sum())
        else:
            asymmetries['spatial_entropy'] = 0
        
        return asymmetries
    
    def _find_out_of_position_players(self, positions: Dict, formation: Formation) -> List[int]:
        """Identifie les joueurs hors de leur position théorique"""
        out_of_position = []
        
        if formation == Formation.F_UNKNOWN:
            return out_of_position
        
        ref_positions = self.reference_formations.get(formation, {})
        if not ref_positions:
            return out_of_position
        
        # Seuil de distance pour considérer un joueur hors position
        threshold = 0.2
        
        for player_id, current_pos in positions.items():
            if player_id in ref_positions:
                ref_pos = ref_positions[player_id]
                distance = np.linalg.norm(np.array(current_pos) - np.array(ref_pos))
                
                if distance > threshold:
                    out_of_position.append(player_id)
        
        return out_of_position
    
    def _analyze_pressing_coordination(self, positions: Dict, 
                                     opponents: Optional[Dict], 
                                     ball_pos: Optional[Tuple]) -> float:
        """Analyse la coordination du pressing"""
        if not opponents or not ball_pos:
            return 0.0
        
        # Identifier les joueurs les plus proches du ballon
        distances_to_ball = {}
        for player_id, pos in positions.items():
            dist = np.linalg.norm(np.array(pos) - np.array(ball_pos))
            distances_to_ball[player_id] = dist
        
        # Les 3 joueurs les plus proches
        pressing_players = sorted(distances_to_ball.items(), key=lambda x: x[1])[:3]
        pressing_ids = [p[0] for p in pressing_players]
        
        # Calculer la compacité du pressing
        pressing_positions = [positions[pid] for pid in pressing_ids]
        if len(pressing_positions) >= 3:
            # Distance moyenne entre les presseurs
            avg_distance = 0
            count = 0
            for i in range(len(pressing_positions)):
                for j in range(i + 1, len(pressing_positions)):
                    dist = np.linalg.norm(
                        np.array(pressing_positions[i]) - np.array(pressing_positions[j])
                    )
                    avg_distance += dist
                    count += 1
            
            avg_distance /= count
            
            # Score de coordination (plus proche = meilleur)
            coordination_score = 1 / (1 + avg_distance * 5)
            
            # Bonus si les presseurs forment un triangle autour du porteur
            ball_carrier = min(opponents.items(), 
                             key=lambda x: np.linalg.norm(np.array(x[1]) - np.array(ball_pos)))
            
            if ball_carrier:
                carrier_pos = ball_carrier[1]
                # Vérifier si le porteur est "encerclé"
                angles = []
                for pos in pressing_positions:
                    angle = np.arctan2(pos[1] - carrier_pos[1], pos[0] - carrier_pos[0])
                    angles.append(angle)
                
                angles.sort()
                # Calculer la couverture angulaire
                max_gap = 0
                for i in range(len(angles)):
                    gap = angles[(i + 1) % len(angles)] - angles[i]
                    if gap < 0:
                        gap += 2 * np.pi
                    max_gap = max(max_gap, gap)
                
                # Si le plus grand écart est < 180°, bon encerclement
                if max_gap < np.pi:
                    coordination_score *= 1.3
            
            return min(coordination_score, 1.0)
        
        return 0.0
    
    def _calculate_transition_speed(self, phase: TacticalPhase) -> Optional[float]:
        """Calcule la vitesse de transition si applicable"""
        if phase not in [TacticalPhase.TRANSITION_DEF_TO_ATT, TacticalPhase.TRANSITION_ATT_TO_DEF]:
            return None
        
        if len(self.position_history) < 10:
            return None
        
        # Analyser le déplacement du centre de gravité
        recent_positions = self.position_history[-10:]
        centers = []
        
        for positions in recent_positions:
            coords = np.array(list(positions.values()))
            center = np.mean(coords, axis=0)
            centers.append(center)
        
        # Vitesse de déplacement vertical
        y_positions = [c[1] for c in centers]
        y_velocity = (y_positions[-1] - y_positions[0]) / len(y_positions)
        
        # Normaliser par rapport à la longueur du terrain
        transition_speed = abs(y_velocity) * 10  # Échelle 0-1
        
        return min(transition_speed, 1.0)
    
    def _generate_heatmaps(self, position_history: List[Dict]) -> Dict[int, np.ndarray]:
        """Génère les heatmaps de position pour chaque joueur"""
        if not position_history:
            return {}
        
        heatmaps = {}
        grid_size = 50
        
        # Collecter toutes les positions par joueur
        player_positions = {}
        for frame_positions in position_history:
            for player_id, pos in frame_positions.items():
                if player_id not in player_positions:
                    player_positions[player_id] = []
                player_positions[player_id].append(pos)
        
        # Créer une heatmap pour chaque joueur
        for player_id, positions in player_positions.items():
            heatmap = np.zeros((grid_size, grid_size))
            
            for x, y in positions:
                grid_x = int(x * (grid_size - 1))
                grid_y = int(y * (grid_size - 1))
                
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    heatmap[grid_y, grid_x] += 1
            
            # Appliquer un filtre gaussien pour lisser
            from scipy.ndimage import gaussian_filter
            heatmap = gaussian_filter(heatmap, sigma=2)
            
            # Normaliser
            if heatmap.max() > 0:
                heatmap /= heatmap.max()
            
            heatmaps[player_id] = heatmap
        
        return heatmaps
    
    def _analyze_pass_network(self) -> Dict:
        """Analyse le réseau de passes (placeholder - nécessite des données de passes)"""
        # Cette méthode nécessiterait des données de passes pour être implémentée
        # Retourne une structure vide pour l'instant
        return {
            'nodes': [],  # Joueurs
            'edges': [],  # Passes entre joueurs
            'centrality': {},  # Importance de chaque joueur dans le réseau
            'clusters': []  # Sous-groupes de joueurs qui jouent ensemble
        }
    
    def visualize_formation(self, analysis: TacticalAnalysis, 
                           save_path: Optional[str] = None) -> None:
        """Visualise la formation tactique"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Formation actuelle avec lignes
        ax1 = axes[0, 0]
        self._plot_formation_lines(ax1, analysis)
        ax1.set_title(f"Formation: {analysis.formation.formation_type.value}")
        
        # 2. Heatmap équipe
        ax2 = axes[0, 1]
        self._plot_team_heatmap(ax2, analysis.heatmaps)
        ax2.set_title("Heatmap de l'équipe")
        
        # 3. Métriques tactiques
        ax3 = axes[1, 0]
        self._plot_tactical_metrics(ax3, analysis)
        ax3.set_title("Métriques tactiques")
        
        # 4. Évolution temporelle
        ax4 = axes[1, 1]
        self._plot_formation_evolution(ax4)
        ax4.set_title("Évolution de la formation")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def _plot_formation_lines(self, ax, analysis: TacticalAnalysis):
        """Trace la formation avec les lignes tactiques"""
        # Dessiner le terrain
        self._draw_pitch(ax)
        
        # Récupérer les positions actuelles
        if self.position_history:
            positions = self.position_history[-1]
            coords = np.array(list(positions.values()))
            
            # Tracer les joueurs
            ax.scatter(coords[:, 0] * self.pitch_length, 
                      coords[:, 1] * self.pitch_width,
                      s=300, c='red', edgecolors='white', linewidth=2, zorder=5)
            
            # Numéros des joueurs
            for i, (player_id, pos) in enumerate(positions.items()):
                ax.text(pos[0] * self.pitch_length, pos[1] * self.pitch_width, 
                       str(player_id), ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='white', zorder=6)
            
            # Tracer les lignes de formation
            lines_coords = self._cluster_into_lines(coords)
            colors = ['blue', 'green', 'orange']
            labels = ['Défense', 'Milieu', 'Attaque']
            
            for (line_type, line_coords), color, label in zip(lines_coords.items(), colors, labels):
                if len(line_coords) > 0:
                    # Ligne horizontale moyenne
                    y_mean = np.mean([c[1] for c in line_coords]) * self.pitch_width
                    x_coords = [c[0] * self.pitch_length for c in line_coords]
                    ax.plot(x_coords, [y_mean] * len(x_coords), 
                           color=color, linewidth=2, alpha=0.7, label=label)
            
            # Surface convexe de l'équipe
            if len(coords) >= 3:
                try:
                    hull = ConvexHull(coords * [self.pitch_length, self.pitch_width])
                    for simplex in hull.simplices:
                        ax.plot(coords[simplex, 0] * self.pitch_length, 
                               coords[simplex, 1] * self.pitch_width,
                               'k-', alpha=0.3, linewidth=1)
                except:
                    pass
            
            ax.legend(loc='upper right')
    
    def _plot_team_heatmap(self, ax, heatmaps: Optional[Dict]):
        """Trace la heatmap combinée de l'équipe"""
        self._draw_pitch(ax)
        
        if heatmaps:
            # Combiner toutes les heatmaps
            combined_heatmap = np.zeros_like(list(heatmaps.values())[0])
            for heatmap in heatmaps.values():
                combined_heatmap += heatmap
            
            # Normaliser
            if combined_heatmap.max() > 0:
                combined_heatmap /= combined_heatmap.max()
            
            # Afficher
            extent = [0, self.pitch_length, 0, self.pitch_width]
            im = ax.imshow(combined_heatmap, extent=extent, origin='lower',
                          cmap='YlOrRd', alpha=0.7, aspect='auto')
            plt.colorbar(im, ax=ax, label='Densité de présence')
    
    def _plot_tactical_metrics(self, ax, analysis: TacticalAnalysis):
        """Affiche les métriques tactiques sous forme de barres"""
        metrics = {
            'Compacité H': analysis.formation.compactness_horizontal,
            'Compacité V': analysis.formation.compactness_vertical,
            'Largeur': analysis.formation.width,
            'Profondeur': analysis.formation.depth,
            'Coordination\nPressing': analysis.pressing_coordination,
            'Confiance\nFormation': analysis.formation.confidence
        }
        
        names = list(metrics.keys())
        values = list(metrics.values())
        
        bars = ax.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylim(0, 1.2)
        ax.set_ylabel('Valeur')
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter des informations textuelles
        info_text = f"Phase: {analysis.phase.value}\n"
        info_text += f"Bloc: {analysis.bloc_height}\n"
        info_text += f"Joueurs hors position: {len(analysis.out_of_position_players)}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_formation_evolution(self, ax):
        """Trace l'évolution de la formation dans le temps"""
        if len(self.formation_history) < 2:
            ax.text(0.5, 0.5, 'Pas assez de données historiques', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Compter les occurrences de chaque formation
        formation_counts = {}
        window_size = 50  # Fenêtre glissante
        
        time_points = []
        formation_data = {f: [] for f in Formation}
        
        for i in range(window_size, len(self.formation_history)):
            window = self.formation_history[i-window_size:i]
            counts = {f: 0 for f in Formation}
            
            for formation in window:
                counts[formation] += 1
            
            time_points.append(i)
            for formation in Formation:
                formation_data[formation].append(counts[formation] / window_size)
        
        # Tracer les lignes
        for formation, data in formation_data.items():
            if formation != Formation.F_UNKNOWN and any(d > 0 for d in data):
                ax.plot(time_points, data, label=formation.value, linewidth=2)
        
        ax.set_xlabel('Temps (frames)')
        ax.set_ylabel('Fréquence d\'utilisation')
        ax.set_ylim(0, 1.1)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def _draw_pitch(self, ax):
        """Dessine un terrain de football"""
        # Terrain
        pitch = patches.Rectangle((0, 0), self.pitch_length, self.pitch_width, 
                                linewidth=2, edgecolor='black', facecolor='green', alpha=0.3)
        ax.add_patch(pitch)
        
        # Ligne médiane
        ax.plot([self.pitch_length/2, self.pitch_length/2], [0, self.pitch_width], 
               'black', linewidth=2)
        
        # Cercle central
        center_circle = patches.Circle((self.pitch_length/2, self.pitch_width/2), 
                                     9.15, linewidth=2, edgecolor='black', 
                                     facecolor='none')
        ax.add_patch(center_circle)
        
        # Surfaces de réparation
        penalty_area_1 = patches.Rectangle((0, self.pitch_width/2 - 20.15), 
                                         16.5, 40.3, linewidth=2, 
                                         edgecolor='black', facecolor='none')
        penalty_area_2 = patches.Rectangle((self.pitch_length - 16.5, self.pitch_width/2 - 20.15), 
                                         16.5, 40.3, linewidth=2, 
                                         edgecolor='black', facecolor='none')
        ax.add_patch(penalty_area_1)
        ax.add_patch(penalty_area_2)
        
        ax.set_xlim(-5, self.pitch_length + 5)
        ax.set_ylim(-5, self.pitch_width + 5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _initialize_reference_formations(self) -> Dict[Formation, Dict]:
        """Initialise les positions de référence pour chaque formation"""
        # Positions normalisées (0-1) pour chaque formation
        # Format: {player_id: (x, y)}
        
        reference = {
            Formation.F_442: {
                # Défense
                2: (0.2, 0.2), 3: (0.2, 0.4), 4: (0.2, 0.6), 5: (0.2, 0.8),
                # Milieu
                6: (0.5, 0.2), 7: (0.5, 0.4), 8: (0.5, 0.6), 11: (0.5, 0.8),
                # Attaque
                9: (0.8, 0.35), 10: (0.8, 0.65)
            },
            Formation.F_433: {
                # Défense
                2: (0.2, 0.2), 3: (0.2, 0.4), 4: (0.2, 0.6), 5: (0.2, 0.8),
                # Milieu
                6: (0.45, 0.3), 8: (0.5, 0.5), 10: (0.45, 0.7),
                # Attaque
                7: (0.8, 0.2), 9: (0.8, 0.5), 11: (0.8, 0.8)
            },
            Formation.F_352: {
                # Défense
                3: (0.2, 0.3), 4: (0.2, 0.5), 5: (0.2, 0.7),
                # Milieu
                2: (0.4, 0.1), 6: (0.45, 0.3), 8: (0.5, 0.5), 10: (0.45, 0.7), 11: (0.4, 0.9),
                # Attaque
                9: (0.8, 0.35), 7: (0.8, 0.65)
            },
            Formation.F_4231: {
                # Défense
                2: (0.2, 0.2), 3: (0.2, 0.4), 4: (0.2, 0.6), 5: (0.2, 0.8),
                # Milieu défensif
                6: (0.4, 0.35), 8: (0.4, 0.65),
                # Milieu offensif
                7: (0.6, 0.2), 10: (0.6, 0.5), 11: (0.6, 0.8),
                # Attaque
                9: (0.85, 0.5)
            }
        }
        
        return reference
    
    def export_tactical_data(self, analysis: TacticalAnalysis, 
                            output_path: str, format: str = 'json') -> None:
        """
        Exporte les données tactiques pour tableau tactique
        
        Args:
            analysis: Analyse tactique à exporter
            output_path: Chemin de sortie
            format: Format d'export ('json', 'csv', 'xml')
        """
        data = {
            'timestamp': datetime.now().isoformat(),
            'formation': {
                'type': analysis.formation.formation_type.value,
                'confidence': analysis.formation.confidence,
                'metrics': {
                    'compactness_h': analysis.formation.compactness_horizontal,
                    'compactness_v': analysis.formation.compactness_vertical,
                    'width': analysis.formation.width,
                    'depth': analysis.formation.depth,
                    'surface_area': analysis.formation.surface_area,
                    'center_of_gravity': analysis.formation.center_of_gravity,
                    'defensive_line': analysis.formation.defensive_line_height,
                    'offensive_line': analysis.formation.offensive_line_height
                }
            },
            'phase': analysis.phase.value,
            'bloc_height': analysis.bloc_height,
            'asymmetries': analysis.asymmetries,
            'out_of_position': analysis.out_of_position_players,
            'pressing_coordination': analysis.pressing_coordination,
            'transition_speed': analysis.transition_speed
        }
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'csv':
            # Aplatir les données pour CSV
            import pandas as pd
            flattened = self._flatten_dict(data)
            df = pd.DataFrame([flattened])
            df.to_csv(output_path, index=False)
        elif format == 'xml':
            import xml.etree.ElementTree as ET
            root = self._dict_to_xml(data, 'tactical_analysis')
            tree = ET.ElementTree(root)
            tree.write(output_path)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Aplatit un dictionnaire imbriqué"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _dict_to_xml(self, data: Dict, root_name: str) -> Any:
        """Convertit un dictionnaire en XML"""
        import xml.etree.ElementTree as ET
        
        root = ET.Element(root_name)
        
        def build_xml(element, data):
            if isinstance(data, dict):
                for key, value in data.items():
                    sub_element = ET.SubElement(element, str(key))
                    build_xml(sub_element, value)
            elif isinstance(data, (list, tuple)):
                for i, item in enumerate(data):
                    sub_element = ET.SubElement(element, f'item_{i}')
                    build_xml(sub_element, item)
            else:
                element.text = str(data)
        
        build_xml(root, data)
        return root
    
    def create_animation(self, position_history: List[Dict], 
                        output_path: str, fps: int = 10) -> None:
        """
        Crée une animation de l'évolution tactique
        
        Args:
            position_history: Historique des positions
            output_path: Chemin de sortie pour l'animation
            fps: Images par seconde
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def update(frame):
            ax.clear()
            self._draw_pitch(ax)
            
            # Positions actuelles
            positions = position_history[frame]
            coords = np.array(list(positions.values()))
            
            # Tracer les joueurs
            ax.scatter(coords[:, 0] * self.pitch_length, 
                      coords[:, 1] * self.pitch_width,
                      s=300, c='red', edgecolors='white', linewidth=2)
            
            # Titre avec le numéro de frame
            ax.set_title(f'Frame {frame}/{len(position_history)}', fontsize=14)
            
            return ax.artists
        
        anim = FuncAnimation(fig, update, frames=len(position_history),
                           interval=1000/fps, blit=False)
        
        anim.save(output_path, writer='pillow', fps=fps)
        plt.close()
    
    def compare_formations(self, analysis1: TacticalAnalysis, 
                          analysis2: TacticalAnalysis) -> Dict:
        """
        Compare deux analyses tactiques
        
        Args:
            analysis1: Première analyse
            analysis2: Deuxième analyse
            
        Returns:
            Dictionnaire de comparaison
        """
        comparison = {
            'formation_similarity': 1.0 if analysis1.formation.formation_type == analysis2.formation.formation_type else 0.0,
            'compactness_diff': {
                'horizontal': abs(analysis1.formation.compactness_horizontal - analysis2.formation.compactness_horizontal),
                'vertical': abs(analysis1.formation.compactness_vertical - analysis2.formation.compactness_vertical)
            },
            'dimension_diff': {
                'width': abs(analysis1.formation.width - analysis2.formation.width),
                'depth': abs(analysis1.formation.depth - analysis2.formation.depth)
            },
            'bloc_height_diff': self._compare_bloc_heights(analysis1.bloc_height, analysis2.bloc_height),
            'pressing_diff': abs(analysis1.pressing_coordination - analysis2.pressing_coordination),
            'phase_match': analysis1.phase == analysis2.phase
        }
        
        # Score de similarité global
        similarity_score = (
            comparison['formation_similarity'] * 0.3 +
            (1 - comparison['compactness_diff']['horizontal']) * 0.15 +
            (1 - comparison['compactness_diff']['vertical']) * 0.15 +
            (1 - comparison['dimension_diff']['width'] / 0.5) * 0.1 +
            (1 - comparison['dimension_diff']['depth'] / 0.5) * 0.1 +
            (1 - comparison['bloc_height_diff']) * 0.1 +
            (1 - comparison['pressing_diff']) * 0.1
        )
        
        comparison['overall_similarity'] = max(0, min(1, similarity_score))
        
        return comparison
    
    def _compare_bloc_heights(self, height1: str, height2: str) -> float:
        """Compare deux hauteurs de bloc"""
        height_values = {'low': 0, 'medium': 0.5, 'high': 1}
        return abs(height_values[height1] - height_values[height2])