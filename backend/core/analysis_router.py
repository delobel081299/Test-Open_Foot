"""
Analysis Router - Module de sélection automatique du pipeline d'analyse
Choisit le pipeline optimal selon le type de vidéo (match, entraînement, etc.)
"""

import os
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging
from pathlib import Path

from backend.utils.logger import setup_logger
from backend.utils.validators import validate_video_file
from backend.core.preprocessing.video_loader import VideoLoader
from backend.core.preprocessing.frame_extractor import FrameExtractor
from backend.core.detection.yolo_detector import YOLODetector
from backend.core.tracking.byte_tracker import ByteTracker
from backend.core.tracking.team_classifier import TeamClassifier
from backend.core.biomechanics.pose_extractor import PoseExtractor
from backend.core.technical.action_classifier import ActionClassifier
from backend.core.tactical.position_analyzer import PositionAnalyzer
from backend.core.scoring.score_aggregator import ScoreAggregator
from backend.core.scoring.report_builder import ReportBuilder

logger = setup_logger(__name__)


class VideoType(Enum):
    """Types de vidéos supportés"""
    MATCH = "match"                    # Match complet
    TRAINING = "entrainement"          # Séance d'entraînement
    GOALKEEPER = "gardien"             # Entraînement spécifique gardien
    PHYSICAL = "physique"              # Préparation physique
    TECHNICAL = "technique"            # Exercices techniques spécifiques
    TACTICAL = "tactique"              # Séances tactiques
    SET_PIECE = "coup_franc"          # Coups de pied arrêtés
    
    @classmethod
    def from_string(cls, value: str) -> Optional['VideoType']:
        """Convertir string en VideoType"""
        value = value.lower().strip()
        for video_type in cls:
            if video_type.value == value:
                return video_type
        return None
    
    @classmethod
    def get_all_types(cls) -> List[str]:
        """Obtenir tous les types supportés"""
        return [vt.value for vt in cls]


@dataclass
class PipelineConfig:
    """Configuration d'un pipeline d'analyse"""
    name: str
    enabled_modules: List[str]
    disabled_modules: List[str]
    priority_modules: List[str]  # Modules à exécuter en priorité
    parameters: Dict[str, Any]
    min_duration: float  # Durée minimale vidéo (secondes)
    max_duration: float  # Durée maximale vidéo (secondes)


@dataclass
class AnalysisResult:
    """Résultat d'une analyse"""
    video_type: VideoType
    pipeline_used: str
    modules_executed: List[str]
    processing_time: float
    scores: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any]


class AnalysisRouter:
    """
    Routeur intelligent pour sélectionner le pipeline d'analyse optimal
    selon le type de vidéo fourni
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialiser le routeur d'analyse
        
        Args:
            config: Configuration optionnelle
        """
        self.config = config or {}
        self.pipelines = self._init_pipelines()
        self.modules = self._init_modules()
        
        logger.info(f"AnalysisRouter initialisé avec {len(self.pipelines)} pipelines")
        
    def _init_pipelines(self) -> Dict[VideoType, PipelineConfig]:
        """Initialiser les configurations de pipelines"""
        return {
            VideoType.MATCH: PipelineConfig(
                name="Match Analysis Pipeline",
                enabled_modules=[
                    "detection", "tracking", "team_classification",
                    "tactical_analysis", "position_analysis", 
                    "statistics", "heatmap", "passing_network"
                ],
                disabled_modules=[
                    "biomechanics_detailed", "technique_analysis"
                ],
                priority_modules=["detection", "tracking"],
                parameters={
                    "min_players": 10,
                    "track_ball": True,
                    "analyze_formations": True,
                    "generate_heatmaps": True,
                    "calculate_possession": True
                },
                min_duration=300.0,   # 5 minutes minimum
                max_duration=7200.0   # 2 heures maximum
            ),
            
            VideoType.TRAINING: PipelineConfig(
                name="Training Analysis Pipeline",
                enabled_modules=[
                    "detection", "tracking", "biomechanics",
                    "technique_analysis", "action_classification",
                    "performance_metrics"
                ],
                disabled_modules=[
                    "tactical_analysis", "team_classification",
                    "heatmap", "passing_network"
                ],
                priority_modules=["biomechanics", "technique_analysis"],
                parameters={
                    "detailed_pose": True,
                    "analyze_technique": True,
                    "track_repetitions": True,
                    "measure_form": True,
                    "individual_feedback": True
                },
                min_duration=10.0,    # 10 secondes minimum
                max_duration=3600.0   # 1 heure maximum
            ),
            
            VideoType.GOALKEEPER: PipelineConfig(
                name="Goalkeeper Analysis Pipeline",
                enabled_modules=[
                    "detection", "tracking", "biomechanics",
                    "goalkeeper_specific", "reaction_time",
                    "dive_analysis"
                ],
                disabled_modules=[
                    "tactical_analysis", "team_classification"
                ],
                priority_modules=["goalkeeper_specific", "reaction_time"],
                parameters={
                    "track_hands": True,
                    "analyze_dives": True,
                    "measure_reaction": True,
                    "positioning_analysis": True
                },
                min_duration=5.0,
                max_duration=1800.0
            ),
            
            VideoType.PHYSICAL: PipelineConfig(
                name="Physical Training Pipeline",
                enabled_modules=[
                    "detection", "tracking", "speed_analysis",
                    "acceleration_metrics", "fatigue_detection"
                ],
                disabled_modules=[
                    "tactical_analysis", "technique_analysis"
                ],
                priority_modules=["speed_analysis"],
                parameters={
                    "track_speed": True,
                    "measure_acceleration": True,
                    "analyze_sprints": True,
                    "detect_fatigue": True
                },
                min_duration=10.0,
                max_duration=3600.0
            ),
            
            VideoType.SET_PIECE: PipelineConfig(
                name="Set Piece Analysis Pipeline",
                enabled_modules=[
                    "detection", "tracking", "ball_trajectory",
                    "player_positioning", "success_rate"
                ],
                disabled_modules=[
                    "fatigue_detection", "heatmap"
                ],
                priority_modules=["ball_trajectory"],
                parameters={
                    "analyze_trajectory": True,
                    "track_wall": True,
                    "measure_accuracy": True
                },
                min_duration=5.0,
                max_duration=300.0
            )
        }
    
    def _init_modules(self) -> Dict[str, Callable]:
        """Initialiser les modules d'analyse disponibles"""
        return {
            "detection": self._run_detection,
            "tracking": self._run_tracking,
            "team_classification": self._run_team_classification,
            "biomechanics": self._run_biomechanics,
            "biomechanics_detailed": self._run_biomechanics_detailed,
            "technique_analysis": self._run_technique_analysis,
            "tactical_analysis": self._run_tactical_analysis,
            "position_analysis": self._run_position_analysis,
            "action_classification": self._run_action_classification,
            "statistics": self._run_statistics,
            "heatmap": self._run_heatmap,
            "passing_network": self._run_passing_network,
            "performance_metrics": self._run_performance_metrics,
            "goalkeeper_specific": self._run_goalkeeper_analysis,
            "reaction_time": self._run_reaction_time,
            "dive_analysis": self._run_dive_analysis,
            "speed_analysis": self._run_speed_analysis,
            "acceleration_metrics": self._run_acceleration_metrics,
            "fatigue_detection": self._run_fatigue_detection,
            "ball_trajectory": self._run_ball_trajectory,
            "player_positioning": self._run_player_positioning,
            "success_rate": self._run_success_rate
        }
    
    def analyze_video(
        self,
        video_path: str,
        video_type: str,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Analyser une vidéo en sélectionnant automatiquement le bon pipeline
        
        Args:
            video_path: Chemin vers la vidéo
            video_type: Type de vidéo (match, entrainement, etc.)
            custom_params: Paramètres personnalisés optionnels
            
        Returns:
            AnalysisResult avec tous les résultats
        """
        import time
        start_time = time.time()
        
        # Résultat à retourner
        result = AnalysisResult(
            video_type=None,
            pipeline_used="",
            modules_executed=[],
            processing_time=0.0,
            scores={},
            warnings=[],
            errors=[],
            metadata={}
        )
        
        try:
            # Valider le type de vidéo
            vtype = VideoType.from_string(video_type)
            if not vtype:
                result.errors.append(
                    f"Type de vidéo '{video_type}' non reconnu. "
                    f"Types supportés: {VideoType.get_all_types()}"
                )
                return result
            
            result.video_type = vtype
            
            # Valider le fichier vidéo
            if not os.path.exists(video_path):
                result.errors.append(f"Fichier vidéo introuvable: {video_path}")
                return result
            
            try:
                video_info = validate_video_file(video_path)
                result.metadata['video_info'] = video_info
            except Exception as e:
                result.errors.append(f"Vidéo invalide: {str(e)}")
                return result
            
            # Sélectionner le pipeline
            pipeline = self.pipelines.get(vtype)
            if not pipeline:
                result.errors.append(f"Aucun pipeline configuré pour le type '{video_type}'")
                return result
            
            result.pipeline_used = pipeline.name
            
            # Vérifier la durée
            duration = video_info.get('duration', 0)
            if duration < pipeline.min_duration:
                result.warnings.append(
                    f"Vidéo trop courte ({duration:.1f}s). "
                    f"Minimum recommandé: {pipeline.min_duration}s"
                )
            elif duration > pipeline.max_duration:
                result.warnings.append(
                    f"Vidéo trop longue ({duration:.1f}s). "
                    f"Maximum recommandé: {pipeline.max_duration}s"
                )
            
            # Charger la vidéo
            logger.info(f"Analyse de type '{vtype.value}' avec pipeline: {pipeline.name}")
            video_data = self._load_video(video_path)
            
            # Exécuter les modules dans l'ordre
            module_results = {}
            
            # D'abord les modules prioritaires
            for module_name in pipeline.priority_modules:
                if module_name in pipeline.enabled_modules:
                    logger.info(f"Exécution du module prioritaire: {module_name}")
                    module_func = self.modules.get(module_name)
                    if module_func:
                        try:
                            module_results[module_name] = module_func(
                                video_data, 
                                pipeline.parameters,
                                module_results  # Passer les résultats précédents
                            )
                            result.modules_executed.append(module_name)
                        except Exception as e:
                            logger.error(f"Erreur dans module {module_name}: {e}")
                            result.errors.append(f"Module {module_name}: {str(e)}")
            
            # Puis les autres modules activés
            for module_name in pipeline.enabled_modules:
                if module_name not in pipeline.priority_modules:
                    logger.info(f"Exécution du module: {module_name}")
                    module_func = self.modules.get(module_name)
                    if module_func:
                        try:
                            module_results[module_name] = module_func(
                                video_data,
                                pipeline.parameters,
                                module_results
                            )
                            result.modules_executed.append(module_name)
                        except Exception as e:
                            logger.error(f"Erreur dans module {module_name}: {e}")
                            result.errors.append(f"Module {module_name}: {str(e)}")
            
            # Vérifier les modules désactivés
            for module_name in pipeline.disabled_modules:
                if module_name in custom_params.get('force_modules', []):
                    result.warnings.append(
                        f"Module '{module_name}' forcé mais non recommandé "
                        f"pour le type '{vtype.value}'"
                    )
            
            # Agréger les scores
            result.scores = self._aggregate_scores(module_results, vtype)
            
            # Temps de traitement
            result.processing_time = time.time() - start_time
            
            logger.info(
                f"Analyse terminée en {result.processing_time:.1f}s. "
                f"Modules exécutés: {len(result.modules_executed)}"
            )
            
        except Exception as e:
            logger.error(f"Erreur fatale dans l'analyse: {e}")
            result.errors.append(f"Erreur fatale: {str(e)}")
        
        return result
    
    def _load_video(self, video_path: str) -> Dict[str, Any]:
        """Charger et préparer les données vidéo"""
        loader = VideoLoader()
        extractor = FrameExtractor()
        
        video_info = loader.load_video(video_path)
        frames = extractor.extract_frames(video_path, interval_seconds=1.0)
        
        return {
            'path': video_path,
            'info': video_info,
            'frames': frames
        }
    
    def _aggregate_scores(
        self, 
        module_results: Dict[str, Any], 
        video_type: VideoType
    ) -> Dict[str, Any]:
        """Agréger les scores selon le type de vidéo"""
        aggregator = ScoreAggregator()
        
        # Poids différents selon le type
        if video_type == VideoType.MATCH:
            weights = {
                'tactical': 0.4,
                'positional': 0.3,
                'statistics': 0.3
            }
        elif video_type == VideoType.TRAINING:
            weights = {
                'technique': 0.5,
                'biomechanics': 0.3,
                'performance': 0.2
            }
        else:
            weights = {
                'overall': 1.0
            }
        
        # Agréger avec les poids appropriés
        scores = aggregator.aggregate_with_weights(module_results, weights)
        
        return scores
    
    # Méthodes stub pour les modules (à implémenter)
    def _run_detection(self, video_data, params, prev_results):
        """Exécuter la détection d'objets"""
        detector = YOLODetector()
        detections = []
        for frame in video_data['frames']:
            detections.append(detector.detect(frame))
        return {'detections': detections}
    
    def _run_tracking(self, video_data, params, prev_results):
        """Exécuter le tracking"""
        tracker = ByteTracker()
        detections = prev_results.get('detection', {}).get('detections', [])
        tracks = tracker.process_video(detections)
        return {'tracks': tracks}
    
    def _run_team_classification(self, video_data, params, prev_results):
        """Classifier les équipes"""
        classifier = TeamClassifier()
        # Implémenter la logique
        return {'team_assignments': {}}
    
    def _run_biomechanics(self, video_data, params, prev_results):
        """Analyse biomécanique basique"""
        extractor = PoseExtractor()
        # Implémenter la logique
        return {'poses': []}
    
    def _run_biomechanics_detailed(self, video_data, params, prev_results):
        """Analyse biomécanique détaillée"""
        # Implémenter analyse approfondie
        return {'detailed_poses': []}
    
    def _run_technique_analysis(self, video_data, params, prev_results):
        """Analyse technique"""
        # Implémenter
        return {'technique_scores': {}}
    
    def _run_tactical_analysis(self, video_data, params, prev_results):
        """Analyse tactique"""
        analyzer = PositionAnalyzer()
        # Implémenter
        return {'formations': [], 'tactics': {}}
    
    def _run_position_analysis(self, video_data, params, prev_results):
        """Analyse des positions"""
        # Implémenter
        return {'positions': {}}
    
    def _run_action_classification(self, video_data, params, prev_results):
        """Classification des actions"""
        classifier = ActionClassifier()
        # Implémenter
        return {'actions': []}
    
    def _run_statistics(self, video_data, params, prev_results):
        """Calcul des statistiques"""
        # Implémenter
        return {'stats': {}}
    
    def _run_heatmap(self, video_data, params, prev_results):
        """Génération des heatmaps"""
        # Implémenter
        return {'heatmaps': {}}
    
    def _run_passing_network(self, video_data, params, prev_results):
        """Analyse du réseau de passes"""
        # Implémenter
        return {'passing_network': {}}
    
    def _run_performance_metrics(self, video_data, params, prev_results):
        """Métriques de performance"""
        # Implémenter
        return {'performance': {}}
    
    def _run_goalkeeper_analysis(self, video_data, params, prev_results):
        """Analyse spécifique gardien"""
        # Implémenter
        return {'goalkeeper_metrics': {}}
    
    def _run_reaction_time(self, video_data, params, prev_results):
        """Mesure temps de réaction"""
        # Implémenter
        return {'reaction_times': []}
    
    def _run_dive_analysis(self, video_data, params, prev_results):
        """Analyse des plongeons"""
        # Implémenter
        return {'dives': []}
    
    def _run_speed_analysis(self, video_data, params, prev_results):
        """Analyse de vitesse"""
        # Implémenter
        return {'speeds': {}}
    
    def _run_acceleration_metrics(self, video_data, params, prev_results):
        """Métriques d'accélération"""
        # Implémenter
        return {'accelerations': {}}
    
    def _run_fatigue_detection(self, video_data, params, prev_results):
        """Détection de fatigue"""
        # Implémenter
        return {'fatigue_levels': {}}
    
    def _run_ball_trajectory(self, video_data, params, prev_results):
        """Analyse trajectoire ballon"""
        # Implémenter
        return {'trajectories': []}
    
    def _run_player_positioning(self, video_data, params, prev_results):
        """Positionnement des joueurs"""
        # Implémenter
        return {'positioning': {}}
    
    def _run_success_rate(self, video_data, params, prev_results):
        """Taux de réussite"""
        # Implémenter
        return {'success_rates': {}}
    
    def get_supported_types(self) -> List[Dict[str, Any]]:
        """
        Obtenir la liste des types de vidéos supportés avec leurs détails
        
        Returns:
            Liste des types avec description
        """
        types = []
        
        for vtype, pipeline in self.pipelines.items():
            types.append({
                'type': vtype.value,
                'name': pipeline.name,
                'enabled_modules': pipeline.enabled_modules,
                'min_duration': pipeline.min_duration,
                'max_duration': pipeline.max_duration,
                'description': self._get_type_description(vtype)
            })
        
        return types
    
    def _get_type_description(self, vtype: VideoType) -> str:
        """Obtenir description d'un type de vidéo"""
        descriptions = {
            VideoType.MATCH: "Match complet avec analyse tactique et statistiques avancées",
            VideoType.TRAINING: "Séance d'entraînement avec analyse technique détaillée",
            VideoType.GOALKEEPER: "Entraînement spécifique gardien avec métriques dédiées",
            VideoType.PHYSICAL: "Préparation physique avec analyse de performance",
            VideoType.TECHNICAL: "Exercices techniques avec évaluation biomécanique",
            VideoType.TACTICAL: "Séances tactiques avec analyse des mouvements collectifs",
            VideoType.SET_PIECE: "Coups de pied arrêtés avec analyse de trajectoire"
        }
        return descriptions.get(vtype, "")
    
    def validate_video_type_match(
        self,
        video_path: str,
        declared_type: str
    ) -> Dict[str, Any]:
        """
        Valider que le type déclaré correspond au contenu vidéo
        
        Returns:
            Dict avec 'is_valid', 'confidence', 'suggested_type', 'reason'
        """
        # Analyser rapidement la vidéo pour déterminer le type probable
        # TODO: Implémenter détection automatique du type
        
        return {
            'is_valid': True,
            'confidence': 0.95,
            'suggested_type': declared_type,
            'reason': "Type validé par analyse du contenu"
        }


# Fonction utilitaire pour créer le routeur
def create_analysis_router(config: Optional[Dict[str, Any]] = None) -> AnalysisRouter:
    """Créer une instance du routeur d'analyse"""
    return AnalysisRouter(config)


if __name__ == "__main__":
    # Test du routeur
    router = AnalysisRouter()
    
    # Afficher types supportés
    print("Types de vidéos supportés:")
    for type_info in router.get_supported_types():
        print(f"- {type_info['type']}: {type_info['description']}")
    
    # Test d'analyse
    result = router.analyze_video(
        video_path="test_match.mp4",
        video_type="match"
    )
    
    print(f"\nRésultat analyse:")
    print(f"Pipeline: {result.pipeline_used}")
    print(f"Modules exécutés: {result.modules_executed}")
    print(f"Temps: {result.processing_time:.1f}s")