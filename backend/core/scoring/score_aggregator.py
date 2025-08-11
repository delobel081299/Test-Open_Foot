"""
Score aggregator for combining different analysis scores
Advanced scoring system with adaptive weighting and normalization
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class VideoType(Enum):
    """Video analysis types"""
    TRAINING = "training"
    MATCH = "match"
    DRILL = "drill"
    FREESTYLE = "freestyle"

class PlayerPosition(Enum):
    """Player positions"""
    GOALKEEPER = "goalkeeper"
    DEFENDER = "defender"
    MIDFIELDER = "midfielder"
    FORWARD = "forward"
    WINGER = "winger"

class PlayerLevel(Enum):
    """Player skill levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PROFESSIONAL = "professional"

@dataclass
class ScoreComponent:
    """Individual score component"""
    name: str
    value: float
    confidence: float
    raw_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PlayerProfile:
    """Player profile for personalization"""
    age: Optional[int] = None
    position: Optional[PlayerPosition] = None
    level: Optional[PlayerLevel] = None
    objectives: List[str] = field(default_factory=list)
    custom_weights: Optional[Dict[str, float]] = None

@dataclass
class ConfidenceInterval:
    """Confidence interval for scores"""
    lower: float
    upper: float
    mean: float
    std_dev: float

class ScoreAggregator:
    """
    Advanced score aggregator for football analysis
    Provides unified scoring system with adaptive weighting
    """
    
    DEFAULT_WEIGHTS = {
        "training": {
            "biomechanics": 0.35,
            "technical": 0.45,
            "tactical": 0.10,
            "physical": 0.10
        },
        "match": {
            "biomechanics": 0.15,
            "technical": 0.30,
            "tactical": 0.35,
            "physical": 0.20
        },
        "drill": {
            "biomechanics": 0.40,
            "technical": 0.50,
            "tactical": 0.05,
            "physical": 0.05
        },
        "freestyle": {
            "biomechanics": 0.30,
            "technical": 0.60,
            "tactical": 0.05,
            "physical": 0.05
        }
    }
    
    POSITION_MODIFIERS = {
        PlayerPosition.GOALKEEPER: {
            "biomechanics": 1.2,
            "technical": 1.1,
            "tactical": 0.8,
            "physical": 1.0
        },
        PlayerPosition.DEFENDER: {
            "biomechanics": 1.0,
            "technical": 0.9,
            "tactical": 1.3,
            "physical": 1.2
        },
        PlayerPosition.MIDFIELDER: {
            "biomechanics": 1.0,
            "technical": 1.2,
            "tactical": 1.3,
            "physical": 1.0
        },
        PlayerPosition.FORWARD: {
            "biomechanics": 1.1,
            "technical": 1.3,
            "tactical": 0.9,
            "physical": 1.1
        },
        PlayerPosition.WINGER: {
            "biomechanics": 1.2,
            "technical": 1.2,
            "tactical": 1.0,
            "physical": 1.3
        }
    }
    
    def __init__(self):
        self.logger = logger
        self.registered_scores: Dict[str, ScoreComponent] = {}
        self.score_history: List[Dict[str, Any]] = []
        self.percentile_data: Dict[str, List[float]] = {
            "biomechanics": [],
            "technical": [],
            "tactical": [],
            "physical": []
        }
    
    def register_score(self, component_name: str, value: float, 
                      confidence: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a score component from analysis modules
        
        Args:
            component_name: Name of the component (biomechanics, technical, tactical, physical)
            value: Raw score value
            confidence: Confidence level (0.0 to 1.0)
            metadata: Additional component metadata
        """
        if metadata is None:
            metadata = {}
            
        normalized_value = self._normalize_score(value, component_name)
        
        self.registered_scores[component_name] = ScoreComponent(
            name=component_name,
            value=normalized_value,
            confidence=confidence,
            raw_value=value,
            metadata=metadata
        )
        
        # Update percentile data
        if component_name in self.percentile_data:
            self.percentile_data[component_name].append(normalized_value)
        
        self.logger.info(f"Registered score for {component_name}: {normalized_value:.2f} (confidence: {confidence:.2f})")
    
    def get_final_score(self, video_type: VideoType = VideoType.TRAINING,
                       player_profile: Optional[PlayerProfile] = None) -> Dict[str, Any]:
        """
        Calculate final aggregated score with adaptive weighting
        
        Args:
            video_type: Type of video analysis
            player_profile: Player profile for personalization
            
        Returns:
            Final score with breakdown and confidence intervals
        """
        if not self.registered_scores:
            return self._empty_score()
        
        weights = self._calculate_adaptive_weights(video_type, player_profile)
        weighted_scores = {}
        total_weight = 0
        total_weighted_score = 0
        
        # Calculate weighted scores
        for component_name, score_component in self.registered_scores.items():
            if component_name in weights:
                weight = weights[component_name]
                weighted_score = score_component.value * weight
                weighted_scores[component_name] = {
                    "score": score_component.value,
                    "weight": weight,
                    "weighted_score": weighted_score,
                    "confidence": score_component.confidence,
                    "raw_value": score_component.raw_value
                }
                
                total_weighted_score += weighted_score * score_component.confidence
                total_weight += weight * score_component.confidence
        
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval()
        
        # Detect strengths and weaknesses
        strengths, weaknesses = self._analyze_strengths_weaknesses(weighted_scores)
        
        # Calculate percentiles
        percentiles = self._calculate_percentiles()
        
        result = {
            "final_score": round(final_score, 2),
            "breakdown": weighted_scores,
            "confidence_interval": confidence_interval,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "percentiles": percentiles,
            "video_type": video_type.value,
            "timestamp": datetime.now().isoformat(),
            "total_components": len(self.registered_scores)
        }
        
        # Store in history
        self.score_history.append(result)
        
        return result
    
    def get_breakdown(self) -> Dict[str, Any]:
        """
        Get detailed breakdown of all registered scores
        
        Returns:
            Detailed component breakdown
        """
        breakdown = {}
        for component_name, score_component in self.registered_scores.items():
            breakdown[component_name] = {
                "score": score_component.value,
                "confidence": score_component.confidence,
                "raw_value": score_component.raw_value,
                "metadata": score_component.metadata
            }
        
        return {
            "components": breakdown,
            "total_registered": len(self.registered_scores),
            "average_confidence": np.mean([s.confidence for s in self.registered_scores.values()]) if self.registered_scores else 0
        }
    
    def export_report(self, include_history: bool = False) -> Dict[str, Any]:
        """
        Export comprehensive analysis report
        
        Args:
            include_history: Whether to include score history
            
        Returns:
            Complete analysis report
        """
        current_breakdown = self.get_breakdown()
        
        report = {
            "summary": {
                "total_analyses": len(self.score_history),
                "components_analyzed": list(self.registered_scores.keys()),
                "average_confidence": current_breakdown.get("average_confidence", 0)
            },
            "current_scores": current_breakdown,
            "percentile_analysis": self._get_percentile_analysis(),
            "trends": self._analyze_trends(),
            "recommendations": self._generate_recommendations(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        if include_history:
            report["history"] = self.score_history
        
        return report
    
    def _normalize_score(self, value: float, component_name: str) -> float:
        """Normalize score to 0-100 scale"""
        # Basic normalization - can be enhanced based on component specifics
        if value < 0:
            return 0.0
        elif value > 100:
            return 100.0
        return float(value)
    
    def _calculate_adaptive_weights(self, video_type: VideoType, 
                                  player_profile: Optional[PlayerProfile]) -> Dict[str, float]:
        """Calculate adaptive weights based on context"""
        base_weights = self.DEFAULT_WEIGHTS.get(video_type.value, self.DEFAULT_WEIGHTS["training"])
        
        if not player_profile:
            return base_weights
        
        # Apply custom weights if specified
        if player_profile.custom_weights:
            return player_profile.custom_weights
        
        # Apply position modifiers
        if player_profile.position:
            position_modifiers = self.POSITION_MODIFIERS.get(player_profile.position, {})
            for component in base_weights:
                if component in position_modifiers:
                    base_weights[component] *= position_modifiers[component]
        
        # Apply age adjustments
        if player_profile.age:
            base_weights = self._apply_age_adjustments(base_weights, player_profile.age)
        
        # Apply level adjustments
        if player_profile.level:
            base_weights = self._apply_level_adjustments(base_weights, player_profile.level)
        
        # Normalize weights to sum to 1
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            base_weights = {k: v / total_weight for k, v in base_weights.items()}
        
        return base_weights
    
    def _apply_age_adjustments(self, weights: Dict[str, float], age: int) -> Dict[str, float]:
        """Apply age-based weight adjustments"""
        if age < 16:  # Youth focus on technique and biomechanics
            weights["technical"] *= 1.2
            weights["biomechanics"] *= 1.1
            weights["tactical"] *= 0.8
        elif age > 30:  # Veteran focus on tactical and experience
            weights["tactical"] *= 1.2
            weights["physical"] *= 0.9
            weights["technical"] *= 1.1
        
        return weights
    
    def _apply_level_adjustments(self, weights: Dict[str, float], level: PlayerLevel) -> Dict[str, float]:
        """Apply skill level adjustments"""
        if level == PlayerLevel.BEGINNER:
            weights["biomechanics"] *= 1.3
            weights["technical"] *= 1.2
            weights["tactical"] *= 0.7
        elif level == PlayerLevel.PROFESSIONAL:
            weights["tactical"] *= 1.2
            weights["physical"] *= 1.1
        
        return weights
    
    def _calculate_confidence_interval(self, confidence_level: float = 0.95) -> ConfidenceInterval:
        """Calculate confidence interval for aggregated score"""
        if not self.registered_scores:
            return ConfidenceInterval(0, 0, 0, 0)
        
        scores = [s.value for s in self.registered_scores.values()]
        confidences = [s.confidence for s in self.registered_scores.values()]
        
        # Weight scores by confidence
        weighted_scores = [s * c for s, c in zip(scores, confidences)]
        mean_score = np.mean(weighted_scores)
        std_dev = np.std(weighted_scores)
        
        # Calculate interval
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        margin = z_score * (std_dev / np.sqrt(len(scores)))
        
        return ConfidenceInterval(
            lower=max(0, mean_score - margin),
            upper=min(100, mean_score + margin),
            mean=mean_score,
            std_dev=std_dev
        )
    
    def _analyze_strengths_weaknesses(self, weighted_scores: Dict[str, Any], 
                                    threshold: float = 10.0) -> Tuple[List[str], List[str]]:
        """Detect player strengths and weaknesses"""
        if not weighted_scores:
            return [], []
        
        scores_list = [(name, data["score"]) for name, data in weighted_scores.items()]
        mean_score = np.mean([score for _, score in scores_list])
        
        strengths = [name for name, score in scores_list if score > mean_score + threshold]
        weaknesses = [name for name, score in scores_list if score < mean_score - threshold]
        
        return strengths, weaknesses
    
    def _calculate_percentiles(self) -> Dict[str, float]:
        """Calculate percentile rankings for current scores"""
        percentiles = {}
        
        for component_name, score_component in self.registered_scores.items():
            if component_name in self.percentile_data and self.percentile_data[component_name]:
                data = self.percentile_data[component_name]
                percentile = (sum(1 for x in data if x <= score_component.value) / len(data)) * 100
                percentiles[component_name] = round(percentile, 1)
        
        return percentiles
    
    def _get_percentile_analysis(self) -> Dict[str, Any]:
        """Get detailed percentile analysis"""
        analysis = {}
        
        for component_name, data in self.percentile_data.items():
            if data:
                analysis[component_name] = {
                    "mean": np.mean(data),
                    "median": np.median(data),
                    "std_dev": np.std(data),
                    "min": min(data),
                    "max": max(data),
                    "sample_size": len(data)
                }
        
        return analysis
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze score trends over time"""
        if len(self.score_history) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        recent_scores = [h["final_score"] for h in self.score_history[-5:]]
        trend = "improving" if recent_scores[-1] > recent_scores[0] else "declining"
        
        return {
            "trend": trend,
            "recent_average": np.mean(recent_scores),
            "improvement_rate": (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations"""
        if not self.registered_scores:
            return ["No data available for recommendations"]
        
        recommendations = []
        
        # Find lowest scoring component
        lowest_component = min(self.registered_scores.items(), key=lambda x: x[1].value)
        if lowest_component[1].value < 60:
            recommendations.append(f"Focus on improving {lowest_component[0]} skills")
        
        # Check confidence levels
        low_confidence = [name for name, score in self.registered_scores.items() 
                         if score.confidence < 0.7]
        if low_confidence:
            recommendations.append(f"Collect more data for: {', '.join(low_confidence)}")
        
        return recommendations
    
    def _empty_score(self) -> Dict[str, Any]:
        """Return empty score structure"""
        return {
            "final_score": 0,
            "breakdown": {},
            "confidence_interval": ConfidenceInterval(0, 0, 0, 0),
            "strengths": [],
            "weaknesses": [],
            "percentiles": {},
            "video_type": "unknown",
            "timestamp": datetime.now().isoformat(),
            "total_components": 0
        }
    
    def clear_scores(self) -> None:
        """Clear all registered scores for new analysis"""
        self.registered_scores.clear()
        self.logger.info("Cleared all registered scores")
    
    def get_score_clustering(self) -> Dict[str, Any]:
        """Perform player profile clustering based on scores"""
        if len(self.score_history) < 5:
            return {"message": "Insufficient data for clustering"}
        
        # Simple clustering based on score patterns
        scores_matrix = []
        for history in self.score_history[-10:]:  # Last 10 analyses
            breakdown = history.get("breakdown", {})
            score_vector = [breakdown.get(comp, {}).get("score", 0) 
                          for comp in ["biomechanics", "technical", "tactical", "physical"]]
            scores_matrix.append(score_vector)
        
        if not scores_matrix:
            return {"message": "No score data for clustering"}
        
        # Calculate similarity to common player types
        avg_scores = np.mean(scores_matrix, axis=0)
        
        player_types = {
            "technical_specialist": [70, 85, 60, 65],
            "tactical_expert": [65, 70, 90, 70],
            "physical_dominant": [75, 65, 65, 90],
            "well_rounded": [75, 75, 75, 75]
        }
        
        similarities = {}
        for ptype, template in player_types.items():
            similarity = 100 - np.linalg.norm(np.array(avg_scores) - np.array(template))
            similarities[ptype] = max(0, similarity)
        
        best_match = max(similarities, key=similarities.get)
        
        return {
            "player_type": best_match,
            "similarities": similarities,
            "average_scores": {
                "biomechanics": avg_scores[0],
                "technical": avg_scores[1],
                "tactical": avg_scores[2],
                "physical": avg_scores[3]
            }
        }