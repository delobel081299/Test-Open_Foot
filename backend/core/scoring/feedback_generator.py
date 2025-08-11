"""
Intelligent feedback generator for football analysis
Generates personalized, constructive feedback based on analysis scores
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import json
import requests
from pathlib import Path
from datetime import datetime
import numpy as np

from .score_aggregator import ScoreAggregator, PlayerProfile, VideoType, PlayerLevel, PlayerPosition
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class FeedbackTone(Enum):
    """Feedback tone styles"""
    ENCOURAGING = "encouraging"
    ANALYTICAL = "analytical"
    MOTIVATIONAL = "motivational"
    PROFESSIONAL = "professional"
    YOUTH_FRIENDLY = "youth_friendly"

class Priority(Enum):
    """Feedback priority levels"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class FeedbackItem:
    """Individual feedback item"""
    category: str
    title: str
    message: str
    priority: Priority
    score_reference: float
    improvement_tip: str
    visual_example: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeedbackReport:
    """Complete feedback report"""
    player_id: str
    total_feedbacks: int
    priority_feedbacks: List[FeedbackItem]
    secondary_feedbacks: List[FeedbackItem]
    overall_tone: FeedbackTone
    generation_timestamp: str
    summary: str
    next_focus_areas: List[str]

class FeedbackGenerator:
    """
    Intelligent feedback generator for football performance analysis
    Uses templates, LLM integration, and adaptive personalization
    """
    
    # Base feedback templates by category
    FEEDBACK_TEMPLATES = {
        "technical": {
            "excellent": [
                "Excellente qualité de passe, maintenir l'angle d'ouverture du pied",
                "Très bonne technique de frappe, précision remarquable",
                "Premier contrôle de qualité, orientation parfaite vers l'espace",
                "Dribbles efficaces, changements de rythme bien maîtrisés",
                "Très bonne qualité technique dans les gestes, fluidité exemplaire"
            ],
            "good": [
                "Bonne technique générale, quelques aspects à peaufiner",
                "Contrôles corrects, chercher plus de variété dans l'orientation",
                "Frappes convenables, travailler la régularité de la précision",
                "Technique de passe satisfaisante, améliorer la vision du jeu"
            ],
            "needs_improvement": [
                "Attention au timing de la frappe, anticiper 0.2s plus tôt",
                "Premier contrôle perfectible, orienter vers l'espace libre",
                "Technique de passe à améliorer, attention au poids du ballon",
                "Contrôles trop lourds, travailler la première touche",
                "Frappes imprécises, revoir le placement d'appui"
            ]
        },
        "tactical": {
            "excellent": [
                "Excellentes prises de décision sous pression",
                "Bon timing des appels, continuer à étirer la défense",
                "Très bonne lecture du jeu, anticipation remarquable",
                "Positionnement offensif intelligent, créateur d'espaces",
                "Excellente communication défensive, leadership visible"
            ],
            "good": [
                "Bonnes décisions tactiques, maintenir cette lucidité",
                "Positionnement correct, chercher plus d'initiative",
                "Lecture du jeu satisfaisante, être plus audacieux",
                "Bon sens tactique défensif, améliorer les transitions"
            ],
            "needs_improvement": [
                "Positionnement défensif à améliorer, réduire distance avec #6",
                "Timing des appels perfectible, observer les défenseurs",
                "Prises de décision hâtives, prendre le temps d'analyser",
                "Couverture défensive insuffisante, surveiller les zones libres",
                "Transitions trop lentes, accélérer la prise de décision"
            ]
        },
        "physical": {
            "excellent": [
                "Bonne intensité maintenue, pic à {speed} km/h",
                "Excellent volume de course : {distance} km",
                "Très bonne endurance, intensité constante sur 90 minutes",
                "Vitesse de pointe remarquable, explosivité excellente",
                "Récupération rapide entre les efforts, condition physique optimale"
            ],
            "good": [
                "Condition physique correcte, maintenir les efforts",
                "Volume de course satisfaisant : {distance} km",
                "Intensité générale bonne, quelques baisses de régime",
                "Vitesse convenable, travailler l'explosivité"
            ],
            "needs_improvement": [
                "Améliorer fréquence des sprints courts",
                "Volume de course insuffisant : {distance} km, viser 2 km de plus",
                "Intensité en baisse après 60 minutes, travailler l'endurance",
                "Vitesse de pointe limitée : {speed} km/h, développer l'explosivité",
                "Récupération lente entre les efforts, améliorer la condition physique"
            ]
        },
        "biomechanics": {
            "excellent": [
                "Biomécanique de course excellente, efficacité maximale",
                "Gestuelle de frappe parfaite, optimisation des forces",
                "Équilibre remarquable dans les changements de direction",
                "Coordination excellente, fluidité des mouvements",
                "Posture idéale, économie d'énergie optimisée"
            ],
            "good": [
                "Biomécanique correcte, quelques optimisations possibles",
                "Gestuelle satisfaisante, chercher plus de fluidité",
                "Équilibre global bon, améliorer les appuis",
                "Coordination correcte, travailler la synchronisation"
            ],
            "needs_improvement": [
                "Biomécanique de course à améliorer, optimiser la foulée",
                "Gestuelle de frappe perfectible, revoir le placement d'appui",
                "Déséquilibres fréquents, renforcer la proprioception",
                "Coordination défaillante, travailler la fluidité des gestes",
                "Posture incorrecte, attention au placement du corps"
            ]
        }
    }
    
    # Tone modifiers for different player profiles
    TONE_MODIFIERS = {
        FeedbackTone.ENCOURAGING: {
            "prefix": ["Bravo pour", "Félicitations pour", "Continue comme ça pour"],
            "connection": ["et", "puis", "également"],
            "improvement": ["Tu peux encore progresser en", "Prochaine étape", "Pour aller plus loin"]
        },
        FeedbackTone.ANALYTICAL: {
            "prefix": ["Analyse technique", "Observation", "Constat"],
            "connection": ["par ailleurs", "en outre", "de plus"],
            "improvement": ["Axe d'amélioration", "Point de progression", "Développement requis"]
        },
        FeedbackTone.MOTIVATIONAL: {
            "prefix": ["Excellent travail sur", "Superbe performance en", "Très impressionnant"],
            "connection": ["maintenant", "désormais", "à présent"],
            "improvement": ["Défi suivant", "Objectif progression", "Prochaine conquête"]
        },
        FeedbackTone.PROFESSIONAL: {
            "prefix": ["Performance", "Résultat", "Niveau"],
            "connection": ["néanmoins", "cependant", "toutefois"],
            "improvement": ["Recommandation", "Suggestion d'entraînement", "Plan de développement"]
        },
        FeedbackTone.YOUTH_FRIENDLY: {
            "prefix": ["Super boulot sur", "Top niveau pour", "Bien joué pour"],
            "connection": ["et en plus", "aussi", "et puis"],
            "improvement": ["Tu peux devenir encore meilleur en", "Prochaine mission", "Nouveau défi"]
        }
    }
    
    # Visual example mappings
    VISUAL_EXAMPLES = {
        "technique_passing": "/static/gifs/passing_technique.gif",
        "technique_shooting": "/static/gifs/shooting_form.gif",
        "technique_control": "/static/gifs/first_touch.gif",
        "tactical_positioning": "/static/gifs/positioning.gif",
        "tactical_movement": "/static/gifs/off_ball_movement.gif",
        "physical_sprint": "/static/gifs/sprint_technique.gif",
        "physical_endurance": "/static/gifs/endurance_training.gif",
        "biomechanics_running": "/static/gifs/running_form.gif",
        "biomechanics_balance": "/static/gifs/balance_drills.gif"
    }
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.logger = logger
        self.ollama_url = ollama_url
        self.ollama_available = self._check_ollama_availability()
        
        # Cache for generated feedback variations
        self.feedback_cache: Dict[str, str] = {}
        
        self.logger.info(f"FeedbackGenerator initialized. Ollama available: {self.ollama_available}")
    
    def generate_feedback(self, score_data: Dict[str, Any], 
                         player_profile: Optional[PlayerProfile] = None,
                         max_feedbacks: int = 4) -> FeedbackReport:
        """
        Generate comprehensive feedback report based on analysis scores
        
        Args:
            score_data: Score data from ScoreAggregator
            player_profile: Player profile for personalization
            max_feedbacks: Maximum number of priority feedbacks (3-5)
            
        Returns:
            Complete feedback report with prioritized items
        """
        self.logger.info("Generating feedback report")
        
        # Step 1: Analyze scores in detail
        analysis_results = self._analyze_scores_detailed(score_data)
        
        # Step 2: Identify priorities
        priorities = self._identify_priorities(analysis_results, max_feedbacks)
        
        # Step 3: Generate feedback items
        all_feedbacks = self._generate_feedback_items(analysis_results, player_profile)
        
        # Step 4: Prioritize and limit feedbacks
        priority_feedbacks, secondary_feedbacks = self._prioritize_feedbacks(
            all_feedbacks, priorities, max_feedbacks
        )
        
        # Step 5: Personalize tone
        tone = self._determine_feedback_tone(player_profile)
        priority_feedbacks = self._personalize_tone(priority_feedbacks, tone)
        
        # Step 6: Add visual examples
        priority_feedbacks = self._add_visual_examples(priority_feedbacks)
        
        # Step 7: Generate summary and next focus areas
        summary = self._generate_summary(priority_feedbacks, score_data)
        next_focus = self._identify_next_focus_areas(analysis_results)
        
        report = FeedbackReport(
            player_id=score_data.get("player_id", "unknown"),
            total_feedbacks=len(all_feedbacks),
            priority_feedbacks=priority_feedbacks,
            secondary_feedbacks=secondary_feedbacks,
            overall_tone=tone,
            generation_timestamp=datetime.now().isoformat(),
            summary=summary,
            next_focus_areas=next_focus
        )
        
        self.logger.info(f"Generated feedback report: {len(priority_feedbacks)} priority items")
        return report
    
    def _analyze_scores_detailed(self, score_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed analysis of scores to identify patterns"""
        breakdown = score_data.get("breakdown", {})
        
        analysis = {
            "scores": {},
            "patterns": {},
            "concerns": [],
            "strengths": [],
            "improvement_areas": []
        }
        
        # Analyze each component
        for component, data in breakdown.items():
            score = data.get("score", 0)
            confidence = data.get("confidence", 1.0)
            
            analysis["scores"][component] = {
                "value": score,
                "confidence": confidence,
                "level": self._categorize_score(score),
                "reliability": "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
            }
            
            # Identify patterns
            if score >= 80:
                analysis["strengths"].append(component)
            elif score <= 50:
                analysis["improvement_areas"].append(component)
            
            if confidence < 0.6:
                analysis["concerns"].append(f"Low confidence in {component} assessment")
        
        # Detect score patterns
        scores_list = [data.get("score", 0) for data in breakdown.values()]
        if scores_list:
            analysis["patterns"]["consistency"] = np.std(scores_list) < 15  # Low std = consistent
            analysis["patterns"]["overall_level"] = np.mean(scores_list)
            analysis["patterns"]["peak_performance"] = max(scores_list)
            analysis["patterns"]["weakest_area"] = min(scores_list)
        
        return analysis
    
    def _identify_priorities(self, analysis: Dict[str, Any], max_items: int) -> List[str]:
        """Identify priority areas for feedback"""
        priorities = []
        
        # Critical: Very low scores (< 40)
        for component, data in analysis["scores"].items():
            if data["value"] < 40:
                priorities.append(f"critical_{component}")
        
        # High: Low scores (40-60) or inconsistent performance
        for component, data in analysis["scores"].items():
            if 40 <= data["value"] <= 60:
                priorities.append(f"high_{component}")
        
        # Medium: Good scores that can be excellent (60-80)
        for component, data in analysis["scores"].items():
            if 60 <= data["value"] <= 80:
                priorities.append(f"medium_{component}")
        
        # Strengths to maintain (> 80)
        for component in analysis["strengths"]:
            priorities.append(f"strength_{component}")
        
        return priorities[:max_items * 2]  # More than needed for selection
    
    def _generate_feedback_items(self, analysis: Dict[str, Any], 
                                player_profile: Optional[PlayerProfile]) -> List[FeedbackItem]:
        """Generate individual feedback items from analysis"""
        feedbacks = []
        
        for component, score_data in analysis["scores"].items():
            score = score_data["value"]
            level = score_data["level"]
            
            # Select appropriate template
            templates = self.FEEDBACK_TEMPLATES.get(component, {})
            template_category = templates.get(level, templates.get("needs_improvement", []))
            
            if not template_category:
                continue
            
            # Select and customize template
            base_message = random.choice(template_category)
            
            # Add specific data if available
            if "{speed}" in base_message and "physical" in component:
                speed = analysis.get("metadata", {}).get("max_speed", 25.0)
                base_message = base_message.format(speed=speed)
            
            if "{distance}" in base_message and "physical" in component:
                distance = analysis.get("metadata", {}).get("total_distance", 8.0)
                base_message = base_message.format(distance=distance)
            
            # Generate improvement tip
            improvement_tip = self._generate_improvement_tip(component, score, level)
            
            # Determine priority
            priority = self._calculate_item_priority(score, score_data["confidence"])
            
            # Use LLM for variation if available
            if self.ollama_available:
                base_message = self._enhance_with_llm(base_message, component, player_profile)
            
            feedback_item = FeedbackItem(
                category=component,
                title=f"{component.capitalize()} Performance",
                message=base_message,
                priority=priority,
                score_reference=score,
                improvement_tip=improvement_tip,
                metadata={"confidence": score_data["confidence"]}
            )
            
            feedbacks.append(feedback_item)
        
        return feedbacks
    
    def _prioritize_feedbacks(self, feedbacks: List[FeedbackItem], 
                             priorities: List[str], max_priority: int) -> Tuple[List[FeedbackItem], List[FeedbackItem]]:
        """Prioritize feedbacks and split into priority and secondary"""
        
        # Sort by priority then by score (lowest first for improvement areas)
        priority_order = {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}
        
        sorted_feedbacks = sorted(
            feedbacks, 
            key=lambda x: (priority_order[x.priority], x.score_reference if x.priority != Priority.CRITICAL else -x.score_reference)
        )
        
        # Split into priority and secondary
        priority_feedbacks = sorted_feedbacks[:max_priority]
        secondary_feedbacks = sorted_feedbacks[max_priority:]
        
        return priority_feedbacks, secondary_feedbacks
    
    def _determine_feedback_tone(self, player_profile: Optional[PlayerProfile]) -> FeedbackTone:
        """Determine appropriate feedback tone based on player profile"""
        if not player_profile:
            return FeedbackTone.ANALYTICAL
        
        # Age-based tone selection
        if player_profile.age and player_profile.age < 16:
            return FeedbackTone.YOUTH_FRIENDLY
        elif player_profile.age and player_profile.age < 20:
            return FeedbackTone.ENCOURAGING
        
        # Level-based tone selection
        if player_profile.level == PlayerLevel.PROFESSIONAL:
            return FeedbackTone.PROFESSIONAL
        elif player_profile.level == PlayerLevel.BEGINNER:
            return FeedbackTone.ENCOURAGING
        
        # Default motivational for intermediate players
        return FeedbackTone.MOTIVATIONAL
    
    def _personalize_tone(self, feedbacks: List[FeedbackItem], tone: FeedbackTone) -> List[FeedbackItem]:
        """Apply tone personalization to feedback messages"""
        modifiers = self.TONE_MODIFIERS.get(tone, self.TONE_MODIFIERS[FeedbackTone.ANALYTICAL])
        
        for feedback in feedbacks:
            # Apply tone modifiers to messages
            if feedback.priority == Priority.CRITICAL or feedback.score_reference < 50:
                # Improvement-focused message
                improvement_prefix = random.choice(modifiers["improvement"])
                feedback.message = f"{improvement_prefix}: {feedback.message}"
            else:
                # Positive reinforcement
                positive_prefix = random.choice(modifiers["prefix"])
                feedback.message = f"{positive_prefix} {feedback.message.lower()}"
        
        return feedbacks
    
    def _add_visual_examples(self, feedbacks: List[FeedbackItem]) -> List[FeedbackItem]:
        """Add visual examples/GIFs to feedback items"""
        for feedback in feedbacks:
            category = feedback.category.lower()
            
            # Map category to visual example
            visual_key = None
            if "technical" in category:
                if "pass" in feedback.message.lower():
                    visual_key = "technique_passing"
                elif "frappe" in feedback.message.lower() or "shoot" in feedback.message.lower():
                    visual_key = "technique_shooting"
                elif "contrôl" in feedback.message.lower():
                    visual_key = "technique_control"
            elif "tactical" in category:
                if "position" in feedback.message.lower():
                    visual_key = "tactical_positioning"
                else:
                    visual_key = "tactical_movement"
            elif "physical" in category:
                if "sprint" in feedback.message.lower() or "vitesse" in feedback.message.lower():
                    visual_key = "physical_sprint"
                else:
                    visual_key = "physical_endurance"
            elif "biomechanics" in category:
                if "course" in feedback.message.lower():
                    visual_key = "biomechanics_running"
                else:
                    visual_key = "biomechanics_balance"
            
            if visual_key and visual_key in self.VISUAL_EXAMPLES:
                feedback.visual_example = self.VISUAL_EXAMPLES[visual_key]
        
        return feedbacks
    
    def _generate_summary(self, priority_feedbacks: List[FeedbackItem], 
                         score_data: Dict[str, Any]) -> str:
        """Generate overall summary of the feedback report"""
        final_score = score_data.get("final_score", 0)
        strengths = score_data.get("strengths", [])
        weaknesses = score_data.get("weaknesses", [])
        
        if final_score >= 80:
            level_desc = "Excellente performance"
        elif final_score >= 60:
            level_desc = "Bonne performance"
        elif final_score >= 40:
            level_desc = "Performance moyenne"
        else:
            level_desc = "Performance à améliorer"
        
        summary_parts = [f"{level_desc} avec un score global de {final_score}/100."]
        
        if strengths:
            summary_parts.append(f"Points forts identifiés : {', '.join(strengths)}.")
        
        if weaknesses:
            summary_parts.append(f"Axes d'amélioration prioritaires : {', '.join(weaknesses)}.")
        
        summary_parts.append(f"{len(priority_feedbacks)} recommandations prioritaires identifiées.")
        
        return " ".join(summary_parts)
    
    def _identify_next_focus_areas(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify next focus areas for development"""
        focus_areas = []
        
        # Primary focus: improvement areas
        for area in analysis.get("improvement_areas", []):
            focus_areas.append(f"Développement {area}")
        
        # Secondary focus: areas with low confidence
        for component, data in analysis["scores"].items():
            if data["reliability"] == "low":
                focus_areas.append(f"Collecte de données supplémentaires pour {component}")
        
        # Tertiary focus: maintaining strengths
        for strength in analysis.get("strengths", []):
            focus_areas.append(f"Maintien du niveau excellent en {strength}")
        
        return focus_areas[:5]  # Limit to top 5
    
    def _categorize_score(self, score: float) -> str:
        """Categorize score into performance level"""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        else:
            return "needs_improvement"
    
    def _calculate_item_priority(self, score: float, confidence: float) -> Priority:
        """Calculate priority level for feedback item"""
        if score < 40:
            return Priority.CRITICAL
        elif score < 60:
            return Priority.HIGH
        elif score < 80:
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def _generate_improvement_tip(self, component: str, score: float, level: str) -> str:
        """Generate specific improvement tip for component"""
        tips = {
            "technical": {
                "needs_improvement": "Concentrez-vous sur les exercices de technique individuelle 15min/jour",
                "good": "Variez les exercices techniques pour gagner en polyvalence",
                "excellent": "Maintenez votre niveau par une pratique régulière"
            },
            "tactical": {
                "needs_improvement": "Étudiez les vidéos de matches pour améliorer la lecture du jeu",
                "good": "Travaillez les situations de match à l'entraînement",
                "excellent": "Développez votre leadership et communication sur le terrain"
            },
            "physical": {
                "needs_improvement": "Plan d'entraînement physique spécifique recommandé",
                "good": "Maintenez votre condition avec des séances régulières",
                "excellent": "Optimisez votre récupération et préparation physique"
            },
            "biomechanics": {
                "needs_improvement": "Travail avec un préparateur physique sur la gestuelle",
                "good": "Exercices de proprioception pour optimiser les mouvements",
                "excellent": "Continuez le travail de prévention des blessures"
            }
        }
        
        return tips.get(component, {}).get(level, "Continuez vos efforts d'amélioration")
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama LLM service is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/version", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _enhance_with_llm(self, message: str, component: str, 
                         player_profile: Optional[PlayerProfile]) -> str:
        """Enhance feedback message using local LLM (Ollama)"""
        if not self.ollama_available:
            return message
        
        # Create cache key
        cache_key = f"{component}_{message[:20]}_{player_profile.level if player_profile else 'none'}"
        
        if cache_key in self.feedback_cache:
            return self.feedback_cache[cache_key]
        
        try:
            # Prepare prompt for LLM
            age_context = f"âgé de {player_profile.age} ans" if player_profile and player_profile.age else "joueur"
            level_context = f"niveau {player_profile.level.value}" if player_profile and player_profile.level else "intermédiaire"
            
            prompt = f"""
            Reformule ce feedback de football pour un {age_context} de {level_context}:
            "{message}"
            
            Critères:
            - Reste constructif et précis
            - Garde la même longueur
            - Utilise un vocabulaire adapté au niveau
            - Reste motivant
            
            Réponse reformulée:
            """
            
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama2",  # ou le modèle de votre choix
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 100
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                # Clean and validate result
                if result and len(result) > 10 and len(result) < 200:
                    self.feedback_cache[cache_key] = result
                    return result
        
        except Exception as e:
            self.logger.warning(f"LLM enhancement failed: {e}")
        
        return message  # Fallback to original
    
    def export_feedback_json(self, report: FeedbackReport, filepath: str) -> None:
        """Export feedback report to JSON file"""
        try:
            # Convert dataclasses to dict
            report_dict = {
                "player_id": report.player_id,
                "total_feedbacks": report.total_feedbacks,
                "priority_feedbacks": [
                    {
                        "category": f.category,
                        "title": f.title,
                        "message": f.message,
                        "priority": f.priority.value,
                        "score_reference": f.score_reference,
                        "improvement_tip": f.improvement_tip,
                        "visual_example": f.visual_example,
                        "timestamp": f.timestamp
                    }
                    for f in report.priority_feedbacks
                ],
                "secondary_feedbacks": [
                    {
                        "category": f.category,
                        "title": f.title,
                        "message": f.message,
                        "priority": f.priority.value,
                        "score_reference": f.score_reference,
                        "improvement_tip": f.improvement_tip,
                        "visual_example": f.visual_example,
                        "timestamp": f.timestamp
                    }
                    for f in report.secondary_feedbacks
                ],
                "overall_tone": report.overall_tone.value,
                "generation_timestamp": report.generation_timestamp,
                "summary": report.summary,
                "next_focus_areas": report.next_focus_areas
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Feedback report exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export feedback report: {e}")
    
    def get_feedback_statistics(self, reports: List[FeedbackReport]) -> Dict[str, Any]:
        """Generate statistics from multiple feedback reports"""
        if not reports:
            return {"message": "No reports available"}
        
        stats = {
            "total_reports": len(reports),
            "categories_distribution": {},
            "priority_distribution": {},
            "average_scores": {},
            "improvement_trends": {},
            "common_strengths": {},
            "common_weaknesses": {}
        }
        
        # Analyze category distribution
        for report in reports:
            for feedback in report.priority_feedbacks:
                category = feedback.category
                stats["categories_distribution"][category] = stats["categories_distribution"].get(category, 0) + 1
                stats["priority_distribution"][feedback.priority.value] = stats["priority_distribution"].get(feedback.priority.value, 0) + 1
        
        return stats