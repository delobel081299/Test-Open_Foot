"""
Example usage of the FeedbackGenerator with ScoreAggregator
Demonstrates complete feedback generation pipeline
"""
from .score_aggregator import ScoreAggregator, PlayerProfile, VideoType, PlayerLevel, PlayerPosition
from .feedback_generator import FeedbackGenerator
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

def generate_example_feedback():
    """Example of complete feedback generation workflow"""
    
    # Initialize systems
    score_aggregator = ScoreAggregator()
    feedback_generator = FeedbackGenerator()
    
    # Simulate score registration from different analysis modules
    logger.info("Registering scores from analysis modules...")
    
    # Biomechanics module scores
    score_aggregator.register_score("biomechanics", 75.5, confidence=0.9, 
                                   metadata={"running_efficiency": 85, "balance_score": 70})
    
    # Technical module scores  
    score_aggregator.register_score("technical", 62.0, confidence=0.85,
                                   metadata={"passing_accuracy": 78, "shooting_precision": 55, "ball_control": 68})
    
    # Tactical module scores
    score_aggregator.register_score("tactical", 45.0, confidence=0.75,
                                   metadata={"positioning": 40, "decision_making": 50, "awareness": 45})
    
    # Physical module scores
    score_aggregator.register_score("physical", 82.0, confidence=0.95,
                                   metadata={"max_speed": 28.5, "total_distance": 8.2, "sprints": 15})
    
    # Create player profile
    player_profile = PlayerProfile(
        age=19,
        position=PlayerPosition.MIDFIELDER,
        level=PlayerLevel.INTERMEDIATE,
        objectives=["improve_tactical_awareness", "enhance_shooting"]
    )
    
    # Generate final aggregated score
    logger.info("Generating final score with adaptive weighting...")
    final_score = score_aggregator.get_final_score(
        video_type=VideoType.MATCH,
        player_profile=player_profile
    )
    
    print(f"Final Score: {final_score['final_score']}/100")
    print(f"Strengths: {final_score['strengths']}")
    print(f"Weaknesses: {final_score['weaknesses']}")
    
    # Generate intelligent feedback
    logger.info("Generating intelligent feedback...")
    feedback_report = feedback_generator.generate_feedback(
        score_data=final_score,
        player_profile=player_profile,
        max_feedbacks=4
    )
    
    # Display results
    print("\n" + "="*60)
    print("RAPPORT DE FEEDBACK INTELLIGENT")
    print("="*60)
    
    print(f"\nRésumé: {feedback_report.summary}")
    print(f"Ton utilisé: {feedback_report.overall_tone.value}")
    
    print(f"\n🎯 FEEDBACKS PRIORITAIRES ({len(feedback_report.priority_feedbacks)}):")
    print("-" * 50)
    
    for i, feedback in enumerate(feedback_report.priority_feedbacks, 1):
        print(f"\n{i}. [{feedback.priority.value.upper()}] {feedback.title}")
        print(f"   📊 Score: {feedback.score_reference:.1f}/100")
        print(f"   💬 {feedback.message}")
        print(f"   💡 Conseil: {feedback.improvement_tip}")
        if feedback.visual_example:
            print(f"   🎬 Exemple: {feedback.visual_example}")
    
    print(f"\n🔄 PROCHAINES ÉTAPES:")
    for i, focus in enumerate(feedback_report.next_focus_areas, 1):
        print(f"   {i}. {focus}")
    
    if feedback_report.secondary_feedbacks:
        print(f"\n📋 FEEDBACKS SECONDAIRES ({len(feedback_report.secondary_feedbacks)}):")
        for feedback in feedback_report.secondary_feedbacks:
            print(f"   • {feedback.category}: {feedback.message[:60]}...")
    
    # Export to JSON
    feedback_generator.export_feedback_json(
        feedback_report, 
        "C:/Web/Test-Open_Foot/feedback_report_example.json"
    )
    
    print(f"\n✅ Rapport exporté vers feedback_report_example.json")
    
    return feedback_report

def demonstrate_different_profiles():
    """Demonstrate feedback generation for different player profiles"""
    
    score_aggregator = ScoreAggregator()
    feedback_generator = FeedbackGenerator()
    
    # Base scores (same for all profiles)
    scores = {
        "biomechanics": (70, 0.9),
        "technical": (55, 0.8),
        "tactical": (75, 0.85),
        "physical": (60, 0.9)
    }
    
    for component, (score, confidence) in scores.items():
        score_aggregator.register_score(component, score, confidence)
    
    # Different player profiles
    profiles = [
        ("Jeune Talent", PlayerProfile(age=16, level=PlayerLevel.BEGINNER, position=PlayerPosition.FORWARD)),
        ("Joueur Pro", PlayerProfile(age=28, level=PlayerLevel.PROFESSIONAL, position=PlayerPosition.MIDFIELDER)),
        ("Vétéran", PlayerProfile(age=34, level=PlayerLevel.ADVANCED, position=PlayerPosition.DEFENDER))
    ]
    
    print("\n" + "="*80)
    print("DÉMONSTRATION - PERSONNALISATION PAR PROFIL")
    print("="*80)
    
    for profile_name, profile in profiles:
        final_score = score_aggregator.get_final_score(VideoType.TRAINING, profile)
        feedback_report = feedback_generator.generate_feedback(final_score, profile, max_feedbacks=3)
        
        print(f"\n🏃 PROFIL: {profile_name}")
        print(f"   Âge: {profile.age}, Niveau: {profile.level.value}, Poste: {profile.position.value}")
        print(f"   Ton de feedback: {feedback_report.overall_tone.value}")
        print(f"   Score final: {final_score['final_score']}/100")
        
        print("   Feedbacks principaux:")
        for feedback in feedback_report.priority_feedbacks[:2]:  # Top 2 seulement
            print(f"   • {feedback.message}")
        
        # Clear scores for next profile
        score_aggregator.clear_scores()
        for component, (score, confidence) in scores.items():
            score_aggregator.register_score(component, score, confidence)

if __name__ == "__main__":
    # Run example
    print("Génération d'exemple de feedback...")
    generate_example_feedback()
    
    print("\n" + "="*80)
    
    # Demonstrate different profiles
    demonstrate_different_profiles()