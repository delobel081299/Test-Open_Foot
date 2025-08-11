#!/usr/bin/env python3
"""
Script de test pour le classificateur d'actions football
Démontre l'utilisation du module avec des vidéos de test
"""

import cv2
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from backend.core.technical.action_classifier import ActionClassifier, ActionConfig, evaluate_classifier
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_test_video_sequence(action_type: str = "pass", duration: float = 2.0, fps: int = 30):
    """Créer une séquence vidéo de test simulant une action"""
    num_frames = int(duration * fps)
    height, width = 480, 640
    frames = []
    
    for i in range(num_frames):
        # Créer une frame avec un fond vert (terrain)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [34, 139, 34]  # Vert terrain
        
        # Simuler un joueur (rectangle rouge)
        player_x = int(width/2 + 100 * np.sin(2 * np.pi * i / num_frames))
        player_y = int(height/2 + 50 * np.cos(2 * np.pi * i / num_frames))
        cv2.rectangle(frame, 
                     (player_x - 20, player_y - 40),
                     (player_x + 20, player_y + 40),
                     (0, 0, 255), -1)
        
        # Simuler un ballon (cercle blanc)
        if action_type in ["pass", "shot"]:
            ball_x = player_x + int(50 * (i / num_frames))
            ball_y = player_y
            cv2.circle(frame, (ball_x, ball_y), 10, (255, 255, 255), -1)
        
        # Ajouter du texte
        cv2.putText(frame, f"Simulated: {action_type}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame {i+1}/{num_frames}", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        frames.append(frame)
    
    return frames


def test_single_action_classification():
    """Test de classification d'une action unique"""
    logger.info("=== Test 1: Classification d'une action unique ===")
    
    # Créer le classificateur
    config = ActionConfig()
    classifier = ActionClassifier(config)
    
    # Créer une séquence de test
    frames = create_test_video_sequence(action_type="shot", duration=2.0)
    
    # Classifier
    start_time = time.time()
    result = classifier.classify(frames)
    inference_time = (time.time() - start_time) * 1000
    
    # Afficher les résultats
    logger.info(f"Action prédite: {result['action']}")
    logger.info(f"Confiance: {result['confidence']:.3f}")
    logger.info(f"Temps d'inférence: {result['inference_time']:.1f}ms")
    
    # Top-3 prédictions
    logger.info("\nTop-3 prédictions:")
    for pred in result['top_k']:
        logger.info(f"  - {pred['action']}: {pred['confidence']:.3f}")
    
    return result


def test_batch_classification():
    """Test de classification en batch"""
    logger.info("\n=== Test 2: Classification en batch ===")
    
    config = ActionConfig()
    classifier = ActionClassifier(config)
    
    # Créer plusieurs séquences
    actions = ["pass", "shot", "dribble", "tackle", "sprint"]
    sequences = []
    
    for action in actions:
        frames = create_test_video_sequence(action_type=action)
        sequences.append(frames)
    
    # Classifier en batch
    start_time = time.time()
    results = classifier.classify_batch(sequences)
    total_time = (time.time() - start_time) * 1000
    
    # Afficher les résultats
    logger.info(f"Temps total pour {len(sequences)} séquences: {total_time:.1f}ms")
    logger.info(f"Temps moyen par séquence: {total_time/len(sequences):.1f}ms")
    
    for i, (action, result) in enumerate(zip(actions, results)):
        logger.info(f"\nSéquence {i+1} (simulée: {action}):")
        logger.info(f"  - Prédiction: {result['action']} (confiance: {result['confidence']:.3f})")


def test_temporal_smoothing():
    """Test du lissage temporel"""
    logger.info("\n=== Test 3: Lissage temporel ===")
    
    config = ActionConfig(smoothing_window=5)
    classifier = ActionClassifier(config)
    
    # Simuler une séquence avec des actions changeantes
    all_frames = []
    true_actions = ["pass", "pass", "shot", "shot", "dribble"]
    
    for action in true_actions:
        frames = create_test_video_sequence(action_type=action, duration=0.5)
        all_frames.extend(frames[:15])  # Prendre 15 frames par action
    
    # Classifier avec fenêtre glissante
    window_size = config.num_frames
    predictions = []
    
    for i in range(0, len(all_frames) - window_size + 1, 5):
        window = all_frames[i:i + window_size]
        result = classifier.classify(window)
        predictions.append({
            'frame_start': i,
            'action': result['action'],
            'confidence': result['confidence'],
            'smoothed': result.get('smoothed', False)
        })
    
    # Afficher les résultats
    logger.info("Prédictions avec lissage temporel:")
    for pred in predictions:
        smoothed_tag = " (lissé)" if pred['smoothed'] else ""
        logger.info(f"Frames {pred['frame_start']}-{pred['frame_start']+window_size}: "
                   f"{pred['action']} ({pred['confidence']:.3f}){smoothed_tag}")


def test_cache_performance():
    """Test des performances du cache"""
    logger.info("\n=== Test 4: Performance du cache ===")
    
    config = ActionConfig(use_cache=True, cache_size=10)
    classifier = ActionClassifier(config)
    
    # Créer une séquence
    frames = create_test_video_sequence(action_type="pass")
    
    # Première classification (sans cache)
    start_time = time.time()
    result1 = classifier.classify(frames)
    time1 = (time.time() - start_time) * 1000
    
    # Deuxième classification (avec cache)
    start_time = time.time()
    result2 = classifier.classify(frames)
    time2 = (time.time() - start_time) * 1000
    
    logger.info(f"Temps sans cache: {time1:.1f}ms")
    logger.info(f"Temps avec cache: {time2:.1f}ms")
    logger.info(f"Accélération: {time1/time2:.1f}x")
    logger.info(f"Résultat depuis cache: {result2.get('from_cache', False)}")


def visualize_action_probabilities(classifier: ActionClassifier, frames: list):
    """Visualiser les probabilités pour toutes les classes"""
    result = classifier.classify(frames)
    
    # Extraire les probabilités
    classes = classifier.config.action_classes
    probs = [result['probabilities'][cls] for cls in classes]
    
    # Créer le graphique
    plt.figure(figsize=(12, 6))
    
    # Barplot des probabilités
    colors = ['green' if p > 0.5 else 'orange' if p > 0.2 else 'red' for p in probs]
    bars = plt.bar(range(len(classes)), probs, color=colors)
    
    # Personnalisation
    plt.xlabel('Actions', fontsize=12)
    plt.ylabel('Probabilité', fontsize=12)
    plt.title(f'Distribution des probabilités - Prédiction: {result["action"]} ({result["confidence"]:.3f})',
              fontsize=14)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Ajouter les valeurs sur les barres
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Ligne de seuil
    plt.axhline(y=classifier.config.confidence_threshold, color='k', 
                linestyle='--', alpha=0.5, label='Seuil de confiance')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('action_probabilities.png', dpi=150)
    plt.close()
    
    logger.info("Graphique des probabilités sauvegardé: action_probabilities.png")


def test_model_statistics():
    """Afficher les statistiques du modèle"""
    logger.info("\n=== Test 5: Statistiques du modèle ===")
    
    config = ActionConfig()
    classifier = ActionClassifier(config)
    
    # Effectuer quelques classifications pour les stats
    for i in range(10):
        frames = create_test_video_sequence()
        classifier.classify(frames)
    
    # Obtenir les stats
    stats = classifier.get_stats()
    
    logger.info("Statistiques du classificateur:")
    logger.info(f"  - Device: {stats['device']}")
    logger.info(f"  - Type de modèle: {stats['model_type']}")
    logger.info(f"  - Nombre de classes: {stats['num_classes']}")
    logger.info(f"  - Taille du cache: {stats['cache_size']}")
    
    if 'inference_stats' in stats:
        logger.info("\nStatistiques d'inférence:")
        logger.info(f"  - Temps moyen: {stats['inference_stats']['mean_ms']:.1f}ms")
        logger.info(f"  - Écart-type: {stats['inference_stats']['std_ms']:.1f}ms")
        logger.info(f"  - Min: {stats['inference_stats']['min_ms']:.1f}ms")
        logger.info(f"  - Max: {stats['inference_stats']['max_ms']:.1f}ms")


def test_augmentations():
    """Test des augmentations de données"""
    logger.info("\n=== Test 6: Augmentations de données ===")
    
    from backend.core.technical.action_classifier import ActionDataAugmentation
    
    # Créer une séquence
    frames = create_test_video_sequence(action_type="dribble")
    frames_array = np.array(frames).transpose(0, 3, 1, 2)  # T, C, H, W
    
    # Appliquer différentes augmentations
    augmentations = {
        "Original": frames_array,
        "Random Crop": ActionDataAugmentation.random_crop(frames_array.copy()),
        "Horizontal Flip": ActionDataAugmentation.random_flip(frames_array.copy()),
        "Speed Variation": ActionDataAugmentation.speed_variation(frames_array.copy()),
        "Add Noise": ActionDataAugmentation.add_noise(frames_array.copy() / 255.0)
    }
    
    # Visualiser
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for ax, (name, aug_frames) in zip(axes, augmentations.items()):
        # Prendre la frame du milieu
        frame = aug_frames[len(aug_frames)//2].transpose(1, 2, 0)
        
        # Normaliser pour l'affichage
        if name == "Add Noise":
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        
        ax.imshow(frame)
        ax.set_title(name)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentations_demo.png', dpi=150)
    plt.close()
    
    logger.info("Démonstration des augmentations sauvegardée: augmentations_demo.png")


def run_full_evaluation():
    """Exécuter une évaluation complète sur un dataset de test"""
    logger.info("\n=== Évaluation complète ===")
    
    config = ActionConfig()
    classifier = ActionClassifier(config)
    
    # Créer un dataset de test synthétique
    test_data = []
    actions = ["pass", "shot", "dribble", "tackle", "sprint"]
    
    for _ in range(5):  # 5 échantillons par action
        for action in actions:
            frames = create_test_video_sequence(action_type=action)
            # Ajouter du bruit pour rendre plus réaliste
            frames_array = np.array(frames)
            noise = np.random.normal(0, 10, frames_array.shape).astype(np.uint8)
            frames_noisy = np.clip(frames_array + noise, 0, 255).astype(np.uint8)
            test_data.append((frames_noisy.tolist(), action))
    
    # Évaluer
    results = evaluate_classifier(
        classifier,
        test_data,
        save_path=Path("evaluation_results.json")
    )
    
    # Afficher la matrice de confusion
    plt.figure(figsize=(10, 8))
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=actions, yticklabels=actions)
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédiction')
    plt.ylabel('Vérité')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    
    logger.info("Matrice de confusion sauvegardée: confusion_matrix.png")


def main():
    """Fonction principale"""
    logger.info("="*60)
    logger.info("Test du classificateur d'actions football")
    logger.info("="*60)
    
    # Exécuter tous les tests
    test_single_action_classification()
    test_batch_classification()
    test_temporal_smoothing()
    test_cache_performance()
    test_model_statistics()
    test_augmentations()
    
    # Visualisation
    config = ActionConfig()
    classifier = ActionClassifier(config)
    frames = create_test_video_sequence(action_type="shot")
    visualize_action_probabilities(classifier, frames)
    
    # Évaluation complète (optionnel car plus long)
    # run_full_evaluation()
    
    logger.info("\n✓ Tous les tests complétés avec succès!")


if __name__ == "__main__":
    main()