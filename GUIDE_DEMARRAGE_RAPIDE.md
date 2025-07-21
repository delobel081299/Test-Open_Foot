# 🚀 GUIDE DE DÉMARRAGE RAPIDE - FOOTBALL AI

## 📋 Table des matières

Ce guide vous permettra de démarrer rapidement le développement de la plateforme d'analyse vidéo IA pour le football en utilisant le vibe coding avec Claude/ChatGPT/Cursor.

---

## 🎯 Note Importante : Approche de Développement

> **⚠️ IMPORTANT** : Ce projet est conçu pour fonctionner d'abord en **mode local** sur vos machines. La décision entre déploiement local chez les clients ou API SaaS sera prise ultérieurement.

### Stratégie de développement :
1. **Phase actuelle** : Application locale complète et fonctionnelle
2. **Phase future** : Adaptation selon le modèle commercial choisi (local ou SaaS)

L'architecture est conçue pour permettre facilement les deux approches sans refonte majeure.

---

## 🎯 Ordre de développement recommandé

### Semaine 1-2 : Infrastructure
1. **Setup projet** → Utilisez `PROMPTS_PHASE_1_INFRASTRUCTURE.md`
2. **Base de données** → Prompt 4 (PostgreSQL Schema)
3. **API de base** → Prompt 5 (FastAPI)
4. **Docker** → Prompt 3 (Docker Production)

### Semaine 3-4 : Traitement Vidéo
1. **Prétraitement** → `PROMPTS_PHASE_2_VIDEO_DETECTION.md` - Prompt 1
2. **Détection YOLO** → Prompt 2 (YOLOv10)
3. **Tracking** → Prompt 3 (ByteTrack)

### Semaine 5 : Analyse Biomécanique
1. **MediaPipe** → `PROMPTS_PHASE_3_BIOMECANIQUE.md` - Prompt 1
2. **Analyse passe** → Prompt 2
3. **Visualisation** → Prompt 6

### Semaine 6-7 : Intelligence
1. **Features** → `PROMPTS_PHASE_4_INTELLIGENCE_SCORING.md` - Prompt 1
2. **XGBoost** → Prompt 2
3. **Feedback LLM** → Prompt 4

### Semaine 8 : Production
1. **Interface web** → `PROMPTS_PHASE_5_PRODUCTION_DATASET.md` - Prompt 2
2. **Kubernetes** → Prompt 3
3. **Monitoring** → Prompt 4

---

## 🛠️ Configuration initiale

### 1. Environnement de développement

```bash
# Cloner le template de base (à créer)
git clone https://github.com/votre-org/football-ai-template.git
cd football-ai

# Installer Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Installer dépendances
poetry install

# Activer environnement
poetry shell
```

### 2. Services requis

```bash
# Lancer services locaux
docker-compose up -d postgres redis minio

# Vérifier que tout fonctionne
docker-compose ps
```

### 3. Configuration IDE (Cursor)

```json
// .cursor/settings.json
{
  "ai.model": "claude-3-opus",
  "ai.temperature": 0.7,
  "ai.contextWindow": "large",
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true
}
```

---

## 💡 Stratégie Vibe Coding

### 1. Utilisation des prompts

**TOUJOURS** :
- Copier le prompt complet depuis les fichiers MD
- Adapter les variables selon votre contexte
- Demander le code complet, pas juste des snippets
- Vérifier la cohérence avec l'architecture globale

**EXEMPLE** :
```
Je travaille sur [MODULE]. 
Voici mon contexte actuel : [CODE_EXISTANT]
[COLLER_PROMPT_DEPUIS_MD]
Génère le code complet avec tests.
```

### 2. Workflow recommandé

1. **Lecture** : Comprendre le module dans `STRUCTURE_TECHNIQUE_FOOTBALL_IA.md`
2. **Prompt** : Utiliser le prompt correspondant dans `PROMPTS_PHASE_X_*.md`
3. **Génération** : Laisser l'IA générer le code complet
4. **Validation** : Tester immédiatement
5. **Itération** : Affiner avec des prompts de suivi

### 3. Prompts de suivi efficaces

```
# Pour debug
"J'ai cette erreur : [ERREUR]. 
Le code actuel est : [CODE].
Corrige et explique le problème."

# Pour optimisation
"Ce code fonctionne mais est lent : [CODE].
Optimise pour traiter des vidéos 1080p en temps réel."

# Pour tests
"Génère des tests unitaires complets pour : [CODE].
Inclus les cas edge et les mocks nécessaires."
```

---

## 📚 Ressources essentielles

### Documentation technique
- **YOLOv10** : https://github.com/THU-MIG/yolov10
- **ByteTrack** : https://github.com/ifzhang/ByteTrack
- **MediaPipe** : https://google.github.io/mediapipe/
- **FastAPI** : https://fastapi.tiangolo.com/

### Datasets
- **SoccerNet** : https://www.soccer-net.org/
- **Kaggle Football** : Rechercher "football detection dataset"
- **Roboflow Universe** : Collections football publiques

### Modèles pré-entraînés
```python
# À télécharger au début
models = {
    "yolov10": "https://github.com/THU-MIG/yolov10/releases/download/v1.0/yolov10x.pt",
    "mediapipe_pose": "automatic_download",
    "ball_detector": "custom_training_required"
}
```

---

## ⚡ Quick Start - Première analyse

### 1. Script minimal fonctionnel

```python
# minimal_analysis.py
import cv2
from ultralytics import YOLO
import mediapipe as mp

# Charger modèles
yolo = YOLO('yolov10x.pt')
mp_pose = mp.solutions.pose.Pose()

# Analyser vidéo
cap = cv2.VideoCapture('test_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Détection
    results = yolo(frame)
    
    # Pose estimation sur première personne détectée
    if len(results[0].boxes) > 0:
        # ... traitement
        pass
```

### 2. Commandes essentielles

```bash
# Lancer API dev
make dev-api

# Lancer interface web
make dev-web

# Tester analyse vidéo
python scripts/test_analysis.py --video sample.mp4

# Générer rapport
make generate-report --analysis-id=123
```

---

## 🚨 Problèmes fréquents

### 1. GPU non détecté
```bash
# Vérifier CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Si problème, réinstaller :
poetry add torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

### 2. Mémoire insuffisante
```python
# Réduire batch size dans configs
VIDEO_CONFIG = {
    "batch_size": 8,  # Réduire si OOM
    "fp16": True,     # Activer mixed precision
}
```

### 3. Modèles lents
- Utiliser YOLOv10-S au lieu de YOLOv10-X pour dev
- Activer TensorRT pour production
- Implémenter cache agressif

---

## 📞 Support & Contact

- **Documentation complète** : `STRUCTURE_TECHNIQUE_FOOTBALL_IA.md`
- **Prompts par phase** : `PROMPTS_PHASE_*.md`
- **Issues GitHub** : [votre-repo]/issues
- **Discord équipe** : [lien-discord]

---

## ✅ Checklist avant de commencer

- [ ] Python 3.11+ installé
- [ ] CUDA 11.8+ (pour GPU)
- [ ] Docker Desktop running
- [ ] 16GB RAM minimum
- [ ] 50GB espace disque
- [ ] Compte AWS/GCP (pour production)
- [ ] Accès aux prompts dans `/PROMPTS_PHASE_*.md`

---

**Rappel** : Utilisez massivement l'IA pour coder ! Les prompts fournis sont testés et optimisés. N'hésitez pas à demander le code complet plutôt que des fragments.

Bon développement ! 🚀⚽ 