# 🎯 FootballAI Analyzer - Plateforme d'Analyse Vidéo IA pour le Football

## 📋 Vue d'ensemble

FootballAI Analyzer est une plateforme **100% locale** d'analyse vidéo intelligente pour le football, capable d'évaluer automatiquement les performances techniques et tactiques des joueurs.

### 🚀 Caractéristiques principales

- **Analyse biomécanique** : Évaluation précise des gestes techniques
- **Analyse tactique** : Compréhension du jeu collectif et des décisions
- **Analyse statistique** : Métriques avancées de performance
- **100% local** : Aucune dépendance cloud, tout fonctionne sur votre machine
- **Interface intuitive** : Application web locale facile à utiliser
- **Rapports détaillés** : PDF avec notes, commentaires et recommandations

### 📁 Structure du projet

```
football-ai-analyzer/
├── docs/                    # Documentation technique
├── backend/                 # Serveur API et logique métier
├── frontend/               # Interface utilisateur
├── models/                 # Modèles IA pré-entraînés
├── data/                   # Données et vidéos
├── scripts/                # Scripts d'installation et utilitaires
└── tests/                  # Tests unitaires et d'intégration
```

### 🛠️ Technologies utilisées

- **Backend** : Python 3.10+, FastAPI, PyTorch
- **Frontend** : React, Vite, Tailwind CSS
- **IA/ML** : YOLOv8, MediaPipe, LightGBM
- **Vidéo** : FFmpeg, OpenCV
- **Base de données** : SQLite (local)

### 🚦 Démarrage rapide

```bash
# Cloner le projet
git clone https://github.com/votre-repo/football-ai-analyzer
cd football-ai-analyzer

# Installer (Windows/Mac/Linux)
python scripts/install.py

# Lancer l'application
python scripts/run.py
```

Accédez ensuite à `http://localhost:3000` dans votre navigateur.

### 📖 Documentation

- [Architecture technique](./ARCHITECTURE.md)
- [Guide d'installation](./INSTALLATION.md)
- [Description des modules](./MODULES.md)
- [Pipeline de traitement](./PIPELINE.md)
- [Prompts IA](./PROMPTS.md)
- [Roadmap](./ROADMAP.md)

### 💡 Pour qui ?

- **Académies de football** : Suivi précis du développement technique des jeunes
- **Clubs professionnels** : Analyse détaillée pour optimiser les performances
- **Clubs semi-professionnels** : Outil d'analyse abordable et complet
- **Clubs amateurs** : Démocratisation de l'analyse vidéo professionnelle
- **Entraîneurs** : Données objectives pour améliorer leurs méthodes
- **Joueurs** : Feedback personnalisé et progression mesurable

### ⚡ Performances (Mode Précision Maximale)

- Traitement vidéo 1080p 60 FPS : ~8-10 min pour 10 min de vidéo (GPU RTX 3060+)
- Traitement vidéo 4K 60 FPS : ~15-20 min pour 10 min de vidéo (GPU RTX 4070+)
- Précision détection : >95% avec modèles SOTA
- GPU obligatoire pour maintenir 60 FPS et précision maximale

### 🤝 Contribution

Ce projet est développé avec l'aide d'assistants IA (Claude, ChatGPT, Cursor). Les contributions sont bienvenues !

### 📝 Licence

MIT License - Voir [LICENSE](./LICENSE) 