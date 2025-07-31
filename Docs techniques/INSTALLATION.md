# 📦 Guide d'Installation - FootballAI Analyzer

## 🔍 Prérequis système

### Configuration minimale (Mode Précision Maximale)
- **OS** : Windows 10/11, Ubuntu 20.04+, macOS 12+
- **CPU** : Intel i7 10ème gen / AMD Ryzen 7 5800X ou supérieur
- **RAM** : 32 GB minimum
- **GPU** : NVIDIA RTX 3060 12GB minimum (OBLIGATOIRE)
- **Stockage** : 100 GB SSD rapide
- **Python** : 3.10 ou 3.11

### Configuration recommandée
- **CPU** : Intel i9 12ème gen / AMD Ryzen 9
- **RAM** : 64 GB
- **GPU** : NVIDIA RTX 4070 Ti 16GB ou supérieur
- **Stockage** : 500 GB NVMe SSD
- **VRAM** : 16GB+ pour traitement 4K 60 FPS

## 🚀 Installation automatique (Recommandée)

### Windows

```powershell
# 1. Ouvrir PowerShell en tant qu'administrateur
# 2. Cloner le projet
git clone https://github.com/votre-repo/football-ai-analyzer.git
cd football-ai-analyzer

# 3. Lancer l'installation automatique
python scripts/install.py

# 4. L'installateur va :
#    - Vérifier votre configuration
#    - Installer Python si nécessaire
#    - Configurer l'environnement virtuel
#    - Installer toutes les dépendances
#    - Télécharger les modèles IA
#    - Configurer la base de données
```

### macOS/Linux

```bash
# 1. Ouvrir un terminal
# 2. Cloner le projet
git clone https://github.com/votre-repo/football-ai-analyzer.git
cd football-ai-analyzer

# 3. Rendre le script exécutable et lancer
chmod +x scripts/install.py
python3 scripts/install.py
```

## 🔧 Installation manuelle (Avancée)

### Étape 1 : Environnement Python

```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Mettre à jour pip
python -m pip install --upgrade pip
```

### Étape 2 : Dépendances Backend

```bash
# Installation des dépendances core
pip install -r requirements.txt

# Si vous avez un GPU NVIDIA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Sans GPU (CPU only)
pip install torch torchvision torchaudio
```

### Étape 3 : Configuration GPU (NVIDIA)

#### Windows
```powershell
# Vérifier CUDA
nvidia-smi

# Si CUDA n'est pas installé, télécharger depuis :
# https://developer.nvidia.com/cuda-11-8-0-download-archive

# Installer cuDNN
# Télécharger depuis : https://developer.nvidia.com/cudnn
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-cuda-toolkit nvidia-cudnn

# Vérifier l'installation
nvcc --version
```

### Étape 4 : Installation FFmpeg

#### Windows
```powershell
# Méthode 1 : Chocolatey (si installé)
choco install ffmpeg

# Méthode 2 : Manuel
# 1. Télécharger depuis https://www.gyan.dev/ffmpeg/builds/
# 2. Extraire dans C:\ffmpeg
# 3. Ajouter C:\ffmpeg\bin au PATH système
```

#### macOS
```bash
# Avec Homebrew
brew install ffmpeg
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

### Étape 5 : Téléchargement des modèles IA

```bash
# Télécharger automatiquement tous les modèles
python scripts/download_models.py

# Ou télécharger manuellement :
# 1. YOLOv8 : https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
# 2. MediaPipe : Téléchargé automatiquement à la première utilisation
# 3. TimeSformer : https://github.com/facebookresearch/TimeSformer/blob/main/MODEL_ZOO.md
```

### Étape 6 : Configuration Frontend

```bash
# Installer Node.js si nécessaire
# Windows : https://nodejs.org/
# macOS : brew install node
# Linux : sudo apt install nodejs npm

# Installer les dépendances frontend
cd frontend
npm install
cd ..
```

### Étape 7 : Configuration base de données

```bash
# Initialiser la base de données
python scripts/init_db.py

# Créer les dossiers nécessaires
python scripts/create_folders.py
```

## ⚙️ Configuration

### Fichier de configuration principal

Créer/éditer `config/app.yaml` :

```yaml
# Configuration générale
app:
  name: "FootballAI Analyzer"
  version: "1.0.0"
  debug: false

# Configuration serveur
server:
  host: "127.0.0.1"
  port: 8000
  workers: 4

# Configuration GPU
gpu:
  enabled: true
  device_id: 0
  memory_fraction: 0.8

# Configuration vidéo
video:
  max_upload_size_mb: 2000
  supported_formats: ["mp4", "avi", "mov", "mkv"]
  output_fps: 60  # Mode précision maximale
  
# Configuration modèles
models:
  detection:
    model: "yolov8x"
    confidence_threshold: 0.5
  pose:
    model: "mediapipe_heavy"
    smooth: true
  action:
    model: "timesformer_base"
    
# Configuration analyse
analysis:
  max_players_tracked: 22
  min_detection_confidence: 0.6
  enable_3d_pose: true
```

## 🏃 Lancement de l'application

### Méthode simple

```bash
# Lancer l'application complète
python scripts/run.py

# L'application sera accessible sur http://localhost:3000
```

### Méthode manuelle

```bash
# Terminal 1 : Lancer le backend
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 : Lancer le frontend
cd frontend
npm run dev
```

## ✅ Vérification de l'installation

```bash
# Lancer le script de vérification
python scripts/check_install.py

# Ce script vérifie :
# ✓ Version Python
# ✓ Dépendances installées
# ✓ GPU disponible
# ✓ FFmpeg fonctionnel
# ✓ Modèles téléchargés
# ✓ Base de données initialisée
# ✓ Ports disponibles
```

## 🐛 Dépannage

### Erreur : "CUDA out of memory"
```bash
# Réduire la taille du batch dans config/app.yaml
# Ou forcer le mode CPU :
python scripts/run.py --cpu-only
```

### Erreur : "ModuleNotFoundError"
```bash
# Réinstaller les dépendances
pip install -r requirements.txt --force-reinstall
```

### Erreur : "Port already in use"
```bash
# Changer les ports dans config/app.yaml
# Ou tuer les processus :
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/macOS
lsof -i :8000
kill -9 <PID>
```

### Performance lente sans GPU
```yaml
# Éditer config/app.yaml pour optimiser CPU :
video:
  output_fps: 15  # Réduire FPS
analysis:
  max_players_tracked: 10  # Réduire nombre de joueurs
```

## 📱 Installation pour utilisateurs finaux

Pour les utilisateurs non-techniques, nous fournirons :

1. **Installateur Windows** (.exe)
   - Installation en un clic
   - Détection automatique GPU
   - Raccourcis bureau/menu

2. **Package macOS** (.dmg)
   - Glisser-déposer dans Applications
   - Configuration automatique

3. **AppImage Linux**
   - Exécutable portable
   - Pas d'installation requise

## 🔄 Mise à jour

```bash
# Mettre à jour l'application
git pull
python scripts/update.py

# Le script va :
# - Sauvegarder vos données
# - Mettre à jour le code
# - Installer nouvelles dépendances
# - Migrer la base de données
# - Télécharger nouveaux modèles
```

## 💡 Tips d'installation

1. **GPU fortement recommandé** : L'analyse est 5-10x plus rapide avec GPU
2. **SSD préférable** : Améliore significativement le chargement vidéo
3. **Fermer autres applications** : Libère RAM et GPU
4. **Antivirus** : Ajouter une exception pour le dossier du projet
5. **VPN** : Peut ralentir le téléchargement des modèles

## 📞 Support

En cas de problème d'installation :
1. Consulter les [issues GitHub](https://github.com/votre-repo/issues)
2. Rejoindre notre [Discord](https://discord.gg/xxxxx)
3. Envoyer logs d'installation : `logs/install.log` 