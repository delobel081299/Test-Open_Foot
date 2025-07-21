# 🖥️ CONFIGURATION DÉPLOIEMENT LOCAL - FOOTBALL AI

## 📋 Vue d'ensemble

Ce document détaille la configuration et le déploiement de Football AI en mode **local** (sur une machine unique avec GPU).

---

## 🔧 Architecture Locale Simplifiée

### Mode Monolithique Modulaire

```
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LOCALE                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   Interface Web  │  │   API FastAPI   │                  │
│  │  (localhost:80)  │  │ (localhost:8000) │                  │
│  └────────┬─────────┘  └────────┬─────────┘                  │
│           └──────────┬───────────┘                          │
│                      ▼                                       │
│  ┌──────────────────────────────────────────────┐          │
│  │           Core Application Engine            │          │
│  ├──────────────────────────────────────────────┤          │
│  │ • Video Processing Module                    │          │
│  │ • ML Inference Module (GPU)                  │          │
│  │ • Analysis & Scoring Module                  │          │
│  │ • Report Generation Module                   │          │
│  └──────────────────────────────────────────────┘          │
│                      ▼                                       │
│  ┌──────────────────────────────────────────────┐          │
│  │              Local Storage                    │          │
│  ├──────────────────────────────────────────────┤          │
│  │ • SQLite/PostgreSQL (embedded)               │          │
│  │ • File System (videos, reports)              │          │
│  │ • Model Cache                                │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Configuration Minimale Requise

### Hardware
- **CPU** : Intel i7/AMD Ryzen 7 ou supérieur
- **RAM** : 16 GB minimum (32 GB recommandé)
- **GPU** : NVIDIA GTX 1660 ou supérieur (6GB VRAM minimum)
- **Stockage** : 500 GB SSD (pour modèles + vidéos)

### Software
- **OS** : Windows 10/11, Ubuntu 20.04+, macOS 11+
- **Python** : 3.10 ou 3.11
- **CUDA** : 11.8+ (pour GPU NVIDIA)
- **Docker** : Optionnel mais recommandé

---

## 🚀 Installation Locale

### 1. Installation One-Click (Windows)

```batch
# setup_local.bat
@echo off
echo "=== Installation Football AI Local ==="

:: Vérifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo "Python non trouvé. Installation..."
    :: Télécharger et installer Python
)

:: Créer environnement virtuel
python -m venv venv
call venv\Scripts\activate

:: Installer dépendances
pip install -r requirements.txt

:: Télécharger modèles ML
python scripts/download_models.py

:: Créer structure dossiers
mkdir data\uploads data\processed data\reports models

:: Lancer application
python src/standalone/main.py --mode gui

echo "Installation terminée ! Accédez à http://localhost:8000"
pause
```

### 2. Installation Docker Local

```yaml
# docker-compose.local.yml
version: '3.8'

services:
  football-ai:
    build: 
      context: .
      dockerfile: docker/Dockerfile.local
    ports:
      - "8000:8000"
      - "80:80"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./config:/app/config
    environment:
      - DEPLOYMENT_MODE=local
      - GPU_ENABLED=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## ⚙️ Configuration Application

### config/local.yaml

```yaml
# Configuration locale Football AI
deployment:
  mode: local
  debug: true
  
server:
  host: 0.0.0.0
  port: 8000
  workers: 1  # Monolithique
  
storage:
  type: local
  base_path: ./data
  max_file_size: 2GB
  
database:
  type: sqlite  # ou postgresql embarqué
  path: ./data/football_ai.db
  
ml_models:
  device: cuda  # ou cpu si pas de GPU
  batch_size: 4
  cache_models: true
  models_path: ./models
  
  yolo:
    model: yolov10x
    confidence: 0.45
    
  mediapipe:
    model_complexity: 2
    min_detection_confidence: 0.5
    
processing:
  max_concurrent_videos: 1  # Limitation locale
  video_formats: [mp4, avi, mov]
  output_format: mp4
  
interface:
  type: web  # ou desktop pour GUI native
  auto_open_browser: true
```

---

## 🔄 Migration Future vers Cloud/API

### Points de Flexibilité

1. **Storage Abstraction**
```python
# src/modules/storage/storage_interface.py
class StorageInterface(ABC):
    @abstractmethod
    def save_video(self, video_data): pass
    
    @abstractmethod  
    def get_video(self, video_id): pass

# Implémentations
class LocalStorage(StorageInterface):
    # Stockage fichier local
    
class S3Storage(StorageInterface):
    # Stockage S3 pour cloud
```

2. **Database Abstraction**
```python
# src/core/database/db_interface.py
class DatabaseInterface(ABC):
    # Interface commune SQLite/PostgreSQL
```

3. **Configuration Dynamique**
```python
# src/core/config/config_loader.py
def load_config():
    mode = os.getenv('DEPLOYMENT_MODE', 'local')
    if mode == 'local':
        return LocalConfig()
    elif mode == 'cloud':
        return CloudConfig()
```

---

## 📊 Monitoring Local

### Dashboard Simple
- Utilisation CPU/GPU/RAM
- Files d'attente de traitement
- Logs temps réel
- Historique analyses

### Métriques Clés
```python
# src/monitoring/local_metrics.py
class LocalMetrics:
    def __init__(self):
        self.videos_processed = 0
        self.average_processing_time = 0
        self.gpu_usage = []
        self.errors = []
```

---

## 🛡️ Sécurité Mode Local

1. **Pas d'authentification** par défaut (usage interne)
2. **Option auth basique** si besoin :
   - Login/password simple
   - Token local
3. **Chiffrement optionnel** des données sensibles
4. **Logs locaux** uniquement

---

## 📝 Scripts Utilitaires

### Benchmark Local
```bash
# Tester performances sur machine locale
python scripts/benchmark.py --video sample.mp4 --iterations 10
```

### Nettoyage Données
```bash
# Nettoyer fichiers temporaires
python scripts/cleanup.py --days 30
```

### Export/Import
```bash
# Exporter analyses pour backup
python scripts/export_data.py --format json --output backup.zip
```

---

## ✅ Checklist Déploiement Local

- [ ] Python 3.10+ installé
- [ ] GPU NVIDIA avec drivers à jour
- [ ] Modèles ML téléchargés (~5GB)
- [ ] Espace disque suffisant (50GB+)
- [ ] Configuration locale adaptée
- [ ] Tests de performance validés
- [ ] Documentation utilisateur local 