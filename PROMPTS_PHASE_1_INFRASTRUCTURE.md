# 🏗️ PROMPTS VIBE CODING - PHASE 1 : INFRASTRUCTURE ET SETUP

## 📅 Durée : 1 semaine

## 🎯 Objectifs
- Créer la structure complète du projet
- Configurer l'environnement de développement
- Mettre en place les outils de base
- Préparer l'architecture modulaire

---

## 1️⃣ Prompt Initial - Structure du Projet

```
Je développe une plateforme d'analyse vidéo IA pour le football professionnel.

Crée-moi une structure de projet Python moderne et professionnelle avec :

STRUCTURE :
- Architecture modulaire claire (src/core, src/modules/*, src/api, src/web)
- Séparation des couches (présentation, métier, données)
- Pattern Repository pour l'accès aux données
- Configuration centralisée

OUTILS :
- Poetry pour la gestion des dépendances
- Pre-commit hooks (black, flake8, mypy, isort)
- Docker multi-stage pour optimisation
- Docker Compose pour développement local
- Makefile avec commandes utiles

QUALITÉ CODE :
- Type hints partout (Python 3.11+)
- Docstrings Google style
- Tests unitaires pytest (structure miroir)
- Coverage minimum 80%
- Logging structuré avec loguru

MODULES PRINCIPAUX :
1. video_processing : traitement vidéo (FFmpeg, OpenCV)
2. detection : détection objets (YOLO, tracking)
3. pose_estimation : analyse biomécanique (MediaPipe)
4. analysis : logique métier football
5. ml_models : modèles ML/DL
6. api : API REST FastAPI
7. storage : gestion fichiers (S3, local)

Génère tous les fichiers de base avec exemples concrets.
```

## 2️⃣ Prompt Configuration Avancée

```
Ajoute à mon projet une configuration complète et professionnelle :

1. CONFIGURATION MULTI-ENVIRONNEMENTS :
   - Fichier .env.example documenté
   - Config Pydantic BaseSettings
   - Validation types et ranges
   - Support dev/staging/prod
   
2. SECRETS MANAGEMENT :
   - AWS Secrets Manager integration
   - Rotation automatique
   - Fallback local pour dev

3. MONITORING :
   - Prometheus metrics
   - Health checks endpoints
   - Performance profiling hooks
   - Sentry error tracking

4. EXEMPLE CONFIG :
```python
class VideoConfig(BaseSettings):
    max_file_size: int = Field(500_000_000, ge=0)  # 500MB
    supported_formats: List[str] = ["mp4", "avi", "mov"]
    fps_extraction: int = Field(30, ge=15, le=60)
    gpu_enabled: bool = True
    
class MLConfig(BaseSettings):
    model_confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    batch_size: int = Field(32, ge=1)
    device: str = Field("cuda", regex="^(cuda|cpu)$")
```

Implémente le système complet avec hot-reload en dev.
```

## 3️⃣ Prompt Docker Production-Ready

```
Configure Docker pour mon application d'analyse vidéo football :

1. DOCKERFILE MULTI-STAGE :
   - Stage 1: Builder avec dépendances compilées
   - Stage 2: Runtime optimisé (alpine si possible)
   - Cache Poetry dependencies
   - Non-root user
   - GPU support (nvidia-docker)

2. DOCKER COMPOSE :
   - Services : app, postgres, redis, minio (S3 local)
   - Networks isolés
   - Volumes pour persistance
   - Healthchecks
   - Restart policies

3. OPTIMISATIONS :
   - Layers caching intelligent
   - Multi-platform build (AMD64/ARM64)
   - Security scanning (Trivy)
   - Size < 2GB avec modèles ML

4. KUBERNETES READY :
   - ConfigMaps/Secrets structure
   - Readiness/Liveness probes
   - Resource limits
   - HPA configuration

Génère tous les fichiers avec best practices 2024.
```

## 4️⃣ Prompt Base de Données

```
Configure une architecture de données robuste pour mon système :

1. POSTGRESQL SCHEMA :
```sql
-- Utilisateurs et organisations
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50), -- 'club', 'academy', 'recruiter'
    subscription_tier VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id),
    email VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50), -- 'admin', 'coach', 'analyst', 'player'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Analyses vidéo
CREATE TABLE video_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    video_url TEXT NOT NULL,
    status VARCHAR(50), -- 'pending', 'processing', 'completed', 'failed'
    metadata JSONB, -- durée, résolution, fps, etc.
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Résultats détaillés
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES video_analyses(id),
    player_id UUID,
    action_type VARCHAR(100), -- 'pass', 'shot', 'dribble', etc.
    timestamp_ms INTEGER,
    biomechanics_score FLOAT,
    tactical_score FLOAT,
    details JSONB, -- Données complètes
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

2. MIGRATIONS ALEMBIC :
   - Auto-génération depuis SQLAlchemy
   - Rollback strategy
   - Data migrations

3. REPOSITORY PATTERN :
   - Interface abstraite
   - Implémentation PostgreSQL
   - Cache Redis layer
   - Tests avec fixtures

Implémente avec SQLAlchemy 2.0 async.
```

## 5️⃣ Prompt API FastAPI

```
Développe une API REST professionnelle FastAPI pour mon système :

1. STRUCTURE API :
```python
# Endpoints principaux
POST   /api/v1/analyses/upload       # Upload vidéo + lancement analyse
GET    /api/v1/analyses/{id}         # Statut et résultats
GET    /api/v1/analyses/{id}/report  # Rapport PDF
WS     /api/v1/analyses/{id}/stream  # Updates temps réel

POST   /api/v1/auth/login
POST   /api/v1/auth/refresh
GET    /api/v1/users/me

GET    /api/v1/stats/player/{id}     # Stats agrégées joueur
GET    /api/v1/stats/team/{id}       # Stats équipe
```

2. FEATURES AVANCÉES :
   - Pagination avec curseurs
   - Filtering/Sorting dynamique
   - Rate limiting par tier
   - Versioning API
   - OpenAPI 3.1 avec exemples

3. UPLOAD OPTIMISÉ :
   - Chunked upload pour gros fichiers
   - Reprise après interruption
   - Progress tracking
   - Validation format/taille

4. AUTHENTIFICATION :
   - JWT avec refresh tokens
   - OAuth2 pour intégrations
   - API keys pour B2B
   - RBAC complet

5. MIDDLEWARE :
   - Request ID tracking
   - Performance logging
   - CORS configuration
   - Compression gzip

Génère code complet avec tests d'intégration.
```

## 6️⃣ Prompt Testing Framework

```
Mets en place une infrastructure de tests complète :

1. STRUCTURE TESTS :
```
tests/
├── unit/           # Tests unitaires rapides
├── integration/    # Tests avec DB/services
├── e2e/           # Tests bout en bout
├── fixtures/      # Données de test
└── conftest.py    # Configuration pytest
```

2. FIXTURES FOOTBALL :
   - Vidéos de test (passes, tirs, dribbles)
   - Annotations ground truth
   - Profils joueurs types
   - Scénarios match complets

3. MOCKS INTELLIGENTS :
   - ML models responses
   - Video processing
   - External APIs
   - AWS services

4. TESTS PERFORMANCE :
   - Locust pour charge API
   - Memory profiling
   - GPU utilization
   - Latence processing

5. CI/CD GITHUB ACTIONS :
```yaml
- Tests parallèles par module
- Coverage report + badge
- Smoke tests GPU
- Build & push Docker
- Deploy staging auto
```

Crée l'infrastructure complète avec exemples.
```

## 7️⃣ Prompt Logging & Monitoring

```
Configure un système de logging et monitoring production-grade :

1. LOGGING STRUCTURÉ :
```python
from loguru import logger

# Configuration par module
logger.add(
    "logs/app_{time}.log",
    rotation="1 day",
    retention="30 days",
    format="{time} | {level} | {module} | {message}",
    serialize=True  # JSON pour parsing
)

# Contexte métier
@logger.contextualize
def analyze_video(video_id: str, user_id: str):
    logger.bind(video_id=video_id, user_id=user_id)
    logger.info("Starting video analysis")
```

2. MÉTRIQUES PROMETHEUS :
   - Latence par endpoint
   - Queue size processing
   - GPU/CPU utilization
   - Model inference time
   - Success/error rates

3. TRACING OPENTELEMETRY :
   - Trace complet pipeline
   - Span par étape
   - Correlation IDs
   - Export Jaeger

4. DASHBOARDS GRAFANA :
   - Vue système global
   - Analyse par client
   - Alerting intelligent
   - SLO tracking

5. ERROR TRACKING :
   - Sentry integration
   - Custom fingerprinting
   - User context
   - Release tracking

Implémente avec exemples concrets football.
```

## 🎮 Commandes Utiles Make

```makefile
# Development
make dev          # Lance tous les services locaux
make test         # Execute tous les tests
make lint         # Vérifie la qualité du code
make format       # Formate le code

# Docker
make build        # Build images Docker
make push         # Push vers registry
make deploy-staging # Déploie en staging

# Database
make db-migrate   # Applique migrations
make db-seed      # Charge données test

# ML
make download-models  # Télécharge modèles pre-trained
make train-custom    # Lance entraînement custom
```

## 📝 Notes pour l'équipe

1. **Commencer par** : Structure de base + Docker
2. **Priorité** : API upload vidéo fonctionnelle
3. **Tests** : Écrire tests en parallèle du code
4. **Documentation** : Mettre à jour au fur et à mesure
5. **Reviews** : Code review systématique via PR 