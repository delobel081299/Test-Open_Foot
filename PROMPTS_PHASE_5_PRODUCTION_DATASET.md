# 🚀 PROMPTS VIBE CODING - PHASE 5 : PRODUCTION & DATASETS

## 📅 Durée : 2 semaines

## 🎯 Objectifs
- Déploiement production scalable
- Création et gestion datasets
- Interface utilisateur finale
- Monitoring et maintenance

---

## 1️⃣ Prompt Création Dataset Football

```
Développe un système complet de création et gestion de dataset pour l'entraînement des modèles :

1. PIPELINE ANNOTATION :
```python
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import labelbox
import roboflow

@dataclass
class FootballAnnotation:
    frame_id: int
    timestamp: float
    
    # Détection objets
    player_bboxes: List[BoundingBox]
    ball_bbox: Optional[BoundingBox]
    player_teams: List[int]
    player_poses: List[PoseKeypoints]
    
    # Labels actions
    action_type: str  # "pass", "shot", "dribble", etc.
    action_quality: float  # 0-10 score expert
    action_phase: str  # "preparation", "execution", "follow_through"
    
    # Contexte tactique
    formation: str  # "4-4-2", "4-3-3", etc.
    game_phase: str  # "attack", "defense", "transition"
    pressure_level: int  # 1-5
    
    # Métadonnées
    annotator_id: str
    confidence: float
    notes: str

class DatasetBuilder:
    def __init__(self):
        self.annotation_tool = self.setup_annotation_platform()
        self.quality_checker = QualityAssurance()
        self.augmenter = FootballAugmenter()
        
    def create_annotation_project(self, videos: List[str]) -> Project:
        # Configuration Labelbox/Roboflow
        project = self.annotation_tool.create_project(
            name="Football Actions Dataset v2",
            media_type="VIDEO",
            ontology=self.create_football_ontology()
        )
        
        # Import vidéos
        for video in videos:
            # Extraction frames clés
            key_frames = self.extract_key_frames(video)
            
            # Pre-annotation avec modèles existants
            pre_annotations = self.generate_pre_annotations(key_frames)
            
            # Upload avec pre-annotations
            project.upload_data(
                video_path=video,
                frames=key_frames,
                pre_annotations=pre_annotations
            )
            
        return project
```

2. STRATÉGIES COLLECTE :
   - Partenariats clubs (données matchs)
   - Crowdsourcing communauté
   - Génération synthétique (Unity/UE)
   - Scraping matchs publics
   - Sessions capture dédiées

3. CONTRÔLE QUALITÉ :
   ```python
   class QualityAssurance:
       def validate_annotations(self, annotations: List[FootballAnnotation]):
           checks = {
               "bbox_overlap": self.check_bbox_consistency(),
               "pose_validity": self.check_pose_biomechanics(),
               "temporal_coherence": self.check_temporal_smoothness(),
               "inter_annotator": self.compute_agreement_score(),
               "completeness": self.check_missing_labels()
           }
           
           return QualityReport(checks)
   ```

4. AUGMENTATION DONNÉES :
   - Variations météo (pluie, neige)
   - Changements éclairage
   - Angles caméra différents
   - Vitesses lecture variées
   - Occlusions artificielles

5. VERSIONING DATASET :
   - DVC pour tracking
   - Splits train/val/test
   - Stratification par scénarios
   - Changelog annotations
   - Benchmarks versions

Génère documentation dataset complète.
```

## 2️⃣ Prompt Interface Web Production

```
Développe l'interface web finale pour les utilisateurs (React + TypeScript) :

1. ARCHITECTURE FRONTEND :
```typescript
// Structure application React
interface AppStructure {
  components: {
    common: {
      Layout: Component
      Navigation: Component
      LoadingStates: Component
      ErrorBoundary: Component
    }
    upload: {
      VideoUploader: Component  // Drag & drop, progress
      QualityChecker: Component // Validation côté client
      MetadataForm: Component   // Infos contextuelles
    }
    analysis: {
      VideoPlayer: Component    // Annotations overlay
      Timeline: Component       // Navigation temporelle  
      StatsPanel: Component    // Métriques temps réel
      Feedback: Component      // Conseils IA
    }
    dashboard: {
      PlayerProfile: Component
      ProgressCharts: Component
      TeamAnalytics: Component
      ExportTools: Component
    }
    coaching: {
      ExerciseLibrary: Component
      TrainingPlanner: Component
      VideoComparison: Component
      NotesEditor: Component
    }
  }
}

// Hook personnalisé pour l'analyse
const useVideoAnalysis = (videoId: string) => {
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<Error | null>(null)
  
  useEffect(() => {
    const ws = new WebSocket(`${WS_URL}/analysis/${videoId}`)
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'progress') {
        setProgress(data.value)
      } else if (data.type === 'complete') {
        setAnalysis(data.analysis)
      }
    }
    
    return () => ws.close()
  }, [videoId])
  
  return { analysis, progress, error }
}
```

2. PLAYER VIDÉO AVANCÉ :
   ```tsx
   const AnnotatedVideoPlayer: React.FC<Props> = ({ video, annotations }) => {
     return (
       <VideoPlayer
         src={video.url}
         overlay={
           <AnnotationLayer
             players={annotations.players}
             ball={annotations.ball}
             tactics={annotations.tactics}
             showSkeleton={settings.showPose}
             showTrajectories={settings.showPaths}
           />
         }
         controls={
           <CustomControls
             onSpeedChange={handleSpeedChange}
             onFrameStep={handleFrameStep}
             bookmarks={annotations.keyMoments}
           />
         }
         sidebar={
           <AnalysisPanel
             currentTime={currentTime}
             metrics={getCurrentMetrics(currentTime)}
             feedback={getCurrentFeedback(currentTime)}
           />
         }
       />
     )
   }
   ```

3. VISUALISATIONS INTERACTIVES :
   - Graphiques D3.js/Victory
   - Terrain 2D/3D Three.js
   - Heatmaps dynamiques
   - Animations trajectoires
   - Comparaisons side-by-side

4. RESPONSIVE DESIGN :
   - Mobile-first approach
   - PWA capabilities
   - Offline mode partiel
   - Touch gestures
   - Adaptation résolution

5. PERFORMANCE :
   - Code splitting routes
   - Lazy loading composants
   - Service worker caching
   - CDN assets statiques
   - WebAssembly pour calculs

Déploie avec CI/CD automatisé.
```

## 3️⃣ Prompt Infrastructure Kubernetes

```
Configure une infrastructure Kubernetes production-ready pour l'application :

1. ARCHITECTURE K8S :
```yaml
# Namespace et RBAC
apiVersion: v1
kind: Namespace
metadata:
  name: football-ai-prod
  
---
# Deployment API principale
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: football-ai-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
    spec:
      containers:
      - name: api
        image: football-ai/api:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          
---
# Service ML avec GPU
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
  namespace: football-ai-prod
spec:
  replicas: 2
  template:
    spec:
      nodeSelector:
        gpu: "true"
      containers:
      - name: inference
        image: football-ai/ml-inference:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

2. AUTOSCALING :
   ```yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: api-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: api-server
     minReplicas: 3
     maxReplicas: 20
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
     - type: Resource
       resource:
         name: memory
         target:
           type: Utilization
           averageUtilization: 80
     - type: Pods
       pods:
         metric:
           name: http_requests_per_second
         target:
           type: AverageValue
           averageValue: "1000"
   ```

3. INGRESS & TLS :
   - Nginx Ingress Controller
   - Cert-manager Let's Encrypt
   - Rate limiting
   - WAF rules
   - DDoS protection

4. MONITORING STACK :
   - Prometheus metrics
   - Grafana dashboards
   - ELK logging
   - Jaeger tracing
   - Alertmanager

5. BACKUP & DR :
   - Velero backups
   - Cross-region replication
   - Disaster recovery plan
   - RTO < 1h, RPO < 15min
   - Chaos engineering tests

Automatise avec Terraform/Pulumi.
```

## 4️⃣ Prompt Monitoring Production

```
Implémente un système de monitoring complet pour la production :

1. MÉTRIQUES MÉTIER :
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Métriques custom
video_processed = Counter(
    'football_videos_processed_total',
    'Total number of videos processed',
    ['status', 'video_type', 'customer_tier']
)

processing_duration = Histogram(
    'football_video_processing_duration_seconds',
    'Video processing duration in seconds',
    ['video_resolution', 'duration_category'],
    buckets=[30, 60, 120, 300, 600, 1200, 3600]
)

active_analyses = Gauge(
    'football_active_analyses',
    'Number of currently running analyses'
)

model_accuracy = Gauge(
    'football_model_accuracy',
    'Current model accuracy score',
    ['model_type', 'version']
)

class MetricsCollector:
    @staticmethod
    def record_video_processing(video_metadata, duration, status):
        video_processed.labels(
            status=status,
            video_type=video_metadata.type,
            customer_tier=video_metadata.customer_tier
        ).inc()
        
        processing_duration.labels(
            video_resolution=video_metadata.resolution,
            duration_category=MetricsCollector.categorize_duration(video_metadata.duration)
        ).observe(duration)
        
    @staticmethod
    def update_model_metrics(model_type, version, metrics):
        model_accuracy.labels(
            model_type=model_type,
            version=version
        ).set(metrics.accuracy)
```

2. DASHBOARDS GRAFANA :
   ```json
   {
     "dashboard": {
       "title": "Football AI Production Metrics",
       "panels": [
         {
           "title": "Video Processing Rate",
           "targets": [{
             "expr": "rate(football_videos_processed_total[5m])"
           }]
         },
         {
           "title": "Processing Latency P95",
           "targets": [{
             "expr": "histogram_quantile(0.95, rate(football_video_processing_duration_seconds_bucket[5m]))"
           }]
         },
         {
           "title": "Model Performance",
           "targets": [{
             "expr": "football_model_accuracy"
           }]
         },
         {
           "title": "Error Rate by Type",
           "targets": [{
             "expr": "rate(football_errors_total[5m]) by (error_type)"
           }]
         }
       ]
     }
   }
   ```

3. ALERTING RULES :
   ```yaml
   groups:
   - name: football_ai_alerts
     rules:
     - alert: HighErrorRate
       expr: rate(football_errors_total[5m]) > 0.05
       for: 5m
       annotations:
         summary: "High error rate detected"
         
     - alert: SlowProcessing
       expr: histogram_quantile(0.95, rate(football_video_processing_duration_seconds_bucket[5m])) > 600
       for: 10m
       annotations:
         summary: "Video processing is slow"
         
     - alert: ModelAccuracyDrop
       expr: football_model_accuracy < 0.8
       for: 15m
       annotations:
         summary: "Model accuracy below threshold"
   ```

4. LOGGING STRUCTURÉ :
   ```python
   import structlog
   
   logger = structlog.get_logger()
   
   logger.info(
       "video_analysis_complete",
       video_id=video.id,
       duration=analysis_duration,
       scores={
           "technical": result.technical_score,
           "tactical": result.tactical_score,
           "physical": result.physical_score
       },
       processing_node=node_id,
       customer_id=customer.id
   )
   ```

5. BUSINESS INTELLIGENCE :
   - Tableau de bord direction
   - KPIs usage par client
   - ROI par fonctionnalité
   - Tendances adoption
   - Prévisions croissance

Intègre avec PagerDuty pour on-call.
```

## 5️⃣ Prompt API Documentation

```
Génère une documentation API complète et interactive :

1. OPENAPI SPECIFICATION :
```yaml
openapi: 3.1.0
info:
  title: Football AI Analysis API
  version: 1.0.0
  description: |
    API pour l'analyse vidéo IA appliquée au football.
    
    ## Authentification
    L'API utilise JWT pour l'authentification. Incluez le token dans le header:
    ```
    Authorization: Bearer <your_token>
    ```
    
    ## Rate Limiting
    - Free tier: 10 requêtes/minute
    - Pro tier: 100 requêtes/minute
    - Enterprise: Illimité
    
    ## Webhooks
    Configurez des webhooks pour recevoir les résultats d'analyse.

servers:
  - url: https://api.football-ai.com/v1
    description: Production
  - url: https://staging-api.football-ai.com/v1
    description: Staging

paths:
  /analyses:
    post:
      summary: Créer une nouvelle analyse vidéo
      operationId: createAnalysis
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                video:
                  type: string
                  format: binary
                  description: Fichier vidéo (max 2GB)
                metadata:
                  $ref: '#/components/schemas/VideoMetadata'
              required:
                - video
      responses:
        '202':
          description: Analyse créée et en cours
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AnalysisResponse'
              examples:
                basic:
                  value:
                    id: "ana_1234567890"
                    status: "processing"
                    estimated_completion: "2024-01-15T10:30:00Z"
                    webhook_url: "https://your-app.com/webhook"
        '400':
          $ref: '#/components/responses/BadRequest'
        '413':
          $ref: '#/components/responses/PayloadTooLarge'
```

2. SDK CLIENTS :
   ```typescript
   // TypeScript SDK
   import { FootballAI } from '@football-ai/sdk';
   
   const client = new FootballAI({
     apiKey: process.env.FOOTBALL_AI_API_KEY,
   });
   
   // Upload et analyse
   const analysis = await client.analyses.create({
     video: videoFile,
     metadata: {
       playerName: 'John Doe',
       position: 'midfielder',
       sessionType: 'training'
     },
     webhookUrl: 'https://myapp.com/webhook'
   });
   
   // Polling status
   const result = await client.analyses.waitForCompletion(analysis.id, {
     pollInterval: 5000,
     timeout: 600000
   });
   
   // Récupération rapport
   const report = await client.reports.generate(analysis.id, {
     format: 'pdf',
     language: 'fr',
     includeVideo: true
   });
   ```

3. EXEMPLES INTERACTIFS :
   - Postman collection
   - Swagger UI intégré
   - Code examples multi-langages
   - Vidéos tutoriels
   - Sandbox environnement

4. GUIDES INTÉGRATION :
   - Quick start guide
   - Best practices
   - Cas d'usage types
   - Troubleshooting
   - Migration guide

5. CHANGELOG VERSIONED :
   - Breaking changes
   - New features
   - Deprecations
   - Security updates
   - Performance improvements

Héberge sur docs.football-ai.com.
```

## 6️⃣ Prompt Tests End-to-End

```
Développe une suite de tests E2E complète pour validation production :

1. TESTS CYPRESS :
```javascript
// cypress/e2e/video-analysis-flow.cy.js
describe('Video Analysis Complete Flow', () => {
  beforeEach(() => {
    cy.login('test@football-ai.com', 'password123')
    cy.visit('/dashboard')
  })
  
  it('should complete full analysis workflow', () => {
    // Upload vidéo
    cy.get('[data-cy=upload-button]').click()
    cy.get('[data-cy=video-dropzone]').attachFile('test-match.mp4')
    
    // Remplir métadonnées
    cy.get('[data-cy=player-name]').type('Test Player')
    cy.get('[data-cy=position-select]').select('Attaquant')
    cy.get('[data-cy=session-type]').click().contains('Match').click()
    
    // Lancer analyse
    cy.get('[data-cy=start-analysis]').click()
    
    // Vérifier progression
    cy.get('[data-cy=progress-bar]', { timeout: 10000 })
      .should('be.visible')
      .should('have.attr', 'aria-valuenow')
      .and('match', /[0-9]+/)
    
    // Attendre completion (avec timeout long)
    cy.get('[data-cy=analysis-complete]', { timeout: 300000 })
      .should('be.visible')
    
    // Vérifier résultats
    cy.get('[data-cy=overall-score]')
      .should('contain', '/10')
    
    cy.get('[data-cy=video-player]')
      .should('be.visible')
      .find('[data-cy=annotation-overlay]')
      .should('exist')
    
    // Tester export
    cy.get('[data-cy=export-pdf]').click()
    cy.verifyDownload('analysis-report.pdf')
  })
  
  it('should handle errors gracefully', () => {
    // Test vidéo corrompue
    cy.intercept('POST', '/api/v1/analyses', {
      statusCode: 400,
      body: { error: 'Invalid video format' }
    })
    
    cy.get('[data-cy=upload-button]').click()
    cy.get('[data-cy=video-dropzone]').attachFile('corrupted.mp4')
    cy.get('[data-cy=start-analysis]').click()
    
    cy.get('[data-cy=error-message]')
      .should('be.visible')
      .and('contain', 'Invalid video format')
  })
})
```

2. TESTS PERFORMANCE :
   ```javascript
   // k6/load-test.js
   import http from 'k6/http';
   import { check, sleep } from 'k6';
   
   export let options = {
     stages: [
       { duration: '2m', target: 100 },  // Ramp up
       { duration: '5m', target: 100 },  // Stay at 100
       { duration: '2m', target: 200 },  // Spike
       { duration: '5m', target: 200 },  // Stay at 200
       { duration: '2m', target: 0 },    // Ramp down
     ],
     thresholds: {
       http_req_duration: ['p(95)<500'], // 95% < 500ms
       http_req_failed: ['rate<0.1'],    // Error rate < 10%
     },
   };
   
   export default function() {
     // Test upload endpoint
     let uploadResponse = http.post(
       'https://api.football-ai.com/v1/analyses',
       {
         video: http.file(open('./test-video.mp4', 'b'), 'video.mp4'),
         metadata: JSON.stringify({ 
           playerName: 'Load Test',
           position: 'midfielder' 
         })
       },
       {
         headers: { 
           'Authorization': `Bearer ${__ENV.API_TOKEN}` 
         }
       }
     );
     
     check(uploadResponse, {
       'upload successful': (r) => r.status === 202,
       'has analysis id': (r) => r.json('id') !== undefined,
     });
     
     sleep(1);
   }
   ```

3. TESTS SÉCURITÉ :
   - OWASP ZAP scanning
   - Injection tests (SQL, XSS)
   - Authentication bypass
   - Rate limiting validation
   - File upload exploits

4. TESTS ACCESSIBILITÉ :
   - WCAG 2.1 AA compliance
   - Screen reader testing
   - Keyboard navigation
   - Color contrast
   - Focus management

5. MONITORING SYNTHÉTIQUE :
   - Pingdom transactions
   - Datadog synthetics
   - Real user monitoring
   - Geographic distribution
   - Mobile performance

Automatise dans CI/CD pipeline.
```

## 🚀 Checklist Mise en Production

```yaml
pre_launch:
  - [ ] Tests E2E passés sur staging
  - [ ] Performance benchmarks validés
  - [ ] Security audit complété
  - [ ] Documentation API finalisée
  - [ ] SDKs publiés (npm, pypi)
  - [ ] Monitoring configuré
  - [ ] Backups testés
  - [ ] Runbooks écrits

launch_day:
  - [ ] DNS switchover
  - [ ] SSL certificates actifs
  - [ ] CDN cache primed
  - [ ] Feature flags ready
  - [ ] Support team briefé
  - [ ] Status page live

post_launch:
  - [ ] Monitor error rates
  - [ ] Check performance metrics
  - [ ] Gather user feedback
  - [ ] Fix critical bugs
  - [ ] Plan v1.1 features
```

## 📊 Success Metrics

- Uptime > 99.9%
- P95 latency < 2s
- User satisfaction > 4.5/5
- Churn rate < 5%
- Processing accuracy > 90% 