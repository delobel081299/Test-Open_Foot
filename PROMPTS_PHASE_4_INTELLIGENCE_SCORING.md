# 🧠 PROMPTS VIBE CODING - PHASE 4 : INTELLIGENCE & SCORING

## 📅 Durée : 2 semaines

## 🎯 Objectifs
- Système de scoring multi-critères
- ML pour évaluation performance
- Génération feedback intelligent
- Analyse tactique avancée

---

## 1️⃣ Prompt Feature Engineering Football

```
Développe un système complet d'extraction de features pour l'analyse football :

1. ARCHITECTURE FEATURES :
```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class FootballFeatures:
    # Features techniques (par geste)
    technical_features: Dict[str, float] = {
        "pass_accuracy": 0.0,           # Précision direction/force
        "first_touch_quality": 0.0,     # Qualité contrôle
        "shot_placement": 0.0,          # Placement tir
        "dribble_success": 0.0,         # Réussite dribbles
        "skill_consistency": 0.0        # Régularité technique
    }
    
    # Features biomécaniques
    biomechanical_features: Dict[str, float] = {
        "kinetic_chain_efficiency": 0.0,
        "balance_score": 0.0,
        "movement_economy": 0.0,
        "power_generation": 0.0,
        "coordination_index": 0.0
    }
    
    # Features tactiques
    tactical_features: Dict[str, float] = {
        "spatial_awareness": 0.0,       # Conscience spatiale
        "decision_speed": 0.0,          # Vitesse décision
        "positioning_quality": 0.0,     # Qualité placement
        "game_reading": 0.0,           # Lecture du jeu
        "team_synchronization": 0.0    # Synchro équipe
    }
    
    # Features physiques
    physical_features: Dict[str, float] = {
        "max_speed": 0.0,
        "acceleration": 0.0,
        "endurance_estimate": 0.0,
        "agility_score": 0.0,
        "strength_indicators": 0.0
    }
    
    # Features contextuelles
    contextual_features: Dict[str, float] = {
        "pressure_level": 0.0,         # Pression adversaire
        "fatigue_level": 0.0,          # Niveau fatigue
        "match_importance": 0.0,       # Importance action
        "time_remaining": 0.0,         # Temps restant
        "score_differential": 0.0      # Différence score
    }

class FeatureExtractor:
    def extract_all_features(self, 
                           video_data: VideoAnalysis,
                           pose_data: BiomechanicalAnalysis,
                           tracking_data: TrackingResults,
                           context: MatchContext) -> FootballFeatures:
        
        features = FootballFeatures()
        
        # Extraction par catégorie
        features.technical_features = self.extract_technical_features(video_data)
        features.biomechanical_features = self.extract_biomechanical_features(pose_data)
        features.tactical_features = self.extract_tactical_features(tracking_data, context)
        features.physical_features = self.extract_physical_features(tracking_data)
        features.contextual_features = self.extract_contextual_features(context)
        
        # Features composites
        features.composite_scores = self.compute_composite_features(features)
        
        return features
```

2. FEATURES AVANCÉES :
   - Heatmaps spatiales
   - Graphes de passes
   - Matrices de transition
   - Séquences temporelles
   - Patterns récurrents

3. NORMALISATION :
   - Standardisation par position
   - Ajustement âge/niveau
   - Compensation contexte
   - Benchmarks relatifs
   - Percentiles population

4. SÉLECTION FEATURES :
   - Importance SHAP
   - Mutual information
   - Recursive elimination
   - Domain knowledge
   - Validation croisée

Implémente pipeline complet avec 200+ features.
```

## 2️⃣ Prompt Modèle XGBoost Multi-Tâches

```
Développe un système de scoring ML avec XGBoost pour évaluation multi-critères :

1. ARCHITECTURE ENSEMBLE :
```python
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import optuna

class FootballScoringSystem:
    def __init__(self):
        # Modèles spécialisés par aspect
        self.models = {
            "technical": self.build_technical_model(),
            "tactical": self.build_tactical_model(),
            "physical": self.build_physical_model(),
            "overall": self.build_overall_model()
        }
        
        # Poids pour score final
        self.weights = {
            "technical": 0.35,
            "tactical": 0.30,
            "physical": 0.20,
            "contextual": 0.15
        }
        
    def build_technical_model(self):
        return MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.01,
                objective='reg:squarederror',
                tree_method='gpu_hist',  # GPU acceleration
                eval_metric=['rmse', 'mae']
            )
        )
        
    def optimize_hyperparameters(self, X_train, y_train):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5)
            }
            
            model = xgb.XGBRegressor(**params, tree_method='gpu_hist')
            # Cross-validation score
            return -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        
        return study.best_params
```

2. STRATÉGIES TRAINING :
   - Stratified K-fold par position
   - Time series split (évolution)
   - Adversarial validation
   - Sample weights (importance)
   - Early stopping patience

3. INTERPRÉTABILITÉ :
   ```python
   def explain_prediction(self, features, prediction):
       # SHAP values
       explainer = shap.TreeExplainer(self.model)
       shap_values = explainer.shap_values(features)
       
       # Feature importance
       importance = self.model.feature_importances_
       
       # Décision path
       path = self.model.decision_path(features)
       
       return ExplanationReport(shap_values, importance, path)
   ```

4. CALIBRATION SCORES :
   - Platt scaling
   - Isotonic regression  
   - Beta calibration
   - Conformal prediction
   - Uncertainty estimation

5. ONLINE LEARNING :
   - Incremental updates
   - Drift detection
   - A/B testing
   - Feedback loop
   - Model versioning

Intègre monitoring MLflow pour tracking expériences.
```

## 3️⃣ Prompt Analyse Tactique Avancée

```
Implémente un système d'analyse tactique intelligent basé sur Graph Neural Networks :

1. REPRÉSENTATION TACTIQUE :
```python
import torch
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool

class TacticalGraphNetwork(torch.nn.Module):
    def __init__(self, num_features=64, hidden_dim=128):
        super().__init__()
        
        # Encodeur joueur
        self.player_encoder = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph convolutions
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 64)
        
        # Décodeur tactique
        self.tactical_decoder = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10)  # 10 scores tactiques
        )
        
    def forward(self, x, edge_index, batch):
        # Encode players
        x = self.player_encoder(x)
        
        # Graph propagation
        x = torch.nn.functional.relu(self.conv1(x, edge_index))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.nn.functional.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        # Pooling par équipe
        x = global_mean_pool(x, batch)
        
        # Scores tactiques
        return self.tactical_decoder(x)

class TacticalAnalyzer:
    def build_match_graph(self, player_positions: List[Position], 
                         team_assignments: List[int]) -> Data:
        # Nodes : joueurs
        # Edges : proximité, passes, marking
        
        nodes = []
        edges = []
        
        for i, player in enumerate(player_positions):
            # Features joueur
            node_features = [
                player.x, player.y,  # Position
                player.velocity_x, player.velocity_y,  # Vitesse
                player.distance_to_ball,
                player.team_role_encoding,  # Rôle tactique
                *player.recent_actions  # Actions récentes
            ]
            nodes.append(node_features)
            
        # Construction edges
        for i in range(len(player_positions)):
            for j in range(i+1, len(player_positions)):
                # Proximité spatiale
                if distance(player_positions[i], player_positions[j]) < 10:
                    edges.append([i, j])
                    
                # Même équipe - lignes de passe
                if team_assignments[i] == team_assignments[j]:
                    if self.is_passing_lane_open(i, j):
                        edges.append([i, j])
```

2. MÉTRIQUES TACTIQUES :
   - Compacité équipe
   - Largeur/Profondeur
   - Distance entre lignes
   - Synchronisation pressing
   - Déséquilibres créés

3. PATTERNS TACTIQUES :
   - Détection formations
   - Transitions phases
   - Mouvements collectifs
   - Stratégies récurrentes
   - Adaptations adversaire

4. PRÉDICTION ACTIONS :
   - Prochain mouvement probable
   - Zones dangereuses
   - Opportunités création
   - Risques défensifs
   - Suggestions positionnement

5. APPRENTISSAGE STYLE :
   - Clustering équipes similaires
   - Profils jeu caractéristiques
   - Evolution tactique match
   - Comparaison entraîneurs
   - Tendances league

Génère visualisations tactiques interactives.
```

## 4️⃣ Prompt LLM Feedback Personnalisé

```
Configure un système de génération de feedback utilisant un LLM fine-tuné :

1. FINE-TUNING MISTRAL/LLAMA :
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model

class FootballFeedbackGenerator:
    def __init__(self, base_model="mistralai/Mistral-7B-Instruct-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Configuration LoRA pour fine-tuning efficace
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
    def prepare_training_data(self):
        # Format données coaching
        training_examples = []
        
        for analysis in coaching_database:
            prompt = f"""
            [CONTEXT]
            Joueur: {analysis.player_profile}
            Geste analysé: {analysis.gesture_type}
            Score biomécanique: {analysis.biomech_score}/10
            Score tactique: {analysis.tactical_score}/10
            Erreurs détectées: {analysis.errors}
            
            [INSTRUCTION]
            Génère un feedback constructif et personnalisé pour aider le joueur à progresser.
            
            [FEEDBACK]
            {analysis.expert_feedback}
            """
            training_examples.append(prompt)
            
        return training_examples
```

2. TEMPLATES FEEDBACK :
   ```python
   feedback_templates = {
       "beginner": """
       👋 Salut {player_name} !
       
       ✅ Points positifs :
       {strengths}
       
       🎯 Axes de progression :
       {improvements}
       
       💡 Conseil du jour :
       {main_tip}
       
       🏃 Exercice recommandé :
       {exercise}
       """,
       
       "intermediate": """
       📊 Analyse détaillée - {gesture_type}
       
       Performance globale : {overall_score}/10
       
       Biomécanique :
       {biomech_details}
       
       Points techniques :
       {technical_points}
       
       Plan de progression :
       {progression_plan}
       """,
       
       "advanced": """
       ⚡ Analyse performance élite
       
       Comparaison références :
       {pro_comparison}
       
       Micro-ajustements :
       {fine_tuning}
       
       Optimisation contextuelle :
       {contextual_optimization}
       """
   }
   ```

3. PERSONNALISATION :
   - Adaptation niveau/âge
   - Historique progression
   - Style apprentissage
   - Objectifs personnels
   - Contexte culturel

4. MULTIMODALITÉ :
   - Feedback textuel
   - Annotations vidéo
   - Schémas tactiques
   - Graphiques progression
   - Audio coaching

5. AMÉLIORATION CONTINUE :
   - Collecte ratings joueurs
   - A/B testing formulations
   - Analyse engagement
   - Ajustement ton/style
   - Métriques impact

Intègre avec système notification push personnalisées.
```

## 5️⃣ Prompt Dashboard Analytics

```
Crée un dashboard analytics complet pour visualisation des performances :

1. ARCHITECTURE DASHBOARD :
```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FootballAnalyticsDashboard:
    def __init__(self):
        st.set_page_config(
            page_title="Football AI Analytics",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def render_player_overview(self, player_data: PlayerAnalytics):
        col1, col2, col3, col4 = st.columns(4)
        
        # KPIs principaux
        with col1:
            st.metric(
                "Score Global", 
                f"{player_data.overall_score:.1f}/10",
                delta=f"{player_data.progression:+.1f}"
            )
            
        with col2:
            st.metric(
                "Précision Technique",
                f"{player_data.technical_accuracy:.0%}",
                delta=f"{player_data.tech_improvement:+.0%}"
            )
            
        # Radar chart compétences
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[
                player_data.passing,
                player_data.shooting, 
                player_data.dribbling,
                player_data.control,
                player_data.physical,
                player_data.tactical
            ],
            theta=['Passe', 'Tir', 'Dribble', 'Contrôle', 'Physique', 'Tactique'],
            fill='toself',
            name='Profil Joueur'
        ))
        
        st.plotly_chart(fig_radar)
        
    def render_progression_timeline(self, historical_data: List[SessionData]):
        # Graphique progression temporelle
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Score Global', 'Métriques Détaillées')
        )
        
        # Score global
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=scores,
                mode='lines+markers',
                name='Score Global',
                line=dict(width=3)
            ),
            row=1, col=1
        )
        
        # Métriques multiples
        for metric in ['technical', 'tactical', 'physical']:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=historical_data[metric],
                    mode='lines',
                    name=metric.capitalize()
                ),
                row=2, col=1
            )
```

2. VISUALISATIONS AVANCÉES :
   - Heatmaps terrain
   - Animations 3D gestes
   - Networks passes
   - Sunburst hiérarchique
   - Sankey flux jeu

3. COMPARAISONS :
   - Vs moyenne position
   - Vs joueurs élite
   - Evolution temporelle
   - Benchmarks équipe
   - Rankings globaux

4. EXPORT RAPPORTS :
   - PDF personnalisé
   - PowerPoint coach
   - Excel données brutes
   - Vidéo highlights
   - Partage sécurisé

5. REAL-TIME UPDATES :
   - WebSocket streaming
   - Notifications push
   - Alertes performance
   - Live coaching
   - Chat intégré

Déploie sur cloud avec auto-scaling.
```

## 6️⃣ Prompt Optimisation Inference

```
Optimise le pipeline d'inférence pour production haute performance :

1. OPTIMISATIONS MODÈLES :
```python
import torch
import tensorrt as trt
import onnxruntime as ort

class OptimizedInferencePipeline:
    def __init__(self):
        self.optimization_strategies = {
            "quantization": self.setup_quantization(),
            "pruning": self.setup_pruning(),
            "distillation": self.setup_distillation(),
            "tensorrt": self.setup_tensorrt()
        }
        
    def quantize_model(self, model, calibration_data):
        # INT8 quantization avec calibration
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={
                torch.nn.Linear: torch.quantization.default_dynamic_qconfig,
                torch.nn.Conv2d: torch.quantization.default_dynamic_qconfig
            },
            dtype=torch.qint8
        )
        
        # Calibration post-training
        quantized_model = self.calibrate_quantized_model(
            quantized_model,
            calibration_data
        )
        
        return quantized_model
        
    def convert_to_tensorrt(self, onnx_model_path):
        # Conversion ONNX -> TensorRT
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        config.set_flag(trt.BuilderFlag.FP16)  # FP16 precision
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        
        # Optimisation profils dynamiques
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",
            (1, 3, 224, 224),    # min
            (8, 3, 736, 736),    # opt
            (32, 3, 1280, 1280)  # max
        )
        
        return engine
```

2. BATCHING INTELLIGENT :
   - Dynamic batching
   - Priority queuing
   - Adaptive batch size
   - Request coalescing
   - Timeout handling

3. CACHING STRATÉGIQUE :
   - Redis résultats
   - Edge caching CDN
   - Feature store
   - Model versioning
   - Invalidation smart

4. PARALLÉLISATION :
   - Multi-GPU inference
   - Pipeline parallelism
   - Model sharding
   - Async processing
   - Stream overlapping

5. MONITORING LATENCE :
   - P50/P95/P99 metrics
   - Request tracing
   - Bottleneck detection
   - Auto-scaling triggers
   - SLA enforcement

Cible <100ms latence E2E pour 95% requêtes.
```

## 🎯 Métriques Cibles

```yaml
accuracy_targets:
  technical_score: MAE < 0.5
  tactical_score: MAE < 0.7  
  overall_correlation: > 0.85 with experts

performance_targets:
  inference_latency_p95: < 100ms
  throughput: > 1000 requests/sec
  gpu_utilization: > 80%
  
quality_targets:
  feedback_satisfaction: > 4.5/5
  actionability_score: > 90%
  progression_correlation: > 0.7
```

## 📝 Checklist Phase

- [ ] Feature extraction pipeline complet
- [ ] XGBoost models trained & validated
- [ ] GNN tactical analysis fonctionnel
- [ ] LLM feedback fine-tuné
- [ ] Dashboard analytics déployé
- [ ] Optimisations production appliquées
- [ ] Tests charge passés
- [ ] Documentation API complète 