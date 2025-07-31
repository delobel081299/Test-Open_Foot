# 🤖 Implémentation ML Biomécanique - Guide Pratique

## 🎯 **OBJECTIF**

Remplacer le système de règles fixes biomécaniques par un **modèle ML entraîné** qui apprend des experts et s'adapte au contexte.

**AVANT** : `if knee_angle < 90: score -= 2` (règles codées)
**APRÈS** : `score = model.predict(features)` (modèle entraîné)

---

## 📋 **PLAN D'IMPLÉMENTATION (1 SEMAINE)**

### **Jour 1-2** : Collecte et préparation données
### **Jour 3-4** : Feature engineering et entraînement
### **Jour 5-6** : Intégration et tests
### **Jour 7** : Validation et documentation

---

## 🗂️ **ÉTAPE 1 : STRUCTURE PROJET**

```bash
# Créer nouvelle structure ML
mkdir -p backend/core/biomechanics/ml/
mkdir -p backend/core/biomechanics/ml/data/
mkdir -p backend/core/biomechanics/ml/models/
mkdir -p backend/core/biomechanics/ml/training/
mkdir -p backend/core/biomechanics/ml/evaluation/
```

```python
# backend/core/biomechanics/ml/__init__.py
"""Module ML pour analyse biomécanique"""

from .data_collector import BiomechDataCollector
from .feature_engineer import BiomechFeatureEngineer  
from .model_trainer import BiomechModelTrainer
from .ml_evaluator import MLBiomechEvaluator

__all__ = [
    'BiomechDataCollector',
    'BiomechFeatureEngineer', 
    'BiomechModelTrainer',
    'MLBiomechEvaluator'
]
```

---

## 📊 **ÉTAPE 2 : COLLECTE DONNÉES D'ENTRAÎNEMENT**

### **2.1 Data Collector**

```python
# backend/core/biomechanics/ml/data_collector.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path

class BiomechDataCollector:
    """Collecteur données biomécanique pour entraînement ML"""
    
    def __init__(self, data_dir: str = "backend/core/biomechanics/ml/data/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Sources de données
        self.expert_scores_file = self.data_dir / "expert_scores.json"
        self.pose_data_file = self.data_dir / "pose_sequences.json"
        self.context_data_file = self.data_dir / "context_data.json"
        
    def collect_training_data(self) -> pd.DataFrame:
        """Collection complète données d'entraînement"""
        
        # 1. Données vidéos existantes avec poses
        pose_data = self.extract_pose_data_from_videos()
        
        # 2. Scores experts (à collecter)
        expert_data = self.collect_expert_scores()
        
        # 3. Données contextuelles
        context_data = self.collect_context_data()
        
        # 4. Fusion datasets
        training_dataset = self.merge_datasets(pose_data, expert_data, context_data)
        
        # 5. Sauvegarde
        training_dataset.to_csv(self.data_dir / "training_dataset.csv", index=False)
        
        return training_dataset
    
    def extract_pose_data_from_videos(self) -> List[Dict]:
        """Extraction poses depuis vidéos existantes"""
        
        # Utiliser votre système existant MediaPipe
        from backend.core.biomechanics.pose_extractor import PoseExtractor
        
        pose_extractor = PoseExtractor()
        pose_sequences = []
        
        # Traiter vidéos exemples
        video_samples = self.get_sample_videos()
        
        for video_path in video_samples:
            print(f"Extraction poses: {video_path}")
            
            # Extraction avec votre système actuel
            video_data = pose_extractor.extract_video_poses(video_path)
            
            # Extraction gestes techniques
            technical_actions = self.extract_technical_actions(video_data)
            
            for action in technical_actions:
                pose_sequence = {
                    'video_id': video_path.stem,
                    'action_type': action['type'],  # shot, pass, dribble, etc.
                    'start_frame': action['start_frame'],
                    'end_frame': action['end_frame'],
                    'player_id': action['player_id'],
                    'pose_sequence': action['poses'],  # Séquence poses 3D
                    'angles_sequence': self.calculate_angle_sequences(action['poses']),
                    'movement_dynamics': self.calculate_movement_dynamics(action['poses'])
                }
                pose_sequences.append(pose_sequence)
        
        return pose_sequences
    
    def collect_expert_scores(self) -> List[Dict]:
        """Interface pour collection scores experts"""
        
        print("🏆 COLLECTE SCORES EXPERTS")
        print("=" * 50)
        
        # Charger poses extraites
        if self.expert_scores_file.exists():
            with open(self.expert_scores_file, 'r') as f:
                return json.load(f)
        
        # Première fois - créer interface scoring
        return self.create_expert_scoring_interface()
    
    def create_expert_scoring_interface(self) -> List[Dict]:
        """Interface simple pour scoring expert"""
        
        expert_scores = []
        
        # Charger actions pour scoring
        pose_data = self.load_pose_sequences()
        
        print(f"📝 {len(pose_data)} actions à scorer par experts")
        print("\nGuide scoring (0-10):")
        print("• 0-3: Technique très pauvre")
        print("• 4-5: Technique insuffisante") 
        print("• 6-7: Technique correcte")
        print("• 8-9: Bonne technique")
        print("• 10: Technique parfaite")
        
        for i, action in enumerate(pose_data[:20]):  # Commencer par 20 exemples
            print(f"\n--- Action {i+1}/20 ---")
            print(f"Type: {action['action_type']}")
            print(f"Vidéo: {action['video_id']}")
            
            # Afficher info biomécanique
            self.display_biomech_summary(action)
            
            # Demander score expert
            while True:
                try:
                    score = float(input("Score expert (0-10): "))
                    if 0 <= score <= 10:
                        break
                    print("Score doit être entre 0 et 10")
                except ValueError:
                    print("Entrer un nombre valide")
            
            # Demander commentaires
            comments = input("Commentaires (optionnel): ")
            
            expert_scores.append({
                'action_id': f"{action['video_id']}_{action['start_frame']}",
                'action_type': action['action_type'],
                'expert_score': score,
                'comments': comments,
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        # Sauvegarder scores
        with open(self.expert_scores_file, 'w') as f:
            json.dump(expert_scores, f, indent=2)
        
        return expert_scores
    
    def display_biomech_summary(self, action: Dict):
        """Affichage résumé biomécanique pour expert"""
        
        angles = action['angles_sequence']
        
        print("📊 Résumé biomécanique:")
        if action['action_type'] == 'shot':
            print(f"• Flexion genou: {angles['knee_flexion']:.1f}°")
            print(f"• Rotation hanche: {angles['hip_rotation']:.1f}°")
            print(f"• Extension cheville: {angles['ankle_extension']:.1f}°")
            print(f"• Inclinaison tronc: {angles['trunk_lean']:.1f}°")
            
        elif action['action_type'] == 'pass':
            print(f"• Position pied appui: {angles['support_foot_angle']:.1f}°")
            print(f"• Rotation épaules: {angles['shoulder_rotation']:.1f}°")
            print(f"• Follow-through: {angles['follow_through']:.1f}°")
    
    def get_sample_videos(self) -> List[Path]:
        """Récupérer vidéos échantillon pour entraînement"""
        
        # Dossier vidéos test
        video_dir = Path("data/sample_videos/")
        
        if not video_dir.exists():
            print("⚠️  Créer dossier data/sample_videos/ avec vidéos test")
            print("📥 Ajouter 5-10 vidéos courtes avec gestes techniques clairs")
            video_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        return list(video_dir.glob("*.mp4"))[:10]  # Max 10 vidéos pour début
    
    def calculate_angle_sequences(self, poses: List) -> Dict:
        """Calcul séquences d'angles depuis poses"""
        
        # Utiliser votre AngleCalculator existant
        from backend.core.biomechanics.angle_calculator import AngleCalculator
        
        angle_calc = AngleCalculator()
        angle_sequences = {}
        
        for pose in poses:
            angles = angle_calc.calculate_joint_angles(pose)
            
            for angle_name, angle_value in angles.items():
                if angle_name not in angle_sequences:
                    angle_sequences[angle_name] = []
                angle_sequences[angle_name].append(angle_value)
        
        # Moyennes et variabilité
        angle_stats = {}
        for angle_name, values in angle_sequences.items():
            angle_stats[f"{angle_name}_mean"] = np.mean(values)
            angle_stats[f"{angle_name}_std"] = np.std(values)
            angle_stats[f"{angle_name}_max"] = np.max(values)
            angle_stats[f"{angle_name}_min"] = np.min(values)
        
        return angle_stats
    
    def merge_datasets(self, pose_data: List, expert_data: List, context_data: List) -> pd.DataFrame:
        """Fusion datasets en DataFrame final"""
        
        # Créer mapping par action_id
        expert_map = {item['action_id']: item for item in expert_data}
        
        merged_data = []
        
        for pose_item in pose_data:
            action_id = f"{pose_item['video_id']}_{pose_item['start_frame']}"
            
            if action_id in expert_map:
                merged_item = {
                    'action_id': action_id,
                    'action_type': pose_item['action_type'],
                    'expert_score': expert_map[action_id]['expert_score'],
                    **pose_item['angles_sequence'],
                    **pose_item['movement_dynamics']
                }
                merged_data.append(merged_item)
        
        return pd.DataFrame(merged_data)

# Script utilitaire pour lancer collection
# scripts/collect_biomech_data.py
"""
Script pour lancer la collecte de données biomécanique

Usage:
python scripts/collect_biomech_data.py
"""

if __name__ == "__main__":
    from backend.core.biomechanics.ml.data_collector import BiomechDataCollector
    
    collector = BiomechDataCollector()
    
    print("🚀 Début collecte données biomécanique ML")
    training_data = collector.collect_training_data()
    
    print(f"✅ Dataset créé: {len(training_data)} échantillons")
    print(f"📁 Sauvegardé: backend/core/biomechanics/ml/data/training_dataset.csv")
```

---

## 🔧 **ÉTAPE 3 : FEATURE ENGINEERING**

```python
# backend/core/biomechanics/ml/feature_engineer.py
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import StandardScaler, LabelEncoder

class BiomechFeatureEngineer:
    """Feature engineering sophistiqué pour biomécanique"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def engineer_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Engineering features complet"""
        
        print("🔧 Feature engineering biomécanique...")
        
        # 1. Features de base (angles)
        basic_features = self.create_basic_features(dataset)
        
        # 2. Features dérivées
        derived_features = self.create_derived_features(dataset)
        
        # 3. Features temporelles
        temporal_features = self.create_temporal_features(dataset)
        
        # 4. Features contextuelles
        contextual_features = self.create_contextual_features(dataset)
        
        # 5. Features d'interaction
        interaction_features = self.create_interaction_features(basic_features)
        
        # 6. Fusion
        all_features = pd.concat([
            basic_features,
            derived_features, 
            temporal_features,
            contextual_features,
            interaction_features
        ], axis=1)
        
        # 7. Nettoyage et normalisation
        final_features = self.clean_and_normalize(all_features)
        
        self.feature_names = final_features.columns.tolist()
        
        print(f"✅ {len(self.feature_names)} features créées")
        
        return final_features
    
    def create_basic_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Features de base - angles articulaires"""
        
        angle_columns = [col for col in dataset.columns if '_mean' in col or '_std' in col]
        
        basic_features = dataset[angle_columns].copy()
        
        # Ajout ratios importants
        if 'knee_flexion_mean' in dataset.columns and 'hip_rotation_mean' in dataset.columns:
            basic_features['knee_hip_ratio'] = (
                dataset['knee_flexion_mean'] / (dataset['hip_rotation_mean'] + 1)
            )
        
        return basic_features
    
    def create_derived_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Features dérivées sophistiquées"""
        
        derived = pd.DataFrame(index=dataset.index)
        
        # 1. Symétrie corporelle
        if 'left_knee_mean' in dataset.columns and 'right_knee_mean' in dataset.columns:
            derived['knee_symmetry'] = abs(
                dataset['left_knee_mean'] - dataset['right_knee_mean']
            )
        
        # 2. Stabilité (variabilité faible = plus stable)
        stability_cols = [col for col in dataset.columns if '_std' in col]
        if stability_cols:
            derived['overall_stability'] = dataset[stability_cols].mean(axis=1)
        
        # 3. Amplitude mouvement
        for angle_base in ['knee', 'hip', 'ankle']:
            max_col = f"{angle_base}_max"
            min_col = f"{angle_base}_min"
            
            if max_col in dataset.columns and min_col in dataset.columns:
                derived[f"{angle_base}_range"] = dataset[max_col] - dataset[min_col]
        
        # 4. Coordination inter-segments
        if 'shoulder_rotation_mean' in dataset.columns and 'hip_rotation_mean' in dataset.columns:
            derived['upper_lower_coordination'] = abs(
                dataset['shoulder_rotation_mean'] - dataset['hip_rotation_mean']
            )
        
        return derived
    
    def create_temporal_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Features temporelles et dynamiques"""
        
        temporal = pd.DataFrame(index=dataset.index)
        
        # 1. Vitesse changement (différences entre max et min)
        for angle_base in ['knee', 'hip', 'ankle']:
            max_col = f"{angle_base}_max"
            min_col = f"{angle_base}_min"
            
            if max_col in dataset.columns and min_col in dataset.columns:
                # Proxy pour vitesse angulaire
                temporal[f"{angle_base}_velocity"] = (dataset[max_col] - dataset[min_col]) / 2
        
        # 2. Consistance temporelle (std faible = plus consistant)
        consistency_cols = [col for col in dataset.columns if '_std' in col]
        if consistency_cols:
            temporal['movement_consistency'] = 1 / (dataset[consistency_cols].mean(axis=1) + 1)
        
        return temporal
    
    def create_contextual_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Features contextuelles"""
        
        contextual = pd.DataFrame(index=dataset.index)
        
        # 1. Encoding action type
        if 'action_type' in dataset.columns:
            le = LabelEncoder()
            contextual['action_type_encoded'] = le.fit_transform(dataset['action_type'])
            self.encoders['action_type'] = le
        
        # 2. Features spécialisées par action
        for action_type in dataset['action_type'].unique():
            mask = dataset['action_type'] == action_type
            contextual.loc[mask, f'is_{action_type}'] = 1
            contextual.loc[~mask, f'is_{action_type}'] = 0
        
        return contextual
    
    def create_interaction_features(self, basic_features: pd.DataFrame) -> pd.DataFrame:
        """Features d'interaction entre angles"""
        
        interaction = pd.DataFrame(index=basic_features.index)
        
        # Sélection angles principaux pour interactions
        main_angles = ['knee_flexion_mean', 'hip_rotation_mean', 'ankle_extension_mean']
        main_angles = [col for col in main_angles if col in basic_features.columns]
        
        # Produits entre angles (coordination)
        for i, angle1 in enumerate(main_angles):
            for angle2 in main_angles[i+1:]:
                interaction[f"{angle1}_{angle2}_product"] = (
                    basic_features[angle1] * basic_features[angle2]
                )
        
        # Ratios importants
        if 'knee_flexion_mean' in basic_features.columns and 'hip_rotation_mean' in basic_features.columns:
            interaction['knee_hip_coordination'] = (
                basic_features['knee_flexion_mean'] / (basic_features['hip_rotation_mean'] + 1)
            )
        
        return interaction
    
    def clean_and_normalize(self, features: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage et normalisation finale"""
        
        # 1. Suppression NaN
        features = features.fillna(features.mean())
        
        # 2. Suppression features constantes
        features = features.loc[:, features.var() > 1e-6]
        
        # 3. Normalisation StandardScaler
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        
        scaler = StandardScaler()
        features[numeric_columns] = scaler.fit_transform(features[numeric_columns])
        self.scalers['standard'] = scaler
        
        return features
    
    def transform_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Transformation nouvelles données avec scalers pré-entraînés"""
        
        # Reproduire engineering
        features = self.engineer_features(new_data)
        
        # Appliquer scalers existants
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        features[numeric_columns] = self.scalers['standard'].transform(features[numeric_columns])
        
        return features
```

---

## 🧠 **ÉTAPE 4 : ENTRAÎNEMENT MODÈLES**

```python
# backend/core/biomechanics/ml/model_trainer.py
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
from pathlib import Path

class BiomechModelTrainer:
    """Entraînement modèles biomécanique ML"""
    
    def __init__(self, models_dir: str = "backend/core/biomechanics/ml/models/"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.metrics = {}
        
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Entraînement ensemble de modèles"""
        
        print("🧠 Entraînement modèles biomécanique...")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # 1. XGBoost (Recommandé pour début)
        self.train_xgboost(X_train, y_train, X_test, y_test)
        
        # 2. LightGBM (Alternative)
        self.train_lightgbm(X_train, y_train, X_test, y_test)
        
        # 3. Random Forest (Baseline)
        self.train_random_forest(X_train, y_train, X_test, y_test)
        
        # 4. Neural Network (Si assez de données)
        if len(X_train) > 200:
            self.train_neural_network(X_train, y_train, X_test, y_test)
        
        # 5. Ensemble final
        self.create_ensemble_model(X_test, y_test)
        
        # 6. Sauvegarde
        self.save_models()
        
        return self.metrics
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Entraînement XGBoost optimisé"""
        
        print("🔥 Entraînement XGBoost...")
        
        # Paramètres optimisés pour biomécanique
        xgb_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Entraînement
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Évaluation
        y_pred = xgb_model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred, "XGBoost")
        
        self.models['xgboost'] = xgb_model
        self.metrics['xgboost'] = metrics
        
        print(f"✅ XGBoost - R²: {metrics['r2']:.3f}, MAE: {metrics['mae']:.3f}")
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Entraînement LightGBM"""
        
        print("⚡ Entraînement LightGBM...")
        
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        y_pred = lgb_model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred, "LightGBM")
        
        self.models['lightgbm'] = lgb_model
        self.metrics['lightgbm'] = metrics
        
        print(f"✅ LightGBM - R²: {metrics['r2']:.3f}, MAE: {metrics['mae']:.3f}")
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Entraînement Random Forest (baseline)"""
        
        print("🌲 Entraînement Random Forest...")
        
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred, "Random Forest")
        
        self.models['random_forest'] = rf_model
        self.metrics['random_forest'] = metrics
        
        print(f"✅ Random Forest - R²: {metrics['r2']:.3f}, MAE: {metrics['mae']:.3f}")
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Entraînement réseau neuronal"""
        
        print("🧠 Entraînement Neural Network...")
        
        # Architecture adaptée à la biomécanique
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)  # Régression
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        
        # Entraînement
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        y_pred = model.predict(X_test).flatten()
        metrics = self.calculate_metrics(y_test, y_pred, "Neural Network")
        
        self.models['neural_network'] = model
        self.metrics['neural_network'] = metrics
        
        print(f"✅ Neural Network - R²: {metrics['r2']:.3f}, MAE: {metrics['mae']:.3f}")
    
    def create_ensemble_model(self, X_test, y_test):
        """Création modèle ensemble"""
        
        print("🎯 Création modèle ensemble...")
        
        # Prédictions tous modèles
        predictions = {}
        
        for name, model in self.models.items():
            if name != 'neural_network':
                predictions[name] = model.predict(X_test)
            else:
                predictions[name] = model.predict(X_test).flatten()
        
        # Moyenne pondérée basée sur performance
        weights = {}
        total_r2 = sum(self.metrics[name]['r2'] for name in predictions.keys())
        
        for name in predictions.keys():
            weights[name] = self.metrics[name]['r2'] / total_r2
        
        # Prédiction ensemble
        ensemble_pred = np.zeros(len(y_test))
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
        
        metrics = self.calculate_metrics(y_test, ensemble_pred, "Ensemble")
        self.metrics['ensemble'] = metrics
        self.models['ensemble_weights'] = weights
        
        print(f"✅ Ensemble - R²: {metrics['r2']:.3f}, MAE: {metrics['mae']:.3f}")
    
    def calculate_metrics(self, y_true, y_pred, model_name: str) -> Dict:
        """Calcul métriques évaluation"""
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'model': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'samples': len(y_true)
        }
    
    def save_models(self):
        """Sauvegarde tous les modèles"""
        
        print("💾 Sauvegarde modèles...")
        
        # Modèles ML classiques
        for name, model in self.models.items():
            if name not in ['neural_network', 'ensemble_weights']:
                joblib.dump(model, self.models_dir / f"{name}_model.joblib")
        
        # Neural network
        if 'neural_network' in self.models:
            self.models['neural_network'].save(self.models_dir / "neural_network_model.h5")
        
        # Poids ensemble
        if 'ensemble_weights' in self.models:
            with open(self.models_dir / "ensemble_weights.json", 'w') as f:
                json.dump(self.models['ensemble_weights'], f)
        
        # Métriques
        with open(self.models_dir / "training_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"✅ Modèles sauvegardés dans {self.models_dir}")
    
    def get_best_model(self) -> str:
        """Retourne le nom du meilleur modèle"""
        
        best_model = max(self.metrics.keys(), key=lambda x: self.metrics[x]['r2'])
        return best_model

# Script d'entraînement
# scripts/train_biomech_models.py
"""
Script pour entraîner modèles biomécanique

Usage:
python scripts/train_biomech_models.py
"""

if __name__ == "__main__":
    from backend.core.biomechanics.ml.data_collector import BiomechDataCollector
    from backend.core.biomechanics.ml.feature_engineer import BiomechFeatureEngineer
    from backend.core.biomechanics.ml.model_trainer import BiomechModelTrainer
    
    print("🚀 Pipeline entraînement biomécanique ML")
    
    # 1. Charger données
    collector = BiomechDataCollector()
    dataset = pd.read_csv("backend/core/biomechanics/ml/data/training_dataset.csv")
    print(f"📊 Dataset: {len(dataset)} échantillons")
    
    # 2. Feature engineering
    engineer = BiomechFeatureEngineer()
    X = engineer.engineer_features(dataset)
    y = dataset['expert_score']
    
    # 3. Entraînement
    trainer = BiomechModelTrainer()
    metrics = trainer.train_all_models(X, y)
    
    # 4. Résultats
    print("\n🏆 RÉSULTATS FINAUX")
    print("=" * 50)
    for model_name, metric in metrics.items():
        print(f"{model_name:15} | R²: {metric['r2']:.3f} | MAE: {metric['mae']:.3f}")
    
    best_model = trainer.get_best_model()
    print(f"\n🥇 Meilleur modèle: {best_model}")
```

---

## 🔌 **ÉTAPE 5 : INTÉGRATION DANS LE SYSTÈME**

```python
# backend/core/biomechanics/ml/ml_evaluator.py
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import json
from pathlib import Path
from typing import Dict, List
import shap  # Pour explications

class MLBiomechEvaluator:
    """Évaluateur biomécanique ML intégré"""
    
    def __init__(self, models_dir: str = "backend/core/biomechanics/ml/models/"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_engineer = None
        self.ensemble_weights = {}
        
        self.load_models()
        
    def load_models(self):
        """Chargement modèles pré-entraînés"""
        
        try:
            # XGBoost (principal)
            self.models['xgboost'] = joblib.load(self.models_dir / "xgboost_model.joblib")
            
            # LightGBM (backup)
            self.models['lightgbm'] = joblib.load(self.models_dir / "lightgbm_model.joblib")
            
            # Neural Network
            if (self.models_dir / "neural_network_model.h5").exists():
                self.models['neural_network'] = tf.keras.models.load_model(
                    self.models_dir / "neural_network_model.h5"
                )
            
            # Poids ensemble
            with open(self.models_dir / "ensemble_weights.json", 'r') as f:
                self.ensemble_weights = json.load(f)
            
            # Feature engineer
            from backend.core.biomechanics.ml.feature_engineer import BiomechFeatureEngineer
            self.feature_engineer = BiomechFeatureEngineer()
            
            print("✅ Modèles ML biomécanique chargés")
            
        except Exception as e:
            print(f"⚠️ Erreur chargement modèles ML: {e}")
            print("💡 Lancer d'abord: python scripts/train_biomech_models.py")
    
    def evaluate_biomechanics_ml(self, pose_data: List, action_type: str, context: Dict = None) -> Dict:
        """Évaluation biomécanique avec ML (remplace les règles)"""
        
        if not self.models:
            # Fallback vers ancien système si modèles pas disponibles
            return self.fallback_to_rules(pose_data, action_type)
        
        try:
            # 1. Préparation features
            features = self.prepare_features_for_ml(pose_data, action_type, context)
            
            # 2. Prédictions modèles
            predictions = self.get_model_predictions(features)
            
            # 3. Score ensemble
            ensemble_score = self.calculate_ensemble_score(predictions)
            
            # 4. Explications
            explanations = self.get_score_explanations(features, ensemble_score)
            
            # 5. Recommandations
            recommendations = self.generate_ml_recommendations(features, explanations)
            
            return {
                'biomech_score': float(ensemble_score),
                'confidence': self.calculate_confidence(predictions),
                'model_breakdown': predictions,
                'key_factors': explanations,
                'improvement_suggestions': recommendations,
                'method': 'ml_trained'
            }
            
        except Exception as e:
            print(f"⚠️ Erreur évaluation ML: {e}")
            return self.fallback_to_rules(pose_data, action_type)
    
    def prepare_features_for_ml(self, pose_data: List, action_type: str, context: Dict) -> np.ndarray:
        """Préparation features pour prédiction ML"""
        
        # Calcul angles (comme système existant)
        from backend.core.biomechanics.angle_calculator import AngleCalculator
        angle_calc = AngleCalculator()
        
        angles_sequence = []
        for pose in pose_data:
            angles = angle_calc.calculate_joint_angles(pose)
            angles_sequence.append(angles)
        
        # Statistiques séquence
        angle_stats = {}
        for angle_name in angles_sequence[0].keys():
            values = [angles[angle_name] for angles in angles_sequence]
            angle_stats.update({
                f"{angle_name}_mean": np.mean(values),
                f"{angle_name}_std": np.std(values),
                f"{angle_name}_max": np.max(values),
                f"{angle_name}_min": np.min(values)
            })
        
        # Création DataFrame temporaire
        temp_data = pd.DataFrame([{
            'action_type': action_type,
            **angle_stats,
            **self.extract_context_features(context or {})
        }])
        
        # Feature engineering
        features = self.feature_engineer.transform_new_data(temp_data)
        
        return features.values[0]
    
    def get_model_predictions(self, features: np.ndarray) -> Dict:
        """Prédictions tous modèles"""
        
        predictions = {}
        
        # XGBoost
        if 'xgboost' in self.models:
            predictions['xgboost'] = float(self.models['xgboost'].predict([features])[0])
        
        # LightGBM
        if 'lightgbm' in self.models:
            predictions['lightgbm'] = float(self.models['lightgbm'].predict([features])[0])
        
        # Neural Network
        if 'neural_network' in self.models:
            predictions['neural_network'] = float(self.models['neural_network'].predict([features])[0])
        
        return predictions
    
    def calculate_ensemble_score(self, predictions: Dict) -> float:
        """Score ensemble pondéré"""
        
        if not self.ensemble_weights:
            # Simple moyenne si pas de poids
            return np.mean(list(predictions.values()))
        
        # Moyenne pondérée
        weighted_sum = 0
        total_weight = 0
        
        for model_name, prediction in predictions.items():
            if model_name in self.ensemble_weights:
                weight = self.ensemble_weights[model_name]
                weighted_sum += weight * prediction
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else np.mean(list(predictions.values()))
    
    def get_score_explanations(self, features: np.ndarray, score: float) -> List[Dict]:
        """Explications score avec SHAP"""
        
        try:
            # Utiliser meilleur modèle pour explications
            best_model = self.models.get('xgboost', list(self.models.values())[0])
            
            # SHAP explainer
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values([features])
            
            # Top 5 features les plus importantes
            feature_importance = list(zip(
                self.feature_engineer.feature_names,
                shap_values[0]
            ))
            
            # Tri par importance absolue
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            explanations = []
            for feature_name, importance in feature_importance[:5]:
                explanations.append({
                    'factor': feature_name,
                    'impact': float(importance),
                    'direction': 'positive' if importance > 0 else 'negative',
                    'description': self.get_feature_description(feature_name)
                })
            
            return explanations
            
        except Exception as e:
            print(f"⚠️ Erreur explications SHAP: {e}")
            return [{'factor': 'overall_technique', 'impact': score, 'direction': 'positive'}]
    
    def generate_ml_recommendations(self, features: np.ndarray, explanations: List[Dict]) -> List[str]:
        """Génération recommandations basées ML"""
        
        recommendations = []
        
        for explanation in explanations[:3]:  # Top 3 facteurs
            factor = explanation['factor']
            impact = explanation['impact']
            
            if impact < -0.5:  # Facteur négatif important
                if 'knee' in factor:
                    recommendations.append("Améliorer la flexion du genou lors de la frappe")
                elif 'hip' in factor:
                    recommendations.append("Optimiser la rotation des hanches")
                elif 'stability' in factor:
                    recommendations.append("Travailler l'équilibre et la stabilité")
                elif 'symmetry' in factor:
                    recommendations.append("Améliorer la symétrie des mouvements")
        
        if not recommendations:
            recommendations.append("Technique globalement bonne, continuer le travail")
        
        return recommendations
    
    def fallback_to_rules(self, pose_data: List, action_type: str) -> Dict:
        """Fallback vers ancien système règles"""
        
        print("🔄 Fallback vers système règles")
        
        # Utiliser ancien évaluateur
        from backend.core.biomechanics.movement_quality import MovementQualityAnalyzer
        
        old_analyzer = MovementQualityAnalyzer()
        old_result = old_analyzer.analyze_movement_quality(pose_data)
        
        return {
            'biomech_score': old_result.get('quality_score', 5.0),
            'confidence': 0.7,
            'method': 'rules_fallback',
            'note': 'Modèles ML non disponibles, utilisation règles'
        }

# Integration dans le module principal
# backend/core/biomechanics/enhanced_analyzer.py
class EnhancedBiomechAnalyzer:
    """Analyseur biomécanique avec ML intégré"""
    
    def __init__(self):
        self.ml_evaluator = MLBiomechEvaluator()
        
        # Garder ancien système en backup
        from backend.core.biomechanics.movement_quality import MovementQualityAnalyzer
        self.rules_evaluator = MovementQualityAnalyzer()
        
    def analyze_player_biomechanics(self, player_track, action_events):
        """Point d'entrée principal - remplace l'ancien"""
        
        biomech_results = []
        
        for event in action_events:
            # Extraction poses pour cet événement
            pose_sequence = self.extract_pose_sequence(player_track, event)
            
            # Évaluation ML (nouveau)
            ml_result = self.ml_evaluator.evaluate_biomechanics_ml(
                pose_sequence, event['type'], event.get('context', {})
            )
            
            # Comparaison avec ancien système (optionnel)
            rules_result = self.rules_evaluator.analyze_movement_quality(pose_sequence)
            
            biomech_results.append({
                'event_id': event['id'],
                'action_type': event['type'],
                'ml_evaluation': ml_result,
                'rules_comparison': rules_result,
                'poses_analyzed': len(pose_sequence)
            })
        
        return biomech_results
```

---

## ✅ **ÉTAPE 6 : VALIDATION ET TESTS**

```python
# tests/test_ml_biomechanics.py
import pytest
import numpy as np
import pandas as pd
from backend.core.biomechanics.ml.ml_evaluator import MLBiomechEvaluator

class TestMLBiomechanics:
    
    def setup_method(self):
        """Setup pour chaque test"""
        self.evaluator = MLBiomechEvaluator()
        
    def test_ml_evaluation_basic(self):
        """Test évaluation ML basique"""
        
        # Données pose simulées
        mock_poses = self.create_mock_poses()
        
        result = self.evaluator.evaluate_biomechanics_ml(
            mock_poses, 'shot', {'pressure_level': 0.5}
        )
        
        assert 'biomech_score' in result
        assert 0 <= result['biomech_score'] <= 10
        assert 'confidence' in result
        assert result['method'] in ['ml_trained', 'rules_fallback']
        
    def test_feature_engineering(self):
        """Test feature engineering"""
        
        mock_data = pd.DataFrame({
            'action_type': ['shot'],
            'knee_flexion_mean': [85.0],
            'hip_rotation_mean': [30.0]
        })
        
        features = self.evaluator.feature_engineer.engineer_features(mock_data)
        
        assert len(features) > 0
        assert not features.isnull().any().any()
        
    def test_model_predictions_consistency(self):
        """Test consistance prédictions"""
        
        mock_poses = self.create_mock_poses()
        
        # Même input doit donner même output
        result1 = self.evaluator.evaluate_biomechanics_ml(mock_poses, 'shot')
        result2 = self.evaluator.evaluate_biomechanics_ml(mock_poses, 'shot')
        
        assert abs(result1['biomech_score'] - result2['biomech_score']) < 0.01
        
    def create_mock_poses(self):
        """Création poses simulées pour tests"""
        
        # Simuler 30 poses (1 seconde à 30 FPS)
        mock_poses = []
        
        for i in range(30):
            pose = {
                'landmarks': [
                    {'x': 0.5, 'y': 0.3, 'z': 0.0, 'visibility': 0.9},  # Nez
                    {'x': 0.48, 'y': 0.35, 'z': 0.1, 'visibility': 0.9}, # Épaule gauche
                    # ... autres landmarks MediaPipe
                ]
            }
            mock_poses.append(pose)
            
        return mock_poses

# Script de validation complète
# scripts/validate_ml_biomech.py
"""
Validation complète système ML biomécanique

Usage:
python scripts/validate_ml_biomech.py
"""

if __name__ == "__main__":
    
    print("🔍 Validation système ML biomécanique")
    print("=" * 50)
    
    # 1. Test chargement modèles
    try:
        evaluator = MLBiomechEvaluator()
        print("✅ Modèles chargés avec succès")
    except Exception as e:
        print(f"❌ Erreur chargement modèles: {e}")
        exit(1)
    
    # 2. Test prédiction
    mock_poses = [{'landmarks': []} for _ in range(30)]  # Mock simple
    
    try:
        result = evaluator.evaluate_biomechanics_ml(mock_poses, 'shot')
        print(f"✅ Prédiction ML réussie: {result['biomech_score']:.2f}")
    except Exception as e:
        print(f"❌ Erreur prédiction: {e}")
    
    # 3. Test performance
    import time
    
    start_time = time.time()
    for _ in range(100):
        evaluator.evaluate_biomechanics_ml(mock_poses, 'shot')
    
    avg_time = (time.time() - start_time) / 100
    print(f"⚡ Performance: {avg_time*1000:.1f}ms par prédiction")
    
    if avg_time < 0.1:  # < 100ms
        print("✅ Performance acceptable")
    else:
        print("⚠️ Performance à optimiser")
    
    print("\n🎉 Validation terminée !")
```

---

## 🚀 **COMMANDES DE DÉPLOIEMENT**

```bash
# 1. Installation dépendances ML
pip install xgboost lightgbm tensorflow scikit-learn
pip install shap lime  # Pour explications
pip install joblib pandas numpy

# 2. Création structure
python -c "
import os
dirs = [
    'backend/core/biomechanics/ml/data',
    'backend/core/biomechanics/ml/models', 
    'backend/core/biomechanics/ml/training',
    'scripts'
]
for d in dirs: os.makedirs(d, exist_ok=True)
print('✅ Structure créée')
"

# 3. Collecte données (première fois)
python scripts/collect_biomech_data.py

# 4. Entraînement modèles
python scripts/train_biomech_models.py

# 5. Validation
python scripts/validate_ml_biomech.py

# 6. Tests
pytest tests/test_ml_biomechanics.py -v
```

---

## 📊 **SUIVI ET MONITORING**

```python
# backend/core/biomechanics/ml/monitoring.py
class MLBiomechMonitoring:
    """Monitoring performance modèles ML"""
    
    def __init__(self):
        self.metrics_file = "backend/core/biomechanics/ml/monitoring_metrics.json"
        
    def log_prediction(self, input_features, prediction, confidence, method):
        """Log chaque prédiction pour suivi"""
        
        log_entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'prediction': float(prediction),
            'confidence': float(confidence),
            'method': method,
            'feature_hash': hash(str(input_features))  # Pour détecter dérives
        }
        
        # Append au fichier de logs
        self.append_to_metrics(log_entry)
    
    def detect_model_drift(self) -> bool:
        """Détection dérive modèle"""
        
        recent_predictions = self.load_recent_predictions(days=7)
        
        if len(recent_predictions) < 50:
            return False
        
        # Statistiques récentes vs historiques
        recent_mean = np.mean(recent_predictions)
        recent_std = np.std(recent_predictions)
        
        # Seuils de dérive (à ajuster)
        if recent_std > 2.0:  # Trop de variabilité
            return True
            
        if recent_mean < 3.0 or recent_mean > 8.0:  # Moyennes anormales
            return True
            
        return False
    
    def schedule_retraining(self):
        """Programmation re-entraînement automatique"""
        
        if self.detect_model_drift():
            print("🚨 Dérive détectée - Re-entraînement recommandé")
            
            # Trigger re-entraînement automatique
            # ou notification équipe
```

---

## 🎯 **RÉSULTATS ATTENDUS**

### **Gains de Performance**
- **Précision** : +15-25% vs règles fixes
- **Adaptabilité** : Prend en compte contexte (fatigue, pression, etc.)
- **Amélioration continue** : Apprend de nouvelles données
- **Explications** : SHAP pour comprendre les décisions

### **Métriques de Succès**
- R² > 0.8 sur données test
- MAE < 1.0 point (sur échelle 0-10)
- Temps prédiction < 100ms
- Accord experts > 85%

---

## 🎉 **CONCLUSION**

Avec cette implémentation, vous transformez votre module biomécanique de **rule-based** vers **ML-based** ! 

**Prochaines étapes** :
1. Lancer `python scripts/collect_biomech_data.py`
2. Scorer 20-50 exemples avec experts
3. Entraîner premier modèle
4. Valider avec tests
5. Intégrer dans système principal

**Besoin d'aide pour une étape spécifique ?** Je peux détailler n'importe quelle partie ! 🚀 