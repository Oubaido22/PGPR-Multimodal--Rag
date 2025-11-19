# ml_model_builder.py - Constructeur de modèles ML pour classification PGPR

import os
import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class PGPRDataset(Dataset):
    """Dataset PyTorch pour les images PGPR"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class PGPRNeuralNetwork(nn.Module):
    """Réseau de neurones pour classification multi-label PGPR"""
    
    def __init__(self, input_dim: int = 2048, hidden_dims: List[int] = [512, 256, 128], 
                 num_classes: int = 4, dropout_rate: float = 0.3):
        super(PGPRNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class PGPRMLModelBuilder:
    """Constructeur de modèles ML pour classification PGPR"""
    
    def __init__(self, models_dir: str = "./ml_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.feature_extractors = {}
        self.bacteria_types = [
            "Bacillus subtilis",
            "Escherichia coli", 
            "Pseudomonas aeruginosa",
            "Staphylococcus aureus"
        ]
        
        # Configuration des modèles
        self.model_configs = {
            "random_forest": {
                "model": RandomForestClassifier,
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42,
                    "n_jobs": -1
                }
            },
            "gradient_boosting": {
                "model": GradientBoostingClassifier,
                "params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42
                }
            },
            "svm": {
                "model": SVC,
                "params": {
                    "kernel": "rbf",
                    "C": 1.0,
                    "probability": True,
                    "random_state": 42
                }
            },
            "mlp": {
                "model": MLPClassifier,
                "params": {
                    "hidden_layer_sizes": (512, 256, 128),
                    "activation": "relu",
                    "solver": "adam",
                    "alpha": 0.001,
                    "max_iter": 500,
                    "random_state": 42
                }
            },
            "neural_network": {
                "model": "custom",  # Notre réseau personnalisé
                "params": {
                    "input_dim": 2048,
                    "hidden_dims": [512, 256, 128],
                    "num_classes": 4,
                    "dropout_rate": 0.3,
                    "learning_rate": 0.001,
                    "epochs": 100,
                    "batch_size": 32
                }
            }
        }
    
    def prepare_data(self, features_dict: Dict[str, np.ndarray], 
                    csv_processor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prépare les données pour l'entraînement"""
        print("Préparation des données pour l'entraînement...")
        
        features_list = []
        labels_list = []
        
        # Traiter les données d'entraînement
        for image_path, features in features_dict.items():
            filename = os.path.basename(image_path)
            
            # Chercher les labels dans train et test
            train_labels = csv_processor.get_bacteria_labels(filename, "train")
            test_labels = csv_processor.get_bacteria_labels(filename, "test")
            
            labels = train_labels if train_labels else test_labels
            
            if labels:
                features_list.append(features)
                label_vector = [labels[bacteria] for bacteria in self.bacteria_types]
                labels_list.append(label_vector)
        
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        
        print(f"Données préparées: {features_array.shape[0]} échantillons, {features_array.shape[1]} features")
        print(f"Labels: {labels_array.shape[1]} classes")
        
        # Diviser en train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            features_array, labels_array, test_size=0.2, random_state=42, stratify=None
        )
        
        return X_train, X_val, y_train, y_val
    
    def train_sklearn_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray):
        """Entraîne les modèles scikit-learn"""
        print("Entraînement des modèles scikit-learn...")
        
        for model_name, config in self.model_configs.items():
            if model_name == "neural_network":
                continue  # Traité séparément
                
            print(f"\nEntraînement de {model_name}...")
            
            # Standardisation des features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Créer et entraîner le modèle
            model_class = config["model"]
            model = model_class(**config["params"])
            
            # Pour les modèles multi-label, entraîner un modèle par classe
            if model_name in ["random_forest", "gradient_boosting"]:
                models = []
                for i in range(y_train.shape[1]):
                    model_i = model_class(**config["params"])
                    model_i.fit(X_train_scaled, y_train[:, i])
                    models.append(model_i)
                
                self.models[model_name] = models
                self.scalers[model_name] = scaler
                
                # Évaluation
                predictions = []
                for i, model_i in enumerate(models):
                    pred = model_i.predict(X_val_scaled)
                    predictions.append(pred)
                
                predictions = np.array(predictions).T
                accuracy = accuracy_score(y_val, predictions)
                f1 = f1_score(y_val, predictions, average='weighted')
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                
            else:
                # Modèles qui supportent directement le multi-label
                if model_name == "svm":
                    # SVM ne supporte pas nativement la classification multi-label
                    # Nous utilisons OneVsRestClassifier pour entraîner un SVM par classe
                    from sklearn.multiclass import OneVsRestClassifier
                    from sklearn.svm import SVC
                    
                    base_svm = SVC(probability=True, random_state=42)
                    model = OneVsRestClassifier(base_svm)
                    model.fit(X_train_scaled, y_train)
                    
                else:
                    # MLPClassifier supporte nativement la classification multi-label
                    model.fit(X_train_scaled, y_train)
                
                self.models[model_name] = model
                self.scalers[model_name] = scaler
                
                # Évaluation
                predictions = model.predict(X_val_scaled)
                accuracy = accuracy_score(y_val, predictions)
                f1 = f1_score(y_val, predictions, average='weighted')
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray):
        """Entraîne le réseau de neurones personnalisé"""
        print("\nEntraînement du réseau de neurones...")
        
        config = self.model_configs["neural_network"]["params"]
        
        # Standardisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Créer les datasets
        train_dataset = PGPRDataset(X_train_scaled, y_train)
        val_dataset = PGPRDataset(X_val_scaled, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        
        # Créer le modèle
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PGPRNeuralNetwork(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            num_classes=config["num_classes"],
            dropout_rate=config["dropout_rate"]
        ).to(device)
        
        # Optimiseur et fonction de perte
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = nn.BCELoss()
        
        # Entraînement
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(config["epochs"]):
            # Entraînement
            model.train()
            train_loss = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_predictions = []
            val_true = []
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    predictions = (outputs > 0.5).float()
                    val_predictions.extend(predictions.cpu().numpy())
                    val_true.extend(batch_labels.cpu().numpy())
            
            val_predictions = np.array(val_predictions)
            val_true = np.array(val_true)
            
            # Métriques
            accuracy = accuracy_score(val_true, val_predictions)
            f1 = f1_score(val_true, val_predictions, average='weighted')
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Val Loss: {val_loss/len(val_loader):.4f}, "
                      f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                torch.save(model.state_dict(), self.models_dir / "best_neural_network.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping à l'epoch {epoch}")
                    break
        
        # Charger le meilleur modèle
        model.load_state_dict(torch.load(self.models_dir / "best_neural_network.pth"))
        self.models["neural_network"] = model
        self.scalers["neural_network"] = scaler
        
        print(f"  Meilleur modèle sauvegardé avec validation loss: {best_val_loss:.4f}")
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """Évalue tous les modèles entraînés"""
        print("\n=== ÉVALUATION DES MODÈLES ===")
        
        results = {}
        
        for model_name in self.models.keys():
            print(f"\nÉvaluation de {model_name}...")
            
            # Standardisation
            X_test_scaled = self.scalers[model_name].transform(X_test)
            
            if model_name == "neural_network":
                # Prédiction avec le réseau de neurones
                model = self.models[model_name]
                model.eval()
                
                test_dataset = PGPRDataset(X_test_scaled, y_test)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                predictions = []
                
                with torch.no_grad():
                    for batch_features, _ in test_loader:
                        batch_features = batch_features.to(device)
                        outputs = model(batch_features)
                        pred = (outputs > 0.5).float()
                        predictions.extend(pred.cpu().numpy())
                
                predictions = np.array(predictions)
                
            elif model_name in ["random_forest", "gradient_boosting"]:
                # Modèles par classe
                models = self.models[model_name]
                predictions = []
                for i, model_i in enumerate(models):
                    pred = model_i.predict(X_test_scaled)
                    predictions.append(pred)
                predictions = np.array(predictions).T
                
            else:
                # Autres modèles scikit-learn
                model = self.models[model_name]
                predictions = model.predict(X_test_scaled)
            
            # Calculer les métriques
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            
            # Rapport de classification détaillé
            report = classification_report(y_test, predictions, 
                                         target_names=self.bacteria_types, 
                                         output_dict=True)
            
            results[model_name] = {
                "accuracy": accuracy,
                "f1_score": f1,
                "classification_report": report,
                "predictions": predictions
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
            # Afficher les métriques par classe
            for i, bacteria in enumerate(self.bacteria_types):
                precision = report[bacteria]['precision']
                recall = report[bacteria]['recall']
                f1_class = report[bacteria]['f1-score']
                print(f"    {bacteria}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_class:.3f}")
        
        return results
    
    def save_models(self):
        """Sauvegarde tous les modèles entraînés"""
        print("\nSauvegarde des modèles...")
        
        for model_name, model in self.models.items():
            if model_name == "neural_network":
                # Le modèle PyTorch est déjà sauvegardé
                continue
            elif model_name in ["random_forest", "gradient_boosting"]:
                # Sauvegarder les modèles par classe
                for i, model_i in enumerate(model):
                    joblib.dump(model_i, self.models_dir / f"{model_name}_class_{i}.pkl")
            else:
                joblib.dump(model, self.models_dir / f"{model_name}.pkl")
        
        # Sauvegarder les scalers
        for model_name, scaler in self.scalers.items():
            joblib.dump(scaler, self.models_dir / f"{model_name}_scaler.pkl")
        
        # Sauvegarder la configuration
        config_data = {
            "bacteria_types": self.bacteria_types,
            "model_configs": self.model_configs
        }
        with open(self.models_dir / "config.pkl", "wb") as f:
            pickle.dump(config_data, f)
        
        print(f"Modèles sauvegardés dans {self.models_dir}")
    
    def load_models(self):
        """Charge les modèles sauvegardés"""
        print("Chargement des modèles...")
        
        # Charger la configuration
        with open(self.models_dir / "config.pkl", "rb") as f:
            config_data = pickle.load(f)
        
        self.bacteria_types = config_data["bacteria_types"]
        
        # Charger les modèles
        for model_name in self.model_configs.keys():
            if model_name == "neural_network":
                # Charger le modèle PyTorch
                config = self.model_configs[model_name]["params"]
                model = PGPRNeuralNetwork(
                    input_dim=config["input_dim"],
                    hidden_dims=config["hidden_dims"],
                    num_classes=config["num_classes"],
                    dropout_rate=config["dropout_rate"]
                )
                model.load_state_dict(torch.load(self.models_dir / "best_neural_network.pth"))
                self.models[model_name] = model
                
            elif model_name in ["random_forest", "gradient_boosting"]:
                # Charger les modèles par classe
                models = []
                for i in range(len(self.bacteria_types)):
                    model_path = self.models_dir / f"{model_name}_class_{i}.pkl"
                    if model_path.exists():
                        model_i = joblib.load(model_path)
                        models.append(model_i)
                if models:
                    self.models[model_name] = models
                    
            else:
                # Charger les autres modèles
                model_path = self.models_dir / f"{model_name}.pkl"
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
            
            # Charger les scalers
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            if scaler_path.exists():
                self.scalers[model_name] = joblib.load(scaler_path)
        
        print(f"Modèles chargés: {list(self.models.keys())}")
    
    def predict(self, features: np.ndarray, model_name: str = "neural_network") -> Tuple[np.ndarray, np.ndarray]:
        """Fait des prédictions avec un modèle spécifique"""
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non trouvé")
        
        # Standardisation
        features_scaled = self.scalers[model_name].transform(features)
        
        if model_name == "neural_network":
            # Prédiction avec le réseau de neurones
            model = self.models[model_name]
            model.eval()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            features_tensor = torch.FloatTensor(features_scaled).to(device)
            
            with torch.no_grad():
                outputs = model(features_tensor)
                probabilities = outputs.cpu().numpy()
                predictions = (outputs > 0.5).float().cpu().numpy()
            
            return predictions, probabilities
            
        elif model_name in ["random_forest", "gradient_boosting"]:
            # Prédiction avec les modèles par classe
            models = self.models[model_name]
            predictions = []
            probabilities = []
            
            for i, model_i in enumerate(models):
                pred = model_i.predict(features_scaled)
                prob = model_i.predict_proba(features_scaled)
                predictions.append(pred)
                probabilities.append(prob[:, 1])  # Probabilité de la classe positive
            
            predictions = np.array(predictions).T
            probabilities = np.array(probabilities).T
            
            return predictions, probabilities
            
        else:
            # Prédiction avec les autres modèles
            model = self.models[model_name]
            predictions = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)
            
            return predictions, probabilities
    
    def get_best_model(self, results: Dict) -> str:
        """Retourne le meilleur modèle basé sur les résultats"""
        best_model = None
        best_f1 = 0
        
        for model_name, result in results.items():
            if result["f1_score"] > best_f1:
                best_f1 = result["f1_score"]
                best_model = model_name
        
        return best_model

def create_ml_models(features_dict: Dict[str, np.ndarray], csv_processor, 
                    test_size: float = 0.2) -> PGPRMLModelBuilder:
    """Fonction utilitaire pour créer et entraîner tous les modèles ML"""
    print("=== CRÉATION DES MODÈLES ML PGPR ===\n")
    
    # Créer le constructeur de modèles
    builder = PGPRMLModelBuilder()
    
    # Préparer les données
    X_train, X_val, y_train, y_val = builder.prepare_data(features_dict, csv_processor)
    
    # Entraîner les modèles scikit-learn
    builder.train_sklearn_models(X_train, y_train, X_val, y_val)
    
    # Entraîner le réseau de neurones
    builder.train_neural_network(X_train, y_train, X_val, y_val)
    
    # Évaluer tous les modèles
    results = builder.evaluate_models(X_val, y_val)
    
    # Sauvegarder les modèles
    builder.save_models()
    
    # Afficher le meilleur modèle
    best_model = builder.get_best_model(results)
    print(f"\n🎉 Meilleur modèle: {best_model} (F1-Score: {results[best_model]['f1_score']:.4f})")
    
    return builder

if __name__ == "__main__":
    print("Constructeur de modèles ML PGPR")
    print("Utilisez create_ml_models() pour entraîner tous les modèles")
