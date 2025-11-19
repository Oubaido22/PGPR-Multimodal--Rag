# image_processor.py - Module de traitement d'images PGPR

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import streamlit as st

class PGPRImageProcessor:
    """Processeur d'images spécialisé pour les bactéries PGPR"""
    
    def __init__(self, model_name: str = "resnet50", feature_dim: int = 2048):
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Charger le modèle pré-entraîné
        self.model = self._load_pretrained_model()
        self.transform = self._get_transforms()
        
        # Cache pour les features
        self.features_cache = {}
        self.cache_path = "./rag_cache/image_features.pkl"
        self._load_features_cache()
    
    def _load_pretrained_model(self) -> nn.Module:
        """Charge un modèle pré-entraîné pour l'extraction de features"""
        if self.model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            # Retirer la dernière couche pour obtenir les features
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == "efficientnet":
            model = models.efficientnet_b0(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Modèle non supporté: {self.model_name}")
        
        model.eval()
        model.to(self.device)
        return model
    
    def _get_transforms(self):
        """Retourne les transformations pour les images"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_features_cache(self):
        """Charge le cache des features depuis le disque"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    self.features_cache = pickle.load(f)
                print(f"Cache d'images chargé: {len(self.features_cache)} images")
            except Exception as e:
                print(f"Erreur lors du chargement du cache: {e}")
                self.features_cache = {}
    
    def _save_features_cache(self):
        """Sauvegarde le cache des features sur le disque"""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.features_cache, f)
            print(f"Cache d'images sauvegardé: {len(self.features_cache)} images")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du cache: {e}")
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extrait les features d'une image PGPR"""
        if image_path in self.features_cache:
            return self.features_cache[image_path]
        
        try:
            # Charger et prétraiter l'image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extraire les features
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().cpu().numpy()
            
            # Normaliser les features
            features = features / np.linalg.norm(features)
            
            # Mettre en cache
            self.features_cache[image_path] = features
            return features
            
        except Exception as e:
            print(f"Erreur lors de l'extraction des features de {image_path}: {e}")
            # Retourner des features vides en cas d'erreur
            return np.zeros(self.feature_dim)
    
    def process_csv_dataset(self, images_dir: str, train_csv: str, test_csv: str) -> Dict[str, np.ndarray]:
        """Traite le dataset d'images CSV et retourne un dictionnaire des features"""
        print("Traitement du dataset CSV...")
        
        features_dict = {}
        
        # Traiter les images d'entraînement
        if os.path.exists(train_csv):
            import pandas as pd
            train_df = pd.read_csv(train_csv)
            
            print(f"Traitement de {len(train_df)} images d'entraînement...")
            for i, row in train_df.iterrows():
                filename = row['filename']
                image_path = os.path.join(images_dir, filename)
                
                if os.path.exists(image_path):
                    try:
                        features = self.extract_features(image_path)
                        features_dict[image_path] = features
                        
                        if (i + 1) % 10 == 0:
                            print(f"  Progression: {i + 1}/{len(train_df)} images traitées")
                            
                    except Exception as e:
                        print(f"Erreur lors du traitement de {filename}: {e}")
                else:
                    print(f"Image manquante: {image_path}")
        
        # Traiter les images de test
        if os.path.exists(test_csv):
            test_df = pd.read_csv(test_csv)
            
            print(f"Traitement de {len(test_df)} images de test...")
            for i, row in test_df.iterrows():
                filename = row['filename']
                image_path = os.path.join(images_dir, filename)
                
                if os.path.exists(image_path):
                    try:
                        features = self.extract_features(image_path)
                        features_dict[image_path] = features
                        
                        if (i + 1) % 10 == 0:
                            print(f"  Progression: {i + 1}/{len(test_df)} images traitées")
                            
                    except Exception as e:
                        print(f"Erreur lors du traitement de {filename}: {e}")
                else:
                    print(f"Image manquante: {image_path}")
        
        # Sauvegarder le cache
        self._save_features_cache()
        
        print(f"Dataset CSV traité: {len(features_dict)} images avec features extraites")
        return features_dict
    
    def process_dataset(self, dataset_path: str, split: str = "train") -> Dict[str, np.ndarray]:
        """Traite un dataset d'images PGPR"""
        print(f"Traitement du dataset {split} depuis {dataset_path}")
        
        features_dict = {}
        dataset_dir = Path(dataset_path) / split
        
        if not dataset_dir.exists():
            print(f"Répertoire {dataset_dir} non trouvé")
            return features_dict
        
        # Parcourir toutes les images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_dir.rglob(f"*{ext}"))
            image_files.extend(dataset_dir.rglob(f"*{ext.upper()}"))
        
        print(f"Trouvé {len(image_files)} images dans {split}")
        
        # Traiter chaque image
        for i, image_path in enumerate(image_files):
            if i % 100 == 0:
                print(f"Traitement: {i}/{len(image_files)}")
            
            features = self.extract_features(str(image_path))
            features_dict[str(image_path)] = features
        
        # Sauvegarder le cache
        self._save_features_cache()
        
        return features_dict
    
    def analyze_image(self, image_path: str) -> Dict:
        """Analyse une image PGPR et retourne des métadonnées"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Impossible de charger l'image"}
            
            # Informations de base
            height, width, channels = image.shape
            
            # Analyse des couleurs
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mean_color = np.mean(hsv, axis=(0, 1))
            
            # Détection de contours (pour identifier les bactéries)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Compter les contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            analysis = {
                "dimensions": {"width": width, "height": height},
                "color_info": {
                    "mean_hue": float(mean_color[0]),
                    "mean_saturation": float(mean_color[1]),
                    "mean_value": float(mean_color[2])
                },
                "bacteria_count": len(contours),
                "image_quality": {
                    "brightness": float(np.mean(gray)),
                    "contrast": float(np.std(gray))
                }
            }
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def find_similar_images(self, query_image_path: str, dataset_features: Dict[str, np.ndarray], 
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """Trouve les images similaires dans le dataset"""
        query_features = self.extract_features(query_image_path)
        
        similarities = []
        for image_path, features in dataset_features.items():
            similarity = np.dot(query_features, features)
            similarities.append((image_path, similarity))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

class PGPRImageClassifier:
    """Classifieur spécialisé pour les bactéries PGPR"""
    
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.model = None
        self.class_names = [
            "Bacillus_subtilis",
            "Pseudomonas_fluorescens", 
            "Azospirillum_brasilense",
            "Rhizobium_leguminosarum",
            "Enterobacter_cloacae"
        ]
    
    def build_model(self, feature_dim: int = 2048) -> nn.Module:
        """Construit un modèle de classification"""
        model = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1)
        )
        return model
    
    def train(self, train_features: List[np.ndarray], train_labels: List[int],
              val_features: List[np.ndarray] = None, val_labels: List[int] = None,
              epochs: int = 50, learning_rate: float = 0.001):
        """Entraîne le modèle de classification"""
        # Convertir en tenseurs
        X_train = torch.FloatTensor(train_features)
        y_train = torch.LongTensor(train_labels)
        
        # Créer le modèle
        self.model = self.build_model(X_train.shape[1])
        self.model.train()
        
        # Optimiseur et fonction de perte
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Entraînement
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def predict(self, features: np.ndarray) -> Tuple[int, str, float]:
        """Prédit la classe d'une image"""
        if self.model is None:
            raise ValueError("Modèle non entraîné")
        
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            outputs = self.model(features_tensor)
            probabilities = outputs.squeeze()
            
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            class_name = self.class_names[predicted_class]
            
            return predicted_class, class_name, confidence

def create_image_document(image_path: str, features: np.ndarray, 
                         analysis: Dict, metadata: Dict = None) -> Dict:
    """Crée un document structuré pour une image PGPR"""
    return {
        "type": "image",
        "path": image_path,
        "features": features,
        "analysis": analysis,
        "metadata": metadata or {},
        "content": f"Image PGPR: {os.path.basename(image_path)} - {analysis.get('bacteria_count', 0)} bactéries détectées"
    }

# Fonctions utilitaires pour Streamlit
def upload_image_interface():
    """Interface Streamlit pour l'upload d'images"""
    uploaded_file = st.file_uploader(
        "Choisissez une image de bactérie PGPR",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Formats supportés: PNG, JPG, JPEG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        # Sauvegarder temporairement
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return temp_path
    return None

def display_image_analysis(image_path: str, processor: PGPRImageProcessor):
    """Affiche l'analyse d'une image dans Streamlit"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_path, caption="Image PGPR", use_column_width=True)
    
    with col2:
        analysis = processor.analyze_image(image_path)
        
        if "error" not in analysis:
            st.subheader("📊 Analyse de l'image")
            
            # Métriques principales
            st.metric("Nombre de bactéries", analysis["bacteria_count"])
            st.metric("Largeur", f"{analysis['dimensions']['width']}px")
            st.metric("Hauteur", f"{analysis['dimensions']['height']}px")
            st.metric("Luminosité", f"{analysis['image_quality']['brightness']:.1f}")
            
            # Informations de couleur
            st.write("**Informations de couleur:**")
            st.write(f"- Teinte moyenne: {analysis['color_info']['mean_hue']:.1f}")
            st.write(f"- Saturation moyenne: {analysis['color_info']['mean_saturation']:.1f}")
            st.write(f"- Valeur moyenne: {analysis['color_info']['mean_value']:.1f}")
        else:
            st.error(f"Erreur d'analyse: {analysis['error']}")
    
    return analysis
