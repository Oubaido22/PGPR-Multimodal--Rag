# enhanced_multimodal_rag.py - Système RAG multimodal enrichi avec ML

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dotenv import load_dotenv

from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from image_processor import PGPRImageProcessor, create_image_document
from dataset_processor import CSVDatasetProcessor
from ml_model_builder import PGPRMLModelBuilder, create_ml_models

class EnhancedMultimodalRAGSystem:
    """Système RAG multimodal enrichi avec modèles ML pour classification PGPR"""
    
    def __init__(self, persist_dir: str = "./rag_cache/"):
        self.persist_dir = persist_dir
        self.vector_store_path = os.path.join(persist_dir, "enhanced_multimodal_vector_store")
        self.chunks_path = os.path.join(persist_dir, "enhanced_multimodal_chunks.pkl")
        self.image_features_path = os.path.join(persist_dir, "enhanced_image_features.pkl")
        
        # Créer les répertoires nécessaires
        Path(persist_dir).mkdir(exist_ok=True)
        Path(self.vector_store_path).mkdir(exist_ok=True)
        
        # Initialiser les processeurs
        self.image_processor = PGPRImageProcessor()
        self.csv_processor = None
        self.ml_builder = None
        self.embeddings = None
        self.llm = None
        self.vector_store = None
        self.rag_chain = None
        
        # Charger la configuration
        load_dotenv()
    
    def initialize_models(self):
        """Initialise les modèles de langage et d'embeddings"""
        print("Initialisation des modèles Ollama...")
        
        self.embeddings = OllamaEmbeddings(model="llama3.1")
        
        self.llm = ChatOllama(
            model="llama3.1",
            temperature=0,
            model_kwargs={
                "num_ctx": 2048,
                "num_thread": 4,
                "num_gpu": 1,
                "repeat_penalty": 1.1
            }
        )
        
        print("Modèles initialisés avec succès")
    
    def setup_csv_dataset(self, images_dir: str, train_csv: str, test_csv: str):
        """Configure le dataset CSV"""
        print("Configuration du dataset CSV...")
        
        self.csv_processor = CSVDatasetProcessor(images_dir, train_csv, test_csv)
        self.image_processor.csv_processor = self.csv_processor
        
        print("Dataset CSV configuré")
    
    def load_or_create_ml_models(self, features_dict: Dict[str, np.ndarray] = None):
        """Charge ou crée les modèles ML"""
        print("Configuration des modèles ML...")
        
        self.ml_builder = PGPRMLModelBuilder()
        
        # Essayer de charger les modèles existants
        try:
            self.ml_builder.load_models()
            print("Modèles ML chargés depuis le cache")
        except:
            print("Modèles ML non trouvés, création de nouveaux modèles...")
            
            if features_dict is None:
                print("Erreur: features_dict requis pour créer les modèles ML")
                return
            
            # Créer et entraîner les modèles
            self.ml_builder = create_ml_models(features_dict, self.csv_processor)
            print("Modèles ML créés et entraînés")
    
    def process_text_documents(self, docs_path: str = "./pgpr_docs/") -> List[Dict]:
        """Traite les documents textuels"""
        print("Traitement des documents textuels...")
        
        # Charger les documents PDF
        loader = PyPDFDirectoryLoader(docs_path)
        documents = loader.load()
        
        # Diviser en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        text_chunks = text_splitter.split_documents(documents)
        
        # Convertir en format multimodal
        multimodal_chunks = []
        for chunk in text_chunks:
            multimodal_chunk = {
                "type": "text",
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "features": None  # Les features textuelles seront générées par l'embedding
            }
            multimodal_chunks.append(multimodal_chunk)
        
        print(f"Documents textuels traités: {len(multimodal_chunks)} chunks")
        return multimodal_chunks
    
    def process_csv_image_dataset(self, images_dir: str, train_csv: str, test_csv: str) -> List[Dict]:
        """Traite le dataset d'images CSV avec enrichissement ML"""
        print("Traitement du dataset d'images CSV avec enrichissement ML...")
        
        # Traiter les images avec le processeur CSV
        features_dict = self.image_processor.process_csv_dataset(images_dir, train_csv, test_csv)
        
        # Charger ou créer les modèles ML
        self.load_or_create_ml_models(features_dict)
        
        multimodal_chunks = []
        
        # Traiter chaque image avec enrichissement ML
        for image_path, features in features_dict.items():
            # Analyse de base de l'image
            analysis = self.image_processor.analyze_image(image_path)
            
            # Prédictions ML
            ml_predictions = {}
            ml_probabilities = {}
            
            if self.ml_builder and self.ml_builder.models:
                try:
                    # Utiliser le meilleur modèle (neural_network par défaut)
                    best_model = "neural_network"
                    if best_model in self.ml_builder.models:
                        predictions, probabilities = self.ml_builder.predict(
                            features.reshape(1, -1), best_model
                        )
                        
                        # Convertir en format lisible
                        for i, bacteria in enumerate(self.ml_builder.bacteria_types):
                            ml_predictions[bacteria] = int(predictions[0][i])
                            ml_probabilities[bacteria] = float(probabilities[0][i])
                        
                        # Ajouter les prédictions à l'analyse
                        analysis["ml_predictions"] = ml_predictions
                        analysis["ml_probabilities"] = ml_probabilities
                        
                        # Créer un contenu enrichi
                        predicted_bacteria = [bacteria for bacteria, pred in ml_predictions.items() if pred == 1]
                        confidence_scores = [ml_probabilities[bacteria] for bacteria in predicted_bacteria]
                        
                        if predicted_bacteria:
                            ml_content = f"Prédictions ML: {', '.join(predicted_bacteria)} "
                            ml_content += f"(confiance: {', '.join([f'{conf:.2f}' for conf in confidence_scores])})"
                        else:
                            ml_content = "Aucune bactérie détectée par le modèle ML"
                        
                        analysis["ml_content"] = ml_content
                        
                except Exception as e:
                    print(f"Erreur lors des prédictions ML pour {image_path}: {e}")
                    analysis["ml_content"] = "Erreur de prédiction ML"
            
            # Créer le document multimodal enrichi
            image_doc = create_image_document(
                image_path=image_path,
                features=features,
                analysis=analysis,
                metadata={
                    "split": "train" if "train" in str(image_path) else "test",
                    "dataset_type": "csv",
                    "file_name": os.path.basename(image_path),
                    "ml_enriched": True
                }
            )
            
            # Enrichir le contenu avec les prédictions ML
            if "ml_content" in analysis:
                image_doc["content"] += f" | {analysis['ml_content']}"
            
            multimodal_chunks.append(image_doc)
        
        print(f"Images traitées avec enrichissement ML: {len(multimodal_chunks)} images")
        return multimodal_chunks
    
    def create_multimodal_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Crée des embeddings pour les chunks multimodaux enrichis"""
        print("Création des embeddings multimodaux enrichis...")
        
        # Traiter les textes en batch pour plus de rapidité
        text_chunks = [chunk for chunk in chunks if chunk["type"] == "text"]
        if text_chunks:
            print(f"Traitement de {len(text_chunks)} chunks textuels...")
            texts = [chunk["content"] for chunk in text_chunks]
            text_embeddings = self.embeddings.embed_documents(texts)
            
            for i, chunk in enumerate(text_chunks):
                chunk["features"] = text_embeddings[i]
        
        # Traiter les images individuellement (plus complexe avec enrichissement ML)
        image_chunks = [chunk for chunk in chunks if chunk["type"] == "image"]
        if image_chunks:
            print(f"Traitement de {len(image_chunks)} chunks d'images...")
            for i, chunk in enumerate(image_chunks):
                if i % 10 == 0:  # Afficher le progrès
                    print(f"  Progression: {i}/{len(image_chunks)} images traitées")
                
                # Enrichir avec les informations ML
                ml_info = ""
                if "ml_predictions" in chunk["analysis"]:
                    predictions = chunk["analysis"]["ml_predictions"]
                    predicted = [bacteria for bacteria, pred in predictions.items() if pred == 1]
                    if predicted:
                        ml_info = f" | Prédictions ML: {', '.join(predicted)}"
                
                metadata_text = f"Image PGPR: {chunk['metadata'].get('file_name', '')} - {chunk['analysis'].get('bacteria_count', 0)} bactéries{ml_info}"
                text_embedding = self.embeddings.embed_query(metadata_text)
                
                # Combiner features d'image et embedding textuel enrichi
                chunk["features"] = chunk["features"]  # Features d'image déjà présentes
                chunk["text_embedding"] = text_embedding  # Embedding textuel enrichi
        
        print("Embeddings multimodaux créés avec succès!")
        return chunks
    
    def build_vector_store(self, chunks: List[Dict]):
        """Construit la base de données vectorielle enrichie"""
        print("Construction de la base de données vectorielle enrichie...")
        
        # Préparer les données pour FAISS
        texts = []
        metadatas = []
        
        for chunk in chunks:
            if chunk["type"] == "text":
                texts.append(chunk["content"])
                metadatas.append({
                    "type": "text",
                    "source": chunk["metadata"].get("source", "unknown"),
                    "page": chunk["metadata"].get("page", 0)
                })
            elif chunk["type"] == "image":
                # Contenu enrichi avec prédictions ML
                content = chunk["content"]
                if "ml_predictions" in chunk["analysis"]:
                    predictions = chunk["analysis"]["ml_predictions"]
                    predicted = [bacteria for bacteria, pred in predictions.items() if pred == 1]
                    if predicted:
                        content += f" | Prédictions ML: {', '.join(predicted)}"
                
                texts.append(content)
                metadatas.append({
                    "type": "image",
                    "path": chunk["path"],
                    "bacteria_count": chunk["analysis"].get("bacteria_count", 0),
                    "split": chunk["metadata"].get("split", "unknown"),
                    "ml_enriched": chunk["metadata"].get("ml_enriched", False),
                    "ml_predictions": chunk["analysis"].get("ml_predictions", {})
                })
        
        # Créer la base vectorielle
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        # Sauvegarder
        self.vector_store.save_local(self.vector_store_path)
        print("Base de données vectorielle enrichie sauvegardée")
    
    def build_rag_chain(self):
        """Construit la chaîne RAG enrichie"""
        print("Construction de la chaîne RAG enrichie...")
        
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5, "fetch_k": 10}
        )
        
        template = """
        Vous êtes un expert en microbiologie spécialisé dans les PGPR (Plant Growth-Promoting Rhizobacteria).
        Vous avez accès à des informations textuelles et des analyses d'images de bactéries PGPR enrichies par des modèles ML.
        
        Les prédictions ML fournissent des informations supplémentaires sur les types de bactéries détectées dans les images.
        
        Répondez de manière claire et précise en vous basant sur le contexte fourni.
        Si des images avec prédictions ML sont mentionnées dans le contexte, utilisez ces informations pour enrichir votre réponse.
        
        Contexte: {context}
        Question: {question}
        
        Réponse détaillée et structurée:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("Chaîne RAG enrichie construite")
    
    def build_system(self, images_dir: str, train_csv: str, test_csv: str):
        """Construit le système RAG multimodal enrichi complet"""
        print("=== CONSTRUCTION DU SYSTÈME RAG MULTIMODAL ENRICHIE ===\n")
        
        # Initialiser les modèles
        self.initialize_models()
        
        # Configurer le dataset CSV
        self.setup_csv_dataset(images_dir, train_csv, test_csv)
        
        # Traiter les documents textuels
        text_chunks = self.process_text_documents()
        
        # Traiter les images avec enrichissement ML
        image_chunks = self.process_csv_image_dataset(images_dir, train_csv, test_csv)
        
        # Combiner tous les chunks
        all_chunks = text_chunks + image_chunks
        
        # Créer les embeddings enrichis
        all_chunks = self.create_multimodal_embeddings(all_chunks)
        
        # Construire la base vectorielle enrichie
        self.build_vector_store(all_chunks)
        
        # Construire la chaîne RAG enrichie
        self.build_rag_chain()
        
        # Sauvegarder les chunks
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(all_chunks, f)
        
        print(f"\nSystème RAG multimodal enrichi construit avec succès!")
        print(f"- Documents textuels: {len(text_chunks)}")
        print(f"- Images enrichies ML: {len(image_chunks)}")
        print(f"- Total: {len(all_chunks)} chunks")
        
        # Statistiques des prédictions ML
        ml_enriched_count = sum(1 for chunk in image_chunks if chunk["metadata"].get("ml_enriched", False))
        print(f"- Images avec prédictions ML: {ml_enriched_count}")
    
    def load_system(self):
        """Charge le système RAG enrichi depuis le cache"""
        if not os.path.exists(self.vector_store_path):
            raise FileNotFoundError("Système RAG enrichi non trouvé. Exécutez build_system() d'abord.")
        
        # Initialiser les modèles
        self.initialize_models()
        
        # Charger la base vectorielle
        self.vector_store = FAISS.load_local(
            folder_path=self.vector_store_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Charger les modèles ML
        try:
            self.ml_builder = PGPRMLModelBuilder()
            self.ml_builder.load_models()
            print("Modèles ML chargés")
        except:
            print("Modèles ML non trouvés")
        
        # Construire la chaîne RAG
        self.build_rag_chain()
        
        print("Système RAG multimodal enrichi chargé depuis le cache")
    
    def query(self, question: str) -> str:
        """Interroge le système RAG enrichi"""
        if self.rag_chain is None:
            raise ValueError("Système RAG non initialisé")
        
        return self.rag_chain.invoke(question)
    
    def predict_image_bacteria(self, image_path: str, model_name: str = "neural_network") -> Dict:
        """Prédit les bactéries dans une image avec les modèles ML"""
        if self.ml_builder is None:
            return {"error": "Modèles ML non chargés"}
        
        try:
            # Extraire les features de l'image
            features = self.image_processor.extract_features(image_path)
            
            # Faire la prédiction
            predictions, probabilities = self.ml_builder.predict(
                features.reshape(1, -1), model_name
            )
            
            # Formater les résultats
            result = {
                "image_path": image_path,
                "model_used": model_name,
                "predictions": {},
                "probabilities": {},
                "detected_bacteria": []
            }
            
            for i, bacteria in enumerate(self.ml_builder.bacteria_types):
                pred = int(predictions[0][i])
                prob = float(probabilities[0][i])
                
                result["predictions"][bacteria] = pred
                result["probabilities"][bacteria] = prob
                
                if pred == 1:
                    result["detected_bacteria"].append({
                        "bacteria": bacteria,
                        "confidence": prob
                    })
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def find_similar_images_with_ml(self, query_image_path: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Trouve des images similaires avec enrichissement ML"""
        # Charger les features d'images depuis le cache
        if os.path.exists(self.image_features_path):
            with open(self.image_features_path, 'rb') as f:
                image_features = pickle.load(f)
            
            # Recherche de similarité
            similar_images = self.image_processor.find_similar_images(
                query_image_path, image_features, top_k
            )
            
            # Enrichir avec les prédictions ML
            enriched_results = []
            for image_path, similarity in similar_images:
                ml_info = {}
                if self.ml_builder:
                    try:
                        features = image_features[image_path]
                        predictions, probabilities = self.ml_builder.predict(
                            features.reshape(1, -1), "neural_network"
                        )
                        
                        detected = []
                        for i, bacteria in enumerate(self.ml_builder.bacteria_types):
                            if predictions[0][i] == 1:
                                detected.append({
                                    "bacteria": bacteria,
                                    "confidence": float(probabilities[0][i])
                                })
                        
                        ml_info = {
                            "detected_bacteria": detected,
                            "model_used": "neural_network"
                        }
                    except:
                        ml_info = {"error": "Prédiction ML échouée"}
                
                enriched_results.append((image_path, similarity, ml_info))
            
            return enriched_results
        else:
            print("Aucune base d'images trouvée")
            return []
    
    def get_system_stats(self) -> Dict:
        """Retourne les statistiques du système enrichi"""
        if self.vector_store is None:
            return {"error": "Système non chargé"}
        
        # Compter les différents types de documents
        stats = {
            "total_documents": len(self.vector_store.docstore._dict),
            "text_documents": 0,
            "image_documents": 0,
            "ml_enriched_images": 0
        }
        
        for doc_id, doc in self.vector_store.docstore._dict.items():
            if doc.metadata.get("type") == "text":
                stats["text_documents"] += 1
            elif doc.metadata.get("type") == "image":
                stats["image_documents"] += 1
                if doc.metadata.get("ml_enriched", False):
                    stats["ml_enriched_images"] += 1
        
        # Ajouter les informations sur les modèles ML
        if self.ml_builder and self.ml_builder.models:
            stats["ml_models_available"] = list(self.ml_builder.models.keys())
        else:
            stats["ml_models_available"] = []
        
        return stats

def build_enhanced_multimodal_rag(images_dir: str, train_csv: str, test_csv: str):
    """Fonction utilitaire pour construire le système RAG multimodal enrichi"""
    system = EnhancedMultimodalRAGSystem()
    system.build_system(images_dir, train_csv, test_csv)
    return system

def load_enhanced_multimodal_rag():
    """Fonction utilitaire pour charger le système RAG multimodal enrichi"""
    system = EnhancedMultimodalRAGSystem()
    system.load_system()
    return system

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration pour votre dataset
    IMAGES_DIR = "./pgpr_images/images/"
    TRAIN_CSV = "./pgpr_images/train_labels.csv"
    TEST_CSV = "./pgpr_images/test_labels.csv"
    
    # Construire le système enrichi
    system = build_enhanced_multimodal_rag(IMAGES_DIR, TRAIN_CSV, TEST_CSV)
    
    # Test du système
    question = "Quels types de bactéries sont présents dans les images et comment les identifier avec les modèles ML?"
    response = system.query(question)
    print(f"Question: {question}")
    print(f"Réponse: {response}")
    
    # Statistiques
    stats = system.get_system_stats()
    print(f"\nStatistiques: {stats}")
