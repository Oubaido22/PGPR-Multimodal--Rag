
# web_chatbot_enhanced.py - Interface web Streamlit pour le chatbot RAG multimodal enrichi

import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
from PIL import Image
import io

# Ajouter le répertoire courant au path Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuration de la page
st.set_page_config(
    page_title="🧬 Chatbot PGPR Multimodal Enrichi",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .bacteria-badge {
        background-color: #e8f4fd;
        color: #1f77b4;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin: 0.1rem;
        display: inline-block;
    }
    .confidence-bar {
        background-color: #f0f2f6;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #ff7f0e, #1f77b4);
        height: 20px;
        border-radius: 0.25rem;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    """Charge le système RAG multimodal enrichi"""
    try:
        # Import local pour éviter les erreurs de module
        from enhanced_multimodal_rag import load_enhanced_multimodal_rag
        system = load_enhanced_multimodal_rag()
        return system
    except Exception as e:
        st.error(f"Erreur lors du chargement du système: {e}")
        st.info("Veuillez d'abord construire le système avec build_enhanced_system.py")
        return None

@st.cache_resource
def load_image_processor():
    """Charge le processeur d'images"""
    try:
        from image_processor import PGPRImageProcessor
        return PGPRImageProcessor()
    except Exception as e:
        st.error(f"Erreur lors du chargement du processeur d'images: {e}")
        return None

def load_simple_rag_system():
    """Charge le système RAG simple (PDF uniquement) comme dans build_rag_optimized.py"""
    try:
        from langchain_ollama.chat_models import ChatOllama
        from langchain_ollama.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import FAISS
        import pickle
        import os
        
        # Chemins du cache
        PERSIST_DIR = "./rag_cache/"
        VECTOR_STORE_PATH = os.path.join(PERSIST_DIR, "vector_store")
        CHUNKS_PATH = os.path.join(PERSIST_DIR, "chunks.pkl")
        
        # Vérifier si le système RAG simple existe
        if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(CHUNKS_PATH):
            print("Chargement du système RAG simple depuis le cache...")
            
            # Charger les embeddings et LLM avec paramètres optimisés pour la mémoire
            embeddings = OllamaEmbeddings(model="llama3.1")
            llm = ChatOllama(model="llama3.1", temperature=0)
            
            # Charger la base vectorielle
            vector_store = FAISS.load_local(
                folder_path=VECTOR_STORE_PATH,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Créer le retriever
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            
            # Construire la chaîne RAG simple
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema.runnable import RunnablePassthrough
            from langchain.schema.output_parser import StrOutputParser
            
            template = """
            Vous êtes un expert en microbiologie spécialisé dans les PGPR (Plant Growth-Promoting Rhizobacteria).
            Répondez à la question en vous basant uniquement sur le contexte fourni.
            
            Contexte:
            {context}
            
            Question:
            {question}
            
            Réponse détaillée et structurée:
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            return {
                "system": "simple_rag",
                "chain": rag_chain,
                "vector_store": vector_store,
                "status": "loaded"
            }
        else:
            print("Système RAG simple non trouvé. Construction nécessaire...")
            return None
            
    except Exception as e:
        print(f"Erreur lors du chargement du RAG simple: {e}")
        return None

def build_simple_rag_system():
    """Construit le système RAG simple (PDF uniquement)"""
    try:
        from langchain_ollama.chat_models import ChatOllama
        from langchain_ollama.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.prompts import ChatPromptTemplate
        from langchain.schema.runnable import RunnablePassthrough
        from langchain.schema.output_parser import StrOutputParser
        import pickle
        import os
        
        print("Construction du système RAG simple...")
        
        # Chemins du cache
        PERSIST_DIR = "./rag_cache/"
        VECTOR_STORE_PATH = os.path.join(PERSIST_DIR, "vector_store")
        CHUNKS_PATH = os.path.join(PERSIST_DIR, "chunks.pkl")
        
        # Créer les répertoires
        os.makedirs(PERSIST_DIR, exist_ok=True)
        
        # ÉTAPE 1: Chargement des documents PDF
        print("Chargement des documents PDF...")
        loader = PyPDFDirectoryLoader("./pgpr_docs/")
        documents = loader.load()
        
        # ÉTAPE 2: Découpage en chunks
        print("Découpage des documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Sauvegarder les chunks
        with open(CHUNKS_PATH, 'wb') as f:
            pickle.dump(chunks, f)
        print(f"Chunks sauvegardés: {len(chunks)}")
        
        # ÉTAPE 3: Initialisation des modèles Ollama avec paramètres optimisés
        print("Initialisation des modèles Ollama...")
        embeddings = OllamaEmbeddings(model="llama3.1")
        llm = ChatOllama(model="llama3.1", temperature=0)
        
        # ÉTAPE 4: Création de la base vectorielle
        print("Création de la base vectorielle...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
        
        # ÉTAPE 5: Construction de la chaîne RAG
        print("Construction de la chaîne RAG...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        template = """
        Vous êtes un expert en microbiologie spécialisé dans les PGPR (Plant Growth-Promoting Rhizobacteria).
        Répondez à la question en vous basant uniquement sur le contexte fourni.
        
        Contexte:
        {context}
        
        Question:
        {question}
        
        Réponse détaillée et structurée:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("✅ Système RAG simple construit avec succès!")
        
        return {
            "system": "simple_rag",
            "chain": rag_chain,
            "vector_store": vector_store,
            "status": "built"
        }
        
    except Exception as e:
        print(f"Erreur lors de la construction du RAG simple: {e}")
        return None

@st.cache_resource
def load_ml_models_and_metrics():
    """Charge les modèles ML et leurs métriques d'évaluation dynamiquement"""
    try:
        from ml_model_builder import PGPRMLModelBuilder
        import os
        import pickle
        
        ml_builder = PGPRMLModelBuilder()
        
        # Essayer de charger les modèles avec gestion d'erreur spécifique
        try:
            ml_builder.load_models()
        except Exception as model_error:
            error_msg = str(model_error)
            if "_pyx_unpickle_CyHalfBinomialLoss" in error_msg or "CyHalfBinomialLoss" in error_msg:
                st.error("❌ Erreur de compatibilité sklearn détectée")
                st.warning("⚠️ Les modèles ont été sauvegardés avec une version différente de sklearn.")
                st.info("💡 **Solution automatique:** Utilisation des métriques par défaut.")
                st.info("🔄 Pour utiliser les vrais modèles, ré-entraînez avec: `python retrain_with_validation.py`")
                
                # Retourner des métriques par défaut au lieu de None
                default_metrics = {
                    "random_forest": {
                        "accuracy": 0.2500,
                        "f1_score": 0.3695,
                        "precision": 0.700,
                        "recall": 0.254,
                        "details": {
                            "Bacillus_subtilis": {"precision": 0.400, "recall": 0.286, "f1": 0.333},
                            "Escherichia_coli": {"precision": 1.000, "recall": 0.273, "f1": 0.429},
                            "Pseudomonas_aeruginosa": {"precision": 1.000, "recall": 0.125, "f1": 0.222},
                            "Staphylococcus_aureus": {"precision": 1.000, "recall": 0.333, "f1": 0.500}
                        }
                    },
                    "gradient_boosting": {
                        "accuracy": 0.2188,
                        "f1_score": 0.3842,
                        "precision": 0.381,
                        "recall": 0.397,
                        "details": {
                            "Bacillus_subtilis": {"precision": 0.400, "recall": 0.571, "f1": 0.471},
                            "Escherichia_coli": {"precision": 0.455, "recall": 0.455, "f1": 0.455},
                            "Pseudomonas_aeruginosa": {"precision": 0.500, "recall": 0.250, "f1": 0.333},
                            "Staphylococcus_aureus": {"precision": 0.167, "recall": 0.333, "f1": 0.222}
                        }
                    },
                    "svm": {
                        "accuracy": 0.2188,
                        "f1_score": 0.2811,
                        "precision": 0.708,
                        "recall": 0.233,
                        "details": {
                            "Bacillus_subtilis": {"precision": 0.833, "recall": 0.714, "f1": 0.769},
                            "Escherichia_coli": {"precision": 1.000, "recall": 0.091, "f1": 0.167},
                            "Pseudomonas_aeruginosa": {"precision": 1.000, "recall": 0.125, "f1": 0.222},
                            "Staphylococcus_aureus": {"precision": 0.000, "recall": 0.000, "f1": 0.000}
                        }
                    },
                    "mlp": {
                        "accuracy": 0.4688,
                        "f1_score": 0.5296,
                        "precision": 0.654,
                        "recall": 0.524,
                        "details": {
                            "Bacillus_subtilis": {"precision": 0.583, "recall": 1.000, "f1": 0.737},
                            "Escherichia_coli": {"precision": 0.700, "recall": 0.636, "f1": 0.667},
                            "Pseudomonas_aeruginosa": {"precision": 0.333, "recall": 0.125, "f1": 0.182},
                            "Staphylococcus_aureus": {"precision": 1.000, "recall": 0.333, "f1": 0.500}
                        }
                    },
                    "neural_network": {
                        "accuracy": 0.5938,
                        "f1_score": 0.6906,
                        "precision": 0.652,
                        "recall": 0.757,
                        "details": {
                            "Bacillus_subtilis": {"precision": 0.538, "recall": 1.000, "f1": 0.700},
                            "Escherichia_coli": {"precision": 0.818, "recall": 0.818, "f1": 0.818},
                            "Pseudomonas_aeruginosa": {"precision": 0.429, "recall": 0.375, "f1": 0.400},
                            "Staphylococcus_aureus": {"precision": 0.833, "recall": 0.833, "f1": 0.833}
                        }
                    }
                }
                return None, default_metrics
            else:
                st.error(f"❌ Erreur lors du chargement des modèles: {model_error}")
                st.info("💡 Les modèles existent mais il y a un problème de compatibilité sklearn.")
                st.info("🔄 Solution: Redémarrez l'application ou utilisez les métriques par défaut.")
                return None, None
        
        # Essayer de charger les métriques depuis un fichier de cache
        metrics_cache_path = "./rag_cache/model_metrics.pkl"
        
        if os.path.exists(metrics_cache_path):
            try:
                with open(metrics_cache_path, 'rb') as f:
                    metrics = pickle.load(f)
                st.info("📊 Métriques chargées depuis le cache (modèles retrainés)")
                return ml_builder, metrics
            except:
                pass
        
            # Si pas de cache, essayer d'évaluer les modèles sur les données de test
        try:
            from dataset_processor import CSVDatasetProcessor
            from image_processor import PGPRImageProcessor
            import numpy as np
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
            
            st.info("🔄 Évaluation des modèles en cours...")
            
            # Charger les données de test
            csv_processor = CSVDatasetProcessor(
                "./pgpr_images/images/",
                "./pgpr_images/train_labels.csv",
                "./pgpr_images/test_labels.csv"
            )
            
            # Extraire les features des images de test
            image_processor = PGPRImageProcessor()
            test_features = []
            test_labels = []
            
            # Récupérer les images de test
            test_data = csv_processor.get_all_image_paths_with_labels("test")
            
            for image_path, labels, split in test_data:
                if os.path.exists(image_path):
                    try:
                        features = image_processor.extract_features(str(image_path))
                        test_features.append(features)
                        
                        # Convertir les labels en format attendu
                        label_vector = [labels[bacteria] for bacteria in ml_builder.bacteria_types]
                        test_labels.append(label_vector)
                    except Exception as e:
                        st.warning(f"Erreur lors du traitement de {image_path}: {e}")
            
            if len(test_features) == 0:
                st.error("❌ Aucune donnée de test trouvée")
                return ml_builder, None
            
            test_features = np.array(test_features)
            test_labels = np.array(test_labels)
            
            # Évaluer chaque modèle
            metrics = {}
            
            for model_name in ml_builder.models.keys():
                try:
                    # Faire les prédictions
                    predictions, probabilities = ml_builder.predict(test_features, model_name)
                    
                    # Calculer les métriques
                    accuracy = accuracy_score(test_labels, predictions)
                    f1 = f1_score(test_labels, predictions, average='weighted')
                    precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
                    recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
                    
                    # Rapport détaillé par classe
                    report = classification_report(test_labels, predictions, 
                                                 target_names=ml_builder.bacteria_types, 
                                                 output_dict=True, zero_division=0)
                    
                    # Détails par bactérie - S'assurer que toutes les bactéries sont incluses
                    details = {}
                    for bacteria in ml_builder.bacteria_types:
                        if bacteria in report:
                            details[bacteria] = {
                                "precision": report[bacteria]['precision'],
                                "recall": report[bacteria]['recall'],
                                "f1": report[bacteria]['f1-score']
                            }
                        else:
                            # Valeurs par défaut si la bactérie n'est pas dans le rapport
                            details[bacteria] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                    
                    metrics[model_name] = {
                        "accuracy": accuracy,
                        "f1_score": f1,
                        "precision": precision,
                        "recall": recall,
                        "details": details
                    }
                    
                except Exception as e:
                    st.warning(f"Erreur lors de l'évaluation de {model_name}: {e}")
                    # Utiliser des métriques par défaut en cas d'erreur
                    default_details = {}
                    for bacteria in ml_builder.bacteria_types:
                        default_details[bacteria] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                    
                    metrics[model_name] = {
                        "accuracy": 0.0,
                        "f1_score": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "details": default_details
                    }
            
            # Sauvegarder les métriques pour la prochaine fois
            try:
                with open(metrics_cache_path, 'wb') as f:
                    pickle.dump(metrics, f)
            except:
                pass
            
            st.success("✅ Modèles évalués avec succès!")
            return ml_builder, metrics
            
        except Exception as e:
            st.warning(f"Impossible d'évaluer les modèles: {e}")
            st.info("Utilisation des métriques par défaut...")
            
            # Métriques par défaut (anciennes valeurs)
            metrics = {
            "random_forest": {
                "accuracy": 0.2500,
                "f1_score": 0.3695,
                    "precision": 0.700,
                    "recall": 0.254,
                "details": {
                    "Bacillus_subtilis": {"precision": 0.400, "recall": 0.286, "f1": 0.333},
                    "Escherichia_coli": {"precision": 1.000, "recall": 0.273, "f1": 0.429},
                    "Pseudomonas_aeruginosa": {"precision": 1.000, "recall": 0.125, "f1": 0.222},
                    "Staphylococcus_aureus": {"precision": 1.000, "recall": 0.333, "f1": 0.500}
                }
            },
            "gradient_boosting": {
                "accuracy": 0.2188,
                "f1_score": 0.3842,
                    "precision": 0.381,
                    "recall": 0.397,
                "details": {
                    "Bacillus_subtilis": {"precision": 0.400, "recall": 0.571, "f1": 0.471},
                    "Escherichia_coli": {"precision": 0.455, "recall": 0.455, "f1": 0.455},
                    "Pseudomonas_aeruginosa": {"precision": 0.500, "recall": 0.250, "f1": 0.333},
                    "Staphylococcus_aureus": {"precision": 0.167, "recall": 0.333, "f1": 0.222}
                }
            },
            "svm": {
                "accuracy": 0.2188,
                "f1_score": 0.2811,
                    "precision": 0.708,
                    "recall": 0.233,
                "details": {
                    "Bacillus_subtilis": {"precision": 0.833, "recall": 0.714, "f1": 0.769},
                    "Escherichia_coli": {"precision": 1.000, "recall": 0.091, "f1": 0.167},
                    "Pseudomonas_aeruginosa": {"precision": 1.000, "recall": 0.125, "f1": 0.222},
                    "Staphylococcus_aureus": {"precision": 0.000, "recall": 0.000, "f1": 0.000}
                }
            },
            "mlp": {
                "accuracy": 0.4688,
                "f1_score": 0.5296,
                    "precision": 0.654,
                    "recall": 0.524,
                "details": {
                    "Bacillus_subtilis": {"precision": 0.583, "recall": 1.000, "f1": 0.737},
                    "Escherichia_coli": {"precision": 0.700, "recall": 0.636, "f1": 0.667},
                    "Pseudomonas_aeruginosa": {"precision": 0.333, "recall": 0.125, "f1": 0.182},
                    "Staphylococcus_aureus": {"precision": 1.000, "recall": 0.333, "f1": 0.500}
                }
            },
            "neural_network": {
                "accuracy": 0.5938,
                "f1_score": 0.6906,
                    "precision": 0.652,
                    "recall": 0.757,
                "details": {
                    "Bacillus_subtilis": {"precision": 0.538, "recall": 1.000, "f1": 0.700},
                    "Escherichia_coli": {"precision": 0.818, "recall": 0.818, "f1": 0.818},
                    "Pseudomonas_aeruginosa": {"precision": 0.429, "recall": 0.375, "f1": 0.400},
                    "Staphylococcus_aureus": {"precision": 0.833, "recall": 0.833, "f1": 0.833}
                }
            }
        }
        
            return ml_builder, metrics
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles ML: {e}")
        st.info("🔄 Utilisation des métriques par défaut...")
        
        # Retourner des métriques par défaut même si les modèles ne peuvent pas être chargés
        default_metrics = {
            "random_forest": {
                "accuracy": 0.2500,
                "f1_score": 0.3695,
                "precision": 0.700,
                "recall": 0.254,
                "details": {
                    "Bacillus_subtilis": {"precision": 0.400, "recall": 0.286, "f1": 0.333},
                    "Escherichia_coli": {"precision": 1.000, "recall": 0.273, "f1": 0.429},
                    "Pseudomonas_aeruginosa": {"precision": 1.000, "recall": 0.125, "f1": 0.222},
                    "Staphylococcus_aureus": {"precision": 1.000, "recall": 0.333, "f1": 0.500}
                }
            },
            "gradient_boosting": {
                "accuracy": 0.2188,
                "f1_score": 0.3842,
                "precision": 0.381,
                "recall": 0.397,
                "details": {
                    "Bacillus_subtilis": {"precision": 0.400, "recall": 0.571, "f1": 0.471},
                    "Escherichia_coli": {"precision": 0.455, "recall": 0.455, "f1": 0.455},
                    "Pseudomonas_aeruginosa": {"precision": 0.500, "recall": 0.250, "f1": 0.333},
                    "Staphylococcus_aureus": {"precision": 0.167, "recall": 0.333, "f1": 0.222}
                }
            },
            "svm": {
                "accuracy": 0.2188,
                "f1_score": 0.2811,
                "precision": 0.708,
                "recall": 0.233,
                "details": {
                    "Bacillus_subtilis": {"precision": 0.833, "recall": 0.714, "f1": 0.769},
                    "Escherichia_coli": {"precision": 1.000, "recall": 0.091, "f1": 0.167},
                    "Pseudomonas_aeruginosa": {"precision": 1.000, "recall": 0.125, "f1": 0.222},
                    "Staphylococcus_aureus": {"precision": 0.000, "recall": 0.000, "f1": 0.000}
                }
            },
            "mlp": {
                "accuracy": 0.4688,
                "f1_score": 0.5296,
                "precision": 0.654,
                "recall": 0.524,
                "details": {
                    "Bacillus_subtilis": {"precision": 0.583, "recall": 1.000, "f1": 0.737},
                    "Escherichia_coli": {"precision": 0.700, "recall": 0.636, "f1": 0.667},
                    "Pseudomonas_aeruginosa": {"precision": 0.333, "recall": 0.125, "f1": 0.182},
                    "Staphylococcus_aureus": {"precision": 1.000, "recall": 0.333, "f1": 0.500}
                }
            },
            "neural_network": {
                "accuracy": 0.5938,
                "f1_score": 0.6906,
                "precision": 0.652,
                "recall": 0.757,
                "details": {
                    "Bacillus_subtilis": {"precision": 0.538, "recall": 1.000, "f1": 0.700},
                    "Escherichia_coli": {"precision": 0.818, "recall": 0.818, "f1": 0.818},
                    "Pseudomonas_aeruginosa": {"precision": 0.429, "recall": 0.375, "f1": 0.400},
                    "Staphylococcus_aureus": {"precision": 0.833, "recall": 0.833, "f1": 0.833}
                }
            }
        }
        
        return None, default_metrics

def display_model_comparison():
    """Affiche la comparaison des modèles ML"""
    st.header("⚖️ Comparaison des Modèles ML")
    
    # Add refresh button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.info("📊 Cette page affiche les performances des modèles ML retrainés avec votre dataset étendu")
    with col2:
        if st.button("🔄 Actualiser", help="Recharger les métriques des modèles"):
            # Clear metrics cache to force reload
            import os
            metrics_cache_path = "./rag_cache/model_metrics.pkl"
            if os.path.exists(metrics_cache_path):
                os.remove(metrics_cache_path)
            st.rerun()
    
    # Ajouter un bouton pour ré-entraîner les modèles
    if st.button("🔧 Ré-entraîner les modèles", help="Résoudre les problèmes de compatibilité sklearn"):
        st.warning("⚠️ Cette action va supprimer les modèles actuels et nécessiter un ré-entraînement.")
        if st.button("✅ Confirmer le ré-entraînement", type="primary"):
            try:
                import subprocess
                result = subprocess.run(["python", "quick_retrain.py"], 
                                     capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    st.success("✅ Préparation terminée! Exécutez maintenant: `python retrain_ml_only.py`")
                    st.info("💡 Après le ré-entraînement, relancez l'application web.")
                    st.info("⚡ retrain_ml_only.py est 5-10x plus rapide que retrain_with_validation.py")
                else:
                    st.error(f"❌ Erreur lors de la préparation: {result.stderr}")
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
            st.rerun()
    with col3:
        if st.button("ℹ️ Info", help="Informations sur les modèles"):
            st.info("""
            **Modèles disponibles:**
            - Random Forest: Rapide, robuste
            - Gradient Boosting: Bonne précision
            - SVM: Linéaire, efficace  
            - MLP: Réseau de neurones simple
            - Neural Network: Réseau personnalisé PyTorch
            """)
    
    # Charger les modèles et métriques
    ml_builder, metrics = load_ml_models_and_metrics()
    
    if metrics is None:
        st.error("❌ Impossible de charger les modèles ML")
        return
    
    # Afficher un message si on utilise les métriques par défaut
    if ml_builder is None and metrics is not None:
        st.warning("⚠️ Utilisation des métriques par défaut (modèles non compatibles)")
        st.info("💡 Pour utiliser les vrais modèles, ré-entraînez avec: `python retrain_with_validation.py`")
    
    if ml_builder is not None:
        st.success("✅ Modèles ML chargés avec succès!")
    else:
        st.info("📊 Métriques par défaut affichées")
    
    # Dataset information
    st.subheader("📊 Informations sur le Dataset")
    try:
        import pandas as pd
        
        # Load dataset info
        train_df = pd.read_csv("./pgpr_images/train_labels.csv")
        test_df = pd.read_csv("./pgpr_images/test_labels.csv")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.metric("Images d'entraînement", len(train_df))
            st.metric("Images de test", len(test_df))
        
        with col_info2:
            bacteria_types = [col for col in train_df.columns if col != 'filename']
            st.write("**Types de bactéries:**")
            for bacteria in bacteria_types:
                count = train_df[bacteria].sum()
                st.write(f"• {bacteria}: {count}")
        
        with col_info3:
            st.write("**Distribution:**")
            total_images = len(train_df) + len(test_df)
            train_pct = (len(train_df) / total_images) * 100
            test_pct = (len(test_df) / total_images) * 100
            st.write(f"• Train: {train_pct:.1f}%")
            st.write(f"• Test: {test_pct:.1f}%")
            
    except Exception as e:
        st.warning(f"Impossible de charger les informations du dataset: {e}")
    
    # Métriques globales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Métriques Globales")
        
        # Créer un DataFrame pour les métriques
        import pandas as pd
        
        metrics_data = []
        for model_name, model_metrics in metrics.items():
            metrics_data.append({
                "Modèle": model_name.replace("_", " ").title(),
                "Accuracy": f"{model_metrics['accuracy']:.3f}",
                "F1-Score": f"{model_metrics['f1_score']:.3f}",
                "Precision": f"{model_metrics['precision']:.3f}",
                "Recall": f"{model_metrics['recall']:.3f}"
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
        
        # Graphique des métriques
        import plotly.express as px
        
        # Préparer les données pour le graphique
        plot_data = []
        for model_name, model_metrics in metrics.items():
            plot_data.extend([
                {"Modèle": model_name.replace("_", " ").title(), "Métrique": "Accuracy", "Valeur": model_metrics['accuracy']},
                {"Modèle": model_name.replace("_", " ").title(), "Métrique": "F1-Score", "Valeur": model_metrics['f1_score']},
                {"Modèle": model_name.replace("_", " ").title(), "Métrique": "Precision", "Valeur": model_metrics['precision']},
                {"Modèle": model_name.replace("_", " ").title(), "Métrique": "Recall", "Valeur": model_metrics['recall']}
            ])
        
        df_plot = pd.DataFrame(plot_data)
        
        fig = px.bar(df_plot, x="Modèle", y="Valeur", color="Métrique", 
                    title="Comparaison des Métriques par Modèle",
                    barmode="group")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(" Classement des Modèles")
        
        # Trier par F1-Score (métrique la plus importante)
        sorted_models = sorted(metrics.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        for i, (model_name, model_metrics) in enumerate(sorted_models, 1):
            model_display = model_name.replace("_", " ").title()
            
            # Badge de position
            if i == 1:
                badge = "🥇"
                color = "success"
            elif i == 2:
                badge = "🥈"
                color = "info"
            elif i == 3:
                badge = "🥉"
                color = "warning"
            else:
                badge = f"#{i}"
                color = "secondary"
            
            with st.container():
                col_badge, col_info = st.columns([1, 4])
                with col_badge:
                    st.markdown(f"<div style='text-align: center; font-size: 24px;'>{badge}</div>", unsafe_allow_html=True)
                
                with col_info:
                    st.markdown(f"**{model_display}**")
                    st.progress(model_metrics['f1_score'])
                    st.caption(f"F1-Score: {model_metrics['f1_score']:.3f} | Accuracy: {model_metrics['accuracy']:.3f}")
    
    # Détails par bactérie
    st.subheader("🦠 Détails par Type de Bactérie")
    
    bacteria_types = ["Bacillus_subtilis", "Escherichia_coli", "Pseudomonas_aeruginosa", "Staphylococcus_aureus"]
    
    for bacteria in bacteria_types:
        with st.expander(f"📋 {bacteria.replace('_', ' ').title()}"):
            bacteria_data = []
            
            # Debug: Check if metrics exist and have details
            if not metrics:
                st.warning("❌ Aucune métrique disponible")
                continue
                
            for model_name, model_metrics in metrics.items():
                try:
                    # Check if details exist for this bacteria
                    if 'details' in model_metrics and bacteria in model_metrics['details']:
                        details = model_metrics['details'][bacteria]
                        bacteria_data.append({
                            "Modèle": model_name.replace("_", " ").title(),
                            "Precision": f"{details.get('precision', 0.0):.3f}",
                            "Recall": f"{details.get('recall', 0.0):.3f}",
                            "F1-Score": f"{details.get('f1', 0.0):.3f}"
                        })
                    else:
                        # Add default values if details not found
                        bacteria_data.append({
                            "Modèle": model_name.replace("_", " ").title(),
                            "Precision": "0.000",
                            "Recall": "0.000", 
                            "F1-Score": "0.000"
                        })
                except Exception as e:
                    st.warning(f"Erreur pour {model_name}: {e}")
                    bacteria_data.append({
                        "Modèle": model_name.replace("_", " ").title(),
                        "Precision": "N/A",
                        "Recall": "N/A",
                        "F1-Score": "N/A"
                    })
            
            if bacteria_data:
                df_bacteria = pd.DataFrame(bacteria_data)
                st.dataframe(df_bacteria, use_container_width=True)
                
                # Try to create chart with numeric values
                try:
                    # Convert back to numeric for plotting
                    df_plot = df_bacteria.copy()
                    for col in ["Precision", "Recall", "F1-Score"]:
                        df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce').fillna(0)
                    
                    fig_bacteria = px.bar(df_plot, x="Modèle", y=["Precision", "Recall", "F1-Score"],
                                        title=f"Performance des Modèles pour {bacteria.replace('_', ' ').title()}",
                                        barmode="group")
                    fig_bacteria.update_layout(height=400)
                    st.plotly_chart(fig_bacteria, use_container_width=True)
                except Exception as e:
                    st.warning(f"Impossible de créer le graphique: {e}")
            else:
                st.warning(f"❌ Aucune donnée disponible pour {bacteria.replace('_', ' ').title()}")
                
                # Show fallback data
                st.info("📊 Affichage des données par défaut...")
                fallback_data = []
                for model_name in ["random_forest", "gradient_boosting", "svm", "mlp", "neural_network"]:
                    fallback_data.append({
                        "Modèle": model_name.replace("_", " ").title(),
                        "Precision": "N/A",
                        "Recall": "N/A",
                        "F1-Score": "N/A"
                    })
                
                df_fallback = pd.DataFrame(fallback_data)
                st.dataframe(df_fallback, use_container_width=True)
                
                # Show debug info
                with st.expander("🔍 Debug Info"):
                    st.write(f"Metrics keys: {list(metrics.keys()) if metrics else 'None'}")
                    if metrics:
                        for model_name, model_metrics in metrics.items():
                            st.write(f"{model_name}: {list(model_metrics.keys()) if isinstance(model_metrics, dict) else 'Not a dict'}")
                            if isinstance(model_metrics, dict) and 'details' in model_metrics:
                                st.write(f"  Details keys: {list(model_metrics['details'].keys())}")
                                st.write(f"  Looking for: {bacteria}")
                                st.write(f"  Found: {bacteria in model_metrics['details']}")
    
    # Recommandations
    st.subheader(" Recommandations")
    
    best_model = sorted_models[0][0]
    best_model_display = best_model.replace("_", " ").title()
    
    st.info(f"""
    ** Modèle Recommandé: {best_model_display}**
    
    - **F1-Score le plus élevé**: {sorted_models[0][1]['f1_score']:.3f}
    - **Accuracy**: {sorted_models[0][1]['accuracy']:.3f}
    - **Performance équilibrée** entre precision et recall
    
    **📈 Observations:**
    - Le réseau de neurones personnalisé montre les meilleures performances globales
    - Les modèles SVM et MLP ont des performances variables selon les types de bactéries
    - Random Forest et Gradient Boosting ont des performances plus modestes sur ce dataset
    """)

def main():
    # En-tête principal
    st.markdown('<h1 class="main-header">🧬 Chatbot PGPR Multimodal Enrichi</h1>', unsafe_allow_html=True)
    
    # Charger le système
    system = load_system()
    image_processor = load_image_processor()
    
    if system is None or image_processor is None:
        st.stop()
    
    # Sidebar avec informations système
    with st.sidebar:
        st.header("📊 Informations Système")
        
        try:
            stats = system.get_system_stats()
            if "error" not in stats:
                st.metric("Documents totaux", stats["total_documents"])
                st.metric("Documents texte", stats["text_documents"])
                st.metric("Images", stats["image_documents"])
                st.metric("Images enrichies ML", stats["ml_enriched_images"])
                
                if stats["ml_models_available"]:
                    st.subheader("🤖 Modèles ML disponibles")
                    for model in stats["ml_models_available"]:
                        st.success(f"✅ {model}")
            else:
                st.warning("Statistiques non disponibles")
        except:
            st.warning("Statistiques non disponibles")
        
        # Sélection du modèle ML actif
        st.subheader(" Modèle ML Actif")
        if hasattr(system, 'ml_builder') and system.ml_builder and system.ml_builder.models:
            available_models = list(system.ml_builder.models.keys())
            active_model = st.selectbox(
                "Choisir le modèle pour les prédictions:",
                available_models,
                index=len(available_models)-1 if "neural_network" in available_models else 0
            )
            st.session_state.active_model = active_model
        else:
            st.warning("Aucun modèle ML disponible")
    
    # Onglets principaux - seulement 4 onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat Textuel", 
        "🖼️ Analyse d'Images", 
        "📊 Statistiques",
        "⚖️ Comparaison Modèles"
    ])
    
    # Onglet 1: Chat Textuel
    with tab1:
        st.subheader("💬 Chat avec le Système RAG Enrichi")
        st.info("Posez des questions sur les PGPR en utilisant le contexte enrichi par les modèles ML")
        
        # Zone de saisie
        user_question = st.text_area(
            "Votre question:",
            placeholder="Ex: Quels types de bactéries PGPR sont présents dans les images et comment les identifier?",
            height=100
        )
        
        if st.button("🚀 Poser la question", type="primary"):
            if user_question.strip():
                with st.spinner("Recherche en cours..."):
                    try:
                        response = system.query(user_question)
                        st.success("✅ Réponse générée!")
                        st.markdown("### 📝 Réponse:")
                        st.markdown(response)
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération de la réponse: {e}")
            else:
                st.warning("⚠️ Veuillez saisir une question")
    
    # Onglet 2: Analyse d'Images
    with tab2:
        st.subheader("🔍 Analyse d'Images PGPR")
        st.info("Uploadez une image pour l'analyser et obtenir des prédictions ML en temps réel")
        
        uploaded_file = st.file_uploader(
            "Choisir une image:",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Formats supportés: PNG, JPG, JPEG, BMP"
        )
        
        if uploaded_file is not None:
            # Afficher l'image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Image uploadée", use_column_width=True)
            
            with col2:
                if st.button(" Analyser l'image", type="primary"):
                    with st.spinner("Analyse en cours..."):
                        try:
                            # Sauvegarder temporairement l'image
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                image.save(tmp_file.name, 'JPEG')
                                tmp_path = tmp_file.name
                            
                            # Analyse de base
                            analysis = image_processor.analyze_image(tmp_path)
                            
                            # Prédictions ML si disponible
                            ml_results = {}
                            if hasattr(system, 'ml_builder') and system.ml_builder:
                                try:
                                    features = image_processor.extract_features(tmp_path)
                                    predictions, probabilities = system.ml_builder.predict(
                                        features.reshape(1, -1), 
                                        st.session_state.get('active_model', 'neural_network')
                                    )
                                    
                                    ml_results = {
                                        "predictions": predictions[0],
                                        "probabilities": probabilities[0]
                                    }
                                except Exception as e:
                                    st.warning(f"Prédictions ML non disponibles: {e}")
                            
                            # Nettoyer le fichier temporaire
                            os.unlink(tmp_path)
                            
                            # Affichage des résultats
                            st.success("✅ Analyse terminée!")
                            
                            # Métriques de base
                            st.metric("Nombre de bactéries détectées", analysis.get("bacteria_count", 0))
                            st.metric("Surface analysée (pixels²)", analysis.get("total_area", 0))
                            
                            # Prédictions ML
                            if ml_results:
                                st.subheader("🤖 Prédictions ML")
                                
                                # Créer un graphique en barres pour les probabilités
                                bacteria_names = ["Bacillus subtilis", "Escherichia coli", "Pseudomonas aeruginosa", "Staphylococcus aureus"]
                                probs = ml_results["probabilities"]
                                
                                fig = px.bar(
                                    x=bacteria_names,
                                    y=probs,
                                    title="Probabilités de détection par type de bactérie",
                                    labels={'x': 'Type de bactérie', 'y': 'Probabilité'},
                                    color=probs,
                                    color_continuous_scale='viridis'
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Bactéries détectées
                                detected = []
                                for i, (name, pred) in enumerate(zip(bacteria_names, ml_results["predictions"])):
                                    if pred == 1:
                                        detected.append({
                                            "name": name,
                                            "confidence": probs[i]
                                        })
                                
                                if detected:
                                    st.subheader("🦠 Bactéries détectées")
                                    for bacteria in detected:
                                        st.markdown(f"""
                                        <div class="confidence-bar">
                                            <strong>{bacteria['name']}</strong>
                                            <div class="confidence-fill" style="width: {bacteria['confidence']*100:.1f}%"></div>
                                            <small>Confiance: {bacteria['confidence']*100:.1f}%</small>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("Aucune bactérie détectée avec confiance suffisante")
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'analyse: {e}")
    
    # Onglet 3: Statistiques
    with tab3:
        st.subheader("📊 Statistiques du Dataset")
        st.info("Visualisez la distribution des images et des types de bactéries")
        
        try:
            # Charger les données CSV
            train_csv = "./pgpr_images/train_labels.csv"
            test_csv = "./pgpr_images/test_labels.csv"
            
            if os.path.exists(train_csv) and os.path.exists(test_csv):
                train_df = pd.read_csv(train_csv)
                test_df = pd.read_csv(test_csv)
                
                # Types de bactéries
                bacteria_types = [col for col in train_df.columns if col != 'filename']
                
                # Statistiques par split
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Distribution par Split")
                    
                    # Données pour le graphique
                    split_data = {
                        'Split': ['Train', 'Test'],
                        'Nombre d\'images': [len(train_df), len(test_df)]
                    }
                    
                    fig = px.pie(
                        values=split_data['Nombre d\'images'],
                        names=split_data['Split'],
                        title="Répartition Train/Test"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("🦠 Distribution par Type de Bactérie")
                    
                    # Compter les occurrences par type
                    bacteria_counts = {}
                    for bacteria in bacteria_types:
                        train_count = train_df[bacteria].sum()
                        test_count = test_df[bacteria].sum()
                        bacteria_counts[bacteria] = train_count + test_count
                    
                    # Graphique en barres
                    fig = px.bar(
                        x=list(bacteria_counts.keys()),
                        y=list(bacteria_counts.values()),
                        title="Nombre d'images par type de bactérie",
                        labels={'x': 'Type de bactérie', 'y': 'Nombre d\'images'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques détaillées
                st.subheader("📋 Statistiques Détaillées")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Train set:**")
                    for bacteria in bacteria_types:
                        count = train_df[bacteria].sum()
                        percentage = (count / len(train_df)) * 100
                        st.write(f"- {bacteria}: {count} ({percentage:.1f}%)")
                
                with col2:
                    st.write("**Test set:**")
                    for bacteria in bacteria_types:
                        count = test_df[bacteria].sum()
                        percentage = (count / len(test_df)) * 100
                        st.write(f"- {bacteria}: {count} ({percentage:.1f}%)")
                
                # Analyse des multi-labels
                st.subheader("🔗 Analyse Multi-labels")
                
                # Compter les échantillons multi-labels
                train_multi = (train_df[bacteria_types].sum(axis=1) > 1).sum()
                test_multi = (test_df[bacteria_types].sum(axis=1) > 1).sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Multi-labels Train", train_multi)
                with col2:
                    st.metric("Multi-labels Test", test_multi)
                
                if train_multi > 0 or test_multi > 0:
                    st.info("⚠️ Ce dataset contient des échantillons multi-labels (une image peut contenir plusieurs types de bactéries)")
                else:
                    st.success("✅ Ce dataset ne contient que des échantillons mono-label")
                    
            else:
                st.error("❌ Fichiers CSV non trouvés")
                
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des statistiques: {e}")
    
    # Onglet 4: Comparaison Modèles
    with tab4:
        display_model_comparison()

if __name__ == "__main__":
    main()