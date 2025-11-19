# build_enhanced_system.py - Script de construction automatique du système RAG multimodal enrichi

import os
import sys
import subprocess
import time
from pathlib import Path

def check_prerequisites():
    """Vérifie les prérequis du système"""
    print("=== VÉRIFICATION DES PRÉREQUIS ===\n")
    
    # Vérifier Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            llama_model = any("llama3.1" in model.get("name", "") for model in models)
            if llama_model:
                print("✅ Ollama est en cours d'exécution avec le modèle llama3.1")
            else:
                print("⚠️ Ollama est en cours d'exécution mais le modèle llama3.1 n'est pas trouvé")
                print("   Installez-le avec: ollama pull llama3.1")
                return False
        else:
            print("❌ Ollama n'est pas accessible sur le port 11434")
            return False
    except Exception as e:
        print(f"❌ Erreur lors de la vérification d'Ollama: {e}")
        print("   Assurez-vous qu'Ollama est installé et en cours d'exécution")
        return False
    
    # Vérifier le dossier des documents PDF
    if not Path("./pgpr_docs/").exists():
        print("❌ Le dossier ./pgpr_docs/ n'existe pas")
        print("   Créez-le et ajoutez vos documents PDF sur les PGPR")
        return False
    else:
        pdf_files = list(Path("./pgpr_docs/").glob("*.pdf"))
        if pdf_files:
            print(f"✅ Dossier pgpr_docs/ trouvé avec {len(pdf_files)} fichiers PDF")
        else:
            print("⚠️ Dossier pgpr_docs/ trouvé mais aucun fichier PDF détecté")
    
    # Vérifier la structure du dataset d'images
    images_dir = Path("./pgpr_images/images/")
    train_csv = Path("./pgpr_images/train_labels.csv")
    test_csv = Path("./pgpr_images/test_labels.csv")
    
    if not images_dir.exists():
        print("❌ Le dossier ./pgpr_images/images/ n'existe pas")
        return False
    
    if not train_csv.exists():
        print("❌ Le fichier ./pgpr_images/train_labels.csv n'existe pas")
        return False
    
    if not test_csv.exists():
        print("❌ Le fichier ./pgpr_images/test_labels.csv n'existe pas")
        return False
    
    # Vérifier le contenu des CSV
    try:
        import pandas as pd
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        
        print(f"✅ Dataset CSV trouvé:")
        print(f"   - Train: {len(train_df)} images")
        print(f"   - Test: {len(test_df)} images")
        
        # Vérifier les types de bactéries
        bacteria_cols = [col for col in train_df.columns if col != 'filename']
        print(f"   - Types de bactéries: {bacteria_cols}")
        
        # Vérifier la présence des images
        missing_train = []
        missing_test = []
        
        for filename in train_df['filename']:
            if not (images_dir / filename).exists():
                missing_train.append(filename)
        
        for filename in test_df['filename']:
            if not (images_dir / filename).exists():
                missing_test.append(filename)
        
        if missing_train or missing_test:
            print(f"⚠️ {len(missing_train) + len(missing_test)} images référencées dans les CSV sont manquantes")
            if missing_train:
                print(f"   Train manquantes: {missing_train[:3]}{'...' if len(missing_train) > 3 else ''}")
            if missing_test:
                print(f"   Test manquantes: {missing_test[:3]}{'...' if len(missing_test) > 3 else ''}")
        else:
            print("✅ Toutes les images référencées sont présentes")
            
    except Exception as e:
        print(f"❌ Erreur lors de la vérification des CSV: {e}")
        return False
    
    print("\n✅ Tous les prérequis sont satisfaits!")
    return True

def install_dependencies():
    """Installe les dépendances Python manquantes"""
    print("\n=== INSTALLATION DES DÉPENDANCES ===\n")
    
    try:
        # Vérifier si les modules essentiels sont installés
        required_modules = [
            'langchain_ollama', 'langchain_community', 'langchain',
            'torch', 'torchvision', 'opencv-python', 'Pillow',
            'scikit-learn', 'joblib', 'plotly', 'streamlit'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module.replace('-', '_'))
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print(f"Installation des modules manquants: {', '.join(missing_modules)}")
            
            # Installer depuis requirements.txt
            if Path("requirements.txt").exists():
                print("Installation depuis requirements.txt...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            else:
                print("Installation des modules manquants...")
                for module in missing_modules:
                    subprocess.run([sys.executable, "-m", "pip", "install", module], check=True)
            
            print("✅ Dépendances installées avec succès")
        else:
            print("✅ Toutes les dépendances sont déjà installées")
            
    except Exception as e:
        print(f"❌ Erreur lors de l'installation des dépendances: {e}")
        return False
    
    return True

def analyze_dataset():
    """Analyse le dataset CSV et affiche des statistiques détaillées"""
    print("\n=== ANALYSE DU DATASET ===\n")
    
    try:
        from dataset_processor import analyze_csv_dataset
        
        train_csv = "./pgpr_images/train_labels.csv"
        test_csv = "./pgpr_images/test_labels.csv"
        
        analyze_csv_dataset(train_csv, test_csv)
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse du dataset: {e}")

def build_ml_models():
    """Construit et entraîne les modèles ML"""
    print("\n=== CONSTRUCTION DES MODÈLES ML ===\n")
    
    try:
        from enhanced_multimodal_rag import build_enhanced_multimodal_rag
        
        images_dir = "./pgpr_images/images/"
        train_csv = "./pgpr_images/train_labels.csv"
        test_csv = "./pgpr_images/test_labels.csv"
        
        print("Construction du système RAG multimodal enrichi...")
        print("Cette étape peut prendre 10-30 minutes selon votre matériel...")
        
        system = build_enhanced_multimodal_rag(images_dir, train_csv, test_csv)
        
        print("\n✅ Système RAG multimodal enrichi construit avec succès!")
        return system
        
    except Exception as e:
        print(f"❌ Erreur lors de la construction du système: {e}")
        return None

def test_system(system):
    """Teste le système construit"""
    print("\n=== TEST DU SYSTÈME ===\n")
    
    try:
        # Test de requête textuelle
        print("Test de requête textuelle...")
        question = "Qu'est-ce que les PGPR et comment fonctionnent-ils?"
        response = system.query(question)
        print(f"Question: {question}")
        print(f"Réponse: {response[:200]}...")
        
        # Test de prédiction ML sur une image
        print("\nTest de prédiction ML...")
        images_dir = Path("./pgpr_images/images/")
        sample_image = next(images_dir.glob("*.jpg"), None)
        
        if sample_image:
            result = system.predict_image_bacteria(str(sample_image))
            if "error" not in result:
                print(f"Image testée: {sample_image.name}")
                print(f"Bactéries détectées: {result['detected_bacteria']}")
            else:
                print(f"Erreur de prédiction: {result['error']}")
        else:
            print("Aucune image trouvée pour le test")
        
        print("\n✅ Tests terminés avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors des tests: {e}")

def create_usage_instructions():
    """Crée un fichier d'instructions d'utilisation"""
    print("\n=== CRÉATION DES INSTRUCTIONS ===\n")
    
    instructions = """# INSTRUCTIONS D'UTILISATION - Système RAG Multimodal Enrichi

## �� Démarrage Rapide

### 1. Interface Web (Recommandé)
```bash
streamlit run web_chatbot_enhanced.py
```
- Ouvrez votre navigateur sur http://localhost:8501
- Utilisez les différents onglets pour explorer le système

### 2. Utilisation Programmée
```python
from enhanced_multimodal_rag import load_enhanced_multimodal_rag

# Charger le système
system = load_enhanced_multimodal_rag()

# Poser une question
response = system.query("Qu'est-ce que les PGPR?")

# Prédire les bactéries dans une image
result = system.predict_image_bacteria("chemin/vers/image.jpg")
```

## 🔧 Fonctionnalités Disponibles

### Chat Textuel
- Questions sur les PGPR basées sur vos documents PDF
- Réponses enrichies par l'analyse d'images

### Analyse d'Images
- Upload d'images pour analyse
- Détection automatique de bactéries
- Prédictions ML en temps réel

### Recherche d'Images
- Trouver des images similaires
- Enrichissement avec prédictions ML

### Prédictions ML
- Comparaison des modèles ML
- Visualisation des probabilités
- Sélection du modèle actif

### Statistiques
- Vue d'ensemble du dataset
- Distribution des types de bactéries
- Performance des modèles

## 📁 Structure des Données

### Images
- Dossier: `./pgpr_images/images/`
- Formats supportés: JPG, PNG, JPEG

### Labels CSV
- `train_labels.csv`: Images d'entraînement
- `test_labels.csv`: Images de test
- Colonnes: filename, Bacillus_subtilis, Escherichia_coli, Pseudomonas_aeruginosa, Staphylococcus_aureus

### Documents PDF
- Dossier: `./pgpr_docs/`
- Contenu scientifique sur les PGPR

## ��️ Modèles ML Disponibles

1. **Random Forest**: Rapide, robuste
2. **Gradient Boosting**: Bonne précision
3. **SVM**: Linéaire, efficace
4. **MLP**: Réseau de neurones simple
5. **Neural Network**: Réseau personnalisé PyTorch

## 🔍 Résolution de Problèmes

### Erreur "Ollama non accessible"
- Vérifiez qu'Ollama est installé et en cours d'exécution
- Testez: `ollama list`

### Erreur "Modèles ML non trouvés"
- Exécutez: `python build_enhanced_system.py`
- Attendez la fin de l'entraînement

### Performance lente
- Les embeddings prennent du temps (normal)
- Utilisez un GPU si disponible

## 📞 Support

Pour toute question ou problème, consultez:
- `README_ENHANCED.md`: Documentation complète
- `requirements.txt`: Dépendances
- Logs de construction pour diagnostiquer les erreurs
"""
    
    with open("INSTRUCTIONS_UTILISATION.md", "w", encoding="utf-8") as f:
        f.write(instructions)
    
    print("✅ Fichier INSTRUCTIONS_UTILISATION.md créé")

def main():
    """Fonction principale"""
    print("�� CONSTRUCTION AUTOMATIQUE DU SYSTÈME RAG MULTIMODAL ENRICHIE")
    print("=" * 70)
    
    # Vérifier les prérequis
    if not check_prerequisites():
        print("\n❌ Prérequis non satisfaits. Corrigez les problèmes et relancez.")
        return
    
    # Installer les dépendances
    if not install_dependencies():
        print("\n❌ Échec de l'installation des dépendances.")
        return
    
    # Analyser le dataset
    analyze_dataset()
    
    # Construire le système
    system = build_ml_models()
    if system is None:
        print("\n❌ Échec de la construction du système.")
        return
    
    # Tester le système
    test_system(system)
    
    # Créer les instructions
    create_usage_instructions()
    
    print("\n" + "=" * 70)
    print("�� SYSTÈME RAG MULTIMODAL ENRICHIE CONSTRUIT AVEC SUCCÈS!")
    print("\n📋 Prochaines étapes:")
    print("1. Lancer l'interface web: streamlit run web_chatbot_enhanced.py")
    print("2. Consulter INSTRUCTIONS_UTILISATION.md pour l'utilisation")
    print("3. Explorer les différentes fonctionnalités")
    print("\n🚀 Bonne exploration!")

if __name__ == "__main__":
    main()
