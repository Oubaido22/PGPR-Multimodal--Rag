#!/usr/bin/env python3
"""
Script pour vérifier la mémoire système et le statut d'Ollama
"""

import psutil
import requests
import subprocess
import sys

def check_system_memory():
    """Vérifie la mémoire système disponible"""
    print("🔍 Vérification de la mémoire système...")
    
    # Mémoire totale
    total_memory = psutil.virtual_memory().total / (1024**3)  # GB
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    used_memory = psutil.virtual_memory().used / (1024**3)  # GB
    
    print(f"📊 Mémoire totale: {total_memory:.1f} GB")
    print(f"📊 Mémoire utilisée: {used_memory:.1f} GB")
    print(f"📊 Mémoire disponible: {available_memory:.1f} GB")
    
    # Recommandations
    if available_memory < 5.0:
        print("⚠️  Mémoire disponible faible (< 5 GB)")
        print("💡 Recommandations:")
        print("   - Fermez d'autres applications")
        print("   - Utilisez les paramètres optimisés (déjà appliqués)")
        print("   - Considérez utiliser un modèle plus petit")
    elif available_memory < 8.0:
        print("✅ Mémoire disponible correcte (5-8 GB)")
        print("💡 Les paramètres optimisés devraient fonctionner")
    else:
        print("✅ Mémoire disponible excellente (> 8 GB)")
        print("💡 Vous pourriez augmenter les paramètres si nécessaire")
    
    return available_memory

def check_ollama_status():
    """Vérifie le statut d'Ollama"""
    print("\n🔍 Vérification d'Ollama...")
    
    try:
        # Vérifier si Ollama est en cours d'exécution
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama est en cours d'exécution")
            
            # Vérifier les modèles disponibles
            models = response.json().get("models", [])
            if models:
                print("📋 Modèles disponibles:")
                for model in models:
                    name = model.get("name", "Unknown")
                    size = model.get("size", 0) / (1024**3)  # GB
                    print(f"   - {name}: {size:.1f} GB")
            else:
                print("⚠️  Aucun modèle trouvé")
                return False
        else:
            print("❌ Ollama ne répond pas correctement")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Ollama n'est pas en cours d'exécution")
        print("💡 Démarrez Ollama avec: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la vérification d'Ollama: {e}")
        return False
    
    return True

def check_llama_model():
    """Vérifie spécifiquement le modèle llama3.1"""
    print("\n🔍 Vérification du modèle llama3.1...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json().get("models", [])
        
        llama_models = [model for model in models if "llama3.1" in model.get("name", "")]
        
        if llama_models:
            model = llama_models[0]
            name = model.get("name", "Unknown")
            size = model.get("size", 0) / (1024**3)  # GB
            print(f"✅ Modèle trouvé: {name}")
            print(f"📊 Taille: {size:.1f} GB")
            
            if size > 4.0:
                print("⚠️  Le modèle est assez volumineux")
                print("💡 Les paramètres optimisés sont nécessaires")
            else:
                print("✅ Taille du modèle acceptable")
            
            return True
        else:
            print("❌ Modèle llama3.1 non trouvé")
            print("💡 Installez-le avec: ollama pull llama3.1")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de la vérification du modèle: {e}")
        return False

def suggest_optimizations():
    """Suggère des optimisations"""
    print("\n💡 Optimisations appliquées:")
    print("✅ Context window réduit: 2048 → 1024")
    print("✅ Threads réduits: 4 → 2")
    print("✅ GPU désactivé: num_gpu = 0")
    print("✅ Réponse limitée: num_predict = 256")
    print("✅ Mode low VRAM activé")
    
    print("\n🚀 Si le problème persiste:")
    print("1. Fermez d'autres applications")
    print("2. Redémarrez Ollama: ollama serve")
    print("3. Utilisez un modèle plus petit: ollama pull llama3.1:8b")
    print("4. Augmentez la mémoire virtuelle si possible")

def main():
    """Fonction principale"""
    print("🧬 Vérificateur de mémoire et Ollama")
    print("=" * 50)
    
    # Vérifier la mémoire
    available_memory = check_system_memory()
    
    # Vérifier Ollama
    ollama_ok = check_ollama_status()
    
    # Vérifier le modèle
    model_ok = check_llama_model()
    
    # Suggestions
    suggest_optimizations()
    
    print("\n📋 Résumé:")
    print(f"   Mémoire disponible: {available_memory:.1f} GB")
    print(f"   Ollama: {'✅' if ollama_ok else '❌'}")
    print(f"   Modèle llama3.1: {'✅' if model_ok else '❌'}")
    
    if ollama_ok and model_ok and available_memory > 4.0:
        print("\n🎉 Système prêt! Vous devriez pouvoir utiliser l'interface.")
    else:
        print("\n⚠️  Problèmes détectés. Vérifiez les recommandations ci-dessus.")

if __name__ == "__main__":
    main()
