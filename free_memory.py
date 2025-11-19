#!/usr/bin/env python3
"""
Script pour libérer de la mémoire et optimiser le système
"""

import psutil
import gc
import os
import sys

def show_memory_usage():
    """Affiche l'utilisation actuelle de la mémoire"""
    memory = psutil.virtual_memory()
    print(f"📊 Mémoire actuelle:")
    print(f"   Total: {memory.total / (1024**3):.1f} GB")
    print(f"   Utilisée: {memory.used / (1024**3):.1f} GB")
    print(f"   Disponible: {memory.available / (1024**3):.1f} GB")
    print(f"   Pourcentage: {memory.percent:.1f}%")
    return memory.available / (1024**3)

def find_memory_hogs():
    """Trouve les processus qui utilisent le plus de mémoire"""
    print("\n🔍 Top 10 des processus utilisant le plus de mémoire:")
    
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Trier par utilisation mémoire
    processes.sort(key=lambda x: x['memory_info'].rss, reverse=True)
    
    for i, proc in enumerate(processes[:10]):
        memory_mb = proc['memory_info'].rss / (1024**2)
        print(f"   {i+1:2d}. {proc['name']:<20} PID:{proc['pid']:<8} {memory_mb:>8.1f} MB")

def suggest_memory_cleanup():
    """Suggère des actions pour libérer de la mémoire"""
    print("\n💡 Actions recommandées pour libérer de la mémoire:")
    print("1. 🗂️  Fermez les applications inutiles:")
    print("   - Navigateurs web (Chrome, Firefox, Edge)")
    print("   - Éditeurs de code (VS Code, PyCharm)")
    print("   - Applications de bureau (Office, Adobe)")
    print("   - Jeux ou autres applications lourdes")
    
    print("\n2. 🔄 Redémarrez les services:")
    print("   - Redémarrez Ollama: ollama serve")
    print("   - Redémarrez l'interface web")
    
    print("\n3. 🧹 Nettoyage système:")
    print("   - Videz la corbeille")
    print("   - Nettoyez les fichiers temporaires")
    print("   - Fermez les onglets inutiles du navigateur")
    
    print("\n4. ⚙️  Optimisations avancées:")
    print("   - Augmentez la mémoire virtuelle")
    print("   - Utilisez un modèle plus petit: ollama pull llama3.1:8b")
    print("   - Fermez les services Windows inutiles")

def cleanup_python_memory():
    """Nettoie la mémoire Python"""
    print("\n🧹 Nettoyage de la mémoire Python...")
    
    # Forcer le garbage collection
    collected = gc.collect()
    print(f"✅ Objets Python nettoyés: {collected}")
    
    # Afficher la mémoire après nettoyage
    memory_after = psutil.virtual_memory()
    print(f"📊 Mémoire après nettoyage: {memory_after.available / (1024**3):.1f} GB disponible")

def check_ollama_memory():
    """Vérifie l'utilisation mémoire d'Ollama"""
    print("\n🔍 Vérification d'Ollama...")
    
    ollama_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if 'ollama' in proc.info['name'].lower():
                memory_mb = proc.info['memory_info'].rss / (1024**2)
                ollama_processes.append((proc.info['pid'], memory_mb))
                print(f"   Ollama PID {proc.info['pid']}: {memory_mb:.1f} MB")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if ollama_processes:
        total_ollama_memory = sum(memory for _, memory in ollama_processes)
        print(f"📊 Total Ollama: {total_ollama_memory:.1f} MB")
        
        if total_ollama_memory > 1000:  # Plus de 1GB
            print("⚠️  Ollama utilise beaucoup de mémoire")
            print("💡 Redémarrez Ollama: ollama serve")
    else:
        print("ℹ️  Aucun processus Ollama trouvé")

def main():
    """Fonction principale"""
    print("🧬 Optimiseur de mémoire pour PGPR RAG")
    print("=" * 50)
    
    # Afficher l'état actuel
    available_memory = show_memory_usage()
    
    # Trouver les processus gourmands
    find_memory_hogs()
    
    # Vérifier Ollama
    check_ollama_memory()
    
    # Nettoyer la mémoire Python
    cleanup_python_memory()
    
    # Suggestions
    suggest_memory_cleanup()
    
    print(f"\n📋 Résumé:")
    print(f"   Mémoire disponible: {available_memory:.1f} GB")
    
    if available_memory < 2.0:
        print("❌ Mémoire très faible! Fermez des applications.")
    elif available_memory < 4.0:
        print("⚠️  Mémoire faible. Les paramètres optimisés sont nécessaires.")
    else:
        print("✅ Mémoire suffisante pour le modèle optimisé.")
    
    print("\n🚀 Prochaines étapes:")
    print("1. Fermez les applications inutiles")
    print("2. Redémarrez Ollama: ollama serve")
    print("3. Relancez l'interface web")
    print("4. Si le problème persiste, utilisez: ollama pull llama3.1:8b")

if __name__ == "__main__":
    main()
