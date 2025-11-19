#!/usr/bin/env python3
"""
Script pour libérer le cache RAM et optimiser la mémoire
"""

import psutil
import subprocess
import os
import gc
import time

def show_current_memory():
    """Affiche l'état actuel de la mémoire"""
    memory = psutil.virtual_memory()
    print(f"📊 Mémoire actuelle:")
    print(f"   Total: {memory.total / (1024**3):.1f} GB")
    print(f"   Utilisée: {memory.used / (1024**3):.1f} GB") 
    print(f"   Disponible: {memory.available / (1024**3):.1f} GB")
    print(f"   Pourcentage: {memory.percent:.1f}%")
    return memory.available / (1024**3)

def find_memory_hogs():
    """Trouve les processus qui consomment le plus de mémoire"""
    print("\n🔍 Top processus consommateurs de mémoire:")
    
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Trier par utilisation mémoire
    processes.sort(key=lambda x: x['memory_info'].rss, reverse=True)
    
    total_memory = 0
    for i, proc in enumerate(processes[:10]):
        memory_mb = proc['memory_info'].rss / (1024**2)
        total_memory += memory_mb
        print(f"   {i+1:2d}. {proc['name']:<20} {memory_mb:>8.1f} MB")
    
    print(f"\n📊 Total top 10: {total_memory/1024:.1f} GB")
    return processes[:10]

def suggest_quick_fixes():
    """Suggère des solutions rapides"""
    print("\n💡 Solutions rapides pour libérer de la mémoire:")
    
    print("1. 🌐 Fermez les onglets de navigateur inutiles:")
    print("   - Edge: ~800MB → Fermez les onglets")
    print("   - Chrome: ~600MB → Fermez les onglets")
    
    print("\n2. 💻 Fermez les applications lourdes:")
    print("   - Cursor/VS Code: ~400MB → Fermez les fenêtres inutiles")
    print("   - Applications Office: ~200MB → Sauvegardez et fermez")
    
    print("\n3. 🔄 Redémarrez les services:")
    print("   - Redémarrez Ollama: ollama serve")
    print("   - Redémarrez l'interface web")
    
    print("\n4. 🧹 Nettoyage système:")
    print("   - Videz la corbeille")
    print("   - Fermez les applications en arrière-plan")

def clear_python_memory():
    """Nettoie la mémoire Python"""
    print("\n🧹 Nettoyage de la mémoire Python...")
    
    # Forcer le garbage collection
    collected = gc.collect()
    print(f"✅ Objets Python nettoyés: {collected}")
    
    # Afficher la mémoire après nettoyage
    memory_after = psutil.virtual_memory()
    print(f"📊 Mémoire après nettoyage: {memory_after.available / (1024**3):.1f} GB")

def restart_ollama_clean():
    """Redémarre Ollama proprement"""
    print("\n🔄 Redémarrage propre d'Ollama...")
    
    try:
        # Arrêter Ollama
        print("⏹️  Arrêt d'Ollama...")
        subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], 
                      capture_output=True, timeout=10)
        time.sleep(2)
        
        # Redémarrer Ollama
        print("🚀 Redémarrage d'Ollama...")
        subprocess.Popen(["ollama", "serve"], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
        time.sleep(3)
        
        print("✅ Ollama redémarré proprement")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du redémarrage: {e}")
        return False

def test_memory_after_cleanup():
    """Teste la mémoire après nettoyage"""
    print("\n🧪 Test de la mémoire après nettoyage...")
    
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    print(f"📊 Mémoire disponible: {available_gb:.1f} GB")
    
    if available_gb >= 5.0:
        print("✅ Mémoire suffisante pour le modèle complet!")
        return True
    elif available_gb >= 3.0:
        print("⚠️  Mémoire limitée - utilisez des paramètres optimisés")
        return False
    else:
        print("❌ Mémoire insuffisante - fermez plus d'applications")
        return False

def main():
    """Fonction principale"""
    print("🧬 Libérateur de cache RAM")
    print("=" * 50)
    
    # État initial
    print("📊 État initial:")
    initial_memory = show_current_memory()
    
    # Identifier les processus gourmands
    memory_hogs = find_memory_hogs()
    
    # Nettoyer la mémoire Python
    clear_python_memory()
    
    # Redémarrer Ollama
    restart_ollama_clean()
    
    # Tester la mémoire après nettoyage
    final_memory = test_memory_after_cleanup()
    
    # Suggestions
    suggest_quick_fixes()
    
    print(f"\n📋 Résumé:")
    print(f"   Mémoire initiale: {initial_memory:.1f} GB")
    
    memory_after = psutil.virtual_memory()
    print(f"   Mémoire finale: {memory_after.available / (1024**3):.1f} GB")
    
    if final_memory:
        print("\n🎉 Mémoire suffisante! Vous pouvez maintenant:")
        print("1. Lancer l'interface web")
        print("2. Tester le chat")
    else:
        print("\n⚠️  Mémoire encore insuffisante. Actions recommandées:")
        print("1. Fermez Edge/Chrome (sauvegarde ~1GB)")
        print("2. Fermez Cursor (sauvegarde ~400MB)")
        print("3. Redémarrez l'ordinateur si nécessaire")
        print("4. Utilisez: ollama pull llama3.1:8b (modèle plus petit)")

if __name__ == "__main__":
    main()
