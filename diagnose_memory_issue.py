#!/usr/bin/env python3
"""
Script pour diagnostiquer ce qui a changé et libérer le cache RAM
"""

import psutil
import subprocess
import os
import time

def check_ollama_memory_usage():
    """Vérifie l'utilisation mémoire d'Ollama"""
    print("🔍 Vérification de l'utilisation mémoire d'Ollama...")
    
    try:
        # Vérifier les processus Ollama
        ollama_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'create_time']):
            try:
                if 'ollama' in proc.info['name'].lower():
                    memory_mb = proc.info['memory_info'].rss / (1024**2)
                    create_time = time.ctime(proc.info['create_time'])
                    ollama_processes.append({
                        'pid': proc.info['pid'],
                        'memory_mb': memory_mb,
                        'created': create_time
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if ollama_processes:
            print("📊 Processus Ollama actifs:")
            for proc in ollama_processes:
                print(f"   PID {proc['pid']}: {proc['memory_mb']:.1f} MB (créé: {proc['created']})")
        else:
            print("ℹ️  Aucun processus Ollama actif")
            
        return ollama_processes
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {e}")
        return []

def clear_ollama_cache():
    """Nettoie le cache d'Ollama"""
    print("\n🧹 Nettoyage du cache Ollama...")
    
    try:
        # Arrêter tous les modèles chargés
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Vérification des modèles chargés")
        
        # Forcer le garbage collection d'Ollama
        print("🔄 Redémarrage d'Ollama pour libérer la mémoire...")
        
        # Arrêter Ollama
        try:
            subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], 
                          capture_output=True, timeout=10)
            print("✅ Ollama arrêté")
        except:
            print("ℹ️  Ollama n'était pas en cours d'exécution")
        
        # Attendre un peu
        time.sleep(2)
        
        # Redémarrer Ollama
        print("🚀 Redémarrage d'Ollama...")
        subprocess.Popen(["ollama", "serve"], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
        
        # Attendre qu'Ollama démarre
        time.sleep(3)
        
        print("✅ Ollama redémarré")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du nettoyage: {e}")
        return False

def check_system_changes():
    """Vérifie ce qui a pu changer dans le système"""
    print("\n🔍 Vérification des changements système...")
    
    # Vérifier la mémoire disponible
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    print(f"📊 Mémoire disponible: {available_gb:.1f} GB")
    
    # Vérifier les processus gourmands
    print("\n🔍 Top 5 des processus utilisant le plus de mémoire:")
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    processes.sort(key=lambda x: x['memory_info'].rss, reverse=True)
    
    for i, proc in enumerate(processes[:5]):
        memory_mb = proc['memory_info'].rss / (1024**2)
        print(f"   {i+1}. {proc['name']:<20} {memory_mb:>8.1f} MB")
    
    # Suggestions spécifiques
    print("\n💡 Actions pour libérer de la mémoire:")
    
    if available_gb < 2.0:
        print("❌ Mémoire très faible!")
        print("1. Fermez Edge/Chrome (sauvegarde ~800MB)")
        print("2. Fermez Cursor/VS Code (sauvegarde ~400MB)")
        print("3. Redémarrez l'ordinateur si nécessaire")
    elif available_gb < 4.0:
        print("⚠️  Mémoire faible")
        print("1. Fermez quelques onglets de navigateur")
        print("2. Fermez les applications inutiles")
    else:
        print("✅ Mémoire suffisante")
        print("1. Le problème pourrait être ailleurs")
        print("2. Vérifiez les paramètres Ollama")

def test_ollama_with_minimal_memory():
    """Teste Ollama avec des paramètres minimaux"""
    print("\n🧪 Test d'Ollama avec paramètres minimaux...")
    
    try:
        # Test simple avec Ollama
        result = subprocess.run([
            "ollama", "run", "llama3.1", 
            "--num_ctx", "512",
            "--num_thread", "1",
            "--num_gpu", "0"
        ], input="Hello", capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Test Ollama réussi avec paramètres minimaux")
            return True
        else:
            print(f"❌ Test Ollama échoué: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test Ollama timeout - modèle trop lourd")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

def main():
    """Fonction principale"""
    print("🧬 Diagnostic de problème mémoire")
    print("=" * 50)
    
    # Vérifier l'état actuel
    ollama_processes = check_ollama_memory_usage()
    
    # Vérifier les changements système
    check_system_changes()
    
    # Nettoyer le cache Ollama
    if clear_ollama_cache():
        print("\n✅ Cache Ollama nettoyé")
    
    # Tester avec paramètres minimaux
    if test_ollama_with_minimal_memory():
        print("\n🎉 Ollama fonctionne avec paramètres optimisés!")
        print("💡 Utilisez ces paramètres dans votre application:")
        print("   num_ctx: 512")
        print("   num_thread: 1") 
        print("   num_gpu: 0")
    else:
        print("\n❌ Ollama ne fonctionne toujours pas")
        print("💡 Solutions alternatives:")
        print("1. Redémarrez l'ordinateur")
        print("2. Utilisez un modèle plus petit: ollama pull llama3.1:8b")
        print("3. Fermez toutes les autres applications")

if __name__ == "__main__":
    main()
