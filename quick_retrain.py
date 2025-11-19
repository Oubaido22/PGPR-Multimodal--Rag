#!/usr/bin/env python3
"""
Script rapide pour ré-entraîner les modèles ML avec la version actuelle de sklearn
"""

import os
import shutil
from pathlib import Path

def quick_retrain():
    """Ré-entraîne rapidement les modèles ML"""
    
    print("🔄 Ré-entraînement rapide des modèles ML...")
    
    # Vérifier si les données existent
    if not os.path.exists("./pgpr_images/train_labels.csv"):
        print("❌ Fichier train_labels.csv non trouvé")
        return False
    
    if not os.path.exists("./pgpr_images/test_labels.csv"):
        print("❌ Fichier test_labels.csv non trouvé")
        return False
    
    if not os.path.exists("./pgpr_images/images/"):
        print("❌ Dossier images non trouvé")
        return False
    
    # Sauvegarder les anciens modèles
    models_dir = Path("./ml_models")
    backup_dir = Path("./ml_models_backup")
    
    if models_dir.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(models_dir, backup_dir)
        print(f"📦 Sauvegarde créée: {backup_dir}")
    
    # Supprimer les anciens modèles
    if models_dir.exists():
        shutil.rmtree(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        print("🗑️ Anciens modèles supprimés")
    
    # Supprimer le cache des métriques
    metrics_cache = Path("./rag_cache/model_metrics.pkl")
    if metrics_cache.exists():
        metrics_cache.unlink()
        print("🗑️ Cache des métriques supprimé")
    
    print("✅ Nettoyage terminé!")
    print("\n📋 Prochaines étapes:")
    print("1. Exécutez: python retrain_ml_only.py")
    print("2. Relancez l'application web")
    print("3. Les modèles seront compatibles avec votre version de sklearn")
    print("\n💡 Note: retrain_ml_only.py est 5-10x plus rapide que retrain_with_validation.py")
    
    return True

def main():
    """Fonction principale"""
    print("🧬 Ré-entraînement rapide des modèles PGPR")
    print("=" * 50)
    
    if quick_retrain():
        print("\n🎉 Préparation terminée!")
        print("💡 Exécutez maintenant: python retrain_with_validation.py")
    else:
        print("\n❌ Erreur lors de la préparation")

if __name__ == "__main__":
    main()
