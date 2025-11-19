#!/usr/bin/env python3
"""
Script pour ré-entraîner UNIQUEMENT les modèles ML (sans reconstruire le RAG)
Résout les problèmes de compatibilité sklearn rapidement
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
import shutil

def retrain_ml_models_only():
    """Ré-entraîne uniquement les modèles ML"""
    
    print("🔄 RETRAINING ML MODELS ONLY (Fast Mode)")
    print("=" * 50)
    
    # Check dataset
    print("\n📊 Checking dataset...")
    
    if not os.path.exists("./pgpr_images/train_labels.csv"):
        print("❌ train_labels.csv not found")
        return False
    
    if not os.path.exists("./pgpr_images/test_labels.csv"):
        print("❌ test_labels.csv not found")
        return False
    
    if not os.path.exists("./pgpr_images/images/"):
        print("❌ Images directory not found")
        return False
    
    # Load CSV files
    train_df = pd.read_csv("./pgpr_images/train_labels.csv")
    test_df = pd.read_csv("./pgpr_images/test_labels.csv")
    
    print(f"✅ Training set: {len(train_df)} images")
    print(f"✅ Test set: {len(test_df)} images")
    
    # Clear ONLY the ML models (keep RAG cache)
    print("\n🧹 Clearing old ML models only...")
    
    if os.path.exists("./ml_models/"):
        shutil.rmtree("./ml_models/")
        print("✅ Removed old ML models")
    
    # Remove metrics cache
    metrics_cache = "./rag_cache/model_metrics.pkl"
    if os.path.exists(metrics_cache):
        os.remove(metrics_cache)
        print("✅ Removed old metrics cache")
    
    # Create ML models directory
    os.makedirs("./ml_models/", exist_ok=True)
    
    print("\n🚀 Starting ML models retraining...")
    print("⏱️  This should take 2-5 minutes (much faster than full retrain)...")
    
    start_time = time.time()
    
    try:
        # Import ML components only
        from ml_model_builder import PGPRMLModelBuilder, create_ml_models
        from dataset_processor import CSVDatasetProcessor
        from image_processor import PGPRImageProcessor
        
        print("\n📚 Processing images for ML training...")
        
        # Initialize processors
        csv_processor = CSVDatasetProcessor(
            "./pgpr_images/images/",
            "./pgpr_images/train_labels.csv", 
            "./pgpr_images/test_labels.csv"
        )
        
        image_processor = PGPRImageProcessor()
        image_processor.csv_processor = csv_processor
        
        # Extract features from all images
        print("🔍 Extracting features from images...")
        features_dict = image_processor.process_csv_dataset(
            "./pgpr_images/images/",
            "./pgpr_images/train_labels.csv",
            "./pgpr_images/test_labels.csv"
        )
        
        print(f"✅ Extracted features from {len(features_dict)} images")
        
        # Create and train ML models
        print("\n🤖 Training ML models...")
        ml_builder = create_ml_models(features_dict, csv_processor)
        
        print("✅ ML models trained successfully!")
        
        # Evaluate models on test data
        print("\n📊 Evaluating models on test data...")
        
        # Get test data
        test_data = csv_processor.get_all_image_paths_with_labels("test")
        test_features = []
        test_labels = []
        
        for image_path, labels, split in test_data:
            if os.path.exists(image_path):
                try:
                    features = image_processor.extract_features(str(image_path))
                    test_features.append(features)
                    
                    label_vector = [labels[bacteria] for bacteria in ml_builder.bacteria_types]
                    test_labels.append(label_vector)
                except Exception as e:
                    print(f"Warning: Error processing {image_path}: {e}")
        
        if len(test_features) > 0:
            test_features = np.array(test_features)
            test_labels = np.array(test_labels)
            
            # Evaluate each model
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
            import pickle
            
            metrics = {}
            
            for model_name in ml_builder.models.keys():
                try:
                    predictions, probabilities = ml_builder.predict(test_features, model_name)
                    
                    accuracy = accuracy_score(test_labels, predictions)
                    f1 = f1_score(test_labels, predictions, average='weighted')
                    precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
                    recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
                    
                    report = classification_report(test_labels, predictions, 
                                                 target_names=ml_builder.bacteria_types, 
                                                 output_dict=True, zero_division=0)
                    
                    details = {}
                    for bacteria in ml_builder.bacteria_types:
                        if bacteria in report:
                            details[bacteria] = {
                                "precision": report[bacteria]['precision'],
                                "recall": report[bacteria]['recall'],
                                "f1": report[bacteria]['f1-score']
                            }
                        else:
                            details[bacteria] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                    
                    metrics[model_name] = {
                        "accuracy": accuracy,
                        "f1_score": f1,
                        "precision": precision,
                        "recall": recall,
                        "details": details
                    }
                    
                    print(f"  {model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
                    
                except Exception as e:
                    print(f"  Error evaluating {model_name}: {e}")
            
            # Save metrics to cache
            metrics_path = "./rag_cache/model_metrics.pkl"
            os.makedirs("./rag_cache/", exist_ok=True)
            with open(metrics_path, 'wb') as f:
                pickle.dump(metrics, f)
            
            print(f"✅ Model metrics saved to {metrics_path}")
            
            # Show best model
            if metrics:
                best_model = max(metrics.items(), key=lambda x: x[1]['f1_score'])
                print(f"🏆 Best model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.3f})")
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n🎉 ML MODELS RETRAINING COMPLETED!")
        print(f"⏱️  Total time: {training_time/60:.1f} minutes")
        print(f"✅ Models saved to ./ml_models/")
        print(f"✅ Metrics saved to ./rag_cache/model_metrics.pkl")
        
        print("\n🎯 Next steps:")
        print("1. Run: streamlit run web_chatbot_enhanced.py")
        print("2. The sklearn compatibility issue should be resolved!")
        print("3. Your RAG system cache is preserved (no need to rebuild)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during ML retraining: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check that all images exist in ./pgpr_images/images/")
        print("2. Verify CSV files have correct format")
        print("3. Check available disk space")
        
        return False

def main():
    """Fonction principale"""
    success = retrain_ml_models_only()
    if success:
        print("\n✅ ML models retraining completed successfully!")
        print("🚀 Your web interface should now work without sklearn errors!")
    else:
        print("\n❌ ML models retraining failed. Check the error messages above.")

if __name__ == "__main__":
    main()
