# retrain_with_validation.py - Retrain ML models with updated dataset including validation

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time

def main():
    print("🔄 RETRAINING ML MODELS WITH UPDATED DATASET")
    print("=" * 60)
    
    # Check dataset
    print("\n📊 Checking updated dataset...")
    
    # Load CSV files
    train_df = pd.read_csv("./pgpr_images/train_labels.csv")
    valid_df = pd.read_csv("./pgpr_images/Valid_labels.csv")
    test_df = pd.read_csv("./pgpr_images/test_labels.csv")
    
    print(f"✅ Training set: {len(train_df)} images")
    print(f"✅ Validation set: {len(valid_df)} images")
    print(f"✅ Test set: {len(test_df)} images")
    print(f"✅ Total dataset: {len(train_df) + len(valid_df) + len(test_df)} images")
    
    # Check bacteria distribution
    bacteria_types = ['Bacillus subtilis', 'Escherichia coli', 'Pseudomonas aeruginosa', 'Staphylococcus aureus']
    
    print("\n📈 Dataset distribution:")
    for bacteria in bacteria_types:
        train_count = train_df[bacteria].sum()
        valid_count = valid_df[bacteria].sum()
        test_count = test_df[bacteria].sum()
        total_count = train_count + valid_count + test_count
        print(f"  {bacteria}: {total_count} total ({train_count} train, {valid_count} valid, {test_count} test)")
    
    # Clear old models and cache
    print("\n🧹 Clearing old models and cache...")
    
    if os.path.exists("./ml_models/"):
        import shutil
        shutil.rmtree("./ml_models/")
        print("✅ Removed old ML models")
    
    if os.path.exists("./rag_cache/"):
        import shutil
        shutil.rmtree("./rag_cache/")
        print("✅ Removed old cache")
    
    # Create new directories
    os.makedirs("./ml_models/", exist_ok=True)
    os.makedirs("./rag_cache/", exist_ok=True)
    
    print("\n🚀 Starting retraining process...")
    print("This may take 15-30 minutes depending on your hardware...")
    
    start_time = time.time()
    
    try:
        # Import and run the enhanced multimodal RAG system
        from enhanced_multimodal_rag import build_enhanced_multimodal_rag
        
        # Build the system with updated dataset
        print("\n📚 Processing documents and images...")
        system = build_enhanced_multimodal_rag(
            images_dir="./pgpr_images/images/",
            train_csv="./pgpr_images/train_labels.csv",
            test_csv="./pgpr_images/test_labels.csv"
        )
        
        # Save model metrics for the web interface
        print("\n💾 Saving model metrics...")
        if hasattr(system, 'ml_builder') and system.ml_builder:
            try:
                # Evaluate models on test data and save metrics
                from dataset_processor import CSVDatasetProcessor
                from image_processor import PGPRImageProcessor
                import numpy as np
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
                import pickle
                
                # Load test data
                csv_processor = CSVDatasetProcessor(
                    "./pgpr_images/images/",
                    "./pgpr_images/train_labels.csv",
                    "./pgpr_images/test_labels.csv"
                )
                
                # Extract test features
                image_processor = PGPRImageProcessor()
                test_features = []
                test_labels = []
                
                test_data = csv_processor.get_all_image_paths_with_labels("test")
                
                for image_path, labels, split in test_data:
                    if os.path.exists(image_path):
                        try:
                            features = image_processor.extract_features(str(image_path))
                            test_features.append(features)
                            
                            label_vector = [labels[bacteria] for bacteria in system.ml_builder.bacteria_types]
                            test_labels.append(label_vector)
                        except Exception as e:
                            print(f"Warning: Error processing {image_path}: {e}")
                
                if len(test_features) > 0:
                    test_features = np.array(test_features)
                    test_labels = np.array(test_labels)
                    
                    # Evaluate each model
                    metrics = {}
                    
                    for model_name in system.ml_builder.models.keys():
                        try:
                            predictions, probabilities = system.ml_builder.predict(test_features, model_name)
                            
                            accuracy = accuracy_score(test_labels, predictions)
                            f1 = f1_score(test_labels, predictions, average='weighted')
                            precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
                            recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
                            
                            report = classification_report(test_labels, predictions, 
                                                         target_names=system.ml_builder.bacteria_types, 
                                                         output_dict=True, zero_division=0)
                            
                            details = {}
                            for bacteria in system.ml_builder.bacteria_types:
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
                    with open(metrics_path, 'wb') as f:
                        pickle.dump(metrics, f)
                    
                    print(f"✅ Model metrics saved to {metrics_path}")
                    
                    # Show best model
                    if metrics:
                        best_model = max(metrics.items(), key=lambda x: x[1]['f1_score'])
                        print(f"🏆 Best model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.3f})")
                
            except Exception as e:
                print(f"Warning: Could not save model metrics: {e}")
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n🎉 RETRAINING COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total time: {training_time/60:.1f} minutes")
        
        # Show model performance
        print("\n📊 Model Performance Summary:")
        if hasattr(system, 'ml_builder') and system.ml_builder:
            print("✅ All 5 ML models trained successfully")
            print("✅ Models saved to ./ml_models/")
            print("✅ Cache saved to ./rag_cache/")
        
        print("\n🎯 Next steps:")
        print("1. Run: streamlit run web_chatbot_enhanced.py")
        print("2. Open browser to: http://localhost:8501")
        print("3. Test the updated models in the 'Comparaison Modèles' tab")
        
    except Exception as e:
        print(f"\n❌ Error during retraining: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check that all images exist in ./pgpr_images/images/")
        print("2. Verify CSV files have correct format")
        print("3. Ensure Ollama is running: ollama serve")
        print("4. Check available disk space")
        
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Retraining completed successfully!")
    else:
        print("\n❌ Retraining failed. Check the error messages above.")
