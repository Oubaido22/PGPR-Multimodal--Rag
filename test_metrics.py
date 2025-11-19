# test_metrics.py - Test script to verify metrics are working

import os
import sys
import pickle

def test_metrics_loading():
    """Test if metrics are being loaded correctly"""
    print("🧪 Testing Metrics Loading...")
    
    # Check if metrics cache exists
    metrics_cache_path = "./rag_cache/model_metrics.pkl"
    
    if os.path.exists(metrics_cache_path):
        print("✅ Metrics cache found")
        try:
            with open(metrics_cache_path, 'rb') as f:
                metrics = pickle.load(f)
            
            print(f"✅ Metrics loaded successfully")
            print(f"📊 Found {len(metrics)} models")
            
            for model_name, model_metrics in metrics.items():
                print(f"\n🔍 Model: {model_name}")
                print(f"  Accuracy: {model_metrics.get('accuracy', 'N/A')}")
                print(f"  F1-Score: {model_metrics.get('f1_score', 'N/A')}")
                
                if 'details' in model_metrics:
                    print(f"  Details for {len(model_metrics['details'])} bacteria:")
                    for bacteria, details in model_metrics['details'].items():
                        print(f"    {bacteria}: P={details.get('precision', 0):.3f}, R={details.get('recall', 0):.3f}, F1={details.get('f1', 0):.3f}")
                else:
                    print("  ❌ No details found")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading metrics: {e}")
            return False
    else:
        print("❌ No metrics cache found")
        print("💡 Run retraining first: python retrain_with_validation.py")
        return False

def test_model_loading():
    """Test if models can be loaded"""
    print("\n🤖 Testing Model Loading...")
    
    try:
        from ml_model_builder import PGPRMLModelBuilder
        
        ml_builder = PGPRMLModelBuilder()
        ml_builder.load_models()
        
        print(f"✅ Models loaded: {list(ml_builder.models.keys())}")
        print(f"✅ Bacteria types: {ml_builder.bacteria_types}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def main():
    print("🔬 PGPR Metrics Test")
    print("=" * 30)
    
    # Test model loading
    models_ok = test_model_loading()
    
    # Test metrics loading
    metrics_ok = test_metrics_loading()
    
    print("\n📋 Summary:")
    print(f"Models: {'✅' if models_ok else '❌'}")
    print(f"Metrics: {'✅' if metrics_ok else '❌'}")
    
    if not metrics_ok:
        print("\n💡 To fix metrics issue:")
        print("1. Run: python retrain_with_validation.py")
        print("2. Wait for completion")
        print("3. Check: python test_metrics.py")
    
    return models_ok and metrics_ok

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the output above.")
