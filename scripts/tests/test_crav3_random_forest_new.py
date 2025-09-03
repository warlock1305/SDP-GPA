#!/usr/bin/env python3
"""
Quick Test for CRAV3 Random Forest NEW
======================================

This script quickly tests the new optimized model on a few repositories
from temp_fetch_repos to verify it works correctly.
"""

import os
import sys
import time
from pathlib import Path

# Add the scripts/analysis directory to the path
sys.path.append('scripts/analysis')

def quick_test_new_model():
    """Quick test of the new CRAV3 Random Forest model."""
    print("🧪 Quick Test: CRAV3 Random Forest NEW")
    print("=" * 60)
    
    # Check if model exists
    model_path = "ml_models/crav3_random_forest_new.joblib"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using crav3_random_forest_new.py")
        return
    
    print("✅ Found trained model")
    
    # Import the model class
    try:
        from crav3_random_forest_new import CRAV3RandomForestNew
        print("✅ Successfully imported CRAV3RandomForestNew")
    except ImportError as e:
        print(f"❌ Failed to import CRAV3RandomForestNew: {e}")
        return
    
    # Initialize model
    try:
        model = CRAV3RandomForestNew()
        print("✅ Successfully initialized model")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return
    
    # Test repositories (simple, medium, complex)
    test_repos = [
        "asweigart_pyscreeze",      # Simple Python library
        "grantjenks_free-python-games",  # Medium complexity games
        "google_protobuf"           # Complex C++ library
    ]
    
    print(f"\n🚀 Testing {len(test_repos)} repositories...")
    print("-" * 60)
    
    for i, repo_name in enumerate(test_repos, 1):
        repo_path = f"temp_fetch_repos/{repo_name}"
        
        if not os.path.exists(repo_path):
            print(f"⚠️  Repository not found: {repo_path}")
            continue
        
        print(f"\n[{i}/{len(test_repos)}] Testing: {repo_name}")
        print(f"📍 Path: {repo_path}")
        
        try:
            start_time = time.time()
            
            # Make prediction
            prediction_result = model.predict(repo_path)
            
            prediction_time = time.time() - start_time
            
            # Extract results
            predictions = prediction_result.get('predictions', [])
            target_count = prediction_result.get('target_count', 0)
            
            print(f"  ✅ Prediction completed in {prediction_time:.2f}s")
            print(f"  🎯 Detected {target_count} targets")
            
            if predictions:
                print(f"  🏷️  Top predictions:")
                for j, pred in enumerate(predictions[:3], 1):  # Show top 3
                    print(f"     {j}. {pred['target']} ({pred['type']}, confidence: {pred['confidence']:.3f})")
            
        except Exception as e:
            print(f"  ❌ Prediction failed: {e}")
    
    print(f"\n✅ Quick test completed!")
    print("The new model is working correctly if all predictions completed successfully.")

if __name__ == "__main__":
    quick_test_new_model()