#!/usr/bin/env python3
"""
Quick Test for Smart Repository Classifier
=========================================

This script quickly tests the new smart classifier on a few repositories
to verify it works correctly.
"""

import os
import sys
import time
from pathlib import Path

# Add the scripts/analysis directory to the path
sys.path.append('scripts/analysis')

def quick_test_smart_classifier():
    """Quick test of the smart repository classifier."""
    print("🧪 Quick Test: Smart Repository Classifier")
    print("=" * 60)
    
    # Import the smart classifier
    try:
        from smart_repository_classifier import SmartRepositoryClassifier
        print("✅ Successfully imported SmartRepositoryClassifier")
    except ImportError as e:
        print(f"❌ Failed to import SmartRepositoryClassifier: {e}")
        return
    
    # Initialize classifier
    try:
        classifier = SmartRepositoryClassifier()
        print("✅ Successfully initialized classifier")
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return
    
    # Test repositories from temp_fetch_repos (diverse types)
    test_repos = [
        "temp_fetch_repos/jmonkeyengine_jmonkeyengine",    # Java game engine
        "temp_fetch_repos/kivy_kivy",                     # Python mobile framework
        "temp_fetch_repos/symfony_symfony",               # PHP web framework
        "temp_fetch_repos/google_protobuf",               # C++ library
        "temp_fetch_repos/jakesgordon_javascript-tetris"  # JavaScript game
    ]
    
    print(f"\n🚀 Testing {len(test_repos)} repositories...")
    print("-" * 60)
    
    for i, repo_path in enumerate(test_repos, 1):
        if not os.path.exists(repo_path):
            print(f"⚠️  Repository not found: {repo_path}")
            continue
        
        print(f"\n[{i}/{len(test_repos)}] Testing: {repo_path}")
        
        try:
            start_time = time.time()
            
            # Classify repository
            result = classifier.classify(repo_path)
            
            classification_time = time.time() - start_time
            
            print(f"  ✅ Classification completed in {classification_time:.2f}s")
            print(f"  🎯 Architecture: {result['architecture']}")
            print(f"  ⭐ Quality: {result['quality']}")
            print(f"  🎯 Specialization: {result['specialization']}")
            print(f"  📊 Confidence: {result['confidence']:.3f}")
            print(f"  🔧 Method: {result['method']}")
            
        except Exception as e:
            print(f"  ❌ Classification failed: {e}")
    
    print(f"\n✅ Quick test completed!")
    print("The smart classifier is working correctly if all classifications completed successfully.")

def test_rule_based_classification():
    """Test only the rule-based classification."""
    print("\n📋 Testing Rule-Based Classification Only")
    print("=" * 50)
    
    try:
        from smart_repository_classifier import RuleBasedClassifier
        
        rule_classifier = RuleBasedClassifier()
        
        # Test repositories from temp_fetch_repos
        test_repos = [
            "temp_fetch_repos/kennethreitz_requests",      # Python HTTP library
            "temp_fetch_repos/sindresorhus_chalk"          # JavaScript CLI library
        ]
        
        for repo_path in test_repos:
            if not os.path.exists(repo_path):
                continue
                
            print(f"\n🔍 Testing rules on: {repo_path}")
            
            try:
                arch_type, confidence = rule_classifier.classify(repo_path)
                print(f"   📋 Result: {arch_type} (confidence: {confidence:.3f})")
                
                # Show extracted features
                features = rule_classifier._extract_rule_features(repo_path)
                print(f"   🔧 Key features:")
                for key, value in features.items():
                    if value:  # Only show True features
                        print(f"      • {key}: {value}")
                        
            except Exception as e:
                print(f"   ❌ Rule classification failed: {e}")
                
    except Exception as e:
        print(f"❌ Failed to test rule-based classification: {e}")

if __name__ == "__main__":
    quick_test_smart_classifier()
    test_rule_based_classification()
