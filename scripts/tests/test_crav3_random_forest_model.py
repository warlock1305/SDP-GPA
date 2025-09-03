#!/usr/bin/env python3
"""
Test CRAV3 Random Forest Model on temp_fetch_repos
==================================================

This script tests the trained CRAV3 Random Forest model on repositories
from temp_fetch_repos and evaluates its predictions and performance.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add the scripts/analysis directory to the path
sys.path.append('scripts/analysis')

def load_expected_patterns():
    """Load expected patterns for different repository types."""
    return {
        # Python libraries
        'asweigart_pyscreeze': {
            'expected_patterns': ['library', 'cli_tool'],
            'expected_languages': ['python'],
            'expected_experience': 'intermediate',
            'expected_specialization': 'backend_specialist'
        },
        'asweigart_pyperclip': {
            'expected_patterns': ['library', 'cli_tool'],
            'expected_languages': ['python'],
            'expected_experience': 'intermediate',
            'expected_specialization': 'backend_specialist'
        },
        'asweigart_pyautogui': {
            'expected_patterns': ['library', 'cli_tool'],
            'expected_languages': ['python'],
            'expected_experience': 'intermediate',
            'expected_specialization': 'backend_specialist'
        },
        'kennethreitz_requests': {
            'expected_patterns': ['library', 'web_application'],
            'expected_languages': ['python'],
            'expected_experience': 'senior',
            'expected_specialization': 'backend_specialist'
        },
        
        # JavaScript libraries
        'sindresorhus_chalk': {
            'expected_patterns': ['library'],
            'expected_languages': ['javascript'],
            'expected_experience': 'senior',
            'expected_specialization': 'frontend_specialist'
        },
        'sindresorhus_update-notifier': {
            'expected_patterns': ['library', 'cli_tool'],
            'expected_languages': ['javascript'],
            'expected_experience': 'senior',
            'expected_specialization': 'frontend_specialist'
        },
        
        # Game development
        'jakesgordon_javascript-pong': {
            'expected_patterns': ['game_development'],
            'expected_languages': ['javascript'],
            'expected_experience': 'intermediate',
            'expected_specialization': 'game_developer'
        },
        'jakesgordon_javascript-breakout': {
            'expected_patterns': ['game_development'],
            'expected_languages': ['javascript'],
            'expected_experience': 'intermediate',
            'expected_specialization': 'game_developer'
        },
        'jakesgordon_javascript-tetris': {
            'expected_patterns': ['game_development'],
            'expected_languages': ['javascript'],
            'expected_experience': 'intermediate',
            'expected_specialization': 'game_developer'
        },
        'grantjenks_free-python-games': {
            'expected_patterns': ['game_development', 'educational'],
            'expected_languages': ['python'],
            'expected_experience': 'intermediate',
            'expected_specialization': 'game_developer'
        },
        
        # Web frameworks
        'zendframework_zendframework': {
            'expected_patterns': ['web_application', 'library'],
            'expected_languages': ['php'],
            'expected_experience': 'senior',
            'expected_specialization': 'backend_specialist'
        },
        'cakephp_cakephp': {
            'expected_patterns': ['web_application', 'library'],
            'expected_languages': ['php'],
            'expected_experience': 'senior',
            'expected_specialization': 'backend_specialist'
        },
        'symfony_symfony': {
            'expected_patterns': ['web_application', 'library'],
            'expected_languages': ['php'],
            'expected_experience': 'senior',
            'expected_specialization': 'backend_specialist'
        },
        
        # Mobile development
        'kivy_kivy': {
            'expected_patterns': ['mobile_app', 'library'],
            'expected_languages': ['python'],
            'expected_experience': 'senior',
            'expected_specialization': 'mobile_developer'
        },
        'kivy_kivy-ios': {
            'expected_patterns': ['mobile_app'],
            'expected_languages': ['python'],
            'expected_experience': 'senior',
            'expected_specialization': 'mobile_developer'
        },
        
        # C++ libraries
        'google_protobuf': {
            'expected_patterns': ['library'],
            'expected_languages': ['cpp'],
            'expected_experience': 'senior',
            'expected_specialization': 'backend_specialist'
        },
        'google_benchmark': {
            'expected_patterns': ['library'],
            'expected_languages': ['cpp'],
            'expected_experience': 'senior',
            'expected_specialization': 'backend_specialist'
        },
        'fmtlib_fmt': {
            'expected_patterns': ['library'],
            'expected_languages': ['cpp'],
            'expected_experience': 'senior',
            'expected_specialization': 'backend_specialist'
        },
        
        # Java libraries
        'square_okhttp': {
            'expected_patterns': ['library'],
            'expected_languages': ['java'],
            'expected_experience': 'senior',
            'expected_specialization': 'backend_specialist'
        },
        'square_retrofit': {
            'expected_patterns': ['library'],
            'expected_languages': ['java'],
            'expected_experience': 'senior',
            'expected_specialization': 'backend_specialist'
        }
    }

def calculate_prediction_accuracy(predictions: List[Dict], expected_patterns: Dict, repo_name: str) -> Dict[str, float]:
    """Calculate accuracy metrics for model predictions."""
    if repo_name not in expected_patterns:
        return {
            'pattern_accuracy': 0.0,
            'experience_accuracy': 0.0,
            'specialization_accuracy': 0.0,
            'overall_accuracy': 0.0,
            'confidence_score': 0.0
        }
    
    expected = expected_patterns[repo_name]
    
    # Extract predicted targets
    predicted_targets = [pred['target'] for pred in predictions]
    predicted_confidences = [pred['confidence'] for pred in predictions]
    
    # Pattern accuracy
    expected_patterns_set = set(expected.get('expected_patterns', []))
    predicted_patterns = set([target for target in predicted_targets if target in [
        'web_application', 'data_science', 'cli_tool', 'mobile_app', 
        'game_development', 'library', 'educational', 'microservices',
        'monolithic', 'api_project', 'testing_project', 'documentation_project'
    ]])
    
    if expected_patterns_set:
        pattern_intersection = predicted_patterns.intersection(expected_patterns_set)
        pattern_accuracy = len(pattern_intersection) / len(expected_patterns_set)
    else:
        pattern_accuracy = 0.0
    
    # Experience level accuracy
    expected_experience = expected.get('expected_experience', '').lower()
    predicted_experience = None
    for target in predicted_targets:
        if target in ['junior', 'intermediate', 'senior']:
            predicted_experience = target
            break
    
    if predicted_experience and expected_experience:
        if predicted_experience == expected_experience:
            experience_accuracy = 1.0
        elif (predicted_experience in ['senior', 'expert'] and expected_experience in ['senior', 'expert']) or \
             (predicted_experience in ['intermediate', 'advanced'] and expected_experience in ['intermediate', 'advanced']) or \
             (predicted_experience in ['junior', 'beginner'] and expected_experience in ['junior', 'beginner']):
            experience_accuracy = 0.5
        else:
            experience_accuracy = 0.0
    else:
        experience_accuracy = 0.0
    
    # Specialization accuracy
    expected_specialization = expected.get('expected_specialization', '').lower()
    predicted_specialization = None
    for target in predicted_targets:
        if target in [
            'frontend_specialist', 'backend_specialist', 'data_scientist',
            'devops_specialist', 'mobile_developer', 'game_developer'
        ]:
            predicted_specialization = target
            break
    
    if predicted_specialization and expected_specialization:
        if predicted_specialization == expected_specialization:
            specialization_accuracy = 1.0
        else:
            specialization_accuracy = 0.0
    else:
        specialization_accuracy = 0.0
    
    # Overall accuracy (weighted average)
    weights = [0.5, 0.3, 0.2]  # pattern, experience, specialization
    values = [pattern_accuracy, experience_accuracy, specialization_accuracy]
    overall_accuracy = sum(w * v for w, v in zip(weights, values))
    
    # Average confidence
    confidence_score = np.mean(predicted_confidences) if predicted_confidences else 0.0
    
    return {
        'pattern_accuracy': pattern_accuracy,
        'experience_accuracy': experience_accuracy,
        'specialization_accuracy': specialization_accuracy,
        'overall_accuracy': overall_accuracy,
        'confidence_score': confidence_score
    }

def test_crav3_random_forest_model():
    """Test the trained CRAV3 Random Forest model on temp_fetch_repos."""
    print("🧪 Testing CRAV3 Random Forest Model on temp_fetch_repos")
    print("=" * 80)
    
    # Check if model exists
    model_path = "ml_models/crav3_random_forest.joblib"
    scaler_path = "ml_models/crav3_random_forest_scaler.joblib"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using crav3_random_forest.py")
        return
    
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler not found at {scaler_path}")
        print("Please train the model first using crav3_random_forest.py")
        return
    
    print("✅ Found trained model and scaler")
    
    # Import the model class
    try:
        from crav3_random_forest import CRAV3RandomForest
        print("✅ Successfully imported CRAV3RandomForest")
    except ImportError as e:
        print(f"❌ Failed to import CRAV3RandomForest: {e}")
        return
    
    # Load expected patterns
    expected_patterns = load_expected_patterns()
    print(f"📋 Loaded expected patterns for {len(expected_patterns)} repositories")
    
    # Initialize model
    try:
        model = CRAV3RandomForest()
        print("✅ Successfully initialized CRAV3RandomForest")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return
    
    # Get all repositories from temp_fetch_repos
    temp_fetch_path = "temp_fetch_repos"
    all_repos = []
    
    if os.path.exists(temp_fetch_path):
        for item in os.listdir(temp_fetch_path):
            item_path = os.path.join(temp_fetch_path, item)
            if os.path.isdir(item_path):
                all_repos.append(item)
    
    print(f"📁 Found {len(all_repos)} total repositories in temp_fetch_repos")
    
    # Test repositories
    test_repos = all_repos[:20]  # Limit to first 20 for testing
    results = []
    
    print(f"\n🚀 Testing {len(test_repos)} repositories...")
    print("-" * 80)
    
    for i, repo_name in enumerate(test_repos, 1):
        repo_path = f"temp_fetch_repos/{repo_name}"
        
        if not os.path.exists(repo_path):
            print(f"⚠️  Repository not found: {repo_path}")
            continue
        
        print(f"\n[{i:2d}/{len(test_repos)}] Testing: {repo_name}")
        print(f"📍 Path: {repo_path}")
        
        try:
            start_time = time.time()
            
            # Make prediction using the trained model
            prediction_result = model.predict(repo_path)
            
            prediction_time = time.time() - start_time
            
            # Extract prediction results
            predictions = prediction_result.get('predictions', [])
            target_count = prediction_result.get('target_count', 0)
            feature_vector_shape = prediction_result.get('feature_vector_shape', (0,))
            
            # Calculate accuracy metrics (only for repos with expected patterns)
            if repo_name in expected_patterns:
                accuracy_metrics = calculate_prediction_accuracy(predictions, expected_patterns, repo_name)
            else:
                # For repos without expected patterns, create basic metrics
                accuracy_metrics = {
                    'pattern_accuracy': 0.0,
                    'experience_accuracy': 0.0,
                    'specialization_accuracy': 0.0,
                    'overall_accuracy': 0.0,
                    'confidence_score': np.mean([pred['confidence'] for pred in predictions]) if predictions else 0.0
                }
            
            # Store results
            result_entry = {
                'repository': repo_name,
                'predictions': predictions,
                'target_count': target_count,
                'feature_vector_shape': feature_vector_shape,
                'expected_patterns': expected_patterns.get(repo_name, {}),
                'accuracy_metrics': accuracy_metrics,
                'prediction_time': prediction_time,
                'success': True
            }
            
            results.append(result_entry)
            
            # Print results
            print(f"  ✅ Prediction completed in {prediction_time:.2f}s")
            print(f"  🎯 Detected {target_count} targets")
            print(f"  📊 Feature vector: {feature_vector_shape}")
            
            if predictions:
                print(f"  🏷️  Top predictions:")
                for j, pred in enumerate(predictions[:5], 1):  # Show top 5
                    print(f"     {j}. {pred['target']} ({pred['type']}, confidence: {pred['confidence']:.3f})")
            
            if repo_name in expected_patterns:
                print(f"  📈 Accuracy: {accuracy_metrics['overall_accuracy']:.3f} ({accuracy_metrics['overall_accuracy']*100:.1f}%)")
                print(f"     • Pattern: {accuracy_metrics['pattern_accuracy']:.3f}")
                print(f"     • Experience: {accuracy_metrics['experience_accuracy']:.3f}")
                print(f"     • Specialization: {accuracy_metrics['specialization_accuracy']:.3f}")
                print(f"     • Confidence: {accuracy_metrics['confidence_score']:.3f}")
            else:
                print(f"  📊 No expected patterns for comparison")
            
        except Exception as e:
            print(f"  ❌ Prediction failed: {e}")
            results.append({
                'repository': repo_name,
                'success': False,
                'error': str(e)
            })
    
    # Calculate overall statistics
    print(f"\n📊 OVERALL PREDICTION STATISTICS")
    print("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        # Calculate average accuracies
        avg_pattern_accuracy = np.mean([r['accuracy_metrics']['pattern_accuracy'] for r in successful_results])
        avg_experience_accuracy = np.mean([r['accuracy_metrics']['experience_accuracy'] for r in successful_results])
        avg_specialization_accuracy = np.mean([r['accuracy_metrics']['specialization_accuracy'] for r in successful_results])
        avg_overall_accuracy = np.mean([r['accuracy_metrics']['overall_accuracy'] for r in successful_results])
        avg_confidence_score = np.mean([r['accuracy_metrics']['confidence_score'] for r in successful_results])
        
        # Calculate average prediction time
        avg_prediction_time = np.mean([r['prediction_time'] for r in successful_results])
        
        # Calculate average target count
        avg_target_count = np.mean([r['target_count'] for r in successful_results])
        
        print(f"📈 Success Rate: {len(successful_results)}/{len(test_repos)} ({len(successful_results)/len(test_repos)*100:.1f}%)")
        print(f"⏱️  Average Prediction Time: {avg_prediction_time:.2f}s")
        print(f"🎯 Average Targets Detected: {avg_target_count:.1f}")
        print(f"\n🎯 Accuracy Metrics:")
        print(f"  Pattern Detection:     {avg_pattern_accuracy:.3f} ({avg_pattern_accuracy*100:.1f}%)")
        print(f"  Experience Level:      {avg_experience_accuracy:.3f} ({avg_experience_accuracy*100:.1f}%)")
        print(f"  Specialization:        {avg_specialization_accuracy:.3f} ({avg_specialization_accuracy*100:.1f}%)")
        print(f"  Overall Accuracy:      {avg_overall_accuracy:.3f} ({avg_overall_accuracy*100:.1f}%)")
        print(f"  Average Confidence:    {avg_confidence_score:.3f} ({avg_confidence_score*100:.1f}%)")
        
        # Category-specific accuracy analysis
        print(f"\n📊 CATEGORY-SPECIFIC ACCURACY ANALYSIS")
        print("-" * 80)
        
        categories = ['library', 'web_application', 'cli_tool', 'game_development', 'mobile_app', 'data_science', 'educational']
        category_accuracies = {}
        
        for category in categories:
            category_results = []
            for result in successful_results:
                if result['repository'] in expected_patterns:
                    expected = expected_patterns[result['repository']]
                    expected_in_category = category in expected.get('expected_patterns', [])
                    predicted_in_category = any(pred['target'] == category for pred in result['predictions'])
                    
                    if expected_in_category and predicted_in_category:
                        category_results.append(1.0)  # True positive
                    elif expected_in_category and not predicted_in_category:
                        category_results.append(0.0)  # False negative
                    elif not expected_in_category and predicted_in_category:
                        category_results.append(0.0)  # False positive
                    else:
                        category_results.append(1.0)  # True negative
            
            if category_results:
                category_accuracies[category] = np.mean(category_results)
                print(f"  {category:20} | Accuracy: {category_accuracies[category]:.3f} ({category_accuracies[category]*100:.1f}%)")
        
        # Target type analysis
        print(f"\n🎯 TARGET TYPE ANALYSIS")
        print("-" * 80)
        
        target_types = {}
        for result in successful_results:
            for pred in result['predictions']:
                pred_type = pred['type']
                if pred_type not in target_types:
                    target_types[pred_type] = []
                target_types[pred_type].append(pred['confidence'])
        
        for target_type, confidences in target_types.items():
            avg_conf = np.mean(confidences)
            count = len(confidences)
            print(f"  {target_type:25} | Count: {count:3d} | Avg Confidence: {avg_conf:.3f}")
        
        # Detailed breakdown
        print(f"\n📋 DETAILED RESULTS BREAKDOWN")
        print("-" * 80)
        for result in successful_results:
            repo_name = result['repository']
            accuracy = result['accuracy_metrics']['overall_accuracy']
            predictions = result['predictions']
            target_count = result['target_count']
            
            print(f"  {repo_name:30} | Accuracy: {accuracy:.3f} | Targets: {target_count}")
            if predictions:
                top_preds = [f"{pred['target']}({pred['confidence']:.2f})" for pred in predictions[:3]]
                print(f"  {'':30} | Top: {', '.join(top_preds)}")
        
        # Save detailed results
        output_file = "crav3_random_forest_test_results.json"
        with open(output_file, "w") as f:
            json.dump({
                'summary': {
                    'total_repositories': len(test_repos),
                    'successful_predictions': len(successful_results),
                    'success_rate': len(successful_results)/len(test_repos),
                    'average_prediction_time': avg_prediction_time,
                    'average_target_count': avg_target_count,
                    'average_accuracies': {
                        'pattern_detection': avg_pattern_accuracy,
                        'experience_level': avg_experience_accuracy,
                        'specialization': avg_specialization_accuracy,
                        'overall': avg_overall_accuracy
                    },
                    'average_confidence': avg_confidence_score,
                    'category_accuracies': category_accuracies,
                    'target_type_analysis': target_types
                },
                'detailed_results': results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Final assessment
        print(f"\n🏆 FINAL ASSESSMENT")
        print("=" * 80)
        
        if avg_overall_accuracy >= 0.8:
            print("🎉 EXCELLENT: CRAV3 Random Forest shows high accuracy across all metrics!")
        elif avg_overall_accuracy >= 0.6:
            print("✅ GOOD: CRAV3 Random Forest performs well with room for improvement")
        elif avg_overall_accuracy >= 0.4:
            print("⚠️  FAIR: CRAV3 Random Forest needs improvements in several areas")
        else:
            print("❌ POOR: CRAV3 Random Forest requires significant improvements")
        
        print(f"Overall Accuracy Score: {avg_overall_accuracy:.3f} ({avg_overall_accuracy*100:.1f}%)")
        
        # Performance analysis
        print(f"\n⚡ PERFORMANCE ANALYSIS")
        print("-" * 80)
        prediction_times = [r['prediction_time'] for r in successful_results]
        if prediction_times:
            print(f"  Fastest Prediction: {min(prediction_times):.2f}s")
            print(f"  Slowest Prediction: {max(prediction_times):.2f}s")
            print(f"  Median Prediction Time: {np.median(prediction_times):.2f}s")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if avg_overall_accuracy < 0.7:
            print(f"  ⚠️  Overall accuracy is below 70%. Consider:")
            print(f"     - Retraining with more diverse data")
            print(f"     - Adjusting hyperparameters")
            print(f"     - Feature engineering improvements")
        
        if avg_confidence_score < 0.6:
            print(f"  ⚠️  Confidence scores are low. Consider:")
            print(f"     - Improving feature extraction")
            print(f"     - Model calibration")
        
        if avg_prediction_time > 10.0:
            print(f"  ⚠️  Prediction time is high. Consider:")
            print(f"     - Feature caching")
            print(f"     - Model optimization")
        
        print(f"\n✅ CRAV3 Random Forest Testing Complete!")
        
    else:
        print("❌ No successful predictions to evaluate")
    
    return results

if __name__ == "__main__":
    test_crav3_random_forest_model()
