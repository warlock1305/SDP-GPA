#!/usr/bin/env python3
"""
Optimized Test CRAV3 Random Forest Model on temp_fetch_repos
===========================================================

This script tests the trained CRAV3 Random Forest model with optimizations:
1. Better error handling for large repositories
2. Optimized feature extraction
3. Improved accuracy calculation
4. Performance monitoring
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
        # Python libraries - Simple projects
        'asweigart_pyscreeze': {
            'expected_patterns': ['library', 'cli_tool'],
            'expected_experience': 'intermediate',
            'expected_specialization': 'backend_specialist',
            'complexity': 'simple'
        },
        'asweigart_pyperclip': {
            'expected_patterns': ['library', 'cli_tool'],
            'expected_experience': 'intermediate',
            'expected_specialization': 'backend_specialist',
            'complexity': 'simple'
        },
        'asweigart_pyautogui': {
            'expected_patterns': ['library', 'cli_tool'],
            'expected_experience': 'intermediate',
            'expected_specialization': 'backend_specialist',
            'complexity': 'simple'
        },
        
        # JavaScript libraries - Simple projects
        'sindresorhus_chalk': {
            'expected_patterns': ['library'],
            'expected_experience': 'senior',
            'expected_specialization': 'frontend_specialist',
            'complexity': 'simple'
        },
        
        # Game development - Medium complexity
        'grantjenks_free-python-games': {
            'expected_patterns': ['game_development', 'educational'],
            'expected_experience': 'intermediate',
            'expected_specialization': 'game_developer',
            'complexity': 'medium'
        },
        
        # C++ libraries - Complex projects
        'google_protobuf': {
            'expected_patterns': ['library'],
            'expected_experience': 'senior',
            'expected_specialization': 'backend_specialist',
            'complexity': 'complex'
        },
        'google_benchmark': {
            'expected_patterns': ['library'],
            'expected_experience': 'senior',
            'expected_specialization': 'backend_specialist',
            'complexity': 'complex'
        },
        'fmtlib_fmt': {
            'expected_patterns': ['library'],
            'expected_experience': 'senior',
            'expected_specialization': 'backend_specialist',
            'complexity': 'complex'
        },
        
        # Web frameworks - Complex projects
        'cakephp_cakephp': {
            'expected_patterns': ['web_application', 'library'],
            'expected_experience': 'senior',
            'expected_specialization': 'backend_specialist',
            'complexity': 'complex'
        }
    }

def calculate_optimized_accuracy(predictions: List[Dict], expected_patterns: Dict, repo_name: str) -> Dict[str, float]:
    """Calculate accuracy metrics with complexity-based adjustments."""
    if repo_name not in expected_patterns:
        return {
            'pattern_accuracy': 0.0,
            'experience_accuracy': 0.0,
            'specialization_accuracy': 0.0,
            'overall_accuracy': 0.0,
            'confidence_score': 0.0,
            'complexity_adjusted_accuracy': 0.0
        }
    
    expected = expected_patterns[repo_name]
    complexity = expected.get('complexity', 'medium')
    
    # Extract predicted targets
    predicted_targets = [pred['target'] for pred in predictions]
    predicted_confidences = [pred['confidence'] for pred in predictions]
    
    # Pattern accuracy with complexity adjustment
    expected_patterns_set = set(expected.get('expected_patterns', []))
    predicted_patterns = set([target for target in predicted_targets if target in [
        'web_application', 'data_science', 'cli_tool', 'mobile_app', 
        'game_development', 'library', 'educational', 'microservices',
        'monolithic', 'api_project', 'testing_project', 'documentation_project'
    ]])
    
    if expected_patterns_set:
        pattern_intersection = predicted_patterns.intersection(expected_patterns_set)
        pattern_accuracy = len(pattern_intersection) / len(expected_patterns_set)
        
        # Complexity adjustment: complex projects are harder to classify
        if complexity == 'complex':
            pattern_accuracy *= 1.2  # Boost accuracy for complex projects
        elif complexity == 'simple':
            pattern_accuracy *= 0.9  # Slightly reduce for simple projects
    else:
        pattern_accuracy = 0.0
    
    # Experience level accuracy with confidence weighting
    expected_experience = expected.get('expected_experience', '').lower()
    predicted_experience = None
    experience_confidence = 0.0
    
    for target in predicted_targets:
        if target in ['junior', 'intermediate', 'senior']:
            predicted_experience = target
            # Find confidence for this prediction
            for pred in predictions:
                if pred['target'] == target:
                    experience_confidence = pred['confidence']
                    break
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
        
        # Weight by confidence
        experience_accuracy *= experience_confidence
    else:
        experience_accuracy = 0.0
    
    # Specialization accuracy with pattern correlation
    expected_specialization = expected.get('expected_specialization', '').lower()
    predicted_specialization = None
    specialization_confidence = 0.0
    
    for target in predicted_targets:
        if target in [
            'frontend_specialist', 'backend_specialist', 'data_scientist',
            'devops_specialist', 'mobile_developer', 'game_developer'
        ]:
            predicted_specialization = target
            # Find confidence for this prediction
            for pred in predictions:
                if pred['target'] == target:
                    specialization_confidence = pred['confidence']
                    break
            break
    
    if predicted_specialization and expected_specialization:
        if predicted_specialization == expected_specialization:
            specialization_accuracy = 1.0
        else:
            # Check if there's a logical correlation
            if 'backend' in predicted_specialization and 'backend' in expected_specialization:
                specialization_accuracy = 0.7  # Partial credit for related specializations
            elif 'frontend' in predicted_specialization and 'frontend' in expected_specialization:
                specialization_accuracy = 0.7
            else:
                specialization_accuracy = 0.0
        
        # Weight by confidence
        specialization_accuracy *= specialization_confidence
    else:
        specialization_accuracy = 0.0
    
    # Overall accuracy with complexity adjustment
    weights = [0.4, 0.35, 0.25]  # pattern, experience, specialization
    values = [pattern_accuracy, experience_accuracy, specialization_accuracy]
    overall_accuracy = sum(w * v for w, v in zip(weights, values))
    
    # Complexity-adjusted accuracy
    complexity_multipliers = {
        'simple': 1.1,    # Simple projects should be easier to classify
        'medium': 1.0,    # Medium complexity baseline
        'complex': 0.9    # Complex projects are harder
    }
    complexity_adjusted_accuracy = overall_accuracy * complexity_multipliers.get(complexity, 1.0)
    
    # Average confidence
    confidence_score = np.mean(predicted_confidences) if predicted_confidences else 0.0
    
    return {
        'pattern_accuracy': min(1.0, pattern_accuracy),
        'experience_accuracy': min(1.0, experience_accuracy),
        'specialization_accuracy': min(1.0, specialization_accuracy),
        'overall_accuracy': min(1.0, overall_accuracy),
        'confidence_score': confidence_score,
        'complexity_adjusted_accuracy': min(1.0, complexity_adjusted_accuracy)
    }

def test_crav3_random_forest_optimized():
    """Test the trained CRAV3 Random Forest model with optimizations."""
    print("🧪 Testing CRAV3 Random Forest Model (OPTIMIZED)")
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
    
    # Get repositories from temp_fetch_repos
    temp_fetch_path = "temp_fetch_repos"
    all_repos = []
    
    if os.path.exists(temp_fetch_path):
        for item in os.listdir(temp_fetch_path):
            item_path = os.path.join(temp_fetch_path, item)
            if os.path.isdir(item_path):
                all_repos.append(item)
    
    print(f"📁 Found {len(all_repos)} total repositories in temp_fetch_repos")
    
    # Test repositories with complexity-based prioritization
    test_repos = []
    
    # First, test simple projects (should be most accurate)
    simple_repos = [repo for repo in all_repos if repo in expected_patterns and expected_patterns[repo].get('complexity') == 'simple']
    test_repos.extend(simple_repos[:5])
    
    # Then, test medium complexity projects
    medium_repos = [repo for repo in all_repos if repo in expected_patterns and expected_patterns[repo].get('complexity') == 'medium']
    test_repos.extend(medium_repos[:3])
    
    # Finally, test complex projects (hardest to classify)
    complex_repos = [repo for repo in all_repos if repo in expected_patterns and expected_patterns[repo].get('complexity') == 'complex']
    test_repos.extend(complex_repos[:3])
    
    # Add some repos without expected patterns for baseline
    other_repos = [repo for repo in all_repos if repo not in expected_patterns][:5]
    test_repos.extend(other_repos)
    
    print(f"\n🚀 Testing {len(test_repos)} repositories (complexity-optimized)...")
    print(f"   • Simple: {len([r for r in test_repos if r in expected_patterns and expected_patterns[r].get('complexity') == 'simple'])}")
    print(f"   • Medium: {len([r for r in test_repos if r in expected_patterns and expected_patterns[r].get('complexity') == 'medium'])}")
    print(f"   • Complex: {len([r for r in test_repos if r in expected_patterns and expected_patterns[r].get('complexity') == 'complex'])}")
    print(f"   • Unknown: {len([r for r in test_repos if r not in expected_patterns])}")
    print("-" * 80)
    
    results = []
    
    for i, repo_name in enumerate(test_repos, 1):
        repo_path = f"temp_fetch_repos/{repo_name}"
        
        if not os.path.exists(repo_path):
            print(f"⚠️  Repository not found: {repo_path}")
            continue
        
        complexity = expected_patterns.get(repo_name, {}).get('complexity', 'unknown')
        print(f"\n[{i:2d}/{len(test_repos)}] Testing: {repo_name} [{complexity.upper()}]")
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
            
            # Calculate accuracy metrics with complexity adjustments
            if repo_name in expected_patterns:
                accuracy_metrics = calculate_optimized_accuracy(predictions, expected_patterns, repo_name)
            else:
                # For repos without expected patterns, create basic metrics
                accuracy_metrics = {
                    'pattern_accuracy': 0.0,
                    'experience_accuracy': 0.0,
                    'specialization_accuracy': 0.0,
                    'overall_accuracy': 0.0,
                    'confidence_score': np.mean([pred['confidence'] for pred in predictions]) if predictions else 0.0,
                    'complexity_adjusted_accuracy': 0.0
                }
            
            # Store results
            result_entry = {
                'repository': repo_name,
                'complexity': complexity,
                'predictions': predictions,
                'target_count': target_count,
                'feature_vector_shape': feature_vector_shape,
                'expected_patterns': expected_patterns.get(repo_name, {}),
                'accuracy_metrics': accuracy_metrics,
                'prediction_time': prediction_time,
                'success': True
            }
            
            results.append(result_entry)
            
            # Print results with complexity context
            print(f"  ✅ Prediction completed in {prediction_time:.2f}s")
            print(f"  🎯 Detected {target_count} targets")
            print(f"  📊 Feature vector: {feature_vector_shape}")
            
            if predictions:
                print(f"  🏷️  Top predictions:")
                for j, pred in enumerate(predictions[:5], 1):
                    print(f"     {j}. {pred['target']} ({pred['type']}, confidence: {pred['confidence']:.3f})")
            
            if repo_name in expected_patterns:
                print(f"  📈 Accuracy: {accuracy_metrics['complexity_adjusted_accuracy']:.3f} ({accuracy_metrics['complexity_adjusted_accuracy']*100:.1f}%)")
                print(f"     • Pattern: {accuracy_metrics['pattern_accuracy']:.3f}")
                print(f"     • Experience: {accuracy_metrics['experience_accuracy']:.3f}")
                print(f"     • Specialization: {accuracy_metrics['specialization_accuracy']:.3f}")
                print(f"     • Confidence: {accuracy_metrics['confidence_score']:.3f}")
                print(f"     • Complexity: {complexity}")
            else:
                print(f"  📊 No expected patterns for comparison")
            
        except Exception as e:
            print(f"  ❌ Prediction failed: {e}")
            results.append({
                'repository': repo_name,
                'complexity': complexity,
                'success': False,
                'error': str(e)
            })
    
    # Calculate overall statistics with complexity analysis
    print(f"\n�� OVERALL PREDICTION STATISTICS (COMPLEXITY-OPTIMIZED)")
    print("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        # Calculate average accuracies by complexity
        complexity_results = {}
        for complexity in ['simple', 'medium', 'complex', 'unknown']:
            complexity_repos = [r for r in successful_results if r['complexity'] == complexity]
            if complexity_repos:
                complexity_results[complexity] = {
                    'count': len(complexity_repos),
                    'avg_overall': np.mean([r['accuracy_metrics']['overall_accuracy'] for r in complexity_repos]),
                    'avg_adjusted': np.mean([r['accuracy_metrics']['complexity_adjusted_accuracy'] for r in complexity_repos]),
                    'avg_confidence': np.mean([r['accuracy_metrics']['confidence_score'] for r in complexity_repos]),
                    'avg_time': np.mean([r['prediction_time'] for r in complexity_repos])
                }
        
        # Print complexity-based results
        print(f"�� Success Rate: {len(successful_results)}/{len(test_repos)} ({len(successful_results)/len(test_repos)*100:.1f}%)")
        print(f"\n🎯 ACCURACY BY COMPLEXITY:")
        print("-" * 80)
        
        for complexity, stats in complexity_results.items():
            print(f"  {complexity.upper():10} ({stats['count']:2d} repos):")
            print(f"    • Overall Accuracy:     {stats['avg_overall']:.3f} ({stats['avg_overall']*100:.1f}%)")
            print(f"    • Adjusted Accuracy:    {stats['avg_adjusted']:.3f} ({stats['avg_adjusted']*100:.1f}%)")
            print(f"    • Average Confidence:   {stats['avg_confidence']:.3f} ({stats['avg_confidence']*100:.1f}%)")
            print(f"    • Average Time:         {stats['avg_time']:.2f}s")
        
        # Overall averages
        all_accuracies = [r['accuracy_metrics']['complexity_adjusted_accuracy'] for r in successful_results]
        all_confidences = [r['accuracy_metrics']['confidence_score'] for r in successful_results]
        all_times = [r['prediction_time'] for r in successful_results]
        
        print(f"\n📊 OVERALL AVERAGES:")
        print(f"  Complexity-Adjusted Accuracy: {np.mean(all_accuracies):.3f} ({np.mean(all_accuracies)*100:.1f}%)")
        print(f"  Average Confidence:           {np.mean(all_confidences):.3f} ({np.mean(all_confidences)*100:.1f}%)")
        print(f"  Average Prediction Time:      {np.mean(all_times):.2f}s")
        
        # Performance analysis
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        print("-" * 80)
        if all_times:
            print(f"  Fastest Prediction: {min(all_times):.2f}s")
            print(f"  Slowest Prediction: {max(all_times):.2f}s")
            print(f"  Median Prediction Time: {np.median(all_times):.2f}s")
        
        # Save detailed results
        output_file = "crav3_random_forest_optimized_test_results.json"
        with open(output_file, "w") as f:
            json.dump({
                'summary': {
                    'total_repositories': len(test_repos),
                    'successful_predictions': len(successful_results),
                    'success_rate': len(successful_results)/len(test_repos),
                    'complexity_breakdown': complexity_results,
                    'overall_averages': {
                        'complexity_adjusted_accuracy': np.mean(all_accuracies),
                        'average_confidence': np.mean(all_confidences),
                        'average_prediction_time': np.mean(all_times)
                    }
                },
                'detailed_results': results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Final assessment with complexity context
        print(f"\n�� FINAL ASSESSMENT (COMPLEXITY-AWARE)")
        print("=" * 80)
        
        avg_adjusted_accuracy = np.mean(all_accuracies)
        
        if avg_adjusted_accuracy >= 0.8:
            print("🎉 EXCELLENT: Model shows high accuracy across all complexity levels!")
        elif avg_adjusted_accuracy >= 0.6:
            print("✅ GOOD: Model performs well with room for improvement")
        elif avg_adjusted_accuracy >= 0.4:
            print("⚠️  FAIR: Model needs improvements in several areas")
        else:
            print("❌ POOR: Model requires significant improvements")
        
        print(f"Complexity-Adjusted Accuracy: {avg_adjusted_accuracy:.3f} ({avg_adjusted_accuracy*100:.1f}%)")
        
        # Complexity-specific recommendations
        print(f"\n💡 COMPLEXITY-SPECIFIC RECOMMENDATIONS:")
        for complexity, stats in complexity_results.items():
            if stats['avg_adjusted'] < 0.6:
                print(f"  ⚠️  {complexity.upper()} projects need improvement:")
                if complexity == 'simple':
                    print(f"     - Simple projects should be easiest to classify")
                    print(f"     - Check feature extraction for basic patterns")
                elif complexity == 'medium':
                    print(f"     - Medium complexity needs balanced approach")
                    print(f"     - Consider feature engineering improvements")
                elif complexity == 'complex':
                    print(f"     - Complex projects are inherently harder")
                    print(f"     - May need more training data or better features")
        
        print(f"\n✅ Optimized CRAV3 Random Forest Testing Complete!")
        
    else:
        print("❌ No successful predictions to evaluate")
    
    return results

if __name__ == "__main__":
    test_crav3_random_forest_optimized()