#!/usr/bin/env python3
"""
Test CRAv3 Accuracy on temp_fetch_repos
=======================================

This script tests the Comprehensive Repository Analyzer v3.0 on repositories
from temp_fetch_repos and evaluates its accuracy based on expected patterns.
"""

import os
import sys
import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add the scripts/analysis directory to the path
sys.path.append('scripts/analysis')

def load_expected_patterns():
    """Load expected patterns for different repository types."""
    return {
        # Python libraries
        'asweigart_pyscreeze': {
            'expected_patterns': ['library', 'cli_tool'],
            'expected_languages': ['python'],
            'expected_experience': 'intermediate'
        },
        'asweigart_pyperclip': {
            'expected_patterns': ['library', 'cli_tool'],
            'expected_languages': ['python'],
            'expected_experience': 'intermediate'
        },
        'asweigart_pyautogui': {
            'expected_patterns': ['library', 'cli_tool'],
            'expected_languages': ['python'],
            'expected_experience': 'intermediate'
        },
        'kennethreitz_requests': {
            'expected_patterns': ['library', 'web_application'],
            'expected_languages': ['python'],
            'expected_experience': 'senior'
        },
        
        # JavaScript libraries
        'sindresorhus_chalk': {
            'expected_patterns': ['library'],
            'expected_languages': ['javascript'],
            'expected_experience': 'senior'
        },
        'sindresorhus_update-notifier': {
            'expected_patterns': ['library', 'cli_tool'],
            'expected_languages': ['javascript'],
            'expected_experience': 'senior'
        },
        
        # Game development
        'jakesgordon_javascript-pong': {
            'expected_patterns': ['game_development'],
            'expected_languages': ['javascript'],
            'expected_experience': 'intermediate'
        },
        'jakesgordon_javascript-breakout': {
            'expected_patterns': ['game_development'],
            'expected_languages': ['javascript'],
            'expected_experience': 'intermediate'
        },
        'jakesgordon_javascript-tetris': {
            'expected_patterns': ['game_development'],
            'expected_languages': ['javascript'],
            'expected_experience': 'intermediate'
        },
        'grantjenks_free-python-games': {
            'expected_patterns': ['game_development', 'educational'],
            'expected_languages': ['python'],
            'expected_experience': 'intermediate'
        },
        
        # Web frameworks
        'zendframework_zendframework': {
            'expected_patterns': ['web_application', 'library'],
            'expected_languages': ['php'],
            'expected_experience': 'senior'
        },
        'cakephp_cakephp': {
            'expected_patterns': ['web_application', 'library'],
            'expected_languages': ['php'],
            'expected_experience': 'senior'
        },
        'symfony_symfony': {
            'expected_patterns': ['web_application', 'library'],
            'expected_languages': ['php'],
            'expected_experience': 'senior'
        },
        
        # Mobile development
        'kivy_kivy': {
            'expected_patterns': ['mobile_app', 'library'],
            'expected_languages': ['python'],
            'expected_experience': 'senior'
        },
        'kivy_kivy-ios': {
            'expected_patterns': ['mobile_app'],
            'expected_languages': ['python'],
            'expected_experience': 'senior'
        },
        
        # C++ libraries
        'google_protobuf': {
            'expected_patterns': ['library'],
            'expected_languages': ['cpp'],
            'expected_experience': 'senior'
        },
        'google_benchmark': {
            'expected_patterns': ['library'],
            'expected_languages': ['cpp'],
            'expected_experience': 'senior'
        },
        'fmtlib_fmt': {
            'expected_patterns': ['library'],
            'expected_languages': ['cpp'],
            'expected_experience': 'senior'
        },
        
        # Java libraries
        'square_okhttp': {
            'expected_patterns': ['library'],
            'expected_languages': ['java'],
            'expected_experience': 'senior'
        },
        'square_retrofit': {
            'expected_patterns': ['library'],
            'expected_languages': ['java'],
            'expected_experience': 'senior'
        }
    }

def calculate_accuracy_metrics(actual_results: Dict, expected_patterns: Dict) -> Dict[str, float]:
    """Calculate accuracy metrics for the analysis results."""
    metrics = {
        'pattern_accuracy': 0.0,
        'experience_accuracy': 0.0,
        'language_detection_accuracy': 0.0,
        'overall_accuracy': 0.0,
        'quality_score_reasonable': 0.0,
        'confidence_accuracy': 0.0,
        'category_specific_accuracy': {}
    }
    
    repo_name = actual_results.get('repository_name', '')
    if repo_name not in expected_patterns:
        return metrics
    
    expected = expected_patterns[repo_name]
    actual = actual_results
    
    # Pattern accuracy with confidence analysis
    detected_patterns = set(actual.get('detected_patterns', []))
    expected_patterns_set = set(expected.get('expected_patterns', []))
    pattern_confidences = actual.get('pattern_confidences', {})
    
    if expected_patterns_set:
        pattern_intersection = detected_patterns.intersection(expected_patterns_set)
        pattern_accuracy = len(pattern_intersection) / len(expected_patterns_set)
        metrics['pattern_accuracy'] = pattern_accuracy
        
        # Calculate confidence accuracy for correctly detected patterns
        correct_confidences = []
        for pattern in pattern_intersection:
            if pattern in pattern_confidences:
                correct_confidences.append(pattern_confidences[pattern])
        
        if correct_confidences:
            metrics['confidence_accuracy'] = np.mean(correct_confidences)
    
    # Experience level accuracy
    actual_experience = actual.get('experience_level', '').lower()
    expected_experience = expected.get('expected_experience', '').lower()
    
    if actual_experience == expected_experience:
        metrics['experience_accuracy'] = 1.0
    elif (actual_experience in ['senior', 'expert'] and expected_experience in ['senior', 'expert']) or \
         (actual_experience in ['intermediate', 'advanced'] and expected_experience in ['intermediate', 'advanced']) or \
         (actual_experience in ['junior', 'beginner'] and expected_experience in ['junior', 'beginner']):
        metrics['experience_accuracy'] = 0.5
    
    # Language detection accuracy
    language_counts = actual.get('language_counts', {})
    if language_counts:
        detected_languages = [lang for lang, count in language_counts.items() if count > 0]
        expected_languages = expected.get('expected_languages', [])
        
        if expected_languages and detected_languages:
            lang_intersection = set(detected_languages).intersection(set(expected_languages))
            if lang_intersection:
                metrics['language_detection_accuracy'] = 1.0
    
    # Quality score reasonableness
    quality_score = actual.get('quality_score', 0)
    if 0 <= quality_score <= 1:
        metrics['quality_score_reasonable'] = 1.0
    
    # Category-specific accuracy
    for category in ['library', 'web_application', 'cli_tool', 'game_development', 'mobile_app', 'data_science', 'educational']:
        expected_in_category = category in expected_patterns_set
        detected_in_category = category in detected_patterns
        
        if expected_in_category and detected_in_category:
            metrics['category_specific_accuracy'][category] = 1.0
        elif expected_in_category and not detected_in_category:
            metrics['category_specific_accuracy'][category] = 0.0
        elif not expected_in_category and detected_in_category:
            metrics['category_specific_accuracy'][category] = 0.0  # False positive
        else:
            metrics['category_specific_accuracy'][category] = 1.0  # True negative
    
    # Overall accuracy (weighted average)
    weights = [0.35, 0.25, 0.15, 0.1, 0.15]  # pattern, experience, language, quality, confidence
    values = [
        metrics['pattern_accuracy'],
        metrics['experience_accuracy'],
        metrics['language_detection_accuracy'],
        metrics['quality_score_reasonable'],
        metrics['confidence_accuracy']
    ]
    
    metrics['overall_accuracy'] = sum(w * v for w, v in zip(weights, values))
    
    return metrics

def test_crav3_accuracy():
    """Test CRAv3 accuracy on repositories."""
    print("🔍 Testing CRAv3 Accuracy on temp_fetch_repos")
    print("=" * 60)
    
    # Import CRAv3
    try:
        from comprehensive_repository_analyzer_v3 import ComprehensiveRepositoryAnalyzerV3
        print("✅ Successfully imported CRAv3")
    except ImportError as e:
        print(f"❌ Failed to import CRAv3: {e}")
        return
    
    # Load expected patterns
    expected_patterns = load_expected_patterns()
    print(f"📋 Loaded expected patterns for {len(expected_patterns)} repositories")
    
    # Initialize analyzer
    analyzer = ComprehensiveRepositoryAnalyzerV3()
    
    # Get all repositories from temp_fetch_repos (resolve path relative to repo root)
    project_root = Path(__file__).resolve().parents[2]
    temp_fetch_path = project_root / "temp_fetch_repos"
    all_repos = []
    
    if temp_fetch_path.exists():
        for item in temp_fetch_path.iterdir():
            if item.is_dir():
                all_repos.append(item.name)
    
    print(f"📁 Found {len(all_repos)} total repositories in temp_fetch_repos")
    
    # Test repositories (use all repos, not just predefined ones)
    test_repos = all_repos
    results = []
    
    print(f"\n🚀 Testing {len(test_repos)} repositories...")
    print("-" * 60)
    
    for i, repo_name in enumerate(test_repos, 1):
        repo_path = temp_fetch_path / repo_name
        
        if not repo_path.exists():
            print(f"⚠️  Repository not found: {repo_path}")
            continue
        
        print(f"\n[{i}/{len(test_repos)}] Testing: {repo_name}")
        print(f"📍 Path: {repo_path}")
        
        try:
            start_time = time.time()
            
            # Analyze repository
            analysis_result = analyzer.analyze_repository(repo_path)
            
            analysis_time = time.time() - start_time
            
            # Extract key results
            actual_results = {
                'repository_name': repo_name,
                'detected_patterns': analysis_result['architecture_analysis']['detected_patterns'],
                'pattern_confidences': analysis_result['architecture_analysis'].get('pattern_confidence', {}),
                'experience_level': analysis_result['programmer_characteristics']['experience_level'],
                'quality_score': analysis_result['quality_assessment']['overall_score'],
                'language_counts': analysis_result['features']['structure_features'].get('language_counts', {}),
                'analysis_time': analysis_time,
                'category_analysis': analysis_result['architecture_analysis'].get('category_analysis', {})
            }
            
            # Calculate accuracy metrics (only for repos with expected patterns)
            if repo_name in expected_patterns:
                accuracy_metrics = calculate_accuracy_metrics(actual_results, expected_patterns)
            else:
                # For repos without expected patterns, create basic metrics
                accuracy_metrics = {
                    'pattern_accuracy': 0.0,
                    'experience_accuracy': 0.0,
                    'language_detection_accuracy': 0.0,
                    'overall_accuracy': 0.0,
                    'quality_score_reasonable': 1.0 if 0 <= actual_results['quality_score'] <= 1 else 0.0,
                    'confidence_accuracy': np.mean(list(actual_results['pattern_confidences'].values())) if actual_results['pattern_confidences'] else 0.0,
                    'category_specific_accuracy': {}
                }
            
            # Store results
            result_entry = {
                'repository': repo_name,
                'actual_results': actual_results,
                'expected_patterns': expected_patterns.get(repo_name, {}),
                'accuracy_metrics': accuracy_metrics,
                'success': True
            }
            
            results.append(result_entry)
            
            # Print results
            print(f"  ✅ Analysis completed in {analysis_time:.2f}s")
            print(f"  🏗️  Detected patterns: {actual_results['detected_patterns']}")
            print(f"  👨‍💻 Experience level: {actual_results['experience_level']}")
            print(f"  ⭐ Quality score: {actual_results['quality_score']:.3f}")
            if repo_name in expected_patterns:
                print(f"  📊 Overall accuracy: {accuracy_metrics['overall_accuracy']:.3f}")
            else:
                print(f"  📊 No expected patterns for comparison")
            
        except Exception as e:
            print(f"  ❌ Analysis failed: {e}")
            results.append({
                'repository': repo_name,
                'success': False,
                'error': str(e)
            })
    
    # Calculate overall statistics
    print(f"\n📊 OVERALL ACCURACY STATISTICS")
    print("=" * 60)
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        # Calculate average accuracies
        avg_pattern_accuracy = np.mean([r['accuracy_metrics']['pattern_accuracy'] for r in successful_results])
        avg_experience_accuracy = np.mean([r['accuracy_metrics']['experience_accuracy'] for r in successful_results])
        avg_language_accuracy = np.mean([r['accuracy_metrics']['language_detection_accuracy'] for r in successful_results])
        avg_overall_accuracy = np.mean([r['accuracy_metrics']['overall_accuracy'] for r in successful_results])
        avg_quality_reasonable = np.mean([r['accuracy_metrics']['quality_score_reasonable'] for r in successful_results])
        avg_confidence_accuracy = np.mean([r['accuracy_metrics']['confidence_accuracy'] for r in successful_results])
        
        # Calculate average analysis time
        avg_analysis_time = np.mean([r['actual_results']['analysis_time'] for r in successful_results])
        
        print(f"📈 Success Rate: {len(successful_results)}/{len(test_repos)} ({len(successful_results)/len(test_repos)*100:.1f}%)")
        print(f"⏱️  Average Analysis Time: {avg_analysis_time:.2f}s")
        print(f"\n🎯 Accuracy Metrics:")
        print(f"  Pattern Detection:     {avg_pattern_accuracy:.3f} ({avg_pattern_accuracy*100:.1f}%)")
        print(f"  Experience Level:      {avg_experience_accuracy:.3f} ({avg_experience_accuracy*100:.1f}%)")
        print(f"  Language Detection:    {avg_language_accuracy:.3f} ({avg_language_accuracy*100:.1f}%)")
        print(f"  Quality Score Reason:  {avg_quality_reasonable:.3f} ({avg_quality_reasonable*100:.1f}%)")
        print(f"  Confidence Accuracy:   {avg_confidence_accuracy:.3f} ({avg_confidence_accuracy*100:.1f}%)")
        print(f"  Overall Accuracy:      {avg_overall_accuracy:.3f} ({avg_overall_accuracy*100:.1f}%)")
        
        # Category-specific accuracy analysis
        print(f"\n📊 CATEGORY-SPECIFIC ACCURACY ANALYSIS")
        print("-" * 60)
        
        categories = ['library', 'web_application', 'cli_tool', 'game_development', 'mobile_app', 'data_science', 'educational']
        category_accuracies = {}
        
        for category in categories:
            category_results = []
            for result in successful_results:
                if category in result['accuracy_metrics']['category_specific_accuracy']:
                    category_results.append(result['accuracy_metrics']['category_specific_accuracy'][category])
            
            if category_results:
                category_accuracies[category] = np.mean(category_results)
                print(f"  {category:20} | Accuracy: {category_accuracies[category]:.3f} ({category_accuracies[category]*100:.1f}%)")
        
        # Confidence analysis
        print(f"\n🎯 CONFIDENCE ANALYSIS")
        print("-" * 60)
        
        all_confidences = []
        for result in successful_results:
            confidences = list(result['actual_results']['pattern_confidences'].values())
            all_confidences.extend(confidences)
        
        if all_confidences:
            avg_confidence = np.mean(all_confidences)
            confidence_std = np.std(all_confidences)
            print(f"  Average Confidence:     {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")
            print(f"  Confidence Std Dev:     {confidence_std:.3f}")
            print(f"  Min Confidence:         {min(all_confidences):.3f}")
            print(f"  Max Confidence:         {max(all_confidences):.3f}")
        
        # Detailed breakdown
        print(f"\n📋 DETAILED RESULTS BREAKDOWN")
        print("-" * 60)
        for result in successful_results:
            repo_name = result['repository']
            accuracy = result['accuracy_metrics']['overall_accuracy']
            patterns = result['actual_results']['detected_patterns']
            experience = result['actual_results']['experience_level']
            confidences = result['actual_results']['pattern_confidences']
            
            print(f"  {repo_name:30} | Accuracy: {accuracy:.3f} | Patterns: {patterns}")
            print(f"  {'':30} | Experience: {experience} | Confidences: {confidences}")
        
        # Save detailed results
        with open("crav3_accuracy_test_results.json", "w") as f:
            json.dump({
                'summary': {
                    'total_repositories': len(test_repos),
                    'successful_analyses': len(successful_results),
                    'success_rate': len(successful_results)/len(test_repos),
                    'average_analysis_time': avg_analysis_time,
                    'average_accuracies': {
                        'pattern_detection': avg_pattern_accuracy,
                        'experience_level': avg_experience_accuracy,
                        'language_detection': avg_language_accuracy,
                        'quality_score_reasonable': avg_quality_reasonable,
                        'confidence_accuracy': avg_confidence_accuracy,
                        'overall': avg_overall_accuracy
                    },
                    'category_accuracies': category_accuracies,
                    'confidence_statistics': {
                        'average_confidence': avg_confidence if all_confidences else 0,
                        'confidence_std': confidence_std if all_confidences else 0,
                        'min_confidence': min(all_confidences) if all_confidences else 0,
                        'max_confidence': max(all_confidences) if all_confidences else 0
                    }
                },
                'detailed_results': results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: crav3_accuracy_test_results.json")
        
        # Final assessment
        print(f"\n🏆 FINAL ASSESSMENT")
        print("=" * 60)
        
        if avg_overall_accuracy >= 0.8:
            print("🎉 EXCELLENT: CRAv3 shows high accuracy across all metrics!")
        elif avg_overall_accuracy >= 0.6:
            print("✅ GOOD: CRAv3 performs well with room for improvement")
        elif avg_overall_accuracy >= 0.4:
            print("⚠️  FAIR: CRAv3 needs improvements in several areas")
        else:
            print("❌ POOR: CRAv3 requires significant improvements")
        
        print(f"Overall Accuracy Score: {avg_overall_accuracy:.3f} ({avg_overall_accuracy*100:.1f}%)")
        
        # Comprehensive analysis summary
        print(f"\n🔍 COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Pattern detection analysis
        print(f"\n🏗️  PATTERN DETECTION ANALYSIS:")
        pattern_detection_results = []
        for result in successful_results:
            if result['repository'] in expected_patterns:
                pattern_detection_results.append(result['accuracy_metrics']['pattern_accuracy'])
        
        if pattern_detection_results:
            avg_pattern_detection = np.mean(pattern_detection_results)
            print(f"  Average Pattern Detection Accuracy: {avg_pattern_detection:.3f} ({avg_pattern_detection*100:.1f}%)")
            
            # Analyze which patterns are most/least accurate
            pattern_accuracy_by_category = {}
            for category in categories:
                category_results = []
                for result in successful_results:
                    if result['repository'] in expected_patterns and category in result['accuracy_metrics']['category_specific_accuracy']:
                        category_results.append(result['accuracy_metrics']['category_specific_accuracy'][category])
                
                if category_results:
                    pattern_accuracy_by_category[category] = np.mean(category_results)
            
            if pattern_accuracy_by_category:
                best_category = max(pattern_accuracy_by_category.items(), key=lambda x: x[1])
                worst_category = min(pattern_accuracy_by_category.items(), key=lambda x: x[1])
                print(f"  Best Detected Category: {best_category[0]} ({best_category[1]*100:.1f}%)")
                print(f"  Worst Detected Category: {worst_category[0]} ({worst_category[1]*100:.1f}%)")
        
        # Experience level analysis
        print(f"\n👨‍💻 EXPERIENCE LEVEL ANALYSIS:")
        experience_results = []
        for result in successful_results:
            if result['repository'] in expected_patterns:
                experience_results.append(result['accuracy_metrics']['experience_accuracy'])
        
        if experience_results:
            avg_experience_detection = np.mean(experience_results)
            print(f"  Average Experience Level Accuracy: {avg_experience_detection:.3f} ({avg_experience_detection*100:.1f}%)")
        
        # Performance analysis
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        analysis_times = [r['actual_results']['analysis_time'] for r in successful_results]
        if analysis_times:
            print(f"  Fastest Analysis: {min(analysis_times):.2f}s")
            print(f"  Slowest Analysis: {max(analysis_times):.2f}s")
            print(f"  Median Analysis Time: {np.median(analysis_times):.2f}s")
        
        # Quality assessment
        print(f"\n⭐ QUALITY ASSESSMENT:")
        quality_scores = [r['actual_results']['quality_score'] for r in successful_results]
        if quality_scores:
            print(f"  Average Quality Score: {np.mean(quality_scores):.3f}")
            print(f"  Quality Score Range: {min(quality_scores):.3f} - {max(quality_scores):.3f}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if avg_overall_accuracy < 0.7:
            print(f"  ⚠️  Overall accuracy is below 70%. Consider:")
            print(f"     - Improving pattern detection algorithms")
            print(f"     - Enhancing keyword analysis")
            print(f"     - Refining experience level assessment")
        
        if avg_confidence_accuracy < 0.6:
            print(f"  ⚠️  Confidence scores are low. Consider:")
            print(f"     - Adjusting confidence calculation methods")
            print(f"     - Improving feature extraction")
        
        if avg_language_accuracy < 0.5:
            print(f"  ⚠️  Language detection needs improvement. Consider:")
            print(f"     - Expanding supported language detection")
            print(f"     - Improving file extension mapping")
        
        print(f"\n✅ CRAv3 Analysis Complete!")
        
    else:
        print("❌ No successful analyses to evaluate")
    
    return results

if __name__ == "__main__":
    test_crav3_accuracy()
