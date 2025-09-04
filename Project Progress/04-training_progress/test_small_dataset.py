"""
Test Comprehensive Repository Analyzer on Small Dataset
=====================================================

Test the trained analyzer on a small subset of repositories to show accuracy and results.
"""

import os
import numpy as np
from pathlib import Path
from comprehensive_repository_analyzer_v2 import ComprehensiveRepositoryAnalyzer

def test_small_dataset():
    """Test the analyzer on a small, diverse subset of repositories."""
    
    print("🧪 Testing Comprehensive Repository Analyzer on Small Dataset")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = ComprehensiveRepositoryAnalyzer()
    analyzer.load_models()
    
    if not analyzer.architectural_classifier or not analyzer.category_classifier:
        print("❌ No trained models found!")
        return
    
    # Select a small, diverse subset of repositories for testing
    test_repositories = [
        # CLI Tools
        ("cli_tool", "dataset/cli_tool/sindresorhus_chalk", "Command Line Interface"),
        ("cli_tool", "dataset/cli_tool/sindresorhus_meow", "Command Line Interface"),
        
        # Data Science
        ("data_science", "dataset/data_science/tensorflow_tensorflow", "Data Science Pipeline"),
        ("data_science", "dataset/data_science/pytorch_pytorch", "Data Science Pipeline"),
        
        # Web Applications
        ("web_application", "dataset/web_application/facebook_react", "Web Application"),
        ("web_application", "dataset/web_application/vuejs_vue", "Web Application"),
        
        # Libraries
        ("library", "dataset/library/sindresorhus_chalk", "Software Library"),
        ("library", "dataset/library/sindresorhus_ora", "Software Library"),
        
        # Game Development
        ("game_development", "dataset/game_development/photonstorm_phaser", "Game Development"),
        ("game_development", "dataset/game_development/craftyjs_Crafty", "Game Development"),
        
        # Mobile Apps
        ("mobile_app", "dataset/mobile_app/react-native-community_react-native-svg", "Mobile Application"),
        ("mobile_app", "dataset/mobile_app/react-native-community_react-native-async-storage", "Mobile Application"),
        
        # Educational
        ("educational", "dataset/educational/vinta_awesome-python", "Educational Project"),
        ("educational", "dataset/educational/avelino_awesome-go", "Educational Project")
    ]
    
    results = []
    
    print(f"📊 Testing {len(test_repositories)} repositories...")
    print("-" * 70)
    
    for category, repo_path, expected_pattern in test_repositories:
        if os.path.exists(repo_path):
            print(f"\n🔍 Testing: {os.path.basename(repo_path)} ({category})")
            print(f"   Expected: {expected_pattern}")
            
            try:
                # Analyze repository
                analysis_results = analyzer.analyze_repository(repo_path)
                
                # Extract predictions
                arch_prediction = analysis_results['predictions'].get('architectural_pattern', {})
                cat_prediction = analysis_results['predictions'].get('category_type', {})
                quality_assessment = analysis_results['quality_assessment']
                programmer_chars = analysis_results['programmer_characteristics']
                
                # Get prediction details
                predicted_pattern = arch_prediction.get('pattern', 'Unknown')
                arch_confidence = arch_prediction.get('confidence', 0)
                predicted_category = cat_prediction.get('category', 'Unknown')
                cat_confidence = cat_prediction.get('confidence', 0)
                quality_score = quality_assessment.get('overall_score', 0)
                
                # Check accuracy
                arch_correct = predicted_pattern == expected_pattern
                cat_correct = predicted_category == category
                
                # Print results
                print(f"   🏗️  Architectural: {predicted_pattern} (Confidence: {arch_confidence:.1%}) {'✅' if arch_correct else '❌'}")
                print(f"   📂  Category: {predicted_category} (Confidence: {cat_confidence:.1%}) {'✅' if cat_correct else '❌'}")
                print(f"   🎯  Quality Score: {quality_score:.3f}")
                print(f"   👨‍💻  Experience: {programmer_chars.get('experience_level', 'Unknown')}")
                print(f"   🎯  Specialization: {programmer_chars.get('specialization', 'Unknown')}")
                
                # Store results
                results.append({
                    'repository': os.path.basename(repo_path),
                    'category': category,
                    'expected_pattern': expected_pattern,
                    'predicted_pattern': predicted_pattern,
                    'arch_confidence': arch_confidence,
                    'arch_correct': arch_correct,
                    'predicted_category': predicted_category,
                    'cat_confidence': cat_confidence,
                    'cat_correct': cat_correct,
                    'quality_score': quality_score,
                    'experience_level': programmer_chars.get('experience_level', 'Unknown'),
                    'specialization': programmer_chars.get('specialization', 'Unknown')
                })
                
            except Exception as e:
                print(f"   ❌ Error analyzing {repo_path}: {e}")
        else:
            print(f"   ❌ Repository not found: {repo_path}")
    
    # Calculate and display accuracy metrics
    print(f"\n" + "="*70)
    print("📈 ACCURACY ANALYSIS")
    print("="*70)
    
    if results:
        # Architectural pattern accuracy
        arch_correct = sum(1 for r in results if r['arch_correct'])
        arch_accuracy = arch_correct / len(results)
        
        # Category accuracy
        cat_correct = sum(1 for r in results if r['cat_correct'])
        cat_accuracy = cat_correct / len(results)
        
        # Overall accuracy
        both_correct = sum(1 for r in results if r['arch_correct'] and r['cat_correct'])
        overall_accuracy = both_correct / len(results)
        
        print(f"🏗️  Architectural Pattern Accuracy: {arch_accuracy:.1%} ({arch_correct}/{len(results)})")
        print(f"📂  Category Classification Accuracy: {cat_accuracy:.1%} ({cat_correct}/{len(results)})")
        print(f"🎯  Overall Accuracy (Both Correct): {overall_accuracy:.1%} ({both_correct}/{len(results)})")
        
        # Average confidence scores
        avg_arch_confidence = np.mean([r['arch_confidence'] for r in results])
        avg_cat_confidence = np.mean([r['cat_confidence'] for r in results])
        avg_quality = np.mean([r['quality_score'] for r in results])
        
        print(f"\n📊 AVERAGE SCORES:")
        print(f"   Architectural Confidence: {avg_arch_confidence:.1%}")
        print(f"   Category Confidence: {avg_cat_confidence:.1%}")
        print(f"   Quality Score: {avg_quality:.3f}")
        
        # Detailed results table
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 100)
        print(f"{'Repository':<25} {'Category':<15} {'Arch Pattern':<20} {'Correct':<8} {'Confidence':<12} {'Quality':<8}")
        print("-" * 100)
        
        for result in results:
            status = "✅" if result['arch_correct'] else "❌"
            print(f"{result['repository']:<25} {result['category']:<15} {result['predicted_pattern']:<20} {status:<8} {result['arch_confidence']:<12.1%} {result['quality_score']:<8.3f}")
        
        # Category-wise accuracy
        print(f"\n📂 CATEGORY-WISE ACCURACY:")
        categories = set(r['category'] for r in results)
        for cat in sorted(categories):
            cat_results = [r for r in results if r['category'] == cat]
            cat_arch_correct = sum(1 for r in cat_results if r['arch_correct'])
            cat_cat_correct = sum(1 for r in cat_results if r['cat_correct'])
            cat_arch_acc = cat_arch_correct / len(cat_results)
            cat_cat_acc = cat_cat_correct / len(cat_results)
            
            print(f"   {cat:<15}: Arch={cat_arch_acc:.1%} ({cat_arch_correct}/{len(cat_results)}), Cat={cat_cat_acc:.1%} ({cat_cat_correct}/{len(cat_results)})")
        
        # Experience level distribution
        print(f"\n👨‍💻 EXPERIENCE LEVEL DISTRIBUTION:")
        experience_counts = {}
        for result in results:
            exp = result['experience_level']
            experience_counts[exp] = experience_counts.get(exp, 0) + 1
        
        for exp, count in sorted(experience_counts.items()):
            percentage = count / len(results) * 100
            print(f"   {exp:<12}: {count} repositories ({percentage:.1f}%)")
        
        # Quality score distribution
        print(f"\n🎯 QUALITY SCORE DISTRIBUTION:")
        quality_scores = [r['quality_score'] for r in results]
        print(f"   Min: {min(quality_scores):.3f}")
        print(f"   Max: {max(quality_scores):.3f}")
        print(f"   Mean: {np.mean(quality_scores):.3f}")
        print(f"   Std: {np.std(quality_scores):.3f}")
        
        # High confidence vs low confidence analysis
        high_conf_arch = [r for r in results if r['arch_confidence'] > 0.7]
        low_conf_arch = [r for r in results if r['arch_confidence'] <= 0.7]
        
        if high_conf_arch:
            high_conf_accuracy = sum(1 for r in high_conf_arch if r['arch_correct']) / len(high_conf_arch)
            print(f"\n🎯 HIGH CONFIDENCE ANALYSIS (Arch > 70%):")
            print(f"   Count: {len(high_conf_arch)} repositories")
            print(f"   Accuracy: {high_conf_accuracy:.1%}")
        
        if low_conf_arch:
            low_conf_accuracy = sum(1 for r in low_conf_arch if r['arch_correct']) / len(low_conf_arch)
            print(f"🎯 LOW CONFIDENCE ANALYSIS (Arch ≤ 70%):")
            print(f"   Count: {len(low_conf_arch)} repositories")
            print(f"   Accuracy: {low_conf_accuracy:.1%}")
    
    else:
        print("❌ No results to analyze!")

def analyze_specific_examples():
    """Analyze a few specific examples in detail."""
    
    print(f"\n" + "="*70)
    print("🔍 DETAILED ANALYSIS EXAMPLES")
    print("="*70)
    
    analyzer = ComprehensiveRepositoryAnalyzer()
    analyzer.load_models()
    
    # Analyze a few specific repositories in detail
    examples = [
        ("CLI Tool Example", "dataset/cli_tool/sindresorhus_chalk"),
        ("Data Science Example", "dataset/data_science/tensorflow_tensorflow"),
        ("Web App Example", "dataset/web_application/facebook_react")
    ]
    
    for example_name, repo_path in examples:
        if os.path.exists(repo_path):
            print(f"\n📁 {example_name}: {os.path.basename(repo_path)}")
            print("-" * 50)
            
            try:
                results = analyzer.analyze_repository(repo_path)
                
                # Predictions
                predictions = results['predictions']
                print(f"🎯 Predictions:")
                for pred_type, pred_data in predictions.items():
                    print(f"   {pred_type.replace('_', ' ').title()}: {pred_data}")
                
                # Quality breakdown
                quality = results['quality_assessment']
                print(f"\n🎯 Quality Assessment:")
                print(f"   Overall Score: {quality.get('overall_score', 0):.3f}")
                
                for metric, score in quality.items():
                    if isinstance(score, dict):
                        print(f"   {metric.replace('_', ' ').title()}:")
                        for sub_metric, sub_score in score.items():
                            print(f"     {sub_metric.replace('_', ' ').title()}: {sub_score:.3f}")
                
                # Programmer characteristics
                chars = results['programmer_characteristics']
                print(f"\n👨‍💻 Programmer Profile:")
                for char, value in chars.items():
                    print(f"   {char.replace('_', ' ').title()}: {value}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"   ❌ Repository not found: {repo_path}")

def main():
    """Main function."""
    
    print("🧪 Small Dataset Testing for Comprehensive Repository Analyzer")
    print("=" * 70)
    
    # Test on small dataset
    test_small_dataset()
    
    # Show detailed examples
    analyze_specific_examples()
    
    print(f"\n✅ Small dataset testing complete!")

if __name__ == "__main__":
    main()
