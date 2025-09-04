"""
Train Comprehensive Repository Analyzer
======================================

This script trains the multi-stage analysis pipeline using our existing dataset.
"""

import os
import json
import numpy as np
from pathlib import Path
from comprehensive_repository_analyzer_v2 import ComprehensiveRepositoryAnalyzer
from typing import List, Tuple, Dict, Any

def load_training_data() -> Tuple[List[Tuple[Dict, str]], List[Tuple[Dict, str]], List[Tuple[Dict, float]]]:
    """Load and prepare training data from our existing dataset."""
    
    print("📊 Loading training data...")
    
    # Initialize analyzer to extract features
    analyzer = ComprehensiveRepositoryAnalyzer()
    
    # Define architectural patterns (based on our analysis)
    architectural_patterns = {
        'cli_tool': 'Command Line Interface',
        'data_science': 'Data Science Pipeline',
        'web_application': 'Web Application',
        'library': 'Software Library',
        'game_development': 'Game Development',
        'mobile_app': 'Mobile Application',
        'educational': 'Educational Project'
    }
    
    # Define quality scores (based on repository characteristics)
    def calculate_quality_score(repo_path: str, category: str) -> float:
        """Calculate quality score based on repository characteristics."""
        quality_score = 0.5  # Base score
        
        # Check for best practices
        if os.path.exists(os.path.join(repo_path, 'README.md')):
            quality_score += 0.1
        if os.path.exists(os.path.join(repo_path, 'tests')) or any('test' in f for f in os.listdir(repo_path)):
            quality_score += 0.15
        if os.path.exists(os.path.join(repo_path, 'requirements.txt')) or os.path.exists(os.path.join(repo_path, 'package.json')):
            quality_score += 0.1
        if os.path.exists(os.path.join(repo_path, '.gitignore')):
            quality_score += 0.05
        
        # Category-specific bonuses
        if category == 'data_science' and os.path.exists(os.path.join(repo_path, 'notebooks')):
            quality_score += 0.1
        if category == 'web_application' and os.path.exists(os.path.join(repo_path, 'src')):
            quality_score += 0.1
        if category == 'library' and os.path.exists(os.path.join(repo_path, 'docs')):
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    # Collect training data
    architectural_data = []
    category_data = []
    quality_data = []
    
    dataset_path = Path("dataset")
    
    if dataset_path.exists():
        for category_dir in dataset_path.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                
                # Skip if not a valid category
                if category_name not in architectural_patterns:
                    continue
                
                print(f"Processing {category_name} repositories...")
                
                for repo_dir in category_dir.iterdir():
                    if repo_dir.is_dir():
                        repo_path = str(repo_dir)
                        
                        try:
                            # Extract features
                            features = analyzer.extract_all_features(repo_path)
                            
                            if features['architectural'] and features['category']:
                                # Architectural pattern data
                                pattern = architectural_patterns[category_name]
                                architectural_data.append((features, pattern))
                                
                                # Category data
                                category_data.append((features, category_name))
                                
                                # Quality data
                                quality_score = calculate_quality_score(repo_path, category_name)
                                quality_data.append((features, quality_score))
                                
                                print(f"  ✅ {repo_dir.name}: Pattern={pattern}, Quality={quality_score:.2f}")
                            
                        except Exception as e:
                            print(f"  ❌ Error processing {repo_dir.name}: {e}")
    
    print(f"\n📈 Training Data Summary:")
    print(f"  Architectural patterns: {len(architectural_data)} samples")
    print(f"  Categories: {len(category_data)} samples")
    print(f"  Quality scores: {len(quality_data)} samples")
    
    return architectural_data, category_data, quality_data

def train_models():
    """Train all models in the comprehensive analyzer."""
    
    print("🚀 Training Comprehensive Repository Analyzer")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ComprehensiveRepositoryAnalyzer()
    
    # Load training data
    architectural_data, category_data, quality_data = load_training_data()
    
    if not architectural_data or not category_data or not quality_data:
        print("❌ Insufficient training data!")
        return
    
    # Train architectural pattern classifier
    print(f"\n🏗️ Training Architectural Pattern Classifier...")
    print(f"   Samples: {len(architectural_data)}")
    
    # Get unique patterns
    patterns = list(set([pattern for _, pattern in architectural_data]))
    print(f"   Patterns: {patterns}")
    
    analyzer.train_architectural_classifier(architectural_data)
    print("   ✅ Architectural classifier trained!")
    
    # Train category classifier
    print(f"\n📂 Training Category Classifier...")
    print(f"   Samples: {len(category_data)}")
    
    # Get unique categories
    categories = list(set([category for _, category in category_data]))
    print(f"   Categories: {categories}")
    
    analyzer.train_category_classifier(category_data)
    print("   ✅ Category classifier trained!")
    
    # Train quality assessor
    print(f"\n🎯 Training Quality Assessor...")
    print(f"   Samples: {len(quality_data)}")
    
    quality_scores = [score for _, score in quality_data]
    print(f"   Quality score range: {min(quality_scores):.2f} - {max(quality_scores):.2f}")
    print(f"   Average quality: {np.mean(quality_scores):.2f}")
    
    analyzer.train_quality_assessor(quality_data)
    print("   ✅ Quality assessor trained!")
    
    # Save models
    print(f"\n💾 Saving trained models...")
    analyzer.save_models()
    print("   ✅ Models saved successfully!")
    
    # Test the trained models
    print(f"\n🧪 Testing trained models...")
    test_models(analyzer, architectural_data, category_data, quality_data)
    
    return analyzer

def test_models(analyzer: ComprehensiveRepositoryAnalyzer, 
                architectural_data: List[Tuple[Dict, str]], 
                category_data: List[Tuple[Dict, str]], 
                quality_data: List[Tuple[Dict, float]]):
    """Test the trained models on a subset of training data."""
    
    print("   Testing architectural pattern classification...")
    
    # Test architectural classifier
    correct_arch = 0
    total_arch = min(10, len(architectural_data))
    
    for i in range(total_arch):
        features, true_pattern = architectural_data[i]
        arch_features = list(features['architectural'].values())
        arch_features_scaled = analyzer.arch_scaler.transform([arch_features])
        predicted_pattern = analyzer.architectural_classifier.predict(arch_features_scaled)[0]
        
        if predicted_pattern == true_pattern:
            correct_arch += 1
    
    arch_accuracy = correct_arch / total_arch
    print(f"   Architectural accuracy: {arch_accuracy:.2%}")
    
    # Test category classifier
    print("   Testing category classification...")
    correct_cat = 0
    total_cat = min(10, len(category_data))
    
    for i in range(total_cat):
        features, true_category = category_data[i]
        cat_features = list(features['category'].values())
        cat_features_scaled = analyzer.cat_scaler.transform([cat_features])
        predicted_category = analyzer.category_classifier.predict(cat_features_scaled)[0]
        
        if predicted_category == true_category:
            correct_cat += 1
    
    cat_accuracy = correct_cat / total_cat
    print(f"   Category accuracy: {cat_accuracy:.2%}")
    
    # Test quality assessor
    print("   Testing quality assessment...")
    quality_errors = []
    total_quality = min(10, len(quality_data))
    
    for i in range(total_quality):
        features, true_quality = quality_data[i]
        combined_features = list(features['architectural'].values()) + list(features['category'].values())
        combined_features_scaled = analyzer.quality_scaler.transform([combined_features])
        predicted_quality = analyzer.quality_regressor.predict(combined_features_scaled)[0]
        
        error = abs(predicted_quality - true_quality)
        quality_errors.append(error)
    
    avg_quality_error = np.mean(quality_errors)
    print(f"   Average quality error: {avg_quality_error:.3f}")
    
    print(f"\n🎉 Model Training Complete!")
    print(f"   Overall Performance:")
    print(f"   - Architectural Pattern Accuracy: {arch_accuracy:.2%}")
    print(f"   - Category Classification Accuracy: {cat_accuracy:.2%}")
    print(f"   - Quality Assessment Error: {avg_quality_error:.3f}")

def demonstrate_analysis():
    """Demonstrate the trained analyzer on sample repositories."""
    
    print("\n🎭 Demonstrating Comprehensive Analysis")
    print("=" * 50)
    
    # Initialize analyzer and load models
    analyzer = ComprehensiveRepositoryAnalyzer()
    analyzer.load_models()
    
    if not analyzer.architectural_classifier or not analyzer.category_classifier:
        print("❌ No trained models found!")
        return
    
    # Sample repositories to analyze
    sample_repos = [
        "dataset/cli_tool/sindresorhus_chalk",
        "dataset/data_science/tensorflow_tensorflow",
        "dataset/web_application/facebook_react",
        "dataset/library/sindresorhus_chalk"
    ]
    
    for repo_path in sample_repos:
        if os.path.exists(repo_path):
            print(f"\n🔍 Analyzing: {repo_path}")
            print("-" * 40)
            
            try:
                results = analyzer.analyze_repository(repo_path)
                
                # Print predictions
                if 'predictions' in results:
                    for pred_type, pred_data in results['predictions'].items():
                        print(f"  {pred_type.replace('_', ' ').title()}: {pred_data}")
                
                # Print quality assessment
                if 'quality_assessment' in results:
                    print(f"  Overall Quality Score: {results['quality_assessment'].get('overall_score', 0):.3f}")
                
                # Print programmer characteristics
                if 'programmer_characteristics' in results:
                    chars = results['programmer_characteristics']
                    print(f"  Experience Level: {chars.get('experience_level', 'Unknown')}")
                    print(f"  Specialization: {chars.get('specialization', 'Unknown')}")
                
            except Exception as e:
                print(f"  ❌ Error analyzing {repo_path}: {e}")
        else:
            print(f"  ❌ Repository not found: {repo_path}")

def main():
    """Main function."""
    
    print("🚀 Comprehensive Repository Analyzer Training")
    print("=" * 60)
    
    # Train models
    analyzer = train_models()
    
    if analyzer:
        # Demonstrate the trained models
        demonstrate_analysis()
        
        print(f"\n✅ Training and demonstration complete!")
        print(f"💡 You can now use the trained models for repository analysis.")

if __name__ == "__main__":
    main()
