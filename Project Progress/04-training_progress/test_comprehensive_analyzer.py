"""
Test Comprehensive Repository Analyzer
=====================================

Demonstrate the trained analyzer on various repository types.
"""

import os
from comprehensive_repository_analyzer_v2 import ComprehensiveRepositoryAnalyzer

def test_comprehensive_analysis():
    """Test the comprehensive analyzer on various repositories."""
    
    print("🧪 Testing Comprehensive Repository Analyzer")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ComprehensiveRepositoryAnalyzer()
    analyzer.load_models()
    
    if not analyzer.architectural_classifier or not analyzer.category_classifier:
        print("❌ No trained models found!")
        return
    
    # Test repositories from different categories
    test_repositories = [
        ("CLI Tool", "dataset/cli_tool/sindresorhus_chalk"),
        ("Data Science", "dataset/data_science/tensorflow_tensorflow"),
        ("Web Application", "dataset/web_application/facebook_react"),
        ("Library", "dataset/library/sindresorhus_chalk"),
        ("Game Development", "dataset/game_development/photonstorm_phaser"),
        ("Mobile App", "dataset/mobile_app/react-native-community_react-native-svg"),
        ("Educational", "dataset/educational/vinta_awesome-python")
    ]
    
    results_summary = []
    
    for repo_type, repo_path in test_repositories:
        if os.path.exists(repo_path):
            print(f"\n🔍 Analyzing {repo_type}: {os.path.basename(repo_path)}")
            print("-" * 50)
            
            try:
                results = analyzer.analyze_repository(repo_path)
                
                # Extract key results
                arch_prediction = results['predictions'].get('architectural_pattern', {})
                cat_prediction = results['predictions'].get('category_type', {})
                quality_assessment = results['quality_assessment']
                programmer_chars = results['programmer_characteristics']
                
                # Print detailed results
                print(f"🏗️  Architectural Pattern: {arch_prediction.get('pattern', 'Unknown')} (Confidence: {arch_prediction.get('confidence', 0):.2%})")
                print(f"📂  Category Type: {cat_prediction.get('category', 'Unknown')} (Confidence: {cat_prediction.get('confidence', 0):.2%})")
                print(f"🎯  Overall Quality Score: {quality_assessment.get('overall_score', 0):.3f}")
                
                # Quality breakdown
                code_quality = quality_assessment.get('code_quality', {})
                arch_quality = quality_assessment.get('architecture_quality', {})
                doc_quality = quality_assessment.get('documentation_quality', {})
                maintainability = quality_assessment.get('maintainability', {})
                
                print(f"   📝 Code Quality: {code_quality.get('complexity', 0):.3f} (complexity), {code_quality.get('structure', 0):.3f} (structure)")
                print(f"   🏛️  Architecture Quality: {arch_quality.get('modularity', 0):.3f} (modularity), {arch_quality.get('abstraction', 0):.3f} (abstraction)")
                print(f"   📚 Documentation Quality: {doc_quality.get('readme_presence', 0):.3f} (README), {doc_quality.get('config_documentation', 0):.3f} (config)")
                print(f"   🔧 Maintainability: {maintainability.get('test_coverage', 0):.3f} (tests), {maintainability.get('code_organization', 0):.3f} (organization)")
                
                # Programmer characteristics
                print(f"👨‍💻 Programmer Profile:")
                print(f"   Experience Level: {programmer_chars.get('experience_level', 'Unknown')}")
                print(f"   Coding Style: {programmer_chars.get('coding_style', 'Unknown')}")
                print(f"   Attention to Detail: {programmer_chars.get('attention_to_detail', 'Unknown')}")
                print(f"   Architectural Thinking: {programmer_chars.get('architectural_thinking', 'Unknown')}")
                print(f"   Best Practices: {programmer_chars.get('best_practices', 'Unknown')}")
                print(f"   Specialization: {programmer_chars.get('specialization', 'Unknown')}")
                
                # Store results for summary
                results_summary.append({
                    'repo_type': repo_type,
                    'repo_name': os.path.basename(repo_path),
                    'arch_pattern': arch_prediction.get('pattern', 'Unknown'),
                    'arch_confidence': arch_prediction.get('confidence', 0),
                    'category': cat_prediction.get('category', 'Unknown'),
                    'cat_confidence': cat_prediction.get('confidence', 0),
                    'quality_score': quality_assessment.get('overall_score', 0),
                    'experience_level': programmer_chars.get('experience_level', 'Unknown'),
                    'specialization': programmer_chars.get('specialization', 'Unknown')
                })
                
            except Exception as e:
                print(f"  ❌ Error analyzing {repo_path}: {e}")
        else:
            print(f"  ❌ Repository not found: {repo_path}")
    
    # Print summary
    print(f"\n📊 Analysis Summary")
    print("=" * 60)
    print(f"{'Repository':<25} {'Arch Pattern':<20} {'Confidence':<12} {'Quality':<8} {'Experience':<12} {'Specialization':<15}")
    print("-" * 100)
    
    for result in results_summary:
        print(f"{result['repo_name']:<25} {result['arch_pattern']:<20} {result['arch_confidence']:<12.1%} {result['quality_score']:<8.3f} {result['experience_level']:<12} {result['specialization']:<15}")
    
    # Calculate overall statistics
    if results_summary:
        avg_quality = sum(r['quality_score'] for r in results_summary) / len(results_summary)
        avg_arch_confidence = sum(r['arch_confidence'] for r in results_summary) / len(results_summary)
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Average Quality Score: {avg_quality:.3f}")
        print(f"   Average Architectural Confidence: {avg_arch_confidence:.1%}")
        print(f"   Repositories Analyzed: {len(results_summary)}")

def analyze_specific_repository(repo_path: str):
    """Analyze a specific repository in detail."""
    
    print(f"🔍 Detailed Analysis: {repo_path}")
    print("=" * 60)
    
    analyzer = ComprehensiveRepositoryAnalyzer()
    analyzer.load_models()
    
    if not os.path.exists(repo_path):
        print(f"❌ Repository not found: {repo_path}")
        return
    
    try:
        results = analyzer.analyze_repository(repo_path)
        
        # Print all available information
        print(f"📁 Repository: {results['repository_path']}")
        
        # Predictions
        print(f"\n🎯 Predictions:")
        for pred_type, pred_data in results['predictions'].items():
            print(f"   {pred_type.replace('_', ' ').title()}: {pred_data}")
        
        # Quality Assessment
        print(f"\n🎯 Quality Assessment:")
        quality = results['quality_assessment']
        print(f"   Overall Score: {quality.get('overall_score', 0):.3f}")
        
        for metric, score in quality.items():
            if isinstance(score, dict):
                print(f"   {metric.replace('_', ' ').title()}:")
                for sub_metric, sub_score in score.items():
                    print(f"     {sub_metric.replace('_', ' ').title()}: {sub_score:.3f}")
            elif metric != 'overall_score':
                print(f"   {metric.replace('_', ' ').title()}: {score:.3f}")
        
        # Programmer Characteristics
        print(f"\n👨‍💻 Programmer Characteristics:")
        chars = results['programmer_characteristics']
        for char, value in chars.items():
            print(f"   {char.replace('_', ' ').title()}: {value}")
        
        # Feature Analysis
        print(f"\n🔧 Feature Analysis:")
        features = results['features']
        
        print(f"   Architectural Features ({len(features['architectural'])}):")
        for feature, value in list(features['architectural'].items())[:10]:  # Show first 10
            print(f"     {feature}: {value}")
        
        print(f"   Category Features ({len(features['category'])}):")
        for feature, value in list(features['category'].items())[:10]:  # Show first 10
            print(f"     {feature}: {value}")
        
    except Exception as e:
        print(f"❌ Error analyzing repository: {e}")

def main():
    """Main function."""
    
    print("🧪 Comprehensive Repository Analyzer Testing")
    print("=" * 60)
    
    # Test on various repositories
    test_comprehensive_analysis()
    
    # Detailed analysis of a specific repository
    print(f"\n" + "="*60)
    specific_repo = "dataset/cli_tool/sindresorhus_chalk"
    analyze_specific_repository(specific_repo)
    
    print(f"\n✅ Testing complete!")

if __name__ == "__main__":
    main()
