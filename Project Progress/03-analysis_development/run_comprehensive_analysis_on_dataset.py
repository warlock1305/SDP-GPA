"""
Run Comprehensive Repository Analyzer v2 on the Whole Dataset
============================================================

This script runs the comprehensive repository analyzer on all repositories
in the dataset across all categories (cli_tool, web_application, data_science, etc.)
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add the scripts directory to the path
sys.path.append('scripts/analysis')

from comprehensive_repository_analyzer_v2 import ComprehensiveRepositoryAnalyzer

def get_all_repositories() -> List[str]:
    """Get all repository paths from the dataset."""
    dataset_path = Path("dataset")
    repositories = []
    
    # Categories in the dataset
    categories = [
        'cli_tool', 'web_application', 'data_science', 'educational',
        'library', 'mobile_app', 'game_development'
    ]
    
    for category in categories:
        category_path = dataset_path / category
        if category_path.exists():
            for repo_dir in category_path.iterdir():
                if repo_dir.is_dir():
                    repositories.append(str(repo_dir))
    
    return repositories

def analyze_dataset():
    """Run comprehensive analysis on the entire dataset."""
    print("🚀 Starting Comprehensive Repository Analysis on Dataset")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = ComprehensiveRepositoryAnalyzer()
    
    # Try to load pre-trained models
    print("📥 Loading pre-trained models...")
    analyzer.load_models()
    
    if not (analyzer.architectural_classifier and analyzer.category_classifier and analyzer.quality_regressor):
        print("❌ Error: Could not load all required models!")
        print("Available models:")
        for model_file in Path("ml_models").glob("*.pkl"):
            print(f"  - {model_file.name}")
        return
    
    print("✅ All models loaded successfully!")
    
    # Get all repositories
    repositories = get_all_repositories()
    print(f"📁 Found {len(repositories)} repositories to analyze")
    
    # Analysis results storage
    all_results = []
    successful_analyses = 0
    failed_analyses = 0
    
    # Analyze each repository
    for i, repo_path in enumerate(repositories, 1):
        repo_name = Path(repo_path).name
        print(f"\n[{i}/{len(repositories)}] 🔍 Analyzing: {repo_name}")
        
        try:
            # Run comprehensive analysis
            results = analyzer.analyze_repository(repo_path)
            
            # Add repository metadata
            results['repo_name'] = repo_name
            results['repo_path'] = repo_path
            results['category'] = Path(repo_path).parent.name
            
            all_results.append(results)
            successful_analyses += 1
            
            # Print summary
            if 'predictions' in results:
                arch_pred = results['predictions'].get('architectural_pattern', {})
                cat_pred = results['predictions'].get('category_type', {})
                
                print(f"   🏗️  Architecture: {arch_pred.get('pattern', 'Unknown')} ({arch_pred.get('confidence', 0):.2f})")
                print(f"   📂 Category: {cat_pred.get('category', 'Unknown')} ({cat_pred.get('confidence', 0):.2f})")
            
            if 'quality_assessment' in results:
                quality_score = results['quality_assessment'].get('overall_score', 0)
                print(f"   ⭐ Quality Score: {quality_score:.3f}")
            
            if 'programmer_characteristics' in results:
                exp_level = results['programmer_characteristics'].get('experience_level', 'Unknown')
                print(f"   👨‍💻 Experience: {exp_level}")
            
        except Exception as e:
            print(f"   ❌ Error analyzing {repo_name}: {str(e)}")
            failed_analyses += 1
    
    # Save results
    print(f"\n💾 Saving analysis results...")
    
    # Create results directory
    results_dir = Path("comprehensive_dataset_analysis")
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(results_dir / "detailed_analysis_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        summary = {
            'repo_name': result['repo_name'],
            'repo_path': result['repo_path'],
            'category': result['category'],
            'architectural_pattern': result.get('predictions', {}).get('architectural_pattern', {}).get('pattern', 'Unknown'),
            'arch_confidence': result.get('predictions', {}).get('architectural_pattern', {}).get('confidence', 0),
            'predicted_category': result.get('predictions', {}).get('category_type', {}).get('category', 'Unknown'),
            'cat_confidence': result.get('predictions', {}).get('category_type', {}).get('confidence', 0),
            'quality_score': result.get('quality_assessment', {}).get('overall_score', 0),
            'experience_level': result.get('programmer_characteristics', {}).get('experience_level', 'Unknown'),
            'coding_style': result.get('programmer_characteristics', {}).get('coding_style', 'Unknown'),
            'attention_to_detail': result.get('programmer_characteristics', {}).get('attention_to_detail', 'Unknown'),
            'architectural_thinking': result.get('programmer_characteristics', {}).get('architectural_thinking', 'Unknown'),
            'best_practices': result.get('programmer_characteristics', {}).get('best_practices', 'Unknown'),
            'specialization': result.get('programmer_characteristics', {}).get('specialization', 'Unknown')
        }
        summary_data.append(summary)
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "analysis_summary.csv", index=False)
    
    # Create analysis report
    report = {
        'analysis_summary': {
            'total_repositories': len(repositories),
            'successful_analyses': successful_analyses,
            'failed_analyses': failed_analyses,
            'success_rate': successful_analyses / len(repositories) if repositories else 0
        },
        'category_distribution': summary_df['category'].value_counts().to_dict(),
        'architectural_pattern_distribution': summary_df['architectural_pattern'].value_counts().to_dict(),
        'predicted_category_distribution': summary_df['predicted_category'].value_counts().to_dict(),
        'experience_level_distribution': summary_df['experience_level'].value_counts().to_dict(),
        'quality_score_statistics': {
            'mean': summary_df['quality_score'].mean(),
            'median': summary_df['quality_score'].median(),
            'std': summary_df['quality_score'].std(),
            'min': summary_df['quality_score'].min(),
            'max': summary_df['quality_score'].max()
        },
        'confidence_statistics': {
            'arch_confidence_mean': summary_df['arch_confidence'].mean(),
            'cat_confidence_mean': summary_df['cat_confidence'].mean()
        }
    }
    
    # Save report
    with open(results_dir / "analysis_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print final summary
    print(f"\n🎉 Analysis Complete!")
    print("=" * 50)
    print(f"📊 Total repositories analyzed: {len(repositories)}")
    print(f"✅ Successful analyses: {successful_analyses}")
    print(f"❌ Failed analyses: {failed_analyses}")
    print(f"📈 Success rate: {successful_analyses/len(repositories)*100:.1f}%")
    
    print(f"\n📁 Results saved to: {results_dir}")
    print(f"   - detailed_analysis_results.json: Full analysis results")
    print(f"   - analysis_summary.csv: Summary table")
    print(f"   - analysis_report.json: Statistical report")
    
    # Print some key statistics
    if summary_data:
        print(f"\n📈 Key Statistics:")
        print(f"   Average Quality Score: {report['quality_score_statistics']['mean']:.3f}")
        print(f"   Average Architecture Confidence: {report['confidence_statistics']['arch_confidence_mean']:.3f}")
        print(f"   Average Category Confidence: {report['confidence_statistics']['cat_confidence_mean']:.3f}")
        
        print(f"\n🏗️  Top Architectural Patterns:")
        for pattern, count in list(report['architectural_pattern_distribution'].items())[:5]:
            print(f"   {pattern}: {count}")
        
        print(f"\n📂 Top Predicted Categories:")
        for category, count in list(report['predicted_category_distribution'].items())[:5]:
            print(f"   {category}: {count}")

if __name__ == "__main__":
    analyze_dataset()
