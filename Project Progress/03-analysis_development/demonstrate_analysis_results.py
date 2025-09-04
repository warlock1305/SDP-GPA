"""
Demonstration of Combined Repository Analysis Results
====================================================

This script demonstrates the comprehensive analysis results from our
AST Features + CodeBERT Embeddings combined analysis system.
"""

import json
import pandas as pd
import numpy as np
from collections import Counter

def load_analysis_results():
    """Load the analysis results from the combined analysis."""
    with open('CombinedAnalysis/comprehensive_analysis_report.json', 'r') as f:
        return json.load(f)

def demonstrate_repository_analysis():
    """Demonstrate the repository analysis results."""
    print("=" * 80)
    print("🔍 COMPREHENSIVE REPOSITORY ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Load results
    results = load_analysis_results()
    summary = results['summary']
    
    print(f"\n📊 ANALYSIS OVERVIEW:")
    print(f"   • Total repositories analyzed: {len(summary)}")
    print(f"   • Analysis approaches: AST Features + CodeBERT Embeddings")
    print(f"   • Metrics calculated: Quality, Complexity, Maintainability, Semantic Richness")
    
    # Quality statistics
    quality_scores = [repo['overall_quality'] for repo in summary]
    print(f"\n🏆 QUALITY ASSESSMENT:")
    print(f"   • Average quality score: {np.mean(quality_scores):.3f}")
    print(f"   • Median quality score: {np.median(quality_scores):.3f}")
    print(f"   • Best quality: {max(quality_scores):.3f}")
    print(f"   • Worst quality: {min(quality_scores):.3f}")
    
    # Architecture patterns
    patterns = Counter([repo['architecture_pattern'] for repo in summary])
    print(f"\n🏗️  ARCHITECTURE PATTERNS DETECTED:")
    for pattern, count in patterns.most_common():
        percentage = (count / len(summary)) * 100
        print(f"   • {pattern}: {count} repositories ({percentage:.1f}%)")
    
    # Project types
    project_types = Counter([repo['project_type'] for repo in summary])
    print(f"\n📁 PROJECT TYPE CLASSIFICATION:")
    for ptype, count in project_types.most_common():
        percentage = (count / len(summary)) * 100
        print(f"   • {ptype}: {count} repositories ({percentage:.1f}%)")
    
    # Languages used
    all_languages = []
    for repo in summary:
        languages = repo['languages'].split(',')
        all_languages.extend(languages)
    
    language_counts = Counter(all_languages)
    print(f"\n💻 PROGRAMMING LANGUAGES:")
    for lang, count in language_counts.most_common():
        print(f"   • {lang}: {count} repositories")
    
    return summary

def demonstrate_individual_repositories(summary):
    """Demonstrate detailed analysis for individual repositories."""
    print(f"\n" + "=" * 80)
    print("📋 DETAILED REPOSITORY ANALYSIS")
    print("=" * 80)
    
    # Sort by quality score
    sorted_repos = sorted(summary, key=lambda x: x['overall_quality'], reverse=True)
    
    for i, repo in enumerate(sorted_repos, 1):
        print(f"\n{i}. 🏆 {repo['repository']}")
        print(f"   📊 Quality Score: {repo['overall_quality']:.3f}")
        print(f"   🏗️  Architecture: {repo['architecture_pattern']}")
        print(f"   📁 Project Type: {repo['project_type']}")
        print(f"   💻 Language: {repo['languages']}")
        print(f"   📈 Metrics:")
        print(f"      • Enhanced Complexity: {repo['enhanced_complexity']:.3f}")
        print(f"      • Enhanced Maintainability: {repo['enhanced_maintainability']:.3f}")
        print(f"      • Semantic Richness: {repo['semantic_richness']:.3f}")
        print(f"      • Technology Diversity: {repo['technology_diversity']:.3f}")
        print(f"   📊 Code Statistics:")
        print(f"      • Total Methods: {repo['total_methods']}")
        print(f"      • Total Files: {repo['total_files']}")

def demonstrate_insights_and_patterns(summary):
    """Demonstrate insights and patterns discovered."""
    print(f"\n" + "=" * 80)
    print("🔍 KEY INSIGHTS & PATTERNS DISCOVERED")
    print("=" * 80)
    
    # Find best and worst quality repositories
    best_repo = max(summary, key=lambda x: x['overall_quality'])
    worst_repo = min(summary, key=lambda x: x['overall_quality'])
    
    print(f"\n🏆 HIGHEST QUALITY REPOSITORY:")
    print(f"   • Repository: {best_repo['repository']}")
    print(f"   • Quality Score: {best_repo['overall_quality']:.3f}")
    print(f"   • Why it's high quality:")
    print(f"     - Low complexity ({best_repo['enhanced_complexity']:.3f})")
    print(f"     - High maintainability ({best_repo['enhanced_maintainability']:.3f})")
    print(f"     - Good semantic richness ({best_repo['semantic_richness']:.3f})")
    
    print(f"\n⚠️  LOWEST QUALITY REPOSITORY:")
    print(f"   • Repository: {worst_repo['repository']}")
    print(f"   • Quality Score: {worst_repo['overall_quality']:.3f}")
    print(f"   • Why it's low quality:")
    print(f"     - High complexity ({worst_repo['enhanced_complexity']:.3f})")
    print(f"     - Low maintainability ({worst_repo['enhanced_maintainability']:.3f})")
    
    # Language-specific insights
    print(f"\n💡 LANGUAGE-SPECIFIC INSIGHTS:")
    js_repos = [repo for repo in summary if 'js' in repo['languages']]
    py_repos = [repo for repo in summary if 'py' in repo['languages']]
    java_repos = [repo for repo in summary if 'java' in repo['languages']]
    
    if js_repos:
        js_avg_quality = np.mean([repo['overall_quality'] for repo in js_repos])
        print(f"   • JavaScript repositories: {len(js_repos)} repos, avg quality: {js_avg_quality:.3f}")
    
    if py_repos:
        py_avg_quality = np.mean([repo['overall_quality'] for repo in py_repos])
        print(f"   • Python repositories: {len(py_repos)} repos, avg quality: {py_avg_quality:.3f}")
    
    if java_repos:
        java_avg_quality = np.mean([repo['overall_quality'] for repo in java_repos])
        print(f"   • Java repositories: {len(java_repos)} repos, avg quality: {java_avg_quality:.3f}")
    
    # Architecture insights
    print(f"\n🏗️  ARCHITECTURE INSIGHTS:")
    web_apps = [repo for repo in summary if repo['architecture_pattern'] == 'web_application']
    if web_apps:
        web_avg_quality = np.mean([repo['overall_quality'] for repo in web_apps])
        print(f"   • Web applications: {len(web_apps)} repos, avg quality: {web_avg_quality:.3f}")
    
    framework_based = [repo for repo in summary if repo['architecture_pattern'] == 'framework_based']
    if framework_based:
        framework_avg_quality = np.mean([repo['overall_quality'] for repo in framework_based])
        print(f"   • Framework-based: {len(framework_based)} repos, avg quality: {framework_avg_quality:.3f}")

def demonstrate_technical_analysis():
    """Demonstrate the technical analysis capabilities."""
    print(f"\n" + "=" * 80)
    print("⚙️  TECHNICAL ANALYSIS CAPABILITIES")
    print("=" * 80)
    
    print(f"\n🔧 AST FEATURES ANALYSIS:")
    print(f"   • Structural code analysis using Abstract Syntax Trees")
    print(f"   • Method count and complexity metrics")
    print(f"   • Path context extraction and analysis")
    print(f"   • Node type diversity assessment")
    print(f"   • Language-specific pattern recognition")
    
    print(f"\n🧠 CODEBERT SEMANTIC ANALYSIS:")
    print(f"   • 768-dimensional semantic embeddings")
    print(f"   • Multi-language code understanding")
    print(f"   • Semantic similarity calculations")
    print(f"   • Code meaning and intent analysis")
    print(f"   • Repository-level semantic representation")
    
    print(f"\n🔄 COMBINED ANALYSIS BENEFITS:")
    print(f"   • Enhanced quality assessment (structural + semantic)")
    print(f"   • Architecture pattern detection")
    print(f"   • Technology stack identification")
    print(f"   • Maintainability and complexity scoring")
    print(f"   • Project type classification")
    print(f"   • Semantic similarity between repositories")

def demonstrate_use_cases():
    """Demonstrate practical use cases for the analysis."""
    print(f"\n" + "=" * 80)
    print("🎯 PRACTICAL USE CASES")
    print("=" * 80)
    
    print(f"\n👥 FOR RECRUITERS & HIRING MANAGERS:")
    print(f"   • Assess candidate's code quality across repositories")
    print(f"   • Identify developers with specific architecture experience")
    print(f"   • Find candidates with high maintainability skills")
    print(f"   • Evaluate technology stack diversity")
    
    print(f"\n👨‍💻 FOR DEVELOPERS:")
    print(f"   • Portfolio quality assessment")
    print(f"   • Code quality improvement insights")
    print(f"   • Architecture pattern recognition")
    print(f"   • Technology stack analysis")
    
    print(f"\n🏢 FOR ORGANIZATIONS:")
    print(f"   • Repository quality benchmarking")
    print(f"   • Architecture pattern analysis")
    print(f"   • Technology stack assessment")
    print(f"   • Code maintainability evaluation")
    
    print(f"\n🔍 FOR RESEARCH:")
    print(f"   • Code quality correlation studies")
    print(f"   • Architecture pattern analysis")
    print(f"   • Language-specific quality metrics")
    print(f"   • Semantic code similarity research")

def main():
    """Main demonstration function."""
    print("🚀 GITHUB PROFILE ANALYZER - ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Load and demonstrate results
    summary = demonstrate_repository_analysis()
    demonstrate_individual_repositories(summary)
    demonstrate_insights_and_patterns(summary)
    demonstrate_technical_analysis()
    demonstrate_use_cases()
    
    print(f"\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETED")
    print("=" * 80)
    print(f"\n📁 Generated Files:")
    print(f"   • CombinedAnalysis/comprehensive_analysis_report.json")
    print(f"   • CombinedAnalysis/comprehensive_analysis_summary.csv")
    print(f"   • CombinedAnalysis/combined_analysis_visualization.png")
    
    print(f"\n🎯 Next Steps in Multi-Model System:")
    print(f"   1. Commit History Analysis")
    print(f"   2. README Analysis & Keyword Extraction")
    print(f"   3. Text Generation for Portfolios")
    print(f"   4. Attention-Based Search System")
    
    print(f"\n💡 This analysis provides a solid foundation for understanding")
    print(f"   repository quality, architecture patterns, and developer skills!")

if __name__ == "__main__":
    main()
