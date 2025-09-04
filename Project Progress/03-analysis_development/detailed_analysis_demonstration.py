"""
Detailed Analysis Demonstration
==============================

This script demonstrates the detailed analysis for a specific repository,
showing how the enhanced architectural pattern detection works in practice.
"""

import json
import numpy as np

def demonstrate_detailed_analysis():
    """Demonstrate detailed analysis for a specific repository."""
    print("=" * 80)
    print("🔍 DETAILED REPOSITORY ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Load enhanced analysis
    with open('CombinedAnalysis/enhanced_architecture_analysis.json', 'r') as f:
        enhanced_data = json.load(f)
    
    # Focus on the most complex repository (ibecir/oop-1002)
    repo_name = "ibecir/oop-1002"
    repo_data = enhanced_data['detailed_analysis'][repo_name]
    
    print(f"\n📁 ANALYZING REPOSITORY: {repo_name}")
    print("=" * 60)
    
    # Show AST Metrics
    print(f"\n🔧 AST METRICS (Structural Analysis):")
    ast_metrics = repo_data['ast_metrics']
    print(f"   • Total Methods: {ast_metrics['total_methods']}")
    print(f"   • Total Path Contexts: {ast_metrics['total_path_contexts']:,}")
    print(f"   • Unique Node Types: {ast_metrics['unique_node_types']:,}")
    print(f"   • Unique Tokens: {ast_metrics['unique_tokens']}")
    print(f"   • Average Path Length: {ast_metrics['avg_path_length']:.2f}")
    print(f"   • Average Path Diversity: {ast_metrics['avg_path_diversity']:.4f}")
    print(f"   • Languages: {', '.join(ast_metrics['languages'])}")
    
    # Show CodeBERT Metrics
    print(f"\n🧠 CODEBERT METRICS (Semantic Analysis):")
    codebert_metrics = repo_data['codebert_metrics']
    print(f"   • Number of Files: {codebert_metrics['num_files']}")
    print(f"   • Embedding Dimension: {codebert_metrics['embedding_dimension']}")
    print(f"   • Repository Embedding: {len(codebert_metrics['repository_embedding'])} dimensions")
    
    # Show Combined Metrics
    print(f"\n🔄 COMBINED METRICS:")
    combined_metrics = repo_data['combined_metrics']
    print(f"   • Enhanced Complexity: {combined_metrics['enhanced_complexity']:.3f}")
    print(f"   • Enhanced Maintainability: {combined_metrics['enhanced_maintainability']:.3f}")
    print(f"   • Semantic Richness: {combined_metrics['semantic_richness']:.3f}")
    print(f"   • Technology Diversity: {combined_metrics['technology_diversity']:.3f}")
    print(f"   • Overall Quality: {combined_metrics['overall_quality']:.3f}")
    
    # Show Pattern Detection Results
    print(f"\n🏗️  ARCHITECTURAL PATTERN DETECTION:")
    pattern_results = repo_data['enhanced_architecture_patterns']
    primary_pattern = pattern_results['primary_pattern']
    
    print(f"   🎯 PRIMARY PATTERN: {primary_pattern['name']}")
    print(f"   📝 Description: {primary_pattern['description']}")
    print(f"   🎯 Confidence: {primary_pattern['confidence']:.2f}")
    print(f"   📊 Total Patterns Detected: {pattern_results['pattern_count']}")
    
    print(f"\n📋 ALL DETECTED PATTERNS:")
    for i, pattern in enumerate(pattern_results['all_patterns'], 1):
        print(f"   {i:2d}. {pattern['name']}")
        print(f"       📝 {pattern['description']}")
        print(f"       🎯 Confidence: {pattern['confidence']:.2f}")
        print()

def demonstrate_pattern_detection_logic():
    """Demonstrate how pattern detection logic works."""
    print("=" * 80)
    print("🔍 PATTERN DETECTION LOGIC EXPLANATION")
    print("=" * 80)
    
    # Load pattern definitions
    with open('CombinedAnalysis/enhanced_architecture_analysis.json', 'r') as f:
        enhanced_data = json.load(f)
    
    pattern_definitions = enhanced_data['pattern_definitions']
    
    # Focus on Monolithic Application pattern
    monolithic_pattern = pattern_definitions['monolithic']
    
    print(f"\n🏗️  PATTERN: {monolithic_pattern['name']}")
    print(f"📝 Description: {monolithic_pattern['description']}")
    
    indicators = monolithic_pattern['indicators']
    print(f"\n🔍 DETECTION INDICATORS:")
    
    if 'complexity_range' in indicators:
        min_comp, max_comp = indicators['complexity_range']
        print(f"   • Complexity Range: {min_comp:.1f} - {max_comp:.1f}")
    
    if 'file_count_range' in indicators:
        min_files, max_files = indicators['file_count_range']
        print(f"   • File Count Range: {min_files} - {max_files}")
    
    if 'method_count_range' in indicators:
        min_methods, max_methods = indicators['method_count_range']
        print(f"   • Method Count Range: {min_methods} - {max_methods}")
    
    if indicators.get('single_language', False):
        print(f"   • Single Language: Required")
    
    # Show how this applies to ibecir/oop-1002
    print(f"\n📊 APPLYING TO ibecir/oop-1002:")
    repo_data = enhanced_data['detailed_analysis']['ibecir/oop-1002']
    
    complexity = repo_data['combined_metrics']['enhanced_complexity']
    file_count = repo_data['codebert_metrics']['num_files']
    method_count = repo_data['ast_metrics']['total_methods']
    languages = repo_data['ast_metrics']['languages']
    
    print(f"   • Actual Complexity: {complexity:.3f} {'✅' if min_comp <= complexity <= max_comp else '❌'}")
    print(f"   • Actual File Count: {file_count} {'✅' if min_files <= file_count <= max_files else '❌'}")
    print(f"   • Actual Method Count: {method_count} {'✅' if min_methods <= method_count <= max_methods else '❌'}")
    print(f"   • Single Language: {len(languages) == 1} {'✅' if len(languages) == 1 else '❌'}")
    
    # Calculate confidence manually
    confidence = 0.0
    total_weight = 0.0
    
    if min_comp <= complexity <= max_comp:
        confidence += 0.3
    total_weight += 0.3
    
    if min_files <= file_count <= max_files:
        confidence += 0.2
    total_weight += 0.2
    
    if min_methods <= method_count <= max_methods:
        confidence += 0.2
    total_weight += 0.2
    
    if len(languages) == 1:
        confidence += 0.1
    total_weight += 0.1
    
    final_confidence = confidence / total_weight if total_weight > 0 else 0.0
    print(f"\n🎯 CALCULATED CONFIDENCE: {final_confidence:.2f}")

def demonstrate_confidence_calculation():
    """Demonstrate confidence calculation methodology."""
    print("=" * 80)
    print("🎯 CONFIDENCE CALCULATION METHODOLOGY")
    print("=" * 80)
    
    print(f"\n📊 CONFIDENCE WEIGHTS:")
    print(f"   • Complexity Range Match: 30% weight")
    print(f"   • File Count Range Match: 20% weight")
    print(f"   • Method Count Range Match: 20% weight")
    print(f"   • Semantic Richness Match: 15% weight")
    print(f"   • Language Requirements: 10% weight")
    print(f"   • Framework Indicators: 5% weight")
    
    print(f"\n🔧 CALCULATION PROCESS:")
    print(f"   1. For each indicator, check if the actual value falls within the expected range")
    print(f"   2. If it matches, add the corresponding weight to the confidence score")
    print(f"   3. Sum all weights to get the total possible score")
    print(f"   4. Divide actual confidence by total weight to get normalized confidence")
    print(f"   5. Only patterns with confidence > 0.3 are considered detected")
    
    print(f"\n📈 CONFIDENCE INTERPRETATION:")
    print(f"   • 0.9 - 1.0: Very High Confidence (Perfect match)")
    print(f"   • 0.7 - 0.9: High Confidence (Strong match)")
    print(f"   • 0.5 - 0.7: Medium Confidence (Good match)")
    print(f"   • 0.3 - 0.5: Low Confidence (Weak match)")
    print(f"   • < 0.3: Not detected (Below threshold)")

def demonstrate_multiple_patterns():
    """Demonstrate how multiple patterns can be detected for one repository."""
    print("=" * 80)
    print("🔍 MULTIPLE PATTERN DETECTION")
    print("=" * 80)
    
    # Load enhanced analysis
    with open('CombinedAnalysis/enhanced_architecture_analysis.json', 'r') as f:
        enhanced_data = json.load(f)
    
    # Show all repositories and their detected patterns
    print(f"\n📋 ALL REPOSITORIES AND THEIR PATTERNS:")
    
    for repo in enhanced_data['summary']:
        print(f"\n📁 {repo['repository']}")
        print(f"   🏗️  Primary: {repo['primary_architecture_pattern']} (confidence: {repo['pattern_confidence']:.2f})")
        print(f"   📊 Total Patterns: {repo['total_patterns_detected']}")
        print(f"   📈 Quality Score: {repo['overall_quality']:.3f}")
        
        # Show all detected patterns for this repository
        repo_data = enhanced_data['detailed_analysis'][repo['repository']]
        patterns = repo_data['enhanced_architecture_patterns']['all_patterns']
        
        if patterns:
            print(f"   🔍 All Patterns:")
            for i, pattern in enumerate(patterns[:5], 1):  # Show top 5
                print(f"      {i}. {pattern['name']} ({pattern['confidence']:.2f})")

def main():
    """Main demonstration function."""
    print("🚀 DETAILED ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Run demonstrations
    demonstrate_detailed_analysis()
    demonstrate_pattern_detection_logic()
    demonstrate_confidence_calculation()
    demonstrate_multiple_patterns()
    
    print(f"\n" + "=" * 80)
    print("✅ DETAILED DEMONSTRATION COMPLETED")
    print("=" * 80)
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Pattern detection uses multiple metrics and indicators")
    print(f"   • Confidence is calculated using weighted scoring")
    print(f"   • Multiple patterns can be detected for the same repository")
    print(f"   • The system provides both structural (AST) and semantic (CodeBERT) analysis")
    print(f"   • This enables comprehensive architectural pattern identification")

if __name__ == "__main__":
    main()

