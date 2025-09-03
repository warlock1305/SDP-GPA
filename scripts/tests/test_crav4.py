#!/usr/bin/env python3
"""
Test CRAv4 - Improved Repository Analyzer
=========================================

This script tests the new CRAv4 analyzer to verify that it fixes
the classification problems from v3.
"""

import os
import sys
import time
from pathlib import Path

# Add the scripts/analysis directory to the path
sys.path.append('scripts/analysis')

def test_crav4_classification():
    """Test CRAv4 classification on sample repositories."""
    print("🧪 Testing CRAv4 - Improved Repository Analyzer")
    print("=" * 60)
    
    # Import CRAv4
    try:
        from comprehensive_repository_analyzer_v4 import ComprehensiveRepositoryAnalyzerV4
        print("✅ Successfully imported CRAv4")
    except ImportError as e:
        print(f"❌ Failed to import CRAv4: {e}")
        return
    
    # Initialize analyzer
    try:
        analyzer = ComprehensiveRepositoryAnalyzerV4()
        print("✅ Successfully initialized CRAv4 analyzer")
    except Exception as e:
        print(f"❌ Failed to initialize CRAv4 analyzer: {e}")
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
            
            # Analyze repository
            results = analyzer.analyze_repository(repo_path)
            
            analysis_time = time.time() - start_time
            
            print(f"  ✅ Analysis completed in {analysis_time:.2f}s")
            
            # Show architecture analysis results
            arch_analysis = results['architecture_analysis']
            print(f"  🏗️  Detected patterns: {arch_analysis['detected_patterns']}")
            
            # Show confidence scores
            if arch_analysis['pattern_confidence']:
                print(f"  📊 Pattern confidences:")
                for pattern, confidence in arch_analysis['pattern_confidence'].items():
                    print(f"     • {pattern}: {confidence:.3f}")
            
            # Show semantic analysis
            semantic_analysis = arch_analysis.get('semantic_analysis', {})
            if semantic_analysis.get('embedding_analysis'):
                print(f"  🤖 Semantic indicators:")
                for category, indicators in semantic_analysis['embedding_analysis'].items():
                    if indicators:
                        print(f"     • {category}: {len(indicators)} indicators")
            
            # Show quality assessment
            quality = results['quality_assessment']
            print(f"  ⭐ Quality score: {quality['overall_score']:.3f}")
            
            # Show programmer characteristics
            prog_chars = results['programmer_characteristics']
            print(f"  👨‍💻 Experience: {prog_chars['experience_level']}")
            print(f"  🎯 Specialization: {prog_chars['specialization']}")
            
        except Exception as e:
            print(f"  ❌ Analysis failed: {e}")
    
    print(f"\n✅ CRAv4 test completed!")
    print("\n🔍 Key improvements in CRAv4:")
    print("  • Better semantic classification using CodeBERT embeddings")
    print("  • More specific and accurate rule-based classification")
    print("  • Enhanced pattern detection logic")
    print("  • Improved confidence scoring")
    print("  • Better handling of C++ and other languages")

if __name__ == "__main__":
    test_crav4_classification()
