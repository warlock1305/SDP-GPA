"""
Combined Repository Analyzer: AST Features + CodeBERT Embeddings
================================================================

This script combines both approaches for comprehensive repository analysis:
1. AST Features (structural analysis) - from extract_ast_features_for_analysis.py
2. CodeBERT Embeddings (semantic analysis) - from extract_codebert_embeddings.py

Provides:
- Enhanced project type classification
- Comprehensive quality metrics
- Semantic similarity analysis
- Architecture pattern detection
- Framework and technology identification
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# === CONFIGURATION ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
AST_FEATURES_DIR = os.path.join(BASE_DIR, "ASTFeaturesForAnalysis")
CODEBERT_DIR = os.path.join(BASE_DIR, "CodeBERTEmbeddings")
OUTPUT_DIR = os.path.join(BASE_DIR, "CombinedAnalysis")

os.makedirs(OUTPUT_DIR, exist_ok=True)


class CombinedRepositoryAnalyzer:
    """Combined analyzer using AST features and CodeBERT embeddings."""
    
    def __init__(self):
        """Initialize the combined analyzer."""
        self.ast_features = {}
        self.codebert_embeddings = {}
        self.combined_analysis = {}
        
    def load_ast_features(self):
        """Load AST features from the analysis results."""
        print("Loading AST features...")
        
        # Load repository features
        ast_features_file = os.path.join(AST_FEATURES_DIR, "repository_features.json")
        if os.path.exists(ast_features_file):
            with open(ast_features_file, 'r') as f:
                self.ast_features = json.load(f)
            print(f"Loaded AST features for {len(self.ast_features)} repositories")
        else:
            print("AST features file not found!")
            return False
        
        # Load project analysis
        project_analysis_file = os.path.join(AST_FEATURES_DIR, "project_analysis.json")
        if os.path.exists(project_analysis_file):
            with open(project_analysis_file, 'r') as f:
                self.project_analysis = json.load(f)
        
        # Load quality metrics
        quality_metrics_file = os.path.join(AST_FEATURES_DIR, "quality_metrics.json")
        if os.path.exists(quality_metrics_file):
            with open(quality_metrics_file, 'r') as f:
                self.quality_metrics = json.load(f)
        
        return True
    
    def load_codebert_embeddings(self):
        """Load CodeBERT embeddings from the extraction results."""
        print("Loading CodeBERT embeddings...")
        
        # Load repository summary
        summary_file = os.path.join(CODEBERT_DIR, "repository_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            # Load individual repository embeddings
            for repo_info in summary_data:
                repo_name = repo_info["repository"]
                embedding_file = os.path.join(CODEBERT_DIR, f"{repo_name.replace('/', '_')}_embeddings.json")
                
                if os.path.exists(embedding_file):
                    with open(embedding_file, 'r') as f:
                        self.codebert_embeddings[repo_name] = json.load(f)
            
            print(f"Loaded CodeBERT embeddings for {len(self.codebert_embeddings)} repositories")
            return True
        else:
            print("CodeBERT summary file not found!")
            return False
    
    def create_combined_analysis(self):
        """Create combined analysis using both AST features and CodeBERT embeddings."""
        print("Creating combined analysis...")
        
        for repo_name in self.ast_features.keys():
            if repo_name in self.codebert_embeddings:
                combined = self._combine_repository_data(repo_name)
                self.combined_analysis[repo_name] = combined
        
        print(f"Created combined analysis for {len(self.combined_analysis)} repositories")
    
    def _combine_repository_data(self, repo_name: str) -> Dict:
        """Combine AST features and CodeBERT embeddings for a single repository."""
        ast_data = self.ast_features[repo_name]
        codebert_data = self.codebert_embeddings[repo_name]
        
        # Extract key metrics
        combined = {
            "repository": repo_name,
            
            # AST Features
            "ast_metrics": {
                "total_methods": ast_data.get("total_methods", 0),
                "total_path_contexts": ast_data.get("total_path_contexts", 0),
                "unique_node_types": ast_data.get("unique_node_types", 0),
                "unique_tokens": ast_data.get("unique_tokens", 0),
                "avg_path_length": ast_data.get("avg_path_length", 0),
                "avg_path_diversity": ast_data.get("avg_path_diversity", 0),
                "languages": ast_data.get("languages", [])
            },
            
            # CodeBERT Metrics
            "codebert_metrics": {
                "num_files": codebert_data.get("num_files", 0),
                "embedding_dimension": codebert_data.get("embedding_dimension", 0),
                "repository_embedding": codebert_data.get("repository_embedding", []),
                "file_info": codebert_data.get("file_info", [])
            },
            
            # Quality Metrics (from AST analysis)
            "quality_metrics": self.quality_metrics.get(repo_name, {}),
            
            # Project Analysis (from AST analysis)
            "project_analysis": self.project_analysis.get(repo_name, {})
        }
        
        # Calculate combined metrics
        combined["combined_metrics"] = self._calculate_combined_metrics(combined)
        
        return combined
    
    def _calculate_combined_metrics(self, repo_data: Dict) -> Dict:
        """Calculate enhanced metrics using both AST and CodeBERT data."""
        ast_metrics = repo_data["ast_metrics"]
        codebert_metrics = repo_data["codebert_metrics"]
        quality_metrics = repo_data["quality_metrics"]
        
        # Enhanced complexity score
        complexity_score = 0.0
        if ast_metrics["avg_path_length"] > 0:
            complexity_score += min(ast_metrics["avg_path_length"] / 10.0, 1.0) * 0.4
        if codebert_metrics["num_files"] > 0:
            complexity_score += min(codebert_metrics["num_files"] / 50.0, 1.0) * 0.3
        if ast_metrics["total_methods"] > 0:
            complexity_score += min(ast_metrics["total_methods"] / 100.0, 1.0) * 0.3
        
        # Enhanced maintainability score
        maintainability_score = 0.0
        if ast_metrics["avg_path_diversity"] > 0:
            maintainability_score += ast_metrics["avg_path_diversity"] * 0.4
        if quality_metrics.get("code_maintainability", 0) > 0:
            maintainability_score += quality_metrics["code_maintainability"] * 0.6
        
        # Semantic richness score (from CodeBERT)
        semantic_richness = 0.0
        if codebert_metrics["repository_embedding"]:
            embedding_norm = np.linalg.norm(codebert_metrics["repository_embedding"])
            semantic_richness = min(embedding_norm / 30.0, 1.0)
        
        # Technology diversity score
        tech_diversity = 0.0
        if ast_metrics["languages"]:
            tech_diversity = min(len(ast_metrics["languages"]) / 3.0, 1.0)
        
        # Overall quality score (weighted combination)
        overall_quality = (
            0.3 * (1.0 - complexity_score) +  # Lower complexity is better
            0.3 * maintainability_score +
            0.2 * semantic_richness +
            0.1 * tech_diversity +
            0.1 * quality_metrics.get("code_consistency", 0)
        )
        
        return {
            "enhanced_complexity": complexity_score,
            "enhanced_maintainability": maintainability_score,
            "semantic_richness": semantic_richness,
            "technology_diversity": tech_diversity,
            "overall_quality": overall_quality
        }
    
    def perform_semantic_similarity_analysis(self):
        """Perform semantic similarity analysis using CodeBERT embeddings."""
        print("Performing semantic similarity analysis...")
        
        # Extract embeddings
        embeddings = []
        repo_names = []
        
        for repo_name, data in self.combined_analysis.items():
            if data["codebert_metrics"]["repository_embedding"]:
                embeddings.append(data["codebert_metrics"]["repository_embedding"])
                repo_names.append(repo_name)
        
        if len(embeddings) < 2:
            print("Not enough embeddings for similarity analysis")
            return
        
        # Calculate similarity matrix
        embeddings_array = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Find most similar repositories
        similarity_analysis = {}
        for i, repo_name in enumerate(repo_names):
            similarities = []
            for j, other_repo in enumerate(repo_names):
                if i != j:
                    similarities.append((other_repo, similarity_matrix[i][j]))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            similarity_analysis[repo_name] = {
                "most_similar": similarities[:3],  # Top 3 most similar
                "similarity_scores": similarities
            }
        
        self.similarity_analysis = similarity_analysis
        print(f"Completed similarity analysis for {len(repo_names)} repositories")
    
    def detect_architecture_patterns(self):
        """Detect architecture patterns using combined analysis."""
        print("Detecting architecture patterns...")
        
        patterns = {}
        for repo_name, data in self.combined_analysis.items():
            ast_metrics = data["ast_metrics"]
            codebert_metrics = data["codebert_metrics"]
            combined_metrics = data["combined_metrics"]
            
            pattern_indicators = []
            
            # Large-scale enterprise pattern
            if (ast_metrics["total_methods"] > 100 and 
                codebert_metrics["num_files"] > 20 and
                ast_metrics["avg_path_length"] > 5):
                pattern_indicators.append(("enterprise_scale", 0.9))
            
            # Microservices pattern
            if (codebert_metrics["num_files"] > 10 and
                ast_metrics["avg_path_diversity"] > 0.7 and
                len(ast_metrics["languages"]) > 1):
                pattern_indicators.append(("microservices", 0.8))
            
            # Simple utility pattern
            if (ast_metrics["total_methods"] < 10 and
                codebert_metrics["num_files"] < 5 and
                ast_metrics["avg_path_length"] < 3):
                pattern_indicators.append(("utility_script", 0.9))
            
            # Framework-heavy pattern
            if (ast_metrics["unique_tokens"] > 500 and
                ast_metrics["total_path_contexts"] > 1000):
                pattern_indicators.append(("framework_based", 0.8))
            
            # Data science pattern
            if ("py" in ast_metrics["languages"] and
                ast_metrics["avg_path_length"] > 4 and
                combined_metrics["semantic_richness"] > 0.7):
                pattern_indicators.append(("data_science", 0.8))
            
            # Web application pattern
            if (("js" in ast_metrics["languages"] or "ts" in ast_metrics["languages"]) and
                codebert_metrics["num_files"] > 5 and
                ast_metrics["total_path_contexts"] > 500):
                pattern_indicators.append(("web_application", 0.8))
            
            # Determine primary pattern
            if pattern_indicators:
                primary_pattern = max(pattern_indicators, key=lambda x: x[1])
                patterns[repo_name] = {
                    "primary_pattern": primary_pattern[0],
                    "confidence": primary_pattern[1],
                    "all_indicators": pattern_indicators
                }
            else:
                patterns[repo_name] = {
                    "primary_pattern": "unknown",
                    "confidence": 0.0,
                    "all_indicators": []
                }
        
        self.architecture_patterns = patterns
        print(f"Detected architecture patterns for {len(patterns)} repositories")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        print("Generating comprehensive report...")
        
        # Create summary data
        summary_data = []
        for repo_name, data in self.combined_analysis.items():
            summary = {
                "repository": repo_name,
                "project_type": data["project_analysis"].get("project_type", "unknown"),
                "architecture_pattern": self.architecture_patterns.get(repo_name, {}).get("primary_pattern", "unknown"),
                "overall_quality": data["combined_metrics"]["overall_quality"],
                "enhanced_complexity": data["combined_metrics"]["enhanced_complexity"],
                "enhanced_maintainability": data["combined_metrics"]["enhanced_maintainability"],
                "semantic_richness": data["combined_metrics"]["semantic_richness"],
                "technology_diversity": data["combined_metrics"]["technology_diversity"],
                "total_methods": data["ast_metrics"]["total_methods"],
                "total_files": data["codebert_metrics"]["num_files"],
                "languages": ",".join(data["ast_metrics"]["languages"])
            }
            summary_data.append(summary)
        
        # Save comprehensive report
        report_file = os.path.join(OUTPUT_DIR, "comprehensive_analysis_report.json")
        with open(report_file, 'w') as f:
            json.dump({
                "summary": summary_data,
                "detailed_analysis": self.combined_analysis,
                "similarity_analysis": getattr(self, 'similarity_analysis', {}),
                "architecture_patterns": getattr(self, 'architecture_patterns', {})
            }, f, indent=2)
        
        # Save CSV summary
        df = pd.DataFrame(summary_data)
        csv_file = os.path.join(OUTPUT_DIR, "comprehensive_analysis_summary.csv")
        df.to_csv(csv_file, index=False)
        
        print(f"Comprehensive report saved to: {report_file}")
        print(f"CSV summary saved to: {csv_file}")
        
        return summary_data
    
    def create_visualizations(self):
        """Create visualizations for the combined analysis."""
        print("Creating visualizations...")
        
        # Load summary data
        csv_file = os.path.join(OUTPUT_DIR, "comprehensive_analysis_summary.csv")
        if not os.path.exists(csv_file):
            print("Summary CSV not found, skipping visualizations")
            return
        
        df = pd.read_csv(csv_file)
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Combined Repository Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Quality Distribution
        axes[0, 0].hist(df['overall_quality'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Overall Quality Distribution')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Number of Repositories')
        
        # 2. Complexity vs Maintainability
        axes[0, 1].scatter(df['enhanced_complexity'], df['enhanced_maintainability'], 
                          alpha=0.7, s=100)
        axes[0, 1].set_title('Complexity vs Maintainability')
        axes[0, 1].set_xlabel('Enhanced Complexity')
        axes[0, 1].set_ylabel('Enhanced Maintainability')
        
        # 3. Architecture Patterns
        pattern_counts = df['architecture_pattern'].value_counts()
        axes[1, 0].pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Architecture Pattern Distribution')
        
        # 4. Technology Diversity vs Semantic Richness
        axes[1, 1].scatter(df['technology_diversity'], df['semantic_richness'], 
                          alpha=0.7, s=100)
        axes[1, 1].set_title('Technology Diversity vs Semantic Richness')
        axes[1, 1].set_xlabel('Technology Diversity')
        axes[1, 1].set_ylabel('Semantic Richness')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = os.path.join(OUTPUT_DIR, "combined_analysis_visualization.png")
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {viz_file}")


def main():
    """Main function for combined repository analysis."""
    print("=== Combined Repository Analysis: AST Features + CodeBERT Embeddings ===")
    
    # Initialize analyzer
    analyzer = CombinedRepositoryAnalyzer()
    
    # Load data
    if not analyzer.load_ast_features():
        print("Failed to load AST features!")
        return
    
    if not analyzer.load_codebert_embeddings():
        print("Failed to load CodeBERT embeddings!")
        return
    
    # Create combined analysis
    analyzer.create_combined_analysis()
    
    # Perform advanced analysis
    analyzer.perform_semantic_similarity_analysis()
    analyzer.detect_architecture_patterns()
    
    # Generate reports
    summary_data = analyzer.generate_comprehensive_report()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Print summary
    print(f"\n=== Analysis Summary ===")
    print(f"Repositories analyzed: {len(summary_data)}")
    
    # Quality statistics
    quality_scores = [row['overall_quality'] for row in summary_data]
    print(f"Quality statistics:")
    print(f"  Average quality: {np.mean(quality_scores):.3f}")
    print(f"  Median quality: {np.median(quality_scores):.3f}")
    print(f"  Best quality: {max(quality_scores):.3f}")
    print(f"  Worst quality: {min(quality_scores):.3f}")
    
    # Architecture patterns
    pattern_counts = Counter([row['architecture_pattern'] for row in summary_data])
    print(f"Architecture patterns found:")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern}: {count}")
    
    print(f"\n✅ Combined analysis completed!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Generated files:")
    print(f"   - comprehensive_analysis_report.json")
    print(f"   - comprehensive_analysis_summary.csv")
    print(f"   - combined_analysis_visualization.png")
    
    print(f"\n🎯 This completes the code analysis foundation!")
    print(f"   Next steps: Commit history analysis, README analysis, text generation")


if __name__ == "__main__":
    main()
