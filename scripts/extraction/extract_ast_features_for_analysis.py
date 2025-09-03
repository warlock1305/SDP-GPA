"""
AST Feature Extractor for Repository Quality Analysis
====================================================

This script extracts AST path contexts as features for:
- NLP models for repository classification
- CodeBERT for code understanding
- Heuristics for quality assessment
- Repository type prediction and consistency analysis
"""

import os
import numpy as np
import json
import pandas as pd
from collections import defaultdict, Counter
import re
import subprocess
import tempfile
import yaml

# === CONFIGURATION ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "ASTFeaturesForAnalysis")
ASTMINER_JAR = os.path.join(BASE_DIR, "astminer-0.9.0", "build", "libs", "astminer.jar")

# Supported file extensions
SUPPORTED_EXTENSIONS = {".java", ".py", ".js", ".c", ".cpp", ".cc"}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_ast_features_from_repo(repo_path, output_path):
    """Extract AST features from a repository using astminer."""
    os.makedirs(output_path, exist_ok=True)
    
    # Detect languages in the repository
    langs = detect_languages_in_repo(repo_path)
    if not langs:
        return False
    
    # Extract features for each language
    for lang in langs:
        lang_output = os.path.join(output_path, lang)
        success = run_astminer_extraction(lang, repo_path, lang_output)
        if not success:
            print(f"[ERROR] Failed to extract {lang} features from {repo_path}")
    
    return True


def detect_languages_in_repo(repo_path):
    """Detect programming languages present in a repository."""
    langs = set()
    
    for root, _, files in os.walk(repo_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                if ext == ".java":
                    langs.add("java")
                elif ext == ".py":
                    langs.add("py")
                elif ext == ".js":
                    langs.add("js")
                elif ext in [".c", ".cpp", ".cc"]:
                    langs.add("cpp")
    
    return langs


def run_astminer_extraction(lang, repo_path, output_path):
    """Run astminer to extract AST features."""
    config = {
        "inputDir": repo_path,
        "outputDir": output_path,
        "parser": {
            "name": "antlr",
            "languages": [lang]
        },
        "filters": [
            {
                "name": "by tree size",
                "maxTreeSize": 1000
            }
        ],
        "label": {
            "name": "file name"
        },
        "storage": {
            "name": "code2seq",
            "length": 9,
            "width": 2
        },
        "numOfThreads": 1
    }
    
    # Save config to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name
    
    try:
        cmd = ["java", "-jar", ASTMINER_JAR, tmp_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[SUCCESS] {repo_path} [{lang}] → {output_path}")
            return True
        else:
            print(f"[ERROR] {repo_path} [{lang}] → {result.stderr.strip()}")
            return False
    finally:
        os.remove(tmp_path)


def extract_path_context_features(astminer_output_path):
    """Extract features from astminer output."""
    features = {
        "path_contexts": [],
        "node_types": set(),
        "tokens": set(),
        "method_count": 0,
        "avg_path_length": 0,
        "path_diversity": 0
    }
    
    # Look for path contexts file
    path_contexts_file = os.path.join(astminer_output_path, "data", "path_contexts.c2s")
    if not os.path.exists(path_contexts_file):
        return features
    
    try:
        with open(path_contexts_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse path context line
                parts = line.split(' ')
                if len(parts) < 2:
                    continue
                
                method_name = parts[0]
                path_contexts = parts[1:]
                
                features["method_count"] += 1
                
                for context in path_contexts:
                    if ',' in context:
                        # Parse: start_token,path,end_token
                        context_parts = context.split(',')
                        if len(context_parts) >= 3:
                            start_token = context_parts[0]
                            path = context_parts[1]
                            end_token = context_parts[2]
                            
                            features["tokens"].add(start_token)
                            features["tokens"].add(end_token)
                            
                            # Extract node types from path
                            path_nodes = path.split('^') + path.split('_')
                            for node in path_nodes:
                                if node and not node.startswith('(') and not node.endswith(')'):
                                    features["node_types"].add(node)
                            
                            features["path_contexts"].append({
                                "start_token": start_token,
                                "path": path,
                                "end_token": end_token
                            })
    except Exception as e:
        print(f"[ERROR] Reading {path_contexts_file}: {e}")
    
    # Calculate statistics
    if features["path_contexts"]:
        path_lengths = [len(ctx["path"].split('^')) + len(ctx["path"].split('_')) for ctx in features["path_contexts"]]
        features["avg_path_length"] = np.mean(path_lengths)
        
        # Calculate path diversity (unique paths / total paths)
        unique_paths = set(ctx["path"] for ctx in features["path_contexts"])
        features["path_diversity"] = len(unique_paths) / len(features["path_contexts"])
    
    # Convert sets to lists for JSON serialization
    features["node_types"] = list(features["node_types"])
    features["tokens"] = list(features["tokens"])
    
    return features


def create_repository_features():
    """Create comprehensive repository features for analysis."""
    print("=== AST Feature Extraction for Repository Analysis ===")
    
    all_repo_features = {}
    
    # Process each repository
    for user in os.listdir(DATASET_ROOT):
        user_path = os.path.join(DATASET_ROOT, user)
        if not os.path.isdir(user_path):
            continue
        
        for repo in os.listdir(user_path):
            repo_path = os.path.join(user_path, repo)
            if not os.path.isdir(repo_path):
                continue
            
            print(f"\n[PROCESSING] {user}/{repo}")
            
            # Extract AST features
            output_path = os.path.join(OUTPUT_DIR, user, repo)
            success = extract_ast_features_from_repo(repo_path, output_path)
            
            if success:
                # Extract features from astminer output
                repo_features = {
                    "repository": f"{user}/{repo}",
                    "languages": [],
                    "total_methods": 0,
                    "total_path_contexts": 0,
                    "unique_node_types": 0,
                    "unique_tokens": 0,
                    "avg_path_length": 0,
                    "avg_path_diversity": 0,
                    "language_features": {}
                }
                
                # Process each language
                for lang_dir in os.listdir(output_path):
                    lang_path = os.path.join(output_path, lang_dir)
                    if os.path.isdir(lang_path):
                        # The astminer creates a nested structure: lang/lang/data/path_contexts.c2s
                        nested_lang_path = os.path.join(lang_path, lang_dir)
                        if os.path.isdir(nested_lang_path):
                            lang_features = extract_path_context_features(nested_lang_path)
                        else:
                            lang_features = extract_path_context_features(lang_path)
                        
                        repo_features["languages"].append(lang_dir)
                        repo_features["total_methods"] += lang_features["method_count"]
                        repo_features["total_path_contexts"] += len(lang_features["path_contexts"])
                        repo_features["unique_node_types"] = max(repo_features["unique_node_types"], len(lang_features["node_types"]))
                        repo_features["unique_tokens"] = max(repo_features["unique_tokens"], len(lang_features["tokens"]))
                        
                        if lang_features["avg_path_length"] > 0:
                            repo_features["avg_path_length"] = max(repo_features["avg_path_length"], lang_features["avg_path_length"])
                        
                        if lang_features["path_diversity"] > 0:
                            repo_features["avg_path_diversity"] = max(repo_features["avg_path_diversity"], lang_features["path_diversity"])
                        
                        repo_features["language_features"][lang_dir] = lang_features
                
                all_repo_features[f"{user}/{repo}"] = repo_features
    
    return all_repo_features


def analyze_project_types(repo_features):
    """Analyze and classify project types based on AST features."""
    print("\n=== Project Type Analysis ===")
    
    project_analysis = {}
    
    for repo_name, features in repo_features.items():
        analysis = {
            "repository": repo_name,
            "project_type": "unknown",
            "confidence": 0.0,
            "indicators": []
        }
        
        # Analyze based on AST features
        indicators = []
        
        # Language-based indicators
        if "java" in features["languages"]:
            if features["total_methods"] > 50:
                indicators.append(("enterprise_application", 0.8))
            elif features["total_methods"] > 20:
                indicators.append(("library_framework", 0.6))
            else:
                indicators.append(("educational", 0.7))
        
        if "py" in features["languages"]:
            if features["avg_path_length"] > 5:
                indicators.append(("data_science", 0.7))
            elif features["total_path_contexts"] > 100:
                indicators.append(("web_application", 0.6))
            else:
                indicators.append(("script_tool", 0.8))
        
        if "js" in features["languages"]:
            if features["unique_tokens"] > 200:
                indicators.append(("web_application", 0.8))
            else:
                indicators.append(("frontend_tool", 0.6))
        
        # Complexity-based indicators
        if features["avg_path_length"] > 7:
            indicators.append(("complex_system", 0.7))
        
        if features.get("avg_path_diversity", 0) > 0.8:
            indicators.append(("diverse_functionality", 0.6))
        
        if features["total_methods"] > 100:
            indicators.append(("large_project", 0.8))
        
        # Determine project type
        if indicators:
            # Group similar indicators
            type_scores = defaultdict(float)
            for project_type, confidence in indicators:
                type_scores[project_type] += confidence
            
            # Get the most likely type
            best_type = max(type_scores.items(), key=lambda x: x[1])
            analysis["project_type"] = best_type[0]
            analysis["confidence"] = min(best_type[1], 1.0)
        
        analysis["indicators"] = indicators
        project_analysis[repo_name] = analysis
    
    return project_analysis


def create_quality_metrics(repo_features):
    """Create quality metrics based on AST features."""
    print("\n=== Quality Metrics Analysis ===")
    
    quality_metrics = {}
    
    for repo_name, features in repo_features.items():
        metrics = {
            "repository": repo_name,
            "code_complexity": 0.0,
            "code_maintainability": 0.0,
            "code_consistency": 0.0,
            "overall_quality": 0.0
        }
        
        # Complexity metric (higher path length = more complex)
        complexity_score = min(features["avg_path_length"] / 10.0, 1.0)
        metrics["code_complexity"] = complexity_score
        
        # Maintainability metric (more methods = harder to maintain, but diversity is good)
        maintainability_score = 0.0
        if features["total_methods"] > 0:
            # Balance between method count and path diversity
            method_factor = min(features["total_methods"] / 50.0, 1.0)
            diversity_factor = features.get("avg_path_diversity", 0)
            maintainability_score = (diversity_factor * 0.7) + ((1.0 - method_factor) * 0.3)
        metrics["code_maintainability"] = maintainability_score
        
        # Consistency metric (more unique node types = less consistent)
        consistency_score = 1.0 - min(features["unique_node_types"] / 100.0, 1.0)
        metrics["code_consistency"] = consistency_score
        
        # Overall quality (weighted average)
        overall_quality = (
            0.3 * (1.0 - complexity_score) +  # Lower complexity is better
            0.4 * maintainability_score +
            0.3 * consistency_score
        )
        metrics["overall_quality"] = overall_quality
        
        quality_metrics[repo_name] = metrics
    
    return quality_metrics


def save_analysis_results(repo_features, project_analysis, quality_metrics):
    """Save all analysis results."""
    print("\n=== Saving Analysis Results ===")
    
    # Convert all sets to lists for JSON serialization
    def convert_sets_to_lists(obj):
        if isinstance(obj, dict):
            return {k: convert_sets_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets_to_lists(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj
    
    repo_features_serializable = convert_sets_to_lists(repo_features)
    
    # Save repository features
    with open(os.path.join(OUTPUT_DIR, "repository_features.json"), "w") as f:
        json.dump(repo_features_serializable, f, indent=2)
    
    # Save project analysis
    with open(os.path.join(OUTPUT_DIR, "project_analysis.json"), "w") as f:
        json.dump(project_analysis, f, indent=2)
    
    # Save quality metrics
    with open(os.path.join(OUTPUT_DIR, "quality_metrics.json"), "w") as f:
        json.dump(quality_metrics, f, indent=2)
    
    # Create summary CSV
    summary_data = []
    for repo_name in repo_features.keys():
        summary_data.append({
            "repository": repo_name,
            "project_type": project_analysis[repo_name]["project_type"],
            "confidence": project_analysis[repo_name]["confidence"],
            "overall_quality": quality_metrics[repo_name]["overall_quality"],
            "code_complexity": quality_metrics[repo_name]["code_complexity"],
            "code_maintainability": quality_metrics[repo_name]["code_maintainability"],
            "code_consistency": quality_metrics[repo_name]["code_consistency"],
            "total_methods": repo_features[repo_name]["total_methods"],
            "total_path_contexts": repo_features[repo_name]["total_path_contexts"],
            "languages": ",".join(repo_features[repo_name]["languages"])
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "repository_analysis_summary.csv"), index=False)
    
    print(f"Results saved to {OUTPUT_DIR}")


def main():
    """Main function for AST feature extraction and analysis."""
    print("=== AST Feature Extraction for Repository Quality Analysis ===")
    
    # Check if astminer is available
    if not os.path.exists(ASTMINER_JAR):
        print(f"ERROR: astminer.jar not found at {ASTMINER_JAR}")
        print("Please ensure astminer is properly installed")
        return
    
    # Extract AST features from all repositories
    repo_features = create_repository_features()
    
    if not repo_features:
        print("No repository features extracted!")
        return
    
    print(f"\nExtracted features from {len(repo_features)} repositories")
    
    # Analyze project types
    project_analysis = analyze_project_types(repo_features)
    
    # Create quality metrics
    quality_metrics = create_quality_metrics(repo_features)
    
    # Save results
    save_analysis_results(repo_features, project_analysis, quality_metrics)
    
    # Print summary
    print(f"\n=== Analysis Summary ===")
    print(f"Repositories analyzed: {len(repo_features)}")
    
    # Project type distribution
    type_counts = Counter(analysis["project_type"] for analysis in project_analysis.values())
    print(f"Project types found:")
    for project_type, count in type_counts.most_common():
        print(f"  {project_type}: {count}")
    
    # Quality statistics
    quality_scores = [metrics["overall_quality"] for metrics in quality_metrics.values()]
    print(f"Quality statistics:")
    print(f"  Average quality: {np.mean(quality_scores):.3f}")
    print(f"  Median quality: {np.median(quality_scores):.3f}")
    print(f"  Best quality: {max(quality_scores):.3f}")
    print(f"  Worst quality: {min(quality_scores):.3f}")
    
    print(f"\n✅ AST feature extraction and analysis completed!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Generated files:")
    print(f"   - repository_features.json")
    print(f"   - project_analysis.json")
    print(f"   - quality_metrics.json")
    print(f"   - repository_analysis_summary.csv")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Use these features with NLP models for classification")
    print(f"   2. Feed to CodeBERT for code understanding")
    print(f"   3. Apply additional heuristics for quality assessment")
    print(f"   4. Analyze consistency between predicted and actual project types")


if __name__ == "__main__":
    main()
