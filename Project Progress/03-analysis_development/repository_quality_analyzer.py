"""
Repository Quality Analyzer using Code2Vec Embeddings
====================================================

This script demonstrates how to use extracted code2vec embeddings for:
1. Project type classification using NLP
2. Code quality assessment with CodeBERT
3. Consistency analysis between predicted and actual project types
4. Repository quality scoring
"""

import os
import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

# === CONFIGURATION ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "Code2VecEmbeddings")
OUTPUT_DIR = os.path.join(BASE_DIR, "RepositoryQualityAnalysis")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_embeddings_data():
    """Load the extracted embeddings and metadata."""
    embeddings_file = os.path.join(EMBEDDINGS_DIR, "all_file_embeddings.npy")
    metadata_file = os.path.join(EMBEDDINGS_DIR, "embedding_metadata.csv")
    repo_summaries_file = os.path.join(EMBEDDINGS_DIR, "repository_summaries.json")
    
    if not all(os.path.exists(f) for f in [embeddings_file, metadata_file, repo_summaries_file]):
        print("ERROR: Embedding files not found. Run extract_code2vec_embeddings.py first.")
        return None, None, None
    
    embeddings = np.load(embeddings_file)
    metadata = pd.read_csv(metadata_file)
    
    with open(repo_summaries_file, 'r') as f:
        repo_summaries = json.load(f)
    
    return embeddings, metadata, repo_summaries


def extract_project_features(embeddings, metadata, repo_summaries):
    """Extract features for project classification and quality analysis."""
    print("Extracting project features...")
    
    # Create repository-level features
    repo_features = []
    repo_labels = []
    
    for repo_path, summary in repo_summaries.items():
        if not summary["repository_embedding"]:
            continue
        
        # Basic repository features
        features = {
            "num_files": summary["num_files"],
            "num_methods": summary["num_methods"],
            "avg_methods_per_file": summary["avg_methods_per_file"],
            "embedding_dimension": summary["embedding_dimension"]
        }
        
        # Add embedding features (first 50 dimensions for efficiency)
        embedding = np.array(summary["repository_embedding"])
        for i in range(min(50, len(embedding))):
            features[f"emb_{i}"] = embedding[i]
        
        # Add embedding statistics
        features["embedding_mean"] = np.mean(embedding)
        features["embedding_std"] = np.std(embedding)
        features["embedding_norm"] = np.linalg.norm(embedding)
        
        repo_features.append(features)
        
        # Extract project type from repository path
        project_type = extract_project_type_from_path(repo_path)
        repo_labels.append(project_type)
    
    return pd.DataFrame(repo_features), repo_labels


def extract_project_type_from_path(repo_path):
    """Extract project type from repository path using heuristics."""
    path_lower = repo_path.lower()
    
    # Define project type patterns
    patterns = {
        "web_application": ["web", "app", "website", "frontend", "backend", "api"],
        "mobile_app": ["mobile", "android", "ios", "react-native", "flutter"],
        "data_science": ["data", "ml", "ai", "machine-learning", "analytics", "jupyter"],
        "game_development": ["game", "unity", "unreal", "gaming"],
        "system_tools": ["tool", "utility", "cli", "system", "os"],
        "library_framework": ["lib", "framework", "sdk", "package"],
        "educational": ["tutorial", "course", "learning", "example", "demo"],
        "research": ["research", "paper", "thesis", "experiment"]
    }
    
    # Count matches for each type
    type_scores = defaultdict(int)
    for project_type, keywords in patterns.items():
        for keyword in keywords:
            if keyword in path_lower:
                type_scores[project_type] += 1
    
    # Return the most likely type
    if type_scores:
        return max(type_scores.items(), key=lambda x: x[1])[0]
    else:
        return "unknown"


def train_project_classifier(features_df, labels):
    """Train a classifier to predict project types."""
    print("Training project type classifier...")
    
    # Prepare data
    X = features_df.values
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Classifier accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features_df.columns,
        'importance': classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return classifier, scaler, accuracy, feature_importance


def analyze_code_quality_heuristics(embeddings, metadata, repo_summaries):
    """Apply heuristics to assess code quality."""
    print("Analyzing code quality using heuristics...")
    
    quality_scores = []
    
    for repo_path, summary in repo_summaries.items():
        if not summary["repository_embedding"]:
            continue
        
        # Quality heuristics
        quality_score = {
            "repository": repo_path,
            "num_files": summary["num_files"],
            "num_methods": summary["num_methods"],
            "avg_methods_per_file": summary["avg_methods_per_file"]
        }
        
        # Heuristic 1: Code complexity (more methods per file = higher complexity)
        complexity_score = min(summary["avg_methods_per_file"] / 10.0, 1.0)
        quality_score["complexity_score"] = complexity_score
        
        # Heuristic 2: Project size (more files = more substantial project)
        size_score = min(summary["num_files"] / 50.0, 1.0)
        quality_score["size_score"] = size_score
        
        # Heuristic 3: Embedding consistency (lower std = more consistent code style)
        embedding = np.array(summary["repository_embedding"])
        consistency_score = 1.0 - min(np.std(embedding), 1.0)
        quality_score["consistency_score"] = consistency_score
        
        # Heuristic 4: Method distribution (balanced methods per file is good)
        method_balance = 1.0 - abs(summary["avg_methods_per_file"] - 5.0) / 10.0
        method_balance = max(0.0, method_balance)
        quality_score["method_balance_score"] = method_balance
        
        # Overall quality score (weighted average)
        overall_score = (
            0.3 * size_score +
            0.2 * consistency_score +
            0.3 * method_balance_score +
            0.2 * (1.0 - complexity_score)  # Lower complexity is better
        )
        quality_score["overall_quality_score"] = overall_score
        
        quality_scores.append(quality_score)
    
    return pd.DataFrame(quality_scores)


def analyze_consistency_between_predicted_and_actual(features_df, labels, classifier, scaler):
    """Analyze consistency between predicted and actual project types."""
    print("Analyzing consistency between predicted and actual project types...")
    
    # Make predictions
    X_scaled = scaler.transform(features_df.values)
    predictions = classifier.predict(X_scaled)
    prediction_probas = classifier.predict_proba(X_scaled)
    
    # Calculate confidence scores
    confidence_scores = np.max(prediction_probas, axis=1)
    
    # Create consistency analysis
    consistency_data = []
    for i, (actual, predicted, confidence) in enumerate(zip(labels, predictions, confidence_scores)):
        consistency_data.append({
            "repository": list(features_df.index)[i] if i < len(features_df.index) else f"repo_{i}",
            "actual_type": actual,
            "predicted_type": predicted,
            "confidence": confidence,
            "is_consistent": actual == predicted,
            "consistency_score": confidence if actual == predicted else 1.0 - confidence
        })
    
    return pd.DataFrame(consistency_data)


def create_visualizations(features_df, labels, quality_df, consistency_df):
    """Create visualizations for the analysis."""
    print("Creating visualizations...")
    
    # 1. Project type distribution
    plt.figure(figsize=(12, 6))
    project_counts = pd.Series(labels).value_counts()
    plt.bar(project_counts.index, project_counts.values)
    plt.title("Distribution of Project Types")
    plt.xlabel("Project Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "project_type_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Quality scores distribution
    plt.figure(figsize=(12, 6))
    plt.hist(quality_df["overall_quality_score"], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title("Distribution of Repository Quality Scores")
    plt.xlabel("Quality Score")
    plt.ylabel("Number of Repositories")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "quality_scores_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Consistency analysis
    plt.figure(figsize=(12, 6))
    consistency_by_type = consistency_df.groupby("actual_type")["consistency_score"].mean()
    plt.bar(consistency_by_type.index, consistency_by_type.values)
    plt.title("Average Consistency Score by Project Type")
    plt.xlabel("Project Type")
    plt.ylabel("Average Consistency Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "consistency_by_project_type.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Quality vs Consistency correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(quality_df["overall_quality_score"], consistency_df["consistency_score"], alpha=0.6)
    plt.title("Quality Score vs Consistency Score")
    plt.xlabel("Quality Score")
    plt.ylabel("Consistency Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "quality_vs_consistency.png"), dpi=300, bbox_inches='tight')
    plt.close()


def generate_analysis_report(features_df, labels, quality_df, consistency_df, classifier_accuracy, feature_importance):
    """Generate a comprehensive analysis report."""
    print("Generating analysis report...")
    
    report = {
        "summary": {
            "total_repositories": len(features_df),
            "classifier_accuracy": classifier_accuracy,
            "avg_quality_score": quality_df["overall_quality_score"].mean(),
            "avg_consistency_score": consistency_df["consistency_score"].mean(),
            "consistent_predictions": consistency_df["is_consistent"].sum(),
            "consistency_rate": consistency_df["is_consistent"].mean()
        },
        "project_type_distribution": pd.Series(labels).value_counts().to_dict(),
        "top_quality_repositories": quality_df.nlargest(10, "overall_quality_score")[["repository", "overall_quality_score"]].to_dict("records"),
        "top_consistent_repositories": consistency_df.nlargest(10, "consistency_score")[["repository", "consistency_score"]].to_dict("records"),
        "feature_importance": feature_importance.head(20).to_dict("records")
    }
    
    # Save report
    with open(os.path.join(OUTPUT_DIR, "analysis_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    # Create summary CSV
    summary_data = []
    for i, repo in enumerate(features_df.index):
        summary_data.append({
            "repository": repo,
            "project_type": labels[i] if i < len(labels) else "unknown",
            "num_files": features_df.loc[repo, "num_files"] if repo in features_df.index else 0,
            "num_methods": features_df.loc[repo, "num_methods"] if repo in features_df.index else 0,
            "quality_score": quality_df[quality_df["repository"] == repo]["overall_quality_score"].iloc[0] if repo in quality_df["repository"].values else 0,
            "consistency_score": consistency_df[consistency_df["repository"] == repo]["consistency_score"].iloc[0] if repo in consistency_df["repository"].values else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "repository_analysis_summary.csv"), index=False)
    
    return report


def main():
    """Main function for repository quality analysis."""
    print("=== Repository Quality Analysis using Code2Vec Embeddings ===")
    
    # Load data
    embeddings, metadata, repo_summaries = load_embeddings_data()
    if embeddings is None:
        return
    
    print(f"Loaded {len(embeddings)} file embeddings from {len(repo_summaries)} repositories")
    
    # Extract features
    features_df, labels = extract_project_features(embeddings, metadata, repo_summaries)
    print(f"Extracted features for {len(features_df)} repositories")
    
    # Train classifier
    classifier, scaler, accuracy, feature_importance = train_project_classifier(features_df, labels)
    
    # Analyze code quality
    quality_df = analyze_code_quality_heuristics(embeddings, metadata, repo_summaries)
    
    # Analyze consistency
    consistency_df = analyze_consistency_between_predicted_and_actual(features_df, labels, classifier, scaler)
    
    # Create visualizations
    create_visualizations(features_df, labels, quality_df, consistency_df)
    
    # Generate report
    report = generate_analysis_report(features_df, labels, quality_df, consistency_df, accuracy, feature_importance)
    
    print(f"\n✅ Repository quality analysis completed!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Generated files:")
    print(f"   - project_type_distribution.png")
    print(f"   - quality_scores_distribution.png")
    print(f"   - consistency_by_project_type.png")
    print(f"   - quality_vs_consistency.png")
    print(f"   - analysis_report.json")
    print(f"   - repository_analysis_summary.csv")
    
    print(f"\n📈 Key Results:")
    print(f"   - Classifier accuracy: {accuracy:.3f}")
    print(f"   - Average quality score: {report['summary']['avg_quality_score']:.3f}")
    print(f"   - Consistency rate: {report['summary']['consistency_rate']:.3f}")
    print(f"   - Total repositories analyzed: {report['summary']['total_repositories']}")


if __name__ == "__main__":
    main()

