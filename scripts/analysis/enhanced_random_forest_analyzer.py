"""
Enhanced Random Forest Architecture Analyzer
============================================

This script demonstrates how Random Forest works for architectural pattern detection,
showing how multiple decision trees with randomly selected features work in parallel
to classify repositories into specific architectural patterns.

Key Features:
- Parallel decision trees with random feature subsets
- Feature importance analysis for architectural patterns
- Tree-by-tree voting visualization
- Confidence scoring based on tree agreement
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedRandomForestAnalyzer:
    """Enhanced Random Forest analyzer for architectural pattern detection."""
    
    def __init__(self, models_dir: str = "ml_models"):
        """Initialize the enhanced Random Forest analyzer."""
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Model components
        self.rf_classifier = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.pattern_names = []
        
        # Analysis results
        self.tree_predictions = []
        self.feature_importance = {}
        self.voting_analysis = {}
        
    def extract_features(self, repo_data: Dict) -> np.ndarray:
        """Extract features from repository data."""
        features = []
        
        # AST Metrics (Structural features)
        ast_metrics = repo_data.get('ast_metrics', {})
        features.extend([
            ast_metrics.get('total_methods', 0),
            ast_metrics.get('total_path_contexts', 0),
            ast_metrics.get('unique_node_types', 0),
            ast_metrics.get('unique_tokens', 0),
            ast_metrics.get('avg_path_length', 0.0),
            ast_metrics.get('avg_path_diversity', 0.0),
            len(ast_metrics.get('languages', []))
        ])
        
        # CodeBERT Metrics (Semantic features)
        codebert_metrics = repo_data.get('codebert_metrics', {})
        features.extend([
            codebert_metrics.get('num_files', 0),
            codebert_metrics.get('embedding_dimension', 0)
        ])
        
        # Combined Metrics (Quality features)
        combined_metrics = repo_data.get('combined_metrics', {})
        features.extend([
            combined_metrics.get('enhanced_complexity', 0.0),
            combined_metrics.get('enhanced_maintainability', 0.0),
            combined_metrics.get('semantic_richness', 0.0),
            combined_metrics.get('technology_diversity', 0.0),
            combined_metrics.get('overall_quality', 0.0)
        ])
        
        # Engineered Features
        features.extend([
            ast_metrics.get('total_methods', 0) / max(codebert_metrics.get('num_files', 1), 1),
            ast_metrics.get('total_path_contexts', 0) / max(ast_metrics.get('total_methods', 1), 1),
            ast_metrics.get('unique_tokens', 0) / max(ast_metrics.get('total_methods', 1), 1),
            ast_metrics.get('avg_path_diversity', 0.0) * ast_metrics.get('total_path_contexts', 0),
            np.log(max(codebert_metrics.get('num_files', 1), 1)),
            np.log(max(ast_metrics.get('total_methods', 1), 1))
        ])
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get descriptive feature names."""
        return [
            # AST Metrics
            "total_methods", "total_path_contexts", "unique_node_types", 
            "unique_tokens", "avg_path_length", "avg_path_diversity", "language_count",
            
            # CodeBERT Metrics
            "num_files", "embedding_dimension",
            
            # Combined Metrics
            "enhanced_complexity", "enhanced_maintainability", "semantic_richness",
            "technology_diversity", "overall_quality",
            
            # Engineered Features
            "methods_per_file_ratio", "paths_per_method_ratio", "tokens_per_method_ratio",
            "diversity_complexity_product", "log_file_count", "log_method_count"
        ]
    
    def get_pattern_names(self) -> List[str]:
        """Get architectural pattern names."""
        return [
            "monolithic", "microservices", "serverless", "mvc_pattern", 
            "clean_architecture", "mvvm_pattern", "react_application",
            "angular_application", "django_application", "spring_application",
            "data_science_project", "blockchain_project", "iot_project",
            "mobile_app", "singleton_pattern", "factory_pattern", "observer_pattern",
            "utility_script", "api_project", "cli_tool", "library_project",
            "testing_project", "documentation_project"
        ]
    
    def train_enhanced_random_forest(self, analysis_data: Dict, n_trees: int = 100) -> Dict:
        """Train enhanced Random Forest with detailed analysis."""
        print("🌲 TRAINING ENHANCED RANDOM FOREST")
        print("=" * 60)
        
        # Prepare data
        features_list = []
        pattern_labels = []
        
        self.feature_names = self.get_feature_names()
        self.pattern_names = self.get_pattern_names()
        
        for repo_name, repo_data in analysis_data.get('detailed_analysis', {}).items():
            # Extract features
            features = self.extract_features(repo_data)
            features_list.append(features)
            
            # Extract pattern labels (multi-label)
            detected_patterns = repo_data.get('enhanced_architecture_patterns', {}).get('all_patterns', [])
            pattern_label = [0] * len(self.pattern_names)
            
            for pattern in detected_patterns:
                pattern_name = pattern.get('name', '').lower().replace(' ', '_')
                if pattern_name in self.pattern_names:
                    idx = self.pattern_names.index(pattern_name)
                    pattern_label[idx] = 1
            
            pattern_labels.append(pattern_label)
        
        X = np.array(features_list)
        y = np.array(pattern_labels)
        
        print(f"📊 Dataset prepared:")
        print(f"   • Features: {X.shape[1]} dimensions")
        print(f"   • Samples: {X.shape[0]} repositories")
        print(f"   • Patterns: {y.shape[1]} architectural patterns")
        print(f"   • Trees: {n_trees} parallel decision trees")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        base_rf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',  # Random feature selection
            bootstrap=True,       # Random sampling with replacement
            random_state=42,
            n_jobs=-1,           # Parallel processing
            verbose=0
        )
        
        self.rf_classifier = MultiOutputClassifier(base_rf)
        self.rf_classifier.fit(X_scaled, y)
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Save model
        model_path = os.path.join(self.models_dir, 'enhanced_rf_classifier.joblib')
        scaler_path = os.path.join(self.models_dir, 'enhanced_rf_scaler.joblib')
        
        joblib.dump(self.rf_classifier, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"💾 Enhanced Random Forest saved to {model_path}")
        
        return {
            "n_trees": n_trees,
            "n_features": X.shape[1],
            "n_patterns": y.shape[1],
            "n_samples": X.shape[0]
        }
    
    def analyze_feature_importance(self):
        """Analyze feature importance across all trees."""
        print("\n🔍 ANALYZING FEATURE IMPORTANCE")
        print("=" * 40)
        
        # Get feature importance from each pattern classifier
        for i, estimator in enumerate(self.rf_classifier.estimators_):
            pattern_name = self.pattern_names[i]
            importance = estimator.feature_importances_
            
            # Store top 5 features for each pattern
            feature_importance = list(zip(self.feature_names, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            self.feature_importance[pattern_name] = feature_importance[:5]
            
            print(f"\n🏗️  {pattern_name.upper()} - Top Features:")
            for j, (feature, imp) in enumerate(feature_importance[:5], 1):
                print(f"   {j}. {feature}: {imp:.4f}")
    
    def predict_with_tree_analysis(self, repo_data: Dict) -> Dict:
        """Predict patterns with detailed tree-by-tree analysis."""
        if self.rf_classifier is None:
            raise ValueError("Random Forest not trained. Call train_enhanced_random_forest first.")
        
        # Extract features
        features = self.extract_features(repo_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions from each tree
        tree_predictions = []
        pattern_votes = {pattern: 0 for pattern in self.pattern_names}
        
        print(f"\n🌳 TREE-BY-TREE ANALYSIS")
        print("=" * 50)
        
        # Analyze each pattern classifier (each has multiple trees)
        for i, estimator in enumerate(self.rf_classifier.estimators_):
            pattern_name = self.pattern_names[i]
            
            # Get predictions from all trees in this estimator
            tree_preds = []
            for tree in estimator.estimators_:
                pred = tree.predict(features_scaled)[0]
                tree_preds.append(pred)
                if pred == 1:
                    pattern_votes[pattern_name] += 1
            
            tree_predictions.append({
                "pattern": pattern_name,
                "tree_predictions": tree_preds,
                "positive_votes": pattern_votes[pattern_name],
                "total_trees": len(estimator.estimators_),
                "confidence": pattern_votes[pattern_name] / len(estimator.estimators_)
            })
        
        # Get final predictions
        final_pred = self.rf_classifier.predict(features_scaled)[0]
        final_proba = self.rf_classifier.predict_proba(features_scaled)
        
        # Format results
        detected_patterns = []
        for i, (pred, proba) in enumerate(zip(final_pred, final_proba)):
            if pred == 1:
                pattern_name = self.pattern_names[i]
                confidence = np.max(proba) if len(proba) > 0 else 0.5
                
                detected_patterns.append({
                    "name": pattern_name.replace('_', ' ').title(),
                    "confidence": confidence,
                    "tree_votes": tree_predictions[i]["positive_votes"],
                    "total_trees": tree_predictions[i]["total_trees"],
                    "tree_confidence": tree_predictions[i]["confidence"]
                })
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "primary_pattern": detected_patterns[0] if detected_patterns else {"name": "Unknown", "confidence": 0.0},
            "all_patterns": detected_patterns,
            "pattern_count": len(detected_patterns),
            "tree_analysis": tree_predictions,
            "voting_summary": pattern_votes
        }
    
    def visualize_tree_voting(self, repo_name: str, tree_analysis: List[Dict]):
        """Visualize how trees voted for each pattern."""
        print(f"\n📊 TREE VOTING VISUALIZATION FOR {repo_name}")
        print("=" * 60)
        
        # Create voting summary
        voting_data = []
        for analysis in tree_analysis:
            pattern = analysis["pattern"]
            votes = analysis["positive_votes"]
            total = analysis["total_trees"]
            confidence = analysis["confidence"]
            
            voting_data.append({
                "Pattern": pattern.replace('_', ' ').title(),
                "Votes": votes,
                "Total Trees": total,
                "Confidence": confidence
            })
        
        # Sort by votes
        voting_data.sort(key=lambda x: x["Votes"], reverse=True)
        
        # Display top patterns
        print(f"\n🏆 TOP PATTERNS BY TREE VOTES:")
        for i, data in enumerate(voting_data[:10], 1):
            percentage = (data["Votes"] / data["Total Trees"]) * 100
            print(f"   {i:2d}. {data['Pattern']:<25} {data['Votes']:3d}/{data['Total Trees']:3d} trees ({percentage:5.1f}%)")
        
        return voting_data
    
    def demonstrate_parallel_trees(self, repo_data: Dict):
        """Demonstrate how parallel trees work with different feature subsets."""
        print(f"\n🌲 PARALLEL DECISION TREES DEMONSTRATION")
        print("=" * 60)
        
        # Get predictions with tree analysis
        results = self.predict_with_tree_analysis(repo_data)
        
        print(f"\n🔍 HOW PARALLEL TREES WORK:")
        print(f"   • Each tree sees a random subset of features")
        print(f"   • Trees make independent decisions")
        print(f"   • Final prediction is majority vote")
        
        # Show example tree decisions
        print(f"\n📋 EXAMPLE TREE DECISIONS:")
        for i, analysis in enumerate(results["tree_analysis"][:5], 1):
            pattern = analysis["pattern"]
            votes = analysis["positive_votes"]
            total = analysis["total_trees"]
            confidence = analysis["confidence"]
            
            print(f"\n   Tree Group {i} - {pattern.upper()}:")
            print(f"   • Trees that voted YES: {votes}/{total}")
            print(f"   • Confidence: {confidence:.3f}")
            
            # Show what features this pattern focuses on
            if pattern in self.feature_importance:
                print(f"   • Key features: {', '.join([f[0] for f in self.feature_importance[pattern][:3]])}")
        
        return results

def demonstrate_enhanced_random_forest():
    """Demonstrate enhanced Random Forest analysis."""
    print("🚀 ENHANCED RANDOM FOREST ARCHITECTURAL PATTERN DETECTION")
    print("=" * 80)
    
    # Load analysis data
    try:
        with open('CombinedAnalysis/enhanced_architecture_analysis.json', 'r') as f:
            analysis_data = json.load(f)
    except FileNotFoundError:
        print("❌ Enhanced architecture analysis not found. Run enhanced analysis first.")
        return
    
    # Initialize enhanced analyzer
    analyzer = EnhancedRandomForestAnalyzer()
    
    # Train enhanced Random Forest
    training_info = analyzer.train_enhanced_random_forest(analysis_data, n_trees=100)
    
    print(f"\n✅ ENHANCED RANDOM FOREST TRAINED SUCCESSFULLY")
    print(f"   • {training_info['n_trees']} parallel decision trees")
    print(f"   • {training_info['n_features']} features per tree (randomly selected)")
    print(f"   • {training_info['n_patterns']} architectural patterns")
    print(f"   • {training_info['n_samples']} training repositories")
    
    # Demonstrate on each repository
    print(f"\n🔍 DEMONSTRATING PARALLEL TREE ANALYSIS:")
    
    for repo_name, repo_data in analysis_data.get('detailed_analysis', {}).items():
        print(f"\n" + "="*80)
        print(f"📁 ANALYZING: {repo_name}")
        print("="*80)
        
        # Get detailed analysis
        results = analyzer.demonstrate_parallel_trees(repo_data)
        
        # Show voting visualization
        voting_data = analyzer.visualize_tree_voting(repo_name, results["tree_analysis"])
        
        # Show final results
        primary = results["primary_pattern"]
        print(f"\n🎯 FINAL PREDICTION:")
        print(f"   • Primary Pattern: {primary['name']}")
        print(f"   • Confidence: {primary['confidence']:.3f}")
        print(f"   • Tree Agreement: {primary.get('tree_votes', 0)}/100 trees")
        
        print(f"\n📋 ALL DETECTED PATTERNS:")
        for i, pattern in enumerate(results["all_patterns"][:5], 1):
            print(f"   {i}. {pattern['name']} (confidence: {pattern['confidence']:.3f}, votes: {pattern['tree_votes']})")

if __name__ == "__main__":
    demonstrate_enhanced_random_forest()
