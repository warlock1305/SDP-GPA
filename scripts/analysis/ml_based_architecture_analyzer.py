"""
Machine Learning-Based Architecture Analyzer
============================================

This script implements a machine learning approach for architectural pattern detection
and project quality assessment, replacing the rule-based system with trained models.

Features:
- Multi-label classification for architectural patterns
- Regression for quality score prediction
- Feature engineering from AST and CodeBERT data
- Model training and evaluation pipeline
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLBasedArchitectureAnalyzer:
    """Machine learning-based architecture pattern detector and quality assessor."""
    
    def __init__(self, models_dir: str = "ml_models"):
        """Initialize the ML-based analyzer."""
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Model components
        self.pattern_classifier = None
        self.quality_regressor = None
        self.pattern_scaler = StandardScaler()
        self.quality_scaler = StandardScaler()
        self.label_encoders = {}
        
        # Pattern definitions (for labeling)
        self.pattern_definitions = self._load_pattern_definitions()
        
    def _load_pattern_definitions(self) -> Dict:
        """Load pattern definitions for labeling."""
        try:
            with open('CombinedAnalysis/enhanced_architecture_analysis.json', 'r') as f:
                data = json.load(f)
            return data.get('pattern_definitions', {})
        except FileNotFoundError:
            print("Warning: Pattern definitions not found. Using default patterns.")
            return {
                "monolithic": {"name": "Monolithic Application"},
                "microservices": {"name": "Microservices"},
                "serverless": {"name": "Serverless Architecture"},
                "mvc_pattern": {"name": "Model-View-Controller"},
                "clean_architecture": {"name": "Clean Architecture"}
            }
    
    def extract_features(self, repo_data: Dict) -> np.ndarray:
        """Extract features from repository data for ML models."""
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
        
        # Additional engineered features
        features.extend([
            # Complexity ratios
            ast_metrics.get('total_methods', 0) / max(codebert_metrics.get('num_files', 1), 1),
            ast_metrics.get('total_path_contexts', 0) / max(ast_metrics.get('total_methods', 1), 1),
            
            # Diversity metrics
            ast_metrics.get('unique_tokens', 0) / max(ast_metrics.get('total_methods', 1), 1),
            ast_metrics.get('avg_path_diversity', 0.0) * ast_metrics.get('total_path_contexts', 0),
            
            # Size indicators
            np.log(max(codebert_metrics.get('num_files', 1), 1)),
            np.log(max(ast_metrics.get('total_methods', 1), 1))
        ])
        
        return np.array(features, dtype=np.float32)
    
    def create_training_dataset(self, analysis_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create training dataset from existing analysis data."""
        print("🔧 Creating training dataset from existing analysis...")
        
        features_list = []
        pattern_labels = []
        quality_scores = []
        
        # Get all pattern names for multi-label classification
        all_patterns = list(self.pattern_definitions.keys())
        
        for repo_name, repo_data in analysis_data.get('detailed_analysis', {}).items():
            # Extract features
            features = self.extract_features(repo_data)
            features_list.append(features)
            
            # Extract pattern labels (multi-label)
            detected_patterns = repo_data.get('enhanced_architecture_patterns', {}).get('all_patterns', [])
            pattern_label = [0] * len(all_patterns)
            
            for pattern in detected_patterns:
                pattern_name = pattern.get('name', '').lower().replace(' ', '_')
                if pattern_name in all_patterns:
                    idx = all_patterns.index(pattern_name)
                    pattern_label[idx] = 1
            
            pattern_labels.append(pattern_label)
            
            # Extract quality score
            quality_score = repo_data.get('combined_metrics', {}).get('overall_quality', 0.0)
            quality_scores.append(quality_score)
        
        X = np.array(features_list)
        y_patterns = np.array(pattern_labels)
        y_quality = np.array(quality_scores)
        
        print(f"📊 Dataset created:")
        print(f"   • Features: {X.shape[1]} dimensions")
        print(f"   • Samples: {X.shape[0]} repositories")
        print(f"   • Pattern labels: {y_patterns.shape[1]} patterns")
        print(f"   • Quality scores: {len(y_quality)} samples")
        
        return X, y_patterns, y_quality
    
    def train_pattern_classifier(self, X: np.ndarray, y_patterns: np.ndarray) -> None:
        """Train multi-label classifier for architectural patterns."""
        print("🤖 Training architectural pattern classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_patterns, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.pattern_scaler.fit_transform(X_train)
        X_test_scaled = self.pattern_scaler.transform(X_test)
        
        # Initialize classifier
        base_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.pattern_classifier = MultiOutputClassifier(base_classifier)
        
        # Train model
        self.pattern_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.pattern_classifier.predict(X_test_scaled)
        y_pred_proba = self.pattern_classifier.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        print(f"📈 Pattern Classification Results:")
        print(f"   • Accuracy: {accuracy:.3f}")
        
        # Save model
        model_path = os.path.join(self.models_dir, 'pattern_classifier.joblib')
        scaler_path = os.path.join(self.models_dir, 'pattern_scaler.joblib')
        
        joblib.dump(self.pattern_classifier, model_path)
        joblib.dump(self.pattern_scaler, scaler_path)
        
        print(f"💾 Model saved to {model_path}")
        
        return accuracy
    
    def train_quality_regressor(self, X: np.ndarray, y_quality: np.ndarray) -> None:
        """Train regressor for quality score prediction."""
        print("🤖 Training quality score regressor...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_quality, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.quality_scaler.fit_transform(X_train)
        X_test_scaled = self.quality_scaler.transform(X_test)
        
        # Initialize regressor
        self.quality_regressor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Train model
        self.quality_regressor.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.quality_regressor.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"📈 Quality Regression Results:")
        print(f"   • R² Score: {r2:.3f}")
        print(f"   • Mean Squared Error: {mse:.4f}")
        print(f"   • Root Mean Squared Error: {np.sqrt(mse):.4f}")
        
        # Save model
        model_path = os.path.join(self.models_dir, 'quality_regressor.joblib')
        scaler_path = os.path.join(self.models_dir, 'quality_scaler.joblib')
        
        joblib.dump(self.quality_regressor, model_path)
        joblib.dump(self.quality_scaler, scaler_path)
        
        print(f"💾 Model saved to {model_path}")
        
        return r2
    
    def predict_patterns(self, repo_data: Dict) -> Dict:
        """Predict architectural patterns for a repository."""
        if self.pattern_classifier is None:
            raise ValueError("Pattern classifier not trained. Call train_pattern_classifier first.")
        
        # Extract features
        features = self.extract_features(repo_data)
        features_scaled = self.pattern_scaler.transform(features.reshape(1, -1))
        
        # Predict patterns
        pattern_probs = self.pattern_classifier.predict_proba(features_scaled)
        pattern_pred = self.pattern_classifier.predict(features_scaled)
        
        # Get pattern names
        all_patterns = list(self.pattern_definitions.keys())
        
        # Format results
        detected_patterns = []
        for i, (pred, probs) in enumerate(zip(pattern_pred[0], pattern_probs)):
            if pred == 1:
                pattern_name = all_patterns[i]
                confidence = np.max(probs) if len(probs) > 0 else 0.5
                
                detected_patterns.append({
                    "name": self.pattern_definitions[pattern_name]["name"],
                    "confidence": confidence,
                    "description": self.pattern_definitions[pattern_name].get("description", "")
                })
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "primary_pattern": detected_patterns[0] if detected_patterns else {"name": "Unknown", "confidence": 0.0},
            "all_patterns": detected_patterns,
            "pattern_count": len(detected_patterns)
        }
    
    def predict_quality(self, repo_data: Dict) -> float:
        """Predict quality score for a repository."""
        if self.quality_regressor is None:
            raise ValueError("Quality regressor not trained. Call train_quality_regressor first.")
        
        # Extract features
        features = self.extract_features(repo_data)
        features_scaled = self.quality_scaler.transform(features.reshape(1, -1))
        
        # Predict quality
        quality_score = self.quality_regressor.predict(features_scaled)[0]
        
        return max(0.0, min(1.0, quality_score))  # Clamp to [0, 1]
    
    def load_models(self) -> bool:
        """Load pre-trained models."""
        try:
            pattern_model_path = os.path.join(self.models_dir, 'pattern_classifier.joblib')
            quality_model_path = os.path.join(self.models_dir, 'quality_regressor.joblib')
            pattern_scaler_path = os.path.join(self.models_dir, 'pattern_scaler.joblib')
            quality_scaler_path = os.path.join(self.models_dir, 'quality_scaler.joblib')
            
            if (os.path.exists(pattern_model_path) and 
                os.path.exists(quality_model_path) and
                os.path.exists(pattern_scaler_path) and
                os.path.exists(quality_scaler_path)):
                
                self.pattern_classifier = joblib.load(pattern_model_path)
                self.quality_regressor = joblib.load(quality_model_path)
                self.pattern_scaler = joblib.load(pattern_scaler_path)
                self.quality_scaler = joblib.load(quality_scaler_path)
                
                print("✅ Pre-trained models loaded successfully")
                return True
            else:
                print("❌ Pre-trained models not found")
                return False
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

def train_ml_models():
    """Train machine learning models on existing analysis data."""
    print("🚀 TRAINING MACHINE LEARNING MODELS")
    print("=" * 60)
    
    # Load existing analysis data
    try:
        with open('CombinedAnalysis/enhanced_architecture_analysis.json', 'r') as f:
            analysis_data = json.load(f)
    except FileNotFoundError:
        print("❌ Enhanced architecture analysis not found. Run enhanced analysis first.")
        return
    
    # Initialize ML analyzer
    ml_analyzer = MLBasedArchitectureAnalyzer()
    
    # Create training dataset
    X, y_patterns, y_quality = ml_analyzer.create_training_dataset(analysis_data)
    
    # Train pattern classifier
    pattern_accuracy = ml_analyzer.train_pattern_classifier(X, y_patterns)
    
    # Train quality regressor
    quality_r2 = ml_analyzer.train_quality_regressor(X, y_quality)
    
    print("\n" + "=" * 60)
    print("✅ MACHINE LEARNING MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)
    print(f"📊 Final Results:")
    print(f"   • Pattern Classification Accuracy: {pattern_accuracy:.3f}")
    print(f"   • Quality Regression R² Score: {quality_r2:.3f}")
    print(f"   • Models saved in: ml_models/")

def demonstrate_ml_analysis():
    """Demonstrate ML-based analysis on existing repositories."""
    print("🔍 MACHINE LEARNING ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Initialize ML analyzer
    ml_analyzer = MLBasedArchitectureAnalyzer()
    
    # Try to load pre-trained models
    if not ml_analyzer.load_models():
        print("❌ No pre-trained models found. Please run train_ml_models() first.")
        return
    
    # Load analysis data
    try:
        with open('CombinedAnalysis/enhanced_architecture_analysis.json', 'r') as f:
            analysis_data = json.load(f)
    except FileNotFoundError:
        print("❌ Enhanced architecture analysis not found.")
        return
    
    # Analyze each repository
    print(f"\n📋 ML-BASED ANALYSIS RESULTS:")
    
    for repo_name, repo_data in analysis_data.get('detailed_analysis', {}).items():
        print(f"\n📁 {repo_name}")
        
        # Predict patterns
        pattern_results = ml_analyzer.predict_patterns(repo_data)
        primary_pattern = pattern_results["primary_pattern"]
        
        print(f"   🏗️  ML Predicted Pattern: {primary_pattern['name']}")
        print(f"   🎯 ML Confidence: {primary_pattern['confidence']:.3f}")
        
        # Predict quality
        predicted_quality = ml_analyzer.predict_quality(repo_data)
        actual_quality = repo_data.get('combined_metrics', {}).get('overall_quality', 0.0)
        
        print(f"   📈 ML Predicted Quality: {predicted_quality:.3f}")
        print(f"   📊 Actual Quality: {actual_quality:.3f}")
        print(f"   🔍 Quality Difference: {abs(predicted_quality - actual_quality):.3f}")

if __name__ == "__main__":
    # Train models
    train_ml_models()
    
    # Demonstrate ML analysis
    demonstrate_ml_analysis()
