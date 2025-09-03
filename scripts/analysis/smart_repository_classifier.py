#!/usr/bin/env python3
"""
Smart Repository Classifier
==========================

Improved architecture with:
1. Hierarchical classification (Architecture -> Quality -> Specialization)
2. Rule-based + ML hybrid approach
3. Feature selection and dimensionality reduction
4. Ensemble methods for better accuracy
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import the comprehensive analyzer
import sys
sys.path.append('scripts/analysis')
from comprehensive_repository_analyzer_v3 import ComprehensiveRepositoryAnalyzerV3

class RuleBasedClassifier:
    """Lightweight rule-based classifier for quick decisions."""
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict[str, List[Dict]]:
        """Initialize classification rules."""
        return {
            'microservices': [
                {'dockerfile': True, 'kubernetes': True, 'confidence': 0.95},
                {'dockerfile': True, 'helm': True, 'confidence': 0.90},
                {'docker_compose': True, 'confidence': 0.85}
            ],
            'data_science': [
                {'requirements_txt': True, 'jupyter_notebooks': True, 'confidence': 0.95},
                {'requirements_txt': True, 'pandas': True, 'confidence': 0.90},
                {'jupyter_notebooks': True, 'matplotlib': True, 'confidence': 0.85}
            ],
            'web_application': [
                {'package_json': True, 'react': True, 'web_keywords': True, 'confidence': 0.95},
                {'package_json': True, 'express': True, 'web_keywords': True, 'confidence': 0.90},
                {'requirements_txt': True, 'django': True, 'web_keywords': True, 'confidence': 0.90},
                {'web_keywords': True, 'frontend_files': True, 'confidence': 0.85}
            ],
            'cli_tool': [
                {'package_json': True, 'bin_folder': True, 'cli_keywords': True, 'confidence': 0.95},
                {'setup_py': True, 'console_scripts': True, 'confidence': 0.90},
                {'bin_folder': True, 'cli_keywords': True, 'confidence': 0.85},
                {'cli_keywords': True, 'small_file_count': True, 'confidence': 0.75}
            ],
            'mobile_app': [
                {'android_manifest': True, 'confidence': 0.95},
                {'ios_project': True, 'confidence': 0.95},
                {'react_native': True, 'confidence': 0.90}
            ],
            'game_development': [
                {'unity_files': True, 'confidence': 0.95},
                {'game_assets': True, 'confidence': 0.90},
                {'game_engine': True, 'confidence': 0.85},
                {'python_games': True, 'confidence': 0.80},
                {'game_related_keywords': True, 'confidence': 0.75}
            ],
            'library': [
                {'setup_py': True, 'tests_folder': True, 'confidence': 0.90},
                {'package_json': True, 'src_folder': True, 'confidence': 0.85},
                {'documentation': True, 'confidence': 0.80}
            ]
        }
    
    def classify(self, repo_path: str) -> Tuple[str, float]:
        """Classify repository using rules."""
        features = self._extract_rule_features(repo_path)
        
        best_match = None
        best_confidence = 0.0
        
        for category, rule_list in self.rules.items():
            for rule in rule_list:
                confidence = self._evaluate_rule(features, rule)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = category
        
        return best_match or "unknown", best_confidence
    
    def _extract_rule_features(self, repo_path: str) -> Dict[str, bool]:
        """Extract features needed for rule evaluation."""
        features = {}
        
        # Check for specific files
        features['dockerfile'] = os.path.exists(os.path.join(repo_path, 'Dockerfile'))
        features['docker_compose'] = os.path.exists(os.path.join(repo_path, 'docker-compose.yml'))
        features['kubernetes'] = any('kubernetes' in f.lower() for f in os.listdir(repo_path) if os.path.isfile(os.path.join(repo_path, f)))
        features['helm'] = any('helm' in f.lower() for f in os.listdir(repo_path) if os.path.isfile(os.path.join(repo_path, f)))
        
        features['requirements_txt'] = os.path.exists(os.path.join(repo_path, 'requirements.txt'))
        features['package_json'] = os.path.exists(os.path.join(repo_path, 'package.json'))
        features['setup_py'] = os.path.exists(os.path.join(repo_path, 'setup.py'))
        
        # Check for specific content patterns
        features['jupyter_notebooks'] = any(f.endswith('.ipynb') for f in os.listdir(repo_path) if os.path.isfile(os.path.join(repo_path, f)))
        features['react'] = self._check_file_content(repo_path, 'package.json', 'react')
        features['express'] = self._check_file_content(repo_path, 'package.json', 'express')
        features['django'] = self._check_file_content(repo_path, 'requirements.txt', 'django')
        features['pandas'] = self._check_file_content(repo_path, 'requirements.txt', 'pandas')
        features['matplotlib'] = self._check_file_content(repo_path, 'requirements.txt', 'matplotlib')
        
        # Game development specific features
        features['python_games'] = self._check_file_content(repo_path, 'requirements.txt', 'pygame') or self._check_file_content(repo_path, 'requirements.txt', 'arcade')
        features['game_related_keywords'] = self._check_game_keywords(repo_path)
        
        # CLI tool specific features
        features['console_scripts'] = self._check_console_scripts(repo_path)
        features['cli_keywords'] = self._check_cli_keywords(repo_path)
        
        # Web application specific features
        features['web_keywords'] = self._check_web_keywords(repo_path)
        features['frontend_files'] = self._check_frontend_files(repo_path)
        
        # Check for folder structures
        features['bin_folder'] = os.path.exists(os.path.join(repo_path, 'bin'))
        features['src_folder'] = os.path.exists(os.path.join(repo_path, 'src'))
        features['tests_folder'] = any('test' in d.lower() for d in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, d)))
        features['documentation'] = any('doc' in d.lower() or 'readme' in d.lower() for d in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, d)))
        
        # Count files for size-based rules
        file_count = len([f for f in os.listdir(repo_path) if os.path.isfile(os.path.join(repo_path, f))])
        features['small_file_count'] = file_count < 50
        
        return features
    
    def _check_file_content(self, repo_path: str, filename: str, content: str) -> bool:
        """Check if file contains specific content."""
        file_path = os.path.join(repo_path, filename)
        if not os.path.exists(file_path):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content_text = f.read().lower()
                return content.lower() in content_text
        except:
            return False
    
    def _evaluate_rule(self, features: Dict[str, bool], rule: Dict) -> float:
        """Evaluate a single rule against features."""
        confidence = rule['confidence']
        
        for key, value in rule.items():
            if key == 'confidence':
                continue
            
            if key in features:
                if features[key] == value:
                    confidence *= 1.0  # Rule matches
                else:
                    confidence *= 0.3  # Rule doesn't match
            else:
                confidence *= 0.5  # Feature not found
        
        return confidence
    
    def _check_game_keywords(self, repo_path: str) -> bool:
        """Check if repository contains game-related keywords."""
        game_keywords = [
            'game', 'player', 'score', 'level', 'sprite', 'collision', 'animation',
            'paddle', 'ball', 'snake', 'tetris', 'puzzle', 'adventure', 'rpg',
            'shooter', 'platform', 'racing', 'strategy', 'simulation'
        ]
        
        # Check in file names and content
        for root, _, files in os.walk(repo_path):
            for file in files:
                file_lower = file.lower()
                # Check file names
                if any(keyword in file_lower for keyword in game_keywords):
                    return True
                
                # Check file content for text files
                if file.endswith(('.py', '.js', '.java', '.cpp', '.md', '.txt')):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().lower()
                            if any(keyword in content for keyword in game_keywords):
                                return True
                    except:
                        continue
        
        return False
    
    def _check_console_scripts(self, repo_path: str) -> bool:
        """Check if setup.py contains console_scripts entry point."""
        setup_py_path = os.path.join(repo_path, 'setup.py')
        if not os.path.exists(setup_py_path):
            return False
        
        try:
            with open(setup_py_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
                return 'console_scripts' in content and 'entry_points' in content
        except:
            return False
    
    def _check_cli_keywords(self, repo_path: str) -> bool:
        """Check if repository contains CLI-related keywords."""
        cli_keywords = [
            'cli', 'command', 'line', 'interface', 'terminal', 'shell', 'script',
            'argparse', 'click', 'typer', 'fire', 'commander', 'yargs', 'minimist',
            'main', 'run', 'execute', 'start', 'stop', 'install', 'uninstall'
        ]
        
        # Check in file names and content
        for root, _, files in os.walk(repo_path):
            for file in files:
                file_lower = file.lower()
                # Check file names
                if any(keyword in file_lower for keyword in cli_keywords):
                    return True
                
                # Check file content for text files
                if file.endswith(('.py', '.js', '.java', '.cpp', '.md', '.txt', '.json')):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().lower()
                            if any(keyword in content for keyword in cli_keywords):
                                return True
                    except:
                        continue
        
        return False
    
    def _check_web_keywords(self, repo_path: str) -> bool:
        """Check if repository contains web-related keywords."""
        web_keywords = [
            'web', 'http', 'server', 'api', 'rest', 'graphql', 'endpoint', 'route',
            'controller', 'view', 'template', 'html', 'css', 'javascript', 'frontend',
            'backend', 'database', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql'
        ]
        
        # Check in file names and content
        for root, _, files in os.walk(repo_path):
            for file in files:
                file_lower = file.lower()
                # Check file names
                if any(keyword in file_lower for keyword in web_keywords):
                    return True
                
                # Check file content for text files
                if file.endswith(('.py', '.js', '.java', '.php', '.md', '.txt', '.json', '.yml')):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().lower()
                            if any(keyword in content for keyword in web_keywords):
                                return True
                    except:
                        continue
        
        return False
    
    def _check_frontend_files(self, repo_path: str) -> bool:
        """Check if repository contains frontend files."""
        frontend_extensions = ['.html', '.css', '.js', '.jsx', '.ts', '.tsx', '.vue', '.svelte']
        frontend_folders = ['public', 'src', 'static', 'assets', 'components', 'pages']
        
        # Check for frontend file extensions
        for root, _, files in os.walk(repo_path):
            for file in files:
                if any(file.endswith(ext) for ext in frontend_extensions):
                    return True
        
        # Check for frontend folders
        for folder in frontend_folders:
            if os.path.exists(os.path.join(repo_path, folder)):
                return True
        
        return False

class FeatureSelector:
    """Feature selection and dimensionality reduction."""
    
    def __init__(self, n_features: int = 100):
        self.n_features = n_features
        self.selector = SelectKBest(score_func=f_classif, k=n_features)
        self.selected_features = []
        self.feature_scores = {}
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit selector and transform features."""
        X_selected = self.selector.fit_transform(X, y)
        self.selected_features = self.selector.get_support()
        self.feature_scores = dict(zip(range(X.shape[1]), self.selector.scores_))
        return X_selected
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted selector."""
        return self.selector.transform(X)
    
    def get_feature_importance(self) -> List[Tuple[int, float]]:
        """Get feature importance scores."""
        return sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)

class SmartRepositoryClassifier:
    """Smart repository classifier with hierarchical approach."""
    
    def __init__(self, models_dir: str = "ml_models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Components
        self.rule_engine = RuleBasedClassifier()
        self.feature_selector = FeatureSelector(n_features=100)
        
        # ML Models
        self.arch_classifier = None
        self.quality_classifier = None
        self.spec_classifier = None
        
        # Feature extractors
        self.analyzer = ComprehensiveRepositoryAnalyzerV3()
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Load pre-trained models if available
        self._load_pretrained_models()
        
        print(f"🔧 Smart Repository Classifier initialized")
    
    def _load_pretrained_models(self):
        """Load pre-trained models if they exist."""
        model_files = {
            'arch': 'smart_arch_classifier.joblib',
            'quality': 'smart_quality_classifier.joblib',
            'spec': 'smart_spec_classifier.joblib'
        }
        
        for model_type, filename in model_files.items():
            model_path = os.path.join(self.models_dir, filename)
            if os.path.exists(model_path):
                try:
                    if model_type == 'arch':
                        self.arch_classifier = joblib.load(model_path)
                    elif model_type == 'quality':
                        self.quality_classifier = joblib.load(model_path)
                    elif model_type == 'spec':
                        self.spec_classifier = joblib.load(model_path)
                    print(f"✅ Loaded {model_type} classifier")
                except Exception as e:
                    print(f"⚠️  Failed to load {model_type} classifier: {e}")
    
    def extract_optimized_features(self, repo_path: str) -> np.ndarray:
        """Extract optimized features using CRAv3."""
        try:
            print(f"🔍 Extracting optimized features from: {repo_path}")
            
            # Extract features using CRAv3
            ast_features = self.analyzer.extract_ast_features(repo_path)
            codebert_features = self.analyzer.extract_codebert_embeddings(repo_path)
            keyword_features = self.analyzer.extract_keyword_features(repo_path)
            structure_features = self.analyzer.extract_file_structure_features(repo_path)
            
            # Build combined features
            all_features = {
                **ast_features,
                **codebert_features.get('semantic_features', {}),
                **codebert_features.get('significant_dimensions', {}),
                **keyword_features,
                **structure_features
            }
            
            # Convert to feature vector
            feature_vector = self._dict_to_feature_vector(all_features)
            
            print(f"   ✅ Extracted {len(feature_vector)} features")
            return feature_vector
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return np.zeros(965)  # Default size
    
    def _dict_to_feature_vector(self, features_dict: Dict[str, Any]) -> np.ndarray:
        """Convert features dictionary to numpy array."""
        # This is a simplified version - in practice, you'd want more robust conversion
        feature_list = []
        
        # Extract values safely
        for key, value in features_dict.items():
            try:
                if isinstance(value, (int, float)):
                    feature_list.append(float(value))
                elif isinstance(value, bool):
                    feature_list.append(float(value))
                elif isinstance(value, list):
                    # Handle lists (like CodeBERT embeddings)
                    if isinstance(value[0], (int, float)):
                        feature_list.extend([float(x) for x in value])
                    else:
                        feature_list.append(0.0)
                else:
                    feature_list.append(0.0)
            except:
                feature_list.append(0.0)
        
        # Pad or truncate to expected size
        expected_size = 965
        if len(feature_list) < expected_size:
            feature_list.extend([0.0] * (expected_size - len(feature_list)))
        elif len(feature_list) > expected_size:
            feature_list = feature_list[:expected_size]
        
        return np.array(feature_list, dtype=np.float32)
    
    def classify(self, repo_path: str) -> Dict[str, Any]:
        """Classify repository using hybrid approach."""
        print(f"🎯 Classifying: {repo_path}")
        
        # Step 1: Rule-based classification
        arch_type, rule_confidence = self.rule_engine.classify(repo_path)
        print(f"   📋 Rule-based: {arch_type} (confidence: {rule_confidence:.2f})")
        
        # If rule-based classification is confident enough, use it
        if rule_confidence > 0.8:
            return {
                'architecture': arch_type,
                'quality': self._estimate_quality_from_rules(repo_path),
                'specialization': self._estimate_specialization_from_rules(repo_path),
                'confidence': rule_confidence,
                'method': 'rule_based'
            }
        
        # Step 2: ML-based classification for uncertain cases
        print(f"   🤖 Using ML classification...")
        return self._ml_classify(repo_path, arch_type)
    
    def _estimate_quality_from_rules(self, repo_path: str) -> str:
        """Estimate quality based on rule-based analysis."""
        # Simple heuristics
        has_tests = any('test' in d.lower() for d in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, d)))
        has_docs = any('readme' in f.lower() for f in os.listdir(repo_path) if os.path.isfile(os.path.join(repo_path, f)))
        has_ci = any('.github' in d for d in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, d)))
        
        score = 0
        if has_tests: score += 1
        if has_docs: score += 1
        if has_ci: score += 1
        
        if score >= 2:
            return "high_quality"
        elif score >= 1:
            return "medium_quality"
        else:
            return "low_quality"
    
    def _estimate_specialization_from_rules(self, repo_path: str) -> str:
        """Estimate specialization based on rule-based analysis."""
        # This would be more sophisticated in practice
        return "generalist"
    
    def _ml_classify(self, repo_path: str, arch_type: str) -> Dict[str, Any]:
        """ML-based classification."""
        try:
            # Check if models and scaler are properly fitted
            if not self._is_ml_ready():
                print(f"   ⚠️  ML models not ready, using rule-based fallback...")
                return self._rule_based_fallback(repo_path, arch_type)
            
            # Extract features
            features = self.extract_optimized_features(repo_path)
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Select features
            features_selected = self.feature_selector.transform(features_scaled)
            
            # Make predictions
            arch_pred = self.arch_classifier.predict(features_selected)[0] if self.arch_classifier else arch_type
            quality_pred = self.quality_classifier.predict(features_selected)[0] if self.quality_classifier else "medium_quality"
            spec_pred = self.spec_classifier.predict(features_selected)[0] if self.spec_classifier else "generalist"
            
            return {
                'architecture': arch_pred,
                'quality': quality_pred,
                'specialization': spec_pred,
                'confidence': 0.7,  # ML confidence
                'method': 'ml_based'
            }
            
        except Exception as e:
            print(f"   ⚠️  ML classification failed: {e}")
            return self._rule_based_fallback(repo_path, arch_type)
    
    def _is_ml_ready(self) -> bool:
        """Check if ML models and scaler are ready for prediction."""
        # Check if scaler is fitted
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            return False
        
        # Check if feature selector is fitted
        if not hasattr(self.feature_selector, 'selector') or not hasattr(self.feature_selector.selector, 'get_support'):
            return False
        
        # Check if models are loaded
        if not all([self.arch_classifier, self.quality_classifier, self.spec_classifier]):
            return False
        
        return True
    
    def _rule_based_fallback(self, repo_path: str, arch_type: str) -> Dict[str, Any]:
        """Fallback to rule-based classification when ML fails."""
        print(f"   📋 Using rule-based fallback...")
        
        # Estimate quality and specialization from rules
        quality = self._estimate_quality_from_rules(repo_path)
        specialization = self._estimate_specialization_from_rules(repo_path)
        
        # Adjust confidence based on rule strength
        rule_confidence = 0.6  # Lower confidence for fallback
        
        return {
            'architecture': arch_type,
            'quality': quality,
            'specialization': specialization,
            'confidence': rule_confidence,
            'method': 'rule_based_fallback'
        }
    
    def train_models(self, dataset_path: str = "dataset"):
        """Train the ML models."""
        print("🚀 TRAINING SMART REPOSITORY CLASSIFIER")
        print("=" * 60)
        
        # Prepare training data
        X, y_arch, y_quality, y_spec = self._prepare_training_data(dataset_path)
        
        if X.shape[0] == 0:
            print("❌ No training data available")
            return
        
        print(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Feature selection
        print("🔧 Selecting best features...")
        X_selected = self.feature_selector.fit_transform(X, y_arch)
        print(f"   Selected {X_selected.shape[1]} features from {X.shape[1]}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Train architecture classifier
        print("🌲 Training architecture classifier...")
        self.arch_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.arch_classifier.fit(X_scaled, y_arch)
        
        # Train quality classifier
        print("⭐ Training quality classifier...")
        self.quality_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.quality_classifier.fit(X_scaled, y_quality)
        
        # Train specialization classifier
        print("🎯 Training specialization classifier...")
        self.spec_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.spec_classifier.fit(X_scaled, y_spec)
        
        # Evaluate models
        self._evaluate_models(X_scaled, y_arch, y_quality, y_spec)
        
        # Save models
        self._save_models()
        
        print("✅ Training completed successfully!")
    
    def _prepare_training_data(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from dataset."""
        X_list = []
        y_arch_list = []
        y_quality_list = []
        y_spec_list = []
        
        categories = os.listdir(dataset_path)
        
        for category in categories:
            category_path = os.path.join(dataset_path, category)
            if not os.path.isdir(category_path):
                continue
            
            for repo_name in os.listdir(category_path):
                repo_path = os.path.join(category_path, repo_name)
                if not os.path.isdir(repo_path):
                    continue
                
                try:
                    # Extract features
                    features = self.extract_optimized_features(repo_path)
                    
                    # Create labels
                    arch_label = self._map_category_to_architecture(category)
                    quality_label = self._estimate_quality_from_rules(repo_path)
                    spec_label = self._map_category_to_specialization(category)
                    
                    X_list.append(features)
                    y_arch_list.append(arch_label)
                    y_quality_list.append(quality_label)
                    y_spec_list.append(spec_label)
                    
                except Exception as e:
                    print(f"⚠️  Error processing {repo_path}: {e}")
                    continue
        
        if not X_list:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        X = np.array(X_list)
        y_arch = np.array(y_arch_list)
        y_quality = np.array(y_quality_list)
        y_spec = np.array(y_spec_list)
        
        return X, y_arch, y_quality, y_spec
    
    def _map_category_to_architecture(self, category: str) -> str:
        """Map dataset category to architecture type."""
        mapping = {
            'cli_tool': 'cli_tool',
            'data_science': 'data_science',
            'web_application': 'web_application',
            'mobile_app': 'mobile_app',
            'game_development': 'game_development',
            'library': 'library',
            'educational': 'educational'
        }
        return mapping.get(category, 'unknown')
    
    def _map_category_to_specialization(self, category: str) -> str:
        """Map dataset category to specialization."""
        mapping = {
            'data_science': 'data_scientist',
            'mobile_app': 'mobile_developer',
            'game_development': 'game_developer',
            'web_application': 'frontend_specialist',
            'library': 'backend_specialist',
            'cli_tool': 'backend_specialist',
            'educational': 'generalist'
        }
        return mapping.get(category, 'generalist')
    
    def _evaluate_models(self, X: np.ndarray, y_arch: np.ndarray, y_quality: np.ndarray, y_spec: np.ndarray):
        """Evaluate model performance."""
        print("\n📊 MODEL EVALUATION")
        print("=" * 40)
        
        # Architecture classifier
        y_arch_pred = self.arch_classifier.predict(X)
        arch_accuracy = accuracy_score(y_arch, y_arch_pred)
        print(f"🏗️  Architecture Classifier: {arch_accuracy:.3f}")
        
        # Quality classifier
        y_quality_pred = self.quality_classifier.predict(X)
        quality_accuracy = accuracy_score(y_quality, y_quality_pred)
        print(f"⭐ Quality Classifier: {quality_accuracy:.3f}")
        
        # Specialization classifier
        y_spec_pred = self.spec_classifier.predict(X)
        spec_accuracy = accuracy_score(y_spec, y_spec_pred)
        print(f"🎯 Specialization Classifier: {spec_accuracy:.3f}")
    
    def _save_models(self):
        """Save trained models."""
        print("\n💾 Saving models...")
        
        # Save architecture classifier
        arch_path = os.path.join(self.models_dir, 'smart_arch_classifier.joblib')
        joblib.dump(self.arch_classifier, arch_path)
        print(f"   ✅ Architecture classifier: {arch_path}")
        
        # Save quality classifier
        quality_path = os.path.join(self.models_dir, 'smart_quality_classifier.joblib')
        joblib.dump(self.quality_classifier, quality_path)
        print(f"   ✅ Quality classifier: {quality_path}")
        
        # Save specialization classifier
        spec_path = os.path.join(self.models_dir, 'smart_spec_classifier.joblib')
        joblib.dump(self.spec_classifier, spec_path)
        print(f"   ✅ Specialization classifier: {spec_path}")
        
        # Save preprocessing components
        scaler_path = os.path.join(self.models_dir, 'smart_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        selector_path = os.path.join(self.models_dir, 'smart_feature_selector.joblib')
        joblib.dump(self.feature_selector, selector_path)

def main():
    """Main function to train the smart classifier."""
    print("🚀 SMART REPOSITORY CLASSIFIER")
    print("=" * 60)
    
    # Initialize classifier
    classifier = SmartRepositoryClassifier()
    
    # Train models
    classifier.train_models()
    
    # Test classification
    print(f"\n🧪 TESTING CLASSIFICATION")
    print("=" * 50)
    
    # Find a test repository
    dataset_path = "dataset"
    test_repo_path = None
    
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            for repo_name in os.listdir(category_path):
                repo_path = os.path.join(category_path, repo_name)
                if os.path.isdir(repo_path):
                    test_repo_path = repo_path
                    test_repo_name = f"{category}/{repo_name}"
                    break
            if test_repo_path:
                break
    
    if test_repo_path:
        print(f"📁 Testing on: {test_repo_name}")
        result = classifier.classify(test_repo_path)
        
        print(f"🎯 Classification Result:")
        print(f"   • Architecture: {result['architecture']}")
        print(f"   • Quality: {result['quality']}")
        print(f"   • Specialization: {result['specialization']}")
        print(f"   • Confidence: {result['confidence']:.3f}")
        print(f"   • Method: {result['method']}")
    else:
        print("⚠️  No test repository found")
    
    print(f"\n🎉 Smart Repository Classifier is ready!")

if __name__ == "__main__":
    main()
