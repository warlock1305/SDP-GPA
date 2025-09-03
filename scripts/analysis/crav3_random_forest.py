#!/usr/bin/env python3
"""
CRAV3 Random Forest Model
=========================

This script creates a Random Forest model based on all features extracted by
comprehensive_repository_analyzer_v3.py, including:
- AST features (structural analysis)
- CodeBERT embeddings (semantic analysis) 
- Keyword analysis (content-based features)
- File structure features
- Semantic features

The model is trained on the dataset folder and can classify repositories into
architectural patterns and other categories.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import the comprehensive analyzer
import sys
sys.path.append('scripts/analysis')
from comprehensive_repository_analyzer_v3 import ComprehensiveRepositoryAnalyzerV3
from enhanced_keyword_analyzer import EnhancedKeywordAnalyzer

class CRAV3RandomForest:
    """Random Forest model based on Comprehensive Repository Analyzer v3 features."""
    
    def __init__(self, models_dir: str = "ml_models"):
        """Initialize the CRAV3 Random Forest model."""
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Model components
        self.rf_classifier = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_names = []
        
        # Analyzer instances
        self.analyzer = ComprehensiveRepositoryAnalyzerV3()
        self.keyword_analyzer = EnhancedKeywordAnalyzer()
        
        # Fix AstMiner path for the analyzer
        self.analyzer.astminer_jar = "astminer-0.9.0/build/libs/astminer.jar"
        
        # Try to load pre-trained model if it exists
        self._load_pretrained_model()
        
        # Feature dimensions
        self.ast_feature_count = 14
        self.codebert_dimension = 768
        self.significant_dimension_count = 50  # Top significant dimensions
        self.keyword_feature_count = 51
        self.structure_feature_count = 20
        self.semantic_feature_count = 9
        
        # Total expected features
        self.total_features = (
            self.ast_feature_count + 
            self.codebert_dimension + 
            self.significant_dimension_count + 
            self.keyword_feature_count + 
            self.structure_feature_count + 
            self.semantic_feature_count
        )
        
        print(f"🔧 CRAV3 Random Forest initialized")
        print(f"📊 Expected features: {self.total_features}")
        print(f"   • AST features: {self.ast_feature_count}")
        print(f"   • CodeBERT embeddings: {self.codebert_dimension}")
        print(f"   • Significant dimensions: {self.significant_dimension_count}")
        print(f"   • Keyword features: {self.keyword_feature_count}")
        print(f"   • Structure features: {self.structure_feature_count}")
        print(f"   • Semantic features: {self.semantic_feature_count}")
        
        # Report model loading status
        if self.rf_classifier is not None:
            print(f"✅ Pre-trained model loaded successfully")
        else:
            print(f"⚠️  No pre-trained model found - training required")
    
    def _load_pretrained_model(self):
        """Load pre-trained model and scaler if they exist."""
        model_path = os.path.join(self.models_dir, 'crav3_random_forest.joblib')
        scaler_path = os.path.join(self.models_dir, 'crav3_random_forest_scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                print("🔄 Loading pre-trained model...")
                self.rf_classifier = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # Load feature and target names if available
                feature_names_path = os.path.join(self.models_dir, 'crav3_random_forest_feature_names.json')
                target_names_path = os.path.join(self.models_dir, 'crav3_random_forest_target_names.json')
                
                if os.path.exists(feature_names_path):
                    with open(feature_names_path, 'r') as f:
                        self.feature_names = json.load(f)
                
                if os.path.exists(target_names_path):
                    with open(target_names_path, 'r') as f:
                        self.target_names = json.load(f)
                
                print("✅ Pre-trained model loaded successfully!")
                
            except Exception as e:
                print(f"⚠️  Failed to load pre-trained model: {e}")
                self.rf_classifier = None
                self.scaler = StandardScaler()
        else:
            print("ℹ️  No pre-trained model found - will need to train")
    
    def extract_comprehensive_features(self, repository_path: str) -> Optional[np.ndarray]:
        """Extract all features using Comprehensive Repository Analyzer v3."""
        try:
            print(f"🔍 Extracting features from: {repository_path}")
            
            # Extract all features using the analyzer with error handling
            try:
                ast_features = self.analyzer.extract_ast_features(repository_path)
            except Exception as e:
                print(f"⚠️  AST extraction failed: {e}, using empty features")
                ast_features = {
                    'avg_path_length': 0.0, 'max_path_length': 0.0, 'path_variety': 0.0,
                    'node_type_diversity': 0.0, 'complexity_score': 0.0, 'nesting_depth': 0.0,
                    'function_count': 0.0, 'class_count': 0.0, 'interface_count': 0.0,
                    'inheritance_depth': 0.0, 'total_ast_nodes': 0, 'unique_node_types': 0,
                    'ast_depth': 0.0, 'branching_factor': 0.0
                }
            
            # Also handle the case where AST features might be None or incomplete
            if ast_features is None:
                print(f"⚠️  AST features returned None, using empty features")
                ast_features = {
                    'avg_path_length': 0.0, 'max_path_length': 0.0, 'path_variety': 0.0,
                    'node_type_diversity': 0.0, 'complexity_score': 0.0, 'nesting_depth': 0.0,
                    'function_count': 0.0, 'class_count': 0.0, 'interface_count': 0.0,
                    'inheritance_depth': 0.0, 'total_ast_nodes': 0, 'unique_node_types': 0,
                    'ast_depth': 0.0, 'branching_factor': 0.0
                }
            
            try:
                codebert_features = self.analyzer.extract_codebert_embeddings(repository_path)
            except Exception as e:
                print(f"⚠️  CodeBERT extraction failed: {e}, using empty features")
                codebert_features = {}
            
            try:
                # Use enhanced keyword analyzer instead of basic one
                keyword_features = self.keyword_analyzer.analyze_repository_content(repository_path)
            except Exception as e:
                print(f"⚠️  Enhanced keyword extraction failed: {e}, using basic extraction")
                try:
                    keyword_features = self.keyword_analyzer._extract_basic_keyword_features(repository_path)
                except Exception as e2:
                    print(f"⚠️  Basic keyword extraction also failed: {e2}, using empty features")
                    keyword_features = {
                        'framework_keywords': 0.0, 'library_keywords': 0.0, 'data_science_keywords': 0.0,
                        'web_keywords': 0.0, 'cli_keywords': 0.0, 'game_keywords': 0.0,
                        'mobile_keywords': 0.0, 'testing_keywords': 0.0, 'database_keywords': 0.0,
                        'cloud_keywords': 0.0, 'total_keywords': 0, 'keyword_diversity': 0.0
                    }
            
            try:
                structure_features = self.analyzer.extract_file_structure_features(repository_path)
            except Exception as e:
                print(f"⚠️  Structure extraction failed: {e}, using empty features")
                structure_features = {
                    'total_files': 0, 'total_directories': 0, 'source_files': 0,
                    'avg_files_per_dir': 0.0, 'max_directory_depth': 0, 'has_tests': 0.0,
                    'has_docs': 0.0, 'has_config': 0.0, 'has_dependencies': 0.0,
                    'has_ci_cd': 0.0, 'has_docker': 0.0, 'language_diversity': 0,
                    'total_source_files': 0, 'main_language_hash': 0
                }
            
            # Combine all features into a single vector
            features = []
            
            # 1. AST Features (14 features)
            ast_feature_list = [
                ast_features.get('avg_path_length', 0.0),
                ast_features.get('max_path_length', 0.0),
                ast_features.get('path_variety', 0.0),
                ast_features.get('node_type_diversity', 0.0),
                ast_features.get('complexity_score', 0.0),
                ast_features.get('nesting_depth', 0.0),
                ast_features.get('function_count', 0.0),
                ast_features.get('class_count', 0.0),
                ast_features.get('interface_count', 0.0),
                ast_features.get('inheritance_depth', 0.0),
                ast_features.get('total_ast_nodes', 0),
                ast_features.get('unique_node_types', 0),
                ast_features.get('ast_depth', 0.0),
                ast_features.get('branching_factor', 0.0)
            ]
            features.extend(ast_feature_list)
            
            # 2. CodeBERT Embeddings (768 dimensions)
            if codebert_features and 'repository_embedding' in codebert_features:
                repo_embedding = codebert_features['repository_embedding']
                if len(repo_embedding) == self.codebert_dimension:
                    features.extend(repo_embedding)
                else:
                    # Pad or truncate to 768 dimensions
                    if len(repo_embedding) < self.codebert_dimension:
                        features.extend(repo_embedding + [0.0] * (self.codebert_dimension - len(repo_embedding)))
                    else:
                        features.extend(repo_embedding[:self.codebert_dimension])
            else:
                # Fill with zeros if no CodeBERT features
                features.extend([0.0] * self.codebert_dimension)
            
            # 3. Significant Dimensions (50 features)
            if codebert_features and 'significant_dimensions' in codebert_features:
                significant_dims = codebert_features['significant_dimensions']
                # Extract top 50 significant dimensions
                significant_values = list(significant_dims.values())[:self.significant_dimension_count]
                if len(significant_values) < self.significant_dimension_count:
                    significant_values.extend([0.0] * (self.significant_dimension_count - len(significant_values)))
                features.extend(significant_values)
            else:
                features.extend([0.0] * self.significant_dimension_count)
            
            # 4. Keyword Features (51 features)
            # Handle both enhanced and basic keyword features
            if 'feature_vector' in keyword_features:
                # Enhanced keyword analyzer result
                keyword_vector = keyword_features['feature_vector']
                if isinstance(keyword_vector, np.ndarray):
                    keyword_vector = keyword_vector.tolist()
                
                if len(keyword_vector) == self.keyword_feature_count:
                    features.extend(keyword_vector)
                else:
                    # Pad or truncate to 51 features
                    if len(keyword_vector) < self.keyword_feature_count:
                        features.extend(keyword_vector + [0.0] * (self.keyword_feature_count - len(keyword_vector)))
                    else:
                        features.extend(keyword_vector[:self.keyword_feature_count])
            else:
                # Basic keyword features (fallback)
                keyword_feature_list = []
                
                # Basic keyword counts (12 features)
                basic_keywords = [
                    keyword_features.get('framework_keywords', 0.0),
                    keyword_features.get('library_keywords', 0.0),
                    keyword_features.get('data_science_keywords', 0.0),
                    keyword_features.get('web_keywords', 0.0),
                    keyword_features.get('cli_keywords', 0.0),
                    keyword_features.get('game_keywords', 0.0),
                    keyword_features.get('mobile_keywords', 0.0),
                    keyword_features.get('testing_keywords', 0.0),
                    keyword_features.get('database_keywords', 0.0),
                    keyword_features.get('cloud_keywords', 0.0),
                    keyword_features.get('total_keywords', 0),
                    keyword_features.get('keyword_diversity', 0.0)
                ]
                keyword_feature_list.extend(basic_keywords)
                
                # Additional keyword features to reach 51 total
                # Language-specific expertise scores
                language_expertise = [
                    keyword_features.get('python_expertise', 0.0),
                    keyword_features.get('java_expertise', 0.0),
                    keyword_features.get('javascript_expertise', 0.0),
                    keyword_features.get('c++_expertise', 0.0),
                    keyword_features.get('ruby_expertise', 0.0),
                    keyword_features.get('go_expertise', 0.0),
                    keyword_features.get('c#_expertise', 0.0),
                    keyword_features.get('php_expertise', 0.0),
                    keyword_features.get('swift_expertise', 0.0),
                    keyword_features.get('kotlin_expertise', 0.0),
                    keyword_features.get('typescript_expertise', 0.0)
                ]
                keyword_feature_list.extend(language_expertise)
                
                # Topic-specific expertise scores
                topic_expertise = [
                    keyword_features.get('testing_expertise', 0.0),
                    keyword_features.get('github_expertise', 0.0),
                    keyword_features.get('security_expertise', 0.0),
                    keyword_features.get('performance_expertise', 0.0),
                    keyword_features.get('architecture_expertise', 0.0),
                    keyword_features.get('data_analysis_expertise', 0.0),
                    keyword_features.get('devops_expertise', 0.0),
                    keyword_features.get('frontend_expertise', 0.0),
                    keyword_features.get('backend_expertise', 0.0),
                    keyword_features.get('cloud_expertise', 0.0),
                    keyword_features.get('ai_ml_expertise', 0.0),
                    keyword_features.get('mobile_expertise', 0.0),
                    keyword_features.get('database_expertise', 0.0)
                ]
                keyword_feature_list.extend(topic_expertise)
                
                # Educational and structural features
                educational_features = [
                    keyword_features.get('educational_course_structure', 0.0),
                    keyword_features.get('educational_academic_patterns', 0.0),
                    keyword_features.get('educational_learning_progression', 0.0)
                ]
                keyword_feature_list.extend(educational_features)
                
                # Code structure features
                code_structure_features = [
                    keyword_features.get('avg_lines_per_file', 0.0),
                    keyword_features.get('avg_functions_per_file', 0.0),
                    keyword_features.get('avg_classes_per_file', 0.0),
                    keyword_features.get('comment_ratio', 0.0),
                    keyword_features.get('function_density', 0.0),
                    keyword_features.get('class_density', 0.0),
                    keyword_features.get('language_count', 0.0),
                    keyword_features.get('file_count', 0.0)
                ]
                keyword_feature_list.extend(code_structure_features)
                
                # Ensure we have exactly 51 features
                if len(keyword_feature_list) < self.keyword_feature_count:
                    keyword_feature_list.extend([0.0] * (self.keyword_feature_count - len(keyword_feature_list)))
                elif len(keyword_feature_list) > self.keyword_feature_count:
                    keyword_feature_list = keyword_feature_list[:self.keyword_feature_count]
                
                features.extend(keyword_feature_list)
            
            # 5. Structure Features (20 features)
            structure_feature_list = [
                structure_features.get('total_files', 0),
                structure_features.get('total_directories', 0),
                structure_features.get('source_files', 0),
                structure_features.get('avg_files_per_dir', 0.0),
                structure_features.get('max_directory_depth', 0),
                structure_features.get('has_tests', 0.0),
                structure_features.get('has_docs', 0.0),
                structure_features.get('has_config', 0.0),
                structure_features.get('has_dependencies', 0.0),
                structure_features.get('has_ci_cd', 0.0),
                structure_features.get('has_docker', 0.0),
                structure_features.get('language_diversity', 0),
                structure_features.get('total_source_files', 0),
                structure_features.get('main_language_hash', 0),
                # Additional structure features
                structure_features.get('total_files', 0) / max(structure_features.get('total_directories', 1), 1),
                structure_features.get('source_files', 0) / max(structure_features.get('total_files', 1), 1),
                min(1.0, structure_features.get('max_directory_depth', 0) / 10.0),
                structure_features.get('has_tests', 0.0) + structure_features.get('has_docs', 0.0),
                structure_features.get('has_config', 0.0) + structure_features.get('has_dependencies', 0.0),
                structure_features.get('has_ci_cd', 0.0) + structure_features.get('has_docker', 0.0)
            ]
            features.extend(structure_feature_list)
            
            # 6. Semantic Features (9 features)
            if codebert_features and 'semantic_features' in codebert_features:
                semantic_features = codebert_features['semantic_features']
                semantic_feature_list = [
                    semantic_features.get('embedding_mean', 0.0),
                    semantic_features.get('embedding_std', 0.0),
                    semantic_features.get('embedding_max', 0.0),
                    semantic_features.get('embedding_min', 0.0),
                    semantic_features.get('embedding_range', 0.0),
                    semantic_features.get('embedding_skewness', 0.0),
                    semantic_features.get('embedding_kurtosis', 0.0),
                    semantic_features.get('semantic_diversity', 0.0),
                    semantic_features.get('semantic_coherence', 0.0)
                ]
            else:
                semantic_feature_list = [0.0] * self.semantic_feature_count
            
            features.extend(semantic_feature_list)
            
            # Ensure we have exactly the expected number of features
            if len(features) != self.total_features:
                print(f"⚠️  Feature count mismatch: expected {self.total_features}, got {len(features)}")
                # Pad or truncate
                if len(features) < self.total_features:
                    features.extend([0.0] * (self.total_features - len(features)))
                else:
                    features = features[:self.total_features]
            
            print(f"✅ Extracted {len(features)} features successfully")
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None
    
    def get_feature_names(self) -> List[str]:
        """Get comprehensive feature names."""
        feature_names = []
        
        # AST feature names
        ast_names = [
            'avg_path_length', 'max_path_length', 'path_variety', 'node_type_diversity',
            'complexity_score', 'nesting_depth', 'function_count', 'class_count',
            'interface_count', 'inheritance_depth', 'total_ast_nodes', 'unique_node_types',
            'ast_depth', 'branching_factor'
        ]
        feature_names.extend([f"ast_{name}" for name in ast_names])
        
        # CodeBERT feature names
        for i in range(self.codebert_dimension):
            feature_names.append(f"codebert_dim_{i}")
        
        # Significant dimension names
        for i in range(self.significant_dimension_count):
            feature_names.append(f"significant_dim_{i}")
        
        # Keyword feature names
        keyword_names = [
            'framework_keywords', 'library_keywords', 'data_science_keywords', 'web_keywords',
            'cli_keywords', 'game_keywords', 'mobile_keywords', 'testing_keywords',
            'database_keywords', 'cloud_keywords', 'total_keywords', 'keyword_diversity'
        ]
        # Pad to 51 features
        keyword_names.extend([f"keyword_extra_{i}" for i in range(self.keyword_feature_count - len(keyword_names))])
        feature_names.extend(keyword_names)
        
        # Structure feature names
        structure_names = [
            'total_files', 'total_directories', 'source_files', 'avg_files_per_dir',
            'max_directory_depth', 'has_tests', 'has_docs', 'has_config', 'has_dependencies',
            'has_ci_cd', 'has_docker', 'language_diversity', 'total_source_files',
            'main_language_hash', 'files_per_dir_ratio', 'source_file_ratio',
            'depth_normalized', 'test_doc_score', 'config_dep_score', 'ci_docker_score'
        ]
        feature_names.extend(structure_names)
        
        # Semantic feature names
        semantic_names = [
            'embedding_mean', 'embedding_std', 'embedding_max', 'embedding_min',
            'embedding_range', 'embedding_skewness', 'embedding_kurtosis',
            'semantic_diversity', 'semantic_coherence'
        ]
        feature_names.extend(semantic_names)
        
        return feature_names
    
    def get_target_names(self) -> List[str]:
        """Get target variable names for classification."""
        return [
            # Architectural patterns
            "web_application", "data_science", "cli_tool", "mobile_app", 
            "game_development", "library", "educational", "microservices",
            "monolithic", "api_project", "testing_project", "documentation_project",
            
            # Quality categories
            "high_quality", "medium_quality", "low_quality",
            
            # Experience levels
            "junior", "intermediate", "senior",
            
            # Specializations
            "frontend_specialist", "backend_specialist", "data_scientist",
            "devops_specialist", "mobile_developer", "game_developer"
        ]
    
    def create_training_dataset(self, dataset_path: str = "dataset") -> Tuple[np.ndarray, np.ndarray]:
        """Create training dataset from the dataset folder with category structure."""
        print("📊 CREATING TRAINING DATASET")
        print("=" * 60)
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path not found: {dataset_path}")
        
        # Get all categories and repositories in the dataset
        categories = []
        repositories = []
        
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if os.path.isdir(category_path):
                categories.append(category)
                
                # Get repositories in this category
                for repo_name in os.listdir(category_path):
                    repo_path = os.path.join(category_path, repo_name)
                    if os.path.isdir(repo_path):
                        repositories.append((category, repo_name, repo_path))
        
        print(f"📁 Found {len(categories)} categories: {', '.join(categories)}")
        print(f"📁 Found {len(repositories)} repositories across all categories")
        
        # Prepare features and labels
        features_list = []
        labels_list = []
        
        self.feature_names = self.get_feature_names()
        self.target_names = self.get_target_names()
        
        for i, (category, repo_name, repo_path) in enumerate(repositories, 1):
            print(f"   [{i:3d}/{len(repositories)}] Processing: {category}/{repo_name}")
            
            try:
                # Extract comprehensive features with better error handling
                features = self.extract_comprehensive_features(repo_path)
                
                if features is not None:
                    # Create labels based on repository characteristics and category
                    labels = self._create_labels_with_category(repo_path, repo_name, category)
                    
                    features_list.append(features)
                    labels_list.append(labels)
                    
                    print(f"      ✅ Features: {features.shape}, Labels: {len(labels)}")
                else:
                    print(f"      ❌ Feature extraction failed")
                    
            except KeyboardInterrupt:
                print(f"      ⚠️  Interrupted, stopping training")
                raise
            except Exception as e:
                print(f"      ❌ Error processing {category}/{repo_name}: {e}")
                # Continue with next repository instead of stopping
                continue
        
        if not features_list:
            raise ValueError("No features extracted from dataset")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"\n📊 Dataset prepared successfully:")
        print(f"   • Features: {X.shape[1]} dimensions")
        print(f"   • Samples: {X.shape[0]} repositories")
        print(f"   • Targets: {y.shape[1]} classification targets")
        print(f"   • Categories: {len(categories)}")
        
        return X, y
    
    def _create_labels_with_category(self, repo_path: str, repo_name: str, category: str) -> List[int]:
        """Create multi-label classification targets for a repository with category information."""
        labels = [0] * len(self.target_names)
        
        try:
            # Analyze repository to determine labels
            analysis_results = self.analyzer.analyze_repository(repo_path)
            
            # Extract patterns and characteristics
            detected_patterns = analysis_results.get('architecture_analysis', {}).get('detected_patterns', [])
            quality_score = analysis_results.get('quality_assessment', {}).get('overall_score', 0.5)
            experience_level = analysis_results.get('programmer_characteristics', {}).get('experience_level', 'junior')
            specialization = analysis_results.get('programmer_characteristics', {}).get('specialization', 'generalist')
            
            # Set architectural pattern labels based on category first, then analysis
            category_pattern = self._map_category_to_pattern(category)
            if category_pattern in self.target_names:
                idx = self.target_names.index(category_pattern)
                labels[idx] = 1
            
            # Also set patterns from analysis
            for pattern in detected_patterns:
                pattern_lower = pattern.lower().replace(' ', '_')
                if pattern_lower in self.target_names:
                    idx = self.target_names.index(pattern_lower)
                    labels[idx] = 1
            
            # Set quality labels
            if quality_score > 0.7:
                quality_idx = self.target_names.index("high_quality")
                labels[quality_idx] = 1
            elif quality_score > 0.4:
                quality_idx = self.target_names.index("medium_quality")
                labels[quality_idx] = 1
            else:
                quality_idx = self.target_names.index("low_quality")
                labels[quality_idx] = 1
            
            # Set experience level labels
            if experience_level.lower() in self.target_names:
                exp_idx = self.target_names.index(experience_level.lower())
                labels[exp_idx] = 1
            
            # Set specialization labels based on category and analysis
            category_specialization = self._map_category_to_specialization(category)
            if category_specialization in self.target_names:
                spec_idx = self.target_names.index(category_specialization)
                labels[spec_idx] = 1
            
            # Also set specialization from analysis
            if 'data_science' in specialization.lower():
                spec_idx = self.target_names.index("data_scientist")
                labels[spec_idx] = 1
            elif 'frontend' in specialization.lower():
                spec_idx = self.target_names.index("frontend_specialist")
                labels[spec_idx] = 1
            elif 'backend' in specialization.lower():
                spec_idx = self.target_names.index("backend_specialist")
                labels[spec_idx] = 1
            elif 'mobile' in specialization.lower():
                spec_idx = self.target_names.index("mobile_developer")
                labels[spec_idx] = 1
            elif 'game' in specialization.lower():
                spec_idx = self.target_names.index("game_developer")
                labels[spec_idx] = 1
            elif 'devops' in specialization.lower():
                spec_idx = self.target_names.index("devops_specialist")
                labels[spec_idx] = 1
            
        except Exception as e:
            print(f"      ⚠️  Error creating labels: {e}")
            # Set default labels based on category and repository name
            category_pattern = self._map_category_to_pattern(category)
            if category_pattern in self.target_names:
                labels[self.target_names.index(category_pattern)] = 1
            
            # Fallback based on repository name
            if 'cli' in repo_name.lower():
                labels[self.target_names.index("cli_tool")] = 1
            elif 'data' in repo_name.lower() or 'science' in repo_name.lower():
                labels[self.target_names.index("data_science")] = 1
            elif 'web' in repo_name.lower():
                labels[self.target_names.index("web_application")] = 1
            elif 'mobile' in repo_name.lower():
                labels[self.target_names.index("mobile_app")] = 1
            elif 'game' in repo_name.lower():
                labels[self.target_names.index("game_development")] = 1
            elif 'library' in repo_name.lower():
                labels[self.target_names.index("library")] = 1
            else:
                # Default based on category
                if category_pattern not in self.target_names:
                    labels[self.target_names.index("web_application")] = 1
        
        return labels
    
    def _map_category_to_pattern(self, category: str) -> str:
        """Map dataset category to architectural pattern."""
        category_lower = category.lower()
        
        if 'cli' in category_lower:
            return "cli_tool"
        elif 'data' in category_lower and 'science' in category_lower:
            return "data_science"
        elif 'web' in category_lower:
            return "web_application"
        elif 'mobile' in category_lower:
            return "mobile_app"
        elif 'game' in category_lower:
            return "game_development"
        elif 'library' in category_lower:
            return "library"
        elif 'educational' in category_lower:
            return "educational"
        else:
            return "web_application"  # default
    
    def _map_category_to_specialization(self, category: str) -> str:
        """Map dataset category to specialization."""
        category_lower = category.lower()
        
        if 'data' in category_lower and 'science' in category_lower:
            return "data_scientist"
        elif 'mobile' in category_lower:
            return "mobile_developer"
        elif 'game' in category_lower:
            return "game_developer"
        elif 'devops' in category_lower:
            return "devops_specialist"
        else:
            return "backend_specialist"  # default
    
    def _create_labels(self, repo_path: str, repo_name: str) -> List[int]:
        """Create multi-label classification targets for a repository (legacy method)."""
        return self._create_labels_with_category(repo_path, repo_name, "unknown")
    
    def train_model(self, X: np.ndarray, y: np.ndarray, n_trees: int = 100) -> Dict:
        """Train the CRAV3 Random Forest model."""
        print("🌲 TRAINING CRAV3 RANDOM FOREST MODEL")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"📊 Training set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")
        
        # Scale features
        print("🔧 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print(f"🌲 Training Random Forest with {n_trees} trees...")
        base_rf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.rf_classifier = MultiOutputClassifier(base_rf)
        self.rf_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate model
        print("📊 Evaluating model...")
        y_pred = self.rf_classifier.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        hamming_loss_score = hamming_loss(y_test, y_pred)
        
        print(f"📊 Model Performance:")
        print(f"   • Accuracy: {accuracy:.4f}")
        print(f"   • Hamming Loss: {hamming_loss_score:.4f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(self.rf_classifier, X_train_scaled, y_train, cv=3)
        print(f"   • Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model
        model_path = os.path.join(self.models_dir, 'crav3_random_forest.joblib')
        scaler_path = os.path.join(self.models_dir, 'crav3_random_forest_scaler.joblib')
        feature_names_path = os.path.join(self.models_dir, 'crav3_random_forest_feature_names.json')
        target_names_path = os.path.join(self.models_dir, 'crav3_random_forest_target_names.json')
        
        joblib.dump(self.rf_classifier, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature and target names
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f)
        with open(target_names_path, 'w') as f:
            json.dump(self.target_names, f)
        
        print(f"💾 CRAV3 Random Forest saved to {model_path}")
        print(f"💾 Feature names saved to {feature_names_path}")
        print(f"💾 Target names saved to {target_names_path}")
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        return {
            "n_trees": n_trees,
            "n_features": X.shape[1],
            "n_targets": y.shape[1],
            "n_samples": X.shape[0],
            "accuracy": accuracy,
            "hamming_loss": hamming_loss_score,
            "cv_score": cv_scores.mean(),
            "cv_std": cv_scores.std()
        }
    
    def analyze_feature_importance(self):
        """Analyze feature importance across all targets."""
        if self.rf_classifier is None:
            print("❌ Model not trained")
            return
        
        print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        # Get feature importance from each target classifier
        for i, estimator in enumerate(self.rf_classifier.estimators_):
            target_name = self.target_names[i]
            importance = estimator.feature_importances_
            
            # Get top 10 features for each target
            feature_importance = list(zip(self.feature_names, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🎯 {target_name.upper()} - Top Features:")
            for j, (feature, imp) in enumerate(feature_importance[:10], 1):
                print(f"   {j:2d}. {feature}: {imp:.4f}")
    
    def predict(self, repository_path: str) -> Dict:
        """Make predictions for a new repository."""
        if self.rf_classifier is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Extract features
        features = self.extract_comprehensive_features(repository_path)
        if features is None:
            raise ValueError("Feature extraction failed")
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions
        predictions = self.rf_classifier.predict(features_scaled)[0]
        probabilities = self.rf_classifier.predict_proba(features_scaled)
        
        # Ensure we have target names
        if not self.target_names:
            self.target_names = self.get_target_names()
        
        # Format results
        detected_targets = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            if pred == 1 and i < len(self.target_names):
                target_name = self.target_names[i]
                confidence = np.max(proba) if len(proba) > 0 else 0.5
                
                detected_targets.append({
                    "target": target_name,
                    "confidence": confidence,
                    "type": self._get_target_type(target_name)
                })
        
        # Sort by confidence
        detected_targets.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "repository_path": repository_path,
            "predictions": detected_targets,
            "target_count": len(detected_targets),
            "feature_vector_shape": features.shape
        }
    
    def _get_target_type(self, target_name: str) -> str:
        """Get the type of a target variable."""
        if target_name in ["web_application", "data_science", "cli_tool", "mobile_app", 
                          "game_development", "library", "educational", "microservices", "monolithic"]:
            return "architectural_pattern"
        elif target_name in ["high_quality", "medium_quality", "low_quality"]:
            return "quality_category"
        elif target_name in ["junior", "intermediate", "senior"]:
            return "experience_level"
        elif target_name in ["frontend_specialist", "backend_specialist", "data_scientist",
                           "devops_specialist", "mobile_developer", "game_developer"]:
            return "specialization"
        else:
            return "unknown"

def main():
    """Main function to train the CRAV3 Random Forest model."""
    print("🚀 CRAV3 RANDOM FOREST MODEL TRAINING")
    print("=" * 80)
    
    # Initialize model
    model = CRAV3RandomForest()
    
    try:
        # Create training dataset
        X, y = model.create_training_dataset()
        
        # Train model
        training_info = model.train_model(X, y, n_trees=100)
        
        print(f"\n✅ CRAV3 RANDOM FOREST TRAINED SUCCESSFULLY")
        print(f"   • {training_info['n_trees']} parallel decision trees")
        print(f"   • {training_info['n_features']} comprehensive features")
        print(f"   • {training_info['n_targets']} classification targets")
        print(f"   • {training_info['n_samples']} training repositories")
        print(f"   • Accuracy: {training_info['accuracy']:.4f}")
        print(f"   • Hamming Loss: {training_info['hamming_loss']:.4f}")
        print(f"   • Cross-validation: {training_info['cv_score']:.4f} (+/- {training_info['cv_std']*2:.4f})")
        
        # Test prediction on a sample repository
        print(f"\n🧪 TESTING PREDICTION")
        print("=" * 50)
        
        # Find a test repository from the category structure
        dataset_path = "dataset"
        test_repo_path = None
        
        # Look for a test repository in any category
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
            prediction = model.predict(test_repo_path)
            
            print(f"🎯 Predictions:")
            for pred in prediction['predictions'][:5]:  # Show top 5
                print(f"   • {pred['target']} ({pred['type']}, confidence: {pred['confidence']:.3f})")
            
            print(f"📊 Total targets detected: {prediction['target_count']}")
        else:
            print("⚠️  No test repository found")
        
        print(f"\n🎉 CRAV3 Random Forest model is ready for use!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
