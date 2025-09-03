#!/usr/bin/env python3
"""
CRAV3 Random Forest NEW - Optimized Version
===========================================

This script creates an optimized Random Forest model that:
1. Uses CRAv3 as a preliminary feature extractor
2. Saves and reuses CodeBERT embeddings and AST features
3. Applies Random Forest for final classification
4. Optimized for performance and accuracy
"""

import os
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import torch for CUDA support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - CUDA acceleration disabled")

# Import the comprehensive analyzer
import sys
sys.path.append('scripts/analysis')
from comprehensive_repository_analyzer_v3 import ComprehensiveRepositoryAnalyzerV3
try:
    from enhanced_keyword_analyzer import EnhancedKeywordAnalyzer
except Exception:
    EnhancedKeywordAnalyzer = None

class CRAV3RandomForestNew:
    """Optimized Random Forest model using CRAv3 as feature extractor."""
    
    def __init__(self, models_dir: str = "ml_models", cache_dir: str = "feature_cache"):
        """Initialize the optimized CRAV3 Random Forest model."""
        self.models_dir = models_dir
        self.cache_dir = cache_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Model components
        self.rf_classifier = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_names = []
        
        # Analyzer instances
        self.analyzer = ComprehensiveRepositoryAnalyzerV3()
        self.keyword_analyzer = EnhancedKeywordAnalyzer() if EnhancedKeywordAnalyzer else None
        
        # CUDA optimization
        self.use_cuda = TORCH_AVAILABLE and torch.cuda.is_available()
        if self.use_cuda:
            print(f"🚀 CUDA detected: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("💻 Using CPU for computations")
        
        # Feature dimensions
        self.ast_feature_count = 14
        self.codebert_dimension = 768
        self.significant_dimension_count = 50
        self.keyword_feature_count = 51
        self.structure_feature_count = 20
        self.semantic_feature_count = 9
        self.crav3_logic_feature_count = 53
        
        # Total expected features
        self.total_features = (
            self.ast_feature_count + 
            self.codebert_dimension + 
            self.significant_dimension_count + 
            self.keyword_feature_count + 
            self.structure_feature_count + 
            self.semantic_feature_count +
            self.crav3_logic_feature_count
        )
        
        # Try to load pre-trained model if it exists
        self._load_pretrained_model()
        
        print(f"🔧 CRAV3 Random Forest NEW initialized")
        print(f"   Expected features: {self.total_features}")
        print(f"   • AST features: {self.ast_feature_count}")
        print(f"   • CodeBERT embeddings: {self.codebert_dimension}")
        print(f"   • Significant dimensions: {self.significant_dimension_count}")
        print(f"   • Keyword features: {self.keyword_feature_count}")
        print(f"   • Structure features: {self.structure_feature_count}")
        print(f"   • Semantic features: {self.semantic_feature_count}")
        print(f"   • CRAv3 logic features: {self.crav3_logic_feature_count}")
        
        # Report model loading status
        if self.rf_classifier is not None:
            print(f"✅ Pre-trained model loaded successfully")
        else:
            print(f"⚠️  No pre-trained model found - training required")
    
    def _load_pretrained_model(self):
        """Load pre-trained model and scaler if they exist."""
        model_path = os.path.join(self.models_dir, 'crav3_random_forest_new.joblib')
        scaler_path = os.path.join(self.models_dir, 'crav3_random_forest_new_scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                print("🔄 Loading pre-trained model...")
                self.rf_classifier = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # Load feature and target names if available
                feature_names_path = os.path.join(self.models_dir, 'crav3_random_forest_new_feature_names.json')
                target_names_path = os.path.join(self.models_dir, 'crav3_random_forest_new_target_names.json')
                
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
    
    def _get_cache_path(self, repository_path: str, feature_type: str) -> str:
        """Get cache path for a specific repository and feature type."""
        repo_name = os.path.basename(repository_path)
        return os.path.join(self.cache_dir, f"{repo_name}_{feature_type}.json")
    
    def _is_cache_valid(self, cache_path: str, max_age_hours: int = 24) -> bool:
        """Check if cached features are still valid."""
        if not os.path.exists(cache_path):
            return False
        
        # Check file age
        file_age = time.time() - os.path.getmtime(cache_path)
        if file_age >= (max_age_hours * 3600):
            return False
        
        # Check if cache file is corrupted or incompatible
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Basic validation - check if data structure is reasonable
            if not isinstance(cached_data, dict):
                return False
            
            # Check for common data corruption patterns
            if any(isinstance(v, str) and v.startswith('[') and v.endswith(']') for v in cached_data.values()):
                print(f"   ⚠️  Cache corrupted (string arrays detected) - will regenerate")
                return False
            
            return True
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"   ⚠️  Cache corrupted ({e}) - will regenerate")
            return False
    
    def _save_features_to_cache(self, repository_path: str, feature_type: str, features: Any):
        """Save features to cache."""
        cache_path = self._get_cache_path(repository_path, feature_type)
        try:
            with open(cache_path, 'w') as f:
                json.dump(features, f, default=str)
        except Exception as e:
            print(f"⚠️  Failed to save {feature_type} features to cache: {e}")
    
    def _load_features_from_cache(self, repository_path: str, feature_type: str) -> Optional[Any]:
        """Load features from cache if valid."""
        cache_path = self._get_cache_path(repository_path, feature_type)
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                
                # Additional validation after loading
                if self._validate_cached_data(cached_data, feature_type):
                    return cached_data
                else:
                    print(f"   ⚠️  Cached {feature_type} data validation failed - will regenerate")
                    # Remove corrupted cache
                    try:
                        os.remove(cache_path)
                        print(f"   🗑️  Removed corrupted cache: {cache_path}")
                    except Exception as e:
                        print(f"   ⚠️  Failed to remove corrupted cache: {e}")
                    return None
                    
            except Exception as e:
                print(f"⚠️  Failed to load {feature_type} features from cache: {e}")
                # Remove corrupted cache
                try:
                    os.remove(cache_path)
                    print(f"   🗑️  Removed corrupted cache: {cache_path}")
                except Exception as e2:
                    print(f"   ⚠️  Failed to remove corrupted cache: {e2}")
        return None

    def _validate_cached_data(self, cached_data: Any, feature_type: str) -> bool:
        """Validate cached data structure and content."""
        if not isinstance(cached_data, dict):
            return False
        
        # Type-specific validation
        if feature_type == "ast":
            # Check for expected AST keys and numeric values
            expected_keys = ['avg_path_length', 'max_path_length', 'complexity_score', 'function_count']
            for key in expected_keys:
                if key in cached_data:
                    value = cached_data[key]
                    if not isinstance(value, (int, float)) or (isinstance(value, str) and value.startswith('[')):
                        return False
        
        elif feature_type == "codebert":
            # Check for expected CodeBERT structure
            if 'repository_embedding' in cached_data:
                repo_emb = cached_data['repository_embedding']
                if not isinstance(repo_emb, list) or not all(isinstance(x, (int, float)) for x in repo_emb):
                    return False
        
        elif feature_type == "keyword":
            # Check for expected keyword structure
            expected_keys = ['framework_keywords', 'library_keywords', 'total_keywords']
            for key in expected_keys:
                if key in cached_data:
                    value = cached_data[key]
                    if not isinstance(value, (int, float)) or (isinstance(value, str) and value.startswith('[')):
                        return False
        
        elif feature_type == "structure":
            # Check for expected structure keys
            expected_keys = ['total_files', 'total_directories', 'source_files']
            for key in expected_keys:
                if key in cached_data:
                    value = cached_data[key]
                    if not isinstance(value, (int, float)) or (isinstance(value, str) and value.startswith('[')):
                        return False
        
        return True
    
    def extract_optimized_features(self, repository_path: str) -> Optional[np.ndarray]:
        """Extract features using CRAv3 with caching and optimization."""
        try:
            print(f"🔍 Extracting optimized features from: {repository_path}")
            
            # 1. AST Features (with caching)
            ast_cache_path = self._get_cache_path(repository_path, "ast")
            if self._is_cache_valid(ast_cache_path):
                print("   📋 Loading AST features from cache...")
                ast_features = self._load_features_from_cache(repository_path, "ast")
            else:
                print("   🌳 Extracting AST features...")
                try:
                    ast_features = self.analyzer.extract_ast_features(repository_path)
                    self._save_features_to_cache(repository_path, "ast", ast_features)
                except Exception as e:
                    print(f"   ⚠️  AST extraction failed: {e}, using empty features")
                    ast_features = self._get_empty_ast_features()
            
            # 2. CodeBERT Embeddings (with caching and CUDA optimization)
            codebert_cache_path = self._get_cache_path(repository_path, "codebert")
            if self._is_cache_valid(codebert_cache_path):
                print("   📋 Loading CodeBERT features from cache...")
                codebert_features = self._load_features_from_cache(repository_path, "codebert")
            else:
                print("   🤖 Extracting optimized CodeBERT embeddings via CRAv3...")
                try:
                    # Use CRAv3's optimized CodeBERT extraction with CUDA support
                    codebert_features = self.analyzer.extract_codebert_embeddings(repository_path)
                    self._save_features_to_cache(repository_path, "codebert", codebert_features)
                except Exception as e:
                    print(f"   ⚠️  CodeBERT extraction failed: {e}, using empty features")
                    codebert_features = self._get_empty_codebert_features()
            
            # 3. Keyword Features (with caching)
            keyword_cache_path = self._get_cache_path(repository_path, "keyword")
            if self._is_cache_valid(keyword_cache_path):
                print("   📋 Loading keyword features from cache...")
                keyword_features = self._load_features_from_cache(repository_path, "keyword")
            else:
                print("   🔎 Extracting keyword features...")
                try:
                    keyword_features = self.analyzer.extract_keyword_features(repository_path)
                    self._save_features_to_cache(repository_path, "keyword", keyword_features)
                except Exception as e:
                    print(f"   ⚠️  Keyword extraction failed: {e}, using basic extraction")
                    try:
                        keyword_features = self.analyzer._extract_basic_keyword_features(repository_path)
                        self._save_features_to_cache(repository_path, "keyword", keyword_features)
                    except Exception as e2:
                        print(f"   ⚠️  Basic keyword extraction also failed: {e2}, using empty features")
                        keyword_features = self._get_empty_keyword_features()
            
            # 4. Structure Features (with caching)
            structure_cache_path = self._get_cache_path(repository_path, "structure")
            if self._is_cache_valid(structure_cache_path):
                print("   📋 Loading structure features from cache...")
                structure_features = self._load_features_from_cache(repository_path, "structure")
            else:
                print("   📁 Extracting structure features...")
                try:
                    structure_features = self.analyzer.extract_file_structure_features(repository_path)
                    self._save_features_to_cache(repository_path, "structure", structure_features)
                except Exception as e:
                    print(f"   ⚠️  Structure extraction failed: {e}, using empty features")
                    structure_features = self._get_empty_structure_features()
            
            # Build combined features dict like CRAv3 does for logic-level analysis
            all_features_dict = self._build_all_features_dict(ast_features, codebert_features, keyword_features, structure_features)

            # Derive CRAv3 logic outputs without re-running heavy extractors
            quality_assessment = self.analyzer.assess_code_quality(all_features_dict)
            programmer_characteristics = self.analyzer.analyze_programmer_characteristics(all_features_dict)
            architecture_analysis = self.analyzer.analyze_architecture_patterns(all_features_dict)

            # Combine all features into a single vector
            features = []
            
            # 1. AST Features (14 features)
            ast_feature_list = self._extract_ast_feature_list(ast_features)
            features.extend(ast_feature_list)
            
            # 2. CodeBERT Embeddings (768 dimensions)
            codebert_feature_list = self._extract_codebert_feature_list(codebert_features)
            features.extend(codebert_feature_list)
            
            # 3. Significant Dimensions (50 features)
            significant_feature_list = self._extract_significant_feature_list(codebert_features)
            features.extend(significant_feature_list)
            
            # 4. Keyword Features (51 features)
            keyword_feature_list = self._extract_keyword_feature_list(keyword_features)
            features.extend(keyword_feature_list)
            
            # 5. Structure Features (20 features)
            structure_feature_list = self._extract_structure_feature_list(structure_features)
            features.extend(structure_feature_list)
            
            # 6. Semantic Features (9 features)
            semantic_feature_list = self._extract_semantic_feature_list(codebert_features)
            features.extend(semantic_feature_list)
            
            # 7. CRAv3 Logic Features (53 features)
            crav3_logic_features = self._extract_crav3_logic_feature_list(
                quality_assessment,
                programmer_characteristics,
                architecture_analysis
            )
            features.extend(crav3_logic_features)
            
            # Ensure we have exactly the expected number of features
            if len(features) != self.total_features:
                print(f"⚠️  Feature count mismatch: expected {self.total_features}, got {len(features)}")
                # Pad or truncate
                if len(features) < self.total_features:
                    features.extend([0.0] * (self.total_features - len(features)))
                else:
                    features = features[:self.total_features]
            
            print(f"   ✅ Extracted {len(features)} features successfully")
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None
    
    def _get_empty_ast_features(self) -> Dict[str, Any]:
        """Get empty AST features."""
        return {
            'avg_path_length': 0.0, 'max_path_length': 0.0, 'path_variety': 0.0,
            'node_type_diversity': 0.0, 'complexity_score': 0.0, 'nesting_depth': 0.0,
            'function_count': 0.0, 'class_count': 0.0, 'interface_count': 0.0,
            'inheritance_depth': 0.0, 'total_ast_nodes': 0, 'unique_node_types': 0,
            'ast_depth': 0.0, 'branching_factor': 0.0
        }
    
    def _get_empty_codebert_features(self) -> Dict[str, Any]:
        """Get empty CodeBERT features."""
        return {
            'repository_embedding': [0.0] * self.codebert_dimension,
            'significant_dimensions': {},
            'semantic_features': {}
        }
    
    def _get_empty_keyword_features(self) -> Dict[str, Any]:
        """Get empty keyword features."""
        return {
            'framework_keywords': 0.0, 'library_keywords': 0.0, 'data_science_keywords': 0.0,
            'web_keywords': 0.0, 'cli_keywords': 0.0, 'game_keywords': 0.0,
            'mobile_keywords': 0.0, 'testing_keywords': 0.0, 'database_keywords': 0.0,
            'cloud_keywords': 0.0, 'total_keywords': 0, 'keyword_diversity': 0.0
        }
    
    def _get_empty_structure_features(self) -> Dict[str, Any]:
        """Get empty structure features."""
        return {
            'total_files': 0, 'total_directories': 0, 'source_files': 0,
            'avg_files_per_dir': 0.0, 'max_directory_depth': 0, 'has_tests': 0.0,
            'has_docs': 0.0, 'has_config': 0.0, 'has_dependencies': 0.0,
            'has_ci_cd': 0.0, 'has_docker': 0.0, 'language_diversity': 0,
            'total_source_files': 0, 'main_language_hash': 0
        }
    
    def _extract_ast_feature_list(self, ast_features: Dict[str, Any]) -> List[float]:
        """Extract AST features as a list."""
        def safe_float(value, default=0.0):
            """Safely convert value to float."""
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        return [
            safe_float(ast_features.get('avg_path_length', 0.0)),
            safe_float(ast_features.get('max_path_length', 0.0)),
            safe_float(ast_features.get('path_variety', 0.0)),
            safe_float(ast_features.get('node_type_diversity', 0.0)),
            safe_float(ast_features.get('complexity_score', 0.0)),
            safe_float(ast_features.get('nesting_depth', 0.0)),
            safe_float(ast_features.get('function_count', 0.0)),
            safe_float(ast_features.get('class_count', 0.0)),
            safe_float(ast_features.get('interface_count', 0.0)),
            safe_float(ast_features.get('inheritance_depth', 0.0)),
            safe_float(ast_features.get('total_ast_nodes', 0)),
            safe_float(ast_features.get('unique_node_types', 0)),
            safe_float(ast_features.get('ast_depth', 0.0)),
            safe_float(ast_features.get('branching_factor', 0.0))
        ]
    
    def _extract_codebert_feature_list(self, codebert_features: Dict[str, Any]) -> List[float]:
        """Extract CodeBERT features as a list."""
        def safe_float(value, default=0.0):
            """Safely convert value to float."""
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        if codebert_features and 'repository_embedding' in codebert_features:
            repo_embedding = codebert_features['repository_embedding']
            if isinstance(repo_embedding, list):
                # Convert all values to float safely
                safe_embeddings = [safe_float(x) for x in repo_embedding]
                
                if len(safe_embeddings) == self.codebert_dimension:
                    return safe_embeddings
                else:
                    # Pad or truncate to 768 dimensions
                    if len(safe_embeddings) < self.codebert_dimension:
                        return safe_embeddings + [0.0] * (self.codebert_dimension - len(safe_embeddings))
                    else:
                        return safe_embeddings[:self.codebert_dimension]
            else:
                return [0.0] * self.codebert_dimension
        else:
            # Fill with zeros if no CodeBERT features
            return [0.0] * self.codebert_dimension
    
    def _extract_significant_feature_list(self, codebert_features: Dict[str, Any]) -> List[float]:
        """Extract significant dimensions as a list."""
        def safe_float(value, default=0.0):
            """Safely convert value to float."""
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        if codebert_features and 'significant_dimensions' in codebert_features:
            significant_dims = codebert_features['significant_dimensions']
            if isinstance(significant_dims, dict):
                # Extract top 50 significant dimensions
                significant_values = list(significant_dims.values())[:self.significant_dimension_count]
                # Convert all values to float safely
                safe_values = [safe_float(x) for x in significant_values]
                
                if len(safe_values) < self.significant_dimension_count:
                    safe_values.extend([0.0] * (self.significant_dimension_count - len(safe_values)))
                return safe_values
            else:
                return [0.0] * self.significant_dimension_count
        else:
            return [0.0] * self.significant_dimension_count
    
    def _extract_keyword_feature_list(self, keyword_features: Dict[str, Any]) -> List[float]:
        """Extract keyword features as a list."""
        # Handle both enhanced and basic keyword features
        if 'feature_vector' in keyword_features:
            # Enhanced keyword analyzer result
            keyword_vector = keyword_features['feature_vector']
            if isinstance(keyword_vector, np.ndarray):
                keyword_vector = keyword_vector.tolist()
            
            if len(keyword_vector) == self.keyword_feature_count:
                return keyword_vector
            else:
                # Pad or truncate to 51 features
                if len(keyword_vector) < self.keyword_feature_count:
                    return keyword_vector + [0.0] * (self.keyword_feature_count - len(keyword_vector))
                else:
                    return keyword_vector[:self.keyword_feature_count]
        else:
            # Basic keyword features (fallback)
            return self._extract_basic_keyword_feature_list(keyword_features)
    
    def _extract_basic_keyword_feature_list(self, keyword_features: Dict[str, Any]) -> List[float]:
        """Extract basic keyword features as a list."""
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
        
        return keyword_feature_list
    
    def _extract_structure_feature_list(self, structure_features: Dict[str, Any]) -> List[float]:
        """Extract structure features as a list."""
        def safe_float(value, default=0.0):
            """Safely convert value to float."""
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        def safe_int(value, default=0):
            """Safely convert value to int."""
            if value is None:
                return default
            try:
                return int(float(value))
            except (ValueError, TypeError):
                return default
        
        # Extract basic features safely
        total_files = safe_int(structure_features.get('total_files', 0))
        total_directories = safe_int(structure_features.get('total_directories', 0))
        source_files = safe_int(structure_features.get('source_files', 0))
        max_directory_depth = safe_int(structure_features.get('max_directory_depth', 0))
        total_source_files = safe_int(structure_features.get('total_source_files', 0))
        main_language_hash = safe_int(structure_features.get('main_language_hash', 0))
        
        return [
            float(total_files),
            float(total_directories),
            float(source_files),
            safe_float(structure_features.get('avg_files_per_dir', 0.0)),
            float(max_directory_depth),
            safe_float(structure_features.get('has_tests', 0.0)),
            safe_float(structure_features.get('has_docs', 0.0)),
            safe_float(structure_features.get('has_config', 0.0)),
            safe_float(structure_features.get('has_dependencies', 0.0)),
            safe_float(structure_features.get('has_ci_cd', 0.0)),
            safe_float(structure_features.get('has_docker', 0.0)),
            safe_float(structure_features.get('language_diversity', 0)),
            float(total_source_files),
            float(main_language_hash),
            # Additional structure features
            float(total_files) / max(float(total_directories), 1.0),
            float(source_files) / max(float(total_files), 1.0),
            min(1.0, float(max_directory_depth) / 10.0),
            safe_float(structure_features.get('has_tests', 0.0)) + safe_float(structure_features.get('has_docs', 0.0)),
            safe_float(structure_features.get('has_config', 0.0)) + safe_float(structure_features.get('has_dependencies', 0.0)),
            safe_float(structure_features.get('has_ci_cd', 0.0)) + safe_float(structure_features.get('has_docker', 0.0))
        ]
    
    def _extract_semantic_feature_list(self, codebert_features: Dict[str, Any]) -> List[float]:
        """Extract semantic features as a list."""
        def safe_float(value, default=0.0):
            """Safely convert value to float."""
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        if codebert_features and 'semantic_features' in codebert_features:
            semantic_features = codebert_features['semantic_features']
            if isinstance(semantic_features, dict):
                return [
                    safe_float(semantic_features.get('embedding_mean', 0.0)),
                    safe_float(semantic_features.get('embedding_std', 0.0)),
                    safe_float(semantic_features.get('embedding_max', 0.0)),
                    safe_float(semantic_features.get('embedding_min', 0.0)),
                    safe_float(semantic_features.get('embedding_range', 0.0)),
                    safe_float(semantic_features.get('embedding_skewness', 0.0)),
                    safe_float(semantic_features.get('embedding_kurtosis', 0.0)),
                    safe_float(semantic_features.get('semantic_diversity', 0.0)),
                    safe_float(semantic_features.get('semantic_coherence', 0.0))
                ]
            else:
                return [0.0] * self.semantic_feature_count
        else:
            return [0.0] * self.semantic_feature_count

    def _extract_optimized_codebert_embeddings(self, repository_path: str) -> Dict[str, Any]:
        """Extract CodeBERT embeddings with CUDA optimization and large repo handling."""
        print(f"🤖 Extracting optimized CodeBERT embeddings from {repository_path}")
        
        # Initialize CodeBERT if not already done
        self.analyzer.initialize_codebert()
        
        if self.analyzer.codebert_model is None:
            print("⚠️  CodeBERT not available, skipping embeddings")
            return self._get_empty_codebert_features()
        
        embeddings = []
        file_info = []
        
        # Check if the path is a directory
        if not os.path.isdir(repository_path):
            print(f"⚠️  Path is not a directory: {repository_path}")
            return self._get_empty_codebert_features()
        
        # Collect files to process
        files_to_process = []
        for root, _, files in os.walk(repository_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.analyzer.codebert_languages:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        rel_path = os.path.relpath(file_path, repository_path)
                        files_to_process.append((file_path, rel_path, ext))
        
        if not files_to_process:
            print("⚠️  No supported files found for CodeBERT analysis")
            return self._get_empty_codebert_features()
        
        print(f"   📁 Processing {len(files_to_process)} files...")
        
        # Process files in batches for memory efficiency
        batch_size = 32 if self.use_cuda else 16
        total_files = len(files_to_process)
        
        for i in range(0, total_files, batch_size):
            batch = files_to_process[i:i + batch_size]
            print(f"   🔄 Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
            
            batch_embeddings = []
            batch_file_info = []
            
            for file_path, rel_path, ext in batch:
                try:
                    # Extract embedding with optimized processing
                    embedding = self._extract_optimized_file_embedding(file_path)
                    if embedding is not None:
                        batch_embeddings.append(embedding)
                        batch_file_info.append({
                            "file_path": rel_path,
                            "language": self.analyzer.codebert_languages[ext],
                            "embedding_dim": len(embedding)
                        })
                except Exception as e:
                    print(f"   ⚠️  Error processing {rel_path}: {e}")
                    continue
            
            # Add batch results to main lists
            embeddings.extend(batch_embeddings)
            file_info.extend(batch_file_info)
            
            # Clear GPU memory if using CUDA
            if self.use_cuda:
                torch.cuda.empty_cache()
        
        if not embeddings:
            print("⚠️  No embeddings extracted successfully")
            return self._get_empty_codebert_features()
        
        # Calculate repository-level embedding
        repo_embedding = np.mean(embeddings, axis=0)
        
        # Calculate semantic features
        semantic_features = self._calculate_optimized_semantic_features(embeddings)
        
        # Extract significant dimensions for enhanced analysis
        significant_features = self._extract_optimized_significant_dimensions(repo_embedding)
        
        print(f"   ✅ Successfully extracted embeddings from {len(embeddings)} files")
        
        return {
            "repository_embedding": repo_embedding.tolist(),
            "num_files": len(embeddings),
            "embedding_dimension": len(repo_embedding),
            "file_embeddings": [emb.tolist() for emb in embeddings],
            "file_info": file_info,
            "semantic_features": semantic_features,
            "significant_dimensions": significant_features
        }

    def _extract_optimized_file_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """Extract CodeBERT embedding for a single file with optimization."""
        try:
            # Handle Jupyter notebooks
            if file_path.endswith('.ipynb'):
                code_content = self._extract_python_from_notebook(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code_content = f.read()
            
            if not code_content.strip():
                return None
            
            # Clean code
            code_content = self._clean_code_for_embedding(code_content)
            
            # Use tokenizer's built-in truncation to avoid token length errors
            inputs = self.analyzer.codebert_tokenizer(
                code_content, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512,
                add_special_tokens=True
            )
            
            # Move to device
            device = torch.device('cuda' if self.use_cuda else 'cpu')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Get embeddings with gradient computation disabled for efficiency
            with torch.no_grad():
                outputs = self.analyzer.codebert_model(input_ids, attention_mask=attention_mask)
                file_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            
            return file_embedding
            
        except Exception as e:
            print(f"⚠️  Error extracting embedding from {file_path}: {e}")
            return None

    def _extract_python_from_notebook(self, notebook_path: str) -> str:
        """Extract Python code from Jupyter notebook."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            python_code = []
            
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        cell_code = ''.join(source)
                    else:
                        cell_code = str(source)
                    
                    if cell_code.strip():
                        python_code.append(cell_code)
            
            return '\n\n'.join(python_code)
            
        except Exception as e:
            print(f"⚠️  Error extracting Python from notebook {notebook_path}: {e}")
            return ""

    def _clean_code_for_embedding(self, code: str) -> str:
        """Clean code for tokenization with optimization."""
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove comments
            if '#' in line:
                line = line.split('#')[0]
            if '//' in line:
                line = line.split('//')[0]
            if '/*' in line and '*/' in line:
                line = re.sub(r'/\*.*?\*/', '', line, flags=re.DOTALL)
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _calculate_optimized_semantic_features(self, embeddings: List[np.ndarray]) -> Dict[str, float]:
        """Calculate semantic features from embeddings with optimization."""
        if not embeddings:
            return {}
        
        embeddings_array = np.array(embeddings)
        
        return {
            'embedding_mean': np.mean(embeddings_array),
            'embedding_std': np.std(embeddings_array),
            'embedding_max': np.max(embeddings_array),
            'embedding_min': np.min(embeddings_array),
            'embedding_range': np.max(embeddings_array) - np.min(embeddings_array),
            'embedding_skewness': self._calculate_skewness(embeddings_array),
            'embedding_kurtosis': self._calculate_kurtosis(embeddings_array),
            'semantic_diversity': np.std(embeddings_array, axis=0).mean(),
            'semantic_coherence': 1.0 / (1.0 + np.std(embeddings_array))
        }

    def _extract_optimized_significant_dimensions(self, embedding: np.ndarray) -> Dict[str, float]:
        """Extract significant dimensions for enhanced analysis with optimization."""
        significant_features = {}
        
        # Extract dimensions for each category pair
        for pair_name, dimensions in self.analyzer.significant_dimensions.items():
            for i, dim in enumerate(dimensions):
                if dim < len(embedding):
                    significant_features[f'{pair_name}_dim_{dim}'] = float(embedding[dim])
        
        # Also extract top significant dimensions across all pairs
        all_significant_dims = set()
        for dims in self.analyzer.significant_dimensions.values():
            all_significant_dims.update(dims)
        
        top_dims = sorted(list(all_significant_dims))[:self.significant_dimension_count]
        for i, dim in enumerate(top_dims):
            if dim < len(embedding):
                significant_features[f'top_dim_{dim}'] = float(embedding[dim])
        
        return significant_features

    def clear_cache(self, feature_type: Optional[str] = None):
        """Clear cache for specific feature type or all features."""
        if feature_type:
            # Clear specific feature type cache
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith(f"_{feature_type}.json")]
            for cache_file in cache_files:
                try:
                    os.remove(os.path.join(self.cache_dir, cache_file))
                    print(f"🗑️  Cleared {feature_type} cache: {cache_file}")
                except Exception as e:
                    print(f"⚠️  Failed to remove cache file {cache_file}: {e}")
        else:
            # Clear all cache
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            for cache_file in cache_files:
                try:
                    os.remove(os.path.join(self.cache_dir, cache_file))
                    print(f"🗑️  Cleared cache: {cache_file}")
                except Exception as e:
                    print(f"⚠️  Failed to remove cache file {cache_file}: {e}")
        
        print(f"✅ Cache clearing completed")

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _build_all_features_dict(self, ast_features: Dict[str, Any], codebert_features: Dict[str, Any], keyword_features: Dict[str, Any], structure_features: Dict[str, Any]) -> Dict[str, Any]:
        """Build a flattened features dict compatible with CRAv3 assess/analyze methods."""
        return {
            **(ast_features or {}),
            **(codebert_features.get('semantic_features', {}) if codebert_features else {}),
            **(codebert_features.get('significant_dimensions', {}) if codebert_features else {}),
            **(keyword_features or {}),
            **(structure_features or {})
        }

    def _extract_crav3_logic_feature_list(self, quality: Dict[str, Any], programmer: Dict[str, Any], architecture: Dict[str, Any]) -> List[float]:
        """Extract CRAv3 logical outputs as numeric feature vector (53 dims)."""
        features: List[float] = []

        # 1) Quality metrics (6 code + 4 arch + 3 doc + 4 maint + overall = 18)
        code = quality.get('code_quality', {})
        arch = quality.get('architecture_quality', {})
        doc = quality.get('documentation_quality', {})
        maint = quality.get('maintainability', {})
        features.extend([
            float(code.get('complexity', 0.0)),
            float(code.get('structure', 0.0)),
            float(code.get('organization', 0.0)),
            float(code.get('consistency', 0.0)),
            float(code.get('semantic_quality', 0.0)),
            float(code.get('code_organization', 0.0)),
        ])
        features.extend([
            float(arch.get('modularity', 0.0)),
            float(arch.get('abstraction', 0.0)),
            float(arch.get('separation', 0.0)),
            float(arch.get('scalability', 0.0)),
        ])
        features.extend([
            float(doc.get('readme_presence', 0.0)),
            float(doc.get('config_documentation', 0.0)),
            float(doc.get('dependency_documentation', 0.0)),
        ])
        features.extend([
            float(maint.get('test_coverage', 0.0)),
            float(maint.get('code_organization', 0.0)),
            float(maint.get('structure_clarity', 0.0)),
            float(maint.get('consistency', 0.0)),
        ])
        features.append(float(quality.get('overall_score', 0.0)))

        # 2) Programmer characteristics one-hot (experience 3, style 4, attention 3, architecture 3, best_practices 3, specialization 6 = 22)
        def one_hot(value: str, choices: List[str]) -> List[float]:
            v = (value or '').lower()
            return [1.0 if v == c else 0.0 for c in choices]

        features.extend(one_hot(programmer.get('experience_level', ''), ['junior', 'intermediate', 'senior']))
        features.extend(one_hot(programmer.get('coding_style', ''), ['professional', 'simple and clean', 'comprehensive', 'basic']))
        features.extend(one_hot(programmer.get('attention_to_detail', ''), ['high', 'medium', 'low']))
        features.extend(one_hot(programmer.get('architectural_thinking', ''), ['strong', 'moderate', 'basic']))
        features.extend(one_hot(programmer.get('best_practices', ''), ['excellent', 'good', 'needs improvement']))

        # specialization collapsed to key roles
        spec_value = (programmer.get('specialization', '') or '').lower()
        spec_choices = ['frontend_specialist', 'backend_specialist', 'data_scientist', 'devops_specialist', 'mobile_developer', 'game_developer']
        mapped = 'backend_specialist'
        if 'data_science' in spec_value or 'data_scientist' in spec_value:
            mapped = 'data_scientist'
        elif 'frontend' in spec_value:
            mapped = 'frontend_specialist'
        elif 'backend' in spec_value or 'generalist' in spec_value:
            mapped = 'backend_specialist'
        elif 'devops' in spec_value:
            mapped = 'devops_specialist'
        elif 'mobile' in spec_value:
            mapped = 'mobile_developer'
        elif 'game' in spec_value:
            mapped = 'game_developer'
        features.extend(one_hot(mapped, spec_choices))

        # 3) Architecture pattern confidences for 8 primary categories (8)
        cat = architecture.get('category_analysis', {})
        cat_order = ['web_application', 'data_science', 'cli_tool', 'mobile_app', 'game_development', 'library', 'educational', 'microservices']
        for k in cat_order:
            features.append(float(cat.get(k, 0.0)))

        # 4) Architectural indicators booleans (8) -> floats
        indicators = architecture.get('architectural_indicators', {})
        indicator_keys = ['modularity', 'abstraction', 'testing', 'documentation', 'configuration', 'dependencies', 'containerization', 'ci_cd']
        for k in indicator_keys:
            features.append(1.0 if indicators.get(k, False) else 0.0)

        # Ensure exact feature count
        if len(features) < self.crav3_logic_feature_count:
            features.extend([0.0] * (self.crav3_logic_feature_count - len(features)))
        elif len(features) > self.crav3_logic_feature_count:
            features = features[:self.crav3_logic_feature_count]

        return features
    
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

        # CRAv3 logic feature names (53)
        feature_names.extend([
            # quality (18)
            'q_complexity', 'q_structure', 'q_organization', 'q_consistency', 'q_semantic_quality', 'q_code_organization',
            'a_modularity', 'a_abstraction', 'a_separation', 'a_scalability',
            'd_readme', 'd_config', 'd_deps',
            'm_tests', 'm_code_org', 'm_struct_clarity', 'm_consistency',
            'overall_quality',
            # programmer one-hots (22)
            'exp_junior', 'exp_intermediate', 'exp_senior',
            'style_professional', 'style_simple', 'style_comprehensive', 'style_basic',
            'detail_high', 'detail_medium', 'detail_low',
            'arch_strong', 'arch_moderate', 'arch_basic',
            'best_excellent', 'best_good', 'best_needs_improvement',
            'spec_frontend', 'spec_backend', 'spec_data', 'spec_devops', 'spec_mobile', 'spec_game',
            # category confidences (8)
            'cat_web', 'cat_data', 'cat_cli', 'cat_mobile', 'cat_game', 'cat_library', 'cat_educational', 'cat_microservices',
            # indicators (8)
            'ind_modularity', 'ind_abstraction', 'ind_testing', 'ind_docs', 'ind_config', 'ind_deps', 'ind_container', 'ind_ci_cd'
        ])
        
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
        print("📊 CREATING TRAINING DATASET (OPTIMIZED)")
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
                # Extract comprehensive features with caching
                features = self.extract_optimized_features(repo_path)
                
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
    
    def train_model(self, X: np.ndarray, y: np.ndarray, n_trees: int = 100) -> Dict:
        """Train the optimized CRAV3 Random Forest model."""
        print("�� TRAINING OPTIMIZED CRAV3 RANDOM FOREST MODEL")
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
        
        # Train Random Forest with optimized parameters
        print(f"🌲 Training Random Forest with {n_trees} trees...")
        base_rf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=20,  # Increased depth for better performance
            min_samples_split=3,  # Slightly higher for stability
            min_samples_leaf=2,  # Slightly higher for stability
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            verbose=0,
            class_weight='balanced'  # Handle class imbalance
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
        model_path = os.path.join(self.models_dir, 'crav3_random_forest_new.joblib')
        scaler_path = os.path.join(self.models_dir, 'crav3_random_forest_new_scaler.joblib')
        feature_names_path = os.path.join(self.models_dir, 'crav3_random_forest_new_feature_names.json')
        target_names_path = os.path.join(self.models_dir, 'crav3_random_forest_new_target_names.json')
        
        joblib.dump(self.rf_classifier, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature and target names
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f)
        with open(target_names_path, 'w') as f:
            json.dump(self.target_names, f)
        
        print(f"💾 Optimized CRAV3 Random Forest saved to {model_path}")
        print(f"💾 Feature names saved to {feature_names_path}")
        print(f"�� Target names saved to {target_names_path}")
        
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
        
        print("\n�� FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        # Get feature importance from each target classifier
        for i, estimator in enumerate(self.rf_classifier.estimators_):
            target_name = self.target_names[i]
            importance = estimator.feature_importances_
            
            # Get top 10 features for each target
            feature_importance = list(zip(self.feature_names, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n�� {target_name.upper()} - Top Features:")
            for j, (feature, imp) in enumerate(feature_importance[:10], 1):
                print(f"   {j:2d}. {feature}: {imp:.4f}")
    
    def predict(self, repository_path: str) -> Dict:
        """Make predictions for a new repository using cached features."""
        if self.rf_classifier is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Extract features (will use cache if available)
        features = self.extract_optimized_features(repository_path)
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
    """Main function to train the optimized CRAV3 Random Forest model."""
    print("🚀 OPTIMIZED CRAV3 RANDOM FOREST MODEL TRAINING")
    print("=" * 80)
    
    # Check for cache corruption and offer to clear
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--clear-cache":
        print("🗑️  Clearing all feature cache...")
        model = CRAV3RandomForestNew()
        model.clear_cache()
        print("✅ Cache cleared. Please run training again.")
        return
    
    # Initialize model
    model = CRAV3RandomForestNew()
    
    try:
        # Create training dataset
        X, y = model.create_training_dataset()
        
        # Train model
        training_info = model.train_model(X, y, n_trees=150)  # More trees for better performance
        
        print(f"\n✅ OPTIMIZED CRAV3 RANDOM FOREST TRAINED SUCCESSFULLY")
        print(f"   • {training_info['n_trees']} parallel decision trees")
        print(f"   • {training_info['n_features']} comprehensive features")
        print(f"   • {training_info['n_targets']} classification targets")
        print(f"   • {training_info['n_samples']} training repositories")
        print(f"   • Accuracy: {training_info['accuracy']:.4f}")
        print(f"   • Hamming Loss: {training_info['hamming_loss']:.4f}")
        print(f"   • Cross-validation: {training_info['cv_score']:.4f} (+/- {training_info['cv_std']*2:.4f})")
        
        # Test prediction on a sample repository
        print(f"\n�� TESTING PREDICTION")
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
        
        print(f"\n🎉 Optimized CRAV3 Random Forest model is ready for use!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()