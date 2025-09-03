#!/usr/bin/env python3
"""
Comprehensive Repository Analyzer v4.0
=====================================

Enhanced analysis pipeline focusing on:
1. AST (Abstract Syntax Tree) Analysis
2. CodeBERT Semantic Embeddings with improved classification
3. Enhanced Keyword and Pattern Analysis
4. Code Quality Assessment
5. Programmer Characteristics Analysis
6. Improved Architecture Pattern Detection

Key improvements over v3:
- Better semantic classification using CodeBERT
- More specific and accurate rule-based classification
- Enhanced pattern detection logic
- Improved confidence scoring
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import ast
import re
import subprocess
import tempfile
import yaml
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

class ComprehensiveRepositoryAnalyzerV4:
    def __init__(self):
        """Initialize the analyzer with all components."""
        self.quality_regressor = None
        self.quality_scaler = StandardScaler()
        
        # AST extraction tools
        self.project_root = Path(__file__).resolve().parents[2]
        self.astminer_jar = str(self.project_root / "astminer-0.9.0" / "build" / "libs" / "astminer.jar")
        
        # CodeBERT components
        self.codebert_model = None
        self.codebert_tokenizer = None
        self._torch = None
        self.device = "cpu"
        
        # Load significant dimensions from v2 analysis for enhanced classification
        self.significant_dimensions = self.load_significant_dimensions()
        
        # Supported languages for AST extraction
        self.supported_languages = {
            ".java": "java",
            ".py": "py", 
            ".js": "js",
            ".jsx": "js",
            ".ts": "js",
            ".tsx": "js",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".kt": "kotlin",
            ".kts": "kotlin",
            ".php": "php",
            ".rb": "ruby",
            ".go": "go"
        }
        
        # Supported languages for CodeBERT
        self.codebert_languages = {
            ".py": "python",
            ".java": "java", 
            ".js": "javascript",
            ".php": "php",
            ".rb": "ruby",
            ".go": "go",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".ipynb": "python"
        }
        
        print(f"🔧 Comprehensive Repository Analyzer v4.0 initialized")
        print(f"📱 Using device: {self.device}")
    
    def load_significant_dimensions(self) -> Dict[str, List[int]]:
        """Load significant dimensions for each category pair from v2 analysis."""
        return {
            'cli_tool_vs_data_science': [448, 720, 644, 588, 540, 97, 39, 34, 461, 657],
            'web_application_vs_library': [588, 498, 720, 77, 688, 363, 270, 155, 608, 670],
            'game_development_vs_mobile_app': [588, 85, 700, 82, 629, 77, 490, 528, 551, 354],
            'data_science_vs_educational': [574, 211, 454, 422, 485, 581, 144, 301, 35, 738],
            'cli_tool_vs_educational': [211, 608, 738, 733, 686, 190, 461, 71, 485, 39],
            'data_science_vs_web_application': [23, 104, 132, 134, 164, 177, 346, 360, 430, 468, 495, 504, 506, 545, 558, 571, 618, 645, 660, 767]
        }
    
    def initialize_codebert(self):
        """Initialize CodeBERT model and tokenizer."""
        if self.codebert_model is None:
            print("🤖 Loading CodeBERT model...")
            try:
                # Lazy import heavy deps
                import torch as _torch
                from transformers import AutoTokenizer, AutoModel

                self._torch = _torch
                self.device = "cuda" if _torch.cuda.is_available() else "cpu"
                device_obj = _torch.device(self.device)

                self.codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
                self.codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
                self.codebert_model.to(device_obj)
                self.codebert_model.eval()
                
                if self.device == "cuda":
                    print(f"🚀 CUDA detected: {_torch.cuda.get_device_name(0)}")
                    print(f"   Memory: {_torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                else:
                    print("💻 Using CPU for computations")
                
                print("✅ CodeBERT model loaded successfully!")
            except Exception as e:
                print(f"❌ Failed to load CodeBERT: {e}")
                print("⚠️  CodeBERT analysis will be skipped")
    
    def extract_ast_features(self, repository_path: str) -> Dict[str, Any]:
        """Extract AST-based structural features using AstMiner."""
        print(f"🌳 Extracting AST features from {repository_path}")
        
        ast_features = {
            'avg_path_length': 0.0,
            'max_path_length': 0.0,
            'path_variety': 0.0,
            'node_type_diversity': 0.0,
            'complexity_score': 0.0,
            'nesting_depth': 0.0,
            'function_count': 0.0,
            'class_count': 0.0,
            'interface_count': 0.0,
            'inheritance_depth': 0.0,
            'total_ast_nodes': 0,
            'unique_node_types': 0,
            'ast_depth': 0.0,
            'branching_factor': 0.0
        }
        
        # Check if AstMiner is available
        if not os.path.exists(self.astminer_jar):
            print(f"⚠️  AstMiner not found at {self.astminer_jar}")
            print("📝 Using basic AST analysis instead...")
            return self._extract_basic_ast_features(repository_path)
        
        # Detect languages in the repository
        detected_languages = self._detect_languages(repository_path)
        
        if not detected_languages:
            print("⚠️  No supported languages detected")
            return ast_features
        
        # Extract AST for each language
        all_ast_data = []
        for lang in detected_languages:
            lang_ast_data = self._extract_ast_for_language(repository_path, lang)
            if lang_ast_data:
                all_ast_data.extend(lang_ast_data)
        
        if all_ast_data:
            ast_features = self._calculate_ast_metrics(all_ast_data)
        
        return ast_features

    def analyze_architecture_patterns(self, features: Dict) -> Dict[str, Any]:
        """Analyze architecture patterns using enhanced semantic analysis with CodeBERT."""
        patterns = {
            'detected_patterns': [],
            'pattern_confidence': {},
            'architectural_indicators': {},
            'category_analysis': {},
            'semantic_analysis': {}
        }
        
        # Enhanced semantic analysis using CodeBERT embeddings
        semantic_analysis = self._analyze_semantic_patterns(features)
        patterns['semantic_analysis'] = semantic_analysis
        
        # Enhanced pattern detection using semantic analysis + structural features
        category_scores = self._analyze_category_scores_enhanced(features, semantic_analysis)
        patterns['category_analysis'] = category_scores
        
        # More specific pattern detection logic
        indicators = self._get_enhanced_pattern_indicators(features, semantic_analysis, category_scores)
        
        # Detect patterns with improved confidence scoring
        for pattern, indicator in indicators.items():
            if indicator['detected']:
                patterns['detected_patterns'].append(pattern)
                patterns['pattern_confidence'][pattern] = indicator['confidence']
        
        # Enhanced architectural indicators
        patterns['architectural_indicators'] = self._get_enhanced_architectural_indicators(features, semantic_analysis)
        
        return patterns
    
    def _analyze_semantic_patterns(self, features: Dict) -> Dict[str, Any]:
        """Analyze semantic patterns using CodeBERT embeddings."""
        semantic_analysis = {
            'code_patterns': {},
            'api_patterns': {},
            'framework_patterns': {},
            'domain_patterns': {},
            'semantic_coherence': features.get('semantic_coherence', 0),
            'embedding_analysis': {}
        }
        
        # Analyze significant dimensions for semantic understanding
        significant_dims = {k: v for k, v in features.items() 
                          if k.startswith(('cli_tool_vs_data_science_dim_', 
                                         'web_application_vs_library_dim_', 
                                         'game_development_vs_mobile_app_dim_', 
                                         'data_science_vs_web_application_dim_'))}
        
        if significant_dims:
            # Extract semantic patterns from embeddings
            semantic_analysis['embedding_analysis'] = self._extract_semantic_patterns_from_embeddings(significant_dims)
            
            # Analyze code patterns based on semantic features
            semantic_analysis['code_patterns'] = self._analyze_code_patterns_from_semantics(features, significant_dims)
            
            # Analyze API and framework patterns
            semantic_analysis['api_patterns'] = self._analyze_api_patterns(features, significant_dims)
            
            # Analyze domain-specific patterns
            semantic_analysis['domain_patterns'] = self._analyze_domain_patterns(features, significant_dims)
        
        return semantic_analysis
    
    def _extract_semantic_patterns_from_embeddings(self, significant_dims: Dict) -> Dict[str, Any]:
        """Extract semantic patterns from CodeBERT embeddings."""
        patterns = {
            'cli_indicators': [],
            'web_indicators': [],
            'data_science_indicators': [],
            'library_indicators': [],
            'game_indicators': [],
            'mobile_indicators': []
        }
        
        # Analyze CLI vs Data Science dimensions
        cli_ds_dims = {k: v for k, v in significant_dims.items() 
                       if k.startswith('cli_tool_vs_data_science_dim_')}
        if cli_ds_dims:
            cli_favored = [448, 720]  # Dimensions that favor CLI tools
            ds_favored = [644, 588, 540]  # Dimensions that favor data science
            
            cli_score = sum(v for k, v in cli_ds_dims.items() 
                           if any(f'dim_{d}' in k for d in cli_favored))
            ds_score = sum(v for k, v in cli_ds_dims.items() 
                          if any(f'dim_{d}' in k for d in ds_favored))
            
            if cli_score > ds_score:
                patterns['cli_indicators'].append(f'CLI-favored semantic pattern (score: {cli_score:.3f})')
            else:
                patterns['data_science_indicators'].append(f'Data science-favored semantic pattern (score: {ds_score:.3f})')
        
        # Analyze Web Application vs Library dimensions
        web_lib_dims = {k: v for k, v in significant_dims.items() 
                        if k.startswith('web_application_vs_library_dim_')}
        if web_lib_dims:
            web_favored = [588, 498]  # Dimensions that favor web applications
            lib_favored = [720, 77, 688]  # Dimensions that favor libraries
            
            web_score = sum(v for k, v in web_lib_dims.items() 
                           if any(f'dim_{d}' in k for d in web_favored))
            lib_score = sum(v for k, v in web_lib_dims.items() 
                           if any(f'dim_{d}' in k for d in lib_favored))
            
            if web_score > lib_score:
                patterns['web_indicators'].append(f'Web application-favored semantic pattern (score: {web_score:.3f})')
            else:
                patterns['library_indicators'].append(f'Library-favored semantic pattern (score: {lib_score:.3f})')
        
        return patterns
    
    def _analyze_code_patterns_from_semantics(self, features: Dict, significant_dims: Dict) -> Dict[str, Any]:
        """Analyze code patterns based on semantic analysis."""
        patterns = {
            'function_patterns': [],
            'class_patterns': [],
            'import_patterns': [],
            'api_patterns': []
        }
        
        # Analyze function patterns
        function_count = features.get('function_count', 0)
        if function_count > 50:
            patterns['function_patterns'].append('High function density - suggests library or framework')
        elif function_count > 20:
            patterns['function_patterns'].append('Moderate function density - suggests application')
        else:
            patterns['function_patterns'].append('Low function density - suggests simple tool or script')
        
        # Analyze class patterns
        class_count = features.get('class_count', 0)
        if class_count > 10:
            patterns['class_patterns'].append('High class density - suggests object-oriented library')
        elif class_count > 5:
            patterns['class_patterns'].append('Moderate class density - suggests structured application')
        else:
            patterns['class_patterns'].append('Low class density - suggests procedural or functional approach')
        
        # Analyze import patterns
        language_counts = features.get('language_counts', {})
        if language_counts.get('py', 0) > 0:
            patterns['import_patterns'].append('Python code detected')
        if language_counts.get('js', 0) > 0:
            patterns['import_patterns'].append('JavaScript/TypeScript code detected')
        if language_counts.get('java', 0) > 0:
            patterns['import_patterns'].append('Java code detected')
        
        return patterns
    
    def _analyze_api_patterns(self, features: Dict, significant_dims: Dict) -> Dict[str, Any]:
        """Analyze API and framework patterns."""
        api_patterns = {
            'web_framework': False,
            'cli_framework': False,
            'data_science_framework': False,
            'mobile_framework': False,
            'game_framework': False
        }
        
        # Check for web framework patterns
        if features.get('web_keywords', 0) > 3:
            api_patterns['web_framework'] = True
        
        # Check for CLI framework patterns
        if features.get('cli_keywords', 0) > 2:
            api_patterns['cli_framework'] = True
        
        # Check for data science framework patterns
        if features.get('data_science_keywords', 0) > 3:
            api_patterns['data_science_framework'] = True
        
        # Check for mobile framework patterns
        if features.get('mobile_keywords', 0) > 2:
            api_patterns['mobile_framework'] = True
        
        # Check for game framework patterns
        if features.get('game_keywords', 0) > 2:
            api_patterns['game_framework'] = True
        
        return api_patterns
    
    def _analyze_domain_patterns(self, features: Dict, significant_dims: Dict) -> Dict[str, Any]:
        """Analyze domain-specific patterns."""
        domain_patterns = {
            'enterprise': False,
            'academic': False,
            'startup': False,
            'open_source': False,
            'commercial': False
        }
        
        # Check for enterprise patterns
        if features.get('has_ci_cd', False) and features.get('has_docker', False):
            domain_patterns['enterprise'] = True
        
        # Check for academic patterns
        if features.get('has_docs', False) and features.get('total_files', 0) < 100:
            domain_patterns['academic'] = True
        
        # Check for open source patterns
        if features.get('has_tests', False) and features.get('has_docs', False):
            domain_patterns['open_source'] = True
        
        return domain_patterns
    
    def _get_enhanced_pattern_indicators(self, features: Dict, semantic_analysis: Dict, category_scores: Dict) -> Dict[str, Dict]:
        """Get enhanced pattern indicators with improved logic."""
        indicators = {}
        
        # CLI Tool detection - more specific
        cli_indicators = semantic_analysis.get('cli_indicators', [])
        cli_keywords = features.get('cli_keywords', 0)
        has_console_scripts = features.get('console_scripts', False)
        has_bin_folder = features.get('bin_folder', False)
        
        cli_detected = (
            len(cli_indicators) > 0 or
            (cli_keywords > 3 and has_console_scripts) or
            (cli_keywords > 2 and has_bin_folder) or
            (cli_keywords > 4 and features.get('total_files', 0) < 100)
        )
        
        indicators['cli_tool'] = {
            'detected': cli_detected,
            'confidence': self._calculate_cli_confidence(features, semantic_analysis, category_scores)
        }
        
        # Web Application detection - more specific
        web_indicators = semantic_analysis.get('web_indicators', [])
        web_keywords = features.get('web_keywords', 0)
        has_frontend_files = features.get('frontend_files', False)
        has_web_framework = semantic_analysis.get('api_patterns', {}).get('web_framework', False)
        
        web_detected = (
            len(web_indicators) > 0 or
            (web_keywords > 5 and has_frontend_files) or
            (web_keywords > 3 and has_web_framework) or
            (web_keywords > 4 and features.get('has_ci_cd', False))
        )
        
        indicators['web_application'] = {
            'detected': web_detected,
            'confidence': self._calculate_web_confidence(features, semantic_analysis, category_scores)
        }
        
        # Library detection - more specific
        library_indicators = semantic_analysis.get('library_indicators', [])
        library_keywords = features.get('library_keywords', 0)
        has_tests = features.get('has_tests', False)
        has_docs = features.get('has_docs', False)
        has_dependencies = features.get('has_dependencies', False)
        
        library_detected = (
            len(library_indicators) > 0 or
            (library_keywords > 3 and has_tests and has_docs) or
            (library_keywords > 2 and has_dependencies and has_docs) or
            (features.get('function_count', 0) > 30 and has_tests)
        )
        
        indicators['library'] = {
            'detected': library_detected,
            'confidence': self._calculate_library_confidence(features, semantic_analysis, category_scores)
        }
        
        # Data Science detection - more specific
        data_science_keywords = features.get('data_science_keywords', 0)
        has_jupyter = features.get('jupyter_notebooks', False)
        has_data_libs = features.get('pandas', False) or features.get('matplotlib', False)
        
        data_science_detected = (
            data_science_keywords > 4 or
            (data_science_keywords > 2 and has_jupyter) or
            (data_science_keywords > 3 and has_data_libs)
        )
        
        indicators['data_science'] = {
            'detected': data_science_detected,
            'confidence': self._calculate_data_science_confidence(features, semantic_analysis, category_scores)
        }
        
        # Game Development detection - more specific
        game_keywords = features.get('game_keywords', 0)
        has_game_assets = features.get('game_assets', False)
        has_unity = features.get('unity_files', False)
        has_pygame = features.get('python_games', False)
        
        game_detected = (
            game_keywords > 3 or
            has_game_assets or
            has_unity or
            has_pygame
        )
        
        indicators['game_development'] = {
            'detected': game_detected,
            'confidence': self._calculate_game_confidence(features, semantic_analysis, category_scores)
        }
        
        # Mobile App detection - more specific
        mobile_keywords = features.get('mobile_keywords', 0)
        has_android = features.get('android_manifest', False)
        has_ios = features.get('ios_project', False)
        has_react_native = features.get('react_native', False)
        
        mobile_detected = (
            mobile_keywords > 2 or
            has_android or
            has_ios or
            has_react_native
        )
        
        indicators['mobile_app'] = {
            'detected': mobile_detected,
            'confidence': self._calculate_mobile_confidence(features, semantic_analysis, category_scores)
        }
        
        return indicators
    
    def _calculate_cli_confidence(self, features: Dict, semantic_analysis: Dict, category_scores: Dict) -> float:
        """Calculate confidence for CLI tool classification."""
        base_confidence = 0.3
        
        # Semantic indicators
        if semantic_analysis.get('cli_indicators'):
            base_confidence += 0.3
        
        # Keyword strength
        cli_keywords = features.get('cli_keywords', 0)
        if cli_keywords > 5:
            base_confidence += 0.2
        elif cli_keywords > 3:
            base_confidence += 0.15
        elif cli_keywords > 1:
            base_confidence += 0.1
        
        # Structural indicators
        if features.get('console_scripts', False):
            base_confidence += 0.15
        if features.get('bin_folder', False):
            base_confidence += 0.1
        if features.get('total_files', 0) < 100:
            base_confidence += 0.1
        
        # Category scores
        if 'cli_tool' in category_scores:
            base_confidence += min(0.2, category_scores['cli_tool'])
        
        return min(1.0, base_confidence)
    
    def _calculate_web_confidence(self, features: Dict, semantic_analysis: Dict, category_scores: Dict) -> float:
        """Calculate confidence for web application classification."""
        base_confidence = 0.3
        
        # Semantic indicators
        if semantic_analysis.get('web_indicators'):
            base_confidence += 0.3
        
        # Keyword strength
        web_keywords = features.get('web_keywords', 0)
        if web_keywords > 7:
            base_confidence += 0.2
        elif web_keywords > 5:
            base_confidence += 0.15
        elif web_keywords > 3:
            base_confidence += 0.1
        
        # Structural indicators
        if features.get('frontend_files', False):
            base_confidence += 0.15
        if features.get('has_ci_cd', False):
            base_confidence += 0.1
        if semantic_analysis.get('api_patterns', {}).get('web_framework', False):
            base_confidence += 0.15
        
        # Category scores
        if 'web_application' in category_scores:
            base_confidence += min(0.2, category_scores['web_application'])
        
        return min(1.0, base_confidence)
    
    def _calculate_library_confidence(self, features: Dict, semantic_analysis: Dict, category_scores: Dict) -> float:
        """Calculate confidence for library classification."""
        base_confidence = 0.3
        
        # Semantic indicators
        if semantic_analysis.get('library_indicators'):
            base_confidence += 0.3
        
        # Keyword strength
        library_keywords = features.get('library_keywords', 0)
        if library_keywords > 5:
            base_confidence += 0.2
        elif library_keywords > 3:
            base_confidence += 0.15
        elif library_keywords > 1:
            base_confidence += 0.1
        
        # Structural indicators
        if features.get('has_tests', False):
            base_confidence += 0.15
        if features.get('has_docs', False):
            base_confidence += 0.1
        if features.get('has_dependencies', False):
            base_confidence += 0.1
        if features.get('function_count', 0) > 30:
            base_confidence += 0.1
        
        # Category scores
        if 'library' in category_scores:
            base_confidence += min(0.2, category_scores['library'])
        
        return min(1.0, base_confidence)
    
    def _calculate_data_science_confidence(self, features: Dict, semantic_analysis: Dict, category_scores: Dict) -> float:
        """Calculate confidence for data science classification."""
        base_confidence = 0.3
        
        # Keyword strength
        data_science_keywords = features.get('data_science_keywords', 0)
        if data_science_keywords > 6:
            base_confidence += 0.3
        elif data_science_keywords > 4:
            base_confidence += 0.2
        elif data_science_keywords > 2:
            base_confidence += 0.15
        
        # Structural indicators
        if features.get('jupyter_notebooks', False):
            base_confidence += 0.2
        if features.get('pandas', False) or features.get('matplotlib', False):
            base_confidence += 0.15
        if features.get('language_counts', {}).get('py', 0) > 0:
            base_confidence += 0.1
        
        # Category scores
        if 'data_science' in category_scores:
            base_confidence += min(0.2, category_scores['data_science'])
        
        return min(1.0, base_confidence)
    
    def _calculate_game_confidence(self, features: Dict, semantic_analysis: Dict, category_scores: Dict) -> float:
        """Calculate confidence for game development classification."""
        base_confidence = 0.3
        
        # Keyword strength
        game_keywords = features.get('game_keywords', 0)
        if game_keywords > 4:
            base_confidence += 0.3
        elif game_keywords > 2:
            base_confidence += 0.2
        elif game_keywords > 1:
            base_confidence += 0.15
        
        # Structural indicators
        if features.get('game_assets', False):
            base_confidence += 0.2
        if features.get('unity_files', False):
            base_confidence += 0.2
        if features.get('python_games', False):
            base_confidence += 0.15
        
        # Category scores
        if 'game_development' in category_scores:
            base_confidence += min(0.2, category_scores['game_development'])
        
        return min(1.0, base_confidence)
    
    def _calculate_mobile_confidence(self, features: Dict, semantic_analysis: Dict, category_scores: Dict) -> float:
        """Calculate confidence for mobile app classification."""
        base_confidence = 0.3
        
        # Keyword strength
        mobile_keywords = features.get('mobile_keywords', 0)
        if mobile_keywords > 3:
            base_confidence += 0.3
        elif mobile_keywords > 2:
            base_confidence += 0.2
        elif mobile_keywords > 1:
            base_confidence += 0.15
        
        # Structural indicators
        if features.get('android_manifest', False):
            base_confidence += 0.25
        if features.get('ios_project', False):
            base_confidence += 0.25
        if features.get('react_native', False):
            base_confidence += 0.2
        
        # Category scores
        if 'mobile_app' in category_scores:
            base_confidence += min(0.2, category_scores['mobile_app'])
        
        return min(1.0, base_confidence)

    def _analyze_category_scores_enhanced(self, features: Dict, semantic_analysis: Dict) -> Dict[str, float]:
        """Enhanced category analysis using semantic analysis and CodeBERT embeddings."""
        category_scores = {}
        
        # Analyze significant dimensions for category classification
        significant_dims = {k: v for k, v in features.items() 
                          if k.startswith(('cli_tool_vs_data_science_dim_', 
                                         'web_application_vs_library_dim_', 
                                         'game_development_vs_mobile_app_dim_', 
                                         'data_science_vs_web_application_dim_'))}
        
        if significant_dims:
            # CLI vs Data Science analysis
            cli_ds_dims = {k: v for k, v in significant_dims.items() 
                          if k.startswith('cli_tool_vs_data_science_dim_')}
            if cli_ds_dims:
                cli_favored = [448, 720, 97, 39, 34]  # CLI-favored dimensions
                ds_favored = [644, 588, 540, 461, 657]  # Data science-favored dimensions
                
                cli_score = sum(v for k, v in cli_ds_dims.items() 
                               if any(f'dim_{d}' in k for d in cli_favored))
                ds_score = sum(v for k, v in cli_ds_dims.items() 
                              if any(f'dim_{d}' in k for d in ds_favored))
                
                # Normalize scores
                total_cli_dims = len([k for k in cli_ds_dims.keys() 
                                    if any(f'dim_{d}' in k for d in cli_favored)])
                total_ds_dims = len([k for k in cli_ds_dims.keys() 
                                   if any(f'dim_{d}' in k for d in ds_favored)])
                
                if total_cli_dims > 0:
                    category_scores['cli_tool'] = max(0, cli_score / total_cli_dims)
                if total_ds_dims > 0:
                    category_scores['data_science'] = max(0, ds_score / total_ds_dims)
            
            # Web Application vs Library analysis
            web_lib_dims = {k: v for k, v in significant_dims.items() 
                           if k.startswith('web_application_vs_library_dim_')}
            if web_lib_dims:
                web_favored = [588, 498, 363, 270, 155]  # Web-favored dimensions
                lib_favored = [720, 77, 688, 608, 670]  # Library-favored dimensions
                
                web_score = sum(v for k, v in web_lib_dims.items() 
                               if any(f'dim_{d}' in k for d in web_favored))
                lib_score = sum(v for k, v in web_lib_dims.items() 
                               if any(f'dim_{d}' in k for d in lib_favored))
                
                # Normalize scores
                total_web_dims = len([k for k in web_lib_dims.keys() 
                                    if any(f'dim_{d}' in k for d in web_favored)])
                total_lib_dims = len([k for k in web_lib_dims.keys() 
                                    if any(f'dim_{d}' in k for d in lib_favored)])
                
                if total_web_dims > 0:
                    category_scores['web_application'] = max(0, web_score / total_web_dims)
                if total_lib_dims > 0:
                    category_scores['library'] = max(0, lib_score / total_lib_dims)
            
            # Game Development vs Mobile App analysis
            game_mobile_dims = {k: v for k, v in significant_dims.items() 
                               if k.startswith('game_development_vs_mobile_app_dim_')}
            if game_mobile_dims:
                game_favored = [588, 85, 700, 82, 629]  # Game-favored dimensions
                mobile_favored = [77, 490, 528, 551, 354]  # Mobile-favored dimensions
                
                game_score = sum(v for k, v in game_mobile_dims.items() 
                                if any(f'dim_{d}' in k for d in game_favored))
                mobile_score = sum(v for k, v in game_mobile_dims.items() 
                                  if any(f'dim_{d}' in k for d in mobile_favored))
                
                # Normalize scores
                total_game_dims = len([k for k in game_mobile_dims.keys() 
                                     if any(f'dim_{d}' in k for d in game_favored)])
                total_mobile_dims = len([k for k in game_mobile_dims.keys() 
                                       if any(f'dim_{d}' in k for d in mobile_favored)])
                
                if total_game_dims > 0:
                    category_scores['game_development'] = max(0, game_score / total_game_dims)
                if total_mobile_dims > 0:
                    category_scores['mobile_app'] = max(0, mobile_score / total_mobile_dims)
        
        # Fallback to keyword-based analysis if no semantic data
        if not category_scores:
            category_scores = {
                'cli_tool': min(1.0, features.get('cli_keywords', 0) / 10.0),
                'data_science': min(1.0, features.get('data_science_keywords', 0) / 10.0),
                'web_application': min(1.0, features.get('web_keywords', 0) / 10.0),
                'library': min(1.0, features.get('library_keywords', 0) / 10.0),
                'game_development': min(1.0, features.get('game_keywords', 0) / 10.0),
                'mobile_app': min(1.0, features.get('mobile_keywords', 0) / 10.0),
                'educational': 0.0
            }
        
        return category_scores
    
    def _get_enhanced_architectural_indicators(self, features: Dict, semantic_analysis: Dict) -> Dict[str, Any]:
        """Get enhanced architectural indicators."""
        indicators = {
            'modularity': features.get('function_count', 0) > 20,
            'abstraction': features.get('class_count', 0) > 5,
            'testing': features.get('has_tests', False),
            'documentation': features.get('has_docs', False),
            'configuration': features.get('has_config', False),
            'dependencies': features.get('has_dependencies', False),
            'containerization': features.get('has_docker', False),
            'ci_cd': features.get('has_ci_cd', False),
            'semantic_coherence': features.get('semantic_coherence', 0) > 0.5,
            'code_complexity': features.get('complexity_score', 0) > 10,
            'semantic_patterns': len(semantic_analysis.get('embedding_analysis', {})) > 0,
            'api_frameworks': any(semantic_analysis.get('api_patterns', {}).values()),
            'domain_specific': any(semantic_analysis.get('domain_patterns', {}).values())
        }
        
        return indicators
    
    def _detect_languages(self, repository_path: str) -> List[str]:
        """Detect programming languages in the repository."""
        languages = set()
        
        # Check if the path is a directory
        if not os.path.isdir(repository_path):
            print(f"⚠️  Path is not a directory: {repository_path}")
            return list(languages)
        
        for root, _, files in os.walk(repository_path):
            for file in files:
                # Only process actual files, not directories with extensions
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in self.supported_languages:
                        languages.add(self.supported_languages[ext])
        
        return list(languages)
    
    def _extract_ast_for_language(self, repository_path: str, language: str) -> List[Dict]:
        """Extract AST data for a specific language using AstMiner."""
        # Check if language is supported by AstMiner
        astminer_supported = {
            "java": True,
            "py": True,
            "js": True,
            "php": True,
            "ruby": True,
            "go": True,
            "kotlin": True
        }
        
        # For unsupported languages, use fallback analysis
        if language not in astminer_supported:
            print(f"⚠️  {language} not supported by AstMiner, using fallback analysis...")
            return self._extract_fallback_ast(repository_path, language)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create AstMiner configuration
                config = {
                    "inputDir": repository_path,
                    "outputDir": temp_dir,
                    "parser": {
                        "name": "antlr",
                        "languages": [language]
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
                config_path = os.path.join(temp_dir, "config.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
                
                # Run AstMiner with timeout
                try:
                    result = subprocess.run(
                        ["java", "-jar", self.astminer_jar, config_path],
                        capture_output=True,
                        text=True,
                        timeout=60  # 60 second timeout
                    )
                    
                    if result.returncode != 0:
                        print(f"❌ AstMiner failed for {language}: {result.stderr}")
                        print(f"   🔄 Falling back to regex-based analysis...")
                        return self._extract_fallback_ast(repository_path, language)
                    
                    # Parse AST output
                    ast_data = self._parse_ast_output(temp_dir)
                    return ast_data
                    
                except subprocess.TimeoutExpired:
                    print(f"⚠️  AstMiner timeout for {language}, using fallback...")
                    return self._extract_fallback_ast(repository_path, language)
                except Exception as e:
                    print(f"⚠️  AstMiner subprocess error for {language}: {e}")
                    print(f"   🔄 Falling back to regex-based analysis...")
                    return self._extract_fallback_ast(repository_path, language)
                
        except Exception as e:
            print(f"❌ Error extracting AST for {language}: {e}")
            return self._extract_fallback_ast(repository_path, language)
    
    def _extract_fallback_ast(self, repository_path: str, language: str) -> List[Dict]:
        """Extract AST features using regex-based analysis for unsupported languages."""
        print(f"   🔍 Using regex-based analysis for {language}...")
        
        ast_data = []
        
        # Language-specific patterns
        if language in ["cpp", "c"]:
            ast_data = self._extract_cpp_ast_features(repository_path)
        elif language in ["cs"]:
            ast_data = self._extract_csharp_ast_features(repository_path)
        elif language in ["rs"]:
            ast_data = self._extract_rust_ast_features(repository_path)
        elif language in ["swift"]:
            ast_data = self._extract_swift_ast_features(repository_path)
        else:
            # Generic fallback for other languages
            ast_data = self._extract_generic_ast_features(repository_path, language)
        
        return ast_data
    
    def _extract_cpp_ast_features(self, repository_path: str) -> List[Dict]:
        """Extract C++ AST features using regex patterns."""
        features = {
            "method_count": 0,
            "path_contexts": [],
            "node_types": set(),
            "tokens": set(),
            "avg_path_length": 0,
            "path_diversity": 0
        }
        
        # C++ specific patterns
        cpp_patterns = {
            "class": r'class\s+\w+',
            "struct": r'struct\s+\w+',
            "function": r'(?:virtual\s+)?(?:inline\s+)?(?:static\s+)?(?:const\s+)?(?:template\s*<[^>]*>\s*)?(?:[\w:<>,\s]+\s+)?\w+\s*\([^)]*\)\s*(?:const\s*)?(?:=\s*0\s*)?(?:override\s*)?(?:final\s*)?\s*\{?',
            "namespace": r'namespace\s+\w+',
            "include": r'#include\s*[<"][^>"]*[>"]',
            "macro": r'#define\s+\w+',
            "typedef": r'typedef\s+[\w\s<>*&]+',
            "enum": r'enum\s+(?:class\s+)?\w+',
            "template": r'template\s*<[^>]*>',
            "constructor": r'\w+\s*\([^)]*\)\s*:\s*[^}]*\{',
            "destructor": r'~\w+\s*\([^)]*\)\s*\{'
        }
        
        total_methods = 0
        all_tokens = set()
        all_node_types = set()
        
        for root, _, files in os.walk(repository_path):
            for file in files:
                if file.endswith(('.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Count patterns
                        for pattern_name, pattern in cpp_patterns.items():
                            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                            if matches:
                                all_node_types.add(pattern_name)
                                all_tokens.update(matches)
                                
                                if pattern_name == "function":
                                    total_methods += len(matches)
                        
                    except Exception as e:
                        print(f"   ⚠️  Error reading {file}: {e}")
                        continue
        
        # Calculate metrics
        features["method_count"] = total_methods
        features["node_types"] = list(all_node_types)
        features["tokens"] = list(all_tokens)
        features["avg_path_length"] = min(10, len(all_node_types))  # Estimate
        features["path_diversity"] = min(1.0, len(all_node_types) / 20.0)  # Normalize
        
        return [features]
    
    def _extract_csharp_ast_features(self, repository_path: str) -> List[Dict]:
        """Extract C# AST features using regex patterns."""
        features = {
            "method_count": 0,
            "path_contexts": [],
            "node_types": set(),
            "tokens": set(),
            "avg_path_length": 0,
            "path_diversity": 0
        }
        
        # C# specific patterns
        csharp_patterns = {
            "class": r'(?:public\s+)?(?:abstract\s+)?(?:sealed\s+)?(?:partial\s+)?class\s+\w+',
            "interface": r'(?:public\s+)?interface\s+\w+',
            "method": r'(?:public\s+|private\s+|protected\s+|internal\s+)?(?:virtual\s+|abstract\s+|override\s+|sealed\s+)?(?:static\s+)?(?:async\s+)?[\w<>,\s]+\s+\w+\s*\([^)]*\)',
            "property": r'(?:public\s+|private\s+|protected\s+|internal\s+)?(?:virtual\s+|abstract\s+|override\s+)?[\w<>,\s]+\s+\w+\s*\{\s*(?:get;\s*)?(?:set;\s*)?\}',
            "namespace": r'namespace\s+[\w.]+',
            "using": r'using\s+[\w.]+;',
            "attribute": r'\[[^\]]+\]',
            "constructor": r'(?:public\s+|private\s+|protected\s+|internal\s+)?\w+\s*\([^)]*\)\s*:\s*[^}]*\{',
            "enum": r'(?:public\s+)?enum\s+\w+',
            "struct": r'(?:public\s+)?struct\s+\w+'
        }
        
        total_methods = 0
        all_tokens = set()
        all_node_types = set()
        
        for root, _, files in os.walk(repository_path):
            for file in files:
                if file.endswith('.cs'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Count patterns
                        for pattern_name, pattern in csharp_patterns.items():
                            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                            if matches:
                                all_node_types.add(pattern_name)
                                all_tokens.update(matches)
                                
                                if pattern_name in ["method", "property"]:
                                    total_methods += len(matches)
                        
                    except Exception as e:
                        print(f"   ⚠️  Error reading {file}: {e}")
                        continue
        
        # Calculate metrics
        features["method_count"] = total_methods
        features["node_types"] = list(all_node_types)
        features["tokens"] = list(all_tokens)
        features["avg_path_length"] = min(10, len(all_node_types))
        features["path_diversity"] = min(1.0, len(all_node_types) / 20.0)
        
        return [features]
    
    def _extract_rust_ast_features(self, repository_path: str) -> List[Dict]:
        """Extract Rust AST features using regex patterns."""
        features = {
            "method_count": 0,
            "path_contexts": [],
            "node_types": set(),
            "tokens": set(),
            "avg_path_length": 0,
            "path_diversity": 0
        }
        
        # Rust specific patterns
        rust_patterns = {
            "struct": r'struct\s+\w+',
            "enum": r'enum\s+\w+',
            "fn": r'fn\s+\w+\s*\([^)]*\)',
            "impl": r'impl\s+(?:[\w:<>]+)\s*\{',
            "trait": r'trait\s+\w+',
            "mod": r'mod\s+\w+',
            "use": r'use\s+[\w:<>]+;',
            "macro": r'macro_rules!\s+\w+',
            "const": r'const\s+\w+:',
            "static": r'static\s+\w+:'
        }
        
        total_methods = 0
        all_tokens = set()
        all_node_types = set()
        
        for root, _, files in os.walk(repository_path):
            for file in files:
                if file.endswith('.rs'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Count patterns
                        for pattern_name, pattern in rust_patterns.items():
                            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                            if matches:
                                all_node_types.add(pattern_name)
                                all_tokens.update(matches)
                                
                                if pattern_name == "fn":
                                    total_methods += len(matches)
                        
                    except Exception as e:
                        print(f"   ⚠️  Error reading {file}: {e}")
                        continue
        
        # Calculate metrics
        features["method_count"] = total_methods
        features["node_types"] = list(all_node_types)
        features["tokens"] = list(all_tokens)
        features["avg_path_length"] = min(10, len(all_node_types))
        features["path_diversity"] = min(1.0, len(all_node_types) / 20.0)
        
        return [features]
    
    def _extract_swift_ast_features(self, repository_path: str) -> List[Dict]:
        """Extract Swift AST features using regex patterns."""
        features = {
            "method_count": 0,
            "path_contexts": [],
            "tokens": set(),
            "node_types": set(),
            "avg_path_length": 0,
            "path_diversity": 0
        }
        
        # Swift specific patterns
        swift_patterns = {
            "class": r'class\s+\w+',
            "struct": r'struct\s+\w+',
            "enum": r'enum\s+\w+',
            "func": r'func\s+\w+\s*\([^)]*\)',
            "protocol": r'protocol\s+\w+',
            "extension": r'extension\s+\w+',
            "import": r'import\s+\w+',
            "var": r'var\s+\w+:',
            "let": r'let\s+\w+:',
            "init": r'init\s*\([^)]*\)'
        }
        
        total_methods = 0
        all_tokens = set()
        all_node_types = set()
        
        for root, _, files in os.walk(repository_path):
            for file in files:
                if file.endswith('.swift'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Count patterns
                        for pattern_name, pattern in swift_patterns.items():
                            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                            if matches:
                                all_node_types.add(pattern_name)
                                all_tokens.update(matches)
                                
                                if pattern_name in ["func", "init"]:
                                    total_methods += len(matches)
                        
                    except Exception as e:
                        print(f"   ⚠️  Error reading {file}: {e}")
                        continue
        
        # Calculate metrics
        features["method_count"] = total_methods
        features["node_types"] = list(all_node_types)
        features["tokens"] = list(all_tokens)
        features["avg_path_length"] = min(10, len(all_node_types))
        features["path_diversity"] = min(1.0, len(all_node_types) / 20.0)
        
        return [features]
    
    def _extract_generic_ast_features(self, repository_path: str, language: str) -> List[Dict]:
        """Extract generic AST features for any language."""
        features = {
            "method_count": 0,
            "path_contexts": [],
            "node_types": set(),
            "tokens": set(),
            "avg_path_length": 0,
            "path_diversity": 0
        }
        
        # Generic patterns that work for most languages
        generic_patterns = {
            "function": r'function\s+\w+\s*\(|def\s+\w+\s*\(|func\s+\w+\s*\(',
            "class": r'class\s+\w+|struct\s+\w+|interface\s+\w+',
            "import": r'import\s+[\w.]+|from\s+[\w.]+\s+import|require\s*\([\'"]?[\w.]+[\'"]?\)',
            "comment": r'//.*$|/\*.*?\*/|#.*$|<!--.*?-->',
            "string": r'["\'][^"\']*["\']',
            "number": r'\b\d+\.?\d*\b'
        }
        
        total_methods = 0
        all_tokens = set()
        all_node_types = set()
        
        for root, _, files in os.walk(repository_path):
            for file in files:
                # Try to detect language by extension
                ext = os.path.splitext(file)[1].lower()
                if ext in ['.txt', '.md', '.json', '.yaml', '.yml']:
                    continue
                
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Count patterns
                    for pattern_name, pattern in generic_patterns.items():
                        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                        if matches:
                            all_node_types.add(pattern_name)
                            all_tokens.update(matches)
                            
                            if pattern_name == "function":
                                total_methods += len(matches)
                    
                except Exception as e:
                    continue
        
        # Calculate metrics
        features["method_count"] = total_methods
        features["node_types"] = list(all_node_types)
        features["tokens"] = list(all_tokens)
        features["avg_path_length"] = min(10, len(all_node_types))
        features["path_diversity"] = min(1.0, len(all_node_types) / 20.0)
        
        return [features]
    
    def _parse_ast_output(self, output_dir: str) -> List[Dict]:
        """Parse AstMiner output files."""
        ast_data = []
        
        # Look for AstMiner output files (.c2s files)
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.c2s'):
                    file_path = os.path.join(root, file)
                    try:
                        ast_metrics = self._extract_ast_metrics_from_c2s(file_path)
                        if ast_metrics:
                            ast_data.append(ast_metrics)
                    except Exception as e:
                        print(f"⚠️  Error parsing AST file {file}: {e}")
        
        return ast_data
    
    def _extract_ast_metrics_from_c2s(self, c2s_file_path: str) -> Optional[Dict]:
        """Extract AST metrics from AstMiner .c2s output file."""
        try:
            features = {
                "path_contexts": [],
                "node_types": set(),
                "tokens": set(),
                "method_count": 0,
                "avg_path_length": 0,
                "path_diversity": 0
            }
            
            with open(c2s_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse path context line: method_name path_context1 path_context2 ...
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
            
        except Exception as e:
            print(f"⚠️  Error extracting AST metrics from {c2s_file_path}: {e}")
            return None
    
    def _calculate_ast_metrics(self, ast_data: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregated AST metrics from multiple files."""
        if not ast_data:
            return {}
        
        # Aggregate metrics from AstMiner output
        total_methods = sum(d.get('method_count', 0) for d in ast_data)
        total_path_contexts = sum(len(d.get('path_contexts', [])) for d in ast_data)
        all_node_types = set()
        all_tokens = set()
        
        for data in ast_data:
            all_node_types.update(data.get('node_types', []))
            all_tokens.update(data.get('tokens', []))
        
        avg_path_lengths = [d.get('avg_path_length', 0) for d in ast_data if d.get('avg_path_length', 0) > 0]
        path_diversities = [d.get('path_diversity', 0) for d in ast_data if d.get('path_diversity', 0) > 0]
        
        # Calculate complexity based on path contexts and methods
        complexity_score = total_path_contexts + total_methods * 2
        
        return {
            'avg_path_length': np.mean(avg_path_lengths) if avg_path_lengths else 0.0,
            'max_path_length': max([d.get('avg_path_length', 0) for d in ast_data]) if ast_data else 0.0,
            'path_variety': np.mean(path_diversities) if path_diversities else 0.0,
            'node_type_diversity': len(all_node_types),
            'complexity_score': complexity_score,
            'nesting_depth': max([d.get('avg_path_length', 0) for d in ast_data]) if ast_data else 0.0,
            'function_count': total_methods,
            'class_count': len([t for t in all_node_types if 'class' in t.lower() or 'Class' in t]),
            'interface_count': len([t for t in all_node_types if 'interface' in t.lower() or 'Interface' in t]),
            'inheritance_depth': 0,  # Would need more detailed parsing
            'total_ast_nodes': total_path_contexts,
            'unique_node_types': len(all_node_types),
            'ast_depth': max([d.get('avg_path_length', 0) for d in ast_data]) if ast_data else 0.0,
            'branching_factor': np.mean(avg_path_lengths) if avg_path_lengths else 0.0
        }
    
    def _extract_basic_ast_features(self, repository_path: str) -> Dict[str, Any]:
        """Extract basic AST features using Python's ast module and simple parsing for other languages."""
        print("📝 Using enhanced basic AST analysis...")
        
        ast_features = {
            'avg_path_length': 0.0,
            'max_path_length': 0.0,
            'path_variety': 0.0,
            'node_type_diversity': 0.0,
            'complexity_score': 0.0,
            'nesting_depth': 0.0,
            'function_count': 0.0,
            'class_count': 0.0,
            'interface_count': 0.0,
            'inheritance_depth': 0.0,
            'total_ast_nodes': 0,
            'unique_node_types': 0,
            'ast_depth': 0.0,
            'branching_factor': 0.0
        }
        
        # Collect files by language
        python_files = []
        javascript_files = []
        java_files = []
        cpp_files = []
        
        for root, _, files in os.walk(repository_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file)[1].lower()
                    if ext == '.py':
                        python_files.append(file_path)
                    elif ext in ['.js', '.jsx', '.ts', '.tsx']:
                        javascript_files.append(file_path)
                    elif ext == '.java':
                        java_files.append(file_path)
                    elif ext in ['.c', '.cpp', '.cc', '.h', '.hpp']:
                        cpp_files.append(file_path)
        
        total_nodes = 0
        node_types = set()
        function_count = 0
        class_count = 0
        max_depth = 0
        
        # Analyze Python files with proper AST
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                analyzer = ASTAnalyzer()
                analyzer.visit(tree)
                
                total_nodes += analyzer.total_nodes
                node_types.update(analyzer.node_types)
                function_count += analyzer.function_count
                class_count += analyzer.class_count
                max_depth = max(max_depth, analyzer.max_depth)
                
            except Exception as e:
                print(f"⚠️  Error parsing {py_file}: {e}")
        
        # Simple analysis for JavaScript files
        for js_file in javascript_files:
            try:
                with open(js_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Simple regex-based analysis for JavaScript
                js_functions = len(re.findall(r'function\s+\w+\s*\(', content)) + len(re.findall(r'const\s+\w+\s*=\s*\([^)]*\)\s*=>', content)) + len(re.findall(r'let\s+\w+\s*=\s*\([^)]*\)\s*=>', content))
                js_classes = len(re.findall(r'class\s+\w+', content))
                js_imports = len(re.findall(r'import\s+', content))
                js_exports = len(re.findall(r'export\s+', content))
                
                function_count += js_functions
                class_count += js_classes
                total_nodes += js_functions + js_classes + js_imports + js_exports
                node_types.update(['function', 'class', 'import', 'export'])
                
            except Exception as e:
                print(f"⚠️  Error parsing {js_file}: {e}")
        
        # Simple analysis for Java files
        for java_file in java_files:
            try:
                with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Simple regex-based analysis for Java
                java_functions = len(re.findall(r'(public|private|protected)?\s*(static\s+)?\w+\s+\w+\s*\([^)]*\)\s*\{', content))
                java_classes = len(re.findall(r'class\s+\w+', content))
                java_interfaces = len(re.findall(r'interface\s+\w+', content))
                java_imports = len(re.findall(r'import\s+', content))
                
                function_count += java_functions
                class_count += java_classes
                ast_features['interface_count'] += java_interfaces
                total_nodes += java_functions + java_classes + java_interfaces + java_imports
                node_types.update(['function', 'class', 'interface', 'import'])
                
            except Exception as e:
                print(f"⚠️  Error parsing {java_file}: {e}")
        
        # Simple analysis for C++ files
        for cpp_file in cpp_files:
            try:
                with open(cpp_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Simple regex-based analysis for C++
                cpp_functions = len(re.findall(r'\w+\s+\w+\s*\(', content))
                cpp_classes = len(re.findall(r'class\s+\w+', content))
                cpp_includes = len(re.findall(r'#include\s+', content))
                
                function_count += cpp_functions
                class_count += cpp_classes
                total_nodes += cpp_functions + cpp_classes + cpp_includes
                node_types.update(['function', 'class', 'include'])
                
            except Exception as e:
                print(f"⚠️  Error parsing {cpp_file}: {e}")
        
        # Calculate features
        ast_features.update({
            'total_ast_nodes': total_nodes,
            'unique_node_types': len(node_types),
            'function_count': function_count,
            'class_count': class_count,
            'nesting_depth': max_depth,
            'complexity_score': function_count + class_count * 2 + ast_features['interface_count'] * 1.5,
            'ast_depth': max_depth,
            'branching_factor': total_nodes / max(max_depth, 1)
        })
        
        return ast_features

    def extract_codebert_embeddings(self, repository_path: str) -> Dict[str, Any]:
        """Extract CodeBERT embeddings with CUDA optimization and large repo handling."""
        print(f"🤖 Extracting optimized CodeBERT embeddings from {repository_path}")
        
        # Initialize CodeBERT if not already done
        self.initialize_codebert()
        
        if self.codebert_model is None:
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
        language_breakdown = {}
        for root, _, files in os.walk(repository_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.codebert_languages:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        rel_path = os.path.relpath(file_path, repository_path)
                        files_to_process.append((file_path, rel_path, ext))
                        lang = self.codebert_languages[ext]
                        language_breakdown[lang] = language_breakdown.get(lang, 0) + 1
        
        if not files_to_process:
            print("⚠️  No supported files found for CodeBERT analysis")
            return self._get_empty_codebert_features()
        
        print(f"   📁 Processing {len(files_to_process)} files...")
        
        # Process files in batches for memory efficiency
        batch_size = 32 if self.device == "cuda" else 16
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
                            "language": self.codebert_languages[ext],
                            "embedding_dim": len(embedding)
                        })
                except Exception as e:
                    print(f"   ⚠️  Error processing {rel_path}: {e}")
                    continue
            
            # Add batch results to main lists
            embeddings.extend(batch_embeddings)
            file_info.extend(batch_file_info)
            
            # Clear GPU memory if using CUDA
            if self.device == "cuda" and self._torch:
                self._torch.cuda.empty_cache()
        
        if not embeddings:
            print("⚠️  No embeddings extracted successfully")
            return self._get_empty_codebert_features()
        
        # Calculate repository-level embedding
        repo_embedding = np.mean(embeddings, axis=0)
        
        # Calculate semantic features
        semantic_features = self._calculate_semantic_features(embeddings)
        
        # Extract significant dimensions for enhanced analysis
        significant_features = self._extract_significant_dimensions(repo_embedding)
        
        print(f"   ✅ Successfully extracted embeddings from {len(embeddings)} files")
        
        return {
            "repository_embedding": repo_embedding.tolist(),
            "num_files": len(embeddings),
            "embedding_dimension": len(repo_embedding),
            "file_embeddings": [emb.tolist() for emb in embeddings],
            "file_info": file_info,
            "language_breakdown": language_breakdown,
            "semantic_features": semantic_features,
            "significant_dimensions": significant_features
        }
    
    def _get_empty_codebert_features(self) -> Dict[str, Any]:
        """Get empty CodeBERT features."""
        return {
            'repository_embedding': [0.0] * 768,  # Standard CodeBERT dimension
            'significant_dimensions': {},
            'semantic_features': {}
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
            code_content = self._clean_code(code_content)
            
            # Use tokenizer's built-in truncation to avoid token length errors
            inputs = self.codebert_tokenizer(
                code_content, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512,
                add_special_tokens=True
            )
            
            # Move to device
            device_obj = self._torch.device(self.device)
            input_ids = inputs['input_ids'].to(device_obj)
            attention_mask = inputs['attention_mask'].to(device_obj)
            
            # Get embeddings with gradient computation disabled for efficiency
            with self._torch.no_grad():
                outputs = self.codebert_model(input_ids, attention_mask=attention_mask)
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
    
    def _clean_code(self, code: str) -> str:
        """Clean code for tokenization."""
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
    
    def _calculate_semantic_features(self, embeddings: List[np.ndarray]) -> Dict[str, float]:
        """Calculate semantic features from embeddings."""
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
    
    def _extract_significant_dimensions(self, embedding: np.ndarray) -> Dict[str, float]:
        """Extract significant dimensions for enhanced category classification."""
        significant_features = {}
        
        # Extract dimensions for each category pair
        for pair_name, dimensions in self.significant_dimensions.items():
            for i, dim in enumerate(dimensions):
                if dim < len(embedding):
                    significant_features[f'{pair_name}_dim_{dim}'] = float(embedding[dim])
        
        # Also extract top significant dimensions across all pairs
        all_significant_dims = set()
        for dims in self.significant_dimensions.values():
            all_significant_dims.update(dims)
        
        top_dims = sorted(list(all_significant_dims))[:50]  # Top 50 dimensions
        for i, dim in enumerate(top_dims):
            if dim < len(embedding):
                significant_features[f'top_dim_{dim}'] = float(embedding[dim])
        
        return significant_features
    
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
    
    def extract_keyword_features(self, repository_path: str) -> Dict[str, Any]:
        """Extract keyword-based features using enhanced keyword analysis."""
        print(f"🔍 Extracting keyword features from {repository_path}")
        
        # Try to use enhanced keyword analyzer from v2
        import sys
        sys.path.append('.')
        
        try:
            from keywords import analyze_repository_keywords
            print("📊 Using enhanced keyword analyzer...")
            keyword_features = analyze_repository_keywords(repository_path)
            return keyword_features
        except ImportError:
            print("⚠️  Enhanced keyword analyzer not available, using basic extraction...")
            return self._extract_basic_keyword_features(repository_path)
    
    def _extract_basic_keyword_features(self, repository_path: str) -> Dict[str, Any]:
        """Basic keyword extraction as fallback."""
        keyword_features = {
            'framework_keywords': 0.0,
            'library_keywords': 0.0,
            'data_science_keywords': 0.0,
            'web_keywords': 0.0,
            'cli_keywords': 0.0,
            'game_keywords': 0.0,
            'mobile_keywords': 0.0,
            'testing_keywords': 0.0,
            'database_keywords': 0.0,
            'cloud_keywords': 0.0,
            'total_keywords': 0,
            'keyword_diversity': 0.0
        }
        
        # Load keyword definitions
        import sys
        sys.path.append('.')
        
        try:
            from keywords import expertise_keywords, topics_keywords
        except ImportError:
            # Fallback to basic keyword definitions
            expertise_keywords = {
                'framework': ['react', 'angular', 'vue', 'django', 'flask', 'express', 'spring', 'laravel'],
                'library': ['library', 'module', 'package', 'dependency', 'import', 'require'],
                'data_science': ['machine learning', 'ml', 'ai', 'neural network', 'tensorflow', 'pytorch', 'scikit-learn'],
                'web': ['http', 'api', 'rest', 'frontend', 'backend', 'server', 'client'],
                'cli': ['command', 'cli', 'terminal', 'console', 'argument', 'option'],
                'game': ['game', 'graphics', 'animation', 'sprite', 'collision', 'physics'],
                'mobile': ['mobile', 'android', 'ios', 'react native', 'flutter'],
                'testing': ['test', 'unit', 'integration', 'spec', 'mock', 'assert'],
                'database': ['database', 'sql', 'mongodb', 'redis', 'query', 'schema'],
                'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'deployment']
            }
            topics_keywords = {
                'framework': ['react', 'angular', 'vue', 'django', 'flask', 'express', 'spring', 'laravel'],
                'library': ['library', 'module', 'package', 'dependency', 'import', 'require'],
                'data_science': ['machine learning', 'ml', 'ai', 'neural network', 'tensorflow', 'pytorch', 'scikit-learn'],
                'web': ['http', 'api', 'rest', 'frontend', 'backend', 'server', 'client'],
                'cli': ['command', 'cli', 'terminal', 'console', 'argument', 'option'],
                'game': ['game', 'graphics', 'animation', 'sprite', 'collision', 'physics'],
                'mobile': ['mobile', 'android', 'ios', 'react native', 'flutter'],
                'testing': ['test', 'unit', 'integration', 'spec', 'mock', 'assert'],
                'database': ['database', 'sql', 'mongodb', 'redis', 'query', 'schema'],
                'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'deployment']
            }
        
        # Collect all text content
        all_content = self._collect_repository_content(repository_path)
        
        if not all_content:
            return keyword_features
        
        # Analyze keywords
        keyword_counts = self._analyze_keywords(all_content, expertise_keywords, topics_keywords)
        
        # Update features
        keyword_features.update(keyword_counts)
        
        return keyword_features
    
    def _collect_repository_content(self, repository_path: str) -> str:
        """Collect all text content from repository."""
        content_parts = []
        
        # Check if the path is a directory
        if not os.path.isdir(repository_path):
            print(f"⚠️  Path is not a directory: {repository_path}")
            return ""
        
        # Text file extensions
        text_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.php', '.rb', '.go', '.rs', '.kt', '.swift', '.ts', '.jsx', '.tsx', '.md', '.txt', '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.scss', '.sass'}
        
        for root, _, files in os.walk(repository_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in text_extensions:
                    file_path = os.path.join(root, file)
                    # Only process actual files
                    if os.path.isfile(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            if content.strip():
                                content_parts.append(content)
                        except Exception as e:
                            print(f"⚠️  Error reading {file_path}: {e}")
        
        return '\n\n'.join(content_parts)
    
    def _analyze_keywords(self, content: str, expertise_keywords: Dict, topics_keywords: Dict) -> Dict[str, float]:
        """Analyze keywords in content."""
        content_lower = content.lower()
        
        keyword_counts = {}
        
        # Count expertise keywords
        for category, keywords in expertise_keywords.items():
            count = sum(1 for keyword in keywords if keyword.lower() in content_lower)
            keyword_counts[f'{category.lower()}_keywords'] = count
        
        # Count topic keywords
        for category, keywords in topics_keywords.items():
            count = sum(1 for keyword in keywords if keyword.lower() in content_lower)
            keyword_counts[f'{category.lower()}_keywords'] = count
        
        # Calculate totals
        total_keywords = sum(keyword_counts.values())
        keyword_counts['total_keywords'] = total_keywords
        keyword_counts['keyword_diversity'] = len([v for v in keyword_counts.values() if v > 0])
        
        return keyword_counts
    
    def extract_file_structure_features(self, repository_path: str) -> Dict[str, Any]:
        """Extract file organization and structure features."""
        print(f"📁 Extracting file structure features from {repository_path}")
        
        repo_path = Path(repository_path)
        structure_features = {}
        
        if repo_path.exists() and repo_path.is_dir():
            # File organization patterns
            files = list(repo_path.rglob('*'))
            directories = [f for f in files if f.is_dir()]
            source_files = [f for f in files if f.is_file() and f.suffix in ['.py', '.js', '.java', '.cpp', '.c', '.h', '.php', '.rb', '.go', '.rs', '.kt', '.swift', '.ts', '.jsx', '.tsx']]
            
            structure_features = {
                'total_files': len(files),
                'total_directories': len(directories),
                'source_files': len(source_files),
                'avg_files_per_dir': len(files) / max(len(directories), 1),
                'max_directory_depth': max([len(f.parts) - len(repo_path.parts) for f in files]),
                'has_tests': 1.0 if any('test' in f.name.lower() for f in files) else 0.0,
                'has_docs': 1.0 if any('doc' in f.name.lower() or 'readme' in f.name.lower() for f in files) else 0.0,
                'has_config': 1.0 if any('config' in f.name.lower() or '.json' in f.name or '.yaml' in f.name for f in files) else 0.0,
                'has_dependencies': 1.0 if any('requirements' in f.name.lower() or 'package.json' in f.name or 'pom.xml' in f.name for f in files) else 0.0,
                'has_ci_cd': 1.0 if any('.github' in str(f) or '.gitlab' in str(f) or 'jenkins' in f.name.lower() for f in files) else 0.0,
                'has_docker': 1.0 if any('dockerfile' in f.name.lower() or 'docker-compose' in f.name.lower() for f in files) else 0.0
            }
            
            # Language distribution
            extensions = [f.suffix.lower() for f in source_files]
            structure_features['language_diversity'] = len(set(extensions))
            
            # Count files by language
            language_counts = {}
            for ext in extensions:
                if ext in self.supported_languages:
                    lang = self.supported_languages[ext]
                    language_counts[lang] = language_counts.get(lang, 0) + 1
            
            structure_features['language_counts'] = language_counts
            structure_features['total_source_files'] = len(source_files)
            
            # Main language (hash-based)
            main_lang = max(set(extensions), key=extensions.count) if extensions else ''
            structure_features['main_language_hash'] = hash(main_lang) % 1000
        
        return structure_features
    
    def analyze_repository(self, repository_path: str) -> Dict[str, Any]:
        """Comprehensive repository analysis."""
        print(f"🚀 Starting comprehensive analysis of: {repository_path}")
        print("=" * 60)
        
        if not os.path.exists(repository_path):
            raise ValueError(f"Repository path does not exist: {repository_path}")
        
        # Extract all features
        print("📊 Extracting features...")
        ast_features = self.extract_ast_features(repository_path)
        codebert_features = self.extract_codebert_embeddings(repository_path)
        keyword_features = self.extract_keyword_features(repository_path)
        structure_features = self.extract_file_structure_features(repository_path)
        
        # Combine all features including significant dimensions
        all_features = {
            **ast_features,
            **codebert_features.get('semantic_features', {}),
            **codebert_features.get('significant_dimensions', {}),
            **keyword_features,
            **structure_features
        }
        
        # Quality assessment
        print("⭐ Assessing code quality...")
        quality_assessment = self.assess_code_quality(all_features)
        
        # Programmer characteristics
        print("👨‍💻 Analyzing programmer characteristics...")
        programmer_characteristics = self.analyze_programmer_characteristics(all_features)
        
        # Architecture analysis
        print("🏗️  Analyzing architecture patterns...")
        architecture_analysis = self.analyze_architecture_patterns(all_features)
        
        # Compile results
        results = {
            'repository_path': repository_path,
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'features': {
                'ast_features': ast_features,
                'codebert_features': codebert_features,
                'keyword_features': keyword_features,
                'structure_features': structure_features
            },
            'quality_assessment': quality_assessment,
            'programmer_characteristics': programmer_characteristics,
            'architecture_analysis': architecture_analysis,
            'summary': self._generate_summary(quality_assessment, programmer_characteristics, architecture_analysis)
        }
        
        print("✅ Analysis complete!")
        return results
    
    def assess_code_quality(self, features: Dict) -> Dict[str, Any]:
        """Assess code quality based on extracted features."""
        quality_metrics = {
            'overall_score': 0.0,
            'code_quality': {},
            'architecture_quality': {},
            'documentation_quality': {},
            'maintainability': {}
        }
        
        # Enhanced code quality metrics using improved features
        complexity_score = min(1.0, features.get('complexity_score', 0) / 50.0)
        structure_score = min(1.0, features.get('total_ast_nodes', 0) / 100.0)
        organization_score = 1.0 if features.get('has_tests', False) else 0.5
        consistency_score = min(1.0, features.get('unique_node_types', 0) / 20.0)
        
        # Additional quality metrics from significant dimensions
        semantic_quality = features.get('semantic_coherence', 0)
        code_organization = min(1.0, features.get('language_diversity', 0) / 5.0)
        
        quality_metrics['code_quality'] = {
            'complexity': complexity_score,
            'structure': structure_score,
            'organization': organization_score,
            'consistency': consistency_score,
            'semantic_quality': semantic_quality,
            'code_organization': code_organization
        }
        
        # Architecture quality metrics
        modularity_score = min(1.0, features.get('function_count', 0) / 50.0)
        abstraction_score = min(1.0, features.get('class_count', 0) / 20.0)
        separation_score = 1.0 / (1.0 + features.get('nesting_depth', 0))
        scalability_score = min(1.0, features.get('total_files', 0) / 100.0)
        
        quality_metrics['architecture_quality'] = {
            'modularity': modularity_score,
            'abstraction': abstraction_score,
            'separation': separation_score,
            'scalability': scalability_score
        }
        
        # Documentation quality metrics
        quality_metrics['documentation_quality'] = {
            'readme_presence': 1.0 if features.get('has_docs', False) else 0.0,
            'config_documentation': 1.0 if features.get('has_config', False) else 0.0,
            'dependency_documentation': 1.0 if features.get('has_dependencies', False) else 0.0
        }
        
        # Maintainability metrics
        quality_metrics['maintainability'] = {
            'test_coverage': 1.0 if features.get('has_tests', False) else 0.0,
            'code_organization': min(1.0, features.get('language_diversity', 0) / 5.0),
            'structure_clarity': 1.0 / (1.0 + features.get('max_directory_depth', 0)),
            'consistency': 1.0 / (1.0 + features.get('embedding_std', 0))
        }
        
        # Calculate overall score
        code_avg = np.mean(list(quality_metrics['code_quality'].values()))
        arch_avg = np.mean(list(quality_metrics['architecture_quality'].values()))
        doc_avg = np.mean(list(quality_metrics['documentation_quality'].values()))
        maint_avg = np.mean(list(quality_metrics['maintainability'].values()))
        
        quality_metrics['overall_score'] = (code_avg + arch_avg + doc_avg + maint_avg) / 4
        
        return quality_metrics
    
    def analyze_programmer_characteristics(self, features: Dict) -> Dict[str, Any]:
        """Analyze programmer characteristics based on code patterns."""
        characteristics = {
            'experience_level': self._assess_experience_level(features),
            'coding_style': self._assess_coding_style(features),
            'attention_to_detail': self._assess_attention_to_detail(features),
            'architectural_thinking': self._assess_architectural_thinking(features),
            'best_practices': self._assess_best_practices(features),
            'specialization': self._assess_specialization(features)
        }
        
        return characteristics
    
    def _assess_experience_level(self, features: Dict) -> str:
        """Assess programmer experience level."""
        experience_score = 0
        
        # Code complexity and structure
        if features.get('complexity_score', 0) > 20:
            experience_score += 3
        elif features.get('complexity_score', 0) > 10:
            experience_score += 2
        elif features.get('complexity_score', 0) > 5:
            experience_score += 1
        
        if features.get('total_ast_nodes', 0) > 100:
            experience_score += 2
        elif features.get('total_ast_nodes', 0) > 50:
            experience_score += 1
        
        # Project organization
        if features.get('has_tests', False):
            experience_score += 2
        if features.get('has_docs', False):
            experience_score += 1
        if features.get('has_config', False):
            experience_score += 1
        if features.get('has_ci_cd', False):
            experience_score += 2
        
        # Code structure
        if features.get('function_count', 0) > 50:
            experience_score += 2
        elif features.get('function_count', 0) > 20:
            experience_score += 1
        
        if features.get('class_count', 0) > 10:
            experience_score += 2
        elif features.get('class_count', 0) > 5:
            experience_score += 1
        if features.get('class_count', 0) > 5:
            experience_score += 1
        
        if experience_score >= 6:
            return "Senior"
        elif experience_score >= 3:
            return "Intermediate"
        else:
            return "Junior"
    
    def _assess_coding_style(self, features: Dict) -> str:
        """Assess coding style."""
        if features.get('has_tests', False) and features.get('has_docs', False):
            return "Professional"
        elif features.get('complexity_score', 0) < 3:
            return "Simple and Clean"
        elif features.get('function_count', 0) > 30:
            return "Comprehensive"
        else:
            return "Basic"
    
    def _assess_attention_to_detail(self, features: Dict) -> str:
        """Assess attention to detail."""
        detail_score = 0
        
        if features.get('has_tests', False):
            detail_score += 2
        if features.get('has_docs', False):
            detail_score += 1
        if features.get('has_config', False):
            detail_score += 1
        if features.get('has_dependencies', False):
            detail_score += 1
        
        if detail_score >= 4:
            return "High"
        elif detail_score >= 2:
            return "Medium"
        else:
            return "Low"
    
    def _assess_architectural_thinking(self, features: Dict) -> str:
        """Assess architectural thinking."""
        arch_score = 0
        
        if features.get('class_count', 0) > 5:
            arch_score += 2
        if features.get('interface_count', 0) > 0:
            arch_score += 2
        if features.get('inheritance_depth', 0) > 2:
            arch_score += 1
        if features.get('max_directory_depth', 0) > 3:
            arch_score += 1
        
        if arch_score >= 4:
            return "Strong"
        elif arch_score >= 2:
            return "Moderate"
        else:
            return "Basic"
    
    def _assess_best_practices(self, features: Dict) -> str:
        """Assess adherence to best practices."""
        practice_score = 0
        
        if features.get('has_tests', False):
            practice_score += 2
        if features.get('has_docs', False):
            practice_score += 1
        if features.get('has_config', False):
            practice_score += 1
        if features.get('nesting_depth', 0) < 5:
            practice_score += 1
        if features.get('complexity_score', 0) < 10:
            practice_score += 1
        
        if practice_score >= 4:
            return "Excellent"
        elif practice_score >= 2:
            return "Good"
        else:
            return "Needs Improvement"
    
    def _assess_specialization(self, features: Dict) -> str:
        """Assess programmer specialization using enhanced analysis."""
        # Analyze keyword patterns to determine specialization
        keyword_counts = {
            'data_science': features.get('data_science_keywords', 0),
            'web_development': features.get('web_keywords', 0),
            'mobile_development': features.get('mobile_keywords', 0),
            'game_development': features.get('game_keywords', 0),
            'cli_tools': features.get('cli_keywords', 0),
            'testing': features.get('testing_keywords', 0),
            'database': features.get('database_keywords', 0),
            'cloud': features.get('cloud_keywords', 0)
        }
        
        # Also consider significant dimensions for specialization
        significant_scores = {}
        for pair_name, dimensions in self.significant_dimensions.items():
            if 'data_science' in pair_name:
                significant_scores['data_science'] = significant_scores.get('data_science', 0) + 1
            if 'web' in pair_name:
                significant_scores['web_development'] = significant_scores.get('web_development', 0) + 1
            if 'cli' in pair_name:
                significant_scores['cli_tools'] = significant_scores.get('cli_tools', 0) + 1
        
        # Combine keyword and significant dimension analysis
        combined_scores = {}
        for category in keyword_counts:
            combined_scores[category] = keyword_counts[category] + significant_scores.get(category, 0) * 2
        
        max_specialization = max(combined_scores.items(), key=lambda x: x[1])
        
        if max_specialization[1] > 15:
            return f"Specialized in {max_specialization[0]}"
        elif max_specialization[1] > 8:
            return f"Focused on {max_specialization[0]}"
        else:
            return "Generalist"
    
    def _generate_summary(self, quality_assessment: Dict, programmer_characteristics: Dict, architecture_analysis: Dict) -> Dict[str, Any]:
        """Generate analysis summary."""
        return {
            'quality_score': quality_assessment['overall_score'],
            'experience_level': programmer_characteristics['experience_level'],
            'coding_style': programmer_characteristics['coding_style'],
            'detected_patterns': architecture_analysis['detected_patterns'],
            'key_strengths': self._identify_strengths(quality_assessment, programmer_characteristics),
            'improvement_areas': self._identify_improvements(quality_assessment, programmer_characteristics)
        }
    
    def _identify_strengths(self, quality_assessment: Dict, programmer_characteristics: Dict) -> List[str]:
        """Identify key strengths."""
        strengths = []
        
        if quality_assessment['overall_score'] > 0.8:
            strengths.append("High overall code quality")
        if programmer_characteristics['experience_level'] == "Senior":
            strengths.append("Senior-level development practices")
        if programmer_characteristics['best_practices'] == "Excellent":
            strengths.append("Excellent adherence to best practices")
        if quality_assessment['documentation_quality']['readme_presence'] > 0:
            strengths.append("Good documentation")
        if quality_assessment['maintainability']['test_coverage'] > 0:
            strengths.append("Testing practices in place")
        
        return strengths
    
    def _identify_improvements(self, quality_assessment: Dict, programmer_characteristics: Dict) -> List[str]:
        """Identify areas for improvement."""
        improvements = []
        
        if quality_assessment['overall_score'] < 0.6:
            improvements.append("Overall code quality needs improvement")
        if not quality_assessment['maintainability']['test_coverage']:
            improvements.append("Add test coverage")
        if not quality_assessment['documentation_quality']['readme_presence']:
            improvements.append("Add documentation")
        if programmer_characteristics['best_practices'] == "Needs Improvement":
            improvements.append("Follow coding best practices")
        if quality_assessment['architecture_quality']['modularity'] < 0.5:
            improvements.append("Improve code modularity")
        
        return improvements


class ASTAnalyzer(ast.NodeVisitor):
    """AST analyzer for Python code."""
    
    def __init__(self):
        self.total_nodes = 0
        self.node_types = set()
        self.function_count = 0
        self.class_count = 0
        self.max_depth = 0
        self.current_depth = 0
    
    def visit(self, node):
        self.total_nodes += 1
        self.node_types.add(type(node).__name__)
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        
        # Count specific node types
        if isinstance(node, ast.FunctionDef):
            self.function_count += 1
        elif isinstance(node, ast.ClassDef):
            self.class_count += 1
        
        self.generic_visit(node)
        self.current_depth -= 1


def main():
    """Main function to demonstrate the analyzer."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python comprehensive_repository_analyzer_v4.py <repository_path>")
        sys.exit(1)
    
    repository_path = sys.argv[1]
    
    # Initialize analyzer
    analyzer = ComprehensiveRepositoryAnalyzerV4()
    
    try:
        # Analyze repository
        results = analyzer.analyze_repository(repository_path)
        
        # Print results
        print("\n" + "="*60)
        print("📊 ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\n🏗️  Architecture Patterns:")
        for pattern in results['architecture_analysis']['detected_patterns']:
            confidence = results['architecture_analysis']['pattern_confidence'].get(pattern, 0)
            print(f"   • {pattern} (confidence: {confidence:.2f})")
        
        print(f"\n⭐ Quality Assessment:")
        print(f"   Overall Score: {results['quality_assessment']['overall_score']:.3f}")
        
        print(f"\n👨‍💻 Programmer Characteristics:")
        print(f"   Experience Level: {results['programmer_characteristics']['experience_level']}")
        print(f"   Coding Style: {results['programmer_characteristics']['coding_style']}")
        print(f"   Best Practices: {results['programmer_characteristics']['best_practices']}")
        print(f"   Specialization: {results['programmer_characteristics']['specialization']}")
        
        print(f"\n📈 Summary:")
        print(f"   Key Strengths: {', '.join(results['summary']['key_strengths'])}")
        print(f"   Improvement Areas: {', '.join(results['summary']['improvement_areas'])}")
        
        # Save results
        output_file = f"analysis_results_{Path(repository_path).name}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
