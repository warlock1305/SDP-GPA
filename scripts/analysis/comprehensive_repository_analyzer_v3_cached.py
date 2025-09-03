"""
Comprehensive Repository Analyzer v3.0 with Caching
==================================================

Enhanced analysis pipeline with caching for resumable training:
1. AST (Abstract Syntax Tree) Analysis
2. CodeBERT Semantic Embeddings
3. Keyword and Pattern Analysis
4. Code Quality Assessment
5. Programmer Characteristics Analysis

Features:
- Saves embeddings and extractions to temp_repo_data folder
- Checks for existing data before re-extracting
- Allows resumable training and optimization
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
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

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
        
        # Visit children
        super().visit(node)
        self.current_depth -= 1
import hashlib
import shutil

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
        
        # Continue visiting children
        self.generic_visit(node)
        self.current_depth -= 1

class ComprehensiveRepositoryAnalyzerV3Cached:
    def __init__(self, cache_dir: str = "temp_repo_data"):
        """Initialize the analyzer with caching capabilities."""
        self.quality_regressor = None
        self.quality_scaler = StandardScaler()
        
        # Cache directory for storing embeddings and extractions
        self.cache_dir = cache_dir
        self.embeddings_dir = os.path.join(cache_dir, "embeddings")
        self.ast_dir = os.path.join(cache_dir, "ast_features")
        self.keywords_dir = os.path.join(cache_dir, "keyword_features")
        self.structure_dir = os.path.join(cache_dir, "structure_features")
        
        # Create cache directories
        for dir_path in [self.embeddings_dir, self.ast_dir, self.keywords_dir, self.structure_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # AST extraction tools
        self.astminer_jar = "../astminer-0.9.0/build/libs/astminer.jar"
        
        # CodeBERT components
        self.codebert_model = None
        self.codebert_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load significant dimensions from v2 analysis for enhanced classification
        self.significant_dimensions = self.load_significant_dimensions()
        
        # Supported languages for AST extraction
        self.supported_languages = {
            ".java": "java",
            ".py": "py", 
            ".js": "js",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".kt": "kotlin",
            ".kts": "kotlin"
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
        
        print(f"🔧 Comprehensive Repository Analyzer v3.0 (Cached) initialized")
        print(f"📱 Using device: {self.device}")
        print(f"💾 Cache directory: {self.cache_dir}")
    
    def _extract_ast_with_astminer(self, repository_path: str) -> Dict[str, Any]:
        """Extract AST features using AstMiner."""
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
        
        # Detect languages in the repository
        languages = self._detect_languages(repository_path)
        if not languages:
            print("⚠️  No supported languages detected")
            return ast_features
        
        print(f"🔍 Detected languages: {languages}")
        
        # Extract AST data for each language
        all_ast_data = []
        for lang in languages:
            lang_ast_data = self._extract_ast_for_language(repository_path, lang)
            if lang_ast_data:
                all_ast_data.extend(lang_ast_data)
        
        if all_ast_data:
            ast_features = self._calculate_ast_metrics(all_ast_data)
        
        return ast_features
    
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
                
                # Run AstMiner
                result = subprocess.run(
                    ["java", "-jar", self.astminer_jar, config_path],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"❌ AstMiner failed for {language}: {result.stderr}")
                    return []
                
                # Parse AST output
                ast_data = self._parse_ast_output(temp_dir)
                return ast_data
                
        except Exception as e:
            print(f"❌ Error extracting AST for {language}: {e}")
            return []
    
    def _parse_ast_output(self, output_dir: str) -> List[Dict]:
        """Parse AstMiner output files."""
        ast_data = []
        
        # Look for AST output files
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.csv') or file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            # Parse AST paths and extract metrics
                            ast_metrics = self._extract_ast_metrics_from_content(content)
                            if ast_metrics:
                                ast_data.append(ast_metrics)
                    except Exception as e:
                        print(f"⚠️  Error parsing AST file {file}: {e}")
        
        return ast_data
    
    def _extract_ast_metrics_from_content(self, content: str) -> Optional[Dict]:
        """Extract AST metrics from AstMiner output content."""
        try:
            lines = content.strip().split('\n')
            if not lines:
                return None
            
            # Extract AST paths and calculate metrics
            paths = []
            node_types = set()
            
            for line in lines:
                if line.strip():
                    # Parse AST path (simplified)
                    parts = line.split()
                    if len(parts) >= 2:
                        path = parts[0]
                        paths.append(path)
                        
                        # Extract node types from path
                        nodes = path.split(',')
                        node_types.update(nodes)
            
            if not paths:
                return None
            
            # Calculate metrics
            path_lengths = [len(path.split(',')) for path in paths]
            
            return {
                'avg_path_length': np.mean(path_lengths),
                'max_path_length': np.max(path_lengths),
                'path_variety': len(set(paths)) / len(paths) if paths else 0,
                'node_type_diversity': len(node_types),
                'total_paths': len(paths),
                'unique_paths': len(set(paths))
            }
            
        except Exception as e:
            print(f"⚠️  Error extracting AST metrics: {e}")
            return None
    
    def _calculate_ast_metrics(self, ast_data: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregated AST metrics from multiple files."""
        if not ast_data:
            return {}
        
        # Aggregate metrics
        avg_path_lengths = [d.get('avg_path_length', 0) for d in ast_data]
        max_path_lengths = [d.get('max_path_length', 0) for d in ast_data]
        path_varieties = [d.get('path_variety', 0) for d in ast_data]
        node_diversities = [d.get('node_type_diversity', 0) for d in ast_data]
        
        return {
            'avg_path_length': np.mean(avg_path_lengths),
            'max_path_length': np.max(max_path_lengths),
            'path_variety': np.mean(path_varieties),
            'node_type_diversity': np.sum(node_diversities),
            'complexity_score': np.sum([d.get('total_paths', 0) for d in ast_data]),
            'nesting_depth': np.max([d.get('max_path_length', 0) for d in ast_data]),
            'function_count': 0,  # Would need more detailed parsing
            'class_count': 0,     # Would need more detailed parsing
            'interface_count': 0, # Would need more detailed parsing
            'inheritance_depth': 0, # Would need more detailed parsing
            'total_ast_nodes': np.sum([d.get('total_paths', 0) for d in ast_data]),
            'unique_node_types': np.sum([d.get('node_type_diversity', 0) for d in ast_data]),
            'ast_depth': np.max([d.get('max_path_length', 0) for d in ast_data]),
            'branching_factor': np.mean([d.get('avg_path_length', 0) for d in ast_data])
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
                with open(js_file, 'r', encoding='utf-8') as f:
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
                with open(java_file, 'r', encoding='utf-8') as f:
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
                with open(cpp_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple regex-based analysis for C++
                cpp_functions = len(re.findall(r'\w+\s+\w+\s*\([^)]*\)\s*\{', content))
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
    
    def _get_repo_hash(self, repository_path: str) -> str:
        """Generate a unique hash for the repository based on its path and modification time."""
        try:
            # Get the most recent modification time of any file in the repository
            latest_mtime = 0
            for root, _, files in os.walk(repository_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    mtime = os.path.getmtime(file_path)
                    latest_mtime = max(latest_mtime, mtime)
            
            # Create hash from path and modification time
            hash_input = f"{repository_path}_{latest_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception as e:
            print(f"⚠️  Error generating repo hash: {e}")
            # Fallback to path-based hash
            return hashlib.md5(repository_path.encode()).hexdigest()
    
    def _get_cache_path(self, repo_hash: str, cache_type: str) -> str:
        """Get the cache file path for a specific repository and cache type."""
        cache_dirs = {
            'embeddings': self.embeddings_dir,
            'ast': self.ast_dir,
            'keywords': self.keywords_dir,
            'structure': self.structure_dir
        }
        
        if cache_type not in cache_dirs:
            raise ValueError(f"Invalid cache type: {cache_type}")
        
        return os.path.join(cache_dirs[cache_type], f"{repo_hash}.json")
    
    def _load_from_cache(self, cache_path: str) -> Optional[Dict]:
        """Load data from cache file."""
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                print(f"📂 Loaded from cache: {cache_path}")
                return data
        except Exception as e:
            print(f"⚠️  Error loading from cache {cache_path}: {e}")
        return None
    
    def _save_to_cache(self, cache_path: str, data: Dict) -> bool:
        """Save data to cache file."""
        try:
            # Convert data to ensure JSON serialization works properly
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            converted_data = convert_for_json(data)
            
            with open(cache_path, 'w') as f:
                json.dump(converted_data, f, indent=2)
            print(f"💾 Saved to cache: {cache_path}")
            return True
        except Exception as e:
            print(f"⚠️  Error saving to cache {cache_path}: {e}")
            return False
    
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
                self.codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
                self.codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
                self.codebert_model.to(self.device)
                self.codebert_model.eval()
                print("✅ CodeBERT model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading CodeBERT: {e}")
                self.codebert_model = None
                self.codebert_tokenizer = None
    
    def extract_ast_features(self, repository_path: str) -> Dict[str, Any]:
        """Extract AST features with caching."""
        repo_hash = self._get_repo_hash(repository_path)
        cache_path = self._get_cache_path(repo_hash, 'ast')
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data
        
        print(f"🌳 Extracting AST features from {repository_path}")
        
        # Extract AST features (this would be the same as in the original analyzer)
        # For now, I'll use a placeholder that calls the original method
        # In practice, you would copy the AST extraction logic here
        
        # Placeholder - in real implementation, copy the AST extraction logic
        ast_features = self._extract_ast_features_impl(repository_path)
        
        # Save to cache
        self._save_to_cache(cache_path, ast_features)
        
        return ast_features
    
    def _extract_ast_features_impl(self, repository_path: str) -> Dict[str, Any]:
        """Implementation of AST feature extraction."""
        print(f"🌳 Extracting AST features from {repository_path}")
        
        # Check if AstMiner is available
        if os.path.exists(self.astminer_jar):
            print("🔧 Using AstMiner for AST extraction...")
            return self._extract_ast_with_astminer(repository_path)
        else:
            print("⚠️  AstMiner not found at", self.astminer_jar)
            print("📝 Using basic AST analysis instead...")
            return self._extract_basic_ast_features(repository_path)
    
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
                with open(js_file, 'r', encoding='utf-8') as f:
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
                with open(java_file, 'r', encoding='utf-8') as f:
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
                with open(cpp_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple regex-based analysis for C++
                cpp_functions = len(re.findall(r'\w+\s+\w+\s*\([^)]*\)\s*\{', content))
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
    
    def _extract_ast_with_astminer(self, repository_path: str) -> Dict[str, Any]:
        """Extract AST features using AstMiner."""
        print("🔧 Using AstMiner for AST extraction...")
        
        # This is a simplified version - in practice, you'd need the full AstMiner setup
        # For now, fall back to basic extraction
        return self._extract_basic_ast_features(repository_path)
    
    def extract_codebert_embeddings(self, repository_path: str) -> Dict[str, Any]:
        """Extract CodeBERT embeddings with caching."""
        repo_hash = self._get_repo_hash(repository_path)
        cache_path = self._get_cache_path(repo_hash, 'embeddings')
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data
        
        print(f"🤖 Extracting CodeBERT embeddings from {repository_path}")
        
        self.initialize_codebert()
        
        if self.codebert_model is None:
            print("⚠️  CodeBERT not available, skipping embeddings")
            return {}
        
        # Extract embeddings (this would be the same as in the original analyzer)
        # For now, I'll use a placeholder that calls the original method
        
        # Placeholder - in real implementation, copy the CodeBERT extraction logic
        embeddings_data = self._extract_codebert_embeddings_impl(repository_path)
        
        # Save to cache
        self._save_to_cache(cache_path, embeddings_data)
        
        return embeddings_data
    
    def _extract_codebert_embeddings_impl(self, repository_path: str) -> Dict[str, Any]:
        """Implementation of CodeBERT embedding extraction."""
        embeddings = []
        file_info = []
        
        # Check if the path is a directory
        if not os.path.isdir(repository_path):
            print(f"⚠️  Path is not a directory: {repository_path}")
            return {}
        
        # Walk through repository
        for root, _, files in os.walk(repository_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.codebert_languages:
                    file_path = os.path.join(root, file)
                    # Only process actual files
                    if os.path.isfile(file_path):
                        rel_path = os.path.relpath(file_path, repository_path)
                        
                        # Extract embedding
                        embedding = self._extract_file_embedding(file_path)
                        if embedding is not None:
                            embeddings.append(embedding)
                            file_info.append({
                                "file_path": rel_path,
                                "language": self.codebert_languages[ext],
                                "embedding_dim": len(embedding)
                            })
        
        if not embeddings:
            return {}
        
        # Calculate repository-level embedding
        repo_embedding = np.mean(embeddings, axis=0)
        
        # Calculate semantic features
        semantic_features = self._calculate_semantic_features(embeddings)
        
        # Extract significant dimensions for enhanced analysis
        significant_features = self._extract_significant_dimensions(repo_embedding)
        
        return {
            "repository_embedding": repo_embedding.tolist(),
            "num_files": len(embeddings),
            "embedding_dimension": len(repo_embedding),
            "file_embeddings": [emb.tolist() for emb in embeddings],
            "file_info": file_info,
            "semantic_features": semantic_features,
            "significant_dimensions": significant_features
        }
    
    def _extract_file_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """Extract CodeBERT embedding for a single file."""
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
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Get embeddings
            with torch.no_grad():
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
        """Extract keyword features with caching."""
        repo_hash = self._get_repo_hash(repository_path)
        cache_path = self._get_cache_path(repo_hash, 'keywords')
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data
        
        print(f"🔍 Extracting keyword features from {repository_path}")
        
        # Extract keyword features (this would be the same as in the original analyzer)
        # For now, I'll use a placeholder that calls the original method
        
        # Placeholder - in real implementation, copy the keyword extraction logic
        keyword_features = self._extract_keyword_features_impl(repository_path)
        
        # Save to cache
        self._save_to_cache(cache_path, keyword_features)
        
        return keyword_features
    
    def _extract_keyword_features_impl(self, repository_path: str) -> Dict[str, Any]:
        """Implementation of keyword feature extraction."""
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
        """Extract file structure features with caching."""
        repo_hash = self._get_repo_hash(repository_path)
        cache_path = self._get_cache_path(repo_hash, 'structure')
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data
        
        print(f"📁 Extracting file structure features from {repository_path}")
        
        # Extract file structure features (this would be the same as in the original analyzer)
        # For now, I'll use a placeholder that calls the original method
        
        # Placeholder - in real implementation, copy the file structure extraction logic
        structure_features = self._extract_file_structure_features_impl(repository_path)
        
        # Save to cache
        self._save_to_cache(cache_path, structure_features)
        
        return structure_features
    
    def _extract_file_structure_features_impl(self, repository_path: str) -> Dict[str, Any]:
        """Implementation of file structure feature extraction."""
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
        """Comprehensive repository analysis with caching."""
        print(f"🚀 Starting comprehensive analysis of: {repository_path}")
        print("=" * 60)
        
        if not os.path.exists(repository_path):
            raise ValueError(f"Repository path does not exist: {repository_path}")
        
        # Extract all features (with caching)
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
            'cache_info': {
                'cache_dir': self.cache_dir,
                'repo_hash': self._get_repo_hash(repository_path)
            },
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
        
        # Enhanced code quality metrics using significant dimensions
        complexity_score = 1.0 / (1.0 + features.get('complexity_score', 0))
        structure_score = min(1.0, features.get('avg_path_length', 0) / 10.0)
        organization_score = 1.0 if features.get('has_tests', False) else 0.5
        consistency_score = 1.0 / (1.0 + features.get('embedding_std', 0))
        
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
            'readme_quality': 1.0 if features.get('has_docs', False) else 0.0,
            'code_comments': 0.5,  # Placeholder
            'api_documentation': 0.5,  # Placeholder
            'examples': 0.5  # Placeholder
        }
        
        # Maintainability metrics
        quality_metrics['maintainability'] = {
            'test_coverage': 1.0 if features.get('has_tests', False) else 0.0,
            'code_consistency': consistency_score,
            'dependency_management': 0.5,  # Placeholder
            'build_system': 1.0 if features.get('has_config', False) else 0.0
        }
        
        # Calculate overall score - ensure all values are numeric
        code_quality_values = [float(v) if isinstance(v, (int, float, str)) else 0.0 for v in quality_metrics['code_quality'].values()]
        arch_quality_values = [float(v) if isinstance(v, (int, float, str)) else 0.0 for v in quality_metrics['architecture_quality'].values()]
        doc_quality_values = [float(v) if isinstance(v, (int, float, str)) else 0.0 for v in quality_metrics['documentation_quality'].values()]
        maint_quality_values = [float(v) if isinstance(v, (int, float, str)) else 0.0 for v in quality_metrics['maintainability'].values()]
        
        overall_score = (
            np.mean(code_quality_values) * 0.3 +
            np.mean(arch_quality_values) * 0.3 +
            np.mean(doc_quality_values) * 0.2 +
            np.mean(maint_quality_values) * 0.2
        )
        
        quality_metrics['overall_score'] = overall_score
        
        return quality_metrics
    
    def analyze_programmer_characteristics(self, features: Dict) -> Dict[str, Any]:
        """Analyze programmer characteristics based on code patterns."""
        characteristics = {
            'experience_level': 'intermediate',
            'coding_style': {},
            'attention_to_detail': 0.5,
            'architectural_thinking': 0.5,
            'best_practices': 0.5,
            'specialization': {}
        }
        
        # Experience level assessment
        complexity = features.get('complexity_score', 0)
        structure_quality = features.get('avg_path_length', 0)
        
        if complexity > 100 and structure_quality > 5:
            characteristics['experience_level'] = 'expert'
        elif complexity > 50 or structure_quality > 3:
            characteristics['experience_level'] = 'advanced'
        elif complexity > 20:
            characteristics['experience_level'] = 'intermediate'
        else:
            characteristics['experience_level'] = 'beginner'
        
        # Coding style analysis
        characteristics['coding_style'] = {
            'consistency': 1.0 / (1.0 + features.get('embedding_std', 0)),
            'readability': 1.0 / (1.0 + features.get('nesting_depth', 0)),
            'modularity': min(1.0, features.get('function_count', 0) / 20.0),
            'abstraction': min(1.0, features.get('class_count', 0) / 10.0)
        }
        
        # Attention to detail
        characteristics['attention_to_detail'] = min(1.0, features.get('total_files', 0) / 50.0)
        
        # Architectural thinking
        characteristics['architectural_thinking'] = min(1.0, features.get('class_count', 0) / 15.0)
        
        # Best practices
        has_tests = features.get('has_tests', False)
        has_docs = features.get('has_docs', False)
        has_config = features.get('has_config', False)
        
        best_practices_score = 0.0
        if has_tests: best_practices_score += 0.4
        if has_docs: best_practices_score += 0.3
        if has_config: best_practices_score += 0.3
        
        characteristics['best_practices'] = best_practices_score
        
        # Specialization analysis
        characteristics['specialization'] = self._assess_specialization(features)
        
        return characteristics
    
    def _assess_specialization(self, features: Dict) -> Dict[str, float]:
        """Assess programmer specialization based on keywords and patterns."""
        specialization = {
            'data_science': 0.0,
            'web_development': 0.0,
            'mobile_development': 0.0,
            'game_development': 0.0,
            'cli_tools': 0.0,
            'libraries': 0.0,
            'system_programming': 0.0
        }
        
        # Use keyword features
        ds_keywords = features.get('data_science_keywords', 0)
        web_keywords = features.get('web_development_keywords', 0)
        mobile_keywords = features.get('mobile_development_keywords', 0)
        game_keywords = features.get('game_development_keywords', 0)
        cli_keywords = features.get('cli_tool_keywords', 0)
        lib_keywords = features.get('library_keywords', 0)
        
        # Normalize scores
        total_keywords = max(1, features.get('total_keywords', 1))
        
        specialization['data_science'] = min(1.0, ds_keywords / total_keywords)
        specialization['web_development'] = min(1.0, web_keywords / total_keywords)
        specialization['mobile_development'] = min(1.0, mobile_keywords / total_keywords)
        specialization['game_development'] = min(1.0, game_keywords / total_keywords)
        specialization['cli_tools'] = min(1.0, cli_keywords / total_keywords)
        specialization['libraries'] = min(1.0, lib_keywords / total_keywords)
        
        return specialization
    
    def analyze_architecture_patterns(self, features: Dict) -> Dict[str, Any]:
        """Analyze architectural patterns in the codebase."""
        patterns = {
            'detected_patterns': [],
            'pattern_confidence': {},
            'architectural_style': 'unknown',
            'complexity_level': 'medium'
        }
        
        # Analyze patterns based on features
        function_count = features.get('function_count', 0)
        class_count = features.get('class_count', 0)
        total_files = features.get('total_files', 0)
        nesting_depth = features.get('nesting_depth', 0)
        
        # Pattern detection logic
        if class_count > 10 and function_count > 50:
            patterns['detected_patterns'].append('object_oriented')
            patterns['pattern_confidence']['object_oriented'] = 0.8
        
        if function_count > 100 and class_count < 5:
            patterns['detected_patterns'].append('procedural')
            patterns['pattern_confidence']['procedural'] = 0.7
        
        if total_files > 50 and features.get('has_tests', False):
            patterns['detected_patterns'].append('modular')
            patterns['pattern_confidence']['modular'] = 0.6
        
        # Architectural style
        if class_count > function_count / 10:
            patterns['architectural_style'] = 'object_oriented'
        elif function_count > 100:
            patterns['architectural_style'] = 'procedural'
        else:
            patterns['architectural_style'] = 'script'
        
        # Complexity level
        if nesting_depth > 5 or function_count > 200:
            patterns['complexity_level'] = 'high'
        elif nesting_depth > 3 or function_count > 100:
            patterns['complexity_level'] = 'medium'
        else:
            patterns['complexity_level'] = 'low'
        
        return patterns
    
    def _generate_summary(self, quality: Dict, characteristics: Dict, architecture: Dict) -> Dict[str, str]:
        """Generate a summary of the analysis."""
        summary = {
            'overall_assessment': 'Good',
            'strengths': [],
            'areas_for_improvement': [],
            'recommendations': []
        }
        
        # Overall assessment
        quality_score = quality.get('overall_score', 0)
        if quality_score > 0.8:
            summary['overall_assessment'] = 'Excellent'
        elif quality_score > 0.6:
            summary['overall_assessment'] = 'Good'
        elif quality_score > 0.4:
            summary['overall_assessment'] = 'Fair'
        else:
            summary['overall_assessment'] = 'Needs Improvement'
        
        # Strengths
        if quality.get('code_quality', {}).get('consistency', 0) > 0.7:
            summary['strengths'].append('Consistent coding style')
        
        if characteristics.get('best_practices', 0) > 0.7:
            summary['strengths'].append('Good development practices')
        
        if architecture.get('complexity_level') == 'high':
            summary['strengths'].append('Handles complex requirements')
        
        # Areas for improvement
        if quality.get('documentation_quality', {}).get('readme_quality', 0) < 0.5:
            summary['areas_for_improvement'].append('Documentation needs improvement')
        
        if quality.get('maintainability', {}).get('test_coverage', 0) < 0.5:
            summary['areas_for_improvement'].append('Test coverage could be improved')
        
        # Recommendations
        if not summary['strengths']:
            summary['recommendations'].append('Focus on code quality and consistency')
        
        if summary['areas_for_improvement']:
            summary['recommendations'].append('Address identified improvement areas')
        
        return summary
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear cache files."""
        if cache_type is None:
            # Clear all caches
            for dir_path in [self.embeddings_dir, self.ast_dir, self.keywords_dir, self.structure_dir]:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    os.makedirs(dir_path)
            print(f"🗑️  Cleared all caches in {self.cache_dir}")
        else:
            # Clear specific cache type
            cache_dirs = {
                'embeddings': self.embeddings_dir,
                'ast': self.ast_dir,
                'keywords': self.keywords_dir,
                'structure': self.structure_dir
            }
            
            if cache_type in cache_dirs:
                dir_path = cache_dirs[cache_type]
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    os.makedirs(dir_path)
                print(f"🗑️  Cleared {cache_type} cache")
            else:
                print(f"❌ Invalid cache type: {cache_type}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cached data."""
        stats = {}
        
        for cache_type, dir_path in [
            ('embeddings', self.embeddings_dir),
            ('ast', self.ast_dir),
            ('keywords', self.keywords_dir),
            ('structure', self.structure_dir)
        ]:
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
                stats[cache_type] = len(files)
            else:
                stats[cache_type] = 0
        
        return stats
    
    def initialize_codebert(self):
        """Initialize CodeBERT model and tokenizer."""
        if self.codebert_model is not None:
            return
        
        try:
            print("🤖 Loading CodeBERT model...")
            self.codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
            self.codebert_model.to(self.device)
            self.codebert_model.eval()
            print("✅ CodeBERT model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading CodeBERT: {e}")
            self.codebert_model = None
            self.codebert_tokenizer = None
    
    def _extract_file_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """Extract CodeBERT embedding for a single file."""
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
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Get embeddings
            with torch.no_grad():
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
            
            code_cells = []
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        code_cells.extend(source)
                    else:
                        code_cells.append(source)
            
            return '\n'.join(code_cells)
        except Exception as e:
            print(f"⚠️  Error extracting Python from notebook {notebook_path}: {e}")
            return ""
    
    def _clean_code(self, code: str) -> str:
        """Clean and preprocess code content."""
        # Remove comments and docstrings (simplified)
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip comment lines
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _calculate_semantic_features(self, embeddings: List[np.ndarray]) -> Dict[str, float]:
        """Calculate semantic features from embeddings."""
        if not embeddings:
            return {}
        
        embeddings_array = np.array(embeddings)
        
        # Calculate semantic diversity (standard deviation of embeddings)
        semantic_diversity = np.std(embeddings_array).item()
        
        # Calculate semantic coherence (average cosine similarity)
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                similarities.append(similarity)
        
        semantic_coherence = np.mean(similarities) if similarities else 0.0
        
        return {
            'semantic_diversity': semantic_diversity,
            'semantic_coherence': semantic_coherence
        }
    
    def _extract_significant_dimensions(self, repo_embedding: np.ndarray) -> Dict[str, float]:
        """Extract significant dimensions for enhanced analysis."""
        significant_features = {}
        
        # Extract dimensions for different category comparisons
        for pair_name, dimensions in self.significant_dimensions.items():
            for dim_name, dim_idx in dimensions.items():
                if dim_idx < len(repo_embedding):
                    feature_name = f"{pair_name}_{dim_name}"
                    significant_features[feature_name] = float(repo_embedding[dim_idx])
        
        return significant_features
    
    def load_significant_dimensions(self) -> Dict[str, Dict[str, int]]:
        """Load significant dimensions from v2 analysis."""
        # This would load from a file, but for now return some example dimensions
        return {
            'cli_tool_vs_data_science': {
                'dim_448': 448,
                'dim_720': 720,
                'dim_644': 644,
                'dim_588': 588
            },
            'web_application_vs_library': {
                'dim_588': 588,
                'dim_498': 498,
                'dim_720': 720,
                'dim_77': 77
            },
            'data_science_vs_web_application': {
                'dim_644': 644,
                'dim_588': 588,
                'dim_498': 498,
                'dim_720': 720
            }
        }

