"""
Comprehensive Repository Analyzer v2.0
=====================================

Multi-stage analysis pipeline:
1. AST + File Structure + CodeBERT → Architectural Pattern Analysis
2. Keywords + CodeBERT Dimensions → Category Type Prediction  
3. Combined Analysis → Quality Assessment & Programmer Characteristics
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import pickle
import os

class ComprehensiveRepositoryAnalyzer:
    def __init__(self):
        self.architectural_classifier = None
        self.category_classifier = None
        self.quality_regressor = None
        self.programmer_analyzer = None
        
        # Load significant dimensions from our analysis
        self.significant_dimensions = self.load_significant_dimensions()
        
        # Initialize scalers
        self.arch_scaler = StandardScaler()
        self.cat_scaler = StandardScaler()
        self.quality_scaler = StandardScaler()
        
    def load_significant_dimensions(self) -> Dict[str, List[int]]:
        """Load significant dimensions for each category pair from our analysis."""
        return {
            'cli_tool_vs_data_science': [448, 720, 644, 588, 540, 97, 39, 34, 461, 657],
            'web_application_vs_library': [588, 498, 720, 77, 688, 363, 270, 155, 608, 670],
            'game_development_vs_mobile_app': [588, 85, 700, 82, 629, 77, 490, 528, 551, 354],
            'data_science_vs_educational': [574, 211, 454, 422, 485, 581, 144, 301, 35, 738],
            'cli_tool_vs_educational': [211, 608, 738, 733, 686, 190, 461, 71, 485, 39]
        }
    
    def extract_ast_features(self, repository_path: str) -> Dict[str, Any]:
        """Extract AST-based structural features."""
        ast_features = {}
        
        # Load AST features from our previous extraction
        ast_file = Path(f"scripts/extraction/ASTFeaturesForAnalysis/{Path(repository_path).name}_ast_features.json")
        
        if ast_file.exists():
            with open(ast_file, 'r') as f:
                ast_data = json.load(f)
                
            # Extract structural metrics (ensure all are numeric)
            ast_features = {
                'avg_path_length': float(ast_data.get('avg_path_length', 0)),
                'max_path_length': float(ast_data.get('max_path_length', 0)),
                'path_variety': float(ast_data.get('path_variety', 0)),
                'node_type_diversity': float(ast_data.get('node_type_diversity', 0)),
                'complexity_score': float(ast_data.get('complexity_score', 0)),
                'nesting_depth': float(ast_data.get('nesting_depth', 0)),
                'function_count': float(ast_data.get('function_count', 0)),
                'class_count': float(ast_data.get('class_count', 0)),
                'interface_count': float(ast_data.get('interface_count', 0)),
                'inheritance_depth': float(ast_data.get('inheritance_depth', 0))
            }
        else:
            # Provide default values if AST file doesn't exist
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
                'inheritance_depth': 0.0
            }
        
        return ast_features
    
    def extract_file_structure_features(self, repository_path: str) -> Dict[str, Any]:
        """Extract file organization and structure features."""
        repo_path = Path(repository_path)
        structure_features = {}
        
        if repo_path.exists():
            # File organization patterns
            files = list(repo_path.rglob('*'))
            directories = [f for f in files if f.is_dir()]
            source_files = [f for f in files if f.is_file() and f.suffix in ['.py', '.js', '.java', '.cpp', '.c', '.h']]
            
            structure_features = {
                'total_files': len(files),
                'total_directories': len(directories),
                'source_files': len(source_files),
                'avg_files_per_dir': len(files) / max(len(directories), 1),
                'max_directory_depth': max([len(f.parts) - len(repo_path.parts) for f in files]),
                'has_tests': 1.0 if any('test' in f.name.lower() for f in files) else 0.0,
                'has_docs': 1.0 if any('doc' in f.name.lower() or 'readme' in f.name.lower() for f in files) else 0.0,
                'has_config': 1.0 if any('config' in f.name.lower() or '.json' in f.name or '.yaml' in f.name for f in files) else 0.0,
                'has_dependencies': 1.0 if any('requirements' in f.name.lower() or 'package.json' in f.name or 'pom.xml' in f.name for f in files) else 0.0
            }
            
            # Language distribution
            extensions = [f.suffix for f in source_files]
            structure_features['language_diversity'] = len(set(extensions))
            # Convert main_language to numeric (hash-based)
            main_lang = max(set(extensions), key=extensions.count) if extensions else ''
            structure_features['main_language_hash'] = hash(main_lang) % 1000  # Convert to numeric
        
        return structure_features
    
    def extract_codebert_architectural_features(self, repository_path: str) -> Dict[str, Any]:
        """Extract CodeBERT embeddings for architectural pattern analysis."""
        repo_name = Path(repository_path).name
        
        # Load CodeBERT embeddings
        embedding_file = Path(f"scripts/extraction/CodeBERTEmbeddings/{repo_name}_embeddings.json")
        
        if embedding_file.exists():
            with open(embedding_file, 'r') as f:
                embedding_data = json.load(f)
            
            # Use full embedding for architectural analysis
            full_embedding = np.array(embedding_data['repository_embedding'])
            
            # Extract architectural-relevant features
            arch_features = {
                'embedding_mean': np.mean(full_embedding),
                'embedding_std': np.std(full_embedding),
                'embedding_max': np.max(full_embedding),
                'embedding_min': np.min(full_embedding),
                'embedding_range': np.max(full_embedding) - np.min(full_embedding),
                'embedding_skewness': self.calculate_skewness(full_embedding),
                'embedding_kurtosis': self.calculate_kurtosis(full_embedding)
            }
            
            # Add specific architectural dimensions (based on our analysis)
            arch_dimensions = [588, 720, 77, 363, 270, 155, 608, 670]  # Architecture-relevant
            for i, dim in enumerate(arch_dimensions):
                arch_features[f'arch_dim_{dim}'] = full_embedding[dim]
            
            return arch_features
        
        return {}
    
    def extract_keyword_features(self, repository_path: str) -> Dict[str, Any]:
        """Extract keyword-based features for category prediction."""
        # Import our enhanced keyword analyzer
        import sys
        sys.path.append('.')
        
        try:
            from keywords import analyze_repository_keywords
            keyword_features = analyze_repository_keywords(repository_path)
            return keyword_features
        except ImportError:
            # Fallback to basic keyword extraction
            return self.basic_keyword_extraction(repository_path)
    
    def extract_codebert_category_features(self, repository_path: str) -> Dict[str, Any]:
        """Extract CodeBERT dimensions relevant for category classification."""
        repo_name = Path(repository_path).name
        
        # Load CodeBERT embeddings
        embedding_file = Path(f"scripts/extraction/CodeBERTEmbeddings/{repo_name}_embeddings.json")
        
        if embedding_file.exists():
            with open(embedding_file, 'r') as f:
                embedding_data = json.load(f)
            
            full_embedding = np.array(embedding_data['repository_embedding'])
            
            # Extract only significant dimensions for category classification
            category_features = {}
            
            # Combine significant dimensions from all category pairs
            all_significant_dims = set()
            for dims in self.significant_dimensions.values():
                all_significant_dims.update(dims)
            
            # Use top significant dimensions
            top_dims = sorted(list(all_significant_dims))[:50]  # Top 50 dimensions
            
            for i, dim in enumerate(top_dims):
                category_features[f'cat_dim_{dim}'] = full_embedding[dim]
            
            return category_features
        
        return {}
    
    def calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def basic_keyword_extraction(self, repository_path: str) -> Dict[str, Any]:
        """Basic keyword extraction as fallback."""
        # This would be a simplified version of our keywords.py functionality
        return {
            'framework_keywords': 0.0,
            'library_keywords': 0.0,
            'data_science_keywords': 0.0,
            'web_keywords': 0.0,
            'cli_keywords': 0.0,
            'game_keywords': 0.0,
            'mobile_keywords': 0.0
        }
    
    def extract_all_features(self, repository_path: str) -> Dict[str, Dict[str, Any]]:
        """Extract all features for comprehensive analysis."""
        features = {
            'architectural': {},
            'category': {}
        }
        
        # Architectural features (AST + File Structure + CodeBERT)
        ast_features = self.extract_ast_features(repository_path)
        file_features = self.extract_file_structure_features(repository_path)
        codebert_arch_features = self.extract_codebert_architectural_features(repository_path)
        
        features['architectural'] = {**ast_features, **file_features, **codebert_arch_features}
        
        # Category features (Keywords + CodeBERT dimensions)
        keyword_features = self.extract_keyword_features(repository_path)
        codebert_cat_features = self.extract_codebert_category_features(repository_path)
        
        features['category'] = {**keyword_features, **codebert_cat_features}
        
        return features
    
    def train_architectural_classifier(self, training_data: List[Tuple[Dict, str]]):
        """Train architectural pattern classifier."""
        X = []
        y = []
        
        for features, pattern in training_data:
            arch_features = features['architectural']
            X.append(list(arch_features.values()))
            y.append(pattern)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.arch_scaler.fit_transform(X)
        
        # Train classifier
        self.architectural_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        self.architectural_classifier.fit(X_scaled, y)
    
    def train_category_classifier(self, training_data: List[Tuple[Dict, str]]):
        """Train category type classifier."""
        X = []
        y = []
        
        for features, category in training_data:
            cat_features = features['category']
            X.append(list(cat_features.values()))
            y.append(category)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.cat_scaler.fit_transform(X)
        
        # Train classifier
        self.category_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        self.category_classifier.fit(X_scaled, y)
    
    def train_quality_assessor(self, training_data: List[Tuple[Dict, float]]):
        """Train quality assessment model."""
        X = []
        y = []
        
        for features, quality_score in training_data:
            # Combine architectural and category features for quality assessment
            combined_features = {**features['architectural'], **features['category']}
            X.append(list(combined_features.values()))
            y.append(quality_score)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.quality_scaler.fit_transform(X)
        
        # Train regressor
        self.quality_regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        self.quality_regressor.fit(X_scaled, y)
    
    def analyze_repository(self, repository_path: str) -> Dict[str, Any]:
        """Comprehensive repository analysis."""
        print(f"🔍 Analyzing repository: {repository_path}")
        
        # Extract all features
        features = self.extract_all_features(repository_path)
        
        results = {
            'repository_path': repository_path,
            'features': features,
            'predictions': {},
            'quality_assessment': {},
            'programmer_characteristics': {}
        }
        
        # 1. Architectural Pattern Analysis
        if self.architectural_classifier:
            arch_features = list(features['architectural'].values())
            arch_features_scaled = self.arch_scaler.transform([arch_features])
            arch_prediction = self.architectural_classifier.predict(arch_features_scaled)[0]
            arch_confidence = np.max(self.architectural_classifier.predict_proba(arch_features_scaled))
            
            results['predictions']['architectural_pattern'] = {
                'pattern': arch_prediction,
                'confidence': arch_confidence
            }
        
        # 2. Category Type Prediction
        if self.category_classifier:
            cat_features = list(features['category'].values())
            cat_features_scaled = self.cat_scaler.transform([cat_features])
            cat_prediction = self.category_classifier.predict(cat_features_scaled)[0]
            cat_confidence = np.max(self.category_classifier.predict_proba(cat_features_scaled))
            
            results['predictions']['category_type'] = {
                'category': cat_prediction,
                'confidence': cat_confidence
            }
        
        # 3. Quality Assessment
        if self.quality_regressor:
            combined_features = list(features['architectural'].values()) + list(features['category'].values())
            combined_features_scaled = self.quality_scaler.transform([combined_features])
            quality_score = self.quality_regressor.predict(combined_features_scaled)[0]
            
            results['quality_assessment'] = {
                'overall_score': quality_score,
                'code_quality': self.assess_code_quality(features),
                'architecture_quality': self.assess_architecture_quality(features),
                'documentation_quality': self.assess_documentation_quality(features),
                'maintainability': self.assess_maintainability(features)
            }
        
        # 4. Programmer Characteristics
        results['programmer_characteristics'] = self.analyze_programmer_characteristics(features, results)
        
        return results
    
    def assess_code_quality(self, features: Dict) -> Dict[str, float]:
        """Assess code quality based on features."""
        arch_features = features['architectural']
        
        quality_metrics = {
            'complexity': 1.0 / (1.0 + arch_features.get('complexity_score', 0)),
            'structure': min(1.0, arch_features.get('avg_path_length', 0) / 10.0),
            'organization': 1.0 if arch_features.get('has_tests', False) else 0.5,
            'consistency': 1.0 / (1.0 + arch_features.get('embedding_std', 0))
        }
        
        return quality_metrics
    
    def assess_architecture_quality(self, features: Dict) -> Dict[str, float]:
        """Assess architecture quality."""
        arch_features = features['architectural']
        
        quality_metrics = {
            'modularity': min(1.0, arch_features.get('function_count', 0) / 50.0),
            'abstraction': min(1.0, arch_features.get('class_count', 0) / 20.0),
            'separation': 1.0 / (1.0 + arch_features.get('nesting_depth', 0)),
            'scalability': min(1.0, arch_features.get('total_files', 0) / 100.0)
        }
        
        return quality_metrics
    
    def assess_documentation_quality(self, features: Dict) -> Dict[str, float]:
        """Assess documentation quality."""
        arch_features = features['architectural']
        
        quality_metrics = {
            'readme_presence': 1.0 if arch_features.get('has_docs', False) else 0.0,
            'config_documentation': 1.0 if arch_features.get('has_config', False) else 0.0,
            'dependency_documentation': 1.0 if arch_features.get('has_dependencies', False) else 0.0
        }
        
        return quality_metrics
    
    def assess_maintainability(self, features: Dict) -> Dict[str, float]:
        """Assess maintainability."""
        arch_features = features['architectural']
        
        quality_metrics = {
            'test_coverage': 1.0 if arch_features.get('has_tests', False) else 0.0,
            'code_organization': min(1.0, arch_features.get('language_diversity', 0) / 5.0),
            'structure_clarity': 1.0 / (1.0 + arch_features.get('max_directory_depth', 0)),
            'consistency': 1.0 / (1.0 + arch_features.get('embedding_std', 0))
        }
        
        return quality_metrics
    
    def analyze_programmer_characteristics(self, features: Dict, results: Dict) -> Dict[str, Any]:
        """Analyze programmer characteristics based on code patterns."""
        arch_features = features['architectural']
        predictions = results.get('predictions', {})
        
        characteristics = {
            'experience_level': self.assess_experience_level(features),
            'coding_style': self.assess_coding_style(features),
            'attention_to_detail': self.assess_attention_to_detail(features),
            'architectural_thinking': self.assess_architectural_thinking(features),
            'best_practices': self.assess_best_practices(features),
            'specialization': self.assess_specialization(predictions)
        }
        
        return characteristics
    
    def assess_experience_level(self, features: Dict) -> str:
        """Assess programmer experience level."""
        arch_features = features['architectural']
        
        # Calculate experience score
        experience_score = 0
        
        # Code complexity and structure
        if arch_features.get('complexity_score', 0) > 5:
            experience_score += 2
        if arch_features.get('avg_path_length', 0) > 8:
            experience_score += 1
        
        # Project organization
        if arch_features.get('has_tests', False):
            experience_score += 2
        if arch_features.get('has_docs', False):
            experience_score += 1
        if arch_features.get('has_config', False):
            experience_score += 1
        
        # Code structure
        if arch_features.get('function_count', 0) > 20:
            experience_score += 1
        if arch_features.get('class_count', 0) > 5:
            experience_score += 1
        
        if experience_score >= 6:
            return "Senior"
        elif experience_score >= 3:
            return "Intermediate"
        else:
            return "Junior"
    
    def assess_coding_style(self, features: Dict) -> str:
        """Assess coding style."""
        arch_features = features['architectural']
        
        # Analyze coding patterns
        if arch_features.get('has_tests', False) and arch_features.get('has_docs', False):
            return "Professional"
        elif arch_features.get('complexity_score', 0) < 3:
            return "Simple and Clean"
        elif arch_features.get('function_count', 0) > 30:
            return "Comprehensive"
        else:
            return "Basic"
    
    def assess_attention_to_detail(self, features: Dict) -> str:
        """Assess attention to detail."""
        arch_features = features['architectural']
        
        detail_score = 0
        
        if arch_features.get('has_tests', False):
            detail_score += 2
        if arch_features.get('has_docs', False):
            detail_score += 1
        if arch_features.get('has_config', False):
            detail_score += 1
        if arch_features.get('has_dependencies', False):
            detail_score += 1
        
        if detail_score >= 4:
            return "High"
        elif detail_score >= 2:
            return "Medium"
        else:
            return "Low"
    
    def assess_architectural_thinking(self, features: Dict) -> str:
        """Assess architectural thinking."""
        arch_features = features['architectural']
        
        arch_score = 0
        
        if arch_features.get('class_count', 0) > 5:
            arch_score += 2
        if arch_features.get('interface_count', 0) > 0:
            arch_score += 2
        if arch_features.get('inheritance_depth', 0) > 2:
            arch_score += 1
        if arch_features.get('max_directory_depth', 0) > 3:
            arch_score += 1
        
        if arch_score >= 4:
            return "Strong"
        elif arch_score >= 2:
            return "Moderate"
        else:
            return "Basic"
    
    def assess_best_practices(self, features: Dict) -> str:
        """Assess adherence to best practices."""
        arch_features = features['architectural']
        
        practice_score = 0
        
        if arch_features.get('has_tests', False):
            practice_score += 2
        if arch_features.get('has_docs', False):
            practice_score += 1
        if arch_features.get('has_config', False):
            practice_score += 1
        if arch_features.get('nesting_depth', 0) < 5:
            practice_score += 1
        if arch_features.get('complexity_score', 0) < 10:
            practice_score += 1
        
        if practice_score >= 4:
            return "Excellent"
        elif practice_score >= 2:
            return "Good"
        else:
            return "Needs Improvement"
    
    def assess_specialization(self, predictions: Dict) -> str:
        """Assess programmer specialization."""
        if 'category_type' in predictions:
            category = predictions['category_type']['category']
            confidence = predictions['category_type']['confidence']
            
            if confidence > 0.8:
                return f"Specialized in {category}"
            elif confidence > 0.6:
                return f"Focused on {category}"
            else:
                return "Generalist"
        
        return "Unknown"
    
    def save_models(self, output_dir: str = "ml_models"):
        """Save trained models."""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.architectural_classifier:
            with open(f"{output_dir}/architectural_classifier.pkl", 'wb') as f:
                pickle.dump(self.architectural_classifier, f)
        
        if self.category_classifier:
            with open(f"{output_dir}/category_classifier.pkl", 'wb') as f:
                pickle.dump(self.category_classifier, f)
        
        if self.quality_regressor:
            with open(f"{output_dir}/quality_regressor.pkl", 'wb') as f:
                pickle.dump(self.quality_regressor, f)
        
        # Save scalers
        with open(f"{output_dir}/arch_scaler.pkl", 'wb') as f:
            pickle.dump(self.arch_scaler, f)
        
        with open(f"{output_dir}/cat_scaler.pkl", 'wb') as f:
            pickle.dump(self.cat_scaler, f)
        
        with open(f"{output_dir}/quality_scaler.pkl", 'wb') as f:
            pickle.dump(self.quality_scaler, f)
    
    def load_models(self, model_dir: str = "ml_models"):
        """Load trained models."""
        if os.path.exists(f"{model_dir}/architectural_classifier.pkl"):
            with open(f"{model_dir}/architectural_classifier.pkl", 'rb') as f:
                self.architectural_classifier = pickle.load(f)
        
        if os.path.exists(f"{model_dir}/category_classifier.pkl"):
            with open(f"{model_dir}/category_classifier.pkl", 'rb') as f:
                self.category_classifier = pickle.load(f)
        
        if os.path.exists(f"{model_dir}/quality_regressor.pkl"):
            with open(f"{model_dir}/quality_regressor.pkl", 'rb') as f:
                self.quality_regressor = pickle.load(f)
        
        # Load scalers
        if os.path.exists(f"{model_dir}/arch_scaler.pkl"):
            with open(f"{model_dir}/arch_scaler.pkl", 'rb') as f:
                self.arch_scaler = pickle.load(f)
        
        if os.path.exists(f"{model_dir}/cat_scaler.pkl"):
            with open(f"{model_dir}/cat_scaler.pkl", 'rb') as f:
                self.cat_scaler = pickle.load(f)
        
        if os.path.exists(f"{model_dir}/quality_scaler.pkl"):
            with open(f"{model_dir}/quality_scaler.pkl", 'rb') as f:
                self.quality_scaler = pickle.load(f)

def main():
    """Main function to demonstrate the comprehensive analyzer."""
    
    print("🚀 Comprehensive Repository Analyzer v2.0")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ComprehensiveRepositoryAnalyzer()
    
    # Try to load pre-trained models
    analyzer.load_models()
    
    # Example analysis
    if analyzer.architectural_classifier and analyzer.category_classifier:
        print("✅ Models loaded successfully!")
        
        # Analyze a sample repository
        sample_repo = "dataset/cli_tool/sindresorhus_chalk"
        
        if os.path.exists(sample_repo):
            results = analyzer.analyze_repository(sample_repo)
            
            print(f"\n📊 Analysis Results for {sample_repo}:")
            print("-" * 50)
            
            # Print predictions
            if 'predictions' in results:
                for pred_type, pred_data in results['predictions'].items():
                    print(f"{pred_type.replace('_', ' ').title()}: {pred_data}")
            
            # Print quality assessment
            if 'quality_assessment' in results:
                print(f"\n🎯 Quality Assessment:")
                for metric, score in results['quality_assessment'].items():
                    if isinstance(score, dict):
                        print(f"  {metric}:")
                        for sub_metric, sub_score in score.items():
                            print(f"    {sub_metric}: {sub_score:.3f}")
                    else:
                        print(f"  {metric}: {score:.3f}")
            
            # Print programmer characteristics
            if 'programmer_characteristics' in results:
                print(f"\n👨‍💻 Programmer Characteristics:")
                for char, value in results['programmer_characteristics'].items():
                    print(f"  {char.replace('_', ' ').title()}: {value}")
        
        else:
            print(f"❌ Sample repository not found: {sample_repo}")
    
    else:
        print("❌ No pre-trained models found. Please train models first.")
        print("💡 You can train models using the training functions in this class.")

if __name__ == "__main__":
    main()
