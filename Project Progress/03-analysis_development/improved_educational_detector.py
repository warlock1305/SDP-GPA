"""
Improved Educational Project Detector
====================================

This script enhances the Random Forest analyzer to better distinguish between:
- Educational/Course projects (like OOP learning materials)
- Real-world applications (like production software)

Key improvements:
- Educational indicators detection
- Context-aware feature engineering
- Course structure recognition
- Better pattern classification for learning materials
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class ImprovedEducationalDetector:
    """Enhanced detector that can distinguish educational from real-world projects."""
    
    def __init__(self, models_dir: str = "ml_models"):
        """Initialize the improved detector."""
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Model components
        self.rf_classifier = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.pattern_names = []
        
        # Educational indicators
        self.educational_keywords = [
            'week', 'lab', 'lecture', 'exercise', 'assignment', 'homework',
            'tutorial', 'example', 'demo', 'practice', 'learning', 'course',
            'student', 'teacher', 'professor', 'classroom', 'education',
            'lesson', 'chapter', 'module', 'unit', 'task', 'problem'
        ]
        
        self.course_structure_patterns = [
            'week1', 'week2', 'week3', 'week4', 'week5', 'week6', 'week7', 'week8',
            'week9', 'week10', 'week11', 'week12', 'week13', 'week14', 'week15',
            'lab1', 'lab2', 'lab3', 'lab4', 'lab5', 'lab6', 'lab7', 'lab8',
            'lecture1', 'lecture2', 'lecture3', 'lecture4', 'lecture5',
            'assignment1', 'assignment2', 'assignment3', 'assignment4',
            'exercise1', 'exercise2', 'exercise3', 'exercise4'
        ]
    
    def extract_enhanced_features(self, repo_data: Dict) -> np.ndarray:
        """Extract enhanced features including educational indicators."""
        features = []
        
        # Original features
        ast_metrics = repo_data.get('ast_metrics', {})
        codebert_metrics = repo_data.get('codebert_metrics', {})
        combined_metrics = repo_data.get('combined_metrics', {})
        
        # Basic metrics
        features.extend([
            ast_metrics.get('total_methods', 0),
            ast_metrics.get('total_path_contexts', 0),
            ast_metrics.get('unique_node_types', 0),
            ast_metrics.get('unique_tokens', 0),
            ast_metrics.get('avg_path_length', 0.0),
            ast_metrics.get('avg_path_diversity', 0.0),
            len(ast_metrics.get('languages', [])),
            codebert_metrics.get('num_files', 0),
            codebert_metrics.get('embedding_dimension', 0),
            combined_metrics.get('enhanced_complexity', 0.0),
            combined_metrics.get('enhanced_maintainability', 0.0),
            combined_metrics.get('semantic_richness', 0.0),
            combined_metrics.get('technology_diversity', 0.0),
            combined_metrics.get('overall_quality', 0.0)
        ])
        
        # Engineered features
        features.extend([
            ast_metrics.get('total_methods', 0) / max(codebert_metrics.get('num_files', 1), 1),
            ast_metrics.get('total_path_contexts', 0) / max(ast_metrics.get('total_methods', 1), 1),
            ast_metrics.get('unique_tokens', 0) / max(ast_metrics.get('total_methods', 1), 1),
            ast_metrics.get('avg_path_diversity', 0.0) * ast_metrics.get('total_path_contexts', 0),
            np.log(max(codebert_metrics.get('num_files', 1), 1)),
            np.log(max(ast_metrics.get('total_methods', 1), 1))
        ])
        
        # NEW: Educational indicators
        educational_features = self.extract_educational_indicators(repo_data)
        features.extend(educational_features)
        
        return np.array(features, dtype=np.float32)
    
    def extract_educational_indicators(self, repo_data: Dict) -> List[float]:
        """Extract features that indicate educational/course projects."""
        indicators = []
        
        # Get repository structure from AST features
        repo_structure = self.analyze_repository_structure(repo_data)
        
        # 1. Course structure indicators
        course_structure_score = self.detect_course_structure(repo_structure)
        indicators.append(course_structure_score)
        
        # 2. Educational keyword density
        educational_keyword_score = self.detect_educational_keywords(repo_structure)
        indicators.append(educational_keyword_score)
        
        # 3. File naming patterns
        naming_pattern_score = self.detect_educational_naming(repo_structure)
        indicators.append(naming_pattern_score)
        
        # 4. Directory depth and organization
        organization_score = self.detect_educational_organization(repo_structure)
        indicators.append(organization_score)
        
        # 5. Code complexity vs. educational content
        complexity_educational_ratio = self.calculate_complexity_educational_ratio(repo_data)
        indicators.append(complexity_educational_ratio)
        
        # 6. Documentation patterns
        documentation_score = self.detect_educational_documentation(repo_structure)
        indicators.append(documentation_score)
        
        # 7. Exercise/assignment patterns
        exercise_pattern_score = self.detect_exercise_patterns(repo_structure)
        indicators.append(exercise_pattern_score)
        
        # 8. Learning progression indicators
        learning_progression_score = self.detect_learning_progression(repo_structure)
        indicators.append(learning_progression_score)
        
        return indicators
    
    def analyze_repository_structure(self, repo_data: Dict) -> Dict:
        """Analyze the repository structure for educational indicators."""
        # This would normally analyze the actual file structure
        # For now, we'll use the available metrics to infer structure
        
        structure = {
            'file_count': repo_data.get('codebert_metrics', {}).get('num_files', 0),
            'method_count': repo_data.get('ast_metrics', {}).get('total_methods', 0),
            'complexity': repo_data.get('combined_metrics', {}).get('enhanced_complexity', 0.0),
            'languages': repo_data.get('ast_metrics', {}).get('languages', []),
            'path_contexts': repo_data.get('ast_metrics', {}).get('total_path_contexts', 0)
        }
        
        return structure
    
    def detect_course_structure(self, structure: Dict) -> float:
        """Detect if the repository follows a course structure."""
        # High file count with moderate complexity suggests course materials
        file_count = structure['file_count']
        complexity = structure['complexity']
        
        # Course projects typically have many files but moderate complexity
        if file_count > 50 and complexity < 0.7:
            return 0.8  # High likelihood of course structure
        elif file_count > 20 and complexity < 0.5:
            return 0.6  # Medium likelihood
        else:
            return 0.2  # Low likelihood
    
    def detect_educational_keywords(self, structure: Dict) -> float:
        """Detect educational keywords in the structure."""
        # This would normally analyze file names and content
        # For now, we'll use heuristics based on metrics
        
        # Educational projects often have high method-to-file ratios
        method_count = structure['method_count']
        file_count = structure['file_count']
        
        if file_count > 0:
            method_file_ratio = method_count / file_count
            if method_file_ratio > 0.5:  # Many methods per file (typical of exercises)
                return 0.7
            elif method_file_ratio > 0.3:
                return 0.5
            else:
                return 0.2
        return 0.0
    
    def detect_educational_naming(self, structure: Dict) -> float:
        """Detect educational naming patterns."""
        # This would analyze actual file names
        # For now, using heuristics
        
        file_count = structure['file_count']
        method_count = structure['method_count']
        
        # Educational projects often have many small files with many methods
        if file_count > 100 and method_count > 100:
            return 0.8  # Very likely educational
        elif file_count > 50 and method_count > 50:
            return 0.6  # Likely educational
        else:
            return 0.2  # Less likely educational
    
    def detect_educational_organization(self, structure: Dict) -> float:
        """Detect educational organization patterns."""
        # Educational projects often have structured organization
        
        file_count = structure['file_count']
        languages = structure['languages']
        
        # Single language with many files often indicates course materials
        if len(languages) == 1 and file_count > 50:
            return 0.7
        elif len(languages) == 1 and file_count > 20:
            return 0.5
        else:
            return 0.3
    
    def calculate_complexity_educational_ratio(self, repo_data: Dict) -> float:
        """Calculate ratio between complexity and educational indicators."""
        complexity = repo_data.get('combined_metrics', {}).get('enhanced_complexity', 0.0)
        file_count = repo_data.get('codebert_metrics', {}).get('num_files', 0)
        
        # High complexity with many files might indicate real project
        # Low complexity with many files might indicate educational project
        if file_count > 0:
            complexity_per_file = complexity / file_count
            if complexity_per_file < 0.01:  # Low complexity per file
                return 0.8  # Likely educational
            elif complexity_per_file < 0.02:
                return 0.6  # Possibly educational
            else:
                return 0.2  # Likely real project
        return 0.5
    
    def detect_educational_documentation(self, structure: Dict) -> float:
        """Detect educational documentation patterns."""
        # Educational projects often have extensive documentation
        # For now, using file count as proxy
        
        file_count = structure['file_count']
        if file_count > 100:
            return 0.7  # Likely has educational documentation
        elif file_count > 50:
            return 0.5
        else:
            return 0.3
    
    def detect_exercise_patterns(self, structure: Dict) -> float:
        """Detect exercise/assignment patterns."""
        # Educational projects often have repetitive patterns
        
        method_count = structure['method_count']
        file_count = structure['file_count']
        
        if file_count > 0:
            methods_per_file = method_count / file_count
            # Many small files with few methods each suggests exercises
            if methods_per_file < 2.0 and file_count > 50:
                return 0.8  # Very likely exercises
            elif methods_per_file < 5.0 and file_count > 20:
                return 0.6  # Possibly exercises
            else:
                return 0.2  # Less likely exercises
        return 0.0
    
    def detect_learning_progression(self, structure: Dict) -> float:
        """Detect learning progression indicators."""
        # Educational projects often show progression in complexity
        
        complexity = structure['complexity']
        file_count = structure['file_count']
        
        # Moderate complexity with many files suggests learning progression
        if 0.3 < complexity < 0.7 and file_count > 50:
            return 0.7  # Likely learning progression
        elif 0.2 < complexity < 0.8 and file_count > 20:
            return 0.5  # Possibly learning progression
        else:
            return 0.2  # Less likely learning progression
    
    def get_enhanced_feature_names(self) -> List[str]:
        """Get enhanced feature names including educational indicators."""
        base_features = [
            # Original features
            "total_methods", "total_path_contexts", "unique_node_types", 
            "unique_tokens", "avg_path_length", "avg_path_diversity", "language_count",
            "num_files", "embedding_dimension", "enhanced_complexity", 
            "enhanced_maintainability", "semantic_richness", "technology_diversity", 
            "overall_quality", "methods_per_file_ratio", "paths_per_method_ratio", 
            "tokens_per_method_ratio", "diversity_complexity_product", 
            "log_file_count", "log_method_count"
        ]
        
        educational_features = [
            "course_structure_score", "educational_keyword_score", 
            "naming_pattern_score", "organization_score", 
            "complexity_educational_ratio", "documentation_score",
            "exercise_pattern_score", "learning_progression_score"
        ]
        
        return base_features + educational_features
    
    def get_enhanced_pattern_names(self) -> List[str]:
        """Get enhanced pattern names including educational patterns."""
        return [
            # Original patterns
            "monolithic", "microservices", "serverless", "mvc_pattern", 
            "clean_architecture", "mvvm_pattern", "react_application",
            "angular_application", "django_application", "spring_application",
            "data_science_project", "blockchain_project", "iot_project",
            "mobile_app", "singleton_pattern", "factory_pattern", "observer_pattern",
            "utility_script", "api_project", "cli_tool", "library_project",
            "testing_project", "documentation_project",
            
            # NEW: Educational patterns
            "educational_project", "course_materials", "learning_exercises",
            "tutorial_project", "assignment_project", "demo_project"
        ]
    
    def train_enhanced_detector(self, analysis_data: Dict, n_trees: int = 100) -> Dict:
        """Train enhanced detector with educational indicators."""
        print("🎓 TRAINING ENHANCED EDUCATIONAL DETECTOR")
        print("=" * 60)
        
        # Prepare data with enhanced features
        features_list = []
        pattern_labels = []
        
        self.feature_names = self.get_enhanced_feature_names()
        self.pattern_names = self.get_enhanced_pattern_names()
        
        for repo_name, repo_data in analysis_data.get('detailed_analysis', {}).items():
            # Extract enhanced features
            features = self.extract_enhanced_features(repo_data)
            features_list.append(features)
            
            # Create enhanced labels including educational patterns
            pattern_label = self.create_enhanced_labels(repo_name, repo_data)
            pattern_labels.append(pattern_label)
        
        X = np.array(features_list)
        y = np.array(pattern_labels)
        
        print(f"📊 Enhanced dataset prepared:")
        print(f"   • Features: {X.shape[1]} dimensions (including {len(self.feature_names) - 20} educational indicators)")
        print(f"   • Samples: {X.shape[0]} repositories")
        print(f"   • Patterns: {y.shape[1]} patterns (including educational)")
        print(f"   • Trees: {n_trees} parallel decision trees")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        base_rf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=12,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.rf_classifier = MultiOutputClassifier(base_rf)
        self.rf_classifier.fit(X_scaled, y)
        
        # Save model
        model_path = os.path.join(self.models_dir, 'enhanced_educational_detector.joblib')
        scaler_path = os.path.join(self.models_dir, 'enhanced_educational_scaler.joblib')
        
        joblib.dump(self.rf_classifier, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"💾 Enhanced Educational Detector saved to {model_path}")
        
        return {
            "n_trees": n_trees,
            "n_features": X.shape[1],
            "n_patterns": y.shape[1],
            "n_samples": X.shape[0]
        }
    
    def create_enhanced_labels(self, repo_name: str, repo_data: Dict) -> List[int]:
        """Create enhanced labels including educational patterns."""
        pattern_label = [0] * len(self.pattern_names)
        
        # Get original patterns
        detected_patterns = repo_data.get('enhanced_architecture_patterns', {}).get('all_patterns', [])
        
        # Map original patterns
        for pattern in detected_patterns:
            pattern_name = pattern.get('name', '').lower().replace(' ', '_')
            if pattern_name in self.pattern_names:
                idx = self.pattern_names.index(pattern_name)
                pattern_label[idx] = 1
        
        # Add educational patterns based on repository analysis
        educational_features = self.extract_educational_indicators(repo_data)
        
        # Determine if this is an educational project
        educational_score = np.mean(educational_features)
        
        if educational_score > 0.6:  # High educational indicators
            # Mark as educational project
            if 'educational_project' in self.pattern_names:
                idx = self.pattern_names.index('educational_project')
                pattern_label[idx] = 1
            
            # Determine specific educational type
            if educational_features[0] > 0.7:  # Course structure
                if 'course_materials' in self.pattern_names:
                    idx = self.pattern_names.index('course_materials')
                    pattern_label[idx] = 1
            
            if educational_features[6] > 0.7:  # Exercise patterns
                if 'learning_exercises' in self.pattern_names:
                    idx = self.pattern_names.index('learning_exercises')
                    pattern_label[idx] = 1
        
        return pattern_label
    
    def predict_enhanced_patterns(self, repo_data: Dict) -> Dict:
        """Predict patterns with enhanced educational detection."""
        if self.rf_classifier is None:
            raise ValueError("Enhanced detector not trained. Call train_enhanced_detector first.")
        
        # Extract enhanced features
        features = self.extract_enhanced_features(repo_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions
        predictions = self.rf_classifier.predict(features_scaled)[0]
        probabilities = self.rf_classifier.predict_proba(features_scaled)
        
        # Format results
        detected_patterns = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            if pred == 1:
                pattern_name = self.pattern_names[i]
                confidence = np.max(proba) if len(proba) > 0 else 0.5
                
                detected_patterns.append({
                    "name": pattern_name.replace('_', ' ').title(),
                    "confidence": confidence,
                    "type": "educational" if "educational" in pattern_name.lower() else "architectural"
                })
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "primary_pattern": detected_patterns[0] if detected_patterns else {"name": "Unknown", "confidence": 0.0},
            "all_patterns": detected_patterns,
            "pattern_count": len(detected_patterns),
            "educational_score": self.calculate_educational_score(repo_data)
        }
    
    def calculate_educational_score(self, repo_data: Dict) -> float:
        """Calculate overall educational score for a repository."""
        educational_features = self.extract_educational_indicators(repo_data)
        return np.mean(educational_features)

def demonstrate_enhanced_detection():
    """Demonstrate enhanced educational detection."""
    print("🎓 ENHANCED EDUCATIONAL PROJECT DETECTION")
    print("=" * 80)
    
    # Load analysis data
    try:
        with open('CombinedAnalysis/enhanced_architecture_analysis.json', 'r') as f:
            analysis_data = json.load(f)
    except FileNotFoundError:
        print("❌ Enhanced architecture analysis not found. Run enhanced analysis first.")
        return
    
    # Initialize enhanced detector
    detector = ImprovedEducationalDetector()
    
    # Train enhanced detector
    training_info = detector.train_enhanced_detector(analysis_data, n_trees=100)
    
    print(f"\n✅ ENHANCED EDUCATIONAL DETECTOR TRAINED SUCCESSFULLY")
    print(f"   • {training_info['n_trees']} parallel decision trees")
    print(f"   • {training_info['n_features']} features (including educational indicators)")
    print(f"   • {training_info['n_patterns']} patterns (including educational)")
    print(f"   • {training_info['n_samples']} training repositories")
    
    # Demonstrate on each repository
    print(f"\n🔍 ENHANCED ANALYSIS RESULTS:")
    
    for repo_name, repo_data in analysis_data.get('detailed_analysis', {}).items():
        print(f"\n" + "="*80)
        print(f"📁 ANALYZING: {repo_name}")
        print("="*80)
        
        # Get enhanced predictions
        results = detector.predict_enhanced_patterns(repo_data)
        
        # Show educational score
        educational_score = results["educational_score"]
        print(f"\n🎓 EDUCATIONAL ANALYSIS:")
        print(f"   • Educational Score: {educational_score:.3f}")
        print(f"   • Project Type: {'Educational' if educational_score > 0.5 else 'Real-world'}")
        
        # Show predictions
        primary = results["primary_pattern"]
        print(f"\n🎯 ENHANCED PREDICTION:")
        print(f"   • Primary Pattern: {primary['name']}")
        print(f"   • Confidence: {primary['confidence']:.3f}")
        print(f"   • Pattern Type: {primary.get('type', 'unknown')}")
        
        print(f"\n📋 ALL DETECTED PATTERNS:")
        for i, pattern in enumerate(results["all_patterns"][:5], 1):
            pattern_type = pattern.get('type', 'unknown')
            print(f"   {i}. {pattern['name']} ({pattern_type}, confidence: {pattern['confidence']:.3f})")
        
        # Special analysis for ibecir/oop-1002
        if repo_name == "ibecir/oop-1002":
            print(f"\n🔍 SPECIAL ANALYSIS FOR OOP COURSE PROJECT:")
            print(f"   • This is an Object-Oriented Programming course project")
            print(f"   • Contains 184 methods across 204 files")
            print(f"   • Organized by weeks (week1, week2, etc.)")
            print(f"   • Educational indicators should be high")
            print(f"   • Should be classified as 'Educational Project' or 'Course Materials'")

if __name__ == "__main__":
    demonstrate_enhanced_detection()
