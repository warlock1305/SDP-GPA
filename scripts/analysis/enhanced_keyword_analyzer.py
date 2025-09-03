"""
Enhanced Keyword Analyzer for Code Content Analysis
==================================================

This script uses the keywords.py file to analyze actual code content and create
quantitative feature vectors that represent the qualitative aspects of code.

Key Features:
- Analyzes actual code files for keyword presence
- Creates quantitative feature vectors from qualitative code characteristics
- Supports multiple programming languages and topics
- Generates feature vectors for Random Forest classification
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import Counter
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import keywords from the existing file
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from keywords import expertise_keywords, topics_keywords

class EnhancedKeywordAnalyzer:
    """Enhanced keyword analyzer for code content analysis."""
    
    def __init__(self):
        """Initialize the enhanced keyword analyzer."""
        self.expertise_keywords = expertise_keywords
        self.topics_keywords = topics_keywords
        
        # File extensions to analyze
        self.code_extensions = {
            '.py', '.java', '.js', '.ts', '.jsx', '.tsx', '.cpp', '.c', '.h', '.hpp',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.r', '.sql', '.sh',
            '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd', '.makefile', '.mk',
            '.dockerfile', '.yaml', '.yml', '.json', '.xml', '.html', '.css', '.scss',
            '.sass', '.less', '.vue', '.svelte', '.md', '.txt', '.cfg', '.conf', '.ini'
        }
        
        # Educational indicators (enhanced from keywords)
        self.educational_indicators = {
            'course_structure': [
                'week', 'lab', 'lecture', 'exercise', 'assignment', 'homework',
                'tutorial', 'example', 'demo', 'practice', 'learning', 'course',
                'student', 'teacher', 'professor', 'classroom', 'education',
                'lesson', 'chapter', 'module', 'unit', 'task', 'problem',
                'quiz', 'test', 'exam', 'project', 'submission', 'grading'
            ],
            'academic_patterns': [
                'main.java', 'test.java', 'example.java', 'demo.java',
                'calculator.java', 'student.java', 'person.java', 'book.java',
                'library.java', 'animal.java', 'vehicle.java', 'shape.java',
                'inheritance', 'polymorphism', 'encapsulation', 'abstraction',
                'interface', 'abstract class', 'constructor', 'getter', 'setter'
            ],
            'learning_progression': [
                'basic', 'intermediate', 'advanced', 'beginner', 'expert',
                'fundamental', 'concept', 'principle', 'theory', 'practice',
                'step by step', 'tutorial', 'guide', 'walkthrough', 'explanation'
            ]
        }
        
        # Create comprehensive keyword mapping
        self.all_keywords = self._create_comprehensive_keyword_mapping()
    
    def _create_comprehensive_keyword_mapping(self) -> Dict[str, List[str]]:
        """Create a comprehensive mapping of all keywords."""
        all_keywords = {}
        
        # Add expertise keywords
        for language, keywords in self.expertise_keywords.items():
            all_keywords[f"lang_{language.lower()}"] = keywords
        
        # Add topic keywords
        for topic, keywords in self.topics_keywords.items():
            all_keywords[f"topic_{topic}"] = keywords
        
        # Add educational indicators
        for category, keywords in self.educational_indicators.items():
            all_keywords[f"edu_{category}"] = keywords
        
        return all_keywords
    
    def analyze_repository_content(self, repo_path: str) -> Dict:
        """Analyze the content of a repository and create feature vectors."""
        print(f"🔍 Analyzing repository content: {repo_path}")
        
        # Get all code files in the repository
        code_files = self._get_code_files(repo_path)
        
        if not code_files:
            print(f"   ⚠️  No code files found in {repo_path}")
            return self._create_empty_analysis()
        
        print(f"   📁 Found {len(code_files)} code files")
        
        # Limit number of files for performance
        max_files = 100  # Process max 100 files
        if len(code_files) > max_files:
            print(f"   ⚠️  Limiting analysis to first {max_files} files for performance")
            code_files = code_files[:max_files]
        
        # Analyze each file efficiently with streaming approach
        file_analyses = []
        total_content = ""
        processed_size = 0
        max_total_size = 50 * 1024 * 1024  # 50MB total limit to prevent memory issues
        
        for i, file_path in enumerate(code_files):
            try:
                # Add progress indicator for large repositories
                if len(code_files) > 20 and i % 20 == 0:
                    print(f"      📄 Processing file {i+1}/{len(code_files)}")
                
                # Get file size first
                file_size = os.path.getsize(file_path)
                
                # Skip very large files that would consume too much memory
                if file_size > 5 * 1024 * 1024:  # 5MB per file limit
                    print(f"      ⚠️  Skipping very large file: {file_path} ({file_size / 1024 / 1024:.1f}MB)")
                    continue
                
                # Check total processed size
                if processed_size + file_size > max_total_size:
                    print(f"      ⚠️  Memory limit reached, stopping file processing")
                    break
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    processed_size += len(content)
                    
                    # Add to total content (but limit if getting too large)
                    if len(total_content) < 10 * 1024 * 1024:  # 10MB limit for total content
                        total_content += content + "\n"
                    
                    file_analysis = self._analyze_file_content(content, file_path)
                    file_analyses.append(file_analysis)
                    
            except Exception as e:
                print(f"   ⚠️  Error reading {file_path}: {e}")
                continue
        
        # Create comprehensive analysis
        analysis = self._create_comprehensive_analysis(file_analyses, total_content, repo_path)
        
        return analysis
    
    def _get_code_files(self, repo_path: str) -> List[str]:
        """Get all code files in the repository."""
        code_files = []
        
        for root, dirs, files in os.walk(repo_path):
            # Skip common directories that don't contain code
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env', '.idea', '.vscode'}]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                
                if file_ext in self.code_extensions:
                    code_files.append(file_path)
        
        return code_files
    
    def _analyze_file_content(self, content: str, file_path: str) -> Dict:
        """Analyze the content of a single file."""
        # Normalize content for analysis
        normalized_content = self._normalize_content(content)
        
        # Count keywords in each category
        keyword_counts = {}
        for category, keywords in self.all_keywords.items():
            count = self._count_keywords_in_content(normalized_content, keywords)
            keyword_counts[category] = count
        
        # Analyze file characteristics
        file_analysis = {
            'file_path': file_path,
            'file_size': len(content),
            'line_count': len(content.split('\n')),
            'keyword_counts': keyword_counts,
            'language_detected': self._detect_language(file_path, normalized_content),
            'complexity_indicators': self._analyze_complexity_indicators(content),
            'educational_indicators': self._analyze_educational_indicators(normalized_content)
        }
        
        return file_analysis
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for keyword analysis."""
        # Convert to lowercase
        content = content.lower()
        
        # Remove comments (basic approach)
        # Remove single-line comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        
        # Remove multi-line comments (basic)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Remove string literals (basic approach)
        content = re.sub(r'"[^"]*"', '', content)
        content = re.sub(r"'[^']*'", '', content)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        return content
    
    def _count_keywords_in_content(self, content: str, keywords: List[str]) -> int:
        """Count how many keywords from a list appear in the content with optimization."""
        count = 0
        
        # Limit content size for performance
        if len(content) > 100000:  # 100KB limit
            content = content[:100000]
        
        # Use faster string operations for simple keywords
        content_lower = content.lower()
        
        for keyword in keywords:
            if len(keyword) < 3:  # Skip very short keywords
                continue
                
            # Use simple string count for better performance
            keyword_lower = keyword.lower()
            count += content_lower.count(keyword_lower)
            
            # Limit total count to prevent excessive processing
            if count > 1000:
                break
        
        return min(count, 1000)  # Cap at 1000
    
    def _detect_language(self, file_path: str, content: str) -> str:
        """Detect the primary language of the file."""
        file_ext = Path(file_path).suffix.lower()
        
        # Map extensions to languages
        ext_to_lang = {
            '.py': 'python', '.java': 'java', '.js': 'javascript', '.ts': 'typescript',
            '.jsx': 'javascript', '.tsx': 'typescript', '.cpp': 'c++', '.c': 'c',
            '.h': 'c', '.hpp': 'c++', '.cs': 'c#', '.php': 'php', '.rb': 'ruby',
            '.go': 'go', '.rs': 'rust', '.swift': 'swift', '.kt': 'kotlin',
            '.r': 'r', '.sql': 'sql', '.sh': 'shell', '.bash': 'shell',
            '.zsh': 'shell', '.fish': 'shell', '.ps1': 'shell', '.bat': 'shell',
            '.cmd': 'shell', '.makefile': 'makefile', '.mk': 'makefile',
            '.dockerfile': 'dockerfile', '.yaml': 'yaml', '.yml': 'yaml',
            '.json': 'json', '.xml': 'xml', '.html': 'html', '.css': 'css',
            '.scss': 'css', '.sass': 'css', '.less': 'css', '.vue': 'vue',
            '.svelte': 'svelte', '.md': 'markdown', '.txt': 'text'
        }
        
        return ext_to_lang.get(file_ext, 'unknown')
    
    def _analyze_complexity_indicators(self, content: str) -> Dict:
        """Analyze code complexity indicators."""
        lines = content.split('\n')
        
        complexity_indicators = {
            'total_lines': len(lines),
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'comment_lines': len([line for line in lines if line.strip().startswith(('#', '//', '/*', '*'))]),
            'function_definitions': len(re.findall(r'\b(def|function|public|private|protected)\s+\w+', content)),
            'class_definitions': len(re.findall(r'\bclass\s+\w+', content)),
            'import_statements': len(re.findall(r'\b(import|from|using|require|include)\b', content)),
            'conditional_statements': len(re.findall(r'\b(if|else|elif|switch|case)\b', content)),
            'loop_statements': len(re.findall(r'\b(for|while|do|foreach)\b', content)),
            'exception_handling': len(re.findall(r'\b(try|catch|except|finally|throw)\b', content))
        }
        
        return complexity_indicators
    
    def _analyze_educational_indicators(self, content: str) -> Dict:
        """Analyze educational indicators in the content."""
        edu_indicators = {}
        
        for category, keywords in self.educational_indicators.items():
            count = self._count_keywords_in_content(content, keywords)
            edu_indicators[category] = count
        
        return edu_indicators
    
    def _create_comprehensive_analysis(self, file_analyses: List[Dict], total_content: str, repo_path: str) -> Dict:
        """Create comprehensive analysis from file analyses."""
        # Aggregate keyword counts across all files
        total_keyword_counts = {}
        for category in self.all_keywords.keys():
            total_keyword_counts[category] = sum(
                analysis['keyword_counts'].get(category, 0) for analysis in file_analyses
            )
        
        # Aggregate complexity indicators
        total_complexity = {
            'total_lines': sum(analysis['complexity_indicators']['total_lines'] for analysis in file_analyses),
            'non_empty_lines': sum(analysis['complexity_indicators']['non_empty_lines'] for analysis in file_analyses),
            'comment_lines': sum(analysis['complexity_indicators']['comment_lines'] for analysis in file_analyses),
            'function_definitions': sum(analysis['complexity_indicators']['function_definitions'] for analysis in file_analyses),
            'class_definitions': sum(analysis['complexity_indicators']['class_definitions'] for analysis in file_analyses),
            'import_statements': sum(analysis['complexity_indicators']['import_statements'] for analysis in file_analyses),
            'conditional_statements': sum(analysis['complexity_indicators']['conditional_statements'] for analysis in file_analyses),
            'loop_statements': sum(analysis['complexity_indicators']['loop_statements'] for analysis in file_analyses),
            'exception_handling': sum(analysis['complexity_indicators']['exception_handling'] for analysis in file_analyses)
        }
        
        # Aggregate educational indicators
        total_educational = {}
        for category in self.educational_indicators.keys():
            total_educational[category] = sum(
                analysis['educational_indicators'].get(category, 0) for analysis in file_analyses
            )
        
        # Detect languages used
        languages_used = set()
        for analysis in file_analyses:
            if analysis['language_detected'] != 'unknown':
                languages_used.add(analysis['language_detected'])
        
        # Create feature vector
        feature_vector = self._create_feature_vector(
            total_keyword_counts, total_complexity, total_educational, languages_used, len(file_analyses)
        )
        
        analysis = {
            'repository_path': repo_path,
            'file_count': len(file_analyses),
            'total_content_size': len(total_content),
            'languages_detected': list(languages_used),
            'keyword_analysis': total_keyword_counts,
            'complexity_analysis': total_complexity,
            'educational_analysis': total_educational,
            'feature_vector': feature_vector,
            'file_analyses': file_analyses
        }
        
        return analysis
    
    def _create_feature_vector(self, keyword_counts: Dict, complexity: Dict, educational: Dict, 
                              languages: Set[str], file_count: int) -> np.ndarray:
        """Create a quantitative feature vector for machine learning."""
        features = []
        
        # 1. Language expertise features (normalized counts)
        for category, keywords in self.all_keywords.items():
            if category.startswith('lang_'):
                # Normalize by number of keywords in category
                normalized_count = keyword_counts.get(category, 0) / len(keywords)
                features.append(normalized_count)
        
        # 2. Topic expertise features (normalized counts)
        for category, keywords in self.all_keywords.items():
            if category.startswith('topic_'):
                normalized_count = keyword_counts.get(category, 0) / len(keywords)
                features.append(normalized_count)
        
        # 3. Educational indicator features (normalized counts)
        for category, keywords in self.educational_indicators.items():
            normalized_count = educational.get(category, 0) / len(keywords)
            features.append(normalized_count)
        
        # 4. Complexity features (normalized by file count)
        if file_count > 0:
            features.extend([
                complexity['total_lines'] / file_count,
                complexity['non_empty_lines'] / file_count,
                complexity['comment_lines'] / file_count,
                complexity['function_definitions'] / file_count,
                complexity['class_definitions'] / file_count,
                complexity['import_statements'] / file_count,
                complexity['conditional_statements'] / file_count,
                complexity['loop_statements'] / file_count,
                complexity['exception_handling'] / file_count
            ])
        else:
            features.extend([0] * 9)
        
        # 5. Language diversity features
        features.append(len(languages))  # Number of languages used
        features.append(file_count)      # Total number of files
        
        # 6. Code quality indicators
        if complexity['total_lines'] > 0:
            features.extend([
                complexity['comment_lines'] / complexity['total_lines'],  # Comment ratio
                complexity['function_definitions'] / complexity['total_lines'],  # Function density
                complexity['class_definitions'] / complexity['total_lines'],  # Class density
                complexity['import_statements'] / complexity['total_lines']  # Import density
            ])
        else:
            features.extend([0] * 4)
        
        return np.array(features, dtype=np.float32)
    
    def _create_empty_analysis(self) -> Dict:
        """Create empty analysis when no code files are found."""
        empty_vector = np.zeros(len(self.all_keywords) + 9 + 2 + 4, dtype=np.float32)
        
        return {
            'repository_path': '',
            'file_count': 0,
            'total_content_size': 0,
            'languages_detected': [],
            'keyword_analysis': {category: 0 for category in self.all_keywords.keys()},
            'complexity_analysis': {
                'total_lines': 0, 'non_empty_lines': 0, 'comment_lines': 0,
                'function_definitions': 0, 'class_definitions': 0, 'import_statements': 0,
                'conditional_statements': 0, 'loop_statements': 0, 'exception_handling': 0
            },
            'educational_analysis': {category: 0 for category in self.educational_indicators.keys()},
            'feature_vector': empty_vector,
            'file_analyses': []
        }
    
    def _extract_basic_keyword_features(self, repository_path: str) -> Dict[str, Any]:
        """Basic keyword extraction as fallback (compatible with comprehensive_repository_analyzer_v3)."""
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
        
        # Collect all text content
        all_content = self._collect_repository_content(repository_path)
        
        if not all_content:
            return keyword_features
        
        # Analyze keywords using the same approach as comprehensive_repository_analyzer_v3
        keyword_counts = self._analyze_basic_keywords(all_content)
        
        # Update features
        keyword_features.update(keyword_counts)
        
        return keyword_features
    
    def _collect_repository_content(self, repository_path: str) -> str:
        """Collect all text content from repository efficiently."""
        content_parts = []
        total_size = 0
        max_total_size = 20 * 1024 * 1024  # 20MB limit for basic analysis
        
        # Check if the path is a directory
        if not os.path.isdir(repository_path):
            print(f"⚠️  Path is not a directory: {repository_path}")
            return ""
        
        # Text file extensions
        text_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.php', '.rb', '.go', '.rs', '.kt', '.swift', '.ts', '.jsx', '.tsx', '.md', '.txt', '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.scss', '.sass'}
        
        for root, _, files in os.walk(repository_path):
            for file in files:
                if total_size >= max_total_size:
                    print(f"      ⚠️  Content size limit reached for basic analysis")
                    break
                    
                ext = os.path.splitext(file)[1].lower()
                if ext in text_extensions:
                    file_path = os.path.join(root, file)
                    # Only process actual files
                    if os.path.isfile(file_path):
                        try:
                            file_size = os.path.getsize(file_path)
                            if file_size > 2 * 1024 * 1024:  # Skip files > 2MB
                                continue
                                
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            if content.strip():
                                content_parts.append(content)
                                total_size += len(content)
                                
                        except Exception as e:
                            print(f"⚠️  Error reading {file_path}: {e}")
        
        return '\n\n'.join(content_parts)
    
    def _analyze_basic_keywords(self, content: str) -> Dict[str, float]:
        """Analyze keywords in content using basic approach."""
        content_lower = content.lower()
        
        keyword_counts = {}
        
        # Count expertise keywords
        for category, keywords in self.expertise_keywords.items():
            count = sum(1 for keyword in keywords if keyword.lower() in content_lower)
            keyword_counts[f'{category.lower()}_keywords'] = count
        
        # Count topic keywords
        for category, keywords in self.topics_keywords.items():
            count = sum(1 for keyword in keywords if keyword.lower() in content_lower)
            keyword_counts[f'{category.lower()}_keywords'] = count
        
        # Calculate totals
        total_keywords = sum(keyword_counts.values())
        keyword_counts['total_keywords'] = total_keywords
        keyword_counts['keyword_diversity'] = len([v for v in keyword_counts.values() if v > 0])
        
        return keyword_counts
    
    def get_feature_names(self) -> List[str]:
        """Get descriptive names for the feature vector."""
        feature_names = []
        
        # Language expertise features
        for category in self.all_keywords.keys():
            if category.startswith('lang_'):
                lang_name = category.replace('lang_', '').title()
                feature_names.append(f"{lang_name}_expertise")
        
        # Topic expertise features
        for category in self.all_keywords.keys():
            if category.startswith('topic_'):
                topic_name = category.replace('topic_', '').replace('_', ' ').title()
                feature_names.append(f"{topic_name}_expertise")
        
        # Educational indicator features
        for category in self.educational_indicators.keys():
            feature_names.append(f"educational_{category}")
        
        # Complexity features
        complexity_names = [
            'avg_lines_per_file', 'avg_non_empty_lines_per_file', 'avg_comment_lines_per_file',
            'avg_functions_per_file', 'avg_classes_per_file', 'avg_imports_per_file',
            'avg_conditionals_per_file', 'avg_loops_per_file', 'avg_exceptions_per_file'
        ]
        feature_names.extend(complexity_names)
        
        # Language diversity features
        feature_names.extend(['language_count', 'file_count'])
        
        # Code quality features
        quality_names = [
            'comment_ratio', 'function_density', 'class_density', 'import_density'
        ]
        feature_names.extend(quality_names)
        
        return feature_names

def analyze_dataset_keywords():
    """Analyze the entire dataset using keyword analysis."""
    print("🔍 ENHANCED KEYWORD ANALYSIS OF DATASET")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = EnhancedKeywordAnalyzer()
    
    # Get dataset path
    dataset_path = "dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    # Analyze each repository
    repository_analyses = {}
    
    for user_dir in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user_dir)
        if os.path.isdir(user_path):
            for repo_dir in os.listdir(user_path):
                repo_path = os.path.join(user_path, repo_dir)
                if os.path.isdir(repo_path):
                    repo_name = f"{user_dir}/{repo_dir}"
                    print(f"\n📁 Analyzing: {repo_name}")
                    
                    # Analyze repository content
                    analysis = analyzer.analyze_repository_content(repo_path)
                    repository_analyses[repo_name] = analysis
    
    # Save comprehensive analysis
    output_path = "KeywordAnalysis"
    os.makedirs(output_path, exist_ok=True)
    
    # Save detailed analysis
    with open(os.path.join(output_path, 'keyword_analysis.json'), 'w') as f:
        json.dump(repository_analyses, f, indent=2, default=str)
    
    # Create summary
    summary = {
        'total_repositories': len(repository_analyses),
        'feature_vector_dimensions': len(analyzer.get_feature_names()),
        'feature_names': analyzer.get_feature_names(),
        'repository_summaries': {}
    }
    
    for repo_name, analysis in repository_analyses.items():
        summary['repository_summaries'][repo_name] = {
            'file_count': analysis['file_count'],
            'languages_detected': analysis['languages_detected'],
            'total_keywords_found': sum(analysis['keyword_analysis'].values()),
            'educational_score': sum(analysis['educational_analysis'].values()),
            'feature_vector_shape': analysis['feature_vector'].shape
        }
    
    with open(os.path.join(output_path, 'keyword_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Keyword analysis completed!")
    print(f"   • Analyzed {len(repository_analyses)} repositories")
    print(f"   • Created {len(analyzer.get_feature_names())} feature dimensions")
    print(f"   • Results saved to {output_path}/")
    
    # Show sample results
    print(f"\n📊 SAMPLE ANALYSIS RESULTS:")
    for repo_name, analysis in list(repository_analyses.items())[:3]:
        print(f"\n🏗️  {repo_name}:")
        print(f"   • Files: {analysis['file_count']}")
        print(f"   • Languages: {', '.join(analysis['languages_detected'])}")
        print(f"   • Total Keywords: {sum(analysis['keyword_analysis'].values())}")
        print(f"   • Educational Score: {sum(analysis['educational_analysis'].values())}")
        print(f"   • Feature Vector: {analysis['feature_vector'].shape}")
    
    return repository_analyses

if __name__ == "__main__":
    analyze_dataset_keywords()
