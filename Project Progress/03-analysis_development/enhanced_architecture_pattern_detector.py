"""
Enhanced Architecture Pattern Detector
=====================================

This script extends the architecture pattern detection capabilities to identify
a comprehensive set of architectural patterns commonly found in software development.

Patterns include:
- Application Architecture: MVC, MVVM, Clean Architecture, etc.
- Design Patterns: Singleton, Factory, Observer, etc.
- System Architecture: Microservices, Monolith, Serverless, etc.
- Framework Patterns: React, Angular, Django, Spring, etc.
- Domain-Specific: Data Science, ML/AI, IoT, Blockchain, etc.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class EnhancedArchitecturePatternDetector:
    """Enhanced detector for comprehensive architectural pattern identification."""
    
    def __init__(self):
        """Initialize the enhanced pattern detector."""
        self.pattern_definitions = self._define_patterns()
        
    def _define_patterns(self) -> Dict:
        """Define comprehensive architectural patterns with detection criteria."""
        return {
            # === APPLICATION ARCHITECTURE PATTERNS ===
            "mvc_pattern": {
                "name": "Model-View-Controller",
                "description": "Separation of data, presentation, and business logic",
                "indicators": {
                    "file_structure": ["models", "views", "controllers", "routes"],
                    "naming_conventions": ["*Controller", "*Model", "*View", "*Route"],
                    "complexity_range": (0.2, 0.6),
                    "file_count_range": (10, 100),
                    "method_count_range": (20, 200)
                }
            },
            
            "mvvm_pattern": {
                "name": "Model-View-ViewModel",
                "description": "Data binding pattern common in modern frameworks",
                "indicators": {
                    "file_structure": ["viewmodels", "views", "models", "services"],
                    "naming_conventions": ["*ViewModel", "*View", "*Model", "*Service"],
                    "framework_indicators": ["react", "vue", "angular", "wpf"],
                    "complexity_range": (0.3, 0.7),
                    "file_count_range": (15, 80)
                }
            },
            
            "clean_architecture": {
                "name": "Clean Architecture",
                "description": "Dependency inversion with layered architecture",
                "indicators": {
                    "file_structure": ["entities", "usecases", "interfaces", "controllers"],
                    "naming_conventions": ["*Entity", "*UseCase", "*Repository", "*Controller"],
                    "complexity_range": (0.4, 0.8),
                    "file_count_range": (20, 150),
                    "method_count_range": (50, 300)
                }
            },
            
            # === SYSTEM ARCHITECTURE PATTERNS ===
            "microservices": {
                "name": "Microservices",
                "description": "Distributed system with independent services",
                "indicators": {
                    "file_structure": ["services", "api", "gateway", "config"],
                    "naming_conventions": ["*Service", "*API", "*Gateway", "*Config"],
                    "complexity_range": (0.5, 0.9),
                    "file_count_range": (30, 200),
                    "method_count_range": (100, 500),
                    "language_diversity": True
                }
            },
            
            "monolithic": {
                "name": "Monolithic Application",
                "description": "Single large application with all functionality",
                "indicators": {
                    "complexity_range": (0.6, 0.9),
                    "file_count_range": (50, 500),
                    "method_count_range": (200, 1000),
                    "single_language": True
                }
            },
            
            "serverless": {
                "name": "Serverless Architecture",
                "description": "Event-driven functions without server management",
                "indicators": {
                    "file_structure": ["functions", "handlers", "events", "triggers"],
                    "naming_conventions": ["*Function", "*Handler", "*Event", "*Trigger"],
                    "framework_indicators": ["aws-lambda", "azure-functions", "google-cloud-functions"],
                    "complexity_range": (0.1, 0.4),
                    "file_count_range": (5, 30)
                }
            },
            
            # === FRAMEWORK-SPECIFIC PATTERNS ===
            "react_application": {
                "name": "React Application",
                "description": "Component-based UI framework",
                "indicators": {
                    "file_structure": ["components", "pages", "hooks", "context"],
                    "naming_conventions": ["*.jsx", "*.tsx", "*Component", "*Hook"],
                    "framework_indicators": ["react", "jsx", "tsx"],
                    "complexity_range": (0.2, 0.6),
                    "file_count_range": (10, 100)
                }
            },
            
            "angular_application": {
                "name": "Angular Application",
                "description": "Full-featured TypeScript framework",
                "indicators": {
                    "file_structure": ["components", "services", "modules", "routing"],
                    "naming_conventions": ["*.component.ts", "*.service.ts", "*.module.ts"],
                    "framework_indicators": ["angular", "@angular"],
                    "complexity_range": (0.3, 0.7),
                    "file_count_range": (15, 120)
                }
            },
            
            "django_application": {
                "name": "Django Application",
                "description": "Python web framework with MVT pattern",
                "indicators": {
                    "file_structure": ["models", "views", "templates", "urls"],
                    "naming_conventions": ["models.py", "views.py", "urls.py", "settings.py"],
                    "framework_indicators": ["django", "djangorestframework"],
                    "complexity_range": (0.3, 0.7),
                    "file_count_range": (20, 150)
                }
            },
            
            "spring_application": {
                "name": "Spring Application",
                "description": "Java enterprise framework",
                "indicators": {
                    "file_structure": ["controllers", "services", "repositories", "entities"],
                    "naming_conventions": ["*Controller", "*Service", "*Repository", "*Entity"],
                    "framework_indicators": ["spring", "@SpringBootApplication"],
                    "complexity_range": (0.4, 0.8),
                    "file_count_range": (25, 200)
                }
            },
            
            # === DOMAIN-SPECIFIC PATTERNS ===
            "data_science_project": {
                "name": "Data Science Project",
                "description": "ML/AI focused project with data processing",
                "indicators": {
                    "file_structure": ["models", "data", "notebooks", "preprocessing"],
                    "naming_conventions": ["*Model", "*Data", "*.ipynb", "*Preprocessing"],
                    "framework_indicators": ["pandas", "numpy", "scikit-learn", "tensorflow", "pytorch"],
                    "complexity_range": (0.4, 0.8),
                    "file_count_range": (10, 80),
                    "semantic_richness_range": (0.6, 1.0)
                }
            },
            
            "blockchain_project": {
                "name": "Blockchain Project",
                "description": "Distributed ledger or cryptocurrency project",
                "indicators": {
                    "file_structure": ["contracts", "nodes", "wallet", "mining"],
                    "naming_conventions": ["*Contract", "*Node", "*Wallet", "*Mining"],
                    "framework_indicators": ["ethereum", "solidity", "web3", "bitcoin"],
                    "complexity_range": (0.5, 0.9),
                    "file_count_range": (15, 100)
                }
            },
            
            "iot_project": {
                "name": "IoT Project",
                "description": "Internet of Things device or system",
                "indicators": {
                    "file_structure": ["sensors", "actuators", "communication", "firmware"],
                    "naming_conventions": ["*Sensor", "*Actuator", "*Communication", "*Firmware"],
                    "framework_indicators": ["arduino", "raspberry-pi", "mqtt", "iot"],
                    "complexity_range": (0.3, 0.7),
                    "file_count_range": (10, 60)
                }
            },
            
            "mobile_app": {
                "name": "Mobile Application",
                "description": "Mobile app development project",
                "indicators": {
                    "file_structure": ["screens", "components", "navigation", "assets"],
                    "naming_conventions": ["*Screen", "*Component", "*Navigation", "*.apk"],
                    "framework_indicators": ["react-native", "flutter", "android", "ios"],
                    "complexity_range": (0.3, 0.7),
                    "file_count_range": (20, 150)
                }
            },
            
            # === DESIGN PATTERNS ===
            "singleton_pattern": {
                "name": "Singleton Pattern",
                "description": "Single instance pattern implementation",
                "indicators": {
                    "naming_conventions": ["*Singleton", "*Instance", "*Manager"],
                    "method_count_range": (1, 10),
                    "complexity_range": (0.1, 0.3)
                }
            },
            
            "factory_pattern": {
                "name": "Factory Pattern",
                "description": "Object creation pattern",
                "indicators": {
                    "naming_conventions": ["*Factory", "*Creator", "*Builder"],
                    "method_count_range": (5, 30),
                    "complexity_range": (0.2, 0.5)
                }
            },
            
            "observer_pattern": {
                "name": "Observer Pattern",
                "description": "Event-driven communication pattern",
                "indicators": {
                    "naming_conventions": ["*Observer", "*Listener", "*Event", "*Callback"],
                    "method_count_range": (10, 50),
                    "complexity_range": (0.3, 0.6)
                }
            },
            
            # === UTILITY PATTERNS ===
            "utility_script": {
                "name": "Utility Script",
                "description": "Simple utility or automation script",
                "indicators": {
                    "complexity_range": (0.1, 0.3),
                    "file_count_range": (1, 10),
                    "method_count_range": (1, 20),
                    "semantic_richness_range": (0.2, 0.6)
                }
            },
            
            "api_project": {
                "name": "API Project",
                "description": "RESTful or GraphQL API service",
                "indicators": {
                    "file_structure": ["routes", "controllers", "middleware", "schemas"],
                    "naming_conventions": ["*Route", "*Controller", "*Middleware", "*Schema"],
                    "framework_indicators": ["express", "fastapi", "flask", "graphql"],
                    "complexity_range": (0.3, 0.7),
                    "file_count_range": (15, 100)
                }
            },
            
            "cli_tool": {
                "name": "CLI Tool",
                "description": "Command-line interface application",
                "indicators": {
                    "file_structure": ["commands", "cli", "utils"],
                    "naming_conventions": ["*Command", "*CLI", "*Utils"],
                    "framework_indicators": ["click", "argparse", "commander"],
                    "complexity_range": (0.2, 0.5),
                    "file_count_range": (5, 40)
                }
            },
            
            "library_project": {
                "name": "Library Project",
                "description": "Reusable code library or package",
                "indicators": {
                    "file_structure": ["src", "lib", "core", "utils"],
                    "naming_conventions": ["*Lib", "*Core", "*Utils", "index"],
                    "complexity_range": (0.3, 0.7),
                    "file_count_range": (10, 80),
                    "method_count_range": (30, 200)
                }
            },
            
            "testing_project": {
                "name": "Testing Project",
                "description": "Test suite or testing framework",
                "indicators": {
                    "file_structure": ["tests", "specs", "fixtures", "mocks"],
                    "naming_conventions": ["*Test", "*Spec", "*Fixture", "*Mock"],
                    "framework_indicators": ["jest", "pytest", "junit", "mocha"],
                    "complexity_range": (0.2, 0.6),
                    "file_count_range": (10, 100)
                }
            },
            
            "documentation_project": {
                "name": "Documentation Project",
                "description": "Documentation or documentation generator",
                "indicators": {
                    "file_structure": ["docs", "documentation", "guides"],
                    "naming_conventions": ["*.md", "*.rst", "*.txt", "README"],
                    "complexity_range": (0.1, 0.4),
                    "file_count_range": (5, 50),
                    "semantic_richness_range": (0.3, 0.8)
                }
            }
        }
    
    def detect_patterns(self, repo_data: Dict) -> Dict:
        """Detect architectural patterns for a repository."""
        ast_metrics = repo_data["ast_metrics"]
        codebert_metrics = repo_data["codebert_metrics"]
        combined_metrics = repo_data["combined_metrics"]
        
        detected_patterns = []
        
        for pattern_key, pattern_info in self.pattern_definitions.items():
            confidence = self._calculate_pattern_confidence(
                pattern_key, pattern_info, ast_metrics, codebert_metrics, combined_metrics
            )
            
            if confidence > 0.3:  # Minimum confidence threshold
                detected_patterns.append({
                    "pattern": pattern_key,
                    "name": pattern_info["name"],
                    "description": pattern_info["description"],
                    "confidence": confidence
                })
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "primary_pattern": detected_patterns[0] if detected_patterns else {"pattern": "unknown", "confidence": 0.0},
            "all_patterns": detected_patterns,
            "pattern_count": len(detected_patterns)
        }
    
    def _calculate_pattern_confidence(self, pattern_key: str, pattern_info: Dict, 
                                    ast_metrics: Dict, codebert_metrics: Dict, 
                                    combined_metrics: Dict) -> float:
        """Calculate confidence score for a specific pattern."""
        indicators = pattern_info["indicators"]
        confidence = 0.0
        total_weight = 0.0
        
        # Check complexity range
        if "complexity_range" in indicators:
            min_comp, max_comp = indicators["complexity_range"]
            complexity = combined_metrics["enhanced_complexity"]
            if min_comp <= complexity <= max_comp:
                confidence += 0.3
            total_weight += 0.3
        
        # Check file count range
        if "file_count_range" in indicators:
            min_files, max_files = indicators["file_count_range"]
            file_count = codebert_metrics["num_files"]
            if min_files <= file_count <= max_files:
                confidence += 0.2
            total_weight += 0.2
        
        # Check method count range
        if "method_count_range" in indicators:
            min_methods, max_methods = indicators["method_count_range"]
            method_count = ast_metrics["total_methods"]
            if min_methods <= method_count <= max_methods:
                confidence += 0.2
            total_weight += 0.2
        
        # Check semantic richness range
        if "semantic_richness_range" in indicators:
            min_semantic, max_semantic = indicators["semantic_richness_range"]
            semantic_richness = combined_metrics["semantic_richness"]
            if min_semantic <= semantic_richness <= max_semantic:
                confidence += 0.15
            total_weight += 0.15
        
        # Check language diversity
        if indicators.get("language_diversity", False):
            if len(ast_metrics["languages"]) > 1:
                confidence += 0.1
            total_weight += 0.1
        
        # Check single language requirement
        if indicators.get("single_language", False):
            if len(ast_metrics["languages"]) == 1:
                confidence += 0.1
            total_weight += 0.1
        
        # Normalize confidence
        if total_weight > 0:
            confidence = confidence / total_weight
        
        return confidence
    
    def get_pattern_categories(self) -> Dict:
        """Get pattern categories for organization."""
        categories = {
            "Application Architecture": [
                "mvc_pattern", "mvvm_pattern", "clean_architecture"
            ],
            "System Architecture": [
                "microservices", "monolithic", "serverless"
            ],
            "Framework-Specific": [
                "react_application", "angular_application", 
                "django_application", "spring_application"
            ],
            "Domain-Specific": [
                "data_science_project", "blockchain_project", 
                "iot_project", "mobile_app"
            ],
            "Design Patterns": [
                "singleton_pattern", "factory_pattern", "observer_pattern"
            ],
            "Utility Patterns": [
                "utility_script", "api_project", "cli_tool", 
                "library_project", "testing_project", "documentation_project"
            ]
        }
        return categories
    
    def generate_pattern_report(self, detected_patterns: Dict) -> str:
        """Generate a human-readable pattern report."""
        if not detected_patterns["all_patterns"]:
            return "No specific architectural patterns detected."
        
        report = f"🏗️  ARCHITECTURAL PATTERNS DETECTED:\n\n"
        
        # Primary pattern
        primary = detected_patterns["primary_pattern"]
        report += f"🎯 PRIMARY PATTERN: {primary['name']}\n"
        report += f"   Confidence: {primary['confidence']:.2f}\n"
        report += f"   Description: {primary['description']}\n\n"
        
        # All patterns
        report += f"📋 ALL DETECTED PATTERNS:\n"
        for i, pattern in enumerate(detected_patterns["all_patterns"][:5], 1):
            report += f"   {i}. {pattern['name']} (confidence: {pattern['confidence']:.2f})\n"
        
        return report


def enhance_combined_analyzer():
    """Enhance the existing combined analyzer with new pattern detection."""
    print("🔧 Enhancing Combined Repository Analyzer with Advanced Pattern Detection...")
    
    # Load existing analysis
    with open('CombinedAnalysis/comprehensive_analysis_report.json', 'r') as f:
        analysis_data = json.load(f)
    
    # Initialize enhanced detector
    detector = EnhancedArchitecturePatternDetector()
    
    # Process each repository
    enhanced_analysis = {}
    for repo_name, repo_data in analysis_data["detailed_analysis"].items():
        print(f"Processing {repo_name}...")
        
        # Detect patterns
        pattern_results = detector.detect_patterns(repo_data)
        
        # Update repository data
        enhanced_repo_data = repo_data.copy()
        enhanced_repo_data["enhanced_architecture_patterns"] = pattern_results
        
        enhanced_analysis[repo_name] = enhanced_repo_data
    
    # Generate enhanced summary
    enhanced_summary = []
    for repo_name, repo_data in enhanced_analysis.items():
        summary_item = {
            "repository": repo_name,
            "project_type": repo_data["project_analysis"].get("project_type", "unknown"),
            "primary_architecture_pattern": repo_data["enhanced_architecture_patterns"]["primary_pattern"]["name"],
            "pattern_confidence": repo_data["enhanced_architecture_patterns"]["primary_pattern"]["confidence"],
            "total_patterns_detected": repo_data["enhanced_architecture_patterns"]["pattern_count"],
            "overall_quality": repo_data["combined_metrics"]["overall_quality"],
            "enhanced_complexity": repo_data["combined_metrics"]["enhanced_complexity"],
            "enhanced_maintainability": repo_data["combined_metrics"]["enhanced_maintainability"],
            "semantic_richness": repo_data["combined_metrics"]["semantic_richness"],
            "technology_diversity": repo_data["combined_metrics"]["technology_diversity"],
            "total_methods": repo_data["ast_metrics"]["total_methods"],
            "total_files": repo_data["codebert_metrics"]["num_files"],
            "languages": ",".join(repo_data["ast_metrics"]["languages"])
        }
        enhanced_summary.append(summary_item)
    
    # Save enhanced analysis
    enhanced_output = {
        "summary": enhanced_summary,
        "detailed_analysis": enhanced_analysis,
        "pattern_definitions": detector.pattern_definitions,
        "pattern_categories": detector.get_pattern_categories()
    }
    
    with open('CombinedAnalysis/enhanced_architecture_analysis.json', 'w') as f:
        json.dump(enhanced_output, f, indent=2)
    
    # Save enhanced CSV
    import pandas as pd
    df = pd.DataFrame(enhanced_summary)
    df.to_csv('CombinedAnalysis/enhanced_architecture_summary.csv', index=False)
    
    print(f"✅ Enhanced analysis saved!")
    print(f"📁 Files generated:")
    print(f"   - CombinedAnalysis/enhanced_architecture_analysis.json")
    print(f"   - CombinedAnalysis/enhanced_architecture_summary.csv")
    
    return enhanced_output


def demonstrate_enhanced_patterns():
    """Demonstrate the enhanced pattern detection capabilities."""
    print("=" * 80)
    print("🏗️  ENHANCED ARCHITECTURAL PATTERN DETECTION DEMONSTRATION")
    print("=" * 80)
    
    # Load enhanced analysis
    with open('CombinedAnalysis/enhanced_architecture_analysis.json', 'r') as f:
        enhanced_data = json.load(f)
    
    print(f"\n📊 PATTERN DETECTION OVERVIEW:")
    print(f"   • Total patterns available: {len(enhanced_data['pattern_definitions'])}")
    print(f"   • Pattern categories: {len(enhanced_data['pattern_categories'])}")
    
    # Show pattern categories
    print(f"\n📂 PATTERN CATEGORIES:")
    for category, patterns in enhanced_data['pattern_categories'].items():
        print(f"   • {category}: {len(patterns)} patterns")
    
    # Show repository results
    print(f"\n🏆 REPOSITORY PATTERN ANALYSIS:")
    for repo in enhanced_data['summary']:
        print(f"\n📁 {repo['repository']}")
        print(f"   🏗️  Primary Pattern: {repo['primary_architecture_pattern']}")
        print(f"   🎯 Confidence: {repo['pattern_confidence']:.2f}")
        print(f"   📊 Total Patterns: {repo['total_patterns_detected']}")
        print(f"   📈 Quality Score: {repo['overall_quality']:.3f}")
    
    # Show pattern distribution
    pattern_counts = {}
    for repo in enhanced_data['summary']:
        pattern = repo['primary_architecture_pattern']
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    print(f"\n📊 PATTERN DISTRIBUTION:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(enhanced_data['summary'])) * 100
        print(f"   • {pattern}: {count} repositories ({percentage:.1f}%)")


if __name__ == "__main__":
    # Enhance the existing analysis
    enhanced_data = enhance_combined_analyzer()
    
    # Demonstrate enhanced patterns
    demonstrate_enhanced_patterns()
    
    print(f"\n🎯 Enhanced pattern detection completed!")
    print(f"   Now detecting {len(enhanced_data['pattern_definitions'])} architectural patterns")
    print(f"   Across {len(enhanced_data['pattern_categories'])} categories")

