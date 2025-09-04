import json
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Tuple
import ast
import re

class LanguageSpecificSchematicExtractor:
    """Extract language-specific schematics for design patterns"""
    
    def __init__(self):
        self.language_patterns = {
            'python': self._extract_python_schematic,
            'java': self._extract_java_schematic,
            'cpp': self._extract_cpp_schematic,
            'javascript': self._extract_javascript_schematic,
            'php': self._extract_php_schematic
        }
        
        # Language-specific pattern indicators
        self.language_indicators = {
            'python': {
                'singleton': {
                    'private_constructor': ['__new__', 'cls'],
                    'static_instance': ['_instance', 'instance'],
                    'get_instance': ['get_instance', 'getInstance', 'instance']
                },
                'factory': {
                    'abstract_method': ['@abstractmethod', 'ABC'],
                    'factory_method': ['create_', 'factory', 'get_'],
                    'interface': ['ABC', 'Protocol']
                },
                'observer': {
                    'observer_interface': ['Observer', 'Subscriber'],
                    'subject_interface': ['Subject', 'Publisher'],
                    'notify_method': ['notify', 'update', 'broadcast']
                }
            },
            'java': {
                'singleton': {
                    'private_constructor': ['private', 'constructor'],
                    'static_instance': ['static', 'instance'],
                    'get_instance': ['getInstance', 'get_instance']
                },
                'factory': {
                    'abstract_method': ['abstract', 'interface'],
                    'factory_method': ['create', 'factory', 'get'],
                    'interface': ['interface', 'implements']
                },
                'observer': {
                    'observer_interface': ['Observer', 'Listener'],
                    'subject_interface': ['Subject', 'Observable'],
                    'notify_method': ['notify', 'update', 'notifyObservers']
                }
            },
            'cpp': {
                'singleton': {
                    'private_constructor': ['private:', 'constructor'],
                    'static_instance': ['static', 'instance'],
                    'get_instance': ['getInstance', 'get_instance']
                },
                'factory': {
                    'abstract_method': ['virtual', 'pure virtual'],
                    'factory_method': ['create', 'factory', 'get'],
                    'interface': ['virtual', '= 0']
                },
                'observer': {
                    'observer_interface': ['Observer', 'Listener'],
                    'subject_interface': ['Subject', 'Observable'],
                    'notify_method': ['notify', 'update', 'notifyObservers']
                }
            },
            'javascript': {
                'singleton': {
                    'private_constructor': ['constructor', 'instance'],
                    'static_instance': ['static', 'instance'],
                    'get_instance': ['getInstance', 'get_instance']
                },
                'factory': {
                    'abstract_method': ['abstract', 'interface'],
                    'factory_method': ['create', 'factory', 'get'],
                    'interface': ['interface', 'implements']
                },
                'observer': {
                    'observer_interface': ['Observer', 'Listener'],
                    'subject_interface': ['Subject', 'Observable'],
                    'notify_method': ['notify', 'update', 'notifyObservers']
                }
            },
            'php': {
                'singleton': {
                    'private_constructor': ['private', '__construct'],
                    'static_instance': ['static', '$instance'],
                    'get_instance': ['getInstance', 'get_instance']
                },
                'factory': {
                    'abstract_method': ['abstract', 'interface'],
                    'factory_method': ['create', 'factory', 'get'],
                    'interface': ['interface', 'implements']
                },
                'observer': {
                    'observer_interface': ['Observer', 'Listener'],
                    'subject_interface': ['Subject', 'Observable'],
                    'notify_method': ['notify', 'update', 'notifyObservers']
                }
            }
        }
    
    def extract_language_specific_schematic(self, cpg_data: Dict, pattern_name: str, language: str) -> Dict:
        """Extract language-specific schematic for a design pattern"""
        
        if language not in self.language_patterns:
            raise ValueError(f"Unsupported language: {language}")
        
        # Build graph from CPG data
        G = nx.MultiDiGraph()
        
        # Add nodes
        for node in cpg_data.get('nodes', []):
            G.add_node(node['id'], **node)
        
        # Add edges
        for edge in cpg_data.get('edges', []):
            G.add_edge(edge['source_id'], edge['target_id'], **edge)
        
        # Extract language-specific schematic
        schematic = self.language_patterns[language](G, pattern_name, language)
        
        return schematic
    
    def _extract_python_schematic(self, G: nx.MultiDiGraph, pattern_name: str, language: str) -> Dict:
        """Extract Python-specific schematic patterns"""
        
        schematic = {
            'pattern_name': pattern_name,
            'language': language,
            'python_specific_patterns': {},
            'structural_patterns': self._extract_structural_patterns(G),
            'relational_patterns': self._extract_relational_patterns(G)
        }
        
        # Python-specific pattern detection
        if pattern_name == 'singleton':
            schematic['python_specific_patterns'] = {
                'has_new_method': self._has_python_new_method(G),
                'has_cls_parameter': self._has_cls_parameter(G),
                'has_static_instance': self._has_static_instance(G, 'python'),
                'has_get_instance': self._has_get_instance(G, 'python')
            }
        
        elif pattern_name == 'factory':
            schematic['python_specific_patterns'] = {
                'has_abc_import': self._has_abc_import(G),
                'has_abstractmethod': self._has_abstractmethod(G),
                'has_factory_methods': self._has_factory_methods(G, 'python')
            }
        
        elif pattern_name == 'observer':
            schematic['python_specific_patterns'] = {
                'has_observer_interface': self._has_observer_interface(G, 'python'),
                'has_subject_interface': self._has_subject_interface(G, 'python'),
                'has_notify_methods': self._has_notify_methods(G, 'python')
            }
        
        return schematic
    
    def _extract_java_schematic(self, G: nx.MultiDiGraph, pattern_name: str, language: str) -> Dict:
        """Extract Java-specific schematic patterns"""
        
        schematic = {
            'pattern_name': pattern_name,
            'language': language,
            'java_specific_patterns': {},
            'structural_patterns': self._extract_structural_patterns(G),
            'relational_patterns': self._extract_relational_patterns(G)
        }
        
        # Java-specific pattern detection
        if pattern_name == 'singleton':
            schematic['java_specific_patterns'] = {
                'has_private_constructor': self._has_private_constructor(G, 'java'),
                'has_static_instance': self._has_static_instance(G, 'java'),
                'has_get_instance': self._has_get_instance(G, 'java'),
                'has_synchronized': self._has_synchronized_method(G)
            }
        
        elif pattern_name == 'factory':
            schematic['java_specific_patterns'] = {
                'has_interface': self._has_interface(G, 'java'),
                'has_abstract_class': self._has_abstract_class(G, 'java'),
                'has_factory_methods': self._has_factory_methods(G, 'java')
            }
        
        elif pattern_name == 'observer':
            schematic['java_specific_patterns'] = {
                'has_observer_interface': self._has_observer_interface(G, 'java'),
                'has_subject_interface': self._has_subject_interface(G, 'java'),
                'has_notify_methods': self._has_notify_methods(G, 'java')
            }
        
        return schematic
    
    def _extract_cpp_schematic(self, G: nx.MultiDiGraph, pattern_name: str, language: str) -> Dict:
        """Extract C++-specific schematic patterns"""
        
        schematic = {
            'pattern_name': pattern_name,
            'language': language,
            'cpp_specific_patterns': {},
            'structural_patterns': self._extract_structural_patterns(G),
            'relational_patterns': self._extract_relational_patterns(G)
        }
        
        # C++-specific pattern detection
        if pattern_name == 'singleton':
            schematic['cpp_specific_patterns'] = {
                'has_private_constructor': self._has_private_constructor(G, 'cpp'),
                'has_static_instance': self._has_static_instance(G, 'cpp'),
                'has_get_instance': self._has_get_instance(G, 'cpp'),
                'has_nullptr_check': self._has_nullptr_check(G)
            }
        
        elif pattern_name == 'factory':
            schematic['cpp_specific_patterns'] = {
                'has_virtual_methods': self._has_virtual_methods(G),
                'has_pure_virtual': self._has_pure_virtual(G),
                'has_factory_methods': self._has_factory_methods(G, 'cpp')
            }
        
        return schematic
    
    def _extract_javascript_schematic(self, G: nx.MultiDiGraph, pattern_name: str, language: str) -> Dict:
        """Extract JavaScript-specific schematic patterns"""
        
        schematic = {
            'pattern_name': pattern_name,
            'language': language,
            'javascript_specific_patterns': {},
            'structural_patterns': self._extract_structural_patterns(G),
            'relational_patterns': self._extract_relational_patterns(G)
        }
        
        # JavaScript-specific pattern detection
        if pattern_name == 'singleton':
            schematic['javascript_specific_patterns'] = {
                'has_constructor_check': self._has_constructor_check(G),
                'has_static_instance': self._has_static_instance(G, 'javascript'),
                'has_get_instance': self._has_get_instance(G, 'javascript')
            }
        
        return schematic
    
    def _extract_php_schematic(self, G: nx.MultiDiGraph, pattern_name: str, language: str) -> Dict:
        """Extract PHP-specific schematic patterns"""
        
        schematic = {
            'pattern_name': pattern_name,
            'language': language,
            'php_specific_patterns': {},
            'structural_patterns': self._extract_structural_patterns(G),
            'relational_patterns': self._extract_relational_patterns(G)
        }
        
        # PHP-specific pattern detection
        if pattern_name == 'singleton':
            schematic['php_specific_patterns'] = {
                'has_private_construct': self._has_private_construct(G),
                'has_static_instance': self._has_static_instance(G, 'php'),
                'has_get_instance': self._has_get_instance(G, 'php'),
                'has_clone_prevention': self._has_clone_prevention(G)
            }
        
        return schematic
    
    # Helper methods for pattern detection
    def _extract_structural_patterns(self, G: nx.MultiDiGraph) -> Dict:
        """Extract structural patterns from graph"""
        try:
            # Convert to undirected for connectivity check
            G_undirected = G.to_undirected()
            is_connected = nx.is_connected(G_undirected) if G.number_of_nodes() > 1 else True
        except:
            is_connected = False
        
        return {
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': is_connected,
            'node_types': self._get_node_type_distribution(G),
            'centrality_patterns': self._get_centrality_patterns(G)
        }
    
    def _extract_relational_patterns(self, G: nx.MultiDiGraph) -> Dict:
        """Extract relational patterns from graph"""
        return {
            'edge_types': self._get_edge_type_distribution(G),
            'inheritance_patterns': self._get_inheritance_patterns(G),
            'composition_patterns': self._get_composition_patterns(G),
            'dependency_patterns': self._get_dependency_patterns(G)
        }
    
    # Language-specific detection methods
    def _has_python_new_method(self, G: nx.MultiDiGraph) -> bool:
        """Check if Python class has __new__ method"""
        for node in G.nodes(data=True):
            if (G.nodes[node[0]].get('node_type') == 'method' and 
                '__new__' in G.nodes[node[0]].get('name', '')):
                return True
        return False
    
    def _has_cls_parameter(self, G: nx.MultiDiGraph) -> bool:
        """Check if Python method has cls parameter"""
        for node in G.nodes(data=True):
            if (G.nodes[node[0]].get('node_type') == 'method' and 
                'cls' in str(G.nodes[node[0]].get('properties', {}))):
                return True
        return False
    
    def _has_abc_import(self, G: nx.MultiDiGraph) -> bool:
        """Check if Python code imports ABC"""
        for node in G.nodes(data=True):
            if (G.nodes[node[0]].get('node_type') == 'import' and 
                'ABC' in str(G.nodes[node[0]].get('name', ''))):
                return True
        return False
    
    def _has_abstractmethod(self, G: nx.MultiDiGraph) -> bool:
        """Check if Python code uses @abstractmethod"""
        for node in G.nodes(data=True):
            if '@abstractmethod' in str(G.nodes[node[0]].get('properties', {})):
                return True
        return False
    
    def _has_private_constructor(self, G: nx.MultiDiGraph, language: str) -> bool:
        """Check for private constructor based on language"""
        indicators = self.language_indicators[language]['singleton']['private_constructor']
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') == 'constructor':
                modifiers = node_data.get('modifiers', [])
                if any(indicator in str(modifiers) for indicator in indicators):
                    return True
        return False
    
    def _has_static_instance(self, G: nx.MultiDiGraph, language: str) -> bool:
        """Check for static instance based on language"""
        indicators = self.language_indicators[language]['singleton']['static_instance']
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') == 'field':
                modifiers = node_data.get('modifiers', [])
                name = node_data.get('name', '')
                if any(indicator in str(modifiers) or indicator in name for indicator in indicators):
                    return True
        return False
    
    def _has_get_instance(self, G: nx.MultiDiGraph, language: str) -> bool:
        """Check for get instance method based on language"""
        indicators = self.language_indicators[language]['singleton']['get_instance']
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') == 'method':
                name = node_data.get('name', '')
                if any(indicator in name for indicator in indicators):
                    return True
        return False
    
    def _has_synchronized_method(self, G: nx.MultiDiGraph) -> bool:
        """Check if Java method is synchronized"""
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') == 'method':
                modifiers = node_data.get('modifiers', [])
                if 'synchronized' in modifiers:
                    return True
        return False
    
    def _has_interface(self, G: nx.MultiDiGraph, language: str) -> bool:
        """Check if Java code has interface"""
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') == 'interface':
                return True
        return False
    
    def _has_abstract_class(self, G: nx.MultiDiGraph) -> bool:
        """Check if Java code has abstract class"""
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') == 'class':
                modifiers = node_data.get('modifiers', [])
                if 'abstract' in modifiers:
                    return True
        return False
    
    def _has_virtual_methods(self, G: nx.MultiDiGraph) -> bool:
        """Check if C++ code has virtual methods"""
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') == 'method':
                modifiers = node_data.get('modifiers', [])
                if 'virtual' in modifiers:
                    return True
        return False
    
    def _has_pure_virtual(self, G: nx.MultiDiGraph) -> bool:
        """Check if C++ code has pure virtual methods"""
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') == 'method':
                properties = node_data.get('properties', {})
                if '= 0' in str(properties):
                    return True
        return False
    
    def _has_nullptr_check(self, G: nx.MultiDiGraph) -> bool:
        """Check if C++ code has nullptr check"""
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            properties = node_data.get('properties', {})
            if 'nullptr' in str(properties):
                return True
        return False
    
    def _has_constructor_check(self, G: nx.MultiDiGraph) -> bool:
        """Check if JavaScript constructor checks for existing instance"""
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') == 'constructor':
                properties = node_data.get('properties', {})
                if 'instance' in str(properties):
                    return True
        return False
    
    def _has_private_construct(self, G: nx.MultiDiGraph) -> bool:
        """Check if PHP has private __construct"""
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') == 'constructor':
                modifiers = node_data.get('modifiers', [])
                if 'private' in modifiers:
                    return True
        return False
    
    def _has_clone_prevention(self, G: nx.MultiDiGraph) -> bool:
        """Check if PHP has clone prevention methods"""
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') == 'method':
                name = node_data.get('name', '')
                if '__clone' in name or '__wakeup' in name:
                    return True
        return False
    
    def _has_factory_methods(self, G: nx.MultiDiGraph, language: str) -> bool:
        """Check for factory methods based on language"""
        indicators = self.language_indicators[language]['factory']['factory_method']
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') == 'method':
                name = node_data.get('name', '')
                if any(indicator in name for indicator in indicators):
                    return True
        return False
    
    def _has_observer_interface(self, G: nx.MultiDiGraph, language: str) -> bool:
        """Check for observer interface based on language"""
        indicators = self.language_indicators[language]['observer']['observer_interface']
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') in ['interface', 'class']:
                name = node_data.get('name', '')
                if any(indicator in name for indicator in indicators):
                    return True
        return False
    
    def _has_subject_interface(self, G: nx.MultiDiGraph, language: str) -> bool:
        """Check for subject interface based on language"""
        indicators = self.language_indicators[language]['observer']['subject_interface']
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') in ['interface', 'class']:
                name = node_data.get('name', '')
                if any(indicator in name for indicator in indicators):
                    return True
        return False
    
    def _has_notify_methods(self, G: nx.MultiDiGraph, language: str) -> bool:
        """Check for notify methods based on language"""
        indicators = self.language_indicators[language]['observer']['notify_method']
        for node in G.nodes(data=True):
            node_data = G.nodes[node[0]]
            if node_data.get('node_type') == 'method':
                name = node_data.get('name', '')
                if any(indicator in name for indicator in indicators):
                    return True
        return False
    
    # Utility methods for structural analysis
    def _get_node_type_distribution(self, G: nx.MultiDiGraph) -> Dict:
        """Get distribution of node types"""
        distribution = {}
        for node in G.nodes(data=True):
            node_type = G.nodes[node[0]].get('node_type', 'unknown')
            distribution[node_type] = distribution.get(node_type, 0) + 1
        return distribution
    
    def _get_edge_type_distribution(self, G: nx.MultiDiGraph) -> Dict:
        """Get distribution of edge types"""
        distribution = {}
        for edge in G.edges(data=True):
            edge_type = edge[2].get('edge_type', 'unknown')
            distribution[edge_type] = distribution.get(edge_type, 0) + 1
        return distribution
    
    def _get_centrality_patterns(self, G: nx.MultiDiGraph) -> Dict:
        """Get centrality patterns"""
        if G.number_of_nodes() == 0:
            return {}
        
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            return {
                'max_degree_centrality': max(degree_centrality.values()) if degree_centrality else 0,
                'avg_degree_centrality': sum(degree_centrality.values()) / len(degree_centrality) if degree_centrality else 0,
                'max_betweenness_centrality': max(betweenness_centrality.values()) if betweenness_centrality else 0,
                'avg_betweenness_centrality': sum(betweenness_centrality.values()) / len(betweenness_centrality) if betweenness_centrality else 0
            }
        except:
            return {}
    
    def _get_inheritance_patterns(self, G: nx.MultiDiGraph) -> Dict:
        """Get inheritance patterns"""
        inheritance_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'inherits']
        return {
            'inheritance_count': len(inheritance_edges),
            'has_inheritance': len(inheritance_edges) > 0
        }
    
    def _get_composition_patterns(self, G: nx.MultiDiGraph) -> Dict:
        """Get composition patterns"""
        composition_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'contains']
        return {
            'composition_count': len(composition_edges),
            'has_composition': len(composition_edges) > 0
        }
    
    def _get_dependency_patterns(self, G: nx.MultiDiGraph) -> Dict:
        """Get dependency patterns"""
        dependency_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'calls']
        return {
            'dependency_count': len(dependency_edges),
            'has_dependencies': len(dependency_edges) > 0
        }

def main():
    """Main function to extract language-specific schematics"""
    
    # Load CPG data from multi-language demos
    multi_lang_dir = Path("design_pattern_dataset/multi_language_demos")
    extractor = LanguageSpecificSchematicExtractor()
    
    all_schematics = {}
    
    for pattern_dir in multi_lang_dir.iterdir():
        if pattern_dir.is_dir():
            pattern_name = pattern_dir.name
            all_schematics[pattern_name] = {}
            
            for lang_dir in pattern_dir.iterdir():
                if lang_dir.is_dir():
                    language = lang_dir.name
                    
                    # Load CPG data for this language implementation
                    cpg_file = lang_dir / f"{pattern_name}_{language}_cpg.json"
                    
                    if cpg_file.exists():
                        with open(cpg_file, 'r') as f:
                            cpg_data = json.load(f)
                        
                        # Extract language-specific schematic
                        schematic = extractor.extract_language_specific_schematic(
                            cpg_data, pattern_name, language
                        )
                        
                        all_schematics[pattern_name][language] = schematic
    
    # Save all schematics
    with open("language_specific_schematics.json", 'w') as f:
        json.dump(all_schematics, f, indent=2)
    
    print("Language-specific schematics extracted successfully!")
    print("Saved to: language_specific_schematics.json")

if __name__ == "__main__":
    main()
