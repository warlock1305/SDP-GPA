"""
AST Extraction and Code2Vec Transformation Explanation
====================================================

This script explains how AST values are extracted and how code2vec transforms them.
"""

import os
import re
from collections import defaultdict

# === EXAMPLE 1: Simple Java Code to AST ===

EXAMPLE_JAVA_CODE = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int multiply(int x, int y) {
        int result = x * y;
        return result;
    }
}
"""

def explain_ast_extraction():
    """Explain how AST extraction works step by step."""
    
    print("=== AST EXTRACTION PROCESS ===\n")
    
    print("1. SOURCE CODE:")
    print(EXAMPLE_JAVA_CODE)
    
    print("\n2. PARSING STEPS:")
    print("   a) Lexical Analysis: Breaks code into tokens")
    print("   b) Syntax Analysis: Builds parse tree")
    print("   c) AST Construction: Creates abstract syntax tree")
    
    print("\n3. AST STRUCTURE (Simplified):")
    print("""
    CompilationUnit
    └── ClassDeclaration (Calculator)
        ├── Modifier (public)
        ├── Identifier (Calculator)
        └── ClassBody
            ├── MethodDeclaration (add)
            │   ├── Modifier (public)
            │   ├── Type (int)
            │   ├── Identifier (add)
            │   ├── Parameters
            │   │   ├── Parameter (int a)
            │   │   └── Parameter (int b)
            │   └── Block
            │       └── ReturnStatement
            │           └── BinaryExpression (+)
            │               ├── VariableReference (a)
            │               └── VariableReference (b)
            └── MethodDeclaration (multiply)
                ├── Modifier (public)
                ├── Type (int)
                ├── Identifier (multiply)
                ├── Parameters
                │   ├── Parameter (int x)
                │   └── Parameter (int y)
                └── Block
                    ├── VariableDeclaration
                    │   ├── Type (int)
                    │   ├── Identifier (result)
                    │   └── BinaryExpression (*)
                    │       ├── VariableReference (x)
                    │       └── VariableReference (y)
                    └── ReturnStatement
                        └── VariableReference (result)
    """)
    
    print("\n4. ASTMINER EXTRACTION:")
    print("   - Uses ANTLR parser to build AST")
    print("   - Assigns unique IDs to each node")
    print("   - Extracts paths between leaf nodes")
    print("   - Stores node types and relationships")


def explain_path_extraction():
    """Explain how AST paths are extracted."""
    
    print("\n=== AST PATH EXTRACTION ===\n")
    
    print("1. PATH CONTEXT DEFINITION:")
    print("   A path context is: <start_token, path, end_token>")
    print("   Where 'path' is the sequence of AST nodes between tokens")
    
    print("\n2. EXAMPLE: For method 'add(int a, int b)':")
    print("   Start token: 'add'")
    print("   End token: 'a'")
    print("   Path: MethodDeclaration → Parameter → VariableDeclaratorId")
    
    print("\n3. ACTUAL ASTMINER OUTPUT FORMAT:")
    print("   Each line: <method_name> <path_contexts>")
    print("   Path context: <start_token,path,end_token>")
    
    print("\n4. SAMPLE EXTRACTION:")
    print("   Method: add")
    print("   Path contexts:")
    print("   - add,MethodDeclaration→Parameter→VariableDeclaratorId,a")
    print("   - add,MethodDeclaration→Parameter→VariableDeclaratorId,b")
    print("   - a,VariableDeclaratorId→BinaryExpression→VariableReference,b")
    print("   - a,VariableDeclaratorId→BinaryExpression→VariableReference,+")


def explain_code2vec_transformation():
    """Explain how code2vec transforms AST paths."""
    
    print("\n=== CODE2VEC TRANSFORMATION ===\n")
    
    print("1. INPUT: AST Path Contexts")
    print("   Format: <start_token, path, end_token>")
    print("   Example: add,MethodDeclaration→Parameter→VariableDeclaratorId,a")
    
    print("\n2. TOKENIZATION:")
    print("   - Start token: 'add' → token ID: 123")
    print("   - Path: 'MethodDeclaration→Parameter→VariableDeclaratorId' → path ID: 456")
    print("   - End token: 'a' → token ID: 789")
    
    print("\n3. EMBEDDING LOOKUP:")
    print("   - Each token gets a learned vector representation")
    print("   - Each path gets a learned vector representation")
    print("   - Vectors are typically 128-512 dimensions")
    
    print("\n4. ATTENTION MECHANISM:")
    print("   - Weights each path context based on relevance")
    print("   - Important paths get higher attention scores")
    print("   - Less relevant paths get lower scores")
    
    print("\n5. AGGREGATION:")
    print("   - Weighted sum of all path context vectors")
    print("   - Creates a single vector representing the method")
    print("   - This vector captures the method's semantic meaning")


def show_real_example():
    """Show a real example from your extracted data."""
    
    print("\n=== REAL EXAMPLE FROM YOUR DATA ===\n")
    
    # Let's look at a real prediction file
    prediction_file = "Code2VecPredictions/ibecir/oop-1002/src/main/java/week1/labs/Main.java.pred.txt"
    
    if os.path.exists(prediction_file):
        with open(prediction_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("1. SOURCE FILE: Main.java")
        print("2. EXTRACTED PATH CONTEXTS:")
        
        # Find the path contexts line
        lines = content.split('\n')
        for line in lines:
            if 'Extracted' in line and 'path contexts' in line:
                print(f"   {line}")
                break
        
        print("\n3. METHOD PREDICTIONS:")
        method_sections = content.split('\n\n')
        for section in method_sections[:3]:  # Show first 3 methods
            if 'Method ' in section and 'Original name:' in section:
                lines = section.strip().split('\n')
                method_name = ""
                predictions = []
                
                for line in lines:
                    if line.startswith('  Original name:'):
                        method_name = line.replace('  Original name:', '').strip()
                    elif line.startswith('    (') and ')' in line:
                        predictions.append(line.strip())
                
                if method_name:
                    print(f"\n   Method: {method_name}")
                    print("   Predictions:")
                    for pred in predictions[:3]:  # Top 3 predictions
                        print(f"     {pred}")
                    
                    print("   Attention Paths:")
                    # Find attention paths
                    for line in lines:
                        if '\t' in line and ',' in line:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                score = parts[0]
                                path_info = parts[1]
                                print(f"     Score: {score} | Path: {path_info}")
                                break
    else:
        print("Prediction file not found. Let me check what's available...")
        
        # List some prediction files
        pred_dir = "Code2VecPredictions"
        if os.path.exists(pred_dir):
            files = [f for f in os.listdir(pred_dir) if f.endswith('.pred.txt')]
            if files:
                print(f"Available prediction files: {files[:5]}")
                
                # Read first available file
                first_file = os.path.join(pred_dir, files[0])
                with open(first_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print(f"\nSample from {files[0]}:")
                lines = content.split('\n')
                for line in lines[:10]:
                    print(f"   {line}")


def explain_attention_mechanism():
    """Explain how the attention mechanism works in code2vec."""
    
    print("\n=== ATTENTION MECHANISM EXPLANATION ===\n")
    
    print("1. WHAT IS ATTENTION?")
    print("   - Attention tells the model which parts of the code are important")
    print("   - Higher attention scores = more important for prediction")
    print("   - Lower attention scores = less important")
    
    print("\n2. HOW IT WORKS:")
    print("   a) Each path context gets an attention weight")
    print("   b) Weights are learned during training")
    print("   c) Important paths get higher weights")
    print("   d) Final prediction uses weighted combination")
    
    print("\n3. EXAMPLE ATTENTION SCORES:")
    print("   - Path: 'method→parameter→variable' → Score: 0.85 (high)")
    print("   - Path: 'method→block→statement' → Score: 0.45 (medium)")
    print("   - Path: 'method→modifier→public' → Score: 0.12 (low)")
    
    print("\n4. WHY THIS MATTERS:")
    print("   - Helps understand what the model focuses on")
    print("   - Shows which code patterns are most predictive")
    print("   - Provides interpretability for predictions")


def main():
    """Run all explanations."""
    print("AST EXTRACTION AND CODE2VEC TRANSFORMATION GUIDE")
    print("=" * 55)
    
    explain_ast_extraction()
    explain_path_extraction()
    explain_code2vec_transformation()
    explain_attention_mechanism()
    show_real_example()
    
    print("\n" + "=" * 55)
    print("SUMMARY:")
    print("1. AST extraction creates a tree representation of code structure")
    print("2. Path contexts capture relationships between code elements")
    print("3. Code2vec transforms paths into learned vector representations")
    print("4. Attention mechanism weights important paths higher")
    print("5. Final prediction combines all weighted path vectors")


if __name__ == "__main__":
    main()

