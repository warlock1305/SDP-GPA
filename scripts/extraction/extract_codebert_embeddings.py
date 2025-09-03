"""
CodeBERT Embeddings Extractor for Multi-Language Repository Analysis
====================================================================

This script integrates CodeBERT to provide semantic embeddings for:
- Python, Java, JavaScript, PHP, Ruby, Go repositories
- Code understanding and architecture analysis
- Framework and pattern detection
- Semantic similarity analysis
"""

import os
import torch
import numpy as np
import json
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import re
import json
from typing import List, Dict, Tuple, Optional

# === CONFIGURATION ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "dataset"))
OUTPUT_DIR = os.path.join(BASE_DIR, "CodeBERTEmbeddings")
AST_FEATURES_DIR = os.path.join(BASE_DIR, "ASTFeaturesForAnalysis")

# Supported file extensions for CodeBERT
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".java": "java", 
    ".js": "javascript",
    ".php": "php",
    ".rb": "ruby",
    ".go": "go",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".ipynb": "python"  # Jupyter notebooks - will extract Python code
}

# CodeBERT model configuration
MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512  # Maximum sequence length for CodeBERT

os.makedirs(OUTPUT_DIR, exist_ok=True)


class CodeBERTExtractor:
    """CodeBERT-based code embedding extractor."""
    
    def __init__(self):
        """Initialize CodeBERT model and tokenizer."""
        print("Loading CodeBERT model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        
        print("CodeBERT model loaded successfully!")
    
    def extract_file_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """Extract embedding for a single source file."""
        try:
            # Handle Jupyter notebooks specially
            if file_path.endswith('.ipynb'):
                code_content = self._extract_python_from_notebook(file_path)
            else:
                # Read the source file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code_content = f.read()
            
            if not code_content.strip():
                return None
            
            # Clean and prepare code
            code_content = self._clean_code(code_content)
            
            # Tokenize
            tokens = self.tokenizer.tokenize(code_content)
            
            # Truncate if too long
            if len(tokens) > MAX_LENGTH - 2:  # Account for [CLS] and [SEP]
                tokens = tokens[:MAX_LENGTH - 2]
            
            # Add special tokens
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # Convert to tensor
            input_ids = torch.tensor([token_ids]).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(input_ids)
                # Use the [CLS] token embedding as file representation
                file_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            
            return file_embedding
            
        except Exception as e:
            print(f"[ERROR] Failed to extract embedding from {file_path}: {str(e)}")
            return None
    
    def _extract_python_from_notebook(self, notebook_path: str) -> str:
        """Extract Python code from Jupyter notebook (.ipynb file)."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            python_code = []
            
            # Extract code from all cells
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'code':
                    # Get source content
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        # Join multiple lines
                        cell_code = ''.join(source)
                    else:
                        cell_code = str(source)
                    
                    if cell_code.strip():
                        python_code.append(cell_code)
            
            # Join all code cells with newlines
            return '\n\n'.join(python_code)
            
        except Exception as e:
            print(f"[WARNING] Failed to extract Python from notebook {notebook_path}: {str(e)}")
            return ""
    
    def extract_repository_embedding(self, repo_path: str) -> Dict:
        """Extract embeddings for all source files in a repository."""
        repo_embeddings = []
        file_info = []
        
        # Walk through repository
        for root, _, files in os.walk(repo_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, repo_path)
                    
                    # Extract embedding
                    embedding = self.extract_file_embedding(file_path)
                    if embedding is not None:
                        repo_embeddings.append(embedding)
                        file_info.append({
                            "file_path": rel_path,
                            "language": SUPPORTED_EXTENSIONS[ext],
                            "embedding_dim": len(embedding),
                            "original_format": "jupyter_notebook" if ext == ".ipynb" else "source_file"
                        })
        
        if not repo_embeddings:
            return None
        
        # Calculate repository-level embedding (average of all file embeddings)
        repo_embedding = np.mean(repo_embeddings, axis=0)
        
        return {
            "repository_embedding": repo_embedding.tolist(),
            "num_files": len(repo_embeddings),
            "embedding_dimension": len(repo_embedding),
            "file_embeddings": [emb.tolist() for emb in repo_embeddings],
            "file_info": file_info
        }
    
    def _clean_code(self, code: str) -> str:
        """Clean and prepare code for tokenization."""
        # Remove comments (basic cleaning)
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove single-line comments
            if '#' in line:
                line = line.split('#')[0]
            if '//' in line:
                line = line.split('//')[0]
            if '/*' in line and '*/' in line:
                line = re.sub(r'/\*.*?\*/', '', line, flags=re.DOTALL)
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def analyze_code_semantics(self, code_content: str) -> Dict:
        """Analyze code semantics using CodeBERT."""
        try:
            # Tokenize code
            tokens = self.tokenizer.tokenize(code_content)
            if len(tokens) > MAX_LENGTH - 2:
                tokens = tokens[:MAX_LENGTH - 2]
            
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([token_ids]).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(input_ids)
                embeddings = outputs.last_hidden_state[0].cpu().numpy()
            
            # Analyze patterns
            analysis = {
                "code_length": len(code_content),
                "token_count": len(tokens),
                "embedding_stats": {
                    "mean": float(np.mean(embeddings)),
                    "std": float(np.std(embeddings)),
                    "min": float(np.min(embeddings)),
                    "max": float(np.max(embeddings))
                }
            }
            
            return analysis
            
        except Exception as e:
            print(f"[ERROR] Semantic analysis failed: {str(e)}")
            return None


def process_all_repositories():
    """Process all repositories in the dataset."""
    print("=== CodeBERT Embeddings Extraction ===")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Initialize CodeBERT extractor
    extractor = CodeBERTExtractor()
    
    processed_count = 0
    error_count = 0
    repository_results = {}
    
    # Process each repository
    for user in os.listdir(DATASET_ROOT):
        user_path = os.path.join(DATASET_ROOT, user)
        if not os.path.isdir(user_path):
            continue
        
        for repo in os.listdir(user_path):
            repo_path = os.path.join(user_path, repo)
            if not os.path.isdir(repo_path):
                continue
            
            print(f"\n[PROCESSING] {user}/{repo}")
            
            # Extract CodeBERT embeddings
            repo_data = extractor.extract_repository_embedding(repo_path)
            
            if repo_data:
                # Save repository embeddings
                repo_key = f"{user}/{repo}"
                output_file = os.path.join(OUTPUT_DIR, f"{repo_key.replace('/', '_')}_embeddings.json")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(repo_data, f, indent=2)
                
                repository_results[repo_key] = repo_data
                processed_count += 1
                print(f"[SUCCESS] {repo_key} → {repo_data['num_files']} files, {repo_data['embedding_dimension']}D embeddings")
            else:
                error_count += 1
                print(f"[ERROR] {user}/{repo} → No embeddings extracted")
    
    # Create summary
    create_repository_summary(repository_results)
    
    print(f"\n=== Summary ===")
    print(f"Processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Total: {processed_count + error_count}")
    
    return repository_results


def create_repository_summary(repository_results: Dict):
    """Create summary of all repository embeddings."""
    print("\nCreating repository summary...")
    
    summary_data = []
    for repo_name, data in repository_results.items():
        summary_data.append({
            "repository": repo_name,
            "num_files": data["num_files"],
            "embedding_dimension": data["embedding_dimension"],
            "avg_embedding_norm": np.linalg.norm(data["repository_embedding"])
        })
    
    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, "repository_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    
    # Create CSV
    df = pd.DataFrame(summary_data)
    csv_file = os.path.join(OUTPUT_DIR, "repository_summary.csv")
    df.to_csv(csv_file, index=False)
    
    print(f"Summary saved to: {summary_file}")
    print(f"CSV saved to: {csv_file}")


def main():
    """Main function for CodeBERT embeddings extraction."""
    print("Starting CodeBERT embeddings extraction...")
    
    # Extract embeddings from all repositories
    repository_results = process_all_repositories()
    
    print("\n✅ CodeBERT embeddings extraction completed!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Generated files:")
    print(f"   - *_embeddings.json (individual repository embeddings)")
    print(f"   - repository_summary.json (summary data)")
    print(f"   - repository_summary.csv (summary CSV)")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Combine with AST features for comprehensive analysis")
    print(f"   2. Use embeddings for semantic similarity analysis")
    print(f"   3. Integrate with repository quality assessment")


if __name__ == "__main__":
    main()
