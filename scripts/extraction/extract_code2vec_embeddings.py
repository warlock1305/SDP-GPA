"""
Code2Vec Embedding Extractor for Repository Quality Analysis
============================================================

This script extracts code2vec embeddings (vectors) from AST paths to be used with:
- NLP models for repository classification
- CodeBERT for code understanding
- Heuristics for quality assessment
"""

import os
import sys
import numpy as np
import json
import pickle
from collections import defaultdict
import pandas as pd

# Add code2vec to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code2vec'))

from vocabularies import VocabType
from config import Config
from model_base import Code2VecModelBase
from tensorflow_model import Code2VecModel
from extractor import Extractor
from common import common

# === CONFIGURATION ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "saved_model_iter8.release")
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "Code2VecEmbeddings")
JAR_PATH = os.path.join(BASE_DIR, 'code2vec/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar')

# Supported file extensions
SUPPORTED_EXTENSIONS = {".java", ".py", ".js", ".c", ".cpp", ".cc"}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_code2vec_model():
    """Load the pre-trained code2vec model for embedding extraction."""
    config = Config(set_defaults=True, load_from_args=False, verify=False)
    config.MODEL_LOAD_PATH = MODEL_FILE.replace("\\", "/")
    config.RELEASE = True
    config.PREDICT = False  # We don't want predictions, just embeddings
    config.DL_FRAMEWORK = 'tensorflow'

    model = Code2VecModel(config)
    return model, config


def extract_embeddings_from_file(file_path, model, config):
    """Extract code2vec embeddings from a source file."""
    try:
        # Extract paths using code2vec's extractor
        path_extractor = Extractor(config, jar_path=JAR_PATH, max_path_length=8, max_path_width=2)
        predict_lines, hash_to_string_dict = path_extractor.extract_paths(file_path)
        
        if not predict_lines:
            return None, None
        
        # Get embeddings by running prediction but extracting the intermediate representations
        # We'll use the model's internal method to get the method vectors
        method_vectors = []
        
        for predict_line in predict_lines:
            try:
                # Get the method vector from the model's internal representation
                # This accesses the method vector before the final prediction layer
                method_vector = model.get_method_vector(predict_line)
                method_vectors.append(method_vector)
            except Exception as e:
                print(f"[VECTOR ERROR] {file_path} → {str(e)}")
                continue
        
        if not method_vectors:
            return None, None
        
        return method_vectors, predict_lines
        
    except Exception as e:
        print(f"[EMBEDDING ERROR] {file_path} → {str(e)}")
        return None, None


def process_single_file_embeddings(file_path, model, config, output_file_path):
    """Process a single source file and save embeddings."""
    print(f"[PROCESSING] {file_path}")

    # Extract embeddings
    method_vectors, predict_lines = extract_embeddings_from_file(file_path, model, config)
    if method_vectors is None:
        return False

    # Save embeddings
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    embedding_data = {
        "source_file": file_path,
        "num_methods": len(method_vectors),
        "embedding_dimension": len(method_vectors[0]) if method_vectors else 0,
        "method_embeddings": [vector.tolist() for vector in method_vectors],
        "path_contexts": predict_lines,
        "file_embedding": np.mean(method_vectors, axis=0).tolist() if method_vectors else None
    }
    
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(embedding_data, f, indent=2)

    print(f"[SUCCESS] {file_path} → {output_file_path}")
    return True


def process_all_files_for_embeddings():
    """Walk through dataset and extract embeddings from all supported source files."""
    print("=== Code2Vec Embedding Extraction for Repository Analysis ===")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Model: {MODEL_FILE}")
    print(f"Output: {OUTPUT_DIR}")

    # Load model
    print("Loading code2vec model...")
    model, config = load_code2vec_model()
    print("Model loaded successfully!")

    processed_count = 0
    error_count = 0
    repository_stats = defaultdict(lambda: {"files": 0, "methods": 0, "embeddings": []})

    for root, _, files in os.walk(DATASET_ROOT):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            source_file_path = os.path.join(root, file)

            # Build relative path from DATASET_ROOT to preserve structure
            rel_path = os.path.relpath(source_file_path, DATASET_ROOT)
            # Output file with .embeddings.json extension
            output_file_path = os.path.join(OUTPUT_DIR, rel_path + ".embeddings.json")

            success = process_single_file_embeddings(source_file_path, model, config, output_file_path)
            if success:
                processed_count += 1
                # Track repository statistics
                repo_path = os.path.dirname(rel_path)
                repository_stats[repo_path]["files"] += 1
                
                # Read the embedding data to count methods
                try:
                    with open(output_file_path, 'r') as f:
                        data = json.load(f)
                        repository_stats[repo_path]["methods"] += data["num_methods"]
                        repository_stats[repo_path]["embeddings"].append(data["file_embedding"])
                except:
                    pass
            else:
                error_count += 1

    # Create repository-level summaries
    create_repository_summaries(repository_stats)

    print(f"\n=== Summary ===")
    print(f"Processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Total: {processed_count + error_count}")

    # Clean up
    model.close_session()


def create_repository_summaries(repository_stats):
    """Create repository-level embedding summaries for analysis."""
    print("\nCreating repository summaries...")
    
    repo_summaries = {}
    
    for repo_path, stats in repository_stats.items():
        if stats["files"] == 0:
            continue
            
        # Calculate repository-level embedding (average of all file embeddings)
        if stats["embeddings"]:
            repo_embedding = np.mean(stats["embeddings"], axis=0).tolist()
        else:
            repo_embedding = None
        
        repo_summaries[repo_path] = {
            "num_files": stats["files"],
            "num_methods": stats["methods"],
            "avg_methods_per_file": stats["methods"] / stats["files"] if stats["files"] > 0 else 0,
            "repository_embedding": repo_embedding,
            "embedding_dimension": len(repo_embedding) if repo_embedding else 0
        }
    
    # Save repository summaries
    summary_file = os.path.join(OUTPUT_DIR, "repository_summaries.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(repo_summaries, f, indent=2)
    
    # Create CSV for easy analysis
    csv_data = []
    for repo_path, summary in repo_summaries.items():
        csv_data.append({
            "repository": repo_path,
            "num_files": summary["num_files"],
            "num_methods": summary["num_methods"],
            "avg_methods_per_file": summary["avg_methods_per_file"],
            "embedding_dimension": summary["embedding_dimension"]
        })
    
    df = pd.DataFrame(csv_data)
    csv_file = os.path.join(OUTPUT_DIR, "repository_statistics.csv")
    df.to_csv(csv_file, index=False)
    
    print(f"Repository summaries saved to: {summary_file}")
    print(f"Repository statistics saved to: {csv_file}")


def create_embedding_dataset_for_nlp():
    """Create a dataset suitable for NLP and CodeBERT analysis."""
    print("\nCreating embedding dataset for NLP analysis...")
    
    # Collect all embeddings
    all_embeddings = []
    all_metadata = []
    
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.endswith('.embeddings.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract file-level embedding
                    if data["file_embedding"]:
                        all_embeddings.append(data["file_embedding"])
                        all_metadata.append({
                            "source_file": data["source_file"],
                            "num_methods": data["num_methods"],
                            "embedding_dimension": data["embedding_dimension"]
                        })
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    # Convert to numpy arrays
    embeddings_array = np.array(all_embeddings)
    metadata_df = pd.DataFrame(all_metadata)
    
    # Save for NLP/CodeBERT analysis
    np.save(os.path.join(OUTPUT_DIR, "all_file_embeddings.npy"), embeddings_array)
    metadata_df.to_csv(os.path.join(OUTPUT_DIR, "embedding_metadata.csv"), index=False)
    
    print(f"Embedding dataset created:")
    print(f"  - Embeddings shape: {embeddings_array.shape}")
    print(f"  - Metadata shape: {metadata_df.shape}")
    print(f"  - Files: all_file_embeddings.npy, embedding_metadata.csv")


def analyze_embedding_quality():
    """Analyze the quality and characteristics of extracted embeddings."""
    print("\nAnalyzing embedding quality...")
    
    embedding_file = os.path.join(OUTPUT_DIR, "all_file_embeddings.npy")
    if not os.path.exists(embedding_file):
        print("No embeddings found. Run extraction first.")
        return
    
    embeddings = np.load(embedding_file)
    
    analysis = {
        "total_files": len(embeddings),
        "embedding_dimension": embeddings.shape[1],
        "mean_embedding_norm": np.mean(np.linalg.norm(embeddings, axis=1)),
        "std_embedding_norm": np.std(np.linalg.norm(embeddings, axis=1)),
        "min_embedding_norm": np.min(np.linalg.norm(embeddings, axis=1)),
        "max_embedding_norm": np.max(np.linalg.norm(embeddings, axis=1)),
        "mean_embedding_values": np.mean(embeddings),
        "std_embedding_values": np.std(embeddings),
        "sparsity": np.mean(embeddings == 0)
    }
    
    # Save analysis
    analysis_file = os.path.join(OUTPUT_DIR, "embedding_quality_analysis.json")
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print("Embedding Quality Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


def main():
    """Main function to extract embeddings for repository analysis."""
    print("Starting Code2Vec embedding extraction for repository quality analysis...")

    # Check if model exists
    if not os.path.exists(MODEL_FILE + ".meta"):
        print(f"ERROR: Model not found at {MODEL_FILE}")
        print("Please ensure the model files are in the models/ directory")
        exit(1)

    # Check if Java extractor exists
    if not os.path.exists(JAR_PATH):
        print(f"ERROR: Java extractor not found at {JAR_PATH}")
        exit(1)

    # Extract embeddings from all files
    process_all_files_for_embeddings()
    
    # Create dataset for NLP analysis
    create_embedding_dataset_for_nlp()
    
    # Analyze embedding quality
    analyze_embedding_quality()
    
    print("\n✅ Embedding extraction completed!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Generated files:")
    print(f"   - *.embeddings.json (individual file embeddings)")
    print(f"   - repository_summaries.json (repo-level summaries)")
    print(f"   - repository_statistics.csv (repo statistics)")
    print(f"   - all_file_embeddings.npy (numpy array for ML)")
    print(f"   - embedding_metadata.csv (metadata for analysis)")
    print(f"   - embedding_quality_analysis.json (quality metrics)")
    
    print(f"\n🎯 Next steps for repository analysis:")
    print(f"   1. Use embeddings with NLP models for project classification")
    print(f"   2. Feed embeddings to CodeBERT for code understanding")
    print(f"   3. Apply heuristics for quality assessment")
    print(f"   4. Analyze consistency between predicted and actual project types")


if __name__ == "__main__":
    main()
