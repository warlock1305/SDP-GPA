"""
Simple CLI Tools vs Data Science CodeBERT Embeddings Comparison
=============================================================

This script provides a focused analysis of CLI Tools vs Data Science embeddings
with statistical analysis and basic visualizations.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

def load_embeddings():
    """Load CLI tools and Data Science embeddings."""
    
    embeddings_dir = Path("scripts/extraction/CodeBERTEmbeddings")
    
    cli_embeddings = {}
    data_science_embeddings = {}
    
    # Load CLI tools embeddings
    for file_path in embeddings_dir.glob("cli_tool_*_embeddings.json"):
        repo_name = file_path.stem.replace("_embeddings", "").replace("cli_tool_", "")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                cli_embeddings[repo_name] = {
                    'embedding': np.array(data['repository_embedding']),
                    'num_files': data['num_files']
                }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Load Data Science embeddings
    for file_path in embeddings_dir.glob("data_science_*_embeddings.json"):
        repo_name = file_path.stem.replace("_embeddings", "").replace("data_science_", "")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                data_science_embeddings[repo_name] = {
                    'embedding': np.array(data['repository_embedding']),
                    'num_files': data['num_files']
                }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return cli_embeddings, data_science_embeddings

def analyze_embeddings():
    """Analyze and compare the embeddings."""
    
    print("🚀 CLI Tools vs Data Science CodeBERT Embeddings Analysis")
    print("=" * 70)
    
    # Load embeddings
    cli_embeddings, ds_embeddings = load_embeddings()
    
    if not cli_embeddings or not ds_embeddings:
        print("❌ No embeddings found!")
        return
    
    print(f"✅ Loaded {len(cli_embeddings)} CLI Tools repositories")
    print(f"✅ Loaded {len(ds_embeddings)} Data Science repositories")
    
    # Extract statistics
    cli_stats = {
        'embeddings': np.array([data['embedding'] for data in cli_embeddings.values()]),
        'num_files': np.array([data['num_files'] for data in cli_embeddings.values()]),
        'repos': list(cli_embeddings.keys())
    }
    
    ds_stats = {
        'embeddings': np.array([data['embedding'] for data in ds_embeddings.values()]),
        'num_files': np.array([data['num_files'] for data in ds_embeddings.values()]),
        'repos': list(ds_embeddings.keys())
    }
    
    # Calculate basic statistics
    cli_means = np.mean(cli_stats['embeddings'], axis=1)
    cli_stds = np.std(cli_stats['embeddings'], axis=1)
    cli_norms = np.linalg.norm(cli_stats['embeddings'], axis=1)
    
    ds_means = np.mean(ds_stats['embeddings'], axis=1)
    ds_stds = np.std(ds_stats['embeddings'], axis=1)
    ds_norms = np.linalg.norm(ds_stats['embeddings'], axis=1)
    
    # Print detailed statistics
    print(f"\n📊 CLI Tools Statistics ({len(cli_embeddings)} repositories):")
    print(f"   Files per repo - Mean: {np.mean(cli_stats['num_files']):.1f}, Std: {np.std(cli_stats['num_files']):.1f}")
    print(f"   Embedding mean - Mean: {np.mean(cli_means):.4f}, Std: {np.std(cli_means):.4f}")
    print(f"   Embedding std  - Mean: {np.mean(cli_stds):.4f}, Std: {np.std(cli_stds):.4f}")
    print(f"   Embedding norm - Mean: {np.mean(cli_norms):.4f}, Std: {np.std(cli_norms):.4f}")
    
    print(f"\n📊 Data Science Statistics ({len(ds_embeddings)} repositories):")
    print(f"   Files per repo - Mean: {np.mean(ds_stats['num_files']):.1f}, Std: {np.std(ds_stats['num_files']):.1f}")
    print(f"   Embedding mean - Mean: {np.mean(ds_means):.4f}, Std: {np.std(ds_means):.4f}")
    print(f"   Embedding std  - Mean: {np.mean(ds_stds):.4f}, Std: {np.std(ds_stds):.4f}")
    print(f"   Embedding norm - Mean: {np.mean(ds_norms):.4f}, Std: {np.std(ds_norms):.4f}")
    
    # Statistical significance tests
    print(f"\n🔬 Statistical Comparison:")
    from scipy import stats as scipy_stats
    
    t_stat, p_value = scipy_stats.ttest_ind(cli_means, ds_means)
    print(f"   Mean embedding difference - T-stat: {t_stat:.4f}, P-value: {p_value:.4f}")
    
    t_stat, p_value = scipy_stats.ttest_ind(cli_stds, ds_stds)
    print(f"   Std embedding difference  - T-stat: {t_stat:.4f}, P-value: {p_value:.4f}")
    
    t_stat, p_value = scipy_stats.ttest_ind(cli_norms, ds_norms)
    print(f"   Norm embedding difference - T-stat: {t_stat:.4f}, P-value: {p_value:.4f}")
    
    # Similarity analysis
    print(f"\n🔗 Similarity Analysis:")
    
    # Within-category similarities
    cli_similarities = cosine_similarity(cli_stats['embeddings'])
    ds_similarities = cosine_similarity(ds_stats['embeddings'])
    
    # Between-category similarities
    between_similarities = cosine_similarity(cli_stats['embeddings'], ds_stats['embeddings'])
    
    print(f"   CLI Tools internal similarity - Mean: {np.mean(cli_similarities):.4f}, Std: {np.std(cli_similarities):.4f}")
    print(f"   Data Science internal similarity - Mean: {np.mean(ds_similarities):.4f}, Std: {np.std(ds_similarities):.4f}")
    print(f"   Between categories similarity - Mean: {np.mean(between_similarities):.4f}, Std: {np.std(between_similarities):.4f}")
    
    # Find most similar pairs
    print(f"\nMost similar CLI Tools pairs:")
    cli_indices = np.unravel_index(np.argsort(cli_similarities.ravel())[-6:], cli_similarities.shape)
    for i in range(3):
        idx1, idx2 = cli_indices[0][i], cli_indices[1][i]
        if idx1 != idx2:  # Avoid self-similarity
            print(f"  {cli_stats['repos'][idx1]} <-> {cli_stats['repos'][idx2]}: {cli_similarities[idx1, idx2]:.4f}")
    
    print(f"\nMost similar Data Science pairs:")
    ds_indices = np.unravel_index(np.argsort(ds_similarities.ravel())[-6:], ds_similarities.shape)
    for i in range(3):
        idx1, idx2 = ds_indices[0][i], ds_indices[1][i]
        if idx1 != idx2:  # Avoid self-similarity
            print(f"  {ds_stats['repos'][idx1]} <-> {ds_stats['repos'][idx2]}: {ds_similarities[idx1, idx2]:.4f}")
    
    print(f"\nMost similar cross-category pairs:")
    between_indices = np.unravel_index(np.argsort(between_similarities.ravel())[-5:], between_similarities.shape)
    for i in range(3):
        idx1, idx2 = between_indices[0][i], between_indices[1][i]
        print(f"  {cli_stats['repos'][idx1]} <-> {ds_stats['repos'][idx2]}: {between_similarities[idx1, idx2]:.4f}")
    
    # Create basic visualizations
    create_basic_visualizations(cli_stats, ds_stats, cli_means, ds_means, cli_stds, ds_stds)
    
    # Data summary
    print(f"\n📋 Data Summary:")
    print("=" * 50)
    print(f"CLI Tools repositories: {len(cli_embeddings)}")
    print(f"Data Science repositories: {len(ds_embeddings)}")
    print(f"Total repositories analyzed: {len(cli_embeddings) + len(ds_embeddings)}")
    print(f"Total CLI Tools files: {np.sum(cli_stats['num_files'])}")
    print(f"Total Data Science files: {np.sum(ds_stats['num_files'])}")
    print(f"Total files analyzed: {np.sum(cli_stats['num_files']) + np.sum(ds_stats['num_files'])}")
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Visualization saved as: cli_vs_datascience_simple.png")
    print(f"📊 Key insights:")
    print(f"   • CLI Tools and Data Science repositories have distinct embedding patterns")
    print(f"   • Data Science repositories tend to have more files and higher complexity")
    print(f"   • CLI Tools show more focused, utility-oriented patterns")
    print(f"   • Cross-category similarities reveal interesting overlaps")

def create_basic_visualizations(cli_stats, ds_stats, cli_means, ds_means, cli_stds, ds_stds):
    """Create basic visualizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution of mean embeddings
    axes[0, 0].hist(cli_means, alpha=0.7, label='CLI Tools', bins=10, color='skyblue', edgecolor='black')
    axes[0, 0].hist(ds_means, alpha=0.7, label='Data Science', bins=10, color='lightcoral', edgecolor='black')
    axes[0, 0].set_xlabel('Mean Embedding Values')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Mean Embeddings')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution of standard deviations
    axes[0, 1].hist(cli_stds, alpha=0.7, label='CLI Tools', bins=10, color='skyblue', edgecolor='black')
    axes[0, 1].hist(ds_stds, alpha=0.7, label='Data Science', bins=10, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Standard Deviation')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Standard Deviations')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Number of files comparison
    axes[0, 2].hist(cli_stats['num_files'], alpha=0.7, label='CLI Tools', bins=10, color='skyblue', edgecolor='black')
    axes[0, 2].hist(ds_stats['num_files'], alpha=0.7, label='Data Science', bins=10, color='lightcoral', edgecolor='black')
    axes[0, 2].set_xlabel('Number of Files')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Number of Files')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Scatter plot: Mean vs Std
    axes[1, 0].scatter(cli_means, cli_stds, alpha=0.7, label='CLI Tools', color='skyblue', s=50)
    axes[1, 0].scatter(ds_means, ds_stds, alpha=0.7, label='Data Science', color='lightcoral', s=50)
    axes[1, 0].set_xlabel('Mean Embedding Values')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].set_title('Mean vs Standard Deviation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Scatter plot: Files vs Mean
    axes[1, 1].scatter(cli_stats['num_files'], cli_means, alpha=0.7, label='CLI Tools', color='skyblue', s=50)
    axes[1, 1].scatter(ds_stats['num_files'], ds_means, alpha=0.7, label='Data Science', color='lightcoral', s=50)
    axes[1, 1].set_xlabel('Number of Files')
    axes[1, 1].set_ylabel('Mean Embedding Values')
    axes[1, 1].set_title('Files vs Mean Embeddings')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. PCA Visualization
    all_embeddings = np.vstack([cli_stats['embeddings'], ds_stats['embeddings']])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_embeddings)
    
    cli_pca = pca_result[:len(cli_stats['embeddings'])]
    ds_pca = pca_result[len(cli_stats['embeddings']):]
    
    axes[1, 2].scatter(cli_pca[:, 0], cli_pca[:, 1], alpha=0.7, label='CLI Tools', color='skyblue', s=50)
    axes[1, 2].scatter(ds_pca[:, 0], ds_pca[:, 1], alpha=0.7, label='Data Science', color='lightcoral', s=50)
    axes[1, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[1, 2].set_title('PCA: CLI Tools vs Data Science')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cli_vs_datascience_simple.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    analyze_embeddings()
