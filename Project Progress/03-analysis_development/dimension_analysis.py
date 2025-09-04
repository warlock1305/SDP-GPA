"""
CodeBERT Embeddings: Dimension Patterns and Cross-Category Analysis
==================================================================

This script analyzes:
1. Specific dimension patterns (architectural signatures)
2. Cross-category similarities (overlapping patterns)
3. Dimension-wise differences between CLI Tools and Data Science
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy import stats
import pandas as pd

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

def analyze_dimension_patterns(cli_embeddings, ds_embeddings):
    """Analyze specific dimension patterns and differences."""
    
    print("🔍 CodeBERT Embeddings: Dimension Pattern Analysis")
    print("=" * 60)
    
    # Convert to numpy arrays
    cli_embeddings_array = np.array([data['embedding'] for data in cli_embeddings.values()])
    ds_embeddings_array = np.array([data['embedding'] for data in ds_embeddings.values()])
    
    # Calculate mean embeddings for each category
    cli_mean_embedding = np.mean(cli_embeddings_array, axis=0)
    ds_mean_embedding = np.mean(ds_embeddings_array, axis=0)
    
    # Calculate dimension-wise differences
    dimension_differences = ds_mean_embedding - cli_mean_embedding
    dimension_ratios = ds_mean_embedding / (cli_mean_embedding + 1e-8)  # Avoid division by zero
    
    # Find most different dimensions
    abs_differences = np.abs(dimension_differences)
    top_different_dimensions = np.argsort(abs_differences)[-20:]  # Top 20 most different
    
    print(f"\n📊 Dimension Analysis Summary:")
    print(f"   CLI Tools mean embedding shape: {cli_mean_embedding.shape}")
    print(f"   Data Science mean embedding shape: {ds_mean_embedding.shape}")
    print(f"   Maximum dimension difference: {np.max(abs_differences):.6f}")
    print(f"   Minimum dimension difference: {np.min(abs_differences):.6f}")
    print(f"   Mean dimension difference: {np.mean(abs_differences):.6f}")
    
    # Statistical significance test for each dimension
    print(f"\n🔬 Statistical Significance Analysis:")
    significant_dimensions = []
    p_values = []
    
    for i in range(768):
        cli_dim_values = cli_embeddings_array[:, i]
        ds_dim_values = ds_embeddings_array[:, i]
        t_stat, p_val = stats.ttest_ind(cli_dim_values, ds_dim_values)
        if p_val < 0.05:  # Significant difference
            significant_dimensions.append(i)
            p_values.append(p_val)
    
    print(f"   Dimensions with significant differences (p < 0.05): {len(significant_dimensions)}")
    print(f"   Percentage of significant dimensions: {len(significant_dimensions)/768*100:.1f}%")
    
    # Show top 10 most significantly different dimensions
    if significant_dimensions:
        significant_differences = [abs_differences[i] for i in significant_dimensions]
        top_significant_indices = np.argsort(significant_differences)[-10:]
        top_significant_dims = [significant_dimensions[i] for i in top_significant_indices]
        
        print(f"\n🏆 Top 10 Most Significantly Different Dimensions:")
        for i, dim in enumerate(reversed(top_significant_dims)):
            diff = dimension_differences[dim]
            p_val = p_values[significant_dimensions.index(dim)]
            print(f"   Dimension {dim:3d}: CLI={cli_mean_embedding[dim]:8.4f}, DS={ds_mean_embedding[dim]:8.4f}, Diff={diff:8.4f}, p={p_val:.6f}")
    
    return {
        'cli_mean': cli_mean_embedding,
        'ds_mean': ds_mean_embedding,
        'differences': dimension_differences,
        'ratios': dimension_ratios,
        'top_different': top_different_dimensions,
        'significant_dimensions': significant_dimensions,
        'p_values': p_values
    }

def analyze_cross_category_similarities(cli_embeddings, ds_embeddings):
    """Analyze cross-category similarities and overlapping patterns."""
    
    print(f"\n🔗 Cross-Category Similarity Analysis")
    print("=" * 50)
    
    # Convert to numpy arrays
    cli_embeddings_array = np.array([data['embedding'] for data in cli_embeddings.values()])
    ds_embeddings_array = np.array([data['embedding'] for data in ds_embeddings.values()])
    
    cli_repos = list(cli_embeddings.keys())
    ds_repos = list(ds_embeddings.keys())
    
    # Calculate cosine similarities between all pairs
    similarities = cosine_similarity(cli_embeddings_array, ds_embeddings_array)
    
    print(f"   Similarity matrix shape: {similarities.shape}")
    print(f"   Mean cross-category similarity: {np.mean(similarities):.4f}")
    print(f"   Std cross-category similarity: {np.std(similarities):.4f}")
    print(f"   Min cross-category similarity: {np.min(similarities):.4f}")
    print(f"   Max cross-category similarity: {np.max(similarities):.4f}")
    
    # Find most similar cross-category pairs
    max_similarity_indices = np.unravel_index(np.argsort(similarities.ravel())[-10:], similarities.shape)
    
    print(f"\n🏆 Top 10 Most Similar Cross-Category Pairs:")
    for i in range(10):
        cli_idx, ds_idx = max_similarity_indices[0][i], max_similarity_indices[1][i]
        similarity = similarities[cli_idx, ds_idx]
        print(f"   {cli_repos[cli_idx]:30s} <-> {ds_repos[ds_idx]:30s} : {similarity:.4f}")
    
    # Find least similar cross-category pairs
    min_similarity_indices = np.unravel_index(np.argsort(similarities.ravel())[:10], similarities.shape)
    
    print(f"\n📉 Top 10 Least Similar Cross-Category Pairs:")
    for i in range(10):
        cli_idx, ds_idx = min_similarity_indices[0][i], min_similarity_indices[1][i]
        similarity = similarities[cli_idx, ds_idx]
        print(f"   {cli_repos[cli_idx]:30s} <-> {ds_repos[ds_idx]:30s} : {similarity:.4f}")
    
    # Analyze similarity distribution
    print(f"\n📊 Similarity Distribution Analysis:")
    similarity_quartiles = np.percentile(similarities, [25, 50, 75])
    print(f"   25th percentile: {similarity_quartiles[0]:.4f}")
    print(f"   50th percentile (median): {similarity_quartiles[1]:.4f}")
    print(f"   75th percentile: {similarity_quartiles[2]:.4f}")
    
    return {
        'similarities': similarities,
        'cli_repos': cli_repos,
        'ds_repos': ds_repos,
        'max_similar_pairs': max_similarity_indices,
        'min_similar_pairs': min_similarity_indices
    }

def create_dimension_visualizations(dim_analysis, similarity_analysis, cli_embeddings, ds_embeddings):
    """Create comprehensive visualizations of dimension patterns and similarities."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Dimension-wise differences heatmap (first 100 dimensions)
    plt.subplot(3, 3, 1)
    differences_100 = dim_analysis['differences'][:100].reshape(10, 10)
    sns.heatmap(differences_100, cmap='RdBu_r', center=0, square=True, 
                cbar_kws={'label': 'DS - CLI Difference'})
    plt.title('Dimension Differences (First 100)\nRed=DS Higher, Blue=CLI Higher')
    plt.xlabel('Dimension (mod 10)')
    plt.ylabel('Dimension (div 10)')
    
    # 2. Top 20 most different dimensions
    plt.subplot(3, 3, 2)
    top_diffs = dim_analysis['differences'][dim_analysis['top_different']]
    plt.barh(range(20), top_diffs, color=['red' if x > 0 else 'blue' for x in top_diffs])
    plt.yticks(range(20), [f'Dim {d}' for d in dim_analysis['top_different']])
    plt.xlabel('Difference (DS - CLI)')
    plt.title('Top 20 Most Different Dimensions')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # 3. Dimension-wise standard deviations comparison
    plt.subplot(3, 3, 3)
    cli_embeddings_array = np.array([data['embedding'] for data in cli_embeddings.values()])
    ds_embeddings_array = np.array([data['embedding'] for data in ds_embeddings.values()])
    cli_stds = np.std(cli_embeddings_array, axis=0)
    ds_stds = np.std(ds_embeddings_array, axis=0)
    plt.plot(cli_stds[:100], label='CLI Tools', alpha=0.7)
    plt.plot(ds_stds[:100], label='Data Science', alpha=0.7)
    plt.xlabel('Dimension')
    plt.ylabel('Standard Deviation')
    plt.title('Dimension-wise Standard Deviations (First 100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Cross-category similarity distribution
    plt.subplot(3, 3, 4)
    similarities_flat = similarity_analysis['similarities'].ravel()
    plt.hist(similarities_flat, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Cross-Category Similarity Distribution')
    plt.axvline(np.mean(similarities_flat), color='red', linestyle='--', label=f'Mean: {np.mean(similarities_flat):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Similarity matrix heatmap
    plt.subplot(3, 3, 5)
    sns.heatmap(similarity_analysis['similarities'], cmap='viridis', 
                xticklabels=False, yticklabels=False, cbar_kws={'label': 'Cosine Similarity'})
    plt.title('Cross-Category Similarity Matrix')
    plt.xlabel('Data Science Repositories')
    plt.ylabel('CLI Tools Repositories')
    
    # 6. Dimension correlation analysis
    plt.subplot(3, 3, 6)
    cli_embeddings_array = np.array([data['embedding'] for data in cli_embeddings.values()])
    ds_embeddings_array = np.array([data['embedding'] for data in ds_embeddings.values()])
    
    # Calculate correlation between CLI and DS mean embeddings
    correlation = np.corrcoef(dim_analysis['cli_mean'], dim_analysis['ds_mean'])[0, 1]
    plt.scatter(dim_analysis['cli_mean'][:100], dim_analysis['ds_mean'][:100], alpha=0.6)
    plt.xlabel('CLI Tools Mean Embedding')
    plt.ylabel('Data Science Mean Embedding')
    plt.title(f'Dimension Correlation (First 100)\nCorrelation: {correlation:.4f}')
    plt.grid(True, alpha=0.3)
    
    # 7. PCA visualization with dimension differences highlighted
    plt.subplot(3, 3, 7)
    all_embeddings = np.vstack([cli_embeddings_array, ds_embeddings_array])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_embeddings)
    
    cli_pca = pca_result[:len(cli_embeddings_array)]
    ds_pca = pca_result[len(cli_embeddings_array):]
    
    plt.scatter(cli_pca[:, 0], cli_pca[:, 1], alpha=0.7, label='CLI Tools', color='skyblue', s=50)
    plt.scatter(ds_pca[:, 0], ds_pca[:, 1], alpha=0.7, label='Data Science', color='lightcoral', s=50)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA: CLI Tools vs Data Science')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Dimension importance ranking
    plt.subplot(3, 3, 8)
    importance_scores = np.abs(dim_analysis['differences'])
    top_important_dims = np.argsort(importance_scores)[-20:]
    top_importance_scores = importance_scores[top_important_dims]
    
    plt.barh(range(20), top_importance_scores, color='purple', alpha=0.7)
    plt.yticks(range(20), [f'Dim {d}' for d in top_important_dims])
    plt.xlabel('Absolute Difference')
    plt.title('Top 20 Most Important Dimensions\nfor Category Discrimination')
    plt.grid(True, alpha=0.3)
    
    # 9. Similarity vs file count relationship
    plt.subplot(3, 3, 9)
    cli_file_counts = [data['num_files'] for data in cli_embeddings.values()]
    ds_file_counts = [data['num_files'] for data in ds_embeddings.values()]
    
    # For each CLI repo, plot its similarity to DS repos vs DS file counts
    for i, cli_repo in enumerate(cli_embeddings.keys()):
        similarities_to_ds = similarity_analysis['similarities'][i, :]
        plt.scatter(ds_file_counts, similarities_to_ds, alpha=0.5, s=30)
    
    plt.xlabel('Data Science Repository File Count')
    plt.ylabel('Similarity to CLI Tools')
    plt.title('Similarity vs File Count Relationship')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('codebert_dimension_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualizations saved as: codebert_dimension_analysis.png")

def main():
    """Main analysis function."""
    
    print("🚀 CodeBERT Embeddings: Dimension Patterns & Cross-Category Analysis")
    print("=" * 80)
    
    # Load embeddings
    cli_embeddings, ds_embeddings = load_embeddings()
    
    if not cli_embeddings or not ds_embeddings:
        print("❌ No embeddings found!")
        return
    
    print(f"✅ Loaded {len(cli_embeddings)} CLI Tools repositories")
    print(f"✅ Loaded {len(ds_embeddings)} Data Science repositories")
    
    # Analyze dimension patterns
    dim_analysis = analyze_dimension_patterns(cli_embeddings, ds_embeddings)
    
    # Analyze cross-category similarities
    similarity_analysis = analyze_cross_category_similarities(cli_embeddings, ds_embeddings)
    
    # Create visualizations
    print(f"\n📊 Creating comprehensive visualizations...")
    create_dimension_visualizations(dim_analysis, similarity_analysis, cli_embeddings, ds_embeddings)
    
    # Summary insights
    print(f"\n🎯 Key Insights:")
    print(f"   • {len(dim_analysis['significant_dimensions'])} dimensions show significant differences")
    print(f"   • Cross-category similarities range from {np.min(similarity_analysis['similarities']):.3f} to {np.max(similarity_analysis['similarities']):.3f}")
    print(f"   • Some CLI Tools and Data Science repos have surprising similarities")
    print(f"   • Dimension patterns reveal architectural signatures")
    print(f"   • These patterns can be used for robust classification")

if __name__ == "__main__":
    main()
