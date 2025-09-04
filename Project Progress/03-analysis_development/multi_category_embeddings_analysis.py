"""
Multi-Category CodeBERT Embeddings Analysis
==========================================

This script performs comprehensive analysis of CodeBERT embeddings across multiple
repository categories to understand architectural differences and similarities.

Categories analyzed:
- Data Science vs Educational
- Game Development vs Mobile Apps  
- Web Applications vs Libraries
- CLI Tools vs All Other Categories
- Cross-category similarity matrix
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
from itertools import combinations

def load_all_embeddings():
    """Load embeddings from all available categories."""
    
    embeddings_dir = Path("scripts/extraction/CodeBERTEmbeddings")
    all_embeddings = {}
    
    # Define category patterns
    category_patterns = {
        'cli_tool': 'cli_tool_*_embeddings.json',
        'data_science': 'data_science_*_embeddings.json',
        'educational': 'educational_*_embeddings.json',
        'game_development': 'game_development_*_embeddings.json',
        'mobile_app': 'mobile_app_*_embeddings.json',
        'web_application': 'web_application_*_embeddings.json',
        'library': 'library_*_embeddings.json'
    }
    
    for category, pattern in category_patterns.items():
        category_embeddings = {}
        for file_path in embeddings_dir.glob(pattern):
            repo_name = file_path.stem.replace("_embeddings", "").replace(f"{category}_", "")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    category_embeddings[repo_name] = {
                        'embedding': np.array(data['repository_embedding']),
                        'num_files': data['num_files'],
                        'category': category
                    }
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if category_embeddings:
            all_embeddings[category] = category_embeddings
            print(f"✅ Loaded {len(category_embeddings)} {category} repositories")
    
    return all_embeddings

def analyze_category_pair(cat1_name, cat1_embeddings, cat2_name, cat2_embeddings):
    """Analyze a pair of categories for differences and similarities."""
    
    print(f"\n🔍 {cat1_name.upper()} vs {cat2_name.upper()} Analysis")
    print("=" * 60)
    
    # Convert to numpy arrays
    cat1_array = np.array([data['embedding'] for data in cat1_embeddings.values()])
    cat2_array = np.array([data['embedding'] for data in cat2_embeddings.values()])
    
    # Calculate mean embeddings
    cat1_mean = np.mean(cat1_array, axis=0)
    cat2_mean = np.mean(cat2_array, axis=0)
    
    # Calculate differences
    differences = cat2_mean - cat1_mean
    abs_differences = np.abs(differences)
    
    # Statistical significance test
    significant_dimensions = []
    p_values = []
    
    for i in range(768):
        cat1_dim_values = cat1_array[:, i]
        cat2_dim_values = cat2_array[:, i]
        t_stat, p_val = stats.ttest_ind(cat1_dim_values, cat2_dim_values)
        if p_val < 0.05:
            significant_dimensions.append(i)
            p_values.append(p_val)
    
    # Cross-category similarities
    similarities = cosine_similarity(cat1_array, cat2_array)
    
    # Summary statistics
    print(f"📊 Summary:")
    print(f"   {cat1_name} repositories: {len(cat1_embeddings)}")
    print(f"   {cat2_name} repositories: {len(cat2_embeddings)}")
    print(f"   Significant dimensions: {len(significant_dimensions)} ({len(significant_dimensions)/768*100:.1f}%)")
    print(f"   Mean similarity: {np.mean(similarities):.4f}")
    print(f"   Similarity range: {np.min(similarities):.4f} - {np.max(similarities):.4f}")
    
    # Top different dimensions
    top_different = np.argsort(abs_differences)[-10:]
    print(f"\n🏆 Top 10 Most Different Dimensions:")
    for i, dim in enumerate(reversed(top_different)):
        diff = differences[dim]
        print(f"   Dimension {dim:3d}: {cat1_name}={cat1_mean[dim]:8.4f}, {cat2_name}={cat2_mean[dim]:8.4f}, Diff={diff:8.4f}")
    
    return {
        'cat1_mean': cat1_mean,
        'cat2_mean': cat2_mean,
        'differences': differences,
        'significant_dimensions': significant_dimensions,
        'similarities': similarities,
        'cat1_repos': list(cat1_embeddings.keys()),
        'cat2_repos': list(cat2_embeddings.keys())
    }

def create_multi_category_visualizations(all_embeddings, pair_analyses):
    """Create comprehensive visualizations for multi-category analysis."""
    
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Cross-category similarity matrix
    plt.subplot(3, 4, 1)
    categories = list(all_embeddings.keys())
    n_categories = len(categories)
    
    # Calculate mean similarity between each category pair
    similarity_matrix = np.zeros((n_categories, n_categories))
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Use the pair analysis if available
                pair_key = f"{cat1}_vs_{cat2}"
                if pair_key in pair_analyses:
                    similarity_matrix[i, j] = np.mean(pair_analyses[pair_key]['similarities'])
                else:
                    # Calculate directly
                    cat1_embeddings = np.array([data['embedding'] for data in all_embeddings[cat1].values()])
                    cat2_embeddings = np.array([data['embedding'] for data in all_embeddings[cat2].values()])
                    similarities = cosine_similarity(cat1_embeddings, cat2_embeddings)
                    similarity_matrix[i, j] = np.mean(similarities)
    
    sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=categories, yticklabels=categories, cbar_kws={'label': 'Mean Similarity'})
    plt.title('Cross-Category Similarity Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 2. Significant dimensions comparison
    plt.subplot(3, 4, 2)
    significant_counts = []
    category_names = []
    for pair_key, analysis in pair_analyses.items():
        significant_counts.append(len(analysis['significant_dimensions']))
        category_names.append(pair_key.replace('_vs_', ' vs '))
    
    plt.barh(range(len(significant_counts)), significant_counts, color='skyblue')
    plt.yticks(range(len(category_names)), category_names)
    plt.xlabel('Number of Significant Dimensions')
    plt.title('Significant Dimensions by Category Pair')
    plt.grid(True, alpha=0.3)
    
    # 3. Mean similarities comparison
    plt.subplot(3, 4, 3)
    mean_similarities = [np.mean(analysis['similarities']) for analysis in pair_analyses.values()]
    plt.barh(range(len(mean_similarities)), mean_similarities, color='lightcoral')
    plt.yticks(range(len(category_names)), category_names)
    plt.xlabel('Mean Similarity')
    plt.title('Mean Similarities by Category Pair')
    plt.grid(True, alpha=0.3)
    
    # 4. PCA visualization of all categories
    plt.subplot(3, 4, 4)
    all_embeddings_combined = []
    all_labels = []
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    for i, (category, embeddings) in enumerate(all_embeddings.items()):
        category_embeddings = np.array([data['embedding'] for data in embeddings.values()])
        all_embeddings_combined.append(category_embeddings)
        all_labels.extend([category] * len(embeddings))
    
    all_embeddings_array = np.vstack(all_embeddings_combined)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_embeddings_array)
    
    for i, category in enumerate(categories):
        category_mask = [label == category for label in all_labels]
        category_pca = pca_result[category_mask]
        plt.scatter(category_pca[:, 0], category_pca[:, 1], 
                   label=category, color=colors[i], alpha=0.7, s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA: All Categories')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 5-8. Individual pair comparisons (dimension differences)
    pair_plots = [(5, 'data_science_vs_educational'), (6, 'game_development_vs_mobile_app'),
                  (7, 'web_application_vs_library'), (8, 'cli_tool_vs_data_science')]
    
    for plot_pos, pair_key in pair_plots:
        if pair_key in pair_analyses:
            plt.subplot(3, 4, plot_pos)
            analysis = pair_analyses[pair_key]
            top_diffs = analysis['differences'][np.argsort(np.abs(analysis['differences']))[-20:]]
            plt.barh(range(20), top_diffs, 
                    color=['red' if x > 0 else 'blue' for x in top_diffs])
            plt.xlabel('Difference')
            plt.title(f'{pair_key.replace("_vs_", " vs ").title()}\nTop 20 Differences')
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.grid(True, alpha=0.3)
    
    # 9. Similarity distribution across all pairs
    plt.subplot(3, 4, 9)
    all_similarities = []
    for analysis in pair_analyses.values():
        all_similarities.extend(analysis['similarities'].ravel())
    
    plt.hist(all_similarities, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Overall Similarity Distribution\nAcross All Category Pairs')
    plt.axvline(np.mean(all_similarities), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_similarities):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Category size comparison
    plt.subplot(3, 4, 10)
    category_sizes = [len(embeddings) for embeddings in all_embeddings.values()]
    plt.bar(categories, category_sizes, color='orange', alpha=0.7)
    plt.xlabel('Category')
    plt.ylabel('Number of Repositories')
    plt.title('Repository Count by Category')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 11. Dimension importance heatmap
    plt.subplot(3, 4, 11)
    # Create a matrix of dimension differences for each pair
    diff_matrix = np.zeros((len(pair_analyses), 768))
    pair_names = list(pair_analyses.keys())
    
    for i, (pair_key, analysis) in enumerate(pair_analyses.items()):
        diff_matrix[i, :] = np.abs(analysis['differences'])
    
    # Show only first 100 dimensions for clarity
    sns.heatmap(diff_matrix[:, :100], cmap='viridis', 
                xticklabels=False, yticklabels=[name.replace('_vs_', ' vs ') for name in pair_names],
                cbar_kws={'label': 'Absolute Difference'})
    plt.title('Dimension Differences Heatmap\n(First 100 dimensions)')
    plt.ylabel('Category Pairs')
    
    # 12. Summary statistics
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    # Calculate summary statistics
    total_repos = sum(len(embeddings) for embeddings in all_embeddings.values())
    avg_significant_dims = np.mean([len(analysis['significant_dimensions']) for analysis in pair_analyses.values()])
    avg_similarity = np.mean([np.mean(analysis['similarities']) for analysis in pair_analyses.values()])
    
    summary_text = f"""
Multi-Category Analysis Summary

Total Repositories: {total_repos}
Categories Analyzed: {len(categories)}
Category Pairs: {len(pair_analyses)}

Average Significant Dimensions: {avg_significant_dims:.1f}
Average Cross-Category Similarity: {avg_similarity:.4f}

Categories with Most Repositories:
{max(all_embeddings.items(), key=lambda x: len(x[1]))[0]}: {max(len(embeddings) for embeddings in all_embeddings.values())}

Categories with Least Repositories:
{min(all_embeddings.items(), key=lambda x: len(x[1]))[0]}: {min(len(embeddings) for embeddings in all_embeddings.values())}

Most Different Pair:
{max(pair_analyses.items(), key=lambda x: len(x[1]['significant_dimensions']))[0]}

Least Different Pair:
{min(pair_analyses.items(), key=lambda x: len(x[1]['significant_dimensions']))[0]}
"""
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('multi_category_embeddings_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Multi-category visualizations saved as: multi_category_embeddings_analysis.png")

def main():
    """Main analysis function."""
    
    print("🚀 Multi-Category CodeBERT Embeddings Analysis")
    print("=" * 80)
    
    # Load all embeddings
    all_embeddings = load_all_embeddings()
    
    if not all_embeddings:
        print("❌ No embeddings found!")
        return
    
    print(f"\n📊 Loaded {len(all_embeddings)} categories")
    
    # Define category pairs to analyze
    category_pairs = [
        ('data_science', 'educational'),
        ('game_development', 'mobile_app'),
        ('web_application', 'library'),
        ('cli_tool', 'data_science'),
        ('cli_tool', 'educational'),
        ('game_development', 'web_application'),
        ('mobile_app', 'library')
    ]
    
    pair_analyses = {}
    
    # Analyze each pair
    for cat1, cat2 in category_pairs:
        if cat1 in all_embeddings and cat2 in all_embeddings:
            pair_key = f"{cat1}_vs_{cat2}"
            analysis = analyze_category_pair(cat1, all_embeddings[cat1], 
                                           cat2, all_embeddings[cat2])
            pair_analyses[pair_key] = analysis
    
    # Create comprehensive visualizations
    print(f"\n📊 Creating multi-category visualizations...")
    create_multi_category_visualizations(all_embeddings, pair_analyses)
    
    # Summary insights
    print(f"\n🎯 Key Insights:")
    print(f"   • Analyzed {len(pair_analyses)} category pairs")
    print(f"   • Total repositories: {sum(len(embeddings) for embeddings in all_embeddings.values())}")
    
    # Find most and least different pairs
    if pair_analyses:
        most_different = max(pair_analyses.items(), 
                           key=lambda x: len(x[1]['significant_dimensions']))
        least_different = min(pair_analyses.items(), 
                            key=lambda x: len(x[1]['significant_dimensions']))
        
        print(f"   • Most different pair: {most_different[0]} ({len(most_different[1]['significant_dimensions'])} significant dimensions)")
        print(f"   • Least different pair: {least_different[0]} ({len(least_different[1]['significant_dimensions'])} significant dimensions)")
        
        # Average statistics
        avg_significant = np.mean([len(analysis['significant_dimensions']) for analysis in pair_analyses.values()])
        avg_similarity = np.mean([np.mean(analysis['similarities']) for analysis in pair_analyses.values()])
        
        print(f"   • Average significant dimensions: {avg_significant:.1f}")
        print(f"   • Average cross-category similarity: {avg_similarity:.4f}")

if __name__ == "__main__":
    main()
