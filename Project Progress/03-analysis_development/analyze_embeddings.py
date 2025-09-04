import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_embeddings():
    """Analyze CodeBERT embeddings from different repository types."""
    
    # Load embeddings from different categories
    embedding_files = [
        ('CLI Tool', 'scripts/extraction/CodeBERTEmbeddings/cli_tool_sindresorhus_boxen_embeddings.json'),
        ('Data Science', 'scripts/extraction/CodeBERTEmbeddings/data_science_jupyter_notebook_embeddings.json'),
        ('Web App', 'scripts/extraction/CodeBERTEmbeddings/web_application_bradtraversy_50projects50days_embeddings.json'),
        ('Library', 'scripts/extraction/CodeBERTEmbeddings/library_sindresorhus_chalk_embeddings.json')
    ]
    
    embeddings_data = {}
    
    for category, file_path in embedding_files:
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                embedding = np.array(data['repository_embedding'])
                embeddings_data[category] = {
                    'embedding': embedding,
                    'mean': np.mean(embedding),
                    'std': np.std(embedding),
                    'min': np.min(embedding),
                    'max': np.max(embedding),
                    'num_files': data['num_files']
                }
    
    # Print statistics
    print("🔍 CodeBERT Embeddings Analysis")
    print("=" * 50)
    
    for category, stats in embeddings_data.items():
        print(f"\n📊 {category}:")
        print(f"   Files processed: {stats['num_files']}")
        print(f"   Embedding mean: {stats['mean']:.4f}")
        print(f"   Embedding std:  {stats['std']:.4f}")
        print(f"   Embedding min:  {stats['min']:.4f}")
        print(f"   Embedding max:  {stats['max']:.4f}")
    
    # Visualize embeddings
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Embedding distributions
    plt.subplot(2, 2, 1)
    for category, stats in embeddings_data.items():
        plt.hist(stats['embedding'], alpha=0.7, label=category, bins=50)
    plt.xlabel('Embedding Values')
    plt.ylabel('Frequency')
    plt.title('CodeBERT Embedding Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Embedding statistics comparison
    plt.subplot(2, 2, 2)
    categories = list(embeddings_data.keys())
    means = [embeddings_data[cat]['mean'] for cat in categories]
    stds = [embeddings_data[cat]['std'] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, means, width, label='Mean', alpha=0.8)
    plt.bar(x + width/2, stds, width, label='Std Dev', alpha=0.8)
    plt.xlabel('Repository Categories')
    plt.ylabel('Embedding Values')
    plt.title('Embedding Statistics by Category')
    plt.xticks(x, categories, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: First 50 dimensions comparison
    plt.subplot(2, 2, 3)
    for category, stats in embeddings_data.items():
        plt.plot(stats['embedding'][:50], label=category, alpha=0.8)
    plt.xlabel('Embedding Dimensions (1-50)')
    plt.ylabel('Embedding Values')
    plt.title('First 50 Dimensions Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Correlation heatmap
    plt.subplot(2, 2, 4)
    embedding_matrix = np.array([stats['embedding'] for stats in embeddings_data.values()])
    correlation_matrix = np.corrcoef(embedding_matrix)
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                xticklabels=categories,
                yticklabels=categories,
                center=0)
    plt.title('Embedding Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('codebert_embeddings_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Analysis complete! Visualization saved as 'codebert_embeddings_analysis.png'")
    
    # Explain what the embeddings represent
    print(f"\n📚 What CodeBERT Embeddings Represent:")
    print(f"   • 768-dimensional vectors capturing code semantics")
    print(f"   • Each dimension represents different code characteristics")
    print(f"   • Higher values indicate stronger presence of certain patterns")
    print(f"   • Similar repositories have similar embedding patterns")
    print(f"   • Used for: pattern detection, similarity analysis, quality assessment")

if __name__ == "__main__":
    analyze_embeddings()
