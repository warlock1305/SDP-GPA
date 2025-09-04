import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
import json
import re

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# === CONFIGURATION ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PREDICTIONS_DIR = os.path.join(BASE_DIR, "Code2VecPredictions")
OUTPUT_DIR = os.path.join(BASE_DIR, "Code2VecVisualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_prediction_file(file_path):
    """Parse a single prediction file and extract structured data."""
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract source file path
        source_match = re.search(r'# Source: (.+)', content)
        source_file = source_match.group(1) if source_match else "Unknown"
        
        # Extract number of path contexts
        contexts_match = re.search(r'# Extracted (\d+) path contexts', content)
        num_contexts = int(contexts_match.group(1)) if contexts_match else 0
        
        # Parse each method prediction
        method_sections = content.split('\n\n')
        
        for section in method_sections:
            if 'Method ' in section and 'Original name:' in section:
                lines = section.strip().split('\n')
                
                # Extract method number
                method_num_match = re.search(r'Method (\d+):', lines[0])
                method_num = int(method_num_match.group(1)) if method_num_match else 0
                
                # Extract original name
                original_name = ""
                predictions = []
                attention_paths = []
                
                for line in lines:
                    if line.startswith('  Original name:'):
                        original_name = line.replace('  Original name:', '').strip()
                    elif line.startswith('    (') and ')' in line and '[' in line:
                        # Parse prediction line
                        prob_match = re.search(r'\(([\d.]+)\)', line)
                        name_match = re.search(r'\[(.+)\]', line)
                        if prob_match and name_match:
                            prob = float(prob_match.group(1))
                            name = name_match.group(1).replace("'", "").replace(", ", "|")
                            predictions.append({'probability': prob, 'name': name})
                    elif '\t' in line and ',' in line:
                        # Parse attention path
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            score = float(parts[0])
                            path_info = parts[1]
                            attention_paths.append({'score': score, 'path': path_info})
                
                if original_name:
                    results.append({
                        'source_file': source_file,
                        'method_num': method_num,
                        'original_name': original_name,
                        'num_contexts': num_contexts,
                        'predictions': predictions,
                        'attention_paths': attention_paths,
                        'top_prediction': predictions[0] if predictions else None,
                        'top_confidence': predictions[0]['probability'] if predictions else 0
                    })
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return results


def collect_all_predictions():
    """Collect all prediction data from the Code2VecPredictions directory."""
    all_results = []
    
    for root, _, files in os.walk(PREDICTIONS_DIR):
        for file in files:
            if file.endswith('.pred.txt'):
                file_path = os.path.join(root, file)
                results = parse_prediction_file(file_path)
                all_results.extend(results)
    
    return all_results


def create_confidence_distribution_plot(data):
    """Create a histogram of prediction confidence scores."""
    confidences = [item['top_confidence'] for item in data if item['top_confidence'] > 0]
    
    plt.figure(figsize=(12, 6))
    plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Methods')
    plt.title('Distribution of Code2Vec Prediction Confidence Scores')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_conf = np.mean(confidences)
    median_conf = np.median(confidences)
    plt.axvline(mean_conf, color='red', linestyle='--', label=f'Mean: {mean_conf:.3f}')
    plt.axvline(median_conf, color='orange', linestyle='--', label=f'Median: {median_conf:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return confidences


def create_method_count_plot(data):
    """Create a bar chart showing number of methods per file."""
    file_method_counts = defaultdict(int)
    
    for item in data:
        file_method_counts[item['source_file']] += 1
    
    # Get top 15 files by method count
    top_files = sorted(file_method_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    
    plt.figure(figsize=(14, 8))
    files, counts = zip(*top_files)
    file_names = [os.path.basename(f) for f in files]
    
    bars = plt.bar(range(len(file_names)), counts, color='lightcoral', alpha=0.8)
    plt.xlabel('Java Files')
    plt.ylabel('Number of Methods')
    plt.title('Number of Methods per Java File (Top 15)')
    plt.xticks(range(len(file_names)), file_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'methods_per_file.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_prediction_accuracy_analysis(data):
    """Analyze prediction accuracy and create visualizations."""
    # Analyze exact matches vs semantic similarity
    exact_matches = 0
    semantic_matches = 0
    total_methods = len(data)
    
    for item in data:
        original = item['original_name'].lower().replace('|', '').replace('_', '')
        top_pred = item['top_prediction']['name'].lower().replace('|', '').replace('_', '') if item['top_prediction'] else ''
        
        if original == top_pred:
            exact_matches += 1
        elif any(word in top_pred for word in original.split()) or any(word in original for word in top_pred.split()):
            semantic_matches += 1
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    labels = ['Exact Matches', 'Semantic Matches', 'No Match']
    sizes = [exact_matches, semantic_matches, total_methods - exact_matches - semantic_matches]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Code2Vec Prediction Accuracy Analysis')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return exact_matches, semantic_matches, total_methods


def create_attention_path_analysis(data):
    """Analyze attention paths and create visualizations."""
    # Collect all attention paths
    all_paths = []
    for item in data:
        for path in item['attention_paths']:
            all_paths.append({
                'score': path['score'],
                'path': path['path'],
                'method': item['original_name']
            })
    
    if not all_paths:
        return
    
    # Create histogram of attention scores
    scores = [path['score'] for path in all_paths]
    
    plt.figure(figsize=(12, 6))
    plt.hist(scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Attention Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Attention Path Scores')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'attention_scores_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find most common path patterns
    path_patterns = []
    for path in all_paths:
        # Extract the path part (between commas)
        path_parts = path['path'].split(',')
        if len(path_parts) >= 3:
            path_patterns.append(path_parts[1])  # The actual path
    
    if path_patterns:
        pattern_counts = Counter(path_patterns)
        top_patterns = pattern_counts.most_common(10)
        
        plt.figure(figsize=(14, 8))
        patterns, counts = zip(*top_patterns)
        
        # Truncate long pattern names for display
        display_patterns = [p[:30] + '...' if len(p) > 30 else p for p in patterns]
        
        bars = plt.bar(range(len(display_patterns)), counts, color='gold', alpha=0.8)
        plt.xlabel('AST Path Patterns')
        plt.ylabel('Frequency')
        plt.title('Most Common AST Path Patterns in Attention')
        plt.xticks(range(len(display_patterns)), display_patterns, rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'common_ast_paths.png'), dpi=300, bbox_inches='tight')
        plt.close()


def create_summary_report(data, confidences, exact_matches, semantic_matches, total_methods):
    """Create a comprehensive summary report."""
    report = {
        "summary": {
            "total_methods_analyzed": total_methods,
            "total_files_processed": len(set(item['source_file'] for item in data)),
            "average_confidence": np.mean(confidences),
            "median_confidence": np.median(confidences),
            "exact_match_rate": exact_matches / total_methods * 100,
            "semantic_match_rate": semantic_matches / total_methods * 100,
            "no_match_rate": (total_methods - exact_matches - semantic_matches) / total_methods * 100
        },
        "top_predictions": [],
        "file_statistics": {}
    }
    
    # Top predictions by confidence
    sorted_by_confidence = sorted(data, key=lambda x: x['top_confidence'], reverse=True)
    for item in sorted_by_confidence[:10]:
        report["top_predictions"].append({
            "method": item['original_name'],
            "prediction": item['top_prediction']['name'] if item['top_prediction'] else "None",
            "confidence": item['top_confidence'],
            "file": os.path.basename(item['source_file'])
        })
    
    # File statistics
    file_stats = defaultdict(lambda: {"methods": 0, "avg_confidence": 0, "confidences": []})
    for item in data:
        file_stats[item['source_file']]["methods"] += 1
        file_stats[item['source_file']]["confidences"].append(item['top_confidence'])
    
    for file, stats in file_stats.items():
        stats["avg_confidence"] = np.mean(stats["confidences"])
        report["file_statistics"][os.path.basename(file)] = {
            "methods": stats["methods"],
            "avg_confidence": stats["avg_confidence"]
        }
    
    # Save report
    with open(os.path.join(OUTPUT_DIR, 'code2vec_analysis_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create a text summary
    with open(os.path.join(OUTPUT_DIR, 'code2vec_summary.txt'), 'w') as f:
        f.write("=== Code2Vec Analysis Summary ===\n\n")
        f.write(f"Total Methods Analyzed: {total_methods}\n")
        f.write(f"Total Files Processed: {report['summary']['total_files_processed']}\n")
        f.write(f"Average Confidence: {report['summary']['average_confidence']:.3f}\n")
        f.write(f"Median Confidence: {report['summary']['median_confidence']:.3f}\n")
        f.write(f"Exact Match Rate: {report['summary']['exact_match_rate']:.1f}%\n")
        f.write(f"Semantic Match Rate: {report['summary']['semantic_match_rate']:.1f}%\n")
        f.write(f"No Match Rate: {report['summary']['no_match_rate']:.1f}%\n\n")
        
        f.write("=== Top 10 Predictions by Confidence ===\n")
        for i, pred in enumerate(report["top_predictions"], 1):
            f.write(f"{i}. {pred['method']} -> {pred['prediction']} ({pred['confidence']:.3f})\n")
    
    return report


def main():
    """Main function to run all visualizations."""
    print("Loading Code2Vec prediction data...")
    data = collect_all_predictions()
    
    if not data:
        print("No prediction data found!")
        return
    
    print(f"Found {len(data)} method predictions from {len(set(item['source_file'] for item in data))} files")
    
    print("Creating visualizations...")
    
    # 1. Confidence distribution
    print("1. Creating confidence distribution plot...")
    confidences = create_confidence_distribution_plot(data)
    
    # 2. Methods per file
    print("2. Creating methods per file plot...")
    create_method_count_plot(data)
    
    # 3. Prediction accuracy analysis
    print("3. Creating prediction accuracy analysis...")
    exact_matches, semantic_matches, total_methods = create_prediction_accuracy_analysis(data)
    
    # 4. Attention path analysis
    print("4. Creating attention path analysis...")
    create_attention_path_analysis(data)
    
    # 5. Summary report
    print("5. Creating summary report...")
    report = create_summary_report(data, confidences, exact_matches, semantic_matches, total_methods)
    
    print(f"\n✅ Visualizations completed!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Generated files:")
    print(f"   - confidence_distribution.png")
    print(f"   - methods_per_file.png") 
    print(f"   - prediction_accuracy.png")
    print(f"   - attention_scores_distribution.png")
    print(f"   - common_ast_paths.png")
    print(f"   - code2vec_analysis_report.json")
    print(f"   - code2vec_summary.txt")
    
    print(f"\n📈 Key Statistics:")
    print(f"   - Average confidence: {report['summary']['average_confidence']:.3f}")
    print(f"   - Exact match rate: {report['summary']['exact_match_rate']:.1f}%")
    print(f"   - Semantic match rate: {report['summary']['semantic_match_rate']:.1f}%")


if __name__ == "__main__":
    main() 