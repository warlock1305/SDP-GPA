"""
Visualize Comprehensive Repository Analysis Results
==================================================

This script creates visualizations and insights from the comprehensive
repository analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import numpy as np

def load_analysis_results():
    """Load the analysis results."""
    results_dir = Path("comprehensive_dataset_analysis")
    
    # Load summary data
    summary_df = pd.read_csv(results_dir / "analysis_summary.csv")
    
    # Load report
    with open(results_dir / "analysis_report.json", 'r') as f:
        report = json.load(f)
    
    return summary_df, report

def create_visualizations(summary_df, report):
    """Create comprehensive visualizations."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Category Distribution
    plt.subplot(3, 3, 1)
    category_counts = summary_df['category'].value_counts()
    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    plt.title('Dataset Category Distribution', fontsize=14, fontweight='bold')
    
    # 2. Architectural Pattern Distribution
    plt.subplot(3, 3, 2)
    arch_counts = summary_df['architectural_pattern'].value_counts()
    plt.bar(range(len(arch_counts)), arch_counts.values, color=sns.color_palette("husl", len(arch_counts)))
    plt.xticks(range(len(arch_counts)), arch_counts.index, rotation=45, ha='right')
    plt.title('Architectural Pattern Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    
    # 3. Quality Score Distribution
    plt.subplot(3, 3, 3)
    plt.hist(summary_df['quality_score'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(summary_df['quality_score'].mean(), color='red', linestyle='--', 
                label=f'Mean: {summary_df["quality_score"].mean():.3f}')
    plt.title('Quality Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 4. Experience Level Distribution
    plt.subplot(3, 3, 4)
    exp_counts = summary_df['experience_level'].value_counts()
    plt.pie(exp_counts.values, labels=exp_counts.index, autopct='%1.1f%%')
    plt.title('Programmer Experience Level Distribution', fontsize=14, fontweight='bold')
    
    # 5. Architecture Confidence vs Quality Score
    plt.subplot(3, 3, 5)
    plt.scatter(summary_df['arch_confidence'], summary_df['quality_score'], alpha=0.6)
    plt.xlabel('Architecture Confidence')
    plt.ylabel('Quality Score')
    plt.title('Architecture Confidence vs Quality Score', fontsize=14, fontweight='bold')
    
    # Add trend line
    z = np.polyfit(summary_df['arch_confidence'], summary_df['quality_score'], 1)
    p = np.poly1d(z)
    plt.plot(summary_df['arch_confidence'], p(summary_df['arch_confidence']), "r--", alpha=0.8)
    
    # 6. Quality Score by Category
    plt.subplot(3, 3, 6)
    quality_by_category = summary_df.groupby('category')['quality_score'].mean().sort_values(ascending=False)
    plt.bar(range(len(quality_by_category)), quality_by_category.values, 
            color=sns.color_palette("husl", len(quality_by_category)))
    plt.xticks(range(len(quality_by_category)), quality_by_category.index, rotation=45, ha='right')
    plt.title('Average Quality Score by Category', fontsize=14, fontweight='bold')
    plt.ylabel('Average Quality Score')
    
    # 7. Best Practices Distribution
    plt.subplot(3, 3, 7)
    practice_counts = summary_df['best_practices'].value_counts()
    plt.bar(range(len(practice_counts)), practice_counts.values, 
            color=sns.color_palette("husl", len(practice_counts)))
    plt.xticks(range(len(practice_counts)), practice_counts.index, rotation=45, ha='right')
    plt.title('Best Practices Adherence', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    
    # 8. Attention to Detail Distribution
    plt.subplot(3, 3, 8)
    attention_counts = summary_df['attention_to_detail'].value_counts()
    plt.pie(attention_counts.values, labels=attention_counts.index, autopct='%1.1f%%')
    plt.title('Attention to Detail Distribution', fontsize=14, fontweight='bold')
    
    # 9. Quality Score by Experience Level
    plt.subplot(3, 3, 9)
    quality_by_exp = summary_df.groupby('experience_level')['quality_score'].mean()
    plt.bar(range(len(quality_by_exp)), quality_by_exp.values, 
            color=sns.color_palette("husl", len(quality_by_exp)))
    plt.xticks(range(len(quality_by_exp)), quality_by_exp.index)
    plt.title('Average Quality Score by Experience Level', fontsize=14, fontweight='bold')
    plt.ylabel('Average Quality Score')
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_detailed_insights(summary_df, report):
    """Create detailed insights and analysis."""
    print("🔍 COMPREHENSIVE ANALYSIS INSIGHTS")
    print("=" * 60)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total repositories analyzed: {report['analysis_summary']['total_repositories']}")
    print(f"   Success rate: {report['analysis_summary']['success_rate']*100:.1f}%")
    print(f"   Average quality score: {report['quality_score_statistics']['mean']:.3f}")
    print(f"   Quality score range: {report['quality_score_statistics']['min']:.3f} - {report['quality_score_statistics']['max']:.3f}")
    
    # Category analysis
    print(f"\n📂 CATEGORY ANALYSIS:")
    for category, count in report['category_distribution'].items():
        avg_quality = summary_df[summary_df['category'] == category]['quality_score'].mean()
        print(f"   {category}: {count} repos, avg quality: {avg_quality:.3f}")
    
    # Architectural pattern analysis
    print(f"\n🏗️  ARCHITECTURAL PATTERN ANALYSIS:")
    for pattern, count in report['architectural_pattern_distribution'].items():
        avg_quality = summary_df[summary_df['architectural_pattern'] == pattern]['quality_score'].mean()
        avg_confidence = summary_df[summary_df['architectural_pattern'] == pattern]['arch_confidence'].mean()
        print(f"   {pattern}: {count} repos, avg quality: {avg_quality:.3f}, avg confidence: {avg_confidence:.3f}")
    
    # Experience level analysis
    print(f"\n👨‍💻 EXPERIENCE LEVEL ANALYSIS:")
    for level, count in report['experience_level_distribution'].items():
        avg_quality = summary_df[summary_df['experience_level'] == level]['quality_score'].mean()
        print(f"   {level}: {count} developers, avg quality: {avg_quality:.3f}")
    
    # Best practices analysis
    print(f"\n✅ BEST PRACTICES ANALYSIS:")
    practice_analysis = summary_df.groupby('best_practices').agg({
        'quality_score': ['mean', 'count'],
        'experience_level': lambda x: x.value_counts().index[0]
    }).round(3)
    print(practice_analysis)
    
    # Top performing repositories
    print(f"\n🏆 TOP 10 HIGHEST QUALITY REPOSITORIES:")
    top_repos = summary_df.nlargest(10, 'quality_score')[['repo_name', 'category', 'quality_score', 'experience_level']]
    for _, repo in top_repos.iterrows():
        print(f"   {repo['repo_name']}: {repo['quality_score']:.3f} ({repo['category']}, {repo['experience_level']})")
    
    # Bottom performing repositories
    print(f"\n⚠️  TOP 10 LOWEST QUALITY REPOSITORIES:")
    bottom_repos = summary_df.nsmallest(10, 'quality_score')[['repo_name', 'category', 'quality_score', 'experience_level']]
    for _, repo in bottom_repos.iterrows():
        print(f"   {repo['repo_name']}: {repo['quality_score']:.3f} ({repo['category']}, {repo['experience_level']})")
    
    # Correlation analysis
    print(f"\n📈 CORRELATION ANALYSIS:")
    correlations = summary_df[['quality_score', 'arch_confidence', 'cat_confidence']].corr()
    print("Quality Score Correlations:")
    print(f"   Architecture Confidence: {correlations.loc['quality_score', 'arch_confidence']:.3f}")
    print(f"   Category Confidence: {correlations.loc['quality_score', 'cat_confidence']:.3f}")
    
    # Model performance insights
    print(f"\n🤖 MODEL PERFORMANCE INSIGHTS:")
    print(f"   Architecture classifier average confidence: {report['confidence_statistics']['arch_confidence_mean']:.3f}")
    print(f"   Category classifier average confidence: {report['confidence_statistics']['cat_confidence_mean']:.3f}")
    
    # Category prediction accuracy
    print(f"\n🎯 CATEGORY PREDICTION ANALYSIS:")
    correct_predictions = (summary_df['category'] == summary_df['predicted_category']).sum()
    accuracy = correct_predictions / len(summary_df)
    print(f"   Category prediction accuracy: {accuracy:.1%}")
    print(f"   All repositories predicted as: {summary_df['predicted_category'].iloc[0]}")
    
    # Quality score distribution insights
    print(f"\n📊 QUALITY SCORE DISTRIBUTION INSIGHTS:")
    quality_quartiles = summary_df['quality_score'].quantile([0.25, 0.5, 0.75])
    print(f"   Q1 (25th percentile): {quality_quartiles[0.25]:.3f}")
    print(f"   Median (50th percentile): {quality_quartiles[0.5]:.3f}")
    print(f"   Q3 (75th percentile): {quality_quartiles[0.75]:.3f}")
    
    # High vs Low quality analysis
    high_quality = summary_df[summary_df['quality_score'] >= quality_quartiles[0.75]]
    low_quality = summary_df[summary_df['quality_score'] <= quality_quartiles[0.25]]
    
    print(f"\n🔝 HIGH QUALITY REPOSITORIES (Top 25%):")
    print(f"   Count: {len(high_quality)}")
    print(f"   Average experience level: {high_quality['experience_level'].mode().iloc[0]}")
    print(f"   Most common category: {high_quality['category'].mode().iloc[0]}")
    
    print(f"\n🔻 LOW QUALITY REPOSITORIES (Bottom 25%):")
    print(f"   Count: {len(low_quality)}")
    print(f"   Average experience level: {low_quality['experience_level'].mode().iloc[0]}")
    print(f"   Most common category: {low_quality['category'].mode().iloc[0]}")

def main():
    """Main function to run the visualization and analysis."""
    print("🚀 Loading Comprehensive Analysis Results...")
    
    # Load results
    summary_df, report = load_analysis_results()
    
    print(f"✅ Loaded analysis results for {len(summary_df)} repositories")
    
    # Create visualizations
    print("📊 Creating visualizations...")
    fig = create_visualizations(summary_df, report)
    
    # Generate detailed insights
    print("🔍 Generating detailed insights...")
    create_detailed_insights(summary_df, report)
    
    print(f"\n🎉 Analysis complete! Visualization saved as 'comprehensive_analysis_visualization.png'")

if __name__ == "__main__":
    main()
