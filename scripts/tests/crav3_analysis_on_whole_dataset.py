#!/usr/bin/env python3
"""
Comprehensive Repository Analyzer v3.0 - Whole Dataset Analysis
==============================================================

This script runs the Comprehensive Repository Analyzer v3.0 on the entire dataset,
analyzing all repositories across all categories and generating comprehensive reports.

Features:
- Analyzes all repositories in the dataset
- Generates individual analysis results
- Creates aggregated summary reports
- Provides performance metrics
- Saves results in organized structure
"""

import os
import sys
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# Add the parent directory to the path to import the analyzer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'analysis'))

from comprehensive_repository_analyzer_v3 import ComprehensiveRepositoryAnalyzerV3

class WholeDatasetAnalyzer:
    """Analyzer for running CRAv3 on the entire dataset."""
    
    def __init__(self):
        """Initialize the whole dataset analyzer."""
        self.analyzer = ComprehensiveRepositoryAnalyzerV3()
        self.dataset_root = Path("dataset")
        self.output_dir = Path("crav3_whole_dataset_analysis")
        self.results = []
        self.errors = []
        self.stats = {
            'total_repositories': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'start_time': None,
            'end_time': None,
            'total_duration': 0
        }
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organization
        (self.output_dir / "individual_results").mkdir(exist_ok=True)
        (self.output_dir / "summary_reports").mkdir(exist_ok=True)
        (self.output_dir / "aggregated_data").mkdir(exist_ok=True)
        
        print(f"🔧 Whole Dataset Analyzer initialized")
        print(f"📁 Dataset root: {self.dataset_root}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def get_all_repositories(self) -> List[str]:
        """Get all repository paths from the dataset."""
        repositories = []
        
        if not self.dataset_root.exists():
            print(f"❌ Dataset root not found: {self.dataset_root}")
            return repositories
        
        # Walk through all categories in the dataset
        for category_dir in self.dataset_root.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                print(f"📂 Found category: {category_name}")
                
                # Get repositories in this category
                for repo_dir in category_dir.iterdir():
                    if repo_dir.is_dir():
                        repo_path = str(repo_dir)
                        repositories.append(repo_path)
                        print(f"   📁 Found repository: {repo_dir.name}")
        
        print(f"📊 Total repositories found: {len(repositories)}")
        return repositories
    
    def analyze_repository(self, repo_path: str) -> Optional[Dict[str, Any]]:
        """Analyze a single repository with error handling."""
        repo_name = Path(repo_path).name
        category = Path(repo_path).parent.name
        
        print(f"\n🔍 Analyzing: {category}/{repo_name}")
        print(f"   📍 Path: {repo_path}")
        
        try:
            # Run the analysis
            start_time = time.time()
            results = self.analyzer.analyze_repository(repo_path)
            analysis_time = time.time() - start_time
            
            # Add metadata
            results['metadata'] = {
                'category': category,
                'repository_name': repo_name,
                'analysis_time_seconds': analysis_time,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Save individual result
            output_file = self.output_dir / "individual_results" / f"{category}_{repo_name}_analysis.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"   ✅ Analysis completed in {analysis_time:.2f}s")
            print(f"   💾 Saved to: {output_file}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error analyzing {category}/{repo_name}: {str(e)}"
            print(f"   ❌ {error_msg}")
            
            # Log error details
            error_info = {
                'category': category,
                'repository_name': repo_name,
                'repository_path': repo_path,
                'error_message': str(e),
                'error_traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
            
            self.errors.append(error_info)
            return None
    
    def analyze_whole_dataset(self):
        """Analyze the entire dataset."""
        print("🚀 Starting Comprehensive Repository Analyzer v3.0 - Whole Dataset Analysis")
        print("=" * 80)
        
        # Record start time
        self.stats['start_time'] = datetime.now()
        
        # Get all repositories
        repositories = self.get_all_repositories()
        self.stats['total_repositories'] = len(repositories)
        
        if not repositories:
            print("❌ No repositories found to analyze!")
            return
        
        print(f"\n📊 Starting analysis of {len(repositories)} repositories...")
        print("-" * 80)
        
        # Analyze each repository
        for i, repo_path in enumerate(repositories, 1):
            print(f"\n[{i}/{len(repositories)}] ", end="")
            
            result = self.analyze_repository(repo_path)
            if result:
                self.results.append(result)
                self.stats['successful_analyses'] += 1
            else:
                self.stats['failed_analyses'] += 1
        
        # Record end time
        self.stats['end_time'] = datetime.now()
        self.stats['total_duration'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        # Generate reports
        self.generate_reports()
        
        # Print final summary
        self.print_final_summary()
    
    def generate_reports(self):
        """Generate comprehensive reports from the analysis results."""
        print(f"\n📊 Generating reports...")
        
        if not self.results:
            print("❌ No successful analyses to report on!")
            return
        
        # 1. Create aggregated data
        self.create_aggregated_data()
        
        # 2. Generate summary reports
        self.generate_summary_reports()
        
        # 3. Create performance analysis
        self.create_performance_analysis()
        
        # 4. Generate error report
        self.generate_error_report()
        
        print("✅ All reports generated successfully!")
    
    def create_aggregated_data(self):
        """Create aggregated data from all analyses."""
        print("   📈 Creating aggregated data...")
        
        # Extract key metrics from all results
        aggregated_data = []
        
        for result in self.results:
            metadata = result.get('metadata', {})
            quality_assessment = result.get('quality_assessment', {})
            programmer_chars = result.get('programmer_characteristics', {})
            architecture_analysis = result.get('architecture_analysis', {})
            
            # Create summary row
            summary_row = {
                'category': metadata.get('category', 'unknown'),
                'repository_name': metadata.get('repository_name', 'unknown'),
                'analysis_time_seconds': metadata.get('analysis_time_seconds', 0),
                'overall_quality_score': quality_assessment.get('overall_score', 0),
                'experience_level': programmer_chars.get('experience_level', 'unknown'),
                'coding_style': programmer_chars.get('coding_style', 'unknown'),
                'best_practices': programmer_chars.get('best_practices', 'unknown'),
                'detected_patterns': ', '.join(architecture_analysis.get('detected_patterns', [])),
                'analysis_timestamp': metadata.get('analysis_timestamp', '')
            }
            
            # Add quality breakdown
            code_quality = quality_assessment.get('code_quality', {})
            arch_quality = quality_assessment.get('architecture_quality', {})
            doc_quality = quality_assessment.get('documentation_quality', {})
            maint_quality = quality_assessment.get('maintainability', {})
            
            summary_row.update({
                'code_complexity': code_quality.get('complexity', 0),
                'code_structure': code_quality.get('structure', 0),
                'code_organization': code_quality.get('organization', 0),
                'code_consistency': code_quality.get('consistency', 0),
                'arch_modularity': arch_quality.get('modularity', 0),
                'arch_abstraction': arch_quality.get('abstraction', 0),
                'arch_separation': arch_quality.get('separation', 0),
                'arch_scalability': arch_quality.get('scalability', 0),
                'doc_readme': doc_quality.get('readme_presence', 0),
                'doc_config': doc_quality.get('config_documentation', 0),
                'doc_dependencies': doc_quality.get('dependency_documentation', 0),
                'maint_test_coverage': maint_quality.get('test_coverage', 0),
                'maint_organization': maint_quality.get('code_organization', 0),
                'maint_structure_clarity': maint_quality.get('structure_clarity', 0),
                'maint_consistency': maint_quality.get('consistency', 0)
            })
            
            aggregated_data.append(summary_row)
        
        # Save as CSV
        df = pd.DataFrame(aggregated_data)
        csv_file = self.output_dir / "aggregated_data" / "analysis_summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"   💾 Aggregated data saved to: {csv_file}")
        
        # Save as JSON for programmatic access
        json_file = self.output_dir / "aggregated_data" / "analysis_summary.json"
        with open(json_file, 'w') as f:
            json.dump(aggregated_data, f, indent=2, default=str)
        print(f"   💾 JSON summary saved to: {json_file}")
    
    def generate_summary_reports(self):
        """Generate summary reports with statistics and insights."""
        print("   📋 Generating summary reports...")
        
        # Calculate statistics
        total_repos = len(self.results)
        quality_scores = [r.get('quality_assessment', {}).get('overall_score', 0) for r in self.results]
        experience_levels = [r.get('programmer_characteristics', {}).get('experience_level', 'unknown') for r in self.results]
        coding_styles = [r.get('programmer_characteristics', {}).get('coding_style', 'unknown') for r in self.results]
        
        # Collect all detected patterns
        all_patterns = []
        for result in self.results:
            patterns = result.get('architecture_analysis', {}).get('detected_patterns', [])
            all_patterns.extend(patterns)
        
        # Create summary report
        summary_report = {
            'analysis_metadata': {
                'total_repositories': total_repos,
                'successful_analyses': self.stats['successful_analyses'],
                'failed_analyses': self.stats['failed_analyses'],
                'success_rate': self.stats['successful_analyses'] / self.stats['total_repositories'] if self.stats['total_repositories'] > 0 else 0,
                'total_duration_seconds': self.stats['total_duration'],
                'average_analysis_time': sum(r.get('metadata', {}).get('analysis_time_seconds', 0) for r in self.results) / len(self.results) if self.results else 0
            },
            'quality_statistics': {
                'average_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                'min_quality_score': min(quality_scores) if quality_scores else 0,
                'max_quality_score': max(quality_scores) if quality_scores else 0,
                'quality_score_distribution': {
                    'excellent': len([s for s in quality_scores if s >= 0.8]),
                    'good': len([s for s in quality_scores if 0.6 <= s < 0.8]),
                    'fair': len([s for s in quality_scores if 0.4 <= s < 0.6]),
                    'poor': len([s for s in quality_scores if s < 0.4])
                }
            },
            'experience_level_distribution': {
                'junior': experience_levels.count('Junior'),
                'intermediate': experience_levels.count('Intermediate'),
                'senior': experience_levels.count('Senior')
            },
            'coding_style_distribution': {
                'professional': coding_styles.count('Professional'),
                'simple_and_clean': coding_styles.count('Simple and Clean'),
                'comprehensive': coding_styles.count('Comprehensive'),
                'basic': coding_styles.count('Basic')
            },
            'architecture_patterns': {
                'total_patterns_detected': len(all_patterns),
                'unique_patterns': list(set(all_patterns)),
                'pattern_frequency': {pattern: all_patterns.count(pattern) for pattern in set(all_patterns)}
            },
            'category_breakdown': {}
        }
        
        # Category breakdown
        categories = {}
        for result in self.results:
            category = result.get('metadata', {}).get('category', 'unknown')
            if category not in categories:
                categories[category] = {
                    'count': 0,
                    'avg_quality': 0,
                    'quality_scores': []
                }
            
            categories[category]['count'] += 1
            quality_score = result.get('quality_assessment', {}).get('overall_score', 0)
            categories[category]['quality_scores'].append(quality_score)
        
        # Calculate averages for categories
        for category, data in categories.items():
            data['avg_quality'] = sum(data['quality_scores']) / len(data['quality_scores']) if data['quality_scores'] else 0
            del data['quality_scores']  # Remove raw scores to keep JSON clean
        
        summary_report['category_breakdown'] = categories
        
        # Save summary report
        summary_file = self.output_dir / "summary_reports" / "comprehensive_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        print(f"   💾 Summary report saved to: {summary_file}")
        
        # Create markdown summary
        self.create_markdown_summary(summary_report)
    
    def create_markdown_summary(self, summary_report: Dict[str, Any]):
        """Create a markdown summary report."""
        print("   📝 Creating markdown summary...")
        
        md_content = f"""# Comprehensive Repository Analyzer v3.0 - Whole Dataset Analysis Report

## 📊 Analysis Overview

- **Total Repositories**: {summary_report['analysis_metadata']['total_repositories']}
- **Successful Analyses**: {summary_report['analysis_metadata']['successful_analyses']}
- **Failed Analyses**: {summary_report['analysis_metadata']['failed_analyses']}
- **Success Rate**: {summary_report['analysis_metadata']['success_rate']:.2%}
- **Total Duration**: {summary_report['analysis_metadata']['total_duration_seconds']:.2f} seconds
- **Average Analysis Time**: {summary_report['analysis_metadata']['average_analysis_time']:.2f} seconds per repository

## 🎯 Quality Assessment

### Overall Quality Statistics
- **Average Quality Score**: {summary_report['quality_statistics']['average_quality_score']:.3f}
- **Minimum Quality Score**: {summary_report['quality_statistics']['min_quality_score']:.3f}
- **Maximum Quality Score**: {summary_report['quality_statistics']['max_quality_score']:.3f}

### Quality Distribution
- **Excellent (≥0.8)**: {summary_report['quality_statistics']['quality_score_distribution']['excellent']} repositories
- **Good (0.6-0.8)**: {summary_report['quality_statistics']['quality_score_distribution']['good']} repositories
- **Fair (0.4-0.6)**: {summary_report['quality_statistics']['quality_score_distribution']['fair']} repositories
- **Poor (<0.4)**: {summary_report['quality_statistics']['quality_score_distribution']['poor']} repositories

## 👨‍💻 Programmer Characteristics

### Experience Level Distribution
- **Junior**: {summary_report['experience_level_distribution']['junior']} developers
- **Intermediate**: {summary_report['experience_level_distribution']['intermediate']} developers
- **Senior**: {summary_report['experience_level_distribution']['senior']} developers

### Coding Style Distribution
- **Professional**: {summary_report['coding_style_distribution']['professional']} repositories
- **Simple and Clean**: {summary_report['coding_style_distribution']['simple_and_clean']} repositories
- **Comprehensive**: {summary_report['coding_style_distribution']['comprehensive']} repositories
- **Basic**: {summary_report['coding_style_distribution']['basic']} repositories

## 🏗️ Architecture Patterns

### Pattern Detection
- **Total Patterns Detected**: {summary_report['architecture_patterns']['total_patterns_detected']}
- **Unique Patterns**: {len(summary_report['architecture_patterns']['unique_patterns'])}

### Pattern Frequency
"""
        
        for pattern, count in summary_report['architecture_patterns']['pattern_frequency'].items():
            md_content += f"- **{pattern}**: {count} repositories\n"
        
        md_content += """
## 📂 Category Breakdown
"""
        
        for category, data in summary_report['category_breakdown'].items():
            md_content += f"""
### {category.title()}
- **Count**: {data['count']} repositories
- **Average Quality**: {data['avg_quality']:.3f}
"""
        
        md_content += f"""
## 📅 Analysis Information

- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Analyzer Version**: Comprehensive Repository Analyzer v3.0
- **Dataset**: Whole repository dataset
- **Output Directory**: {self.output_dir}

## 📁 Generated Files

- `individual_results/`: Individual analysis results for each repository
- `summary_reports/`: Aggregated summary reports
- `aggregated_data/`: CSV and JSON data for further analysis
- `performance_analysis/`: Performance metrics and timing information
- `error_reports/`: Error logs and failed analysis details

---
*Generated by Comprehensive Repository Analyzer v3.0*
"""
        
        # Save markdown report
        md_file = self.output_dir / "summary_reports" / "analysis_report.md"
        with open(md_file, 'w') as f:
            f.write(md_content)
        print(f"   💾 Markdown report saved to: {md_file}")
    
    def create_performance_analysis(self):
        """Create performance analysis report."""
        print("   ⚡ Creating performance analysis...")
        
        if not self.results:
            return
        
        # Extract performance data
        analysis_times = [r.get('metadata', {}).get('analysis_time_seconds', 0) for r in self.results]
        categories = [r.get('metadata', {}).get('category', 'unknown') for r in self.results]
        
        # Calculate performance metrics
        performance_data = {
            'overall_performance': {
                'total_analysis_time': sum(analysis_times),
                'average_analysis_time': sum(analysis_times) / len(analysis_times),
                'min_analysis_time': min(analysis_times),
                'max_analysis_time': max(analysis_times),
                'total_repositories': len(self.results)
            },
            'category_performance': {}
        }
        
        # Category-wise performance
        category_times = {}
        for category, time_taken in zip(categories, analysis_times):
            if category not in category_times:
                category_times[category] = []
            category_times[category].append(time_taken)
        
        for category, times in category_times.items():
            performance_data['category_performance'][category] = {
                'count': len(times),
                'total_time': sum(times),
                'average_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        
        # Save performance analysis
        perf_file = self.output_dir / "summary_reports" / "performance_analysis.json"
        with open(perf_file, 'w') as f:
            json.dump(performance_data, f, indent=2, default=str)
        print(f"   💾 Performance analysis saved to: {perf_file}")
    
    def generate_error_report(self):
        """Generate error report for failed analyses."""
        print("   ❌ Generating error report...")
        
        if not self.errors:
            print("   ✅ No errors to report!")
            return
        
        error_report = {
            'total_errors': len(self.errors),
            'error_summary': {},
            'detailed_errors': self.errors
        }
        
        # Categorize errors
        error_types = {}
        for error in self.errors:
            error_msg = error['error_message']
            error_type = error_msg.split(':')[0] if ':' in error_msg else 'Unknown'
            
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
        
        error_report['error_summary'] = error_types
        
        # Save error report
        error_file = self.output_dir / "summary_reports" / "error_report.json"
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2, default=str)
        print(f"   💾 Error report saved to: {error_file}")
    
    def print_final_summary(self):
        """Print final summary to console."""
        print("\n" + "=" * 80)
        print("🎉 COMPREHENSIVE REPOSITORY ANALYZER v3.0 - WHOLE DATASET ANALYSIS COMPLETE")
        print("=" * 80)
        
        print(f"\n📊 Analysis Summary:")
        print(f"   Total Repositories: {self.stats['total_repositories']}")
        print(f"   Successful Analyses: {self.stats['successful_analyses']}")
        print(f"   Failed Analyses: {self.stats['failed_analyses']}")
        print(f"   Success Rate: {self.stats['successful_analyses'] / self.stats['total_repositories']:.2%}")
        print(f"   Total Duration: {self.stats['total_duration']:.2f} seconds")
        
        if self.results:
            avg_quality = sum(r.get('quality_assessment', {}).get('overall_score', 0) for r in self.results) / len(self.results)
            print(f"   Average Quality Score: {avg_quality:.3f}")
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"   📂 Individual Results: {self.output_dir}/individual_results/")
        print(f"   📂 Summary Reports: {self.output_dir}/summary_reports/")
        print(f"   📂 Aggregated Data: {self.output_dir}/aggregated_data/")
        
        if self.errors:
            print(f"\n⚠️  {len(self.errors)} analyses failed. Check error report for details.")
        
        print(f"\n✅ Analysis complete! Check the output directory for detailed results.")


def main():
    """Main function to run the whole dataset analysis."""
    print("🚀 Comprehensive Repository Analyzer v3.0 - Whole Dataset Analysis")
    print("=" * 80)
    
    # Initialize analyzer
    whole_dataset_analyzer = WholeDatasetAnalyzer()
    
    try:
        # Run analysis
        whole_dataset_analyzer.analyze_whole_dataset()
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user.")
        print("💾 Partial results have been saved.")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("📋 Error traceback:")
        traceback.print_exc()
        
        # Save partial results if available
        if whole_dataset_analyzer.results:
            print(f"\n💾 Saving partial results...")
            partial_file = whole_dataset_analyzer.output_dir / "partial_results.json"
            with open(partial_file, 'w') as f:
                json.dump(whole_dataset_analyzer.results, f, indent=2, default=str)
            print(f"   Partial results saved to: {partial_file}")


if __name__ == "__main__":
    main()
