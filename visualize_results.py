#!/usr/bin/env python3
"""
Visualization and Analysis of Simulation Results
Creates plots to understand system dynamics over time.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(filepath: str = "simulation_results.json"):
    """Load simulation results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_metrics_over_time(metrics_history: dict, output_path: str = "outputs/metrics_plot.png"):
    """Create a comprehensive plot of all metrics over time."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('AI Content & Recommender System Dynamics', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Define metrics to plot
    metrics_to_plot = [
        ('content_diversity', 'Content Diversity', 'Higher is better'),
        ('engagement_gini', 'Engagement Inequality (Gini)', 'Lower is better'),
        ('topic_concentration', 'Topic Concentration (HHI)', 'Lower is better'),
        ('opinion_variance', 'Opinion Variance', 'Metric of spread'),
        ('polarization', 'Polarization', 'Lower is better'),
    ]
    
    for idx, (metric_key, title, interpretation) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        if metric_key in metrics_history and metrics_history[metric_key]:
            values = metrics_history[metric_key]
            rounds = list(range(1, len(values) + 1))
            
            ax.plot(rounds, values, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Simulation Round', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(f'{title}\n({interpretation})', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(values) > 2:
                z = np.polyfit(rounds, values, 1)
                p = np.poly1d(z)
                ax.plot(rounds, p(rounds), "--", alpha=0.5, color='red', 
                       label=f'Trend: {"↑" if z[0] > 0 else "↓"}')
                ax.legend()
    
    # Plot sentiment distribution in the last subplot
    ax = axes[5]
    if 'sentiment_distribution' in metrics_history:
        sentiment_data = metrics_history['sentiment_distribution']
        if sentiment_data:
            means = [s['mean'] for s in sentiment_data if isinstance(s, dict)]
            stds = [s['std'] for s in sentiment_data if isinstance(s, dict)]
            
            if means:
                rounds = list(range(1, len(means) + 1))
                ax.errorbar(rounds, means, yerr=stds, marker='o', capsize=5, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('Simulation Round', fontsize=10)
                ax.set_ylabel('Mean Sentiment', fontsize=10)
                ax.set_title('Sentiment Distribution\n(Mean ± Std)', fontsize=11)
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to: {output_path}")
    
    return fig


def analyze_content_characteristics(results: dict, output_path: str = "outputs/content_analysis.png"):
    """Analyze characteristics of generated content."""
    
    content_items = results['content_items']
    
    # Extract data
    topics = [item.get('topic') for item in content_items if item.get('topic')]
    sentiments = [item.get('sentiment_score') for item in content_items if item.get('sentiment_score') is not None]
    engagements = [item.get('engagement_score', 0) for item in content_items]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Content Characteristics Analysis', fontsize=16, fontweight='bold')
    
    # Topic distribution
    ax = axes[0, 0]
    if topics:
        unique_topics, counts = np.unique(topics, return_counts=True)
        ax.bar(range(len(unique_topics)), counts, color='steelblue')
        ax.set_xticks(range(len(unique_topics)))
        ax.set_xticklabels(unique_topics, rotation=45, ha='right')
        ax.set_ylabel('Number of Posts')
        ax.set_title('Content Distribution by Topic')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Sentiment distribution
    ax = axes[0, 1]
    if sentiments:
        ax.hist(sentiments, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=2, label='Neutral')
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Sentiment Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Engagement distribution
    ax = axes[1, 0]
    if engagements:
        ax.hist(engagements, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Engagement Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Engagement Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentile lines
        percentiles = [50, 90, 95]
        for p in percentiles:
            val = np.percentile(engagements, p)
            ax.axvline(x=val, color='red', linestyle='--', alpha=0.5, 
                      label=f'{p}th percentile')
        ax.legend()
    
    # Engagement vs Sentiment
    ax = axes[1, 1]
    if sentiments and engagements and len(sentiments) == len(engagements):
        ax.scatter(sentiments, engagements, alpha=0.6, s=50, color='purple')
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Engagement Score')
        ax.set_title('Engagement vs Sentiment')
        ax.grid(True, alpha=0.3)
        
        # Add correlation
        if len(sentiments) > 1:
            corr = np.corrcoef(sentiments, engagements)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Content analysis plot saved to: {output_path}")
    
    return fig


def generate_summary_report(results: dict, output_path: str = "outputs/summary_report.txt"):
    """Generate a text summary report."""
    
    content_items = results['content_items']
    metrics_summary = results.get('metrics_summary', {})
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("SIMULATION SUMMARY REPORT")
    report_lines.append("="*70)
    report_lines.append("")
    
    # Basic statistics
    report_lines.append("CONTENT GENERATION STATISTICS")
    report_lines.append("-"*70)
    report_lines.append(f"Total content items generated: {len(content_items)}")
    
    # Count by type
    initial_posts = sum(1 for item in content_items if item.get('parent_id') is None)
    responses = len(content_items) - initial_posts
    report_lines.append(f"  Initial posts: {initial_posts}")
    report_lines.append(f"  Responses: {responses}")
    
    # Topics
    topics = [item.get('topic') for item in content_items if item.get('topic')]
    if topics:
        unique_topics = set(topics)
        report_lines.append(f"  Unique topics: {len(unique_topics)}")
        report_lines.append(f"  Topics: {', '.join(unique_topics)}")
    
    report_lines.append("")
    
    # Metrics summary
    report_lines.append("SYSTEM DYNAMICS METRICS")
    report_lines.append("-"*70)
    
    for metric_name, stats in metrics_summary.items():
        metric_display = metric_name.replace('_', ' ').title()
        report_lines.append(f"\n{metric_display}:")
        report_lines.append(f"  Final value: {stats['final']:.4f}")
        report_lines.append(f"  Average: {stats['mean']:.4f}")
        report_lines.append(f"  Trend: {stats['trend']}")
    
    report_lines.append("")
    report_lines.append("="*70)
    
    # Key findings
    report_lines.append("\nKEY FINDINGS")
    report_lines.append("-"*70)
    
    # Analyze trends
    findings = []
    
    if 'content_diversity' in metrics_summary:
        div_trend = metrics_summary['content_diversity']['trend']
        div_final = metrics_summary['content_diversity']['final']
        if div_trend == 'decreasing':
            findings.append(f"⚠ Content diversity is decreasing (final: {div_final:.3f}), "
                          "suggesting potential echo chamber formation.")
        else:
            findings.append(f"✓ Content diversity is maintaining or increasing (final: {div_final:.3f}).")
    
    if 'engagement_gini' in metrics_summary:
        gini_final = metrics_summary['engagement_gini']['final']
        if gini_final > 0.5:
            findings.append(f"⚠ High engagement inequality (Gini: {gini_final:.3f}), "
                          "indicating winner-take-all dynamics.")
        else:
            findings.append(f"✓ Moderate engagement inequality (Gini: {gini_final:.3f}).")
    
    if 'polarization' in metrics_summary:
        pol_trend = metrics_summary['polarization']['trend']
        pol_final = metrics_summary['polarization']['final']
        if pol_trend == 'increasing':
            findings.append(f"⚠ Opinion polarization is increasing (final: {pol_final:.3f}).")
        else:
            findings.append(f"✓ Opinion polarization is stable or decreasing (final: {pol_final:.3f}).")
    
    for finding in findings:
        report_lines.append(f"  {finding}")
    
    report_lines.append("")
    report_lines.append("="*70)
    
    # Write report
    report_text = "\n".join(report_lines)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"Summary report saved to: {output_path}")
    print("\n" + report_text)
    
    return report_text


def main():
    """Main visualization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize simulation results")
    parser.add_argument("--input", type=str, default="simulation_results.json",
                       help="Path to simulation results JSON file")
    
    args = parser.parse_args()
    
    print("Loading simulation results...")
    results = load_results(args.input)
    
    print("\nGenerating visualizations...")
    
    # Create output directory if it doesn't exist
    Path("outputs").mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_metrics_over_time(results['metrics_history'])
    analyze_content_characteristics(results)
    generate_summary_report(results)
    
    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)


if __name__ == "__main__":
    main()