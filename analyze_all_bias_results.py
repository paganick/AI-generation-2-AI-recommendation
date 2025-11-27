"""
Comprehensive analysis pipeline for all bias test results.
Generates comparative visualizations and summary reports.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

def load_bias_analysis(scenario_path: Path) -> dict:
    """Load bias analysis from a scenario."""
    analysis_file = scenario_path / "comprehensive_analysis.json"
    if analysis_file.exists():
        with open(analysis_file, 'r') as f:
            return json.load(f)
    return None

def analyze_self_preference_results(results_dir: Path):
    """Analyze self-preference bias across scenarios."""

    print("\n" + "="*70)
    print("SELF-PREFERENCE BIAS ANALYSIS")
    print("="*70)

    sp_dir = results_dir / "self_preference"
    if not sp_dir.exists():
        print("No self-preference results found.")
        return None

    results = {}

    for scenario_dir in sp_dir.iterdir():
        if scenario_dir.is_dir():
            analysis = load_bias_analysis(scenario_dir)
            if analysis and 'llm_favoritism' in analysis:
                results[scenario_dir.name] = analysis['llm_favoritism']

    if not results:
        print("No favoritism data found.")
        return None

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract favoritism scores for mixed scenarios
    test_scenarios = {k: v for k, v in results.items() if 'test' in k}

    if test_scenarios:
        scenario_names = []
        llama_bias = []
        mistral_bias = []

        for name, data in test_scenarios.items():
            if 'favoritism_bias' in data:
                scenario_names.append(name.replace('self_pref_test_', '').replace('_rec', ' rec'))
                bias = data['favoritism_bias']
                llama_bias.append(bias.get('llama', 0))
                mistral_bias.append(bias.get('mistral', 0))

        x = np.arange(len(scenario_names))
        width = 0.35

        axes[0].bar(x - width/2, llama_bias, width, label='Llama', color='#e74c3c')
        axes[0].bar(x + width/2, mistral_bias, width, label='Mistral', color='#3498db')
        axes[0].set_xlabel('Recommender')
        axes[0].set_ylabel('Favoritism Bias (%)')
        axes[0].set_title('Self-Preference Bias by Recommender Architecture')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(scenario_names)
        axes[0].legend()
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[0].grid(True, alpha=0.3)

    # Statistical significance
    p_values = []
    scenarios_sig = []

    for name, data in test_scenarios.items():
        if 'statistical_test' in data:
            p_val = data['statistical_test'].get('p_value', 1.0)
            p_values.append(p_val)
            scenarios_sig.append(name.replace('self_pref_test_', '').replace('_rec', '\nrec'))

    if p_values:
        colors = ['red' if p < 0.05 else 'gray' for p in p_values]
        axes[1].bar(scenarios_sig, p_values, color=colors)
        axes[1].axhline(y=0.05, color='red', linestyle='--', label='p=0.05 threshold')
        axes[1].set_xlabel('Recommender')
        axes[1].set_ylabel('P-value')
        axes[1].set_title('Statistical Significance of Self-Preference Bias')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = results_dir / "self_preference_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    return results

def analyze_gender_bias_results(results_dir: Path):
    """Analyze gender bias across scenarios."""

    print("\n" + "="*70)
    print("GENDER BIAS ANALYSIS")
    print("="*70)

    gb_dir = results_dir / "gender_bias"
    if not gb_dir.exists():
        print("No gender bias results found.")
        return None

    results = {}

    for scenario_dir in gb_dir.iterdir():
        if scenario_dir.is_dir():
            content_file = scenario_dir / "all_content.csv"
            if content_file.exists():
                df = pd.read_csv(content_file)
                results[scenario_dir.name] = df

    if not results:
        print("No data found.")
        return None

    # Analyze gender representation in recommendations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for idx, (name, df) in enumerate(results.items()):
        if 'gender' not in df.columns:
            continue

        row = idx // 2
        col = idx % 2

        # Gender distribution in content
        gender_counts = df['gender'].value_counts()
        axes[row, col].bar(gender_counts.index, gender_counts.values)
        axes[row, col].set_title(f"{name.replace('gender_bias_', '').replace('_', ' ')}")
        axes[row, col].set_xlabel('Gender')
        axes[row, col].set_ylabel('Content Count')
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = results_dir / "gender_bias_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    return results

def analyze_topic_bias_results(results_dir: Path):
    """Analyze topic bias across scenarios."""

    print("\n" + "="*70)
    print("TOPIC BIAS ANALYSIS")
    print("="*70)

    tb_dir = results_dir / "topic_bias"
    if not tb_dir.exists():
        print("No topic bias results found.")
        return None

    results = {}

    for scenario_dir in tb_dir.iterdir():
        if scenario_dir.is_dir():
            analysis = load_bias_analysis(scenario_dir)
            if analysis:
                results[scenario_dir.name] = analysis

    if not results:
        print("No data found.")
        return None

    # Create visualization comparing topic distributions
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for idx, (name, analysis) in enumerate(results.items()):
        if 'content_diversity' in analysis and 'topic_distribution' in analysis['content_diversity']:
            topic_dist = analysis['content_diversity']['topic_distribution']
            topics = list(topic_dist.keys())
            counts = list(topic_dist.values())

            axes[idx].bar(topics, counts)
            axes[idx].set_title(f"{name.replace('topic_bias_', '').replace('_', ' ')}")
            axes[idx].set_xlabel('Topic')
            axes[idx].set_ylabel('Content Count')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = results_dir / "topic_bias_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    return results

def analyze_political_bias_results(results_dir: Path):
    """Analyze political bias across scenarios."""

    print("\n" + "="*70)
    print("POLITICAL BIAS ANALYSIS")
    print("="*70)

    pb_dir = results_dir / "political_bias"
    if not pb_dir.exists():
        print("No political bias results found.")
        return None

    results = {}

    for scenario_dir in pb_dir.iterdir():
        if scenario_dir.is_dir():
            content_file = scenario_dir / "all_content.csv"
            if content_file.exists():
                df = pd.read_csv(content_file)
                results[scenario_dir.name] = df

    if not results:
        print("No data found.")
        return None

    # Visualize political position distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, df) in enumerate(results.items()):
        if 'political_position' not in df.columns:
            continue

        pos_counts = df['political_position'].value_counts()
        axes[idx].bar(pos_counts.index, pos_counts.values)
        axes[idx].set_title(f"{name.replace('political_bias_', '').replace('_', ' ')}")
        axes[idx].set_xlabel('Political Position')
        axes[idx].set_ylabel('Content Count')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = results_dir / "political_bias_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    return results

def analyze_dialect_bias_results(results_dir: Path):
    """Analyze dialect bias across scenarios."""

    print("\n" + "="*70)
    print("DIALECT BIAS ANALYSIS")
    print("="*70)

    db_dir = results_dir / "dialect_bias"
    if not db_dir.exists():
        print("No dialect bias results found.")
        return None

    results = {}

    for scenario_dir in db_dir.iterdir():
        if scenario_dir.is_dir():
            content_file = scenario_dir / "all_content.csv"
            if content_file.exists():
                df = pd.read_csv(content_file)
                results[scenario_dir.name] = df

    if not results:
        print("No data found.")
        return None

    # Visualize dialect distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, df) in enumerate(results.items()):
        if 'dialect' not in df.columns:
            continue

        dialect_counts = df['dialect'].value_counts()
        axes[idx].bar(dialect_counts.index, dialect_counts.values)
        axes[idx].set_title(f"{name.replace('dialect_bias_', '').replace('_', ' ')}")
        axes[idx].set_xlabel('Dialect')
        axes[idx].set_ylabel('Content Count')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = results_dir / "dialect_bias_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    return results

def generate_comprehensive_report(results_dir: Path):
    """Generate comprehensive JSON report of all analyses."""

    report = {
        "test_suite": "Complete Bias Analysis",
        "results_directory": str(results_dir),
        "bias_analyses": {}
    }

    # Analyze each bias type
    sp_results = analyze_self_preference_results(results_dir)
    if sp_results:
        report["bias_analyses"]["self_preference"] = {
            "scenarios_analyzed": len(sp_results),
            "summary": "Self-preference bias analysis completed"
        }

    gb_results = analyze_gender_bias_results(results_dir)
    if gb_results:
        report["bias_analyses"]["gender_bias"] = {
            "scenarios_analyzed": len(gb_results),
            "summary": "Gender bias analysis completed"
        }

    tb_results = analyze_topic_bias_results(results_dir)
    if tb_results:
        report["bias_analyses"]["topic_bias"] = {
            "scenarios_analyzed": len(tb_results),
            "summary": "Topic bias analysis completed"
        }

    pb_results = analyze_political_bias_results(results_dir)
    if pb_results:
        report["bias_analyses"]["political_bias"] = {
            "scenarios_analyzed": len(pb_results),
            "summary": "Political bias analysis completed"
        }

    db_results = analyze_dialect_bias_results(results_dir)
    if db_results:
        report["bias_analyses"]["dialect_bias"] = {
            "scenarios_analyzed": len(db_results),
            "summary": "Dialect bias analysis completed"
        }

    # Save report
    report_file = results_dir / "comprehensive_bias_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Comprehensive report saved: {report_file}")

    return report

def main():
    """Main analysis pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze all bias test results")
    parser.add_argument("--results-dir", type=str, default="./complete_bias_results",
                       help="Directory containing bias test results")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print("\n" + "="*70)
    print("COMPREHENSIVE BIAS ANALYSIS PIPELINE")
    print("="*70)
    print(f"Results directory: {results_dir}")
    print()

    # Run all analyses
    report = generate_comprehensive_report(results_dir)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nGenerated visualizations:")
    for viz_file in results_dir.glob("*_summary.png"):
        print(f"  • {viz_file.name}")
    print(f"\nComprehensive report: comprehensive_bias_report.json")
    print("="*70)

if __name__ == "__main__":
    main()
