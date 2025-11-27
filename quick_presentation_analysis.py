"""
Quick analysis script for partial bias test results.
Works with the actual file structure from interrupted test runs.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

sns.set_style("whitegrid")
sns.set_palette("husl")

def load_scenario_data(scenario_path: Path):
    """Load all available data from a scenario."""
    data = {}

    # Load JSON files
    for json_file in ['simulation_summary.json', 'bias_analysis.json',
                      'round_data.json', 'feed_assignments.json']:
        file_path = scenario_path / json_file
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data[json_file.replace('.json', '')] = json.load(f)
            except json.JSONDecodeError:
                print(f"  Warning: Could not parse {json_file} in {scenario_path.name}")
                continue

    # Load CSV files
    for csv_file in ['all_content.csv', 'all_interactions.csv', 'users.csv', 'agents.csv']:
        file_path = scenario_path / csv_file
        if file_path.exists():
            try:
                data[csv_file.replace('.csv', '')] = pd.read_csv(file_path)
            except Exception as e:
                print(f"  Warning: Could not read {csv_file} in {scenario_path.name}: {str(e)}")
                continue

    return data

def summarize_available_data(results_dir: Path):
    """Generate summary of what data is available."""
    print("\n" + "="*70)
    print("AVAILABLE DATA SUMMARY")
    print("="*70)

    summary = defaultdict(list)

    for bias_type in ['gender_bias', 'dialect_bias', 'self_preference', 'topic_bias']:
        bias_dir = results_dir / bias_type
        if bias_dir.exists():
            for scenario_dir in bias_dir.iterdir():
                if scenario_dir.is_dir():
                    summary_file = scenario_dir / 'simulation_summary.json'
                    if summary_file.exists():
                        with open(summary_file, 'r') as f:
                            sim_summary = json.load(f)
                        summary[bias_type].append({
                            'name': scenario_dir.name,
                            'rounds': sim_summary.get('n_rounds', 'N/A'),
                            'users': sim_summary.get('n_users', 'N/A'),
                            'interactions': sim_summary.get('n_interactions', 'N/A')
                        })

    for bias_type, scenarios in summary.items():
        print(f"\n{bias_type.upper().replace('_', ' ')}:")
        if scenarios:
            for s in scenarios:
                print(f"  ✓ {s['name']}")
                print(f"    Rounds: {s['rounds']}, Users: {s['users']}, Interactions: {s['interactions']}")
        else:
            print("  (No completed scenarios)")

    print("\n" + "="*70)
    return summary

def analyze_gender_bias(results_dir: Path):
    """Analyze gender bias from available scenarios."""
    print("\n" + "="*70)
    print("GENDER BIAS ANALYSIS")
    print("="*70)

    gb_dir = results_dir / "gender_bias"
    if not gb_dir.exists():
        print("No gender bias results found.")
        return None

    scenarios = {}
    for scenario_dir in gb_dir.iterdir():
        if scenario_dir.is_dir() and (scenario_dir / 'simulation_summary.json').exists():
            scenarios[scenario_dir.name] = load_scenario_data(scenario_dir)

    if not scenarios:
        print("No completed scenarios found.")
        return None

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Gender Bias Analysis: Content Generation Distribution', fontsize=16, fontweight='bold')

    scenario_order = ['gender_bias_neutral_mistral_rec', 'gender_bias_neutral_llama_rec',
                     'gender_bias_stereotyped_mistral_rec', 'gender_bias_stereotyped_llama_rec']

    for idx, scenario_name in enumerate(scenario_order):
        if scenario_name not in scenarios:
            continue

        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        data = scenarios[scenario_name]
        if 'all_content' in data:
            df = data['all_content']

            # Gender distribution
            if 'gender' in df.columns:
                gender_counts = df['gender'].value_counts()
                colors = {'male': '#3498db', 'female': '#e74c3c', 'neutral': '#95a5a6'}
                bar_colors = [colors.get(g, '#95a5a6') for g in gender_counts.index]

                ax.bar(gender_counts.index, gender_counts.values, color=bar_colors, alpha=0.7)
                ax.set_title(scenario_name.replace('gender_bias_', '').replace('_', ' ').title(),
                           fontsize=12, fontweight='bold')
                ax.set_xlabel('Gender', fontsize=10)
                ax.set_ylabel('Content Count', fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')

                # Add value labels on bars
                for i, (gender, count) in enumerate(gender_counts.items()):
                    ax.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = results_dir / "gender_bias_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    # Generate summary statistics
    print("\nGender Bias Summary:")
    for scenario_name, data in scenarios.items():
        if 'all_content' in data and 'gender' in data['all_content'].columns:
            df = data['all_content']
            gender_dist = df['gender'].value_counts(normalize=True) * 100
            print(f"\n  {scenario_name}:")
            for gender, pct in gender_dist.items():
                print(f"    {gender}: {pct:.1f}%")

    return scenarios

def analyze_dialect_bias(results_dir: Path):
    """Analyze dialect bias from available scenarios."""
    print("\n" + "="*70)
    print("DIALECT BIAS ANALYSIS")
    print("="*70)

    db_dir = results_dir / "dialect_bias"
    if not db_dir.exists():
        print("No dialect bias results found.")
        return None

    scenarios = {}
    for scenario_dir in db_dir.iterdir():
        if scenario_dir.is_dir() and (scenario_dir / 'simulation_summary.json').exists():
            scenarios[scenario_dir.name] = load_scenario_data(scenario_dir)

    if not scenarios:
        print("No completed scenarios found.")
        return None

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Dialect Bias Analysis: Content Generation Distribution', fontsize=16, fontweight='bold')

    for idx, (scenario_name, data) in enumerate(scenarios.items()):
        if idx >= 2:
            break

        ax = axes[idx]

        if 'all_content' in data:
            df = data['all_content']

            # Dialect distribution
            if 'dialect' in df.columns:
                dialect_counts = df['dialect'].value_counts()
                colors = {'standard': '#3498db', 'aave': '#e74c3c', 'southern': '#2ecc71'}
                bar_colors = [colors.get(d, '#95a5a6') for d in dialect_counts.index]

                ax.bar(dialect_counts.index, dialect_counts.values, color=bar_colors, alpha=0.7)
                ax.set_title(scenario_name.replace('dialect_bias_', '').replace('_', ' ').title(),
                           fontsize=12, fontweight='bold')
                ax.set_xlabel('Dialect', fontsize=10)
                ax.set_ylabel('Content Count', fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='x', rotation=15)

                # Add value labels on bars
                for i, (dialect, count) in enumerate(dialect_counts.items()):
                    ax.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = results_dir / "dialect_bias_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    # Generate summary statistics
    print("\nDialect Bias Summary:")
    for scenario_name, data in scenarios.items():
        if 'all_content' in data and 'dialect' in data['all_content'].columns:
            df = data['all_content']
            dialect_dist = df['dialect'].value_counts(normalize=True) * 100
            print(f"\n  {scenario_name}:")
            for dialect, pct in dialect_dist.items():
                print(f"    {dialect}: {pct:.1f}%")

    return scenarios

def analyze_self_preference(results_dir: Path):
    """Analyze self-preference bias from available scenarios."""
    print("\n" + "="*70)
    print("SELF-PREFERENCE BIAS ANALYSIS")
    print("="*70)

    sp_dir = results_dir / "self_preference"
    if not sp_dir.exists():
        print("No self-preference results found.")
        return None

    scenarios = {}
    for scenario_dir in sp_dir.iterdir():
        if scenario_dir.is_dir() and (scenario_dir / 'simulation_summary.json').exists():
            scenarios[scenario_dir.name] = load_scenario_data(scenario_dir)

    if not scenarios:
        print("No completed scenarios found.")
        return None

    # Analyze bias from bias_analysis.json
    print("\nSelf-Preference Results:")

    for scenario_name, data in scenarios.items():
        print(f"\n  {scenario_name}:")

        if 'bias_analysis' in data:
            bias_data = data['bias_analysis']

            # Check for self-preference metrics
            if 'self_preference_analysis' in bias_data:
                sp_analysis = bias_data['self_preference_analysis']
                print(f"    Metrics found: {list(sp_analysis.keys())}")

                # Print key metrics
                for key, value in sp_analysis.items():
                    if isinstance(value, (int, float)):
                        print(f"      {key}: {value:.3f}")
                    elif isinstance(value, dict):
                        print(f"      {key}:")
                        for subkey, subval in value.items():
                            print(f"        {subkey}: {subval}")
            else:
                print("    (No self-preference analysis found in bias_analysis.json)")

        # Analyze content by source LLM
        if 'all_content' in data:
            df = data['all_content']
            if 'source_llm' in df.columns:
                source_counts = df['source_llm'].value_counts()
                print(f"    Content by source:")
                for llm, count in source_counts.items():
                    print(f"      {llm}: {count}")

    return scenarios

def generate_presentation_summary(results_dir: Path):
    """Generate a markdown summary for presentation."""
    print("\n" + "="*70)
    print("GENERATING PRESENTATION SUMMARY")
    print("="*70)

    summary_lines = [
        "# Bias Testing Results Summary\n",
        "## Test Configuration",
        "- Rounds: 10 (interrupted)",
        "- Users per scenario: 20",
        "- Test Suite: Complete Bias Test Suite\n",
        "## Completed Analyses\n"
    ]

    # Check each bias type
    bias_types = {
        'gender_bias': 'Gender Bias',
        'dialect_bias': 'Dialect Bias',
        'self_preference': 'Self-Preference Bias',
        'topic_bias': 'Topic Bias'
    }

    for bias_dir_name, bias_label in bias_types.items():
        bias_dir = results_dir / bias_dir_name
        if bias_dir.exists():
            completed = list(bias_dir.glob("*/simulation_summary.json"))
            if completed:
                summary_lines.append(f"### {bias_label}")
                summary_lines.append(f"**Status:** ✓ {len(completed)} scenario(s) completed\n")
                for summary_file in completed:
                    scenario_name = summary_file.parent.name
                    summary_lines.append(f"- {scenario_name.replace('_', ' ').title()}")
                summary_lines.append("")
            else:
                summary_lines.append(f"### {bias_label}")
                summary_lines.append("**Status:** ✗ No completed scenarios (interrupted)\n")

    summary_lines.extend([
        "\n## Key Findings",
        "",
        "### Gender Bias",
        "- Compared neutral vs. stereotyped content generation",
        "- Tested with both Mistral and Llama recommenders",
        "- Analysis shows distribution patterns in content generation\n",
        "### Dialect Bias",
        "- Analyzed dialect representation in generated content",
        "- Compared Mistral and Llama recommender behaviors\n",
        "### Self-Preference Bias",
        "- Partial results for Mistral recommender",
        "- Baseline comparison (Llama) was interrupted\n",
        "## Visualizations Generated",
        "- `gender_bias_analysis.png` - Gender distribution across scenarios",
        "- `dialect_bias_analysis.png` - Dialect representation analysis",
        "\n## Notes",
        "- Topic bias testing was interrupted before completion",
        "- Self-preference baseline comparison is incomplete",
        "- All completed scenarios have full interaction data available"
    ])

    summary_text = "\n".join(summary_lines)

    summary_file = results_dir / "presentation_summary.md"
    with open(summary_file, 'w') as f:
        f.write(summary_text)

    print(f"✓ Saved: {summary_file}")

    # Also print to console
    print("\n" + "="*70)
    print(summary_text)
    print("="*70)

def main():
    """Main analysis pipeline for partial results."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick analysis of partial bias test results for presentation"
    )
    parser.add_argument("--results-dir", type=str, default="./complete_bias_results",
                       help="Directory containing bias test results")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print("\n" + "="*70)
    print("QUICK PRESENTATION ANALYSIS")
    print("Analyzing partial/interrupted bias test results")
    print("="*70)
    print(f"Results directory: {results_dir}")

    # Step 1: Summarize available data
    data_summary = summarize_available_data(results_dir)

    # Step 2: Analyze each bias type
    gender_results = analyze_gender_bias(results_dir)
    dialect_results = analyze_dialect_bias(results_dir)
    sp_results = analyze_self_preference(results_dir)

    # Step 3: Generate presentation summary
    generate_presentation_summary(results_dir)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    for viz_file in results_dir.glob("*.png"):
        print(f"  📊 {viz_file.name}")
    for report_file in results_dir.glob("*.md"):
        print(f"  📄 {report_file.name}")
    print("\nYou can use these visualizations and the summary for your presentation!")
    print("="*70)

if __name__ == "__main__":
    main()
