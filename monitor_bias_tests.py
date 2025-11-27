"""
Monitor progress of bias test suite execution.
"""

import os
import json
from pathlib import Path
from datetime import datetime

def check_progress(results_dir='./complete_bias_results'):
    """Check which scenarios have completed."""

    results_path = Path(results_dir)

    if not results_path.exists():
        print("Results directory doesn't exist yet.")
        return

    # Expected scenarios
    expected_scenarios = {
        'self_preference': [
            'self_pref_baseline_llama',
            'self_pref_baseline_mistral',
            'self_pref_test_llama_rec',
            'self_pref_test_mistral_rec'
        ],
        'topic_bias': [
            'topic_bias_llama_rec',
            'topic_bias_mistral_rec'
        ],
        'political_bias': [
            'political_bias_llama_rec',
            'political_bias_mistral_rec'
        ],
        'dialect_bias': [
            'dialect_bias_llama_rec',
            'dialect_bias_mistral_rec'
        ],
        'gender_bias': [
            'gender_bias_neutral_llama_rec',
            'gender_bias_stereotyped_llama_rec',
            'gender_bias_neutral_mistral_rec',
            'gender_bias_stereotyped_mistral_rec'
        ]
    }

    print("\n" + "="*70)
    print("BIAS TEST SUITE PROGRESS REPORT")
    print("="*70)
    print(f"Results directory: {results_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_expected = sum(len(scenarios) for scenarios in expected_scenarios.values())
    total_completed = 0

    for bias_type, scenarios in expected_scenarios.items():
        print(f"\n📊 {bias_type.replace('_', ' ').title()}")
        print("-" * 70)

        type_path = results_path / bias_type

        for scenario in scenarios:
            scenario_path = type_path / scenario

            # Check if scenario directory exists
            if not scenario_path.exists():
                status = "⏳ Not started"
            else:
                # Check for completion markers
                has_csv = (scenario_path / "all_content.csv").exists()
                has_analysis = (scenario_path / "comprehensive_analysis.json").exists()
                has_plots = (scenario_path / "analysis_plots").exists()

                if has_csv and has_analysis and has_plots:
                    status = "✅ Complete"
                    total_completed += 1
                elif has_csv:
                    status = "🔄 In progress (simulation done)"
                else:
                    status = "🔄 In progress (running)"

            print(f"  {scenario:40s} {status}")

    print("\n" + "="*70)
    print(f"Overall Progress: {total_completed}/{total_expected} scenarios completed")
    print(f"Completion: {100 * total_completed / total_expected:.1f}%")
    print("="*70)

    return total_completed, total_expected


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor bias test suite progress")
    parser.add_argument("--results-dir", type=str, default="./complete_bias_results",
                       help="Results directory to monitor")
    parser.add_argument("--watch", action="store_true",
                       help="Continuously monitor (update every 30s)")

    args = parser.parse_args()

    if args.watch:
        import time
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                completed, total = check_progress(args.results_dir)

                if completed == total:
                    print("\n✨ All scenarios completed!")
                    break

                print("\n(Refreshing in 30 seconds... Press Ctrl+C to stop)")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        check_progress(args.results_dir)
