#!/usr/bin/env python3
"""
Quick Demo Script
Runs a small simulation with mock LLM to demonstrate the system quickly.
"""

import sys
sys.path.insert(0, '/home/claude')

from run_simulation import run_simulation
from visualize_results import (
    load_results, 
    plot_metrics_over_time, 
    analyze_content_characteristics,
    generate_summary_report
)


def main():
    print("\n" + "="*70)
    print("AI CONTENT & RECOMMENDER DYNAMICS - QUICK DEMO")
    print("="*70)
    print("\nThis demo runs a small simulation with mock LLM (fast, no GPU needed)")
    print("to demonstrate the system's capabilities.\n")
    
    # Run simulation
    print("Starting simulation...\n")
    content_pool, metrics, users = run_simulation(
        n_rounds=3,
        n_agents=4,
        n_users=6,
        recommender_strategy="diversity",
        use_mock_llm=True
    )
    
    # Generate visualizations
    print("\n\nGenerating visualizations...")
    
    try:
        results = load_results()
        plot_metrics_over_time(results['metrics_history'])
        analyze_content_characteristics(results)
        generate_summary_report(results)
        
        print("\n" + "="*70)
        print("DEMO COMPLETE!")
        print("="*70)
        print("\nCheck these files:")
        print("  - simulation_results.json (raw data)")
        print("  - /mnt/user-data/outputs/metrics_plot.png")
        print("  - /mnt/user-data/outputs/content_analysis.png")
        print("  - /mnt/user-data/outputs/summary_report.txt")
        
    except Exception as e:
        print(f"\nNote: Visualization failed (matplotlib may not be available): {e}")
        print("But simulation results are saved in simulation_results.json")
    
    print("\n" + "="*70)
    print("\nTo run a full simulation with Llama-3.1-8B:")
    print("  python run_simulation.py --rounds 10 --agents 6 --users 10")
    print("\nTo test different recommender strategies:")
    print("  python run_simulation.py --mock --recommender relevance")
    print("  python run_simulation.py --mock --recommender diversity")
    print("  python run_simulation.py --mock --recommender random")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()