#!/bin/bash

# Convenience script to run both scenarios and compare results
# Uses the REALISTIC simulation with probabilistic agent behavior

set -e  # Exit on error

echo "======================================================================="
echo "REALISTIC SOCIAL MEDIA SIMULATION - BOTH SCENARIOS"
echo "======================================================================="
echo ""
echo "This runs REALISTIC simulations where:"
echo "  - ALL agents create seed content at initialization"
echo "  - Each round, agents probabilistically decide to post/respond/stay quiet"
echo "  - Simulated users (passive audience) consume and engage with content"
echo "  - User opinions evolve based on exposure (bounded confidence model)"
echo ""

# Parse arguments
ROUNDS=10
USERS=20

while [[ $# -gt 0 ]]; do
    case $1 in
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --users)
            USERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--rounds N] [--users N]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Rounds: $ROUNDS"
echo "  Users (passive audience): $USERS"
echo ""

# Run Scenario 1
echo "======================================================================="
echo "RUNNING SCENARIO 1: Llama-only"
echo "======================================================================="
python run_realistic_scenario.py --scenario 1 --rounds $ROUNDS --users $USERS --output ./scenario1_output

echo ""
echo "✓ Scenario 1 complete!"
echo ""

# Run Scenario 2
echo "======================================================================="
echo "RUNNING SCENARIO 2: Mixed Llama + Mistral"
echo "======================================================================="
python run_realistic_scenario.py --scenario 2 --rounds $ROUNDS --users $USERS --output ./scenario2_output

echo ""
echo "✓ Scenario 2 complete!"
echo ""

# Compare scenarios
echo "======================================================================="
echo "COMPARING SCENARIOS"
echo "======================================================================="
python analyze_scenarios.py --output-dir ./scenario1_output --compare ./scenario2_output

echo ""
echo "======================================================================="
echo "ALL DONE!"
echo "======================================================================="
echo "Results:"
echo "  Scenario 1 output: ./scenario1_output/"
echo "  Scenario 2 output: ./scenario2_output/"
echo "  Comparison: ./comparison_results/"
echo ""
echo "Visualizations:"
echo "  Scenario 1: ./scenario1_output/analysis_plots/"
echo "  Scenario 2: ./scenario2_output/analysis_plots/"
echo "======================================================================="
