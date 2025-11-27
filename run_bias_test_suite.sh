#!/bin/bash

# Bias Test Suite Batch Runner
# Executes all scenarios in a bias test suite and generates comparative analysis

set -e  # Exit on error

# Default values
TEST_SUITE_DIR=""
OUTPUT_BASE_DIR="./bias_test_results"
ROUNDS=10
USERS=20
PARALLEL=false
EFFECTS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-suite)
            TEST_SUITE_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --users)
            USERS="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --effects)
            EFFECTS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --test-suite <dir> [options]"
            echo ""
            echo "Options:"
            echo "  --test-suite <dir>   Path to test suite directory (required)"
            echo "  --output <dir>       Output base directory (default: ./bias_test_results)"
            echo "  --rounds <n>         Number of simulation rounds (default: 10)"
            echo "  --users <n>          Number of simulated users (default: 20)"
            echo "  --parallel           Run scenarios in parallel (experimental)"
            echo "  --effects <list>     Comma-separated list of effects to run (default: all)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$TEST_SUITE_DIR" ]; then
    echo "Error: --test-suite is required"
    echo "Run with --help for usage information"
    exit 1
fi

if [ ! -d "$TEST_SUITE_DIR" ]; then
    echo "Error: Test suite directory not found: $TEST_SUITE_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

echo "========================================"
echo "Bias Test Suite Batch Runner"
echo "========================================"
echo "Test Suite: $TEST_SUITE_DIR"
echo "Output: $OUTPUT_BASE_DIR"
echo "Rounds: $ROUNDS"
echo "Users: $USERS"
echo "Parallel: $PARALLEL"
echo ""

# Load test suite index
INDEX_FILE="$TEST_SUITE_DIR/test_suite_index.json"
if [ ! -f "$INDEX_FILE" ]; then
    echo "Error: Test suite index not found: $INDEX_FILE"
    exit 1
fi

# Determine which effects to run
if [ -z "$EFFECTS" ]; then
    # Run all effects found in the test suite
    EFFECT_DIRS=$(find "$TEST_SUITE_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)
else
    # Run only specified effects
    IFS=',' read -ra EFFECT_DIRS <<< "$EFFECTS"
fi

# Function to check if a scenario is already completed
is_scenario_complete() {
    local output_dir=$1

    # Check for key completion markers
    if [ -f "$output_dir/all_content.csv" ] && \
       [ -f "$output_dir/comprehensive_analysis.json" ] && \
       [ -d "$output_dir/analysis_plots" ]; then
        return 0  # Complete
    else
        return 1  # Not complete
    fi
}

# Function to run a single scenario
run_scenario() {
    local scenario_file=$1
    local effect_type=$2
    local scenario_id=$(basename "$scenario_file" .json)
    local output_dir="$OUTPUT_BASE_DIR/$effect_type/$scenario_id"

    # Check if already completed
    if is_scenario_complete "$output_dir"; then
        echo "  ⏭  Skipping (already complete): $effect_type/$scenario_id"
        return 0
    fi

    echo "Running: $effect_type/$scenario_id"

    python run_multi_llm_simulation.py \
        --config "$scenario_file" \
        --rounds "$ROUNDS" \
        --users "$USERS" \
        --output "$output_dir"

    if [ $? -eq 0 ]; then
        echo "  ✓ Completed: $effect_type/$scenario_id"

        # Run bias analysis
        echo "  Analyzing: $effect_type/$scenario_id"
        python bias_effect_analyzer.py \
            --output-dir "$output_dir" \
            --bias-types "$effect_type" \
            --save "$output_dir/bias_analysis.json"
    else
        echo "  ✗ Failed: $effect_type/$scenario_id"
        return 1
    fi
}

# Run scenarios for each effect type
TOTAL_SCENARIOS=0
COMPLETED_SCENARIOS=0
FAILED_SCENARIOS=0

for effect_dir in $EFFECT_DIRS; do
    effect_path="$TEST_SUITE_DIR/$effect_dir"

    if [ ! -d "$effect_path" ]; then
        echo "Warning: Effect directory not found: $effect_path"
        continue
    fi

    echo ""
    echo "========================================"
    echo "Running scenarios for: $effect_dir"
    echo "========================================"

    # Find all scenario files in this effect directory
    scenario_files=$(find "$effect_path" -name "*.json" -type f)

    if [ -z "$scenario_files" ]; then
        echo "Warning: No scenario files found in $effect_path"
        continue
    fi

    # Create effect output directory
    mkdir -p "$OUTPUT_BASE_DIR/$effect_dir"

    # Run each scenario
    for scenario_file in $scenario_files; do
        TOTAL_SCENARIOS=$((TOTAL_SCENARIOS + 1))

        if [ "$PARALLEL" = true ]; then
            # Run in background
            run_scenario "$scenario_file" "$effect_dir" &
        else
            # Run sequentially
            if run_scenario "$scenario_file" "$effect_dir"; then
                COMPLETED_SCENARIOS=$((COMPLETED_SCENARIOS + 1))
            else
                FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
            fi
        fi
    done

    # If running in parallel, wait for all scenarios in this effect to complete
    if [ "$PARALLEL" = true ]; then
        wait
        # Count completed vs failed (simplified - just count output directories)
        completed_count=$(find "$OUTPUT_BASE_DIR/$effect_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
        COMPLETED_SCENARIOS=$((COMPLETED_SCENARIOS + completed_count))
    fi
done

echo ""
echo "========================================"
echo "Batch Execution Complete"
echo "========================================"
echo "Total scenarios: $TOTAL_SCENARIOS"
echo "Completed: $COMPLETED_SCENARIOS"
echo "Failed: $FAILED_SCENARIOS"
echo ""
echo "Results saved to: $OUTPUT_BASE_DIR"
echo ""

# Generate comparative analysis for each effect type
echo "========================================"
echo "Generating Comparative Analysis"
echo "========================================"

for effect_dir in $EFFECT_DIRS; do
    effect_output_path="$OUTPUT_BASE_DIR/$effect_dir"

    if [ ! -d "$effect_output_path" ]; then
        continue
    fi

    echo "Analyzing: $effect_dir"

    # Find all scenario output directories
    scenario_dirs=$(find "$effect_output_path" -mindepth 1 -maxdepth 1 -type d)

    if [ -z "$scenario_dirs" ]; then
        echo "  Warning: No scenario outputs found for $effect_dir"
        continue
    fi

    # Run multi-scenario comparison (if analyze_scenarios.py supports it)
    python analyze_scenarios.py \
        --multi-compare $scenario_dirs \
        --output "$effect_output_path/comparative_analysis" \
        2>/dev/null || echo "  Note: Comparative analysis not available"
done

echo ""
echo "========================================"
echo "All Done!"
echo "========================================"
echo "Review results in: $OUTPUT_BASE_DIR"
echo ""
echo "Next steps:"
echo "  1. Review individual scenario results in $OUTPUT_BASE_DIR/<effect>/<scenario>/"
echo "  2. Review comparative analyses in $OUTPUT_BASE_DIR/<effect>/comparative_analysis/"
echo "  3. Generate visualizations and summary reports"
echo ""
