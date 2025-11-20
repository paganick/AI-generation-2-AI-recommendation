# Multi-LLM Recommendation System Scenarios Guide

## Overview

This guide describes the available scenarios for testing different combinations of LLM architectures for content generation and recommendation in a social media simulation environment.

## Available Scenarios

### Scenario 1: Llama-3.1-8B Only
- **Config File**: `scenario1_llama_only.json`
- **Output Directory**: `./scenario1_output`
- **Description**: All content generation and recommendations use Llama-3.1-8B-Instruct
- **Use Case**: Baseline scenario with uniform architecture

### Scenario 2: Mixed Generation (Llama + Mistral), Llama Recommendation
- **Config File**: `scenario2_mixed_generation.json`
- **Output Directory**: `./scenario2_output`
- **Description**:
  - Content generation: 50% Llama-3.1-8B, 50% Mistral-7B
  - Recommendations: Llama-3.1-8B
- **Use Case**: Test whether Llama recommender favors Llama-generated content

### Scenario 3: Mistral-7B Only
- **Config File**: `scenario3_mistral_only.json`
- **Output Directory**: `./scenario3_output`
- **Description**: All content generation and recommendations use Mistral-7B-Instruct
- **Use Case**: Baseline scenario with Mistral architecture for comparison with Scenario 1

### Scenario 4: Mixed Generation (Llama + Mistral), Mistral Recommendation
- **Config File**: `scenario4_mixed_mistral_rec.json`
- **Output Directory**: `./scenario4_output`
- **Description**:
  - Content generation: 50% Llama-3.1-8B, 50% Mistral-7B
  - Recommendations: Mistral-7B
- **Use Case**: Test whether Mistral recommender favors Mistral-generated content

## Running Scenarios

### Single Scenario

```bash
# Run scenario 1
python run_scenario.py --scenario 1 --rounds 10 --users 20

# Run scenario 2
python run_scenario.py --scenario 2 --rounds 10 --users 20

# Run scenario 3
python run_scenario.py --scenario 3 --rounds 10 --users 20

# Run scenario 4
python run_scenario.py --scenario 4 --rounds 10 --users 20
```

### Custom Output Directory

```bash
python run_scenario.py --scenario 1 --rounds 15 --users 30 --output ./custom_output
```

## Analysis Capabilities

### Single Scenario Analysis

```bash
python analyze_scenarios.py --output-dir ./scenario1_output
```

This performs comprehensive analysis including:

1. **Content Diversity Analysis**
   - Topic distribution and entropy
   - Architecture distribution
   - Semantic similarity (if sentence-transformers available)

2. **Promotion Patterns Analysis**
   - Visibility by architecture
   - Engagement rate by architecture
   - Filter bubble analysis

3. **Sentiment Dynamics Analysis**
   - Temporal sentiment evolution
   - Sentiment by architecture
   - Polarization measures

4. **Engagement Patterns Analysis**
   - Engagement distribution (Gini coefficient)
   - Virality analysis
   - Echo chamber detection

5. **Polarization Analysis**
   - Opinion evolution
   - Between-group interactions

6. **Architecture Effects Analysis**
   - Performance comparison across architectures
   - Statistical significance tests

7. **NEW: Recommendation-Topic-Engagement Correlations**
   - Topic recommendation frequency
   - Engagement metrics by topic
   - Correlation between recommendation frequency and engagement
   - Topic diversity in user feeds

8. **NEW: LLM Favoritism in Mixed Scenarios**
   - Distribution of recommended content by architecture
   - Favoritism bias calculation (recommendation % - generation %)
   - Statistical significance testing (Chi-square)
   - Favoritism analysis by topic
   - Favoritism analysis by sentiment
   - Engagement outcomes of favoritism

### Two-Scenario Comparison

```bash
python analyze_scenarios.py --output-dir ./scenario1_output --compare ./scenario2_output
```

### Multi-Scenario Comparison

```bash
# Compare all four scenarios
python analyze_scenarios.py --multi-compare \
    ./scenario1_output \
    ./scenario2_output \
    ./scenario3_output \
    ./scenario4_output

# With custom names
python analyze_scenarios.py --multi-compare \
    ./scenario1_output \
    ./scenario2_output \
    ./scenario3_output \
    ./scenario4_output \
    --scenario-names "Llama Only" "Mixed-Llama Rec" "Mistral Only" "Mixed-Mistral Rec"
```

## Key Research Questions

### 1. Recommendation-Topic-Engagement Correlations

**Research Questions:**
- Which topics get recommended most frequently?
- Is there a correlation between recommendation frequency and engagement?
- Does topic diversity in feeds affect user engagement?

**Relevant Metrics:**
- Topic recommendation counts
- Correlation coefficients (recommendation ↔ engagement/views/likes)
- Average topic diversity in user feeds

**Where to Find:**
- Analysis output: `recommendation_topic_engagement` section
- Visualization: `recommendation_topic_engagement.png`

### 2. LLM Favoritism in Mixed Scenarios

**Research Questions:**
- Does the recommendation system favor content from specific LLM architectures?
- Is favoritism statistically significant?
- Does favoritism vary by topic or sentiment?
- Does favoritism correlate with engagement outcomes?

**Relevant Metrics:**
- Baseline proportions (generation %)
- Recommendation proportions (recommendation %)
- Favoritism bias (recommendation % - generation %)
- Chi-square test p-value
- Topic-specific and sentiment-specific favoritism

**Where to Find:**
- Analysis output: `llm_favoritism` section
- Visualization: `llm_favoritism_analysis.png` (only for mixed scenarios)

## Output Files

After running analysis, you'll find:

### Data Files
- `all_content.csv` - All generated content with metadata
- `all_interactions.csv` - User interaction logs
- `agents.csv` - Agent configurations
- `users.csv` - User profiles
- `feed_assignments.json` - Content recommended to each user per round
- `round_data.json` - Round-by-round simulation state
- `comprehensive_analysis.json` - Complete analysis results

### Visualization Files
- `content_overview.png` - Topic, architecture, sentiment, engagement distributions
- `temporal_dynamics.png` - Evolution over time
- `architecture_comparison.png` - Performance by architecture
- `recommendation_topic_engagement.png` - NEW: Topic-engagement correlations
- `llm_favoritism_analysis.png` - NEW: Favoritism analysis (mixed scenarios only)

### Comparison Files (for multi-scenario comparison)
- `multi_scenario_comparison.png` - Side-by-side comparison visualization
- `multi_scenario_comparison.json` - Detailed comparison data

## Example Workflow

### Complete Analysis Pipeline

```bash
# Step 1: Run all scenarios
python run_scenario.py --scenario 1 --rounds 10 --users 20
python run_scenario.py --scenario 2 --rounds 10 --users 20
python run_scenario.py --scenario 3 --rounds 10 --users 20
python run_scenario.py --scenario 4 --rounds 10 --users 20

# Step 2: Compare all scenarios
python analyze_scenarios.py --multi-compare \
    ./scenario1_output \
    ./scenario2_output \
    ./scenario3_output \
    ./scenario4_output \
    --scenario-names \
    "Llama Only" \
    "Mixed (Llama Rec)" \
    "Mistral Only" \
    "Mixed (Mistral Rec)"

# Step 3: Review results
# - Check console output for key metrics
# - Open ./multi_comparison_results/multi_scenario_comparison.png
# - Review individual scenario visualizations in each output directory
```

## Interpreting Results

### Understanding Favoritism Bias

- **Positive bias**: Architecture is over-represented in recommendations (favored)
- **Negative bias**: Architecture is under-represented in recommendations (disfavored)
- **Near-zero bias**: Proportional representation (no favoritism)

Example:
```
Favoritism Bias (Recommendation % - Generation %):
  llama: +5.2% (favored)
  mistral: -5.2% (disfavored)
```

This indicates the Llama-based recommender is recommending 5.2% more Llama content than would be expected based on generation proportions.

### Understanding Correlations

- **Positive correlation** (0 to 1): As one metric increases, the other increases
- **Negative correlation** (-1 to 0): As one metric increases, the other decreases
- **Correlation magnitude**:
  - 0.0-0.3: Weak
  - 0.3-0.7: Moderate
  - 0.7-1.0: Strong

### Statistical Significance

- **p < 0.05**: Significant result (effect is likely real)
- **p ≥ 0.05**: Not significant (effect could be due to chance)

## Notes

1. **GPU Requirements**: Running real LLM models requires a GPU with sufficient VRAM
2. **Computation Time**: Each scenario takes 10-30 minutes depending on rounds/users/hardware
3. **Semantic Analysis**: Install `sentence-transformers` for semantic similarity analysis:
   ```bash
   pip install sentence-transformers
   ```
4. **Mock Mode**: For testing without GPU, set `use_mock: true` in scenario config files

## Troubleshooting

### Out of Memory
- Reduce number of users: `--users 10`
- Reduce number of rounds: `--rounds 5`
- Use smaller models or mock mode

### Import Errors
```bash
pip install pandas numpy scipy matplotlib seaborn
pip install sentence-transformers  # optional
```

### Analysis Fails
- Ensure simulation completed successfully
- Check that output directory contains all required CSV/JSON files
- Verify file paths in command

## Contact & Support

For issues or questions, please refer to the main project documentation or create an issue in the repository.
