# Scenario-Based Simulation Guide

This guide explains how to run comprehensive simulations comparing different LLM architectures for AI content generation and recommendation in social media applications.

## Overview

We have two predefined scenarios for comparing AI architectures:

### **Scenario 1: Llama-only (Baseline)**
- All agents use **Llama-3.1-8B-Instruct** for content generation
- Recommendation also uses **Llama-3.1-8B-Instruct**
- **Purpose**: Establish baseline performance metrics

### **Scenario 2: Mixed Generation**
- 50% of agents use **Llama-3.1-8B-Instruct**
- 50% of agents use **Mistral-7B-Instruct**
- Recommendation uses **Llama-3.1-8B-Instruct**
- **Purpose**: Test effects of architectural diversity on content ecosystem

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

Key packages:
- `transformers` - HuggingFace models
- `torch` - PyTorch for model inference
- `sentence-transformers` - Semantic similarity analysis
- `pandas`, `numpy` - Data analysis
- `matplotlib`, `seaborn` - Visualization
- `scipy` - Statistical tests

### 2. Authenticate with HuggingFace (for Llama models)

```bash
huggingface-cli login
```

### 3. Hardware Requirements

- **GPU**: Recommended 16GB+ VRAM (for running both Llama and Mistral simultaneously)
- **RAM**: 32GB+ recommended
- **Storage**: ~30GB for models

## Quick Start

### Option 1: Run Both Scenarios and Compare (Recommended)

```bash
# Make script executable
chmod +x run_and_compare_scenarios.sh

# Run with default settings (10 rounds, 20 users)
./run_and_compare_scenarios.sh

# Custom settings
./run_and_compare_scenarios.sh --rounds 15 --users 30
```

This will:
1. Run Scenario 1 (Llama-only)
2. Run Scenario 2 (Mixed generation)
3. Generate comprehensive analysis and comparison
4. Create visualizations

### Option 2: Run Individual Scenarios

```bash
# Run Scenario 1
python run_scenario.py --scenario 1 --rounds 10 --users 20

# Run Scenario 2
python run_scenario.py --scenario 2 --rounds 10 --users 20

# Analyze individual scenario
python analyze_scenarios.py --output-dir ./scenario1_output

# Compare scenarios
python analyze_scenarios.py --output-dir ./scenario1_output --compare ./scenario2_output
```

## Analysis Outputs

### 1. Content Diversity Metrics

**Topic Distribution**
- Shannon entropy of topic distribution
- Diversity across rounds

**Semantic Similarity**
- Average pairwise cosine similarity (using sentence transformers)
- Semantic diversity index (1 - similarity)

**Novelty Measures**
- Temporal diversity trends
- Round-by-round entropy

### 2. Promotion Patterns

**Visibility Analysis**
- Total and average views by architecture
- Engagement rates by model type

**Filter Bubble Detection**
- Herfindahl-Hirschman Index (HHI) per user
- Topic concentration measures
- Exposure diversity

### 3. Sentiment Dynamics

**Temporal Evolution**
- Mean sentiment by round
- Sentiment variance trends
- Extremity evolution

**Extremity Trends**
- Absolute sentiment scores over time
- Polarization indicators

**Architecture Comparison**
- Mean sentiment by model
- Sentiment variance by architecture

### 4. Engagement Patterns

**Network Formation**
- Response networks (who responds to whom)
- Cross-architecture interactions

**Virality Analysis**
- Top viral content identification
- Engagement distribution (Gini coefficient)
- Cascade detection

**Echo Chamber Metrics**
- Same-architecture response rate
- Cross-group interaction frequency

### 5. Polarization Indices

**Opinion Clustering**
- Temporal opinion evolution
- Between-group vs. within-group variance
- Polarization index

**Interaction Patterns**
- Cross-architecture engagement
- Opinion alignment in interactions

### 6. Architecture Effects

**Performance Comparison**
- Engagement scores by architecture
- Statistical significance tests (Mann-Whitney U)
- Agent-level performance analysis

**Model-specific Metrics**
- Generation statistics (tokens, count)
- Content quality indicators
- User preference patterns

## Output Files

After running simulations, you'll find:

```
scenario1_output/
├── all_content.csv           # All generated content with metadata
├── all_interactions.csv      # User-content interactions
├── agents.csv                # Agent configurations
├── users.csv                 # User profiles
├── feed_assignments.json     # What content each user saw
├── round_data.json          # Round-by-round state
├── comprehensive_analysis.json  # Full analysis results
└── analysis_plots/          # Visualizations
    ├── content_overview.png
    ├── temporal_dynamics.png
    └── architecture_comparison.png

scenario2_output/
└── (same structure)

comparison_results/
└── scenario_comparison.json  # Side-by-side comparison
```

## Visualization Outputs

### 1. Content Overview
- Topic distribution bar chart
- Architecture distribution
- Sentiment histogram
- Engagement distribution

### 2. Temporal Dynamics
- Sentiment evolution over rounds
- Engagement trends
- Opinion evolution (mean ± std)
- Content volume by round

### 3. Architecture Comparison
- Engagement boxplots by architecture
- Sentiment boxplots by architecture
- Total views/likes by architecture

## Customizing Scenarios

### Modifying Existing Scenarios

Edit the JSON configuration files:

**scenario1_llama_only.json**
```json
{
  "backends": [...],
  "agent_assignments": [
    {
      "agent_id": "agent_0",
      "backend_id": "llama",
      "persona": "Your Persona",
      "objective": "Your Objective",
      "temperature": 0.8,
      "response_style": "your_style"
    }
  ],
  "recommender": {
    "type": "llm",
    "backend_id": "llama",
    "use_personalization": true
  }
}
```

### Creating New Scenarios

1. Create new configuration file (e.g., `scenario3_custom.json`)
2. Define backends and agent assignments
3. Run with:
   ```bash
   python run_multi_llm_simulation.py --config scenario3_custom.json --rounds 10
   ```

## Analysis Options

### Single Scenario Analysis

```bash
python analyze_scenarios.py --output-dir ./scenario1_output
```

Generates:
- Comprehensive metrics JSON
- All visualization plots
- Statistical summaries

### Comparative Analysis

```bash
python analyze_scenarios.py --output-dir ./scenario1_output --compare ./scenario2_output
```

Additional outputs:
- Side-by-side metric comparisons
- Statistical significance tests
- Comparative visualizations

## Interpreting Results

### Content Diversity
- **Higher entropy** → more diverse topics
- **Lower semantic similarity** → more unique content
- **Stable diversity** → consistent ecosystem

### Promotion Patterns
- **Lower HHI** → less filter bubble effect
- **Uniform visibility** → fair content distribution
- **Architecture bias** → some models over-promoted

### Sentiment Dynamics
- **Increasing extremity** → polarization warning
- **High variance** → diverse perspectives
- **Convergence** → echo chamber formation

### Engagement Patterns
- **Low Gini** → equal engagement distribution
- **High echo chamber rate** → insular groups
- **Diverse networks** → healthy interaction

### Architecture Effects
- **Significant p-values** → real performance differences
- **Uniform engagement** → architecture-agnostic users
- **Architecture clustering** → preference patterns

## Research Questions

Use these scenarios to investigate:

1. **Does architectural diversity increase content diversity?**
   - Compare topic entropy between scenarios

2. **Do certain architectures create filter bubbles?**
   - Examine HHI scores and user exposure patterns

3. **Which architecture generates more engaging content?**
   - Compare engagement metrics and statistical tests

4. **Does mixing architectures affect polarization?**
   - Analyze opinion evolution and polarization indices

5. **Are there echo chamber effects based on model type?**
   - Check same-architecture response rates

## Performance Tips

### Reducing Simulation Time

1. **Reduce rounds**: Start with 5-10 rounds for testing
2. **Fewer users**: 10-15 users for quick experiments
3. **Skip embeddings**: Set `EMBEDDINGS_AVAILABLE=False` in analyze_scenarios.py
4. **Batch generation**: Already optimized in backends

### Managing Memory

1. **Sequential scenario runs**: Run scenarios separately
2. **Clear GPU cache**: Between scenarios
3. **Reduce context length**: Edit `max_new_tokens` in configs

### Scaling Up

For production runs:
- 20-50 rounds for stable metrics
- 30-50 users for robust statistics
- Multiple random seeds for reproducibility

## Troubleshooting

### Out of Memory (OOM)

```python
# In run_scenario.py, reduce batch sizes or switch models
# Use smaller models: Llama-3.2-3B instead of 8B
```

### Slow Generation

- Check if GPU is being used: `nvidia-smi`
- Reduce `max_new_tokens` in backend configs
- Use mock backends for testing logic first

### Missing Embeddings

```bash
pip install sentence-transformers
```

Or disable in `analyze_scenarios.py`:
```python
EMBEDDINGS_AVAILABLE = False
```

## Citation

If you use this framework in research:

```bibtex
@software{multi_llm_scenarios,
  title = {Multi-LLM Social Media Simulation Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/AI-generation-2-AI-recommendation}
}
```

## Support

For issues or questions:
- Check documentation in README.md
- Review configuration files for examples
- Open GitHub issue for bugs

Happy experimenting! 🚀
