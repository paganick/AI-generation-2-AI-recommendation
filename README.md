# AI Content & Recommender System Dynamics Simulator

A research prototype for studying feedback loops between AI-generated social media content and AI-powered recommendation systems.

## Overview

This simulator models a three-layer system:

1. **Content Generation Layer**: Multiple AI agents with diverse personas generate social media posts and responses
2. **Recommendation Layer**: An AI-powered recommender system ranks and filters content for users
3. **Human Impact Layer**: Simulated users whose opinions evolve based on content exposure

The system captures feedback loops: recommender outputs influence what content is seen, which influences future content generation, creating complex dynamics.

## Features

- **Multi-agent content generation** with configurable personas and objectives
- **Multiple recommender strategies**: relevance-based, diversity-aware, LLM-powered, or random baseline
- **Opinion dynamics modeling** using bounded confidence models
- **Comprehensive metrics** including diversity, polarization, engagement inequality
- **Visualization tools** for analyzing system evolution

## Installation

### Prerequisites

```bash
# Install Python dependencies
pip install transformers torch accelerate numpy matplotlib --break-system-packages

# For Llama models, authenticate with HuggingFace
huggingface-cli login
```

### Files

- `simulation_framework.py`: Core simulation components
- `llm_backend.py`: LLM inference backend (Llama-3.1-8B support)
- `run_simulation.py`: Main simulation runner
- `visualize_results.py`: Analysis and visualization tools

## Quick Start

### 1. Test with Mock LLM (no GPU required)

```bash
python run_simulation.py --mock --rounds 3 --agents 4 --users 8
```

### 2. Run with Llama-3.1-8B

```bash
python run_simulation.py --rounds 5 --agents 6 --users 10 --recommender diversity
```

### 3. Visualize Results

```bash
python visualize_results.py
```

## Usage

### Basic Simulation

```bash
python run_simulation.py [OPTIONS]
```

**Options:**
- `--rounds N`: Number of simulation rounds (default: 5)
- `--agents N`: Number of AI agents (default: 6)
- `--users N`: Number of simulated users (default: 10)
- `--recommender STRATEGY`: Recommender strategy
  - `relevance`: Keyword-based relevance matching
  - `diversity`: Diversity-aware ranking
  - `llm`: LLM-powered ranking (expensive)
  - `random`: Random baseline
- `--mock`: Use mock LLM for testing (fast, no model loading)
- `--model NAME`: HuggingFace model name (default: meta-llama/Llama-3.1-8B-Instruct)

### Example Configurations

**Fast testing:**
```bash
python run_simulation.py --mock --rounds 3
```

**Full simulation with diversity promotion:**
```bash
python run_simulation.py --rounds 10 --recommender diversity --agents 8 --users 15
```

**Compare recommender strategies:**
```bash
# Run multiple simulations with different strategies
for strategy in relevance diversity random; do
    python run_simulation.py --rounds 5 --recommender $strategy
    mv simulation_results.json results_${strategy}.json
done
```

## Output

### simulation_results.json

Contains:
- All generated content items with metadata
- Complete metrics history for each round
- Summary statistics
- Timestamp

### Visualizations (in /mnt/user-data/outputs/)

1. **metrics_plot.png**: Time series of all system metrics
2. **content_analysis.png**: Content characteristics (topics, sentiment, engagement)
3. **summary_report.txt**: Textual analysis with key findings

## Metrics

### Content-Level Metrics

- **Content Diversity**: Shannon entropy of topic distribution (higher = more diverse)
- **Sentiment Distribution**: Mean and variance of sentiment scores
- **Topic Concentration**: Herfindahl-Hirschman Index (lower = less concentrated)

### System-Level Metrics

- **Engagement Gini**: Inequality in attention distribution (0 = equal, 1 = winner-take-all)
- **Opinion Variance**: Spread of user opinions
- **Polarization**: Average distance from opinion center

## Architecture

### Agent Personas (Default Configuration)

1. **Progressive Activist**: Advocates for social/environmental causes (temp: 0.8)
2. **Tech Entrepreneur**: Discusses innovation and opportunities (temp: 0.7)
3. **Academic Researcher**: Evidence-based, analytical insights (temp: 0.5)
4. **Skeptical Journalist**: Critical questioning, accountability (temp: 0.6)
5. **Community Organizer**: Consensus-building, mobilization (temp: 0.7)
6. **Contrarian Thinker**: Challenges mainstream narratives (temp: 0.8)

### Simulation Flow

```
Round N:
  1. Recommender generates personalized feeds for each user
  2. Users' opinions updated based on content exposure (bounded confidence)
  3. Engagement simulated based on feed assignments
  4. Agents generate responses to popular content
  5. New content added to pool
  6. Metrics collected
```

## Extending the Prototype

### Add New Agent Personas

Edit `create_diverse_agents()` in `run_simulation.py`:

```python
{
    "persona": "Your persona description",
    "objective": "What this agent aims to achieve",
    "temperature": 0.7,  # 0.0-1.0, higher = more random
    "response_style": "descriptive adjective"
}
```

### Implement Custom Recommender Strategy

Add to `RecommenderSystem` class in `simulation_framework.py`:

```python
def _rank_by_custom(self, content_pool, user, k):
    # Your ranking logic here
    return ranked_items[:k]
```

### Add New Metrics

Add to `MetricsCollector` class:

```python
def compute_your_metric(self, content_pool, users):
    # Compute your metric
    return metric_value
```

## Research Applications

### Experimental Questions

1. **Feedback loop stability**: Do AI-AI systems converge or diverge?
2. **Diversity-polarization tradeoff**: Does diversity promotion reduce polarization?
3. **Agent heterogeneity**: How does persona diversity affect ecosystem health?
4. **Recommendation strategies**: Which strategies prevent filter bubbles?
5. **Temporal dynamics**: Are there critical transitions or phase shifts?

### Parameter Sweeps

Study sensitivity to:
- Number of agents and personas
- Recommender strategy
- User tolerance (bounded confidence parameter)
- Temperature settings for generation
- Initial topic distribution

### Validation Approaches

1. **Qualitative**: Compare generated content with real social media
2. **Distributional**: Match statistical properties (engagement, sentiment)
3. **Dynamic**: Compare temporal patterns with real-world data
4. **Theoretical**: Derive equilibria, stability conditions

## Performance Considerations

### Computational Requirements

- **Mock LLM**: ~1 second per round (CPU only)
- **Llama-3.1-8B**: ~30-60 seconds per round (GPU recommended)
- **Memory**: ~8GB GPU VRAM for Llama-3.1-8B

### Optimization Tips

1. Use mock LLM for prototyping and debugging
2. Reduce `max_new_tokens` for faster generation
3. Use smaller models (e.g., Llama-3.2-1B) for experiments
4. Batch generation when possible
5. Profile with fewer rounds/agents initially

### HPC Cluster Usage

For large-scale experiments:

```bash
#!/bin/bash
#SBATCH --job-name=ai_dynamics
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

module load python/3.10
module load cuda/11.8

python run_simulation.py --rounds 20 --agents 10 --users 50 --recommender diversity
```

## Known Limitations

1. **Simplified sentiment analysis**: Uses keyword matching (could integrate BERT-based models)
2. **1D opinion space**: Could extend to multi-dimensional opinions
3. **No user-to-user interaction**: Only AI agents generate content
4. **Static agent personas**: Could implement learning/adaptation
5. **Coarse-grained time**: Discrete rounds vs. continuous time

## Future Extensions

### High Priority

1. **Integrate BERT for sentiment/topic classification** (leverage your existing pipeline)
2. **Add user-generated content** (mixed AI-human ecosystem)
3. **Implement temporal decay** (older content less likely to be shown)
4. **Network effects** (users influence each other)
5. **Multi-platform simulation** (cross-platform dynamics)

### Research Features

1. **Interventions**: Test policy changes (e.g., diversity quotas)
2. **Adversarial agents**: Study misinformation dynamics
3. **Platform comparison**: Model different algorithmic designs
4. **Heterogeneous recommenders**: Different users see different algorithms
5. **Real-world initialization**: Bootstrap from actual social media data

## Citation

If you use this code in research, please cite:

```bibtex
@software{ai_content_dynamics_simulator,
  author = {[Your Name]},
  title = {AI Content and Recommender System Dynamics Simulator},
  year = {2024},
  institution = {University of Zurich, Social Computing Group}
}
```

## License

[Specify license - e.g., MIT, Apache 2.0, etc.]

## Contact

Nick - Social Computing Group, University of Zurich

For questions, suggestions, or collaborations, please reach out!

## Acknowledgments

This work builds on research in:
- Opinion dynamics (Hegselmann-Krause, voter models)
- Algorithmic fairness and feedback loops
- Multi-agent systems
- Computational social science

Special thanks to the Social Computing Group for feedback and support.`