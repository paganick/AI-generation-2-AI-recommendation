# Bias Testing Framework - Implementation Summary

## What Was Created

I've extended your recommendation system simulation framework to support systematic testing of 5 different bias effects. Here's a complete overview:

---

## 🎯 Core Components

### 1. **bias_effects.py** - Bias Effect Specification System
- **Purpose**: Define and configure different bias effect tests
- **Key Classes**:
  - `BiasEffectType`: Enum of 5 bias types (self-preference, topic, political, dialect, gender)
  - `BiasEffectSpec`: Specification for a bias effect test
  - `AgentAttributeSpec`: Agent attributes to control (gender, dialect, political stance)
  - `ContentAttributeSpec`: Content attributes to track (topics, political positions)
  - `BiasTestConfiguration`: Complete test configuration
- **Pre-built Functions**:
  - `build_self_preference_test()`: Create self-preference test
  - `build_topic_bias_test()`: Create topic bias test
  - `build_political_bias_test()`: Create political bias test
  - `build_dialect_bias_test()`: Create dialect bias test
  - `build_gender_bias_test()`: Create gender bias test

### 2. **generate_bias_test_suite.py** - Test Suite Generator
- **Purpose**: Generate comprehensive scenario configurations for bias testing
- **Key Class**: `BiasTestSuiteGenerator`
- **Methods**:
  - `generate_self_preference_suite()`: 4+ scenarios testing LLM self-preference
  - `generate_topic_bias_suite()`: Scenarios testing topic favoritism
  - `generate_political_bias_suite()`: Scenarios testing political bias
  - `generate_dialect_bias_suite()`: Scenarios testing language/dialect bias
  - `generate_gender_bias_suite()`: Scenarios testing gender bias
  - `generate_full_test_suite()`: Generate all tests at once
- **CLI Usage**:
  ```bash
  python generate_bias_test_suite.py --output ./test_suite --effects all --rounds 10 --users 20
  ```

### 3. **bias_effect_analyzer.py** - Bias Analysis Module
- **Purpose**: Analyze simulation results for each bias effect
- **Key Class**: `BiasEffectAnalyzer`
- **Analysis Methods**:
  - `analyze_self_preference_bias()`: LLM architecture favoritism analysis
  - `analyze_topic_preference_bias()`: Topic over/under-representation
  - `analyze_political_bias()`: Political position bias and echo chambers
  - `analyze_dialect_bias()`: Language/dialect visibility bias
  - `analyze_gender_bias()`: Gender visibility and topic interactions
  - `generate_full_bias_report()`: Comprehensive report for all applicable biases
- **Statistical Tests**:
  - Chi-square tests for distribution comparisons
  - Kruskal-Wallis H-test for ranking comparisons
  - Entropy calculations for diversity metrics
- **CLI Usage**:
  ```bash
  python bias_effect_analyzer.py --output-dir ./results --bias-types gender_bias --save report.json
  ```

### 4. **run_bias_test_suite.sh** - Batch Execution Script
- **Purpose**: Run entire test suites in batch
- **Features**:
  - Sequential or parallel execution
  - Automatic analysis after each scenario
  - Progress tracking and error handling
  - Comparative analysis generation
- **Usage**:
  ```bash
  ./run_bias_test_suite.sh --test-suite ./test_suite --output ./results --rounds 10 --users 20
  ```

### 5. **demo_bias_testing.py** - Demonstration Script
- **Purpose**: Show how to use the framework programmatically
- **Demos**:
  - Creating single bias test configurations
  - Generating complete test suites
  - Inspecting scenario configurations
  - Understanding bias concepts
- **Usage**:
  ```bash
  python demo_bias_testing.py
  ```

---

## 📚 Documentation

### 1. **BIAS_EFFECTS_DESIGN.md** - Design Document
- Complete architecture overview
- Detailed specification for each bias effect
- Implementation phases
- Statistical rigor considerations
- Scalability and extensibility design

### 2. **BIAS_TESTING_GUIDE.md** - User Guide
- Quick start tutorial
- Detailed usage instructions
- Explanation of each bias effect
- Interpreting results and metrics
- Statistical significance interpretation
- Troubleshooting guide
- Best practices
- Example workflows

### 3. **IMPLEMENTATION_SUMMARY.md** - This Document
- Overview of what was created
- Quick reference for all components
- Example usage patterns

---

## 🚀 Quick Start

### Step 1: Generate Test Suite

```bash
# Generate test suite for all 5 bias effects
python generate_bias_test_suite.py \
    --output ./my_test_suite \
    --effects all \
    --rounds 10 \
    --users 20 \
    --agents 6
```

This creates:
```
my_test_suite/
├── self_preference/
│   ├── self_pref_baseline_llama.json
│   ├── self_pref_baseline_mistral.json
│   ├── self_pref_test_llama_rec.json
│   └── self_pref_test_mistral_rec.json
├── topic_bias/
│   ├── topic_bias_llama_rec.json
│   └── topic_bias_mistral_rec.json
├── political_bias/
├── dialect_bias/
├── gender_bias/
└── test_suite_index.json
```

### Step 2: Run Test Suite

```bash
# Run all scenarios
./run_bias_test_suite.sh \
    --test-suite ./my_test_suite \
    --output ./bias_results \
    --rounds 10 \
    --users 20
```

### Step 3: Analyze Results

```bash
# Analysis is automatic during execution
# Results are in: ./bias_results/<effect>/<scenario>/bias_analysis.json

# Example: View gender bias results
cat ./bias_results/gender_bias/gender_bias_neutral_llama_rec/bias_analysis.json | jq '.bias_analyses.gender_bias'
```

---

## 📊 Understanding the 5 Bias Effects

### 1. Self-Preference for LLM-Produced Content

**Tests whether LLM recommenders prefer content from their own architecture.**

**Example Scenario**:
- 50% content from Llama, 50% from Mistral
- Llama recommender
- **Expected if biased**: Llama content gets > 50% of recommendations

**Key Metrics**:
- `favoritism_score`: (% recommended - % generated)
- Positive score = favoritism, negative = discrimination

### 2. Topic Preference Bias

**Tests whether certain topics are systematically over/under-recommended.**

**Example Scenario**:
- Agents evenly distributed across topics (climate, AI ethics, politics, etc.)
- Measure: Are all topics equally represented in recommendations?

**Key Metrics**:
- `topic_bias_scores`: Over/under-representation per topic
- `temporal_trends`: How bias evolves over time

### 3. Political Position Bias

**Tests whether recommenders favor certain political positions.**

**Example Scenario**:
- Agents balanced across political stances (progressive, conservative, libertarian, centrist)
- Measure: Representation and echo chamber formation

**Key Metrics**:
- `position_bias_scores`: Over/under-representation by political stance
- `echo_chamber_analysis`: Cross-position exposure (entropy)

### 4. Language/Dialect Visibility Bias

**Tests whether minority dialect communities are less visible.**

**Example Scenario**:
- Human agents using different dialects (AAVE, SAE, Southern, General American)
- Same content quality across dialects
- Measure: Visibility and ranking position

**Key Metrics**:
- `visibility_bias_scores`: Recommendation rate difference
- `ranking_position_analysis`: Average ranking by dialect

### 5. Gender Bias

**Tests whether author gender affects content visibility.**

**Example Scenario**:
- Agents balanced across genders (male, female, non-binary)
- Two variants: gender-neutral topics and gender-stereotyped topics
- Measure: Visibility and topic-gender interactions

**Key Metrics**:
- `visibility_bias_scores`: Recommendation rate difference
- `topic_gender_interaction`: How gender bias varies by topic

---

## 🔧 Customization Examples

### Test Only Specific Bias Effects

```bash
# Only test gender and dialect bias
python generate_bias_test_suite.py \
    --output ./focused_test_suite \
    --effects gender_bias dialect_bias \
    --rounds 15 \
    --users 30
```

### Scale Up for Production

```bash
# Large-scale testing
python generate_bias_test_suite.py \
    --output ./production_suite \
    --effects all \
    --rounds 100 \
    --users 200 \
    --agents 12
```

### Programmatic Usage

```python
from generate_bias_test_suite import BiasTestSuiteGenerator

# Create custom test suite
generator = BiasTestSuiteGenerator(output_dir="./custom_suite")

# Generate only gender bias scenarios
gender_scenarios = generator.generate_gender_bias_suite(
    num_agents=8,
    num_rounds=20,
    num_users=40
)

# Inspect first scenario
print(gender_scenarios[0]['name'])
print(gender_scenarios[0]['agent_assignments'])

# Save to disk
generator.save_test_suite({"gender_bias": gender_scenarios})
```

---

## 📈 Analyzing Results

### Automated Analysis

Each scenario automatically generates `bias_analysis.json` with:

```json
{
  "metadata": {
    "total_content": 120,
    "total_interactions": 2400
  },
  "bias_analyses": {
    "gender_bias": {
      "visibility_bias_scores": {
        "male": 7.2,
        "female": -5.8,
        "non-binary": -1.4
      },
      "statistical_test": {
        "p_value": 0.003,
        "significant": true
      },
      "ranking_position_analysis": {...},
      "engagement_by_gender": {...}
    }
  }
}
```

### Manual Analysis

```bash
# Analyze specific scenario
python bias_effect_analyzer.py \
    --output-dir ./results/my_scenario \
    --bias-types gender_bias topic_bias \
    --save ./analysis_report.json
```

### Interpreting Results

**Statistical Significance**:
- p < 0.05 = significant bias
- p < 0.01 = strong bias
- p < 0.001 = very strong bias

**Effect Sizes**:
- Bias score < 2% = minimal
- 2-5% = small but noticeable
- 5-10% = moderate
- > 10% = large, concerning

---

## 🛠 Integration with Existing Code

### Required Extensions to Existing Classes

To fully utilize the bias testing framework, you'll need to extend your existing `Agent` and `ContentItem` classes:

#### Agent Extensions (in `simulation_framework.py`)

```python
@dataclass
class Agent:
    # ... existing fields ...

    # NEW: Add these optional fields
    gender: Optional[str] = None
    dialect: Optional[str] = None
    political_stance: Optional[str] = None
    is_human: bool = False
```

#### ContentItem Extensions (in `simulation_framework.py`)

```python
@dataclass
class ContentItem:
    # ... existing fields ...

    # NEW: Add these optional fields
    political_position: Optional[str] = None
    dialect_type: Optional[str] = None
    author_gender: Optional[str] = None
    author_is_human: bool = False
```

#### Configuration Extensions (in `multi_llm_config.py`)

```python
@dataclass
class AgentLLMAssignment:
    # ... existing fields ...

    # NEW: Add these optional fields
    gender: Optional[str] = None
    dialect: Optional[str] = None
    political_stance: Optional[str] = None
    is_human: bool = False
```

**Note**: The bias testing framework will work with scenarios that have these attributes. Scenarios without these attributes will return informative error messages when analyzing those specific bias types.

---

## 📁 File Reference

| File | Purpose | Type |
|------|---------|------|
| `bias_effects.py` | Bias effect specifications | Core Module |
| `generate_bias_test_suite.py` | Test suite generator | Tool |
| `bias_effect_analyzer.py` | Bias analysis | Tool |
| `run_bias_test_suite.sh` | Batch execution script | Script |
| `demo_bias_testing.py` | Demonstration | Example |
| `BIAS_EFFECTS_DESIGN.md` | Design document | Documentation |
| `BIAS_TESTING_GUIDE.md` | User guide | Documentation |
| `IMPLEMENTATION_SUMMARY.md` | This file | Documentation |

---

## 🎓 Example Workflows

### Workflow 1: Quick Exploration

```bash
# 1. Generate small test suite
python generate_bias_test_suite.py --output ./quick_test --effects gender_bias --rounds 5 --users 10

# 2. Run one scenario manually
python run_multi_llm_simulation.py \
    --config ./quick_test/gender_bias/gender_bias_neutral_llama_rec.json \
    --rounds 5 --users 10 --output ./results/test1

# 3. Analyze
python bias_effect_analyzer.py --output-dir ./results/test1 --bias-types gender_bias
```

### Workflow 2: Comprehensive Study

```bash
# 1. Generate full test suite
python generate_bias_test_suite.py --output ./full_study --effects all --rounds 50 --users 100

# 2. Run all scenarios
./run_bias_test_suite.sh --test-suite ./full_study --output ./full_results --rounds 50 --users 100

# 3. Review all bias analyses
find ./full_results -name "bias_analysis.json" -exec echo "=== {} ===" \; -exec cat {} \;
```

### Workflow 3: Focused Investigation

```bash
# 1. Generate only self-preference tests
python generate_bias_test_suite.py --output ./self_pref_study --effects self_preference --rounds 20 --users 40

# 2. Run scenarios
./run_bias_test_suite.sh --test-suite ./self_pref_study --output ./self_pref_results

# 3. Compare across recommender architectures
# (Results will show if Llama or Mistral has stronger self-preference)
```

---

## ✅ Validation Checklist

Before running large-scale tests:

- [ ] Run `python demo_bias_testing.py` to verify framework works
- [ ] Generate a small test suite (5 rounds, 10 users)
- [ ] Run one scenario manually to verify it completes
- [ ] Check that analysis produces expected output format
- [ ] Review statistical test results for reasonableness
- [ ] Scale up gradually (10 → 20 → 50 → 100 rounds)

---

## 🐛 Known Limitations

1. **LLM API Costs**: Large-scale tests with many rounds/users can be expensive
2. **Computation Time**: 100 rounds with 200 users can take hours per scenario
3. **Mock Mode**: For rapid iteration, use `use_mock: true` in backend config
4. **Statistical Power**: Small tests (< 10 rounds) may not detect subtle biases

---

## 🚧 Future Enhancements

Potential extensions to consider:

1. **Intersectional Bias Analysis**: Gender × Political Stance × Topic
2. **Temporal Bias Evolution**: How bias changes over time
3. **User-Level Bias**: Do different users experience different biases?
4. **Comparative Dashboards**: Interactive visualization of bias across scenarios
5. **Automated Reporting**: HTML/PDF reports with charts and tables
6. **Confidence Intervals**: Bootstrap methods for effect size uncertainty
7. **Multi-Architecture Comparisons**: Test GPT, Claude, etc. alongside Llama/Mistral

---

## 📞 Support

- **Documentation**: See `BIAS_TESTING_GUIDE.md` for detailed usage
- **Design**: See `BIAS_EFFECTS_DESIGN.md` for architecture details
- **Examples**: Run `python demo_bias_testing.py` for demos
- **Issues**: Check error messages in bias_analysis.json

---

## 🎉 Summary

You now have a **complete, extensible framework** for systematically testing 5 different bias effects in LLM-powered recommendation systems:

1. ✅ **Self-Preference Bias** - Do LLMs favor their own architecture?
2. ✅ **Topic Preference Bias** - Are certain topics over/under-recommended?
3. ✅ **Political Bias** - Do recommenders favor political positions?
4. ✅ **Dialect Bias** - Are minority dialects less visible?
5. ✅ **Gender Bias** - Does gender affect content visibility?

The framework is:
- **Modular**: Each bias effect is independently testable
- **Scalable**: From 5-round demos to 100-round production studies
- **Statistical**: Includes significance testing and effect sizes
- **Extensible**: Easy to add new bias dimensions
- **Automated**: Batch execution and analysis

**Next Steps**: Start with the demo, then generate and run your first test suite!
