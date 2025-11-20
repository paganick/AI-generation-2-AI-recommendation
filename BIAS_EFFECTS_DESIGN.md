# Bias Effects Testing Framework - Design Document

## Overview
This document describes the extension to the recommendation system simulation framework to support systematic testing of multiple bias effects. The design enables scalable, reproducible testing of different bias dimensions while maintaining backward compatibility.

## Bias Effects to Test

### 1. Self-Preference for LLM-Produced Content
**Question**: Do LLM recommenders prefer content from their own architecture vs. other LLMs vs. humans?
- **Variants**:
  - Same architecture (Llama → Llama content)
  - Different architecture (Llama → Mistral content)
  - LLM vs. human content
- **Controls**: Topic distribution, content quality, temporal effects
- **Metrics**: Recommendation ratio, ranking position, engagement outcomes

### 2. Topic/Issue Preference Bias
**Question**: Do recommenders systematically prefer or suppress certain topics?
- **Test Topics**: Climate change, AI ethics, politics, healthcare, education, entertainment
- **Variants**:
  - Uniform topic distribution in generation
  - Measure topic distribution in recommendations
- **Controls**: Content quality, engagement potential, author diversity
- **Metrics**: Topic representation ratio (recommended/generated), temporal trends

### 3. Political Position Bias
**Question**: Do recommenders favor certain political positions?
- **Test Positions**: Progressive, conservative, libertarian, centrist
- **Variants**:
  - Explicit political content
  - Implicit political framing
- **Controls**: Topic, sentiment, argument quality
- **Metrics**: Position representation, echo chamber formation, cross-position exposure

### 4. Language/Dialect Bias (Visibility of Minority Communities)
**Question**: Are human creators using minority dialects less visible?
- **Test Dialects**:
  - African American Vernacular English (AAVE)
  - Standard American English (SAE)
  - Other regional/social dialects
- **Variants**:
  - Human-only content with varied dialects
  - Mixed human/LLM with dialect markers
- **Controls**: Content substance, topic, author expertise
- **Metrics**: Visibility (recommendation rate), engagement, ranking position

### 5. Gender Bias
**Question**: Does author gender affect content visibility and recommendation?
- **Test Attributes**:
  - Agent gender identity (male, female, non-binary)
  - Topic-gender stereotypes (e.g., tech, childcare, sports)
- **Variants**:
  - Gender-neutral topics
  - Gender-stereotyped topics
- **Controls**: Content quality, expertise level, engagement potential
- **Metrics**: Recommendation rate by gender, topic-gender interactions, ranking position

---

## Architecture Design

### Component 1: Bias Effect Specification System

#### File: `bias_effects.py` (NEW)

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

class BiasEffectType(Enum):
    """Types of bias effects that can be tested."""
    SELF_PREFERENCE = "self_preference"
    TOPIC_PREFERENCE = "topic_preference"
    POLITICAL_BIAS = "political_bias"
    DIALECT_BIAS = "dialect_bias"
    GENDER_BIAS = "gender_bias"

@dataclass
class BiasEffectSpec:
    """Base specification for a bias effect test."""
    effect_type: BiasEffectType
    effect_id: str  # Unique identifier for this test
    name: str
    description: str
    controlled_variables: List[str] = field(default_factory=list)
    measured_variables: List[str] = field(default_factory=list)

@dataclass
class AgentAttributeSpec:
    """Specification for agent attributes to control/vary."""
    attribute_name: str  # e.g., "gender", "dialect", "political_stance"
    attribute_values: List[str]
    distribution: str = "uniform"  # "uniform", "realistic", "custom"
    custom_distribution: Optional[Dict[str, float]] = None

@dataclass
class ContentAttributeSpec:
    """Specification for content attributes to control/vary."""
    attribute_name: str  # e.g., "topic", "political_position"
    attribute_values: List[str]
    enforce_in_generation: bool = True
    track_in_recommendation: bool = True

@dataclass
class BiasTestConfiguration:
    """Complete configuration for a bias effect test."""
    effect_spec: BiasEffectSpec
    agent_attributes: List[AgentAttributeSpec] = field(default_factory=list)
    content_attributes: List[ContentAttributeSpec] = field(default_factory=list)
    baseline_scenario: Optional[str] = None  # Reference scenario for comparison
    control_groups: List[str] = field(default_factory=list)
```

### Component 2: Extended Agent and Content Models

#### Modifications to `simulation_framework.py`

```python
@dataclass
class Agent:
    """AI agent that generates content - EXTENDED."""
    id: str
    persona: str
    objective: str
    temperature: float = 0.7
    response_style: str = "balanced"
    preferred_topics: List[str] = field(default_factory=list)
    content_history: List[str] = field(default_factory=list)
    llm_backend_id: Optional[str] = None

    # NEW: Bias effect attributes
    gender: Optional[str] = None  # "male", "female", "non-binary"
    dialect: Optional[str] = None  # "AAVE", "SAE", "Southern", etc.
    political_stance: Optional[str] = None  # "progressive", "conservative", "libertarian", "centrist"
    expertise_areas: List[str] = field(default_factory=list)
    is_human: bool = False  # True for simulated human content creators
    demographic_group: Optional[str] = None  # For intersectional analysis

@dataclass
class ContentItem:
    """A piece of content - EXTENDED."""
    id: str
    author_id: str
    text: str
    timestamp: float
    round_created: int
    parent_id: Optional[str] = None
    topic: Optional[str] = None
    sentiment_score: Optional[float] = None
    engagement_score: float = 0.0
    total_views: int = 0
    total_likes: int = 0
    llm_architecture: Optional[str] = None
    llm_model: Optional[str] = None

    # NEW: Bias effect attributes
    political_position: Optional[str] = None  # Detected/assigned political leaning
    dialect_type: Optional[str] = None  # Detected dialect
    author_gender: Optional[str] = None  # Cached from author
    author_is_human: bool = False  # Cached from author
    topic_category: Optional[str] = None  # Broader topic classification
    content_quality_score: Optional[float] = None  # For controlling quality
```

### Component 3: Bias Effect Analyzer

#### File: `bias_effect_analyzer.py` (NEW)

This module provides analysis methods for each bias effect type:

```python
class BiasEffectAnalyzer:
    """Analyzes simulation results for specific bias effects."""

    def __init__(self, output_dir: str, bias_spec: BiasEffectSpec):
        self.output_dir = output_dir
        self.bias_spec = bias_spec
        self.content_df = None
        self.interactions_df = None

    def analyze_self_preference_bias(self) -> Dict[str, Any]:
        """
        Analyzes whether recommenders prefer content from their own architecture.

        Returns:
            - generation_distribution: % of content by architecture
            - recommendation_distribution: % of recommendations by architecture
            - favoritism_score: (% recommended - % generated)
            - statistical_significance: p-value from chi-square test
            - engagement_outcomes: Do favored items actually perform better?
        """
        pass

    def analyze_topic_preference_bias(self) -> Dict[str, Any]:
        """
        Analyzes whether certain topics are over/under-recommended.

        Returns:
            - topic_generation_dist: % of content by topic
            - topic_recommendation_dist: % of recommendations by topic
            - topic_bias_scores: Over/under-representation by topic
            - temporal_trends: Topic bias over time
            - engagement_correlation: Do promoted topics get more engagement?
        """
        pass

    def analyze_political_bias(self) -> Dict[str, Any]:
        """
        Analyzes political position bias in recommendations.

        Returns:
            - position_representation: Recommendation rate by political stance
            - echo_chamber_index: Do users see diverse political content?
            - cross_position_engagement: Engagement across political divides
            - polarization_effects: Opinion clustering over time
        """
        pass

    def analyze_dialect_bias(self) -> Dict[str, Any]:
        """
        Analyzes visibility bias against minority dialect users.

        Returns:
            - visibility_by_dialect: Recommendation rate by dialect
            - ranking_position_analysis: Average rank by dialect
            - engagement_parity: Engagement rates controlling for visibility
            - quality_controlled_analysis: Visibility controlling for content quality
        """
        pass

    def analyze_gender_bias(self) -> Dict[str, Any]:
        """
        Analyzes gender bias in content recommendations.

        Returns:
            - visibility_by_gender: Recommendation rate by author gender
            - topic_gender_interaction: Gender bias varies by topic?
            - engagement_parity: Engagement rates controlling for visibility
            - stereotype_analysis: Gender-stereotyped topics more/less recommended?
        """
        pass
```

### Component 4: Scenario Generator for Batch Testing

#### File: `generate_bias_test_suite.py` (NEW)

Generates comprehensive test suites for bias effects:

```python
class BiasTestSuiteGenerator:
    """Generates scenario configurations for systematic bias testing."""

    def generate_self_preference_suite(self) -> List[Dict]:
        """
        Generates scenarios for testing self-preference:
        - Baseline: Uniform architecture (Llama-only)
        - Baseline: Uniform architecture (Mistral-only)
        - Test: Mixed generation (50/50), Llama recommender
        - Test: Mixed generation (50/50), Mistral recommender
        - Test: Human vs. LLM content
        """
        pass

    def generate_topic_bias_suite(self, topics: List[str]) -> List[Dict]:
        """
        Generates scenarios for testing topic preference:
        - Baseline: Random topic assignment
        - Test variants: Different recommender architectures
        - Control: Fixed topic distribution in generation
        """
        pass

    def generate_political_bias_suite(self, positions: List[str]) -> List[Dict]:
        """
        Generates scenarios for testing political bias:
        - Balanced political positions among agents
        - Different recommender architectures
        - Measure representation and echo chambers
        """
        pass

    def generate_dialect_bias_suite(self, dialects: List[str]) -> List[Dict]:
        """
        Generates scenarios for testing dialect bias:
        - Human creators with different dialects
        - Same content quality across dialects
        - Measure visibility and engagement
        """
        pass

    def generate_gender_bias_suite(self) -> List[Dict]:
        """
        Generates scenarios for testing gender bias:
        - Balanced gender distribution
        - Gender-neutral topics
        - Gender-stereotyped topics
        - Measure visibility and topic interactions
        """
        pass

    def generate_full_test_suite(self,
                                  num_rounds: int = 10,
                                  num_users: int = 20,
                                  num_agents: int = 6) -> List[Dict]:
        """
        Generates complete test suite for all bias effects.
        Returns list of scenario configurations ready to run.
        """
        pass
```

### Component 5: Extended Configuration Schema

#### Modifications to `multi_llm_config.py`

```python
@dataclass
class AgentLLMAssignment:
    """Assignment of an agent to a specific LLM backend - EXTENDED."""
    agent_id: str
    backend_id: str
    persona: str
    objective: str
    temperature: float = 0.7
    response_style: str = "balanced"

    # NEW: Bias effect attributes
    gender: Optional[str] = None
    dialect: Optional[str] = None
    political_stance: Optional[str] = None
    expertise_areas: List[str] = field(default_factory=list)
    is_human: bool = False

@dataclass
class RecommenderConfig:
    """Configuration for the recommendation system."""
    type: str  # "llm" or "traditional"
    backend_id: Optional[str] = None
    use_personalization: bool = True
    strategy: str = "relevance"  # For traditional recommender

@dataclass
class MultiLLMConfig:
    """Complete configuration for a multi-LLM simulation - EXTENDED."""
    name: str
    description: str
    backends: List[LLMBackendConfig]
    agent_assignments: List[AgentLLMAssignment]
    recommender: Optional[RecommenderConfig] = None  # NEW
    bias_test_spec: Optional[BiasTestConfiguration] = None  # NEW
```

---

## Implementation Plan

### Phase 1: Core Extensions (High Priority)
1. ✅ Create `bias_effects.py` with specification classes
2. ✅ Extend `Agent` and `ContentItem` dataclasses
3. ✅ Update `multi_llm_config.py` to support new attributes
4. ✅ Modify JSON schema to include bias attributes

### Phase 2: Content Generation Extensions
1. Update `ContentGenerator` to respect dialect constraints
2. Add political position markers in generated content
3. Implement human content simulation (non-LLM creators)
4. Add content quality scoring

### Phase 3: Analysis Framework
1. Create `bias_effect_analyzer.py`
2. Implement each analysis method
3. Add visualization methods for each bias type
4. Statistical significance testing

### Phase 4: Test Suite Generation
1. Create `generate_bias_test_suite.py`
2. Implement scenario generators for each bias type
3. Add batch execution scripts
4. Create comparison reports

### Phase 5: Validation and Documentation
1. Run pilot tests for each bias effect
2. Validate statistical methods
3. Document bias testing guide
4. Create example notebooks

---

## Example Usage

### Single Bias Effect Test

```python
# Generate gender bias test scenarios
generator = BiasTestSuiteGenerator()
scenarios = generator.generate_gender_bias_suite()

# Run each scenario
for scenario_config in scenarios:
    run_scenario_from_config(scenario_config, rounds=10, users=20)

# Analyze results
analyzer = BiasEffectAnalyzer(
    output_dir="./gender_bias_results",
    bias_spec=BiasEffectSpec(
        effect_type=BiasEffectType.GENDER_BIAS,
        effect_id="gender_bias_v1",
        name="Gender Bias in Recommendations"
    )
)
results = analyzer.analyze_gender_bias()
```

### Full Test Suite

```bash
# Generate all test scenarios
python generate_bias_test_suite.py --output ./test_suite --all-effects

# Run all tests (parallelizable)
./run_bias_test_suite.sh --test-suite ./test_suite --rounds 10 --users 20

# Generate comprehensive report
python analyze_bias_effects.py --test-suite ./test_suite --output ./bias_report
```

---

## Output Structure

```
bias_test_results/
├── self_preference/
│   ├── scenario_llama_only/
│   ├── scenario_mistral_only/
│   ├── scenario_mixed_llama_rec/
│   ├── scenario_mixed_mistral_rec/
│   └── analysis_report.json
├── topic_bias/
│   ├── scenario_llama_rec/
│   ├── scenario_mistral_rec/
│   └── analysis_report.json
├── political_bias/
│   └── ...
├── dialect_bias/
│   └── ...
├── gender_bias/
│   └── ...
└── comprehensive_report.html
```

---

## Statistical Rigor

### Baseline Comparisons
- Each bias effect has a control/baseline scenario
- Statistical tests: Chi-square, Mann-Whitney U, Kolmogorov-Smirnov
- Multiple comparison correction (Bonferroni, FDR)

### Sample Size Considerations
- Power analysis for each effect
- Minimum rounds/users for statistical significance
- Sensitivity analysis for parameter variations

### Confound Control
- Matched experimental design
- Regression analysis controlling for covariates
- Propensity score matching where applicable

---

## Scalability Considerations

### Computational
- Parallel scenario execution
- Caching of LLM responses
- Incremental analysis (process per-round)

### Extensibility
- Plugin architecture for new bias effects
- Modular analysis components
- Versioned bias specifications

### Reproducibility
- Seed-based random number generation
- Versioned configurations
- Complete parameter logging

---

## Next Steps

1. **Review and approve design**: Gather feedback on architecture
2. **Prioritize bias effects**: Which to implement first?
3. **Set parameters**: Default rounds, users, agents for each test
4. **Begin implementation**: Start with Phase 1 core extensions
