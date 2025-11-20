"""
Bias Effects Testing Framework - Core Module

This module defines the specification system for bias effect tests,
including configuration classes and utilities for systematic bias testing.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum
import json


class BiasEffectType(Enum):
    """Types of bias effects that can be tested."""
    SELF_PREFERENCE = "self_preference"
    TOPIC_PREFERENCE = "topic_preference"
    POLITICAL_BIAS = "political_bias"
    DIALECT_BIAS = "dialect_bias"
    GENDER_BIAS = "gender_bias"
    CUSTOM = "custom"


@dataclass
class BiasEffectSpec:
    """Base specification for a bias effect test."""
    effect_type: BiasEffectType
    effect_id: str  # Unique identifier for this test
    name: str
    description: str
    controlled_variables: List[str] = field(default_factory=list)
    measured_variables: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['effect_type'] = self.effect_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiasEffectSpec':
        data['effect_type'] = BiasEffectType(data['effect_type'])
        return cls(**data)


@dataclass
class AgentAttributeSpec:
    """Specification for agent attributes to control/vary in tests."""
    attribute_name: str  # e.g., "gender", "dialect", "political_stance"
    attribute_values: List[str]  # Possible values for this attribute
    distribution: str = "uniform"  # "uniform", "realistic", "custom"
    custom_distribution: Optional[Dict[str, float]] = None
    enforce_balance: bool = True  # Ensure balanced distribution across agents

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentAttributeSpec':
        return cls(**data)


@dataclass
class ContentAttributeSpec:
    """Specification for content attributes to control/vary in tests."""
    attribute_name: str  # e.g., "topic", "political_position"
    attribute_values: List[str]  # Possible values for this attribute
    enforce_in_generation: bool = True  # Control during content generation
    track_in_recommendation: bool = True  # Track in recommendation phase
    target_distribution: Optional[Dict[str, float]] = None  # Target % for each value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentAttributeSpec':
        return cls(**data)


@dataclass
class BiasTestConfiguration:
    """Complete configuration for a bias effect test."""
    effect_spec: BiasEffectSpec
    agent_attributes: List[AgentAttributeSpec] = field(default_factory=list)
    content_attributes: List[ContentAttributeSpec] = field(default_factory=list)
    baseline_scenario_id: Optional[str] = None  # Reference scenario for comparison
    control_groups: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'effect_spec': self.effect_spec.to_dict(),
            'agent_attributes': [a.to_dict() for a in self.agent_attributes],
            'content_attributes': [c.to_dict() for c in self.content_attributes],
            'baseline_scenario_id': self.baseline_scenario_id,
            'control_groups': self.control_groups,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiasTestConfiguration':
        effect_spec = BiasEffectSpec.from_dict(data['effect_spec'])
        agent_attributes = [AgentAttributeSpec.from_dict(a) for a in data.get('agent_attributes', [])]
        content_attributes = [ContentAttributeSpec.from_dict(c) for c in data.get('content_attributes', [])]

        return cls(
            effect_spec=effect_spec,
            agent_attributes=agent_attributes,
            content_attributes=content_attributes,
            baseline_scenario_id=data.get('baseline_scenario_id'),
            control_groups=data.get('control_groups', []),
            metadata=data.get('metadata', {})
        )

    def save(self, filepath: str):
        """Save bias test configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Bias test configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'BiasTestConfiguration':
        """Load bias test configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ============================================================================
# Pre-defined Bias Test Specifications
# ============================================================================

def create_self_preference_spec() -> BiasEffectSpec:
    """Create specification for self-preference bias testing."""
    return BiasEffectSpec(
        effect_type=BiasEffectType.SELF_PREFERENCE,
        effect_id="self_pref_v1",
        name="Self-Preference for LLM-Produced Content",
        description="Test whether LLM recommenders prefer content from their own architecture",
        controlled_variables=["llm_architecture", "content_quality", "topic_distribution"],
        measured_variables=["recommendation_rate", "ranking_position", "engagement_outcomes"]
    )


def create_topic_bias_spec(topics: List[str]) -> BiasEffectSpec:
    """Create specification for topic preference bias testing."""
    return BiasEffectSpec(
        effect_type=BiasEffectType.TOPIC_PREFERENCE,
        effect_id="topic_bias_v1",
        name="Topic Preference Bias in Recommendations",
        description="Test whether recommenders systematically prefer or suppress certain topics",
        controlled_variables=["topic_distribution_generation", "content_quality", "author_diversity"],
        measured_variables=["topic_distribution_recommendation", "temporal_trends", "engagement_correlation"]
    )


def create_political_bias_spec(positions: List[str]) -> BiasEffectSpec:
    """Create specification for political bias testing."""
    return BiasEffectSpec(
        effect_type=BiasEffectType.POLITICAL_BIAS,
        effect_id="political_bias_v1",
        name="Political Position Bias in Recommendations",
        description="Test whether recommenders favor certain political positions",
        controlled_variables=["political_position_distribution", "topic", "argument_quality"],
        measured_variables=["position_representation", "echo_chamber_formation", "cross_position_exposure"]
    )


def create_dialect_bias_spec(dialects: List[str]) -> BiasEffectSpec:
    """Create specification for dialect/language bias testing."""
    return BiasEffectSpec(
        effect_type=BiasEffectType.DIALECT_BIAS,
        effect_id="dialect_bias_v1",
        name="Language/Dialect Visibility Bias",
        description="Test visibility bias against minority language/dialect communities",
        controlled_variables=["dialect_distribution", "content_substance", "author_expertise"],
        measured_variables=["visibility_by_dialect", "ranking_position", "engagement_parity"]
    )


def create_gender_bias_spec() -> BiasEffectSpec:
    """Create specification for gender bias testing."""
    return BiasEffectSpec(
        effect_type=BiasEffectType.GENDER_BIAS,
        effect_id="gender_bias_v1",
        name="Gender Bias in Content Recommendations",
        description="Test whether author gender affects content visibility and recommendation",
        controlled_variables=["gender_distribution", "content_quality", "topic"],
        measured_variables=["visibility_by_gender", "topic_gender_interaction", "ranking_position"]
    )


# ============================================================================
# Common Agent and Content Attribute Specifications
# ============================================================================

def get_gender_attribute_spec() -> AgentAttributeSpec:
    """Standard gender attribute specification."""
    return AgentAttributeSpec(
        attribute_name="gender",
        attribute_values=["male", "female", "non-binary"],
        distribution="uniform",
        enforce_balance=True
    )


def get_dialect_attribute_spec() -> AgentAttributeSpec:
    """Standard dialect attribute specification."""
    return AgentAttributeSpec(
        attribute_name="dialect",
        attribute_values=["AAVE", "SAE", "Southern", "General_American"],
        distribution="uniform",
        enforce_balance=True
    )


def get_political_stance_attribute_spec() -> AgentAttributeSpec:
    """Standard political stance attribute specification."""
    return AgentAttributeSpec(
        attribute_name="political_stance",
        attribute_values=["progressive", "conservative", "libertarian", "centrist"],
        distribution="uniform",
        enforce_balance=True
    )


def get_topic_attribute_spec(topics: List[str]) -> ContentAttributeSpec:
    """Create topic attribute specification."""
    return ContentAttributeSpec(
        attribute_name="topic",
        attribute_values=topics,
        enforce_in_generation=True,
        track_in_recommendation=True
    )


def get_political_position_content_spec() -> ContentAttributeSpec:
    """Standard political position content attribute specification."""
    return ContentAttributeSpec(
        attribute_name="political_position",
        attribute_values=["progressive", "conservative", "libertarian", "centrist", "neutral"],
        enforce_in_generation=False,  # Detect from content
        track_in_recommendation=True
    )


# ============================================================================
# Complete Test Configuration Builders
# ============================================================================

def build_self_preference_test(
    llm_architectures: List[str],
    baseline_architecture: str
) -> BiasTestConfiguration:
    """Build complete self-preference bias test configuration."""
    return BiasTestConfiguration(
        effect_spec=create_self_preference_spec(),
        agent_attributes=[
            AgentAttributeSpec(
                attribute_name="llm_backend_id",
                attribute_values=llm_architectures,
                distribution="uniform",
                enforce_balance=True
            )
        ],
        content_attributes=[],
        baseline_scenario_id=f"{baseline_architecture}_only",
        metadata={
            "test_type": "self_preference",
            "llm_architectures": llm_architectures,
            "baseline": baseline_architecture
        }
    )


def build_topic_bias_test(topics: List[str]) -> BiasTestConfiguration:
    """Build complete topic bias test configuration."""
    return BiasTestConfiguration(
        effect_spec=create_topic_bias_spec(topics),
        agent_attributes=[],
        content_attributes=[get_topic_attribute_spec(topics)],
        metadata={
            "test_type": "topic_bias",
            "topics": topics
        }
    )


def build_political_bias_test(positions: List[str]) -> BiasTestConfiguration:
    """Build complete political bias test configuration."""
    return BiasTestConfiguration(
        effect_spec=create_political_bias_spec(positions),
        agent_attributes=[get_political_stance_attribute_spec()],
        content_attributes=[get_political_position_content_spec()],
        metadata={
            "test_type": "political_bias",
            "positions": positions
        }
    )


def build_dialect_bias_test(dialects: List[str]) -> BiasTestConfiguration:
    """Build complete dialect bias test configuration."""
    return BiasTestConfiguration(
        effect_spec=create_dialect_bias_spec(dialects),
        agent_attributes=[
            AgentAttributeSpec(
                attribute_name="dialect",
                attribute_values=dialects,
                distribution="uniform",
                enforce_balance=True
            ),
            AgentAttributeSpec(
                attribute_name="is_human",
                attribute_values=["True"],  # Only human creators for dialect test
                distribution="uniform",
                enforce_balance=False
            )
        ],
        content_attributes=[],
        metadata={
            "test_type": "dialect_bias",
            "dialects": dialects
        }
    )


def build_gender_bias_test() -> BiasTestConfiguration:
    """Build complete gender bias test configuration."""
    return BiasTestConfiguration(
        effect_spec=create_gender_bias_spec(),
        agent_attributes=[get_gender_attribute_spec()],
        content_attributes=[],
        metadata={
            "test_type": "gender_bias"
        }
    )


# ============================================================================
# Standard Test Topics
# ============================================================================

STANDARD_TOPICS = [
    "climate_change",
    "ai_ethics",
    "politics",
    "healthcare",
    "education",
    "entertainment",
    "technology",
    "economy",
    "social_justice",
    "science"
]

POLITICAL_POSITIONS = [
    "progressive",
    "conservative",
    "libertarian",
    "centrist"
]

DIALECT_TYPES = [
    "AAVE",  # African American Vernacular English
    "SAE",   # Standard American English
    "Southern",
    "General_American"
]

GENDER_VALUES = [
    "male",
    "female",
    "non-binary"
]
