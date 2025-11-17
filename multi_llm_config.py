"""
Multi-LLM Configuration System
Provides utilities for configuring multiple LLM backends in simulations.
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class LLMBackendConfig:
    """Configuration for a single LLM backend."""
    backend_id: str
    type: str  # "llama", "gpt", "small", "mock"
    model: str
    use_mock: bool = False
    device: str = "auto"
    description: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class AgentLLMAssignment:
    """Assignment of an agent to a specific LLM backend."""
    agent_id: str
    backend_id: str
    persona: str
    objective: str
    temperature: float = 0.7
    response_style: str = "balanced"

    def to_dict(self):
        return asdict(self)


@dataclass
class MultiLLMConfig:
    """Complete configuration for a multi-LLM simulation."""
    name: str
    description: str
    backends: List[LLMBackendConfig]
    agent_assignments: List[AgentLLMAssignment]

    def to_dict(self):
        return {
            'name': self.name,
            'description': self.description,
            'backends': [b.to_dict() for b in self.backends],
            'agent_assignments': [a.to_dict() for a in self.agent_assignments]
        }

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'MultiLLMConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        backends = [LLMBackendConfig(**b) for b in data['backends']]
        agent_assignments = [AgentLLMAssignment(**a) for a in data['agent_assignments']]

        return cls(
            name=data['name'],
            description=data['description'],
            backends=backends,
            agent_assignments=agent_assignments
        )


def create_example_config() -> MultiLLMConfig:
    """Create an example configuration with multiple LLM architectures."""

    # Define multiple LLM backends
    backends = [
        LLMBackendConfig(
            backend_id="mock_general",
            type="mock",
            model="mock",
            use_mock=True,
            description="Mock backend for general agents"
        ),
        LLMBackendConfig(
            backend_id="mock_creative",
            type="mock",
            model="mock-creative",
            use_mock=True,
            description="Mock backend for creative agents"
        ),
        LLMBackendConfig(
            backend_id="mock_analytical",
            type="mock",
            model="mock-analytical",
            use_mock=True,
            description="Mock backend for analytical agents"
        )
    ]

    # Define agent-to-backend assignments
    agent_assignments = [
        AgentLLMAssignment(
            agent_id="agent_0",
            backend_id="mock_general",
            persona="Progressive Activist",
            objective="Advocate for social and environmental causes",
            temperature=0.8,
            response_style="passionate"
        ),
        AgentLLMAssignment(
            agent_id="agent_1",
            backend_id="mock_creative",
            persona="Tech Entrepreneur",
            objective="Discuss innovation and business opportunities",
            temperature=0.7,
            response_style="optimistic"
        ),
        AgentLLMAssignment(
            agent_id="agent_2",
            backend_id="mock_analytical",
            persona="Academic Researcher",
            objective="Provide evidence-based analytical insights",
            temperature=0.5,
            response_style="analytical"
        ),
        AgentLLMAssignment(
            agent_id="agent_3",
            backend_id="mock_general",
            persona="Skeptical Journalist",
            objective="Question narratives and demand accountability",
            temperature=0.6,
            response_style="critical"
        ),
        AgentLLMAssignment(
            agent_id="agent_4",
            backend_id="mock_creative",
            persona="Community Organizer",
            objective="Build consensus and mobilize action",
            temperature=0.7,
            response_style="collaborative"
        ),
        AgentLLMAssignment(
            agent_id="agent_5",
            backend_id="mock_general",
            persona="Contrarian Thinker",
            objective="Challenge mainstream narratives",
            temperature=0.8,
            response_style="provocative"
        )
    ]

    return MultiLLMConfig(
        name="Multi-LLM Social Media Simulation",
        description="Simulation with different LLM architectures generating content from diverse perspectives",
        backends=backends,
        agent_assignments=agent_assignments
    )


def create_real_llm_config() -> MultiLLMConfig:
    """Create a configuration using real LLM models (requires GPU)."""

    backends = [
        LLMBackendConfig(
            backend_id="llama",
            type="llama",
            model="meta-llama/Llama-3.1-8B-Instruct",
            use_mock=False,
            description="Llama 3.1 8B for nuanced, context-aware generation"
        ),
        LLMBackendConfig(
            backend_id="gpt2",
            type="gpt",
            model="gpt2",
            use_mock=False,
            description="GPT-2 for baseline comparison"
        ),
        LLMBackendConfig(
            backend_id="small",
            type="small",
            model="microsoft/phi-2",
            use_mock=False,
            description="Phi-2 small model for faster generation"
        )
    ]

    agent_assignments = [
        AgentLLMAssignment(
            agent_id="agent_0",
            backend_id="llama",
            persona="Progressive Activist",
            objective="Advocate for social and environmental causes",
            temperature=0.8
        ),
        AgentLLMAssignment(
            agent_id="agent_1",
            backend_id="llama",
            persona="Tech Entrepreneur",
            objective="Discuss innovation and opportunities",
            temperature=0.7
        ),
        AgentLLMAssignment(
            agent_id="agent_2",
            backend_id="gpt2",
            persona="Academic Researcher",
            objective="Provide evidence-based insights",
            temperature=0.5
        ),
        AgentLLMAssignment(
            agent_id="agent_3",
            backend_id="gpt2",
            persona="Skeptical Journalist",
            objective="Question narratives critically",
            temperature=0.6
        ),
        AgentLLMAssignment(
            agent_id="agent_4",
            backend_id="small",
            persona="Community Organizer",
            objective="Build consensus and action",
            temperature=0.7
        ),
        AgentLLMAssignment(
            agent_id="agent_5",
            backend_id="small",
            persona="Contrarian Thinker",
            objective="Challenge mainstream views",
            temperature=0.8
        )
    ]

    return MultiLLMConfig(
        name="Multi-Architecture LLM Comparison",
        description="Compare content generation across Llama, GPT-2, and Phi-2 architectures",
        backends=backends,
        agent_assignments=agent_assignments
    )


def validate_config(config: MultiLLMConfig) -> bool:
    """
    Validate a configuration.

    Checks:
    - All backend IDs are unique
    - All agent assignments reference valid backends
    - No duplicate agent IDs
    """
    # Check unique backend IDs
    backend_ids = [b.backend_id for b in config.backends]
    if len(backend_ids) != len(set(backend_ids)):
        print("Error: Duplicate backend IDs found")
        return False

    # Check unique agent IDs
    agent_ids = [a.agent_id for a in config.agent_assignments]
    if len(agent_ids) != len(set(agent_ids)):
        print("Error: Duplicate agent IDs found")
        return False

    # Check all agent assignments reference valid backends
    backend_id_set = set(backend_ids)
    for assignment in config.agent_assignments:
        if assignment.backend_id not in backend_id_set:
            print(f"Error: Agent {assignment.agent_id} references unknown backend {assignment.backend_id}")
            return False

    print("Configuration is valid!")
    return True


if __name__ == "__main__":
    # Example: Create and save configurations

    # 1. Mock LLM configuration (for testing)
    mock_config = create_example_config()
    mock_config.save("config_mock_multi_llm.json")

    # 2. Real LLM configuration (requires GPU)
    real_config = create_real_llm_config()
    real_config.save("config_real_multi_llm.json")

    # 3. Validate configurations
    print("\nValidating mock configuration:")
    validate_config(mock_config)

    print("\nValidating real LLM configuration:")
    validate_config(real_config)
