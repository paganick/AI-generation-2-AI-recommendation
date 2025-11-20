"""
Demo: Bias Testing Framework Usage

This script demonstrates how to use the bias testing framework programmatically.
"""

import json
from pathlib import Path
from bias_effects import (
    build_self_preference_test,
    build_topic_bias_test,
    build_political_bias_test,
    build_dialect_bias_test,
    build_gender_bias_test,
    STANDARD_TOPICS,
    POLITICAL_POSITIONS,
    DIALECT_TYPES,
    GENDER_VALUES
)
from generate_bias_test_suite import BiasTestSuiteGenerator


def demo_create_single_bias_test():
    """Demonstrate creating a single bias test configuration."""
    print("=" * 60)
    print("DEMO 1: Creating Single Bias Test Configuration")
    print("=" * 60)

    # Create a gender bias test
    gender_test = build_gender_bias_test()

    print("\nGender Bias Test Configuration:")
    print(f"  Effect Type: {gender_test.effect_spec.effect_type.value}")
    print(f"  Effect ID: {gender_test.effect_spec.effect_id}")
    print(f"  Name: {gender_test.effect_spec.name}")
    print(f"  Description: {gender_test.effect_spec.description}")

    print("\n  Agent Attributes:")
    for attr in gender_test.agent_attributes:
        print(f"    - {attr.attribute_name}: {attr.attribute_values}")

    print("\n  Controlled Variables:")
    for var in gender_test.effect_spec.controlled_variables:
        print(f"    - {var}")

    print("\n  Measured Variables:")
    for var in gender_test.effect_spec.measured_variables:
        print(f"    - {var}")

    # Save to file
    output_dir = Path("./demo_output")
    output_dir.mkdir(exist_ok=True)
    gender_test.save(str(output_dir / "gender_bias_test.json"))

    print(f"\n✓ Saved to: {output_dir / 'gender_bias_test.json'}")


def demo_generate_test_suite():
    """Demonstrate generating a complete test suite."""
    print("\n" + "=" * 60)
    print("DEMO 2: Generating Complete Test Suite")
    print("=" * 60)

    # Create generator
    generator = BiasTestSuiteGenerator(output_dir="./demo_test_suite")

    # Generate test suite for specific effects
    effects = ["self_preference", "gender_bias"]

    print(f"\nGenerating test suite for: {', '.join(effects)}")

    suite = generator.generate_full_test_suite(
        effects=effects,
        num_rounds=5,  # Small for demo
        num_users=10,
        num_agents=4
    )

    print("\nGenerated Scenarios:")
    for effect_type, scenarios in suite.items():
        print(f"\n  {effect_type}:")
        for scenario in scenarios:
            print(f"    - {scenario['name']}")
            print(f"      ID: {scenario['scenario_id']}")
            print(f"      Backends: {[b['backend_id'] for b in scenario['backends']]}")
            print(f"      Agents: {len(scenario['agent_assignments'])}")

    # Save test suite
    generator.save_test_suite(suite)

    print(f"\n✓ Test suite saved to: ./demo_test_suite")


def demo_inspect_scenario():
    """Demonstrate inspecting a generated scenario configuration."""
    print("\n" + "=" * 60)
    print("DEMO 3: Inspecting Scenario Configuration")
    print("=" * 60)

    # Generate a simple scenario
    generator = BiasTestSuiteGenerator()

    scenarios = generator.generate_gender_bias_suite(
        num_agents=6,
        num_rounds=10,
        num_users=20
    )

    # Inspect first scenario
    scenario = scenarios[0]

    print(f"\nScenario: {scenario['name']}")
    print(f"Description: {scenario['description']}")
    print(f"Scenario ID: {scenario['scenario_id']}")

    print("\nSimulation Parameters:")
    print(f"  Rounds: {scenario['simulation_params']['num_rounds']}")
    print(f"  Users: {scenario['simulation_params']['num_users']}")

    print("\nBackends:")
    for backend in scenario['backends']:
        print(f"  - {backend['backend_id']}: {backend['model']}")

    print("\nAgents:")
    for agent in scenario['agent_assignments']:
        print(f"  - {agent['agent_id']}:")
        print(f"      Backend: {agent['backend_id']}")
        print(f"      Persona: {agent['persona']}")
        if 'gender' in agent:
            print(f"      Gender: {agent['gender']}")
        if 'preferred_topics' in agent:
            print(f"      Topics: {agent['preferred_topics']}")

    print("\nRecommender:")
    print(f"  Type: {scenario['recommender']['type']}")
    print(f"  Backend: {scenario['recommender']['backend_id']}")

    if 'bias_test' in scenario:
        print("\nBias Test Specification:")
        for key, value in scenario['bias_test'].items():
            print(f"  {key}: {value}")


def demo_analyze_bias_concepts():
    """Demonstrate the different bias effect concepts."""
    print("\n" + "=" * 60)
    print("DEMO 4: Understanding Bias Effect Concepts")
    print("=" * 60)

    print("\n1. SELF-PREFERENCE BIAS")
    print("   Question: Do LLM recommenders prefer their own architecture's content?")
    print("   Example: Llama recommender showing more Llama-generated content")
    print("   Key Metric: favoritism_score = (% recommended - % generated)")

    print("\n2. TOPIC PREFERENCE BIAS")
    print("   Question: Are certain topics over/under-recommended?")
    print("   Example: AI ethics content recommended 15% more than generated")
    print(f"   Topics tested: {', '.join(STANDARD_TOPICS[:5])}...")

    print("\n3. POLITICAL BIAS")
    print("   Question: Do recommenders favor certain political positions?")
    print("   Example: Progressive content recommended 10% more than conservative")
    print(f"   Positions tested: {', '.join(POLITICAL_POSITIONS)}")

    print("\n4. DIALECT/LANGUAGE BIAS")
    print("   Question: Are minority dialect speakers less visible?")
    print("   Example: AAVE content recommended 20% less than SAE content")
    print(f"   Dialects tested: {', '.join(DIALECT_TYPES)}")

    print("\n5. GENDER BIAS")
    print("   Question: Does author gender affect content visibility?")
    print("   Example: Male authors' tech content recommended 15% more")
    print(f"   Genders tested: {', '.join(GENDER_VALUES)}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("BIAS TESTING FRAMEWORK - DEMO")
    print("=" * 60)

    # Run demos
    demo_analyze_bias_concepts()
    demo_create_single_bias_test()
    demo_generate_test_suite()
    demo_inspect_scenario()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Explore generated files in ./demo_output and ./demo_test_suite")
    print("2. Run a test scenario:")
    print("   python run_multi_llm_simulation.py \\")
    print("       --config ./demo_test_suite/<effect>/<scenario>.json \\")
    print("       --rounds 10 --users 20")
    print("3. Analyze results:")
    print("   python bias_effect_analyzer.py \\")
    print("       --output-dir <simulation_output> \\")
    print("       --bias-types <effect_type>")
    print("4. Review the full guide: BIAS_TESTING_GUIDE.md")
    print()


if __name__ == "__main__":
    main()
