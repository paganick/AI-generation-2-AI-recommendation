"""
Enhanced Scenario Runner with LLM-based Recommendation Support

Runs the two specified scenarios:
1. Scenario 1: All Llama-3.1-8B (generation + recommendation)
2. Scenario 2: Mixed generation (Llama + Mistral), Llama recommendation
"""

import argparse
import numpy as np
from typing import List
from pathlib import Path

from llm_backend import create_multi_llm_setup, MultiLLMManager, get_llm_backend
from llm_recommender import LLMRecommender
from multi_llm_config import MultiLLMConfig
from simulation_framework import (
    Agent, SimulatedUser, EnhancedDataTracker, ContentGenerator,
    RecommenderSystem, UserContentInteractionSimulator,
    add_sentiment_scores, save_results
)


def create_agents_from_config(config: MultiLLMConfig) -> List[Agent]:
    """Create Agent objects from configuration."""
    agents = []
    for assignment in config.agent_assignments:
        agent = Agent(
            id=assignment.agent_id,
            persona=assignment.persona,
            objective=assignment.objective,
            temperature=assignment.temperature,
            response_style=assignment.response_style,
            llm_backend_id=assignment.backend_id
        )
        agents.append(agent)
    return agents


def create_backend_manager_from_config(config: MultiLLMConfig) -> MultiLLMManager:
    """Create MultiLLMManager from configuration."""
    backend_config_dict = {}
    for backend in config.backends:
        backend_config_dict[backend.backend_id] = {
            'type': backend.type,
            'model': backend.model,
            'use_mock': backend.use_mock
        }

    manager = create_multi_llm_setup(backend_config_dict)

    for assignment in config.agent_assignments:
        manager.assign_backend_to_agent(assignment.agent_id, assignment.backend_id)

    return manager


def create_simulated_users(n_users: int = 10) -> List[SimulatedUser]:
    """Create simulated users with diverse interests."""
    interests_pool = [
        ["climate change", "environment", "sustainability"],
        ["technology", "AI", "innovation"],
        ["healthcare", "medicine", "public health"],
        ["education", "learning", "universities"],
        ["politics", "policy", "governance"],
        ["economics", "business", "markets"],
        ["social justice", "equality", "rights"],
        ["science", "research", "discovery"]
    ]

    users = []
    for i in range(n_users):
        interests = interests_pool[i % len(interests_pool)]
        opinion = np.random.uniform(-1, 1, size=1)
        tolerance = np.random.uniform(0.2, 0.5)

        user = SimulatedUser(
            id=f"user_{i}",
            interests=interests,
            opinion_vector=opinion,
            tolerance=tolerance
        )
        users.append(user)

    return users


def run_scenario(config: MultiLLMConfig, n_rounds: int = 10, n_users: int = 20,
                output_dir: str = "./outputs"):
    """
    Run a simulation scenario with given configuration.

    Args:
        config: Scenario configuration
        n_rounds: Number of simulation rounds
        n_users: Number of simulated users
        output_dir: Directory for outputs
    """
    print(f"\n{'='*70}")
    print(f"RUNNING SCENARIO: {config.name}")
    print(f"{'='*70}")
    print(f"Description: {config.description}")
    print(f"Rounds: {n_rounds}")
    print(f"Users: {n_users}")
    print(f"Agents: {len(config.agent_assignments)}")
    print(f"LLM Backends: {len(config.backends)}")
    print(f"{'='*70}\n")

    # Setup
    data_tracker = EnhancedDataTracker(output_dir=output_dir)
    backend_manager = create_backend_manager_from_config(config)
    agents = create_agents_from_config(config)
    users = create_simulated_users(n_users)

    data_tracker.all_users = {user.id: user for user in users}

    # Initialize content generator
    content_generator = ContentGenerator(agents, backend_manager, data_tracker)

    # Initialize recommender
    recommender_config = getattr(config, 'recommender', None)
    if recommender_config and recommender_config.get('type') == 'llm':
        # Use LLM-based recommender
        print(f"Initializing LLM-based recommender with backend: {recommender_config['backend_id']}")
        recommender_backend = backend_manager.get_backend_for_agent("agent_0")  # Use any backend as fallback
        if recommender_config['backend_id'] in backend_manager.backends:
            recommender_backend = backend_manager.backends[recommender_config['backend_id']]

        recommender = LLMRecommender(
            llm_backend=recommender_backend,
            use_personalization=recommender_config.get('use_personalization', True)
        )
        use_llm_recommender = True
    else:
        # Use traditional recommender
        print("Using traditional keyword-based recommender")
        recommender = RecommenderSystem(strategy='relevance')
        use_llm_recommender = False

    interaction_simulator = UserContentInteractionSimulator(data_tracker)

    # Initial topics
    topics = ["climate change", "technology", "healthcare", "education", "politics"]
    content_pool = []

    # Run simulation
    for round_num in range(n_rounds):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num + 1}/{n_rounds}")
        print(f"{'='*70}")

        # Generate initial posts
        new_content = content_generator.generate_initial_posts(
            topics=topics,
            current_round=round_num,
            posts_per_topic=2
        )
        content_pool.extend(new_content)

        # Add sentiment scores
        add_sentiment_scores(content_pool)

        # Generate recommendations
        print(f"\n→ Generating recommendations for {len(users)} users")
        feeds = {}

        if use_llm_recommender:
            # Use LLM-based recommender
            for user in users:
                ranked_content = recommender.rank_content_for_user(
                    content_pool=content_pool,
                    user=user,
                    k=10
                )
                feeds[user.id] = ranked_content
        else:
            # Use traditional recommender
            feeds = recommender.rank_content_for_all_users(content_pool, users, k=10)

        # Store feed assignments
        data_tracker.feed_assignments[round_num] = {
            user_id: [item.id for item in feed]
            for user_id, feed in feeds.items()
        }

        # Simulate user interactions
        interaction_simulator.simulate_interactions(feeds, data_tracker.all_users, round_num)

        # Store user opinions
        data_tracker.round_data['user_opinions'][round_num] = {
            user.id: user.opinion_vector.copy()
            for user in users
        }

        # Generate responses
        if len(content_pool) > 0:
            all_feed_items = []
            for feed in feeds.values():
                all_feed_items.extend(feed)

            responses = content_generator.generate_responses(
                content_pool=content_pool,
                feed_items=all_feed_items,
                current_round=round_num,
                n_responses=min(5, len(agents))
            )
            content_pool.extend(responses)
            add_sentiment_scores(responses)

        print(f"\nRound {round_num + 1} complete:")
        print(f"  Total content items: {len(content_pool)}")
        print(f"  Total interactions: {len(data_tracker.interactions)}")

    # Print LLM statistics
    backend_manager.print_stats()

    # Analyze content by architecture
    print("\n" + "="*70)
    print("CONTENT GENERATION BY LLM ARCHITECTURE")
    print("="*70)

    architecture_stats = {}
    for item in content_pool:
        arch = item.llm_architecture or "unknown"
        if arch not in architecture_stats:
            architecture_stats[arch] = {
                'count': 0,
                'total_likes': 0,
                'total_views': 0,
                'avg_engagement': 0
            }
        architecture_stats[arch]['count'] += 1
        architecture_stats[arch]['total_likes'] += item.total_likes
        architecture_stats[arch]['total_views'] += item.total_views

    for arch, stats in architecture_stats.items():
        if stats['count'] > 0:
            stats['avg_engagement'] = stats['total_likes'] / stats['count']
        print(f"\n{arch}:")
        print(f"  Content items: {stats['count']}")
        print(f"  Total likes: {stats['total_likes']}")
        print(f"  Total views: {stats['total_views']}")
        print(f"  Avg engagement: {stats['avg_engagement']:.2f}")

    print("\n" + "="*70 + "\n")

    # Save all data
    data_tracker.save_all_data()
    save_results(data_tracker, output_path=f"{output_dir}/simulation_summary.json")

    print(f"\n✓ Simulation complete! Results saved to {output_dir}/")

    return data_tracker, architecture_stats


def main():
    parser = argparse.ArgumentParser(description="Run AI content generation scenarios")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4], required=True,
                       help="Scenario number (1-4)")
    parser.add_argument("--rounds", type=int, default=10,
                       help="Number of simulation rounds (default: 10)")
    parser.add_argument("--users", type=int, default=20,
                       help="Number of simulated users (default: 20)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: auto-generated)")

    args = parser.parse_args()

    # Load appropriate configuration
    scenario_configs = {
        1: ("scenario1_llama_only.json", "./scenario1_output"),
        2: ("scenario2_mixed_generation.json", "./scenario2_output"),
        3: ("scenario3_mistral_only.json", "./scenario3_output"),
        4: ("scenario4_mixed_mistral_rec.json", "./scenario4_output")
    }

    config_file, default_output = scenario_configs[args.scenario]
    output_dir = args.output if args.output else default_output

    print(f"Loading configuration from {config_file}...")
    config = MultiLLMConfig.load(config_file)

    # Run scenario
    run_scenario(
        config=config,
        n_rounds=args.rounds,
        n_users=args.users,
        output_dir=output_dir
    )

    print(f"\n{'='*70}")
    print("TO ANALYZE RESULTS, RUN:")
    print(f"  python analyze_scenarios.py --output-dir {output_dir}")
    print("\nTO COMPARE MULTIPLE SCENARIOS, RUN:")
    print("  python analyze_scenarios.py --multi-compare ./scenario1_output ./scenario2_output ./scenario3_output ./scenario4_output")
    print("="*70)


if __name__ == "__main__":
    main()
