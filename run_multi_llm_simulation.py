"""
Multi-LLM Simulation Runner
Demonstrates how to run simulations with multiple LLM architectures.
"""

import argparse
import numpy as np
from typing import List
from pathlib import Path

from llm_backend import create_multi_llm_setup, MultiLLMManager
from multi_llm_config import MultiLLMConfig, create_example_config
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
    # Convert config backends to dictionary format
    backend_config_dict = {}
    for backend in config.backends:
        backend_config_dict[backend.backend_id] = {
            'type': backend.type,
            'model': backend.model,
            'use_mock': backend.use_mock
        }

    # Create manager
    manager = create_multi_llm_setup(backend_config_dict)

    # Assign backends to agents
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
        # Random interests
        interests = interests_pool[i % len(interests_pool)]

        # Random initial opinion (between -1 and 1)
        opinion = np.random.uniform(-1, 1, size=1)

        # Random tolerance (how open they are to different views)
        tolerance = np.random.uniform(0.2, 0.5)

        user = SimulatedUser(
            id=f"user_{i}",
            interests=interests,
            opinion_vector=opinion,
            tolerance=tolerance
        )
        users.append(user)

    return users


def run_simulation(config: MultiLLMConfig, n_rounds: int = 5, n_users: int = 10,
                  recommender_strategy: str = "relevance", output_dir: str = "./outputs"):
    """
    Run a multi-LLM simulation.

    Args:
        config: Multi-LLM configuration
        n_rounds: Number of simulation rounds
        n_users: Number of simulated users
        recommender_strategy: Recommendation strategy ('relevance', 'diversity', 'engagement', 'random')
        output_dir: Directory for outputs
    """
    print(f"\n{'='*70}")
    print(f"MULTI-LLM SIMULATION: {config.name}")
    print(f"{'='*70}")
    print(f"Description: {config.description}")
    print(f"Rounds: {n_rounds}")
    print(f"Users: {n_users}")
    print(f"Agents: {len(config.agent_assignments)}")
    print(f"LLM Backends: {len(config.backends)}")
    print(f"Recommender: {recommender_strategy}")
    print(f"{'='*70}\n")

    # Setup
    data_tracker = EnhancedDataTracker(output_dir=output_dir)
    backend_manager = create_backend_manager_from_config(config)
    agents = create_agents_from_config(config)
    users = create_simulated_users(n_users)

    # Store users in tracker
    data_tracker.all_users = {user.id: user for user in users}

    # Initialize components
    content_generator = ContentGenerator(agents, backend_manager, data_tracker)
    recommender = RecommenderSystem(strategy=recommender_strategy)
    interaction_simulator = UserContentInteractionSimulator(data_tracker)

    # Initial topics
    topics = ["climate change", "technology", "healthcare"]
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

        # Generate recommendations for all users
        feeds = recommender.rank_content_for_all_users(content_pool, users, k=10)

        # Store feed assignments
        data_tracker.feed_assignments[round_num] = {
            user_id: [item.id for item in feed]
            for user_id, feed in feeds.items()
        }

        # Simulate user interactions
        interaction_simulator.simulate_interactions(feeds, data_tracker.all_users, round_num)

        # Store user opinions for this round
        data_tracker.round_data['user_opinions'][round_num] = {
            user.id: user.opinion_vector.copy()
            for user in users
        }

        # Generate responses to popular content
        if len(content_pool) > 0:
            # Get popular content from feeds
            all_feed_items = []
            for feed in feeds.values():
                all_feed_items.extend(feed)

            responses = content_generator.generate_responses(
                content_pool=content_pool,
                feed_items=all_feed_items,
                current_round=round_num,
                n_responses=3
            )
            content_pool.extend(responses)
            add_sentiment_scores(responses)

        print(f"\nRound {round_num + 1} complete:")
        print(f"  Total content items: {len(content_pool)}")
        print(f"  Total interactions: {len(data_tracker.interactions)}")

    # Print LLM statistics
    backend_manager.print_stats()

    # Analyze content by LLM architecture
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


def main():
    parser = argparse.ArgumentParser(description="Run multi-LLM social media simulation")
    parser.add_argument("--config", type=str, help="Path to configuration file (JSON)")
    parser.add_argument("--example", action="store_true", help="Use example mock configuration")
    parser.add_argument("--rounds", type=int, default=5, help="Number of simulation rounds")
    parser.add_argument("--users", type=int, default=10, help="Number of simulated users")
    parser.add_argument("--recommender", type=str, default="relevance",
                       choices=["relevance", "diversity", "engagement", "random"],
                       help="Recommender strategy")
    parser.add_argument("--output", type=str, default="./outputs_multi_llm",
                       help="Output directory")

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = MultiLLMConfig.load(args.config)
    elif args.example:
        print("Using example mock configuration...")
        config = create_example_config()
    else:
        print("Error: Please specify --config or --example")
        return

    # Run simulation
    run_simulation(
        config=config,
        n_rounds=args.rounds,
        n_users=args.users,
        recommender_strategy=args.recommender,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
