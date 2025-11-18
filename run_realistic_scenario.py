"""
REALISTIC Social Media Simulation with AI Content Generation

This simulation models a more realistic social media ecosystem:

POPULATIONS:
1. AI AGENTS (6 content creators):
   - Generate posts and responses using LLMs (Llama, Mistral, etc.)
   - Probabilistically decide each round: post new content, respond, or stay quiet
   - Each agent has a distinct persona and uses a specific LLM backend

2. SIMULATED USERS (20 passive audience members):
   - Consume content via personalized recommendation feeds
   - Engage by liking content that aligns with their opinions
   - Their opinions evolve based on exposure (bounded confidence model)
   - They do NOT create content - they represent human audience influence

SIMULATION FLOW:
  Initialization:
    - Each AI agent creates 1-2 seed posts
    - Users initialized with diverse interests and opinions

  Each Round:
    1. CONTENT DISCOVERY PHASE:
       - Recommender generates personalized feeds for users
       - Users view and engage with content (likes)
       - Engagement tracked (views, likes)

    2. AGENT DECISION PHASE:
       - Each agent probabilistically decides:
         * Post new content (30% chance)
         * Respond to engaging content (50% chance if content exists)
         * Stay quiet (20% chance)

    3. CONTENT GENERATION PHASE:
       - Agents execute their decisions
       - New content enters the pool

    4. OPINION UPDATE PHASE:
       - Users' opinions shift based on content exposure
       - Polarization metrics computed

This models AI content → human consumption → opinion influence dynamics.
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


def create_simulated_users(n_users: int = 20) -> List[SimulatedUser]:
    """
    Create simulated users (passive audience members).

    These users:
    - Consume content through personalized feeds
    - Engage by liking content that aligns with their opinions
    - Their opinions evolve based on exposure
    - They do NOT create content
    """
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


def run_realistic_scenario(config: MultiLLMConfig, n_rounds: int = 10, n_users: int = 20,
                          engagement_threshold: int = 2, output_dir: str = "./outputs"):
    """
    Run a REALISTIC social media simulation.

    Flow:
      Initialization → Each agent creates seed content
      Each Round → Content discovery → Agent decisions → Content generation → Opinion update

    Args:
        config: Scenario configuration
        n_rounds: Number of simulation rounds
        n_users: Number of simulated users (passive audience)
        engagement_threshold: Minimum engagement to be considered "engaging"
        output_dir: Directory for outputs
    """
    print(f"\n{'='*70}")
    print(f"REALISTIC SOCIAL MEDIA SIMULATION")
    print(f"{'='*70}")
    print(f"Scenario: {config.name}")
    print(f"Description: {config.description}")
    print(f"\nPopulations:")
    print(f"  AI Agents (content creators): {len(config.agent_assignments)}")
    print(f"  Simulated Users (audience): {n_users}")
    print(f"\nSimulation:")
    print(f"  Rounds: {n_rounds}")
    print(f"  LLM Backends: {len(config.backends)}")
    print(f"{'='*70}\n")

    # Setup
    data_tracker = EnhancedDataTracker(output_dir=output_dir)
    backend_manager = create_backend_manager_from_config(config)
    agents = create_agents_from_config(config)
    users = create_simulated_users(n_users)

    data_tracker.all_users = {user.id: user for user in users}

    # Initialize components
    content_generator = ContentGenerator(agents, backend_manager, data_tracker)

    # Initialize recommender
    recommender_config = getattr(config, 'recommender', None)
    if recommender_config and recommender_config.get('type') == 'llm':
        print(f"Recommender: LLM-based (backend: {recommender_config['backend_id']})")
        recommender_backend = backend_manager.backends[recommender_config['backend_id']]
        recommender = LLMRecommender(
            llm_backend=recommender_backend,
            use_personalization=recommender_config.get('use_personalization', True)
        )
        use_llm_recommender = True
    else:
        print("Recommender: Traditional keyword-based")
        recommender = RecommenderSystem(strategy='relevance')
        use_llm_recommender = False

    interaction_simulator = UserContentInteractionSimulator(data_tracker)

    # Topics for content generation
    topics = ["climate change", "technology", "healthcare", "education", "politics", "economics"]

    # =================================================================
    # INITIALIZATION: All agents create seed content
    # =================================================================
    print(f"\n{'='*70}")
    print("INITIALIZATION PHASE")
    print(f"{'='*70}")

    content_pool = content_generator.generate_seed_content(topics=topics, posts_per_agent=2)
    add_sentiment_scores(content_pool)

    print(f"✓ Initial content pool: {len(content_pool)} posts from {len(agents)} agents\n")

    # =================================================================
    # SIMULATION ROUNDS
    # =================================================================
    for round_num in range(1, n_rounds + 1):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num}/{n_rounds}")
        print(f"{'='*70}")

        # -----------------------------------------------------------------
        # PHASE 1: CONTENT DISCOVERY (Users consume content)
        # -----------------------------------------------------------------
        print(f"\n→ PHASE 1: CONTENT DISCOVERY")
        print(f"   Generating personalized feeds for {len(users)} users")

        feeds = {}
        if use_llm_recommender:
            for user in users:
                ranked_content = recommender.rank_content_for_user(
                    content_pool=content_pool,
                    user=user,
                    k=10
                )
                feeds[user.id] = ranked_content
        else:
            feeds = recommender.rank_content_for_all_users(content_pool, users, k=10)

        # Store feed assignments
        data_tracker.feed_assignments[round_num] = {
            user_id: [item.id for item in feed]
            for user_id, feed in feeds.items()
        }

        # Users interact with content
        print(f"   Users viewing and engaging with content")
        interaction_simulator.simulate_interactions(feeds, data_tracker.all_users, round_num)

        # Store user opinions
        data_tracker.round_data['user_opinions'][round_num] = {
            user.id: user.opinion_vector.copy()
            for user in users
        }

        # Update engagement scores based on interactions
        for item in content_pool:
            # Engagement score increases with views and likes
            if item.total_views > 0:
                item.engagement_score = item.total_likes + (item.total_views * 0.1)

        # -----------------------------------------------------------------
        # PHASE 2: AGENT DECISION (What will agents do?)
        # -----------------------------------------------------------------
        print(f"\n→ PHASE 2: AGENT DECISION")
        decisions = content_generator.agent_decide_actions(
            content_pool=content_pool,
            engagement_threshold=engagement_threshold
        )

        # -----------------------------------------------------------------
        # PHASE 3: CONTENT GENERATION (Agents create content)
        # -----------------------------------------------------------------
        new_content = content_generator.generate_round_content(
            current_round=round_num,
            action_decisions=decisions
        )

        if new_content:
            add_sentiment_scores(new_content)
            content_pool.extend(new_content)

        # -----------------------------------------------------------------
        # ROUND SUMMARY
        # -----------------------------------------------------------------
        print(f"\n{'─'*70}")
        print(f"Round {round_num} Summary:")
        print(f"  Content pool: {len(content_pool)} items")
        print(f"  New content this round: {len(new_content)}")
        print(f"  Total interactions: {len(data_tracker.interactions)}")
        print(f"  Active agents: {sum(1 for d in decisions.values() if d['post_new'] or d['respond'])}/{len(agents)}")
        print(f"{'─'*70}")

    # =================================================================
    # FINAL STATISTICS
    # =================================================================
    print(f"\n{'='*70}")
    print("SIMULATION COMPLETE")
    print(f"{'='*70}")

    # LLM statistics
    backend_manager.print_stats()

    # Content by architecture
    print(f"\n{'='*70}")
    print("CONTENT GENERATION BY ARCHITECTURE")
    print(f"{'='*70}")

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
        print(f"  Posts: {stats['count']}")
        print(f"  Total likes: {stats['total_likes']}")
        print(f"  Total views: {stats['total_views']}")
        print(f"  Avg engagement: {stats['avg_engagement']:.2f}")

    print(f"\n{'='*70}\n")

    # Save all data
    data_tracker.save_all_data()
    save_results(data_tracker, output_path=f"{output_dir}/simulation_summary.json")

    print(f"\n✓ Simulation complete! Results saved to {output_dir}/")
    print(f"\nTo analyze results:")
    print(f"  python analyze_scenarios.py --output-dir {output_dir}")

    return data_tracker, architecture_stats


def main():
    parser = argparse.ArgumentParser(description="Run realistic AI content generation scenario")
    parser.add_argument("--scenario", type=int, choices=[1, 2], required=True,
                       help="Scenario number (1=Llama-only, 2=Mixed)")
    parser.add_argument("--rounds", type=int, default=10,
                       help="Number of simulation rounds (default: 10)")
    parser.add_argument("--users", type=int, default=20,
                       help="Number of simulated users/audience (default: 20)")
    parser.add_argument("--engagement-threshold", type=int, default=2,
                       help="Min engagement score for agents to respond (default: 2)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: auto-generated)")

    args = parser.parse_args()

    # Load configuration
    if args.scenario == 1:
        config_file = "scenario1_llama_only.json"
        default_output = "./scenario1_output"
    else:
        config_file = "scenario2_mixed_generation.json"
        default_output = "./scenario2_output"

    output_dir = args.output if args.output else default_output

    print(f"Loading configuration from {config_file}...")
    config = MultiLLMConfig.load(config_file)

    # Run realistic scenario
    run_realistic_scenario(
        config=config,
        n_rounds=args.rounds,
        n_users=args.users,
        engagement_threshold=args.engagement_threshold,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
