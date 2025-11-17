#!/usr/bin/env python3
"""
Main Simulation Runner
Demonstrates the AI content generation and recommendation dynamics system.
"""

import numpy as np
import argparse
from typing import List
import json

from simulation_framework import (
    Agent, SimulatedUser, ContentItem,
    ContentGenerator, RecommenderSystem, MetricsCollector, HumanDynamicsModel,
    simulate_engagement, save_results
)
from llm_backend import get_llm_backend


def create_diverse_agents(n_agents: int = 6) -> List[Agent]:
    """Create a diverse set of AI agents with different personas."""
    
    agent_configs = [
        {
            "persona": "Progressive activist",
            "objective": "Advocate for social and environmental causes",
            "temperature": 0.8,
            "response_style": "passionate"
        },
        {
            "persona": "Tech entrepreneur",
            "objective": "Discuss innovation and market opportunities",
            "temperature": 0.7,
            "response_style": "optimistic"
        },
        {
            "persona": "Academic researcher",
            "objective": "Share evidence-based insights and nuanced analysis",
            "temperature": 0.5,
            "response_style": "analytical"
        },
        {
            "persona": "Skeptical journalist",
            "objective": "Question claims and seek accountability",
            "temperature": 0.6,
            "response_style": "critical"
        },
        {
            "persona": "Community organizer",
            "objective": "Build consensus and mobilize collective action",
            "temperature": 0.7,
            "response_style": "collaborative"
        },
        {
            "persona": "Contrarian thinker",
            "objective": "Challenge mainstream narratives",
            "temperature": 0.8,
            "response_style": "provocative"
        }
    ]
    
    agents = []
    for i in range(min(n_agents, len(agent_configs))):
        config = agent_configs[i]
        agent = Agent(
            id=f"agent_{i}",
            persona=config["persona"],
            objective=config["objective"],
            temperature=config["temperature"],
            response_style=config["response_style"]
        )
        agents.append(agent)
    
    return agents


def create_simulated_users(n_users: int = 10) -> List[SimulatedUser]:
    """Create simulated human users with diverse interests."""
    
    interest_pools = [
        ["climate change", "environment", "sustainability"],
        ["technology", "AI", "innovation"],
        ["healthcare", "medicine", "public health"],
        ["politics", "policy", "governance"],
        ["education", "science", "research"],
        ["economy", "business", "finance"],
        ["arts", "culture", "creativity"],
        ["sports", "fitness", "wellness"]
    ]
    
    users = []
    for i in range(n_users):
        # Randomly select interests
        interests = interest_pools[i % len(interest_pools)]
        
        # Initialize opinion vector (1D for simplicity, representing left-right or positive-negative)
        opinion = np.random.uniform(-1, 1, size=1)
        
        # Random tolerance for bounded confidence
        tolerance = np.random.uniform(0.2, 0.5)
        
        user = SimulatedUser(
            id=f"user_{i}",
            interests=interests,
            opinion_vector=opinion,
            tolerance=tolerance
        )
        users.append(user)
    
    return users


def add_simple_sentiment_scores(content_pool: List[ContentItem]) -> dict:
    """Add simple sentiment scores to content (mock implementation)."""
    sentiment_map = {}
    
    for item in content_pool:
        # Simple heuristic: look for positive/negative words
        text_lower = item.text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'important', 'positive', 
                         'improve', 'better', 'solution', 'opportunity', 'success']
        negative_words = ['bad', 'terrible', 'wrong', 'crisis', 'problem', 
                         'danger', 'risk', 'threat', 'concern', 'failure']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Simple sentiment score: -1 to 1
        total_words = len(text_lower.split())
        sentiment = (pos_count - neg_count) / max(total_words, 1)
        sentiment = np.clip(sentiment, -1, 1)
        
        item.sentiment_score = sentiment
        sentiment_map[item.id] = sentiment
    
    return sentiment_map


def run_simulation(n_rounds: int = 5, 
                   n_agents: int = 6,
                   n_users: int = 10,
                   recommender_strategy: str = "relevance",
                   use_mock_llm: bool = False,
                   model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """
    Run the full simulation.
    
    Args:
        n_rounds: Number of simulation rounds
        n_agents: Number of AI agents
        n_users: Number of simulated human users
        recommender_strategy: Strategy for recommendation ('relevance', 'diversity', 'llm', 'random')
        use_mock_llm: Whether to use mock LLM (for testing)
        model_name: LLM model to use
    """
    
    print("="*60)
    print("AI Content & Recommender System Dynamics Simulation")
    print("="*60)
    print(f"Configuration:")
    print(f"  Rounds: {n_rounds}")
    print(f"  Agents: {n_agents}")
    print(f"  Users: {n_users}")
    print(f"  Recommender: {recommender_strategy}")
    print(f"  Model: {'Mock' if use_mock_llm else model_name}")
    print("="*60)
    
    # Initialize components
    print("\n[1/6] Initializing LLM backend...")
    llm = get_llm_backend(use_mock=use_mock_llm, model_name=model_name)
    
    print("\n[2/6] Creating agents and users...")
    agents = create_diverse_agents(n_agents)
    users = create_simulated_users(n_users)
    print(f"  Created {len(agents)} agents and {len(users)} users")
    
    print("\n[3/6] Initializing simulation components...")
    generator = ContentGenerator(agents, llm)
    recommender = RecommenderSystem(llm, strategy=recommender_strategy)
    metrics = MetricsCollector()
    human_dynamics = HumanDynamicsModel(users, model_type="bounded_confidence")
    
    # Define topics for discussion
    topics = ["climate change", "artificial intelligence", "healthcare reform"]
    
    # Initialize content pool
    print("\n[4/6] Generating initial content...")
    content_pool = generator.generate_initial_posts(topics, posts_per_topic=2)
    print(f"  Generated {len(content_pool)} initial posts")
    
    # Add sentiment scores
    sentiment_map = add_simple_sentiment_scores(content_pool)
    
    # Simulation loop
    print("\n[5/6] Running simulation rounds...")
    for round_num in range(n_rounds):
        print(f"\n--- Round {round_num + 1}/{n_rounds} ---")
        
        # Step 1: Recommender system generates feeds for each user
        print(f"  Generating personalized feeds...")
        feed_assignments = {}
        for user in users:
            feed = recommender.rank_content(content_pool, user, k=5)
            feed_assignments[user.id] = feed
            # Update user exposure history
            user.exposure_history.extend([item.id for item in feed])
        
        # Step 2: Update human opinions based on exposure
        print(f"  Updating user opinions...")
        human_dynamics.update_opinions(feed_assignments, sentiment_map)
        
        # Step 3: Simulate engagement
        print(f"  Simulating engagement...")
        simulate_engagement(content_pool, feed_assignments)
        
        # Step 4: Agents generate new content (responses to feed)
        print(f"  Generating agent responses...")
        # Collect all feed items
        all_feed_items = []
        for feed in feed_assignments.values():
            all_feed_items.extend(feed)
        # Remove duplicates
        unique_feed_items = list({item.id: item for item in all_feed_items}.values())
        
        new_responses = generator.generate_responses(
            content_pool, 
            unique_feed_items, 
            n_responses=min(4, n_agents)
        )
        
        # Add new content to pool
        content_pool.extend(new_responses)
        
        # Add sentiment scores to new content
        new_sentiment_map = add_simple_sentiment_scores(new_responses)
        sentiment_map.update(new_sentiment_map)
        
        print(f"  Generated {len(new_responses)} new responses")
        print(f"  Total content items: {len(content_pool)}")
        
        # Step 5: Collect metrics
        metrics.update_metrics(content_pool, users)
        
        # Print current metrics
        print(f"  Content diversity: {metrics.history['content_diversity'][-1]:.3f}")
        print(f"  Engagement Gini: {metrics.history['engagement_gini'][-1]:.3f}")
        print(f"  Opinion variance: {metrics.history['opinion_variance'][-1]:.3f}")
        print(f"  Polarization: {metrics.history['polarization'][-1]:.3f}")
    
    # Save results
    print("\n[6/6] Saving results...")
    save_results(content_pool, metrics, output_path="./simulation_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    
    summary = metrics.get_summary()
    for metric_name, stats in summary.items():
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        print(f"  Final value: {stats['final']:.3f}")
        print(f"  Average: {stats['mean']:.3f}")
        print(f"  Trend: {stats['trend']}")
    
    print(f"\nTotal content generated: {len(content_pool)} items")
    print(f"Total LLM calls: {generator.generation_count}")
    
    print("\n" + "="*60)
    print("Results saved to: /home/claude/simulation_results.json")
    print("="*60)
    
    return content_pool, metrics, users


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Content & Recommender Dynamics Simulator")
    parser.add_argument("--rounds", type=int, default=5, help="Number of simulation rounds")
    parser.add_argument("--agents", type=int, default=6, help="Number of AI agents")
    parser.add_argument("--users", type=int, default=10, help="Number of simulated users")
    parser.add_argument("--recommender", type=str, default="relevance", 
                       choices=["relevance", "diversity", "llm", "random"],
                       help="Recommender strategy")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM for testing")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="LLM model name")
    
    args = parser.parse_args()
    
    try:
        run_simulation(
            n_rounds=args.rounds,
            n_agents=args.agents,
            n_users=args.users,
            recommender_strategy=args.recommender,
            use_mock_llm=args.mock,
            model_name=args.model
        )
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()