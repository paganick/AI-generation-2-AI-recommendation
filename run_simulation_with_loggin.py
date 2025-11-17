#!/usr/bin/env python3
"""
Main Simulation Runner with Comprehensive Logging
Logs all interactions: initialization, agent creation, content generation, recommendation, etc.
"""

import numpy as np
import argparse
from typing import List
import json
import logging
from datetime import datetime
from pathlib import Path

from simulation_framework import (
    Agent, SimulatedUser, ContentItem,
    ContentGenerator, RecommenderSystem, MetricsCollector, HumanDynamicsModel,
    simulate_engagement, save_results
)
from llm_backend import get_llm_backend


class SimulationLogger:
    """Enhanced logger that tracks all simulation interactions."""
    
    def __init__(self, log_file: str = "./simulation_log.jsonl"):
        self.log_file = log_file
        self.log_entries = []
        
        # Also set up Python logging for console output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create directory if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear previous log file
        with open(log_file, 'w') as f:
            pass
    
    def log_event(self, event_type: str, data: dict):
        """Log an event with timestamp."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        self.log_entries.append(entry)
        
        # Write to file immediately (JSONL format - one JSON object per line)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')
        
        # Also print to console
        self.logger.info(f"{event_type}: {self._format_data(data)}")
    
    def _format_data(self, data: dict) -> str:
        """Format data for console output."""
        # Create a short summary for console
        if 'summary' in data:
            return data['summary']
        return str(data)[:100]
    
    def save_full_log(self, output_file: str = "./simulation_log_full.json"):
        """Save complete log as single JSON file."""
        # Create directory if it doesn't exist
        log_dir = Path(output_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.log_entries, f, indent=2, default=str)
        self.logger.info(f"Full log saved to {output_file}")


def create_diverse_agents(n_agents: int = 6, sim_logger: SimulationLogger = None) -> List[Agent]:
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
        
        # Log agent creation
        if sim_logger:
            sim_logger.log_event("AGENT_CREATED", {
                'agent_id': agent.id,
                'persona': agent.persona,
                'objective': agent.objective,
                'temperature': agent.temperature,
                'response_style': agent.response_style,
                'summary': f"Created {agent.persona} agent"
            })
    
    return agents


def create_simulated_users(n_users: int = 10, sim_logger: SimulationLogger = None) -> List[SimulatedUser]:
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
        
        # Initialize opinion vector
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
        
        # Log user creation
        if sim_logger:
            sim_logger.log_event("USER_CREATED", {
                'user_id': user.id,
                'interests': user.interests,
                'initial_opinion': float(user.opinion_vector[0]),
                'tolerance': user.tolerance,
                'summary': f"Created user with interests: {', '.join(interests[:2])}"
            })
    
    return users


def add_simple_sentiment_scores(content_pool: List[ContentItem], 
                                sim_logger: SimulationLogger = None) -> dict:
    """Add simple sentiment scores to content."""
    sentiment_map = {}
    
    for item in content_pool:
        text_lower = item.text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'important', 'positive', 
                         'improve', 'better', 'solution', 'opportunity', 'success']
        negative_words = ['bad', 'terrible', 'wrong', 'crisis', 'problem', 
                         'danger', 'risk', 'threat', 'concern', 'failure']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text_lower.split())
        sentiment = (pos_count - neg_count) / max(total_words, 1)
        sentiment = np.clip(sentiment, -1, 1)
        
        item.sentiment_score = sentiment
        sentiment_map[item.id] = sentiment
        
        # Log sentiment analysis
        if sim_logger:
            sim_logger.log_event("SENTIMENT_ANALYZED", {
                'content_id': item.id,
                'sentiment_score': float(sentiment),
                'positive_words': pos_count,
                'negative_words': neg_count,
                'summary': f"Sentiment: {sentiment:.2f} for content {item.id[:8]}"
            })
    
    return sentiment_map


def run_simulation(n_rounds: int = 5, 
                   n_agents: int = 6,
                   n_users: int = 10,
                   recommender_strategy: str = "relevance",
                   use_mock_llm: bool = False,
                   model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                   log_file: str = "./simulation_log.jsonl"):
    """
    Run the full simulation with comprehensive logging.
    """
    
    # Initialize logger
    sim_logger = SimulationLogger(log_file=log_file)
    
    sim_logger.log_event("SIMULATION_STARTED", {
        'n_rounds': n_rounds,
        'n_agents': n_agents,
        'n_users': n_users,
        'recommender_strategy': recommender_strategy,
        'use_mock_llm': use_mock_llm,
        'model_name': model_name if not use_mock_llm else "mock",
        'summary': f"Starting simulation with {n_agents} agents, {n_users} users, {n_rounds} rounds"
    })
    
    print("="*60)
    print("AI Content & Recommender System Dynamics Simulation")
    print("="*60)
    print(f"Configuration:")
    print(f"  Rounds: {n_rounds}")
    print(f"  Agents: {n_agents}")
    print(f"  Users: {n_users}")
    print(f"  Recommender: {recommender_strategy}")
    print(f"  Model: {'Mock' if use_mock_llm else model_name}")
    print(f"  Log file: {log_file}")
    print("="*60)
    
    # Initialize components
    print("\n[1/6] Initializing LLM backend...")
    sim_logger.log_event("LLM_INITIALIZATION_STARTED", {
        'model_name': model_name if not use_mock_llm else "mock",
        'summary': "Initializing LLM backend"
    })
    
    llm = get_llm_backend(use_mock=use_mock_llm, model_name=model_name)
    
    sim_logger.log_event("LLM_INITIALIZATION_COMPLETED", {
        'model_name': model_name if not use_mock_llm else "mock",
        'summary': "LLM backend ready"
    })
    
    print("\n[2/6] Creating agents and users...")
    agents = create_diverse_agents(n_agents, sim_logger=sim_logger)
    users = create_simulated_users(n_users, sim_logger=sim_logger)
    print(f"  Created {len(agents)} agents and {len(users)} users")
    
    print("\n[3/6] Initializing simulation components...")
    generator = ContentGenerator(agents, llm)
    recommender = RecommenderSystem(llm, strategy=recommender_strategy)
    metrics = MetricsCollector()
    human_dynamics = HumanDynamicsModel(users, model_type="bounded_confidence")
    
    sim_logger.log_event("COMPONENTS_INITIALIZED", {
        'content_generator': 'initialized',
        'recommender_system': recommender_strategy,
        'metrics_collector': 'initialized',
        'human_dynamics_model': 'bounded_confidence',
        'summary': f"Initialized all components with {recommender_strategy} recommender"
    })
    
    # Define topics for discussion
    topics = ["climate change", "artificial intelligence", "healthcare reform"]
    
    sim_logger.log_event("TOPICS_DEFINED", {
        'topics': topics,
        'summary': f"Discussion topics: {', '.join(topics)}"
    })
    
    # Initialize content pool
    print("\n[4/6] Generating initial content...")
    sim_logger.log_event("INITIAL_CONTENT_GENERATION_STARTED", {
        'topics': topics,
        'posts_per_topic': 2,
        'summary': "Starting initial content generation"
    })
    
    content_pool = generator.generate_initial_posts(topics, posts_per_topic=2)
    
    # Log each generated post
    for item in content_pool:
        sim_logger.log_event("CONTENT_GENERATED", {
            'content_id': item.id,
            'author_id': item.author_id,
            'topic': item.topic,
            'text': item.text,
            'timestamp': item.timestamp,
            'type': 'initial_post',
            'summary': f"{item.author_id} posted about {item.topic}"
        })
    
    print(f"  Generated {len(content_pool)} initial posts")
    
    # Add sentiment scores
    sentiment_map = add_simple_sentiment_scores(content_pool, sim_logger=sim_logger)
    
    # Simulation loop
    print("\n[5/6] Running simulation rounds...")
    for round_num in range(n_rounds):
        print(f"\n--- Round {round_num + 1}/{n_rounds} ---")
        
        sim_logger.log_event("ROUND_STARTED", {
            'round_number': round_num + 1,
            'total_rounds': n_rounds,
            'content_pool_size': len(content_pool),
            'summary': f"Starting round {round_num + 1}/{n_rounds}"
        })
        
        # Step 1: Recommender system generates feeds for each user
        print(f"  Generating personalized feeds...")
        sim_logger.log_event("FEED_GENERATION_STARTED", {
            'round': round_num + 1,
            'n_users': len(users),
            'strategy': recommender_strategy,
            'summary': f"Generating feeds for {len(users)} users"
        })
        
        feed_assignments = {}
        for user in users:
            feed = recommender.rank_content(content_pool, user, k=5)
            feed_assignments[user.id] = feed
            
            # Log feed assignment
            sim_logger.log_event("FEED_ASSIGNED", {
                'round': round_num + 1,
                'user_id': user.id,
                'user_interests': user.interests,
                'feed_size': len(feed),
                'feed_items': [{'id': item.id, 'topic': item.topic, 'author': item.author_id} 
                              for item in feed],
                'summary': f"Feed of {len(feed)} items assigned to {user.id}"
            })
            
            # Update user exposure history
            user.exposure_history.extend([item.id for item in feed])
        
        sim_logger.log_event("FEED_GENERATION_COMPLETED", {
            'round': round_num + 1,
            'total_feeds': len(feed_assignments),
            'summary': f"Generated {len(feed_assignments)} personalized feeds"
        })
        
        # Step 2: Update human opinions based on exposure
        print(f"  Updating user opinions...")
        sim_logger.log_event("OPINION_UPDATE_STARTED", {
            'round': round_num + 1,
            'summary': "Updating user opinions based on content exposure"
        })
        
        # Store pre-update opinions for logging
        pre_update_opinions = {user.id: float(user.opinion_vector[0]) for user in users}
        
        human_dynamics.update_opinions(feed_assignments, sentiment_map)
        
        # Log opinion changes
        for user in users:
            post_update_opinion = float(user.opinion_vector[0])
            opinion_change = post_update_opinion - pre_update_opinions[user.id]
            
            sim_logger.log_event("OPINION_UPDATED", {
                'round': round_num + 1,
                'user_id': user.id,
                'pre_opinion': pre_update_opinions[user.id],
                'post_opinion': post_update_opinion,
                'change': opinion_change,
                'tolerance': user.tolerance,
                'summary': f"{user.id} opinion: {pre_update_opinions[user.id]:.3f} → {post_update_opinion:.3f}"
            })
        
        # Step 3: Simulate engagement
        print(f"  Simulating engagement...")
        sim_logger.log_event("ENGAGEMENT_SIMULATION_STARTED", {
            'round': round_num + 1,
            'summary': "Simulating content engagement"
        })
        
        # Store pre-engagement scores
        pre_engagement = {item.id: item.engagement_score for item in content_pool}
        
        simulate_engagement(content_pool, feed_assignments)
        
        # Log engagement changes
        for item in content_pool:
            engagement_change = item.engagement_score - pre_engagement[item.id]
            if engagement_change > 0:
                sim_logger.log_event("ENGAGEMENT_RECORDED", {
                    'round': round_num + 1,
                    'content_id': item.id,
                    'pre_engagement': pre_engagement[item.id],
                    'post_engagement': item.engagement_score,
                    'change': engagement_change,
                    'summary': f"Content {item.id[:8]} gained {engagement_change:.2f} engagement"
                })
        
        # Step 4: Agents generate new content (responses to feed)
        print(f"  Generating agent responses...")
        
        # Collect all feed items
        all_feed_items = []
        for feed in feed_assignments.values():
            all_feed_items.extend(feed)
        # Remove duplicates
        unique_feed_items = list({item.id: item for item in all_feed_items}.values())
        
        sim_logger.log_event("RESPONSE_GENERATION_STARTED", {
            'round': round_num + 1,
            'n_agents_responding': min(4, n_agents),
            'n_feed_items': len(unique_feed_items),
            'summary': f"Agents generating responses to {len(unique_feed_items)} feed items"
        })
        
        new_responses = generator.generate_responses(
            content_pool, 
            unique_feed_items, 
            n_responses=min(4, n_agents)
        )
        
        # Log each new response
        for response in new_responses:
            sim_logger.log_event("CONTENT_GENERATED", {
                'content_id': response.id,
                'author_id': response.author_id,
                'parent_id': response.parent_id,
                'topic': response.topic,
                'text': response.text,
                'timestamp': response.timestamp,
                'type': 'response',
                'summary': f"{response.author_id} responded to {response.parent_id[:8]}"
            })
        
        # Add new content to pool
        content_pool.extend(new_responses)
        
        # Add sentiment scores to new content
        new_sentiment_map = add_simple_sentiment_scores(new_responses, sim_logger=sim_logger)
        sentiment_map.update(new_sentiment_map)
        
        print(f"  Generated {len(new_responses)} new responses")
        print(f"  Total content items: {len(content_pool)}")
        
        sim_logger.log_event("RESPONSE_GENERATION_COMPLETED", {
            'round': round_num + 1,
            'new_responses': len(new_responses),
            'total_content': len(content_pool),
            'summary': f"Generated {len(new_responses)} responses (total: {len(content_pool)})"
        })
        
        # Step 5: Collect metrics
        metrics.update_metrics(content_pool, users)
        
        sim_logger.log_event("METRICS_COMPUTED", {
            'round': round_num + 1,
            'content_diversity': metrics.history['content_diversity'][-1],
            'engagement_gini': metrics.history['engagement_gini'][-1],
            'opinion_variance': metrics.history['opinion_variance'][-1],
            'polarization': metrics.history['polarization'][-1],
            'summary': f"Metrics computed for round {round_num + 1}"
        })
        
        # Print current metrics
        print(f"  Content diversity: {metrics.history['content_diversity'][-1]:.3f}")
        print(f"  Engagement Gini: {metrics.history['engagement_gini'][-1]:.3f}")
        print(f"  Opinion variance: {metrics.history['opinion_variance'][-1]:.3f}")
        print(f"  Polarization: {metrics.history['polarization'][-1]:.3f}")
        
        sim_logger.log_event("ROUND_COMPLETED", {
            'round_number': round_num + 1,
            'summary': f"Completed round {round_num + 1}/{n_rounds}"
        })
    
    # Save results
    print("\n[6/6] Saving results...")
    save_results(content_pool, metrics, output_path="./simulation_results_with_logging.json")
    
    # Save full log
    sim_logger.save_full_log()
    
    sim_logger.log_event("SIMULATION_COMPLETED", {
        'total_content': len(content_pool),
        'total_llm_calls': generator.generation_count,
        'final_metrics': metrics.get_summary(),
        'summary': f"Simulation completed: {len(content_pool)} items, {generator.generation_count} LLM calls"
    })
    
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
    print("Results saved to:")
    print("  - simulation_results.json (final results)")
    print(f"  - {log_file} (interaction log, JSONL format)")
    print("  - simulation_log_full.json (complete log, JSON format)")
    print("="*60)
    
    return content_pool, metrics, users


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Content & Recommender Dynamics Simulator with Logging")
    parser.add_argument("--rounds", type=int, default=5, help="Number of simulation rounds")
    parser.add_argument("--agents", type=int, default=6, help="Number of AI agents")
    parser.add_argument("--users", type=int, default=10, help="Number of simulated users")
    parser.add_argument("--recommender", type=str, default="relevance", 
                       choices=["relevance", "diversity", "llm", "random"],
                       help="Recommender strategy")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM for testing")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="LLM model name")
    parser.add_argument("--logfile", type=str, default="./simulation_log.jsonl",
                       help="Path to log file")
    
    args = parser.parse_args()
    
    try:
        run_simulation(
            n_rounds=args.rounds,
            n_agents=args.agents,
            n_users=args.users,
            recommender_strategy=args.recommender,
            use_mock_llm=args.mock,
            model_name=args.model,
            log_file=args.logfile
        )
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()