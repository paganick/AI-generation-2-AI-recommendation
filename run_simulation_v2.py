#!/usr/bin/env python3
"""
Improved Simulation Runner - Clear Workflow
Everything is tracked, saved, and ready for analysis.
"""

import numpy as np
import argparse
from typing import List

from simulation_framework_v2 import (
    Agent, SimulatedUser, ContentItem,
    ContentGenerator, EnhancedDataTracker, add_sentiment_scores, save_results
)
from llm_recommender import LLMRecommender, EngagementModel

# Import LLM backend
import sys
sys.path.insert(0, './')
try:
    from llm_backend import get_llm_backend
except:
    print("Warning: Could not import llm_backend, will need to specify path")
    from simulation_framework_v2 import get_llm_backend  # Fallback


def create_agents(n_agents: int = 6) -> List[Agent]:
    """
    Create AI agents with clear personas and objectives.
    Each agent has a distinct viewpoint and style.
    """
    
    agent_configs = [
        {
            "persona": "Progressive activist",
            "objective": "Advocate for social and environmental causes",
            "temperature": 0.8,
            "response_style": "passionate",
            "preferred_topics": ["climate change", "social justice"]
        },
        {
            "persona": "Tech entrepreneur",
            "objective": "Discuss innovation and market opportunities",
            "temperature": 0.7,
            "response_style": "optimistic",
            "preferred_topics": ["artificial intelligence", "technology"]
        },
        {
            "persona": "Academic researcher",
            "objective": "Share evidence-based insights",
            "temperature": 0.5,
            "response_style": "analytical",
            "preferred_topics": ["artificial intelligence", "healthcare"]
        },
        {
            "persona": "Skeptical journalist",
            "objective": "Question claims and seek accountability",
            "temperature": 0.6,
            "response_style": "critical",
            "preferred_topics": ["politics", "healthcare"]
        },
        {
            "persona": "Community organizer",
            "objective": "Build consensus and mobilize action",
            "temperature": 0.7,
            "response_style": "collaborative",
            "preferred_topics": ["climate change", "politics"]
        },
        {
            "persona": "Contrarian thinker",
            "objective": "Challenge mainstream narratives",
            "temperature": 0.8,
            "response_style": "provocative",
            "preferred_topics": ["technology", "politics"]
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
            response_style=config["response_style"],
            preferred_topics=config.get("preferred_topics", [])
        )
        agents.append(agent)
        print(f"  ✓ Created {agent.persona} (agent_{i})")
    
    return agents


def create_users(n_users: int = 10) -> List[SimulatedUser]:
    """
    Create simulated users with diverse interests and opinions.
    Each user has:
    - Interests (what content they want to see)
    - Opinion (their current stance: -1 = negative, +1 = positive)
    - Tolerance (how open to different opinions)
    """
    
    interest_pools = [
        ["climate change", "environment", "sustainability"],
        ["technology", "AI", "innovation"],
        ["healthcare", "medicine", "public health"],
        ["politics", "policy", "governance"],
    ]
    
    users = []
    for i in range(n_users):
        # Randomly select interests
        interests = interest_pools[i % len(interest_pools)]
        
        # Random initial opinion
        opinion = np.random.uniform(-0.5, 0.5, size=1)
        
        # Random tolerance
        tolerance = np.random.uniform(0.2, 0.5)
        
        user = SimulatedUser(
            id=f"user_{i}",
            interests=interests,
            opinion_vector=opinion,
            tolerance=tolerance
        )
        users.append(user)
        print(f"  ✓ Created user_{i} (interests: {interests[0]}, opinion: {opinion[0]:.2f})")
    
    return users


def run_simulation(
    n_rounds: int = 5,
    n_agents: int = 6,
    n_users: int = 10,
    use_personalization: bool = True,
    use_mock_llm: bool = False,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    output_dir: str = "./simulation_data"
):
    """
    Run the complete simulation.
    
    Workflow per round:
    1. Agents generate content on topics
    2. Content gets sentiment scores
    3. LLM recommender ranks content for each user
    4. Users view content probabilistically, like based on engagement model
    5. Users update opinions based on viewed content
    6. Agents respond to popular content
    7. Everything is saved for analysis
    
    Args:
        n_rounds: Number of simulation rounds
        n_agents: Number of AI content creators
        n_users: Number of simulated users
        use_personalization: Whether recommender uses user history
        use_mock_llm: Use mock LLM for testing (fast, no GPU)
        model_name: Which LLM to use
        output_dir: Where to save all data
    """
    
    print("="*80)
    print("AI CONTENT & RECOMMENDER SYSTEM SIMULATION - LLM-BASED VERSION")
    print("="*80)
    print(f"Configuration:")
    print(f"  Rounds: {n_rounds}")
    print(f"  Agents: {n_agents}")
    print(f"  Users: {n_users}")
    print(f"  Personalization: {'Enabled' if use_personalization else 'Disabled'}")
    print(f"  Model: {'Mock' if use_mock_llm else model_name}")
    print(f"  Output: {output_dir}")
    print("="*80)
    
    # Initialize components
    print("\n[1/5] Initializing system...")
    
    # LLM backend
    print("  → Loading LLM backend...")
    llm = get_llm_backend(use_mock=use_mock_llm, model_name=model_name)
    
    # Data tracker
    print("  → Initializing data tracker...")
    data_tracker = EnhancedDataTracker(output_dir=output_dir)
    
    # Create agents
    print("\n[2/5] Creating agents...")
    agents = create_agents(n_agents)
    
    # Create users
    print("\n[3/5] Creating users...")
    users = create_users(n_users)
    users_dict = {u.id: u for u in users}
    data_tracker.all_users = users_dict
    
    # Initialize simulation components
    print("\n[4/5] Initializing simulation components...")
    generator = ContentGenerator(agents, llm, data_tracker)
    recommender = LLMRecommender(llm, use_personalization=use_personalization)
    engagement_model = EngagementModel()
    
    # Discussion topics
    topics = ["climate change", "artificial intelligence", "healthcare reform"]
    print(f"  → Discussion topics: {', '.join(topics)}")
    
    # Content pool (accumulates over rounds)
    content_pool = []
    
    print("\n[5/5] Running simulation...")
    print("="*80)
    
    # SIMULATION LOOP
    for round_num in range(1, n_rounds + 1):
        print(f"\n{'='*80}")
        print(f"ROUND {round_num}/{n_rounds}")
        print(f"{'='*80}")
        
        # STEP 1: CONTENT GENERATION
        print(f"\n[Round {round_num} - Step 1/5] CONTENT GENERATION")
        print("-" * 60)
        
        # Generate initial posts
        new_posts = generator.generate_initial_posts(
            topics=topics,
            current_round=round_num,
            posts_per_topic=2
        )
        content_pool.extend(new_posts)
        
        # Add sentiment scores
        print(f"\n  → Adding sentiment scores...")
        sentiment_map = add_sentiment_scores(content_pool)
        print(f"    ✓ Analyzed sentiment for {len(content_pool)} items")
        
        # STEP 2: CONTENT RECOMMENDATION (LLM-BASED)
        print(f"\n[Round {round_num} - Step 2/5] LLM-BASED CONTENT RECOMMENDATION")
        print("-" * 60)
        
        # Rank content for each user using LLM
        feed_assignments = {}
        print(f"  → Ranking content for {len(users)} users...")
        
        for user in users:
            ranked_content = recommender.rank_content_for_user(
                content_pool=content_pool,
                user=user,
                k=5  # Top 5 items per user
            )
            feed_assignments[user.id] = ranked_content
            print(f"    • {user.id}: Ranked {len(content_pool)} items → top 5 selected")
        
        # Save feed assignments
        data_tracker.feed_assignments[round_num] = {
            user_id: [item.id for item in feed]
            for user_id, feed in feed_assignments.items()
        }
        
        # Track what content was shown
        data_tracker.round_data['content_shown'][round_num] = {
            user_id: [item.id for item in feed]
            for user_id, feed in feed_assignments.items()
        }
        
        # STEP 3: USER-CONTENT INTERACTION (PROBABILISTIC)
        print(f"\n[Round {round_num} - Step 3/5] USER-CONTENT INTERACTION")
        print("-" * 60)
        
        total_views = 0
        total_likes = 0
        
        for user_id, feed in feed_assignments.items():
            user = users_dict[user_id]
            opinion_before = float(user.opinion_vector[0])
            
            for rank, content_item in enumerate(feed):
                # Compute engagement probability
                engagement_result = engagement_model.simulate_engagement(
                    user=user,
                    content_item=content_item,
                    rank_in_feed=rank,
                    content_pool=content_pool
                )
                
                viewed = engagement_result['viewed']
                liked = engagement_result['liked']
                
                # Update content metrics
                if viewed:
                    content_item.total_views += 1
                    total_views += 1
                
                if liked:
                    content_item.total_likes += 1
                    content_item.engagement_score += 1.0
                    total_likes += 1
                
                # Update user opinion (ONLY if viewed and within tolerance)
                if viewed and content_item.sentiment_score is not None:
                    distance = abs(user.opinion_vector[0] - content_item.sentiment_score)
                    
                    if distance < user.tolerance:
                        # Bounded confidence: user influenced by this content
                        alpha = 0.1  # Learning rate
                        user.opinion_vector[0] = (1 - alpha) * user.opinion_vector[0] + \
                                                alpha * content_item.sentiment_score
                
                # Record interaction
                from simulation_framework_v2 import UserContentInteraction
                interaction = UserContentInteraction(
                    user_id=user_id,
                    content_id=content_item.id,
                    round=round_num,
                    rank_in_feed=rank,
                    viewed=viewed,
                    liked=liked,
                    opinion_before=opinion_before,
                    opinion_after=float(user.opinion_vector[0]),
                    timestamp=round_num * 100.0 + rank
                )
                
                data_tracker.interactions.append(interaction)
                user.exposure_history.append(content_item.id)
                user.engagement_history.append({
                    'content_id': content_item.id,
                    'round': round_num,
                    'liked': liked,
                    'viewed': viewed
                })
        
        print(f"  ✓ Simulated {len(data_tracker.interactions)} interactions")
        print(f"    Views: {total_views}, Likes: {total_likes}")
        
        # Track user opinions
        data_tracker.round_data['user_opinions'][round_num] = {
            user.id: user.opinion_vector.copy()
            for user in users
        }
        
        # STEP 4: RESPONSE GENERATION
        print(f"\n[Round {round_num} - Step 4/5] RESPONSE GENERATION")
        print("-" * 60)
        
        # Agents respond to popular content
        # Collect all content from feeds
        all_feed_items = []
        for feed in feed_assignments.values():
            all_feed_items.extend(feed)
        unique_feed_items = list({item.id: item for item in all_feed_items}.values())
        
        new_responses = generator.generate_responses(
            content_pool=content_pool,
            feed_items=unique_feed_items,
            current_round=round_num,
            n_responses=min(4, n_agents)
        )
        content_pool.extend(new_responses)
        
        # Add sentiment to new responses
        add_sentiment_scores(new_responses)
        
        # STEP 5: ROUND SUMMARY
        print(f"\n[Round {round_num} - Step 5/5] ROUND SUMMARY")
        print("-" * 60)
        
        # Compute statistics
        avg_opinion = np.mean([u.opinion_vector[0] for u in users])
        opinion_variance = np.var([u.opinion_vector[0] for u in users])
        
        total_likes = sum(item.total_likes for item in content_pool)
        total_views = sum(item.total_views for item in content_pool)
        
        print(f"  Content pool size: {len(content_pool)}")
        print(f"  New content: {len(new_posts) + len(new_responses)}")
        print(f"  Total interactions: {len(data_tracker.interactions)}")
        print(f"  Total likes: {total_likes}")
        print(f"  Total views: {total_views}")
        print(f"  Avg user opinion: {avg_opinion:.3f}")
        print(f"  Opinion variance: {opinion_variance:.3f}")
    
    # SAVE ALL DATA
    print("\n" + "="*80)
    print("SIMULATION COMPLETE - SAVING DATA")
    print("="*80)
    
    data_tracker.save_all_data()
    save_results(data_tracker)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total content generated: {len(content_pool)}")
    print(f"Total user-content interactions: {len(data_tracker.interactions)}")
    print(f"Total likes: {sum(item.total_likes for item in content_pool)}")
    print(f"Total views: {sum(item.total_views for item in content_pool)}")
    print(f"\nFinal user opinions:")
    for user in users[:5]:  # Show first 5
        print(f"  {user.id}: {user.opinion_vector[0]:.3f} (tolerance: {user.tolerance:.2f})")
    if len(users) > 5:
        print(f"  ... and {len(users) - 5} more users")
    
    print("\n" + "="*80)
    print("DATA SAVED TO:")
    print("="*80)
    print(f"  {output_dir}/all_content.csv")
    print(f"  {output_dir}/all_interactions.csv")
    print(f"  {output_dir}/agents.csv")
    print(f"  {output_dir}/users.csv")
    print(f"  {output_dir}/feed_assignments.json")
    print(f"  {output_dir}/round_data.json")
    print("="*80)
    
    return content_pool, data_tracker, users


def main():
    parser = argparse.ArgumentParser(
        description="AI Content & Recommender Simulation - LLM-Based Version"
    )
    parser.add_argument("--rounds", type=int, default=5, 
                       help="Number of simulation rounds")
    parser.add_argument("--agents", type=int, default=6, 
                       help="Number of AI agents (content creators)")
    parser.add_argument("--users", type=int, default=10, 
                       help="Number of simulated users")
    parser.add_argument("--no-personalization", action="store_true",
                       help="Disable personalization (no user history in recommendations)")
    parser.add_argument("--mock", action="store_true",
                       help="Use mock LLM (fast, no GPU)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="LLM model name")
    parser.add_argument("--output", type=str, default="./simulation_data",
                       help="Output directory for data")
    
    args = parser.parse_args()
    
    try:
        run_simulation(
            n_rounds=args.rounds,
            n_agents=args.agents,
            n_users=args.users,
            use_personalization=not args.no_personalization,
            use_mock_llm=args.mock,
            model_name=args.model,
            output_dir=args.output
        )
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()