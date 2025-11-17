"""
Enhanced Simulation Framework - Improved Version
Clear philosophy: Track everything, save everything, analyze everything.
"""

import numpy as np
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd


@dataclass
class ContentItem:
    """A piece of content (post/tweet) in the system."""
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
    llm_architecture: Optional[str] = None  # NEW: Track which LLM generated this content
    llm_model: Optional[str] = None  # NEW: Track specific model used

    def to_dict(self):
        d = asdict(self)
        return d


@dataclass
class Agent:
    """AI agent that generates content."""
    id: str
    persona: str
    objective: str
    temperature: float = 0.7
    response_style: str = "balanced"
    preferred_topics: List[str] = field(default_factory=list)
    content_history: List[str] = field(default_factory=list)
    llm_backend_id: Optional[str] = None  # NEW: Which LLM backend this agent uses

    def to_dict(self):
        return asdict(self)


@dataclass
class SimulatedUser:
    """Human user consuming content."""
    id: str
    interests: List[str]
    opinion_vector: np.ndarray
    tolerance: float = 0.3
    exposure_history: List[str] = field(default_factory=list)
    engagement_history: List[Dict] = field(default_factory=list)  # NEW: Track each interaction
    
    def to_dict(self):
        d = asdict(self)
        d['opinion_vector'] = self.opinion_vector.tolist()
        return d


@dataclass
class UserContentInteraction:
    """Records a specific user-content interaction."""
    user_id: str
    content_id: str
    round: int
    rank_in_feed: int  # Position in user's feed (0-indexed)
    viewed: bool
    liked: bool  # Simulated based on opinion alignment
    opinion_before: float
    opinion_after: float
    timestamp: float


class EnhancedDataTracker:
    """
    Tracks ALL data for analysis:
    - Individual content items
    - User feeds per round
    - User-content interactions
    - Agent generation patterns
    """
    
    def __init__(self, output_dir: str = "./simulation_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data stores
        self.all_content: List[ContentItem] = []
        self.all_agents: Dict[str, Agent] = {}
        self.all_users: Dict[str, SimulatedUser] = {}
        self.interactions: List[UserContentInteraction] = []
        self.feed_assignments: Dict[int, Dict[str, List[str]]] = {}  # round -> user_id -> [content_ids]
        
        # Round-by-round tracking
        self.round_data = {
            'content_generated': {},  # round -> [content_items]
            'content_shown': {},       # round -> user_id -> [content_ids]
            'user_opinions': {},       # round -> user_id -> opinion
        }
    
    def save_all_data(self):
        """Save everything to files for analysis."""
        
        # 1. Save all content as CSV
        content_df = pd.DataFrame([c.to_dict() for c in self.all_content])
        content_df.to_csv(self.output_dir / "all_content.csv", index=False)
        
        # 2. Save all interactions as CSV
        interaction_df = pd.DataFrame([asdict(i) for i in self.interactions])
        if not interaction_df.empty:
            interaction_df.to_csv(self.output_dir / "all_interactions.csv", index=False)
        
        # 3. Save agents info
        agents_df = pd.DataFrame([a.to_dict() for a in self.all_agents.values()])
        agents_df.to_csv(self.output_dir / "agents.csv", index=False)
        
        # 4. Save users info
        users_data = []
        for user in self.all_users.values():
            user_dict = user.to_dict()
            user_dict['final_opinion'] = user.opinion_vector[0]
            user_dict['n_exposures'] = len(user.exposure_history)
            users_data.append(user_dict)
        users_df = pd.DataFrame(users_data)
        users_df.to_csv(self.output_dir / "users.csv", index=False)
        
        # 5. Save feed assignments by round
        with open(self.output_dir / "feed_assignments.json", 'w') as f:
            json.dump(self.feed_assignments, f, indent=2)
        
        # 6. Save round-by-round data
        with open(self.output_dir / "round_data.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for round_num, user_opinions in self.round_data['user_opinions'].items():
                serializable_data[round_num] = {
                    uid: float(op[0]) if isinstance(op, np.ndarray) else float(op)
                    for uid, op in user_opinions.items()
                }
            json.dump({
                'content_generated': self.round_data['content_generated'],
                'content_shown': self.round_data['content_shown'],
                'user_opinions': serializable_data
            }, f, indent=2)
        
        print(f"\n✓ All data saved to {self.output_dir}/")
        print(f"  - all_content.csv ({len(self.all_content)} items)")
        print(f"  - all_interactions.csv ({len(self.interactions)} interactions)")
        print(f"  - agents.csv ({len(self.all_agents)} agents)")
        print(f"  - users.csv ({len(self.all_users)} users)")
        print(f"  - feed_assignments.json")
        print(f"  - round_data.json")


class ContentGenerator:
    """Manages AI agents that generate content."""

    def __init__(self, agents: List[Agent], llm_backend, data_tracker: EnhancedDataTracker):
        """
        Initialize ContentGenerator.

        Args:
            agents: List of Agent objects
            llm_backend: Either a single LLM backend OR a MultiLLMManager
            data_tracker: EnhancedDataTracker instance
        """
        self.agents = {agent.id: agent for agent in agents}
        self.llm = llm_backend
        self.generation_count = 0
        self.data_tracker = data_tracker

        # Check if we're using MultiLLMManager
        self.is_multi_llm = hasattr(llm_backend, 'get_backend_for_agent')

        # Store agents in tracker
        self.data_tracker.all_agents = self.agents

    def _get_backend_for_agent(self, agent_id: str):
        """Get the appropriate LLM backend for a given agent."""
        if self.is_multi_llm:
            return self.llm.get_backend_for_agent(agent_id)
        else:
            return self.llm

    def generate_initial_posts(self, topics: List[str], current_round: int,
                               posts_per_topic: int = 2) -> List[ContentItem]:
        """
        Generate initial posts on given topics.
        Each agent is prompted with their persona and a topic.
        """
        content_items = []
        timestamp = current_round * 100.0  # Simple timestamp
        
        print(f"\n→ Generating initial posts for round {current_round}")
        
        for topic in topics:
            # Select agents for this topic
            selected_agents = list(self.agents.keys())[:posts_per_topic]
            
            for agent_id in selected_agents:
                agent = self.agents[agent_id]

                # Get the appropriate backend for this agent
                backend = self._get_backend_for_agent(agent_id)

                # Create generation prompt
                prompt = self._create_generation_prompt(agent, topic)
                print(f"  Agent {agent_id} ({agent.persona}) posting about '{topic}'...")

                # Generate content
                text = backend.generate(prompt, temperature=agent.temperature)

                # Create content item
                content_id = f"c_{current_round}_{self.generation_count}"
                item = ContentItem(
                    id=content_id,
                    author_id=agent_id,
                    text=text,
                    timestamp=timestamp,
                    round_created=current_round,
                    topic=topic,
                    llm_architecture=backend.architecture_type,
                    llm_model=backend.model_name
                )

                content_items.append(item)
                agent.content_history.append(content_id)
                self.generation_count += 1
                timestamp += 1.0
        
        # Save to tracker
        self.data_tracker.all_content.extend(content_items)
        if current_round not in self.data_tracker.round_data['content_generated']:
            self.data_tracker.round_data['content_generated'][current_round] = []
        self.data_tracker.round_data['content_generated'][current_round].extend(
            [c.id for c in content_items]
        )
        
        print(f"  ✓ Generated {len(content_items)} posts")
        return content_items
    
    def generate_responses(self, content_pool: List[ContentItem], 
                          feed_items: List[ContentItem], 
                          current_round: int,
                          n_responses: int = 5) -> List[ContentItem]:
        """
        Generate responses to popular content.
        Agents respond to content they find in the feed.
        """
        responses = []
        timestamp = current_round * 100.0 + 50.0
        
        print(f"\n→ Generating {n_responses} responses to feed content")
        
        if not feed_items:
            print("  (No feed items to respond to)")
            return responses
        
        # Select random agents to respond
        responding_agents = np.random.choice(
            list(self.agents.keys()), 
            size=min(n_responses, len(self.agents)), 
            replace=False
        )
        
        for agent_id in responding_agents:
            agent = self.agents[agent_id]

            # Get the appropriate backend for this agent
            backend = self._get_backend_for_agent(agent_id)

            # Select a random item from feed to respond to
            target_item = np.random.choice(feed_items)

            prompt = self._create_response_prompt(agent, target_item)
            print(f"  Agent {agent_id} responding to {target_item.id[:10]}...")

            text = backend.generate(prompt, temperature=agent.temperature)

            content_id = f"c_{current_round}_{self.generation_count}"
            response = ContentItem(
                id=content_id,
                author_id=agent_id,
                text=text,
                timestamp=timestamp,
                round_created=current_round,
                parent_id=target_item.id,
                topic=target_item.topic,
                llm_architecture=backend.architecture_type,
                llm_model=backend.model_name
            )

            responses.append(response)
            agent.content_history.append(content_id)
            self.generation_count += 1
            timestamp += 1.0
        
        # Save to tracker
        self.data_tracker.all_content.extend(responses)
        self.data_tracker.round_data['content_generated'][current_round].extend(
            [r.id for r in responses]
        )
        
        print(f"  ✓ Generated {len(responses)} responses")
        return responses
    
    def _create_generation_prompt(self, agent: Agent, topic: str) -> str:
        """Create a prompt for initial post generation."""
        return f"""You are a social media user with the following characteristics:
Persona: {agent.persona}
Objective: {agent.objective}
Style: {agent.response_style}

Write a social media post (2-3 sentences) about: {topic}

Post:"""
    
    def _create_response_prompt(self, agent: Agent, target: ContentItem) -> str:
        """Create a prompt for responding to existing content."""
        return f"""You are a social media user with the following characteristics:
Persona: {agent.persona}
Objective: {agent.objective}
Style: {agent.response_style}

Someone posted: "{target.text}"

Write a brief response (2-3 sentences):

Response:"""
    
    def _generate_content_id(self, current_round: int) -> str:
        """Generate a unique content ID."""
        return f"c_{current_round}_{self.generation_count}"


class RecommenderSystem:
    """
    Recommender system that ranks and filters content for users.
    Currently uses simple relevance-based ranking.
    """
    
    def __init__(self, strategy: str = "relevance"):
        self.strategy = strategy
    
    def rank_content_for_all_users(self, content_pool: List[ContentItem], 
                                   users: List[SimulatedUser], 
                                   k: int = 10) -> Dict[str, List[ContentItem]]:
        """
        Rank content for ALL users.
        Returns: {user_id: [top k content items]}
        """
        print(f"\n→ Ranking content for {len(users)} users (strategy: {self.strategy})")
        
        feeds = {}
        for user in users:
            feeds[user.id] = self.rank_content(content_pool, user, k)
        
        print(f"  ✓ Generated {len(feeds)} personalized feeds")
        return feeds
    
    def rank_content(self, content_pool: List[ContentItem], 
                     user: SimulatedUser, k: int = 10) -> List[ContentItem]:
        """Rank content for a specific user and return top-k items."""
        
        if self.strategy == "relevance":
            return self._rank_by_relevance(content_pool, user, k)
        elif self.strategy == "diversity":
            return self._rank_by_diversity(content_pool, user, k)
        elif self.strategy == "engagement":
            return self._rank_by_engagement(content_pool, user, k)
        else:
            # Random baseline
            return list(np.random.choice(content_pool, size=min(k, len(content_pool)), replace=False))
    
    def _rank_by_relevance(self, content_pool: List[ContentItem], 
                           user: SimulatedUser, k: int) -> List[ContentItem]:
        """
        Simple relevance-based ranking using keyword matching.
        
        Score = (keyword_matches) + (engagement_score * 0.1)
        """
        scores = []
        for item in content_pool:
            score = 0.0
            # Check if user interests appear in content
            for interest in user.interests:
                if interest.lower() in item.text.lower():
                    score += 1.0
            # Add engagement score (popular content gets boost)
            score += item.engagement_score * 0.1
            scores.append(score)
        
        # Sort by score and return top-k
        ranked_indices = np.argsort(scores)[::-1][:k]
        return [content_pool[i] for i in ranked_indices]
    
    def _rank_by_diversity(self, content_pool: List[ContentItem], 
                           user: SimulatedUser, k: int) -> List[ContentItem]:
        """Diversity-aware ranking."""
        # (Same as before - omitted for brevity)
        return self._rank_by_relevance(content_pool, user, k)
    
    def _rank_by_engagement(self, content_pool: List[ContentItem],
                            user: SimulatedUser, k: int) -> List[ContentItem]:
        """Rank purely by engagement (popularity)."""
        sorted_content = sorted(content_pool, key=lambda x: x.engagement_score, reverse=True)
        return sorted_content[:k]


class UserContentInteractionSimulator:
    """
    Simulates how users interact with content:
    - Viewing
    - Liking
    - Opinion updates
    """
    
    def __init__(self, data_tracker: EnhancedDataTracker):
        self.data_tracker = data_tracker
    
    def simulate_interactions(self, feed_assignments: Dict[str, List[ContentItem]],
                            users: Dict[str, SimulatedUser],
                            current_round: int):
        """
        Simulate user interactions with their feeds.
        
        For each user:
          - Views content in feed
          - Likes content based on opinion alignment
          - Updates opinion based on viewed content
        """
        print(f"\n→ Simulating user-content interactions")
        
        for user_id, feed in feed_assignments.items():
            user = users[user_id]
            opinion_before = float(user.opinion_vector[0])
            
            for rank, content_item in enumerate(feed):
                # User views the content
                viewed = True
                content_item.total_views += 1
                
                # User likes if opinion aligns with content sentiment
                if content_item.sentiment_score is not None:
                    opinion_distance = abs(opinion_before - content_item.sentiment_score)
                    like_probability = max(0, 1.0 - opinion_distance)  # Closer opinion = higher like prob
                    liked = np.random.random() < like_probability
                    
                    if liked:
                        content_item.total_likes += 1
                        content_item.engagement_score += 1.0
                else:
                    liked = False
                
                # Update user opinion (bounded confidence)
                if content_item.sentiment_score is not None:
                    distance = abs(user.opinion_vector[0] - content_item.sentiment_score)
                    
                    if distance < user.tolerance:
                        # User is influenced by this content
                        alpha = 0.1  # Learning rate
                        user.opinion_vector[0] = (1 - alpha) * user.opinion_vector[0] + \
                                                alpha * content_item.sentiment_score
                
                # Record interaction
                interaction = UserContentInteraction(
                    user_id=user_id,
                    content_id=content_item.id,
                    round=current_round,
                    rank_in_feed=rank,
                    viewed=viewed,
                    liked=liked,
                    opinion_before=opinion_before,
                    opinion_after=float(user.opinion_vector[0]),
                    timestamp=current_round * 100.0 + rank
                )
                
                self.data_tracker.interactions.append(interaction)
                user.exposure_history.append(content_item.id)
                user.engagement_history.append({
                    'content_id': content_item.id,
                    'round': current_round,
                    'liked': liked
                })
        
        print(f"  ✓ Simulated {len(self.data_tracker.interactions)} interactions")


def add_sentiment_scores(content_items: List[ContentItem]) -> Dict[str, float]:
    """Add simple sentiment scores to content (keyword-based)."""
    sentiment_map = {}
    
    positive_words = ['good', 'great', 'excellent', 'important', 'positive', 
                     'improve', 'better', 'solution', 'opportunity', 'success']
    negative_words = ['bad', 'terrible', 'wrong', 'crisis', 'problem', 
                     'danger', 'risk', 'threat', 'concern', 'failure']
    
    for item in content_items:
        text_lower = item.text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text_lower.split())
        sentiment = (pos_count - neg_count) / max(total_words, 1)
        sentiment = np.clip(sentiment, -1, 1)
        
        item.sentiment_score = sentiment
        sentiment_map[item.id] = sentiment
    
    return sentiment_map


# Save function updated
def save_results(data_tracker: EnhancedDataTracker, 
                output_path: str = "./simulation_results.json"):
    """Save simulation results summary."""
    results = {
        'n_content': len(data_tracker.all_content),
        'n_agents': len(data_tracker.all_agents),
        'n_users': len(data_tracker.all_users),
        'n_interactions': len(data_tracker.interactions),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Summary saved to {output_path}")