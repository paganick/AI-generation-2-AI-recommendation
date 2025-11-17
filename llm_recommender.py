"""
LLM-Based Recommender System with Personalization
Core improvement: Recommender uses LLM to rank content, not simple keyword matching.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import json


class LLMRecommender:
    """
    Pure LLM-based recommender system.
    Uses an LLM to rank content for users, with optional personalization.
    """
    
    def __init__(self, llm_backend, use_personalization: bool = True):
        """
        Args:
            llm_backend: LLM for ranking content
            use_personalization: Whether to include user history in prompts
        """
        self.llm = llm_backend
        self.use_personalization = use_personalization
        self.ranking_history = []
    
    def rank_content_for_user(self, 
                              content_pool: List,
                              user,
                              k: int = 10) -> List:
        """
        Rank content for a specific user using LLM.
        
        The LLM sees:
        - User interests
        - User interaction history (if personalization enabled)
        - All available content
        
        Returns: Top k ranked content items
        """
        
        # Build the ranking prompt
        prompt = self._create_ranking_prompt(content_pool, user, k)
        
        # Get LLM ranking
        response = self.llm.generate(prompt, temperature=0.3)
        
        # Parse the ranking
        ranked_items = self._parse_ranking(response, content_pool, k)
        
        # Store for analysis
        self.ranking_history.append({
            'user_id': user.id,
            'n_items': len(content_pool),
            'prompt_length': len(prompt),
            'response': response
        })
        
        return ranked_items
    
    def _create_ranking_prompt(self, content_pool: List, user, k: int) -> str:
        """
        Create the LLM prompt for ranking content.
        
        Prompt structure:
        1. User profile (interests)
        2. User history (if personalization enabled)
        3. Available content
        4. Task: rank content
        """
        
        # Limit content pool size for prompt (LLM context limits)
        max_items = 20
        if len(content_pool) > max_items:
            # Sample recent + popular content
            recent_items = sorted(content_pool, key=lambda x: x.timestamp, reverse=True)[:max_items//2]
            popular_items = sorted(content_pool, key=lambda x: x.engagement_score, reverse=True)[:max_items//2]
            sampled_pool = list(set(recent_items + popular_items))[:max_items]
        else:
            sampled_pool = content_pool
        
        # Start building prompt
        prompt_parts = []
        
        # 1. User profile
        prompt_parts.append("# USER PROFILE")
        prompt_parts.append(f"User interests: {', '.join(user.interests)}")
        
        # 2. Personalization (optional)
        if self.use_personalization and len(user.engagement_history) > 0:
            prompt_parts.append("\n# USER INTERACTION HISTORY")
            
            # Get recently liked content
            recent_likes = [
                h for h in user.engagement_history[-10:]  # Last 10 interactions
                if h.get('liked', False)
            ]
            
            if recent_likes:
                prompt_parts.append("Recently liked content:")
                for i, interaction in enumerate(recent_likes[-5:], 1):  # Last 5 likes
                    # Find the content item
                    content_item = next(
                        (item for item in content_pool if item.id == interaction['content_id']),
                        None
                    )
                    if content_item:
                        prompt_parts.append(f"  {i}. \"{content_item.text[:80]}...\"")
            else:
                prompt_parts.append("(User hasn't liked any content yet)")
        
        # 3. Available content
        prompt_parts.append("\n# CONTENT TO RANK")
        prompt_parts.append(f"Rank the following {len(sampled_pool)} posts for this user.\n")
        
        for i, item in enumerate(sampled_pool, 1):
            # Include metadata that might be useful
            metadata = []
            if item.topic:
                metadata.append(f"topic: {item.topic}")
            if item.engagement_score > 0:
                metadata.append(f"engagement: {item.engagement_score:.1f}")
            
            metadata_str = f" [{', '.join(metadata)}]" if metadata else ""
            
            prompt_parts.append(f"{i}. \"{item.text}\"{metadata_str}")
        
        # 4. Task
        prompt_parts.append("\n# TASK")
        prompt_parts.append(f"Rank these posts from most to least relevant for this user.")
        prompt_parts.append(f"Return ONLY a comma-separated list of numbers (e.g., \"3,1,5,2,4\").")
        prompt_parts.append(f"Return the top {k} most relevant posts.\n")
        prompt_parts.append("Ranking:")
        
        return "\n".join(prompt_parts)
    
    def _parse_ranking(self, llm_response: str, content_pool: List, k: int) -> List:
        """
        Parse LLM response to extract ranked content.
        
        Expected format: "3,1,5,2,4" (1-indexed positions)
        """
        import re
        
        # Extract numbers from response
        numbers = re.findall(r'\d+', llm_response)
        
        try:
            # Convert to 0-indexed positions
            ranking_indices = [int(n) - 1 for n in numbers]
            
            # Validate indices
            valid_indices = [idx for idx in ranking_indices 
                           if 0 <= idx < len(content_pool)]
            
            # Return ranked content
            ranked_content = [content_pool[idx] for idx in valid_indices[:k]]
            
            # If we got fewer than k items, pad with remaining content
            if len(ranked_content) < k:
                remaining = [item for item in content_pool 
                           if item not in ranked_content]
                ranked_content.extend(remaining[:k - len(ranked_content)])
            
            return ranked_content
        
        except Exception as e:
            # Fallback: return first k items if parsing fails
            print(f"Warning: Failed to parse LLM ranking: {e}")
            return content_pool[:k]


class EngagementModel:
    """
    Models the probability that a user engages with content.
    
    Based on:
    - Cosine similarity with previously liked content
    - Topic matching with user interests
    - Content popularity/virality
    - Position in feed (top items more likely)
    """
    
    def __init__(self):
        self.engagement_history = []
    
    def compute_engagement_probability(self,
                                      user,
                                      content_item,
                                      rank_in_feed: int,
                                      content_pool: List) -> Dict[str, float]:
        """
        Compute probability that user engages with (likes) this content.
        
        Returns:
            Dict with:
            - 'like_prob': Overall probability of liking
            - 'view_prob': Probability of viewing (position-dependent)
            - 'components': Breakdown of probability components
        """
        
        components = {}
        
        # 1. Topic matching (0 to 1)
        components['topic_match'] = self._compute_topic_match(user, content_item)
        
        # 2. Historical similarity (0 to 1)
        components['history_similarity'] = self._compute_history_similarity(
            user, content_item, content_pool
        )
        
        # 3. Popularity/virality (0 to 1)
        components['popularity'] = self._compute_popularity_score(
            content_item, content_pool
        )
        
        # 4. Position bias (0 to 1)
        components['position_bias'] = self._compute_position_bias(rank_in_feed)
        
        # 5. Opinion alignment (0 to 1)
        components['opinion_alignment'] = self._compute_opinion_alignment(
            user, content_item
        )
        
        # Combine components with weights
        weights = {
            'topic_match': 0.25,
            'history_similarity': 0.25,
            'popularity': 0.15,
            'position_bias': 0.10,
            'opinion_alignment': 0.25
        }
        
        # Weighted sum
        like_prob = sum(components[k] * weights[k] for k in components)
        
        # View probability (mostly position-dependent)
        view_prob = 0.9 * components['position_bias'] + 0.1
        
        return {
            'like_prob': like_prob,
            'view_prob': view_prob,
            'components': components
        }
    
    def _compute_topic_match(self, user, content_item) -> float:
        """
        Compute how well content topic matches user interests.
        
        Returns: 0 to 1 (1 = perfect match)
        """
        if not content_item.topic:
            return 0.5  # Neutral if no topic
        
        # Simple keyword matching
        matches = 0
        for interest in user.interests:
            if interest.lower() in content_item.topic.lower():
                matches += 1
            # Also check if interest is in the text
            if interest.lower() in content_item.text.lower():
                matches += 0.5
        
        # Normalize by number of interests
        return min(1.0, matches / max(len(user.interests), 1))
    
    def _compute_history_similarity(self, user, content_item, content_pool) -> float:
        """
        Compute cosine similarity with previously liked content.
        
        For simplicity, use text-based similarity (could use embeddings).
        Returns: 0 to 1 (1 = very similar to liked content)
        """
        if not user.engagement_history:
            return 0.5  # Neutral if no history
        
        # Get liked content
        liked_content_ids = [
            h['content_id'] for h in user.engagement_history 
            if h.get('liked', False)
        ]
        
        if not liked_content_ids:
            return 0.5  # Neutral if nothing liked yet
        
        # Find liked content items
        liked_items = [
            item for item in content_pool 
            if item.id in liked_content_ids
        ]
        
        if not liked_items:
            return 0.5
        
        # Simple text similarity (word overlap)
        current_words = set(content_item.text.lower().split())
        
        similarities = []
        for liked_item in liked_items[-5:]:  # Last 5 liked items
            liked_words = set(liked_item.text.lower().split())
            
            # Jaccard similarity
            intersection = len(current_words & liked_words)
            union = len(current_words | liked_words)
            
            if union > 0:
                similarities.append(intersection / union)
        
        if similarities:
            # Average similarity with liked content
            return np.mean(similarities)
        else:
            return 0.5
    
    def _compute_popularity_score(self, content_item, content_pool) -> float:
        """
        Compute virality/popularity score.
        
        Based on engagement relative to other content.
        Returns: 0 to 1 (1 = most viral)
        """
        if not content_pool or len(content_pool) == 1:
            return 0.5
        
        # Get engagement scores
        all_engagement = [item.engagement_score for item in content_pool]
        
        if max(all_engagement) == 0:
            return 0.5  # No engagement yet
        
        # Percentile rank
        percentile = sum(1 for e in all_engagement if e < content_item.engagement_score) / len(all_engagement)
        
        return percentile
    
    def _compute_position_bias(self, rank_in_feed: int) -> float:
        """
        Position bias: users more likely to engage with top items.
        
        Returns: 0 to 1 (1 = top position)
        """
        # Exponential decay with position
        # Position 0: 1.0
        # Position 5: 0.5
        # Position 10: 0.25
        return np.exp(-0.15 * rank_in_feed)
    
    def _compute_opinion_alignment(self, user, content_item) -> float:
        """
        Compute how well content aligns with user opinion.
        
        Returns: 0 to 1 (1 = perfect alignment)
        """
        if content_item.sentiment_score is None:
            return 0.5
        
        # Distance between user opinion and content sentiment
        distance = abs(user.opinion_vector[0] - content_item.sentiment_score)
        
        # Convert to similarity (0 distance = 1 similarity)
        # Distance of 2.0 (max) → similarity 0
        similarity = 1.0 - (distance / 2.0)
        
        return max(0.0, similarity)
    
    def simulate_engagement(self, user, content_item, rank_in_feed: int,
                           content_pool: List) -> Dict[str, bool]:
        """
        Simulate whether user views and likes this content.
        
        Returns:
            Dict with 'viewed' and 'liked' boolean values
        """
        probs = self.compute_engagement_probability(
            user, content_item, rank_in_feed, content_pool
        )
        
        # Simulate viewing (probabilistic based on position)
        viewed = np.random.random() < probs['view_prob']
        
        # Simulate liking (only if viewed)
        liked = False
        if viewed:
            liked = np.random.random() < probs['like_prob']
        
        # Store engagement
        self.engagement_history.append({
            'user_id': user.id,
            'content_id': content_item.id,
            'rank': rank_in_feed,
            'viewed': viewed,
            'liked': liked,
            'probs': probs
        })
        
        return {'viewed': viewed, 'liked': liked}


# Example usage
USAGE_EXAMPLE = """
# Create LLM-based recommender
recommender = LLMRecommender(llm_backend, use_personalization=True)

# Rank content for user
ranked_content = recommender.rank_content_for_user(
    content_pool=all_content,
    user=current_user,
    k=10
)

# Create engagement model
engagement_model = EngagementModel()

# For each ranked item
for rank, content_item in enumerate(ranked_content):
    # Compute engagement probability
    probs = engagement_model.compute_engagement_probability(
        user=current_user,
        content_item=content_item,
        rank_in_feed=rank,
        content_pool=all_content
    )
    
    print(f"Content: {content_item.text[:50]}")
    print(f"  Like probability: {probs['like_prob']:.2%}")
    print(f"  Components:")
    print(f"    Topic match: {probs['components']['topic_match']:.2f}")
    print(f"    History similarity: {probs['components']['history_similarity']:.2f}")
    print(f"    Popularity: {probs['components']['popularity']:.2f}")
    print(f"    Position bias: {probs['components']['position_bias']:.2f}")
    print(f"    Opinion alignment: {probs['components']['opinion_alignment']:.2f}")
    
    # Simulate engagement
    result = engagement_model.simulate_engagement(
        user=current_user,
        content_item=content_item,
        rank_in_feed=rank,
        content_pool=all_content
    )
    
    print(f"  Result: viewed={result['viewed']}, liked={result['liked']}")
"""