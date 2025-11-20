"""
Comprehensive Post-Processing Analysis Suite for Multi-LLM Scenarios

Analyzes:
- Content Diversity: Topic distribution, semantic similarity, novelty measures
- Promotion Patterns: Which content types gain visibility, filter bubble formation
- Sentiment Dynamics: Temporal evolution of sentiment scores, extremity trends
- Engagement Patterns: Network formation, virality cascades, echo chamber emergence
- Polarization Indices: Opinion clustering, between-group interaction frequency
- Architecture Effects: Performance differences across model combinations
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from scipy import stats
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')


def sanitize_for_json(obj):
    """
    Recursively sanitize data for JSON serialization.
    Converts tuple keys to strings and handles numpy/pandas types.
    """
    if isinstance(obj, dict):
        return {
            str(k) if isinstance(k, tuple) else k: sanitize_for_json(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


# Optional: sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")


class ScenarioAnalyzer:
    """Comprehensive analyzer for simulation scenarios."""

    def __init__(self, output_dir: str):
        """
        Initialize analyzer.

        Args:
            output_dir: Directory containing simulation outputs
        """
        self.output_dir = Path(output_dir)
        self.content_df = None
        self.interactions_df = None
        self.agents_df = None
        self.users_df = None
        self.feed_assignments = None
        self.round_data = None

        # Load embedding model if available
        if EMBEDDINGS_AVAILABLE:
            print("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedding_model = None

        self.load_data()

    def load_data(self):
        """Load all simulation data."""
        print(f"Loading data from {self.output_dir}...")

        try:
            self.content_df = pd.read_csv(self.output_dir / "all_content.csv")
            self.interactions_df = pd.read_csv(self.output_dir / "all_interactions.csv")
            self.agents_df = pd.read_csv(self.output_dir / "agents.csv")
            self.users_df = pd.read_csv(self.output_dir / "users.csv")

            with open(self.output_dir / "feed_assignments.json", 'r') as f:
                self.feed_assignments = json.load(f)

            with open(self.output_dir / "round_data.json", 'r') as f:
                self.round_data = json.load(f)

            print(f"✓ Loaded {len(self.content_df)} content items, "
                  f"{len(self.interactions_df)} interactions, "
                  f"{len(self.agents_df)} agents, {len(self.users_df)} users")

        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise

    # =================================================================
    # 1. CONTENT DIVERSITY ANALYSIS
    # =================================================================

    def analyze_content_diversity(self) -> Dict:
        """
        Analyze content diversity metrics.

        Returns:
            Dictionary containing diversity metrics
        """
        print("\n" + "="*70)
        print("CONTENT DIVERSITY ANALYSIS")
        print("="*70)

        results = {}

        # Topic distribution
        topic_counts = self.content_df['topic'].value_counts()
        topic_entropy = stats.entropy(topic_counts / topic_counts.sum())
        results['topic_entropy'] = topic_entropy
        results['topic_distribution'] = topic_counts.to_dict()

        print(f"\nTopic Entropy: {topic_entropy:.3f} (higher = more diverse)")
        print("\nTopic Distribution:")
        for topic, count in topic_counts.items():
            print(f"  {topic}: {count} ({count/len(self.content_df)*100:.1f}%)")

        # Architecture distribution
        arch_counts = self.content_df['llm_architecture'].value_counts()
        results['architecture_distribution'] = arch_counts.to_dict()

        print("\nContent by Architecture:")
        for arch, count in arch_counts.items():
            print(f"  {arch}: {count} ({count/len(self.content_df)*100:.1f}%)")

        # Semantic similarity analysis (if embeddings available)
        if self.embedding_model is not None:
            print("\nComputing semantic similarities...")
            embeddings = self.embedding_model.encode(self.content_df['text'].tolist())

            # Average pairwise cosine similarity
            n = len(embeddings)
            similarities = []
            for i in range(min(n, 100)):  # Sample 100 items for efficiency
                for j in range(i+1, min(n, 100)):
                    sim = 1 - cosine(embeddings[i], embeddings[j])
                    similarities.append(sim)

            avg_similarity = np.mean(similarities)
            results['avg_semantic_similarity'] = float(avg_similarity)
            results['semantic_diversity'] = 1 - avg_similarity

            print(f"Average Semantic Similarity: {avg_similarity:.3f}")
            print(f"Semantic Diversity: {1-avg_similarity:.3f} (higher = more diverse)")

        # Novelty measures (temporal analysis)
        self.content_df_sorted = self.content_df.sort_values('timestamp')
        round_diversity = []

        for round_num in self.content_df['round_created'].unique():
            round_content = self.content_df[self.content_df['round_created'] == round_num]
            if len(round_content) > 1:
                topics_in_round = round_content['topic'].value_counts()
                round_entropy = stats.entropy(topics_in_round / topics_in_round.sum())
                round_diversity.append(round_entropy)

        results['round_diversity_trend'] = round_diversity
        results['avg_round_diversity'] = float(np.mean(round_diversity))

        print(f"\nAverage Round Diversity: {np.mean(round_diversity):.3f}")

        return results

    # =================================================================
    # 2. PROMOTION PATTERNS ANALYSIS
    # =================================================================

    def analyze_promotion_patterns(self) -> Dict:
        """Analyze which content types gain visibility."""
        print("\n" + "="*70)
        print("PROMOTION PATTERNS ANALYSIS")
        print("="*70)

        results = {}

        # Visibility by architecture
        visibility_by_arch = self.content_df.groupby('llm_architecture')['total_views'].agg([
            'sum', 'mean', 'std', 'count'
        ])
        results['visibility_by_architecture'] = visibility_by_arch.to_dict()

        print("\nVisibility by Architecture:")
        print(visibility_by_arch)

        # Engagement rate by architecture
        self.content_df['engagement_rate'] = self.content_df['total_likes'] / (self.content_df['total_views'] + 1)
        engagement_by_arch = self.content_df.groupby('llm_architecture')['engagement_rate'].agg([
            'mean', 'std'
        ])
        results['engagement_by_architecture'] = engagement_by_arch.to_dict()

        print("\nEngagement Rate by Architecture:")
        print(engagement_by_arch)

        # Topic promotion patterns
        topic_visibility = self.content_df.groupby('topic')['total_views'].agg([
            'sum', 'mean'
        ]).sort_values('sum', ascending=False)
        results['topic_visibility'] = topic_visibility.to_dict()

        print("\nTopic Visibility (sorted by total views):")
        print(topic_visibility)

        # Filter bubble analysis - measure concentration of content exposure
        user_topic_exposure = defaultdict(Counter)

        for round_str, feeds in self.feed_assignments.items():
            for user_id, content_ids in feeds.items():
                for content_id in content_ids:
                    content_row = self.content_df[self.content_df['id'] == content_id]
                    if not content_row.empty:
                        topic = content_row.iloc[0]['topic']
                        user_topic_exposure[user_id][topic] += 1

        # Calculate Herfindahl-Hirschman Index for each user (concentration measure)
        filter_bubble_scores = {}
        for user_id, topic_counts in user_topic_exposure.items():
            total = sum(topic_counts.values())
            if total > 0:
                proportions = [count/total for count in topic_counts.values()]
                hhi = sum(p**2 for p in proportions)
                filter_bubble_scores[user_id] = hhi

        results['filter_bubble_scores'] = filter_bubble_scores
        results['avg_filter_bubble_score'] = float(np.mean(list(filter_bubble_scores.values())))

        print(f"\nFilter Bubble Analysis (HHI):")
        print(f"  Average HHI: {results['avg_filter_bubble_score']:.3f}")
        print(f"  (0.33 = uniform, 1.0 = single topic)")

        return results

    # =================================================================
    # 3. SENTIMENT DYNAMICS ANALYSIS
    # =================================================================

    def analyze_sentiment_dynamics(self) -> Dict:
        """Analyze temporal evolution of sentiment."""
        print("\n" + "="*70)
        print("SENTIMENT DYNAMICS ANALYSIS")
        print("="*70)

        results = {}

        # Temporal sentiment evolution
        sentiment_by_round = self.content_df.groupby('round_created')['sentiment_score'].agg([
            'mean', 'std', 'min', 'max'
        ])
        results['sentiment_by_round'] = sentiment_by_round.to_dict()

        print("\nSentiment Evolution by Round:")
        print(sentiment_by_round)

        # Sentiment extremity trend (absolute values)
        self.content_df['sentiment_extremity'] = np.abs(self.content_df['sentiment_score'])
        extremity_by_round = self.content_df.groupby('round_created')['sentiment_extremity'].mean()
        results['extremity_trend'] = extremity_by_round.tolist()

        print(f"\nSentiment Extremity Trend:")
        for round_num, extremity in extremity_by_round.items():
            print(f"  Round {round_num}: {extremity:.3f}")

        # Sentiment by architecture
        sentiment_by_arch = self.content_df.groupby('llm_architecture')['sentiment_score'].agg([
            'mean', 'std'
        ])
        results['sentiment_by_architecture'] = sentiment_by_arch.to_dict()

        print("\nSentiment by Architecture:")
        print(sentiment_by_arch)

        # Polarization in sentiment (bimodality)
        sentiment_values = self.content_df['sentiment_score'].dropna()
        if len(sentiment_values) > 0:
            skewness = stats.skew(sentiment_values)
            kurtosis = stats.kurtosis(sentiment_values)
            results['sentiment_skewness'] = float(skewness)
            results['sentiment_kurtosis'] = float(kurtosis)

            print(f"\nSentiment Distribution:")
            print(f"  Skewness: {skewness:.3f} (0 = symmetric)")
            print(f"  Kurtosis: {kurtosis:.3f} (0 = normal, >0 = heavy tails)")

        return results

    # =================================================================
    # 4. ENGAGEMENT PATTERNS ANALYSIS
    # =================================================================

    def analyze_engagement_patterns(self) -> Dict:
        """Analyze network formation, virality, echo chambers."""
        print("\n" + "="*70)
        print("ENGAGEMENT PATTERNS ANALYSIS")
        print("="*70)

        results = {}

        # Engagement distribution (Gini coefficient)
        engagement_scores = self.content_df['engagement_score'].sort_values()
        n = len(engagement_scores)
        if n > 0:
            cumsum = engagement_scores.cumsum()
            gini = (2 * sum((i+1) * engagement_scores.iloc[i] for i in range(n))) / (n * cumsum.iloc[-1]) - (n+1)/n
            results['engagement_gini'] = float(gini)

            print(f"\nEngagement Gini Coefficient: {gini:.3f}")
            print(f"  (0 = equal, 1 = winner-take-all)")

        # Virality analysis (content with high engagement growth)
        viral_content = self.content_df.nlargest(10, 'engagement_score')[
            ['id', 'author_id', 'llm_architecture', 'topic', 'engagement_score', 'total_views', 'total_likes']
        ]
        results['viral_content'] = viral_content.to_dict('records')

        print("\nTop 10 Viral Content:")
        for i, row in viral_content.iterrows():
            print(f"  {row['id']}: {row['engagement_score']:.1f} engagement "
                  f"({row['llm_architecture']}, {row['topic']})")

        # Response network analysis
        response_network = self.content_df[self.content_df['parent_id'].notna()][
            ['id', 'parent_id', 'author_id', 'llm_architecture']
        ]
        results['n_responses'] = len(response_network)
        results['response_rate'] = float(len(response_network) / len(self.content_df))

        print(f"\nResponse Network:")
        print(f"  Total responses: {len(response_network)}")
        print(f"  Response rate: {len(response_network)/len(self.content_df)*100:.1f}%")

        # Echo chamber detection - same architecture responding to same architecture
        if len(response_network) > 0:
            response_network_with_parent = response_network.merge(
                self.content_df[['id', 'llm_architecture']],
                left_on='parent_id',
                right_on='id',
                suffixes=('_response', '_parent')
            )

            same_arch_responses = response_network_with_parent[
                response_network_with_parent['llm_architecture_response'] ==
                response_network_with_parent['llm_architecture_parent']
            ]

            echo_chamber_rate = len(same_arch_responses) / len(response_network)
            results['echo_chamber_rate'] = float(echo_chamber_rate)

            print(f"\nEcho Chamber Analysis:")
            print(f"  Same-architecture responses: {len(same_arch_responses)}/{len(response_network)}")
            print(f"  Echo chamber rate: {echo_chamber_rate*100:.1f}%")

        return results

    # =================================================================
    # 5. POLARIZATION INDICES ANALYSIS
    # =================================================================

    def analyze_polarization(self) -> Dict:
        """Analyze opinion clustering and polarization."""
        print("\n" + "="*70)
        print("POLARIZATION INDICES ANALYSIS")
        print("="*70)

        results = {}

        # Opinion evolution analysis
        opinion_evolution = {}
        for round_str, opinions in self.round_data['user_opinions'].items():
            round_num = int(round_str)
            opinion_values = list(opinions.values())
            opinion_evolution[round_num] = {
                'mean': float(np.mean(opinion_values)),
                'std': float(np.std(opinion_values)),
                'range': float(np.max(opinion_values) - np.min(opinion_values))
            }

        results['opinion_evolution'] = opinion_evolution

        print("\nOpinion Evolution:")
        for round_num, stats_dict in sorted(opinion_evolution.items()):
            print(f"  Round {round_num}: mean={stats_dict['mean']:.3f}, "
                  f"std={stats_dict['std']:.3f}, range={stats_dict['range']:.3f}")

        # Final opinion clustering (using k-means-like approach)
        final_round = max(int(r) for r in self.round_data['user_opinions'].keys())
        final_opinions = list(self.round_data['user_opinions'][str(final_round)].values())

        if len(final_opinions) > 0:
            # Simple bimodality check
            median_opinion = np.median(final_opinions)
            below_median = [o for o in final_opinions if o < median_opinion]
            above_median = [o for o in final_opinions if o >= median_opinion]

            if len(below_median) > 0 and len(above_median) > 0:
                between_group_distance = abs(np.mean(above_median) - np.mean(below_median))
                within_group_variance = (np.var(below_median) + np.var(above_median)) / 2

                polarization_index = between_group_distance / (within_group_variance + 0.01)
                results['polarization_index'] = float(polarization_index)

                print(f"\nPolarization Index: {polarization_index:.3f}")
                print(f"  (higher = more polarized)")

        # Between-group interaction frequency
        if not self.interactions_df.empty:
            # Merge with content to get architecture info
            interactions_with_arch = self.interactions_df.merge(
                self.content_df[['id', 'llm_architecture']],
                left_on='content_id',
                right_on='id'
            )

            # Count interactions by architecture
            arch_interactions = interactions_with_arch.groupby('llm_architecture').size()
            results['interactions_by_architecture'] = arch_interactions.to_dict()

            print("\nInteractions by Content Architecture:")
            for arch, count in arch_interactions.items():
                print(f"  {arch}: {count} interactions")

        return results

    # =================================================================
    # 6. ARCHITECTURE EFFECTS ANALYSIS
    # =================================================================

    def analyze_architecture_effects(self) -> Dict:
        """Compare performance across model combinations."""
        print("\n" + "="*70)
        print("ARCHITECTURE EFFECTS ANALYSIS")
        print("="*70)

        results = {}

        # Performance by architecture
        arch_performance = self.content_df.groupby('llm_architecture').agg({
            'engagement_score': ['mean', 'std', 'max'],
            'total_views': ['mean', 'sum'],
            'total_likes': ['mean', 'sum'],
            'sentiment_score': ['mean', 'std']
        })

        results['architecture_performance'] = arch_performance.to_dict()

        print("\nArchitecture Performance Comparison:")
        print(arch_performance)

        # Statistical significance tests (if multiple architectures)
        architectures = self.content_df['llm_architecture'].unique()
        if len(architectures) > 1:
            print("\nStatistical Significance Tests (Mann-Whitney U):")

            archs = list(architectures)
            for i in range(len(archs)):
                for j in range(i+1, len(archs)):
                    arch1_engagement = self.content_df[
                        self.content_df['llm_architecture'] == archs[i]
                    ]['engagement_score']
                    arch2_engagement = self.content_df[
                        self.content_df['llm_architecture'] == archs[j]
                    ]['engagement_score']

                    if len(arch1_engagement) > 0 and len(arch2_engagement) > 0:
                        statistic, pvalue = stats.mannwhitneyu(
                            arch1_engagement, arch2_engagement, alternative='two-sided'
                        )

                        print(f"  {archs[i]} vs {archs[j]}: p={pvalue:.4f} "
                              f"({'significant' if pvalue < 0.05 else 'not significant'})")

        # Agent-level analysis
        content_by_agent = self.content_df.groupby('author_id').agg({
            'engagement_score': ['mean', 'sum', 'count'],
            'total_views': 'sum',
            'total_likes': 'sum'
        })

        # Flatten multi-level columns for easier handling
        content_by_agent_flat = content_by_agent.copy()
        content_by_agent_flat.columns = ['_'.join(col).strip() for col in content_by_agent_flat.columns.values]
        content_by_agent_flat = content_by_agent_flat.reset_index()

        # Merge with agent info to get architecture (if available)
        if 'llm_backend_id' in self.agents_df.columns:
            content_by_agent_with_arch = content_by_agent_flat.merge(
                self.agents_df[['id', 'llm_backend_id']],
                left_on='author_id',
                right_on='id',
                how='left'
            )
            print(f"\nAgent-Backend Mapping:")
            for _, row in content_by_agent_with_arch.iterrows():
                if pd.notna(row.get('llm_backend_id')):
                    print(f"  {row['author_id']} uses {row['llm_backend_id']} backend")

        print("\nTop Performing Agents:")
        top_agents = content_by_agent.nlargest(5, ('engagement_score', 'sum'))
        for agent_id, row in top_agents.iterrows():
            print(f"  {agent_id}: {row[('engagement_score', 'sum')]:.1f} total engagement "
                  f"({row[('engagement_score', 'count')]} posts)")

        return results

    # =================================================================
    # 7. RECOMMENDATION-TOPIC-ENGAGEMENT CORRELATIONS ANALYSIS
    # =================================================================

    def analyze_recommendation_topic_engagement_correlations(self) -> Dict:
        """
        Analyze correlations between recommendations, topics, and engagement.

        Investigates:
        - Which topics get recommended most frequently
        - Engagement rates by topic
        - Correlation between topic diversity in feeds and user engagement
        """
        print("\n" + "="*70)
        print("RECOMMENDATION-TOPIC-ENGAGEMENT CORRELATIONS ANALYSIS")
        print("="*70)

        results = {}

        # Analyze topic distribution in recommendations
        recommended_content_ids = set()
        topic_recommendation_counts = Counter()

        for round_str, feeds in self.feed_assignments.items():
            for user_id, content_ids in feeds.items():
                for content_id in content_ids:
                    recommended_content_ids.add(content_id)
                    content_row = self.content_df[self.content_df['id'] == content_id]
                    if not content_row.empty:
                        topic = content_row.iloc[0]['topic']
                        topic_recommendation_counts[topic] += 1

        results['topic_recommendation_counts'] = dict(topic_recommendation_counts)

        print("\nTopic Recommendation Frequency:")
        total_recommendations = sum(topic_recommendation_counts.values())
        for topic, count in topic_recommendation_counts.most_common():
            percentage = (count / total_recommendations * 100) if total_recommendations > 0 else 0
            print(f"  {topic}: {count} recommendations ({percentage:.1f}%)")

        # Analyze engagement by topic
        topic_engagement = self.content_df.groupby('topic').agg({
            'engagement_score': ['mean', 'std', 'sum'],
            'total_views': ['mean', 'sum'],
            'total_likes': ['mean', 'sum']
        })
        results['topic_engagement'] = topic_engagement.to_dict()

        print("\nEngagement Metrics by Topic:")
        print(topic_engagement)

        # Calculate correlation between recommendation frequency and engagement
        topic_stats = []
        for topic in self.content_df['topic'].unique():
            topic_content = self.content_df[self.content_df['topic'] == topic]
            rec_count = topic_recommendation_counts.get(topic, 0)
            avg_engagement = topic_content['engagement_score'].mean()
            avg_views = topic_content['total_views'].mean()
            avg_likes = topic_content['total_likes'].mean()

            topic_stats.append({
                'topic': topic,
                'recommendation_count': rec_count,
                'avg_engagement': avg_engagement,
                'avg_views': avg_views,
                'avg_likes': avg_likes
            })

        topic_stats_df = pd.DataFrame(topic_stats)

        # Correlation between recommendation frequency and engagement
        if len(topic_stats_df) > 1:
            corr_engagement = topic_stats_df['recommendation_count'].corr(topic_stats_df['avg_engagement'])
            corr_views = topic_stats_df['recommendation_count'].corr(topic_stats_df['avg_views'])
            corr_likes = topic_stats_df['recommendation_count'].corr(topic_stats_df['avg_likes'])

            results['correlation_rec_engagement'] = float(corr_engagement) if not pd.isna(corr_engagement) else None
            results['correlation_rec_views'] = float(corr_views) if not pd.isna(corr_views) else None
            results['correlation_rec_likes'] = float(corr_likes) if not pd.isna(corr_likes) else None

            print(f"\nCorrelations between Recommendation Frequency and Engagement:")
            print(f"  Rec Frequency ↔ Engagement Score: {corr_engagement:.3f}")
            print(f"  Rec Frequency ↔ Views: {corr_views:.3f}")
            print(f"  Rec Frequency ↔ Likes: {corr_likes:.3f}")

        # Analyze topic diversity in user feeds
        user_topic_diversity = {}
        for round_str, feeds in self.feed_assignments.items():
            for user_id, content_ids in feeds.items():
                topics_in_feed = []
                for content_id in content_ids:
                    content_row = self.content_df[self.content_df['id'] == content_id]
                    if not content_row.empty:
                        topics_in_feed.append(content_row.iloc[0]['topic'])

                if len(topics_in_feed) > 0:
                    # Calculate topic entropy for this feed
                    topic_counts = Counter(topics_in_feed)
                    proportions = [count/len(topics_in_feed) for count in topic_counts.values()]
                    feed_entropy = stats.entropy(proportions)

                    if user_id not in user_topic_diversity:
                        user_topic_diversity[user_id] = []
                    user_topic_diversity[user_id].append(feed_entropy)

        avg_user_topic_diversity = {
            user_id: float(np.mean(diversities))
            for user_id, diversities in user_topic_diversity.items()
        }
        results['avg_user_topic_diversity'] = avg_user_topic_diversity
        results['overall_avg_topic_diversity'] = float(np.mean(list(avg_user_topic_diversity.values())))

        print(f"\nAverage Topic Diversity in User Feeds: {results['overall_avg_topic_diversity']:.3f}")
        print(f"  (higher = more diverse topics in recommendations)")

        return results

    # =================================================================
    # 8. LLM FAVORITISM IN MIXED SCENARIOS ANALYSIS
    # =================================================================

    def analyze_llm_favoritism_in_recommendations(self) -> Dict:
        """
        Analyze whether the recommendation system disproportionately favors
        content from specific LLM architectures in mixed scenarios.

        Investigates:
        - Distribution of recommended content by LLM architecture
        - Whether favoritism correlates with content characteristics (topic, sentiment, etc.)
        - Exposure bias in mixed-LLM scenarios
        """
        print("\n" + "="*70)
        print("LLM FAVORITISM IN RECOMMENDATIONS ANALYSIS")
        print("="*70)

        results = {}

        # Check if this is a mixed scenario
        architectures = self.content_df['llm_architecture'].unique()
        if len(architectures) <= 1:
            print("\nThis is not a mixed-LLM scenario (only one architecture detected).")
            print("Favoritism analysis requires multiple LLM architectures.")
            results['is_mixed_scenario'] = False
            return results

        results['is_mixed_scenario'] = True
        results['architectures'] = list(architectures)

        print(f"\nDetected mixed scenario with architectures: {', '.join(architectures)}")

        # Content generation distribution (baseline)
        content_by_arch = self.content_df['llm_architecture'].value_counts()
        total_content = len(self.content_df)
        baseline_proportions = {arch: count/total_content for arch, count in content_by_arch.items()}

        print("\nBaseline Content Generation by Architecture:")
        for arch, count in content_by_arch.items():
            print(f"  {arch}: {count} items ({baseline_proportions[arch]*100:.1f}%)")

        results['baseline_proportions'] = baseline_proportions

        # Recommendation distribution
        recommended_content_ids = []
        for round_str, feeds in self.feed_assignments.items():
            for user_id, content_ids in feeds.items():
                recommended_content_ids.extend(content_ids)

        # Get architectures of recommended content
        recommended_arch_counts = Counter()
        for content_id in recommended_content_ids:
            content_row = self.content_df[self.content_df['id'] == content_id]
            if not content_row.empty:
                arch = content_row.iloc[0]['llm_architecture']
                recommended_arch_counts[arch] += 1

        total_recommendations = sum(recommended_arch_counts.values())
        recommendation_proportions = {
            arch: count/total_recommendations
            for arch, count in recommended_arch_counts.items()
        }

        print("\nRecommendation Distribution by Architecture:")
        for arch, count in recommended_arch_counts.items():
            print(f"  {arch}: {count} recommendations ({recommendation_proportions[arch]*100:.1f}%)")

        results['recommendation_proportions'] = recommendation_proportions

        # Calculate favoritism bias (difference from baseline)
        favoritism_bias = {}
        print("\nFavoritism Bias (Recommendation % - Generation %):")
        for arch in architectures:
            baseline = baseline_proportions.get(arch, 0)
            recommended = recommendation_proportions.get(arch, 0)
            bias = recommended - baseline
            favoritism_bias[arch] = float(bias)

            bias_direction = "favored" if bias > 0 else "disfavored" if bias < 0 else "neutral"
            print(f"  {arch}: {bias:+.1%} ({bias_direction})")

        results['favoritism_bias'] = favoritism_bias

        # Statistical significance test (Chi-square test)
        expected_counts = [baseline_proportions.get(arch, 0) * total_recommendations for arch in architectures]
        observed_counts = [recommended_arch_counts.get(arch, 0) for arch in architectures]

        if all(c > 5 for c in expected_counts):  # Chi-square validity check
            chi2_stat, p_value = stats.chisquare(observed_counts, expected_counts)
            results['chi2_statistic'] = float(chi2_stat)
            results['chi2_pvalue'] = float(p_value)

            print(f"\nChi-Square Test for Favoritism:")
            print(f"  χ² = {chi2_stat:.3f}, p = {p_value:.4f}")
            print(f"  {'Significant' if p_value < 0.05 else 'Not significant'} favoritism detected")

        # Analyze favoritism by topic
        print("\nFavoritism by Topic:")
        topic_arch_recommendations = defaultdict(Counter)

        for content_id in recommended_content_ids:
            content_row = self.content_df[self.content_df['id'] == content_id]
            if not content_row.empty:
                arch = content_row.iloc[0]['llm_architecture']
                topic = content_row.iloc[0]['topic']
                topic_arch_recommendations[topic][arch] += 1

        topic_favoritism = {}
        for topic, arch_counts in topic_arch_recommendations.items():
            total_topic_recs = sum(arch_counts.values())
            topic_proportions = {
                arch: count/total_topic_recs
                for arch, count in arch_counts.items()
            }
            topic_favoritism[topic] = topic_proportions

            print(f"\n  Topic: {topic}")
            for arch, proportion in topic_proportions.items():
                baseline = baseline_proportions.get(arch, 0)
                bias = proportion - baseline
                print(f"    {arch}: {proportion*100:.1f}% (bias: {bias:+.1%})")

        results['favoritism_by_topic'] = topic_favoritism

        # Analyze favoritism by sentiment
        print("\nFavoritism by Sentiment Quartile:")
        try:
            self.content_df['sentiment_quartile'] = pd.qcut(
                self.content_df['sentiment_score'],
                q=4,
                labels=['Q1 (Most Negative)', 'Q2', 'Q3', 'Q4 (Most Positive)'],
                duplicates='drop'
            )
        except ValueError:
            # If we can't create 4 quartiles, use fewer bins
            self.content_df['sentiment_quartile'] = pd.qcut(
                self.content_df['sentiment_score'],
                q=4,
                duplicates='drop'
            )

        sentiment_arch_recommendations = defaultdict(Counter)
        for content_id in recommended_content_ids:
            content_row = self.content_df[self.content_df['id'] == content_id]
            if not content_row.empty:
                arch = content_row.iloc[0]['llm_architecture']
                sentiment_q = content_row.iloc[0]['sentiment_quartile']
                if pd.notna(sentiment_q):
                    sentiment_arch_recommendations[sentiment_q][arch] += 1

        sentiment_favoritism = {}
        for sentiment_q, arch_counts in sentiment_arch_recommendations.items():
            total_sentiment_recs = sum(arch_counts.values())
            sentiment_proportions = {
                arch: count/total_sentiment_recs
                for arch, count in arch_counts.items()
            }
            sentiment_favoritism[str(sentiment_q)] = sentiment_proportions

            print(f"\n  Sentiment: {sentiment_q}")
            for arch, proportion in sentiment_proportions.items():
                baseline = baseline_proportions.get(arch, 0)
                bias = proportion - baseline
                print(f"    {arch}: {proportion*100:.1f}% (bias: {bias:+.1%})")

        results['favoritism_by_sentiment'] = sentiment_favoritism

        # Analyze engagement outcomes of favoritism
        arch_engagement_in_recommendations = {}
        for arch in architectures:
            arch_recommended_ids = [
                content_id for content_id in recommended_content_ids
                if not self.content_df[self.content_df['id'] == content_id].empty
                and self.content_df[self.content_df['id'] == content_id].iloc[0]['llm_architecture'] == arch
            ]

            if arch_recommended_ids:
                arch_recommended_content = self.content_df[self.content_df['id'].isin(arch_recommended_ids)]
                arch_engagement_in_recommendations[arch] = {
                    'mean_engagement': float(arch_recommended_content['engagement_score'].mean()),
                    'mean_views': float(arch_recommended_content['total_views'].mean()),
                    'mean_likes': float(arch_recommended_content['total_likes'].mean())
                }

        results['architecture_engagement_in_recommendations'] = arch_engagement_in_recommendations

        print("\nEngagement of Recommended Content by Architecture:")
        for arch, metrics in arch_engagement_in_recommendations.items():
            print(f"  {arch}:")
            print(f"    Mean Engagement: {metrics['mean_engagement']:.2f}")
            print(f"    Mean Views: {metrics['mean_views']:.2f}")
            print(f"    Mean Likes: {metrics['mean_likes']:.2f}")

        return results

    # =================================================================
    # VISUALIZATION
    # =================================================================

    def create_visualizations(self, save_dir: str = None):
        """Create comprehensive visualization plots."""
        if save_dir is None:
            save_dir = self.output_dir / "analysis_plots"
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(exist_ok=True)

        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)

        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")

        # 1. Content diversity over time
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Topic distribution
        self.content_df['topic'].value_counts().plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Topic Distribution')
        axes[0, 0].set_xlabel('Topic')
        axes[0, 0].set_ylabel('Count')

        # Architecture distribution
        self.content_df['llm_architecture'].value_counts().plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Content by Architecture')
        axes[0, 1].set_xlabel('Architecture')
        axes[0, 1].set_ylabel('Count')

        # Sentiment distribution
        self.content_df['sentiment_score'].hist(bins=20, ax=axes[1, 0])
        axes[1, 0].set_title('Sentiment Distribution')
        axes[1, 0].set_xlabel('Sentiment Score')
        axes[1, 0].set_ylabel('Frequency')

        # Engagement distribution
        self.content_df['engagement_score'].hist(bins=20, ax=axes[1, 1])
        axes[1, 1].set_title('Engagement Distribution')
        axes[1, 1].set_xlabel('Engagement Score')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(save_dir / 'content_overview.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved content_overview.png")
        plt.close()

        # 2. Temporal dynamics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Sentiment evolution
        sentiment_by_round = self.content_df.groupby('round_created')['sentiment_score'].mean()
        sentiment_by_round.plot(marker='o', ax=axes[0, 0])
        axes[0, 0].set_title('Sentiment Evolution')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Mean Sentiment')

        # Engagement evolution
        engagement_by_round = self.content_df.groupby('round_created')['engagement_score'].mean()
        engagement_by_round.plot(marker='o', ax=axes[0, 1])
        axes[0, 1].set_title('Engagement Evolution')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Mean Engagement')

        # Opinion evolution
        rounds = sorted([int(r) for r in self.round_data['user_opinions'].keys()])
        opinion_means = [np.mean(list(self.round_data['user_opinions'][str(r)].values())) for r in rounds]
        opinion_stds = [np.std(list(self.round_data['user_opinions'][str(r)].values())) for r in rounds]

        axes[1, 0].plot(rounds, opinion_means, marker='o', label='Mean')
        axes[1, 0].fill_between(rounds,
                                 [m-s for m,s in zip(opinion_means, opinion_stds)],
                                 [m+s for m,s in zip(opinion_means, opinion_stds)],
                                 alpha=0.3)
        axes[1, 0].set_title('Opinion Evolution')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Opinion Value')
        axes[1, 0].legend()

        # Content volume by round
        content_by_round = self.content_df.groupby('round_created').size()
        content_by_round.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Content Volume by Round')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Number of Posts')

        plt.tight_layout()
        plt.savefig(save_dir / 'temporal_dynamics.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved temporal_dynamics.png")
        plt.close()

        # 3. Architecture comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Engagement by architecture
        self.content_df.boxplot(column='engagement_score', by='llm_architecture', ax=axes[0, 0])
        axes[0, 0].set_title('Engagement by Architecture')
        axes[0, 0].set_xlabel('Architecture')
        axes[0, 0].set_ylabel('Engagement Score')
        plt.sca(axes[0, 0])
        plt.xticks(rotation=45)

        # Sentiment by architecture
        self.content_df.boxplot(column='sentiment_score', by='llm_architecture', ax=axes[0, 1])
        axes[0, 1].set_title('Sentiment by Architecture')
        axes[0, 1].set_xlabel('Architecture')
        axes[0, 1].set_ylabel('Sentiment Score')
        plt.sca(axes[0, 1])
        plt.xticks(rotation=45)

        # Views by architecture
        arch_views = self.content_df.groupby('llm_architecture')['total_views'].sum()
        arch_views.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Total Views by Architecture')
        axes[1, 0].set_xlabel('Architecture')
        axes[1, 0].set_ylabel('Total Views')
        plt.sca(axes[1, 0])
        plt.xticks(rotation=45)

        # Likes by architecture
        arch_likes = self.content_df.groupby('llm_architecture')['total_likes'].sum()
        arch_likes.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Total Likes by Architecture')
        axes[1, 1].set_xlabel('Architecture')
        axes[1, 1].set_ylabel('Total Likes')
        plt.sca(axes[1, 1])
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(save_dir / 'architecture_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved architecture_comparison.png")
        plt.close()

        # 4. Recommendation-Topic-Engagement Correlations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Topic recommendation frequency
        topic_rec_counts = {}
        for round_str, feeds in self.feed_assignments.items():
            for user_id, content_ids in feeds.items():
                for content_id in content_ids:
                    content_row = self.content_df[self.content_df['id'] == content_id]
                    if not content_row.empty:
                        topic = content_row.iloc[0]['topic']
                        topic_rec_counts[topic] = topic_rec_counts.get(topic, 0) + 1

        if topic_rec_counts:
            topics = list(topic_rec_counts.keys())
            counts = list(topic_rec_counts.values())
            axes[0, 0].bar(topics, counts)
            axes[0, 0].set_title('Topic Recommendation Frequency')
            axes[0, 0].set_xlabel('Topic')
            axes[0, 0].set_ylabel('Number of Recommendations')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # Engagement by topic
        topic_engagement = self.content_df.groupby('topic')['engagement_score'].mean().sort_values(ascending=False)
        topic_engagement.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Average Engagement by Topic')
        axes[0, 1].set_xlabel('Topic')
        axes[0, 1].set_ylabel('Average Engagement Score')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Topic vs engagement scatter
        topic_stats = []
        for topic in self.content_df['topic'].unique():
            topic_content = self.content_df[self.content_df['topic'] == topic]
            rec_count = topic_rec_counts.get(topic, 0)
            avg_engagement = topic_content['engagement_score'].mean()
            topic_stats.append({'topic': topic, 'rec_count': rec_count, 'engagement': avg_engagement})

        topic_stats_df = pd.DataFrame(topic_stats)
        if len(topic_stats_df) > 0:
            axes[1, 0].scatter(topic_stats_df['rec_count'], topic_stats_df['engagement'])
            for _, row in topic_stats_df.iterrows():
                axes[1, 0].annotate(row['topic'], (row['rec_count'], row['engagement']),
                                   fontsize=8, alpha=0.7)
            axes[1, 0].set_title('Recommendation Frequency vs Engagement')
            axes[1, 0].set_xlabel('Recommendation Count')
            axes[1, 0].set_ylabel('Average Engagement Score')

        # Views by topic
        topic_views = self.content_df.groupby('topic')['total_views'].mean().sort_values(ascending=False)
        topic_views.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Average Views by Topic')
        axes[1, 1].set_xlabel('Topic')
        axes[1, 1].set_ylabel('Average Views')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(save_dir / 'recommendation_topic_engagement.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved recommendation_topic_engagement.png")
        plt.close()

        # 5. LLM Favoritism Analysis (only for mixed scenarios)
        architectures = self.content_df['llm_architecture'].unique()
        if len(architectures) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Content generation distribution (baseline)
            content_by_arch = self.content_df['llm_architecture'].value_counts()
            axes[0, 0].bar(content_by_arch.index, content_by_arch.values)
            axes[0, 0].set_title('Content Generation by Architecture (Baseline)')
            axes[0, 0].set_xlabel('Architecture')
            axes[0, 0].set_ylabel('Number of Content Items')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Recommendation distribution
            recommended_content_ids = []
            for round_str, feeds in self.feed_assignments.items():
                for user_id, content_ids in feeds.items():
                    recommended_content_ids.extend(content_ids)

            recommended_arch_counts = Counter()
            for content_id in recommended_content_ids:
                content_row = self.content_df[self.content_df['id'] == content_id]
                if not content_row.empty:
                    arch = content_row.iloc[0]['llm_architecture']
                    recommended_arch_counts[arch] += 1

            if recommended_arch_counts:
                axes[0, 1].bar(recommended_arch_counts.keys(), recommended_arch_counts.values())
                axes[0, 1].set_title('Recommendation Distribution by Architecture')
                axes[0, 1].set_xlabel('Architecture')
                axes[0, 1].set_ylabel('Number of Recommendations')
                axes[0, 1].tick_params(axis='x', rotation=45)

            # Favoritism bias
            total_content = len(self.content_df)
            baseline_proportions = {arch: count/total_content for arch, count in content_by_arch.items()}
            total_recommendations = sum(recommended_arch_counts.values())
            recommendation_proportions = {
                arch: count/total_recommendations
                for arch, count in recommended_arch_counts.items()
            }
            favoritism_bias = {
                arch: recommendation_proportions.get(arch, 0) - baseline_proportions.get(arch, 0)
                for arch in architectures
            }

            axes[1, 0].bar(favoritism_bias.keys(), [v*100 for v in favoritism_bias.values()])
            axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('Favoritism Bias (% Recommended - % Generated)')
            axes[1, 0].set_xlabel('Architecture')
            axes[1, 0].set_ylabel('Bias (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Engagement of recommended content by architecture
            arch_engagement = {}
            for arch in architectures:
                arch_recommended_ids = [
                    content_id for content_id in recommended_content_ids
                    if not self.content_df[self.content_df['id'] == content_id].empty
                    and self.content_df[self.content_df['id'] == content_id].iloc[0]['llm_architecture'] == arch
                ]
                if arch_recommended_ids:
                    arch_recommended_content = self.content_df[self.content_df['id'].isin(arch_recommended_ids)]
                    arch_engagement[arch] = arch_recommended_content['engagement_score'].mean()

            if arch_engagement:
                axes[1, 1].bar(arch_engagement.keys(), arch_engagement.values())
                axes[1, 1].set_title('Average Engagement of Recommended Content')
                axes[1, 1].set_xlabel('Architecture')
                axes[1, 1].set_ylabel('Average Engagement Score')
                axes[1, 1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(save_dir / 'llm_favoritism_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved llm_favoritism_analysis.png")
            plt.close()
        else:
            print(f"✓ Skipped llm_favoritism_analysis.png (single architecture scenario)")

        print(f"\nAll visualizations saved to {save_dir}/")

    # =================================================================
    # MAIN ANALYSIS RUNNER
    # =================================================================

    def _sanitize_for_json(self, obj):
        """
        Recursively sanitize data for JSON serialization.
        Converts tuple keys to strings and handles numpy/pandas types.
        """
        return sanitize_for_json(obj)

    def run_full_analysis(self, save_results: bool = True) -> Dict:
        """Run complete analysis suite."""
        print("\n" + "="*70)
        print("RUNNING COMPREHENSIVE ANALYSIS")
        print("="*70)

        all_results = {}

        all_results['content_diversity'] = self.analyze_content_diversity()
        all_results['promotion_patterns'] = self.analyze_promotion_patterns()
        all_results['sentiment_dynamics'] = self.analyze_sentiment_dynamics()
        all_results['engagement_patterns'] = self.analyze_engagement_patterns()
        all_results['polarization'] = self.analyze_polarization()
        all_results['architecture_effects'] = self.analyze_architecture_effects()
        all_results['recommendation_topic_engagement'] = self.analyze_recommendation_topic_engagement_correlations()
        all_results['llm_favoritism'] = self.analyze_llm_favoritism_in_recommendations()

        # Create visualizations
        self.create_visualizations()

        # Save results
        if save_results:
            results_file = self.output_dir / "comprehensive_analysis.json"
            # Sanitize results for JSON serialization
            sanitized_results = self._sanitize_for_json(all_results)
            with open(results_file, 'w') as f:
                json.dump(sanitized_results, f, indent=2, default=str)
            print(f"\n✓ Analysis results saved to {results_file}")

        return all_results


def compare_scenarios(scenario1_dir: str, scenario2_dir: str, output_dir: str = "./comparison_results"):
    """
    Compare two scenarios side-by-side.

    Args:
        scenario1_dir: Output directory for scenario 1
        scenario2_dir: Output directory for scenario 2
        output_dir: Where to save comparison results
    """
    print("\n" + "="*70)
    print("COMPARING SCENARIOS")
    print("="*70)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Analyze both scenarios
    print("\n### Analyzing Scenario 1 ###")
    analyzer1 = ScenarioAnalyzer(scenario1_dir)
    results1 = analyzer1.run_full_analysis(save_results=True)

    print("\n### Analyzing Scenario 2 ###")
    analyzer2 = ScenarioAnalyzer(scenario2_dir)
    results2 = analyzer2.run_full_analysis(save_results=True)

    # Create comparison report
    comparison = {
        'scenario1': {
            'path': scenario1_dir,
            'results': results1
        },
        'scenario2': {
            'path': scenario2_dir,
            'results': results2
        },
        'comparison_summary': {}
    }

    # Key comparisons
    print("\n" + "="*70)
    print("SCENARIO COMPARISON SUMMARY")
    print("="*70)

    # Content diversity
    print("\nContent Diversity:")
    print(f"  Scenario 1 - Topic Entropy: {results1['content_diversity']['topic_entropy']:.3f}")
    print(f"  Scenario 2 - Topic Entropy: {results2['content_diversity']['topic_entropy']:.3f}")

    if 'semantic_diversity' in results1['content_diversity'] and 'semantic_diversity' in results2['content_diversity']:
        print(f"  Scenario 1 - Semantic Diversity: {results1['content_diversity']['semantic_diversity']:.3f}")
        print(f"  Scenario 2 - Semantic Diversity: {results2['content_diversity']['semantic_diversity']:.3f}")

    # Engagement
    print("\nEngagement:")
    print(f"  Scenario 1 - Gini: {results1['engagement_patterns']['engagement_gini']:.3f}")
    print(f"  Scenario 2 - Gini: {results2['engagement_patterns']['engagement_gini']:.3f}")

    # Polarization
    if 'polarization_index' in results1['polarization'] and 'polarization_index' in results2['polarization']:
        print("\nPolarization:")
        print(f"  Scenario 1 - Index: {results1['polarization']['polarization_index']:.3f}")
        print(f"  Scenario 2 - Index: {results2['polarization']['polarization_index']:.3f}")

    # Save comparison
    comparison_file = output_path / "scenario_comparison.json"
    with open(comparison_file, 'w') as f:
        # Convert any tuple keys to strings before JSON serialization
        comparison_serializable = sanitize_for_json(comparison)
        json.dump(comparison_serializable, f, indent=2, default=str)

    print(f"\n✓ Comparison saved to {comparison_file}")

    return comparison


def compare_multiple_scenarios(scenario_dirs: List[str], scenario_names: List[str] = None,
                              output_dir: str = "./multi_comparison_results"):
    """
    Compare multiple scenarios side-by-side.

    Args:
        scenario_dirs: List of output directories for each scenario
        scenario_names: Optional list of names for each scenario (for display)
        output_dir: Where to save comparison results
    """
    print("\n" + "="*70)
    print("COMPARING MULTIPLE SCENARIOS")
    print("="*70)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Analyze all scenarios
    analyzers = []
    results_list = []

    if scenario_names is None:
        scenario_names = [f"Scenario {i+1}" for i in range(len(scenario_dirs))]

    for i, scenario_dir in enumerate(scenario_dirs):
        print(f"\n### Analyzing {scenario_names[i]} ###")
        analyzer = ScenarioAnalyzer(scenario_dir)
        results = analyzer.run_full_analysis(save_results=True)
        analyzers.append(analyzer)
        results_list.append(results)

    # Create comprehensive comparison report
    comparison = {
        'scenarios': [
            {
                'name': scenario_names[i],
                'path': scenario_dirs[i],
                'results': results_list[i]
            }
            for i in range(len(scenario_dirs))
        ],
        'comparison_summary': {}
    }

    # Key comparisons
    print("\n" + "="*70)
    print("MULTI-SCENARIO COMPARISON SUMMARY")
    print("="*70)

    # Content diversity comparison
    print("\n1. Content Diversity:")
    print(f"{'Scenario':<30} {'Topic Entropy':<15} {'Semantic Diversity':<20}")
    print("-" * 70)
    for i, results in enumerate(results_list):
        topic_entropy = results['content_diversity']['topic_entropy']
        semantic_div = results['content_diversity'].get('semantic_diversity', 'N/A')
        if semantic_div != 'N/A':
            print(f"{scenario_names[i]:<30} {topic_entropy:<15.3f} {semantic_div:<20.3f}")
        else:
            print(f"{scenario_names[i]:<30} {topic_entropy:<15.3f} {'N/A':<20}")

    # Engagement comparison
    print("\n2. Engagement:")
    print(f"{'Scenario':<30} {'Gini Coefficient':<20}")
    print("-" * 70)
    for i, results in enumerate(results_list):
        gini = results['engagement_patterns']['engagement_gini']
        print(f"{scenario_names[i]:<30} {gini:<20.3f}")

    # Polarization comparison
    print("\n3. Polarization:")
    print(f"{'Scenario':<30} {'Polarization Index':<20}")
    print("-" * 70)
    for i, results in enumerate(results_list):
        pol_index = results['polarization'].get('polarization_index', 'N/A')
        if pol_index != 'N/A':
            print(f"{scenario_names[i]:<30} {pol_index:<20.3f}")
        else:
            print(f"{scenario_names[i]:<30} {'N/A':<20}")

    # Recommendation-Topic-Engagement correlations
    print("\n4. Recommendation-Topic-Engagement Correlations:")
    print(f"{'Scenario':<30} {'Rec↔Engagement':<20} {'Rec↔Views':<15} {'Rec↔Likes':<15}")
    print("-" * 70)
    for i, results in enumerate(results_list):
        if 'recommendation_topic_engagement' in results:
            corr_eng = results['recommendation_topic_engagement'].get('correlation_rec_engagement', 'N/A')
            corr_views = results['recommendation_topic_engagement'].get('correlation_rec_views', 'N/A')
            corr_likes = results['recommendation_topic_engagement'].get('correlation_rec_likes', 'N/A')

            corr_eng_str = f"{corr_eng:.3f}" if corr_eng != 'N/A' and corr_eng is not None else 'N/A'
            corr_views_str = f"{corr_views:.3f}" if corr_views != 'N/A' and corr_views is not None else 'N/A'
            corr_likes_str = f"{corr_likes:.3f}" if corr_likes != 'N/A' and corr_likes is not None else 'N/A'

            print(f"{scenario_names[i]:<30} {corr_eng_str:<20} {corr_views_str:<15} {corr_likes_str:<15}")
        else:
            print(f"{scenario_names[i]:<30} {'N/A':<20} {'N/A':<15} {'N/A':<15}")

    # LLM Favoritism (only for mixed scenarios)
    print("\n5. LLM Favoritism in Recommendations (Mixed Scenarios Only):")
    for i, results in enumerate(results_list):
        if 'llm_favoritism' in results and results['llm_favoritism'].get('is_mixed_scenario', False):
            print(f"\n{scenario_names[i]}:")
            favoritism_bias = results['llm_favoritism']['favoritism_bias']
            for arch, bias in favoritism_bias.items():
                print(f"  {arch}: {bias:+.1%} bias")

            if 'chi2_pvalue' in results['llm_favoritism']:
                p_value = results['llm_favoritism']['chi2_pvalue']
                significance = "Significant" if p_value < 0.05 else "Not significant"
                print(f"  Statistical test: p={p_value:.4f} ({significance})")
        else:
            print(f"\n{scenario_names[i]}: Single-architecture scenario (no favoritism analysis)")

    # Architecture performance comparison
    print("\n6. Architecture Performance:")
    for i, results in enumerate(results_list):
        print(f"\n{scenario_names[i]}:")
        if 'architecture_performance' in results['architecture_effects']:
            arch_perf = results['architecture_effects']['architecture_performance']
            for arch in arch_perf.get('engagement_score', {}).get('mean', {}).keys():
                mean_eng = arch_perf['engagement_score']['mean'].get(arch, 0)
                mean_views = arch_perf['total_views']['mean'].get(arch, 0)
                mean_likes = arch_perf['total_likes']['mean'].get(arch, 0)
                print(f"  {arch}:")
                print(f"    Avg Engagement: {mean_eng:.2f}")
                print(f"    Avg Views: {mean_views:.2f}")
                print(f"    Avg Likes: {mean_likes:.2f}")

    # Create comparative visualizations
    print("\n" + "="*70)
    print("CREATING COMPARATIVE VISUALIZATIONS")
    print("="*70)

    # 1. Compare key metrics across scenarios
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Multi-Scenario Comparison', fontsize=16)

    # Topic entropy
    topic_entropies = [r['content_diversity']['topic_entropy'] for r in results_list]
    axes[0, 0].bar(scenario_names, topic_entropies)
    axes[0, 0].set_title('Topic Entropy')
    axes[0, 0].set_ylabel('Entropy')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Engagement Gini
    gini_coeffs = [r['engagement_patterns']['engagement_gini'] for r in results_list]
    axes[0, 1].bar(scenario_names, gini_coeffs)
    axes[0, 1].set_title('Engagement Gini Coefficient')
    axes[0, 1].set_ylabel('Gini')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Polarization index
    pol_indices = [r['polarization'].get('polarization_index', 0) for r in results_list]
    axes[0, 2].bar(scenario_names, pol_indices)
    axes[0, 2].set_title('Polarization Index')
    axes[0, 2].set_ylabel('Index')
    axes[0, 2].tick_params(axis='x', rotation=45)

    # Filter bubble scores
    filter_bubbles = [r['promotion_patterns']['avg_filter_bubble_score'] for r in results_list]
    axes[1, 0].bar(scenario_names, filter_bubbles)
    axes[1, 0].set_title('Filter Bubble Score (HHI)')
    axes[1, 0].set_ylabel('HHI')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Topic diversity in feeds
    topic_diversities = []
    for r in results_list:
        if 'recommendation_topic_engagement' in r:
            topic_diversities.append(r['recommendation_topic_engagement']['overall_avg_topic_diversity'])
        else:
            topic_diversities.append(0)

    axes[1, 1].bar(scenario_names, topic_diversities)
    axes[1, 1].set_title('Topic Diversity in User Feeds')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # Echo chamber rate
    echo_rates = [r['engagement_patterns'].get('echo_chamber_rate', 0) for r in results_list]
    axes[1, 2].bar(scenario_names, [r*100 for r in echo_rates])
    axes[1, 2].set_title('Echo Chamber Rate')
    axes[1, 2].set_ylabel('Rate (%)')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path / 'multi_scenario_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved multi_scenario_comparison.png")
    plt.close()

    # Save comparison report
    comparison_file = output_path / "multi_scenario_comparison.json"
    comparison_serializable = sanitize_for_json(comparison)
    with open(comparison_file, 'w') as f:
        json.dump(comparison_serializable, f, indent=2, default=str)

    print(f"\n✓ Multi-scenario comparison saved to {comparison_file}")

    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze simulation scenarios")
    parser.add_argument("--output-dir", type=str,
                       help="Directory containing simulation outputs (for single scenario analysis)")
    parser.add_argument("--compare", type=str, help="Second scenario directory for comparison")
    parser.add_argument("--multi-compare", nargs='+',
                       help="Multiple scenario directories for comparison (space-separated)")
    parser.add_argument("--scenario-names", nargs='+',
                       help="Optional names for scenarios in multi-compare (space-separated)")

    args = parser.parse_args()

    if args.multi_compare:
        # Compare multiple scenarios
        scenario_names = args.scenario_names if args.scenario_names else None
        compare_multiple_scenarios(args.multi_compare, scenario_names)
    elif args.compare and args.output_dir:
        # Compare two scenarios
        compare_scenarios(args.output_dir, args.compare)
    elif args.output_dir:
        # Analyze single scenario
        analyzer = ScenarioAnalyzer(args.output_dir)
        analyzer.run_full_analysis()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  Single scenario: python analyze_scenarios.py --output-dir ./scenario1_output")
        print("  Two scenarios: python analyze_scenarios.py --output-dir ./scenario1_output --compare ./scenario2_output")
        print("  Multiple scenarios: python analyze_scenarios.py --multi-compare ./scenario1_output ./scenario2_output ./scenario3_output")
        print("  With names: python analyze_scenarios.py --multi-compare ./s1 ./s2 ./s3 --scenario-names 'Llama Only' 'Mixed' 'Mistral Only'")
