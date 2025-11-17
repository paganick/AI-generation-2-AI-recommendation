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

        # Merge with agent info to get architecture
        content_by_agent_with_arch = content_by_agent.merge(
            self.agents_df[['id', 'llm_backend_id']],
            left_index=True,
            right_on='id',
            how='left'
        )

        print("\nTop Performing Agents:")
        top_agents = content_by_agent.nlargest(5, ('engagement_score', 'sum'))
        for agent_id, row in top_agents.iterrows():
            print(f"  {agent_id}: {row[('engagement_score', 'sum')]:.1f} total engagement "
                  f"({row[('engagement_score', 'count')]} posts)")

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

        print(f"\nAll visualizations saved to {save_dir}/")

    # =================================================================
    # MAIN ANALYSIS RUNNER
    # =================================================================

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

        # Create visualizations
        self.create_visualizations()

        # Save results
        if save_results:
            results_file = self.output_dir / "comprehensive_analysis.json"
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
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
        json.dump(comparison, f, indent=2, default=str)

    print(f"\n✓ Comparison saved to {comparison_file}")

    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze simulation scenarios")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory containing simulation outputs")
    parser.add_argument("--compare", type=str, help="Second scenario directory for comparison")

    args = parser.parse_args()

    if args.compare:
        # Compare two scenarios
        compare_scenarios(args.output_dir, args.compare)
    else:
        # Analyze single scenario
        analyzer = ScenarioAnalyzer(args.output_dir)
        analyzer.run_full_analysis()
