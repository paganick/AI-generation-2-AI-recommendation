"""
Bias Effect Analyzer

Provides specialized analysis methods for different bias effects.
Each bias effect has dedicated analysis functions with appropriate metrics and statistical tests.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from collections import Counter
import json


class BiasEffectAnalyzer:
    """Analyzes simulation results for specific bias effects."""

    def __init__(self, output_dir: str):
        """
        Initialize analyzer with simulation output directory.

        Args:
            output_dir: Path to simulation output directory
        """
        self.output_dir = Path(output_dir)
        self.content_df = None
        self.interactions_df = None
        self.agents_df = None
        self.users_df = None
        self.feed_assignments = None

        self._load_data()

    def _load_data(self):
        """Load all simulation data files."""
        try:
            self.content_df = pd.read_csv(self.output_dir / "all_content.csv")
            self.interactions_df = pd.read_csv(self.output_dir / "all_interactions.csv")
            self.agents_df = pd.read_csv(self.output_dir / "agents.csv")
            self.users_df = pd.read_csv(self.output_dir / "users.csv")

            feed_file = self.output_dir / "feed_assignments.json"
            if feed_file.exists():
                with open(feed_file, 'r') as f:
                    self.feed_assignments = json.load(f)

            print(f"Loaded data from {self.output_dir}")
            print(f"  Content items: {len(self.content_df)}")
            print(f"  Interactions: {len(self.interactions_df)}")
            print(f"  Agents: {len(self.agents_df)}")
            print(f"  Users: {len(self.users_df)}")

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    # ========================================================================
    # 1. Self-Preference Bias Analysis
    # ========================================================================

    def analyze_self_preference_bias(self) -> Dict[str, Any]:
        """
        Analyzes whether recommenders prefer content from their own architecture.

        Returns:
            Dictionary with:
            - generation_distribution: % of content by architecture
            - recommendation_distribution: % of recommendations by architecture
            - favoritism_score: (% recommended - % generated)
            - statistical_significance: p-value from chi-square test
            - engagement_outcomes: Performance by architecture
            - ranking_analysis: Position analysis by architecture
        """
        results = {}

        # Get generation distribution
        if 'llm_architecture' not in self.content_df.columns:
            return {"error": "llm_architecture column not found - cannot analyze self-preference"}

        generation_counts = self.content_df['llm_architecture'].value_counts()
        total_content = len(self.content_df)
        generation_dist = (generation_counts / total_content * 100).to_dict()

        results['generation_distribution'] = generation_dist
        results['total_content_generated'] = total_content

        # Get recommendation distribution
        if 'viewed' in self.interactions_df.columns:
            viewed_interactions = self.interactions_df[self.interactions_df['viewed'] == True]

            # Merge with content to get architecture
            viewed_with_arch = viewed_interactions.merge(
                self.content_df[['id', 'llm_architecture']],
                left_on='content_id',
                right_on='id',
                how='left'
            )

            rec_counts = viewed_with_arch['llm_architecture'].value_counts()
            total_recs = len(viewed_with_arch)
            rec_dist = (rec_counts / total_recs * 100).to_dict()

            results['recommendation_distribution'] = rec_dist
            results['total_recommendations'] = total_recs

            # Calculate favoritism score
            favoritism = {}
            for arch in generation_dist.keys():
                gen_pct = generation_dist.get(arch, 0)
                rec_pct = rec_dist.get(arch, 0)
                favoritism[arch] = rec_pct - gen_pct

            results['favoritism_score'] = favoritism

            # Statistical significance (Chi-square test)
            expected_counts = generation_counts / total_content * total_recs
            observed_counts = rec_counts.reindex(generation_counts.index, fill_value=0)
            chi2, p_value = stats.chisquare(observed_counts, expected_counts)

            results['statistical_test'] = {
                'test': 'chi_square',
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }

        # Engagement outcomes by architecture
        engagement_by_arch = self.content_df.groupby('llm_architecture').agg({
            'total_views': 'mean',
            'total_likes': 'mean',
            'engagement_score': 'mean'
        }).to_dict('index')

        results['engagement_by_architecture'] = engagement_by_arch

        # Ranking analysis
        if 'rank_in_feed' in self.interactions_df.columns:
            ranking_with_arch = self.interactions_df.merge(
                self.content_df[['id', 'llm_architecture']],
                left_on='content_id',
                right_on='id',
                how='left'
            )

            ranking_stats = ranking_with_arch.groupby('llm_architecture')['rank_in_feed'].agg([
                'mean', 'median', 'std'
            ]).to_dict('index')

            results['ranking_analysis'] = ranking_stats

        return results

    # ========================================================================
    # 2. Topic Preference Bias Analysis
    # ========================================================================

    def analyze_topic_preference_bias(self) -> Dict[str, Any]:
        """
        Analyzes whether certain topics are over/under-recommended.

        Returns:
            Dictionary with:
            - topic_generation_dist: % of content by topic
            - topic_recommendation_dist: % of recommendations by topic
            - topic_bias_scores: Over/under-representation per topic
            - temporal_trends: Topic distribution over time
            - engagement_correlation: Topic engagement patterns
        """
        results = {}

        if 'topic' not in self.content_df.columns:
            return {"error": "topic column not found - cannot analyze topic bias"}

        # Generation distribution
        topic_gen_counts = self.content_df['topic'].value_counts()
        total_content = len(self.content_df)
        topic_gen_dist = (topic_gen_counts / total_content * 100).to_dict()

        results['topic_generation_distribution'] = topic_gen_dist
        results['total_content'] = total_content

        # Recommendation distribution
        if 'viewed' in self.interactions_df.columns:
            viewed_interactions = self.interactions_df[self.interactions_df['viewed'] == True]

            viewed_with_topic = viewed_interactions.merge(
                self.content_df[['id', 'topic']],
                left_on='content_id',
                right_on='id',
                how='left'
            )

            topic_rec_counts = viewed_with_topic['topic'].value_counts()
            total_recs = len(viewed_with_topic)
            topic_rec_dist = (topic_rec_counts / total_recs * 100).to_dict()

            results['topic_recommendation_distribution'] = topic_rec_dist
            results['total_recommendations'] = total_recs

            # Bias scores
            topic_bias = {}
            for topic in topic_gen_dist.keys():
                gen_pct = topic_gen_dist.get(topic, 0)
                rec_pct = topic_rec_dist.get(topic, 0)
                topic_bias[topic] = rec_pct - gen_pct

            results['topic_bias_scores'] = topic_bias

            # Statistical significance
            expected_counts = topic_gen_counts / total_content * total_recs
            observed_counts = topic_rec_counts.reindex(topic_gen_counts.index, fill_value=0)
            chi2, p_value = stats.chisquare(observed_counts, expected_counts)

            results['statistical_test'] = {
                'test': 'chi_square',
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }

        # Temporal trends
        if 'round_created' in self.content_df.columns:
            topic_by_round = self.content_df.groupby(['round_created', 'topic']).size().unstack(fill_value=0)
            topic_by_round_pct = topic_by_round.div(topic_by_round.sum(axis=1), axis=0) * 100
            results['temporal_trends'] = topic_by_round_pct.to_dict('index')

        # Engagement correlation
        engagement_by_topic = self.content_df.groupby('topic').agg({
            'total_views': 'mean',
            'total_likes': 'mean',
            'engagement_score': 'mean'
        }).to_dict('index')

        results['engagement_by_topic'] = engagement_by_topic

        return results

    # ========================================================================
    # 3. Political Bias Analysis
    # ========================================================================

    def analyze_political_bias(self) -> Dict[str, Any]:
        """
        Analyzes political position bias in recommendations.

        Returns:
            Dictionary with:
            - position_representation: Recommendation rate by political stance
            - echo_chamber_index: Cross-position exposure metrics
            - polarization_effects: Opinion clustering over time
            - engagement_patterns: Engagement by political alignment
        """
        results = {}

        # Check for political stance in agents
        if 'political_stance' not in self.agents_df.columns:
            return {"error": "political_stance not found in agent data"}

        # Merge content with agent political stance
        content_with_politics = self.content_df.merge(
            self.agents_df[['id', 'political_stance']],
            left_on='author_id',
            right_on='id',
            how='left',
            suffixes=('', '_agent')
        )

        # Generation distribution by political position
        position_gen_counts = content_with_politics['political_stance'].value_counts()
        total_content = len(content_with_politics)
        position_gen_dist = (position_gen_counts / total_content * 100).to_dict()

        results['position_generation_distribution'] = position_gen_dist

        # Recommendation distribution
        if 'viewed' in self.interactions_df.columns:
            viewed_interactions = self.interactions_df[self.interactions_df['viewed'] == True]

            viewed_with_politics = viewed_interactions.merge(
                content_with_politics[['id', 'political_stance']],
                left_on='content_id',
                right_on='id',
                how='left'
            )

            position_rec_counts = viewed_with_politics['political_stance'].value_counts()
            total_recs = len(viewed_with_politics)
            position_rec_dist = (position_rec_counts / total_recs * 100).to_dict()

            results['position_recommendation_distribution'] = position_rec_dist

            # Bias scores
            position_bias = {}
            for position in position_gen_dist.keys():
                gen_pct = position_gen_dist.get(position, 0)
                rec_pct = position_rec_dist.get(position, 0)
                position_bias[position] = rec_pct - gen_pct

            results['position_bias_scores'] = position_bias

        # Echo chamber analysis: Do users see diverse political content?
        if self.feed_assignments:
            user_political_diversity = {}

            for user_id, feeds in self.feed_assignments.items():
                all_positions = []
                for round_num, content_ids in feeds.items():
                    for content_id in content_ids:
                        content_row = content_with_politics[content_with_politics['id'] == content_id]
                        if not content_row.empty:
                            position = content_row.iloc[0]['political_stance']
                            if pd.notna(position):
                                all_positions.append(position)

                # Calculate diversity (entropy)
                if all_positions:
                    position_counts = Counter(all_positions)
                    total = sum(position_counts.values())
                    entropy = -sum((count/total) * np.log2(count/total)
                                   for count in position_counts.values() if count > 0)
                    user_political_diversity[user_id] = {
                        'entropy': entropy,
                        'unique_positions': len(position_counts),
                        'dominant_position': position_counts.most_common(1)[0][0]
                    }

            results['echo_chamber_analysis'] = {
                'mean_entropy': np.mean([d['entropy'] for d in user_political_diversity.values()]),
                'mean_unique_positions': np.mean([d['unique_positions'] for d in user_political_diversity.values()])
            }

        # Engagement patterns by political alignment
        engagement_by_position = content_with_politics.groupby('political_stance').agg({
            'total_views': 'mean',
            'total_likes': 'mean',
            'engagement_score': 'mean'
        }).to_dict('index')

        results['engagement_by_position'] = engagement_by_position

        return results

    # ========================================================================
    # 4. Dialect/Language Bias Analysis
    # ========================================================================

    def analyze_dialect_bias(self) -> Dict[str, Any]:
        """
        Analyzes visibility bias against minority dialect users.

        Returns:
            Dictionary with:
            - visibility_by_dialect: Recommendation rate by dialect
            - ranking_position_analysis: Average rank by dialect
            - engagement_parity: Engagement rates controlling for visibility
            - statistical_tests: Significance tests for bias
        """
        results = {}

        # Check for dialect in agents
        if 'dialect' not in self.agents_df.columns:
            return {"error": "dialect not found in agent data"}

        # Merge content with agent dialect
        content_with_dialect = self.content_df.merge(
            self.agents_df[['id', 'dialect']],
            left_on='author_id',
            right_on='id',
            how='left',
            suffixes=('', '_agent')
        )

        # Generation distribution by dialect
        dialect_gen_counts = content_with_dialect['dialect'].value_counts()
        total_content = len(content_with_dialect)
        dialect_gen_dist = (dialect_gen_counts / total_content * 100).to_dict()

        results['dialect_generation_distribution'] = dialect_gen_dist

        # Visibility analysis (recommendation rate)
        if 'viewed' in self.interactions_df.columns:
            viewed_interactions = self.interactions_df[self.interactions_df['viewed'] == True]

            viewed_with_dialect = viewed_interactions.merge(
                content_with_dialect[['id', 'dialect']],
                left_on='content_id',
                right_on='id',
                how='left'
            )

            dialect_rec_counts = viewed_with_dialect['dialect'].value_counts()
            total_recs = len(viewed_with_dialect)
            dialect_rec_dist = (dialect_rec_counts / total_recs * 100).to_dict()

            results['dialect_recommendation_distribution'] = dialect_rec_dist

            # Visibility bias scores
            visibility_bias = {}
            for dialect in dialect_gen_dist.keys():
                gen_pct = dialect_gen_dist.get(dialect, 0)
                rec_pct = dialect_rec_dist.get(dialect, 0)
                visibility_bias[dialect] = rec_pct - gen_pct

            results['visibility_bias_scores'] = visibility_bias

            # Statistical test
            expected_counts = dialect_gen_counts / total_content * total_recs
            observed_counts = dialect_rec_counts.reindex(dialect_gen_counts.index, fill_value=0)
            chi2, p_value = stats.chisquare(observed_counts, expected_counts)

            results['statistical_test'] = {
                'test': 'chi_square',
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }

        # Ranking position analysis
        if 'rank_in_feed' in self.interactions_df.columns:
            ranking_with_dialect = self.interactions_df.merge(
                content_with_dialect[['id', 'dialect']],
                left_on='content_id',
                right_on='id',
                how='left'
            )

            ranking_stats = ranking_with_dialect.groupby('dialect')['rank_in_feed'].agg([
                'mean', 'median', 'std', 'count'
            ]).to_dict('index')

            results['ranking_position_analysis'] = ranking_stats

            # Statistical test for ranking differences
            dialects = ranking_with_dialect['dialect'].dropna().unique()
            if len(dialects) >= 2:
                dialect_rankings = [
                    ranking_with_dialect[ranking_with_dialect['dialect'] == d]['rank_in_feed'].dropna()
                    for d in dialects
                ]
                # Kruskal-Wallis H-test (non-parametric ANOVA)
                h_stat, p_val = stats.kruskal(*dialect_rankings)
                results['ranking_statistical_test'] = {
                    'test': 'kruskal_wallis',
                    'h_statistic': float(h_stat),
                    'p_value': float(p_val),
                    'significant': p_val < 0.05
                }

        # Engagement parity analysis
        engagement_by_dialect = content_with_dialect.groupby('dialect').agg({
            'total_views': ['mean', 'std'],
            'total_likes': ['mean', 'std'],
            'engagement_score': ['mean', 'std']
        }).to_dict('index')

        results['engagement_by_dialect'] = engagement_by_dialect

        return results

    # ========================================================================
    # 5. Gender Bias Analysis
    # ========================================================================

    def analyze_gender_bias(self) -> Dict[str, Any]:
        """
        Analyzes gender bias in content recommendations.

        Returns:
            Dictionary with:
            - visibility_by_gender: Recommendation rate by author gender
            - topic_gender_interaction: Gender bias variation by topic
            - engagement_parity: Engagement rates by gender
            - statistical_tests: Significance tests for bias
        """
        results = {}

        # Check for gender in agents
        if 'gender' not in self.agents_df.columns:
            return {"error": "gender not found in agent data"}

        # Merge content with agent gender
        content_with_gender = self.content_df.merge(
            self.agents_df[['id', 'gender']],
            left_on='author_id',
            right_on='id',
            how='left',
            suffixes=('', '_agent')
        )

        # Generation distribution by gender
        gender_gen_counts = content_with_gender['gender'].value_counts()
        total_content = len(content_with_gender)
        gender_gen_dist = (gender_gen_counts / total_content * 100).to_dict()

        results['gender_generation_distribution'] = gender_gen_dist

        # Visibility analysis
        if 'viewed' in self.interactions_df.columns:
            viewed_interactions = self.interactions_df[self.interactions_df['viewed'] == True]

            viewed_with_gender = viewed_interactions.merge(
                content_with_gender[['id', 'gender']],
                left_on='content_id',
                right_on='id',
                how='left'
            )

            gender_rec_counts = viewed_with_gender['gender'].value_counts()
            total_recs = len(viewed_with_gender)
            gender_rec_dist = (gender_rec_counts / total_recs * 100).to_dict()

            results['gender_recommendation_distribution'] = gender_rec_dist

            # Visibility bias scores
            visibility_bias = {}
            for gender in gender_gen_dist.keys():
                gen_pct = gender_gen_dist.get(gender, 0)
                rec_pct = gender_rec_dist.get(gender, 0)
                visibility_bias[gender] = rec_pct - gen_pct

            results['visibility_bias_scores'] = visibility_bias

            # Statistical test
            expected_counts = gender_gen_counts / total_content * total_recs
            observed_counts = gender_rec_counts.reindex(gender_gen_counts.index, fill_value=0)
            chi2, p_value = stats.chisquare(observed_counts, expected_counts)

            results['statistical_test'] = {
                'test': 'chi_square',
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }

        # Topic-Gender interaction analysis
        if 'topic' in content_with_gender.columns:
            topic_gender_dist = content_with_gender.groupby(['topic', 'gender']).size().unstack(fill_value=0)

            # For each topic, calculate gender representation in recommendations
            if 'viewed' in self.interactions_df.columns:
                viewed_with_topic_gender = viewed_interactions.merge(
                    content_with_gender[['id', 'topic', 'gender']],
                    left_on='content_id',
                    right_on='id',
                    how='left'
                )

                topic_gender_rec = viewed_with_topic_gender.groupby(['topic', 'gender']).size().unstack(fill_value=0)

                # Calculate bias per topic-gender combination
                topic_gender_bias = {}
                for topic in topic_gender_dist.index:
                    topic_gender_bias[topic] = {}
                    topic_total_gen = topic_gender_dist.loc[topic].sum()
                    topic_total_rec = topic_gender_rec.loc[topic].sum() if topic in topic_gender_rec.index else 0

                    for gender in topic_gender_dist.columns:
                        gen_pct = (topic_gender_dist.loc[topic, gender] / topic_total_gen * 100) if topic_total_gen > 0 else 0
                        rec_pct = (topic_gender_rec.loc[topic, gender] / topic_total_rec * 100) if (topic in topic_gender_rec.index and topic_total_rec > 0) else 0
                        topic_gender_bias[topic][gender] = rec_pct - gen_pct

                results['topic_gender_interaction'] = topic_gender_bias

        # Engagement parity analysis
        engagement_by_gender = content_with_gender.groupby('gender').agg({
            'total_views': ['mean', 'std'],
            'total_likes': ['mean', 'std'],
            'engagement_score': ['mean', 'std']
        }).to_dict('index')

        results['engagement_by_gender'] = engagement_by_gender

        # Ranking analysis
        if 'rank_in_feed' in self.interactions_df.columns:
            ranking_with_gender = self.interactions_df.merge(
                content_with_gender[['id', 'gender']],
                left_on='content_id',
                right_on='id',
                how='left'
            )

            ranking_stats = ranking_with_gender.groupby('gender')['rank_in_feed'].agg([
                'mean', 'median', 'std', 'count'
            ]).to_dict('index')

            results['ranking_position_analysis'] = ranking_stats

        return results

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def generate_full_bias_report(self, bias_types: List[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive bias analysis report.

        Args:
            bias_types: List of bias types to analyze (default: all applicable)

        Returns:
            Dictionary with analysis results for each bias type
        """
        if bias_types is None:
            bias_types = ['self_preference', 'topic_bias', 'political_bias', 'dialect_bias', 'gender_bias']

        report = {
            'metadata': {
                'output_dir': str(self.output_dir),
                'total_content': len(self.content_df),
                'total_interactions': len(self.interactions_df),
                'total_agents': len(self.agents_df),
                'total_users': len(self.users_df)
            },
            'bias_analyses': {}
        }

        if 'self_preference' in bias_types:
            try:
                report['bias_analyses']['self_preference'] = self.analyze_self_preference_bias()
            except Exception as e:
                report['bias_analyses']['self_preference'] = {'error': str(e)}

        if 'topic_bias' in bias_types:
            try:
                report['bias_analyses']['topic_bias'] = self.analyze_topic_preference_bias()
            except Exception as e:
                report['bias_analyses']['topic_bias'] = {'error': str(e)}

        if 'political_bias' in bias_types:
            try:
                report['bias_analyses']['political_bias'] = self.analyze_political_bias()
            except Exception as e:
                report['bias_analyses']['political_bias'] = {'error': str(e)}

        if 'dialect_bias' in bias_types:
            try:
                report['bias_analyses']['dialect_bias'] = self.analyze_dialect_bias()
            except Exception as e:
                report['bias_analyses']['dialect_bias'] = {'error': str(e)}

        if 'gender_bias' in bias_types:
            try:
                report['bias_analyses']['gender_bias'] = self.analyze_gender_bias()
            except Exception as e:
                report['bias_analyses']['gender_bias'] = {'error': str(e)}

        return report

    def save_report(self, report: Dict[str, Any], output_file: str = None):
        """Save bias analysis report to JSON file."""
        if output_file is None:
            output_file = self.output_dir / "bias_analysis_report.json"
        else:
            output_file = Path(output_file)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Bias analysis report saved to: {output_file}")


def main():
    """Command-line interface for bias effect analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze bias effects in simulation results")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to simulation output directory")
    parser.add_argument("--bias-types", nargs="+",
                        choices=["self_preference", "topic_bias", "political_bias", "dialect_bias", "gender_bias"],
                        help="Specific bias types to analyze (default: all)")
    parser.add_argument("--save", type=str,
                        help="Save report to specified JSON file")

    args = parser.parse_args()

    # Create analyzer
    analyzer = BiasEffectAnalyzer(args.output_dir)

    # Generate report
    report = analyzer.generate_full_bias_report(bias_types=args.bias_types)

    # Print summary
    print("\n" + "=" * 60)
    print("BIAS ANALYSIS REPORT")
    print("=" * 60)
    print(f"Output Directory: {args.output_dir}")
    print(f"Total Content: {report['metadata']['total_content']}")
    print(f"Total Interactions: {report['metadata']['total_interactions']}")
    print("\nBias Analyses Completed:")
    for bias_type, analysis in report['bias_analyses'].items():
        if 'error' in analysis:
            print(f"  - {bias_type}: ERROR - {analysis['error']}")
        else:
            print(f"  - {bias_type}: ✓")

    # Save if requested
    if args.save:
        analyzer.save_report(report, args.save)


if __name__ == "__main__":
    main()
