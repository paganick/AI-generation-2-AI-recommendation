#!/usr/bin/env python3
"""
Analysis Script for Simulation Data
Simple statistics and visualizations of the simulation results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


class SimulationAnalyzer:
    """Analyze simulation data and generate statistics."""
    
    def __init__(self, data_dir: str = "./simulation_data"):
        self.data_dir = Path(data_dir)
        self.load_data()
    
    def load_data(self):
        """Load all simulation data."""
        print(f"Loading data from {self.data_dir}...")
        
        self.content_df = pd.read_csv(self.data_dir / "all_content.csv")
        self.interactions_df = pd.read_csv(self.data_dir / "all_interactions.csv")
        self.agents_df = pd.read_csv(self.data_dir / "agents.csv")
        self.users_df = pd.read_csv(self.data_dir / "users.csv")
        
        with open(self.data_dir / "feed_assignments.json") as f:
            self.feed_assignments = json.load(f)
        
        with open(self.data_dir / "round_data.json") as f:
            self.round_data = json.load(f)
        
        print(f"  ✓ Loaded {len(self.content_df)} content items")
        print(f"  ✓ Loaded {len(self.interactions_df)} interactions")
        print(f"  ✓ Loaded {len(self.agents_df)} agents")
        print(f"  ✓ Loaded {len(self.users_df)} users")
    
    def content_statistics(self):
        """Print content statistics."""
        print("\n" + "="*80)
        print("CONTENT STATISTICS")
        print("="*80)
        
        print(f"\nTotal content items: {len(self.content_df)}")
        print(f"  By round:")
        for round_num in sorted(self.content_df['round_created'].unique()):
            count = len(self.content_df[self.content_df['round_created'] == round_num])
            print(f"    Round {round_num}: {count} items")
        
        print(f"\n  By type:")
        n_initial = len(self.content_df[self.content_df['parent_id'].isna()])
        n_responses = len(self.content_df[self.content_df['parent_id'].notna()])
        print(f"    Initial posts: {n_initial}")
        print(f"    Responses: {n_responses}")
        
        print(f"\n  By topic:")
        topic_counts = self.content_df['topic'].value_counts()
        for topic, count in topic_counts.items():
            print(f"    {topic}: {count}")
        
        print(f"\n  By author:")
        author_counts = self.content_df['author_id'].value_counts()
        for author, count in author_counts.head(6).items():
            agent_info = self.agents_df[self.agents_df['id'] == author].iloc[0]
            print(f"    {author} ({agent_info['persona']}): {count}")
        
        print(f"\n  Engagement:")
        print(f"    Total views: {self.content_df['total_views'].sum()}")
        print(f"    Total likes: {self.content_df['total_likes'].sum()}")
        print(f"    Avg views per item: {self.content_df['total_views'].mean():.2f}")
        print(f"    Avg likes per item: {self.content_df['total_likes'].mean():.2f}")
        
        print(f"\n  Sentiment:")
        print(f"    Mean sentiment: {self.content_df['sentiment_score'].mean():.3f}")
        print(f"    Sentiment std: {self.content_df['sentiment_score'].std():.3f}")
        print(f"    Min sentiment: {self.content_df['sentiment_score'].min():.3f}")
        print(f"    Max sentiment: {self.content_df['sentiment_score'].max():.3f}")
    
    def interaction_statistics(self):
        """Print interaction statistics."""
        print("\n" + "="*80)
        print("INTERACTION STATISTICS")
        print("="*80)
        
        print(f"\nTotal interactions: {len(self.interactions_df)}")
        
        print(f"\n  By round:")
        for round_num in sorted(self.interactions_df['round'].unique()):
            count = len(self.interactions_df[self.interactions_df['round'] == round_num])
            likes = self.interactions_df[self.interactions_df['round'] == round_num]['liked'].sum()
            print(f"    Round {round_num}: {count} interactions, {likes} likes")
        
        print(f"\n  Likes:")
        total_likes = self.interactions_df['liked'].sum()
        like_rate = total_likes / len(self.interactions_df)
        print(f"    Total likes: {total_likes}")
        print(f"    Like rate: {like_rate:.1%}")
        
        print(f"\n  Per user:")
        user_interactions = self.interactions_df.groupby('user_id').size()
        user_likes = self.interactions_df.groupby('user_id')['liked'].sum()
        print(f"    Avg interactions per user: {user_interactions.mean():.1f}")
        print(f"    Avg likes per user: {user_likes.mean():.1f}")
        
        print(f"\n  Opinion changes:")
        self.interactions_df['opinion_change'] = \
            self.interactions_df['opinion_after'] - self.interactions_df['opinion_before']
        print(f"    Mean opinion change: {self.interactions_df['opinion_change'].mean():.4f}")
        print(f"    Max opinion change: {self.interactions_df['opinion_change'].abs().max():.4f}")
    
    def user_statistics(self):
        """Print user statistics."""
        print("\n" + "="*80)
        print("USER STATISTICS")
        print("="*80)
        
        print(f"\nTotal users: {len(self.users_df)}")
        
        print(f"\n  Final opinions:")
        opinions = self.users_df['final_opinion']
        print(f"    Mean: {opinions.mean():.3f}")
        print(f"    Std: {opinions.std():.3f}")
        print(f"    Min: {opinions.min():.3f}")
        print(f"    Max: {opinions.max():.3f}")
        
        print(f"\n  By interest group:")
        # Extract first interest
        self.users_df['primary_interest'] = self.users_df['interests'].apply(
            lambda x: eval(x)[0] if isinstance(x, str) else x[0]
        )
        for interest in self.users_df['primary_interest'].unique():
            users = self.users_df[self.users_df['primary_interest'] == interest]
            print(f"    {interest}: {len(users)} users, "
                  f"avg opinion: {users['final_opinion'].mean():.3f}")
    
    def opinion_evolution(self):
        """Analyze how opinions evolved over rounds."""
        print("\n" + "="*80)
        print("OPINION EVOLUTION")
        print("="*80)
        
        print(f"\n  By round:")
        for round_num in sorted([int(r) for r in self.round_data['user_opinions'].keys()]):
            opinions = list(self.round_data['user_opinions'][str(round_num)].values())
            print(f"    Round {round_num}: mean={np.mean(opinions):.3f}, "
                  f"std={np.std(opinions):.3f}, "
                  f"variance={np.var(opinions):.4f}")
    
    def generate_report(self, output_file: str = "./analysis_report.txt"):
        """Generate complete text report."""
        import sys
        original_stdout = sys.stdout
        
        with open(output_file, 'w') as f:
            sys.stdout = f
            self.content_statistics()
            self.interaction_statistics()
            self.user_statistics()
            self.opinion_evolution()
        
        sys.stdout = original_stdout
        print(f"\n✓ Report saved to {output_file}")
    
    def plot_opinion_evolution(self, output_file: str = "./opinion_evolution.png"):
        """Plot how user opinions evolved over time."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Individual user opinion trajectories
        rounds = sorted([int(r) for r in self.round_data['user_opinions'].keys()])
        for user_id in list(self.round_data['user_opinions'][str(rounds[0])].keys())[:10]:
            opinions = [self.round_data['user_opinions'][str(r)][user_id] for r in rounds]
            ax1.plot(rounds, opinions, marker='o', alpha=0.6, label=user_id[:6])
        
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Opinion', fontsize=12)
        ax1.set_title('User Opinion Trajectories (first 10 users)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Plot 2: Opinion distribution statistics
        means = []
        stds = []
        for round_num in rounds:
            opinions = list(self.round_data['user_opinions'][str(round_num)].values())
            means.append(np.mean(opinions))
            stds.append(np.std(opinions))
        
        ax2.plot(rounds, means, marker='o', linewidth=2, label='Mean Opinion')
        ax2.fill_between(rounds, 
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.3, label='±1 Std Dev')
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Opinion', fontsize=12)
        ax2.set_title('Opinion Statistics Over Time', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {output_file}")
        plt.close()
    
    def plot_content_engagement(self, output_file: str = "./content_engagement.png"):
        """Plot content engagement patterns."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Views vs Likes
        ax1.scatter(self.content_df['total_views'], 
                   self.content_df['total_likes'],
                   alpha=0.6, s=50)
        ax1.set_xlabel('Total Views', fontsize=12)
        ax1.set_ylabel('Total Likes', fontsize=12)
        ax1.set_title('Content Engagement: Views vs Likes', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Engagement by topic
        topic_engagement = self.content_df.groupby('topic').agg({
            'total_views': 'sum',
            'total_likes': 'sum'
        })
        
        x = np.arange(len(topic_engagement))
        width = 0.35
        ax2.bar(x - width/2, topic_engagement['total_views'], width, 
               label='Views', alpha=0.8)
        ax2.bar(x + width/2, topic_engagement['total_likes'], width,
               label='Likes', alpha=0.8)
        ax2.set_xlabel('Topic', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Engagement by Topic', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(topic_engagement.index, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {output_file}")
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze simulation results")
    parser.add_argument("--data-dir", type=str, default="./simulation_data",
                       help="Directory with simulation data")
    parser.add_argument("--report", action="store_true",
                       help="Generate text report")
    parser.add_argument("--plots", action="store_true",
                       help="Generate plots")
    parser.add_argument("--all", action="store_true",
                       help="Generate everything")
    
    args = parser.parse_args()
    
    # Default: print statistics
    if not any([args.report, args.plots, args.all]):
        args.report = True
        args.plots = True
    
    if args.all:
        args.report = True
        args.plots = True
    
    try:
        analyzer = SimulationAnalyzer(data_dir=args.data_dir)
        
        # Print to console
        analyzer.content_statistics()
        analyzer.interaction_statistics()
        analyzer.user_statistics()
        analyzer.opinion_evolution()
        
        if args.report:
            analyzer.generate_report()
        
        if args.plots:
            analyzer.plot_opinion_evolution()
            analyzer.plot_content_engagement()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files in {args.data_dir}")
        print(f"Make sure you've run the simulation first!")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()