#!/usr/bin/env python3
"""
Simulation Log Analyzer
Tools for analyzing and visualizing the detailed interaction logs.
"""

import json
from collections import Counter, defaultdict
from typing import List, Dict
import argparse


class LogAnalyzer:
    """Analyze simulation logs to extract insights."""
    
    def __init__(self, log_file: str = "/mnt/user-data/outputs/simulation_log.jsonl"):
        self.log_file = log_file
        self.events = []
        self.load_logs()
    
    def load_logs(self):
        """Load logs from JSONL file."""
        self.events = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self.events.append(json.loads(line))
            print(f"Loaded {len(self.events)} events from {self.log_file}")
        except FileNotFoundError:
            print(f"Log file not found: {self.log_file}")
        except Exception as e:
            print(f"Error loading logs: {e}")
    
    def get_event_types(self) -> Counter:
        """Get count of each event type."""
        return Counter(event['event_type'] for event in self.events)
    
    def get_events_by_type(self, event_type: str) -> List[Dict]:
        """Get all events of a specific type."""
        return [e for e in self.events if e['event_type'] == event_type]
    
    def get_events_by_round(self, round_num: int) -> List[Dict]:
        """Get all events from a specific round."""
        return [e for e in self.events 
                if e.get('data', {}).get('round') == round_num or 
                   e.get('data', {}).get('round_number') == round_num]
    
    def analyze_content_generation(self) -> Dict:
        """Analyze content generation patterns."""
        content_events = self.get_events_by_type("CONTENT_GENERATED")
        
        analysis = {
            'total_content': len(content_events),
            'by_type': Counter(e['data'].get('type', 'unknown') for e in content_events),
            'by_author': Counter(e['data'].get('author_id') for e in content_events),
            'by_topic': Counter(e['data'].get('topic') for e in content_events),
        }
        
        return analysis
    
    def analyze_opinion_dynamics(self) -> Dict:
        """Analyze how opinions evolved."""
        opinion_events = self.get_events_by_type("OPINION_UPDATED")
        
        # Track opinion changes by round
        changes_by_round = defaultdict(list)
        for event in opinion_events:
            round_num = event['data'].get('round', 0)
            change = abs(event['data'].get('change', 0))
            changes_by_round[round_num].append(change)
        
        # Calculate average change per round
        avg_changes = {
            round_num: sum(changes) / len(changes) if changes else 0
            for round_num, changes in changes_by_round.items()
        }
        
        analysis = {
            'total_updates': len(opinion_events),
            'avg_change_by_round': avg_changes,
            'users_with_changes': len(set(e['data'].get('user_id') for e in opinion_events)),
        }
        
        return analysis
    
    def analyze_recommendations(self) -> Dict:
        """Analyze recommendation patterns."""
        feed_events = self.get_events_by_type("FEED_ASSIGNED")
        
        # Track what topics were recommended to users with different interests
        recommendations_by_interest = defaultdict(lambda: Counter())
        
        for event in feed_events:
            user_interests = event['data'].get('user_interests', [])
            feed_items = event['data'].get('feed_items', [])
            
            for interest in user_interests:
                for item in feed_items:
                    topic = item.get('topic')
                    if topic:
                        recommendations_by_interest[interest][topic] += 1
        
        analysis = {
            'total_feeds_assigned': len(feed_events),
            'recommendations_by_interest': dict(recommendations_by_interest),
        }
        
        return analysis
    
    def analyze_engagement(self) -> Dict:
        """Analyze engagement patterns."""
        engagement_events = self.get_events_by_type("ENGAGEMENT_RECORDED")
        
        # Track engagement by content
        engagement_by_content = defaultdict(float)
        for event in engagement_events:
            content_id = event['data'].get('content_id')
            change = event['data'].get('change', 0)
            engagement_by_content[content_id] += change
        
        # Get top engaged content
        sorted_engagement = sorted(engagement_by_content.items(), 
                                  key=lambda x: x[1], reverse=True)
        
        analysis = {
            'total_engagement_events': len(engagement_events),
            'unique_content_engaged': len(engagement_by_content),
            'top_engaged_content': sorted_engagement[:10] if sorted_engagement else [],
            'total_engagement': sum(engagement_by_content.values()),
        }
        
        return analysis
    
    def generate_report(self, output_file: str = "/mnt/user-data/outputs/log_analysis_report.txt"):
        """Generate comprehensive analysis report."""
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("SIMULATION LOG ANALYSIS REPORT")
        report_lines.append("="*70)
        report_lines.append("")
        
        # Event types summary
        report_lines.append("EVENT TYPES SUMMARY")
        report_lines.append("-"*70)
        event_types = self.get_event_types()
        for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"  {event_type}: {count}")
        report_lines.append("")
        
        # Content generation analysis
        report_lines.append("CONTENT GENERATION ANALYSIS")
        report_lines.append("-"*70)
        content_analysis = self.analyze_content_generation()
        report_lines.append(f"Total content items: {content_analysis['total_content']}")
        report_lines.append("\nBy type:")
        for content_type, count in content_analysis['by_type'].items():
            report_lines.append(f"  {content_type}: {count}")
        report_lines.append("\nBy author:")
        for author, count in sorted(content_analysis['by_author'].items(), 
                                    key=lambda x: x[1], reverse=True):
            report_lines.append(f"  {author}: {count}")
        report_lines.append("\nBy topic:")
        for topic, count in content_analysis['by_topic'].items():
            report_lines.append(f"  {topic}: {count}")
        report_lines.append("")
        
        # Opinion dynamics analysis
        report_lines.append("OPINION DYNAMICS ANALYSIS")
        report_lines.append("-"*70)
        opinion_analysis = self.analyze_opinion_dynamics()
        report_lines.append(f"Total opinion updates: {opinion_analysis['total_updates']}")
        report_lines.append(f"Users with opinion changes: {opinion_analysis['users_with_changes']}")
        report_lines.append("\nAverage opinion change by round:")
        for round_num, avg_change in sorted(opinion_analysis['avg_change_by_round'].items()):
            report_lines.append(f"  Round {round_num}: {avg_change:.4f}")
        report_lines.append("")
        
        # Recommendation analysis
        report_lines.append("RECOMMENDATION ANALYSIS")
        report_lines.append("-"*70)
        rec_analysis = self.analyze_recommendations()
        report_lines.append(f"Total feeds assigned: {rec_analysis['total_feeds_assigned']}")
        report_lines.append("\nRecommendations by user interest:")
        for interest, topics in rec_analysis['recommendations_by_interest'].items():
            report_lines.append(f"\n  Interest: {interest}")
            for topic, count in topics.most_common(3):
                report_lines.append(f"    → {topic}: {count} recommendations")
        report_lines.append("")
        
        # Engagement analysis
        report_lines.append("ENGAGEMENT ANALYSIS")
        report_lines.append("-"*70)
        engagement_analysis = self.analyze_engagement()
        report_lines.append(f"Total engagement events: {engagement_analysis['total_engagement_events']}")
        report_lines.append(f"Unique content engaged: {engagement_analysis['unique_content_engaged']}")
        report_lines.append(f"Total engagement score: {engagement_analysis['total_engagement']:.2f}")
        report_lines.append("\nTop 5 most engaged content:")
        for content_id, engagement in engagement_analysis['top_engaged_content'][:5]:
            report_lines.append(f"  {content_id[:12]}: {engagement:.2f}")
        report_lines.append("")
        
        report_lines.append("="*70)
        
        # Write report
        report_text = "\n".join(report_lines)
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"\nAnalysis report saved to: {output_file}")
        print("\n" + report_text)
        
        return report_text
    
    def export_timeline(self, output_file: str = "/mnt/user-data/outputs/event_timeline.txt"):
        """Export a human-readable timeline of events."""
        with open(output_file, 'w') as f:
            f.write("SIMULATION EVENT TIMELINE\n")
            f.write("="*70 + "\n\n")
            
            for i, event in enumerate(self.events, 1):
                timestamp = event.get('timestamp', 'unknown')
                event_type = event.get('event_type', 'unknown')
                summary = event.get('data', {}).get('summary', 'No summary')
                
                f.write(f"[{i:4d}] {timestamp}\n")
                f.write(f"       Type: {event_type}\n")
                f.write(f"       {summary}\n")
                f.write("\n")
        
        print(f"Event timeline saved to: {output_file}")
    
    def filter_logs(self, event_types: List[str] = None, 
                   round_num: int = None,
                   user_id: str = None,
                   agent_id: str = None,
                   output_file: str = "/mnt/user-data/outputs/filtered_log.json"):
        """Filter logs based on criteria and save to file."""
        filtered_events = self.events.copy()
        
        if event_types:
            filtered_events = [e for e in filtered_events 
                             if e['event_type'] in event_types]
        
        if round_num is not None:
            filtered_events = [e for e in filtered_events 
                             if e.get('data', {}).get('round') == round_num or
                                e.get('data', {}).get('round_number') == round_num]
        
        if user_id:
            filtered_events = [e for e in filtered_events 
                             if e.get('data', {}).get('user_id') == user_id]
        
        if agent_id:
            filtered_events = [e for e in filtered_events 
                             if e.get('data', {}).get('agent_id') == agent_id or
                                e.get('data', {}).get('author_id') == agent_id]
        
        with open(output_file, 'w') as f:
            json.dump(filtered_events, f, indent=2)
        
        print(f"Filtered {len(filtered_events)} events to: {output_file}")
        return filtered_events


def main():
    parser = argparse.ArgumentParser(description="Analyze simulation logs")
    parser.add_argument("--logfile", type=str, 
                       default="/mnt/user-data/outputs/simulation_log.jsonl",
                       help="Path to log file (JSONL format)")
    parser.add_argument("--report", action="store_true",
                       help="Generate analysis report")
    parser.add_argument("--timeline", action="store_true",
                       help="Export event timeline")
    parser.add_argument("--filter-type", nargs="+",
                       help="Filter by event types")
    parser.add_argument("--filter-round", type=int,
                       help="Filter by round number")
    parser.add_argument("--filter-user", type=str,
                       help="Filter by user ID")
    parser.add_argument("--filter-agent", type=str,
                       help="Filter by agent ID")
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(log_file=args.logfile)
    
    if not analyzer.events:
        print("No events loaded. Make sure the log file exists and is valid.")
        return
    
    # Default: generate report if no specific action requested
    if not any([args.report, args.timeline, args.filter_type, 
               args.filter_round, args.filter_user, args.filter_agent]):
        args.report = True
    
    if args.report:
        analyzer.generate_report()
    
    if args.timeline:
        analyzer.export_timeline()
    
    if any([args.filter_type, args.filter_round, args.filter_user, args.filter_agent]):
        analyzer.filter_logs(
            event_types=args.filter_type,
            round_num=args.filter_round,
            user_id=args.filter_user,
            agent_id=args.filter_agent
        )
    
    # Print event type summary
    print("\n" + "="*70)
    print("EVENT TYPES SUMMARY")
    print("="*70)
    event_types = analyzer.get_event_types()
    for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
        print(f"{event_type:35s}: {count:4d}")


if __name__ == "__main__":
    main()