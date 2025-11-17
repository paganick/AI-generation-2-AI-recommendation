#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
Run this before running the full simulation.
"""

print("Testing imports...")
print("-" * 60)

try:
    print("1. Testing llm_backend import...")
    from llm_backend import get_llm_backend, MockLLMBackend, LlamaBackend
    print("   ✓ llm_backend imported successfully")
except Exception as e:
    print(f"   ✗ Error importing llm_backend: {e}")
    exit(1)

try:
    print("2. Testing simulation_framework_v2 import...")
    from simulation_framework_v2 import (
        Agent, SimulatedUser, ContentItem, UserContentInteraction,
        ContentGenerator, EnhancedDataTracker, add_sentiment_scores, save_results
    )
    print("   ✓ simulation_framework_v2 imported successfully")
except Exception as e:
    print(f"   ✗ Error importing simulation_framework_v2: {e}")
    exit(1)

try:
    print("3. Testing llm_recommender import...")
    from llm_recommender import LLMRecommender, EngagementModel
    print("   ✓ llm_recommender imported successfully")
except Exception as e:
    print(f"   ✗ Error importing llm_recommender: {e}")
    exit(1)

print("-" * 60)
print("All imports successful! ✓")
print("\nYou can now run:")
print("  python run_simulation_v2.py --mock --rounds 2 --agents 3 --users 5")