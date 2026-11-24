#!/usr/bin/env python3
"""
Test latent MAS strategies with complex agent configurations.
Tests: more agents, different roles, longer prompts, complex tasks.
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_with_more_agents(strategy):
    """Test with 4+ agents instead of 2-3."""
    print(f"\nTest: More agents (4) with {strategy}")
    print("-" * 60)
    
    payload = {
        "model_id": "gemma-4-E4B-it-Q4_K_S",
        "task": "Plan and execute a project",
        "roles": ["planner", "executor", "reviewer", "tester"],
        "max_tokens": 100,
        "shared_prompt_kv": True,
        "create_temporary_agents": True,
        "destroy_temporary_agents": True,
        "system_prompts": {
            "planner": "You are a planning agent.",
            "executor": "You are an execution agent.",
            "reviewer": "You are a reviewer agent.",
            "tester": "You are a tester agent."
        },
        "strategy": strategy,
        "adapter_type": "recursive",
        "type_k": "turbo4",
        "type_v": "turbo4"
    }
    
    for i in range(3):
        try:
            response = requests.post(f"{BASE_URL}/mas/run", json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                response_text = json.dumps(result)
                if "decode/init failure" in response_text:
                    print(f"  Retry {i+1}: ✗ decode/init failure")
                    return False
                print(f"  Retry {i+1}: ✓")
            else:
                if "decode/init failure" in response.text:
                    print(f"  Retry {i+1}: ✗ decode/init failure")
                    return False
                print(f"  Retry {i+1}: ✗ HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"  Retry {i+1}: ✗ {e}")
            return False
        time.sleep(0.5)
    
    print("✓ All retries succeeded with 4 agents")
    return True

def test_with_long_prompt(strategy):
    """Test with longer, more complex prompts."""
    print(f"\nTest: Long complex prompt with {strategy}")
    print("-" * 60)
    
    long_task = """
    You are working on a complex software project. Please analyze the following requirements:
    
    1. The system must handle real-time data processing
    2. It needs to support multiple concurrent users
    3. Security is a critical requirement
    4. The system should be scalable to handle growth
    5. Performance must be optimized for low latency
    
    Please provide a detailed analysis and implementation plan for each requirement.
    Consider trade-offs and provide recommendations for the best approach.
    """
    
    payload = {
        "model_id": "gemma-4-E4B-it-Q4_K_S",
        "task": long_task,
        "roles": ["analyst", "architect"],
        "max_tokens": 150,
        "shared_prompt_kv": True,
        "create_temporary_agents": True,
        "destroy_temporary_agents": True,
        "system_prompts": {
            "analyst": "You are a business analyst.",
            "architect": "You are a system architect."
        },
        "strategy": strategy,
        "adapter_type": "recursive",
        "type_k": "turbo4",
        "type_v": "turbo4"
    }
    
    for i in range(3):
        try:
            response = requests.post(f"{BASE_URL}/mas/run", json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                response_text = json.dumps(result)
                if "decode/init failure" in response_text:
                    print(f"  Retry {i+1}: ✗ decode/init failure")
                    return False
                print(f"  Retry {i+1}: ✓")
            else:
                if "decode/init failure" in response.text:
                    print(f"  Retry {i+1}: ✗ decode/init failure")
                    return False
                print(f"  Retry {i+1}: ✗ HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"  Retry {i+1}: ✗ {e}")
            return False
        time.sleep(0.5)
    
    print("✓ All retries succeeded with long prompt")
    return True

def test_with_different_adapter_types(strategy):
    """Test with different adapter types."""
    print(f"\nTest: Different adapter types with {strategy}")
    print("-" * 60)
    
    adapter_types = ["recursive", "cross", "mean"]
    
    for adapter_type in adapter_types:
        payload = {
            "model_id": "gemma-4-E4B-it-Q4_K_S",
            "task": "Test task",
            "roles": ["agent1", "agent2"],
            "max_tokens": 80,
            "shared_prompt_kv": True,
            "create_temporary_agents": True,
            "destroy_temporary_agents": True,
            "system_prompts": {
                "agent1": "You are agent 1.",
                "agent2": "You are agent 2."
            },
            "strategy": strategy,
            "adapter_type": adapter_type,
            "type_k": "turbo4",
            "type_v": "turbo4"
        }
        
        try:
            response = requests.post(f"{BASE_URL}/mas/run", json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                response_text = json.dumps(result)
                if "decode/init failure" in response_text:
                    print(f"  adapter={adapter_type}: ✗ decode/init failure")
                    return False
                print(f"  adapter={adapter_type}: ✓")
            else:
                if "decode/init failure" in response.text:
                    print(f"  adapter={adapter_type}: ✗ decode/init failure")
                    return False
                print(f"  adapter={adapter_type}: ✗ HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"  adapter={adapter_type}: ✗ {e}")
            return False
        time.sleep(0.3)
    
    print("✓ All adapter types succeeded")
    return True

def test_mixed_strategies_in_sequence():
    """Test switching between different strategies in rapid sequence."""
    print(f"\nTest: Mixed strategies in rapid sequence")
    print("-" * 60)
    
    strategies = ["latent_mas", "latent_mas_no_align", "latent_mas_embeddings", "chain", "broadcast"]
    
    for strategy in strategies:
        payload = {
            "model_id": "gemma-4-E4B-it-Q4_K_S",
            "task": "Quick test",
            "roles": ["agent1", "agent2"],
            "max_tokens": 50,
            "shared_prompt_kv": True,
            "create_temporary_agents": True,
            "destroy_temporary_agents": True,
            "system_prompts": {
                "agent1": "You are agent 1.",
                "agent2": "You are agent 2."
            },
            "strategy": strategy,
            "adapter_type": "recursive",
            "type_k": "turbo4",
            "type_v": "turbo4"
        }
        
        try:
            response = requests.post(f"{BASE_URL}/mas/run", json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                response_text = json.dumps(result)
                if "decode/init failure" in response_text:
                    print(f"  strategy={strategy}: ✗ decode/init failure")
                    return False
                print(f"  strategy={strategy}: ✓")
            else:
                if "decode/init failure" in response.text:
                    print(f"  strategy={strategy}: ✗ decode/init failure")
                    return False
                print(f"  strategy={strategy}: ✗ HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"  strategy={strategy}: ✗ {e}")
            return False
        # No sleep - rapid sequence
    
    print("✓ All strategies succeeded in rapid sequence")
    return True

def main():
    """Run all complex configuration tests."""
    strategies = ["latent_mas", "latent_mas_no_align", "latent_mas_embeddings"]
    
    print("=" * 60)
    print("COMPLEX CONFIGURATION TESTS")
    print("=" * 60)
    
    all_results = {}
    
    for strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"STRATEGY: {strategy}")
        print(f"{'=' * 60}")
        
        results = {
            "more_agents": test_with_more_agents(strategy),
            "long_prompt": test_with_long_prompt(strategy),
            "different_adapters": test_with_different_adapter_types(strategy)
        }
        
        all_results[strategy] = results
        
        print(f"\nSummary for {strategy}:")
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {test_name}: {status}")
    
    # Test mixed strategies
    print(f"\n{'=' * 60}")
    print("MIXED STRATEGY SEQUENCE TEST")
    print(f"{'=' * 60}")
    mixed_result = test_mixed_strategies_in_sequence()
    
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    
    for strategy, results in all_results.items():
        all_passed = all(results.values())
        status = "✓ ALL PASS" if all_passed else "✗ SOME FAIL"
        print(f"  {strategy}: {status}")
    
    print(f"  mixed_sequence: {'✓ PASS' if mixed_result else '✗ FAIL'}")
    
    # Check if all tests passed
    total_passed = sum(sum(results.values()) for results in all_results.values()) + (1 if mixed_result else 0)
    total_tests = sum(len(results) for results in all_results.values()) + 1
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✓✓✓ ALL COMPLEX CONFIGURATION TESTS PASSED ✓✓✓")
        return True
    else:
        print("\n✗ Some tests failed")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Test script error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
