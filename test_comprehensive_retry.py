#!/usr/bin/env python3
"""
Comprehensive test for latent MAS retry fix.
Tests: more retries, different prompts, rapid successive calls, edge cases.
"""

import requests
import time
import json
import threading

BASE_URL = "http://localhost:8000"

def test_single_request(strategy, task, max_tokens=100):
    """Test a single request."""
    payload = {
        "model_id": "gemma-4-E4B-it-Q4_K_S",
        "task": task,
        "roles": ["planner", "executor"],
        "max_tokens": max_tokens,
        "shared_prompt_kv": True,
        "create_temporary_agents": True,
        "destroy_temporary_agents": True,
        "system_prompts": {
            "planner": "You are a planning agent.",
            "executor": "You are an execution agent."
        },
        "strategy": strategy,
        "adapter_type": "recursive",
        "type_k": "turbo4",
        "type_v": "turbo4"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/mas/run",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = json.dumps(result)
            if "decode/init failure" in response_text:
                return False, "decode/init failure in response"
            return True, result.get('time_ms', 'N/A')
        else:
            error_text = response.text
            if "decode/init failure" in error_text:
                return False, "decode/init failure error"
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_many_retries(strategy, num_retries=10):
    """Test with many consecutive retries."""
    print(f"\nTest: {num_retries} consecutive retries with {strategy}")
    print("-" * 60)
    
    task = "Explain quantum physics briefly"
    failures = []
    
    for i in range(num_retries):
        success, info = test_single_request(strategy, task)
        if success:
            print(f"  Retry {i+1}/{num_retries}: ✓ ({info}ms)")
        else:
            print(f"  Retry {i+1}/{num_retries}: ✗ ({info})")
            failures.append((i+1, info))
        time.sleep(0.5)
    
    if failures:
        print(f"✗ {len(failures)} failures out of {num_retries}")
        for retry_num, error in failures:
            print(f"    Retry {retry_num}: {error}")
        return False
    else:
        print(f"✓ All {num_retries} retries succeeded")
        return True

def test_different_prompts(strategy):
    """Test with different prompts to ensure robustness."""
    print(f"\nTest: Different prompts with {strategy}")
    print("-" * 60)
    
    prompts = [
        "What is the capital of France?",
        "Explain machine learning",
        "Write a haiku about code",
        "Calculate 2+2",
        "Describe the color blue"
    ]
    
    failures = []
    for i, prompt in enumerate(prompts):
        success, info = test_single_request(strategy, prompt)
        if success:
            print(f"  Prompt {i+1}: ✓ ({info}ms)")
        else:
            print(f"  Prompt {i+1}: ✗ ({info})")
            failures.append((i+1, info))
        time.sleep(0.5)
    
    if failures:
        print(f"✗ {len(failures)} failures out of {len(prompts)}")
        return False
    else:
        print(f"✓ All {len(prompts)} different prompts succeeded")
        return True

def test_rapid_succession(strategy):
    """Test rapid successive requests without delay."""
    print(f"\nTest: Rapid succession (no delay) with {strategy}")
    print("-" * 60)
    
    task = "Quick test"
    num_requests = 5
    failures = []
    
    for i in range(num_requests):
        success, info = test_single_request(strategy, task)
        if success:
            print(f"  Request {i+1}/{num_requests}: ✓")
        else:
            print(f"  Request {i+1}/{num_requests}: ✗ ({info})")
            failures.append((i+1, info))
        # No sleep between requests
    
    if failures:
        print(f"✗ {len(failures)} failures out of {num_requests}")
        return False
    else:
        print(f"✓ All {num_requests} rapid requests succeeded")
        return True

def test_different_token_counts(strategy):
    """Test with different max_tokens values."""
    print(f"\nTest: Different max_tokens with {strategy}")
    print("-" * 60)
    
    token_counts = [50, 100, 200, 500]
    failures = []
    
    for tokens in token_counts:
        success, info = test_single_request(strategy, "Test", max_tokens=tokens)
        if success:
            print(f"  max_tokens={tokens}: ✓")
        else:
            print(f"  max_tokens={tokens}: ✗ ({info})")
            failures.append((tokens, info))
        time.sleep(0.5)
    
    if failures:
        print(f"✗ {len(failures)} failures")
        return False
    else:
        print(f"✓ All token count variations succeeded")
        return True

def main():
    """Run all comprehensive tests."""
    strategies = ["latent_mas", "latent_mas_no_align", "latent_mas_embeddings"]
    
    print("=" * 60)
    print("COMPREHENSIVE LATENT MAS RETRY TESTS")
    print("=" * 60)
    
    all_results = {}
    
    for strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"STRATEGY: {strategy}")
        print(f"{'=' * 60}")
        
        results = {
            "many_retries": test_many_retries(strategy, num_retries=10),
            "different_prompts": test_different_prompts(strategy),
            "rapid_succession": test_rapid_succession(strategy),
            "different_tokens": test_different_token_counts(strategy)
        }
        
        all_results[strategy] = results
        
        print(f"\nSummary for {strategy}:")
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {test_name}: {status}")
    
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    
    for strategy, results in all_results.items():
        all_passed = all(results.values())
        status = "✓ ALL PASS" if all_passed else "✗ SOME FAIL"
        print(f"  {strategy}: {status}")
    
    # Check if all tests passed
    total_passed = sum(sum(results.values()) for results in all_results.values())
    total_tests = sum(len(results) for results in all_results.values())
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✓✓✓ ALL COMPREHENSIVE TESTS PASSED ✓✓✓")
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
