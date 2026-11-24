#!/usr/bin/env python3
"""
Test script to verify all latent MAS strategies work on multiple retries.
Tests: latent_mas, latent_mas_no_align, latent_mas_embeddings
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_strategy(strategy_name, num_retries=3):
    """Test a specific strategy with multiple retries."""
    
    payload = {
        "model_id": "gemma-4-E4B-it-Q4_K_S",
        "task": "Explain quantum physics in simple terms",
        "roles": ["planner", "executor"],
        "max_tokens": 150,
        "shared_prompt_kv": True,
        "create_temporary_agents": True,
        "destroy_temporary_agents": True,
        "system_prompts": {
            "planner": "You are a planning agent.",
            "executor": "You are an execution agent."
        },
        "strategy": strategy_name,
        "adapter_type": "recursive",
        "type_k": "turbo4",
        "type_v": "turbo4"
    }
    
    print(f"\nTesting strategy: {strategy_name}")
    print("=" * 60)
    
    for attempt in range(1, num_retries + 1):
        print(f"--- Retry {attempt}/{num_retries} ---")
        
        try:
            response = requests.post(
                f"{BASE_URL}/mas/run",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Success (time: {result.get('time_ms', 'N/A')}ms)")
                
                # Check for decode/init failure in the response
                response_text = json.dumps(result)
                if "decode/init failure" in response_text:
                    print("ERROR: decode/init failure found in response!")
                    return False
                    
            else:
                error_text = response.text
                print(f"✗ Failed: {error_text[:200]}")
                
                if "decode/init failure" in error_text:
                    print("ERROR: The decode/init failure bug still exists!")
                    return False
                return False
                    
        except requests.exceptions.Timeout:
            print("✗ Timed out")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
        
        time.sleep(1)
    
    print(f"✓ All {num_retries} retries succeeded for {strategy_name}")
    return True

def main():
    """Test all latent MAS strategies."""
    strategies = [
        "latent_mas",
        "latent_mas_no_align",
        "latent_mas_embeddings"
    ]
    
    print("Testing all latent MAS strategies with multiple retries")
    print("=" * 60)
    
    results = {}
    for strategy in strategies:
        results[strategy] = test_strategy(strategy, num_retries=3)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    for strategy, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {strategy}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All strategies passed the retry test!")
        return True
    else:
        print("\n✗ Some strategies failed")
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
