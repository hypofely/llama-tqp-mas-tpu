#!/usr/bin/env python3
"""
Test the exact scenario that was failing:
Retrying the "quantum physics explain prompt" multiple times.
This simulates the user's original bug report.
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_quantum_physics_retry(strategy, num_retries=5):
    """
    Test the exact scenario from the bug report:
    "quantum physics explain prompt" with multiple retries.
    """
    print(f"\n{'=' * 70}")
    print(f"Testing EXACT BUG REPORT SCENARIO")
    print(f"Strategy: {strategy}")
    print(f"Prompt: 'Explain quantum physics in simple terms'")
    print(f"Retries: {num_retries}")
    print(f"{'=' * 70}")
    
    # Exact prompt from bug report
    task = "Explain quantum physics in simple terms"
    
    # Configuration matching the error message
    payload = {
        "model_id": "gemma-4-E4B-it-Q4_K_S",
        "task": task,
        "roles": ["planner", "executor", "outer"],
        "max_tokens": 200,  # Reduced from 20000 for faster testing
        "shared_prompt_kv": True,
        "create_temporary_agents": True,
        "destroy_temporary_agents": True,
        "system_prompts": {
            "planner": "You are a planning agent.",
            "executor": "You are an execution agent.",
            "outer": "You are an outer agent."
        },
        "strategy": strategy,
        "adapter_type": "recursive",
        "type_k": "turbo4",
        "type_v": "turbo4"
    }
    
    failures = []
    decode_init_failures = []
    
    for retry_num in range(1, num_retries + 1):
        print(f"\n--- Retry {retry_num}/{num_retries} ---")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/mas/run",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = json.dumps(result)
                
                # Check for the specific error from bug report
                if "decode/init failure" in response_text:
                    error_msg = "MAS_UNAVAILABLE: MAS run failed rc=-1 ... hint=decode/init failure"
                    print(f"✗ FAILED: {error_msg}")
                    decode_init_failures.append(retry_num)
                    failures.append((retry_num, error_msg))
                else:
                    print(f"✓ SUCCESS")
                    print(f"  Roles: {result.get('roles', [])}")
                    print(f"  Time: {result.get('time_ms', 'N/A')}ms")
                    
            else:
                error_text = response.text
                
                # Check for the specific error from bug report
                if "decode/init failure" in error_text:
                    error_msg = "MAS_UNAVAILABLE: MAS run failed rc=-1 ... hint=decode/init failure"
                    print(f"✗ FAILED: {error_msg}")
                    print(f"  Full error: {error_text[:200]}")
                    decode_init_failures.append(retry_num)
                    failures.append((retry_num, error_msg))
                else:
                    print(f"✗ FAILED: HTTP {response.status_code}")
                    print(f"  Error: {error_text[:200]}")
                    failures.append((retry_num, f"HTTP {response.status_code}"))
                    
        except requests.exceptions.Timeout:
            print(f"✗ FAILED: Timeout")
            failures.append((retry_num, "Timeout"))
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failures.append((retry_num, str(e)))
        
        # Small delay between retries (simulating user clicking retry)
        time.sleep(1)
    
    print(f"\n{'=' * 70}")
    print(f"RESULTS FOR {strategy}")
    print(f"{'=' * 70}")
    
    if decode_init_failures:
        print(f"✗✗✗ BUG STILL EXISTS ✗✗✗")
        print(f"decode/init failure occurred on retries: {decode_init_failures}")
        print(f"This is the EXACT error from your bug report!")
        return False
    elif failures:
        print(f"✗ Some failures occurred (but NOT decode/init failure):")
        for retry_num, error in failures:
            print(f"  Retry {retry_num}: {error}")
        return False
    else:
        print(f"✓✓✓ ALL {num_retries} RETRIES SUCCEEDED ✓✓✓")
        print(f"The 'decode/init failure' bug is FIXED!")
        return True

def main():
    """Test the exact scenario with all latent MAS strategies."""
    strategies = ["latent_mas", "latent_mas_no_align", "latent_mas_embeddings"]
    
    print("=" * 70)
    print("EXACT BUG REPORT SCENARIO TEST")
    print("Simulating: 'on the chatpanel current session click the retry")
    print("on the quantum physics explain prompt'")
    print("=" * 70)
    
    results = {}
    for strategy in strategies:
        results[strategy] = test_quantum_physics_retry(strategy, num_retries=5)
    
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    
    for strategy, passed in results.items():
        status = "✓ FIXED" if passed else "✗ BUG EXISTS"
        print(f"  {strategy}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print(f"\n✓✓✓ THE BUG IS FIXED FOR ALL STRATEGIES ✓✓✓")
        print(f"You can now retry prompts without decode/init failure!")
        return True
    else:
        print(f"\n✗ The bug still exists for some strategies")
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
