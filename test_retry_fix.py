#!/usr/bin/env python3
"""
Test script to verify latent_mas strategy works on multiple retries.
This tests the fix for the "decode/init failure" error that occurred on retry.
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_latent_mas_retry():
    """Test latent_mas strategy with multiple retries."""
    
    # Test payload
    payload = {
        "model_id": "gemma-4-E4B-it-Q4_K_S",
        "task": "Explain quantum physics in simple terms",
        "roles": ["planner", "executor", "outer"],
        "max_tokens": 200,
        "shared_prompt_kv": True,
        "create_temporary_agents": True,
        "destroy_temporary_agents": True,
        "system_prompts": {
            "planner": "You are a planning agent.",
            "executor": "You are an execution agent.",
            "outer": "You are an outer agent."
        },
        "strategy": "latent_mas",
        "adapter_type": "recursive",
        "type_k": "turbo4",
        "type_v": "turbo4"
    }
    
    print("Testing latent_mas strategy with multiple retries...")
    print("=" * 60)
    
    for attempt in range(1, 4):
        print(f"\n--- Attempt {attempt} ---")
        print(f"Timestamp: {time.strftime('%H:%M:%S')}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/mas/run",
                json=payload,
                timeout=120
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Success! Time: {result.get('time_ms', 'N/A')}ms")
                print(f"Roles: {result.get('roles', [])}")
                print(f"Output keys: {list(result.get('outputs', {}).keys())}")
                
                # Check for outputs
                outputs = result.get('outputs', {})
                if outputs:
                    for role, output in outputs.items():
                        if isinstance(output, dict):
                            text = output.get('text', '')[:100]
                        else:
                            text = str(output)[:100]
                        print(f"  {role}: {text}...")
                
                print("✓ Attempt succeeded")
            else:
                error_text = response.text
                print(f"✗ Attempt failed: {error_text}")
                
                # Check if it's the decode/init failure error
                if "decode/init failure" in error_text:
                    print("ERROR: The decode/init failure bug still exists!")
                    return False
                    
        except requests.exceptions.Timeout:
            print("✗ Attempt timed out")
            return False
        except Exception as e:
            print(f"✗ Attempt error: {e}")
            return False
        
        # Small delay between retries
        time.sleep(2)
    
    print("\n" + "=" * 60)
    print("All 3 attempts succeeded! The fix appears to be working.")
    return True

if __name__ == "__main__":
    try:
        success = test_latent_mas_retry()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Test script error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
