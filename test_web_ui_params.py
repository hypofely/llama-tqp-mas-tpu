#!/usr/bin/env python3
"""
Test with web UI parameters to reproduce the decode/init failure error.
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_with_web_ui_params():
    """Test with the exact parameters used by the web UI."""
    print("Testing with web UI parameters...")
    print("max_tokens=2048, n_ctx=8192, n_ctx_seq=10240")
    
    payload = {
        "model_id": "gemma-4-E4B-it-Q4_K_S",
        "task": "Explain quantum physics in simple terms",
        "roles": ["planner", "executor", "outer"],
        "max_tokens": 2048,
        "n_ctx": 8192,
        "n_ctx_seq": 10240,
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
        "type_v": "turbo4",
        "agent_params": {
            "planner": {"latent_steps": 80, "is_latent": 1},
            "executor": {"latent_steps": 80, "is_latent": 1},
            "outer": {"latent_steps": 0, "is_latent": 0}
        }
    }
    
    for i in range(3):
        print(f"\n--- Retry {i+1}/3 ---")
        try:
            response = requests.post(f"{BASE_URL}/mas/run", json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                response_text = json.dumps(result)
                if "decode/init failure" in response_text:
                    print(f"✗ FAILED: decode/init failure")
                    print(f"  Error: {response_text[:500]}")
                    return False
                print(f"✓ SUCCESS")
            else:
                if "decode/init failure" in response.text:
                    print(f"✗ FAILED: decode/init failure")
                    print(f"  Error: {response.text[:500]}")
                    return False
                print(f"✗ FAILED: HTTP {response.status_code}")
                print(f"  Error: {response.text[:500]}")
                return False
                
        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False
        time.sleep(1)
    
    print("\n✓ All retries succeeded with web UI parameters")
    return True

if __name__ == "__main__":
    try:
        success = test_with_web_ui_params()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Test script error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
