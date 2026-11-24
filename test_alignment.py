#!/usr/bin/env python3
"""Test alignment matrix functionality via mas-server API"""

import requests
import json
import time
import signal

MAS_SERVER_URL = "http://localhost:8000"
TIMEOUT = 30  # 30 second timeout for all requests

def test_alignment_compute():
    """Test alignment matrix computation"""
    print("\n=== Test 1: Alignment Matrix Computation ===")
    try:
        response = requests.post(f"{MAS_SERVER_URL}/mas/alignment/compute", timeout=TIMEOUT)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        if response.status_code == 200:
            print("✓ PASSED: Alignment matrix computed successfully")
            return True
        else:
            print("✗ FAILED: Could not compute alignment matrix")
            return False
    except requests.exceptions.Timeout:
        print(f"✗ FAILED: Request timed out after {TIMEOUT} seconds")
        return False
    except Exception as e:
        print(f"✗ FAILED: Exception: {e}")
        return False

def test_alignment_toggle():
    """Test alignment toggle"""
    print("\n=== Test 2: Alignment Toggle ===")
    try:
        # Test enabling
        response = requests.post(f"{MAS_SERVER_URL}/mas/alignment/toggle",
                                json={"use_alignment": True}, timeout=TIMEOUT)
        print(f"Enable Status: {response.status_code}")
        print(f"Enable Response: {response.json()}")

        # Test disabling
        response = requests.post(f"{MAS_SERVER_URL}/mas/alignment/toggle",
                                json={"use_alignment": False}, timeout=TIMEOUT)
        print(f"Disable Status: {response.status_code}")
        print(f"Disable Response: {response.json()}")

        if response.status_code == 200:
            print("✓ PASSED: Alignment toggle works")
            return True
        else:
            print("✗ FAILED: Could not toggle alignment")
            return False
    except requests.exceptions.Timeout:
        print(f"✗ FAILED: Request timed out after {TIMEOUT} seconds")
        return False
    except Exception as e:
        print(f"✗ FAILED: Exception: {e}")
        return False

def test_alignment_status():
    """Test alignment status"""
    print("\n=== Test 3: Alignment Status ===")
    try:
        response = requests.get(f"{MAS_SERVER_URL}/mas/alignment/status", timeout=TIMEOUT)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        if response.status_code == 200:
            print("✓ PASSED: Alignment status retrieved")
            return True
        else:
            print("✗ FAILED: Could not get alignment status")
            return False
    except requests.exceptions.Timeout:
        print(f"✗ FAILED: Request timed out after {TIMEOUT} seconds")
        return False
    except Exception as e:
        print(f"✗ FAILED: Exception: {e}")
        return False

def test_latent_mas_strategy():
    """Test LATENT_MAS strategy with alignment"""
    print("\n=== Test 4: LATENT_MAS Strategy ===")
    try:
        # First compute alignment matrix
        requests.post(f"{MAS_SERVER_URL}/mas/alignment/compute", timeout=TIMEOUT)
        requests.post(f"{MAS_SERVER_URL}/mas/alignment/toggle", json={"use_alignment": True}, timeout=TIMEOUT)

        # Run a simple test with LATENT_MAS strategy
        payload = {
            "task": "What is 2+2?",
            "roles": ["MathExpert"],
            "max_tokens": 50,
            "strategy": "latent_mas",
            "clear_agents": True
        }
        response = requests.post(f"{MAS_SERVER_URL}/mas/run", json=payload, timeout=TIMEOUT)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response keys: {result.keys()}")
            print("✓ PASSED: LATENT_MAS strategy executed")
            return True
        else:
            print(f"✗ FAILED: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"✗ FAILED: Request timed out after {TIMEOUT} seconds")
        return False
    except Exception as e:
        print(f"✗ FAILED: Exception: {e}")
        return False

def test_latent_mas_no_align_strategy():
    """Test LATENT_MAS_NO_ALIGN strategy"""
    print("\n=== Test 5: LATENT_MAS_NO_ALIGN Strategy ===")
    try:
        # Disable alignment
        requests.post(f"{MAS_SERVER_URL}/mas/alignment/toggle", json={"use_alignment": False}, timeout=TIMEOUT)

        # Run a simple test with LATENT_MAS_NO_ALIGN strategy
        payload = {
            "task": "What is 2+2?",
            "roles": ["MathExpert"],
            "max_tokens": 50,
            "strategy": "latent_mas_no_align",
            "clear_agents": True
        }
        response = requests.post(f"{MAS_SERVER_URL}/mas/run", json=payload, timeout=TIMEOUT)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response keys: {result.keys()}")
            print("✓ PASSED: LATENT_MAS_NO_ALIGN strategy executed")
            return True
        else:
            print(f"✗ FAILED: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"✗ FAILED: Request timed out after {TIMEOUT} seconds")
        return False
    except Exception as e:
        print(f"✗ FAILED: Exception: {e}")
        return False

def test_latent_mas_embeddings_strategy():
    """Test LATENT_MAS_EMBEDDINGS strategy"""
    print("\n=== Test 6: LATENT_MAS_EMBEDDINGS Strategy ===")
    try:
        # Run a simple test with LATENT_MAS_EMBEDDINGS strategy
        payload = {
            "task": "What is 2+2?",
            "roles": ["MathExpert"],
            "max_tokens": 50,
            "strategy": "latent_mas_embeddings",
            "clear_agents": True
        }
        response = requests.post(f"{MAS_SERVER_URL}/mas/run", json=payload, timeout=TIMEOUT)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response keys: {result.keys()}")
            print("✓ PASSED: LATENT_MAS_EMBEDDINGS strategy executed")
            return True
        else:
            print(f"✗ FAILED: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"✗ FAILED: Request timed out after {TIMEOUT} seconds")
        return False
    except Exception as e:
        print(f"✗ FAILED: Exception: {e}")
        return False

def main():
    print("=" * 60)
    print("Alignment Matrix Test Suite")
    print("=" * 60)
    
    # Wait for server to be ready
    print("\nWaiting for server to be ready...")
    time.sleep(2)
    
    results = []
    results.append(("Alignment Compute", test_alignment_compute()))
    results.append(("Alignment Toggle", test_alignment_toggle()))
    results.append(("Alignment Status", test_alignment_status()))
    results.append(("LATENT_MAS Strategy", test_latent_mas_strategy()))
    results.append(("LATENT_MAS_NO_ALIGN Strategy", test_latent_mas_no_align_strategy()))
    results.append(("LATENT_MAS_EMBEDDINGS Strategy", test_latent_mas_embeddings_strategy()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:40s} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)

if __name__ == "__main__":
    main()
