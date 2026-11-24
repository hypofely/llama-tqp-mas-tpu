#!/usr/bin/env python3
"""Test /v1/chat/completions endpoint on mas-server.

Usage:
  1. Start the server manually:  start_mas_server.bat
  2. Wait for "listening on http://0.0.0.0:8080"
  3. Run this test:               python test_v1_chat.py
"""
import time
import requests
import sys

BASE_URL = "http://127.0.0.1:8080"

def wait_for_server(timeout=600):
    print("Waiting for server to respond on 127.0.0.1:8080 ...")
    for i in range(timeout * 2):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                print("Server is ready!")
                return True
        except Exception:
            pass
        if i % 4 == 0:
            print(".", end="", flush=True)
        time.sleep(0.5)
    print("\nTimeout! Is the server running?")
    return False

def chat_completion(messages, max_tokens=256, temperature=0.7):
    payload = {
        "model": "gemma-4-E4B-it",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
    }
    r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def is_coherent(text):
    """Heuristic: reject if >30% of words are the same repeated word."""
    words = text.lower().split()
    if len(words) < 5:
        return False
    from collections import Counter
    most_common = Counter(words).most_common(1)[0]
    return most_common[1] / len(words) < 0.30

def main():
    print("=" * 60)
    print("MAS Server /v1/chat/completions Test")
    print("=" * 60)

    if not wait_for_server():
        sys.exit(1)

    all_passed = True

    # Test 1: Single completion
    print("\n[Test 1] Single /v1/chat/completions request")
    try:
        resp = chat_completion([
            {"role": "user", "content": "Explain quantum physics in one sentence."}
        ], max_tokens=64)
        content = resp["choices"][0]["message"]["content"]
        print(f"  Response: {content[:200]}...")
        if is_coherent(content):
            print("  PASS: Response is coherent")
        else:
            print("  FAIL: Response appears repetitive/echoing")
            all_passed = False
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False

    # Test 2: Multiple sequential requests (the memory slot bug repro)
    print("\n[Test 2] 5 sequential requests (memory slot stress test)")
    prompts = [
        "What is the speed of light?",
        "Explain gravity simply.",
        "How do lasers work?",
        "What is DNA?",
        "Explain quantum physics",
    ]
    for i, prompt in enumerate(prompts, 1):
        try:
            resp = chat_completion([
                {"role": "user", "content": prompt}
            ], max_tokens=48)
            content = resp["choices"][0]["message"]["content"]
            status = "OK" if is_coherent(content) else "REPETITIVE"
            print(f"  [{i}/5] {prompt[:30]:<30} -> {status} ({len(content)} chars)")
            if not is_coherent(content):
                all_passed = False
        except Exception as e:
            print(f"  [{i}/5] {prompt[:30]:<30} -> FAIL: {e}")
            all_passed = False

    # Test 3: Multi-turn conversation
    print("\n[Test 3] Multi-turn conversation")
    try:
        resp = chat_completion([
            {"role": "system", "content": "You are a helpful physics tutor."},
            {"role": "user", "content": "What is quantum entanglement?"},
        ], max_tokens=128)
        content = resp["choices"][0]["message"]["content"]
        print(f"  Response: {content[:200]}...")
        if is_coherent(content):
            print("  PASS: Multi-turn response is coherent")
        else:
            print("  FAIL: Multi-turn response appears repetitive")
            all_passed = False
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
