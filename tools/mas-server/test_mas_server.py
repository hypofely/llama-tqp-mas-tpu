#!/usr/bin/env python3
"""Test suite for the C++ mas-server.exe (port 9000)."""
import json, time, subprocess, sys, os
import requests

import os as _os
BASE_URL = _os.environ.get("MAS_BASE", "http://127.0.0.1:9000")
RESULTS = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tests": {}, "errors": []}

def check(name, ok, detail):
    RESULTS["tests"][name] = {"ok": ok, **detail}
    status = "PASS" if ok else "FAIL"
    print(f"  [{name}] {status}")

def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    check("health", r.status_code == 200 and r.json().get("status") == "ok",
          {"model_loaded": r.json().get("model_loaded")})

def test_generate():
    payload = {"prompt": "Write about space.", "max_tokens": 32, "temperature": 0.7}
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=60)
    elapsed = time.perf_counter() - t0
    ok = r.status_code == 200
    text = r.json().get("text", "") if ok else ""
    check("generate", ok and len(text) > 10,
          {"sec": round(elapsed, 2), "tok_per_sec": round(32 / elapsed, 1) if elapsed else 0,
           "text_preview": text[:80]})

def test_generate_greedy():
    payload = {"prompt": "Write about space.", "max_tokens": 64, "temperature": 0.0}
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=60)
    elapsed = time.perf_counter() - t0
    ok = r.status_code == 200
    text = r.json().get("text", "") if ok else ""
    check("generate_greedy", ok and len(text) > 10,
          {"sec": round(elapsed, 2), "tok_per_sec": round(64 / elapsed, 1) if elapsed else 0,
           "text_preview": text[:80]})

def test_mas_broadcast():
    payload = {"task": "Write about space.", "roles": ["writer"], "max_tokens": 32, "temperature": 0.7, "strategy": "broadcast"}
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/mas/run", json=payload, timeout=60)
    elapsed = time.perf_counter() - t0
    ok = r.status_code == 200
    out = r.json().get("outputs", {}).get("writer", "") if ok else ""
    check("mas_broadcast", ok and len(out) > 10,
          {"sec": round(elapsed, 2), "output_preview": out[:80]})

def test_mas_chain():
    payload = {"task": "Explain photosynthesis.", "roles": ["planner", "executor"], "max_tokens": 32, "temperature": 0.7, "strategy": "chain"}
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/mas/run", json=payload, timeout=60)
    elapsed = time.perf_counter() - t0
    ok = r.status_code == 200
    out = r.json().get("outputs", {}) if ok else {}
    planner = out.get("planner", "")
    executor = out.get("executor", "")
    check("mas_chain", ok and len(planner) > 5 and len(executor) > 5,
          {"sec": round(elapsed, 2), "planner_preview": planner[:60], "executor_preview": executor[:60]})

def test_mas_dtas():
    # DTAS should be auto-routed to broadcast and not crash
    payload = {"task": "Write about AI.", "roles": ["writer", "editor"], "max_tokens": 32, "temperature": 0.7, "strategy": "dtas"}
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/mas/run", json=payload, timeout=60)
    elapsed = time.perf_counter() - t0
    ok = r.status_code == 200
    out = r.json().get("outputs", {}) if ok else {}
    check("mas_dtas", ok,
          {"sec": round(elapsed, 2), "outputs": {k: v[:60] for k, v in out.items()}})

def test_perf():
    perf = {}
    for mt in [32, 64, 128]:
        payload = {"prompt": "Write about space.", "max_tokens": mt, "temperature": 0.7}
        t0 = time.perf_counter()
        r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=60)
        elapsed = time.perf_counter() - t0
        ok = r.status_code == 200
        perf[f"{mt}tok"] = {"sec": round(elapsed, 2), "tok_per_sec": round(mt / elapsed, 1) if elapsed else 0}
    check("perf", all(v["tok_per_sec"] > 0 for v in perf.values()), perf)

def main():
    print("=" * 60)
    print("MAS C++ Server Test Suite")
    print("=" * 60)
    print(f"\nServer: {BASE_URL}\n")

    try:
        test_health()
        test_generate()
        test_generate_greedy()
        test_mas_broadcast()
        test_mas_chain()
        test_mas_dtas()
        test_perf()
    except Exception as e:
        RESULTS["errors"].append(str(e))
        print(f"\nERROR: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, res in RESULTS["tests"].items():
        status = "PASS" if res["ok"] else "FAIL"
        print(f"  {status:4}  {name}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "test_results_cpp.json")
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
