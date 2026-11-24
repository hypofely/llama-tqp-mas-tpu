import requests, time, json

BASE_URL = "http://127.0.0.1:8080"
MODEL_PATH = r"E:\LMStudio\models\unsloth\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-Q4_K_S.gguf"

# Load model
print("Loading model...")
r = requests.post(f"{BASE_URL}/models", json={"model_id":"test","model_path":MODEL_PATH,"n_gpu_layers":-1}, timeout=60)
print(f"Model load: {r.status_code}")
if r.status_code != 200:
    print(r.text)
    exit(1)

# Create agent
print("Creating agent...")
r = requests.post(f"{BASE_URL}/agents", json={"agent_id":"test","role":"writer","model_id":"test","system_prompt":"","n_ctx":2048,"use_inner_link":False,"adapter_type":"none"}, timeout=30)
print(f"Agent create: {r.status_code}")

# Test generation
results = {}
for max_tok in [64, 128]:
    print(f"Testing {max_tok} tokens...")
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/agents/generate", json={"agent_id":"test","text":"Explain quantum computing:","max_tokens":max_tok,"temperature":0.7}, timeout=120)
    elapsed = time.perf_counter() - t0
    if r.status_code == 200:
        stats = r.json().get("stats", {})
        gen_tok = stats.get("generated_tokens", 0)
        results[f"{max_tok}tok"] = {"sec": round(elapsed, 2), "gen_tok": gen_tok, "tok/sec": round(gen_tok/elapsed, 2)}
        print(f"  {max_tok}tok: {results[f'{max_tok}tok']['tok/sec']} tok/sec")
    else:
        print(f"  Error: {r.text}")

# Cleanup
requests.post(f"{BASE_URL}/agents/destroy", json={"agent_id":"test"}, timeout=10)

# Save results
with open("speed_result.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to speed_result.json")
print(json.dumps(results, indent=2))
