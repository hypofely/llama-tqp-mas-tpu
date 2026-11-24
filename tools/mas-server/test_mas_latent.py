#!/usr/bin/env python3
"""LatentMAS feature tests for the C++ mas-server (port 9000).

Covers the latent-space primitives this fork adds on top of TurboQuant:
  - hidden-state capture (per-layer API + HTTP)
  - latent handoff via input embeddings (agent runs without tokenization)
  - cross-agent projection (CrossLink adapter, hidden -> embeddings)
  - KV cache clone / agent spawn (single-context, seq-id based)
  - chain + stop-after-prompt, DTAS (KV-share spawn), SSE streaming

Usage:
    mas-server.exe -m <model.gguf> -p 9000 [--n-gpu-layers 99]
    py -3.12 test_mas_latent.py
"""
import json, math, sys, time
import requests

import os as _os
BASE = _os.environ.get("MAS_BASE", "http://127.0.0.1:9000")
RESULTS = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tests": {}, "errors": []}


def check(name, ok, detail):
    RESULTS["tests"][name] = {"ok": bool(ok), **detail}
    print(f"  [{name}] {'PASS' if ok else 'FAIL'}  {json.dumps(detail)[:160]}")


def post(path, payload=None, timeout=90):
    if payload is None:
        return requests.post(f"{BASE}{path}", timeout=timeout)
    return requests.post(f"{BASE}{path}", json=payload, timeout=timeout)


def main():
    print("=" * 60)
    print("MAS LatentMAS Feature Test Suite")
    print("=" * 60)

    s = requests.Session()
    try:
        # 0. health + warmup: /mas/run lazily initializes the MAS context,
        #    which /v1/agents requires to exist
        r = s.get(f"{BASE}/health", timeout=5)
        j = r.json()
        check("health", r.status_code == 200 and j.get("status") == "ok",
              {"cuda": j.get("cuda"), "model": j.get("model_loaded")})

        r = post("/mas/run", {"task": "warmup", "roles": ["warmup"], "strategy": "chain",
                              "max_tokens": 2, "temperature": 0.0})
        check("warmup_init", r.status_code == 200, {"preview": r.text[:120]})

        # 1. three agents
        idx = {}
        for role in ("encoder", "decoder", "spawn"):
            r = post("/v1/agents", {"role": role, "max_tokens": 16})
            j = r.json()
            idx[role] = j.get("id")
            check(f"create_agent_{role}", r.status_code == 200 and idx[role] is not None,
                  {"response": j})

        # 2. chain run with hidden-state capture, stop after prompt
        r = post("/mas/run", {
            "task": "Write the word hello.",
            "roles": ["encoder"], "strategy": "chain",
            "capture_hidden_states": True, "stop_after_prompt": True,
            "max_tokens": 8, "temperature": 0.0,
        })
        j = r.json()
        out = j.get("outputs", {}).get("encoder", "") if r.status_code == 200 else ""
        check("chain_stop_after_prompt", r.status_code == 200 and isinstance(out, str),
              {"output_preview": out[:50], "timing": j.get("timing", {})})

        # 2b. /mas/run REBUILDS the MAS context, which drops every agent created
        #     before it - so the three agents from step 1 are gone here and the
        #     server is left holding only the "encoder" the run itself made.
        #     Re-create the two peers the later steps project into / feed, and
        #     re-read their indices instead of reusing the stale ones.
        for role in ("decoder", "spawn"):
            r = post("/v1/agents", {"role": role, "max_tokens": 16})
            j = r.json()
            idx[role] = j.get("id")
            check(f"recreate_agent_{role}", r.status_code == 200 and idx[role] is not None,
                  {"response": j})

        # 3. hidden-state retrieval (last token, last layer)
        r = s.get(f"{BASE}/v1/agents/0/hidden-state", timeout=30)
        j = r.json()
        hs = j.get("hidden_state", [])
        n_embd = j.get("n_embd", 0)
        finite = all(math.isfinite(v) for v in hs[:64])
        nonzero = any(abs(v) > 1e-6 for v in hs[:64])
        check("hidden_state", r.status_code == 200 and n_embd > 0 and len(hs) == n_embd and nonzero and finite,
              {"n_embd": n_embd, "len": len(hs), "sample": hs[:3], "nonzero": nonzero})

        # 4. cross-agent projection (hidden -> embeddings via CrossLink adapter)
        dst = idx.get("decoder", 1)
        r = post(f"/v1/agents/0/project-to/{dst}")
        j = r.json()
        check("project_to", r.status_code == 200 and j.get("status") == "ok", {"response": j})

        # 5. latent prompt: feed the captured hidden state back as input
        #    embeddings (1 token) so agent 1 generates WITHOUT tokenization
        r = post(f"/v1/agents/{dst}/embeddings", {"embeddings": hs[:n_embd], "n_embd": n_embd})
        j = r.json()
        check("embeddings_feed", r.status_code == 200 and j.get("n_tokens") == 1,
              {"response": j})

        r = post("/mas/run", {
            "task": "", "roles": ["decoder"], "strategy": "chain",
            "capture_hidden_states": True, "max_tokens": 8, "temperature": 0.0,
        })
        j = r.json()
        out2 = j.get("outputs", {}).get("decoder", "") if r.status_code == 200 else ""
        check("generate_from_latent_embeddings",
              r.status_code == 200 and isinstance(out2, str),
              {"output_preview": out2[:50]})

        # 6. KV cache clone / spawn via the HTTP endpoint (rc reported)
        r = post("/kv/clone", {"src_agent_id": "src", "dst_agent_id": "dst"})
        j = r.json()
        check("kv_clone_endpoint", r.status_code == 200 and "status" in j, {"response": j})

        # 7. DTAS strategy (KV-clone then batched decode) must complete
        r = post("/mas/run", {
            "task": "hello", "roles": ["spawn", "decoder"], "strategy": "dtas",
            "max_tokens": 8, "temperature": 0.0,
        })
        j = r.json()
        ok = r.status_code == 200 and len(j.get("outputs", {})) == 2
        check("dtas_multiagent", ok,
              {"outputs": {k: v[:40] for k, v in j.get("outputs", {}).items()}})

        # 8. latent strategy must complete
        r = post("/mas/run", {
            "task": "say hello", "roles": ["encoder"], "strategy": "latent_mas",
            "max_tokens": 8, "temperature": 0.0,
        })
        j = r.json()
        check("latent_mas_strategy", r.status_code == 200,
              {"outputs": {k: str(v)[:40] for k, v in j.get("outputs", {}).items()}})

        # 9. streaming endpoint returns at least start + done
        r = s.post(f"{BASE}/mas/run/stream",
                   json={"task": "hi", "roles": ["encoder"], "max_tokens": 4, "temperature": 0.0},
                   stream=True, timeout=90)
        events = []
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                try:
                    events.append(json.loads(line[5:].strip()))
                except Exception:
                    pass
        kinds = [e.get("event") for e in events]
        check("stream_events", r.status_code == 200 and "start" in kinds and "done" in kinds,
              {"kinds": kinds})

    except Exception as e:
        RESULTS["errors"].append(str(e))
        print(f"\nERROR: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = failed = 0
    for name, res in RESULTS["tests"].items():
        status = "PASS" if res["ok"] else "FAIL"
        passed += res["ok"]
        failed += (not res["ok"])
        print(f"  {status:4}  {name}")
    print(f"\n{passed} passed, {failed} failed")
    out_path = "test_results_latent.json"
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"Saved: {out_path}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
