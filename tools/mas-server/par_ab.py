"""A/B the per-step projection thread count on the 8B.

The projection is n_embd^2 MACs every latent step (16.7M at 4096). This measures
whether parallelising it actually reduces ms/step, and confirms the output is
UNCHANGED (a speed knob must not alter results).

Loads the cached matrix (0.1 s) instead of re-solving (~40 min).
"""
import hashlib
import json
import statistics
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9026
B = "http://127.0.0.1:%d" % PORT
CACHE = r"C:\+run\GPT_WEB\llama-tqp-mas-cpp\tools\mas-server\align_8b_s32.bin"
DEPTH = 32
NTOK = 48
REPS = 3


def req(path, body=None, method=None, t=900):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        B + path, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method or ("POST" if data else "GET"))
    with urllib.request.urlopen(r, timeout=t) as resp:
        raw = resp.read().decode()
        try:
            return resp.status, json.loads(raw)
        except Exception:
            return resp.status, raw


for i in range(90):
    try:
        if req("/health", t=10)[1].get("model_loaded"):
            break
    except Exception:
        pass
    time.sleep(10)
else:
    print("server never healthy")
    sys.exit(2)

# The cache key embeds stride AND ridge. A fresh server defaults to stride 1,
# so loading a stride-32 file WITHOUT passing vocab_stride is a 404 miss.
st, b = req("/mas/alignment/load", {"path": CACHE, "vocab_stride": 32}, t=300)
print("cache load HTTP %s %s" % (st, json.dumps(b)[:110]), flush=True)
if st != 200:
    print("FATAL: cache miss - cannot A/B without a matrix")
    sys.exit(2)

PROMPT = ("You are a senior performance engineer. p50 42 ms, p99 1900 ms, QPS 3200, "
          "8 workers, pool 10, mean query 6 ms, cache hit 0.61, GC p99 180 ms. "
          "Identify the p99 tail cause and give one fix.\n")

results = []
for nthr in (1, 2, 4, 8):
    req("/mas/alignment/toggle",
        {"use_alignment": True, "rescale_norm": True, "parallel_threads": nthr}, t=60)
    walls, steps_ms, hashes, texts = [], [], set(), []
    for rep in range(REPS):
        try:
            req("/agents/clear", {}, t=60)
        except Exception:
            pass
        req("/v1/agents", {"role": "ab", "max_tokens": 4096,
                           "temperature": 0.0, "top_k": 1}, t=120)
        req("/mas/latent/stats", method="DELETE", t=20)
        segs = [{"kind": "tokens", "text": PROMPT},
                {"kind": "latent", "latent_steps": DEPTH, "align": 1}]
        for _ in range(NTOK):
            segs.append({"kind": "tokens", "text": "", "sample": 1})
        t0 = time.time()
        st, r = req("/turn", {"agent_idx": 0, "segments": segs}, t=900)
        walls.append(time.time() - t0)
        s = req("/mas/latent/stats", t=20)[1]
        steps_ms.append(s.get("avg_ms_per_step", 0.0))
        txt = r.get("text") or ""
        hashes.add(hashlib.sha256(txt.encode()).hexdigest()[:12])
        texts.append(txt)
    w = statistics.median(walls)
    ms = statistics.median(steps_ms)
    print("threads=%-2d wall=%.3fs  ms/step=%.2f  distinct_outputs=%d"
          % (nthr, w, ms, len(hashes)), flush=True)
    print("   %r" % texts[0][:130], flush=True)
    results.append({"threads": nthr, "wall_s": round(w, 3),
                    "ms_per_step": round(ms, 2),
                    "distinct_outputs": len(hashes),
                    "text": texts[0]})

base = results[0]
print("\n=== vs threads=1 ===", flush=True)
for r in results:
    sp = base["ms_per_step"] / r["ms_per_step"] if r["ms_per_step"] else 0
    same = "SAME" if r["text"] == base["text"] else "DIFFERENT(!)"
    print("threads=%-2d ms/step %.2f -> %.2f  speedup %.2fx  output %s"
          % (r["threads"], base["ms_per_step"], r["ms_per_step"], sp, same),
          flush=True)

json.dump(results, open("par_ab_results.json", "w"), indent=1)
print("\nsaved -> par_ab_results.json", flush=True)