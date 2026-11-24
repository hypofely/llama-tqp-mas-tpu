"""Perf comparison on Qwen3-8B: latent-hybrid pipelines vs full text.

Answers three questions with real wall-clock numbers:
  1. Per-unit cost: ms per text token vs ms per latent step.
  2. The requested pipeline shape:
        thinking(latent) -> resource collection(text) -> pre-plan(latent)
        -> [latent, string, latent, string]
     versus a pure-text turn of comparable output length.
  3. Whether latent thinking actually helps, judged by READING the output.

Methodology rules learned the hard way:
  - Reset counters with DELETE /mas/latent/stats (POST .../reset is a silent no-op).
  - Print RAW strings. A unique-word coherence proxy rated 'the\\nthe\\nthe'
    higher than real sentences, so never rank by a proxy alone.
  - Time the REQUEST, not a poll loop.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=9026)
ap.add_argument("--ntok", type=int, default=128, help="text tokens per arm")
ap.add_argument("--depths", type=str, default="0,2,4,8")
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--json", type=str, default="")
args = ap.parse_args()

B = "http://127.0.0.1:%d" % args.port


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


def stats():
    try:
        return req("/mas/latent/stats", t=20)[1]
    except Exception:
        return {}


def reset():
    try:
        req("/mas/latent/stats", method="DELETE", t=20)
    except Exception:
        pass


def new_agent(role, ctx=8192):
    """Clear then create ONE agent, so every arm starts at index 0 with empty KV.

    Two traps this avoids:
      - KV accumulates on a reused agent; the per-agent quota is
        n_ctx / n_agents_max (1024 at 8192/8), so a sweep reusing one agent dies
        with "failed to find a memory slot for batch of size 1" -> HTTP 500.
      - /v1/agents APPENDS and n_agents_max caps the count, so one-agent-per-arm
        without clearing runs out of slots after 8 arms.
    """
    try:
        req("/agents/clear", {}, t=60)
    except Exception:
        pass
    req("/v1/agents", {"role": role, "max_tokens": ctx,
                       "temperature": 0.0, "top_k": 1}, t=120)


# A long-ish prompt so the context is not trivially small.
PROMPT = (
    "You are a senior performance engineer. Below is a profile of a web service.\n"
    "p50 latency 42 ms, p99 latency 1900 ms, QPS 3200, 8 worker threads, "
    "Postgres connection pool size 10, mean query time 6 ms, "
    "cache hit ratio 0.61, GC pause p99 180 ms, heap 6 GB, "
    "CPU utilisation 0.74, disk await 11 ms, network RTT to DB 0.4 ms.\n"
    "Identify the most likely cause of the p99 tail and propose one concrete fix.\n"
)

TASK_TEXT = "Explain the bottleneck, then give the fix."


def run_turn(segments, tag):
    # A FRESH AGENT PER ARM. KV positions accumulate on a reused agent, and the
    # per-agent quota is n_ctx / n_agents_max (1024 at 8192/8), so a sweep that
    # reuses one agent dies partway with "failed to find a memory slot".
    new_agent("perf_" + tag)
    reset()
    t0 = time.time()
    st, r = req("/turn", {"agent_idx": 0, "segments": segments}, t=900)
    wall = time.time() - t0
    s = stats()
    txt = r.get("text") or ""
    return {
        "tag": tag,
        "wall_s": round(wall, 3),
        "http": st,
        "n_tokens": r.get("n_tokens", 0),
        "n_latent": r.get("n_latent_steps", 0),
        "n_positions": r.get("n_positions", 0),
        "latent_ms_total": s.get("latent_ms", 0),
        "latent_steps_counted": s.get("latent_steps", 0),
        "avg_ms_per_step": s.get("avg_ms_per_step", 0),
        "words": len(txt.split()),
        "text": txt,
    }


def text_arm(ntok):
    segs = [{"kind": "tokens", "text": PROMPT + TASK_TEXT}]
    for _ in range(ntok):
        segs.append({"kind": "tokens", "text": "", "sample": 1})
    return segs


def latent_arm(depth, ntok):
    """thinking(latent) -> speak(text)."""
    segs = [{"kind": "tokens", "text": PROMPT + TASK_TEXT}]
    if depth:
        segs.append({"kind": "latent", "latent_steps": depth, "align": 1})
    for _ in range(ntok):
        segs.append({"kind": "tokens", "text": "", "sample": 1})
    return segs


def pipeline_arm(depth, ntok):
    """The requested shape:
       thinking(latent) -> resource collection(text) -> pre-plan(latent)
       -> [latent, string, latent, string]
    """
    segs = [{"kind": "tokens", "text": PROMPT}]
    if depth:
        segs.append({"kind": "latent", "latent_steps": depth, "align": 1})
    # resource collection: speak a short chunk
    for _ in range(max(4, ntok // 4)):
        segs.append({"kind": "tokens", "text": "", "sample": 1})
    # pre-plan thinking
    if depth:
        segs.append({"kind": "latent", "latent_steps": depth, "align": 1})
    # interleaved latent / string x2
    for _ in range(2):
        if depth:
            segs.append({"kind": "latent", "latent_steps": depth, "align": 1})
        for _ in range(max(4, ntok // 4)):
            segs.append({"kind": "tokens", "text": "", "sample": 1})
    return segs


print("=== server / alignment ===", flush=True)
st, h = req("/health", t=20)
st, a = req("/mas/alignment/status", t=20)
print("model_loaded=%s  align n_embd=%s use=%s rescale=%s target_norm=%.5f"
      % (h.get("model_loaded"), a.get("n_embd"), a.get("use_alignment"),
         a.get("rescale_norm"), a.get("target_norm") or 0.0), flush=True)
if not a.get("n_embd"):
    print("FATAL: no alignment matrix; run the compute + save first")
    sys.exit(2)

new_agent("perf")

results = []

# ---- 1. per-unit cost -------------------------------------------------------
print("\n=== 1. PER-UNIT COST (8B) ===", flush=True)
N = 32
r_text = run_turn(text_arm(N), "text%d" % N)
# latent-only steps: measure by comparing a depth-N latent turn to depth 0
r_lat = run_turn(latent_arm(N, 0), "latent%d" % N)
ms_tok = r_text["wall_s"] * 1000.0 / max(1, r_text["n_tokens"])
lat_total = r_lat["latent_ms_total"] or 0
lat_n = r_lat["latent_steps_counted"] or N
ms_step = (lat_total / lat_n) if lat_n else 0.0
print("text  : %d tok in %.3f s -> %.1f ms/token"
      % (r_text["n_tokens"], r_text["wall_s"], ms_tok), flush=True)
print("latent: %d steps, server-reported %.1f ms total -> %.1f ms/step"
      % (lat_n, lat_total, ms_step), flush=True)
if ms_step and ms_tok:
    print("ratio : 1 latent step = %.2f text tokens of time"
          % (ms_step / ms_tok), flush=True)
results += [r_text, r_lat]

# ---- 2. depth sweep, same output length ------------------------------------
print("\n=== 2. LATENT DEPTH SWEEP (%d text tokens each) ===" % args.ntok,
      flush=True)
for d in [int(x) for x in args.depths.split(",")]:
    runs = [run_turn(latent_arm(d, args.ntok), "L%d" % d)
            for _ in range(args.reps)]
    w = statistics.median(r["wall_s"] for r in runs)
    best = runs[0]
    print("L=%-3d wall=%.3fs tok=%d latent=%d pos=%d words=%d"
          % (d, w, best["n_tokens"], best["n_latent"],
             best["n_positions"], best["words"]), flush=True)
    print("      %r" % best["text"][:200], flush=True)
    results += runs

# ---- 3. the requested pipeline vs full text --------------------------------
print("\n=== 3. PIPELINE vs FULL TEXT ===", flush=True)
for d in (0, 4):
    runs = [run_turn(pipeline_arm(d, args.ntok), "pipe_L%d" % d)
            for _ in range(args.reps)]
    w = statistics.median(r["wall_s"] for r in runs)
    best = runs[0]
    print("pipeline L=%-2d wall=%.3fs tok=%d latent=%d pos=%d words=%d"
          % (d, w, best["n_tokens"], best["n_latent"],
             best["n_positions"], best["words"]), flush=True)
    print("      %r" % best["text"][:220], flush=True)
    results += runs

runs = [run_turn(text_arm(args.ntok), "fulltext") for _ in range(args.reps)]
w = statistics.median(r["wall_s"] for r in runs)
print("full text     wall=%.3fs tok=%d words=%d"
      % (w, runs[0]["n_tokens"], runs[0]["words"]), flush=True)
print("      %r" % runs[0]["text"][:220], flush=True)
results += runs

if args.json:
    json.dump({"per_unit": {"ms_per_token": ms_tok, "ms_per_latent_step": ms_step},
               "runs": results}, open(args.json, "w"), indent=1)
    print("\nsaved -> " + args.json, flush=True)