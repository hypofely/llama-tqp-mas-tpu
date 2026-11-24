#!/usr/bin/env python3
"""latent_bench.py - quality + speed of the MAS latent path vs a text baseline.

WHY THIS EXISTS
The measurement log accumulated many A/B numbers but each was a one-off script
in the scratch dir, so nothing was reproducible and the quality proxy that got
used (unique-words / total-words) turned out to be GAMEABLE: it rated the
degenerate output `the\\nthe\\nthe` as BETTER than a real sentence. This script
therefore does two things differently:

  1. Reports the RAW generated strings, not just a score, so a human can see
     whether a number is real.
  2. Scores with GSM8K-style exact-match on the final numeric answer where the
     prompt has one, which cannot be satisfied by repeating a word.

It sweeps latent depth and reports, per depth:
  - exact-match accuracy on a small GSM8K-style set
  - mean latency and tokens/s for the whole turn
  - mean latent steps actually taken (so an early stop is visible)
  - KV positions consumed (the real cost of a latent step)

Usage:
    python tools/mas-server/latent_bench.py [--base http://127.0.0.1:9024]
                                            [--depths 0,2,4,8,16,32]
                                            [--samples 3] [--json out.json]
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

# A small, self-contained GSM8K-style set: arithmetic and one-step reasoning with
# a single unambiguous integer answer. Kept inline so the bench needs no dataset
# download and stays deterministic.
QA = [
    ("Natalia sold clips to 20 friends in April, and then she sold half as many "
     "clips in May. How many clips did she sell altogether?",
     "30"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes "
     "of babysitting. How much did she earn?",
     "10"),
    ("Betty is saving money for a new wallet which costs $100. She has only half "
     "of the money she needs. Her parents decided to give her $15, and her "
     "grandparents twice as much as her parents. How much more money does Betty "
     "need to buy the wallet?",
     "5"),
    ("James writes a 3-page letter to 2 different friends twice a week. How many "
     "pages does he write a year?",
     "624"),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many "
     "bolts in total does it take?",
     "3"),
    ("A bakery sells 8 trays of cookies a day, 12 cookies per tray. How many "
     "cookies in 5 days?",
     "480"),
]


def _post(base, path, payload, timeout=600):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get(base, path, timeout=60):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _extract_number(text):
    """Last integer in the string - GSM8K answers are the final number."""
    if not text:
        return None
    nums = re.findall(r"-?\d[\d,]*", text)
    if not nums:
        return None
    return int(nums[-1].replace(",", ""))


def run_one(base, question, depth, n_tok, transform, stop, pool_window):
    """Run one question at one latent depth. Returns a record dict."""
    try:
        _post(base, "/agents/clear", {}, timeout=30)
    except Exception:
        pass
    _post(base, "/v1/agents", {"role": "bench", "max_tokens": 512,
                               "temperature": 0.0, "top_k": 1}, timeout=60)

    segs = [{"kind": "tokens", "text": question + " "}]
    if depth > 0:
        seg = {"kind": "latent", "latent_steps": depth, "align": 1,
               "latent_transform": transform}
        if stop > 0:
            seg["latent_stop_on_repeat"] = stop
        if pool_window > 1:
            seg["latent_pool_window"] = pool_window
        segs.append(seg)
    segs.append({"kind": "tokens", "text": "", "sample": 1})
    for _ in range(n_tok - 1):
        segs.append({"kind": "tokens", "text": "", "sample": 1})

    # The bench must report ntok EMITTED text tokens on both arms. A latent depth
    # displaces text tokens (the latent span consumes what the text baseline would
    # have emitted), so comparing wall time without pinning the token count is
    # meaningless: a latent run can look faster purely by emitting fewer tokens.
    # ntok is therefore only an upper bound; the actual emitted count is whatever
    # /turn reports as n_tokens, and both arms are normalised by it below.
    try:
        _post(base, "/mas/latent/stats/reset", {}, timeout=15)
    except Exception:
        # Correct verb is DELETE; the wrong verb silently leaves counters
        # cumulative, so always try it.
        try:
            req = urllib.request.Request(base + "/mas/latent/stats", method="DELETE")
            with urllib.request.urlopen(req, timeout=15) as r:
                r.read()
        except Exception:
            pass

    t0 = time.time()
    r = _post(base, "/turn", {"agent_idx": 0, "segments": segs}, timeout=600)
    wall = time.time() - t0

    text = (r.get("text") or "").strip()
    try:
        stats = _get(base, "/mas/latent/stats", timeout=15)
    except Exception:
        stats = {}

    return {
        "text": text,
        "wall_s": round(wall, 3),
        "n_tokens": int(r.get("n_tokens", 0) or 0),
        "latent_steps": stats.get("latent_steps", 0),
        "latent_ms": stats.get("latent_ms", 0),
        "kv_positions": stats.get("kv_positions", 0),
        "tokens_sampled": stats.get("tokens_sampled", 0),
        "text_ms": stats.get("text_ms", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:9024")
    ap.add_argument("--depths", default="0,2,4,8,16")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--ntok", type=int, default=32)
    ap.add_argument("--transform", type=int, default=2,
                    help="latent_transform: 0 raw, 1 prior, 2 delta, 3 ema, 4 pool")
    ap.add_argument("--stop", type=float, default=0.99,
                    help="latent_stop_on_repeat (0 disables)")
    ap.add_argument("--pool-window", type=int, default=1)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    base = args.base.rstrip("/")
    try:
        health = _get(base, "/health", timeout=15)
    except Exception as e:
        print("server not reachable at %s: %s" % (base, e), file=sys.stderr)
        return 2
    print("server: model_loaded=%s cuda=%s" %
          (health.get("model_loaded"), health.get("cuda")))

    # Alignment must be live or every latent step is off-distribution.
    try:
        al = _get(base, "/mas/alignment/status", timeout=15)
        print("alignment: use=%s n_embd=%s rescale=%s target_norm=%s" %
              (al.get("use_alignment"), al.get("n_embd"),
               al.get("rescale_norm"), al.get("target_norm")))
        if not al.get("use_alignment"):
            print("WARNING: alignment is OFF - latent quality will be far worse.")
    except Exception as e:
        print("alignment status unavailable: %s" % e)

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    out = {"base": base, "depths": depths, "results": []}

    print()
    hdr = "%-6s %-9s %-9s %-6s %-8s %-7s %-7s %s" % (
        "depth", "exact", "wall_s", "tok", "lat_ms", "steps", "kv_pos", "sample")
    print(hdr)
    print("-" * len(hdr))

    for d in depths:
        exact = 0
        total = 0
        walls, lat_ms, steps, kvs, samples, toks = [], [], [], [], [], []
        for sample in range(args.samples):
            for q, want in QA:
                rec = run_one(base, q, d, args.ntok, args.transform,
                              args.stop, args.pool_window)
                total += 1
                got = _extract_number(rec["text"])
                if got is not None and got == int(want):
                    exact += 1
                walls.append(rec["wall_s"])
                lat_ms.append(rec["latent_ms"])
                steps.append(rec["latent_steps"])
                kvs.append(rec["kv_positions"])
                toks.append(rec["n_tokens"])
                samples.append(rec["text"])
        acc = (exact / total) if total else 0.0
        # Token-normalised speed is the honest comparator: wall time alone rewards
        # a run that emitted fewer tokens.
        tok_mean = (sum(toks) / len(toks)) if toks else 0.0
        tps_mean = (tok_mean / (sum(walls) / len(walls))) if (toks and walls and
                    sum(walls) > 0) else 0.0
        row = {
            "depth": d, "exact": exact, "total": total,
            "accuracy": round(acc, 3),
            "wall_s_mean": round(sum(walls) / len(walls), 3) if walls else 0,
            "tokens_mean": round(tok_mean, 1),
            "tokens_per_s_mean": round(tps_mean, 1),
            "latent_ms_mean": round(sum(lat_ms) / len(lat_ms), 1) if lat_ms else 0,
            "steps_mean": round(sum(steps) / len(steps), 1) if steps else 0,
            "kv_pos_mean": round(sum(kvs) / len(kvs), 1) if kvs else 0,
            "samples": samples,
        }
        out["results"].append(row)
        print("%-6d %-9s %-9.3f %-6.1f %-8.0f %-7.1f %-7.1f %s" % (
            d, "%d/%d" % (exact, total), row["wall_s_mean"],
            row["tokens_mean"], row["latent_ms_mean"], row["steps_mean"],
            row["kv_pos_mean"],
            (samples[0][:46] if samples else "")))
        # Every sample string is printed: a single 'best' sample can hide a run
        # that is degenerate 5 times out of 6, and the whole point of this bench
        # is that the human READS the raw text.
        for si, s in enumerate(samples):
            print("       [%d] %r" % (si, s[:120]))

    # Token-savings / wall-clock summary vs the depth-0 text baseline.
    base_row = next((r for r in out["results"] if r["depth"] == 0), None)
    if base_row:
        print()
        print("=== vs text baseline (depth 0) ===")
        print("wall_x  is RAW wall speedup (misleading: a latent run emits fewer")
        print("tokens, so it can win on wall while doing less work).")
        print("tok_x   is the HONEST comparator: tokens/s ratio at equal work.")
        print()
        print("%-6s %-12s %-12s %-12s %s" % ("depth", "wall_x", "tok_x",
                                             "steps", "note"))
        b_tps = base_row.get("tokens_per_s_mean", 0) or 0
        for r in out["results"]:
            if r["depth"] == 0:
                continue
            w = base_row["wall_s_mean"]
            mult = (w / r["wall_s_mean"]) if r["wall_s_mean"] else 0
            r_tps = r.get("tokens_per_s_mean", 0) or 0
            tps_mult = (r_tps / b_tps) if b_tps else 0
            note = ""
            if r["steps_mean"] and r["depth"]:
                note = "%.2f steps/depth (early stop %s)" % (
                    r["steps_mean"] / r["depth"],
                    "ACTIVE" if r["steps_mean"] < r["depth"] * 0.9 else "off")
            print("%-6d %-12.2f %-12.2f %-12.1f %s" % (r["depth"], mult,
                                                       tps_mult,
                                                       r["steps_mean"], note))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())