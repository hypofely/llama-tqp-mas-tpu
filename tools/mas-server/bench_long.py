"""Long-text generation benchmark.

The short-answer benchmark cannot answer the question actually asked: does
latent thinking help when the OUTPUT IS LONG? On a 12-question arithmetic set
there is nothing to compress and no room for reasoning to compound.

Runs multi-hundred-token tasks and measures, per arm:

  * perplexity  - the model's own log-softmax over the tokens it emitted. The
    only objective fluency/quality signal for free-form text: length and
    repetition proxies cannot separate real prose from fluent-sounding loops.
    (Computed in llama-mas.cpp at each sample site, exposed as out.perplexity.)
  * tokens / words / wall - the cost axis
  * ceiling hit - did it degenerate into looping until the cap (a real failure)
  * distinct-4gram ratio - repetition proxy, cross-check on perplexity
  * sentence stats - count, mean and stdev of sentence length

Arms are the configurations named by the user, plus the chain variants:
  text          plain text, no latent
  lat10_t       10 latent steps, then text
  t_lat10_t     text -> latent -> text   (the "thinking, string" interleaving)
  t_l10_t_l10   longer interleave
  ag3_l0        3 agents chained by KV, no latent
  ag3_l10       3 agents chained, 10 latent each
  ag5_l0        5 agents chained, no latent

Usage: python bench_long.py --port 9028 [--ntok 700] [--json ...]
"""
import argparse
import json
import re
import statistics
import time
import urllib.request

TASKS = [
    {
        "id": "regen",
        "prompt": ("Write a detailed technical explanation of how a hybrid vehicle's "
                   "regenerative braking system works. Cover the energy flow, the "
                   "control logic that decides between friction and regenerative "
                   "braking, and the failure modes an engineer should test for."),
    },
    {
        "id": "planning",
        "prompt": ("You are planning the migration of a 20-year-old monolithic inventory "
                   "application to a service architecture. The system has 400k lines of "
                   "code, no test suite, and a live customer base. Produce a step-by-step "
                   "migration plan that keeps the old system running throughout, names "
                   "the risk of each step, and states the rollback criteria."),
    },
    {
        "id": "debug",
        "prompt": ("A distributed job scheduler occasionally loses tasks. The scheduler "
                   "stores jobs in PostgreSQL, dispatches them over NATS, and workers "
                   "ack with a heartbeat. Symptoms: tasks vanish after a deploy, and a "
                   "restart recovers some but not all. Reason through the plausible "
                   "causes in order of likelihood and give a concrete diagnostic step "
                   "for each, then propose a fix."),
    },
    {
        "id": "design",
        "prompt": ("Design a rate limiter for a multi-tenant API gateway that must handle "
                   "100k requests per second across 40 tenants, where each tenant has "
                   "its own quota and burst allowance. Compare token bucket, sliding "
                   "window and fixed window under this load, justify the choice, and "
                   "describe how you would make it correct under clock skew and "
                   "network partition."),
    },
    {
        "id": "trace",
        "prompt": ("Given this short Python function, trace its execution for the input "
                   "data = [{'a': 3}, {'a': 5}, {'b': 1}], state the exact output, and "
                   "explain every branch that was and was not taken:\n\n"
                   "def f(items):\n"
                   "    out = []\n"
                   "    for it in items:\n"
                   "        if 'a' in it:\n"
                   "            out.append(it['a'] * 2)\n"
                   "        else:\n"
                   "            out.append(0)\n"
                   "    return sum(out), out\n"),
    },
    {
        "id": "tradeoff",
        "prompt": ("An engineering team must choose between three storage engines for a "
                   "write-heavy telemetry ingest pipeline: LSM-tree, B-tree, and "
                   "append-only log with periodic compaction. Enumerate the workload "
                   "characteristics that matter, derive a decision for each, and state "
                   "what measurement would settle the choice empirically."),
    },
]

SENT = re.compile(r"[.!?](?:\s|$)")


def metrics(txt, tok, wall, ppl, ntok):
    words = txt.split()
    sents = [s for s in SENT.split(txt) if s.strip()]
    slen = [len(s.split()) for s in sents] or [0]
    grams = []
    n = 4
    for i in range(max(0, len(words) - n + 1)):
        grams.append(tuple(words[i:i + n]))
    distinct = len(set(grams)) / len(grams) if grams else 1.0
    uniq = len(set(words)) / len(words) if words else 1.0
    return {
        "tokens": tok, "words": len(words), "wall_s": round(wall, 2),
        "perplexity": round(ppl, 2) if ppl and ppl > 0 else None,
        "logprob_n": 0,
        "n_latent": 0,
        "distinct_4gram": round(distinct, 4),
        "type_token": round(uniq, 4),
        "sentences": len(sents),
        "sent_len_mean": round(statistics.mean(slen), 1),
        "sent_len_sd": round(statistics.pstdev(slen), 1),
        "ceiling_hit": tok >= ntok,
        "chars_per_tok": round(len(txt) / tok, 2) if tok else 0.0,
        "raw": txt,
    }


def segs_for(prompt, plan):
    """Build a /turn segment list.

    The /turn schema has NO per-segment max_tokens: the prompt is a text segment,
    a latent segment carries latent_steps, and TEXT OUTPUT is one
    {"kind":"tokens","sample":1} segment PER TOKEN. Found the hard way - a text
    segment with sample unset decodes but emits nothing, so every text arm
    silently returned an empty string.
    """
    segs = [{"kind": "tokens", "text": prompt}]
    for kind, n in plan:
        if kind == "latent":
            segs.append({"kind": "latent", "latent_steps": n, "align": 1})
        else:
            for _ in range(n):
                segs.append({"kind": "tokens", "text": "", "sample": 1})
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9028)
    ap.add_argument("--ntok", type=int, default=700)
    ap.add_argument("--cache", type=str, default=None)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--json", type=str, default="bench_long_results.json")
    ap.add_argument("--tasks", type=str, default=None)
    ap.add_argument("--arms", type=str, default=None)
    args = ap.parse_args()

    B = "http://127.0.0.1:%d" % args.port

    def req(path, body=None, t=2400):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(B + path, data=data,
                                   headers={"Content-Type": "application/json"} if data else {},
                                   method="POST" if data else "GET")
        with urllib.request.urlopen(r, timeout=t) as resp:
            return resp.status, json.loads(resp.read().decode())

    tasks = TASKS
    if args.tasks:
        want = set(args.tasks.split(","))
        tasks = [t for t in TASKS if t["id"] in want]

    print("=== LONG-TEXT BENCHMARK: %d tasks, ceiling=%d, :%d" %
          (len(tasks), args.ntok, args.port))

    if args.cache:
        st, b = req("/mas/alignment/load", {"path": args.cache, "vocab_stride": args.stride})
        print("alignment cache loaded=%s" % b.get("loaded"))
        req("/mas/alignment/toggle",
            {"use_alignment": True, "rescale_norm": True, "parallel_threads": 4})

    def fresh_agent(tag):
        """Clear then create ONE agent. /turn rejects an out-of-range agent_idx, and
        KV accumulates on a reused agent (per-agent quota = n_ctx/n_agents_max), so
        every arm must start from an empty agent at index 0."""
        try:
            req("/agents/clear", {})
        except Exception:
            pass
        req("/v1/agents", {"role": "bench_" + tag, "max_tokens": 16384,
                           "temperature": 0.0, "top_k": 1,
                           "repeat_penalty": 1.3, "repeat_last_n": 64}, t=180)

    def run_single(task, plan):
        # The prompt MUST be a text segment, not the `prompt` field: the field
        # prefills without marking an output row, so n_outputs stays 0 and the
        # sampler's llama_get_logits_ith(-1) hits the fork's fatal
        # "corrupt output buffer" guard. perf_8b.py had this right all along.
        body = {"segments": segs_for(task["prompt"], plan),
                "agent_idx": 0, "stop_on_eog": True}
        # Wall time is measured CLIENT-side: /turn reports tokens/sec but no
        # wall_seconds field, and the server-side figure excludes HTTP and
        # agent setup. The request time is what a caller actually pays.
        t0 = time.time()
        st, b = req("/turn", body)
        wall = time.time() - t0
        ppl = b.get("perplexity") or 0.0
        m = metrics(b.get("text", ""), b.get("n_tokens", 0),
                    wall, ppl, args.ntok)
        m["logprob_n"] = b.get("logprob_n", 0)
        m["n_latent"] = b.get("n_latent_steps", 0)
        return m

    def tok_count(b):
        """Token COUNT. Chain returns `total_tokens` (int) and `wall_ms`; /turn
        returns `tokens` as an id array. Reading "tokens" alone yielded 0."""
        for k in ("total_tokens", "tokens", "n_tokens"):
            v = b.get(k)
            if v is None:
                continue
            return len(v) if isinstance(v, list) else v
        return 0

    def run_chain(task, agents, lat):
        body = {"prompt": task["prompt"], "agents": agents, "latent_steps": lat,
                "ntok": args.ntok, "max_tokens": args.ntok, "stop_on_eog": True,
                "feedback_source": 0}
        t0 = time.time()
        st, b = req("/mas/latent-chain", body)
        wall = time.time() - t0
        txt = b.get("text") or ""
        if isinstance(txt, list):
            txt = "\n".join(txt)
        m = metrics(txt, tok_count(b), wall, 0.0, args.ntok)
        m["n_latent"] = b.get("total_latent_steps", b.get("latent_steps", 0))
        return m

    n = args.ntok
    ARMS = [
        ("text",      lambda t: (fresh_agent("text"), run_single(t, [("text", n)]))[1]),
        ("lat10_t",   lambda t: (fresh_agent("lat10"), run_single(t, [("latent", 10), ("text", n)]))[1]),
        ("t_lat10_t", lambda t: (fresh_agent("tlat"), run_single(
            t, [("text", 150), ("latent", 10), ("text", max(1, n - 150))]))[1]),
        ("t_l10_t_l10", lambda t: (fresh_agent("tl10"), run_single(
            t, [("text", 120), ("latent", 10), ("text", 120), ("latent", 10),
                ("text", max(1, n - 240))]))[1]),
        ("ag3_l0",    lambda t: run_chain(t, 3, 0)),
        ("ag3_l10",   lambda t: run_chain(t, 3, 10)),
        ("ag5_l0",    lambda t: run_chain(t, 5, 0)),
    ]
    if args.arms:
        want = set(args.arms.split(","))
        ARMS = [a for a in ARMS if a[0] in want]

    out = []
    for name, fn in ARMS:
        t0 = time.time()
        rows = []
        for task in tasks:
            try:
                m = fn(task)
            except Exception as e:
                m = {"tokens": 0, "words": 0, "wall_s": 0.0, "perplexity": None,
                     "distinct_4gram": 0.0, "type_token": 0.0, "sentences": 0,
                     "sent_len_mean": 0.0, "sent_len_sd": 0.0, "ceiling_hit": False,
                     "chars_per_tok": 0.0, "logprob_n": 0, "n_latent": 0,
                     "raw": "ERROR " + str(e)[:200]}
            m["task"] = task["id"]
            rows.append(m)
            print("  [%s/%s] tok=%4d words=%4d ppl=%s 4g=%.3f wall=%.1fs ceil=%s"
                  % (name, task["id"], m["tokens"], m["words"],
                     ("%.1f" % m["perplexity"]) if m.get("perplexity") else "-",
                     m["distinct_4gram"], m["wall_s"], m["ceiling_hit"]), flush=True)
        ceil = sum(1 for r in rows if r["ceiling_hit"])
        ppls = [r["perplexity"] for r in rows if r.get("perplexity")]
        toks = [r["tokens"] for r in rows]
        wall = [r["wall_s"] for r in rows]
        d4 = [r["distinct_4gram"] for r in rows]
        rec = {
            "arm": name, "n_tasks": len(rows), "ceiling_hits": ceil,
            "tokens_median": statistics.median(toks), "tokens_total": sum(toks),
            "wall_median": round(statistics.median(wall), 2),
            "wall_total": round(sum(wall), 2),
            "perplexity_median": round(statistics.median(ppls), 2) if ppls else None,
            "distinct4_median": round(statistics.median(d4), 4),
            "rows": rows,
        }
        out.append(rec)
        print("--> %-12s tok/med=%5.0f  ppl=%s  distinct4=%.3f  wall/med=%6.2fs  "
              "ceil=%d  [%.0fs]"
              % (name, rec["tokens_median"],
                 ("%.1f" % rec["perplexity_median"]) if rec["perplexity_median"] else "-",
                 rec["distinct4_median"], rec["wall_median"], ceil, time.time() - t0),
              flush=True)

    print("")
    print("=== SUMMARY ===")
    print("%-12s %8s %8s %8s %8s %8s %6s" %
          ("arm", "tok/med", "ppl", "dist4", "words", "wall_s", "ceil"))
    for r in out:
        print("%-12s %8.0f %8s %8.3f %8.0f %8.2f %6d"
              % (r["arm"], r["tokens_median"],
                 ("%.1f" % r["perplexity_median"]) if r["perplexity_median"] else "-",
                 r["distinct4_median"],
                 statistics.median([x["words"] for x in r["rows"]]),
                 r["wall_median"], r["ceiling_hits"]))

    json.dump({"args": vars(args), "results": out}, open(args.json, "w"))
    print("")
    print("wrote %s" % args.json)


if __name__ == "__main__":
    main()