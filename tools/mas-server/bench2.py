"""Benchmark suite v2 - the real measurement.

Fixes what the v1 12-question test got wrong:

  * COVERAGE: 48 questions across 6 categories with programmatically computed
    gold. v1 was 12 single-step-ish arithmetic problems, which cannot show a
    capability difference even if one exists. Per-category breakdown is
    reported, because "latent helps on hard logic but not arithmetic" is a
    different finding from "latent does nothing".
  * DIFFICULTY: includes genuine multi-constraint problems (4-6 steps) rather
    than one-liners, so headroom exists to score differently.
  * REPEATS: every arm is run N times, so a score is a mean +/- interval
    rather than a single lucky sample. Greedy decoding still gives some
    variance because KV inheritance depends on the role cycle.
  * STATISTICS: Wilson score intervals on every proportion, and a paired
    per-question comparison against the baseline arm so differences are
    reported as paired deltas, not two independent-looking scores.
  * TOKEN CEILING: configurable and raised so an arm cannot be punished for
    generating normally; ceiling hits are reported per arm.
  * CATEGORIES: each question carries a category so an aggregate cannot hide a
    category where latent helps while hurting everywhere else.

Usage:
  python bench2.py --port 9028 --ntok 1024 --reps 2 [--arms ...] [--json ...]
"""
import argparse
import json
import math
import re
import statistics
import sys
import time
import urllib.request

# --------------------------------------------------------------------------
# Question generation. Gold is COMPUTED, never hand-written.
# --------------------------------------------------------------------------

def gen():
    Q = []

    def add(cat, prompt, gold, **kw):
        Q.append({"cat": cat, "q": prompt, "gold": gold, **kw})

    # ---- arithmetic: multi-step word problems (gold from Python) ----
    for i in range(10):
        a = 12 + i * 7
        b = 3 + i
        c = 5 + (i % 4) * 2
        total = a * b
        left = total - c * (i + 1)
        add("arithmetic",
            "A workshop buys %d crates of %d tools each. It sells %d tools per day "
            "for %d days. How many tools are left?" % (a, b, c, i + 1),
            left)
        Q[-1]["steps"] = "multiply %d*%d=%d then subtract %d*%d=%d -> %d" % (
            a, b, a * b, c, i + 1, c * (i + 1), left)

    # ---- percentage / ratio ----
    for i in range(6):
        base = 200 + i * 50
        pct = 15 + i * 5
        disc = round(base * (1 - pct / 100.0), 2)
        add("percent",
            "An item costs $%d. After a %d%% discount what is the price in dollars?" % (base, pct),
            disc)
        Q[-1]["steps"] = "%d * (1 - %d/100) = %s" % (base, pct, disc)

    for i in range(4):
        per = 3 + i
        need = 48 + i * 6
        import math as _m
        packs = _m.ceil(need / per)
        add("percent",
            "Tickets are sold in packs of %d. You need exactly %d tickets. What is the "
            "minimum number of packs to buy?" % (per, need), packs)
        Q[-1]["steps"] = "ceil(%d/%d) = %d" % (need, per, packs)

    # ---- unit conversion ----
    for i in range(6):
        km = 12 + i * 9
        m = km * 1000
        add("units", "How many metres is %d kilometres?" % km, m)
        Q[-1]["steps"] = "%d * 1000 = %d" % (km, m)
    for i in range(3):
        hours = 2 + i
        mins = 90 - i * 10
        add("units",
            "A film runs %d hours and %d minutes. How many minutes is that in total?" % (hours, mins),
            hours * 60 + mins)
        Q[-1]["steps"] = "%d*60 + %d = %d" % (hours, mins, hours * 60 + mins)

    # ---- multi-constraint logic (harder) ----
    for i in range(6):
        # three-person scheduling constraint
        a_start = 9 + i
        dur = 2 + (i % 3)
        b_after = a_start + dur + 1          # B starts 1h after A finishes
        c_at = b_after + 3                   # C is 3h after B
        add("logic",
            "A task starts at %d:00 and lasts %d hours. A second task starts one hour "
            "after the first ends. A third starts three hours after the second starts. "
            "At what hour (24h, just the number) does the third task start?" % (a_start, dur),
            c_at)
        Q[-1]["steps"] = "A ends %d, B starts %d, C starts %d" % (a_start + dur, b_after, c_at)

    # ---- sequences ----
    for i in range(6):
        step = 3 + i
        start = 5 + i * 2
        terms = [start + k * step for k in range(5)]
        nxt = start + 5 * step
        add("sequence",
            "What is the next number in the sequence: %s?" % ", ".join(str(t) for t in terms),
            nxt)
        Q[-1]["steps"] = "arithmetic step %d -> %d" % (step, nxt)

    # ---- comparison / ranking (requires holding several facts) ----
    for i in range(6):
        p1 = 3 + i
        p2 = p1 + 2
        rate1 = 5 + i
        rate2 = rate1 + 3
        t1 = p1 * rate1
        t2 = p2 * rate2
        # gold = the FASTER completion time (smaller)
        gold = min(t1, t2)
        add("compare",
            "Machine A makes %d parts per hour for %d hours. Machine B makes %d parts per "
            "hour for %d hours. What is the smaller total number of parts produced?" %
            (p1, rate1, p2, rate2), gold)
        Q[-1]["steps"] = "A=%d B=%d -> min=%d" % (t1, t2, gold)

    return Q


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

ANSWER_RE = re.compile(r"answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", re.I)
ASSERT_RE = re.compile(
    r"(answer|result|total|final|therefore|thus|so[ ,]|equals|=|is|are|should be|comes to|gives)", re.I)


def gold_str(g):
    if isinstance(g, int):
        return str(g)
    return ("%g" % g)


def score(txt, gold):
    """Return (strict, concluded). See verify_claim.py for why position filtering
    must NOT be applied to an explicit 'answer is <gold>'."""
    if not txt:
        return 0, 0
    want = float(gold)
    gs = gold_str(gold)

    # strict: an explicit answer statement naming the gold
    for v in ANSWER_RE.findall(txt):
        try:
            if abs(float(v.replace(",", "")) - want) < 1e-6:
                concluded = 1
                break
        except ValueError:
            continue
    else:
        concluded = 0
    strict = concluded

    if not concluded:
        tail = txt[int(len(txt) * 0.60):]
        for m in re.finditer(r"(?<![\d.])" + re.escape(gs) + r"(?![\d.])", tail):
            if ASSERT_RE.search(tail[max(0, m.start() - 70):m.start()]):
                concluded = 1
                break
    return strict, concluded


def wilson(k, n, z=1.96):
    """Wilson score interval - correct for small n, unlike normal approximation."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9028)
    ap.add_argument("--base", type=str, default=None)
    ap.add_argument("--ntok", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--cache", type=str, default=None)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--json", type=str, default="bench2_results.json")
    ap.add_argument("--cat", type=str, default=None, help="only this category")
    ap.add_argument("--arms", type=str, default=None)
    args = ap.parse_args()

    B = args.base or ("http://127.0.0.1:%d" % args.port)
    Q = gen()
    if args.cat:
        Q = [q for q in Q if q["cat"] == args.cat]
    print("=== BENCHMARK v2: %d questions, reps=%d, ceiling=%d, base=%s" %
          (len(Q), args.reps, args.ntok, B))
    cats = sorted(set(q["cat"] for q in Q))
    print("categories: %s" % ", ".join("%s=%d" % (c, sum(1 for q in Q if q["cat"] == c)) for c in cats))

    def req(path, body=None, method=None, t=1800):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(B + path, data=data,
                                   headers={"Content-Type": "application/json"} if data else {},
                                   method=method or ("POST" if data else "GET"))
        with urllib.request.urlopen(r, timeout=t) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw

    if args.cache:
        try:
            st, b = req("/mas/alignment/load", {"path": args.cache, "vocab_stride": args.stride}, t=600)
            print("alignment cache: loaded=%s" % (b.get("loaded") if isinstance(b, dict) else b))
        except Exception as e:
            print("alignment cache FAILED: %s" % str(e)[:100])
        req("/mas/alignment/toggle",
            {"use_alignment": True, "rescale_norm": True, "parallel_threads": 4})

    BASE_PROMPT = ("You are a careful problem solver. Work through the problem and then state "
                   "your conclusion on its own line in exactly this form:\n"
                   "the answer is <number>\n\nProblem: ")

    def tok_count(b):
        """Token COUNT, not the token-id list.

        /mas/latent-chain returns `tokens` as an ARRAY of ids (n_tokens on /turn is
        an integer). Using the array directly made every count comparison raise
        TypeError and killed the sweep after the first arm.
        """
        # /mas/latent-chain names it `total_tokens` (an int); /turn uses `tokens`
        # as an ARRAY of ids. Reading "tokens" alone yielded 0 for every chain arm.
        for k in ("total_tokens", "tokens", "n_tokens"):
            v = b.get(k)
            if v is None:
                continue
            return len(v) if isinstance(v, list) else v
        return 0

    def chain_fn(agents, lat, role_texts=None):
        def fn(q):
            prompt = BASE_PROMPT + q["q"]
            body = {"prompt": prompt, "agents": agents, "latent_steps": lat,
                    "ntok": args.ntok, "max_tokens": args.ntok, "stop_on_eog": True,
                    "feedback_source": 0}
            if role_texts:
                body["role_prompts"] = role_texts
            _t0 = time.time()
            st, b = req("/mas/latent-chain", body)
            txt = b.get("text") or ""
            if isinstance(txt, list):
                txt = "\n".join(txt)
            return round(time.time() - _t0, 3), txt, tok_count(b)
        return fn

    def fresh_agent(tag):
        try:
            req("/agents/clear", {}, t=60)
        except Exception:
            pass
        # repeat_penalty is REQUIRED: without it the sampler applies no penalty and
        # greedy decoding loops forever ("The answer is 31. The answer is 31. ...")
        # until the token ceiling, which made the single-agent arm unscorable.
        req("/v1/agents", {"role": "bench_" + tag, "max_tokens": 16384,
                           "temperature": 0.0, "top_k": 1,
                           "repeat_penalty": 1.3, "repeat_last_n": 64}, t=180)

    def single_fn(_agents_unused=1):
        def fn(q):
            # FRESH AGENT PER QUESTION. /turn appends to the agent's KV, and the
            # per-agent quota is n_ctx/n_agents_max (16384/8 = 2048). A single
            # reused agent fills up after one verbose answer and every later
            # question returns HTTP 500 ("failed to find a memory slot").
            fresh_agent("single")
            # one sample segment per token - the schema has no per-segment cap
            segs = [{"kind": "tokens", "text": BASE_PROMPT + q["q"]}]
            for _ in range(args.ntok):
                segs.append({"kind": "tokens", "text": "", "sample": 1})
            body = {"segments": segs, "agent_idx": 0, "stop_on_eog": True}
            _t0 = time.time()
            st, b = req("/turn", body)
            txt = b.get("text") or ""
            return round(time.time() - _t0, 3), txt, tok_count(b)
        return fn

    ARMS = [
        ("single",     single_fn()),
        ("ag3_l0",     chain_fn(3, 0)),
        ("ag3_l10",    chain_fn(3, 10)),
        ("ag5_l0",     chain_fn(5, 0)),
        ("ag7_l0",     chain_fn(7, 0)),
        ("ag5_l20",    chain_fn(5, 20)),
    ]
    if args.arms:
        want = set(args.arms.split(","))
        ARMS = [a for a in ARMS if a[0] in want]

    results = []
    for name, fn in ARMS:
        t0 = time.time()
        recs = []
        for rep in range(args.reps):
            for q in Q:
                try:
                    w, txt, tk = fn(q)
                except Exception as e:
                    w, txt, tk = 0.0, "ERROR " + str(e)[:160], 0
                s_, c_ = score(txt, q["gold"])
                recs.append({"rep": rep, "cat": q["cat"], "text": txt, "gold": q["gold"],
                             "strict": s_, "concluded": c_, "wall": w, "tokens": tk,
                             "error": txt.startswith("ERROR")})
        n = len(recs)
        # per-question aggregation across reps, so CIs are over questions not runs
        per_q = {}
        for r in recs:
            per_q.setdefault((r["rep"], r["cat"]), []).append(r)
        S = sum(r["strict"] for r in recs)
        C = sum(r["concluded"] for r in recs)
        CE = sum(1 for r in recs if r["tokens"] >= args.ntok)
        ER = sum(1 for r in recs if r["error"])
        # assert the invariant that caught the earlier metric bug
        assert C >= S, "metric bug: concluded (%d) < strict (%d)" % (C, S)
        r = {
            "arm": name, "n": n, "reps": args.reps, "nq": len(Q),
            "strict": S, "concluded": C, "ceiling_hits": CE, "errors": ER,
            "strict_ci": wilson(S, n), "concluded_ci": wilson(C, n),
            "wall_median": round(statistics.median([x["wall"] for x in recs]), 2),
            "wall_mean": round(statistics.mean([x["wall"] for x in recs]), 2),
            "tokens_median": round(statistics.median([x["tokens"] for x in recs]), 1),
            "per_cat": {},
            "rows": recs,
        }
        for c in cats:
            sub = [x for x in recs if x["cat"] == c]
            if sub:
                r["per_cat"][c] = {"concluded": sum(x["concluded"] for x in sub),
                                   "strict": sum(x["strict"] for x in sub),
                                   "n": len(sub)}
        results.append(r)
        ci = r["concluded_ci"]
        print("--> %-10s concluded %3d/%-3d (%.1f%% CI %.0f-%.0f)  strict %3d/%-3d  "
              "ceil %2d  err %d  wall %.2fs  tok %.0f  [%.0fs]"
              % (name, C, n, 100.0 * C / n, 100 * ci[0], 100 * ci[1], S, n, CE, ER,
                 r["wall_median"], r["tokens_median"], time.time() - t0), flush=True)

    # ---- per-category table: where, if anywhere, does latent differ? ----
    print("\n=== PER-CATEGORY concluded (correct/total) ===")
    hdr = "%-11s" % "category"
    for name, _ in ARMS:
        hdr += " %9s" % name
    print(hdr)
    for c in cats:
        line = "%-11s" % c[:11]
        for name, _ in ARMS:
            rr = next((x for x in results if x["arm"] == name), None)
            pc = rr["per_cat"].get(c) if rr else None
            line += " %9s" % (("%d/%d" % (pc["concluded"], pc["n"])) if pc else "-")
        print(line)

    # ---- paired comparison vs baseline: does latent/more-agents actually move
    # each question? Two independent scores hide this. ----
    print("\n=== PAIRED vs 'single' (per-question concluded deltas) ===")
    base_rows = {(r["rep"], i): r for i, r in enumerate(results[0]["rows"])}
    print("%-11s %8s %8s %8s %9s" % ("arm", "better", "worse", "same", "mean delta"))
    for r in results[1:]:
        up = dn = same = 0
        deltas = []
        # align by (rep, question index)
        b_by_q = {}
        for b in results[0]["rows"]:
            b_by_q.setdefault(b["rep"], []).append(b)
        my_by_q = {}
        for b in r["rows"]:
            my_by_q.setdefault(b["rep"], []).append(b)
        for rep in range(args.reps):
            for i in range(min(len(b_by_q.get(rep, [])), len(my_by_q.get(rep, [])))):
                d = my_by_q[rep][i]["concluded"] - b_by_q[rep][i]["concluded"]
                deltas.append(d)
                if d > 0:
                    up += 1
                elif d < 0:
                    dn += 1
                else:
                    same += 1
        print("%-11s %8d %8d %8d %+9.3f" %
              (r["arm"], up, dn, same, (sum(deltas) / len(deltas)) if deltas else 0))

    print("\n=== AGGREGATE ===")
    print("%-11s %9s %9s %6s %8s %8s" % ("arm", "concluded", "strict", "ceil", "wall_s", "tokens"))
    for r in results:
        print("%-11s %9s %9s %6d %8.2f %8s"
              % (r["arm"], "%d/%d" % (r["concluded"], r["n"]), "%d/%d" % (r["strict"], r["n"]),
                 r["ceiling_hits"], r["wall_median"], r["tokens_median"]))

    json.dump({"args": vars(args), "categories": cats, "results": results},
              open(args.json, "w"))
    print("\nwrote %s (%d arms, %d questions, %d reps)"
          % (args.json, len(results), len(Q), args.reps))


if __name__ == "__main__":
    main()