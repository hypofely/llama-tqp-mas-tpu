"""Scale the winning axis: agent count, and harder problems.

Established (GSM8K-easy, 6 questions, Qwen3-8B, ChatML, EOG stop, 512 ceiling):
  1 agent no roles                 1/6   22.09 s   512 tok
  3 agents + KV handoff, 0 latent  6/6    8.00 s   184 tok

Open questions this answers:
  1. Does MORE agents keep helping, or does it saturate / degrade?
  2. Does the win survive HARDER problems (multi-step, not one-operation)?
  3. On harder problems, does latent thinking finally earn its cost?

Every arm uses ChatML and EOG stopping, so ntok is a ceiling not a target.
"""
import argparse
import json
import re
import statistics
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=9026)
ap.add_argument("--ntok", type=int, default=768)
ap.add_argument("--agents", type=str, default="1,3,5")
ap.add_argument("--steps", type=str, default="0", help="latent steps per agent, comma list")
ap.add_argument("--set", type=str, default="hard", choices=["hard", "easy"])
ap.add_argument("--json", type=str, default="scale_results.json")
args = ap.parse_args()

B = "http://127.0.0.1:%d" % args.port
CACHE = r"C:\+run\GPT_WEB\llama-tqp-mas-cpp\tools\mas-server\align_8b_s32.bin"

# Harder: multi-step, unit conversion, rates, combinatorics. Gold verified by hand.
HARD = [
    ("A train travels 90 km in 1.5 hours, then 120 km in 2 hours. What is its "
     "average speed for the whole trip in km/h?", 60),
    ("A shop marks up a $40 item by 25%, then offers a 20% discount on the marked "
     "price. What is the final price in dollars?", 40),
    ("Three pipes fill a tank. Pipe A alone takes 6 hours, B alone 12 hours, "
     "C alone 4 hours. How many hours do all three take together?", 2),
    ("A rectangle's length is 3 times its width. Its perimeter is 64 cm. "
     "What is its area in square centimetres?", 192),
    ("In a class of 30 students, 18 play football, 15 play basketball, and 5 play "
     "neither. How many play both sports?", 8),
    ("A car's value drops 20% each year. It costs $25000 new. What is its value "
     "after 2 years in dollars?", 16000),
]
EASY = [
    ("Natalia sold clips to 48 friends in April, and half as many in May. "
     "How many clips did she sell altogether?", 72),
    ("Weng earns $12 an hour for babysitting. She did 50 minutes yesterday. "
     "How much did she earn in dollars?", 10),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. "
     "How many bolts in total?", 3),
]
QS = HARD if args.set == "hard" else EASY

ROLE_POOL = [
    "Plan: list the given quantities, the unknown, and the operations needed. Do not compute yet.",
    "Execute the plan step by step, computing each intermediate value explicitly.",
    "Check the intermediate values for arithmetic or unit errors and correct them.",
    "Re-derive the result by a different route to cross-check it.",
    "State the final answer as 'The answer is <number>.' with no further commentary.",
]


def req(path, body=None, method=None, t=1800):
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


def chatml(q):
    return ("<|im_start|>user\n" + q +
            "\nSolve step by step, then state 'The answer is <number>.'"
            "<|im_end|>\n<|im_start|>assistant\n")


def correct(text, gold):
    m = re.findall(r"answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", text, re.I)
    got = m[-1].replace(",", "") if m else None
    if got is None:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        got = nums[-1].replace(",", "") if nums else None
    if got is None:
        return False
    try:
        return abs(float(got) - float(gold)) < 1e-6
    except ValueError:
        return False


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

st, b = req("/mas/alignment/load", {"path": CACHE, "vocab_stride": 32}, t=300)
print("align cache loaded=%s" % b.get("loaded"), flush=True)
req("/mas/alignment/toggle",
    {"use_alignment": True, "rescale_norm": True, "parallel_threads": 4}, t=60)

results = []


def single(q, ntok):
    try:
        req("/agents/clear", {}, t=60)
    except Exception:
        pass
    req("/v1/agents", {"role": "s", "max_tokens": 8192,
                       "temperature": 0.0, "top_k": 1}, t=120)
    segs = [{"kind": "tokens", "text": chatml(q)}]
    for _ in range(ntok):
        segs.append({"kind": "tokens", "text": "", "sample": 1, "stop_on_eog": 1})
    t0 = time.time()
    st, r = req("/turn", {"agent_idx": 0, "segments": segs}, t=1800)
    return time.time() - t0, (r.get("text") or ""), r.get("n_tokens", 0), 0


def chain(q, n_agents, steps, ntok):
    t0 = time.time()
    st, r = req("/mas/latent-chain", {
        "prompt": chatml(q), "agents": n_agents, "latent_steps": steps,
        "ntok": ntok, "feedback_source": 1, "align": True,
        # Cycle the pool: ROLE_POOL has 5 entries, so agents > 5 would get
        # NO role and (with latent_steps=0) an empty segment list -> rc=-1.
        "role_prompts": [ROLE_POOL[i % len(ROLE_POOL)] for i in range(n_agents)],
        "stop_on_eog": True,
    }, t=1800)
    if st != 200:
        return time.time() - t0, "ERROR " + json.dumps(r)[:150], 0, 0
    return (time.time() - t0, r.get("text") or "",
            r.get("total_tokens", 0), r.get("total_latent_steps", 0))


print("\n=== baseline: 1 agent, no roles (%s set, ceiling %d) ===" % (args.set, args.ntok),
      flush=True)
ok, walls, toks, rows = 0, [], [], []
for qi, (q, gold) in enumerate(QS):
    w, txt, tk, lt = single(q, args.ntok)
    c = correct(txt, gold)
    ok += int(c); walls.append(w); toks.append(tk)
    rows.append({"q": qi, "gold": gold, "correct": c, "wall_s": round(w, 3), "tokens": tk})
    print("q%d gold=%-6s %s wall=%.2fs tok=%d" % (qi, gold, "OK " if c else "MISS", w, tk), flush=True)
results.append({"arm": "single", "agents": 1, "steps": 0, "correct": ok,
                "n": len(QS), "wall_median": round(statistics.median(walls), 3),
                "tokens_median": statistics.median(toks), "rows": rows})
print("--> single %d/%d  %.2fs  %s tok" % (ok, len(QS), statistics.median(walls),
                                          statistics.median(toks)), flush=True)

for na in [int(x) for x in args.agents.split(",")]:
    if na < 2:
        continue
    for stp in [int(x) for x in args.steps.split(",")]:
        label = "chain a=%d lat=%d" % (na, stp)
        print("\n=== %s ===" % label, flush=True)
        ok, walls, toks, lats, rows = 0, [], [], [], []
        for qi, (q, gold) in enumerate(QS):
            w, txt, tk, lt = chain(q, na, stp, args.ntok)
            c = correct(txt, gold)
            ok += int(c); walls.append(w); toks.append(tk); lats.append(lt)
            rows.append({"q": qi, "gold": gold, "correct": c,
                         "wall_s": round(w, 3), "tokens": tk, "latent": lt})
            print("q%d gold=%-6s %s wall=%.2fs tok=%d lat=%d"
                  % (qi, gold, "OK " if c else "MISS", w, tk, lt), flush=True)
            if txt.startswith("ERROR"):
                print("   %s" % txt[:150], flush=True)
        results.append({"arm": "chain", "agents": na, "steps": stp, "correct": ok,
                        "n": len(QS), "wall_median": round(statistics.median(walls), 3),
                        "tokens_median": statistics.median(toks),
                        "latent_median": statistics.median(lats), "rows": rows})
        print("--> %s %d/%d  %.2fs  %s tok" % (label, ok, len(QS),
                                              statistics.median(walls),
                                              statistics.median(toks)), flush=True)

print("\n=== SUMMARY (%s set) ===" % args.set, flush=True)
base = results[0]
for r in results:
    wx = base["wall_median"] / r["wall_median"] if r["wall_median"] else 0
    tx = base["tokens_median"] / r["tokens_median"] if r["tokens_median"] else 0
    name = r["arm"] if r["arm"] == "single" else "chain a=%d lat=%d" % (r["agents"], r["steps"])
    print("%-18s %d/%d  wall %6.2fs (%.2fx)  tok %6s (%.2fx)"
          % (name, r["correct"], r["n"], r["wall_median"], wx,
             r["tokens_median"], tx), flush=True)
json.dump({"args": vars(args), "results": results}, open(args.json, "w"), indent=1)
print("saved -> " + args.json, flush=True)