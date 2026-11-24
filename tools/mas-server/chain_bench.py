"""Sequential latent MAS vs text baselines — the paper's actual comparison.

Gen-Verse/LatentMAS claims its accuracy gain from MULTI-AGENT latent
collaboration (layer-wise KV handed between agents), not single-agent latent
depth. This measures that shape directly:

  text_single : one agent, prompt + N tokens of chain-of-thought
  text_mas    : K agents, each RE-DECODES the prompt + prior answers, then speaks
  latent_chain: K agents, each thinks in latent space and INHERITS the previous
                agent's full layer-wise KV (no prompt re-decode); only the last
                agent decodes text

Token budget defaults to 512, NOT 64: at 64 the model is still mid-reasoning and
never emits an answer, so a 64-token accuracy number measures the budget rather
than the method. Upstream runs --max_new_tokens 2048.

Reports exact-match accuracy, wall time, and tokens actually decoded per arm.
Always prints raw text — a proxy coherence score was proven gameable.
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
ap.add_argument("--ntok", type=int, default=512)
ap.add_argument("--agents", type=int, default=3)
ap.add_argument("--steps", type=int, default=10, help="latent steps per agent")
ap.add_argument("--feedback", type=int, default=1, help="0=layer0 tap, 1=upstream final layer")
ap.add_argument("--limit", type=int, default=6)
ap.add_argument("--arms", type=str, default="text_single,latent_chain")
ap.add_argument("--stop-eog", dest="stop_eog", type=int, default=1,
                help="1=stop at EOG (ntok is a ceiling), 0=burn the full budget")
ap.add_argument("--json", type=str, default="chain_bench.json")
args = ap.parse_args()

B = "http://127.0.0.1:%d" % args.port
CACHE = r"C:\+run\GPT_WEB\llama-tqp-mas-cpp\tools\mas-server\align_8b_s32.bin"

QUESTIONS = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as "
     "many clips in May. How many clips did she sell altogether in April and May?", 72),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes "
     "of babysitting. How much did she earn?", 10),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half "
     "of the money she needs. Her parents decided to give her $15 for that purpose, "
     "and her grandparents twice as much as her parents. How much more money does "
     "Betty need to buy the wallet?", 5),
    ("James writes a 3-page letter to 2 different friends twice a week. How many "
     "pages does he write a year?", 624),
    ("Mark has a garden with flowers. He planted plants of three different colors "
     "in it. Ten of them are yellow, and there are 80% more of those in purple. "
     "There are only 25% as many green flowers as there are yellow and purple "
     "flowers. How many flowers does Mark have in his garden?", 35),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many "
     "bolts in total does it take?", 3),
]

ROLES = [
    "Plan the calculation: list the quantities and the operations needed. Do not compute yet.",
    "Carry out the plan step by step and compute the intermediate values.",
    "Verify the arithmetic and state the final answer as 'The answer is <number>.'",
]


def chatml(question):
    """Qwen3 needs manual ChatML on /turn.

    /turn sends RAW tokens - it does not apply a chat template (only
    /v1/chat/completions does). Without ChatML the model treats the question as
    a text continuation and rambles off-topic (measured: it emitted React import
    statements for a fibre-arithmetic question). That, plus a 64-token budget,
    is why earlier accuracy runs scored 1/6 - both were harness bugs, not model
    or latent failures.
    """
    return ("<|im_start|>user\n" + question +
            "\nSolve step by step, then state 'The answer is <number>.'")


PROMPT_FN = lambda q: chatml(q) + "<|im_end|>\n<|im_start|>assistant\n"


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


def last_number(text):
    """Prefer an explicit 'answer is N', else the last number in the text."""
    m = re.findall(r"answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", text, re.I)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def is_correct(text, gold):
    got = last_number(text)
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
print("alignment cache: HTTP %s loaded=%s" % (st, b.get("loaded")), flush=True)
req("/mas/alignment/toggle",
    {"use_alignment": True, "rescale_norm": True, "parallel_threads": 4}, t=60)
st, a = req("/mas/alignment/status", t=30)
print("align n_embd=%s rescale=%s threads=%s target_norm=%.5f"
      % (a.get("n_embd"), a.get("rescale_norm"), a.get("parallel_threads"),
         a.get("target_norm") or 0), flush=True)

QS = QUESTIONS[:args.limit]
results = []


def arm_text_single(q, gold):
    try:
        req("/agents/clear", {}, t=60)
    except Exception:
        pass
    req("/v1/agents", {"role": "single", "max_tokens": 8192,
                       "temperature": 0.0, "top_k": 1}, t=120)
    segs = [{"kind": "tokens", "text": PROMPT_FN(q)}]
    for _ in range(args.ntok):
        segs.append({"kind": "tokens", "text": "", "sample": 1,
                     "stop_on_eog": 1 if args.stop_eog else 0})
    t0 = time.time()
    st, r = req("/turn", {"agent_idx": 0, "segments": segs}, t=1800)
    return {"wall_s": time.time() - t0, "text": r.get("text") or "",
            "tokens": r.get("n_tokens", 0), "latent": r.get("n_latent_steps", 0),
            "positions": r.get("n_positions", 0)}


def arm_latent_chain(q, gold):
    t0 = time.time()
    st, r = req("/mas/latent-chain", {
        "prompt": PROMPT_FN(q),
        "agents": args.agents, "latent_steps": args.steps, "ntok": args.ntok,
        "feedback_source": args.feedback, "align": True,
        "role_prompts": ROLES[:args.agents],
        "stop_on_eog": bool(args.stop_eog),
    }, t=1800)
    if st != 200:
        return {"error": "HTTP %s %s" % (st, json.dumps(r)[:200]),
                "wall_s": time.time() - t0, "text": "", "tokens": 0,
                "latent": 0, "positions": 0}
    pa = r.get("per_agent", [])
    return {"wall_s": time.time() - t0, "text": r.get("text") or "",
            "tokens": r.get("total_tokens", 0),
            "latent": r.get("total_latent_steps", 0),
            "positions": (pa[-1].get("n_positions", 0) if pa else 0),
            "per_agent": [{"a": p.get("agent"), "pos": p.get("n_positions"),
                           "tok": p.get("n_tokens"), "lat": p.get("n_latent_steps"),
                           "cloned": p.get("kv_cloned"), "ms": round(p.get("wall_ms", 0))}
                          for p in pa]}


ARMS = {"text_single": arm_text_single, "latent_chain": arm_latent_chain}

for name in args.arms.split(","):
    fn = ARMS[name.strip()]
    print("\n=== %s (ntok=%d) ===" % (name, args.ntok), flush=True)
    correct, walls, toks, lats, rows = 0, [], [], [], []
    for qi, (q, gold) in enumerate(QS):
        r = fn(q, gold)
        ok = is_correct(r["text"], gold)
        correct += int(ok)
        walls.append(r["wall_s"])
        toks.append(r["tokens"])
        lats.append(r["latent"])
        rows.append({"q": qi, "gold": gold, "correct": ok, **r})
        print("q%d gold=%-5s %s wall=%.2fs tok=%d lat=%d pos=%d"
              % (qi, gold, "OK " if ok else "MISS", r["wall_s"],
                 r["tokens"], r["latent"], r["positions"]), flush=True)
        if r.get("per_agent"):
            print("   chain: %s" % json.dumps(r["per_agent"]), flush=True)
        if r.get("error"):
            print("   ERROR %s" % r["error"], flush=True)
        print("   %r" % r["text"][-220:], flush=True)
    summary = {"arm": name, "correct": correct, "n": len(QS),
               "wall_median": round(statistics.median(walls), 3),
               "tokens_median": statistics.median(toks),
               "latent_median": statistics.median(lats), "rows": rows}
    results.append(summary)
    print("--> %s: %d/%d  median wall %.2fs  median tokens %s"
          % (name, correct, len(QS), summary["wall_median"],
             summary["tokens_median"]), flush=True)

print("\n=== SUMMARY ===", flush=True)
base = results[0]
for r in results:
    wx = base["wall_median"] / r["wall_median"] if r["wall_median"] else 0
    tx = (base["tokens_median"] / r["tokens_median"]) if r["tokens_median"] else 0
    print("%-14s %d/%d  wall %.2fs (%.2fx vs %s)  tokens %s (%.2fx)  latent %s"
          % (r["arm"], r["correct"], r["n"], r["wall_median"], wx, base["arm"],
             r["tokens_median"], tx, r["latent_median"]), flush=True)

json.dump({"args": vars(args), "results": results}, open(args.json, "w"), indent=1)
print("saved -> " + args.json, flush=True)