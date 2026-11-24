"""Isolate WHAT actually produces the accuracy gain.

Measured so far (GSM8K 6 questions, Qwen3-8B, ChatML, 512 tokens):
  text_single (no roles)          1/6   20.90 s
  chain roles + 10 latent/agent   6/6   23.45 s
  chain roles +  0 latent/agent   6/6   20.53 s   <-- latent adds cost, no accuracy

So the gain is NOT latent. This script separates the two remaining candidates:
  roles_single : ONE agent, all three role instructions concatenated, no KV
                 handoff, no latent. If this also scores 6/6, the entire gain is
                 prompt decomposition and neither latent nor KV transfer matters.
  roles_notok  : same but with a much smaller budget, to see how much of the
                 text_single failure was simply running out of tokens.
"""
import json
import re
import statistics
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9026
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 512
B = "http://127.0.0.1:%d" % PORT

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


def run(prompt, ntok):
    try:
        req("/agents/clear", {}, t=60)
    except Exception:
        pass
    req("/v1/agents", {"role": "iso", "max_tokens": 8192,
                       "temperature": 0.0, "top_k": 1}, t=120)
    segs = [{"kind": "tokens", "text": prompt}]
    for _ in range(ntok):
        segs.append({"kind": "tokens", "text": "", "sample": 1})
    t0 = time.time()
    st, r = req("/turn", {"agent_idx": 0, "segments": segs}, t=1800)
    return time.time() - t0, (r.get("text") or "")


out = []
for label, mk in [
    ("roles_single", lambda q: "<|im_start|>user\n" + q + "\n" +
     "\n".join("%d. %s" % (i + 1, r) for i, r in enumerate(ROLES)) +
     "<|im_end|>\n<|im_start|>assistant\n"),
    ("plain_long", lambda q: "<|im_start|>user\n" + q +
     "\nSolve step by step, then state 'The answer is <number>.'"
     "<|im_end|>\n<|im_start|>assistant\n"),
]:
    print("\n=== %s (ntok=%d) ===" % (label, NTOK), flush=True)
    ok, walls, rows = 0, [], []
    for qi, (q, gold) in enumerate(QUESTIONS):
        w, txt = run(mk(q), NTOK)
        c = correct(txt, gold)
        ok += int(c)
        walls.append(w)
        rows.append({"q": qi, "gold": gold, "correct": c, "wall_s": round(w, 3),
                     "text": txt})
        print("q%d gold=%-5s %s wall=%.2fs" % (qi, gold, "OK " if c else "MISS", w),
              flush=True)
        print("   %r" % txt[-160:], flush=True)
    med = statistics.median(walls)
    print("--> %s: %d/%d  median wall %.2fs" % (label, ok, len(QUESTIONS), med),
          flush=True)
    out.append({"arm": label, "ntok": NTOK, "correct": ok, "n": len(QUESTIONS),
                "wall_median": round(med, 3), "rows": rows})

json.dump(out, open("isolate_results_%d.json" % NTOK, "w"), indent=1)
print("\nsaved -> isolate_results_%d.json" % NTOK, flush=True)