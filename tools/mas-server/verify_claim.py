"""Adversarial verification of the chain-vs-single claim.

The claimed win (6/6 vs 1/6) has three plausible confounds. This tests all three
instead of trusting the headline:

 C1 PROMPT PARITY. The chain's last role says "State the final answer ... with no
    further commentary". The single baseline only got "Solve step by step, then
    state ...". Every single-arm run hit the token CEILING (768/768), i.e. it
    never stopped to answer. So the chain may simply be better PROMPTED.
    Fix: give the single agent ALL FIVE roles verbatim, same ceiling, same stop.

 C2 SCORING. The extractor falls back to "last number in the text" when there is
    no "answer is N". A cut-off single-arm run ends on an INTERMEDIATE value, so
    it scores MISS even if the right number appears earlier. Report three scores:
      strict  = explicit "answer is N" matches gold
      lenient = gold appears anywhere as a standalone number
      lastnum = the original (possibly unfair) rule
    If strict/lenient diverge sharply, the headline was a scoring artifact.

 C3 SAMPLE SIZE. 6 questions cannot separate 1/6 from 6/6 reliably. Use 12.

Also stores FULL text for every run so any score can be audited by hand.
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
ap.add_argument("--agents", type=int, default=5)
ap.add_argument("--steps", type=int, default=0)
ap.add_argument("--cache", type=str,
                default=r"C:\+run\GPT_WEB\llama-tqp-mas-cpp\tools\mas-server\align_8b_s32.bin",
                help="alignment cache for THIS model (key embeds stride+ridge)")
ap.add_argument("--stride", type=int, default=32)
ap.add_argument("--json", type=str, default="verify_results.json")
args = ap.parse_args()

B = "http://127.0.0.1:%d" % args.port
CACHE = args.cache

# 12 questions, gold computed in Python (not by hand) so the answer key is auditable.
def _q(text, gold):
    return (text, float(gold))

QUESTIONS = [
    _q("A train travels 90 km in 1.5 hours, then 120 km in 2 hours. What is its "
       "average speed for the whole trip in km/h?", (90 + 120) / (1.5 + 2)),
    _q("A shop marks up a $40 item by 25%, then offers a 20% discount on the marked "
       "price. What is the final price in dollars?", 40 * 1.25 * 0.8),
    _q("Three pipes fill a tank. Pipe A alone takes 6 hours, B alone 12 hours, "
       "C alone 4 hours. How many hours do all three take together?", 1 / (1/6 + 1/12 + 1/4)),
    _q("A rectangle's length is 3 times its width. Its perimeter is 64 cm. "
       "What is its area in square centimetres?", (64/8) * (3 * 64/8)),
    _q("In a class of 30 students, 18 play football, 15 play basketball, and 5 play "
       "neither. How many play both sports?", 18 + 15 - (30 - 5)),
    _q("A car's value drops 20% each year. It costs $25000 new. What is its value "
       "after 2 years in dollars?", 25000 * 0.8 ** 2),
    _q("Natalia sold clips to 48 friends in April, and half as many in May. "
       "How many clips did she sell altogether?", 48 + 24),
    _q("Weng earns $12 an hour for babysitting. She worked 50 minutes. "
       "How much did she earn in dollars?", 12 * 50 / 60),
    _q("A robe takes 2 bolts of blue fiber and half that much white fiber. "
       "How many bolts in total?", 2 + 1),
    _q("James writes a 3-page letter to 2 friends, twice a week. "
       "How many pages does he write in a year (52 weeks)?", 3 * 2 * 2 * 52),
    _q("A tank holds 240 litres. It is 3/8 full. How many more litres are needed "
       "to fill it?", 240 * (1 - 3/8)),
    _q("A worker earns $18/hour for the first 8 hours and 1.5x that rate beyond. "
       "She works 11 hours. What is her total pay in dollars?", 18*8 + 18*1.5*3),
]

ROLES = [
    "Plan: list the given quantities, the unknown, and the operations needed. Do not compute yet.",
    "Execute the plan step by step, computing each intermediate value explicitly.",
    "Check the intermediate values for arithmetic or unit errors and correct them.",
    "Re-derive the result by a different route to cross-check it.",
    "State the final answer as 'The answer is <number>.' with no further commentary.",
]
SUFFIX = "\nSolve step by step, then state 'The answer is <number>.'"


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


def chatml(q, extra=""):
    return ("<|im_start|>user\n" + q + extra + SUFFIX +
            "<|im_end|>\n<|im_start|>assistant\n")


def nums(text):
    return [float(x.replace(",", "")) for x in
            re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)]


ANSWER_LANG = re.compile(
    r"(answer|result|total|final|therefore|thus|so[ ,]|equals|=|is|are|should be|comes to|gives)",
    re.I)


def concluded(txt, gold):
    """Gold asserted as a RESULT, not restated as a given.

    Guards learned from auditing raw text: look only in the LAST 40% (the
    opening restates the question's own numbers - gold=40 false-matched 11
    restatements of a given "$40 item"), and require answer-assertion language
    within ~70 chars before the number so a bare intermediate does not count.
    """
    if not txt:
        return False
    # An explicit "the answer is <gold>" is a conclusion ANYWHERE. The first
    # version scanned only the last 40%, so a model that answered and then kept
    # talking scored BELOW its own strict score (strict 12/12 vs concluded 9/12 -
    # logically impossible). That bug fabricated a "-3 correct conclusions"
    # verdict against latent on the 8B. Position filtering belongs ONLY on the
    # weaker fallback below, where it suppresses restated question numbers.
    for v in re.findall(r"answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", txt, re.I):
        try:
            if abs(float(v.replace(",", "")) - gold) < 1e-6:
                return True
        except ValueError:
            pass
    tail = txt[int(len(txt) * 0.60):]
    g = "%g" % gold
    for m in re.finditer(r"(?<![\d.])" + re.escape(g) + r"(?![\d.])", tail):
        if ANSWER_LANG.search(tail[max(0, m.start() - 70):m.start()]):
            return True
    return False


def score(text, gold):
    """Three scorings, so a scoring artifact is visible rather than assumed."""
    m = re.findall(r"answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", text, re.I)
    strict = False
    if m:
        try:
            strict = abs(float(m[-1].replace(",", "")) - gold) < 1e-6
        except ValueError:
            pass
    ns = nums(text)
    lenient = any(abs(n - gold) < 1e-6 for n in ns)
    lastnum = bool(ns) and abs(ns[-1] - gold) < 1e-6
    return strict, lenient, lastnum


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
st, b = req("/mas/alignment/load", {"path": CACHE, "vocab_stride": args.stride}, t=300)
print("align loaded=%s | questions=%d | ceiling=%d" % (b.get("loaded"), len(QUESTIONS), args.ntok),
      flush=True)
req("/mas/alignment/toggle",
    {"use_alignment": True, "rescale_norm": True, "parallel_threads": 4}, t=60)


def single(q, roles_all):
    try:
        req("/agents/clear", {}, t=60)
    except Exception:
        pass
    req("/v1/agents", {"role": "v", "max_tokens": 8192,
                       "temperature": 0.0, "top_k": 1}, t=120)
    extra = ""
    if roles_all:
        extra = "\n" + "\n".join("%d. %s" % (i + 1, r) for i, r in enumerate(ROLES))
    segs = [{"kind": "tokens", "text": chatml(q, extra)}]
    for _ in range(args.ntok):
        segs.append({"kind": "tokens", "text": "", "sample": 1, "stop_on_eog": 1})
    t0 = time.time()
    st, r = req("/turn", {"agent_idx": 0, "segments": segs}, t=1800)
    return time.time() - t0, (r.get("text") or ""), r.get("n_tokens", 0)


def chain(q):
    t0 = time.time()
    st, r = req("/mas/latent-chain", {
        "prompt": chatml(q), "agents": args.agents, "latent_steps": args.steps,
        "ntok": args.ntok, "feedback_source": 1, "align": True,
        "role_prompts": [ROLES[i % len(ROLES)] for i in range(args.agents)],
        "stop_on_eog": True,
    }, t=1800)
    if st != 200:
        return time.time() - t0, "ERROR " + json.dumps(r)[:200], 0
    return time.time() - t0, (r.get("text") or ""), r.get("total_tokens", 0)


def chain_n(q, na, stp):
    t0 = time.time()
    st, r = req("/mas/latent-chain", {
        "prompt": chatml(q), "agents": na, "latent_steps": stp,
        "ntok": args.ntok, "feedback_source": 1, "align": True,
        "role_prompts": [ROLES[i % len(ROLES)] for i in range(na)],
        "stop_on_eog": True,
    }, t=1800)
    if st != 200:
        return time.time() - t0, "ERROR " + json.dumps(r)[:200], 0
    return time.time() - t0, (r.get("text") or ""), r.get("total_tokens", 0)


ARMS = [
    ("chain3_lat0",  lambda q: chain_n(q, 3, 0)),
    ("chain3_lat10", lambda q: chain_n(q, 3, 10)),
    ("chain5_lat0",  lambda q: chain_n(q, 5, 0)),
]

results = []
for name, fn in ARMS:
    print("\n=== %s ===" % name, flush=True)
    S = L = N = C = CEIL = ERR = 0
    walls, toks, rows = [], [], []
    for qi, (q, gold) in enumerate(QUESTIONS):
        try:
            w, txt, tk = fn(q)
        except Exception as e:
            # One failed question must not destroy the whole sweep. Record it as
            # a miss with the error so partial results survive and the failure is
            # visible rather than silently dropping the arm.
            w, txt, tk = 0.0, "ERROR " + str(e)[:160], 0
            ERR += 1
        s_, l_, n_ = score(txt, gold)
        c_ = concluded(txt, gold)
        S += int(s_); L += int(l_); N += int(n_); C += int(c_)
        if tk >= args.ntok:
            CEIL += 1
        walls.append(w); toks.append(tk)
        rows.append({"q": qi, "gold": gold, "strict": s_, "lenient": l_,
                     "lastnum": n_, "concluded": c_, "ceiling": tk >= args.ntok,
                     "wall_s": round(w, 3), "tokens": tk, "text": txt})
        print("q%-2d gold=%-8g strict=%d concl=%d wall=%5.2fs tok=%d%s"
              % (qi, gold, s_, c_, w, tk, "  CEILING" if tk >= args.ntok else ""),
              flush=True)
    r = {"arm": name, "strict": S, "lenient": L, "lastnum": N, "concluded": C,
         "ceiling_hits": CEIL, "errors": ERR, "n": len(QUESTIONS),
         "wall_median": round(statistics.median(walls), 3),
         "tokens_median": statistics.median(toks), "rows": rows}
    assert C >= S, "metric bug: concluded (%d) < strict (%d) is impossible" % (C, S)
    results.append(r)
    print("--> %s  strict %d/%d  CONCLUDED %d/%d  ceiling %d/%d  errors %d  wall %.2fs  tok %s"
          % (name, S, len(QUESTIONS), C, len(QUESTIONS), CEIL, len(QUESTIONS),
             ERR, r["wall_median"], r["tokens_median"]), flush=True)

print("\n=== VERDICT TABLE ===", flush=True)
print("%-18s %8s %9s %9s %9s %8s"
      % ("arm", "strict", "CONCLUDED", "ceiling", "wall_s", "tokens"), flush=True)
for r in results:
    print("%-18s %5d/%-2d %6d/%-2d %6d/%-2d %9.2f %8s"
          % (r["arm"], r["strict"], r["n"], r["concluded"], r["n"],
             r["ceiling_hits"], r["n"], r["wall_median"], r["tokens_median"]), flush=True)
json.dump({"args": vars(args), "results": results}, open(args.json, "w"), indent=1)
print("saved -> " + args.json, flush=True)