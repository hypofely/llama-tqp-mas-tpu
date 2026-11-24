"""A/B the latent FEEDBACK SOURCE against upstream LatentMAS.

Upstream (Gen-Verse/LatentMAS models.py generate_latent_batch) feeds the
LAST-LAYER hidden state through Wa each step:

    last_hidden = outputs.hidden_states[-1][:, -1, :]
    latent_vec  = realign(last_hidden)          # Wa, then norm rescale
    model(inputs_embeds=latent_vec, past_key_values=past)

hidden_states[0] (the layer-0 input embedding) is captured but only appended to
an analysis list - it is NEVER fed back. This fork's default does the opposite.
Since Wa maps OUTPUT space -> INPUT space, applying it to an already-input-space
vector is the wrong domain. This measures whether that matters.

Arms: source 0 (layer-0 tap, fork default) vs source 1 (final layer, upstream),
each at several depths, on GSM8K-style exact match AND raw prose.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=9026)
ap.add_argument("--depths", type=str, default="0,4,16")
ap.add_argument("--ntok", type=int, default=64)
ap.add_argument("--json", type=str, default="src_ab_results.json")
args = ap.parse_args()

B = "http://127.0.0.1:%d" % args.port
CACHE = r"C:\+run\GPT_WEB\llama-tqp-mas-cpp\tools\mas-server\align_8b_s32.bin"

# Same question set as latent_bench.py so numbers are comparable.
QUESTIONS = [
    ("Natalia sold clips to 48 friends in April, and then she sold half as many "
     "clips in May. How many clips did she sell altogether in April and May?", 72),
    ("Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes of "
     "babysitting. How much did she earn?", 10),
    ("Betty is saving money for a new wallet which costs $100. Betty has only "
     "half of the money she needs. Her parents gave her $15, and her "
     "grandparents twice as much as her parents. How much more does Betty need?", 5),
    ("James writes a 3-page letter to 2 different friends twice a week. "
     "How many pages does he write a year?", 624),
    ("Mark has a garden with flowers. He planted plants of three different "
     "colors. Ten of them are yellow, and there are 80% more of those in purple. "
     "There are only 25% as many green flowers as there are yellow and purple "
     "flowers. How many flowers does Mark have in his garden?", 35),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. "
     "How many bolts in total does it take?", 3),
]


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


def extract_number(text):
    import re
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


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
print("cache load HTTP %s loaded=%s" % (st, b.get("loaded")), flush=True)
req("/mas/alignment/toggle",
    {"use_alignment": True, "rescale_norm": True, "parallel_threads": 4}, t=60)

results = []
for source in (0, 1):
    label = "layer0_tap(fork)" if source == 0 else "final_layer(upstream)"
    print("\n=== source=%d  %s ===" % (source, label), flush=True)
    for depth in [int(x) for x in args.depths.split(",")]:
        correct, walls, texts = 0, [], []
        for q, gold in QUESTIONS:
            try:
                req("/agents/clear", {}, t=60)
            except Exception:
                pass
            req("/v1/agents", {"role": "src", "max_tokens": 4096,
                               "temperature": 0.0, "top_k": 1}, t=120)
            segs = [{"kind": "tokens", "text": q}]
            if depth:
                segs.append({"kind": "latent", "latent_steps": depth,
                             "align": 1, "latent_feedback_source": source})
            for _ in range(args.ntok):
                segs.append({"kind": "tokens", "text": "", "sample": 1})
            t0 = time.time()
            st, r = req("/turn", {"agent_idx": 0, "segments": segs}, t=900)
            walls.append(time.time() - t0)
            txt = r.get("text") or ""
            texts.append(txt)
            got = extract_number(txt)
            if got is not None:
                try:
                    if abs(float(got) - gold) < 1e-6:
                        correct += 1
                except ValueError:
                    pass
        w = statistics.median(walls)
        print("depth=%-3d exact=%d/%d  wall=%.3fs" % (depth, correct, len(QUESTIONS), w),
              flush=True)
        print("   %r" % texts[0][:150], flush=True)
        results.append({"source": source, "label": label, "depth": depth,
                        "correct": correct, "n": len(QUESTIONS),
                        "wall_s": round(w, 3), "texts": texts})

print("\n=== SUMMARY ===", flush=True)
for r in results:
    print("source=%d %-22s depth=%-3d %d/%d  %.3fs"
          % (r["source"], r["label"], r["depth"], r["correct"], r["n"], r["wall_s"]),
          flush=True)
json.dump(results, open(args.json, "w"), indent=1)
print("saved -> " + args.json, flush=True)