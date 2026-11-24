"""Save the 8B alignment matrix, enable rescale, then measure latent vs text.

The 4096-dim solve costs ~40 min, so the matrix MUST be cached to disk.
Cache key embeds stride AND ridge, so both must match on load.
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9026
B = "http://127.0.0.1:%d" % PORT
CACHE = r"C:\+run\GPT_WEB\llama-tqp-mas-cpp\tools\mas-server\align_8b_s32.bin"


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


st, s = req("/mas/alignment/status", t=30)
print("status: n_embd=%s use=%s target_norm=%s ridge=%s stride=%s"
      % (s.get("n_embd"), s.get("use_alignment"), s.get("target_norm"),
         s.get("ridge"), s.get("vocab_stride")), flush=True)
if not s.get("n_embd"):
    print("NO MATRIX - compute first")
    sys.exit(2)

# 1. Save to disk (repo path, NOT scratch - scratch is pruned after 24h idle).
st, b = req("/mas/alignment/save", {"path": CACHE}, t=300)
print("save HTTP %s %s" % (st, json.dumps(b)[:160]), flush=True)

# 2. Turn on the norm re-scale (the LatentMAS ingredient).
st, b = req("/mas/alignment/toggle",
            {"use_alignment": True, "rescale_norm": True}, t=60)
print("rescale HTTP %s %s" % (st, json.dumps(b)[:160]), flush=True)

# 3. Verify the cache round-trips (this is the 40-minute -> 0.03 s win).
t0 = time.time()
st, b = req("/mas/alignment/load", {"path": CACHE, "vocab_stride": 32}, t=300)
print("load HTTP %s in %.2f s %s" % (st, time.time() - t0, json.dumps(b)[:140]),
      flush=True)