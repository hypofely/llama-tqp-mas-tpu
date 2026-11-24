"""Compute + cache the alignment matrix for ANY model/port.

The solve is O(n_embd^3): 2048 -> ~272 s, 2560 -> ~10 min, 4096 -> ~40 min
(all at stride 32). Always cache the result; the cache key embeds stride AND
ridge, so both must be passed on load or it 404s.

Usage: python align_any.py PORT STRIDE CACHE_PATH
"""
import json
import os
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9028
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 32
CACHE = sys.argv[3] if len(sys.argv) > 3 else \
    r"C:\+run\GPT_WEB\llama-tqp-mas-cpp\tools\mas-server\align_4b instruct_s32.bin".replace(" ", "_")
B = "http://127.0.0.1:%d" % PORT
OUT = "align_any_%d.json" % PORT


def req(path, body=None, method=None, t=3000):
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


res = {"port": PORT, "stride": STRIDE, "cache": CACHE}
for i in range(90):
    try:
        if req("/health", t=10)[1].get("model_loaded"):
            break
    except Exception:
        pass
    time.sleep(10)
else:
    res["error"] = "server never healthy"
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res))
    sys.exit(2)

# Try the cache first - a hit saves the whole solve.
if os.path.exists(CACHE):
    try:
        st, b = req("/mas/alignment/load", {"path": CACHE, "vocab_stride": STRIDE}, t=300)
        if st == 200 and b.get("loaded"):
            res["cache_hit"] = True
            res["status"] = req("/mas/alignment/status", t=30)[1]
            json.dump(res, open(OUT, "w"), indent=1)
            print("CACHE HIT n_embd=%s" % res["status"].get("n_embd"))
            sys.exit(0)
    except Exception as e:
        print("cache load failed (%s) - computing" % str(e)[:60])

t0 = time.time()
try:
    st, b = req("/mas/alignment/compute", {"vocab_stride": STRIDE}, t=2900)
    res["compute_seconds"] = round(time.time() - t0, 1)
    res["http"] = st
    print("COMPUTE %.1f s HTTP %s" % (res["compute_seconds"], st), flush=True)
except Exception as e:
    res["compute_seconds"] = round(time.time() - t0, 1)
    res["error"] = str(e)[:200]
    print("COMPUTE FAILED after %.1f s: %s" % (res["compute_seconds"], str(e)[:120]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    sys.exit(1)

st, stat = req("/mas/alignment/status", t=30)
res["status"] = stat
print("n_embd=%s target_norm=%s" % (stat.get("n_embd"), stat.get("target_norm")), flush=True)
if stat.get("n_embd"):
    st, b = req("/mas/alignment/save", {"path": CACHE}, t=300)
    res["saved"] = b.get("saved")
    print("saved=%s -> %s" % (b.get("saved"), CACHE), flush=True)
    st, b = req("/mas/alignment/toggle",
                {"use_alignment": True, "rescale_norm": True, "parallel_threads": 4}, t=60)
    print("rescale=%s threads=%s" % (b.get("rescale_norm"), b.get("parallel_threads")), flush=True)
json.dump(res, open(OUT, "w"), indent=1)