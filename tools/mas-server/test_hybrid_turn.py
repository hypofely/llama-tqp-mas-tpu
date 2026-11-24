#!/usr/bin/env python3
"""Hybrid turn: latent reasoning + text/tool-call output in one KV sequence.

Covers the three properties that matter:
  1. a LATENT segment writes positions but emits NO tokens
  2. a following TOKENS segment continues from the carried hidden state
  3. a second turn on the same session_id reuses the cached prefix
"""
import json
import sys
import urllib.request

import os as _os
BASE = _os.environ.get("MAS_BASE", "http://127.0.0.1:9000")
VERBOSE = True


def post(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else b"{}"
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}


def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"raw": e.read().decode()}


FAILS = []
PASSES = []
RUN_ID = "0" 


def check(name, cond, detail=None):
    if cond:
        PASSES.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILS.append(name)
        print(f"  [FAIL] {name}  {json.dumps(detail)[:400] if detail else ''}")


def main():
    global RUN_ID
    import time as _time
    RUN_ID = _time.strftime("%H%M%S")
    print(f"== hybrid turn (run {RUN_ID}) ==")
    rc, h = get("/health")
    check("health", rc == 200 and h.get("model_loaded"))

    # The turn driver works on an existing agent; a fresh server has none, and
    # the MAS context is created lazily on the first agent request.
    post("/reset", {})
    rc, ca = post("/v1/agents", {"role": "hybrid_writer", "max_tokens": 96})
    check("agent_created", rc == 200 and ca.get("status") == "ok", ca)

    # --- 1. latent-only turn: 12 latent steps, no sampling, no text ----------
    rc, j = post("/turn", {
        "session_id": f"hyb_a_{RUN_ID}",
        "agent_idx": 0,
        "segments": [
            {"kind": "tokens", "text": "Once upon a time there was a girl named Lily who loved to play outside with her dog."},
            {"kind": "latent", "latent_steps": 12},
        ],
    })
    check("latent_turn_ok", rc == 200 and j.get("status") == "ok", j)
    check("latent_emitted_no_tokens",
          j.get("n_latent_steps") == 12 and j.get("n_tokens") == 0,
          {"latent": j.get("n_latent_steps"), "tokens": j.get("n_tokens")})
    check("latent_advanced_positions",
          j.get("n_positions", 0) >= 12,
          {"n_positions": j.get("n_positions")})
    if VERBOSE:
        print(f"       text='{j.get('text','')}' think={j.get('think_steps_per_sec')} t/s")

    # --- 2. session state after turn 1 -------------------------------------
    rc, s = get(f"/sessions/hyb_a_{RUN_ID}")
    check("session_saved", rc == 200 and s.get("kv_bytes", 0) > 0,
          {"kv_bytes": s.get("kv_bytes"), "segs": len(s.get("segments", []))})
    kinds = [seg["kind"] for seg in s.get("segments", [])]
    check("session_ledger_has_latent", 1 in kinds, {"kinds": kinds})
    if VERBOSE:
        print(f"       ledger={[(x['kind'], x['pos_start'], x['pos_end']) for x in s.get('segments', [])]}")

    # --- 3. turn 2 on the same session: text continues after latent --------
    rc, j2 = post("/turn", {
        "session_id": f"hyb_a_{RUN_ID}",
        "agent_idx": 0,
        "segments": [
            {"kind": "tokens", "text": "Once upon a time there was a girl named Lily who loved to play outside with her dog. She was happy."},
            {"kind": "tokens", "text": " And then the sun went down.", "sample": 1},
        ],
    })
    check("turn2_ok", rc == 200 and j2.get("status") == "ok", j2)
    check("turn2_restored_kv", j2.get("kv_restored") is True,
          {"restored": j2.get("kv_restored"), "note": j2.get("reuse_note")})
    if VERBOSE:
        print(f"       reuse_note={j2.get('reuse_note')}")
        print(f"       text2='{j2.get('text','')}'")

    # --- 4. latent-only resume: no tokens, positions still grow ------------
    rc, s2 = get(f"/sessions/hyb_a_{RUN_ID}")
    check("session_grew", rc == 200 and s2.get("pos", 0) >= s.get("pos", 0),
          {"pos": s2.get("pos"), "prev": s.get("pos")})

    # --- 4b. latent continuation AFTER a restore ---------------------------
    # The KV alone is not enough: the next latent step consumes the last
    # residual vector, which llama_state_seq_* does NOT serialize. If that tail
    # hidden state is lost on save/restore, the driver silently falls back to
    # decoding a seed token, which writes that token to a second position.
    # Detect it by growth: a pure latent resume adds EXACTLY latent_steps
    # positions, with no extra seed decode.
    rc, before = get(f"/sessions/hyb_a_{RUN_ID}")
    pos_before = before.get("pos", 0)
    LATENT_STEPS = 6
    rc, j4 = post("/turn", {
        "session_id": f"hyb_a_{RUN_ID}",
        "agent_idx": 0,
        "segments": [{"kind": "latent", "latent_steps": LATENT_STEPS}],
    })
    check("latent_resume_ok", rc == 200 and j4.get("status") == "ok", j4)
    # n_positions is the ABSOLUTE KV tail after the turn, not the delta.
    check("latent_resume_exact_growth",
          j4.get("n_latent_steps") == LATENT_STEPS and
          j4.get("n_positions") == pos_before + LATENT_STEPS,
          {"latent": j4.get("n_latent_steps"), "positions": j4.get("n_positions"),
           "pos_before": pos_before, "expected": pos_before + LATENT_STEPS})
    check("latent_resume_no_tokens", j4.get("n_tokens") == 0, {"n_tokens": j4.get("n_tokens")})

    # --- 5. a brand-new session id must NOT inherit state -----------------
    rc, j3 = post("/turn", {
        "session_id": f"hyb_fresh_{RUN_ID}",
        "agent_idx": 0,
        "segments": [{"kind": "tokens", "text": "The quick brown fox."}],
    })
    check("fresh_session_no_reuse", rc == 200 and j3.get("kv_restored") is False,
          {"restored": j3.get("kv_restored")})

    print()
    print("=" * 60)
    print(f"{len(PASSES)} passed, {len(FAILS)} failed")
    if FAILS:
        for f in FAILS:
            print(f"  FAIL {f}")
    print("=" * 60)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())