#!/usr/bin/env python3
"""
bench_multiagent.py — comparative multi-agent benchmark against the mas-server.

Runs the SAME 3-role workflow (planner -> coder -> reviewer) on one task and prints
what each execution mode spends in PRE-ENCODED PROMPT TOKENS and wall time:

  naive      3 sequential single-agent runs; every turn re-encodes the whole
             transcript (the normal non-MAS orchestrator behaviour).
  chain      one /mas/run with 3 roles sharing one context (MAS engine).
  latent     one /mas/run with the plan/review agents latent (is_latent=1):
             the next agent continues from the previous agent's projected
             hidden states instead of re-reading the transcript text.

Token/time numbers come from the server's own llama_mas_timing
(n_prefill_tokens / n_generated_tokens / prefill_ms / generation_ms).

CLI:  python bench_multiagent.py [--url http://127.0.0.1:9032] [--spec-chars 6000] [--max-tokens 48]
"""
import argparse, json, sys, time, urllib.request

def post(url, payload, timeout=600):
    req = urllib.request.Request(url + "/mas/run", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body, (time.time() - t0) * 1000.0

ROLES = ["planner", "coder", "reviewer"]
ROLE_PROMPTS = {
    "planner":  "You are the Planner. Produce a concise step-by-step plan. Do NOT write code.",
    "coder":    "You are the Coder. Implement the plan. Output only code and 1-2 line file names.",
    "reviewer": "You are the Reviewer. Check the code against the plan. Output PASS or FAIL plus at most 2 issues.",
}

def make_spec(n_chars):
    base = ("This project is an offline-first task manager with a local encrypted store, "
            "a command palette, keyboard-first navigation, background sync, and a plugin API. ")
    return (base * (n_chars // len(base) + 1))[:n_chars]

def agent_params(role, max_tokens, latent=False, latent_steps=0):
    p = {"max_tokens": max_tokens, "is_latent": bool(latent), "latent_steps": latent_steps}
    return p

def run_naive(url, spec, task, max_tokens):
    """Three sequential single-agent runs; each turn re-encodes full transcript."""
    turns = []
    transcript = spec
    for role in ROLES:
        payload = {
            "task": task,
            "roles": [role],
            "system_prompts": {role: ROLE_PROMPTS[role] + "\n\nProject spec:\n" + transcript[:20000]},
            "strategy": "chain",
            "max_tokens": max_tokens,
            "clear_agents": True,
            "agent_params": {role: agent_params(role, max_tokens)},
        }
        body, wall = post(url, payload)
        if "error" in body:
            return {"error": body["error"]}, []
        timing = body.get("timing", {})
        outp = body.get("outputs", {}).get(role, "")
        turns.append({
            "role": role, "wall_ms": wall,
            "prefill_tokens": timing.get("n_prefill_tokens", 0),
            "gen_tokens": timing.get("n_generated_tokens", 0),
        })
        transcript += "\n\n" + role.upper() + ":\n" + outp
    return {"ok": True}, turns

def run_mas(url, spec, task, max_tokens, strategy, latent_map, share_carrier=False):
    """One /mas/run with all roles in one shared context.

    share_carrier=True models the DTAS/latent design: only the FIRST role
    carries the big shared handout; the others get plain role prompts and
    reuse it via KV-clone / hidden-state continuation.
    """
    if share_carrier:
        prompts = {}
        for i, r in enumerate(ROLES):
            prompts[r] = ROLE_PROMPTS[r] + ("\n\nProject spec:\n" + spec[:20000]) if i == 0 else ROLE_PROMPTS[r]
    else:
        prompts = {r: ROLE_PROMPTS[r] + "\n\nProject spec:\n" + spec[:20000] for r in ROLES}
    payload = {
        "task": task,
        "roles": ROLES,
        "system_prompts": prompts,
        "strategy": strategy,
        "max_tokens": max_tokens,
        "clear_agents": True,
        "agent_params": {r: agent_params(r, max_tokens, "latent" in latent_map.get(r, ""), latent_map.get(r) == "latent" and 12 or 0) for r in ROLES},
    }
    body, wall = post(url, payload)
    if "error" in body:
        return {"error": body["error"]}, []
    timing = body.get("timing", {})
    return {"ok": True, "timing": timing, "outputs": body.get("outputs", {}), "wall_ms": wall}, []

def fmt(ms):
    return f"{ms/1000.0:.2f}s" if ms >= 1000 else f"{ms:.0f}ms"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9032")
    ap.add_argument("--spec-chars", type=int, default=6000)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--task", default="Design a todo board schema, implement the core model, then review it.")
    args = ap.parse_args()

    spec = make_spec(args.spec_chars)
    print(f"== multi-agent benchmark | {args.url} | spec ~{len(spec)} chars "
          f"| max_tokens/agent={args.max_tokens} | roles={ROLES} ==")
    print("task:", args.task, "\n")

    rows = []

    # 1) naive
    res, turns = run_naive(args.url, spec, args.task, args.max_tokens)
    if "error" in res:
        print("naive FAILED:", res["error"])
    else:
        tot_ms = sum(t["wall_ms"] for t in turns)
        tot_pre = sum(t["prefill_tokens"] for t in turns)
        tot_gen = sum(t["gen_tokens"] for t in turns)
        print(f"[naive ] total {fmt(tot_ms)} | prefill tokens {tot_pre} | generated {tot_gen}")
        for t in turns:
            print(f"         {t['role']:9s} wall {fmt(t['wall_ms']):>8s} prefill {t['prefill_tokens']:>6} gen {t['gen_tokens']}")
        rows.append(("naive", tot_ms, tot_pre, tot_gen))

    # 2) MAS chain (shared context)
    res, _ = run_mas(args.url, spec, args.task, args.max_tokens, "chain", {})
    if "error" in res:
        print("chain FAILED:", res["error"])
    else:
        tm = res["timing"]
        print(f"[chain ] total {fmt(res['wall_ms'])} | prefill tokens {tm.get('n_prefill_tokens',0)} | generated {tm.get('n_generated_tokens',0)} "
              f"| prefill {fmt(tm.get('prefill_ms',0))} gen {fmt(tm.get('generation_ms',0))}")
        rows.append(("chain", res["wall_ms"], tm.get("n_prefill_tokens", 0), tm.get("n_generated_tokens", 0)))

    # 3) dtas (KV-clone from first agent; all agents decode from one shared prefill)
    res, _ = run_mas(args.url, spec, args.task, args.max_tokens, "dtas", {}, share_carrier=True)
    if "error" in res:
        print("dtas FAILED:", res["error"])
    else:
        tm = res["timing"]
        print(f"[dtas  ] total {fmt(res['wall_ms'])} | prefill tokens {tm.get('n_prefill_tokens',0)} | generated {tm.get('n_generated_tokens',0)} "
              f"| prefill {fmt(tm.get('prefill_ms',0))} gen {fmt(tm.get('generation_ms',0))}")
        rows.append(("dtas", res["wall_ms"], tm.get("n_prefill_tokens", 0), tm.get("n_generated_tokens", 0)))

    # 4) latent_mas (LatentMAS engine: agents continue from projected hidden states)
    res, _ = run_mas(args.url, spec, args.task, args.max_tokens, "latent_mas", {}, share_carrier=True)
    if "error" in res:
        print("latent FAILED:", res["error"])
    else:
        tm = res["timing"]
        print(f"[latent] total {fmt(res['wall_ms'])} | prefill tokens {tm.get('n_prefill_tokens',0)} | generated {tm.get('n_generated_tokens',0)} "
              f"| prefill {fmt(tm.get('prefill_ms',0))} gen {fmt(tm.get('generation_ms',0))}")
        rows.append(("latent", res["wall_ms"], tm.get("n_prefill_tokens", 0), tm.get("n_generated_tokens", 0)))

    if len(rows) >= 2:
        base_ms, base_pre, _ = rows[0][1], rows[0][2], 0
        print("\n== savings vs naive ==")
        for name, ms, pre, _gen in rows[1:]:
            print(f"{name:8s} time {fmt(ms)} ({100*(1-ms/base_ms):+.1f}%) | prefill tokens {pre} ({100*(1-pre/base_pre):+.1f}%)")

if __name__ == "__main__":
    main()