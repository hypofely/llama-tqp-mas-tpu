# LatentMAS / MAS Server — Test Scenarios

> **Last updated:** 2026-06-25  
> **Purpose:** Real-world test suite for debugging and validating current + future MAS implementation  
> **Run against:** `localhost:8000` (mas-server) + GPT_WEB chat panel

---

## 1. Quick Smoke Tests (30 seconds each)

### Test 1: Server Health
```powershell
curl http://localhost:8000/health
# Expected: "ok" or JSON with status
```

### Test 2: MAS C++ Library Loads
```powershell
# Build the library (Release)
cmake --build build-codex --target mas-server --config Release -j 8

# Run a minimal bench
build-codex/bin/latent-bench.exe -m "C:\gemma-4-E4B-it-Q4_K_S.gguf" --roles planner,executor,outer --max-tokens 32
# Expected: All 3 agents produce output, no rc=-1, no "inconsistent sequence positions"
```

### Test 3: Frontend Can Reach Server
```javascript
// In browser console on localhost:5002
fetch("http://localhost:8000/health").then(r => r.text()).then(console.log)
// Expected: "ok"
```

### Test 4: MAS Provider + Default Crew (No Latent)
1. Open GPT_WEB chat panel
2. Set LLM Provider = "MAS"
3. Set Crew = "Default Crew" (planner, executor, outer)
4. **Uncheck** "Latent MAS"
5. Send message: "What is 2+2?"
6. **Expected:** All 3 agents produce text, no console errors, response visible

### Test 5: MAS Provider + Default Crew (With Latent) — THE CRITICAL TEST
1. Same as Test 4 but **Check** "Latent MAS"
2. Send message: "What is 2+2?"
3. **Expected:** All 3 agents produce text, no console errors
4. **Console MUST be clean of:**
   - `rc=-1`
   - `decode/init failure`
   - `inconsistent sequence positions`
   - `Failed to fetch`
   - `192.168.1.64:8000` (stale IP)

---

## 2. Progressive Debugging Scenarios

### Scenario A: KV Type Mismatch

**Setup:**
```powershell
# Start server with planar3 (type 47)
mas-server.exe -m model.gguf -p 8000 --type-k planar3 --type-v planar3
```

**Test:** Send request with `type_k=44, type_v=44` (turbo4 mismatch)

**Expected behavior:**
- Server should detect mismatch
- Recreate context with new types
- New context should have `type_k=44, type_v=44`
- Generation succeeds

**Bug to watch for:**
- After failed recreation: `type_k=1` (corrupted state)
- This means `g_state.type_k` was committed before checking success

**Breakpoint locations:**
- `mas-server.cpp:722` — `ensure_mas_context` entry
- `mas-server.cpp:935-960` — KV type switch logic
- `mas-server.cpp:1861` — `recreate_mas_context` call

### Scenario B: Latent Transfer KV Wipe

**Setup:**
- Server running with correct KV types
- Crew: planner (256 tok) → executor (256 tok) → outer (256 tok)
- Latent MAS enabled

**Test:** Send a multi-step reasoning prompt

**Expected behavior:**
```
[MAS/stream] strategy=5 (LATENT_MAS)
[MAS/latent_mas] KV cloned planner -> executor
[MAS/latent_mas] KV cloned executor -> outer
[MAS] outer agent produces final answer
```

**Bug to watch for (before fix):**
```
init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
  - the last position stored in the memory module of the context (i.e. the
    last decoded token) is 20
  - but the batch contains a token at position 22
```

**Root cause:** `llama_memory_seq_rm()` wiping KV cache after clone

**Breakpoint locations:**
- `llama-mas.cpp:1886` — first `llama_decode` in wave 0
- `llama-mas.cpp:1910` — latent transfer skip logic (THE FIX SITE)
- `llama-mas.cpp:2016` — KV clone call
- `llama-mas.cpp:2250` — hidden state extraction failure point

### Scenario C: GQA / SWA Model Edge Case

**Setup:**
- Model: Gemma 4 E2B (GQA: 8 heads, 1 KV head; SWA on first 20 layers)
- Context: `n_ctx=30720`, `n_agents_max=3`

**Test:** Long-context prompt (>4096 tokens) with Latent MAS

**Expected behavior:**
- Context preflights from 4096 → 30720
- SWA layers (layers 0-19) handle long context
- Non-SWA layers (20-34) handle full attention
- All agents complete without shape assertion failures

**Bug to watch for:**
- `ggml_reshape_3d` assertion failure during `/mas/run`
- `n_embd` mismatch between model and batch
- `GGML_ASSERT` in `ggml-cuda/mas-adapter.cu`

**Breakpoint locations:**
- `llama-context.cpp:3390` — context creation with layer graph
- `llama-kv-cache.cpp:1005` — `set_cached_shift_res` (if exists)
- `ggml/src/ggml.c:3699` — `ggml_reshape_3d` assertion

### Scenario D: Stress Test (Many Agents)

**Setup:**
```json
{
  "n_agents_max": 4,
  "roles": ["planner", "executor", "critic", "outer"],
  "strategy": 5
}
```

**Test:** Agent A finishes, clones KV to B; B finishes, clones to C; etc.

**Expected behavior:**
- Each transfer preserves KV state
- No `seq_id` conflicts
- Memory stays within `n_agents_max × n_ctx` budget

**Failure mode:**
- `llama_decode` returns -1 at step 2+ (position gap or memory exhaustion)
- Server returns 500 with `rc=-1` in body

### Scenario E: Async Run + Cancellation

**Setup:**
```json
{
  "async": true,
  "max_tokens": 2048,
  "cancel_after_tokens": 100
}
```

**Test:** Start async run, cancel mid-generation

**Expected behavior:**
- Server returns task_id immediately
- Streaming stops on cancel
- Partial output preserved (not discarded)
- No memory leak from orphaned context

---

## 3. Frontend Tests (Playwright / Browser)

### F-1: MAS Provider Selection
1. Navigate to `localhost:5002`
2. Open chat panel → click LLM Provider dropdown
3. Select "MAS"
4. **Verify:** No dropdown crash, provider name shows as "MAS"

### F-2: Crew Selection
1. With MAS selected, open Crew dropdown
2. Select "Default Crew"
3. **Verify:** Crew name shows, latent checkbox appears

### F-3: Latent Toggle
1. Check "Latent MAS" checkbox
2. Send message
3. **Verify:**
   - Console shows `strategy=5` (LATENT_MAS) in MAS logs
   - No `rc=-1` errors
   - All 3 agents produce output

### F-4: 3-Agent Output Verification
1. After F-3 completes, click the info/details button on assistant message
2. **Verify:** Three sections visible — planner, executor, outer
3. **Verify:** Outer section is NOT just "Please provide the answer"
4. **Verify:** Planner text → executor text → outer answer (not empty)

### F-5: Switch Between Latent / Non-Latent
1. Toggle Latent MAS off
2. Send same message
3. Compare latency (non-latent should be slower due to re-encoding)
4. Toggle on, send again
5. **Verify:** Latent mode is faster and produces equivalent output

### F-6: Stale Endpoint Recovery
1. In DevTools Network tab, find MAS request
2. Note the URL (should be `localhost:8000/agent/encode` or similar)
3. **Verify:** URL is `localhost:8000`, NOT `192.168.1.64:8000`

### F-7: Error Message Quality
1. Force an error (send max_tokens=0 or invalid role)
2. **Verify:** Error message includes `hint=` field with actionable description
3. **Verify:** No raw exception stack traces leaked to user

---

## 4. Automated Regression Tests

### R-1: KV Type Round-Trip
```powershell
# Start with type_k=47, type_v=47 (PLANAR3)
$r1 = Invoke-RestMethod -Uri "http://localhost:8000/mas/run" -Method POST -Body $json1
# Check response contains correct KV types
$r1.agents | ForEach-Object { $_.kv_type_k; $_.kv_type_v }
# Expected: 47, 47
```

### R-2: Latent Transfer Integrity
```python
# After planner generates 80 tokens:
# 1. Check executor's KV cache contains planner's tokens
# 2. Check executor's position starts at 80 (not 0)
# 3. Verify executor's first 5 tokens ≠ planner's first 5 tokens
#    (they continue, not repeat)
```

### R-3: Layer Graph Isolation
```python
# Run with planner on layers [0-10], executor on [10-20]
# Verify:
# 1. Different layer graphs produce different outputs (not identical)
# 2. Both outputs are coherent English (not garbled)
# 3. No crashes or assertions
```

### R-4: Context Recreation Mid-Run
```powershell
# 1. Send run with type_k=47
# 2. Immediately send run with type_k=44
# 3. Second run should recreate context, not crash
# 4. Server health should remain "ok"
```

### R-5: VRAM Budget
```powershell
# Measure VRAM before/after MAS run:
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
# Expected: Difference ≤ n_agents × model_size / n_agents_max
# For Gemma 4 E2B + 3 agents: ~1.4 GB VRAM total context
```

### R-6: Adapter Kernel Correctness
```powershell
# Compare CPU vs GPU adapter output:
# 1. Run with GGML_CUDA_FORCE_GRAPHS=0 (CPU adapters)
# 2. Run with GGML_CUDA_FORCE_GRAPHS=1 (GPU adapters)
# 3. Outputs should match within floating-point tolerance
```

---

## 5. Debugging Playbook (When Things Break)

### "rc=-1 with no clear error"
1. Check `mas-server.cpp:1040` — `describe_mas_failure()` — does it have a `hint=`?
2. If hint says `decode/init failure` → check `llama-mas.cpp` line 1886/1970/2000
3. Add breakpoint at `llama_mas_decode_step()` (line ~2000)
4. Step into — is `llama_decode` returning -1? Why?
5. Check `batch.n_tokens` and `batch.sequences[0].pos` — any position gaps?
6. If position gap → Bug 2 (KV wipe) is back

### "type_k=1 after KV switch"
1. Check `mas-server.cpp:937` — is `g_state.type_k` being set before/before success check?
2. If before → Bug 1 fix not applied or not compiled
3. Clean rebuild: `cmake --build build-codex --target mas-server --config Debug -j 8 --clean-first`

### "Memory leaks / OOM"
1. Check `n_seq_max` vs `n_agents_max`
2. Verify `llama_kv_cache` dealloc on context destroy
3. Check `mas-adapter.cu` for `device_malloc` without `device_free`

### "Adapter output garbage"
1. Check `model->type` vs expected adapter type (RECURSIVE vs CROSS)
2. Verify `adapter_weights` are uploaded to GPU before forward
3. Check `d2h` copy after forward — is the host buffer populated?

### "SWA layers don't have KV cache"
1. `llama_kv_cache_iswa` creates filtered caches — they might not have certain layers
2. KV clone must handle missing layers (skip, don't crash)
3. Check `mas-server.cpp` layer graph handling for SWA models

---

## 6. Performance Baseline Targets

| Metric | Target | Current (GTX 1080 + Gemma 4 E2B) |
|--------|--------|----------------------------------|
| First token latency (latent) | < 500ms | TBD |
| First token latency (non-latent) | < 800ms | ~1000ms |
| Tokens/second (decode) | > 10 tok/s | ~20 tok/s (single agent) |
| MAS overhead vs single agent | < 20% | ~15% |
| VRAM (3 agents, 32K ctx) | < 3 GB | ~1.4 GB (16K ctx) |
| Memory leak (100 runs) | 0 MB | 0 MB |

---

## 7. Test Checklist Before Marking Changes as DONE

- [ ] Server starts on correct port with correct model
- [ ] `/health` returns 200
- [ ] KV types match between batch file and frontend defaults
- [ ] LATENT_MAS with 3 agents produces output from all 3
- [ ] Outer agent does NOT say "Please provide the answer"
- [ ] Zero `rc=-1` errors in console for clean run
- [ ] Zero `inconsistent sequence positions` errors
- [ ] Zero `192.168.x.x` stale IP errors
- [ ] Adapter kernels run on GPU (if enabled)
- [ ] Layer graph per-agent JSON works end-to-end
- [ ] Context recreation handles KV type change without corruption
- [ ] 100-run stress test: no memory leaks, no crashes
- [ ] Frontend Playwright test: F-1 through F-7 all pass
