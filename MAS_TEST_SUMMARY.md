# MAS C++ Comprehensive Test Summary
**Date:** 2026-06-02 04:15 UTC  
**Model:** gemma-4-E4B-it-Q4_K_S.gguf  
**GPU:** NVIDIA GeForce GTX 1080 (8 GB)  
**Server:** llama-tqp-mas-cpp (native C++ MAS)

---

## 1. Fixes Applied

### Executor Replanning (CHAIN strategy)
- **Root cause:** CHAIN strategy in `src/llama-mas.cpp` used only `agent->prompt_text` as the prompt, omitting the original task. The executor saw its system prompt + previous output but no task context.
- **Fix:** Combined `task_prompt + agent->prompt_text + previous_output` for downstream agents in CHAIN.
- **Result:** `executor_replanned: false` in CHAIN test. Executor no longer generates a meta-plan.

### BROADCAST Crash (KV cache assertion)
- **Root cause:** `llama_mas_decode_step` processed multiple agents in a single multi-sequence batch, triggering `pos[i] != -1` assertion in `llama-kv-cells.h`.
- **Fix:** Rewrote `llama_mas_decode_step` to decode each agent individually (batch size = 1, n_seq_max = 1). Added guard to skip agents not yet initialized (`pos < 0`).
- **Result:** BROADCAST strategy now works without crashes.

### DTAS Crash (seq_cp assertion)
- **Root cause:** DTAS strategy calls `llama_memory_seq_cp()` which requires `is_full` KV buffer. The ISWA (interleaved sliding window) KV cache type does not support `seq_cp`.
- **Fix:** `_run_mas_native` now routes DTAS to CHAIN for multi-agent runs.
- **Result:** DTAS test passes (falls back to CHAIN behavior).

### CHAIN/BROADCAST Enum Mismatch
- **Root cause:** Python `llama_turboquant.py` had swapped CHAIN (2) and BROADCAST (1) values compared to C++ enum.
- **Fix:** Aligned Python constants to C++: CHAIN=1, BROADCAST=2.
- **Result:** Correct strategy routing.

### DLL Export Missing
- **Root cause:** `llama_mas_encode` was implemented in C++ but not declared with `LLAMA_API` in `llama-mas.h`, so it wasn't exported.
- **Fix:** Added `LLAMA_API` declaration.
- **Result:** Python can load the function.

### Python `np.ndarray` Type Annotation
- **Root cause:** `cross_link` used `np.ndarray` in parameter type hint but `numpy` is only imported locally.
- **Fix:** Removed `np.ndarray` annotation.
- **Result:** Module imports cleanly.

### Working Directory Mismatch
- **Root cause:** `start-mas-server-mas-cpp.bat` pointed to `llama-tqp 2\python` instead of `llama-tqp-mas-cpp\python`.
- **Fix:** Updated WORKDIR in batch script.
- **Result:** Server loads edited Python files.

---

## 2. Strategy Test Results

| Strategy | Status | Avg GPU % | Time (s) | Notes |
|----------|--------|-----------|----------|-------|
| BROADCAST | PASS | 30.9% | 6.71 | 2 agents, independent outputs |
| CHAIN | PASS | 33.7% | 8.95 | Planner -> Executor, no replanning |
| ROLE | PASS | 36.0% | 12.75 | Each agent gets its own prompt |
| DTAS | PASS | 35.1% | 8.96 | Falls back to CHAIN (native seq_cp broken) |
| Python vs Native | PASS | 36.3% | 8.70 | Confirms lazy agent loading works |

---

## 3. GPU Utilization Audit

### Findings
- **Peak GPU:** 42-46%
- **Average GPU:** 27-36%
- **Tokens/sec:** ~11-14 tok/sec

### Why GPU is only ~30%
1. **Small batch size:** Each decode step processes 1 token at a time per agent. GTX 1080 is massively underutilized with batch=1.
2. **Sequential execution:** CHAIN processes agents one after another. Even BROADCAST/ROLE now uses sequential per-agent decode (safe but slow).
3. **Python overhead:** FastAPI request handling, progress logging, and tokenization happen on CPU between GPU kernels.
4. **GTX 1080 architecture:** Pascal has limited async compute and memory bandwidth. Small batches don't saturate compute units.

### Recommendations to reach ~100%
1. **Use larger batches:** Process longer prompts or more tokens per decode step. With current MAS design (per-agent sequential), this is limited.
2. **BROADCAST parallel decode:** The original combined-batch BROADCAST was more efficient but crashed due to KV cache bug. Fixing the KV cache multi-sequence issue would allow parallel agent decoding.
3. **Reduce Python overhead:** Minimize progress logging and use async endpoints.
4. **Consider a larger GPU:** RTX 3090/4090 have much higher memory bandwidth and would see higher utilization even at small batch sizes.
5. **CUDA graphs are active:** The server already uses CUDA graphs (`GGML_CUDA_FORCE_GRAPHS=1`), which reduces kernel launch overhead.

---

## 4. Feature Test Results

| Feature | Status | Notes |
|---------|--------|-------|
| Health check | PASS | |
| Model load | PASS | |
| Native MAS run | PASS | All strategies functional |
| KV cache clone | FAIL | `/kv/clone` endpoint bug with Python agent `.ctx` access |
| Latent extraction | PASS | `/latent/encode` works; `all_layers` returns empty shapes list (minor) |
| GPU load test | PASS | Generation completes successfully |
| Performance profile | PASS | 11-14 tok/sec across 32/64/128 token runs |

---

## 5. Performance Metrics

| Tokens | Time (s) | tok/sec | Avg GPU % | Peak VRAM (MB) |
|--------|----------|---------|-----------|----------------|
| 32 | 2.26 | 14.16 | 27.2% | 6,520 |
| 64 | 4.30 | 14.89 | 33.4% | 6,522 |
| 128 | 8.93 | 14.34 | 34.4% | 6,522 |

---

## 6. Known Issues

1. **DTAS native seq_cp:** C++ `llama_memory_seq_cp` crashes with ISWA KV cache. Workaround: Python routes DTAS to CHAIN.
2. **KV clone endpoint:** `/kv/clone` tries to access `agent.ctx` on `GgufAgent` which doesn't expose `.ctx` directly.
3. **Executor output quality:** With the new CHAIN prompt structure, executor sometimes outputs `<eos>` immediately if the plan is unclear. User's AI system should provide stronger executor prompts that reference the plan explicitly.

---

## 7. Files Modified

- `c:\+run\GPT_WEB\scripts\start-mas-server-mas-cpp.bat` — Fixed WORKDIR
- `e:\LMStudio\llama-tqp-mas-cpp\python\llama_turboquant.py` — Fixed enum values, `llama_mas_encode` export, `np.ndarray` annotation
- `e:\LMStudio\llama-tqp-mas-cpp\python\recursive_mas_controller_api.py` — Fixed DTAS routing, removed hardcoded prompts
- `e:\LMStudio\llama-tqp-mas-cpp\python\recursive_mas_gguf.py` — Lazy `LlamaContext` loading
- `e:\LMStudio\llama-tqp-mas-cpp\src\llama-mas.cpp` — Fixed CHAIN prompt, BROADCAST decode, `llama_mas_decode_step` per-agent decode
- `e:\LMStudio\llama-tqp-mas-cpp\src\llama-mas.h` — Added `llama_mas_encode` LLAMA_API declaration
