# MAS / DTAS / Latent / Layer Graph — Implementation Status

> Last updated: 2026-07-04 — Fixed native MAS garbage-output bug (zero-init context params). Native MAS now produces coherent output.
> Build: `build-codex\bin\llama.dll` + `ggml-cuda.dll` compiled successfully, server running on port 8000.

---

## 1. What Was Implemented

### 1.1 Per-Agent Layer Graphs (NATIVE MAS)

**Problem:** GPT_WEB Crew Builder lets users configure per-agent layer graphs (All / Range / Custom layers), but the native C++ MAS engine ignored them and logged a warning.

**Solution:**
- Added `llama_layer_graph_context * layer_graph` field to `llama_mas_agent` struct.
- Added C API `llama_mas_set_agent_layer_graph(mas, agent_idx, lg)` in `llama-mas.h`, `llama.h`, `llama-mas.cpp`.
- Added C API `llama_mas_get_context(mas)` so Python can create layer graphs against the internal context.
- Exported `llama_context_set_layer_graph(ctx, lg)` in `llama.h` (was internal-only).
- Refactored `llama_mas_decode_step()` to **group active agents by layer graph** and decode each group separately:
  - Agents with the same layer graph (or all `nullptr`) share a batched decode.
  - Agents with different graphs each get their own decode call with the correct graph active.
  - Preserves batching within groups; only splits when graphs differ.

**Files touched:**
- `src/llama-mas.h` — new declarations
- `src/llama-mas.cpp` — agent struct, set/get APIs, decode_step grouping
- `include/llama.h` — exported C APIs
- `src/llama-context.cpp` — fixed `llama_context_set_layer_graph` signature to match export
- `src/llama-layer-graph.cpp` — fixed `extern` declaration signature

**Python / FastAPI wiring:**
- `python/llama_turboquant.py` — `LlamaMasContext.set_agent_layer_graph(role, json)` now creates a real `LayerGraphContext` via `llama_mas_get_context()` and loads the JSON into it.
- `python/recursive_mas_controller_api.py` — `_run_mas_native()` now calls `mas.set_agent_layer_graph(role, json)` for every agent that has a `layerConfig` in the request. Removed the old "not supported" warning.

**End-to-end flow:**
GPT_WEB CrewBuilder (`layerConfig` UI) → `RecursiveMasClient.js` (`buildLayerGraphJson`) → POST `/mas/run` (`agent_layer_graphs` JSON) → FastAPI `_run_mas_native()` → `LlamaMasContext.set_agent_layer_graph()` → C++ `llama_mas_set_agent_layer_graph()` → `llama_mas_decode_step()` groups by graph → each group decoded with correct layer subset.

---

### 1.2 KV Cache Row-Level Sharing (clone_range)

**Problem:** Native MAS had `llama_mas_kv_clone()` (full clone) but no way to copy only a sub-range of KV cells (e.g. "only take KV cache from row X to the new agent").

**Solution:**
- Added C API `llama_mas_kv_clone_range(mas, src_agent, dst_agent, src_pos_start, src_pos_end, dst_pos_start)`.
- In the single-context design, this calls `llama_memory_seq_cp(mem, src_seq, dst_seq, p0, p1)` which just adds the destination seq_id to the existing cells (zero-copy).
- Documented that `dst_pos_start` must equal `src_pos_start` for in-context copies; cross-context positional shifting is reserved for future work.

**Files touched:**
- `src/llama-mas.h`, `include/llama.h`, `src/llama-mas.cpp`
- `python/llama_turboquant.py` — `LlamaMasContext.kv_clone_range()` wrapper

---

### 1.3 CUDA Adapter Kernels (infrastructure)

**Problem:** Adapter MLP (RecursiveLink / CrossLink) runs on CPU, causing 5-10 ms per forward. User wants everything possible pushed to CUDA.

**Solution:**
- Created `ggml/src/ggml-cuda/mas-adapter.cuh` and `.cu` with raw CUDA kernels:
  - `kernel_layernorm`, `kernel_rms_norm`, `kernel_gelu`, `kernel_mat_vec_mul`, `kernel_vec_add`, `kernel_vec_copy`
  - `ggml_cuda_mas_recursive_forward()` — full RecursiveLink forward on GPU
  - `ggml_cuda_mas_cross_forward()` — full CrossLink forward on GPU
- These compile automatically into `ggml-cuda` because `CMakeLists.txt` uses `file(GLOB *.cu)`.
- **Now wired into `llama-mas.cpp`** via `sample_agent_group()`. When a hidden-state tensor lives on a CUDA backend, adapter weights are uploaded to GPU once and `ggml_cuda_mas_recursive_forward()` / `cross_forward()` are called on device pointers.
- Device memory wrappers (`ggml_cuda_mas_device_malloc/free/memcpy_h2d/memcpy_d2h`) keep all `cudaMalloc`/`cudaFree`/`cudaMemcpy` inside `ggml-cuda.dll`; `llama.dll` only imports adapter forward symbols.

**Files touched:**
- `ggml/src/ggml-cuda/mas-adapter.cuh`
- `ggml/src/ggml-cuda/mas-adapter.cu`

---

### 1.4 Build Verification

- `src/llama-mas.cpp.obj` — compiled cleanly (single object test)
- Full `llama.dll` link — **succeeded** after fixing `llama_context_set_layer_graph` signature mismatch between `llama.h` export and `llama-context.cpp` definition.
- New DLL copied to:
  - `E:\LMStudio\llama-tqp-mas-cpp\build-codex\bin\llama.dll`
  - `E:\LMStudio\llama-tqp 2\build-codex-vs2019\bin\Release\llama.dll`
  - `E:\LMStudio\llama-tqp 2\build\bin\Release\llama.dll`

---

### 1.5 Native MAS Garbage Output Bug — FIXED

**Problem:** Native MAS (`LlamaMasContext`) produced completely garbled tokens (e.g. 'href Observatory Scholar City dengan') while Python MAS (`GgufAgent`) worked fine with the same model.

**Root Cause:** `LlamaMasContext.__init__` in `python/llama_turboquant.py` created a **zero-initialized** `llama_context_params` struct:
```python
params = llama_context_params()  # WRONG — zeroes RoPE, yarn, flash-attn fields
```
Python's `ctypes.Structure()` default-initializes all fields to `0`. This zeroed out critical parameters such as:
- `rope_freq_base` (must be `500000.0` for Llama 3.1)
- `rope_freq_scale` (must be `1.0`)
- `yarn_ext_factor` (must be `1.0`)
- And many other defaults set by `llama_context_default_params()`.

`LlamaContext` (used by Python MAS) correctly called `llama_context_default_params()` first, which is why Python MAS always worked.

**Fix:** Changed line 470 in `python/llama_turboquant.py`:
```python
params = _lib.llama_context_default_params()  # CORRECT — inherits all defaults
```

**Verification:** After fix, native MAS produces coherent English output identical in quality to Python MAS (differences are only due to temperature sampling vs greedy argmax).

```
Native MAS:  '.' — Code flows like a stream / Algorithms dance with ease / Logic's gentle
Python MAS:  'A code is written / Syntax errors abound here / Logic is my guide'
```

**Files touched:**
- `python/llama_turboquant.py` — `LlamaMasContext.__init__` (1 line)

## 2. What Is Still TODO / Future Work

### 2.1 GPU-Resident Hidden States (Phase 3)
- Currently `llama_context::get_hidden_states_layer()` copies from GPU to CPU.
- Need a new API `llama_get_hidden_state_gpu()` that returns a `ggml_tensor*` (or device pointer) so adapter kernels can consume it directly without copy overhead.
- Once that exists, wire `ggml_cuda_mas_recursive_forward()` / `cross_forward()` into `llama_mas_decode_step()`.

### 2.2 KV Cache Row-Level Copy With Position Shift
- `llama_mas_kv_clone_range()` currently keeps the same positions (`dst_pos_start` ignored for same-stream copies).
- For true "copy tail to position 0" semantics, we need to either:
  - Add `llama_kv_cache_shift_range()` to adjust cell positions after clone, or
  - Implement cross-context physical buffer copy (more expensive).

### 2.3 VRAM Optimizations
- ✅ **KV cache quantization for native MAS** — `MasRunRequest` now accepts `type_k` and `type_v` (e.g. `"turbo4"`, `"f16"`, `"q8_0"`); `_run_mas_native()` passes them to `LlamaMasContext()`, which forwards them to `llama_context_params` and `llama_mas_init()`.
- KV cache memory pooling / shared buffers for agents with identical prefixes.
- KV cache defragmentation hooks in native MAS.

### 2.4 In-Graph Adapter Compute (Phase 2)
- Running adapter MLPs inside the GGML compute graph would eliminate the separate adapter call entirely.
- Requires adding custom GGML ops or extending the graph builder in `models/llama.cpp`.
- Much larger scope; CUDA kernels (1.3) are the pragmatic first step.

### 2.5 Comprehensive Test Suite
- ✅ Short token-count verification (32-64 tokens) passed with native MAS.
- ✅ Benchmark: native MAS vs Python MAS — **native ~1.15x faster** (2.1s vs 2.5s for 32 tokens, single agent, Meta-Llama-3.1-8B-Q4_K_S on GTX 1080).
- ⚠️ Correctness test: ensure layer graph skip actually skips layers and produces different (but valid) outputs.

---

## 3. How to Restart the Server to Pick Up the New DLL

After rebuilding `llama.dll`, stop the running server and restart:

```powershell
# 1. Stop the old server
$server = Get-Process -Name python | Where-Object { $_.CommandLine -like "*uvicorn*" }
if ($server) { Stop-Process -Id $server.Id -Force }

# 2. Start fresh
$env:PYTHONPATH="E:\LMStudio\llama-tqp-mas-cpp\python;$env:PYTHONPATH"
python -m uvicorn recursive_mas_controller_api:app --host 0.0.0.0 --port 8000 --log-level info
```

The new `llama.dll`, `ggml.dll`, and `ggml-cuda.dll` must be copied to all load paths (`build-codex-vs2019/bin/Release`, `build/bin/Release`, etc.) before restarting.

---

## 4. Quick Reference: New C APIs

```c
// Per-agent layer graph
int llama_mas_set_agent_layer_graph(struct llama_mas_context * mas,
                                      int32_t agent_idx,
                                      struct llama_layer_graph_context * lg);

// Get underlying llama_context from MAS (for creating layer graphs)
struct llama_context * llama_mas_get_context(struct llama_mas_context * mas);

// KV cache row-level clone
int llama_mas_kv_clone_range(struct llama_mas_context * mas,
                             int32_t src_agent_idx, int32_t dst_agent_idx,
                             int32_t src_pos_start, int32_t src_pos_end,
                             int32_t dst_pos_start);

// Set layer graph on any context (exported for Python/ctypes)
void llama_context_set_layer_graph(struct llama_context * ctx,
                                   struct llama_layer_graph_context * lg);
```

---

## 5. Quick Reference: New Python APIs

```python
from llama_turboquant import LlamaMasContext

mas = LlamaMasContext(shared_model, n_ctx=8192, n_agents_max=4)
mas.add_agent(role="planner", ...)

# Set per-agent layer graph from JSON
mas.set_agent_layer_graph("planner", '{"layers":[{"id":0,"enabled":true}],"order":[0]}')

# Clone KV range from one agent to another
mas.kv_clone_range("planner", "executor", src_pos_start=10, src_pos_end=50)
```

---

*End of status document.*
