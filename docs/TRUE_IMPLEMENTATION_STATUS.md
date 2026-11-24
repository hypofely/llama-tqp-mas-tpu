# True Implementation Status — MAS C++ Fork

> **Date**: 2026-06-02
> **Method**: Source code audit of `src/`, `ggml/src/ggml-cuda/`, `python/`, and build artifacts.
> **Scope**: What is *actually* compiled, exported, and wired end-to-end vs what is only declared or stubbed.

---

## 1. Native MAS Engine (`src/llama-mas.cpp/h`)

### 1.1 Implemented & Compiled

| Function | Status | Notes |
|----------|--------|-------|
| `llama_mas_init` | ✅ | Creates single shared context, sets n_seq_max |
| `llama_mas_free` | ✅ | Frees samplers, context, agents |
| `llama_mas_add_agent` | ✅ | Full config + deep-copy adapter weights |
| `llama_mas_add_agent_flat` | ✅ | Simplified flat API used by Python |
| `llama_mas_remove_agent` | ✅ | |
| `llama_mas_set_agent_prompt` | ✅ | |
| `llama_mas_set_strategy` | ✅ | |
| `llama_mas_get_context` | ✅ | For Python to create layer graphs |
| `llama_mas_set_agent_layer_graph` | ✅ | |
| `llama_mas_kv_clone_range` | ✅ | `llama_memory_seq_cp` + position shift |
| `llama_mas_all_finished` | ✅ | |
| `llama_mas_decode_step` | ✅ | **Fixed 2026-06-02** — groups by layer graph, batched decode, samples per agent |
| `llama_mas_run` | ✅ | **Fixed 2026-06-02** — CHAIN, DTAS, BROADCAST/ROLE strategies all implemented |
| `llama_mas_get_text` | ✅ | **Fixed 2026-06-02** — simple detokenize |
| `llama_mas_free_text` | ✅ | **Fixed 2026-06-02** |
| `llama_mas_kv_clone` | ✅ | **Fixed 2026-06-02** — full seq_cp |
| `llama_mas_encode` | ✅ | Returns last-layer hidden state (CPU copy) |
| `llama_mas_encode_all_layers` | ✅ | Returns all layers (CPU copy) |
| `llama_mas_free_latent` | ✅ | |
| `llama_mas_free_layers` | ✅ | |
| `llama_context_set_layer_graph` | ✅ | Exported C API |
| Adapter weight init / free (CPU) | ✅ | Identity defaults |
| Adapter recursive_forward (CPU) | ✅ | LayerNorm→Linear→GELU→Linear→residual→LN |
| Adapter cross_forward (CPU) | ✅ | Cross projection with residual |
| Adapter mean_pool / concat | ✅ | |
| CUDA adapter weight upload/free | ✅ | `ggml_cuda_mas_device_malloc/free/memcpy` |

### 1.2 Declared in Header but NOT Implemented

| Function | Status | Impact |
|----------|--------|--------|
| `llama_mas_decode_step` GPU adapter call | ❌ | **Critical gap**: decode_step groups agents by layer graph but **never applies adapters**. It samples directly from logits. The per-agent `adapter_type` field is stored but unused during generation. |

### 1.3 Never Declared / Not Started

| Feature | Status | Notes |
|---------|--------|-------|
| `llama_get_hidden_state_gpu()` | ❌ | Phase 3 from architecture plan. No declaration in `llama.h`. Needed for zero-copy adapter on GPU. |
| In-graph adapter compute (Phase 2) | ❌ | No `ggml-adapter.h/.cpp` files exist. |

---

## 2. Layer Graph Engine (`src/llama-layer-graph.cpp/h`)

### 2.1 Implemented & Compiled

| Function | Status |
|----------|--------|
| `llama_layer_graph_init` | ✅ |
| `llama_layer_graph_free` | ✅ |
| `llama_layer_graph_get` | ✅ |
| `llama_layer_graph_n_layers` | ✅ |
| `llama_layer_graph_reset` | ✅ |
| `llama_layer_graph_set_enabled` | ✅ |
| `llama_layer_graph_reorder` | ✅ |
| `llama_layer_graph_duplicate` | ✅ |
| `llama_layer_graph_remove` | ✅ |
| `llama_layer_graph_load_json` | ✅ | Parses `{"layers":[...],"order":[...]}` |
| `llama_layer_graph_export_json` | ✅ |
| `llama_layer_graph_free_json` | ✅ |
| `llama_layer_graph_forward` | ✅ | Sets active graph, calls `llama_decode` |
| `llama_layer_graph_set_capture` | ✅ |
| `llama_layer_graph_get_snapshot_count` | ✅ |
| `llama_layer_graph_get_snapshot` | ✅ |
| `llama_layer_graph_clear_snapshots` | ✅ |
| `llama_layer_graph_capture_hidden_state` | ✅ | Called from model hook |
| `llama_layer_graph_get_timings` | ✅ | Per-node timing (distributed evenly) |
| `llama_layer_graph_free_timings` | ✅ |
| `llama_layer_graph_reset_timings` | ✅ |
| `llama_layer_graph_is_layer_enabled` | ✅ |
| Model hook in `models/llama.cpp` | ✅ | Checks `params.layer_graph_ctx`, skips disabled layers |

### 2.2 Python Wrapper

| Class | Status | Notes |
|-------|--------|-------|
| `LayerGraphContext` | ❌ | **Not present** in `llama_turboquant.py`. Only `llama_mas_set_agent_layer_graph` ctypes binding exists. No way for Python to create a layer graph context. |

---

## 3. Hidden States API

### 3.1 Implemented & Exported in `llama.h`

| Function | Status |
|----------|--------|
| `llama_set_output_hidden_states` | ✅ |
| `llama_get_hidden_states_layer_count` | ✅ |
| `llama_get_hidden_states_layer` | ✅ | **CPU copy** — copies GPU→CPU on every call |
| `llama_get_hidden_states_layer_n_outputs` | ✅ |
| `llama_get_hidden_states_layer_ith` | ✅ |
| `llama_get_hidden_states_last_token` | ✅ |

### 3.2 Missing

| Function | Status | Notes |
|----------|--------|-------|
| `llama_get_hidden_state_gpu()` | ❌ | Phase 3. Needed to pass GPU-resident tensor to CUDA adapter kernels without CPU copy. |

---

## 4. KV Cache Operations

### 4.1 Implemented & Exported

| Function | Status |
|----------|--------|
| `llama_kv_cache_clone` | ✅ |
| `llama_kv_cache_serialize` | ✅ |
| `llama_kv_cache_deserialize` | ✅ |
| `llama_kv_cache_layer_export` | ✅ |
| `llama_kv_cache_layer_import` | ✅ |

---

## 5. CUDA Adapter Kernels (`ggml/src/ggml-cuda/mas-adapter.cu`)

### 5.1 Implemented & Compiled into `ggml-cuda.dll`

| Kernel / Function | Status |
|-------------------|--------|
| `kernel_layernorm` | ✅ |
| `kernel_rms_norm` | ✅ |
| `kernel_gelu` | ✅ |
| `kernel_mat_vec_mul` | ✅ |
| `kernel_vec_add` | ✅ |
| `kernel_vec_copy` | ✅ |
| `ggml_cuda_mas_recursive_forward` | ✅ | Full pipeline |
| `ggml_cuda_mas_cross_forward` | ✅ | Full pipeline |
| `ggml_cuda_mas_device_malloc/free` | ✅ |
| `ggml_cuda_mas_memcpy_h2d/d2h` | ✅ |

### 5.2 Integration Status

| Integration Point | Status | Notes |
|-------------------|--------|-------|
| Wired into `llama_mas_decode_step` | ❌ | **Never called**. decode_step extracts logits and samples; adapter is completely bypassed. |
| Wired into `llama-mas.cpp` weight upload | ✅ | GPU weights uploaded on agent creation when `GGML_USE_CUDA` |

---

## 6. Python Wrappers (`python/llama_turboquant.py`)

### 6.1 Implemented

| Class / Function | Status |
|------------------|--------|
| `LlamaSharedModel` | ✅ |
| `LlamaContext` | ✅ |
| `LlamaMasContext` | ✅ |
| `LlamaMasContext.add_agent` | ✅ |
| `LlamaMasContext.run` | ✅ |
| `LlamaMasContext.set_agent_prompt` | ✅ | **Added 2026-06-02** |
| `LlamaMasContext.set_agent_layer_graph` | ❌ | **Missing** — no `LayerGraphContext` wrapper to create graphs |
| `LlamaContext.kv_clone_to` | ✅ |
| `LlamaContext.kv_serialize/deserialize` | ✅ |
| `LlamaContext.kv_layer_export/import` | ✅ |
| `LlamaContext.get_hidden_states_layer` | ✅ |

### 6.2 Missing

| Feature | Status |
|---------|--------|
| `LayerGraphContext` Python class | ❌ |
| `LlamaMasContext.encode` / `encode_all_layers` | ❌ | Only exists in C++ (`llama_mas_encode`) but not exposed to Python |

---

## 7. FastAPI Controller (`python/recursive_mas_controller_api.py`)

### 7.1 Implemented

| Endpoint | Status |
|----------|--------|
| `POST /models` | ✅ |
| `DELETE /models/{id}` | ✅ |
| `POST /agents` | ✅ |
| `DELETE /agents/{id}` | ✅ |
| `POST /agents/destroy-all` | ✅ |
| `POST /agents/clear` | ✅ |
| `POST /latent/encode` | ✅ |
| `POST /agents/generate` | ✅ |
| `POST /kv/clone` | ✅ |
| `POST /kv/export` | ✅ |
| `POST /kv/import` | ✅ |
| `POST /kv/layer/export` | ✅ |
| `POST /kv/layer/import` | ✅ |
| `POST /moe/router` | ✅ |
| `POST /dtas/dispatch` | ✅ |
| `POST /mas/run` (sync) | ✅ | Native MAS path fixed 2026-06-02 |
| `POST /mas/run-async` | ⚠️ | **Always uses `_run_mas_sync`** (Python MAS), never native |
| `GET /mas/status/{job_id}` | ✅ |
| `POST /mas/cancel/{job_id}` | ✅ |

### 7.2 Missing / Broken

| Issue | Status |
|-------|--------|
| `/mas/run` with `use_native_mas=true` did not pass `system_prompts` | **Fixed 2026-06-02** |
| `/mas/run` native did not use CHAIN strategy for multi-agent | **Fixed 2026-06-02** |
| `/mas/run-async` ignores `use_native_mas` | ❌ |
| No endpoint to set per-agent layer graphs | ❌ |
| No endpoint for `llama_mas_encode` (hidden state extraction) | ❌ |

---

## 8. Build Artifacts

| Directory | llama.dll | MAS objs | CUDA adapter objs | Status |
|-----------|-----------|----------|-------------------|--------|
| `build-codex/bin/` | ✅ 3.5MB | ✅ | ✅ | **Primary build** — includes all patches |
| `build-codex-vs2019/bin/Release/` | ✅ 3.5MB | ❌ | ❌ | **Outdated** — no MAS code |
| `build/bin/Release/` | ✅ 3.5MB | ❌ | ❌ | **Outdated** — no MAS code |

**Conclusion**: The server batch file (`start-mas-server-mas-cpp.bat`) points to `build-codex/bin/llama.dll` as the override, but if that fails to load, the older DLLs in the other directories may be picked up instead, causing silent absence of MAS features.

---

## 9. Summary: What Blocks the User's Issues

| User Complaint | Root Cause | Fix Status |
|----------------|-----------|------------|
| "Slow speed" | Native MAS `decode_step` was missing → batched decode not working. Adapters bypassed. No GPU hidden states. | `decode_step` restored. True speedup needs GPU hidden states (Phase 3) or adapter integration. |
| "Planner result not passed to executor" | `_run_mas_native` used `strategy="role"` (isolated agents) and never called `set_agent_prompt`. | **Fixed** — now forces CHAIN for multi-agent and passes system prompts. |
| "Agent instructions not passed" | Same as above — `system_prompts` ignored, no `set_agent_prompt` exposed in Python. | **Fixed** — `set_agent_prompt` added to Python wrapper and called from API. |

---

## 10. Recommended Implementation Order

### Phase A — Immediate (fixes correctness)
1. ✅ Restore corrupted `llama-mas.cpp` (`decode_step`, `run`, `get_text`, `free_text`, `kv_clone`)
2. ✅ Expose `set_agent_prompt` in Python
3. ✅ Fix `_run_mas_native` to pass prompts and use CHAIN
4. Rebuild `llama.dll` from `build-codex`

### Phase B — Python Layer Graph (1 day)
5. Add `LayerGraphContext` Python wrapper in `llama_turboquant.py`
6. Wire per-agent layer graphs through `_run_mas_native`
7. Add FastAPI endpoint for layer graph creation

### Phase C — Adapter Integration in Decode (2-3 days)
8. Extract hidden state from last layer in `llama_mas_decode_step`
9. Apply CPU adapter (if no GPU tensor available) before sampling
10. Wire CUDA adapter kernels into decode step when hidden states are GPU-backed

### Phase D — GPU Hidden States (1-2 weeks)
11. Add `llama_get_hidden_state_gpu()` to C API
12. Modify `llama.cpp` to save GPU tensor reference after compute
13. Eliminate CPU copy path entirely for MAS mode

---

*End of audit.*
