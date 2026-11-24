# Custom Patches & Implementations in llama-tqp-mas-cpp

> This document catalogs every custom patch, extension, and new subsystem added to the llama.cpp fork at `llama-tqp-mas-cpp`.
> Last updated: 2026-07-04 — native MAS garbage-output bug fixed

---

## Table of Contents

1. [Dynamic Layer Graph Engine](#1-dynamic-layer-graph-engine)
2. [Multi-Agent System (MAS) Native Engine](#2-multi-agent-system-mas-native-engine)
3. [Hidden States API](#3-hidden-states-api)
4. [KV Cache Clone](#4-kv-cache-clone)
5. [CUDA Graph Enablement](#5-cuda-graph-enablement)
6. [TurboQuant Custom Quantization Types](#6-turboquant-custom-quantization-types)
7. [Pending / Needs Additional Work](#7-pending--needs-additional-work)

---

## 1. Dynamic Layer Graph Engine

**Purpose:** Allow selective loading, reordering, duplication, and skipping of individual transformer layers at runtime. Each agent in a crew can use a different subset of layers.

**Files:**
- `src/llama-layer-graph.h` — C API declarations
- `src/llama-layer-graph.cpp` — Implementation
- `src/llama-context.h` — Added `set_layer_graph(void*)` setter
- `src/llama-context.cpp` — `llama_context_set_layer_graph()` free function
- `src/models/llama.cpp` — Forward-pass hook checks `layer_graph_ctx`

**Key C API:**
```c
struct llama_layer_graph_context * llama_layer_graph_init(struct llama_context * ctx);
void llama_layer_graph_free(struct llama_layer_graph_context * lg);
int llama_layer_graph_set_enabled(struct llama_layer_graph_context * lg, int32_t instance_id, bool enabled);
int llama_layer_graph_reorder(struct llama_layer_graph_context * lg, int32_t from_index, int32_t to_index);
int llama_layer_graph_load_json(struct llama_layer_graph_context * lg, const char * json);
char * llama_layer_graph_export_json(const struct llama_layer_graph_context * lg);
int llama_layer_graph_forward(struct llama_layer_graph_context * lg, const llama_token * tokens, int32_t n_tokens);
```

**JSON Format:**
```json
{"layers":[{"id":0,"enabled":true},{"id":1,"enabled":false}],"order":[0,2,3]}
```

**Implementation State:**
- ✅ Core graph struct, mutation, and JSON import/export are fully implemented
- ✅ Forward pass hook in `models/llama.cpp` skips disabled layers via `llama_layer_graph_is_layer_enabled()`
- ✅ Python bindings (`llama_turboquant.py`) expose `LayerGraphContext` wrapper
- ✅ Per-agent layer graphs in **native MAS mode** are now wired via `llama_mas_set_agent_layer_graph()`.
- ✅ `llama_context_set_layer_graph()` is exported in `llama.h` and available to Python via ctypes.
- ⚠️ Hidden-state snapshot memory is bounded (max 64 snapshots as of latest patch) but still high for large models.

---

## 2. Multi-Agent System (MAS) Native Engine

**Purpose:** Run multiple agents in a single batched `llama_decode` call for speed, with per-agent adapters, prompts, and strategies.

**Files:**
- `src/llama-mas.h` — MAS agent config, context, strategy enums
- `src/llama-mas.cpp` — Full native MAS implementation
- `python/llama_turboquant.py` — Python `LlamaMasContext` wrapper
- `python/recursive_mas_gguf.py` — Python `GgufAgent` / `MasSystem`
- `python/recursive_mas_controller_api.py` — FastAPI controller

**Key C API:**
```c
struct llama_mas_context * llama_mas_init(struct llama_model * model, struct llama_context_params params, int32_t n_agents_max);
void llama_mas_free(struct llama_mas_context * mas);
int32_t llama_mas_add_agent(struct llama_mas_context * mas, const struct llama_mas_agent_config * config);
void llama_mas_set_agent_prompt(struct llama_mas_context * mas, int32_t agent_idx, const char * prompt);
int llama_mas_run(struct llama_mas_context * mas, const char * task_prompt, int32_t max_tokens, enum llama_mas_strategy strategy);
```

**Strategies:**
- `LLAMA_MAS_STRATEGY_CHAIN` — Sequential, each agent sees previous output
- `LLAMA_MAS_STRATEGY_BROADCAST` — All agents decode in parallel
- `LLAMA_MAS_STRATEGY_ROLE` — Single agent runs (role-based dispatch)
- `LLAMA_MAS_STRATEGY_DTAS` — KV clone from first agent, then batched decode

**Adapter Types:**
- `LLAMA_MAS_ADAPTER_NONE` — Pass-through
- `LLAMA_MAS_ADAPTER_RECURSIVE` — LayerNorm→Linear→GELU→Linear→residual→LayerNorm
- `LLAMA_MAS_ADAPTER_CROSS` — Cross-agent projection with residual
- `LLAMA_MAS_ADAPTER_MEAN` — Mean pool across tokens
- `LLAMA_MAS_ADAPTER_CAT` — Concatenate tokens

**Implementation State:**
- ✅ Init, add_agent, set_prompt, run, get_text are functional
- ✅ Chain strategy copies previous agent outputs into next agent's prompt
- ✅ Broadcast and DTAS strategies use single batched decode for all agents
- ✅ Adapter forward kernels implemented in C (CPU)
- ⚠️ Chain prompt template was vague; recently hardened to explicitly say "ACT ON the above output. Do NOT create a new plan."
- ✅ Per-agent layer graphs are wired via `llama_mas_set_agent_layer_graph()`
- ✅ CUDA adapter kernels (`ggml_cuda_mas_recursive_forward`, `ggml_cuda_mas_cross_forward`) are wired into `llama-mas.cpp` decode step; adapter compute runs on GPU when hidden states are CUDA-backed
- ⚠️ `llama_mas_kv_clone` is a no-op because native MAS uses a single context with multiple seq_ids

---

## 3. Hidden States API

**Purpose:** Extract hidden-state vectors from arbitrary layers during decode, enabling adapter MLPs and latent-state exchange between agents.

**Files:**
- `include/llama.h` — Public C API declarations
- `src/llama-context.h` — Internal storage for hidden-state arrays
- `src/llama.cpp` — `llama_set_output_hidden_states()` and getters

**Key C API:**
```c
void llama_set_output_hidden_states(struct llama_context * ctx, bool output_hidden_states);
int32_t llama_get_hidden_states_layer_count(struct llama_context * ctx);
float * llama_get_hidden_states_layer(struct llama_context * ctx, int32_t il);
int32_t llama_get_hidden_states_layer_n_outputs(struct llama_context * ctx, int32_t il);
float * llama_get_hidden_states_layer_ith(struct llama_context * ctx, int32_t il, int32_t i);
float * llama_get_hidden_states_last_token(struct llama_context * ctx);
```

**Implementation State:**
- ✅ Fully implemented and exported in `llama.h`
- ✅ Python bindings available via `llama_turboquant`
- ✅ `LlamaContext.decode()` automatically enables hidden states when `output_hidden_states=True`
- ✅ Used by `GgufAgent.encode()` and `LlamaMasContext.encode()` for adapter MLPs

---

## 4. KV Cache Clone

**Purpose:** Copy KV cache from one context to another so later agents can reuse prefix tokens without re-computing them.

**Files:**
- `include/llama.h` — `llama_kv_cache_clone` declaration
- `src/llama-kv-cache.h` / `src/llama-kv-cache.cpp` — Implementation

**Key C API:**
```c
int32_t llama_kv_cache_clone(struct llama_context * src, struct llama_context * dst);
```

**Implementation State:**
- ✅ Implemented and exported
- ✅ Used by Python `LlamaContext.kv_clone_to()` and DTAS dispatch
- ⚠️ Native MAS uses single context with seq_ids, so `llama_mas_kv_clone` is currently a no-op

---

## 5. CUDA Graph Enablement

**Purpose:** Capture and replay GPU command sequences to reduce CPU overhead on repeated decode calls.

**Files:**
- `ggml/src/ggml-cuda/` — CUDA graph capture/replay in `ggml_cuda_graph` helpers
- Build script sets `GGML_CUDA_FORCE_GRAPHS=1` env var

**Key Behavior:**
- Compute Capability 6.x (Pascal, e.g. GTX 1080) is architecturally supported but was being rejected by an outdated guard.
- The guard was removed/updated so Pascal GPUs can use CUDA graphs.
- Environment variable `GGML_CUDA_FORCE_GRAPHS=1` overrides any remaining disable logic.

**Implementation State:**
- ✅ Enabled for Pascal (CC 6.1) and newer
- ✅ `start-mas-server-mas-cpp.bat` sets `GGML_CUDA_FORCE_GRAPHS=1`
- ✅ Verified working: server starts without `ggml_cuda_graph_set_enabled: disabling CUDA graphs due to GPU architecture` warning
- ⚠️ Graphs are only beneficial for repeated identical-shape decode calls; first-run overhead exists

---

## 6. TurboQuant Custom Quantization Types

**Purpose:** Custom GGML types for compressed KV cache storage.

**Files:**
- `ggml/include/ggml.h` — Type enum additions
- `ggml/src/ggml-cuda/` — CUDA kernels for Turbo4 dequant

**Key Types:**
- `GGML_TYPE_TURBO4_0` — 4-bit per-weight quantization for KV cache

**Implementation State:**
- ✅ Declared in `ggml.h` and used throughout Python API as default `type_k`/`type_v`
- ✅ CUDA kernels exist for decode
- ⚠️ Not all backends (CPU, Vulkan, Metal) may have full Turbo4 support

---

- **Native MAS Garbage Output** — FIXED. LlamaMasContext.__init__ was using llama_context_params() which zero-initializes all fields, breaking RoPE and attention. Changed to _lib.llama_context_default_params() in python/llama_turboquant.py line 470. Native MAS now produces coherent output on par with Python MAS.

## 7. Pending / Needs Additional Work

| Feature | Gap | Next Step |
|---------|-----|-----------|
| Per-agent layer graphs in native MAS | ✅ Implemented via `llama_mas_set_agent_layer_graph()` C API; decode step groups agents by layer graph |
| `llama_context_set_layer_graph` in Python | ✅ Exported in `llama.h`; Python/ctypes can call it directly |
| GPU adapter compute | ✅ CUDA kernels (`ggml_cuda_mas_recursive_forward` / `cross_forward`) wired into `llama-mas.cpp` decode step; runs on GPU when hidden states are CUDA-backed |
| Native MAS KV clone | Single-context design makes `llama_mas_kv_clone` a no-op | Use separate contexts per agent (memory cost) or implement seq_id-based KV copy |
| Executor replanning | Prompt engineering mitigated but not guaranteed | Consider grammar-based JSON plan format + structured executor prompt |
| Memory utilization | Native MAS shares 1 context; Python MAS creates N contexts | Default `useNativeMas=true` helps; investigate n_ctx tuning per model size |
