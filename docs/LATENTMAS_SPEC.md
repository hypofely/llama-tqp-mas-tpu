# LatentMAS — Complete Specification Guide

> **Last updated:** 2026-06-25  
> **Status:** Production-ready for single-context KV-clone mode (`LATENT_MAS` strategy)  
> **Reference:** https://arxiv.org/html/2511.20639v3 | https://github.com/Gen-Verse/LatentMAS

---

## 1. What LatentMAS Is

LatentMAS is a **training-free multi-agent reasoning framework** that moves LLM collaboration from token space into continuous latent space. Instead of agents passing text back and forth (expensive, lossy), they pass **last-layer hidden states** or **KV cache fragments** through a shared working memory.

### Key Advantages
- **14.6% higher accuracy** on 9 benchmarks vs token-based MAS
- **70–83% fewer tokens** generated (no re-encoding between agents)
- **4–4.3× faster inference** (no context re-serialization between agents)

---

## 2. Core Algorithm

### 2.1 Problem with Token-Based Multi-Agent Systems

Traditional MAS (e.g., ChatDev, MetaGPT):
```
Agent A generates 100 tokens → text passes to Agent B → Agent B re-encodes 100 tokens → generates 50 more
```
Problems:
- **Context bloat**: Every agent re-encodes everything before it
- **Lossy transfer**: Text extraction loses subtle signals
- **Slow**: Re-encoding dominates latency

### 2.2 LatentMAS Solution

LatentMAS operates on **KV cache fragments** and **hidden states** directly:

```
Agent A generates 80 tokens → extract last-layer hidden state → project to embedding space →
prepend to Agent B's KV cache → Agent B continues from Agent A's latent state
```

No token round-trip. No re-encoding.

### 2.3 Mathematical Basis

**Alignment projection** (from paper):
```
Wa ≈ W_out† · W_in
```
- `W_in` — token embedding projection (input)
- `W_out` — output head projection (to logits/vocab)
- `W_out†` — pseudo-inverse of `W_out`
- `Wa` — adapter that projects hidden states back into the input space

**Why this works:** In decoder-only transformers, the residual stream is the same at every layer. If you take agent A's last-layer hidden state and can project it into agent B's input space, agent B can continue generation from that exact state.

---

## 3. Agent Roles (Standard Taxonomy)

| Role | Function | Token Budget | Temperature |
|------|----------|-------------|-------------|
| **planner** | Breaks the task into steps, plans approach | ~256 tokens | 0.7–0.9 |
| **executor** | Performs calculations, writes code, executes tasks | ~2048 tokens | 0.3–0.6 |
| **critic/reviewer** | Reviews executor output, checks correctness | ~512 tokens | 0.5–0.7 |
| **outer/synthesizer** | Produces final answer from all agents' work | ~1024 tokens | 0.7–0.9 |
| **refiner** (optional) | Polishes and formats final output | ~512 tokens | 0.5–0.8 |
| **solver** (optional) | Focused specialist for math/logic sub-tasks | ~1024 tokens | 0.3–0.5 |

### In This Implementation (`GEMMA_4_E2B` crew)

| Agent ID | Role | System Prompt | Max Tokens |
|----------|------|--------------|-----------|
| `crew_planner_xxx` | planner | "You are a planner. Break tasks into clear, actionable steps." | 256 |
| `crew_executor_xxx` | executor | "You are an executor. Complete tasks with precision." | 256 |
| `crew_outer_xxx` | outer | "You are the synthesizer. Use planner's steps and executor's results to give the final answer." | 256 |

**Critical:** The **outer/synthesizer** agent MUST produce the final visible output to the user. If outer only says "Please provide the answer", the system is broken.

---

## 4. KV Cache Mechanics

### 4.1 Single-Context Multi-Sequence Design

The C++ implementation uses a **single `llama_context`** with multiple sequences (one per agent):
- `n_agents_max=4` → context allocates `n_seq_max=4` internally
- Each sequence (`seq_id`) has its own KV cache slots
- All agents share the same model weights and compute graph

### 4.2 KV Cache Cloning (`llama_mas_kv_clone`)

```cpp
// Copies full KV cache from src agent to dst agent
// After this, dst can continue generation from src's state
int llama_mas_kv_clone(mas, src_agent_idx, dst_agent_idx);
```

**What happens internally:**
```
for each cell in src KV cache:
    add dst_seq_id to cell (no data copy — zero-copy pointer append)
next decode of dst starts at: dst_position = src_position + src_generated_len
```

### 4.3 KV Cache Row-Level Clone (`llama_mas_kv_clone_range`)

```cpp
// Copy only a sub-range of KV cells (e.g. last N tokens)
int llama_mas_kv_clone_range(mas, src, dst, src_start, src_end, dst_start);
```

Use case: partial KV transfer, pruning stale tokens between agents.

### 4.4 KV Cache Quantization Types

KV cache can be quantized to save VRAM:

| Type | Constant | Bits/channel | Rotation | Notes |
|------|----------|--------------|----------|-------|
| `f16` | `GGML_TYPE_F16` | 16 | None | Full precision |
| `q8_0` | `GGML_TYPE_Q8_0` | 8.5 | None | Standard 8-bit |
| `q4_0` | `GGML_TYPE_Q4_0` | 4.5 | None | Basic 4-bit |
| `turbo4` | `GGML_TYPE_TURBO4_0` | 4 | WHT + PolarQuant | Best quality at 4-bit |
| `turbo3` | `GGML_TYPE_TURBO3_0` | 3 | WHT + PolarQuant | More aggressive |
| `planar3` | `GGML_TYPE_PLANAR3_0` | 3 | 2D Givens | RotorQuant family |
| `iso3` | `GGML_TYPE_ISO3_0` | 3 | 4D quaternion | RotorQuant family |

**KV type must be set at context creation time** (before any tokens are decoded). Changing it requires context recreation.

---

## 5. Strategies Implemented

### 5.1 Strategy Matrix

| Strategy | Value | Behavior | Agent Communication | Use Case |
|----------|-------|----------|---------------------|----------|
| `CHAIN` | 0 | Sequential: A→B→C | Text only | Fallback |
| `BROADCAST` | 1 | All agents see same input | Text only | Parallel evaluation |
| `ROLE` | 2 | Sequential with role prompts | Text only | Standard crew |
| `DTAS` | 3 | Dynamic Task Allocation | Text + KV optional | Adaptive routing |
| `LATENT_MAS` | 5 | Sequential with KV clone | **KV cache transfer** | **Production mode** |
| `LATENT_MAS_NO_ALIGN` | 6 | Like LATENT_MAS but no alignment matrix | **KV cache transfer only** | Debugging |

### 5.2 LATENT_MAS Execution Flow

```
1. LOAD_PHASE:
   For each agent in wave[0]:
     - decode its prompt_tokens through context
     - state becomes: prompt decoded, KV cache populated

2. GENERATE_PHASE (per wave):
   For each agent in wave:
     - llama_mas_decode_step() → generate 1 token at a time
     - check for stop condition (EOS, max_tokens)
     - if not last agent in wave:
         llama_mas_kv_clone(src=last_agent, dst=first_agent_of_next_wave)
         dst.prompt_pos += src.generated_len

3. LATENT_TRANSFER:
   - is_latent_transfer = true for wave[1..N]
   - src agent's KV cells are physically shared with dst agent
   - dst agent's llama_decode continues from src's final position
   - NO re-encoding of src's generated tokens
```

### 5.3 Wave Structure

In the current implementation, each agent is its own wave (strict chain):
```cpp
// Auto-detect: each agent is its own wave (strict latent chain).
for (int i = 0; i < (int)mas->agents.size(); ++i) {
    waves.push_back({i});
}
```

- Wave 0: planner generates
- Wave 1: planner's KV cloned to executor → executor continues
- Wave 2: executor's KV cloned to outer → outer generates final answer

---

## 6. Hidden States API

### 6.1 llm_get_hidden_states_* (C++)

```cpp
// Enable hidden state capture on a context (before decode)
void llama_set_output_hidden_states(llama_context * ctx, bool enabled);

// Get the last token's hidden state from the final layer
// Returns pointer to float array of size llama_model_n_embd(ctx->model)
float * llama_get_hidden_states_last_token(llama_context * ctx, int * n_embd_out);
```

### 6.2 When Hidden States Are Used

Currently only used by:
- `LATENT_MAS_EMBEDDINGS` strategy (embeddings injection, not KV clone)
- Adapter forward pass (for CrossLink/RecursiveLink projections)

### 6.3 The Alignment Matrix

For true hidden-state injection (not just KV clone), the alignment matrix `Wa` projects hidden states back to input space:
```cpp
// Conceptual: not yet fully implemented for Gemma
float * Wa = compute_alignment_matrix(model->W_in, model->W_out);
// When agent A finishes, project its hidden state through Wa,
// then prepend as input embeddings to agent B
```

**Current implementation note:** `LATENT_MAS` in production uses KV cache cloning, not true hidden-state injection. The `LATENT_MAS_EMBEDDINGS` strategy attempts true injection but is **experimental** and prone to failures on models with unusual architectures (GQA, SWA, shared KV layers).

---

## 7. Layer Graph System

### 7.1 What It Is

The **layer graph** allows per-agent control over which transformer layers are active during decode. This enables:
- Skipping early layers for agents that don't need deep reasoning
- Using different computation graphs per agent role
- Dynamic graph switching during MAS execution

### 7.2 Layer Graph Context

```cpp
struct llama_layer_graph_context {
    int    n_layers;                // Total layers in model
    bool   enabled[MAX_LAYERS];     // Which layers to run
    bool   order[MAX_LAYERS];       // Execution order
    bool   skip;                    // If true, skip entire graph
    float  snapshots[MAX_LAYERS];   // Timing/telemetry
};
```

### 7.3 How It's Wired

```
CrewBuilder UI (layerConfig JSON)
    → RecursiveMasClient.js (buildLayerGraphJson)
    → POST /mas/run { agent_layer_graphs: { "planner": {...}, ... } }
    → FastAPI _run_mas_native()
    → llama_mas_set_agent_layer_graph(agent_idx, lg)
    → llama_mas_decode_step():
        - Group agents by layer_graph
        - Each group gets its own decode() call with correct graph active
```

---

## 8. CUDA Adapter Kernels

### 8.1 Purpose

After the main model forward pass, adapter MLPs (CrossLink/RecursiveLink) run on GPU to project hidden states between agents. Without GPU kernels, these run on CPU → 5–10ms overhead per agent transition.

### 8.2 Kernel Inventory

| Kernel | File | Function |
|--------|------|----------|
| `kernel_layernorm` | `mas-adapter.cu` | LayerNorm over hidden states |
| `kernel_rms_norm` | `mas-adapter.cu` | RMSNorm (used by Gemma/LLaMA) |
| `kernel_gelu` | `mas-adapter.cu` | GELU activation |
| `kernel_mat_vec_mul` | `mas-adapter.cu` | Linear projection (matrix × vector) |
| `kernel_vec_add` | `mas-adapter.cu` | Residual addition |
| `kernel_vec_copy` | `mas-adapter.cu` | Memory copy (H2D/D2H) |
| `ggml_cuda_mas_recursive_forward` | `mas-adapter.cu` | Full RecursiveLink pipeline |
| `ggml_cuda_mas_cross_forward` | `mas-adapter.cu` | Full CrossLink pipeline |

### 8.3 Device Memory API

```c
void * ggml_cuda_mas_device_malloc(size_t size);
void   ggml_cuda_mas_device_free(void * ptr);
void   ggml_cuda_mas_memcpy_h2d(void * dst, const void * src, size_t size);
void   ggml_cuda_mas_memcpy_d2h(void * dst, const void * src, size_t size);
```

All CUDA memory operations are confined to `ggml-cuda.dll`. `llama.dll` only imports adapter forward symbols.

---

## 9. MAS Server API Reference

### 9.1 Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Server health check |
| `/v1/models` | GET | List loaded models |
| `/agents` | GET | List registered agents |
| `/agents` | POST | Register a new agent |
| `/mas/run` | POST | Execute MAS inference |
| `/mas/run/async` | POST | Async MAS with task ID |
| `/kv/transfer` | POST | KV cache transfer between agents |
| `/kv/clone` | POST | Full KV clone |
| `/kv/clone_range` | POST | Sub-range KV clone |
| `/hidden/encode` | POST | Encode hidden state to embedding |
| `/hidden/decode` | POST | Decode embedding back to text |
| `/context/recreate` | POST | Recreate context with new params |

### 9.2 /mas/run Request Body

```json
{
  "model": "gemma-4-e2b",
  "agents": [
    {
      "agent_id": "crew_planner_xxx",
      "role": "planner",
      "system_prompt": "...",
      "max_tokens": 256,
      "temperature": 0.8,
      "adapter": "none"
    }
  ],
  "prompt": "User question here",
  "strategy": 5,
  "n_ctx": 30720,
  "n_agents_max": 4,
  "type_k": 47,
  "type_v": 47,
  "shared_kv": true,
  "agent_layer_graphs": {
    "crew_planner_xxx": {"layers": [...], "order": [...]},
    "crew_executor_xxx": {"layers": [...], "order": [...]}
  }
}
```

---

## 10. Known Bugs and Fixes Applied

### Bug 1: KV Type Corruption on Context Recreation

**Symptom:** After switching KV type, server enters `type_k=1` (invalid) state, all future runs fail.

**Root Cause:** `g_state.type_k` was set *before* `recreate_mas_context()`. If recreation failed, the new type was committed but the old context was freed.

**Fix (mas-server.cpp:939-947):**
```cpp
int saved_k = g_state.type_k;
int saved_v = g_state.type_v;
g_state.type_k = new_type_k;
g_state.type_v = new_type_v;
if (!recreate_mas_context({})) {
    g_state.type_k = saved_k;  // restore on failure
    g_state.type_v = saved_v;
    return true;
}
```

### Bug 2: KV Cache Wipe During Latent Transfer

**Symptom:** `init: the tokens of sequence 0 ... have inconsistent sequence positions: X=20, Y=22`

**Root Cause:** In `llama-mas.cpp:1910`, the latent-transfer skip condition was:
```cpp
if (agent->prompt_tokens.empty() && agent->is_latent) { ... }
```
For agents with `is_latent=false` (default), empty prompt tokens fell through to:
```cpp
llama_memory_seq_rm(mem, agent->seq_id, 0, -1); // WIPES the just-cloned KV cache
```

**Fix (llama-mas.cpp:1910-1921):**
```cpp
if (is_latent_transfer) {
    agent->finished = false;
    continue;  // ALL agents skip prompt-decode, preserve cloned KV
}
```

### Bug 3: Start Script Line Wrapping

**Symptom:** `--type-v` argument split across two lines in `start_gemma4_e2b.bat`, causing argument parser to ignore it.

**Fix:** Ensure `--type-k` and `--type-v` are on the same line:
```bat
mas-server.exe -m "C:\gemma-4-E4B-it-Q4_K_S.gguf" -p 8000 --n-ctx 4096 --n-gpu-layers 999 --flash-attn on --type-k turbo4 --type-v turbo4
```

---

## 11. Architektur Model Check: Gemma 4 Specifics

### Why LatentMAS Was Failing on Gemma 4

| Property | Value | Impact |
|----------|-------|--------|
| `head_count` | 8 | Standard |
| `head_count_kv` | 1 | **GQA: 8 query heads share 1 KV head** |
| `embedding_length` | 1536 | Standard |
| `key_length` / `value_length` | 512 | SWA uses 256 |
| `shared_kv_layers` | 20 | **First 20 layers use full KV, last 15 use SWA** |
| `sliding_window` | 512 | Non-standard attention pattern |
| `rope.dimension_count` | 512 | Non-standard RoPE |

**The failure was NOT due to architecture mismatch.** It was the KV cache wipe bug (Bug 2) combined with KV type mismatch (Bug 1). Once both are fixed, LATENT_MAS works correctly on Gemma 4 E2B.

### What to Watch For on Other Models

- **Sliding window attention (SWA):** `llama_kv_cache_iswa` creates filtered caches for SWA layers. KV clone must handle SWA cells correctly.
- **Shared KV layers:** First N layers share KV between heads. KV clone should copy the shared cells, not per-head.
- **GQA/MQA:** Attention computes Q (many heads) × K,V (few heads). KV cache layout differs from MHA.
- **Mixed layer patterns:** Models like Gemma 4 have SWA for early layers, full attention for late. `n_embd_k_gqa` array varies per layer.

---

## 12. Build System & Upstream Sync

### Repository Structure

```
llama-tqp-mas-cpp/
├── _build_mas_server.bat          # Release build
├── _build_mas_server_debug.bat    # Debug build (with PDB)
├── start_gemma4_e2b.bat           # Launch script with all model params
├── scripts/
│   └── sync_upstream.sh           # <-- Port changes on top of new upstream
├── docs/
│   ├── LATENTMAS_SPEC.md          # <-- This file
│   ├── TEST_SCENARIOS.md          # <-- Testing guide
│   └── MAS_IMPLEMENTATION_STATUS.md
├── src/
│   ├── llama-mas.cpp              # CORE: native MAS engine
│   ├── llama-mas.h
│   ├── llama-layer-graph.cpp      # CORE: layer graph engine
│   ├── llama-kv-cache.cpp         # CORE: KV clone/range/export
│   └── models/                    # All model implementations
├── tools/
│   └── mas-server/
│       ├── mas-server.cpp         # CORE: HTTP API
│       └── CMakeLists.txt
└── ggml/src/ggml-cuda/
    ├── mas-adapter.cu/.cuh        # CORE: CUDA adapter kernels
    ├── cpy-planar-iso.cu/.cuh     # RotorQuant KV conversion
    └── turbo-quant.cuh/wht.cu      # TurboQuant kernels
```

### How to Apply Custom Changes on Top of New Upstream

See `scripts/sync_upstream.sh` for the automated procedure. High-level:

1. **Fetch upstream** into `upstream/` directory (separate worktree)
2. **Generate patch** of custom files vs upstream base:
   ```
   diff -ruN llama.cpp-base llama-tqp-mas-cpp > custom.diff
   ```
3. **Apply to new upstream:**
   ```
   cd new-llama.cpp
   patch -p1 < custom.diff
   ```
4. **Resolve conflicts** only in modified files (not new files)
5. **Rebuild** and verify

### Core Files That MUST Be Ported

| File | Priority | Notes |
|------|----------|-------|
| `src/llama-mas.cpp/.h` | **BLOCKER** | Entire MAS engine |
| `src/llama-layer-graph.cpp/.h` | **BLOCKER** | Layer graph system |
| `include/llama.h` additions | **BLOCKER** | C API exports |
| `src/llama-context.cpp` | **BLOCKER** | Layer graph wiring |
| `src/llama-kv-cache.cpp` | **BLOCKER** | KV clone_range |
| `tools/mas-server/` | **BLOCKER** | HTTP server |
| `ggml/src/ggml-cuda/mas-adapter.cu/.cuh` | **HIGH** | GPU adapter kernels |
| `ggml/src/ggml-cuda/CMakeLists.txt` | **HIGH** | Build integration |
| `ggml/src/ggml-cuda/` quant kernels | **MEDIUM** | Turbo/Rotor quant |

### Optional/Extras (May Skip)

| File/Dir | Notes |
|----------|-------|
| `examples/mas-bench/` | Benchmarks only |
| `python/` | Python bindings wrapper |
| `models/templates/` | Chat templates |
| `.devops/` | Docker/Nix configs |
| `.gemini/` | IDE config |
