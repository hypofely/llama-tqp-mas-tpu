# Recursive MAS → Native llama.cpp C++ Architecture Plan

## 1. Current Architecture Map

### 1.1 Data Flow Diagram (Python Wrapper)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI SERVER                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────────────┐  │
│  │ POST /models│   │POST /agents │   │ /mas/run    │   │ /dtas/dispatch   │  │
│  │   load      │   │   create    │   │  orchestrate│   │  route+clone     │  │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └────────┬─────────┘  │
│         │                 │                  │                    │           │
│         ▼                 ▼                  ▼                    ▼           │
│  ┌──────────────────────────────────────────────────────────────────────┐     │
│  │                    PYTHON ORCHESTRATION LAYER                         │     │
│  │  _run_mas_sync()  │  _select_agents()  │  ThreadPoolExecutor(max=2) │     │
│  │  shared_prompt_kv │  share_kv_from_first│  strategy: role/broadcast  │     │
│  └──────────────────────────────────────────────────────────────────────┘     │
│         │                      │                     │                        │
│         ▼                      ▼                     ▼                        │
│  ┌──────────────────────────────────────────────────────────────────────┐     │
│  │                    GGUF AGENT LAYER (per agent)                       │     │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────────┐ │     │
│  │  │ tokenize  │→ │ decode()  │→ │get_logits │→ │ generate_with_    │ │     │
│  │  │           │  │           │  │ (sync)    │  │ stats() loop      │ │     │
│  │  └───────────┘  └─────┬─────┘  └─────┬─────┘  │  for i in range() │ │     │
│  │                       │              │         └───────────────────┘ │     │
│  │                       ▼              ▼                            │     │
│  │  ┌─────────────────────────────────────────────────────────────┐    │     │
│  │  │         LLAMA_TURBOQUANT (ctypes wrapper)                  │    │     │
│  │  │  LlamaSharedModel (weights)  │  LlamaContext (KV cache)   │    │     │
│  │  │  ┌─────────────┐              │  ┌─────────────────────┐   │    │     │
│  │  │  │ llama_model │              │  │ llama_context       │   │    │     │
│  │  │  │  (shared)   │              │  │  (per agent)        │   │    │     │
│  │  │  └──────┬──────┘              │  │  decode()           │   │    │     │
│  │  │         │                      │  │  get_logits()       │   │    │     │
│  │  │         │                      │  │  get_hidden_states()│   │    │     │
│  │  │         │                      │  │  kv_clone_to()      │   │    │     │
│  │  │         │                      │  │  kv_serialize()     │   │    │     │
│  │  │         │                      │  │  kv_layer_export()  │   │    │     │
│  │  │         │                      │  └─────────────────────┘   │    │     │
│  │  └─────────┼──────────────────────┼────────────────────────────┘    │     │
│  │            │                      │                                  │     │
│  │            ▼                      ▼                                  │     │
│  │  ┌─────────────────────────────────────────────────────────────┐    │     │
│  │  │              llama.cpp C++ ENGINE (llama.dll)               │    │     │
│  │  │                                                             │    │     │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │    │     │
│  │  │  │ ggml_backend │  │ llama_decode │  │ llama_synchronize│   │    │     │
│  │  │  │   (CUDA)     │  │  (compute)   │  │  (barrier)       │   │    │     │
│  │  │  └──────────────┘  └──────────────┘  └──────────────────┘   │    │     │
│  │  │                                                             │    │     │
│  │  │  ┌────────────────────────────────────────────────────────┐   │    │     │
│  │  │  │ KV CACHE (per-context)                               │   │    │     │
│  │  │  │  - K tensors: [n_layer, n_kv, head_dim]              │   │    │     │
│  │  │  │  - V tensors: [n_layer, n_kv, head_dim]              │   │    │     │
│  │  │  │  - cloned via llama_kv_cache_clone(src, dst)         │   │    │     │
│  │  │  │  - exported via llama_kv_cache_layer_export()        │   │    │     │
│  │  │  └────────────────────────────────────────────────────────┘   │    │     │
│  │  │                                                             │    │     │
│  │  │  ┌────────────────────────────────────────────────────────┐   │    │     │
│  │  │  │ HIDDEN STATES (per-decode, all layers → CPU copy)    │   │    │     │
│  │  │  │  - llama_get_hidden_state(layer, pos)                │   │    │     │
│  │  │  │  - copied layer-by-layer GPU→CPU in Python loop      │   │    │     │
│  │  │  └────────────────────────────────────────────────────────┘   │    │     │
│  │  └─────────────────────────────────────────────────────────────┘    │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │              PYTORCH ADAPTER LAYER (CPU or CUDA)                      │     │
│  │  RecursiveLink(hidden_size):  LayerNorm→Linear→GELU→Linear→LN      │     │
│  │  CrossLink(in_dim, out_dim):  LayerNorm→Linear→GELU→Linear→residual │     │
│  │  Currently: CPU-bound, causing 5-10ms per forward                    │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 KV Cache Operations

| Operation | API | Data Movement | Used By |
|-----------|-----|---------------|---------|
| `kv_clone_to(dst)` | `llama_kv_cache_clone(src, dst)` | GPU→GPU (within CUDA, if same device) | `shared_prompt_kv`, `share_kv_from_first` |
| `kv_serialize()` | `llama_kv_cache_serialize(ctx, buf, size)` | GPU→CPU→Python bytes | `/kv/export`, persistence |
| `kv_deserialize(data)` | `llama_kv_cache_deserialize(ctx, buf, size)` | CPU→GPU | `/kv/import`, restoration |
| `kv_layer_export(layer, flags)` | `llama_kv_cache_layer_export()` | GPU→CPU (TurboQuant-4 compressed) | Cross-model KV transfer |
| `kv_layer_import(data)` | `llama_kv_cache_layer_import()` | CPU→GPU (decompress) | Cross-model KV restore |

### 1.3 Shared Prompt KV Flow (`/mas/run` with `shared_prompt_kv=true`)

```
Role A (first agent):
  1. encode("Task: {task}\n") 
     → decode(tokens) → KV cache populated
     → hidden state extracted

Role B (second agent):
  2. src_model.kv_clone_to(dst_model)
     → llama_kv_cache_clone copies K/V tensors from A→B
     → B's position counter set to A's
  3. generate_with_stats(prompt)
     → decode continues from cloned position
     → No re-encoding of shared prefix

Role C, D, ...: same as B
```

**Performance gain**: Avoids re-decoding the system prompt + task for each agent.
For a 200-token prompt × 4 agents = 600 tokens of compute saved.

### 1.4 DTAS KV Clone Flow (`/dtas/dispatch` with `share_kv_from_first=true`)

```
Agent 1: decode(full_prompt) → generate() → KV cache has full prompt
Agent 2: clone KV from Agent 1 → generate()  (no re-encode)
Agent 3: clone KV from Agent 1 → generate()  (no re-encode)
```

**Difference from MAS**: DTAS is parallel dispatch; MAS is sequential role chain.

---

## 2. Bottleneck Analysis

### 2.1 Current Python-Layer Bottlenecks

| # | Bottleneck | Time Cost | Root Cause |
|---|-----------|-----------|------------|
| 1 | `llama_synchronize()` per token | 1-3ms | CUDA barrier before `get_logits()` |
| 2 | `get_hidden_states()` all layers copy | 10-50ms | GPU→CPU copy of `n_layer × n_embd × n_tokens` |
| 3 | PyTorch adapter on CPU | 5-10ms | `RecursiveLink`/`CrossLink` in CPU PyTorch |
| 4 | Python `for` loop per token | 0.5-2ms | GIL, ctypes overhead, numpy argmax |
| 5 | `get_logits()` CPU copy | 0.5-1ms | 128K float array GPU→CPU for numpy |
| 6 | KV clone through Python | ~1ms | ctypes call overhead + Python exception handling |
| 7 | ThreadPoolExecutor contention | Variable | GIL, max_workers=2 default |

### 2.2 Theoretical Speed Ceiling

| Mode | Tokens/s | Bottleneck |
|------|----------|------------|
| Direct completion (no hidden states) | ~32 | decode() only, minimal sync |
| MAS generation (current) | ~10 | hidden states + adapter + sync |
| MAS with CUDA adapters | ~15-18 | sync + Python loop still present |
| Native C++ MAS (target) | ~28-30 | decode() + in-graph adapter, no sync |
| Batch-parallel C++ MAS | ~50-80 | Multiple sequences batched in one decode |

---

## 3. C++ Native Architecture Plan

### 3.1 Design Goal

Move the entire MAS orchestration **into llama.cpp's compute graph**, eliminating:
- Python per-token loops
- GPU→CPU hidden state copies
- `llama_synchronize()` per token
- ctypes/Python overhead

### 3.2 Proposed C++ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NATIVE LLAMA.CPP MAS ENGINE                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  llama_mas_context (NEW)                                            │   │
│  │  - Owns multiple llama_context (one per agent)                       │   │
│  │  - Shared llama_model across all agents                              │   │
│  │  - Central KV cache pool with clone/transfer support                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  llama_mas_agent (NEW)                                              │   │
│  │  - Role name, system prompt, adapter weights                       │   │
│  │  - Owns: llama_context (KV cache slot)                             │   │
│  │  - Adapter: llama_adapter_graph (in-graph MLP)                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  llama_mas_adapter (NEW) — In-Graph Adapter MLP                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  GGML compute graph (runs on GPU)                          │    │   │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐          │    │   │
│  │  │  │ ggml_norm  │→ │ ggml_mul_mat│→ │ ggml_gelu  │→ ...     │    │   │
│  │  │  │ (LayerNorm)│  │ (Linear)   │  │ (activation)│          │    │   │
│  │  │  └────────────┘  └────────────┘  └────────────┘          │    │   │
│  │  │                                                            │    │   │
│  │  │  Input:  hidden_state tensor [1, n_embd] (GPU)             │    │   │
│  │  │  Output: adapted_state tensor [1, n_embd] (GPU)          │    │   │
│  │  │                                                            │    │   │
│  │  │  Weights stored as GGML tensors (q4_0 / q8_0 / f16)      │    │   │
│  │  │  Loaded from: <model_name>_adapter.gguf                 │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  llama_mas_decode (NEW) — Batched Multi-Agent Decode              │   │
│  │                                                                     │   │
│  │  Instead of: for each token: decode(1) + sync + get_logits      │   │
│  │  Do:         for each step:                                      │   │
│  │              1. Build batch with tokens from ALL agents           │   │
│  │              2. Single llama_decode(batch) call                   │   │
│  │              3. Extract logits for all agents in one shot       │   │
│  │              4. Sample next tokens for all agents (GPU or CPU)  │   │
│  │              5. Repeat until all agents hit EOS/max_tokens      │   │
│  │                                                                     │   │
│  │  Key insight: llama_decode already supports batch>1 (n_batch)   │   │
│  │  We just need proper seq_id assignment per agent                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Hidden State Extraction (MODIFIED llama.cpp)                     │   │
│  │                                                                     │   │
│  │  Current: llama_get_hidden_state(layer, pos) → CPU copy           │   │
│  │  New:     llama_get_hidden_state_gpu(layer, pos) → GPU tensor    │   │
│  │           (returns a ggml_tensor view, no copy)                   │   │
│  │                                                                     │   │
│  │  The adapter graph consumes this GPU tensor directly              │   │
│  │  Zero CPU involvement in hidden state path                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  KV Cache Transfer (ENHANCED existing)                            │   │
│  │                                                                     │   │
│  │  llama_kv_cache_clone(src, dst)  — already exists                  │   │
│  │  llama_kv_cache_share_slot(src, dst, offset) — NEW                │   │
│  │    (share prefix KV between agents without full copy)             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Plan (Phased)

### Phase 1: Adapter Weight Format (2-3 days)

**Goal**: Define and implement the adapter weight storage format.

```
File: <model>_adapter.gguf

Tensors:
  adapter.recursive.pre_ln.weight    [n_embd]        f32
  adapter.recursive.pre_ln.bias      [n_embd]        f32  (optional)
  adapter.recursive.proj1.weight     [n_embd, n_embd] q4_0/q8_0
  adapter.recursive.proj1.bias       [n_embd]        f32
  adapter.recursive.proj2.weight     [n_embd, n_embd] q4_0/q8_0
  adapter.recursive.proj2.bias       [n_embd]        f32
  adapter.recursive.post_ln.weight   [n_embd]        f32
  adapter.recursive.post_ln.bias     [n_embd]        f32  (optional)
  
  adapter.cross.ln_source.weight     [in_dim]        f32
  adapter.cross.proj1.weight         [in_dim, h_dim] q4_0/q8_0
  adapter.cross.proj1.bias           [h_dim]         f32
  adapter.cross.proj2.weight         [h_dim, out_dim]q4_0/q8_0
  adapter.cross.proj2.bias           [out_dim]       f32
  adapter.cross.residual.weight      [in_dim,out_dim]q4_0/q8_0
  adapter.cross.residual.bias        [out_dim]       f32
  adapter.cross.ln_target.weight     [out_dim]       f32

Metadata:
  adapter.architecture = "recursive_mas"
  adapter.hidden_size  = <n_embd>
  adapter.version      = 1
```

**Files to create/modify**:
- `common/adapter.h` — adapter tensor definitions
- `common/adapter.cpp` — GGUF load/save for adapters
- `convert_adapter.py` — PyTorch → GGUF converter

### Phase 2: In-Graph Adapter Compute (3-5 days)

**Goal**: Execute adapter MLP inside the GGML compute graph.

```cpp
// ggml_adapter.h (NEW)
struct llama_adapter {
    struct ggml_tensor * w_pre_ln;   // [n_embd]
    struct ggml_tensor * w_proj1;  // [n_embd, n_embd]
    struct ggml_tensor * b_proj1;  // [n_embd]
    struct ggml_tensor * w_proj2;  // [n_embd, n_embd]
    struct ggml_tensor * b_proj2;  // [n_embd]
    struct ggml_tensor * w_post_ln;  // [n_embd]
    ggml_backend_buffer_t buffer;
};

// Build compute graph for adapter forward
static struct ggml_cgraph * build_adapter_graph(
    struct ggml_context * ctx,
    struct llama_adapter * adapter,
    struct ggml_tensor * input,   // [1, n_embd] — hidden state
    struct ggml_tensor * output   // [1, n_embd] — adapted state
) {
    // h = pre_ln(x)
    struct ggml_tensor * h = ggml_norm(ctx, input, adapter->w_pre_ln);
    
    // out = proj2(gelu(proj1(h)))
    struct ggml_tensor * tmp = ggml_mul_mat(ctx, adapter->w_proj1, h);
    tmp = ggml_add(ctx, tmp, adapter->b_proj1);
    tmp = ggml_gelu(ctx, tmp);
    tmp = ggml_mul_mat(ctx, adapter->w_proj2, tmp);
    tmp = ggml_add(ctx, tmp, adapter->b_proj2);
    
    // residual: out = x + tmp
    tmp = ggml_add(ctx, input, tmp);
    
    // post_ln
    output = ggml_norm(ctx, tmp, adapter->w_post_ln);
    
    // Build and return graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);
    return gf;
}
```

**Files to create**:
- `ggml-adapter.h/.cpp` — GGML graph construction for adapter
- `llama-adapter.h/.cpp` — llama.cpp integration

**Files to modify**:
- `llama.cpp` — hook adapter into decode pipeline

### Phase 3: GPU-Resident Hidden States (2-3 days)

**Goal**: Keep hidden states on GPU; expose as GGML tensor.

```cpp
// Current (slow):
float * llama_get_hidden_state(struct llama_context * ctx, int32_t layer, int32_t pos);
// → Copies from GPU to CPU buffer, returns CPU pointer

// New (fast):
struct ggml_tensor * llama_get_hidden_state_gpu(
    struct llama_context * ctx,
    int32_t layer,
    int32_t pos
);
// → Returns a GPU-resident ggml_tensor view, NO copy
```

**Implementation**:
- llama.cpp already computes hidden states in `llama_build_graph()`
- After `ggml_graph_compute()`, the hidden state tensor is GPU-resident
- We just need to expose a pointer to it before the next decode resets it

**Files to modify**:
- `llama.h` — add `llama_get_hidden_state_gpu()` declaration
- `llama.cpp` — implement, save tensor reference after compute

### Phase 4: Batched Multi-Agent Decode (4-6 days)

**Goal**: Decode tokens from ALL agents in a single `llama_decode()` call.

```cpp
// llama_mas.h (NEW)
struct llama_mas_agent {
    const char * role;
    llama_seq_id seq_id;      // Unique sequence ID within batch
    llama_token * prompt_tokens;
    size_t prompt_len;
    size_t generated_len;
    bool finished;
    struct llama_adapter * inner_adapter;
    struct llama_adapter * cross_adapter;  // from prev agent
};

struct llama_mas_context {
    struct llama_model * model;
    struct llama_context * ctx;  // Single context, multiple seqs
    struct llama_mas_agent * agents;
    size_t n_agents;
    size_t max_tokens;
    enum llama_mas_strategy {
        LLAMA_MAS_STRATEGY_ROLE,       // Sequential
        LLAMA_MAS_STRATEGY_BROADCAST,  // Parallel
        LLAMA_MAS_STRATEGY_CHAIN       // Output feeds next
    } strategy;
};

// Main decode loop
int llama_mas_decode(struct llama_mas_context * mas) {
    while (!all_agents_finished(mas)) {
        // Build batch from all active agents
        llama_batch batch = {0};
        int n_active = 0;
        for (int i = 0; i < mas->n_agents; i++) {
            if (!mas->agents[i].finished) {
                // Add agent's next token to batch
                batch.token[n_active] = get_next_token(&mas->agents[i]);
                batch.pos[n_active] = mas->agents[i].prompt_len + mas->agents[i].generated_len;
                batch.n_seq_id[n_active] = 1;
                batch.seq_id[n_active][0] = mas->agents[i].seq_id;
                batch.logits[n_active] = 1;  // compute logits for all
                n_active++;
            }
        }
        batch.n_tokens = n_active;
        
        // Single decode for ALL agents
        int ret = llama_decode(mas->ctx, batch);
        if (ret != 0) return ret;
        
        // Extract logits for all agents
        for (int i = 0; i < n_active; i++) {
            float * logits = llama_get_logits_ith(mas->ctx, i);
            int next_token = sample_token(logits, ...);
            // ... apply adapter if needed ...
            advance_agent(&mas->agents[i], next_token);
        }
    }
    return 0;
}
```

**Key insight**: llama.cpp already supports `n_seq_max > 1` and `seq_id` arrays.
We just need proper batch construction and per-sequence logits extraction.

**Files to create**:
- `llama-mas.h/.cpp` — Multi-agent context and decode

**Files to modify**:
- `llama.cpp` — ensure `llama_get_logits_ith()` works with multi-seq batches

### Phase 5: KV Cache Sharing Optimization (1-2 days)

**Goal**: Avoid full KV clone for shared prefix.

```cpp
// Current: llama_kv_cache_clone copies ALL K/V tensors
// New: llama_kv_cache_share_prefix creates a view

int llama_kv_cache_share_prefix(
    struct llama_context * src,
    struct llama_context * dst,
    llama_pos prefix_len
);
```

**Implementation**:
- llama.cpp KV cache is a list of "cells" per position
- For prefix sharing, we can point dst's first N cells to src's cells
- Only divergent positions need unique cells
- This is how vLLM's PagedAttention works

**Alternative (simpler)**: Just use `llama_kv_cache_clone()` which is already native and fast.
The clone is GPU→GPU, so it's already near-optimal.

### Phase 6: Python API Wrapper (2-3 days)

**Goal**: Minimal Python ctypes wrapper around the new C++ API.

```python
# llama_mas_native.py (NEW — thin wrapper)
from ctypes import CDLL, c_void_p, c_int, c_char_p, POINTER

_lib = CDLL("llama.dll")

class LlamaMasContext:
    def __init__(self, model_path, n_agents):
        self.ctx = _lib.llama_mas_init(model_path.encode(), n_agents)
    
    def add_agent(self, role, adapter_path, system_prompt):
        _lib.llama_mas_add_agent(self.ctx, role.encode(), ...)
    
    def run(self, task, max_tokens, strategy="role"):
        return _lib.llama_mas_run(self.ctx, task.encode(), max_tokens, strategy.encode())
    
    def get_outputs(self):
        # Returns dict {role: text}
        pass
```

**The FastAPI layer becomes THIN**:
- `/mas/run` → `llama_mas_run()` (one C call)
- `/dtas/dispatch` → `llama_mas_run()` with strategy="broadcast"
- `/kv/clone` → `llama_kv_cache_clone()` (already native)
- No Python loops, no per-token overhead

---

## 5. Estimated Performance Impact

| Change | Current | After C++ | Speedup |
|--------|---------|-----------|---------|
| Per-token Python loop | 0.5-2ms | 0ms | ∞ |
| `llama_synchronize()` | 1-3ms/token | 0 (batched) | ∞ |
| Hidden states GPU→CPU | 10-50ms/token | 0 (GPU tensor) | ∞ |
| Adapter CPU compute | 5-10ms/token | <0.1ms (GPU) | 50-100x |
| Batch decode (4 agents) | 4 × 3ms = 12ms | 1 × 3ms = 3ms | 4x |
| **Overall MAS throughput** | **~10 tok/s** | **~25-30 tok/s** | **2.5-3x** |
| Batch-parallel agents | 10 tok/s each | 25 tok/s each × 4 | **10x total** |

---

## 6. File Inventory for C++ Changes

### New Files (in llama.cpp repo)

| File | Purpose | Lines est. |
|------|---------|------------|
| `ggml-adapter.h/.cpp` | GGML graph for adapter MLP | ~300 |
| `llama-adapter.h/.cpp` | llama.cpp adapter integration | ~400 |
| `llama-mas.h/.cpp` | Multi-agent context + batched decode | ~600 |
| `examples/mas/mas.cpp` | CLI example for native MAS | ~200 |

### Modified Files (in llama.cpp repo)

| File | Change | Lines est. |
|------|--------|------------|
| `llama.h` | Add `llama_get_hidden_state_gpu()`, `llama_mas_*` | ~50 |
| `llama.cpp` | Expose hidden state tensor, hook adapter | ~200 |
| `ggml.h` | (maybe) `ggml_norm()` with learned weights | ~0 (exists) |
| `common/common.h` | Adapter load/save helpers | ~100 |

### New Files (in Python repo)

| File | Purpose |
|------|---------|
| `convert_adapter.py` | PyTorch state_dict → GGUF adapter |
| `llama_mas_native.py` | Thin ctypes wrapper |
| `create_adapter_for_model.ps1` | Updated to emit GGUF |

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GGML doesn't support learned LayerNorm params | Low | High | Use `ggml_mul()` + `ggml_add()` manually |
| Adapter quantization degrades quality | Medium | Medium | Test fp16 vs q8_0 vs q4_0 |
| Multi-seq batching has bugs in llama.cpp | Medium | High | Test with existing `perplexity` example |
| CUDA graph capture breaks with adapters | Medium | High | Disable CUDA graphs for MAS mode |
| Training data mismatch (adapter not fine-tuned) | High | Low | Adapters are randomly initialized anyway |

---

## 8. Decision Checkpoints

1. **Do we really need C++?** 
   - If PyTorch CUDA adapters + batched Python gets you to 20 tok/s, maybe not.
   - If you want 30+ tok/s or multi-agent batching, yes.

2. **Can we skip the adapter and just use raw hidden states?**
   - Yes: remove `RecursiveLink`/`CrossLink`, just `mean()` or `cat()` hidden states.
   - Much simpler C++: just batch decode + KV sharing.

3. **Should we use vLLM instead?**
   - vLLM has native multi-LoRA + continuous batching.
   - But: no hidden states API, no KV clone (different architecture).
   - Trade-off: vLLM = better throughput, llama.cpp = more control.

---

## 9. Recommended First Step

**Before writing any C++**, validate the adapter concept:

1. Train a small `RecursiveLink` on your model's hidden states
2. Measure: does the adapter actually improve output quality vs raw hidden states?
3. If yes → full C++ plan above
4. If no → just implement batched decode + KV sharing (Phase 4 only, much simpler)

Most of the speed gain comes from **batching + removing sync**, not from the adapter being in C++.
