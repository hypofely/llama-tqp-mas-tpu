# CUDA & Architecture Performance Audit

**Project:** llama-tqp-mas-cpp (MAS Server)  
**Date:** 2026-06-03  
**Scope:** Identify CPU-bound operations that can be moved to CUDA and architectural improvements for inference throughput.

---

## 1. Executive Summary

The MAS server already offloads the majority of heavy compute (matrix multiplications, flash attention, RoPE, norms) to CUDA. However, **significant CPU bottlenecks remain in the multi-agent orchestration layer, KV cache metadata management, sampling, and graph building**. The biggest single opportunity is **batched multi-agent decoding**, which could yield 2-8x throughput improvements depending on agent count.

### Top 5 Opportunities (by impact)

| # | Opportunity | Est. Speedup | Effort |
|---|-------------|--------------|--------|
| 1 | Batched multi-agent decode (single `llama_decode` call) | 2-8x | High |
| 2 | MAS adapter forward on CUDA (end-to-end, no CPU fallback) | 1.2-1.5x | Medium |
| 3 | KV cache metadata on GPU (shift/idx bookkeeping) | 1.1-1.3x | High |
| 4 | Graph caching / reduce `sched_reserve` calls | 1.1-1.2x | Medium |
| 5 | Sampler backend offload (full chain on GPU) | 1.05-1.15x | Low-Medium |

---

## 2. CUDA Coverage Analysis

### 2.1 Well-Covered (CUDA-native)

- **Matrix ops:** `MUL_MAT`, `MUL_MAT_ID`, `OUT_PROD`, `SCALE`
- **Activations:** `GELU`, `SILU`, `RELU`, `SIGMOID`, `TANH`, `EXP`, `SWIGLU`, `GEGLU`
- **Normalization:** `RMS_NORM`, `GROUP_NORM`, `L2_NORM`, `NORM`
- **Attention:** `FLASH_ATTN_EXT` (primary path), `ROPE`, `SOFT_MAX`
- **Custom quant ops:** `TURBO_WHT`, `SET_ROWS` (planar3/iso3/turbo3)
- **Reductions:** `SUM`, `SUM_ROWS`, `MEAN`, `TOP_K`, `ARGMAX`, `ARGSort`
- **Conv/SSM:** `CONV_2D`, `SSM_CONV`, `SSM_SCAN`, `RWKV_WKV6/7`
- **Optimizers:** `OPT_STEP_ADAMW`, `OPT_STEP_SGD`

### 2.2 CPU Fallback Traps

Any `ggml_op` not explicitly handled in `ggml_cuda_compute_forward` (`ggml-cuda.cu:2660-2991`) falls back to the CPU backend, triggering:
1. **Device-to-host copy** of inputs
2. **CPU compute**
3. **Host-to-device copy** of outputs

This is especially painful for small tensors where copy overhead exceeds compute time.

**Known unsupported paths:**
- Niche unary ops not in the switch (e.g. `HARDSIGMOID_BACK` if it existed)
- Any new custom ops added without CUDA kernels
- `GGML_OP_NONE / RESHAPE / VIEW / PERMUTE / TRANSPOSE` — no-ops, fine

---

## 3. MAS Layer Bottlenecks

### 3.1 Agent Decode: Sequential Single-Token Batches

**Location:** `src/llama-mas.cpp:702-735` (`llama_mas_decode_step`)

```cpp
for (int i = 0; i < (int)mas->agents.size(); ++i) {
    // ... each agent gets its own llama_decode with batch.n_tokens == 1
    int ret = llama_decode(mas->ctx, batch);  // <─ GPU kernel launch per agent
}
```

**Problem:** With N active agents, this launches N separate GPU graphs per timestep. Each launch has:
- Graph building overhead (~0.1-0.5ms)
- Kernel launch latency (~0.02ms)
- CUDA stream synchronization
- Missing batching benefits (no `MUL_MAT` batching across agents)

**Impact:** For 8 agents, this is ~8x slower than a single batched decode of 8 tokens.

**Fix:** Merge agents into a single `llama_batch` with distinct `seq_id`s and call `llama_decode` once. The KV cache already supports multi-sequence; the assertion that blocked this should be revisited.

---

### 3.2 MAS Adapter Forward: CPU Fallback for Encode Path

**Location:** `src/llama-mas.cpp:206-284` (`recursive_forward`, `cross_forward`)

The adapter forward pass has **two implementations:**

1. **CUDA path:** `ggml_cuda_mas_recursive_forward` / `ggml_cuda_mas_cross_forward` — exists but is only used in `llama_mas_encode_agent` when `gpu_data` is present.
2. **CPU path:** `llama_mas_adapter_recursive_forward` — pure CPU loops over `n_embd` (typically 3072-8192).

**Problem in `llama_mas_encode_agent` (`src/llama-mas.cpp:1064-1093`):**

```cpp
if (agent->adapter_type == LLAMA_MAS_ADAPTER_RECURSIVE && agent->adapter_weights) {
#ifdef GGML_USE_CUDA
    if (agent->adapter_weights->gpu_data) {
        // CUDA path: h2d copy → kernel → d2h copy
    }
#endif
    // else: CPU path (not even called; encode path lacks CPU adapter application)
}
```

Even the CUDA path does:
1. `malloc` + `memcpy` of hidden states to GPU
2. One-off kernel launch
3. `malloc` + `memcpy` back to CPU

**Impact:** For 3072-dim embeddings, the copies alone are ~12us per direction, but the kernel launch and sync dominate. With many agents, this adds up.

**Fix:** Keep adapter weights permanently on GPU. Apply adapters as part of the compute graph (ggml custom op) rather than manual CUDA calls. This eliminates all H2D/D2H copies.

---

### 3.3 Adapter Weight Upload: Per-Agent, Not Shared

**Location:** `src/llama-mas.cpp:342-378`

Each agent that provides adapter weights gets a full GPU copy. If 16 agents share the same adapter topology, there are 16 duplicate GPU allocations.

**Fix:** Reference-counted GPU weight cache keyed by adapter signature.

---

### 3.4 Hidden State Extraction: CPU Copies

**Location:** `src/llama-mas.cpp:938-942`

```cpp
float * result = (float *)malloc(size);
std::memcpy(result, layer_data, size);
```

`layer_data` comes from `llama_get_hidden_states_layer`, which returns a pointer to GPU memory (when using CUDA). The `memcpy` here is a **blocking D2H copy** of the entire last-layer hidden state.

**Impact:** For 128 tokens × 4096 dims × 4 bytes = 2 MiB per encode. With 100 encode calls = 200 MiB of PCIe traffic.

**Fix:** If downstream processing (kNN, AVP) can run on GPU, keep the tensor on device. Only copy back when returning to the user.

---

## 4. KV Cache Bottlenecks

### 4.1 Metadata Management: Entirely CPU-Side

**Locations:**
- `src/llama-kv-cache.cpp:518-543` (`clear`)
- `src/llama-kv-cache.cpp:545-606` (`seq_rm`)
- `src/llama-kv-cache.cpp:608-714` (`seq_cp`)
- `src/llama-kv-cache.cpp:717-763` (`seq_add`)
- `src/llama-kv-cache.cpp:1020-1358` (`find_slot`)

The KV cache maintains `v_cells` (per-cell metadata: sequence IDs, positions, shifts) as `std::vector` of CPU structs. Every token insertion, deletion, or shift requires:
- CPU loops over cells
- Bitmap/sequence set operations
- Then a separate `ggml_set_rows` graph op to actually write the tensor data

**Impact:** For 128K context × 32 layers, `find_slot` and `seq_rm` can scan millions of cells. This is negligible at small contexts but becomes a bottleneck at 32K+.

**Fix:** Store cell metadata in a GPU-resident sparse structure (e.g. CUDA `thrust::device_vector` of free lists) and manage allocations via GPU kernels. The `ggml_set_rows` op already runs on CUDA; only the metadata bookkeeping needs to move.

---

### 4.2 K-Shift Graph Rebuild

**Location:** `src/llama-kv-cache.cpp:979-1015`

When positions shift (e.g. context sliding), the KV cache:
1. Resets the scheduler (`ggml_backend_sched_reset`)
2. Builds a new `build_graph_shift` graph
3. Runs `graph_compute` for every layer

This happens **synchronously** before the main decode can proceed.

**Impact:** ~1-3ms per shift, stalls the token generation pipeline.

**Fix:** Pre-allocate and cache the shift graph. The graph topology is identical every time; only the `k_shift` tensor values change.

---

### 4.3 No KV Defragmentation

**Observation:** No `defrag` or `compact` function exists in the KV cache. As sequences are added and removed, the cache may fragment. `find_slot` does first-fit search; fragmentation increases search time.

**Fix:** Implement periodic defragmentation as a CUDA memcpy of KV buffers + metadata reordering. This is complex but important for long-running servers.

---

## 5. Graph Building Overhead

### 5.1 Graph Rebuild on Sampler Changes

**Location:** `src/llama-context.cpp:1162-1182`

Changing a sampler (`set_sampler`) triggers:
```cpp
sched_need_reserve = true;
```

Which causes a full `sched_reserve()` on the next decode, rebuilding both PP and TG graphs.

**Impact:** `sched_reserve` takes 50-200ms in the logs. If agents change samplers frequently, this is catastrophic.

**Fix:** Separate sampler graphs from model graphs. The sampler graph is tiny (softmax + top-k + temperature); it should not force a full model graph rebuild.

---

### 5.2 Graph Rebuild on Adapter / Context Changes

Same pattern for:
- `set_causal_attn` → `sched_need_reserve = true`
- `adapters_lora_apply` → `sched_need_reserve = true`
- `set_output_hidden_states` → commented out, but should be considered

**Fix:** Use `ggml_backend_sched`’s graph update API to mutate inputs without rebuilding the topology. Most of these changes only affect tensor values, not shapes.

---

### 5.3 Per-Decode Graph Building (Non-Reuse Path)

**Location:** `src/llama-context.cpp:1278`

```cpp
gf = model.build_graph(gparams);
```

When `graph_reuse_disable` is true or the graph cannot be reused, this builds the full transformer graph from scratch every decode.

**Impact:** For a 32-layer model, this constructs ~1000+ `ggml_tensor` nodes. At ~1us per node, that's ~1ms overhead per token.

**Fix:** Ensure graph reuse is enabled in production. The `gf_res_prev` / `can_reuse` mechanism should handle this, but verify it triggers for MAS decode patterns.

---

## 6. Sampling Bottlenecks

### 6.1 CPU Sampling Fallback

**Location:** `src/llama-sampler.cpp:808-816`

```cpp
// If a backend sampler has already sampled a token, return it.
if (sampled_token != LLAMA_TOKEN_NULL) {
    return sampled_token;
}
```

If the sampler chain contains any op unsupported by CUDA (e.g. `penalties` with non-standard parameters), the entire chain falls back to CPU.

**CPU sampler path (`llama_sampler_chain_apply`):**
- `softmax` over vocab (32K-128K elements) — CPU loop
- `top_k` — `std::partial_sort` on CPU
- `top_p` — CPU accumulation loop
- `temperature` — CPU elementwise

**Impact:** For 128K vocab, CPU softmax is ~0.5-1ms. With many agents, this serializes.

**Fix:** Ensure all samplers in the chain implement `backend_init` / `backend_apply` and use only CUDA-supported ops. The `llama_sampler_greedy` and `llama_sampler_temp` already have backend paths; verify penalties and top_p do too.

---

### 6.2 Per-Agent Sampler Allocation

**Location:** `src/llama-mas.cpp:443-463`

Each agent gets its own `llama_sampler_chain`. For 16 agents, that's 16 separate sampler states, 16 separate backend init checks, and 16 separate CPU fallback paths.

**Fix:** If agents share sampling parameters (common in role-based MAS), use a single backend sampler graph and only vary logits per agent. This requires architectural change to share sampler state.

---

## 7. Tokenization Bottlenecks

### 7.1 CPU-Only Tokenization

**Location:** `src/llama-vocab.cpp` (all tokenization)

`llama_tokenize` and `llama_detokenize` are pure CPU. For long prompts (4K-32K tokens), this can take 1-10ms.

**Impact:** In the MAS `llama_mas_run` loop, every agent re-tokenizes its prompt independently. With 8 agents and 2K tokens each, that's 16K tokens of CPU tokenization before any GPU work begins.

**Fix:**
- **Short-term:** Cache tokenizations. If the same prompt is used repeatedly, store the token vector.
- **Medium-term:** Use a faster CPU tokenizer (e.g. `tiktoken` Rust core) — out of scope for this codebase.
- **Long-term:** BPE/SentencePiece tokenization is not GPU-friendly; this is a fundamental limit.

---

## 8. Memory Architecture Issues

### 8.1 Hidden State Double Buffering

**Location:** `src/llama-mas.cpp:938-942`

The MAS encode functions allocate a CPU buffer and `memcpy` from the GPU hidden states. Then, if an adapter is applied on GPU, they allocate *another* GPU buffer and copy back.

**Flow:**
```
GPU hidden states → CPU malloc+memcpy → (optional) CPU malloc+GPU H2D → CUDA kernel → D2H → CPU
```

**Fix:** Keep everything on GPU. Adapters should be ggml custom ops in the graph, not manual CUDA.

---

### 8.2 Adapter Weight Duplication

**Location:** `src/llama-mas.cpp:342-378`

Each agent's adapter weights are `calloc`'d on CPU, then `cudaMalloc` + `cudaMemcpy` to GPU. With 16 agents × 8192-dim adapters, this is ~1GB of duplicated GPU memory.

**Fix:** Weight sharing + reference counting. Most agents in a role-based MAS use the same adapter topology.

---

### 8.3 KV Cache Stream Copies

**Location:** `src/llama-kv-cache.cpp:608-714`

`seq_cp` across streams requires a full buffer copy (`sc_info` enqueued copy). This is done via `ggml_backend_tensor_set` / `get`, which may go through CPU staging buffers depending on backend.

**Fix:** Use `cudaMemcpyPeerAsync` for cross-device copies or `cudaMemcpyAsync` for same-device. Ensure the copy stays on GPU without host staging.

---

## 9. Recommendations by Priority

### P0 — Critical (Do First)

#### 9.1 Batched Multi-Agent Decode
**File:** `src/llama-mas.cpp:702-735`  
**Action:** Replace the per-agent `llama_decode` loop with a single batched decode containing all active agents' tokens, each with a distinct `seq_id`.  
**Expected Impact:** 2-8x throughput increase with multiple agents.  
**Risk:** KV cache multi-sequence assertions need to be validated. The comment says "multi-sequence KV cache assertion" blocks this — find and fix that assertion.

---

### P1 — High Impact

#### 9.2 MAS Adapter as ggml Custom Op
**File:** `src/llama-mas.cpp` (adapter forward)  
**Action:** Replace manual `ggml_cuda_mas_recursive_forward` / `ggml_cuda_mas_cross_forward` with a proper `ggml_op` (e.g. `GGML_OP_MAS_ADAPTER`) that the scheduler can place on GPU automatically.  
**Expected Impact:** Eliminates all H2D/D2H copies for adapter application. ~1.2-1.5x end-to-end for adapter-heavy workflows.  
**Risk:** Requires adding a new op to ggml; moderate complexity.

#### 9.3 Cache Shift Graph
**File:** `src/llama-kv-cache.cpp:979-1015`  
**Action:** Pre-build and cache the `build_graph_shift` graph in `llama_kv_cache`. On shift, only update the `k_shift` input tensor values.  
**Expected Impact:** ~1-3ms saved per shift event.  
**Risk:** Low. Graph topology is static.

#### 9.4 Sampler Backend Enforcement
**File:** `src/llama-sampler.cpp`  
**Action:** Audit all sampler types for `backend_apply` completeness. Ensure `penalties`, `top_k`, `top_p`, `dist` all have CUDA paths. Add a debug assert or warning when CPU fallback occurs.  
**Expected Impact:** ~0.5-1ms per token saved for large vocab.  
**Risk:** Low. Existing infrastructure supports this.

---

### P2 — Medium Impact

#### 9.5 KV Cache Metadata on GPU
**File:** `src/llama-kv-cache.cpp` (cell management)  
**Action:** Migrate `v_cells` metadata (seq_id bitmaps, positions, shifts) to GPU-resident structures. Implement `find_slot`, `seq_rm`, `seq_add` as CUDA kernels.  
**Expected Impact:** ~1.1-1.3x at 32K+ context. Essential for 128K context targets.  
**Risk:** High complexity. Metadata operations are irregular (set ops, bitmaps).

#### 9.6 Graph Reuse Hardening
**File:** `src/llama-context.cpp` (graph building)  
**Action:** Add metrics/logging for graph reuse rate. Prevent `sched_need_reserve` from being set by sampler-only changes.  
**Expected Impact:** ~1.1-1.2x by avoiding reserve stalls.  
**Risk:** Low.

#### 9.7 Tokenization Cache
**File:** `src/llama-mas.cpp`  
**Action:** Add an `std::unordered_map<std::string, std::vector<llama_token>>` cache in `llama_mas_context`.  
**Expected Impact:** Eliminates repeated tokenization of system prompts and role prefixes.  
**Risk:** Negligible.

---

### P3 — Nice to Have

#### 9.8 Adapter Weight Deduplication
**File:** `src/llama-mas.cpp`  
**Action:** Hash adapter weights and share GPU buffers across agents with identical weights.  
**Expected Impact:** Reduces GPU memory pressure.  
**Risk:** Low.

#### 9.9 KV Cache Defragmentation
**File:** `src/llama-kv-cache.cpp`  
**Action:** Implement periodic compaction of KV buffers when fragmentation exceeds a threshold.  
**Expected Impact:** Improves long-horizon `find_slot` performance.  
**Risk:** Medium complexity.

#### 9.10 Async Tokenization
**File:** `src/llama-mas.cpp`  
**Action:** Tokenize agent N+1's prompt while agent N is decoding.  
**Expected Impact:** Hides tokenization latency in decode time.  
**Risk:** Requires thread safety in vocab access.

---

## 10. Measurement Plan

To validate improvements, add these metrics to the server logs or a `/metrics` endpoint:

1. **Decode batch size histogram** — confirm batched decode is working
2. **Graph reuse rate** — % of decodes that reuse the previous graph
3. **sched_reserve time** — ms spent in `sched_reserve`
4. **Sampler backend fallback count** — how often CPU sampling is used
5. **Adapter H2D/D2H bytes** — PCIe traffic from MAS adapters
6. **KV cache shift time** — ms spent in K-shift per decode
7. **Tokenization time** — ms per `llama_tokenize` call

---

## Appendix A: Files Audited

| File | Lines | Focus |
|------|-------|-------|
| `src/llama-mas.cpp` | 1-1360 | MAS orchestration, adapters, encode/decode |
| `src/llama-context.cpp` | 1600-2350 | Decode pipeline, graph building, scheduling |
| `src/llama-kv-cache.cpp` | 518-1358 | KV cache metadata, shifts, copies |
| `src/llama-sampler.cpp` | 1-1200 | Sampling backend/CPU fallback |
| `src/llama-graph.cpp` | 2100-2400 | Attention graph building |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 2660-2991 | CUDA op dispatch |
| `ggml/src/ggml-cuda/set-rows.cu` | 530-659 | KV cache SET_ROWS on CUDA |

---

*End of audit.*
