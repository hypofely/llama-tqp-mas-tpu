# Changes Summary — MAS C++ Implementation Fix & Audit

**Date**: 2026-06-02

---

## Critical Bug Fixed: Corrupted `llama-mas.cpp`

The C++ source file `src/llama-mas.cpp` was accidentally corrupted — several function bodies were either missing or pasted into the wrong functions:

- **`llama_mas_decode_step`** was **completely missing** (declared in header, called 3 times, but no body)
- **`llama_mas_run`** had **no body** — the CHAIN/DTAS/BROADCAST logic was accidentally inside `llama_mas_get_text`
- **`llama_mas_get_text`** contained ~300 lines of generation logic instead of simple detokenization
- **`llama_mas_free_text`** was missing
- **`llama_mas_kv_clone`** was missing

**Impact**: You were likely running a stale pre-compiled DLL. The corrupted source meant no one could rebuild the DLL with the MAS code included.

**Fix**: Restored all five functions with correct implementations. Rebuilt `build-codex/bin/llama.dll` successfully.

---

## Correctness Fixes

### 1. Planner result now passed to executor

**File**: `python/recursive_mas_controller_api.py`

**Before**: `_run_mas_native` used strategy `"role"` (isolated agents), so each agent only saw the raw task text. The planner's output was never appended to the executor's prompt.

**After**: Multi-agent runs now automatically use `LLAMA_MAS_STRATEGY_CHAIN`, which feeds each agent's full output to the next agent with the instruction:

```
Your job is to ACT ON the above output. Do NOT create a new plan or repeat the previous agent's work.
```

### 2. Agent instructions / system prompts now passed

**File**: `python/recursive_mas_controller_api.py` + `python/llama_turboquant.py`

**Before**: `req.system_prompts` was completely ignored in the native MAS path. There was no Python method to call `llama_mas_set_agent_prompt`.

**After**:
- Added `set_agent_prompt(role, prompt)` to `LlamaMasContext`
- `_run_mas_native` now calls it for every agent, combining `system_prompts[role]` with the task text

### 3. Layer graphs now wired end-to-end

**File**: `python/recursive_mas_controller_api.py` + `python/llama_turboquant.py`

**Before**: The status docs claimed layer graphs were wired, but:
- No `LayerGraphContext` Python wrapper existed
- No `agent_layer_graphs` field in `MasRunRequest`
- `_run_mas_native` never called `set_agent_layer_graph`

**After**:
- Added `LlamaMasContext.set_agent_layer_graph(role, json_str)` which creates a C++ layer graph from JSON and attaches it to the agent
- Added `agent_layer_graphs: Optional[Dict[str, str]]` to `MasRunRequest`
- `_run_mas_native` reads per-role layer graphs from the request and attaches them

### 4. Async MAS now supports native mode

**File**: `python/recursive_mas_controller_api.py`

**Before**: `POST /mas/run-async` hardcoded `_run_mas_sync` (Python MAS) regardless of `use_native_mas` flag.

**After**: When `use_native_mas=true`, the async background thread calls `_run_mas_native` instead.

---

## New Python APIs Exposed

| API | File | Purpose |
|-----|------|---------|
| `LlamaMasContext.set_agent_prompt(role, prompt)` | `llama_turboquant.py` | Per-agent system/instruction text |
| `LlamaMasContext.set_agent_layer_graph(role, json)` | `llama_turboquant.py` | Per-agent dynamic layer subset |
| `LlamaMasContext.encode(prompt)` | `llama_turboquant.py` | Returns numpy array `[n_tokens, n_embd]` of last-layer hidden states |
| `LlamaMasContext.get_hidden_state_tensor(layer=-1)` | `llama_turboquant.py` | Returns GPU-resident `ggml_tensor*` address (for CUDA adapter kernels) |
| `agent_layer_graphs` field | `recursive_mas_controller_api.py` | Per-role JSON layer graph in `/mas/run` request |
| `LLAMA_MAS_STRATEGY_DTAS` constant | `llama_turboquant.py` | Strategy enum value 3 |

---

## Speed Improvements

With `llama_mas_decode_step` restored, the native MAS engine now performs **true batched multi-agent decode**:

- **BROADCAST / ROLE strategies**: All agents' prompt tokens are fed in a single `llama_decode` batch, then generation proceeds with one batched decode step per token across all agents
- **CHAIN strategy**: Agents run sequentially, but each agent's decode uses the full batch size (1 token per agent)
- **Layer graph grouping**: Agents with the same layer graph share a decode call; only different graphs trigger separate decodes

This eliminates the Python-level per-token synchronization overhead that was causing slowdown.

**Note**: The largest remaining speed bottleneck is **GPU→CPU hidden state copy** on every decode. True 2.5–3× speedup requires Phase 3 (GPU-resident hidden states with in-graph adapter compute), which is a larger architectural project.

---

## Build Verification

`build-codex/bin/llama.dll` was successfully rebuilt at **2026-06-02 03:07:40** with all MAS code included.

The Python wrapper now loads `build-codex/bin/llama.dll` **first** in the candidate list, ensuring the updated DLL is always used.

---

## Remaining Architectural Work (Not Bugs)

These are NOT bugs — they are advanced features from the architecture plan that were always intended for future phases:

### Phase 2: In-Graph Adapter Compute
- **Status**: CUDA adapter kernels exist and compile, but are **never called** during generation
- **Why**: Adapters in this codebase are designed for **inter-agent hidden state communication**, not for modifying logits during token generation. The native MAS uses text/KV-based communication (CHAIN/DTAS). Adding in-graph adapter compute requires modifying the transformer forward pass to inject MLP adapters between layers.

### Phase 3: GPU-Resident Hidden States
- **Status**: C API `llama_get_hidden_state_tensor()` exists and is now exposed to Python
- **Why needed**: Currently `llama_get_hidden_states_layer()` copies GPU→CPU. For adapter kernels to consume hidden states without copy overhead, the tensor must stay on GPU.
- **Current workaround**: Use `get_hidden_state_tensor()` to get the GPU pointer and pass it directly to CUDA kernels from Python (advanced usage).

---

## Files Modified

1. `src/llama-mas.cpp` — Restored missing functions
2. `python/llama_turboquant.py` — Added `set_agent_prompt`, `set_agent_layer_graph`, `encode`, `get_hidden_state_tensor`, DLL path fix, `LLAMA_MAS_STRATEGY_DTAS`
3. `python/recursive_mas_controller_api.py` — Fixed `_run_mas_native` prompts/strategy, added `agent_layer_graphs`, fixed async native path
4. `docs/TRUE_IMPLEMENTATION_STATUS.md` — Comprehensive audit of implemented vs claimed vs missing features

---

## How to Deploy

1. Restart the MAS server using your batch script:
   ```bat
   c:\+run\GPT_WEB\scripts\start-mas-server-mas-cpp.bat
   ```

2. The server will now automatically load the rebuilt `build-codex/bin/llama.dll`.

3. Send a `/mas/run` request with:
   - `use_native_mas: true`
   - `roles: ["planner", "executor"]`
   - `system_prompts: {"planner": "...", "executor": "..."}`
   - (optional) `agent_layer_graphs: {"planner": "{\"layers\":...", ...}`

4. The planner's output will now flow to the executor, and each agent will receive its specific instructions.
