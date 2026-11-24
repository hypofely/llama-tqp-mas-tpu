# Current Status — MAS Multi-Model & Hidden State Transfer

## What Works

### 1. Model Swapping (`POST /config/model`)
- Swap to a different GGUF at runtime
- Agent metadata (roles, UUIDs, params) preserved across swaps
- KV cache serialized to CPU RAM on swap-out, restored on swap-back
- Old model state cached in RAM (`g_state.state_cache`) for fast switch-back

### 2. Agent Input Embeddings
- `llama_mas_set_agent_input_embeddings()` — feed float vectors as `batch.embd`
- Agents with embeddings skip tokenization entirely
- Works in both CHAIN and BROADCAST strategies
- HTTP: `POST /v1/agents/{id}/embeddings`

### 3. Hidden State Capture
- `llama_mas_get_agent_hidden_state()` — last token, last layer
- `llama_mas_set_output_hidden_states()` — enable hidden state extraction
- `llama_mas_set_stop_after_prompt()` — stop after prompt decode (prevents generation from overwriting hidden state)
- HTTP: `GET /v1/agents/{id}/hidden-state`

### 4. Cross-Agent Projection (CrossLink)
- `llama_mas_project_agent_to_embeddings()` — captures hidden state, applies source agent's CrossLink adapter, sets as dst embeddings
- HTTP: `POST /v1/agents/{src}/project-to/{dst}`
- Both CPU (`llama_mas_adapter_cross_forward`) and CUDA (`ggml_cuda_mas_cross_forward`) implementations exist

### 5. CrossLink Adapter Architecture
```
LayerNorm(source) -> Linear(in_dim -> hidden_dim) -> GELU -> Linear(hidden_dim -> out_dim)
  -> residual_proj(in_dim -> out_dim) -> add -> LayerNorm(target)
```
- `llama_mas_adapter_weights_init(n_embd, in_dim, out_dim, hidden_dim)` — create projection weights
- Default weights are identity-like (diagonal = 1.0) but **NOT TRAINED**
- For meaningful cross-model transfer, weights must be trained on aligned corpus

### 6. Chain + Broadcast Strategy (`LLAMA_MAS_STRATEGY_CHAIN_BROADCAST`)
- Agents are processed in sequential **waves** (groups)
- Within each wave, agents decode in parallel (broadcast)
- After each wave, the last agent's hidden state is captured and passed as **input embeddings** to all agents in the next wave
- Waves are explicit via `broadcast_waves` parameter, or auto-detected (each agent = its own wave)
- Auto-enables `capture_hidden_states` when used via HTTP API
- HTTP: `POST /mas/run` with `"strategy": "chain_broadcast"` and `"broadcast_waves": [[0], [1,2], [3]]`
- C API: `llama_mas_set_broadcast_waves(mas, waves, wave_sizes, n_waves)` before `llama_mas_run()`

**Example:** `"broadcast_waves": [[0], [1,2], [3]]` means:
- Wave 0: Agent 0 runs with task prompt
- Wave 1: Agents 1 and 2 run in parallel, receiving Agent 0's hidden state as embeddings
- Wave 2: Agent 3 runs, receiving the last agent of wave 1's hidden state as embeddings

### 7. Real-Time Token Streaming (`POST /mas/run/stream`)
- SSE (Server-Sent Events) endpoint streams every token as it is generated
- Events: `start`, `token`, `done`, `error`
- Multi-agent: tokens from different agents are interleaved in the stream (batched decode)
- Final `done` event includes full outputs + timing stats
- C API: `llama_mas_set_stream_callback(mas, cb, user_data)` before `llama_mas_run()`

**Example SSE events:**
```
data: {"event":"start","roles":["analyzer","critic"],"strategy":"broadcast"}
data: {"event":"token","agent":0,"token":123,"step":0}
data: {"event":"token","agent":1,"token":456,"step":1}
data: {"event":"done","outputs":{"analyzer":"...","critic":"..."},"timing":{...}}
```

### 8. Timing / Performance Logging
- Every `llama_mas_run()` now logs timing to stderr:
  `[MAS TIMING] total=...ms prefill=...ms (n=..., ... t/s) generation=...ms (n=..., ... t/s)`
- `llama_mas_get_timing()` returns structured stats:
  - `prefill_ms` / `generation_ms`
  - `n_prefill_tokens` / `n_generated_tokens`
  - `prefill_tokens_per_sec` / `tokens_per_sec`
- Timing is included in JSON responses from both `/mas/run` and `/mas/run/stream`

### 9. Trained CrossLink Weights (`train_crosslink.py`)
- Python training script collects (source_hidden, target_embedding) pairs via MAS server HTTP API
- Trains a PyTorch model matching the C++ CrossLink architecture exactly
- Saves weights in a custom binary format (`crosslink.bin`) with magic `LNCR`
- HTTP endpoint `POST /v1/agents/{id}/adapter/load` uploads trained weights to an agent
- C APIs for programmatic weight upload:
  - `llama_mas_adapter_weights_set_cross_ln_source/proj1/proj2/residual/ln_target`
  - `llama_mas_set_agent_adapter(mas, agent_idx, adapter_type, weights)`

**Usage:**
```bash
# 1. Start MAS server with source model
# 2. Train
python train_crosslink.py \
  --source-model gemma-4b.gguf \
  --target-model qwen-9b.gguf \
  --dataset demo_prompts.txt \
  --output crosslink_gemma_qwen.bin \
  --hidden-dim 4096 \
  --epochs 200

# 3. Load trained weights into agent 0
Invoke-WebRequest -Uri "http://127.0.0.1:8080/v1/agents/0/adapter/load" \
  -Method POST -Body '{"weights_file":"crosslink_gemma_qwen.bin","adapter_type":"cross"}'
```

### 10. Multi-Model Dimension Mismatch — Fixed
- CrossLink naturally handles mismatched embedding dimensions via `in_dim` / `out_dim`
- Training script automatically detects source and target dimensions from collected hidden states
- No manual dimension configuration needed

### 11. Layer-Graph KV Cache Preservation — Fixed
- `llama_layer_graph_forward` now tracks `kv_pos` across calls (starts at 0, advances by n_tokens)
- Uses explicit `seq_id` for KV storage instead of implicit default
- New APIs to control position and sequence:
  - `llama_layer_graph_reset_kv_pos(lg, pos)` — reset to start a fresh sequence
  - `llama_layer_graph_set_seq_id(lg, seq_id)` — use a specific sequence ID
- This allows running partial layer graphs in sequence (e.g. layers 0-10, then 11-20) while preserving the KV cache

## File Map

### Current (root level, actively modified)
- `src/llama-mas.cpp` — MAS runtime, decode, embeddings, projection, timing, streaming callback
- `src/llama-mas.h` — MAS APIs (broadcast waves, stream callback, timing, weight upload)
- `tools/mas-server/mas-server.cpp` — HTTP server, streaming endpoint, adapter load endpoint
- `src/llama-layer-graph.cpp/.h` — layer-graph forward with KV position tracking
- `src/llama-context.cpp` — KV cache compatible deserialization
- `src/llama-kv-cache.cpp/.h` — compatible state read helpers
- `_build_mas_server.bat` — build script
- `start_mas_server.bat` — launch script
- `train_crosslink.py` — CrossLink adapter training script (PyTorch)
- `demo_prompts.txt` — sample prompt dataset for training
- `test_v1_chat.py` — API test script
- `test_mas_cuda.cpp` — CUDA smoke test

### Archive (one-time / legacy)
- `archive/patches/` — 16 Python patch scripts (already applied)
- `archive/build-scripts/` — old build batch files
- `archive/logs/` — old build/runtime logs
- `archive/test-results/` — old JSON test outputs

## Quick Test Commands

```powershell
# Start server
start_mas_server.bat

# Create agents
Invoke-WebRequest -Uri "http://127.0.0.1:8080/v1/agents" -Method POST -Body '{"role":"encoder","max_tokens":32}'
Invoke-WebRequest -Uri "http://127.0.0.1:8080/v1/agents" -Method POST -Body '{"role":"decoder","max_tokens":32}'

# Run with hidden state capture, stop after prompt
Invoke-WebRequest -Uri "http://127.0.0.1:8080/mas/run" -Method POST -Body '{"task":"test","roles":["encoder","decoder"],"strategy":"chain","capture_hidden_states":true,"stop_after_prompt":true}'

# Project encoder hidden state to decoder embeddings
Invoke-WebRequest -Uri "http://127.0.0.1:8080/v1/agents/0/project-to/1" -Method POST

# Chain + Broadcast: analyzer solo, then critics 1+2 in parallel from analyzer's hidden state
Invoke-WebRequest -Uri "http://127.0.0.1:8080/mas/run" -Method POST -Body '{"task":"Analyze this claim","roles":["analyzer","critic1","critic2"],"strategy":"chain_broadcast","broadcast_waves":[[0],[1,2]],"max_tokens":32}'

# Streaming: real-time tokens via SSE (PowerShell example)
$req = [System.Net.HttpWebRequest]::Create("http://127.0.0.1:8080/mas/run/stream")
$req.Method = "POST"; $req.ContentType = "application/json"; $req.Accept = "text/event-stream"
$body = [System.Text.Encoding]::UTF8.GetBytes('{"task":"Hello","roles":["bot"],"max_tokens":16}')
$req.ContentLength = $body.Length; $s = $req.GetRequestStream(); $s.Write($body, 0, $body.Length); $s.Close()
$resp = $req.GetResponse(); $reader = New-Object System.IO.StreamReader($resp.GetResponseStream())
while ($null -ne ($line = $reader.ReadLine())) { Write-Host $line }
$reader.Close(); $resp.Close()

# Load trained CrossLink adapter weights into agent 0
Invoke-WebRequest -Uri "http://127.0.0.1:8080/v1/agents/0/adapter/load" -Method POST -Body '{"weights_file":"crosslink_gemma_qwen.bin","adapter_type":"cross"}'

# Swap models
Invoke-WebRequest -Uri "http://127.0.0.1:8080/config/model" -Method POST -Body '{"model_path":"...","keep_uuids":["..."]}'
```

## Session 2026-09-29 — hardening + verification

### Root-caused crashes (all verified fixed)
- **All Windows builds were compiling C++ WITHOUT `/EHsc`** (upstream bug: C4530 warnings present since forever). Every nlohmann `JSON_THROW` degraded to `std::abort()` → `0xc0000409` crashes in `test-json-partial`. Fix: `add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/EHsc>")` in root `CMakeLists.txt`. `test-json-partial` now passes; also removes the abort-on-malformed-JSON class of failures from mas-server.
- **`test-jinja` crash** = `std::stoi(str, base)` with base -663 (template `"test_string".int(chars=-663.5, true)`) → CRT invalid-parameter abort. Fix: `common/jinja/value.cpp` int() validates base ∈ {0} ∪ 2..36, throws `raised_exception` otherwise. Test: 305 tests / 1426 assertions / 0 failures (also covers the `-py` variant).
- **Jinja unbounded recursion** (CVE-2026-18581-adjacent): added `MAX_PARSE_DEPTH=512` + RAII `depth_guard` in `common/jinja/parser.cpp` (`parse_any()`, `parse_if_expression()`).

### Test-suite fixes
- `tests/test-quantize-fns.cpp` — fork's planar3/iso3 (3-bit) KV types now use the 3-bit tolerance table (they exceed the old 2-bit cap). Green.
- `tests/test-quant-type-selection.cpp` — `model_specs[]` trimmed to models that fit this fork's target hardware (GTX 1080 8 GB): `ggml-org/Qwen3-0.6B-GGUF` Q8_0 + `ggml-org/gemma-3-4b-it-GGUF` Q8_0. Deleted 10 oversized schemas from `tests/snapshots/`. Fully offline, 2/2 green.
- `tests/gguf-model-data.cpp` — optional `Authorization: Bearer $HF_TOKEN` on metadata fetch (gated repos), plus `LLAMA_GGUF_CACHE_DIR` env to redirect the metadata cache off a full C: drive (defaults: `E:\LMStudio\models\.gguf-headers`).

### mas-server runtime bugs found by the new test suite
- **SSE stream use-after-free** (`/mas/run/stream`): `delete st` in the chunked provider raced the detached generator thread's final `st->mtx` lock → Debug CRT "mutex destroyed while busy" modal. Fix: `st` is now `std::shared_ptr<mas_stream_state>`; raw pointer (`st.get()`) passed to `llama_mas_set_stream_callback`; nothing is deleted explicitly.
- **`/v1/agents` broken after any `/mas/run`**: auto-resize overwrote `n_agents_max` with the run's role count (a 1-role warmup permanently shrank 8 → 1, blocking further agent creation). Fix: auto-resize only ever grows (`std::max`), never shrinks below the configured `--n-agents-max`.

### New test suite
- `tools/mas-server/test_mas_latent.py` — 14 checks covering: health, warmup init, agent creation, chain+stop-after-prompt, hidden-state retrieval (n_embd vector, finite/nonzero), CrossLink `project-to`, embeddings feed (no tokenization), generation from latent embeddings, kv-clone endpoint, DTAS multi-agent spawn, `latent_mas` strategy, SSE streaming (`start`/`token`/`done`). **14/14 PASS** on the rebuilt server (stories15M, turbo4 K/V, CPU build; CUDA variant passes the original 7/7 suite).
- Original suite `tools/mas-server/test_mas_server.py`: **7/7 PASS**.

### Build automation
- `scripts/llama-tqp-mas.sh` — maintainer script: `fetch` (TurboQuant origin + ggml-org upstream, drift report), `merge` (origin, optional `--with-upstream`, abort-on-conflict), `build` (auto-detects VS via vswhere → vcvars; CUDA=ON when nvcc present; uses `build-cuda/` so the CPU `build/` test tree stays intact), `test` (subset + both server suites + latent-bench), `fresh <dir>` (reproduce the fork from scratch), `all`.
- Upstream remote repaired to `https://github.com/ggml-org/llama.cpp.git` (was a stale local mirror path).
