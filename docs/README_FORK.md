# llama-tqp-mas-cpp — TurboQuant × LatentMAS fork

Your custom llama.cpp: TheTom's **TurboQuant** fork (`feature/turboquant-kv-cache` —
turbo4 KV-cache types, CUDA graphs for Pascal, MoE cache, faster quantized kernels)
plus the local **MAS overlay** (multi-agent latent-space engine): hidden-state
capture, latent-embedding prompting (no tokenization), CrossLink cross-agent
projection, KV-clone agent spawn (DTAS), per-sequence KV handoff, layer-graph
forward, and the HTTP `mas-server`.

Everything is CUDA-first where it can be (GGML_CUDA + CUDA graphs; verified on a
GTX 1080 / Pascal, nvcc 12.6).

## Layout

| Path | What |
|---|---|
| `tools/mas-server/` | `mas-server.cpp` — HTTP server (OpenAI-ish `/v1/*` + `/mas/*`), SSE streaming |
| `src/llama-mas.*` | MAS runtime: agents, strategies (chain, broadcast, dtas, latent_mas), embeddings, projection, KV clone |
| `src/llama-layer-graph.*` | partial layer-graph forward with KV position tracking |
| `src/llama-kv-cache.*`, `src/llama-context.cpp` | KV cache state read/serialize helpers |
| `scripts/llama-tqp-mas.sh` | maintain: fetch/merge/build/test/fresh |
| `CURRENT_STATUS.md`, `CUSTOM_PATCHES.md`, `docs/LATENTMAS_INTEGRATION.md` | fork docs (yes, they're all in the root) |

## Day-to-day

```bash
# build (auto-detects VS via vswhere, CUDA if nvcc found; uses build-cuda/)
./scripts/llama-tqp-mas.sh build

# test everything (CPU test tree + both server suites + latent-bench)
./scripts/llama-tqp-mas.sh test

# update from TheTom / upstream llama.cpp (aborts cleanly on conflicts)
./scripts/llama-tqp-mas.sh fetch
./scripts/llama-tqp-mas.sh merge            # TurboQuant latest
./scripts/llama-tqp-mas.sh merge --with-upstream

# reproduce: latest llama.cpp (upstream) + TurboQuant + MAS overlay
./scripts/llama-tqp-mas.sh fresh /path/to/new-clone
```

Windows notes: run under git-bash; build trees are `build/` (CPU + tests, Release),
`build-codex/` (Debug CUDA), `build-cuda/` (script default). The root
`CMakeLists.txt` carries the `/EHsc` fix — Windows C++ builds without it silently
turn every JSON exception into a process abort.

## mas-server (the LatentMAS gateway)

```bash
# tiny model for CI-ish smoke
./build/bin/mas-server.exe -m build/tinyllamas/stories15M-q4_0.gguf -p 9000 \
    --n-ctx 512 --type-k turbo4 --type-v turbo4 --flash-attn on --n-agents-max 8

# real model (fit for an 8 GB GTX 1080)
./build-codex/bin/mas-server.exe -m "E:/LMStudio/models/Qwen3-0.6B-Q8_0.gguf" -p 9000 \
    --n-ctx 1024 --n-gpu-layers 99 --type-k turbo4 --type-v turbo4 \
    --flash-attn on --n-agents-max 8
```

Latent handoff in one curl sequence (capture → project → feed → generate):

```bash
# 1. run a prompt, capture hidden states, stop after the prompt
curl -s -X POST localhost:9000/mas/run -H 'Content-Type: application/json' \
  -d '{"task":"Write the word hello.","roles":["encoder"],"strategy":"chain",
       "capture_hidden_states":true,"stop_after_prompt":true,"max_tokens":8}'

# 2. hidden state of agent 0 (last token, last layer, n_embd floats)
curl -s localhost:9000/v1/agents/0/hidden-state

# 3. project hidden state into agent 1's embedding space (CrossLink)
curl -s -X POST localhost:9000/v1/agents/0/project-to/1

# 4. hand agent 1 a latent prompt (NO tokenization)
curl -s -X POST localhost:9000/v1/agents/1/embeddings -H 'Content-Type: application/json' \
  -d '{"embeddings":[...256 floats...],"n_embd":256}'
curl -s -X POST localhost:9000/mas/run -H 'Content-Type: application/json' \
  -d '{"task":"","roles":["decoder"],"strategy":"chain","max_tokens":8}'
```

Streaming: `POST /mas/run/stream` (SSE: `start`, `token*`, `done`).
Agent spawn with KV handoff: strategy `dtas` (first agent's KV is cloned for the
spawned agent, then batched decode).

## Tests (all verified 2026-09-29)

| Suite | Command | Result |
|---|---|---|
| ctest (CPU tree) | `ctest --test-dir build` | 41/42 (only `test-eval-callback` "Not Run" — it downloads a model) |
| Server API | `py -3.12 tools/mas-server/test_mas_server.py` | 7/7 (CUDA + Qwen3-0.6B: 7/7) |
| LatentMAS features | `py -3.12 tools/mas-server/test_mas_latent.py` | 14/14 (CPU + Qwen3-0.6B: 14/14) |
| latent-bench | `build-codex/bin/latent-bench.exe <model> "2 + 2 = ?" 48` | 4/4 sweeps |

Test models: `build/tinyllamas/stories15M-q4_0.gguf` (tiny), real model at
`E:\LMStudio\models\Qwen3-0.6B-Q8_0.gguf` (767 MB, fits 8 GB VRAM).