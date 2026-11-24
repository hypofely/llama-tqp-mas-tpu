# Rebuilding TurboQuant llama.cpp for MAS

This folder is a **git worktree** of the TurboQuant+ fork at commit `4d24ad8`
(tag `tqp-v0.1.1`, llama.cpp base `b8979`, April 2026). It is the known-working
source for Gemma-4 E4B + `turbo4` KV cache on CUDA.

## Verified working configurations

| Model                        | type_k   | type_v   | KV size   |
|------------------------------|----------|----------|-----------|
| Gemma-4 E4B-it Q4_K_S, n_ctx 8192 | turbo4 | turbo4 | 119 MiB (vs 1406 MiB q4_0) |
| Qwen3.6-35B-A3B (prebuilt tested)  | turbo4 | turbo3 | similar reduction |

## How `mas_server.py` finds this build

```text
mas_server.py  →  ./llama-tqp/python/llama_turboquant.py
                     │
                     ├─ loads ./llama-tqp/build/bin/Release/llama.dll
                     │  (compiled from the C/C++ source in this worktree)
                     │
                     └─ ctypes structs match include/llama.h at 4d24ad8
                        (see python/ABI-NOTES.md before swapping DLLs)
```

## To pull a newer TurboQuant release

The fork is the upstream-of-record for TurboQuant:
**https://github.com/TheTom/llama-cpp-turboquant**

Releases are tagged `tqp-vX.Y.Z`. The fork maintainers handle upstream llama.cpp
syncs (b8979 → b9418 → …) and resolve all merge conflicts, so you do **not**
maintain a patch series against raw `ggml-org/llama.cpp` yourself.

### Update procedure

```powershell
# 1. Fetch latest from the fork's local mirror
cd C:\+run\llama-cpp-turboquant
git fetch origin

# 2. Pick a release tag (or HEAD if you want bleeding edge)
git tag -l 'tqp-v*' | Select-Object -Last 5

# 3. Replace the worktree pointer (this folder)
cd E:\LMStudio\llama-tqp
git fetch C:\+run\llama-cpp-turboquant
git checkout <new-tag-or-commit>

# 4. Adapt python/llama_turboquant.py ctypes structs if the ABI changed.
#    See python/ABI-NOTES.md.

# 5. Rebuild (10-25 min on a GTX 1080, single arch)
cmake -S . -B build -G "Visual Studio 16 2019" -A x64 `
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61-real `
      -DBUILD_SHARED_LIBS=ON `
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON
cmake --build build --config Release --target llama --parallel 8
```

For a different GPU, change `CMAKE_CUDA_ARCHITECTURES`:

| GPU family         | arch flag         |
|--------------------|-------------------|
| Pascal (GTX 10x0)  | `61-real`         |
| Turing (RTX 20x0)  | `75-real`         |
| Ampere (RTX 30x0)  | `86-real`         |
| Ada (RTX 40x0)     | `89-real`         |
| Hopper (H100)      | `90-real`         |
| Multi-arch         | `61-real;75-real;86-real;89-real` |

## Why not patch raw upstream?

`b8979..tqp-v0.1.1` is a 14 MB diff covering 649 files — most of it is upstream
churn, not turbo logic. Reproducing what `TheTom/llama-cpp-turboquant` does
(merge upstream, resolve conflicts, fix kernels per architecture, retest on
~7 model families) is a maintenance job, not a patch. **Track the fork's
release tags instead.**

The fork's branch history shows the work it already does for you:

```text
2b61ea24e  Merge upstream b9190 into feature/turboquant-kv-cache
e30bbcfe5  merge: upstream/master @ 4f13cb742 (b9190)
69d8e4be4  Merge fix/cve-2026-21869-n-discard-clamp
4f331667d  Merge fix/issue-81-nix-spirv-dupe
e0954d1b9  Merge fix/enable-attn-rot-by-default
0fa61d724  Merge pr53-auto-asymmetric
cde3e1ad4  Merge fix/issue-78-gqa-reshape
…
```

That's the value of using the fork.
