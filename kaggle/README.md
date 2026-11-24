# llama-tqp-mas-tpu

Source snapshot of the local `llama-tqp-mas-cpp` tree (llama.cpp + TurboQuant KV cache + latent-MAS overlay),
plus a self-contained Kaggle TPU v5e-8 smoke test.

This is a source-only snapshot: build directories, virtualenvs, logs, models and binaries were excluded.

## Kaggle smoke test (Phase 1)

1. Create a Kaggle notebook with accelerator **TPU v5e-8** (identity verification required by Kaggle).
2. Import `kaggle/tpu_probe.ipynb` from this repo, or clone this repo in a cell and run `kaggle/probe_tpu.py`.
3. The probe checks that a TPU is attached, downloads xbill9's prebuilt `llama-server` + `libggml-tpu.so`
   (release v0.2.0), serves a small Q4_0 model on `TPU0`, and prints decode tok/s.

The probe deliberately uses the **stock** prebuilt backend. It answers one question first: does this Kaggle
account actually get a TPU v5e-8 and does an llama.cpp graph run on it? The TurboQuant/MAS build comes next.

## Local tree contents

- `tools/mas-server/` - mas-server overlay (C++ + test harness)
- `scripts/llama-tqp-mas.sh` - fetch/patch/sync workflow for the TurboQuant + MAS tree
- `scripts/sync_upstream.sh`, `turboquant-patches/` - upstream porting material
