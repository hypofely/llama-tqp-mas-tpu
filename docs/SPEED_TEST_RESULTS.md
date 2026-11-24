# Speed Test Results

## Test Configuration
- Model: `E:\LMStudio\models\unsloth\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-Q4_K_S.gguf`
- Flags used: `--flash-attn on --no-mmap`
- GPU: CUDA with all layers on GPU
- Test date: 2026-06-02

## Fixes Applied

### 1. Fixed mas-server cparams (2026-06-02)
- `load_model()`: Added `flash_attn_type`, `offload_kqv` to context params
- `handle_mas_run()`: Added missing `type_k`, `type_v`, `flash_attn_type`, `offload_kqv` to MAS context params
- These were causing MAS context to use default F16 KV cache and auto flash attention

### 2. Optimized llama_generate_single batch reuse (2026-06-02)
- Changed from allocating/freeing `llama_batch` for every token to reusing a pre-allocated batch
- Speed improvement: 17 tok/sec -> 24 tok/sec (~40% faster)

## Benchmark Results

| Quant Type | Speed (tok/sec) | Status | Notes |
|------------|----------------|--------|-------|
| **q4_k_s** | **28.48** | **WORKS** | Fastest - native CUDA kernels |
| turbo3 | 23.8 | WORKS | WHT rotation adds ~5ms/token overhead |
| turbo4 | 23.78 | WORKS | WHT rotation adds ~5ms/token overhead |
| **planar3** | **CRASH** | **BROKEN** | Missing CUDA flash attention dequantization |
| **iso3** | **CRASH** | **BROKEN** | Missing CUDA flash attention dequantization |

## Root Cause Analysis

### Planar3/Iso3 Crash
RotorQuant types (planar3, iso3) lack CUDA flash attention dequantization kernels in `fattn-common.cuh`. The integration was attempted but reverted due to `extern __constant__` linkage errors. Without these kernels, the CUDA flash attention code path crashes during graph reserve.

### Performance Bottleneck (TurboQuant vs Q4)
Q4_K_S is ~20% faster than TurboQuant (28.5 vs 24 tok/sec). This indicates:
1. TurboQuant's WHT rotation + dequantization in flash attention adds ~5ms per token
2. Native Q4_K_S uses highly optimized CUDA kernels without custom rotation
3. The batch reuse fix recovered significant overhead (17->24 tok/sec)

## To Fix RotorQuant CUDA

Following files need modification per `CUDA_FLASH_ATTENTION_INTEGRATION.md`:
- `ggml/src/ggml-cuda/fattn-common.cuh` - Add planar/iso dequantization functions
- `ggml/src/ggml-cuda/fattn-vec.cuh` - Update type classification
- `ggml/src/ggml-cuda/CMakeLists.txt` - Add template instances

The main blocker is the `extern __constant__` linkage issue for rotation matrices.

## Files Modified

- `tools/mas-server/mas-server.cpp` - Fixed cparams handling
- `src/llama-mas.cpp` - Optimized batch reuse in generation loop
