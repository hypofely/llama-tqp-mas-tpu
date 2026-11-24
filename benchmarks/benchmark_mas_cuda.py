#!/usr/bin/env python3
"""
MAS CUDA Benchmark — standalone ctypes script.

Compares batched vs per-agent decode, CUDA adapters vs CPU, and backend sampling.
Does NOT depend on llama_turboquant.py.

Usage:
    python benchmarks/benchmark_mas_cuda.py -m model.gguf [-n 32] [-a 4]
"""

import ctypes
import os
import sys
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Find and load llama.dll
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).parent.parent
_candidates = [
    _repo_root / "build-codex" / "bin" / "llama.dll",
    _repo_root / "build" / "bin" / "Release" / "llama.dll",
]
_DLL_PATH = None
for c in _candidates:
    if c.exists():
        _DLL_PATH = c
        break
if _DLL_PATH is None:
    raise FileNotFoundError(f"llama.dll not found; checked: {_candidates}")

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_DLL_PATH.parent))
    for cuda_bin in [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    ]:
        if os.path.isdir(cuda_bin):
            os.add_dll_directory(cuda_bin)

_lib = ctypes.CDLL(str(_DLL_PATH))

# ---------------------------------------------------------------------------
# C function signatures
# ---------------------------------------------------------------------------
llama_token = ctypes.c_int32
llama_seq_id = ctypes.c_int32

_lib.llama_backend_init.argtypes = []
_lib.llama_backend_init.restype = None

_lib.llama_model_default_params.argtypes = []
_lib.llama_model_default_params.restype = ctypes.c_void_p

_lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
_lib.llama_model_load_from_file.restype = ctypes.c_void_p

_lib.llama_model_free.argtypes = [ctypes.c_void_p]
_lib.llama_model_free.restype = None

_lib.llama_context_default_params.argtypes = []
_lib.llama_context_default_params.restype = ctypes.c_void_p

_lib.llama_mas_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
_lib.llama_mas_init.restype = ctypes.c_void_p

_lib.llama_mas_free.argtypes = [ctypes.c_void_p]
_lib.llama_mas_free.restype = None

_lib.llama_mas_add_agent_flat.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, llama_seq_id,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
    ctypes.c_float, ctypes.c_int32,
]
_lib.llama_mas_add_agent_flat.restype = ctypes.c_int32

_lib.llama_mas_set_cuda_adapters.argtypes = [ctypes.c_void_p, ctypes.c_bool]
_lib.llama_mas_set_cuda_adapters.restype = None
_lib.llama_mas_get_cuda_adapters.argtypes = [ctypes.c_void_p]
_lib.llama_mas_get_cuda_adapters.restype = ctypes.c_bool

_lib.llama_mas_set_batched_decode.argtypes = [ctypes.c_void_p, ctypes.c_bool]
_lib.llama_mas_set_batched_decode.restype = None
_lib.llama_mas_get_batched_decode.argtypes = [ctypes.c_void_p]
_lib.llama_mas_get_batched_decode.restype = ctypes.c_bool

_lib.llama_mas_set_backend_sampling.argtypes = [ctypes.c_void_p, ctypes.c_bool]
_lib.llama_mas_set_backend_sampling.restype = None
_lib.llama_mas_get_backend_sampling.argtypes = [ctypes.c_void_p]
_lib.llama_mas_get_backend_sampling.restype = ctypes.c_bool

_lib.llama_mas_run.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32]
_lib.llama_mas_run.restype = ctypes.c_int32

_lib.llama_mas_get_tokens.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_size_t)]
_lib.llama_mas_get_tokens.restype = ctypes.POINTER(llama_token)

_lib.llama_backend_free.argtypes = []
_lib.llama_backend_free.restype = None

_lib.ggml_backend_load_all.argtypes = []
_lib.ggml_backend_load_all.restype = None

# ---------------------------------------------------------------------------
# Struct wrappers
# ---------------------------------------------------------------------------
class LlamaModelParams(ctypes.Structure):
    _fields_ = [
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("rpc_servers", ctypes.c_char_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
    ]

class LlamaContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("logits_all", ctypes.c_bool),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("op_offload_0", ctypes.c_bool),
    ]

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_bench(model_path: str, n_agents: int, n_predict: int, ngl: int,
              batched: bool, cuda_adapters: bool, backend_sampling: bool):
    _lib.ggml_backend_load_all()
    _lib.llama_backend_init()

    # Model params
    mparams = LlamaModelParams()
    ctypes.memset(ctypes.byref(mparams), 0, ctypes.sizeof(mparams))
    mparams.n_gpu_layers = ngl
    mparams.use_mmap = True

    model = _lib.llama_model_load_from_file(model_path.encode("utf-8"), ctypes.byref(mparams))
    if not model:
        raise RuntimeError("Failed to load model")

    # Context params
    cparams = LlamaContextParams()
    ctypes.memset(ctypes.byref(cparams), 0, ctypes.sizeof(cparams))
    cparams.n_ctx = 4096
    cparams.n_batch = 2048
    cparams.n_ubatch = 2048
    cparams.n_seq_max = n_agents
    cparams.no_perf = True

    mas = _lib.llama_mas_init(model, ctypes.byref(cparams), n_agents)
    if not mas:
        raise RuntimeError("llama_mas_init failed")

    # Apply toggles
    _lib.llama_mas_set_batched_decode(mas, batched)
    _lib.llama_mas_set_cuda_adapters(mas, cuda_adapters)
    _lib.llama_mas_set_backend_sampling(mas, backend_sampling)

    roles = ["planner", "coder", "reviewer", "tester"]
    for i in range(n_agents):
        role = roles[i] if i < len(roles) else f"agent_{i}"
        _lib.llama_mas_add_agent_flat(
            mas, role.encode("utf-8"), i,
            0,  # LLAMA_MAS_ADAPTER_NONE
            n_predict, 0.7, 0.9, 40,
        )

    task = b"Write a Python function to sort a list of integers."

    # Warm-up
    _lib.llama_mas_run(mas, task, 2, 0)  # LLAMA_MAS_STRATEGY_BROADCAST = 0

    # Timed run
    t0 = time.perf_counter()
    ret = _lib.llama_mas_run(mas, task, n_predict, 0)
    decode_ms = (time.perf_counter() - t0) * 1000.0

    # Count tokens generated
    total_tokens = 0
    for i in range(n_agents):
        count = ctypes.c_size_t(0)
        toks = _lib.llama_mas_get_tokens(mas, i, ctypes.byref(count))
        if toks and count.value:
            total_tokens += count.value

    _lib.llama_mas_free(mas)
    _lib.llama_model_free(model)

    tok_per_sec = total_tokens / (decode_ms / 1000.0) if decode_ms > 0 else 0.0
    return {
        "decode_ms": decode_ms,
        "total_tokens": total_tokens,
        "tok_per_sec": tok_per_sec,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MAS CUDA Benchmark")
    parser.add_argument("-m", "--model", required=True, help="Path to GGUF model")
    parser.add_argument("-n", "--tokens", type=int, default=32, help="Tokens per agent")
    parser.add_argument("-a", "--agents", type=int, default=4, help="Number of agents")
    parser.add_argument("--ngl", type=int, default=99, help="GPU layers")
    args = parser.parse_args()

    print("=" * 60)
    print("MAS CUDA Benchmark")
    print(f"Model: {args.model}")
    print(f"Agents: {args.agents}  Tokens: {args.tokens}  GPU layers: {args.ngl}")
    print("=" * 60)

    configs = [
        ("CUDA defaults (all on)", True, True, True),
        ("No batched decode", False, True, True),
        ("No backend sampling", True, True, False),
        ("All CPU fallback", False, False, False),
    ]

    results = []
    for name, batched, cuda_adapters, backend_sampling in configs:
        print(f"\n[RUN] {name}")
        try:
            r = run_bench(args.model, args.agents, args.tokens, args.ngl,
                          batched, cuda_adapters, backend_sampling)
            print(f"  Decode: {r['decode_ms']:.1f} ms | Tokens: {r['total_tokens']} | Speed: {r['tok_per_sec']:.2f} tok/s")
            results.append((name, r))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, None))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<30} {'Decode(ms)':<12} {'Tok/Sec':<12} {'Speedup':<10}")
    print("-" * 60)
    baseline = results[0][1]["tok_per_sec"] if results[0][1] else 1.0
    for name, r in results:
        if r:
            speedup = r["tok_per_sec"] / baseline if baseline > 0 else 0.0
            print(f"{name:<30} {r['decode_ms']:<12.1f} {r['tok_per_sec']:<12.2f} {speedup:<10.2f}x")
        else:
            print(f"{name:<30} {'ERROR':<12}")
    print("=" * 60)


if __name__ == "__main__":
    main()
