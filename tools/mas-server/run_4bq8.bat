@echo off
REM Qwen3-4B Q8_0 (official base) on :9029 - isolates QUANTIZATION.
REM n_ctx 16384 (not 8192): the per-agent KV quota is n_ctx/n_agents_max,
REM so 8192/8 = 1024 positions. A chain inherits KV and then speaks up to
REM 768 tokens, which overflowed 1024 and returned HTTP 500 mid-benchmark.
REM 16384/8 = 2048 per agent. KV cost ~2.4 GB + model 4.7 GB fits 8 GB.
REM Same base family as the 8B Q4_K_M already measured, and same 4B size as the
REM -Instruct Q4_K_M, so this separates quant loss from size and tuning.
cd /d C:\+run\GPT_WEB\llama-tqp-mas-cpp
set GGML_CUDA_FORCE_GRAPHS=0
build-codex\bin\mas-server.exe -m E:\LMStudio\models\Qwen3-4B-Q8_0.gguf -p 9029 --n-ctx 16384 --n-batch 2048 --n-gpu-layers 99 --n-agents-max 8 > server_4bq8.log 2>&1
echo SERVER_EXIT=%ERRORLEVEL% >> server_4bq8.log