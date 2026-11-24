@echo off
REM Qwen3-4B-Instruct-2507 Q4_K_M on :9028.
REM Repo-resident: the Hermes scratch dir prunes entries idle 24h.
REM NOTE: this is the -Instruct-2507 release, which has NO thinking mode (only the
REM April base Qwen3-4B does). It is also Q4_K_M, same quant class as the 8B, so
REM it varies SIZE and instruct-tuning but NOT quantization.
cd /d C:\+run\GPT_WEB\llama-tqp-mas-cpp
set GGML_CUDA_FORCE_GRAPHS=0
build-codex\bin\mas-server.exe -m E:\LMStudio\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf -p 9028 --n-ctx 16384 --n-batch 2048 --n-gpu-layers 99 --n-agents-max 8 > server_4b.log 2>&1
echo SERVER_EXIT=%ERRORLEVEL% >> server_4b.log