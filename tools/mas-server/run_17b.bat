@echo off
REM Qwen3-1.7B Q8_0 on :9024 — the regression-suite server.
REM Repo-resident on purpose: scratch is pruned after 24h idle.
cd /d C:\+run\GPT_WEB\llama-tqp-mas-cpp
set GGML_CUDA_FORCE_GRAPHS=0
build-codex\bin\mas-server.exe -m E:\LMStudio\models\Qwen3-1.7B-Q8_0.gguf -p 9024 --n-ctx 4096 --n-batch 2048 --n-gpu-layers 99 --n-agents-max 8 > server_lo.log 2>&1
echo SERVER_EXIT=%ERRORLEVEL% >> server_lo.log