@echo off
REM Qwen3-8B Q4_K_M on :9026 for the MAS latent path.
REM Lives in the repo, NOT scratch: the Hermes scratch dir prunes entries idle
REM for 24h, which silently deleted the launcher and made every start fail with
REM exit 1 while the old server log looked like a model-load crash.
cd /d C:\+run\GPT_WEB\llama-tqp-mas-cpp
set GGML_CUDA_FORCE_GRAPHS=0
build-codex\bin\mas-server.exe -m E:\LMStudio\models\Qwen3-8B-Q4_K_M.gguf -p 9026 --n-ctx 8192 --n-batch 2048 --n-gpu-layers 99 --n-agents-max 8 > server_8b.log 2>&1
echo SERVER_EXIT=%ERRORLEVEL% >> server_8b.log