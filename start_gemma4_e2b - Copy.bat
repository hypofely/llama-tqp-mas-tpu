@echo off
setlocal

:: Kill any stale server instances before starting
:: (prevents DLL lock errors during rebuild and old-code confusion)
taskkill /f /im mas-server.exe >nul 2>&1

:: Force-enable CUDA graphs on this GPU architecture
:: (otherwise ggml disables them and warns every decode)
set "GGML_CUDA_FORCE_GRAPHS=1"

set "MAS_DIR=C:\+run\GPT_WEB\llama-tqp-mas-cpp"
set "MODEL=E:\LMStudio\models\unsloth\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_S.gguf"
set "MMPROJ=E:\LMStudio\models\unsloth\gemma-4-E2B-it-GGUF\mmproj-F32.gguf"
set "BIN_DIR=%MAS_DIR%\build-codex\bin"

:: Ensure DLLs (ggml, CUDA) are on PATH
set "PATH=%BIN_DIR%;%PATH%"

:: RotorQuant 3-bit = planar3 (or iso3) KV cache quantization
set "ROTQUANT=planar3"

set "SERVER=%BIN_DIR%\mas-server.exe"

:: n_ctx controls the KV cache reservation. 8192 uses ~2-3 GB VRAM for a 4B model
:: with flash-attn + planar3. Lower to 2048 or 4096 if VRAM is tight.
:: Agents are NOT created at startup — they are created per-request by the client.
set "N_CTX=4096"

echo Starting MAS server...
echo   Model: %MODEL%
echo   MMProj: %MMPROJ%
echo   KV quant: %ROTQUANT%
echo   n_ctx: %N_CTX%
echo   Port: 8088
echo   Bin: %SERVER%
echo.

:: Note: mas-server does not currently support --mmproj argument
:: Vision models require mmproj support which may not be available in mas-server
:: This script starts the server with the text model only
"%SERVER%" -m "%MODEL%" -p 8088 --n-ctx %N_CTX% --n-gpu-layers 999 --flash-attn on --type-k %ROTQUANT% --type-v %ROTQUANT%

endlocal
