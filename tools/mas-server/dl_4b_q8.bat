@echo off
REM Official Qwen3-4B Q8_0 (base, has thinking mode) - isolates QUANTIZATION.
REM The local 4B is -Instruct-2507 Q4_K_M, which varies size AND tuning but not
REM quant. This one is the same base family as the 8B Q4_K_M we already measured.
cd /d E:\LMStudio\models
curl -L --retry 5 --retry-delay 5 -C - -o Qwen3-4B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf
echo DL_EXIT=%ERRORLEVEL%
dir Qwen3-4B-Q8_0.gguf