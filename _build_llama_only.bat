@echo off
set GGML_CUDA_FORCE_GRAPHS=1
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
cd /d C:\+run\GPT_WEB\llama-tqp-mas-cpp
cmake --build build-codex --target llama --config Debug -j 8