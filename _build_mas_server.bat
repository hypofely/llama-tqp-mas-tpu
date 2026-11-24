@echo off
set GGML_CUDA_FORCE_GRAPHS=1
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
:: Force rebuild of changed MAS source even if ninja cache is stale
powershell -NoProfile -Command "(Get-Item 'C:\+run\GPT_WEB\llama-tqp-mas-cpp\src\llama-mas.cpp').LastWriteTime = Get-Date"
cmake --build "C:\+run\GPT_WEB\llama-tqp-mas-cpp\build-codex" --target mas-server latent-bench --config Release -j16
