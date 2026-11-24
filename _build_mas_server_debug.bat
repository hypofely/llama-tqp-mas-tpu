@echo off
set GGML_CUDA_FORCE_GRAPHS=1
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cmake --build "C:\+run\GPT_WEB\llama-tqp-mas-cpp\build-codex" --target mas-server latent-bench --config Debug -j16
