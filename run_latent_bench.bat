@echo off
set GGML_CUDA_FORCE_GRAPHS=1
cd /d %~dp0build-codex\bin
.\latent-bench.exe %*
