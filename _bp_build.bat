@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cmake --build "C:\+run\GPT_WEB\llama-tqp-mas-cpp\build-probe" --target mas-server latent-bench --config Debug -j16 > "C:\+run\GPT_WEB\llama-tqp-mas-cpp\bp_build.log" 2>&1
echo EXIT=%ERRORLEVEL% >> "C:\+run\GPT_WEB\llama-tqp-mas-cpp\bp_build.log"