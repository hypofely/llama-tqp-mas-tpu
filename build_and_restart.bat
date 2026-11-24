@echo off
setlocal

set GGML_CUDA_FORCE_GRAPHS=1
set MAS_DIR=C:\+run\GPT_WEB\llama-tqp-mas-cpp
set BUILD_DIR=%MAS_DIR%\build-codex

echo Building MAS server...
echo.

:: Setup VS build environment (includes Windows SDK)
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" x64

:: Touch source to force ninja rebuild
powershell -NoProfile -Command "(Get-Item '%MAS_DIR%\src\llama-mas.cpp').LastWriteTime = Get-Date"

:: Build only mas-server
cmake --build "%BUILD_DIR%" --target mas-server --config Release -j 4

if %ERRORLEVEL% EQU 0 (
    echo.
    echo BUILD SUCCESS
) else (
    echo.
    echo BUILD FAILED
)

endlocal
