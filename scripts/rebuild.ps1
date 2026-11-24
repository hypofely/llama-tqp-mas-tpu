#requires -Version 5
<#
.SYNOPSIS
    Rebuild llama.dll + ggml-cuda.dll for the MAS server.

.DESCRIPTION
    Wraps the cmake configure + build steps for c:\+run\GPT_WEB\llama-tqp-mas-cpp.
    Defaults to GTX 1080 (compute_61). Override with -CudaArch.

.PARAMETER CudaArch
    CMAKE_CUDA_ARCHITECTURES value. Examples:
      61-real          GTX 1080
      75-real          RTX 2060/2070/2080
      86-real          RTX 30x0
      89-real          RTX 40x0
      "61-real;86-real" Multi-arch fat binary

.PARAMETER Clean
    Wipe build/ before configuring. Use after a git checkout to a new commit.

.PARAMETER Generator
    Visual Studio generator. Default: "Visual Studio 16 2019".

.EXAMPLE
    .\scripts\rebuild.ps1
    Default rebuild for GTX 1080.

.EXAMPLE
    .\scripts\rebuild.ps1 -Clean -CudaArch 86-real
    Clean rebuild for RTX 3090.
#>
param(
    [string]$CudaArch = "61-real",
    [switch]$Clean,
    [string]$Generator = "Visual Studio 16 2019"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root "build"

Write-Host "TurboQuant llama.cpp rebuild" -ForegroundColor Cyan
Write-Host "  Source : $root"
Write-Host "  Build  : $build"
Write-Host "  Arch   : $CudaArch"
Write-Host ""

if ($Clean -and (Test-Path $build)) {
    Write-Host "Cleaning $build ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $build
}

Write-Host "Configuring ..." -ForegroundColor Cyan
cmake -S $root -B $build -G $Generator -A x64 `
    -DGGML_CUDA=ON `
    -DCMAKE_CUDA_ARCHITECTURES=$CudaArch `
    -DBUILD_SHARED_LIBS=ON `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_BUILD_TOOLS=ON
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

Write-Host "Building llama target (this takes 10-25 min) ..." -ForegroundColor Cyan
cmake --build $build --config Release --target llama --parallel 8
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

$dll = Join-Path $build "bin\Release\llama.dll"
if (-not (Test-Path $dll)) { throw "Expected DLL not found: $dll" }

Write-Host ""
Write-Host "Build OK." -ForegroundColor Green
Get-ChildItem (Join-Path $build "bin\Release") -Filter "*.dll" |
    Select-Object Name, @{N="MB"; E={[math]::Round($_.Length / 1MB, 1)}}, LastWriteTime |
    Format-Table -AutoSize

Write-Host "Smoke test command:" -ForegroundColor Cyan
Write-Host '  $env:MAS_DEFAULT_MODEL_PATH = "<path to gguf>"'
Write-Host '  .\start_mas_server.bat'
