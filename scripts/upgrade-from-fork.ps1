#requires -Version 5
<#
.SYNOPSIS
    Pull a newer TurboQuant fork commit/tag into this worktree and rebuild.

.DESCRIPTION
    The fork at C:\+run\llama-cpp-turboquant tracks
    https://github.com/TheTom/llama-cpp-turboquant and handles all upstream
    llama.cpp syncs and merge conflicts for you. This script:

      1. Fetches the latest from the fork mirror.
      2. Lists recent tqp-v* tags.
      3. Checks out the requested ref into this worktree.
      4. Reminds you to verify python/ ctypes ABI against the new include/llama.h.
      5. Rebuilds.

    Run this whenever you want a newer upstream-aligned turboquant build.

.PARAMETER Ref
    Tag or commit to check out. Default: latest tqp-v* tag.

.PARAMETER ForkPath
    Path to the local fork mirror. Default: C:\+run\llama-cpp-turboquant.

.PARAMETER CudaArch
    Forwarded to rebuild.ps1.

.PARAMETER SkipBuild
    Do the checkout but skip the rebuild step.

.EXAMPLE
    .\scripts\upgrade-from-fork.ps1
    Latest tagged release, default GPU.

.EXAMPLE
    .\scripts\upgrade-from-fork.ps1 -Ref tqp-v0.2.0 -CudaArch 86-real
    Specific release, RTX 30x0.
#>
param(
    [string]$Ref,
    [string]$ForkPath = "C:\+run\llama-cpp-turboquant",
    [string]$CudaArch = "61-real",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $ForkPath)) {
    throw "Fork mirror not found at $ForkPath. Clone https://github.com/TheTom/llama-cpp-turboquant there first."
}

Push-Location $ForkPath
try {
    Write-Host "Fetching fork updates ..." -ForegroundColor Cyan
    git fetch --tags origin
    if ($LASTEXITCODE -ne 0) { throw "git fetch failed" }

    if (-not $Ref) {
        # Pick the latest tqp-v* tag by version sort
        $Ref = (git tag -l 'tqp-v*' | Sort-Object { [version]($_ -replace '^tqp-v','') } |
                Select-Object -Last 1)
        if (-not $Ref) { throw "No tqp-v* tags found in fork" }
        Write-Host "Auto-selected latest release: $Ref" -ForegroundColor Green
    }

    $tagSha = git rev-parse $Ref
    if ($LASTEXITCODE -ne 0) { throw "Unknown ref: $Ref" }
    Write-Host "Target: $Ref ($tagSha)" -ForegroundColor Cyan
} finally {
    Pop-Location
}

Push-Location $root
try {
    Write-Host "Checking out $Ref in worktree $root ..." -ForegroundColor Cyan
    # The worktree shares git refs with the fork mirror.
    git checkout $tagSha
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed" }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Worktree updated." -ForegroundColor Green
Write-Host ""
Write-Host "REVIEW REQUIRED before rebuild:" -ForegroundColor Yellow
Write-Host "  1. Open $root\include\llama.h"
Write-Host "  2. Compare struct llama_model_params + llama_context_params with"
Write-Host "     $root\python\llama_turboquant.py ctypes structs."
Write-Host "  3. See $root\python\ABI-NOTES.md for the field history."
Write-Host ""

if ($SkipBuild) {
    Write-Host "SkipBuild set - stopping." -ForegroundColor Yellow
    return
}

$ans = Read-Host "ABI verified? Run rebuild now? (y/N)"
if ($ans -ne 'y' -and $ans -ne 'Y') {
    Write-Host "Skipping rebuild. When ready: .\scripts\rebuild.ps1 -Clean -CudaArch $CudaArch"
    return
}

& (Join-Path $PSScriptRoot "rebuild.ps1") -Clean -CudaArch $CudaArch
