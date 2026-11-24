#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Sync llama-tqp-mas-cpp custom changes on top of a fresh upstream llama.cpp checkout.
.DESCRIPTION
    This script patches a new upstream llama.cpp release with the custom MAS/LatentMAS
    changes from this repo. It generates a diff, applies it to the new upstream, and
    copies any new (non-upstream) files that can't be patch-applied.
    
.USAGE
    # Sync from C:\+run\GPT_WEB\llama-tqp-mas-cpp onto new upstream:
    .\sync_upstream.ps1 -NewUpstreamPath "C:\+run\llama.cpp-new"
    
    # Or just generate the diff for manual apply:
    .\sync_upstream.ps1 -GenerateDiffOnly
#>

param(
    [string]$NewUpstreamPath = "",
    [switch]$GenerateDiffOnly,
    [switch]$DryRun,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path "$ScriptDir\.."  # llama-tqp-mas-cpp root
$UpstreamBase = Resolve-Path "$ScriptDir\..\upstream-reference"  # existing base
$DiffOutput = "$ScriptDir\..\custom.diff"
$LogPath = "$ScriptDir\sync_upstream.log"

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] $msg"
    Add-Content -Path $LogPath -Value "[$ts] $msg"
}

function Get-UpstreamBaseCommit {
    # Try to find the upstream base commit from git history
    pushd $RepoRoot
    try {
        $commit = git log --all --oneline --grep="upstream" | Select-Object -First 1
        if ($commit) {
            return $commit.ToString().Split(' ')[0]
        }
    } finally {
        popd
    }
    return ""
}

# ============================================================
# Step 1: Generate diff of custom changes
# ============================================================
function Generate-Diff {
    Log "Generating custom diff..."
    
    $diffFile = $DiffOutput
    if (Test-Path $diffFile) { Remove-Item $diffFile }
    
    pushd $RepoRoot
    try {
        # Diff against upstream base, or against HEAD if no base
        $baseCommit = Get-UpstreamBaseCommit
        if ($baseCommit) {
            Log "Using upstream base commit: $baseCommit"
            git diff "$baseCommit"..HEAD --no-color -- . `
                ":(exclude)build*/" `
                ":(exclude).vs/" `
                ":(exclude)archive/" `
                ":(exclude)compile_commands.json" |
                Out-File -FilePath $diffFile -Encoding utf8
        } else {
            Log "No upstream base found, diffing against empty tree"
            git diff --no-color HEAD -- . `
                ":(exclude)build*/" `
                ":(exclude).vs/" `
                ":(exclude)archive/" |
                Out-File -FilePath $diffFile -Encoding utf8
        }
    } finally {
        popd
    }
    
    $stats = git diff --stat "$baseCommit..HEAD" -- . `
        ":(exclude)build*/" ":(exclude).vs/" ":(exclude)archive/" 2>$null
    Log "Diff generated: $stats"
    Log "Diff file: $diffFile"
}

# ============================================================
# Step 2: List new (non-upstream) files that need manual copy
# ============================================================
function Get-NewFiles {
    Log "Identifying new (non-upstream) files..."
    
    $newFiles = @()
    $corePaths = @(
        "src\llama-mas.cpp",
        "src\llama-mas.h",
        "src\llama-layer-graph.cpp",
        "src\llama-layer-graph.h",
        "src\llama-kv-cache.cpp",
        "src\llama-kv-cache.h",
        "src\llama-context.cpp",
        "src\models\llama.cpp",
        "tools\mas-server\mas-server.cpp",
        "tools\mas-server\mas-server-common.cpp",
        "tools\mas-server\mas-server-common.h",
        "tools\mas-server\server-task.cpp",
        "tools\mas-server\server-task.h",
        "tools\mas-server\server-queue.cpp",
        "tools\mas-server\server-queue.h",
        "tools\mas-server\server-models.cpp",
        "tools\mas-server\server-models.h",
        "tools\mas-server\server-http.cpp",
        "tools\mas-server\server-http.h",
        "tools\mas-server\server-context.cpp",
        "tools\mas-server\server-context.h",
        "tools\mas-server\server-tools.cpp",
        "tools\mas-server\server-tools.h",
        "tools\mas-server\CMakeLists.txt",
        "ggml\src\ggml-cuda\mas-adapter.cu",
        "ggml\src\ggml-cuda\mas-adapter.cuh",
        "ggml\src\ggml-cuda\cpy-planar-iso.cu",
        "ggml\src\ggml-cuda\cpy-planar-iso.cuh",
        "ggml\src\ggml-cuda\cpy-utils.cuh",
        "ggml\src\ggml-cuda\set-rows.cu",
        "ggml\src\ggml-cuda\set-rows.cuh",
        "ggml\src\ggml-cuda\planar-iso-constants.cuh",
        "ggml\src\ggml-cuda\turbo-quant.cuh",
        "ggml\src\ggml-cuda\turbo-wht.cu",
        "ggml\src\ggml-cuda\turbo-innerq.cu",
        "include\llama.h"
    )
    
    foreach ($path in $corePaths) {
        $fullPath = Join-Path $RepoRoot $path
        if (Test-Path $fullPath) {
            $newFiles += $path
        }
    }
    
    Log "Core files to preserve:"
    $newFiles | ForEach-Object { Log "  $_" }
    
    # Also list all modified files from git
    pushd $RepoRoot
    try {
        $modified = git diff --name-only HEAD 2>$null
        $modified | ForEach-Object { Log "  MODIFIED: $_" }
    } finally {
        popd
    }
    
    return $newFiles
}

# ============================================================
# Step 3: Apply diff to new upstream
# ============================================================
function Apply-Patch {
    param([string]$TargetPath)
    
    Log "Applying patch to: $TargetPath"
    
    if (-not (Test-Path $TargetPath)) {
        Log "ERROR: Target path does not exist: $TargetPath"
        exit 1
    }
    
    # Copy core new files first (they don't exist in upstream)
    $newFiles = Get-NewFiles
    foreach ($file in $newFiles) {
        $src = Join-Path $RepoRoot $file
        $dst = Join-Path $TargetPath $file
        
        if (-not (Test-Path $dst)) {
            $dstDir = Split-Path -Parent $dst
            if (-not (Test-Path $dstDir)) {
                New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
            }
            Copy-Item $src $dst -Force
            Log "  COPIED (new): $file"
        }
    }
    
    # Apply the diff
    $diffFile = $DiffOutput
    if (Test-Path $diffFile) {
        pushd $TargetPath
        try {
            $patchResult = git apply --whitespace=nowarn "$diffFile" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Log "  PATCH applied successfully"
            } else {
                Log "  PATCH conflicts detected, see above"
                if ($Verbose) { $patchResult | ForEach-Object { Log "    $_" } }
            }
        } finally {
            popd
        }
    } else {
        Log "  No diff file found, skipping patch step"
    }
    
    # Copy over any additional new files that patch missed
    $extraNew = @(
        "tools\mas-server\CMakeLists.txt",
        "examples\mas-bench\mas-bench.cpp",
        "examples\mas-bench\latent-bench.cpp",
        "examples\mas-bench\test-alignment.cpp",
        "examples\mas-bench\CMakeLists.txt"
    )
    foreach ($file in $extraNew) {
        $src = Join-Path $RepoRoot $file
        $dst = Join-Path $TargetPath $file
        if (Test-Path $src -and -not (Test-Path $dst)) {
            $dstDir = Split-Path -Parent $dst
            if (-not (Test-Path $dstDir)) {
                New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
            }
            Copy-Item $src $dst -Force
            Log "  COPIED (extra): $file"
        }
    }
}

# ============================================================
# Step 4: Build verification
# ============================================================
function Test-Build {
    param([string]$TargetPath)
    
    Log "Testing build in: $TargetPath"
    
    pushd $TargetPath
    try {
        # Check if CMakeLists.txt has mas-server
        $cmake = Get-Content "tools\CMakeLists.txt" -ErrorAction SilentlyContinue
        if ($cmake -match "mas-server") {
            Log "  CMakeLists.txt includes mas-server ✓"
        } else {
            Log "  WARNING: mas-server not found in tools/CMakeLists.txt"
        }
        
        # Check if include/llama.h has MAS exports
        $llama_h = Get-Content "include\llama.h" -ErrorAction SilentlyContinue
        if ($llama_h -match "llama_mas_") {
            Log "  include/llama.h has MAS exports ✓"
        } else {
            Log "  WARNING: MAS exports missing from include/llama.h"
        }
        
        # Check CMake build dir
        if (Test-Path "build") {
            Log "  Build directory exists"
        } else {
            Log "  No build directory — run cmake configure first"
        }
    } finally {
        popd
    }
}

# ============================================================
# Main
# ============================================================
Log "=== sync_upstream.ps1 starting ==="
Log "Repo root: $RepoRoot"

if ($GenerateDiffOnly) {
    Generate-Diff
    Log "=== Diff-only mode complete ==="
    exit 0
}

if (-not $NewUpstreamPath) {
    Log "ERROR: -NewUpstreamPath is required (or use -GenerateDiffOnly)"
    Log "Usage: .\sync_upstream.ps1 -NewUpstreamPath 'C:\path\to\new\llama.cpp'"
    exit 1
}

$Target = Resolve-Path $NewUpstreamPath
if (-not $DryRun) {
    Generate-Diff
    Apply-Patch -TargetPath $Target
    Test-Build -TargetPath $Target
} else {
    Log "DRY RUN — would apply patch to: $Target"
    Get-NewFiles | ForEach-Object { Log "  Would copy: $_" }
}

Log "=== sync_upstream.ps1 complete ==="
