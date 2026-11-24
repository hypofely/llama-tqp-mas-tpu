#Requires -Version 5.1
<#
.SYNOPSIS
    Auto-creates PyTorch adapter weights for a GGUF model by detecting its hidden size.

.DESCRIPTION
    Loads the model via llama_turboquant to read n_embd, then creates and saves:
      - <model_name>_adapter_config.json  (metadata: hidden_size, n_layer, etc.)
      - <model_name>_adapter.pt           (RecursiveLink state_dict)
      - <model_name>_crosslink.pt         (CrossLink state_dict for a pair)

.PARAMETER ModelPath
    Full path to the .gguf model file.

.PARAMETER OutputDir
    Directory to save adapter files. Defaults to same folder as model.

.PARAMETER NoWeights
    Only create config JSON, skip saving .pt weights.

.EXAMPLE
    .\create_adapter_for_model.ps1 -ModelPath "E:\Models\qwen3.5-9b-q4.gguf"
    .\create_adapter_for_model.ps1 -ModelPath "E:\Models\gemma-4b.gguf" -OutputDir "C:\Adapters"
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$ModelPath,

    [string]$OutputDir = "",

    [switch]$NoWeights
)

$ErrorActionPreference = "Stop"

# ── Resolve paths ──────────────────────────────────────────────
$ModelPath = Resolve-Path $ModelPath -ErrorAction Stop
$ModelDir  = Split-Path $ModelPath -Parent
$ModelName = [System.IO.Path]::GetFileNameWithoutExtension($ModelPath)

if (-not $OutputDir) { $OutputDir = $ModelDir }
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null }
$OutputDir = Resolve-Path $OutputDir

Write-Host "[create-adapter] Model: $ModelPath" -ForegroundColor Cyan
Write-Host "[create-adapter] Output: $OutputDir" -ForegroundColor Cyan

# ── Inline Python: detect n_embd and save adapters ───────────
$PythonScript = @"
import sys, json, os, pathlib

model_path   = r"$($ModelPath)"
output_dir   = r"$($OutputDir)"
model_name   = "$($ModelName)"
save_weights = not $($NoWeights.IsPresent.ToString().ToLower())

# Add repo python/ to path so llama_turboquant can be imported
sys.path.insert(0, r"$(Split-Path $PSScriptRoot -Parent)\python")

from llama_turboquant import LlamaSharedModel

print(f"[python] Loading model to detect dimensions...")
model = LlamaSharedModel(model_path, n_gpu_layers=0)  # load briefly

meta = {
    "model_path": model_path,
    "model_name": model_name,
    "n_embd":     model.n_embd,
    "n_layer":    model.n_layer,
    "n_vocab":    model.n_vocab,
}
print(f"[python] Detected: n_embd={model.n_embd}, n_layer={model.n_layer}, n_vocab={model.n_vocab}")

# Save config JSON
config_path = os.path.join(output_dir, f"{model_name}_adapter_config.json")
with open(config_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"[python] Saved config: {config_path}")

if save_weights:
    import torch
    import torch.nn as nn

    # ── RecursiveLink (inner adapter) ──
    class RecursiveLink(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.pre_ln  = nn.LayerNorm(hidden_size)
            self.proj1   = nn.Linear(hidden_size, hidden_size)
            self.act     = nn.GELU()
            self.proj2   = nn.Linear(hidden_size, hidden_size)
            self.post_ln = nn.LayerNorm(hidden_size)

        def forward(self, x):
            h = self.pre_ln(x)
            out = self.proj2(self.act(self.proj1(h)))
            out = x + out
            return self.post_ln(out)

    inner = RecursiveLink(model.n_embd)
    inner_path = os.path.join(output_dir, f"{model_name}_adapter.pt")
    torch.save(inner.state_dict(), inner_path)
    print(f"[python] Saved RecursiveLink weights: {inner_path}")

    # ── CrossLink (cross-agent adapter) ──
    class CrossLink(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            hidden_dim = out_dim * 2
            self.ln_source     = nn.LayerNorm(in_dim)
            self.ln_target     = nn.LayerNorm(out_dim)
            self.proj1         = nn.Linear(in_dim, hidden_dim)
            self.act           = nn.GELU()
            self.proj2         = nn.Linear(hidden_dim, out_dim)
            self.residual_proj = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            h = self.ln_source(x)
            out = self.proj2(self.act(self.proj1(h)))
            out = out + self.residual_proj(x)
            return self.ln_target(out)

    # Same-dimension crosslink (most common case)
    cross = CrossLink(model.n_embd, model.n_embd)
    cross_path = os.path.join(output_dir, f"{model_name}_crosslink.pt")
    torch.save(cross.state_dict(), cross_path)
    print(f"[python] Saved CrossLink weights: {cross_path}")

    # ── Loader snippet ──
    snippet = f'''# Auto-generated loader for {model_name}
import torch
from recursive_mas_gguf import RecursiveLink, CrossLink, AgentConfig, GgufAgent

hidden_size = {model.n_embd}

# Load pre-created adapters
def load_inner_adapter(path="{inner_path.replace('\\','/')}", device="cuda"):
    link = RecursiveLink(hidden_size).to(device)
    link.load_state_dict(torch.load(path, map_location=device))
    return link

def load_crosslink(path="{cross_path.replace('\\','/')}", device="cuda"):
    cross = CrossLink(hidden_size, hidden_size).to(device)
    cross.load_state_dict(torch.load(path, map_location=device))
    return cross
'''
    snippet_path = os.path.join(output_dir, f"{model_name}_loader.py")
    with open(snippet_path, "w") as f:
        f.write(snippet)
    print(f"[python] Saved loader snippet: {snippet_path}")

print("[python] Done.")
"@

# ── Run Python ───────────────────────────────────────────────
$PythonExe = "C:\Users\zsolt\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction SilentlyContinue)?.Source
    if (-not $PythonExe) { throw "Python not found. Set `$PythonExe manually." }
}

Write-Host "[create-adapter] Running Python detection..." -ForegroundColor DarkGray
$Proc = Start-Process -FilePath $PythonExe -ArgumentList "-c", $PythonScript `
    -PassThru -Wait -NoNewWindow -WorkingDirectory (Split-Path $PSScriptRoot -Parent)

if ($Proc.ExitCode -ne 0) {
    throw "Python script failed with exit code $($Proc.ExitCode)"
}

Write-Host "`n[create-adapter] All files created successfully." -ForegroundColor Green
Write-Host "Use these values when creating agents:`n"
Get-Content (Join-Path $OutputDir "$ModelName`_adapter_config.json") -Raw | ConvertFrom-Json | Format-List
