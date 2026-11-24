#!/usr/bin/env python3
"""
Train CrossLink adapter weights for cross-model hidden-state projection.

Usage:
    python train_crosslink.py \
        --source-model "path/to/gemma-4b.gguf" \
        --target-model "path/to/qwen-9b.gguf" \
        --dataset prompts.txt \
        --output crosslink_weights.npz \
        --hidden-dim 4096 \
        --epochs 200 \
        --lr 1e-4

The script:
  1. Connects to the running MAS server (default http://127.0.0.1:8080)
  2. Loads source model, collects hidden states for each prompt
  3. Swaps to target model, collects input embeddings for same prompts
  4. Trains a small PyTorch MLP to map source hidden -> target embedding
  5. Saves trained weights to an .npz file
  6. (Optional) Uploads weights to a specified agent via HTTP

Requirements: torch, numpy, requests
"""

import argparse
import base64
import json
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def mas_post(endpoint, data, server_url="http://127.0.0.1:8080"):
    """POST JSON to MAS server, return parsed JSON."""
    url = f"{server_url}{endpoint}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[ERROR] POST {endpoint} failed: {e}", file=sys.stderr)
        return None


def mas_get(endpoint, server_url="http://127.0.0.1:8080"):
    """GET from MAS server, return parsed JSON."""
    url = f"{server_url}{endpoint}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[ERROR] GET {endpoint} failed: {e}", file=sys.stderr)
        return None


def wait_for_server(server_url, timeout=120):
    print(f"[WAIT] Waiting for server at {server_url} ...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = mas_get("/health", server_url)
            if r and r.get("status") == "ok":
                print("[OK] Server is ready")
                return True
        except Exception:
            pass
        time.sleep(0.5)
    print("[FAIL] Server not ready within timeout", file=sys.stderr)
    return False


def swap_model(model_path, server_url="http://127.0.0.1:8080"):
    print(f"[SWAP] Loading model: {model_path}")
    r = mas_post("/config/model", {"model_path": model_path}, server_url)
    if r and r.get("status") == "ok":
        print("[OK] Model loaded")
        return True
    print(f"[FAIL] Model swap failed: {r}", file=sys.stderr)
    return False


def collect_hidden_states(prompts, server_url="http://127.0.0.1:8080", max_tokens=0):
    """
    Run each prompt through the current model with stop_after_prompt=true
    and capture_hidden_states=true.  Returns list of float arrays (n_embd,).
    """
    states = []
    n_embd = None
    for prompt in prompts:
        print(f"  [COLLECT] prompt: {prompt[:60]}...")
        # Single-agent run, stop after prompt to get hidden state before generation
        r = mas_post("/mas/run", {
            "task": prompt,
            "roles": ["encoder"],
            "strategy": "broadcast",
            "max_tokens": max_tokens,
            "clear_agents": True,
            "capture_hidden_states": True,
            "stop_after_prompt": True,
            "temperature": 0.0
        }, server_url)
        if not r or r.get("status") != "ok":
            print(f"  [WARN] run failed: {r}")
            states.append(None)
            continue

        # Get hidden state
        hs = mas_get("/v1/agents/0/hidden-state", server_url)
        if not hs:
            print("  [WARN] hidden state not available")
            states.append(None)
            continue

        arr = np.array(hs["data"], dtype=np.float32)
        states.append(arr)
        if n_embd is None:
            n_embd = arr.shape[0]
        print(f"  [OK] hidden shape={arr.shape}")
    return states, n_embd


def collect_target_embeddings(prompts, server_url="http://127.0.0.1:8080"):
    """
    For the target model we need the *input embedding* of the prompt,
    not the hidden state.  We get this by tokenizing the prompt on the
    target model and then running a dummy decode to capture the last
    token's hidden state (which for the first decode step is the prompt
    embedding transformed by the model).

    A simpler proxy: run stop_after_prompt and take the hidden state.
    The hidden state after prompt decode is the model's representation
    of the prompt, which is a reasonable training target.
    """
    return collect_hidden_states(prompts, server_url, max_tokens=0)


# ---------------------------------------------------------------------------
# CrossLink PyTorch model (must match C++ architecture exactly)
# ---------------------------------------------------------------------------
class CrossLink(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.ln_source = nn.LayerNorm(in_dim, elementwise_affine=True)
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, out_dim)
        self.residual = nn.Linear(in_dim, out_dim)
        self.ln_target = nn.LayerNorm(out_dim, elementwise_affine=True)

    def forward(self, x):
        # x: [batch, in_dim]
        normed = self.ln_source(x)
        h = F.gelu(self.proj1(normed))
        out = self.proj2(h)
        res = self.residual(x)
        out = out + res
        out = self.ln_target(out)
        return out


def train(source_states, target_states, in_dim, out_dim, hidden_dim,
          epochs=200, lr=1e-4, batch_size=8, device="cuda"):
    print(f"[TRAIN] in_dim={in_dim} out_dim={out_dim} hidden_dim={hidden_dim}")
    print(f"[TRAIN] dataset size={len(source_states)} epochs={epochs} lr={lr}")

    X = torch.tensor(np.stack(source_states), dtype=torch.float32, device=device)
    Y = torch.tensor(np.stack(target_states), dtype=torch.float32, device=device)

    model = CrossLink(in_dim, out_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(dataset)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}  loss={avg_loss:.6f}  best={best_loss:.6f}")

    print(f"[TRAIN] Finished. Best MSE={best_loss:.6f}")
    model.load_state_dict(best_state)
    return model


def export_weights(model, path):
    """Save trained weights in CrossLink binary format (.bin)."""
    import struct
    state = model.state_dict()
    in_dim = state["ln_source.weight"].shape[0]
    out_dim = state["ln_target.weight"].shape[0]
    hidden_dim = state["proj1.weight"].shape[0]
    n_embd = in_dim  # not used by CrossLink, set to in_dim

    def flatten(t):
        return t.cpu().numpy().astype(np.float32).ravel().tobytes()

    with open(path, "wb") as f:
        # Header
        f.write(struct.pack("<I", 0x4C4E4352))  # magic 'LNCR'
        f.write(struct.pack("<I", 1))           # version
        f.write(struct.pack("<I", in_dim))
        f.write(struct.pack("<I", out_dim))
        f.write(struct.pack("<I", hidden_dim))
        f.write(struct.pack("<I", n_embd))
        # Weight arrays (PyTorch Linear weight is [out_features, in_features] which matches C row-major)
        f.write(flatten(state["ln_source.weight"]))
        f.write(flatten(state["ln_source.bias"]))
        f.write(flatten(state["ln_target.weight"]))
        f.write(flatten(state["ln_target.bias"]))
        f.write(flatten(state["proj1.weight"]))
        f.write(flatten(state["proj1.bias"]))
        f.write(flatten(state["proj2.weight"]))
        f.write(flatten(state["proj2.bias"]))
        f.write(flatten(state["residual.weight"]))
        f.write(flatten(state["residual.bias"]))
    print(f"[SAVE] Weights saved to {path}  (in={in_dim} out={out_dim} hid={hidden_dim})")


def upload_weights_to_agent(agent_idx, bin_path, adapter_type="cross",
                              server_url="http://127.0.0.1:8080"):
    """Upload trained weights to a running MAS agent via HTTP."""
    payload = {
        "weights_file": bin_path,
        "adapter_type": adapter_type,
    }
    r = mas_post(f"/v1/agents/{agent_idx}/adapter/load", payload, server_url)
    if r and r.get("status") == "ok":
        print(f"[UPLOAD] Adapter uploaded to agent {agent_idx}")
        return True
    print(f"[FAIL] Upload failed: {r}", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(description="Train CrossLink adapter")
    parser.add_argument("--source-model", required=True, help="Path to source GGUF")
    parser.add_argument("--target-model", required=True, help="Path to target GGUF")
    parser.add_argument("--dataset", required=True, help="Text file with one prompt per line")
    parser.add_argument("--output", default="crosslink_weights.npz", help="Output .npz file")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="CrossLink hidden dimension")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--upload-to-agent", type=int, default=-1,
                        help="After training, upload weights to this agent index")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load prompts
    with open(args.dataset, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if len(prompts) < 2:
        print("[ERROR] Need at least 2 prompts in dataset", file=sys.stderr)
        sys.exit(1)
    print(f"[DATA] Loaded {len(prompts)} prompts")

    if not wait_for_server(args.server_url):
        sys.exit(1)

    # ---------------------------------------------------------------
    # 1. Collect source hidden states
    # ---------------------------------------------------------------
    print("\n--- Collecting source hidden states ---")
    if not swap_model(args.source_model, args.server_url):
        sys.exit(1)
    source_states, source_n_embd = collect_hidden_states(prompts, args.server_url)
    source_states = [s for s in source_states if s is not None]
    if not source_states:
        print("[ERROR] No source hidden states collected", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] Source states: {len(source_states)} x {source_states[0].shape}")

    # ---------------------------------------------------------------
    # 2. Collect target embeddings
    # ---------------------------------------------------------------
    print("\n--- Collecting target embeddings ---")
    if not swap_model(args.target_model, args.server_url):
        sys.exit(1)
    target_states, target_n_embd = collect_target_embeddings(prompts, args.server_url)
    target_states = [s for s in target_states if s is not None]
    if not target_states:
        print("[ERROR] No target embeddings collected", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] Target states: {len(target_states)} x {target_states[0].shape}")

    # Align lengths (in case some prompts failed on one side)
    n = min(len(source_states), len(target_states))
    source_states = source_states[:n]
    target_states = target_states[:n]
    print(f"[ALIGN] Using {n} paired samples")

    # ---------------------------------------------------------------
    # 3. Train
    # ---------------------------------------------------------------
    print("\n--- Training CrossLink ---")
    model = train(
        source_states, target_states,
        in_dim=source_n_embd,
        out_dim=target_n_embd,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device
    )

    # ---------------------------------------------------------------
    # 4. Save
    # ---------------------------------------------------------------
    export_weights(model, args.output)

    # ---------------------------------------------------------------
    # 5. Optionally upload
    # ---------------------------------------------------------------
    if args.upload_to_agent >= 0:
        print(f"\n--- Uploading to agent {args.upload_to_agent} ---")
        # Need source model loaded again for the adapter to work on that model
        swap_model(args.source_model, args.server_url)
        upload_weights_to_agent(args.upload_to_agent, args.output, server_url=args.server_url)

    print("\n[done]")


if __name__ == "__main__":
    main()
