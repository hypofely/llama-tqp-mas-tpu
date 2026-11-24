"""
True latent reasoning finetune.

WHAT "TRUE LATENT" MEANS HERE
-----------------------------
The model is trained so that a CONTINUOUS VECTOR can occupy an input position and
stand in for reasoning that would otherwise be emitted as text tokens.

Mechanically, for a record with latent depth d:
  - tokenise  [question] [placeholder x d] [answer]
  - embed everything normally
  - REPLACE each placeholder's embedding with a vector produced from the model's
    own hidden state at that position (projected through a trainable W)
  - compute loss ONLY on the answer tokens
  - backprop into BOTH LoRA and W

The key point: the placeholder is not a token the model learns to emit. Its
embedding is overwritten by a continuous vector derived from the running hidden
state. The model therefore learns "when the input at this slot is a dense vector
rather than an embedding row, continue from it" - which is the capability that
discrete-token training cannot teach.

WHY THE PROJECTION IS NON-TRIVIAL
---------------------------------
Feeding a raw hidden state back as an embedding is off-manifold (already
established: it changes output but does not create capability). W is the trainable
bridge that makes the vector land where the model expects. Without W in the
gradient path this degenerates to token training, so W must receive gradients.

TRAINING DETAIL THAT MATTERS
----------------------------
Qwen3 ties input and output embeddings. A LoRA adapter on the embedding matrix
would therefore perturb the output projection too, so embedding/lm_head modules
are excluded from LoRA targeting; only attention/MLP projections plus W adapt.
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


PLACEHOLDER = "<|object_ref_start|>"  # real unused Qwen token; see note below
PH_ID = None  # set in main() once the tokenizer is known


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_batch(examples, tokenizer, device, max_len=1024):
    """
    Tokenise a batch into the tensors the latent forward pass needs.

    Returns:
      input_ids    [B,T]  token ids, placeholders included
      placeholder  [B,T]  bool mask, True where a latent vector must be injected
      answer_mask  [B,T]  bool mask, True where loss is computed
      attn         [B,T]  attention mask

    Depth 0 records carry no placeholder; their reasoning is plain text, which
    gives the model a baseline and keeps the curriculum mixed.
    """
    bs = len(examples)
    seqs = []
    for ex in examples:
        msgs = ex["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        asst = next(m["content"] for m in msgs if m["role"] == "assistant")

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        full = prompt + asst + (tokenizer.eos_token or "")

        ids = tokenizer(full, add_special_tokens=False,
                        truncation=True, max_length=max_len)["input_ids"]
        p_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        seqs.append((ids, p_len))

    T = max(len(s) for s, _ in seqs)
    input_ids = torch.full((bs, T), tokenizer.pad_token_id or 0, dtype=torch.long)
    placeholder = torch.zeros((bs, T), dtype=torch.bool)
    answer_mask = torch.zeros((bs, T), dtype=torch.bool)
    attn = torch.zeros((bs, T), dtype=torch.long)

    for i, (ids, n_prompt) in enumerate(seqs):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids)
        attn[i, :L] = 1
        # Loss on the answer only: everything from the end of the prompt on.
        answer_mask[i, n_prompt:L] = True
        # Mark placeholder slots inside the assistant span.
        if PH_ID is not None:
            for pos in range(n_prompt, L):
                if ids[pos] == PH_ID:
                    placeholder[i, pos] = True

    return {
        "input_ids": input_ids.to(device),
        "placeholder": placeholder.to(device),
        "answer_mask": answer_mask.to(device),
        "attn": attn.to(device),
    }


class LatentProjection(nn.Module):
    """
    Maps a hidden state to an input-space embedding.

    Small on purpose: LayerNorm -> Linear -> GELU -> Linear with a residual
    straight through, so at init it behaves near identity and training starts
    from something sane rather than noise.
    """

    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h):
        n = self.ln(h)
        return n * self.scale + self.proj(n)


def compute_loss(logits, input_ids, answer_mask):
    """
    Next-token cross entropy, masked to the answer span.

    Position t predicts token t+1, so logits and labels are both shifted by one.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = answer_mask[:, 1:]

    if shift_mask.sum() == 0:
        return None

    vocab = shift_logits.size(-1)
    flat_logits = shift_logits.reshape(-1, vocab)
    flat_labels = shift_labels.reshape(-1)
    flat_mask = shift_mask.reshape(-1)

    losses = F.cross_entropy(flat_logits.float(), flat_labels,
                             reduction="none", ignore_index=-100)
    losses = losses * flat_mask.float()
    return losses.sum() / flat_mask.float().sum().clamp(min=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data", default="tools/training/gsm8k_latent.jsonl")
    ap.add_argument("--out", default="latent_adapter")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--proj-lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dry-run", action="store_true",
                    help="build one batch and report shapes, train nothing")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        sys.exit("CUDA required: no GPU visible to torch")
    free, total = torch.cuda.mem_get_info()
    print("device: %s (%.2f GB free of %.2f)"
          % (torch.cuda.get_device_name(0), free / 1e9, total / 1e9))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print("loading tokenizer + model: %s" % args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # The placeholder must be ONE real token in the base vocab. Qwen3 has no
    # <|latent|> token, and inventing one by adding to the tokenizer would shift
    # nothing but would require embedding-resize surgery on tied weights. So use
    # an existing special token the chat template never emits:
    # <|object_ref_start|> (id 151646). It is outside the text distribution, so
    # its embedding is untrained for prose - and because we OVERWRITE that
    # embedding with the projected hidden state anyway, what it means as a token
    # is irrelevant. Only its id matters, as a marker to locate the slot.
    global PH_ID
    PH_ID = tok.convert_tokens_to_ids(PLACEHOLDER)
    if PH_ID is None or PH_ID == tok.unk_token_id:
        sys.exit("placeholder %r is not a single token in this tokenizer" % PLACEHOLDER)
    print("placeholder %r -> id %d" % (PLACEHOLDER, PH_ID))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    model.to(dev := torch.device("cuda"))

    # LoRA on attention/MLP projections only. Embeddings stay frozen: Qwen3 ties
    # them to lm_head, so adapting them would silently change the output head.
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    hidden = model.config.hidden_size
    proj = LatentProjection(hidden).to(dev, dtype=torch.bfloat16)
    print("projection: %d -> %d, params=%d"
          % (hidden, hidden, sum(p.numel() for p in proj.parameters())))

    data = load_jsonl(args.data)
    if args.limit:
        data = data[:args.limit]
    n_latent = sum(1 for e in data if e.get("latent_depth", 0) > 0)
    print("examples: %d  (latent %d, plain-text %d)"
          % (len(data), n_latent, len(data) - n_latent))
    if n_latent == 0:
        sys.exit("no latent records - nothing to train")

    # ---- dry run -----------------------------------------------------------
    if args.dry_run:
        sample = [e for e in data if e.get("latent_depth", 0) > 0][:2]
        b = build_batch(sample, tok, dev, args.max_len)
        print("\ndry run batch:")
        for k, v in b.items():
            print("  %-12s %s" % (k, tuple(v.shape)))
        n_ph = int(b["placeholder"].sum())
        n_ans = int(b["answer_mask"].sum())
        print("  placeholder positions: %d" % n_ph)
        print("  answer positions     : %d" % n_ans)
        if n_ph == 0:
            sys.exit("dry run found no placeholder positions - check PLACEHOLDER")
        if n_ans == 0:
            sys.exit("dry run found no answer positions - loss would be undefined")

        ids = b["input_ids"]
        emb = model.get_input_embeddings()(ids).to(torch.bfloat16)
        with torch.no_grad():
            h = model(inputs_embeds=emb, attention_mask=b["attn"],
                      use_cache=False, output_hidden_states=True).hidden_states[-1]
        vec = proj(h)
        emb2 = emb.clone()
        m = b["placeholder"]
        emb2[m] = vec[m].to(emb2.dtype)
        logits = model(inputs_embeds=emb2, attention_mask=b["attn"],
                       use_cache=False, return_dict=True).logits
        loss = compute_loss(logits, ids, b["answer_mask"])
        print("  logits %s  loss %s"
              % (tuple(logits.shape), ("%.4f" % loss.item()) if loss is not None else "None"))
        print("  VRAM now %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
        return 0

    # ---- training ----------------------------------------------------------
    params_lora = [p for p in model.parameters() if p.requires_grad]
    params_proj = list(proj.parameters())
    opt = torch.optim.AdamW([
        {"params": params_lora, "lr": args.lr},
        {"params": params_proj, "lr": args.proj_lr},
    ], weight_decay=0.01)

    model.train()
    proj.train()

    rng = random.Random(args.seed)
    step = 0
    running = 0.0
    t0 = time.time()

    for epoch in range(args.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            exs = [data[j] for j in order[i:i + args.batch_size]]
            b = build_batch(exs, tok, dev, args.max_len)

            ids = b["input_ids"]
            emb = model.get_input_embeddings()(ids).to(torch.bfloat16)

            # Pass 1 (no grad): hidden states the latent vector is derived from.
            with torch.no_grad():
                h = model(inputs_embeds=emb, attention_mask=b["attn"],
                          use_cache=False, output_hidden_states=True).hidden_states[-1]

            # Inject the projected vector at every placeholder slot.
            m = b["placeholder"]
            if m.any():
                vec = proj(h)
                emb = emb.clone()
                emb[m] = vec[m].to(emb.dtype)

            # Pass 2: forward with latent slots filled, backprop the answer loss.
            logits = model(inputs_embeds=emb, attention_mask=b["attn"],
                           use_cache=False, return_dict=True).logits
            loss = compute_loss(logits, ids, b["answer_mask"])
            if loss is None:
                continue

            (loss / args.grad_accum).backward()
            running += loss.item()
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params_lora + params_proj, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % 10 == 0:
                print("  step %4d  loss %.4f  %.1fs  %.2f GB"
                      % (step, running / 10, time.time() - t0,
                         torch.cuda.max_memory_allocated() / 1e9))
                running = 0.0

    model.save_pretrained(args.out)
    torch.save({"proj": proj.state_dict(), "hidden": hidden},
               os.path.join(args.out, "latent_proj.pt"))
    print("\nsaved adapter -> %s" % args.out)
    print("  latent_proj.pt holds the trainable projection")
    print("total time %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())