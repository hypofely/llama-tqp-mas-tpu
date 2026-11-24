"""
LoRA finetune for LATENT reasoning: teach a model to consume its own hidden
states as "thought" tokens instead of emitting them as text.

WHY THIS SHAPE OF TRAINER
-------------------------
Naive approach that does NOT work: fine-tune on (question, reasoning, answer)
text. That teaches better reasoning in TEXT. It does not make the model accept a
continuous vector in the input slot, because the loss never sees one.

What this trainer does instead: build each example so that the reasoning span is
REPLACED by a placeholder token whose input embedding is a learned continuous
vector. The model is trained to predict the answer given:
    [question tokens] [THOUGHT placeholder] [answer tokens]
and the THOUGHT slot receives an embedding produced by a small projection of the
model's own hidden state at that position. Backprop through that projection is
what teaches the network that "a vector here means reasoning already happened".

Concretely per example:
  h = hidden state at the last question token          (no grad; frozen base)
  e = W_proj(h)                                        (trainable)
  logits = model(inputs_embeds = [emb(question), e, emb(answer)])
  loss  = cross_entropy(logits, answer_tokens)         (masked to answer only)

THE TRAP THIS AVOIDS: training on question+answer only, with the thought as
plain text, produces a model that is better at reasoning but still cannot take a
continuous input. The projection W_proj is the part that makes latent work, so
it must be in the gradient path.

FRAMEWORK CHOICE: this uses llama.cpp's finetune example, not HF Transformers.
The target is a GGUF model that llama.cpp serves, and HF Transformers would need
a separate fp16 download plus a conversion step. llama.cpp is already built here
with CUDA and reads GGUF directly.

STATUS: skeleton + data pipeline complete; the forward/backward wiring below is
the part under construction.
"""

import argparse
import json
import os
import sys

# Placeholder used in the instruction text so a human can see where the latent
# vector goes. It is replaced by the projected embedding before the forward pass.
THOUGHT_PLACEHOLDER = "<|latent|>"


def load_examples(path, limit=None):
    """
    Load training examples.

    Each line is JSON:
      {"messages": [{"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."}],
       "reasoning": "optional chain of thought to compress into a latent vector"}

    When "reasoning" is present, the trainer inserts THOUGHT_PLACEHOLDER between
    the last user turn and the assistant turn.
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def build_training_text(ex, tokenizer):
    """
    Render one example to (prompt_text, answer_text).

    Returns the prompt with the placeholder appended, and the answer that the
    loss is computed over. The placeholder is what the latent embedding replaces.
    """
    msgs = ex["messages"]
    has_reasoning = bool(ex.get("reasoning"))
    return msgs, has_reasoning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="GGUF model to finetune")
    ap.add_argument("--data", required=True, help="JSONL training data")
    ap.add_argument("--out", required=True, help="output GGUF")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--gpu-layers", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit("missing model: %s" % args.model)
    if not os.path.exists(args.data):
        sys.exit("missing data: %s" % args.data)

    examples = load_examples(args.data, args.limit)
    print("loaded %d examples" % len(examples))

    with_reason = sum(1 for e in examples if e.get("reasoning"))
    print("with reasoning: %d  (these become latent-training examples)" % with_reason)
    if with_reason == 0:
        print("\nWARNING: no example has a 'reasoning' field.")
        print("Without it every example trains the ordinary text path and the")
        print("model never sees a continuous input vector - i.e. this would not")
        print("teach latent reasoning at all.")
        print("Add \"reasoning\" to some examples, or pass --allow-plain-text.")

    print("\nplanned run:")
    print("  model      : %s" % args.model)
    print("  data       : %s" % args.data)
    print("  out        : %s" % args.out)
    print("  epochs     : %d" % args.epochs)
    print("  batch      : %d" % args.batch_size)
    print("  lr         : %g" % args.lr)
    print("  lora r/alpha: %d/%d" % (args.lora_rank, args.lora_alpha))
    print("  ctx        : %d" % args.ctx)
    print("\nNOT IMPLEMENTED YET - see module docstring for the forward/backward")
    print("design. Refusing to write a partial GGUF that looks trained but is not.")
    return 1


if __name__ == "__main__":
    sys.exit(main())