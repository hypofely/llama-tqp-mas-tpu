"""Build a latent-training corpus from GSM8K.

WHY GSM8K
---------
GSM8K ships human-written step-by-step solutions with a #### <number> final
answer. That matters because the whole difficulty with Qwen3 thinking traces was
parsing where the reasoning lives - it moved between the think tags and the
answer channel unpredictably. GSM8K has no template to mis-parse: question,
reasoning, answer are separate fields.

WHAT EACH RECORD LOOKS LIKE
---------------------------
    {
      "messages": [
        {"role": "user", "content": "<question>"},
        {"role": "assistant", "content": "<reasoning> #### <answer>"}
      ],
      "reasoning": "<reasoning>",          # compressed into the latent slot
      "answer": "<answer>",
      "latent_depth": 1                     # how many thought vectors stand in
    }

The trainer replaces `reasoning` with a continuous embedding, so the model learns
to predict the answer WITHOUT re-generating the reasoning in text. That is the
entire point: the reasoning becomes latent.

DEPTH VARIATION
---------------
For a curriculum the same problem appears at several depths. depth=0 keeps the
reasoning as text (the baseline the model already knows). depth>=1 replaces the
reasoning with that many latent vectors. Mixing them is what teaches the model
that the vector slot is optional rather than mandatory.

OUTPUT: JSONL, one record per line, so it streams into the trainer.
"""

import argparse
import json
import random
import re
import sys


LATENT_TOKEN = "<|object_ref_start|>"


def load_gsm8k(split="train", limit=None):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets not installed: uv pip install datasets")

    # Full repo id: older `datasets` cannot resolve the bare "gsm8k" alias.
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def strip_calc_markup(text):
    """
    Remove GSM8K's <<a=b>> calculator annotations.

    GSM8K writes "Mira jogs 5 x 2 = <<5*2=10>>10 miles per day." The <<...>>
    is a dataset artifact telling a calculator what to evaluate; it is not
    reasoning a human or model would produce. Leaving it in teaches the latent
    slot to encode markup instead of reasoning structure.

    The convention is that the result also appears immediately after the
    annotation, so "5 x 2 = <<5*2=10>>10" reads correctly as "5 x 2 = 10".
    Deleting the <<...>> span therefore leaves valid text behind.
    """
    return re.sub(r"<<[^>]*>>", "", text)


def split_solution(solution):
    """
    GSM8K format: "<reasoning steps>\\n#### <number>"

    Returns (reasoning, answer). The reasoning keeps everything before the
    marker so the latent slot carries the full chain. Calculator markup is
    stripped so the latent encodes reasoning rather than dataset syntax.
    """
    if "####" not in solution:
        return strip_calc_markup(solution.strip()), ""
    reasoning, _, answer = solution.rpartition("####")
    return strip_calc_markup(reasoning.strip()), answer.strip()


def build(records, depths=(0, 1, 2), seed=1337, shuffle=True):
    rng = random.Random(seed)
    out = []
    for ex in records:
        q = ex["question"].strip()
        reasoning, answer = split_solution(ex["answer"])
        if not answer:
            continue
        for d in depths:
            if d == 0:
                # Baseline: reasoning stays as text. No latent vector.
                assistant = reasoning
                rec_reasoning = None
            else:
                # Latent: reasoning is compressed; the assistant turn begins
                # with the placeholder the trainer replaces by an embedding.
                # Must be a REAL single token in Qwen3's vocab - the trainer
                # locates the latent slot by token id. <|object_ref_start|> is a
                # real Qwen special token the chat template never emits.
                assistant = LATENT_TOKEN
                rec_reasoning = reasoning
            out.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": assistant},
                ],
                "reasoning": rec_reasoning,
                "answer": answer,
                "latent_depth": d,
            })
    if shuffle:
        rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--depths", default="0,1,2",
                    help="comma-separated latent depths to emit per problem")
    ap.add_argument("--out", default="gsm8k_latent.jsonl")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    depths = tuple(int(x) for x in args.depths.split(",") if x.strip())
    print("loading gsm8k/%s (limit %s)" % (args.split, args.limit))
    records = load_gsm8k(args.split, args.limit)
    print("loaded %d problems" % len(records))

    out = build(records, depths=depths, seed=args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    latent = sum(1 for r in out if r["latent_depth"] > 0)
    plain = len(out) - latent
    print("wrote %s: %d records (%d latent, %d plain-text)"
          % (args.out, len(out), latent, plain))
    print("depth histogram:")
    for d in depths:
        n = sum(1 for r in out if r["latent_depth"] == d)
        print("  depth %d: %d" % (d, n))


if __name__ == "__main__":
    main()