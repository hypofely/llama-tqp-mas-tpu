"""Extract reasoning traces from Qwen3-style thinking output.

WHY THIS EXISTS
---------------
Qwen3 "thinking mode" (chat_template_kwargs.enable_thinking=true) does NOT
populate `reasoning_content`. That field comes back EMPTY. The trace lives in
`content`, delimited by think tags. Reading message["reasoning_content"]
silently yields nothing.

Observed shape (Qwen3-1.7B, "What is 17*3?"):

    <think>
</think>

    The problem is straightforward. 17 multiplied by 3 is 51. ...

    </think>

    A restated answer follows.

So there is an EMPTY opening block, the real reasoning sits between the tags,
and the answer follows the close. A silent mis-parse is indistinguishable from
"the model produced no reasoning", hence the selftest below.

RULES:
  1. Answer = everything after the LAST close tag.
  2. Reasoning zone = between the first open and the last close.
  3. Zone split on blank lines; whitespace-only segments dropped (discards the
     empty opening block).
  4. No tags at all -> whole thing is the answer, no thoughts.
"""

import re

OPEN_RE = re.compile(r"<think\b[^>]*>")
CLOSE_RE = re.compile(r"</think\s*>")
TAG_RE = re.compile(r"</?think\b[^>]*>")


def split_trace(content: str) -> dict:
    """Split thinking output into reasoning segments and the final answer."""
    out = {"thoughts": [], "answer": "", "blocks": [], "had_tags": False, "note": ""}

    if not content:
        out["note"] = "empty content"
        return out

    opens = [(m.start(), m.end()) for m in OPEN_RE.finditer(content)]
    closes = [(m.start(), m.end()) for m in CLOSE_RE.finditer(content)]

    if not opens and not closes:
        out["answer"] = content.strip()
        out["note"] = "no think tags - plain response"
        return out

    out["had_tags"] = True

    # 1. Answer: after the LAST close. If no close, after the LAST open.
    if closes:
        out["answer"] = TAG_RE.sub("", content[closes[-1][1]:]).strip()
    elif opens:
        out["answer"] = TAG_RE.sub("", content[opens[-1][1]:]).strip()

    # 2. Reasoning zone: first open -> last close.
    if opens and closes and closes[-1][1] > opens[0][1]:
        zone = content[opens[0][1]: closes[-1][0]]
    elif opens:
        zone = content[opens[0][1]:]
    else:
        zone = content[: closes[0][0]] if closes else ""

    # 3. Split on blank lines, drop whitespace-only segments.
    for para in re.split(r"\n\s*\n", zone):
        p = TAG_RE.sub("", para).strip()
        if p:
            out["thoughts"].append(p)

    # 4. Record block pairs for debugging.
    for (o_start, o_end) in opens:
        later = [c_end for (c_start, c_end) in closes if c_end > o_end]
        if later:
            out["blocks"].append((o_start, min(later)))

    if not out["thoughts"]:
        out["note"] = "tags present but reasoning zone empty"
    return out


def is_usable(thoughts, min_chars: int = 40) -> bool:
    """A trace counts as usable only with a real amount of reasoning text."""
    return sum(len(t) for t in thoughts) >= min_chars


def _selftest() -> int:
    """Check against the exact shapes seen in practice. Returns failure count."""
    fails = 0

    # 1. Real Qwen3-1.7B shape, verbatim from /v1/chat/completions:
    #    empty opening block, reasoning, closing tag, restated answer.
    real = (
        "%s\n%s\n\n" % (O, C) +
        "The problem is straightforward. 17 multiplied by 3 is 51. "
        "Let me explain it step by step.\n\n"
        "First, lets multiply 17 by 3.\n\n"
        "%s\n\n" % C +
        "17 multiplied by 3 is 517 * 3 = 51. \n\n"
        "Let me explain it step by step:\n\n1. Multiply 7 by 3 = 21."
    )
    r = split_trace(real)
    if len(r["thoughts"]) < 1:
        print("FAIL real-shape: thoughts=%r" % (r["thoughts"],))
        fails += 1
    joined = " ".join(r["thoughts"])
    if "straightforward" not in joined:
        print("FAIL real-shape: reasoning not captured -> %r" % joined[:80])
        fails += 1
    if not is_usable(r["thoughts"]):
        print("FAIL real-shape: not usable")
        fails += 1
    if "1. Multiply" not in r["answer"]:
        print("FAIL real-shape: answer=%r" % r["answer"][:70])
        fails += 1

    # 2. Empty opening block must NOT become a thought.
    r2 = split_trace("%s\n%s\n\nThe answer is 51." % (O, C))
    if any(t.strip() == "" for t in r2["thoughts"]):
        print("FAIL empty-block: empty string kept as thought")
        fails += 1

    # 3. Plain response, no tags.
    r3 = split_trace("Paris is the capital of France.")
    if r3["had_tags"] or r3["thoughts"] or r3["answer"] != "Paris is the capital of France.":
        print("FAIL plain: had_tags=%r thoughts=%r answer=%r"
              % (r3["had_tags"], r3["thoughts"], r3["answer"]))
        fails += 1

    # 4. Well-formed thinking with a real answer after the close.
    good = ("%s\n" % O + "Let me work through this.\n\n"
            + "17 * 3 = 51.\n" + "%s\n\n" % C + "The answer is 51.")
    r4 = split_trace(good)
    if not r4["thoughts"]:
        print("FAIL good: no thoughts")
        fails += 1
    if "51" not in r4["answer"]:
        print("FAIL good: answer=%r" % r4["answer"])
        fails += 1

    return fails


O = "%s" % OPEN_RE.pattern.replace(chr(92) + "b[^>]*>", ">")
C = CLOSE_RE.pattern.replace(chr(92) + "s*>", ">")


if __name__ == "__main__":
    import sys

    fails = _selftest()
    print("selftest: %s" % ("OK" if fails == 0 else "%d FAILURE(S)" % fails))
    sys.exit(1 if fails else 0)
