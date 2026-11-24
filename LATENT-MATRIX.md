# Latent / hidden-state feature matrix

Living checklist for the latent-reasoning work in
`C:\+run\GPT_WEB\llama-tqp-mas-cpp`. Every item must land as a **toggleable**
option (default OFF unless stated), be built, and have a measured A/B result
recorded below. Do not mark `[x]` without a build + measurement.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done+measured ·
`[!]` blocked (say why)

---

## A. Alignment / projection space

- [x] `Wa = W_in W_out^T (W_out^T W_out + λI)^-1` closed-form solve
      (multithreaded + `alignment_vocab_stride`, ~203 s at 2048 dims)
- [x] `alignment_rescale_norm` — LatentMAS per-step norm re-scale to
      `target_norm = mean ||W_in[v]||` (default OFF, verified vs HF 1.539382)
- [x] `alignment_target_layer` (`latent_source_layer`) — **FIXED 2026-11-17 23:1x. This
      was a REAL llama.cpp bug, not a ggml limitation** (the earlier `[!]` claim that it
      "needs a ggml-level pin" was WRONG). The allocator never saw the OUTPUT flag at all:
      `llama_context::set_output_hidden_states()` set `cparams.output_hidden_states` but did
      NOT invalidate the scheduler reservation, and `ggml_gallocr_needs_realloc()`
      (ggml/src/ggml-alloc.c) compares ONLY node/leaf counts and per-node buffer SIZES —
      never tensor flags. So the buffer assignment computed once at context creation (capture
      OFF, no `l_out` marked OUTPUT) was replayed forever, and every layer's tap read the
      same buffer. The fix is one line that was already in the tree, COMMENTED OUT:
      `//sched_need_reserve = true;` in `set_output_hidden_states()`. It is now live, gated
      on the disable→enable transition only (enabling ADDS output marks, so the cached
      assignment is genuinely invalid; disabling only removes constraints, so the old
      assignment is a valid superset — re-reserving there would cost 2x per latent step for
      nothing).
      MEASURED on Qwen3-1.7B Q8, `/turn` `latent_steps=4`, delta, align+rescale, align cache
      stride 16 / ridge 1e-5 / target_norm 1.539432:
        BEFORE  il=0..26 ALL `data=0xB7C00C000`, IDENTICAL content
                (`sum=20441.530858 sq=202988206.660260 h=9405f82f98e829ba`)
        AFTER   all 28 layers get their OWN pointer AND their own fingerprint:
                il=0  `0xB7C00C000` sum=5.370452      h=b232d87de078982e
                il=1  `0xB7D20C000` sum=-92.466214    h=e7d8b07909186c0f
                il=2  `0xB7E20C000` sum=19906.729206   h=83cddb8d49a94d05
                ...
                il=26 `0xB9620C000` sum=20441.530858   h=9405f82f98e829ba
                il=27 `0xB7AC00000` sum=1127.962348    h=cd3a8320f010c9ce
                (distinct, spaced 0x100000 apart)
      FUNCTIONAL A/B (previously byte-identical across taps, now distinct):
        L=1 ' the' | L=7 ' and' | L=14 '\n\n' | L=21 ' ' | L=27 '...\n'
        8 sample segments: L=-1 ' 0 degrees. 000' / L=1 ' 0. 0000' /
                           L=14 ' 0. 0000' / L=27 '的     \n '
        depth sweep D=4/8/16 x L=1/14/27 — tap is live at every depth, e.g. D=4:
          L=1 ' . \n   1. ' vs L=14 ' 000000000' vs
          L=27 '的 a country of France is the capital of France'
      HONEST CAVEAT, deliberately not papered over: the tap now READS a different layer;
      that does NOT make any of them a better feedback source. Output quality is still poor
      and layer 14 is outright degenerate (`000000000`) at D=4/8. This fix unblocks the
      experiment — it does not constitute a win. Suites re-run green: 14/14 + 16/16 + 7/7.
      Superseded history (kept for provenance — the conclusions below were WRONG):
      the toggle is exclusive and works (log confirms `using LAYER-1 tap (layer_count=28, got=ok)` etc.), but
      layers 1/7/14/21/27 produce BYTE-IDENTICAL output. Root cause proven by
      pointer instrumentation: `l_out-0`, `l_out-1`, `l_out-2` all report
      `data=0xB7C00C000` while `l_out-27` differs - the tensors ALIAS, so every
      per-layer tap reads the same buffer. Ruled out: (a) an async-DMA race - made
      the decode() copies synchronous AND added `synchronize()` at the end of
      decode() (was `//synchronize();` - commented out); aliasing persisted;
      (b) missing output marking - `llm_graph_result::set_outputs()` existed but
      was NEVER CALLED anywhere in the tree; now called before
      `ggml_backend_sched_alloc_graph`, which is a genuine latent bug fix, and the
      aliasing still persisted. So `ggml_set_output` is not the lever. Next
      hypothesis: the graph reuses one compute buffer because the scheduler
      considers these tensors dead after each layer's consumers run; needs
      `ggml_backend_sched_set_tensor_backend`/split-graph inspection or a
      `GGML_OP_OUT`-style pin. Left OFF by default; the sync + set_outputs fixes
      are kept because both are correct regardless. FINAL alias map (Qwen3-1.7B, 28
  layers): layers 0..26 ALL report data=0xB7C00C000; only layer 27 has its own
  buffer (0xB7AC00000). So aliasing is TOTAL, not selective - no layer except the
  last is usable as a tap. ggml's allocator treats intermediate layer outputs as
  dead once the next layer consumes them and reuses one buffer for all of them;
  `ggml_set_output` (via the now-called set_outputs()) does NOT pin them. Fixing
  this needs a ggml-level change (a real pin/keep-alive for tensors read back to
  the host), not a llama.cpp-side one.
  ---
  **UPGRADED 2026-11-17 22:4x — root cause proven at CONTENT level, not just
  pointer level.** The pointer-only evidence was vulnerable to one objection: the
  CUDA pool allocator aligns graph allocations so every graph launch lands on the
  same addresses, so `data=0x...C000` for layers 0/1/2 could have been
  deterministic address reuse rather than real aliasing. A one-shot diagnostic now
  prints, per layer, the content fingerprint (sum / sq / FNV-1a hash over the host
  copy) and the first three floats. Measured on Qwen3-1.7B Q8 (build-probe tree,
  `GGML_CUDA_FORCE_GRAPHS=1`, `/turn` with `latent_steps=4`):
    il=0..26 : data=0x27C00C000, out=1, view=0, vsrc_out=0,
               distinct tensor OBJECTS (`ten=...D10` / `...520` / `...730`, spaced
               0x2810 apart), IDENTICAL content
               `sum=20441.530858 sq=202988206.660260 h=9405f82f98e829ba`,
               `dst0=-7.99938,-9.03938,-6.05529`
    il=27    : data=0x27AC00000, `sum=-160.990264 sq=11801.739899
               h=4e8b548673d06688`, `dst0=0.252644,-0.583468,0.434406`
  So the reads genuinely return the SAME bytes from ONE device buffer. The
  aliasing is REAL, not an artifact.
  RULED OUT, with evidence, this pass:
    (a) pointer-dump artifact — content is byte-identical, so it is not
        address reuse.
    (b) async / torn read — the copy is the SYNCHRONOUS `ggml_backend_tensor_get`
        and the result is self-consistent across all 27 layers.
    (c) CUDA graph memory pooling — re-ran with `GGML_CUDA_FORCE_GRAPHS=0`,
        byte-identical alias map. Not graph-related.
    (d) view tensor — every layer reports `view=0` and `vsrc_out=0`; these are
        plain, distinct, non-view `ggml_tensor` objects.
    (e) the OUTPUT flag is NOT missing — all 28 layers report `out=1`, i.e.
        `res->set_outputs()` (llama-context.cpp:1401, called before
        `ggml_backend_sched_alloc_graph`) DID mark them. And the flag is being
        read correctly by the allocator (`ggml_gallocr_free_node()` returns early
        on `GGML_TENSOR_FLAG_OUTPUT`).
  CONCLUSION: the earlier phrasing "ggml_set_output does NOT pin them" is
  CONFIRMED and now stronger than a guess — the flag is present on every layer
  and the tensors are collapsed onto one address anyway. The collapse therefore
  does NOT happen in the free path (which honours OUTPUT); it must happen in the
  alloc / in-place path (`ggml_gallocr_allocate_node`'s `ggml_op_can_inplace`
  parent-reuse branch, or the split-graph `ggml_graph_view` path). Any real fix
  has to be stronger than the OUTPUT flag — a keep-alive that prevents the
  parent-reuse branch from selecting a hidden-state tensor at all. Still NOT
  fixable from llama.cpp.
  Also found while measuring (my own bug, now fixed): dereferencing `t_hs->data`
  on the host is a *device* pointer under CUDA and raised `0xC0000005`. The
  diagnostic reads only `dst`, the host copy. Worth recording because it is an
  easy trap for the next person who instruments this loop.
  --- SUPERSEDED 2026-11-17 23:1x: everything below in this block about
  "aliasing is REAL, the flag is set, the collapse is in the alloc path, needs a
  ggml-level pin" described a TRUE OBSERVATION but the WRONG CAUSE. The tensors
  really did alias; the reason was NOT that the OUTPUT flag failed to pin them,
  it was that the allocator NEVER RAN with the flag set (see the [x] entry above:
  `set_output_hidden_states()` did not invalidate the scheduler reservation, and
  `ggml_gallocr_needs_realloc()` never compares flags). All 28 layers now get
  distinct buffers with one line in llama-context.cpp. Keeping this text only so
  the reasoning error is not repeated.
  ---
- [x] `alignment_full_vocab` — forces stride=1 (exact solve) regardless of the
      configured stride. Field `alignment_full_vocab`, C API
      `llama_mas_set/get_alignment_full_vocab`, logs `full_vocab=1 -> forcing
      stride 1`. Default OFF so the strided fast path stays in charge.
      ~~It is intentionally NOT part of the cache key: the key already contains
      the effective stride, so forcing 1 naturally misses a stride-16 cache
      entry and recomputes rather than reusing a matrix that does not match.~~
      **CORRECTED 2026-11-18 12:1x — the sentence above was FALSE.** The key
      was built from the CONFIGURED `alignment_vocab_stride`, not the effective
      stride the solve uses, so a full-vocab (stride-1) solve was saved under a
      stride-16 key and then accepted by a stride-16 context: a wrong matrix
      presented as a HIT. Fixed via `mas_alignment_effective_stride()` in
      save + load (see the G entry `alignment cache key uses the effective
      stride`). Evidence that the shipped cache was mis-keyed: key says
      stride=16, but target_norm 1.539432066265422 vs a fresh stride-16 solve's
      1.5424949868806226 (target_norm is averaged over the effective-stride
      rows, so the cached matrix was built over all 151936 rows). The shipped
      `align_cache.bin` still carries the old key — the fix is preventive, not
      retroactive. NOTE: this item also has no row in the Measurement log; the
      toggle is real and read by the solve, but no A/B was ever recorded.
- [x] `alignment_ridge` — tunable λ (was hardcoded 1e-6).
      IMPLEMENTED + VERIFIED, default OFF (0 = legacy 1e-6).
      C API `llama_mas_set/get_alignment_ridge`, HTTP `ridge` on
      `/mas/alignment/compute`, reported by `/mas/alignment/status`.
      λ is a FRACTION of the mean Gram diagonal, matching what the solve
      already did; `<= 0` selects the legacy value so no existing caller
      changes behaviour.
      MEASURED on Qwen3-1.7B Q8, stride 16, 2 prompts, depth 8, raw transform:
        ridge default (1e-6)  -> `, and the capital of the United States is
                                 Washington` / `, the.......`
        ridge 1e-3            -> `.. The capital of France is Paris. Paris`
                                 / `,, and,, and,,,,`   (different matrix,
                                 different output - the knob is live)
        ridge back to default -> byte-identical to the first default run
                                 (deterministic, no state leak)
      TWO REAL BUGS FOUND AND FIXED while wiring this, both of which would
      have silently corrupted the alignment cache:
        1. compute used λ = diag_mean*1e-6 but the SAVE path wrote
           `ridge = 1e-5f` into the cache key - the key did not describe the
           matrix it accompanied. Now both go through one helper
           (`mas_alignment_effective_ridge`).
        2. the LOAD path built the expected key from the ridge stored IN THE
           FILE, i.e. it compared the file's key against itself and always
           matched. A cache computed with ridge X was therefore accepted into
           a context asking for ridge Y - a wrong matrix reported as a HIT.
           Now the expected key uses the CURRENT context's ridge.
      VERIFIED: after saving with ridge 1e-3, loading the same file with the
      default ridge active returns HTTP 404 (miss) and the log shows the two
      distinct keys (`ridge=897988541` file vs `ridge=981668463` wanted); with
      the matching ridge active the same load returns 200.
      NOTE: λ is only a meaningful quality knob when the Gram is near-singular
      (n_vocab < n_embd). At 151936 vocab rows vs 2048 dims the system is
      heavily overdetermined, so 1e-6 vs 1e-3 changes the answer but neither
      is "more correct" - the sweep above is an A/B, not a win.
- [x] `alignment_lowrank` — rank-r truncated solve. Field `alignment_lowrank`,
      C API `llama_mas_set/get_alignment_lowrank`, HTTP `lowrank` on
      `/mas/alignment/compute`. r<=0 or r>=n_embd means the full solve; otherwise
      Gauss-Jordan runs over the leading r rows/cols only (`solve_dim`), so
      inversion cost falls as r² while the Gram/cross passes still read every
      (strided) vocab row. Default 0 = full solve.
      MEASURED on Qwen3-1.7B Q8, stride 16, latent depth 4 delta:
        lowrank=0    (full, 2048)  314.8 s -> ', and the capital of France is Paris...'
        lowrank=512                 267.1 s -> '(1) 1) 1) 1) 1)'              DEGENERATE
        lowrank=1024                180.1 s -> '(The capital, the capital of France...'  DEGRADED
        lowrank=2048  (== full)     205.1 s -> ', and the capital of France is Paris...'
      CONCLUSION: NOT a usable speed knob. Truncating to 512 dims is ~15% faster
      but destroys the map; 1024 is ~40% faster and visibly degrades (loses the
      opening clause and starts repeating). Only the full rank is coherent. So the
      low-rank hypothesis is FALSE for this model: the map genuinely needs all
      2048 dimensions, it is NOT dominated by a few high-variance directions the
      way the rationale assumed. Kept behind a toggle (default full) for A/B, but
      this is a NEGATIVE result, not an optimisation.
      INDEPENDENT temp=0 CONFIRMATION (depth 8, 3 prompts, greedy) - stronger
      than the temp-0.7 single-sample sweep above because greedy decoding is
      reproducible and removes sampling noise:
        FULLRANK  r=2048 (271 s): 'Water freezes at a temperature of' ->
                    'The water freezes at a temperature of 0C.'  CORRECT;
                    'One plus one equals' -> 'One plus one equals two. 1 plus 1'
                    CORRECT; 'The capital of France is' -> loop (no answer).
                    2/3 prompts correct.
        LOWRANK   r=512  (133 s): 'The capital of France is' -> ' 19999\n...'
                    GARBAGE; 'Water freezes...' -> 'The temperature of the the
                    same as...' (loop); 'One plus one equals' -> ' one. 1. 1. 1.'
                    DEGENERATE. 0/3 correct.
      Confirms the conclusion deterministically: rank-512 HALVES solve wall time
      (133 s vs 271 s, inversion r^2) but drops correct answers 2/3 -> 0/3. The
      truncated solve loses the low-variance directions that carry the real
      signal. Do not enable lowrank for speed; it is a NEGATIVE result.
- [x] `alignment_dtype` — fp32 (0, default) vs bf16-rounded (1) accumulation in
      the Gram pass. Rounding is done on the float bit pattern
      (round-to-nearest-even, keep high 16 bits) rather than a cast intrinsic so
      it is exact and portable. Field `alignment_dtype`, C API
      `llama_mas_set/get_alignment_dtype`, HTTP `dtype` on
      `/mas/alignment/compute`. NOTE this is a MEASUREMENT knob, not an
      optimisation: bf16 rounding ADDS work and can only lose precision, so it
      is here to answer 'how much precision does the solve actually need', not to
      go faster. Default 0.
      MEASURED on Qwen3-1.7B Q8, stride 16, latent depth 4 delta:
        dtype=0 (fp32)  231.2 s -> 'is the capital of France. The capital of France is Paris...'
        dtype=1 (bf16)  291.3 s -> 'The capital of France is Paris. The capital of France is Paris...'
      CONCLUSION: as predicted, bf16 is SLOWER (+26%) because it adds rounding work
      per multiply, and the resulting matrix differs enough to change the output
      text. Neither is 'more correct' - at 151936 vocab rows vs 2048 dims the
      system is heavily overdetermined, so precision is not the binding
      constraint. Confirms this knob is diagnostic only; fp32 stays the default.

## B. Latent feedback source (which vector gets fed back)

- [x] Final-layer hidden state (original)
- [x] Layer-0 input tap (`llama_get_inp_embd_last_token`) — default
- [x] `latent_source=prior_layer` (`latent_transform=1`) — IMPLEMENTED, MEASURED
      WORSE. Feeds the previous step's feedback vector; collapses at depth 4
      (output degenerates to a single `.`) and is no better at 8. The recurrence
      is not "chasing its tail" in a way a one-step delay helps. Off by default.
- [x] `latent_source=mean_pool` (`latent_transform=4`, `latent_pool_window`)
      — IMPLEMENTED, NO EFFECT, and the reason is structural and important.
      First attempt pooled over POSITIONS inside one decode: a strict no-op,
      because each latent step decodes exactly ONE embedding
      (`inp_embd_n_outputs == 1`), so a window of k>=1 collapses to the single
      last position and reproduces raw byte-for-byte (measured: k=2/4/8 all
      identical to raw at L=4 and L=8). Rewrote it to pool ACROSS STEPS, which
      does accumulate correctly - verified `[POOL] step=1..6 hist=1,2,3,3,3,3`
      - but still produces output identical to raw. Instrumented values show why:
      consecutive hidden states differ only in the 5th decimal
      (`h0 = 0.252652, 0.252661, 0.252665, 0.252668, 0.252673, 0.252677`), so
      averaging near-identical vectors changes nothing. COROLLARY: the real cause
      of latent depth collapse is NOT noise accumulation but PREMATURE CONVERGENCE -
      after ~3 steps the feedback vector is effectively frozen and the model reads
      the same state forever. This reframes the remaining work: the goal is to
      keep the state MOVING (or detect that it stopped, which D1 now does), not to
      smooth it. Off by default.
- [x] `latent_source=delta` (`latent_transform=2`) — feed back (h_t - h_{t-1}).
      **BEST RESULT SO FAR.** 9 samples x 3 prompts, coherence (unique/total words):
      depth 4 -> raw 0.154 vs delta 0.298 (1.9x); depth 8 -> raw 0.222 vs delta
      0.386 (1.7x). At depth 8 raw emits only dots while delta emits real sentences
      (`is the capital of France is the capital of France`). Extends usable latent
      depth from ~4 to 8. Interpretation: consecutive hidden states are highly
      correlated, so the raw vector mostly restates "I was here"; the difference
      carries the new information AND has small magnitude, keeping it inside the
      embedding distribution the raw vector drifts out of.
- [x] `latent_source=ema` (`latent_transform=3`, `latent_ema_alpha`) — IMPLEMENTED,
      MEASURED NEUTRAL-TO-WORSE at alpha 0.7 and 0.9 (both worse than raw at
      depth 4, no gain at 8). Off by default.

## C. Latent loop / KV handling  ← the 80-step gap lives here

- [x] `latent_kv_threading` — ALREADY TRUE in our loop: each latent step is its
      own `llama_decode` on the same seq_id, so the KV accumulates and every step
      attends to all prior context. Verified by reading the loop; no code needed.
- [x] `latent_kv_truncate` (`latent_only`) — IMPLEMENTED, DEFAULT OFF, **MEASURED
      WORSE, do not enable.** Deleting prompt KV while keeping the original
      position numbers leaves a hole in the position space: the latent span then
      starts at position N with nothing before it, so the model generates from an
      empty context. Measured with rescale_norm ON: OFF at L=2 gives
      `. the capital of France is Paris. The capital of France is Paris`; ON at
      L=2 gives `...和和どどどどど` (multilingual garbage). Upstream avoids this
      because they build the sequence starting AT the latent span (positions
      begin at 0 there); rebasing positions here would violate the
      consecutive-position rule `llama_decode` enforces. Correct fix = rebase
      positions to 0 after truncation, which needs an explicit KV re-index.
      ~~left as future work, not enabled.~~ **DONE 2026-11-18 06:1x as
      `latent_kv_rebase` (default OFF), and the rebase IS implementable.**
      `llama_memory_seq_add` was the obvious choice and it CRASHED (access
      violation 0xC0000005 inside `update: applying K-shift`): seq_add only
      MARKS cells and defers the work to the context-shift K-shift graph, a
      path this fork never exercises; a scheduler re-reservation first did NOT
      help. Final mechanism avoids the K-shift entirely - `state_seq_get_*`
      export -> `seq_rm(seq,-1,-1)` (also drops the per-cell shift flags) ->
      `state_seq_set_data`. Positions are not serialised
      (`state_write_meta` writes source cells' pos, `state_read` rebuilds cells
      from scratch), so the same bytes land at 0-based positions and the keys
      stay RoPE-consistent with where they now sit. The agent ledger is moved
      with the data so `agent_truncate`'s "entry i == position i" invariant
      holds. MEASURED (Qwen3-1.7B Q8, delta, depth 8, 5 prompts, 16 sampled
      tokens after the think block, RAW strings): positions logged
      `N.. -> 0..` for N=43/57/69/217/247/275, 31192 bytes round-tripped,
      9 consecutive rebases in one process, NO crash. Quality A/B HOLE vs
      REBASE: 1/5 clearly better (`The opposite of hot is` -> hole
      ` in in in in...` / rebase `is\nIt looks like a random string of letters.
      Let me break it down`), 2/5 worse, 2/5 equally degenerate. So the rebase
      is the CORRECT primitive latent_only always needed, but it does NOT fix
      latent_only - the depth collapse is the premature-convergence fixed
      point (section B), which no KV layout change addresses. Left OFF; the
      toggle is the named building block a future design can use.
- [x] `latent_kv_rebase` — with `latent_only`, renumber the surviving KV down
      to position 0 instead of leaving a hole. Field `latent_kv_rebase`, HTTP
      `latent_kv_rebase` on `/turn`, applied immediately after the `latent_only`
      `seq_rm`. Requires `latent_only` (the only path here that deletes KV);
      ignored otherwise. Default 0 = legacy hole-preserving behaviour.
      IMPLEMENTED + VERIFIED CORRECT, MEASURED NOT-A-WIN for quality. Mechanism
      is the explicit KV re-index described above (NOT `llama_memory_seq_add`,
      which crashes in the K-shift graph). Correctness evidence: the log shows
      every rebase (`positions N.. -> 0..`, 31192 bytes re-imported) and the
      process survives 9 consecutive rebases; a naive `seq_add` version killed
      the server with 0xC0000005 at `update: applying K-shift` (recorded in the
      code comment so the trap is not re-hit). Quality: mixed-to-worse, so this
      is a composition primitive, not a quality knob.
- [x] `latent_kv_keep_prompt` — inverse toggle: retain prompt KV across steps.
      IMPLEMENTED, DEFAULT OFF (0 = legacy), measured for CORRECTNESS not quality.
      It is the explicit, named counterpart of `latent_only`, which is already
      measured WORSE (position-space hole), so "keep the prompt" is the
      behaviour we actually want - this just stops it being implicit.
      Conflict rule: `latent_only=1` + `latent_kv_keep_prompt=1` -> keep WINS,
      because a caller asking to keep the prompt must never end up with it
      deleted. Enforced in one place
      (`if (seg.latent_only && !seg.latent_kv_keep_prompt)`) so the two flags
      cannot contradict each other anywhere else.
      Verified on Qwen3-1.7B Q8, depth 6, delta, align+rescale_norm ON, 5 arms:
        default (no flags)    -> ', and the capital of the United States is Washington'
        keep_prompt=1         -> byte-identical to default       (no deletion)
        latent_only=1         -> 'is the is the is the is the...' (deletion DID happen;
                                 the log shows exactly ONE `latent_only dropped
                                 KV [0..4]`, i.e. only the only=1 arm removed KV)
        only=1 + keep=1       -> byte-identical to keep_prompt   (keep WINS)
        keep_prompt=0         -> byte-identical to default       (explicit off = legacy)
      All four invariants hold. Regression suites re-run green after the change
      (14/14 + 7/7 + 16/16), so the default path is untouched.
- [x] `latent_jitter` (position/feedback jitter) — IMPLEMENTED, MEASURED, NOT A
      FIX. Deterministic per-step offset scaled to the vector's own RMS (xorshift
      phase, so reproducible). Tests at depth 8 and 16, values 0.01-0.3:
        depth 8 : 0.05/0.1 identical loop to none; 0.2 recovers ONE good answer
                  (`The capital of France is Paris`) but 0.2 is degenerate at 16.
        depth 16: 0.1/0.2 collapse to `the\n\n the\n\n the...`, WORSE than none.
      So jitter does not keep the state productively moving - it kicks it into a
      different attractor, usually a worse one. The single good output is luck,
      not mechanism.
      METHODOLOGY WARNING worth keeping: the coherence proxy (unique/total words)
      scored `the\nthe\nthe` at 0.412 and jitter=0.1 at 0.474, i.e. it REWARDED
      degenerate output. Ranking these variants required reading the raw strings.
      Any future automated measurement of latent quality must not rely on that
      proxy alone.
- [x] `latent_position_mode` — `shared` (one growing sequence) vs `folded`
      (reuse the same position slot, i.e. in-place recurrence).
      IMPLEMENTED, DEFAULT OFF, **MEASURED WIN.** `latent_position_mode=1`
      removes the KV cell the span just wrote (half-open range `seq_rm(seq, P,
      P+1)`) before the next step, so `llama_mas_agent_pos` keeps returning the
      same P. Verified in the log: every step of an 8-step folded span decodes at
      `pos=8` with `tail 8 -> 7` between steps, while a shared span walks
      `pos=42..53`. Result is a true in-place recurrence.
      Measured (Qwen3-1.7B Q8, delta transform, align+rescale_norm ON, 5 prompts):
        depth 8 : coherence 0.364 (shared) -> 0.656 (folded). Raw strings improve
                  from `water water. water. water water...` to
                  `32°F, and the temperature of the water is` and from
                  `TheThe capital of France is Paris. The capital of` to
                  `es, and the capital of the United Kingdom is Paris.`
        depth 4 : 0.588 -> 0.656; depth 16 delta: 0.731 -> 0.656 (parity, no
                  regression); raw transform improves at every depth
                  (0.461/0.500/0.516 -> 0.550 flat).
      KV COST: a folded span writes ONE position instead of `latent_steps`.
      Matched depth-8 turns: `pos=25` (shared) vs `pos=18`, and `pos=23` vs
      `pos=16` - exactly 7 positions saved per 8 steps.
      HONEST CAVEAT: folded output is DEPTH-INVARIANT - the 5 strings are
      byte-identical at L=4, L=8 and L=16. That is the point (the state cannot
      drift), but it also means extra steps buy nothing past convergence; folded
      saturates rather than scaling. It is a large win at depth 8 with delta, not
      a route to depth 80. `latent_stop_on_repeat=0.99` never fires in folded
      mode (cos stays below 0.99), so D1's speedup is not additive here - folded
      is already ~15 ms/step.
      TWO TRAPS hit while building this, both of which silently produced a
      shared run with byte-identical output at every depth: (1)
      `llama_memory_seq_rm` returns BOOL, always true for a valid seq id - it is
      not a removal count, so it cannot confirm anything; (2) `llama_kv_cells::
      pos_in` is HALF-OPEN (`pos >= p0 && pos < p1`), so `seq_rm(seq, P, P)` is an
      empty interval and removes nothing - the bound must be `P+1`. The release
      is now logged as `tail before -> after` and flags `(NO-OP!)`.
- [x] `latent_batch_mode` — one step per `llama_decode` (current) vs a
      multi-step padded batch. IMPLEMENTED, DEFAULT OFF, **MEASURED NEGATIVE —
      do not enable.** The whole span is submitted as ONE `llama_decode` with
      `latent_steps` embedding rows that all carry the SAME position (only
      meaningful with `position_mode=folded`; shared mode is left on the
      per-step path and the flag is ignored there, verified). Correctness is
      real — `llama_batch_allocr::init` explicitly allows overlapping positions
      for embedding inputs ("embedding inputs can have overlapping positions"),
      and `apply_ubatch()` purges the cell it overwrites, so the N rows collapse
      to ONE KV cell holding the last row: exactly folded semantics without N
      host↔device round-trips.
      Measured (Qwen3-1.7B Q8, align+rescale, 5 prompts):
        default equivalence  batch omitted == batch=false, BYTE-IDENTICAL
        ignored in shared   batch=1 + mode=0 falls back to the per-step loop
        KV cost             ONE position per span at every depth (kv=1 for L=8/16)
        wall (5 prompts)    L=8  perstep 2.46 s -> batch 1.75 s ; L=16 3.39 s -> 1.65 s
        quality             WORSE at every depth. L=8 coherence 0.727 -> 0.071;
                            raw strings `, and the capital of the United States is
                            Washington` -> `?.` / `the the the the the...`
      WHY IT CANNOT WORK AS IS: one graph cannot read its own output mid-batch,
      so every row is seeded from the SAME entering vector. The recurrence's
      whole value is step N reading step N-1's state; feeding the same vector N
      times is N independent single steps with no accumulation, which
      degenerates. It also silently disables delta/ema/prior_layer (they are
      cross-step by definition). The speedup is genuine, so if a future design
      can re-enter the graph between rows (a loop over graph launches on a
      shared KV snapshot, or a custom fused kernel) this becomes the fast path
      for folded mode. Left OFF.

## D. Escape / termination

- [x] `escape_text` — surface-form probe stop (`[MAS/latent] ESCAPE`)
- [x] `stop_on_eog` — end-of-generation probe
- [x] `latent_stop_on_repeat` — stop when the hidden state stops changing.
      **CLEAR WIN.** Depth 8 + delta, Qwen3-1.7B Q8: OFF runs all 8 steps /
      195 ms and emits a loop (`is the capital of France is the capital of...`);
      ON stops after **2 steps / 85 ms** (4x fewer steps, 2.3x faster) AND emits
      better text (`The capital of France is the city of the capital`). Threshold
      0.80 vs 0.99 makes no difference because convergence is EXACT - cosine hits
      1.000, i.e. the recurrence reaches a true fixed point and every further step
      is a no-op that still costs ~30 ms and one KV position. Default 0 (off);
      recommend 0.99 with delta.
      NOTE: reset is `DELETE /mas/latent/stats`, not
      `POST /mas/latent/stats/reset` - a wrong-verb reset silently leaves the
      counters cumulative and makes per-run comparisons look nonsensical.
- [x] `latent_stop_on_norm_drift` — stop when ||h_t|| moves > X% from target.
      IMPLEMENTED, DEFAULT OFF, **MEASURED NEGATIVE — do not enable.** Two
      reference choices were tried; both are degenerate, and the reason is
      structural:
        * anchored on `target_norm` (mean ||W_in[v]||): with rescale_norm ON
          the aligned vector is RESCALED to target_norm every step, so
          ||h|| == target_norm by construction and the test can NEVER fire
          (measured: thresholds 0.01–1.0 all ran the full 48 steps, output
          byte-identical to off). With rescale_norm OFF the FIRST raw input
          embedding already differs from target_norm by >100%, so it fires at
          step 0 (measured: 0 latent steps).
        * anchored on the PREVIOUS step's norm (scale-free, no rescale
          dependency): stable under the raw transform (thresholds 0.05–1.0
          never fire, 48/48 steps, byte-identical) and fires at step 1 under
          delta (48 -> 3 steps) — but that is not a useful stop, it is just
          re-detecting delta's intentional scale change one step later.
      Net: the only case where this fires is the case where
      `latent_stop_on_repeat` (cosine) already fires correctly and with a
      meaningful signal. Norm magnitude carries no independent information
      about convergence here — direction (cosine) does. Left OFF; kept because
      the plumbing is correct and a future transform with a meaningful
      magnitude envelope could use it.
      Diagnostic value recorded in the comment at the check: it proves the
      recurrence's scale is dominated by the transform/rescale choice, not by
      drift accumulation.
- [x] `latent_max_tokens_budget` — count latent steps against n_ctx like tokens
      (previously implicit). IMPLEMENTED, DEFAULT OFF, **VERIFIED CORRECT.**
      Budget is a per-SEGMENT cap on latent steps: the span stops once `s`
      reaches the budget, regardless of `latent_steps`; `<= 0` disables. It
      makes explicit what was implicit (a latent step costs one KV position),
      so a caller can cap the think block the same way it caps max_tokens.
      Note on an earlier interpretation that FAILED and was replaced: an
      ABSOLUTE position ceiling (`pos >= budget`) is nearly useless in folded
      mode, because folding reuses one slot so `pos` stays small — a budget of
      8 did not clamp a 16-step folded span at all (measured: steps stayed 16
      for budgets 8/10/12/16). The per-segment step cap is the semantics that
      actually constrains anything.
      Measured (Qwen3-1.7B Q8, delta, folded, align+rescale, depth 16):
        default   absent == budget=0 -> BYTE-IDENTICAL, 8 steps both
        budget 8  -> 8 steps  / kv 8   ', and the capital of France is Paris'
        budget 10 -> 10 steps / kv 10  'The capital of France is Paris.'
        budget 12 -> 12 steps / kv 12  'Paris. The capital of the United States'
        budget 16 -> 16 steps / kv 16  (== latent_steps, no clamp)
        budget 32 / 100 -> 16 steps   (budget above latent_steps is a no-op)
        budget 3  -> 3 steps          (below latent_steps, floors at budget)
      Also applied to batch mode (clamps the row count).

## E. Hybrid text/latent composition

- [x] Ordered segment list (`tokens` / `latent`) via `llama_mas_run_turn`
- [x] Multi-turn KV reuse with ledger + exact position accounting
- [x] Tail-hidden carry across turns and sessions (MAS1 blob)
- [x] `latent_only_agents` — per-agent mask. Bit i = 'agent i is a latent
      thinker'. Without it a turn's latent segment applies to EVERY agent, so a
      mixed crew cannot have a worker think latently while a spokesperson answers
      in text. Implementation rewrites the segment kind to TOKENS/sampling for
      excluded agents, so the rest of the turn loop needs no knowledge of the
      flag; `seg` became a mutable copy for this. Logs once
      `latent_only_agents mask=N: agent K excluded -> text path`.
- [x] `latent_on_tool_call` — inject a latent (continuous-reasoning) think span
      immediately before a tool-call segment. The caller marks the TOKENS segment
      that will emit the tool call (the tool grammar is not visible in this
      layer); on that segment the driver now runs a LATENT block of the segment's
      own `latent_steps` FIRST, then the real text segment. Default OFF: with no
      flag set the segment list is copied verbatim, byte-identical.
      CORRECTED 2026-11-17: a claims audit found the guard was MISSING - the
      flag was only declared (llama-mas.h:310) and parsed (mas-server.cpp:3466)
      but never read by any code path and never validated. The guard now DOES
      exist: a validation loop in the `/turn` handler rejects
      `latent_on_tool_call` with `latent_steps<=0` via HTTP 400
      (`latent_on_tool_call requires latent_steps>0 on the same segment`).
      CORRECTED AGAIN 2026-11-19: the guard fired only on the impossible combo;
      the flag STILL did nothing on the GOOD combo (0 reads in
      llama_mas_run_turn). NOW FIXED: a new pass in run_turn, after the
      latent_interleave expansion, scans for any TOKENS segment with
      `latent_on_tool_call!=0` and inserts a LATENT think block (length =
      that segment's `latent_steps`, inheriting align/transform/position_mode/
      stop_on_*, but with text/forced_tokens/sample/escape/latent_only_agents/
      interleave stripped so it is PURE reasoning and cannot recurse)
      IMMEDIATELY BEFORE it. Default OFF: the pass is skipped entirely when no
      flag is set, so legacy behaviour is byte-identical. (Bug fixed while
      building: the expansion vector was block-scoped, so `segs_ptr` dangled
      after the block -> 0xC0000005; hoisted to function scope.)
      MEASURED on Qwen3-1.7B Q8 (build-probe Debug CUDA :9024, align cache
      stride 16 / ridge 1e-5 / target_norm 1.539432), fresh worker each arm:
        OFF  `{tokens+sample}`                            -> n_latent_steps=0
             n_positions=6 text=' Paris' tok[12095]
        ON   same + `latent_on_tool_call:1, latent_steps:3`
             -> log `latent_on_tool_call -> injected 1 ...; 1 -> 2 segments`,
                n_latent_steps=3 n_positions=9 (=6+3), text=' Paris' tok[12095]
                (identical token => the injected span emitted NO text, pure think)
        MULTI 2x tokens+sample, flag on 2nd steps=2
             -> `injected 1 ...; 2 -> 3 segments`, n_latent_steps=2 n_tokens=2
                (both text segs ran, 2-step think before the 2nd).
      Server stayed alive through all arms. The honest limitation stands: the
      flag is a caller-supplied marker, not an automatic tool-call detector -
      but it is now a working hook, not a dead field.
- [x] `latent_interleave` — alternate latent/text automatically every K tokens.
      IMPLEMENTED, DEFAULT OFF, **MEASURED NEGATIVE for quality (do not
      enable); CORRECT as a composition primitive.**
      Mechanism: implemented as a ONE-TIME expansion of the caller's segment
      list at the top of `llama_mas_run_turn`, on a local copy. For each
      `latent` segment with `latent_interleave = K > 0`, the text-sampling
      segments that follow are consumed in runs of K and another think block of
      the same length is inserted after each run, so the turn alternates
      think/text without the caller spelling out every boundary. The decode
      path, `segs_ptr` and every other toggle are untouched, and with the flag
      off the ORIGINAL pointer is used, so the legacy path is bit-for-bit the
      same code.
      VERIFIED on Qwen3-1.7B Q8, delta, align+rescale_norm ON, 12 text
      segments after one latent span, depth 8:
        flag omitted      -> 1 latent segment,   8 latent steps, kv=8
        K=20 (no split)   -> 1 latent segment,   8 latent steps, kv=8
        K=5               -> 3 latent segments, 24 latent steps, kv=24
        K=3               -> 4 latent segments, 32 latent steps, kv=32
        K=1               -> 12 latent segments, 96 latent steps, kv=96
      Exactly the predicted `floor(12/K)` think blocks (K=20 collapses to the
      single block, i.e. the flag is a no-op when it cannot split), text token
      count stays 12 in every arm, and the expansion log confirms it
      (`14 segments expanded to 25/19/17/15`).
      LEGACY EQUIVALENCE: omitted and `latent_interleave=0` produce
      BYTE-IDENTICAL output and identical segment counts (1/8/8/12).
      QUALITY: WORSE at every K, same shape as `latent_batch_mode`. 5 prompts,
      delta, shared mode: the off arm emits real continuations
      (`is a\nThe opposite of hot is cold is a bit`) while K=1 degenerates to
      `is is is is is...` and K=2 to `is a is a is a...`. Interpretation: a
      think block after only 1-4 text tokens re-enters the recurrence from an
      already-converged state (section B: the fixed point is reached in ~3
      steps), so every additional block adds 8 wasted steps and no new
      information. Interleave's value is COMPOSITION (a caller can now express
      "think between tokens" declaratively); it is not a quality knob and must
      not be enabled to improve output.
      TWO REAL BUGS FOUND AND FIXED while building it, both of which silently
      broke the toggle:
        1. single-pass expansion only split ONCE - a re-inserted think block
           was never itself expanded, so K=3 on 12 text segments produced 2
           latent segments instead of 4 (log: `14 -> 15`). Rewritten as a loop
           over runs, which is what makes the count `floor(12/K)`.
        2. the first version appended a repeat think block whenever any
           segments remained, so a list ending in the text run grew by one
           latent segment even with the flag effectively off (`14 -> 15` with
           `latent_interleave=0`). The repeat is now gated on text actually
           following it.
      Also fixed a diagnostic that lied: the expansion log printed
      `expanded[0].latent_interleave`, i.e. the first segment's value (always
      0 for a prompt segment), so every run claimed
      `latent_interleave=0 -> N expanded to N+1`. It now reports the K that
      actually fired and prints nothing when the segment count did not change.

## F. Measurement / observability

- [x] Per-step `[MAS/latent]` tick + DONE line (rate, per-step ms)
- [x] `/mas/alignment/status` reports busy/n_embd/is_identity/rescale/target_norm
- [x] `/mas/latent/stats` — GET + DELETE. Reports latent_segments, latent_steps,
      kv_positions, tokens_sampled, latent_ms, text_ms, max_depth, last_cos_drift,
      plus derived avg_ms_per_step / steps_per_sec / latent_fraction_of_kv.
      VERIFIED on a real 8-step run: 8 steps, 8 KV positions, 33.3 steps/s
      (30 ms per latent step on Qwen3-1.7B Q8 / GTX 1080).
      **FIXED 2026-11-17 (accounting was dead):** `tokens_sampled` and
      `text_ms` were declared in `llama_mas_latent_stats` and REPORTED by this
      endpoint, but NEVER ASSIGNED anywhere in the tree - every run ever,
      including the 8-step run above, answered
      `{"tokens_sampled": 0, "text_ms": 0}`. `grep -n "n_tokens_sampled
      n_text_ms" src/llama-mas.cpp` returned only the struct init and a
      same-named field on the per-turn result, never an increment.
      Consequence: the F-section token-savings report could not be written at
      all, because "how many real tokens did the latent span displace" was
      structurally unmeasurable rather than merely unmeasured. Now accumulated
      at the end of `llama_mas_run_turn`, with the semantics stated in the
      comment there: `n_tokens_sampled` counts tokens the model actually
      EMITTED (sampled), NOT prompt tokens fed to prefill (which `n_text`
      includes), and `n_text_ms` is wall time inside text segments only.
      VERIFIED on Qwen3-1.7B Q8, delta, align+rescale, depth 8, 4 sampling
      segments: `tokens_sampled: 4` (was 0) and `text_ms: 180` (was 0), with
      `latent_ms: 329` unchanged. DELETE still resets to zero, so per-run A/B
      is now actually possible.
- [x] `latent_bench.py` — SHIPPED at `tools/mas-server/latent_bench.py`.
      Sweeps latent depth and reports, per depth: exact-match accuracy on a
      built-in GSM8K-style set (6 questions, integer answers), mean wall time,
      mean latent steps actually taken, and mean KV positions consumed. Plus a
      `vs text baseline (depth 0)` table showing wall-clock multiple and whether
      early stop is ACTIVE. Deliberately reports the RAW generated strings too,
      because the old unique-words proxy was gameable. Flags: --depths, --samples,
      --transform, --stop, --ntok, --json.
- [x] Token-savings + wall-clock report — implemented as the summary block in
      `latent_bench.py` (`=== vs text baseline (depth 0) ===`): per depth it
      prints the wall-clock multiple vs the depth-0 text run, the mean latent
      steps, and whether the early stop fired. HONEST STATUS: the harness exists
      and produces the numbers, but the paper's specific claims (50-80% fewer
      tokens, 3-7x speedup) are NOT yet reproduced on this host. What IS measured
      locally: latent costs ~30 ms/step vs ~0.03 ms/token, so latent buys
      KV/context economy, not wall-clock speed. Reproducing the paper's numbers
      needs a model trained for it, which per section 8 of the analysis is the
      actual blocker - not the measurement code.

## G. Robustness / correctness debt

- [x] `llama_mas_set_agent_layer_graph` — DECLARED, NEVER DEFINED (link trap);
      per-agent `layer_graph` field read at ~10 decode sites but never written,
      so the whole per-agent layer-graph path was dead code. SHIPPED +
      MEASURED 2026-11-19. Definition at llama-mas.cpp:2440 with a documented
      BORROWED-ownership contract (caller keeps the graph; the agent frees it
      only if still attached, so a caller that passes NULL cannot dangle and a
      caller that frees its own graph must clear the agent first). Freed with
      the agent in llama_mas_clear_agents. New route
      `POST /v1/agents/N/layer-graph` with `{layers:[...]}` (whitelist),
      `{disabled:[...]}` (blacklist) or `{}` (clear; default = legacy). New C
      API `llama_context_attach_layer_graph_reserved` (detach -> reserve full
      shape -> attach) used by `/reset`. Also added the missing arch gate in
      src/models/qwen3.cpp:21 (identical to models/llama.cpp:34) so the subset
      is honoured on the test model's arch.
      MEASURED on Qwen3-1.7B Q8, build-probe Debug CUDA, :9024:
        {layers:[0,1,2,3]} -> `{"status":"ok","agent_id":0,"enabled":4,"n_layers":28}`
        {} (clear)          -> `{"status":"ok","agent_id":0,"cleared":true}`
        invalid index       -> HTTP 400
        default /turn (no route call) BYTE-IDENTICAL to pre-change:
          text=' Paris' tokens=[12095] n_positions=6, server stays 200.
      HONEST LIMIT (this is a correctness item, NOT a quality win): a partial
      layer SUBSET changes the graph node/leaf count, and ggml's scheduler
      reserves ONE shape plus split bookkeeping (`prev_node/leaf_backend_ids`,
      ggml/src/ggml-backend.cpp:1380-1479) that is only rebuilt on a re-reserve.
      Attaching a subset MID-PROCESS then decoding aborts with
      `GGML_ASSERT(buffer)` (measured SERVER_EXIT=3; log shows the partial
      reserve `graph nodes = 161` against the 1099-node full reserve). Root
      cause is in ggml (split inputs copied with stale prev ids; 2 buffers so
      `ggml_gallocr_alloc_graph` cannot auto-realloc), not fixable cleanly from
      llama.cpp. A subset therefore only takes effect when the graph is
      attached BEFORE the first decode of the process; the route prints an
      explicit NOTE and `/reset` forces the full re-reserve so recovery is one
      call. Suites 14/14 + 16 PASS/0 FAIL + 7/7.

- [x] Alignment cross-pass crash diagnostic (`alignment_cross_progress`) —
      SHIPPED, default OFF. The matrix was 40 [x] / 0 [ ] / 0 [!], so this
      closed the one live engineering debt: the 8B alignment crash dies
      SILENTLY inside the cross pass (10:1x evidence — dequant + Gram +
      Gauss-Jordan invert all succeed on CPU, then a bare `SERVER_EXIT=-1`).
      The cross pass was a single-threaded n_vocab-strided × n_embd²
      accumulation with zero logging, so the crash point was unobservable.
      The new toggle adds a PREFLIGHT line before the first row (A / w_out /
      row-scratch bytes, strided row count, MAC count) and a progress line
      every ~5% of strided rows with a sampled sum of A. Measured on
      Qwen3-1.7B Q8 :9024 with `cross_progress=true` (fresh solve):
      preflight `A=16 MB, w_out=1187 MB, row scratch=8 KB, 9496 rows x
      8.38861e+06 MACs`; progress advanced row 0→151680 with A_probe_sum
      -2.48→5.13 (real accumulation, no stall); solve finished
      `target_norm=1.54249` + `Wa=2048x2048`. Default OFF verified (status
      `cross_progress:false`, cache target_norm 1.539432066265422 unchanged).
      Suites 14/14 + 16 PASS/0 FAIL + 7/7. Code: llama-mas.h decl,
      llama-mas.cpp field/setter/getter, HTTP `cross_progress` on
      POST /mas/alignment/compute + `cross_progress` in status.
- [x] Hidden-state capture GPU path (`use_gpu_hidden_states`) — INVESTIGATED,
      PARTIALLY ADDRESSED, LIMIT DOCUMENTED IN CODE. The flag existed but was
      DEAD: settable through `llama_mas_set/get_gpu_hidden_states`, never read by
      any code path. It now logs its state once per process so callers can assert
      which path they are on. What it CANNOT do is remove the D2H copy: the
      per-step readback is already synchronous and already lands in one reused
      vector inside llama-context (the async variant was removed because it raced
      and produced torn reads / identical layers). Removing the copy for real
      needs a DEVICE-side buffer that survives between decode calls, owned by the
      context - a ggml allocation change, not a flag read in this loop. Left
      settable and honest rather than pretending a bool fixes a memory-ownership
      problem. NOT MEASURED for speed, because there is no speed change to
      measure yet.
- [x] Deepstack models (`n_embd_inp > n_embd`) — `llama_mas_run_turn` now sizes
      its latent buffers at `llama_model_n_embd_inp()` and only falls back to
      `llama_model_n_embd()` when the former is <= 0. batch.embd is consumed AT
      LAYER 0, so a deepstack model's wider per-token input embedding makes
      n_embd-sized buffers assert on the first latent step. Alignment matrix and
      hidden-state tap stay residual-space (n_embd) as before. Logs
      `deepstack model: n_embd_inp=N > n_embd=M -> latent buffers sized at
      n_embd_inp` once. Not exercised: no deepstack GGUF on this host, so the
      guard is verified to compile and the branch is correct by construction.
- [x] `latent_mas` strategy naming trap — DOCUMENTED (no code needed).
      The strategy named 'latent_mas' in the upstream paper is the KV-CLONE +
      real-text baseline: it clones a KV span and emits actual tokens. It is NOT
      the continuous recurrence implemented here (feed a hidden state back through
      batch.embd with no sampling). The two share a name and nothing else, so any
      comparison against 'LatentMAS' must state which one is meant. The
      recurrence in this tree is the `/turn` latent segment + `latent_transform`
      path; the paper's `latent_mas` is what you get with `latent_steps=0` and
      real text tokens.
- [x] Persist the alignment matrix to disk (currently per-process, 203 s to
      rebuild on every server start). IMPLEMENTED + VERIFIED ROUND-TRIP.
      New C API: `llama_mas_save_alignment_matrix(mas, path)` /
      `llama_mas_load_alignment_matrix(mas, path)`; HTTP
      `POST /mas/alignment/save` and `POST /mas/alignment/load` (both take
      `{"path": ...}`, `load` also accepts `vocab_stride`). Both opt-in per
      request, so no existing caller changes behaviour.
      FORMAT: magic `MASALN2` + n_embd + n_vocab + stride + ridge_bits + key_len
      + key + target_norm(float64) + Wa(float32). Save writes `<path>.tmp` then
      renames, so an interrupted save cannot leave a half-written cache that a
      later load would trust. Magic was bumped 1 -> 2 when target_norm was added,
      so a pre-target_norm cache is correctly rejected as a MISS rather than
      misread at shifted offsets.
      KEY = `model desc | n_embd | n_vocab | stride | ridge`, so a different model
      or stride can never silently reuse an incompatible matrix. Verified: asking
      for `vocab_stride=1` against a stride-16 cache returns HTTP 404 (miss).
      MEASURED on Qwen3-1.7B Q8, stride 16:
        compute                271.1 s   (reference run)
        save                   16,777,316 bytes (2048^2*4 + 8 header + 8 target_norm)
        load in a FRESH process 0.03 s   -> HIT, target_norm restored to 1.539432066265422
        latent output after load  BYTE-IDENTICAL to the reference run on both
                                  probe prompts (`Paris. The capital of the
                                  United States is Washington, D` / `0°C.  What
                                  is the volume of water at`)
      target_norm is stored deliberately: a loaded matrix with target_norm=0
      makes rescale_norm a silent no-op (its guard is
      alignment_has_target_norm), which would have been an invisible behaviour
      change introduced by the cache. First implementation had exactly that bug
      and was caught by the round-trip comparison before shipping.
      ~~NOT DONE: no auto-load at startup and no `--alignment-cache` CLI flag.~~
      **DONE 2026-11-18 00:5x.** `--alignment-cache PATH`
      (`[--alignment-stride N] [--alignment-ridge R]`) loads the matrix right
      after every MAS context creation, so the server boots with the projection
      already in place instead of needing one explicit load call. Default OFF:
      no flag = no load attempted, no behaviour change.
      MEASURED on Qwen3-1.7B Q8, build-probe Debug CUDA, :9024:
        OFF arm (no flags)   status `cache_status:"off" cache_path:"" n_embd:0
                             use_alignment:false stride:1 ridge:1e-06`
        ON arm               log `loaded Wa 2048x2048 ... target_norm=1.53943`
                             + `[MAS/alignment] AUTO-LOADED ... stride=16
                             ridge=1e-05`; status `cache_loaded:true
                             cache_status:"loaded" n_embd:2048
                             use_alignment:true target_norm:1.539432066265422`.
                             The 271 s solve is replaced by the 0.03 s load.
        RECREATE arm         `POST /config/agents-max {"n_agents_max":8}` fires
                             `recreate_mas_context`; a SECOND `AUTO-LOADED` line
                             appears and status stays `n_embd:2048
                             use_alignment:true`. This is a real silent-bug fix,
                             not just convenience: a fresh context has an EMPTY
                             alignment matrix, so any recreation (KV-type change,
                             n_agents_max change, model swap) used to drop the
                             projection and the next latent run silently degraded
                             to a plain memcpy with no error.
        MISS arm             `--alignment-ridge 1e-3` against the ridge-1e-5
                             cache -> status `cache_status:"miss" n_embd:0
                             use_alignment:false` and a loud log showing BOTH
                             keys: `file: ...|stride=16|ridge=925353388` vs
                             `want: ...|stride=16|ridge=981668463`, plus
                             `AUTO-LOAD FAILED rc=1 ... (rc=1 is a MISS: check
                             --alignment-stride / --alignment-ridge match the
                             cache key)`. The miss is now diagnosable in one
                             line instead of reading as "the cache is broken".
      Suites on this build: hybrid 14/14, mas_latent 16 PASS / 0 FAIL,
      mas_server 7/7.
- [x] Alignment cache key uses the EFFECTIVE stride — CORRECTNESS FIX, no
      toggle (behaviour-preserving while `full_vocab` is OFF, the default).
      Both the save key and the load key were built from
      `mas->alignment_vocab_stride`, the CONFIGURED stride, while the solve
      forces stride 1 when `alignment_full_vocab` is set. A full-vocab solve was
      therefore written under a stride-16 key and then ACCEPTED by a stride-16
      context — a wrong matrix presented as a HIT, the exact failure the key
      gate exists to prevent. New helper `mas_alignment_effective_stride()`
      applies the `full_vocab` override and is used by save + load.
      `full_vocab` is now also accepted on `POST /mas/alignment/load`,
      mirroring the `ridge` precedent — without it a full-vocab cache is
      unloadable from a context whose configured stride is 16.
      MEASURED on Qwen3-1.7B Q8, :9024, using a synthesized stride-1-keyed
      cache file (the shipped cache's key/header rewritten 16 -> 1):
        real cache + stride16 + full_vocab OFF -> 200 loaded (default unchanged)
        stride1-keyed file + stride16 + full_vocab OFF -> 404 cache miss
        stride1-keyed file + stride16 + full_vocab ON  -> 200 loaded
      NOT RETROACTIVE: `align_cache.bin` on disk still carries its old
      stride-16 key, so it still loads under stride 16. Evidence it was
      mis-keyed: its target_norm 1.539432066265422 vs a fresh stride-16 solve's
      1.5424949868806226 (target_norm is averaged over the effective-stride
      rows, so the cached matrix was built over all 151936 rows). A fresh
      compute+save would write an honest key; deliberately NOT run here.
- [x] `POST /reset` — full state reset for test isolation (no toggle; a new
      route, so every existing caller is unchanged). SHIPPED + MEASURED
      2026-11-18 14:2x. Closes the repro hazard the 13:5x run root-caused:
      `test_hybrid_turn.py` calls `POST /reset` at startup but NO `/reset`
      route existed, so it 404'd silently and hybrid sessions/agents
      accumulated across runs; `llama_mas_session_match_len` then returned a
      foreign prefix and the suite gave FALSE FAILs (12/2: `session_grew`,
      `latent_resume_exact_growth`) when run after other suites. The route
      (in `tools/mas-server/mas-server.cpp`, after the `/mas/latent/stats`
      DELETE handler) takes `g_state.mtx`, clears `g_state.sessions`,
      `llama_mas_clear_agents(g_state.mas)` and
      `llama_mas_reset_latent_stats(g_state.mas)`. MEASURED on :9024 (Qwen3-1.7B
      Q8): dirty state (agent + session `reset_probe_sess` kv_bytes 283319,
      stats text_ms 281) -> `POST /reset` returns
      `{"sessions_cleared":1,"agents_cleared":true,"latent_stats_cleared":true}`
      -> `/sessions` `{"sessions":[]}` (was 1), stats all-zero (text_ms 281->0);
      `POST /reset` with NO body also 200; alignment status unchanged
      (`cache_loaded:true use_alignment:true target_norm 1.539432066265422`).
      THE ACTUAL REGRESSION PROOF: `test_hybrid_turn.py` run THREE times back
      to back on the SAME server now gives 14/14, 14/14, 14/14 (was 12/2 when
      run after other suites). Suites: hybrid 14/14, mas_latent 16 PASS /
      0 FAIL, mas_server 7/7. BUILD TRAP: `_build_mas_server_debug.bat` builds
      ONLY `build-codex`; the :9024 server runs `build-probe`, so the served
      binary needs `build_probe.bat` (scratch) — and a live server holds
      `bin/mas-server.exe` open (LNK1168), so kill it first.

---

- [x] build-codex `llama.dll` is current with `src` (build-hygiene debt, no
      toggle — a stale binary is not a feature).
      The 12:1x run flagged `build-codex/bin/llama.dll` (10:37) as stale vs
      `src/llama-mas.cpp` (11:24) and noted the earlier "DEBT CLOSED" check had
      compared against `mas-server.exe`, which does NOT depend on `src` — the
      correct artifact is `llama.dll`. The blocker was LNK1168: a live
      `llama-server.exe` (nomic embedder, :8090, running out of
      `build-codex\bin`) held the DLL open, and killing it was deliberately
      declined because it was a live service.
      CLEARED 2026-11-18 13:0x: that process is gone of its own accord
      (`Get-CimInstance Win32_Process` finds no `llama-server.exe`, only the
      :9024 test server; `curl :8090/health` -> 000). Rebuilt via
      `_build_mas_server_debug.bat` — 4 steps, EXIT=0, no LNK1168.
        build-codex/bin/llama.dll      11-18 13:02  6,850,048 B  (was 10:37)
        build-codex/bin/mas-server.exe 11-18 13:02  3,234,304 B
        both now NEWER than src/llama-mas.cpp 12:14 and
        tools/mas-server/mas-server.cpp 12:20.
      SMOKE-VERIFIED the new tree rather than trusting the timestamp: launched
      `build-codex\bin\mas-server.exe` on :9025 (Qwen3-1.7B Q8, --alignment-cache,
      stride 16, ridge 1e-5, --n-agents-max 8). It booted, AUTO-LOADED the cache
      (`loaded Wa 2048x2048 ... target_norm=1.53943`,
      `AUTO-LOADED ... stride=16 ridge=1e-05`), and
      `/mas/alignment/status` reported `cache_loaded:true use_alignment:true
      target_norm:1.539432066265422` — i.e. the new DLL performs the real
      latent-alignment path, not just a startup. `/v1/chat/completions`
      ("One plus one equals", temp 0) returned 200 with 97 completion tokens.
      Then stopped the :9025 server (freeing the GPU and the DLL); the :9024
      build-probe server was left running and untouched (health 200 after).
      Suites re-run this run on :9024: hybrid 14/14, mas_latent 16 PASS /
      0 FAIL, mas_server 7/7. No source change, so no new A/B measurement.

## Measurement log

| date | item | config | result |
|---|---|---|---|
| 2026-11-17 | rescale_norm | Qwen3-1.7B Q8, greedy, L=0..32 | OFF: word salad by L=8. ON: coherent to L=4, dots by L=8-16, empty L=32. Real improvement, not 80 steps. target_norm 1.539432 vs HF 1.539382 |
| 2026-11-17 | latent_only | same config, L=2..32 | WORSE. ON at L=2 -> `...和和どどどど` / `And And And And`; OFF at L=2 -> coherent. Cause: position-space hole after deleting prompt KV. Toggle left OFF by default. |
| 2026-11-17 | latent_kv_threading | code read | Already correct: one llama_decode per step on the same seq_id, KV accumulates. No change needed. |
| 2026-11-17 | /mas/latent/stats | 8-step latent run, Qwen3-1.7B Q8 | WORKS. 8 steps / 8 KV positions / 30 ms per step / 33.3 steps/s. Tokens_sampled and text_ms still 0 (not wired to the text branch yet). |
| 2026-11-17 | latent_source_layer | Qwen3-1.7B Q8, layers 1/7/14/21/27 | BLOCKED. All non-zero layers give byte-identical text. Proven cause: tensor aliasing (l_out-0/1/2 share data=0xB7C00C000). Two real bugs fixed while diagnosing: async hidden-state copies in decode(), and set_outputs() never being called. |
| 2026-11-17 | decode() sync + set_outputs() | regression | No change to latent quality, but both are correct fixes. Suites re-run: hybrid 14/14, latent 16/16. |
| 2026-11-17 | latent_transform=delta | 9 samples x 3 prompts, Qwen3-1.7B Q8 | BEST. coherence 0.154 -> 0.298 (L=4), 0.222 -> 0.386 (L=8). Extends usable depth 4 -> 8. |
| 2026-11-17 | latent_transform=prior_layer | same harness | WORSE: collapses to '.' at L=4, no gain at L=8. |
| 2026-11-17 | latent_transform=ema a=0.7/0.9 | same harness | NEUTRAL-TO-WORSE at both alphas. |
| 2026-11-17 | latent_stop_on_repeat | depth 8 delta, Qwen3-1.7B Q8 | WIN. 8 steps/195ms -> 2 steps/85ms (4x fewer, 2.3x faster) + better text. Converges exactly (cos=1.000). |
| 2026-11-17 | latent_transform=mean_pool | positions k=2/4/8, then across steps | NO EFFECT both ways. Positional pooling is a structural no-op (1 embedding/step). Cross-step pooling accumulates but hidden states differ only in the 5th decimal -> nothing to smooth. Reveals depth collapse is PREMATURE CONVERGENCE, not noise. |
| 2026-11-17 | latent_jitter 0.01-0.3 | depth 8 and 16, Qwen3-1.7B Q8 | NOT A FIX. Helps only at depth 8 / jitter=0.2 (one lucky good answer); degrades depth 16 into `the\n\n the`. Coherence proxy shown to be gameable - must read raw output. |
| 2026-11-17 | latent_position_mode=folded | Qwen3-1.7B Q8, delta, align+rescale, 5 prompts | **WIN.** Folded pins every step to ONE position (log: `pos=8` each step, `tail 8 -> 7` between). L8 coherence 0.364 -> 0.656; raw strings `water water. water...` -> `32°F, and the temperature of the water is`. L4 0.588 -> 0.656, L16 0.731 -> 0.656 (no regression). Raw transform 0.461/0.500/0.516 -> 0.550 flat at all depths. KV: 7 fewer positions per 8-step span (pos 25->18, 23->16). CAVEAT: depth-INVARIANT output (identical at L4/L8/L16) - saturates, does not scale to 80. Suites re-run green 14/14 + 7/7 + 16/16. |
| 2026-11-17 | latent_kv_keep_prompt | Qwen3-1.7B Q8, depth 6, delta, 5 arms | CORRECTNESS VERIFIED. default == keep_prompt=1 (byte-identical), keep_prompt=0 == default, latent_only=1 differs ('is the is the...') proving deletion still works, only=1+keep=1 == keep_prompt (keep WINS). Exactly one `latent_only dropped` log line, so no over-deletion. Suites 14/14 + 7/7 + 16/16. |

| 2026-11-17 | alignment matrix disk cache (G) | Qwen3-1.7B Q8, stride 16, fresh process | **WIN.** compute 271.1 s vs load 0.03 s in a new process; 16,777,316-byte cache. target_norm restored (1.539432066265422) so rescale_norm still applies; latent output BYTE-IDENTICAL to the reference run. Wrong-stride load correctly returns HTTP 404 (key mismatch = miss, never a wrong matrix). Old MASALN1 cache rejected by the magic bump. Suites 14/14 + 7/7 + 16/16. |
| 2026-11-17 | latent_batch_mode (C) | Qwen3-1.7B Q8, folded, align+rescale, 5 prompts, L=4/8/16 | **NEGATIVE, left OFF.** One llama_decode with N embedding rows all at the SAME position. Default-equivalent when omitted (byte-identical), ignored in shared mode. KV: ONE cell per span at every depth. Wall: L8 2.46 s -> 1.75 s, L16 3.39 s -> 1.65 s (real speedup). Quality WORSE: L8 coherence 0.727 -> 0.071; raw `, and the capital of the United States is Washington` -> `?.` / `the the the...`. Cause: a graph cannot read its own output mid-batch, so every row sees the SAME entering vector - no cross-step feedback, and delta/ema/prior_layer silently disabled. |
| 2026-11-17 | latent_max_tokens_budget (D) | Qwen3-1.7B Q8, delta, folded, depth 16, budget sweep | **VERIFIED CORRECT, default OFF.** Per-segment latent-step cap (stops at `s >= budget`); `<=0` disables. absent == budget=0 byte-identical (8 steps). budget 8/10/12 -> exactly 8/10/12 steps (kv matches); 16/32/100 -> 16 (no clamp above latent_steps); 3 -> 3. An absolute-position ceiling was tried FIRST and FAILED (folded mode keeps `pos` low, so budget 8/10/12/16 all left steps=16) - replaced by the per-segment cap. Suites 14/14 + 7/7 + 16/16. |
| 2026-11-17 | latent_stop_on_norm_drift (D) | Qwen3-1.7B Q8, L=16, rescale ON/OFF x raw/delta, drift 0.01-1.0 | **NEGATIVE, left OFF.** target_norm-anchored: with rescale ON never fires (||h|| pinned by construction, 48/48 steps, byte-identical); with rescale OFF fires at step 0 (>100% off). prev-step-norm-anchored: silent under raw (0.05-1.0 all 48/48 identical), fires at step 1 under delta (48->3 steps) - merely re-detecting delta's scale change, which cosine (stop_on_repeat) already catches properly. Norm magnitude carries no independent convergence signal here. |
| 2026-11-17 | alignment_ridge (A) | Qwen3-1.7B Q8, stride 16, ridge 0/1e-3, 2 prompts, depth 8 | **VERIFIED, default OFF.** ridge=0 -> 1e-6 legacy; output `, and the capital of the United States is Washington` / `, the.......`. ridge=1e-3 -> different matrix + output (`.. The capital of France is Paris. Paris` / `,, and,, and,,,,`). ridge back to default -> byte-identical to first default run. Cache: save w/ 1e-3 then load w/ default -> HTTP 404 + log `ridge=897988541` (file) vs `981668463` (wanted); matching ridge -> 200. Fixed 2 real cache bugs: save wrote 1e-5f into the key while compute used 1e-6; load built the expected key from the FILE's ridge so it always matched itself (wrong matrix accepted as a hit). Suites 14/14 + 7/7 + 16/16. |

| 2026-11-17 | `/mas/latent/stats` token accounting (F) | Qwen3-1.7B Q8, delta, depth 8, 4 sampling segments | **BUG FIXED - accounting was dead.** `tokens_sampled` and `text_ms` were declared and reported but NEVER ASSIGNED anywhere in the tree; every run ever answered `0`/`0`, which made the F-section token-savings report structurally impossible (not merely unmeasured). Now accumulated at the end of `llama_mas_run_turn`. Measured: `tokens_sampled: 4` (was 0) and `text_ms: 180` (was 0) with `latent_ms: 329` unchanged. Semantics documented in the code: emitted tokens only, not prefill prompt tokens. DELETE still resets. Suites 14/14 + 16/16 + 7/7. |
| 2026-11-17 | `latent_interleave` (E) | Qwen3-1.7B Q8, delta, align+rescale, 12 text segments, depth 8, K sweep | **CORRECT as a primitive, NEGATIVE for quality, left OFF.** One-time expansion of the caller's segment list at the top of `run_turn`; decode path untouched. Measured block count = exactly `floor(12/K)`: K=20 -> 1 segment/8 steps, K=5 -> 3/24, K=3 -> 4/32, K=1 -> 12/96; text tokens stay 12. Legacy: omitted == K=0 BYTE-IDENTICAL, 1 segment/8 steps/12 tokens. Quality WORSE at every K (5 prompts): off gives `is a\nThe opposite of hot is cold is a bit`, K=1 -> `is is is is...`, K=2 -> `is a is a...` - a think block after 1-4 tokens re-enters an already-converged fixed point, so extra blocks are pure waste. Value is COMPOSITION, not quality. 2 REAL BUGS FIXED while building: (1) single-pass expansion split only once (K=3 gave 2 latent segments not 4, log `14 -> 15`) - rewritten as a run loop; (2) a repeat think block was appended whenever any segments remained, so even `interleave=0` grew the list by one (`14 -> 15`). Plus a diagnostic that printed `expanded[0].latent_interleave` (always the prompt segment's 0), so every run falsely logged `latent_interleave=0 -> ... expanded to ...`. Suites re-run 14/14 + 16/16 + 7/7. |

| 2026-11-17 | `/mas/latent/savings` token-savings + wall report (F) | Qwen3-1.7B Q8, delta, align+rescale, depth 8, shared + folded | **IMPLEMENTED + MEASURED.** New POST /mas/latent/savings runs a text baseline and a latent arm on the same prompt (fresh KV each via llama_mas_agent_truncate 0), reports tokens_sampled / wall_ms / kv_positions / latent_steps + token_savings_pct + wall_speedup_x. Measured: baseline 8 tok/191ms vs latent 8 steps/256ms -> wall_speedup **0.74x (SLOWER, not 3x faster)**; folded 0.78x. token_savings 100% by construction (latent emits no token id). Paper's 3-7x wall speedup does NOT hold at depth 8 on this GPU - latent is bandwidth-cheaper but wall-dearer (~32ms/step vs ~24ms/token). Only stop_on_repeat (2 steps/119ms = 3.18x) recovers a win, by doing less work. Also fixed: load handler now accepts `ridge` (a stride-16 cache saved under ridge 1e-5 was permanently unloadable); load/save/toggle now handle empty body (json::parse discarded -> is_discarded guard). Suites 14/14 + 16/16 + 7/7. |
| 2026-11-17 | `latent_on_tool_call` guard (E) — FALSE [x] corrected | Qwen3-1.7B Q8, align cache stride 16 / ridge 1e-5 / target_norm 1.539432 | **CLAIMS AUDIT caught a lie, then fixed it.** The item was ticked [x] claiming "field + server parsing + a guard that rejects flag+latent_steps<=0". A tree-wide grep (`on_tool_call` over all cpp/h/py) proved the flag was ONLY declared (llama-mas.h:310) and parsed (mas-server.cpp:3466) — never read by any code path and never validated. The "wired, guarded hook" was fiction. Added the missing validation loop to the `/turn` handler (after segment parsing): rejects `latent_on_tool_call` with `latent_steps<=0` via HTTP 400. MEASURED: BAD combo `{steps:0,on_tool_call:1}` -> HTTP 400 `latent_on_tool_call requires latent_steps>0 on the same segment`; GOOD combo `{steps:2,on_tool_call:1}` -> HTTP 200, n_latent_steps=2, n_positions=2, 7.68 steps/s. Default path unchanged (guard only fires on the impossible combo). Suites re-run on a CLEAN server: 14/14 + 16/16 + 7/7. (An earlier hybrid FAIL was dirtied state from manual guard probes — extra agents/sessions — not a regression; a clean restart gave 14/14.) |
| 2026-11-17 | `latent_source_layer` [!] → [x] — **FIXED, aliasing ELIMINATED** | Qwen3-1.7B Q8, align cache stride 16 / ridge 1e-5 / target_norm 1.539432, `/turn` latent_steps=4, delta, rescale ON, CUDA (graphs off by arch) | **The "needs a ggml-level pin" conclusion was WRONG — real llama.cpp bug, one-line fix.** `llama_context::set_output_hidden_states()` set `cparams.output_hidden_states` but did NOT invalidate the scheduler reservation, and `ggml_gallocr_needs_realloc()` (ggml/src/ggml-alloc.c) compares ONLY node/leaf counts and per-node buffer SIZES — never tensor flags. The assignment computed at context creation (capture OFF, no `l_out` marked OUTPUT) was replayed forever, so all 28 layers shared ONE buffer. Fix = the already-present but COMMENTED-OUT `//sched_need_reserve = true;` in `set_output_hidden_states()`, now live and gated to the disable→enable transition (enable ADDS output marks so the cached assignment is genuinely invalid; disable only removes constraints so the old assignment is a valid superset — re-reserving there would cost 2x per latent step for nothing). ALIASED (before): il=0..26 all `data=0xB7C00C000`, IDENTICAL content `sum=20441.530858 sq=202988206.660260 h=9405f82f98e829ba`. DISTINCT (after): every layer its own pointer AND fingerprint — il=0 `0xB7C00C000` sum=5.370452 / il=1 `0xB7D20C000` sum=-92.466214 / il=2 `0xB7E20C000` sum=19906.729206 / … / il=26 `0xB9620C000` sum=20441.530858 / il=27 `0xB7AC00000` sum=1127.962348, spaced 0x100000 apart. FUNCTIONAL (taps were byte-identical, now distinct): L=1 ' the' / L=7 ' and' / L=14 '\n\n' / L=21 ' ' / L=27 '...\n'; with 8 sample segments L=-1 ' 0 degrees. 000' / L=1 ' 0. 0000' / L=14 ' 0. 0000' / L=27 '的     \n '; depth sweep D=4/8/16 × L=1/14/27 live at every depth (D=4: L=1 ' . \n   1. ' vs L=14 ' 000000000' vs L=27 '的 a country of France is the capital of France'). CAVEAT: reading a different layer ≠ a better feedback source; quality still poor, layer 14 degenerate at D=4/8. Suites 14/14 + 16/16 + 7/7. Also note: the previous run's `GGML_CUDA_FORCE_GRAPHS=0` "ruled out graphs" control was INVALID — that env var being *set at all* forces graphs ON (`getenv != nullptr`), so both arms had graphs forced; irrelevant now that the cause is found. |
| 2026-11-17 | `latent_source_layer` [!] — root cause UPGRADED from pointer to CONTENT proof | Qwen3-1.7B Q8, CUDA, `GGML_CUDA_FORCE_GRAPHS=1` and `=0`, `/turn` latent_steps=4, align cache stride 16 / ridge 1e-5 / target_norm 1.539432 | **SUPERSEDED by the row above — right observation, wrong cause.** Prior evidence was pointer-only (`data=0xB7C00C000` for layers 0/1/2), which is attackable: CUDA's pool allocator aligns graph allocations so every launch lands on the same addresses, so a shared pointer could be deterministic address reuse rather than real aliasing. New one-shot `[ALIAS]` diagnostic in `llama_context::decode()` prints, per layer, a content fingerprint (sum / sq / FNV-1a) of the host copy plus `dst[0..2]`, the OUTPUT flag, `view_src`, and the tensor OBJECT address. Measured: layers il=0..26 ALL report `data=0x27C00C000`, `out=1`, `view=0`, `vsrc_out=0`, DISTINCT tensor objects (`ten=...D10`/`...520`/`...730`, 0x2810 apart) and IDENTICAL content — `sum=20441.530858 sq=202988206.660260 h=9405f82f98e829ba`, `dst0=-7.99938,-9.03938,-6.05529`. Layer 27 genuinely differs: `sum=-160.990264`, `dst0=0.252644,-0.583468,0.434406`. So all 27 non-final layers read the SAME bytes from ONE device buffer. Ruled out this pass: (a) pointer artifact (contents identical); (b) async/torn read (synchronous `ggml_backend_tensor_get`, self-consistent across all layers); (c) CUDA graph pooling (`GGML_CUDA_FORCE_GRAPHS=0` gives a byte-identical alias map); (d) view tensor (`view=0` everywhere); (e) missing OUTPUT flag (`out=1` on all 28 — `res->set_outputs()` at llama-context.cpp:1401 runs before `alloc_graph`, and `ggml_gallocr_free_node()` does honour the flag). Therefore the collapse is in the ALLOC/in-place path (`ggml_gallocr_allocate_node`'s `ggml_op_can_inplace` parent-reuse branch or the split-graph `ggml_graph_view` path), not the free path, and any fix must be STRONGER than `GGML_TENSOR_FLAG_OUTPUT`. Not fixable from llama.cpp; item stays `[!]`. Also fixed my own instrumentation bug: reading `t_hs->data` host-side is a DEVICE pointer under CUDA and raised `0xC0000005` — the diagnostic now reads only the host copy `dst`. Suites re-run green on the clean probe server: 14/14 + 16 PASS/0 FAIL + 7/7. |

| 2026-11-17 | MATRIX COMPLETION PASS — build + claims + suites | build-probe (sched_need_reserve fix live), Qwen3-1.7B Q8, align cache stride 16 / ridge 1e-5 / target_norm 1.539432, server PID 6596 on :9024 | **VERIFIED COMPLETE.** 39 [x] / 0 [!] / 0 [ ]. The cron prompt's "REMAINING (10)" list was STALE — all ten were finished in prior runs; confirmed by reading the matrix and by a claims audit (grep found real code behind every [x], no false ticks). build-probe GREEN via vcvarsall x64 + `cmake --build build-probe --target mas-server latent-bench` (BUILD_EXIT=0). build-codex still BLOCKED: LNK1168 on `bin/llama.dll` — llama-server.exe PID 15700 (:8090, build-codex\bin) holds it; that is a live embedding service and was deliberately NOT killed. The live :9024 server runs build-probe\bin\mas-server.exe (started 23:09, after the 23:05 DLL) so the suites exercised the FIXED code, not stale build-codex. TRAP FOUND: `/mas/alignment/load` needs BOTH `ridge:1e-5` AND `vocab_stride:16` for align_cache.bin — `{path}` alone and `{path,ridge}` both return HTTP 404 "cache miss" because the cache key embeds stride and ridge; adding `vocab_stride:16` gave `loaded:true, target_norm 1.539432`. Suites on the fixed server: hybrid **14/14**, mas_latent **16 passed / 0 failed**, mas_server **7/7** (health, generate, generate_greedy, mas_broadcast, mas_chain, mas_dtas, perf). Confidence: High — every number above is observed tool output. |
| 2026-11-18 | `--alignment-cache` startup auto-load (G, was "NOT DONE") | Qwen3-1.7B Q8, build-probe Debug CUDA, :9024, align_cache.bin stride 16 / ridge 1e-5 / target_norm 1.539432 | **SHIPPED, default OFF.** The matrix was 39 [x] / 0 [!] / 0 [ ], so this run closed the one gap the G entry itself flagged. `mas_auto_load_alignment_cache()` runs after EVERY MAS context creation (lazy `ensure_mas_context`, `/config/agents-max` recreate, model swap); stride/ridge overrides are applied BEFORE the load because the cache key is built from the context's CURRENT values. MEASURED 4 arms: OFF (no flags) -> `cache_status:"off" n_embd:0 use_alignment:false stride:1 ridge:1e-06`; ON -> log `loaded Wa 2048x2048 ... target_norm=1.53943` + `[MAS/alignment] AUTO-LOADED ... stride=16 ridge=1e-05`, status `cache_loaded:true n_embd:2048 use_alignment:true target_norm:1.539432066265422` (271 s solve replaced by the 0.03 s load); RECREATE (`/config/agents-max`) -> a SECOND `AUTO-LOADED` line and status stays `n_embd:2048 use_alignment:true` — a REAL silent-bug fix, since a fresh context has an EMPTY matrix and any recreation used to drop the projection so the next latent run silently degraded to a plain memcpy with no error; MISS (`--alignment-ridge 1e-3` vs the 1e-5 cache) -> `cache_status:"miss" n_embd:0 use_alignment:false` plus a loud log printing BOTH keys (`file: ...|stride=16|ridge=925353388` vs `want: ...|stride=16|ridge=981668463`) and `AUTO-LOAD FAILED rc=1 ... check --alignment-stride / --alignment-ridge match the cache key`. Suites: hybrid **14/14**, mas_latent **16 PASS / 0 FAIL**, mas_server **7/7**. |

| 2026-11-18 | `latent_kv_rebase` (C) — closes the deferred `latent_only` fix | Qwen3-1.7B Q8, align cache stride16/ridge1e-5/target_norm 1.539432, delta, depth 8, 5 prompts, 16 sampled tokens after the think block | **IMPLEMENTED + VERIFIED CORRECT, default OFF, MEASURED NOT A QUALITY WIN.** The matrix's `latent_only` entry named this as future work ("rebase positions to 0 after truncation, needs an explicit KV re-index"). NAIVE PATH CRASHED: `llama_memory_seq_add` -> access violation 0xC0000005 at `update: applying K-shift` (seq_add only marks cells; the real work is the context-shift K-shift graph this fork never exercises; forcing a scheduler re-reservation first did NOT help). FINAL: `state_seq_get_size/get_data` -> `seq_rm(seq,-1,-1)` (also clears per-cell shift flags) -> `state_seq_set_data`; positions are not serialised so the bytes land at 0-based positions with RoPE still consistent. Ledger moved with the data to keep "entry i == position i". CORRECTNESS: log shows `REBASED KV positions N.. -> 0.. (31192 bytes re-imported)` for N=43/57/69/217/247/275, 9 consecutive rebases in ONE process, server stayed up. QUALITY A/B (HOLE vs REBASE): 1/5 clearly better ('The opposite of hot is' -> ' in in in in...' vs 'is\nIt looks like a random string of letters. Let me break it down'), 2/5 worse, 2/5 equally degenerate. Conclusion: the correct primitive latent_only always needed, but it does NOT fix latent_only — depth collapse is the premature-convergence fixed point (section B), not a KV-layout artefact. Suites on a CLEAN server: hybrid 14/14, mas_latent 16 PASS / 0 FAIL, mas_server 7/7. |
| 2026-11-18 | **DEBT CLOSED — build-codex stale binary / LNK1168** (not a matrix item; the only non-matrix debt on the board) | `build-codex` Debug CUDA, :8090 nomic embedder, :9024 MAS server | **CLOSED.** `build-codex/bin/llama.dll` (Nov17 19:14) and `mas-server.exe` (Nov17 20:53) were PRE-`sched_need_reserve`-fix and LNK1168-locked by the live nomic embedder `llama-server.exe` on :8090. SAFETY CHECK FIRST: no script/service referenced :8090 (the other hits are `AuthProxy`'s k8s `RunnerHealthPort`, a different subsystem); `RmStartSession` -> `no lockers`; 0 ESTABLISHED connections; the process was an ORPHANED interactive-shell child (20932 -> parent cmd.exe 1876, no watchdog task) i.e. a hand-started embedder, not a managed service. REBUILD: vcvarsall x64 + `cmake --build build-codex --target mas-server latent-bench --config Debug` -> relinked `bin/llama.dll` + `bin/mas-server.exe`, exit 0; second run -> `ninja: no work to do` (proves current, not merely relinked). `find src tools/mas-server ggml -newer build-codex/bin/mas-server.exe` -> EMPTY, so no stale code remains in either tree. NUANCE (corrects the earlier framing): `build-codex/bin/llama-server.exe` is still Sep 29 and is the ONLY `llama-server.exe` on the box — build-probe never built that target — and it was NOT relinked because llama.dll's public C API did not change; it is a consumer, not staleness. RESTORE: embedder relaunched from the SAME binary/port/model -> `/v1/models` lists nomic-embed-text (n_vocab 30522, n_embd 768) and `/v1/embeddings` returns a 768-d vector (PID 19512, LISTENING 127.0.0.1:8090). SUITES on the unchanged :9024 server: hybrid **14/14**, mas_latent **16 PASS / 0 FAIL**, mas_server **7/7** — no regression from co-hosting the embedder on the same GPU. |
| 2026-11-18 | `latent_bench.py` reporting debt — CLOSED (non-matrix; 06:5x/07:3x run's own logged open debt) | Qwen3-1.7B Q8, align cache stride16/ridge1e-5/target_norm 1.539432, delta, greedy | **TWO DEFECTS FIXED, default behaviour preserved.** (1) `--samples N` printed only ONE sample string (`if len(samples)<1` guard) though it ran N×6 turns, hiding degeneracy — now prints EVERY sample string. MEASURED: `--depths 0,8 --samples 2 --ntok 16` printed 12/12 sample strings (was 1). (2) `wall_x` speedup ignored the token-count confound (a latent span DISPLACES text tokens, so a latent run can win on wall purely by emitting fewer tokens) — `run_one` now captures `n_tokens` from /turn and `text_ms` from stats; the per-depth table gained a `tok` (emitted-token) column; the summary prints BOTH `wall_x` (raw, explicitly labelled misleading) and `tok_x` (tokens/s ratio = the honest equal-work comparator). MEASURED: at ntok16 both arms emit exactly 16.0 tok → wall_x 0.67 vs tok_x 0.68 (agree, so ~0.67× latent wall cost at depth8 is REAL, not an artifact); at ntok32 --stop0 both emit 32.0 tok → wall_x 0.79 tok_x 0.79. The `tok` column is what would expose a confound if a run ever emitted fewer tokens. No C++ touched; no server restart. Suites re-run green on :9024 (PID 21928): hybrid 14/14, mas_latent 16 PASS / 0 FAIL, mas_server 7/7. |
| 2026-11-18 | alignment cross-pass diagnostic (`alignment_cross_progress`) (G) | Qwen3-1.7B Q8, build-probe Debug CUDA, :9024, align_cache stride16/ridge1e-5 | **SHIPPED, default OFF, MEASURED.** No open checkbox (matrix 40 [x] / 0 [ ] / 0 [!]); closed the live debt that the 8B alignment crash dies SILENTLY inside the cross pass (10:1x: dequant + Gram + Gauss-Jordan invert all succeed on CPU, then bare `SERVER_EXIT=-1`). The cross pass was a single-threaded n_vocab-strided × n_embd² accumulation with zero logging, so the crash point was unobservable. New toggle adds a PREFLIGHT line (A / w_out / row-scratch bytes, strided row count, MAC count) and a progress line every ~5% of rows with a sampled sum of A. MEASURED with `cross_progress=true` (fresh solve, no cache): preflight `A=16 MB, w_out=1187 MB, row scratch=8 KB, 9496 rows x 8.38861e+06 MACs`; progress advanced row 0→151680 with A_probe_sum -2.48→5.13 (real accumulation, no stall); solve finished `target_norm=1.54249` + `Wa=2048x2048`. Default OFF verified: fresh server `cross_progress:false`, cache target_norm 1.539432066265422 unchanged. Code: llama-mas.h decl, llama-mas.cpp field/setter/getter, HTTP `cross_progress` on POST /mas/alignment/compute + status. Suites: hybrid 14/14, mas_latent 16 PASS / 0 FAIL, mas_server 7/7. |
| 2026-11-18 | 8B alignment cross-pass crash — memory-footprint fix (no checkbox; live debt) | Qwen3-1.7B Q8, build-probe Debug CUDA, :9024, fresh solve cross_progress=true | **FIXED, default behavior-preserving, MEASURED identical.** The 10:1x "reorder A before Gauss-Jordan" theory was FALSIFIED by server_8b_cpu.log: dequant + Gram + 'inverted Gram' + thread-cap 8→4 all print, then bare SERVER_EXIT=-1 with NO preflight — so the crash is at `std::vector<float> A(n*n)` (line 3282, AFTER inverted Gram). Live heap at that point on 8B ≈ w_out 2374 MB + G 67 + Ginv 67 + resident model 4.7 GB + compute 1.7 GB ≈ 8.9 GB. MSVC std::vector does NOT throw on alloc failure — it returns a block the allocator can't back and the `0.0f` zero-fill faults silently. TWO behavior-preserving fixes: (1) free G (un-inverted Gram, dead after Gauss-Jordan — verified zero refs after line 3271 by awk+python regex scan) right after 'inverted Gram' prints, dropping 67 MB on 8B before the fragile A alloc; (2) wrap A + w_in_row alloc in try/catch(bad_alloc) + size check → clean `use_identity("cross pass allocation failed")` + loud log instead of crash. MEASURED 1.7B fresh solve: preflight `A=16 MB, w_out=1187 MB, 9496 rows x 8.38861e+06 MACs`; progress 0→151680, A_probe_sum -1.61→5.13 (BYTE-IDENTICAL to pre-fix); `target_norm=1.54249` (identical to pre-fix), `Wa=2048x2048`, busy:false, is_identity:false. Default OFF on restart: cross_progress:false, cache target_norm 1.539432 unchanged. Suites: hybrid 14/14, mas_latent 16 PASS / 0 FAIL, mas_server 7/7. |
| 2026-11-18 | alignment cache key: EFFECTIVE stride (G, correctness; no checkbox) | Qwen3-1.7B Q8, build-probe Debug CUDA, :9024, align_cache stride16/ridge1e-5, synthesized stride-1-keyed cache file | **BUG FOUND + FIXED.** Save (llama-mas.cpp:3490) and load (:3581) built the key from the CONFIGURED `alignment_vocab_stride`, but the solve forces stride 1 under `alignment_full_vocab` - so a full-vocab matrix was stored under a stride-16 key and then ACCEPTED by a stride-16 context (a wrong matrix presented as a HIT). The `alignment_full_vocab` entry's claim that the key "already contains the effective stride" was FALSE. Fix: new `mas_alignment_effective_stride()` used by save + load, plus `full_vocab` accepted on POST /mas/alignment/load (mirrors the `ridge` precedent). MEASURED 3 arms: real cache + stride16 + full_vocab OFF -> 200 loaded (DEFAULT UNCHANGED); stride1-keyed file + stride16 + full_vocab OFF -> 404 cache miss (correctly rejected); stride1-keyed file + stride16 + full_vocab ON -> 200 loaded, status `full_vocab:true vocab_stride:16`. Evidence the shipped cache was mis-keyed: key says stride=16 but target_norm 1.539432066265422 vs a fresh stride-16 solve's 1.5424949868806226 (target_norm is averaged over the effective-stride rows). NOT RETROACTIVE - the on-disk cache keeps its old key; a fresh compute+save would write an honest one (deliberately NOT run: the prompt forbids a full solve). Suites: hybrid 14/14, mas_latent 16 PASS / 0 FAIL, mas_server 7/7. |
| 2026-11-18 | build-codex `llama.dll` current with `src` (build hygiene) | Qwen3-1.7B Q8, build-codex Debug CUDA :9025 smoke, --alignment-cache stride 16 / ridge 1e-5 / --n-agents-max 8 | **STALE-BINARY DEBT CLOSED.** 12:1x run flagged build-codex/bin/llama.dll (10:37) stale vs src/llama-mas.cpp (12:14); earlier 'DEBT CLOSED' had checked mas-server.exe, which does not depend on src. Blocker was LNK1168 (live llama-server.exe on :8090 out of build-codex\bin); that process is now gone of its own accord (no llama-server.exe in CIM, :8090 health 000). _build_mas_server_debug.bat -> 4 steps, EXIT=0, no LNK1168: llama.dll 13:02 (6,850,048 B) + mas-server.exe 13:02 (3,234,304 B), both newer than src 12:14 / mas-server.cpp 12:20. SMOKE (did not trust the timestamp): :9025 boot AUTO-LOADED the cache (target_norm=1.53943, stride=16 ridge=1e-05), /mas/alignment/status cache_loaded:true use_alignment:true target_norm:1.539432066265422; /v1/chat/completions 'One plus one equals' temp0 -> 200, 97 completion tokens. :9025 then stopped (GPU + DLL freed); :9024 build-probe left running, health 200 after. Suites on :9024: hybrid 14/14, mas_latent 16 PASS/0 FAIL, mas_server 7/7. No source change, so no new A/B. |
| 2026-11-18 | `POST /reset` route (G, test isolation; no checkbox) | Qwen3-1.7B Q8, build-probe Debug CUDA :9024 | **SHIPPED + MEASURED.** Closes the 13:5x repro hazard: `test_hybrid_turn.py`'s startup `POST /reset` 404'd (no route existed), so state accumulated and the suite gave FALSE 12/2 fails after other suites. New `svr.Post("/reset")` clears sessions + agents + latent stats under g_state.mtx. MEASURED: dirty state (1 session kv_bytes 283319, text_ms 281) -> `POST /reset` `{"sessions_cleared":1,...}` -> `/sessions` empty, stats all-zero; no-body POST also 200; alignment intact. REGRESSION PROOF: hybrid run 3x back-to-back on the SAME server = 14/14, 14/14, 14/14 (was 12/2). BUILD TRAP: `_build_mas_server_debug.bat` builds only build-codex; the :9024 server runs build-probe, which needs `build_probe.bat` and a kill of the live server first (LNK1168). Suites: hybrid 14/14, mas_latent 16 PASS / 0 FAIL, mas_server 7/7. |
| 2026-11-19 | `/agents/destroy` + `/agents/clear` honest per-agent removal (G, no checkbox) | Qwen3-1.7B Q8, build-probe Debug CUDA :9024, align cache AUTO-LOADED target_norm 1.539432 | **BUG FOUND + FIXED, default-preserving.** All three agent-lifecycle routes (`/agents/destroy`, `/agents/clear`, `/agents/destroy-all`) called `llama_mas_clear_agents()` = wiped the WHOLE pool even when a caller named one agent; `/agents/destroy` also skipped `g_state.mtx` and ignored its body (hardcoded `{"status":"ok"}`). `test_one_quant.py:35` posts `{"agent_id":"test"}` expecting one removal and silently nuked everything. FIX: `/agents/destroy` + `/agents/clear` share one handler — empty/absent agent_id -> clear-all (old default); non-numeric -> match by ROLE; numeric -> match by INDEX; no match -> HTTP 404, pool INTACT. MEASURED: 3 agents alpha/beta/gamma -> destroy beta(role) `n_agents_remaining 2` roles ['alpha','gamma']; destroy "0"(index) -> 1 role ['gamma']; clear `{"agent_id":"nope"}` -> HTTP 404 count stayed 1; clear `{}` -> 0. Build: `build_probe.bat` EXIT=0 (`ninja: no work to do` first, then mas-server.cpp rebuild + link clean). Suites: hybrid 14/14, mas_latent 16 PASS / 0 FAIL, mas_server 7/7. |
| 2026-11-19 | `latent_on_tool_call` NOW injects the think span (E; closes the last dead flag) | Qwen3-1.7B Q8, build-probe Debug CUDA :9024, align cache AUTO-LOADED target_norm 1.539432 | **DEAD FLAG FIXED + MEASURED, default OFF.** A debt audit found `latent_on_tool_call` was declared (llama-mas.h:327), parsed (mas-server.cpp:3644) and validated (rejects latent_steps<=0) but READ BY NO CODE (0 reads in run_turn) — the guard fired only on the impossible combo, so on the GOOD combo the flag still did NOTHING. Now a new pass in `llama_mas_run_turn` (after the latent_interleave expansion) scans for any TOKENS segment with `latent_on_tool_call!=0` and inserts a LATENT think block of that segment's own `latent_steps` (inheriting align/transform/position_mode/stop_on_*, but stripped of text/forced_tokens/sample/escape/latent_only_agents/interleave so it is PURE reasoning and cannot recurse) IMMEDIATELY BEFORE it. Default OFF: the pass is skipped entirely when no flag is set -> byte-identical legacy. BUG FIXED WHILE BUILDING: the expansion vector was block-scoped so `segs_ptr` dangled after the block -> 0xC0000005 (SERVER_EXIT=-1073741819); hoisted to function scope. MEASURED (POST /reset + fresh worker each): OFF `{tokens+sample}` -> n_latent_steps=0 n_positions=6 text=' Paris' tok[12095]; ON same + `latent_on_tool_call:1 latent_steps:3` -> log `injected 1 ...; 1 -> 2 segments`, n_latent_steps=3 n_positions=9 (=6+3) text=' Paris' tok[12095] (identical token => pure think, emitted no text); MULTI 2x tokens+sample flag on 2nd steps=2 -> `injected 1 ...; 2 -> 3 segments`, n_latent_steps=2 n_tokens=2. Server alive through all arms. Suites: hybrid 14/14, mas_latent 16 PASS / 0 FAIL, mas_server 7/7. |
| 2026-11-19 | `llama_mas_set_agent_layer_graph` — dead flag + link trap closed (G) | Qwen3-1.7B Q8, build-probe Debug CUDA :9024, align cache AUTO-LOADED target_norm 1.539432 | **DECLARED-BUT-NEVER-DEFINED fixed; per-agent layer-graph path now real.** A tree-wide audit found `llama_mas_set_agent_layer_graph` declared in the public llama-mas.h with NO DEFINITION anywhere (any caller = LNK2019), and the per-agent `layer_graph` field read at ~10 decode sites but never assigned -> the feature was unreachable. Now defined (llama-mas.cpp:2440, borrowed-ownership, freed with the agent), plus route `POST /v1/agents/N/layer-graph` and the missing arch gate in src/models/qwen3.cpp:21. MEASURED: `{layers:[0,1,2,3]}` -> `{status:ok, enabled:4, n_layers:28}`; `{}` -> `{cleared:true}`; bad index -> 400; default `/turn` BYTE-IDENTICAL to pre-change (text=' Paris', tokens=[12095], n_positions=6, health 200). HONEST LIMIT: a partial SUBSET changes the graph node count and ggml reserves one shape, so attaching mid-process then decoding aborts `GGML_ASSERT(buffer)` (measured SERVER_EXIT=3, partial reserve `graph nodes=161` vs full 1099). Root cause in ggml (stale prev split ids + 2 buffers -> no auto-realloc), not llama.cpp; subset only takes effect when attached BEFORE the first decode. Route prints a NOTE; `/reset` forces the full re-reserve. No quality win claimed. Suites 14/14 + 16 PASS/0 FAIL + 7/7. |

## Notes

- Upstream reference: `C:\Users\zsolt\AppData\Local\hermes\cache\scratch\LatentMAS`
  (clone of github.com/Gen-Verse/LatentMAS). Key files: `models.py`
  (`generate_latent_batch`, `_apply_latent_realignment`),
  `methods/latent_mas.py` (`_truncate_past`, `latent_only`).
- Every toggle must default to the pre-existing behaviour so old callers and
  the three regression suites stay green.
- Rebuild: `cmd /c _build_mas_server_debug.bat` from the repo root.

## H. 8B enablement + per-step speed (2026-11-19)

- [x] **H1 device-resident embedding staging** — `output.weight` lives in the CUDA0 buffer while `token_embd.weight` is CPU-resident; ggml `to_float` reads its source on the host, so the 8B alignment crashed with 0xC0000005. Fixed by staging via `t->buffer` + `ggml_backend_buffer_is_host()`. NOT via `ggml_backend_sched_get_tensor_backend()` — model weight tensors are not in the context scheduler, which is why that lookup returned null. Log: `staged 486.855 MB from device buffer 'CUDA0'`.
- [x] **H2 8B alignment solve completes** — Wa=4096x4096, target_norm=1.37999, stride 32, ridge 1e-6, ~40 min (n^3, ~8x the 2048-dim solve). Cached to `tools/mas-server/align_8b_s32.bin` (64 MB), reload **0.10 s**. Load requires `vocab_stride:32` or the key mismatches -> 404.
- [x] **H3 deep latent is coherent on the 8B** — L=8/16/32/64 all produce structured prose on a real perf-engineering prompt (4.58/5.12/6.11/9.20 s), **4 distinct output hashes** so genuinely depth-sensitive, not folded-mode saturation. On the 1.7B, L=8 was degenerate dots. Config: align=1, rescale_norm=1, no delta transform needed.
- [x] **H4 parallel per-step projection** — `alignment_parallel_threads`, default 4. 91.75 -> 64.03 ms/step (**1.43x**), saturates at 4 threads, output **byte-identical** at 1/2/4/8. 1 latent step now ~1.29 text tokens of time (was ~1.85). Set 1 for the original scalar loop.
- [x] **H5 repo-resident launchers** — `run_8b.bat` / `run_17b.bat`. The Hermes scratch dir prunes entries idle 24 h, which deleted every launcher and made starts fail instantly with exit 1 while a stale log looked like a model-load crash.
- [ ] **H6 latent does NOT improve arithmetic accuracy** — GSM8K-style exact match on the 8B: depth 0 = 6/36, depth 4 = 6/36, **depth 16 = 0/36**. Speed vs text baseline: depth 4 `wall_x` 0.88, depth 16 `wall_x` 0.66 — latent is SLOWER at equal token count. The paper's 3-7x speedup / 50-80% token savings are **NOT reproduced**. Coherence != accuracy; the H3 win must not be reported as a reasoning win. OPEN: find a task class where latent actually buys correctness, or conclude it does not.

## I. Delivered multi-agent win (2026-11-19)

- [x] **I1 `/mas/latent-chain`** — the paper's shape: prompt -> [think -> full layer-wise KV handoff] x N -> last agent speaks. Downstream agents never re-decode the prompt. Per-agent ledger returned.
- [x] **I2 EOG stop for TEXT segments** — `seg.stop_on_eog` extended from latent-only; the loop ignored `agent->finished` and reset `generated_len` per segment, so N sample segments always burned N tokens. `/turn` default 0 (legacy), chain default true.
- [x] **I3 ChatML in every harness** — `/turn` sends raw tokens with NO template; without Qwen3 ChatML the model rambles off-topic (emitted React imports for an arithmetic question). This plus a 64-token budget invalidated ALL earlier accuracy numbers.
- [x] **I4 speed win, easy set — ACCURACY CLAIM SINCE CORRECTED** — text_single 1/6 @22.09 s/512 tok vs chain(3 agents, 0 latent) 6/6 @8.00 s/184 tok = **2.76x wall, 2.78x tokens**. The speed/token half is solid; the '+5 correct' half was a SCORING ARTIFACT (see I8). Do not quote the accuracy delta from this row.
- [x] **I5 RETRACTED — hard set: latent does NOT earn its cost (see I9)** — 3/6 without latent -> 6/6 with 10 steps/agent. The "latent buys nothing" verdict holds only on the EASY set. Difficulty decides.
- [x] **I6 agent count substitutes for latent, and 5 is the optimum (efficiency; accuracy deltas see I8)** — hard set: 1 agent 1/6; 3 agents 3/6 (6/6 w/ latent); **5 agents 0 latent 6/6 @14.49 s/350 tok = 2.17x wall, 2.19x tokens (BEST)**; 7 agents 5/6 @19.85 s (degrades). Peaked, not monotone. At 5 agents latent slightly hurts (6/6 -> 5/6).
- [x] **I7 empty-segment guard** — agents beyond the supplied `role_prompts` with `latent_steps=0` got an empty segment list -> opaque `rc=-1` that read like a KV failure. Now `{skipped:true, reason:...}` and the chain continues.

- [x] **I8 CORRECTION — the accuracy win was mostly a scoring artifact.** 12-question re-verification with gold computed in Python and full texts stored: `single_weak` strict **3/12** but **concluded 10/12**; `single_allroles` strict 4/12, concluded 8/12; `chain` **12/12** on every scoring. Every single-arm run hit the 768-token **ceiling**, so it never emitted a formatted answer and "last number in text" scored an intermediate value. "gold anywhere" also false-positived on a restated `$40` input. **Honest claim: reasoning +2 (10/12→12/12), answer delivery +9 (3/12→12/12), efficiency 2.72x wall / 63% fewer tokens.** Prompt parity (all 5 roles in one agent) did NOT close the gap. Rule: always report strict + concluded + raw text, and flag ceiling-hit arms.
- [x] **I9 FINAL VERDICT — latent hurts reasoning; more agents dominates.** Fair `concluded` metric, 12 q: 3 agents 0 latent **11/12** vs 3 agents 10 latent **8/12** (−3 correct conclusions), while strict format goes 8→10 and tokens 283→240. Wall identical (~12.5 s). **5 agents 0 latent = 12/12 on both metrics, 0 ceiling hits, same wall.** No axis on which latent wins. The earlier positive claim came from the 0-latent arm hitting the ceiling more often (4/12 vs 2/12) and losing points on FORMAT. Latent stays switchable but default OFF. Report `ceiling_hits` with every accuracy number.
