#pragma once

#include <stdint.h>

#include "llama.h"
#include <vector>
#include <string>
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif

//
// Multi-Agent System (MAS) internal header.
//

// MAS adapter types
enum llama_mas_adapter_type {
    LLAMA_MAS_ADAPTER_NONE     = 0,
    LLAMA_MAS_ADAPTER_RECURSIVE = 1,
    LLAMA_MAS_ADAPTER_CROSS    = 2,
    LLAMA_MAS_ADAPTER_MEAN     = 3,
    LLAMA_MAS_ADAPTER_CAT      = 4,
};

// MAS strategies
enum llama_mas_strategy {
    LLAMA_MAS_STRATEGY_ROLE            = 0,
    LLAMA_MAS_STRATEGY_CHAIN           = 1,
    LLAMA_MAS_STRATEGY_BROADCAST       = 2,
    LLAMA_MAS_STRATEGY_DTAS           = 3,
    LLAMA_MAS_STRATEGY_CHAIN_BROADCAST = 4, // Chain of broadcast waves: each wave runs in parallel, waves are sequential
    LLAMA_MAS_STRATEGY_LATENT_MAS      = 5, // Original LatentMAS: KV cache transfer + alignment matrix (ICML 2026)
    LLAMA_MAS_STRATEGY_LATENT_MAS_NO_ALIGN = 6, // Experimental: KV cache transfer without alignment matrix
    LLAMA_MAS_STRATEGY_LATENT_MAS_EMBEDDINGS = 7, // Experimental: Embedding transfer instead of KV cache
};

//
// Adapter weight storage (for RECURSIVE and CROSS types)
// Weights are stored as standard float arrays, computed on CPU/GPU via backend
//

struct llama_mas_adapter_weights {
    // RecursiveLink weights (all [n_embd] or [n_embd, n_embd])
    float * w_pre_ln;      // [n_embd]     LayerNorm weight
    float * b_pre_ln;      // [n_embd]     LayerNorm bias
    float * w_proj1;       // [n_embd * n_embd]  Linear 1 weight
    float * b_proj1;       // [n_embd]     Linear 1 bias
    float * w_proj2;       // [n_embd * n_embd]  Linear 2 weight
    float * b_proj2;       // [n_embd]     Linear 2 bias
    float * w_post_ln;     // [n_embd]     LayerNorm weight
    float * b_post_ln;     // [n_embd]     LayerNorm bias

    // CrossLink weights (in_dim -> out_dim, with residual projection)
    float * w_cross_ln_source;  // [in_dim]
    float * b_cross_ln_source;  // [in_dim]
    float * w_cross_proj1;      // [in_dim * hidden_dim]
    float * b_cross_proj1;      // [hidden_dim]
    float * w_cross_proj2;      // [hidden_dim * out_dim]
    float * b_cross_proj2;      // [out_dim]
    float * w_cross_residual;   // [in_dim * out_dim]
    float * b_cross_residual;   // [out_dim]
    float * w_cross_ln_target;  // [out_dim]
    float * b_cross_ln_target;  // [out_dim]

    int32_t n_embd;
    int32_t in_dim;
    int32_t out_dim;
    int32_t hidden_dim;
    void * gpu_data;  // opaque pointer for GPU weight cache (GGML_USE_CUDA)
};

//
// MAS Agent configuration (per agent)
// Each agent has its own adapter type and optional weights.
// Configs are isolated — no agent sees another's adapter weights.
//

struct llama_mas_agent_config {
    const char * role;                    // Agent role name
    llama_seq_id seq_id;                  // Sequence ID within the batch
    enum llama_mas_adapter_type adapter_type; // Which adapter to use

    // Optional: adapter weights for RECURSIVE/CROSS types
    // If NULL, the adapter acts as identity (pass-through)
    struct llama_mas_adapter_weights * adapter_weights;

    // Per-agent generation parameters
    int32_t max_tokens;
    float   temperature;
    float   top_p;
    int32_t top_k;

    // Token control
    llama_token * prompt_tokens;
    size_t        prompt_len;
    bool          finished;
    size_t        generated_len;

    // Per-agent stop-after-prompt (encode only, do not generate)
    bool          stop_after_prompt;

    // LatentMAS: if true, agent generates tokens but does not produce text output.
    // Its generated tokens are kept in the shared KV cache for downstream agents.
    bool          is_latent;

    // LatentMAS: number of latent thought tokens to generate (only used when is_latent=true).
    // If 0 and is_latent=true, the agent is encode-only (no generation).
    int32_t       latent_steps;
};

//
// MAS Context (orchestrates multi-agent decode)
//

struct llama_mas_context;

// Create/destroy MAS context
LLAMA_API struct llama_mas_context * llama_mas_init(
    struct llama_model * model,
    struct llama_context_params params,
    int32_t n_agents_max,
    bool use_batched_decode
);

LLAMA_API void llama_mas_free(struct llama_mas_context * mas);

// Add an agent with its isolated config.
// Returns the agent index, or -1 on error.
LLAMA_API int32_t llama_mas_add_agent(
    struct llama_mas_context * mas,
    const struct llama_mas_agent_config * config
);

// Flat API: add agent without constructing a config struct.
LLAMA_API int32_t llama_mas_add_agent_flat(
    struct llama_mas_context * mas,
    const char * role,
    int32_t seq_id,
    int32_t adapter_type,
    int32_t max_tokens,
    float temperature,
    float top_p,
    int32_t top_k,
    int32_t stop_after_prompt,
    int32_t is_latent,
    int32_t latent_steps
);

// Remove an agent and free its isolated config/weights.
LLAMA_API void llama_mas_remove_agent(
    struct llama_mas_context * mas,
    int32_t agent_idx
);

// Set the UUID of an agent (for preserving identity across recreations)
LLAMA_API void llama_mas_set_agent_uuid(
    struct llama_mas_context * mas,
    int32_t agent_idx,
    const char * uuid
);

// Remove all agents and reset the MAS context for a new run.
// The underlying llama_context and KV cache are preserved.
LLAMA_API void llama_mas_clear_agents(struct llama_mas_context * mas);

// Set per-agent input embeddings (instead of tokenized prompt).
// embeddings: float array of shape [n_tokens, n_embd]
// Set embeddings=NULL or n_tokens=0 to clear.
LLAMA_API void llama_mas_set_agent_input_embeddings(
    struct llama_mas_context * mas,
    int32_t agent_idx,
    const float * embeddings,
    int32_t n_tokens,
    int32_t n_embd
);

// Enable hidden state extraction from all layers during decode.
// Call before llama_mas_run() if you need to capture hidden states.
LLAMA_API void llama_mas_set_output_hidden_states(
    struct llama_mas_context * mas,
    bool output_hidden_states
);

// Stop generation after prompt decode (useful for capturing prompt hidden states).
// When true, llama_mas_run() decodes prompts but does not generate tokens.
LLAMA_API void llama_mas_set_stop_after_prompt(
    struct llama_mas_context * mas,
    bool stop_after_prompt
);

// ---------------------------------------------------------------------------
// Hybrid text/latent turn driver
//
// A turn is an ordered list of segments. Each segment is decoded with its own
// llama_decode() call because llama_context::decode asserts batches are XOR
// (all-token or all-embedding) - see llama-context.cpp:1729. Both kinds write
// into the SAME sequence and the SAME KV cache, and both advance the position
// counter by exactly one, so the cache stays a single contiguous run.
//
//   LLAMA_MAS_SEGMENT_TOKENS : text segment. prompt is tokenized and decoded as
//                              one batch; if sample != 0 the last position is
//                              sampled and the token appended to out_tokens.
//   LLAMA_MAS_SEGMENT_LATENT : latent segment. latent_steps iterations of
//                              decode(embedding) with no sampling. Step 1 is
//                              seeded from seed_token (or the agent's last
//                              generated token). The returned hidden state is
//                              optionally aligned with Wa before feedback.
//
// Hidden-state capture is toggled PER SEGMENT: enabling it makes llama_context
// copy every layer x every output position to host memory, so it must stay off
// during text prefill of long code contexts.
// ---------------------------------------------------------------------------

enum llama_mas_segment_kind {
    LLAMA_MAS_SEGMENT_TOKENS = 0,
    LLAMA_MAS_SEGMENT_LATENT = 1,
};

struct llama_mas_segment {
    enum llama_mas_segment_kind kind;

    // TOKENS
    const char * text;          // tokenized with the agent's prompt settings (NULL = empty)
    const llama_token * forced_tokens; // forced ids decoded before the text; NULL = none
    int32_t      n_forced_tokens;
    llama_token  seed_token;    // forced first token instead of the last generated one (<0 = auto)
    int32_t      sample;        // 1 = sample the last position and emit a token

    // LATENT
    int32_t      latent_steps;  // number of hidden-state feedback iterations
    int32_t      align;         // 1 = apply Wa before feeding the state back
    int32_t      stop_on_eog;   // LATENT: 1 = stop early if the last step decodes to EOG

    // LATENT escape: stop thinking as soon as the model would emit this text.
    // This is the "I'm done thinking, hand control back" signal. It is matched
    // per step by detokenizing the probe token, so the caller does not need to
    // know the model's token id for it - pass the surface form, e.g. the model's
    // own end-of-thinking marker. NULL/empty disables it. Checked in addition to
    // stop_on_eog, and only when a probe was actually sampled.
    const char * escape_text;

    // ── Upstream LatentMAS loop-shape toggles (0 = legacy, the default) ─────
    // latent_only: drop the prompt KV before the latent loop so each step
    // attends to the latent span alone (upstream 'latent_only'). Without it the
    // model re-reads a growing prompt every step.
    // latent_truncate_after: after the loop, rewind the KV to before the latent
    // span so a long think block does not permanently occupy n_ctx.
    int32_t      latent_only;
    int32_t      latent_truncate_after;

    // Inverse of latent_only: explicitly RETAIN the prompt KV across latent
    // steps. latent_only (deleting the prompt) was measured WORSE because it
    // leaves a position-space hole; this toggle makes the intended behaviour
    // explicit and self-documenting rather than implicit, and gives the A/B a
    // named control. Setting both latent_only=1 and latent_kv_keep_prompt=1 is
    // contradictory, so keep_prompt WINS (latent_only is skipped) - a caller
    // asking to keep the prompt must never end up with it deleted.
    int32_t      latent_kv_keep_prompt;

    // `latent_kv_rebase`: with `latent_only`, rebase the SURVIVING KV positions
    // down to 0 instead of leaving a position-space hole. latent_only deletes
    // the prompt KV but the memory module refuses non-consecutive positions, so
    // the span historically started at position N with nothing before it - the
    // measured cause of the `...和和どどどど` garbage. llama_memory_seq_add()
    // is the explicit KV re-index this needs: it shifts the remaining cells and
    // llama_kv_cache applies a RoPE K-shift on the next decode, so positions
    // are relative and the result is exact rather than approximate.
    // Requires latent_only (it is the only path that deletes KV here); ignored
    // otherwise. Default 0 = legacy hole-preserving behaviour.
    // CAVEAT: the agent's generated_tokens ledger is "prompt then generated",
    // i.e. entry i == position i. A rebase breaks that identity for the prompt
    // prefix (those positions no longer exist), so llama_mas_agent_truncate's
    // keep_total arithmetic is only approximate after a rebase. It is already
    // KV-authoritative for positions, so decoding stays correct.
    int32_t      latent_kv_rebase;

    // Feedback transform applied to the tapped vector before it is fed back:
    //   0 = raw (default), 1 = prior step's vector, 2 = delta (h_t - h_{t-1}),
    //   3 = EMA across steps (latent_ema_alpha).
    // These act on the vector the loop already holds, so ggml buffer aliasing
    // (which defeats latent_source_layer) cannot affect them.
    int32_t      latent_transform;
    float        latent_ema_alpha;

    // Stop early when the recurrence has converged (cosine similarity between
    // consecutive hidden states >= this). 0 disables. Without it a loop keeps
    // emitting the same phrase for every remaining step, burning n_ctx.
    float        latent_stop_on_repeat;

    // Stop early when the feedback vector's L2 norm drifts more than this
    // FRACTION away from the alignment target_norm (mean ||W_in[v]||), i.e.
    // |  ||h||/target_norm - 1 |  > latent_stop_on_norm_drift. 0 disables.
    // Rationale: the embedding distribution has a characteristic magnitude;
    // a latent vector whose norm runs far away from it is out-of-distribution
    // and the recurrence has left the region the model was trained on, so
    // further steps degrade rather than refine. Needs alignment computed
    // (target_norm > 0); without it the check is a silent no-op.
    float        latent_stop_on_norm_drift;

    // Window size for latent_transform=4 (mean_pool): how many trailing layer-0
    // input positions to average. 1 = last token only (same as the default tap).
    int32_t      latent_pool_window;

    // Deterministic per-step perturbation applied to the feedback vector, as a
    // multiple of its own RMS. 0 = off. Keeps the recurrence from converging to a
    // frozen state after ~3 steps (the measured cause of depth collapse).
    float        latent_jitter;

    // ── E-section composition knobs ────────────────────────────────────────
    // `latent_only_agents`: make this segment expand into a LATENT-ONLY run for
    // agents that are configured `is_latent`, while non-latent agents keep
    // answering in text. Without it every agent in a turn goes latent, which is
    // wrong for a mixed crew (a "worker" agent should think, a "spokesperson"
    // should answer). Implementation: segments carry an agent mask, and a
    // latent-only segment is skipped for agents not in the mask.
    int32_t      latent_only_agents;

    // `latent_on_tool_call`: on a TOKENS segment, run a latent (continuous
    // reasoning) span IMMEDIATELY BEFORE this segment instead of after. The
    // idea is to spend a little continuous reasoning before committing to a
    // structured action, which is where a wrong-but-confident tool call is most
    // expensive. The span length is the segment's own `latent_steps` and its
    // feedback transform / alignment come from the usual fields on this segment
    // (align, latent_transform, latent_position_mode, ...). The flag lives on
    // the text/tool-call segment itself - the segment that FOLLOWS the injected
    // think block - because the caller knows which segment emits the tool call;
    // the model's tool grammar is not visible in this layer. Default 0 = legacy
    // behaviour (no injected span). Rejected by the server if latent_steps<=0.
    int32_t      latent_on_tool_call;

    // Position accounting for the latent span:
    //   0 = shared   : every step appends a NEW position (legacy default; the
    //                  growing sequence that measured premature convergence)
    //   1 = folded   : reuse the SAME position slot for every step (in-place
    //                  recurrence). Implemented by removing the KV cell written
    //                  by step s before step s+1, so the memory module's tail
    //                  (seq_pos_max) stays pinned and the next step's embedding
    //                  lands on the identical position with the identical mask
    //                  shape. RoPE therefore sees the same relative offsets at
    //                  every step and the recurrence cannot drift the way a
    //                  growing span does. Costs ONE position for the whole
    //                  segment instead of latent_steps.
    // Folding deliberately contradicts the "keep the state MOVING" hypothesis
    // for depth collapse: it tests whether the collapse is instead an
    // O(n)-drift artefact of an ever-lengthening sequence.
    int32_t      latent_position_mode;

    // Execute the whole latent span as ONE llama_decode call carrying
    // latent_steps embedding rows, instead of latent_steps separate one-row
    // decodes. Only meaningful with position_mode=folded (every row re-uses
    // the same KV slot, which llama_batch_allocr explicitly permits for
    // embedding inputs); shared mode would need N distinct positions and is
    // therefore left on the per-step path. Default 0 = per-step (legacy).
    int32_t      latent_batch_mode;

    // Count latent steps against the agent's token budget: stop the span once
    // (already-generated positions + steps taken this segment) would exceed
    // this many. <= 0 disables (legacy). This makes "a latent step costs one
    // token of context" explicit instead of implicit, so a long think block
    // cannot silently overrun max_tokens / n_ctx. Independent of the hard
    // n_ctx guard in the loop, which only catches the absolute ceiling.
    int32_t      latent_max_tokens_budget;

    // Alternate latent (thinking) and text segments automatically every K text
    // tokens. <= 0 disables (legacy: the caller writes the segment list).
    //
    // This is the composition-level half of the hybrid loop. Today the caller
    // must spell out every tokens/latent boundary, which means the "how long do
    // I think" decision is hard-coded per call site and cannot react to what
    // the model is producing. With this set, a `latent` segment followed by a
    // sampling segment will emit at most K text tokens before running another
    // latent span of latent_steps, alternating until max_tokens is reached or
    // the model emits EOG. K <= 0 keeps the legacy one-shot behaviour.
    int32_t      latent_interleave;

    // Which layer's hidden state feeds the latent recurrence.
    //   0 = layer-0 INPUT embedding (default; the space batch.embd enters)
    //  -1 = final-layer output state (the legacy fallback)
    //   N = output of layer N (1..n_layer), i.e. res->get_layer_inp(N)
    // Rationale: the layer-0 tap is in the right SPACE but is the least
    // processed representation. A deeper layer has more task-relevant content
    // while still being a pre-output residual, so it may survive recurrence
    // better. This is the knob to A/B if depth is the goal.
    int32_t      latent_source_layer;

    // Which vector is fed back each latent step.
    //   0 = layer-0 INPUT tap (this fork's historical default)
    //   1 = FINAL-LAYER hidden state, as upstream LatentMAS does
    // Upstream applies Wa to the LAST layer (Wa is the output->input map) and
    // never feeds hidden_states[0] back. Mode 0 applies Wa to a vector already
    // in input space, which is the wrong domain - kept as the default only so
    // existing behaviour is unchanged until measured.
    int32_t      latent_feedback_source;

    // filled in by the call
    int32_t      pos_start;     // first position written
    int32_t      n_positions;   // positions written
    int32_t      n_tokens_out;  // TOKENS: tokens sampled (0 or 1)
    int32_t      n_hidden_out;  // LATENT: hidden vectors captured (== latent_steps)
};

/*
 * Per-process latent statistics. Every latent segment updates these, so an A/B
 * over toggles can be read from one endpoint instead of inferred from logs.
 *   n_latent_steps   - total latent vectors fed back
 *   n_kv_positions   - total positions written (tokens + latent)
 *   n_tokens_sampled - real tokens emitted (the thing latent is meant to save)
 *   n_wall_ms        - wall time inside latent loops only
 *   max_depth        - deepest latent segment seen
 */
struct llama_mas_latent_stats {
    uint64_t n_latent_segments;
    uint64_t n_latent_steps;
    uint64_t n_kv_positions;
    uint64_t n_tokens_sampled;
    uint64_t n_latent_ms;
    uint64_t n_text_ms;
    int32_t  max_depth;
    double   last_cos_drift;   // cosine vs previous step, last step only
};

struct llama_mas_turn_result {
    int32_t     n_positions;            // total positions written across all segments
    int32_t     n_tokens;               // total tokens sampled
    int32_t     n_latent_steps;         // total latent steps executed
    // Perplexity of the emitted text under the model itself:
    //   perplexity = exp(-logprob_sum / logprob_n)
    // Objective quality signal for free-form long text, where repetition proxies
    // cannot tell fluent prose from fluent-sounding degenerate loops. Latent
    // steps emit no tokens, so they do not enter the average.
    double      logprob_sum;
    int64_t     logprob_n;
    float       think_tokens_per_sec;   // latent steps / second (0 when unavailable)
    float       text_tokens_per_sec;    // sampled tokens / second (0 when unavailable)
    char *      text;                   // sampled text, malloc'd (caller frees with free())
    llama_token * tokens;               // sampled token ids, malloc'd (caller frees with free())
    int32_t     n_tokens_sampled;
};

// Drive one hybrid turn on agent_idx.
// segments/n_segments describe the turn; out_result receives timings and text.
// out_tokens/out_n_tokens may be NULL. Returns 0 on success, negative on error.
LLAMA_API int llama_mas_run_turn(
    struct llama_mas_context * mas,
    int32_t agent_idx,
    const struct llama_mas_segment * segments,
    int32_t n_segments,
    struct llama_mas_turn_result * out_result
);

// Free a llama_mas_turn_result's text/tokens (NULL-safe).
LLAMA_API void llama_mas_free_turn_result(struct llama_mas_turn_result * result);

// Enable hidden-state capture on the context (thin passthrough used by the
// turn driver to toggle capture per segment).
LLAMA_API void llama_mas_set_capture_hidden_states(
    struct llama_mas_context * mas,
    bool enable
);

// ---------------------------------------------------------------------------
// Session reuse (hybrid turns)
//
// One agent's KV sequence IS the session. These let a caller resume it on a
// later turn without re-decoding the shared prefix.
//
// The hit test is TOKEN ids, not text: the cache holds tokens, and an edit that
// looks cosmetic can tokenize differently. Text is kept in the ledger only for
// humans (diffing, debugging a cache miss).
// ---------------------------------------------------------------------------

// Current tail position of an agent's sequence (prefilled prompt + generated).
LLAMA_API size_t llama_mas_agent_pos(const struct llama_mas_context * mas, int32_t agent_idx);

// Drop everything past `keep` positions. Reuse = truncate(keep) + re-decode.
LLAMA_API int llama_mas_agent_truncate(struct llama_mas_context * mas, int32_t agent_idx, size_t keep);

// Length of the leading sequence that `new_tokens` still matches. Positions
// [0, keep) are reusable; [keep, ...) must be re-decoded.
LLAMA_API size_t llama_mas_session_match_len(const struct llama_mas_context * mas,
                                             const llama_token * new_tokens, size_t n_new);

// Ledger introspection (index 0..llama_mas_turn_log_count()-1).
LLAMA_API int    llama_mas_turn_log_count(const struct llama_mas_context * mas);
LLAMA_API int    llama_mas_turn_log_kind(const struct llama_mas_context * mas, int32_t i);
LLAMA_API const char * llama_mas_turn_log_text(const struct llama_mas_context * mas, int32_t i);
LLAMA_API size_t llama_mas_turn_log_pos_start(const struct llama_mas_context * mas, int32_t i);
LLAMA_API size_t llama_mas_turn_log_pos_end(const struct llama_mas_context * mas, int32_t i);

// Save/restore an agent's KV sequence TOGETHER with the position ledger.
// llama_mas_session_export/import only move KV bytes; without the ledger the
// next turn cannot tell which positions are valid, so use this pair to persist
// a hybrid session across a restart. save returns 0 on success, -2 if cap is
// too small (*out_size still holds the needed size), -3 on a short read.
LLAMA_API int llama_mas_session_save(struct llama_mas_context * mas, int32_t agent_idx,
                                     uint8_t * dst, size_t cap, size_t * out_size);
LLAMA_API int llama_mas_session_restore(struct llama_mas_context * mas, int32_t agent_idx,
                                        const uint8_t * src, size_t size);

// Serialize/restore an agent's KV sequence (for surviving a process restart).
// export returns 0 on success, -2 if the buffer is too small (*out_size is
// still filled so the caller can retry), -3 on a short read.
LLAMA_API int llama_mas_session_export(struct llama_mas_context * mas, int32_t agent_idx,
                                       uint8_t * dst, size_t cap, size_t * out_size);
LLAMA_API int llama_mas_session_import(struct llama_mas_context * mas, int32_t agent_idx,
                                       const uint8_t * src, size_t size);

// Get the hidden state of the last output token for an agent.
// Returns float[n_embd] pointer into context memory (valid until next decode).
// Works best with CHAIN strategy where each agent decodes individually.
// Returns NULL if hidden states are not available.
LLAMA_API float * llama_mas_get_agent_hidden_state(
    struct llama_mas_context * mas,
    int32_t agent_idx,
    int32_t * n_embd
);

// Compute the alignment matrix Wa = W_in * W_out^† for latent thought injection.
// This maps last-layer hidden states to valid input embeddings to avoid OOD activations.
// Returns 0 on success, -1 on failure.
LLAMA_API int llama_mas_compute_alignment_matrix(
    struct llama_mas_context * mas
);

// Disk cache for the alignment matrix (G: persist across processes).
//
// The solve costs 200-380 s on Qwen3-1.7B Q8, and the matrix is currently
// per-process, so every server restart pays it again. These let a caller load a
// previously computed matrix instead.
//
// The cache key is (model description, n_embd, n_vocab, stride, ridge), so a
// different model or a different stride can never silently reuse an
// incompatible matrix - a key mismatch is a MISS, not a wrong answer.
//
// Returns 0 on success (matrix loaded and active), 1 on cache miss, negative on
// error. The cache is used only when use_alignment is enabled, so it can never
// change the behaviour of a caller that never asked for alignment.
LLAMA_API int llama_mas_load_alignment_matrix(
    struct llama_mas_context * mas,
    const char * path
);

// Write the CURRENT in-memory alignment matrix to `path`. Returns 0 on success.
// Writes to a temp file and renames, so an interrupted save cannot leave a
// half-written cache that a later load would trust.
LLAMA_API int llama_mas_save_alignment_matrix(
    struct llama_mas_context * mas,
    const char * path
);

// Apply alignment matrix to a hidden state vector.
// dst[n_embd] = Wa * src[n_embd]
// Returns 0 on success, -1 on failure.
LLAMA_API int llama_mas_apply_alignment(
    struct llama_mas_context * mas,
    const float * src,
    float * dst,
    int32_t n_embd
);

// Enable/disable alignment matrix usage for latent thought injection.
// When enabled, hidden states are aligned with Wa before injection.
LLAMA_API void llama_mas_set_use_alignment(
    struct llama_mas_context * mas,
    bool use_alignment
);

// Get whether alignment matrix is currently enabled.
LLAMA_API bool llama_mas_get_use_alignment(
    llama_mas_context * mas
);

// Vocab stride used by the alignment least-squares. 1 = exact (every vocab row);
// higher values subsample and trade accuracy for speed. The exact solve costs
// n_vocab*n_embd^2 MACs, which is minutes-to-never on CPU above ~512 dims.
LLAMA_API void llama_mas_set_alignment_vocab_stride(
    llama_mas_context * mas, int32_t stride
);
LLAMA_API int32_t llama_mas_get_alignment_vocab_stride(
    const llama_mas_context * mas
);

// Ridge (lambda) as a FRACTION of the mean Gram diagonal, used when inverting
// G = W_out^T W_out + lambda*I. <= 0 selects the legacy default (1e-6). It is
// part of the alignment cache key, so changing it and re-loading an old cache
// is a MISS rather than a silently different matrix. Requires a recompute to
// take effect.
LLAMA_API void llama_mas_set_alignment_ridge(
    llama_mas_context * mas, double ridge
);
LLAMA_API double llama_mas_get_alignment_ridge(
    const llama_mas_context * mas
);

// ── A-section alignment solve knobs (all default OFF / legacy) ─────────────
LLAMA_API void llama_mas_set_alignment_full_vocab(llama_mas_context * mas, bool on);
LLAMA_API bool llama_mas_get_alignment_full_vocab(const llama_mas_context * mas);
LLAMA_API void llama_mas_set_alignment_lowrank(llama_mas_context * mas, int32_t r);
LLAMA_API int32_t llama_mas_get_alignment_lowrank(const llama_mas_context * mas);
LLAMA_API void llama_mas_set_alignment_dtype(llama_mas_context * mas, int32_t dtype);
LLAMA_API int32_t llama_mas_get_alignment_dtype(const llama_mas_context * mas);

// Cross-pass diagnostic (default OFF, behaviour-neutral). When enabled, the
// alignment cross pass A = W_in * W_out^T prints a preflight of the peak
// working set and then a progress line every ~5% of strided vocab rows,
// including a sampled sum of A. This exists to make the 8B alignment crash
// observable: on that model the dequant + Gram + invert all succeed and the
// process dies SILENTLY inside the cross pass (bare SERVER_EXIT=-1). Off by
// default; when off the only change is the one-line preflight print, which is
// unconditional because it costs nothing and attributes an OOM to the cross pass.
LLAMA_API void llama_mas_set_alignment_cross_progress(llama_mas_context * mas, bool on);
LLAMA_API bool llama_mas_get_alignment_cross_progress(const llama_mas_context * mas);

// LatentMAS-style per-step norm re-scale: rescale each aligned latent vector to
// the mean norm of the real token embeddings (target_norm), so its magnitude
// stays on-distribution across steps. Off by default; enabling it reproduces
// LatentMAS 'latent_space_realign'. Requires the alignment matrix to have been
// computed (target_norm is derived there).
// Threads for the per-step projection matmul (dst = Wa * src).
// Default 4. Set <= 1 to restore the original single-threaded scalar loop.
// Output is bit-identical at any thread count (rows are independent).
// The projection costs n_embd^2 MACs every latent step (16.7M at 4096), so this
// is a measurable speed knob on wide models.
// Set an agent's repetition penalty after creation (rebuilds its sampler).
// repeat_penalty 1.0 = disabled; repeat_last_n <= 0 means 64.
LLAMA_API int llama_mas_set_repeat_penalty(struct llama_mas_context * mas,
                                           int32_t agent_idx, float repeat_penalty,
                                           int32_t repeat_last_n,
                                           float penalty_freq, float penalty_present);

LLAMA_API void llama_mas_set_alignment_parallel_threads(
        llama_mas_context * mas, int32_t n_threads);
LLAMA_API int32_t llama_mas_get_alignment_parallel_threads(
        const llama_mas_context * mas);

LLAMA_API void llama_mas_set_alignment_rescale_norm(
    llama_mas_context * mas, bool on
);
LLAMA_API bool llama_mas_get_alignment_rescale_norm(
    const llama_mas_context * mas
);
LLAMA_API double llama_mas_get_alignment_target_norm(
    const llama_mas_context * mas
);

// Copy the process-wide latent statistics into *out. Returns 0 on success.
LLAMA_API int llama_mas_get_latent_stats(
    const llama_mas_context * mas,
    struct llama_mas_latent_stats * out
);

// Reset the statistics (used between A/B runs).
LLAMA_API void llama_mas_reset_latent_stats(
    llama_mas_context * mas
);

// Dimensions/state of the computed alignment matrix. n_embd is 0 when no
// matrix has been computed yet; is_identity is true when the tap is passing
// input through unchanged (i.e. alignment is a no-op).
LLAMA_API int32_t llama_mas_get_alignment_n_embd(
    llama_mas_context * mas
);
LLAMA_API bool llama_mas_alignment_is_identity(
    llama_mas_context * mas
);

// Set per-agent prompt text (overrides global task_prompt for that agent).
LLAMA_API void llama_mas_set_agent_prompt(
    struct llama_mas_context * mas,
    int32_t agent_idx,
    const char * prompt
);

// Set strategy for the next run.
LLAMA_API void llama_mas_set_strategy(
    struct llama_mas_context * mas,
    enum llama_mas_strategy strategy
);

// Configure broadcast wave groups for CHAIN_BROADCAST strategy.
// waves: array of wave definitions, each wave is a list of agent indices.
// n_waves: number of waves.
// e.g. waves = [[0], [1,2], [3]] means: agent 0 solo, then agents 1+2 broadcast, then agent 3 solo.
// Call before llama_mas_run(). Pass NULL or n_waves=0 to auto-detect from agent order.
LLAMA_API void llama_mas_set_broadcast_waves(
    struct llama_mas_context * mas,
    int32_t ** waves,
    int32_t * wave_sizes,
    int32_t n_waves
);

// Run one decode step for all active agents.
// Builds a single batch, calls llama_decode once, extracts logits,
// applies per-agent adapter, samples next tokens.
// Returns 0 on success, non-zero on error.
LLAMA_API int llama_mas_decode_step(struct llama_mas_context * mas);

// Check if all agents have finished (hit EOS or max_tokens).
LLAMA_API bool llama_mas_all_finished(const struct llama_mas_context * mas);

// Get the generated tokens for an agent.
LLAMA_API const llama_token * llama_mas_get_tokens(
    const struct llama_mas_context * mas,
    int32_t agent_idx,
    size_t * out_count
);

// Get the generated text for an agent (detokenized).
// Caller must free() the returned string.
LLAMA_API char * llama_mas_get_text(
    const struct llama_mas_context * mas,
    int32_t agent_idx
);

// Free text returned by llama_mas_get_text (avoids CRT heap mismatch)
LLAMA_API void llama_mas_free_text(char * text);

// KV cache: clone from source agent to destination agent.
LLAMA_API int llama_mas_kv_clone(
    struct llama_mas_context * mas,
    int32_t src_agent_idx,
    int32_t dst_agent_idx
);

// KV cache: clone a specific row range from source agent to destination agent.
// Copies KV cells from src_agent at [src_pos_start, src_pos_end) to dst_agent at dst_pos_start.
LLAMA_API int llama_mas_kv_clone_range(
    struct llama_mas_context * mas,
    int32_t src_agent_idx,
    int32_t dst_agent_idx,
    int32_t src_pos_start,
    int32_t src_pos_end,
    int32_t dst_pos_start
);

// Set per-agent layer graph (dynamic transformer layer execution).
// Each agent can use a different subset of layers. If NULL, all layers are used.
//
// DEFINITION 2026-11-19: this was declared in the public header but had NO
// DEFINITION anywhere in the tree - a link trap for any caller, and the reason
// the per-agent `layer_graph` field (read at ~10 decode sites) was always NULL:
// the whole per-agent layer-graph path was dead code. Now defined.
//
// The caller retains ownership of `lg` (same contract as the layer-graph API,
// which frees only graphs it created via llama_layer_graph_init). Passing NULL
// clears the agent's graph (full layer set). The graph is only consulted for
// LLM_ARCH_LLAMA-family models: models/llama.cpp gates the enabled/disabled
// branch on `params.layer_graph_ctx`, so an agent's graph has no effect on a
// model whose graph builder ignores it - it is not a universal layer switch.
//
// SAFETY: the graph outlives the agent. llama_mas_free() and
// llama_mas_clear_agents() free any NON-NULL layer graph owned by an agent, so
// a caller that never clears it cannot leak; a caller that frees the graph
// itself must clear the agent's pointer (or pass NULL) BEFORE freeing it, or
// the decode path will dereference a dangling pointer. Passing NULL is the
// documented "release" path.
LLAMA_API int llama_mas_set_agent_layer_graph(
    struct llama_mas_context * mas,
    int32_t agent_idx,
    struct llama_layer_graph_context * lg
);

// Get the underlying llama_context from the MAS context.
// This allows creating per-agent layer graphs.
LLAMA_API struct llama_context * llama_mas_get_context(struct llama_mas_context * mas);

// Attach a layer graph to the shared context, sizing the scheduler for the
// WORST CASE (the full layer set). A partial layer subset changes the graph
// shape, so the boot-time buffer assignment is stale; but reserving WHILE the
// partial graph is attached sizes the buffers for the SMALL graph and the next
// full-graph decode aborts with GGML_ASSERT(buffer). So this detaches, reserves
// the full shape, then re-attaches. Returns 0 on success.
LLAMA_API int llama_context_attach_layer_graph_reserved(
    struct llama_context * ctx,
    struct llama_layer_graph_context * lg
);

// Agent introspection (for state persistence across context recreation)
struct llama_mas_agent_info {
    const char * role;
    llama_seq_id seq_id;
    int32_t      adapter_type;
    int32_t      max_tokens;
    float        temperature;
    float        top_p;
    int32_t      top_k;
    const char * prompt_text; // NULL if empty
    const char * uuid;        // Persistent agent identifier
    bool         stop_after_prompt; // Encode-only agent (no generation)
    bool         is_latent;         // Latent agent: generates tokens but no text output
    int32_t      latent_steps;      // How many latent thought tokens to generate
    double       runtime_ms;        // Wall-clock duration for this agent in current run
};

// Get number of active agents.
LLAMA_API int32_t llama_mas_agent_count(const struct llama_mas_context * mas);

// Get agent info by index. Returns 0 on success, -1 if idx is out of bounds.
LLAMA_API int32_t llama_mas_agent_get_info(
    const struct llama_mas_context * mas,
    int32_t agent_idx,
    struct llama_mas_agent_info * out_info
);

// Single-context generation: encode prompt + generate tokens in one native C++ call.
// Zero Python overhead during generation. Returns a malloc'd string.
// Caller must free() the result. Returns NULL on error.
LLAMA_API char * llama_generate_single(
    struct llama_context * ctx,
    struct llama_model * model,
    const char * prompt,
    int32_t max_tokens,
    float temperature,
    float top_p,
    int32_t top_k,
    float repeat_penalty
);

// Convenience: run full generation loop until all agents finish.
// task_prompt is tokenized and set as the prompt for all agents.
LLAMA_API int llama_mas_run(
    struct llama_mas_context * mas,
    const char * task_prompt,
    int32_t max_tokens,
    enum llama_mas_strategy strategy
);

//
// Adapter helper: build adapter weights from PyTorch float arrays.
// Used from Python/ctypes to upload trained adapter weights.
//

LLAMA_API struct llama_mas_adapter_weights * llama_mas_adapter_weights_init(
    int32_t n_embd,
    int32_t in_dim,
    int32_t out_dim,
    int32_t hidden_dim
);

LLAMA_API void llama_mas_adapter_weights_free(
    struct llama_mas_adapter_weights * weights
);

// ---------------------------------------------------------------------------
// Adapter weight upload (for training / Python binding)
// ---------------------------------------------------------------------------
// Each setter copies data from the provided float arrays into the adapter
// weight structure.  Call these after llama_mas_adapter_weights_init()
// and before attaching the adapter to an agent.
//
// Naming convention:  w=weight, b=bias
//
LLAMA_API void llama_mas_adapter_weights_set_cross_ln_source(
    struct llama_mas_adapter_weights * weights,
    const float * w, const float * b, int32_t in_dim
);
LLAMA_API void llama_mas_adapter_weights_set_cross_ln_target(
    struct llama_mas_adapter_weights * weights,
    const float * w, const float * b, int32_t out_dim
);
LLAMA_API void llama_mas_adapter_weights_set_cross_proj1(
    struct llama_mas_adapter_weights * weights,
    const float * w, const float * b, int32_t in_dim, int32_t hidden_dim
);
LLAMA_API void llama_mas_adapter_weights_set_cross_proj2(
    struct llama_mas_adapter_weights * weights,
    const float * w, const float * b, int32_t hidden_dim, int32_t out_dim
);
LLAMA_API void llama_mas_adapter_weights_set_cross_residual(
    struct llama_mas_adapter_weights * weights,
    const float * w, const float * b, int32_t in_dim, int32_t out_dim
);

// Attach adapter weights to an existing agent.
// agent_idx: index returned by llama_mas_add_agent_flat / llama_mas_add_agent
// If adapter_type is LLAMA_MAS_ADAPTER_NONE, weights are ignored.
LLAMA_API void llama_mas_set_agent_adapter(
    struct llama_mas_context * mas,
    int32_t agent_idx,
    enum llama_mas_adapter_type adapter_type,
    struct llama_mas_adapter_weights * weights
);

//
// Adapter compute: apply selected adapter type to a hidden state vector.
// These operate on CPU float arrays (small, fast for single-token vectors).
// For GPU-resident hidden states, the MAS decode loop will call these
// after synchronizing or use ggml graph variants.
//

// RecursiveLink forward on CPU float arrays.
// hidden_in  : [n_embd]
// hidden_out : [n_embd]  (can alias hidden_in)
LLAMA_API void llama_mas_adapter_recursive_forward(
    const struct llama_mas_adapter_weights * weights,
    const float * hidden_in,
    float * hidden_out
);

// CrossLink forward on CPU float arrays.
// hidden_in  : [in_dim]
// hidden_out : [out_dim]
LLAMA_API void llama_mas_adapter_cross_forward(
    const struct llama_mas_adapter_weights * weights,
    const float * hidden_in,
    float * hidden_out
);

// Mean pooling: output = mean(hidden_states across n_tokens)
// states: [n_tokens * n_embd]
// output: [n_embd]
LLAMA_API void llama_mas_adapter_mean_pool(
    const float * states,
    int32_t n_tokens,
    int32_t n_embd,
    float * output
);

// Concatenate hidden states across tokens (no-op if n_tokens==1)
// output must have size n_tokens * n_embd
LLAMA_API void llama_mas_adapter_concat(
    const float * states,
    int32_t n_tokens,
    int32_t n_embd,
    float * output
);

// ---------------------------------------------------------------------------
// Hidden state encoding (C++ native, replaces Python agent.encode)
// ---------------------------------------------------------------------------

// Encode a prompt with the default agent (agent 0) and return the last-layer hidden state.
// Returns a float[n_tokens * n_embd] array. Caller must free with llama_mas_free_latent.
LLAMA_API float * llama_mas_encode(
    struct llama_mas_context * mas,
    const char * prompt,
    int32_t * out_n_tokens,
    int32_t * out_n_embd
);

// Encode a prompt with a specific agent and return the last-layer hidden state.
// If the agent has an adapter, it is applied to the hidden state.
// Returns a float[n_tokens * n_embd] array. Caller must free with llama_mas_free_latent.
LLAMA_API float * llama_mas_encode_agent(
    struct llama_mas_context * mas,
    int32_t agent_idx,
    const char * prompt,
    int32_t * out_n_tokens,
    int32_t * out_n_embd
);

// CrossLink: apply source agent's adapter to target agent's hidden state.
// src_hidden: [n_tokens * n_embd] from source agent.
// Returns a newly allocated float array [n_tokens * target_n_embd].
// Caller must free with llama_mas_free_latent.
LLAMA_API float * llama_mas_cross_link(
    struct llama_mas_context * mas,
    int32_t src_agent_idx,
    int32_t dst_agent_idx,
    const float * src_hidden,
    int32_t n_tokens,
    int32_t n_embd
);

// Capture hidden state from src agent, apply CrossLink projection, and set as dst agent embeddings.
// Uses src agent's adapter_weights for projection. If src has no adapter, copies state as-is.
// Returns 0 on success, -1 on failure.
LLAMA_API int llama_mas_project_agent_to_embeddings(
    struct llama_mas_context * mas,
    int32_t src_agent_idx,
    int32_t dst_agent_idx
);

// Free latent / hidden state arrays returned by encode functions.
LLAMA_API void llama_mas_free_latent(float * latent);

// ---------------------------------------------------------------------------
// GPU-resident hidden state tensor access (zero-copy for CUDA adapters)
// ---------------------------------------------------------------------------

// Get the GPU tensor pointer for an agent's hidden state after the last decode.
// Returns a ggml_tensor* cast to void*, or NULL if not available.
// The tensor is valid until the next llama_decode() call.
LLAMA_API void * llama_mas_agent_get_hidden_state_tensor(
    struct llama_mas_context * mas,
    int32_t agent_idx,
    int32_t layer
);

// ---------------------------------------------------------------------------
// CUDA / Performance toggles (default = true when CUDA is available)
// ---------------------------------------------------------------------------

// Enable/disable CUDA adapter forward (applies to encode_agent and cross_link).
// When true (default), adapter weights are kept on GPU and applied via CUDA kernels.
LLAMA_API void llama_mas_set_cuda_adapters(struct llama_mas_context * mas, bool use_cuda);
LLAMA_API bool llama_mas_get_cuda_adapters(const struct llama_mas_context * mas);

// Enable/disable batched multi-agent decode.
// When true (default), llama_mas_decode_step merges all active agents into a single llama_decode call.
// Falls back to per-agent decode if agents have incompatible layer graphs.
LLAMA_API void llama_mas_set_batched_decode(struct llama_mas_context * mas, bool use_batched);
LLAMA_API bool llama_mas_get_batched_decode(const struct llama_mas_context * mas);

// Enable/disable backend (GPU) sampling.
// When true (default), sampler chains are initialized on the backend and run on GPU.
LLAMA_API void llama_mas_set_backend_sampling(struct llama_mas_context * mas, bool use_backend);
LLAMA_API bool llama_mas_get_backend_sampling(const struct llama_mas_context * mas);

// Re-create samplers for all agents to apply backend_sampling toggle.
LLAMA_API void llama_mas_refresh_samplers(struct llama_mas_context * mas);

// Enable/disable tokenization cache.
// When true (default), repeated prompts are tokenized once and cached.
LLAMA_API void llama_mas_set_tokenization_cache(struct llama_mas_context * mas, bool use_cache);
LLAMA_API bool llama_mas_get_tokenization_cache(const struct llama_mas_context * mas);

// Enable/disable cached KV shift graph.
// When true (default), the K-shift graph is built once and reused, only input values are updated.
LLAMA_API void llama_mas_set_cached_shift_graph(struct llama_mas_context * mas, bool use_cache);
LLAMA_API bool llama_mas_get_cached_shift_graph(const struct llama_mas_context * mas);

// Enable/disable GPU hidden state retention.
// When true (default), hidden states are kept on GPU and D2H copies are deferred.
LLAMA_API void llama_mas_set_gpu_hidden_states(struct llama_mas_context * mas, bool use_gpu);
LLAMA_API bool llama_mas_get_gpu_hidden_states(const struct llama_mas_context * mas);

// Enable/disable async tokenization.
// When true (default), agent N+1 prompt is tokenized while agent N decodes.
LLAMA_API void llama_mas_set_async_tokenization(struct llama_mas_context * mas, bool use_async);
LLAMA_API bool llama_mas_get_async_tokenization(const struct llama_mas_context * mas);

// ---------------------------------------------------------------------------
// Streaming callback
// ---------------------------------------------------------------------------

// Called after every token generation step for every active agent.
// agent_idx: which agent produced the token
// token: the sampled llama_token
// step: monotonically increasing step counter (resets each llama_mas_run)
// user_data: opaque pointer passed through from llama_mas_set_stream_callback
//
// Note: the callback is invoked synchronously from within the decode loop,
// so it should be fast (e.g. push to a lock-free queue). Heavy work should
// be deferred to another thread.
typedef void (*llama_mas_stream_cb)(
    int32_t agent_idx,
    llama_token token,
    int32_t step,
    void * user_data
);

LLAMA_API void llama_mas_set_stream_callback(
    struct llama_mas_context * mas,
    llama_mas_stream_cb cb,
    void * user_data
);

// ---------------------------------------------------------------------------
// Timing / profiling
// ---------------------------------------------------------------------------

struct llama_mas_timing {
    double prefill_ms;       // time spent decoding prompts
    double generation_ms;    // time spent generating tokens
    int64_t n_prefill_tokens;
    int64_t n_generated_tokens;
    double tokens_per_sec;   // n_generated_tokens / (generation_ms / 1000.0)
    double prefill_tokens_per_sec; // n_prefill_tokens / (prefill_ms / 1000.0)
};

// Fill the timing struct with stats from the last llama_mas_run() call.
// If no run has happened yet, all fields are zero.
LLAMA_API void llama_mas_get_timing(
    const struct llama_mas_context * mas,
    struct llama_mas_timing * timing
);

#ifdef __cplusplus
}
#endif
