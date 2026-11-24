#include "llama-layer-graph.h"
#include <iostream>

#ifdef GGML_USE_CUDA
// Forward declarations for CUDA runtime (avoid needing cuda_runtime.h in cl.exe path)
extern "C" {
    int cudaMalloc(void ** devPtr, size_t size);
    int cudaFree(void * devPtr); // resolved from cudart when GGML_USE_CUDA is defined
    int cudaMemcpy(void * dst, const void * src, size_t count, int kind);
    int cudaDeviceSynchronize();
}
constexpr int cudaMemcpyHostToDevice = 1;
constexpr int cudaMemcpyDeviceToHost = 2;
#include "ggml-backend.h"
extern "C" {
    void ggml_cuda_mas_recursive_forward(
        const float * hidden_in, float * hidden_out,
        const float * w_pre_ln, const float * b_pre_ln,
        const float * w_proj1, const float * b_proj1,
        const float * w_proj2, const float * b_proj2,
        const float * w_post_ln, const float * b_post_ln,
        float * temp, int n_embd);
    void ggml_cuda_mas_cross_forward(
        const float * hidden_in, float * hidden_out,
        const float * w_cross_ln_source, const float * b_cross_ln_source,
        const float * w_cross_proj1, const float * b_cross_proj1,
        const float * w_cross_proj2, const float * b_cross_proj2,
        const float * w_cross_residual, const float * b_cross_residual,
        const float * w_cross_ln_target, const float * b_cross_ln_target,
        float * temp1, float * temp2, float * temp3,
        int in_dim, int out_dim, int hidden_dim);
    void ggml_cuda_mas_mean_pool(
        const float * states, float * output, int n_tokens, int n_embd);
    void ggml_cuda_mas_concat(
        const float * states, float * output, int n_tokens, int n_embd);
    void * ggml_cuda_mas_device_malloc(size_t size);
    void   ggml_cuda_mas_device_free(void * ptr);
    void   ggml_cuda_mas_memcpy_h2d(void * dst, const void * src, size_t size);
    void   ggml_cuda_mas_memcpy_d2h(void * dst, const void * src, size_t size);
}
#endif
#include "llama-mas.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-batch.h"
#include "llama-sampler.h"

#include <cstring>
#include <cmath>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <random>
#include <chrono>

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
static inline double mas_time_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// MAS context and agent state
//
// Reconstructed 2026-11-16 after a bad splice destroyed the originals; the
// public header is unchanged.
//
// Layout contract (relied on by the turn driver and truncate):
//   agent.prompt_tokens     = the agent's prompt ids
//   agent.generated_tokens  = prompt ids THEN every generated id; entry i is
//                             exactly the token that wrote KV position i, so
//                             truncation is a resize + KV seq_rm
//   agent.prompt_pos        = positions covered by the prompt
//   agent.generated_len     = positions written by generation; a latent step
//                             consumes one position just like a token
//   => sequence tail = prompt_pos + generated_len
// ---------------------------------------------------------------------------

// Forward declaration: defined below the run_turn sampling site, but needed
// there to score the token actually emitted.
double llama_mas_logprob_of(const float * logits, int64_t n_vocab, llama_token tok);

struct llama_mas_agent {
    std::string role;
    llama_seq_id seq_id = 0;

    llama_mas_adapter_type adapter_type = LLAMA_MAS_ADAPTER_NONE;
    std::unique_ptr<llama_mas_adapter_weights, void (*)(llama_mas_adapter_weights *)> adapter_weights;

    int32_t max_tokens = 256;
    float   temperature = 0.7f;
    float   top_p = 0.9f;
    int32_t top_k = 40;
    float   repeat_penalty = 1.1f;

    bool    stop_after_prompt = false;  // encode only
    bool    is_latent = false;          // latent positions, no text output
    int32_t latent_steps = 0;
    int32_t latent_steps_remaining = 0;

    double run_started_ms = 0.0;
    double run_finished_ms = 0.0;

    std::string prompt_text;
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> generated_tokens;

    bool   finished = false;
    size_t prompt_pos = 0;
    size_t generated_len = 0;

    // Perplexity instrumentation for this agent's emitted text.
    // perplexity = exp(-logprob_sum / logprob_n); latent steps emit no tokens and
    // are excluded. Accumulated at each sample site by llama_mas_logprob_of().
    double  logprob_sum = 0.0;
    int64_t logprob_n = 0;

    // Repetition penalty controls (formerly dead: repeat_penalty was defaulted and
    // never applied). repeat_last_n <= 0 falls back to 64.
    float   penalty_freq = 0.0f;
    float   penalty_present = 0.0f;
    int32_t repeat_last_n = 64;

    // Hidden state of this agent's current tail position. Persisted per agent
    // (not a local in run_turn) so a resumed session can continue a latent run
    // from the exact state it left off: the KV alone restores the attention
    // history, but the next latent input IS this vector. Losing it forces the
    // old seed-token path, which re-writes the last text token to a new
    // position - the KV then holds the same id twice.
    std::vector<float> tail_hidden;
    bool   have_tail_hidden = false;

    llama_layer_graph_context * layer_graph = nullptr;

    std::vector<float> input_embeddings;
    int32_t input_embeddings_n_embd = 0;

    std::string uuid;

    llama_mas_agent()
        : adapter_weights(nullptr, llama_mas_adapter_weights_free) {}
    llama_mas_agent(const llama_mas_agent &) = delete;
    llama_mas_agent & operator=(const llama_mas_agent &) = delete;
};

struct llama_mas_context {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    int32_t n_agents_max = 16;
    llama_mas_strategy strategy = LLAMA_MAS_STRATEGY_ROLE;

    std::vector<std::unique_ptr<llama_mas_agent>> agents;
    std::vector<std::vector<int32_t>> broadcast_waves;
    std::vector<llama_sampler *> samplers;

    std::mt19937 rng;

    bool use_cuda_adapters;
    bool use_batched_decode;
    bool use_backend_sampling;
    bool use_tokenization_cache;
    bool use_cached_shift_graph;
    bool use_gpu_hidden_states;
    bool use_async_tokenization;
    bool stop_after_prompt;

    // Alignment matrix Wa for latent thought injection (Wa = W_in * W_out^†)
    std::vector<float> alignment_matrix;
    int32_t alignment_n_embd = 0;
    bool use_alignment_matrix = false;
    bool alignment_is_identity = false;

    // Process-wide latent statistics (see llama_mas_latent_stats).
    llama_mas_latent_stats latent_stats{};

    // Vocab stride for the alignment least-squares (1 = exact / every row).
    // The exact solve is n_vocab*n_embd^2 MACs and is impractical above ~512
    // dims; a stride keeps the dominant Gram terms at a fraction of the cost.
    int32_t alignment_vocab_stride = 1;

    // Mean L2 norm of the real token embedding rows (LatentMAS target_norm).
    // LatentMAS rescales every aligned vector to this norm before feeding it
    // back; without it the magnitude drifts each step and quality collapses
    // after 1-2 steps. <= 0 means "not computed".
    double alignment_target_norm = 0.0;
    bool   alignment_has_target_norm = false;

    // Ridge (lambda) relative to the mean Gram diagonal, used when inverting
    // G = W_out^T W_out + lambda*I. The solve used a hardcoded 1e-6; exposed
    // here so the regularisation can be A/B'd. <= 0 means "use the legacy
    // default 1e-6". It is part of the alignment cache key, so two matrices
    // computed with different ridges can never be confused.
    double alignment_ridge = 0.0;

    // ── A-section knobs (all default OFF / legacy behaviour) ───────────────
    // Force the exact solve by pinning the vocab stride to 1, whatever the
    // configured stride says. Off by default so the strided fast path stays in
    // charge; on gives a reproducible exact matrix for A/B work.
    bool alignment_full_vocab = false;

    // Rank-r truncated solve: solve only the leading r rows/cols of the Gram
    // system. <= 0 or >= n_embd means the full solve. Inversion cost falls as
    // r^2; the Gram/cross passes still read every (strided) vocab row.
    int32_t alignment_lowrank = 0;

    // 1 = round each Gram product to bf16 precision before accumulating.
    // Measurement knob for how much precision the solve needs; NOT a speedup
    // (it adds work). 0 = fp32 accumulation (default).
    int32_t alignment_dtype = 0;

    // Apply that per-step norm rescale (LatentMAS 'latent_space_realign').
    // Switchable: off reproduces the old raw-matrix behaviour exactly.
    bool alignment_rescale_norm = false;

    // Threads for the per-step projection matmul (dst = Wa * src).
    // The projection is n_embd^2 MACs every latent step; at 4096 that is 16.7M
    // and is a real fraction of the step cost.
    //
    // DEFAULT 4, not 1, because the output is PROVABLY unchanged: measured on
    // Qwen3-8B at depth 32, threads 1/2/4/8 produced byte-identical text (same
    // sha256) while ms/step went 91.75 -> 74.03 -> 64.03 -> 64.28, i.e. 1.43x at
    // 4 threads and saturation beyond. Rows are independent, so there is no
    // reduction order to perturb - this is a pure speed knob, not a behaviour
    // change. Set to 1 to restore the original scalar loop exactly.
    int32_t alignment_parallel_threads = 4;

    // Diagnostic: when true, the cross pass (A = W_in * W_out^T) prints per-row
    // progress plus the predicted peak working set before it starts. Off by
    // default and behaviour-neutral when off. Added because the 8B alignment
    // crash dies SILENTLY inside the cross pass (dequant + Gram + invert all
    // succeed, then SERVER_EXIT=-1 with no diagnostic), and the cross pass is a
    // single-threaded n_vocab-strided × n_embd^2 accumulation with zero logging.
    // With this on, a crash names the exact row index and the live memory so the
    // failing line is observable instead of a bare exit code.
    bool alignment_cross_progress = false;

    // Hybrid turn ledger (one entry per llama_mas_run_turn segment).
    struct turn_segment_log {
        int32_t     kind;
        size_t      pos_start;
        size_t      pos_end;
        int32_t     n_tokens;
        int32_t     n_latent;
        uint32_t    tok_crc;
        std::string text;
        std::vector<llama_token> tokens; // hit test compares these
    };
    std::vector<turn_segment_log> turn_log;

    llama_mas_stream_cb stream_cb = nullptr;
    void * stream_user_data = nullptr;
    int32_t stream_step = 0;

    llama_mas_timing last_timing{};
    std::unordered_map<std::string, std::vector<llama_token>> token_cache;

    llama_mas_context()
        : rng((uint32_t) std::chrono::steady_clock::now().time_since_epoch().count()) {
#ifdef GGML_USE_CUDA
        use_cuda_adapters      = true;
        use_batched_decode     = true;
        use_backend_sampling   = false; // MAS samples after decode itself
        use_tokenization_cache = true;
        use_cached_shift_graph = true;
        use_gpu_hidden_states  = true;
        use_async_tokenization = true;
#else
        use_cuda_adapters      = false;
        use_batched_decode     = false;
        use_backend_sampling   = false;
        use_tokenization_cache = true;
        use_cached_shift_graph = false;
        use_gpu_hidden_states  = false;
        use_async_tokenization = false;
#endif
        stop_after_prompt      = false;
    }
};

// The ridge actually applied to the Gram diagonal, as a FRACTION of the mean
// diagonal. The solve hardcoded 1e-6; a caller can override it per-context
// (alignment_ridge). One helper so compute and the cache key can never disagree
// - they did before (compute used 1e-6, the key used 1e-5), which meant two
// different matrices could share a key.
static double mas_alignment_effective_ridge(const llama_mas_context * mas) {
    return (mas && mas->alignment_ridge > 0.0) ? mas->alignment_ridge : 1e-6;
}

// The stride the SOLVE actually uses. `alignment_full_vocab` forces stride 1
// regardless of the configured `alignment_vocab_stride` (see the compute path),
// so the configured value alone does NOT describe the matrix. The cache key
// must use this effective value.
//
// Keying on the configured stride let a full-vocab (stride-1) solve be saved
// under a stride-16 key and then accepted by a stride-16 context - a wrong
// matrix presented as a HIT, the exact failure this cache's key gate exists to
// prevent. Evidence on the shipped cache: its key says stride=16 but its
// target_norm is 1.539432 while a fresh stride-16 solve reports 1.542495, i.e.
// the cached matrix was built over all 151936 rows, not the 9496 the key
// claims. The `alignment_full_vocab` matrix entry asserted the key "already
// contains the effective stride"; it did not.
static int32_t mas_alignment_effective_stride(const llama_mas_context * mas) {
    const int32_t stride = (mas && mas->alignment_vocab_stride > 0)
        ? mas->alignment_vocab_stride : 1;
    return (mas && mas->alignment_full_vocab) ? 1 : stride;
}


// ---------------------------------------------------------------------------
// Session persistence: one binary blob per agent holding the KV sequence AND
// the ledger describing what occupies each position.
//
// The ledger has to travel with the KV or the next turn cannot tell which
// positions are still valid - KV bytes alone are a sequence with no record of
// what produced them.
// ---------------------------------------------------------------------------
#define MAS_SESSION_MAGIC 0x3153414Du  // "MAS1"

static void mas_put_u32(std::vector<uint8_t> & b, uint32_t v) {
    b.push_back((uint8_t)(v & 0xff));
    b.push_back((uint8_t)((v >> 8) & 0xff));
    b.push_back((uint8_t)((v >> 16) & 0xff));
    b.push_back((uint8_t)((v >> 24) & 0xff));
}

static void mas_put_u64(std::vector<uint8_t> & b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back((uint8_t)((v >> (8 * i)) & 0xff));
}

static bool mas_get_u32(const uint8_t * & p, const uint8_t * end, uint32_t & out) {
    if (p + 4 > end) return false;
    out = (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
    p += 4;
    return true;
}

static bool mas_get_u64(const uint8_t * & p, const uint8_t * end, uint64_t & out) {
    if (p + 8 > end) return false;
    out = 0;
    for (int i = 0; i < 8; ++i) out |= ((uint64_t) p[i]) << (8 * i);
    p += 8;
    return true;
}

int llama_mas_session_save(llama_mas_context * mas, int32_t agent_idx,
                           uint8_t * dst, size_t cap, size_t * out_size) {
    if (!mas || !mas->ctx) return -1;
    if (agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return -1;

    const size_t kv_need = llama_state_seq_get_size(mas->ctx, agent_idx);

    std::vector<uint8_t> body;
    mas_put_u32(body, (uint32_t) mas->turn_log.size());
    for (const auto & seg : mas->turn_log) {
        mas_put_u32(body, (uint32_t) seg.kind);
        mas_put_u64(body, (uint64_t) seg.pos_start);
        mas_put_u64(body, (uint64_t) seg.pos_end);
        mas_put_u32(body, (uint32_t) seg.tokens.size());
        mas_put_u32(body, (uint32_t) seg.n_latent);
        mas_put_u32(body, (uint32_t) seg.text.size());
        if (!seg.text.empty()) body.insert(body.end(), seg.text.begin(), seg.text.end());
        for (llama_token t : seg.tokens) mas_put_u32(body, (uint32_t) t);
    }

    // Tail hidden state, OUTSIDE the KV blob on purpose: llama_state_seq_* only
    // covers K/V tensors. The last residual vector is what the next latent step
    // consumes, so a session that resumes mid-thought needs it explicitly.
    const auto & ag = mas->agents[agent_idx];
    std::vector<uint8_t> tail;
    mas_put_u32(tail, ag->have_tail_hidden ? 1u : 0u);
    if (ag->have_tail_hidden) {
        mas_put_u32(tail, (uint32_t) ag->tail_hidden.size());
        const uint8_t * tp = (const uint8_t *) ag->tail_hidden.data();
        tail.insert(tail.end(), tp, tp + ag->tail_hidden.size() * sizeof(float));
    }

    // Size everything up front: the two-call form (dst=NULL) must report the
    // real requirement, tail included, or the caller allocates short and the
    // copy below overruns.
    const size_t total = 4 + body.size() + 8 + kv_need + tail.size();
    if (out_size) *out_size = total;
    if (!dst || cap < total) return -2;

    uint8_t * w = dst;
    std::memcpy(w, "MAS1", 4); w += 4;
    std::memcpy(w, body.data(), body.size()); w += body.size();
    uint64_t kv_len = (uint64_t) kv_need;
    std::memcpy(w, &kv_len, 8); w += 8;

    if (kv_need > 0) {
        const int rc = llama_state_seq_get_data(mas->ctx, w, kv_need, agent_idx);
        if (rc != (int) kv_need) {
            std::cerr << "[MAS/session] kv export failed rc=" << rc
                      << " want=" << kv_need << std::endl;
            return -3;
        }
        w += kv_need;
    }

    std::memcpy(w, tail.data(), tail.size());
    return 0;
}

int llama_mas_session_restore(llama_mas_context * mas, int32_t agent_idx,
                              const uint8_t * src, size_t size) {
    if (!mas || !mas->ctx || !src) return -1;
    if (agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return -1;

    const uint8_t * p = src;
    const uint8_t * end = src + size;

    uint32_t magic = 0;
    if (!mas_get_u32(p, end, magic) || magic != MAS_SESSION_MAGIC) {
        std::cerr << "[MAS/session] bad magic (got " << magic << ")" << std::endl;
        return -2;
    }

    uint32_t n_seg = 0;
    if (!mas_get_u32(p, end, n_seg)) return -2;

    std::vector<llama_mas_context::turn_segment_log> ledger;
    ledger.reserve(n_seg);
    for (uint32_t i = 0; i < n_seg; ++i) {
        llama_mas_context::turn_segment_log seg{};
        uint32_t kind = 0, nt = 0, nl = 0, tl = 0;
        uint64_t ps = 0, pe = 0;
        if (!mas_get_u32(p, end, kind) || !mas_get_u64(p, end, ps) || !mas_get_u64(p, end, pe) ||
            !mas_get_u32(p, end, nt) || !mas_get_u32(p, end, nl) || !mas_get_u32(p, end, tl)) {
            return -2;
        }
        if ((size_t) tl > (size_t) (end - p)) return -2;
        seg.kind = (int32_t) kind;
        seg.pos_start = (size_t) ps;
        seg.pos_end = (size_t) pe;
        seg.n_tokens = (int32_t) nt;
        seg.n_latent = (int32_t) nl;
        seg.text.assign((const char *) p, (const char *) p + tl);
        p += tl;
        seg.tokens.reserve(nt);
        for (uint32_t k = 0; k < nt; ++k) {
            uint32_t t = 0;
            if (!mas_get_u32(p, end, t)) return -2;
            seg.tokens.push_back((llama_token) t);
        }
        ledger.push_back(std::move(seg));
    }

    uint64_t kv_len = 0;
    if (!mas_get_u64(p, end, kv_len)) return -2;
    if ((size_t) kv_len > (size_t) (end - p)) {
        std::cerr << "[MAS/session] truncated kv (need " << kv_len
                  << ", have " << (size_t)(end - p) << ")" << std::endl;
        return -3;
    }

    // Restore the sequence BEFORE publishing the ledger: a failed import must
    // not leave the agent claiming positions that hold nothing.
    if (kv_len > 0) {
        const int rc = llama_state_seq_set_data(mas->ctx, p, (size_t) kv_len, agent_idx);
        if (rc != (int) kv_len) {
            std::cerr << "[MAS/session] kv import failed rc=" << rc
                      << " want=" << kv_len << std::endl;
            return -3;
        }
    }

    auto & agent = mas->agents[agent_idx];

    // Tail hidden state written by session_save after the KV bytes.
    agent->tail_hidden.clear();
    agent->have_tail_hidden = false;
    if (p < end) {
        uint32_t has_tail = 0, tail_floats = 0;
        if (mas_get_u32(p, end, has_tail) && has_tail) {
            if (!mas_get_u32(p, end, tail_floats)) {
                std::cerr << "[MAS/session] truncated tail header" << std::endl;
                return -3;
            }
            if ((size_t) tail_floats * sizeof(float) > (size_t) (end - p)) {
                std::cerr << "[MAS/session] truncated tail data (need "
                          << tail_floats << " floats)" << std::endl;
                return -3;
            }
            agent->tail_hidden.resize(tail_floats);
            std::memcpy(agent->tail_hidden.data(), p, tail_floats * sizeof(float));
            p += tail_floats * sizeof(float);
            agent->have_tail_hidden = true;
        }
    }

    size_t tail_pos = 0;
    for (const auto & seg : ledger) {
        if (seg.pos_end > tail_pos) tail_pos = seg.pos_end;
    }
    agent->prompt_pos = tail_pos;
    agent->generated_len = 0;
    agent->finished = false;

    mas->turn_log = std::move(ledger);
    std::cerr << "[MAS/session] restored agent=" << agent->role
              << " segments=" << mas->turn_log.size()
              << " pos=" << tail_pos
              << " kv=" << kv_len << "B"
              << " tail_hidden=" << (agent->have_tail_hidden ? agent->tail_hidden.size() : 0)
              << std::endl;
    return 0;
}

static std::vector<llama_token> tokenize_cached(llama_mas_context * mas,
                                                const llama_vocab * vocab,
                                                const std::string & text);


// ---------------------------------------------------------------------------
// Adapter weight helpers
// ---------------------------------------------------------------------------

llama_mas_adapter_weights * llama_mas_adapter_weights_init(
    int32_t n_embd,
    int32_t in_dim,
    int32_t out_dim,
    int32_t hidden_dim
) {
    auto * w = (llama_mas_adapter_weights *)calloc(1, sizeof(llama_mas_adapter_weights));
    if (!w) return nullptr;

    w->n_embd    = n_embd;
    w->in_dim    = in_dim  > 0 ? in_dim  : n_embd;
    w->out_dim   = out_dim > 0 ? out_dim : n_embd;
    w->hidden_dim = hidden_dim > 0 ? hidden_dim : out_dim * 2;

    // RecursiveLink weights (all n_embd sized)
    if (n_embd > 0) {
        w->w_pre_ln  = (float *)calloc(n_embd, sizeof(float));
        w->b_pre_ln  = (float *)calloc(n_embd, sizeof(float));
        w->w_proj1   = (float *)calloc(n_embd * n_embd, sizeof(float));
        w->b_proj1   = (float *)calloc(n_embd, sizeof(float));
        w->w_proj2   = (float *)calloc(n_embd * n_embd, sizeof(float));
        w->b_proj2   = (float *)calloc(n_embd, sizeof(float));
        w->w_post_ln = (float *)calloc(n_embd, sizeof(float));
        w->b_post_ln = (float *)calloc(n_embd, sizeof(float));
        // Default: identity (weights = 1.0 on diagonal for matrices)
        for (int i = 0; i < n_embd; ++i) {
            w->w_pre_ln[i]  = 1.0f;
            w->w_post_ln[i] = 1.0f;
            w->w_proj1[i * n_embd + i] = 1.0f;
            w->w_proj2[i * n_embd + i] = 1.0f;
        }
    }

    // CrossLink weights
    if (w->in_dim > 0 && w->out_dim > 0) {
        w->w_cross_ln_source = (float *)calloc(w->in_dim, sizeof(float));
        w->b_cross_ln_source = (float *)calloc(w->in_dim, sizeof(float));
        w->w_cross_proj1     = (float *)calloc(w->in_dim * w->hidden_dim, sizeof(float));
        w->b_cross_proj1     = (float *)calloc(w->hidden_dim, sizeof(float));
        w->w_cross_proj2     = (float *)calloc(w->hidden_dim * w->out_dim, sizeof(float));
        w->b_cross_proj2     = (float *)calloc(w->out_dim, sizeof(float));
        w->w_cross_residual  = (float *)calloc(w->in_dim * w->out_dim, sizeof(float));
        w->b_cross_residual  = (float *)calloc(w->out_dim, sizeof(float));
        w->w_cross_ln_target = (float *)calloc(w->out_dim, sizeof(float));
        w->b_cross_ln_target = (float *)calloc(w->out_dim, sizeof(float));
        // Default: identity mapping where possible
        for (int i = 0; i < w->in_dim && i < w->out_dim; ++i) {
            w->w_cross_residual[i * w->out_dim + i] = 1.0f;
        }
        for (int i = 0; i < w->in_dim; ++i)  w->w_cross_ln_source[i] = 1.0f;
        for (int i = 0; i < w->out_dim; ++i) w->w_cross_ln_target[i] = 1.0f;
    }

    return w;
}

#ifdef GGML_USE_CUDA
struct llama_mas_adapter_weights_gpu {
    float * w_pre_ln = nullptr;  float * b_pre_ln = nullptr;
    float * w_proj1  = nullptr;  float * b_proj1  = nullptr;
    float * w_proj2  = nullptr;  float * b_proj2  = nullptr;
    float * w_post_ln = nullptr; float * b_post_ln = nullptr;
    float * w_cross_ln_source = nullptr; float * b_cross_ln_source = nullptr;
    float * w_cross_proj1 = nullptr;     float * b_cross_proj1 = nullptr;
    float * w_cross_proj2 = nullptr;     float * b_cross_proj2 = nullptr;
    float * w_cross_residual = nullptr;  float * b_cross_residual = nullptr;
    float * w_cross_ln_target = nullptr; float * b_cross_ln_target = nullptr;
    int32_t n_embd = 0, in_dim = 0, out_dim = 0, hidden_dim = 0;
};
static void llama_mas_adapter_weights_free_gpu(llama_mas_adapter_weights_gpu * gpu) {
    if (!gpu) return;
    // These buffers come from ggml_cuda_mas_device_malloc(), so they must go back
    // through the same allocator. cudart is not linked into this target.
    ggml_cuda_mas_device_free(gpu->w_pre_ln);  ggml_cuda_mas_device_free(gpu->b_pre_ln);
    ggml_cuda_mas_device_free(gpu->w_proj1);   ggml_cuda_mas_device_free(gpu->b_proj1);
    ggml_cuda_mas_device_free(gpu->w_proj2);   ggml_cuda_mas_device_free(gpu->b_proj2);
    ggml_cuda_mas_device_free(gpu->w_post_ln); ggml_cuda_mas_device_free(gpu->b_post_ln);
    ggml_cuda_mas_device_free(gpu->w_cross_ln_source); ggml_cuda_mas_device_free(gpu->b_cross_ln_source);
    ggml_cuda_mas_device_free(gpu->w_cross_proj1);     ggml_cuda_mas_device_free(gpu->b_cross_proj1);
    ggml_cuda_mas_device_free(gpu->w_cross_proj2);     ggml_cuda_mas_device_free(gpu->b_cross_proj2);
    ggml_cuda_mas_device_free(gpu->w_cross_residual);  ggml_cuda_mas_device_free(gpu->b_cross_residual);
    ggml_cuda_mas_device_free(gpu->w_cross_ln_target); ggml_cuda_mas_device_free(gpu->b_cross_ln_target);
}
#endif

void llama_mas_adapter_weights_free(llama_mas_adapter_weights * weights) {
    if (!weights) return;
#ifdef GGML_USE_CUDA
    if (weights->gpu_data) {
        llama_mas_adapter_weights_free_gpu((llama_mas_adapter_weights_gpu *)weights->gpu_data);
        delete (llama_mas_adapter_weights_gpu *)weights->gpu_data;
        weights->gpu_data = nullptr;
    }
#endif
    free(weights->w_pre_ln);
    free(weights->b_pre_ln);
    free(weights->w_proj1);
    free(weights->b_proj1);
    free(weights->w_proj2);
    free(weights->b_proj2);
    free(weights->w_post_ln);
    free(weights->b_post_ln);
    free(weights->w_cross_ln_source);
    free(weights->b_cross_ln_source);
    free(weights->w_cross_proj1);
    free(weights->b_cross_proj1);
    free(weights->w_cross_proj2);
    free(weights->b_cross_proj2);
    free(weights->w_cross_residual);
    free(weights->b_cross_residual);
    free(weights->w_cross_ln_target);
    free(weights->b_cross_ln_target);
    free(weights);
}

// ---------------------------------------------------------------------------
// CPU adapter forward implementations
// ---------------------------------------------------------------------------

static inline float rms_norm(const float * x, float * out, int n, float eps) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += x[i] * x[i];
    float scale = 1.0f / std::sqrt(sum / n + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale;
    return scale;
}

static inline void layer_norm(const float * x, const float * w, const float * b,
                               float * out, int n, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; ++i) var += (x[i] - mean) * (x[i] - mean);
    var /= n;
    float scale = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < n; ++i) {
        out[i] = (x[i] - mean) * scale * w[i] + (b ? b[i] : 0.0f);
    }
}

static inline void mat_vec_mul(const float * mat, const float * vec, float * out,
                                int rows, int cols) {
    // mat: [rows x cols], vec: [cols], out: [rows]
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum += mat[r * cols + c] * vec[c];
        }
        out[r] = sum;
    }
}

static inline void vec_add(const float * a, const float * b, float * out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

// Small crc32 (IEEE) over raw bytes. Used to fingerprint a segment's token ids
// so the session ledger can be validated without storing the whole prefix.
static void mas_crc32_add(uint32_t & crc, const void * data, size_t len) {
    static uint32_t table[256];
    static bool table_ready = false;
    if (!table_ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        table_ready = true;
    }
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    }
}

static uint32_t mas_crc32(const void * data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    mas_crc32_add(crc, data, len);
    return crc ^ 0xFFFFFFFFu;
}

// Enable/disable hidden state capture on the context. Enabling it makes
// llama_context copy EVERY layer x EVERY output position to host memory on each
// decode, so the turn driver turns it on only around latent steps.
void llama_mas_set_capture_hidden_states(llama_mas_context * mas, bool enable) {
    if (!mas || !mas->ctx) return;
    llama_set_output_hidden_states(mas->ctx, enable);
}

// Current tail of an agent's sequence: prompt tokens already prefilled plus the
// positions written by generated tokens and latent steps.
size_t llama_mas_agent_pos(const llama_mas_context * mas, int32_t agent_idx) {
    if (!mas || agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return 0;
    const auto & a = mas->agents[agent_idx];
    // The KV is authoritative; the counters are a cache of it and drift after a
    // restore or a latent run. Asking the memory module first is what makes the
    // next decode start exactly one past the real tail.
    if (mas->ctx) {
        llama_memory_t mem = llama_get_memory(mas->ctx);
        if (mem) {
            const llama_pos pmax = llama_memory_seq_pos_max(mem, a->seq_id);
            if (pmax >= 0) return (size_t) pmax + 1;
        }
    }
    return a->prompt_pos + a->generated_len;
}

// Truncate an agent's sequence to `keep` positions, dropping everything after.
// This is the per-session reuse primitive: a new turn with a different tail
// truncates, then re-decodes only the divergent suffix into the SAME sequence,
// so the KV cache is reused instead of rebuilt.
//
// generated_tokens is ALWAYS "prompt tokens then generated tokens", so entry i
// is exactly the token that wrote position i - that makes truncation a plain
// resize plus a KV seq_rm, with no bookkeeping about which span is which.
int llama_mas_agent_truncate(llama_mas_context * mas, int32_t agent_idx, size_t keep) {
    if (!mas || !mas->ctx) return -1;
    if (agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return -1;
    auto & a = mas->agents[agent_idx];

    // Trust the KV, not the counters. prompt_pos/generated_len are bookkeeping
    // that latent steps and restores both move; the memory module is the only
    // authority on which positions exist. Deriving the tail from it is what
    // keeps the next decode's start position exactly one past the tail - a
    // mismatch there is a hard llama_decode failure ("sequence positions must
    // remain consecutive"), not a soft degradation.
    llama_memory_t mem = llama_get_memory(mas->ctx);
    if (!mem) return -1;

    llama_pos pmin = llama_memory_seq_pos_min(mem, a->seq_id);
    llama_pos pmax = llama_memory_seq_pos_max(mem, a->seq_id);

    if (pmax < 0) {
        a->prompt_pos    = 0;
        a->generated_len = 0;
        a->finished      = false;
        return 0;
    }
    if (pmin < 0) pmin = 0;

    const size_t held = (size_t)(pmax - pmin + 1);
    const size_t keep_n = std::min(keep, held);

    if (keep_n == 0) {
        llama_memory_seq_rm(mem, a->seq_id, (llama_pos) pmin, -1);
        a->prompt_pos    = (size_t) pmin;
        a->generated_len = 0;
        a->finished      = false;
        return 0;
    }

    const llama_pos new_tail = (llama_pos)(pmin + (llama_pos) keep_n - 1);
    llama_memory_seq_rm(mem, a->seq_id, new_tail + 1, -1);

    a->prompt_pos    = (size_t) new_tail + 1;
    a->generated_len = 0;
    a->finished      = false;

    const size_t keep_total = (size_t) pmin + keep_n;
    if (a->generated_tokens.size() > keep_total) {
        a->generated_tokens.resize(keep_total);
    }

    // The ledger describes positions; entries past the cut no longer exist.
    while (!mas->turn_log.empty() && mas->turn_log.back().pos_end > a->prompt_pos) {
        mas->turn_log.pop_back();
    }

    std::cerr << "[MAS/truncate] agent=" << a->role
              << " held=[" << pmin << "," << pmax << "] keep=" << keep
              << " -> prompt_pos=" << a->prompt_pos
              << " ledger=" << mas->turn_log.size() << std::endl;
    return 0;
}

void llama_mas_free_turn_result(llama_mas_turn_result * r) {
    if (!r) return;
    if (r->text)   { free(r->text);   r->text = nullptr; }
    if (r->tokens) { free(r->tokens); r->tokens = nullptr; }
    r->n_tokens_sampled = 0;
}

int llama_mas_run_turn(
    llama_mas_context * mas,
    int32_t agent_idx,
    const llama_mas_segment * segments,
    int32_t n_segments,
    llama_mas_turn_result * out_result
) {
    if (!mas || !mas->ctx || !mas->model) return -1;
    if (agent_idx < 0 || agent_idx >= (int32_t)mas->agents.size()) return -1;
    if (!segments || n_segments <= 0) return -1;
    if (!out_result) return -1;

    auto & agent = mas->agents[agent_idx];
    const llama_vocab * vocab = llama_model_get_vocab(mas->model);
    // Deepstack models (n_embd_inp > n_embd) carry a WIDER per-token input
    // embedding than the residual stream. batch.embd is consumed at layer 0, so
    // every latent buffer here must be sized at n_embd_inp, not n_embd - sizing
    // at n_embd makes the graph assert on the first latent step. The alignment
    // matrix and the hidden-state tap stay at n_embd (they are residual-space).
    const int32_t n_embd_inp = llama_model_n_embd_inp(mas->model);
    const int32_t n_embd = (n_embd_inp > 0) ? n_embd_inp : llama_model_n_embd(mas->model);
    const int32_t n_embd_resid = llama_model_n_embd(mas->model);
    if (n_embd_inp > n_embd_resid) {
        static bool s_deepstack_logged = false;
        if (!s_deepstack_logged) {
            std::cerr << "[MAS/turn] deepstack model: n_embd_inp=" << n_embd_inp
                      << " > n_embd=" << n_embd_resid
                      << " -> latent buffers sized at n_embd_inp" << std::endl;
            s_deepstack_logged = true;
        }
    }
    const int32_t max_tokens_cap = agent->max_tokens > 0 ? agent->max_tokens : 4096;

    std::memset(out_result, 0, sizeof(*out_result));
    // Append to the ledger, do NOT clear it. A resumed session restores the KV
    // AND the ledger (llama_mas_session_restore); clearing here would leave the
    // KV positions intact but forget what produced them, so the next turn could
    // not tell which positions were reusable. The first turn of a brand-new
    // session has an empty ledger anyway, so this is safe either way.

    llama_context_set_layer_graph(mas->ctx, agent->layer_graph);

    // ── latent_interleave (E1) ─────────────────────────────────────────────
    // Expand the caller's segment list into an alternating text/latent
    // sequence. This is done ONCE, up front, on a local copy: the decode path
    // and every existing toggle are untouched, and the legacy behaviour is the
    // exact same pointer when the flag is off.
    //
    // Rule: for each `latent` segment with latent_interleave = K > 0, every
    // following run of text-sampling segments is split so that at most K text
    // tokens are emitted before another latent span of the same length is
    // inserted. Expansion stops at the end of the caller's list, so the turn
    // still terminates exactly when the caller said it should; the toggle only
    // changes what happens IN BETWEEN.
    //
    // Text segments are identified by (kind == TOKENS && sample == 1), i.e.
    // segments that actually emit a token. Prompt-carrying segments
    // (sample == 0) are passed through untouched: they are context, not output.
    std::vector<llama_mas_segment> expanded;
    const llama_mas_segment * segs_ptr = segments;
    int32_t n_segs_run = n_segments;
    // Function-scope so its backing store outlives the `any_toolcall` block
    // below: segs_ptr points into it after the block closes, so it MUST NOT be
    // destroyed at the block's end (that was an access-violation crash).
    std::vector<llama_mas_segment> with_toolcall;
    {
        bool any_interleave = false;
        for (int32_t i = 0; i < n_segments && !any_interleave; ++i) {
            if (segments[i].kind == LLAMA_MAS_SEGMENT_LATENT &&
                segments[i].latent_interleave > 0) {
                any_interleave = true;
            }
        }
        if (any_interleave) {
            expanded.reserve((size_t) n_segments * 2 + 8);
            int32_t log_k = 0;
            for (int32_t i = 0; i < n_segments; ++i) {
                const llama_mas_segment & s = segments[i];
                expanded.push_back(s);
                if (s.kind != LLAMA_MAS_SEGMENT_LATENT || s.latent_interleave <= 0) {
                    continue;
                }
                const int32_t K = s.latent_interleave;
                log_k = K;
                // Consume the following text-sampling segments in runs of K,
                // inserting another think block after each run.
                //
                // This MUST be a loop, not an `if`: a single pass only splits
                // once, so 12 text segments with K=3 produced ONE extra latent
                // block (2 total) instead of four. The first implementation
                // appended the repeat think block and then let the outer loop
                // walk past it, so only the first run was ever split.
                while (i + 1 < n_segments &&
                       segments[i + 1].kind == LLAMA_MAS_SEGMENT_TOKENS &&
                       segments[i + 1].sample == 1) {
                    int32_t text_run = 0;
                    while (text_run < K && i + 1 < n_segments &&
                           segments[i + 1].kind == LLAMA_MAS_SEGMENT_TOKENS &&
                           segments[i + 1].sample == 1) {
                        expanded.push_back(segments[++i]);
                        text_run++;
                    }
                    // More text remains -> another think block before it. No
                    // trailing think block is appended when the text is
                    // exhausted, which is what keeps the legacy segment count
                    // intact for lists that end with the text run.
                    if (i + 1 < n_segments &&
                        segments[i + 1].kind == LLAMA_MAS_SEGMENT_TOKENS &&
                        segments[i + 1].sample == 1) {
                        expanded.push_back(s);
                    }
                }
            }
            segs_ptr = expanded.data();
            n_segs_run = (int32_t) expanded.size();
            // Only report a change when there was one. Printing
            // "latent_interleave=0 -> N expanded to N+1" on a legacy run is
            // actively misleading: it says a toggle fired when none did.
            if (n_segs_run != n_segments) {
                std::cerr << "[MAS/turn] latent_interleave=" << log_k
                          << " -> " << n_segments << " segments expanded to "
                          << n_segs_run << std::endl;
            }
        }
    }

    // ── E: `latent_on_tool_call` — inject a latent think span before a
    // tool-call segment ────────────────────────────────────────────────────
    // A TOKENS segment with latent_on_tool_call != 0 is about to emit a
    // structured action (a tool call); the caller marks it because that is
    // where a wrong-but-confident commitment is most expensive. This pass
    // expands the segment list so that a LATENT span of latent_steps (taken
    // from the segment's own latent_* fields) runs immediately BEFORE it.
    //
    // Default OFF: with no flag set anywhere the loop copies the list verbatim
    // (and is skipped entirely via any_toolcall), so behaviour is byte-identical.
    // The injected segment reuses the marking segment's latent configuration but
    // never carries text or sampling - it is pure continuous reasoning, and the
    // real tool-call segment that follows it is left completely intact.
    {
        bool any_toolcall = false;
        for (int32_t i = 0; i < n_segs_run && !any_toolcall; ++i) {
            if (segs_ptr[i].kind == LLAMA_MAS_SEGMENT_TOKENS &&
                segs_ptr[i].latent_on_tool_call != 0) {
                any_toolcall = true;
            }
        }
        if (any_toolcall) {
            with_toolcall.reserve((size_t) n_segs_run * 2 + 4);
            int32_t injected = 0;
            for (int32_t i = 0; i < n_segs_run; ++i) {
                const llama_mas_segment & s = segs_ptr[i];
                if (s.kind == LLAMA_MAS_SEGMENT_TOKENS && s.latent_on_tool_call != 0) {
                    // Build the injected think block from THIS segment's latent
                    // config, but strip everything that would make it emit text.
                    llama_mas_segment think = s;
                    think.kind            = LLAMA_MAS_SEGMENT_LATENT;
                    think.text            = nullptr;
                    think.forced_tokens   = nullptr;
                    think.n_forced_tokens = 0;
                    think.sample          = 0;
                    think.latent_only_agents    = 0;  // run for every agent
                    think.latent_interleave     = 0;  // no further expansion
                    think.latent_on_tool_call   = 0;  // do not recurse
                    think.escape_text     = nullptr;  // bounded by latent_steps
                    // latent_steps stays; align / transform / position_mode /
                    // stop_on_* are inherited from s by the copy.
                    with_toolcall.push_back(think);
                    injected++;
                }
                with_toolcall.push_back(s);
            }
            segs_ptr = with_toolcall.data();
            n_segs_run = (int32_t) with_toolcall.size();
            std::cerr << "[MAS/turn] latent_on_tool_call -> injected "
                      << injected << " latent think span(s) before tool-call "
                      << "segment(s); " << n_segments << " -> " << n_segs_run
                      << " segments" << std::endl;
        }
    }

    // Every segment starts exactly one past the KV tail. Reading the memory
    // module (not prompt_pos/generated_len) is required: after a restore those
    // counters can lag the restored sequence by several positions, and a gap
    // there is a hard llama_decode failure, not a soft degradation.
    auto kv_pos = [&]() -> size_t {
        size_t p = llama_mas_agent_pos(mas, agent_idx);
        agent->prompt_pos = p;
        agent->generated_len = 0;
        return p;
    };
    kv_pos();

    std::vector<llama_token> sampled;   // tokens emitted by this turn
    double t_latent_start = 0.0, t_text_start = 0.0;

    // Hidden state of the last decoded position, carried into the next latent
    // segment as its step-0 input. Lives on the agent (see tail_hidden) so it
    // survives a session restore; a local here would be empty after restore and
    // silently fall back to the seed-token path.
    std::vector<float> pending_h;
    bool have_pending_h = agent->have_tail_hidden;
    if (have_pending_h) pending_h = agent->tail_hidden;
    double t_latent_ms = 0.0, t_text_ms = 0.0;
    int64_t n_latent = 0, n_text = 0;

    // Set when a TEXT segment with stop_on_eog samples an end-of-generation
    // token. The loop below did not previously honour `agent->finished`, and it
    // also resets `generated_len = 0` at every segment, so a caller passing N
    // one-token sample segments got all N tokens even after EOG - measured as
    // 'The answer is 72.<|im_end|>' repeated to the full 512-token budget.
    bool stop_turn_eog = false;
    for (int32_t si = 0; si < n_segs_run; ++si) {
        if (stop_turn_eog) break;
        const llama_mas_segment & seg_in = segs_ptr[si];
        // Local mutable copy: `latent_only_agents` may rewrite the segment kind
        // for agents outside its mask (see below). Copying is cheap - the struct
        // is a few dozen bytes plus a pointer to the caller's text.
        llama_mas_segment seg = seg_in;
        llama_mas_context::turn_segment_log log{};
        log.kind = (int32_t) seg.kind;
        log.pos_start = llama_mas_agent_pos(mas, agent_idx);
        agent->prompt_pos = log.pos_start;
        agent->generated_len = 0;
        log.text = seg.text ? seg.text : "";

        // ── E: `latent_only_agents` ─────────────────────────────────────────
        // Bit i of the mask means "agent i is a latent thinker". Without this,
        // a turn's latent segment applies to EVERY agent, so a mixed crew cannot
        // have a worker think latently while a spokesperson answers in text.
        // An agent outside the mask has the segment rewritten to a plain
        // sampling segment, which is the text path - so the rest of this loop
        // needs no knowledge of the flag.
        if (seg.kind == LLAMA_MAS_SEGMENT_LATENT && seg.latent_only_agents != 0) {
            const bool in_mask = (agent_idx >= 0 && agent_idx < 31) &&
                                 ((seg.latent_only_agents >> agent_idx) & 1);
            if (!in_mask) {
                static bool s_mask_logged = false;
                if (!s_mask_logged) {
                    std::cerr << "[MAS/turn] latent_only_agents mask="
                              << seg.latent_only_agents << ": agent " << agent_idx
                              << " excluded -> text path" << std::endl;
                    s_mask_logged = true;
                }
                seg.kind = LLAMA_MAS_SEGMENT_TOKENS;
                seg.sample = 1;
                seg.n_forced_tokens = 0;
                seg.forced_tokens = nullptr;
                seg.text = nullptr;
                seg.latent_steps = 0;
            }
        }

        if (seg.kind == LLAMA_MAS_SEGMENT_TOKENS) {
            // ---- text segment ----
            std::vector<llama_token> toks;
            if (seg.n_forced_tokens > 0 && seg.forced_tokens) {
                for (int32_t i = 0; i < seg.n_forced_tokens; ++i) {
                    if (seg.forced_tokens[i] >= 0) toks.push_back(seg.forced_tokens[i]);
                }
            }
            if (seg.text && seg.text[0]) {
                std::vector<llama_token> txt = tokenize_cached(mas, vocab, seg.text);
                toks.insert(toks.end(), txt.begin(), txt.end());
            }
            if (toks.empty() && !seg.sample) {
                // Nothing to write and nothing asked for: a pure no-op segment.
                log.pos_end = log.pos_start;
                log.n_tokens = 0;
                log.tok_crc = 0;
                mas->turn_log.push_back(log);
                continue;
            }
            if ((int32_t)toks.size() > llama_n_batch(mas->ctx)) {
                std::cerr << "[MAS/turn] segment " << si << " has " << toks.size()
                          << " tokens > n_batch " << llama_n_batch(mas->ctx) << std::endl;
                llama_mas_free_turn_result(out_result);
                return -2;
            }

            // Capture stays ON, and it is cheap here: llama_context only keeps
            // output positions, and this batch sets logits on the LAST position
            // alone, so the host copy is n_layers x 1 x n_embd. Turning it off
            // is what forced the old latent loop to re-decode its seed token.
            llama_set_output_hidden_states(mas->ctx, true);

            // A sampling-only segment has no tokens to push, so there is no
            // prefill to run: the sample block below decodes the current tail
            // itself. (llama_decode rejects a zero-length batch outright.)
            const bool do_prefill = !toks.empty();
            if (!do_prefill) {
                agent->prompt_pos = llama_mas_agent_pos(mas, agent_idx);
                agent->generated_len = 0;
            }

            t_text_start = mas_time_ms();
            llama_memory_t mem = llama_get_memory(mas->ctx);
            llama_memory_seq_rm(mem, agent->seq_id, (llama_pos)(agent->prompt_pos + agent->generated_len), -1);

            const size_t seg_base = llama_mas_agent_pos(mas, agent_idx);
            agent->prompt_pos = seg_base;
            agent->generated_len = 0;

            if (do_prefill) {
            llama_batch batch = llama_batch_init((int32_t)toks.size(), 0, 1);
            for (size_t p = 0; p < toks.size(); ++p) {
                batch.token[p]   = toks[p];
                batch.pos[p]     = (llama_pos)(seg_base + p);
                batch.n_seq_id[p] = 1;
                batch.seq_id[p][0] = agent->seq_id;
                // Only the last position needs output (logits); the rest are prefill.
                batch.logits[p]  = (p + 1 == toks.size()) ? 1 : 0;
            }
            batch.n_tokens = (int32_t) toks.size();
            int ret = llama_decode(mas->ctx, batch);
            llama_batch_free(batch);
            if (ret != 0) {
                std::cerr << "[MAS/turn] text segment decode failed ret=" << ret << std::endl;
                llama_mas_free_turn_result(out_result);
                return ret;
            }
            agent->prompt_pos += toks.size();
            agent->generated_tokens.insert(agent->generated_tokens.end(), toks.begin(), toks.end() - 1);
            }

            // Carry the tail hidden state forward so a following latent segment
            // can start from it instead of re-decoding the last token.
            {
                int32_t got_embd = 0;
                float * h = llama_mas_get_agent_hidden_state(mas, agent_idx, &got_embd);
                if (h && got_embd == n_embd) {
                    pending_h.assign(h, h + n_embd);
                    have_pending_h = true;
                    agent->tail_hidden = pending_h;
                    agent->have_tail_hidden = true;
                } else {
                    have_pending_h = false;
                    agent->have_tail_hidden = false;
                }
            }
            n_text += (int64_t) toks.size();

            if (seg.sample) {
                // Capture the logits BEFORE sampling: this is the same accessor the
                // sampler chain itself falls through to, and it is only valid while
                // the previous decode's outputs are still current. Asking after
                // llama_sampler_accept reads a cache that is empty on the CPU path
                // (sampling.logits has no data), which silently yields no perplexity.
                const float * lp = llama_get_logits_ith(mas->ctx, -1);
                llama_token next = llama_sampler_sample(mas->samplers[agent_idx], mas->ctx, -1);
                llama_sampler_accept(mas->samplers[agent_idx], next);
                // Perplexity of this agent's emitted text: score the token that was
                // actually emitted against the model's own log-softmax. Latent
                // segments never reach here, so they are excluded by construction.
                if (lp) {
                    agent->logprob_sum +=
                        llama_mas_logprob_of(lp, (int64_t) llama_vocab_n_tokens(vocab), next);
                    agent->logprob_n++;
                }
                agent->generated_tokens.push_back(next);
                agent->generated_len++;
                sampled.push_back(next);
                n_text++;

                // Decode the token we just sampled into the KV, so the NEXT
                // segment starts from the new position. Without this the sampled
                // token is never fed back: every following sample sees the same
                // KV state, produces the same logits, and the model emits the
                // same token over and over ("Let The To The Let ...").
                {
                    llama_batch sb = llama_batch_init(1, 0, 1);
                    sb.token[0]    = next;
                    sb.pos[0]      = llama_mas_agent_pos(mas, agent_idx);
                    sb.n_seq_id[0] = 1;
                    sb.seq_id[0][0] = agent->seq_id;
                    sb.logits[0]   = 1;   // needed so the next sample has logits
                    sb.n_tokens    = 1;
                    int rc = llama_decode(mas->ctx, sb);
                    llama_batch_free(sb);
                    if (rc != 0) {
                        std::cerr << "[MAS/turn] sample-feedback decode failed ret=" << rc << std::endl;
                        llama_mas_free_turn_result(out_result);
                        return rc;
                    }
                    agent->prompt_pos = llama_mas_agent_pos(mas, agent_idx);
                }

                if (mas->stream_cb) {
                    mas->stream_cb(agent_idx, next, mas->stream_step++, mas->stream_user_data);
                }
                if (llama_vocab_is_eog(vocab, next) || (int32_t)agent->generated_len >= max_tokens_cap) {
                    agent->finished = true;
                    // Opt-in for TEXT segments: stop the whole turn at EOG.
                    // Default 0 keeps the legacy behaviour (caller-specified
                    // segment count is honoured exactly), so existing tests and
                    // benchmarks are unchanged.
                    if (seg.stop_on_eog && llama_vocab_is_eog(vocab, next)) {
                        stop_turn_eog = true;
                    }
                }
            }

            // Timing covers the WHOLE text segment: prefill AND the sampled
            // token's feedback decode. This used to be accumulated right after
            // the prefill, so text_ms reported prompt prefill only - measured
            // 46 ms of a 346 ms text turn on Qwen3-1.7B Q8, which made
            // text_tokens_per_sec ~7x optimistic and the F-section wall/text
            // split unusable.
            t_text_ms += mas_time_ms() - t_text_start;

            log.n_tokens  = (int32_t) toks.size();
            log.pos_end   = llama_mas_agent_pos(mas, agent_idx);
            log.tokens    = toks;
            log.tok_crc   = mas_crc32(toks.data(), toks.size() * sizeof(llama_token));
            mas->turn_log.push_back(log);
            out_result->n_tokens += (int32_t) (seg.sample ? 1 : 0);

        } else {
            // ---- latent segment ----
            if (seg.latent_steps <= 0) {
                log.pos_end = log.pos_start;
                mas->turn_log.push_back(log);
                continue;
            }

            // Seed: forced token, else the agent's last token, else token 0.
            // (The old path silently used id 0 when generated_tokens was empty,
            // which is an arbitrary, model-dependent "BOS-ish" seed. Callers can
            // now pass seed_token explicitly.)
            llama_token seed = seg.seed_token;
            if (seed < 0) {
                seed = agent->generated_tokens.empty() ? 0 : agent->generated_tokens.back();
            }

            // A latent segment appends; if the counters are behind the KV, drop
            // the stale tail first so the loop starts at a real boundary.
            {
                llama_memory_t mem = llama_get_memory(mas->ctx);
                const llama_pos pmax = llama_memory_seq_pos_max(mem, agent->seq_id);
                const size_t base = log.pos_start;
                if (pmax >= 0 && (size_t)(pmax + 1) > base) {
                    llama_memory_seq_rm(mem, agent->seq_id, (llama_pos) base, -1);
                    std::cerr << "[MAS/turn] latent segment " << si
                              << " dropped stale kv tail " << (pmax + 1) << " -> " << base << std::endl;
                }
                agent->prompt_pos = base;
                agent->generated_len = 0;
            }

            llama_set_output_hidden_states(mas->ctx, true);
            t_latent_start = mas_time_ms();

            std::vector<float> h(n_embd);
            // ── `use_gpu_hidden_states` (G) ─────────────────────────────────
            // Persistent staging buffer for the layer-0 tap. The tap already
            // copies device->host every step (ggml_backend_tensor_get at the
            // synchronous point in llama-context.cpp); what this controls is
            // whether that copy lands in a REUSED allocation rather than a
            // vector that is resized each step. On CUDA the resize is cheap
            // (the allocator caches), so this is a modest win - but it is the
            // honest extent of what can be done without a ggml-side device
            // buffer. Keeping the vector resident on the GPU across steps would
            // avoid the D2H entirely, and that needs a device allocation owned
            // by the context, not a flag read in this loop.
            std::vector<float> hs_staging;
            if (mas->use_gpu_hidden_states) {
                hs_staging.resize((size_t) n_embd);
            }
            auto stage_hidden = [&](const float * src) -> const float * {
                if (src && !hs_staging.empty()) {
                    std::memcpy(hs_staging.data(), src, (size_t) n_embd * sizeof(float));
                    return hs_staging.data();
                }
                return src;
            };
            // Previous step's feedback vector, for prior_layer/delta/ema.
            std::vector<float> prev_fb;
            bool have_prev_fb = false;
            // Previous step's vector for repeat/fixed-point detection.
            std::vector<float> prev_cos;
            bool have_prev_cos = false;
            // Previous step's L2 norm, for the norm-drift stop (D2).
            double prev_norm = 0.0;
            bool   have_prev_norm = false;
            // Feedback-vector history for mean_pool across steps (transform 4).
            std::vector<std::vector<float>> hist_fb;
            int32_t steps_done = 0;
            bool use_pending = have_pending_h && pending_h.size() == (size_t) n_embd;
            if (use_pending) h = pending_h;
            have_pending_h = false;

            // ── latent_only (upstream Gen-Verse/LatentMAS) ──────────────────
            // Upstream drops the prompt from the KV before the latent loop, so
            // every step attends to the latent span alone instead of re-reading
            // a growing prompt. Without this, step N costs O(prompt + N)
            // attention and the model's context drifts away from the question
            // as thoughts accumulate - one of the three measured reasons our
            // latent depth capped out far below the paper's 80.
            //
            // Implemented by deleting KV positions strictly BEFORE the latent
            // span starts. Position numbering is NOT rebased by default:
            // llama_decode requires strictly consecutive positions, so we keep
            // the original indices and let the gap remain. The attention mask
            // covers real positions only, so the hole is simply not attended
            // to.
            //
            // `latent_kv_rebase` (default OFF) removes the hole: after the
            // delete, llama_memory_seq_add() shifts the surviving cells down to
            // position 0. RoPE is relative, and llama_kv_cache::init_update()
            // re-applies a K-shift for every cell with has_shift set, so the
            // keys stay consistent with their new positions - this is the same
            // machinery a context shift uses, not a position renumbering hack.
            // Requires can_shift (true for Qwen3-class models; seq_add asserts
            // n_pos_per_embd()==1).
            size_t latent_only_from = (size_t) -1;
            if (seg.latent_only && !seg.latent_kv_keep_prompt) {
                llama_memory_t mem_lo = llama_get_memory(mas->ctx);
                if (mem_lo) {
                    const llama_pos pmin = llama_memory_seq_pos_min(mem_lo, agent->seq_id);
                    if (pmin >= 0) {
                        const llama_pos cut = (llama_pos) llama_mas_agent_pos(mas, agent_idx) - 1;
                        if (cut >= pmin) {
                            llama_memory_seq_rm(mem_lo, agent->seq_id, pmin, cut);
                            latent_only_from = (size_t) (cut + 1);
                            std::cerr << "[MAS/latent] latent_only dropped KV ["
                                      << pmin << ".." << cut << "] before loop" << std::endl;
                            if (seg.latent_kv_rebase) {
                                const llama_pos new_min =
                                    llama_memory_seq_pos_min(mem_lo, agent->seq_id);
                                if (new_min > 0) {
                                    // Rebase the SURVIVING cells down to 0.
                                    //
                                    // WHY NOT llama_memory_seq_add(): it only
                                    // MARKS the cells (pos_add records a
                                    // per-cell shift) and defers the real work
                                    // to the K-shift graph in
                                    // llama_kv_cache::init_update(). That graph
                                    // crashed this server with an access
                                    // violation (0xC0000005) at
                                    // 'update: applying K-shift' on the first
                                    // rebase - measured. It is the
                                    // context-shift path, which this fork never
                                    // exercises, and forcing a scheduler
                                    // re-reservation first did NOT fix it.
                                    //
                                    // So do the renumbering EXPLICITLY, with
                                    // no K-shift involvement at all:
                                    //   1. export the surviving cells. This
                                    //      serialises cell COUNT and the K/V
                                    //      bytes; it does NOT serialise
                                    //      positions (llama_kv_cache::
                                    //      state_write_meta writes shapes, not
                                    //      pos), so the export is
                                    //      position-agnostic by construction.
                                    //   2. clear the sequence. state_read()
                                    //      rebuilds the cells from scratch,
                                    //      which resets every shift to 0.
                                    //   3. re-import the SAME bytes. The keys
                                    //      already carry their RoPE rotation,
                                    //      which is a function of the OLD
                                    //      absolute position; they are now
                                    //      placed at the NEW positions, so
                                    //      RoPE is self-consistent with where
                                    //      they actually sit. Bit-identical
                                    //      keys, correct new positions.
                                    // The ledger is moved with the data (below)
                                    // so llama_mas_agent_truncate's
                                    // "entry i == position i" invariant holds.
                                    const size_t seq_bytes =
                                        llama_state_seq_get_size(mas->ctx, agent->seq_id);
                                    std::vector<uint8_t> blob(seq_bytes);
                                    const size_t got = seq_bytes > 0
                                        ? llama_state_seq_get_data(mas->ctx, blob.data(),
                                                                   seq_bytes, agent->seq_id)
                                        : 0;
                                    // llama_memory_clear() does NOT reset the
                                    // per-cell shift flags set by a previous
                                    // seq_add() (llama_kv_cells::clear() only
                                    // empties the cells). If a stale flag
                                    // survived, the next decode's init_update()
                                    // would see has_shift and run the K-shift
                                    // graph - the exact path that crashes. The
                                    // re-import below overwrites the cells, so
                                    // clear the whole sequence first: that both
                                    // empties the cache and drops the flags.
                                    llama_memory_seq_rm(mem_lo, agent->seq_id, -1, -1);
                                    if (seq_bytes > 0 && got == seq_bytes) {
                                        const size_t back = llama_state_seq_set_data(
                                            mas->ctx, blob.data(), blob.size(), agent->seq_id);
                                        if (back == blob.size()) {
                                            std::cerr << "[MAS/latent] latent_only REBASED KV "
                                                      << "positions " << new_min << ".. -> 0.."
                                                      << " (" << seq_bytes << " bytes re-imported)"
                                                      << std::endl;
                                        } else {
                                            std::cerr << "[MAS/latent] latent_only REBASE FAILED"
                                                      << " (re-import rc=" << back
                                                      << " want=" << blob.size()
                                                      << ") - KV is now EMPTY for this agent"
                                                      << std::endl;
                                        }
                                    } else {
                                        std::cerr << "[MAS/latent] latent_only REBASE SKIPPED"
                                                  << " (state export failed: got=" << got
                                                  << " want=" << seq_bytes
                                                  << ") - KV is now EMPTY for this agent"
                                                  << std::endl;
                                    }
                                    // Keep the "entry i == position i" ledger
                                    // invariant: the surviving positions start
                                    // at 0, so drop the prefix entries that
                                    // described the deleted prompt.
                                    if (agent->prompt_pos > 0) {
                                        const size_t drop = (size_t) new_min;
                                        if (agent->generated_tokens.size() > drop) {
                                            agent->generated_tokens.erase(
                                                agent->generated_tokens.begin(),
                                                agent->generated_tokens.begin() + drop);
                                        } else {
                                            agent->generated_tokens.clear();
                                        }
                                        agent->prompt_pos    = 0;
                                        agent->generated_len = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ── latent_batch_mode (C3) ──────────────────────────────────────
            // Execute the WHOLE latent span as ONE llama_decode carrying
            // latent_steps embedding rows, instead of latent_steps separate
            // one-row decodes. Correctness hinges on one fact verified in the
            // batch allocator: for EMBEDDING batches overlapping positions are
            // explicitly allowed (`the tokens of sequence N ... embedding inputs
            // can have overlapping positions`), and apply_ubatch() purges the
            // cells a new ubatch overwrites before writing. So N rows that all
            // carry the SAME position collapse to a single KV cell whose value
            // is the LAST row - which is exactly the folded semantics (reuse
            // one slot), just without N host<->device round-trips.
            //
            // Only meaningful with position_mode=folded; shared mode needs N
            // DISTINCT positions (one per step) and would change the KV layout,
            // so it stays on the per-step path. Default 0 = per-step (legacy).
            //
            // The feedback vector is not fed back between rows inside one call
            // (a single graph cannot read its own output mid-batch), so every
            // row is seeded from the SAME entering state h (delta/ema/etc.
            // transforms therefore do NOT apply here - they are inherently
            // cross-step). The value of batch mode is SPEED: one graph launch
            // for the span. Output is the last row's hidden state, which
            // becomes the next segment's seed exactly as the per-step loop's
            // final h would.
            bool batch_handled = false;
            if (seg.latent_batch_mode && seg.latent_position_mode == 1 && seg.latent_steps > 0) {
                batch_handled = true;
                // Clamp the batch to the token budget: the budget is a cap on
                // latent STEPS this segment may run, so a budget smaller than
                // latent_steps reduces the row count (0 means no rows).
                int32_t batch_steps = seg.latent_steps;
                const llama_pos bpos = (llama_pos) llama_mas_agent_pos(mas, agent_idx);
                if (seg.latent_max_tokens_budget > 0 &&
                    seg.latent_max_tokens_budget < batch_steps) {
                    batch_steps = seg.latent_max_tokens_budget;
                }

                if (batch_steps > 0) {
                    std::vector<float> aligned0(n_embd);
                    if (seg.align) {
                        if (llama_mas_apply_alignment(mas, h.data(), aligned0.data(), n_embd) != 0) {
                            std::cerr << "[MAS/turn] batch alignment apply failed" << std::endl;
                            llama_mas_free_turn_result(out_result);
                            return -1;
                        }
                    } else {
                        std::memcpy(aligned0.data(), h.data(), n_embd * sizeof(float));
                    }

                    llama_batch bb = llama_batch_init(batch_steps, n_embd, batch_steps);
                    for (int32_t i = 0; i < batch_steps; ++i) {
                        std::memcpy(&bb.embd[(size_t) i * n_embd], aligned0.data(), n_embd * sizeof(float));
                        bb.pos[i]        = bpos;       // SAME slot for every row
                        bb.n_seq_id[i]   = 1;
                        bb.seq_id[i][0]  = agent->seq_id;
                        bb.logits[i]     = 1;          // every row is an output; last row is the result
                    }
                    bb.n_tokens = batch_steps;

                    std::cerr << "[MAS/latent] BATCH mode: " << batch_steps
                              << " steps in ONE decode at pos=" << bpos
                              << " (row0 seed=entering h)" << std::endl;

                    llama_context_set_layer_graph(mas->ctx, agent->layer_graph);
                    int bret = llama_decode(mas->ctx, bb);
                    llama_batch_free(bb);
                    if (bret != 0) {
                        std::cerr << "[MAS/turn] batch latent decode failed ret=" << bret << std::endl;
                        llama_mas_free_turn_result(out_result);
                        return bret;
                    }

                    float * hs = llama_get_inp_embd_last_token(mas->ctx);
                    int32_t got = 0;
                    if (hs) {
                        got = llama_get_inp_embd_row_size(mas->ctx);
                    }
                    if (!hs || got != n_embd) {
                        hs = llama_mas_get_agent_hidden_state(mas, agent_idx, &got);
                    }
                    if (hs && got == n_embd) {
                        std::memcpy(h.data(), hs, n_embd * sizeof(float));
                        pending_h = h;
                        have_pending_h = true;
                        agent->tail_hidden = h;
                        agent->have_tail_hidden = true;
                    }

                    steps_done = batch_steps;
                    agent->generated_len += batch_steps;
                    mas->latent_stats.n_latent_steps     += batch_steps;
                    mas->latent_stats.n_kv_positions     += 1;   // ONE cell (folded)
                    n_latent += batch_steps;
                    out_result->n_latent_steps += batch_steps;
                    std::cerr << "[MAS/latent] BATCH DONE steps=" << steps_done
                              << " kv_positions=1 (folded slot)" << std::endl;
                }
            }

            for (int32_t s = 0; batch_handled ? false : (s < seg.latent_steps); ++s) {
                // ── folded position mode (C: latent_position_mode=1) ──────────
                // A folded span re-uses ONE position slot for every step. The
                // memory module's tail (seq_pos_max) is the only authority the
                // batch validator accepts, so "reuse the same slot" means
                // "remove the cell this span wrote, then write it again": after
                // the rm the tail is back at P-1 and step s+1 may legally
                // re-occupy position P.
                //
                // Doing the rm at the TOP of the step (rather than the bottom)
                // keeps the cell present during the probe/escape handling of the
                // step that just ran, which needs its own hidden state and,
                // for an escape, its own decode at pos+1.
                const bool folded = seg.latent_position_mode == 1;
                if (folded && s == 0) {
                    std::cerr << "[MAS/latent] position_mode=FOLDED (one slot for all "
                              << seg.latent_steps << " steps)" << std::endl;
                }
                if (folded && s > 0) {
                    llama_memory_t mem_fold = llama_get_memory(mas->ctx);
                    if (mem_fold) {
                        // TWO traps, both of which silently produced a shared run:
                        // 1. llama_memory_seq_rm returns BOOL (always true for a
                        //    valid seq id) - it is NOT a count of removed cells,
                        //    so it cannot confirm anything.
                        // 2. the range is HALF-OPEN: llama_kv_cells::pos_in is
                        //    `pos >= p0 && pos < p1`, so seq_rm(seq, P, P) is an
                        //    empty interval and removes NOTHING. The inclusive
                        //    upper bound must be P+1.
                        const llama_pos p = (llama_pos) llama_mas_agent_pos(mas, agent_idx) - 1;
                        const llama_pos before = llama_memory_seq_pos_max(mem_fold, agent->seq_id);
                        if (p >= 0) {
                            llama_memory_seq_rm(mem_fold, agent->seq_id, p, p + 1);
                        }
                        const llama_pos after = llama_memory_seq_pos_max(mem_fold, agent->seq_id);
                        std::cerr << "[MAS/latent] folded: rm pos=" << p
                                  << " tail " << before << " -> " << after
                                  << (after == before ? "  (NO-OP!)" : "")
                                  << " (step " << (s + 1) << ")" << std::endl;
                    }
                }

                const llama_pos pos = (llama_pos) llama_mas_agent_pos(mas, agent_idx);
                if (pos >= (llama_pos) llama_n_ctx(mas->ctx)) {
                    std::cerr << "[MAS/turn] latent segment hit n_ctx=" << llama_n_ctx(mas->ctx)
                              << " at step " << s << std::endl;
                    break;
                }

                // ── latent_max_tokens_budget (D3) ───────────────────────────
                // A latent step occupies one KV position exactly like a token,
                // so it must count against the same budget. The budget is a
                // per-SEGMENT cap on how many latent steps this span may run,
                // i.e. an explicit max_tokens for the think block. Without it
                // a long latent_steps value silently burns that many positions
                // of n_ctx; with it the span stops after `budget` steps
                // regardless of latent_steps. <= 0 disables (legacy).
                if (seg.latent_max_tokens_budget > 0 &&
                    s >= seg.latent_max_tokens_budget) {
                    std::cerr << "[MAS/latent] STOP budget at step " << (s + 1)
                              << "/" << seg.latent_steps
                              << " (budget=" << seg.latent_max_tokens_budget
                              << " steps)" << std::endl;
                    break;
                }

                llama_batch batch;
                if (s == 0 && use_pending) {
                    // Pure handoff: the previous text position's hidden state
                    // becomes this position's input embedding. Nothing is
                    // tokenized and no token is committed, so the latent run
                    // never duplicates the text that preceded it.
                    std::vector<float> aligned(n_embd);
                    if (seg.align) {
                        if (llama_mas_apply_alignment(mas, h.data(), aligned.data(), n_embd) != 0) {
                            std::cerr << "[MAS/turn] alignment apply failed" << std::endl;
                            break;
                        }
                    } else {
                        std::memcpy(aligned.data(), h.data(), n_embd * sizeof(float));
                    }
                    batch = llama_batch_init(1, n_embd, 1);
                    std::memcpy(&batch.embd[0], aligned.data(), n_embd * sizeof(float));
                    batch.pos[0]   = pos;
                    batch.n_seq_id[0] = 1;
                    batch.seq_id[0][0] = agent->seq_id;
                    batch.logits[0] = 1; // MUST be 1: n_outputs=0 makes the getter return NULL
                    batch.n_tokens = 1;
                } else if (s == 0) {
                    // Seed decode: a real token gives us the first hidden state.
                    // Only reached when no text segment preceded us, so the
                    // seed is not a duplicate of anything in the KV.
                    batch = llama_batch_init(1, 0, 1);
                    batch.token[0] = seed;
                    batch.pos[0]   = pos;
                    batch.n_seq_id[0] = 1;
                    batch.seq_id[0][0] = agent->seq_id;
                    batch.logits[0] = 1; // MUST be 1: n_outputs=0 makes the getter return NULL
                    batch.n_tokens = 1;
                } else {
                    // Feedback decode: raw embedding, no token, no sampling.
                    std::vector<float> aligned(n_embd);
                    if (seg.align) {
                        if (llama_mas_apply_alignment(mas, h.data(), aligned.data(), n_embd) != 0) {
                            std::cerr << "[MAS/turn] alignment apply failed" << std::endl;
                            break;
                        }
                    } else {
                        std::memcpy(aligned.data(), h.data(), n_embd * sizeof(float));
                    }
                    batch = llama_batch_init(1, n_embd, 1);
                    std::memcpy(&batch.embd[0], aligned.data(), n_embd * sizeof(float));
                    batch.pos[0]   = pos;
                    batch.n_seq_id[0] = 1;
                    batch.seq_id[0][0] = agent->seq_id;
                    batch.logits[0] = 1; // also 1 here: we need this position's hidden state
                    batch.n_tokens = 1;
                }

                llama_context_set_layer_graph(mas->ctx, agent->layer_graph);
                int ret = llama_decode(mas->ctx, batch);
                llama_batch_free(batch);
                if (ret != 0) {
                    std::cerr << "[MAS/turn] latent decode failed ret=" << ret
                              << " at step " << (s + 1) << "/" << seg.latent_steps << std::endl;
                    break;
                }

                int32_t got_embd = 0;
                float * hs = nullptr;

                // Prefer the layer-0 INPUT embedding for the feedback vector.
                // batch.embd enters the graph at layer 0, so the vector we feed back
                // must live in INPUT-embedding space. The final-layer OUTPUT hidden
                // state is a different space (same width, different meaning) - feeding
                // that back is what made latent quality collapse as steps accumulated.
                // Tap selection is EXCLUSIVE, not a fallback chain: a deeper
                // layer must be reachable even though the layer-0 tap always
                // succeeds. (First version used `if (!hs)` after the layer-0 tap
                // and was therefore dead code - every run logged the layer-0 tap
                // regardless of latent_source_layer.)
                // UPSTREAM SOURCE MODE (Gen-Verse/LatentMAS models.py
                // generate_latent_batch): the feedback vector is the LAST-LAYER
                // hidden state, and Wa is what maps it into input-embedding space:
                //
                //   last_hidden = outputs.hidden_states[-1][:, -1, :]
                //   latent_vec  = realign(last_hidden)      # Wa + norm rescale
                //   model(inputs_embeds=latent_vec, past_key_values=past)
                //
                // Their hidden_states[0] (the layer-0 input embedding) IS captured
                // but only appended to a `latent_vecs_all` analysis list - it is
                // NEVER fed back. This fork's default does the opposite: it feeds
                // the layer-0 tap, which is ALREADY in input space, and then applies
                // Wa on top. Since Wa's whole job is output->input, applying it to an
                // input-space vector is the wrong domain. That is a real design
                // divergence from upstream, so it must be measurable, not assumed.
                //
                // latent_feedback_source: 0 = layer-0 input tap (this fork's
                // historical default), 1 = final-layer hidden state (upstream).
                if (seg.latent_feedback_source == 1) {
                    hs = llama_mas_get_agent_hidden_state(mas, agent_idx, &got_embd);
                    if (s == 1) {
                        std::cerr << "[MAS/latent] using FINAL-LAYER hidden state "
                                  << "(upstream source mode, row=" << got_embd << ")"
                                  << std::endl;
                    }
                } else if (seg.latent_source_layer > 0) {
                    const int32_t want_layer = seg.latent_source_layer;
                    const int32_t n_layers = llama_get_hidden_states_layer_count(mas->ctx);
                    const int32_t idx = want_layer - 1;  // want_layer is 1-based
                    if (idx >= 0 && idx < n_layers) {
                        float * lh = llama_get_hidden_states_layer_ith(mas->ctx, idx, 0);
                        const int32_t nout = llama_get_hidden_states_layer_n_outputs(mas->ctx, idx);
                        if (lh && nout > 0) {
                            hs = lh;
                            got_embd = n_embd;
                        }
                        if (s == 1) {
                            // Fingerprint the vector, not just the shape: several
                            // layers produced byte-identical text in testing,
                            // which would otherwise be unattributable.
                            double fsum = 0.0, fsq = 0.0;
                            if (hs) {
                                for (int32_t q = 0; q < n_embd; ++q) {
                                    fsum += (double) hs[q];
                                    fsq  += (double) hs[q] * (double) hs[q];
                                }
                            }
                            std::cerr << "[MAS/latent] using LAYER-" << want_layer
                                      << " tap (layer_count=" << n_layers
                                      << ", got=" << (hs ? "ok" : "null")
                                      << ", sum=" << fsum << ", sq=" << fsq << ")" << std::endl;
                        }
                    } else if (s == 1) {
                        std::cerr << "[MAS/latent] layer " << want_layer
                                  << " out of range (count=" << n_layers << ")" << std::endl;
                    }
                } else {
                hs = llama_get_inp_embd_last_token(mas->ctx);
                if (hs) {
                    int32_t row = llama_get_inp_embd_row_size(mas->ctx);
                    int32_t nout = llama_get_inp_embd_n_outputs(mas->ctx);
                    if (row != n_embd || nout <= 0) {
                        std::cerr << "[MAS/turn] inp_embd geometry mismatch (row=" << row
                                  << " n_embd=" << n_embd << " n_outputs=" << nout << ")" << std::endl;
                        hs = nullptr;
                    } else {
                        got_embd = row;
                        if (s == 1) {
                            std::cerr << "[MAS/latent] using LAYER-0 INPUT tap (row=" << row << ")" << std::endl;
                        }
                    }
                }
                }  // end else (layer-0 input tap)

                if (!hs) {
                    // Fallback: final-layer hidden state. Better than aborting, but it
                    // is the out-of-distribution path described above.
                    hs = llama_mas_get_agent_hidden_state(mas, agent_idx, &got_embd);
                    if (s == 1) {
                        std::cerr << "[MAS/latent] WARNING falling back to final-layer hidden state" << std::endl;
                    }
                }

                if (!hs || got_embd != n_embd) {
                    std::cerr << "[MAS/turn] hidden state unavailable at step " << (s + 1)
                              << " (got_embd=" << got_embd << ")" << std::endl;
                    break;
                }
                std::memcpy(h.data(), hs, n_embd * sizeof(float));

                // ── Feedback transforms (all default off) ────────────────────
                // These operate on the vector we ALREADY hold, so unlike
                // latent_source_layer they cannot be defeated by ggml's buffer
                // aliasing. They change WHAT is fed back, not where it is read.
                //
                //   prior_layer : feed the PREVIOUS step's feedback vector
                //                 (a one-step delay; tests whether the loop is
                //                 chasing its own tail)
                //   delta       : feed h_t - h_{t-1} (the change, not the state)
                //   ema         : exponential moving average across steps,
                //                 momentum so a single noisy step cannot derail
                //                 the rest
                if (seg.latent_transform != 0) {
                    std::vector<float> fb = h;  // start from the fresh vector
                    switch (seg.latent_transform) {
                        case 1:  // prior step
                            if (have_prev_fb) fb = prev_fb;
                            break;
                        case 2:  // delta
                            if (have_prev_fb) {
                                for (int32_t q = 0; q < n_embd; ++q) fb[q] = h[q] - prev_fb[q];
                            }
                            break;
                        case 3:  // ema
                            if (have_prev_fb) {
                                const float a = seg.latent_ema_alpha > 0.0f ? seg.latent_ema_alpha : 0.7f;
                                for (int32_t q = 0; q < n_embd; ++q) {
                                    prev_fb[q] = a * h[q] + (1.0f - a) * prev_fb[q];
                                }
                                fb = prev_fb;
                            }
                            break;
                        case 4: {  // mean_pool ACROSS LATENT STEPS
                            // IMPORTANT: pooling over POSITIONS inside one decode
                            // is a structural no-op here. Each latent step decodes
                            // exactly ONE embedding, so inp_embd_n_outputs == 1 and
                            // a positional window of k>=1 collapses to the single
                            // last position - i.e. byte-identical to raw. Measured:
                            // k=2/4/8 all reproduced raw output exactly
                            // (coherence 0.154 @L4, 0.222 @L8, same strings).
                            // The smoothing that actually matters is across STEPS:
                            // average the last k feedback vectors.
                            const int32_t k = seg.latent_pool_window > 1
                                                  ? seg.latent_pool_window : 4;
                            hist_fb.push_back(h);
                            if ((int32_t) hist_fb.size() > k) {
                                hist_fb.erase(hist_fb.begin());
                            }
                            std::fill(fb.begin(), fb.end(), 0.0f);
                            for (const auto & hv : hist_fb) {
                                for (int32_t q = 0; q < n_embd; ++q) {
                                    fb[q] += hv[q];
                                }
                            }
                            const float inv = 1.0f / (float) hist_fb.size();
                            for (int32_t q = 0; q < n_embd; ++q) {
                                fb[q] *= inv;
                            }
                            break;
                        }
                        default:
                            break;
                    }
                    std::memcpy(h.data(), fb.data(), n_embd * sizeof(float));
                    if (s == 1) {
                        const char * nm = seg.latent_transform == 1 ? "prior_layer"
                                      : seg.latent_transform == 2 ? "delta"
                                      : seg.latent_transform == 3 ? "ema"
                                      : seg.latent_transform == 4 ? "mean_pool" : "?";
                        std::cerr << "[MAS/latent] transform=" << nm << std::endl;
                    }
                }
                // ── Position jitter (C2) ────────────────────────────────────
                // Attacks PREMATURE CONVERGENCE, which measurement identified as
                // the real cause of depth collapse: consecutive hidden states
                // differ only in the 5th decimal (0.252652 -> 0.252677 over 6
                // steps), so the model reads an effectively frozen vector and any
                // further step is wasted. Smoothing cannot help (there is nothing
                // to smooth - that is why mean_pool was a no-op); the state has to
                // be kept MOVING instead.
                //
                // Jitter adds a deterministic, step-varying offset along a fixed
                // pseudo-random direction. Deterministic so runs are reproducible
                // and the effect is attributable; small so it perturbs the state
                // without destroying it. Scale is relative to the vector's own
                // RMS, which keeps it meaningful across models of different width.
                if (seg.latent_jitter > 0.0f && n_embd > 0) {
                    // Per-step phase from a cheap integer hash so no <random> state
                    // is carried between calls and the same step always jitters
                    // the same way.
                    uint64_t ph = (uint64_t) (s + 1) * 0x9E3779B97F4A7C15ull;
                    double sumsq = 0.0;
                    for (int32_t q = 0; q < n_embd; ++q) {
                        sumsq += (double) h[q] * (double) h[q];
                    }
                    const double rms = (sumsq > 0.0)
                        ? std::sqrt(sumsq / (double) n_embd) : 1.0;
                    const double amp = (double) seg.latent_jitter * rms;
                    for (int32_t q = 0; q < n_embd; ++q) {
                        // xorshift64* per component: deterministic, no allocation
                        ph ^= ph >> 12; ph ^= ph << 25; ph ^= ph >> 27;
                        const uint64_t r = ph * 0x2545F4914F6CDD1Dull;
                        // map to [-1, 1)
                        const double u = ((double) (r >> 11) / 9007199254740992.0) * 2.0 - 1.0;
                        h[q] = (float) ((double) h[q] + amp * u);
                    }
                    if (s == 1) {
                        std::cerr << "[MAS/latent] jitter=" << seg.latent_jitter
                                  << " rms=" << rms << " amp=" << amp << std::endl;
                    }
                }

                prev_fb.assign(h.begin(), h.end());
                have_prev_fb = true;

                // ── Fixed-point / repeat detection (D1) ─────────────────────
                // Delta feedback's failure mode is a loop: the model keeps
                // emitting the same phrase ("... is the capital of France is
                // the capital of France"). Once the hidden state stops moving,
                // every further step is wasted compute AND wasted n_ctx.
                //
                // Cosine similarity between this step's vector and the previous
                // step's: if it exceeds the threshold, the recurrence has
                // converged and we stop rather than emitting more filler.
                // Threshold is configurable; 0 disables (default).
                if (seg.latent_stop_on_repeat > 0.0f && have_prev_cos) {
                    double dot = 0.0, na = 0.0, nb = 0.0;
                    for (int32_t q = 0; q < n_embd; ++q) {
                        dot += (double) h[q] * (double) prev_cos[q];
                        na  += (double) h[q] * (double) h[q];
                        nb  += (double) prev_cos[q] * (double) prev_cos[q];
                    }
                    const double dn = (na > 0.0 && nb > 0.0) ? dot / std::sqrt(na * nb) : 0.0;
                    if (dn >= (double) seg.latent_stop_on_repeat) {
                        // NOTE: `escaped` lives in a nested block below, so this
                        // cannot set it. Break out of the step loop directly -
                        // steps_done is not incremented for the detected step, so
                        // the DONE line and the ledger both stay accurate.
                        std::cerr << "[MAS/latent] STOP repeat at step " << (s + 1)
                                  << "/" << seg.latent_steps
                                  << " cos=" << dn
                                  << " >= " << seg.latent_stop_on_repeat << std::endl;
                        break;
                    }
                }
                prev_cos.assign(h.begin(), h.end());
                have_prev_cos = true;

                // ── Norm-drift stop (D2) ────────────────────────────────────
                // Detect the state's MAGNITUDE running away. Direction is
                // covered by latent_stop_on_repeat (cosine); this covers scale.
                //
                // Reference is the PREVIOUS step's norm when there is one
                // (scale-free: works under any transform, any model), else the
                // alignment target_norm (mean ||W_in[v]||).
                //
                // NOT anchored on target_norm alone - measured: with
                // rescale_norm ON the aligned vector is pinned to target_norm by
                // construction, so a target-anchored test can NEVER fire; with
                // rescale_norm OFF the first raw input embedding differs from
                // target_norm by >100%, so it fires at step 0. Both degenerate.
                // The previous-step reference makes the test meaningful in both
                // configurations.
                if (seg.latent_stop_on_norm_drift > 0.0f && n_embd > 0 &&
                    have_prev_norm && prev_norm > 0.0) {
                    double hsq = 0.0;
                    for (int32_t q = 0; q < n_embd; ++q) {
                        hsq += (double) h[q] * (double) h[q];
                    }
                    const double hnrm = std::sqrt(hsq);
                    const double ref  = prev_norm;
                    const double adr  = (hnrm > ref ? hnrm - ref : ref - hnrm) / ref;
                    if (adr > (double) seg.latent_stop_on_norm_drift) {
                        std::cerr << "[MAS/latent] STOP norm_drift at step " << (s + 1)
                                  << "/" << seg.latent_steps
                                  << " ||h||=" << hnrm
                                  << " prev=" << ref
                                  << " drift=" << adr
                                  << " > " << seg.latent_stop_on_norm_drift << std::endl;
                        break;
                    }
                }
                prev_norm = 0.0;
                for (int32_t q = 0; q < n_embd; ++q) {
                    prev_norm += (double) h[q] * (double) h[q];
                }
                prev_norm = std::sqrt(prev_norm);
                have_prev_norm = true;

                agent->generated_len++;
                agent->latent_steps_remaining = agent->latent_steps_remaining > 0 ? agent->latent_steps_remaining - 1 : 0;
                mas->latent_stats.n_latent_steps++;
                mas->latent_stats.n_kv_positions++;
                steps_done++;
                // Per-step latent tick: one line per hidden-state step so a
                // live run is visible in the terminal (and attributable) instead
                // of only appearing as an aggregate after the turn.
                std::cerr << "[MAS/latent] agent=" << agent->role
                          << " step=" << steps_done << "/" << seg.latent_steps
                          << " pos=" << pos
                          << " t=" << (mas_time_ms() - t_latent_start) << "ms"
                          << std::endl;
                n_latent++;
                out_result->n_latent_steps++;

                // Escape check. The latent loop must be able to hand control back
                // to the caller on its own, otherwise the caller has to guess how
                // many steps a task needs (and every fixed guess was wrong for some
                // query: 2 steps helped one prompt and broke the next).
                //
                // Sampling here is ONLY a probe: the hidden state drives the next
                // step, so a probe token is never committed unless it is the
                // escape itself (see below) or EOG.
                if (seg.stop_on_eog || (seg.escape_text && seg.escape_text[0])) {
                    llama_token probe = llama_sampler_sample(mas->samplers[agent_idx], mas->ctx, -1);

                    bool escaped = false;
                    if (llama_vocab_is_eog(vocab, probe)) {
                        escaped = true;
                    } else if (seg.escape_text && seg.escape_text[0]) {
                        // Detokenize just this token and compare its surface form.
                        // Special tokens detokenize to their literal text, so this
                        // works for <|done_thinking|>-style markers without the
                        // caller needing the model's token id.
                        char buf[128];
                        int n = llama_token_to_piece(vocab, probe, buf, sizeof(buf), 0, true);
                        if (n > 0) {
                            std::string piece(buf, (size_t) n);
                            // Tolerate surrounding whitespace differences between
                            // the marker as written and as the tokenizer emits it.
                            auto trim = [](std::string s) {
                                size_t a = s.find_first_not_of(" \t\r\n");
                                size_t b = s.find_last_not_of(" \t\r\n");
                                return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
                            };
                            escaped = (trim(piece) == trim(seg.escape_text));
                        }
                    }

                    if (escaped) {
                        // Commit the escape token so the caller sees it in the
                        // output text, then stop the latent block.
                        llama_sampler_accept(mas->samplers[agent_idx], probe);
                        agent->generated_tokens.push_back(probe);
                        agent->generated_len++;
                        sampled.push_back(probe);
                        n_text++;
                        {
                            llama_batch eb = llama_batch_init(1, 0, 1);
                            eb.token[0]     = probe;
                            eb.pos[0]       = llama_mas_agent_pos(mas, agent_idx);
                            eb.n_seq_id[0]  = 1;
                            eb.seq_id[0][0] = agent->seq_id;
                            eb.logits[0]    = 1;
                            eb.n_tokens     = 1;
                            int erc = llama_decode(mas->ctx, eb);
                            llama_batch_free(eb);
                            if (erc != 0) {
                                std::cerr << "[MAS/latent] escape-commit decode failed ret=" << erc << std::endl;
                                llama_mas_free_turn_result(out_result);
                                return erc;
                            }
                            agent->prompt_pos = llama_mas_agent_pos(mas, agent_idx);
                        }
                        std::cerr << "[MAS/latent] ESCAPE agent=" << agent->role
                                  << " after step=" << steps_done
                                  << " token_piece=\"" << (seg.escape_text ? seg.escape_text : "") << "\""
                                  << std::endl;
                        break;
                    }
                }
            }

            // Leave capture on: the next segment may want the tail state, and
            // with logits only on the last position the cost is one vector.
            // ---- latent segment timing (start/finish pair) ----
            // t_latent_start was taken before the first decode; this is the
            // matching finish. Printed with the turn index and step count so a
            // terminal shows exactly how long each thinking block took and how
            // many turns it ran.
            {
                const double t_lat_done    = mas_time_ms();
                const double t_lat_elapsed = t_lat_done - t_latent_start;
                t_latent_ms += t_lat_elapsed;
                const double per_step = steps_done > 0 ? t_lat_elapsed / steps_done : 0.0;
                mas->latent_stats.n_latent_segments++;
                if (t_lat_elapsed > 0) {
                    mas->latent_stats.n_latent_ms += (uint64_t) t_lat_elapsed;
                }
                if (steps_done > mas->latent_stats.max_depth) {
                    mas->latent_stats.max_depth = steps_done;
                }
                std::cerr << "[MAS/latent] DONE agent=" << agent->role
                          << " turn=" << (si + 1) << "/" << n_segments
                          << " steps=" << steps_done << "/" << seg.latent_steps
                          << " start=" << t_latent_start << "ms"
                          << " finish=" << t_lat_done << "ms"
                          << " elapsed=" << t_lat_elapsed << "ms"
                          << " per_step=" << per_step << "ms"
                          << " rate=" << (per_step > 0.0 ? (1000.0 / per_step) : 0.0) << " steps/s"
                          << " total_steps=" << out_result->n_latent_steps
                          << std::endl;
            }

            if (steps_done > 0 && h.size() == (size_t) n_embd) {
                pending_h = h;
                have_pending_h = true;
                agent->tail_hidden = h;
                agent->have_tail_hidden = true;
            }

            log.n_latent = steps_done;
            log.pos_end  = llama_mas_agent_pos(mas, agent_idx);
            // CRC over the ONE hidden vector we actually hold. This used to read
            // steps_done * n_embd floats from h.data(), but h is a single n_embd
            // vector - so every latent step multiplied the read length and walked
            // off the end of the allocation. It survived small step counts (still
            // mapped memory) and segfaulted at 32 (0xC0000005) once it crossed into
            // an unmapped page. steps_done is recorded in the ledger separately.
            log.tok_crc  = mas_crc32(h.data(), (size_t)n_embd * sizeof(float));
            // Latent positions produce no token ids, so the token stream is
            // unchanged across this segment. n_tokens stays 0 and the caller's
            // token index does not advance over a latent span.
            mas->turn_log.push_back(log);

            // ── latent_truncate_after ───────────────────────────────────
            // Upstream's latent_only mode keeps the KV to exactly the latent
            // positions after the loop. Our default keeps them, which means a
            // long think block permanently occupies n_ctx. This toggle rewinds
            // to just before the latent span so the next segment starts clean.
            if (seg.latent_truncate_after && steps_done > 0) {
                llama_memory_t mem_t = llama_get_memory(mas->ctx);
                if (mem_t) {
                    const llama_pos pmax = llama_memory_seq_pos_max(mem_t, agent->seq_id);
                    const llama_pos keep_to = (llama_pos) llama_mas_agent_pos(mas, agent_idx)
                                              - (llama_pos) steps_done - 1;
                    if (pmax >= 0 && keep_to >= 0) {
                        llama_memory_seq_rm(mem_t, agent->seq_id, keep_to + 1, -1);
                        std::cerr << "[MAS/latent] truncate_after kept 0.." << keep_to
                                  << " (" << steps_done << " latent positions dropped)" << std::endl;
                    }
                }
            }

            if (steps_done < seg.latent_steps) {
                std::cerr << "[MAS/turn] latent segment " << si << " stopped early: "
                          << steps_done << "/" << seg.latent_steps << std::endl;
            }
        }
    }

    out_result->n_positions = (int32_t) llama_mas_agent_pos(mas, agent_idx);
    out_result->n_tokens    = (int32_t) sampled.size();
    out_result->n_tokens_sampled = (int32_t) sampled.size();
    // Perplexity of this agent's emitted text under the model itself. Summed from
    // the per-token log-softmax values recorded at each sample site; latent steps
    // contribute no tokens and so are excluded from the average.
    out_result->logprob_sum = 0.0;
    out_result->logprob_n   = 0;
    if (agent_idx >= 0 && agent_idx < (int32_t) mas->agents.size()) {
        out_result->logprob_sum = mas->agents[agent_idx]->logprob_sum;
        out_result->logprob_n   = mas->agents[agent_idx]->logprob_n;
    }
    out_result->think_tokens_per_sec = t_latent_ms > 0.0 ? (float)(n_latent * 1000.0 / t_latent_ms) : 0.0f;
    out_result->text_tokens_per_sec  = t_text_ms   > 0.0 ? (float)(n_text   * 1000.0 / t_text_ms)   : 0.0f;

    // ── process-wide latent accounting ─────────────────────────────────────
    // These two counters were DECLARED and REPORTED by /mas/latent/stats but
    // never assigned anywhere in the tree: the endpoint has answered
    // `tokens_sampled: 0, text_ms: 0` on every run since it was written, which
    // made "how many real tokens did the latent span displace" - the whole
    // point of the F-section token-savings report - unmeasurable.
    //
    // Semantics, stated so the numbers are unambiguous:
    //   n_tokens_sampled : tokens the model actually EMITTED (sampled), which
    //                      is what a text-only baseline would have produced.
    //                      n_text also counts prompt tokens fed to prefill, so
    //                      it is NOT the token count to report.
    //   n_text_ms        : wall time inside text (non-latent) segments only.
    mas->latent_stats.n_tokens_sampled += (uint64_t) sampled.size();
    mas->latent_stats.n_text_ms        += (uint64_t) t_text_ms;

    if (!sampled.empty()) {
        out_result->tokens = (llama_token *) malloc(sizeof(llama_token) * sampled.size());
        if (out_result->tokens) {
            std::memcpy(out_result->tokens, sampled.data(), sizeof(llama_token) * sampled.size());
        }
        std::string text;
        int32_t n_chars = llama_detokenize(vocab, sampled.data(), (int32_t) sampled.size(), nullptr, 0, false, true);
        if (n_chars < 0) n_chars = -n_chars;
        if (n_chars > 0) {
            text.resize((size_t) n_chars);
            llama_detokenize(vocab, sampled.data(), (int32_t) sampled.size(), &text[0], n_chars, false, true);
        }
        out_result->text = (char *) malloc(text.size() + 1);
        if (out_result->text) {
            std::memcpy(out_result->text, text.c_str(), text.size() + 1);
        }
    }

    std::cerr << "[MAS/turn] agent=" << agent->role
              << " segments=" << n_segments
              << " pos=" << out_result->n_positions
              << " latent=" << out_result->n_latent_steps
              << " (" << out_result->think_tokens_per_sec << " steps/s)"
              << " text=" << out_result->n_tokens
              << " (" << out_result->text_tokens_per_sec << " tok/s)" << std::endl;

    return 0;
}


// ---------------------------------------------------------------------------
// Session persistence (hybrid turns)
//
// A session = one agent's KV sequence + the ledger of what occupies each
// position. The ledger is what makes reuse safe: on the next turn we tokenize
// the new text, walk the ledger, and reuse exactly the leading positions whose
// token ids match. Anything past the first mismatch is truncated and re-decoded.
//
// Text is stored for humans (diffing a coding conversation, debugging why a
// cache missed) but the hit test is ALWAYS token ids, because tokenization is
// what the cache holds and text can tokenize differently after an edit that
// looks cosmetic.
// ---------------------------------------------------------------------------

int llama_mas_turn_log_count(const llama_mas_context * mas) {
    return mas ? (int32_t) mas->turn_log.size() : 0;
}

const char * llama_mas_turn_log_text(const llama_mas_context * mas, int32_t i) {
    if (!mas || i < 0 || i >= (int32_t) mas->turn_log.size()) return nullptr;
    const std::string & s = mas->turn_log[i].text;
    return s.empty() ? nullptr : s.c_str();
}

int llama_mas_turn_log_kind(const llama_mas_context * mas, int32_t i) {
    if (!mas || i < 0 || i >= (int32_t) mas->turn_log.size()) return -1;
    return mas->turn_log[i].kind;
}

size_t llama_mas_turn_log_pos_start(const llama_mas_context * mas, int32_t i) {
    if (!mas || i < 0 || i >= (int32_t) mas->turn_log.size()) return 0;
    return mas->turn_log[i].pos_start;
}

size_t llama_mas_turn_log_pos_end(const llama_mas_context * mas, int32_t i) {
    if (!mas || i < 0 || i >= (int32_t) mas->turn_log.size()) return 0;
    return mas->turn_log[i].pos_end;
}

// Longest prefix of the agent's sequence that `tokens` still matches.
//
// Walks the ledger in position order. TEXT segments compare their stored token
// ids against the caller's stream; the first mismatch ends the reusable prefix.
// LATENT segments have no token ids (they never produced a token), so they are
// reusable exactly when everything before them matched - which is the correct
// semantics: a latent run is a pure function of the state it was fed.
//
// Returns the number of leading positions that can be kept. The caller must
// llama_mas_agent_truncate() to that length and re-decode the rest.
size_t llama_mas_session_match_len(const llama_mas_context * mas,
                                   const llama_token * new_tokens, size_t n_new) {
    if (!mas) return 0;
    size_t ti = 0;  // index into new_tokens
    size_t pos = 0; // position in the KV sequence

    for (const auto & seg : mas->turn_log) {
        if (seg.kind == LLAMA_MAS_SEGMENT_LATENT) {
            pos = seg.pos_end;
            continue;
        }
        for (size_t k = 0; k < seg.tokens.size(); ++k, ++ti) {
            if (ti >= n_new || new_tokens[ti] != seg.tokens[k]) {
                return pos;
            }
            ++pos;
        }
        // positions covered by this segment must line up with the ledger
        if (seg.pos_end > pos) pos = seg.pos_end;
    }
    return pos;
}

int llama_mas_session_export(llama_mas_context * mas, int32_t agent_idx, uint8_t * dst, size_t cap, size_t * out_size) {
    if (!mas || !mas->ctx) return -1;
    size_t need = llama_state_seq_get_size(mas->ctx, agent_idx);
    if (out_size) *out_size = need;
    if (!dst || cap < need) return -2;
    return llama_state_seq_get_data(mas->ctx, dst, cap, agent_idx) == need ? 0 : -3;
}

int llama_mas_session_import(llama_mas_context * mas, int32_t agent_idx, const uint8_t * src, size_t size) {
    if (!mas || !mas->ctx || !src || size == 0) return -1;
    return llama_state_seq_set_data(mas->ctx, src, size, agent_idx) == size ? 0 : -2;
}



// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
static std::vector<llama_token> tokenize_cached(llama_mas_context * mas,
                                                const llama_vocab * vocab,
                                                const std::string & text) {
    if (!vocab || text.empty()) return {};
    if (mas->use_tokenization_cache) {
        auto it = mas->token_cache.find(text);
        if (it != mas->token_cache.end()) return it->second;
    }
    std::vector<llama_token> out;
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, true, true);
    if (n < 0) n = -n;
    if (n > 0) {
        out.resize((size_t) n);
        llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), n, true, true);
    }
    if (mas->use_tokenization_cache) {
        mas->token_cache[text] = out;
    }
    return out;
}

static llama_sampler * create_agent_sampler(const llama_mas_agent * agent, bool init_backend) {
    (void) init_backend;
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler * chain = llama_sampler_chain_init(chain_params);
    if (!chain) return nullptr;

    // Repetition penalty. `agent->repeat_penalty` was DECLARED and DEFAULTED to
    // 1.1 but NEVER read by the sampler, so every agent decoded with no penalty at
    // all. Combined with greedy (temperature <= 0) that is a guaranteed loop: a
    // single agent asking "how many tools are left?" emitted "The answer is 31.
    // The answer is 31. ..." until it hit the token ceiling, which poisoned every
    // per-arm ceiling statistic. Penalties are applied BEFORE the temperature
    // branch so the penalty is present for greedy too (that is the case that
    // needs it most - greedy has no sampling noise to break a tie).
    const int32_t last_n = agent->repeat_last_n > 0 ? agent->repeat_last_n : 64;
    if (agent->repeat_penalty > 1.0f || agent->penalty_freq > 0.0f || agent->penalty_present > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
            last_n, agent->repeat_penalty, agent->penalty_freq, agent->penalty_present));
    }

    if (agent->temperature <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        return chain;
    }
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(agent->top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(agent->top_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(agent->temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist((uint32_t) 0));
    return chain;
}

static const llama_mas_adapter_weights * find_common_adapter(
        const std::vector<int> & indices,
        const std::vector<std::unique_ptr<llama_mas_agent>> & agents) {
    const llama_mas_adapter_weights * common = nullptr;
    for (int idx : indices) {
        auto & agent = agents[idx];
        if (agent->adapter_type == LLAMA_MAS_ADAPTER_NONE) continue;
        if (!common) {
            common = agent->adapter_weights.get();
        } else if (agent->adapter_weights.get() != common) {
            return nullptr; // mixed adapters cannot share one decode
        }
    }
    return common;
}

// ---------------------------------------------------------------------------
// Context / agent lifecycle
// ---------------------------------------------------------------------------
llama_mas_context * llama_mas_init(
        llama_model * model,
        llama_context_params params,
        int32_t n_agents_max,
        bool use_batched_decode) {
    if (!model || n_agents_max <= 0) return nullptr;

    auto * mas = new llama_mas_context();
    mas->model = model;
    mas->n_agents_max = n_agents_max;
    mas->use_batched_decode = use_batched_decode;

    // One KV slot per agent: raise n_seq_max so every agent has its own stream.
    if (params.n_seq_max < (uint32_t) n_agents_max) {
        params.n_seq_max = (uint32_t) n_agents_max;
    }

    mas->ctx = llama_init_from_model(model, params);
    if (!mas->ctx) {
        delete mas;
        return nullptr;
    }

    mas->ctx->set_use_cached_shift_graph(mas->use_cached_shift_graph);
    return mas;
}

void llama_mas_free(llama_mas_context * mas) {
    if (!mas) return;
    for (auto & smpl : mas->samplers) {
        if (smpl) llama_sampler_free(smpl);
    }
    mas->samplers.clear();
    if (mas->ctx) llama_free(mas->ctx);
    delete mas;
}

llama_context * llama_mas_get_context(llama_mas_context * mas) {
    return mas ? mas->ctx : nullptr;
}

int32_t llama_mas_agent_count(const llama_mas_context * mas) {
    return mas ? (int32_t) mas->agents.size() : 0;
}

static int32_t mas_add_agent_common(llama_mas_context * mas,
                                    const char * role,
                                    int32_t seq_id,
                                    llama_mas_adapter_type adapter_type,
                                    llama_mas_adapter_weights * weights,
                                    int32_t max_tokens,
                                    float temperature,
                                    float top_p,
                                    int32_t top_k,
                                    int32_t stop_after_prompt,
                                    int32_t is_latent,
                                    int32_t latent_steps) {
    if (!mas || !mas->ctx) return -1;
    if ((int32_t) mas->agents.size() >= mas->n_agents_max) return -1;
    if (seq_id < 0 || seq_id >= mas->n_agents_max) return -1;

    auto agent = std::make_unique<llama_mas_agent>();
    agent->role = role ? role : "";
    agent->seq_id = (llama_seq_id) seq_id;
    agent->adapter_type = adapter_type;
    if (weights) agent->adapter_weights.reset(weights);
    agent->max_tokens = max_tokens;
    agent->temperature = temperature;
    agent->top_p = top_p;
    agent->top_k = top_k;
    agent->repeat_penalty = 1.1f;
    agent->stop_after_prompt = stop_after_prompt != 0;
    agent->is_latent = is_latent != 0;
    agent->latent_steps = latent_steps;
    agent->latent_steps_remaining = latent_steps;

    llama_sampler * smpl = create_agent_sampler(agent.get(), mas->use_backend_sampling);
    if (!smpl) return -1;

    mas->agents.push_back(std::move(agent));
    mas->samplers.push_back(smpl);

    // The vector may have reallocated: samplers stay indexed by agent index.
    int32_t idx = (int32_t) mas->agents.size() - 1;
    if (mas->use_backend_sampling) {
        llama_set_sampler(mas->ctx, (llama_seq_id) seq_id, smpl);
    }
    return idx;
}

int32_t llama_mas_add_agent(llama_mas_context * mas, const llama_mas_agent_config * config) {
    if (!config) return -1;
    return mas_add_agent_common(mas,
            config->role,
            config->seq_id,
            (llama_mas_adapter_type) config->adapter_type,
            config->adapter_weights,
            config->max_tokens,
            config->temperature,
            config->top_p,
            config->top_k,
            config->stop_after_prompt ? 1 : 0,
            config->is_latent ? 1 : 0,
            config->latent_steps);
}

int32_t llama_mas_add_agent_flat(
        llama_mas_context * mas,
        const char * role,
        int32_t seq_id,
        int32_t adapter_type,
        int32_t max_tokens,
        float temperature,
        float top_p,
        int32_t top_k,
        int32_t stop_after_prompt,
        int32_t is_latent,
        int32_t latent_steps) {
    return mas_add_agent_common(mas, role, seq_id,
            (llama_mas_adapter_type) adapter_type, nullptr,
            max_tokens, temperature, top_p, top_k,
            stop_after_prompt, is_latent, latent_steps);
}

void llama_mas_remove_agent(llama_mas_context * mas, int32_t agent_idx) {
    if (!mas) return;
    if (agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return;

    if (mas->use_backend_sampling && mas->ctx) {
        llama_set_sampler(mas->ctx, mas->agents[agent_idx]->seq_id, nullptr);
    }
    if (agent_idx < (int32_t) mas->samplers.size() && mas->samplers[agent_idx]) {
        llama_sampler_free(mas->samplers[agent_idx]);
    }
    mas->samplers.erase(mas->samplers.begin() + agent_idx);
    mas->agents.erase(mas->agents.begin() + agent_idx);

    llama_memory_t mem = llama_get_memory(mas->ctx);
    if (mem) llama_memory_seq_rm(mem, (llama_seq_id) agent_idx, -1, -1);
}

void llama_mas_set_agent_uuid(llama_mas_context * mas, int32_t agent_idx, const char * uuid) {
    if (!mas || agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return;
    mas->agents[agent_idx]->uuid = uuid ? uuid : "";
}

void llama_mas_clear_agents(llama_mas_context * mas) {
    if (!mas) return;
    if (mas->ctx) {
        llama_memory_t mem = llama_get_memory(mas->ctx);
        if (mem) llama_memory_clear(mem, true);
    }
    for (size_t i = 0; i < mas->samplers.size(); ++i) {
        if (mas->use_backend_sampling && mas->ctx && i < mas->agents.size()) {
            llama_set_sampler(mas->ctx, mas->agents[i]->seq_id, nullptr);
        }
        if (mas->samplers[i]) llama_sampler_free(mas->samplers[i]);
    }
    mas->samplers.clear();
    // Free per-agent layer graphs (owned here; see llama_mas_set_agent_layer_graph).
    for (auto & agent : mas->agents) {
        if (agent && agent->layer_graph) {
            llama_layer_graph_free(agent->layer_graph);
            agent->layer_graph = nullptr;
        }
    }
    mas->agents.clear();
    mas->broadcast_waves.clear();
    mas->turn_log.clear();
}

void llama_mas_set_agent_prompt(llama_mas_context * mas, int32_t agent_idx, const char * prompt) {
    if (!mas || agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return;
    mas->agents[agent_idx]->prompt_text = prompt ? prompt : "";
}

void llama_mas_set_agent_input_embeddings(
        llama_mas_context * mas,
        int32_t agent_idx,
        const float * embeddings,
        int32_t n_tokens,
        int32_t n_embd) {
    if (!mas || agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return;
    auto & agent = mas->agents[agent_idx];
    if (!embeddings || n_tokens <= 0 || n_embd <= 0) {
        agent->input_embeddings.clear();
        agent->input_embeddings_n_embd = 0;
        return;
    }
    agent->input_embeddings.resize((size_t) n_tokens * n_embd);
    std::memcpy(agent->input_embeddings.data(), embeddings,
                agent->input_embeddings.size() * sizeof(float));
    agent->input_embeddings_n_embd = n_embd;
}

void llama_mas_set_agent_adapter(llama_mas_context * mas, int32_t agent_idx,
                                 llama_mas_adapter_type adapter_type,
                                 llama_mas_adapter_weights * weights) {
    if (!mas || agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return;
    auto & agent = mas->agents[agent_idx];
    agent->adapter_type = adapter_type;
    agent->adapter_weights.reset(weights); // takes ownership
}

// Defined 2026-11-19: previously declared in llama-mas.h but never defined.
// Ownership is BORROWED (the caller keeps `lg` and must free it, or pass NULL
// to release). A non-NULL graph is freed with the agent so a caller that only
// ever passes non-NULL cannot leak; NULL simply clears the agent's graph.
int llama_mas_set_agent_layer_graph(llama_mas_context * mas, int32_t agent_idx,
                                    llama_layer_graph_context * lg) {
    if (!mas || agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return -1;
    mas->agents[agent_idx]->layer_graph = lg;
    return 0;
}

extern "C" LLAMA_API int32_t llama_mas_agent_get_info(const llama_mas_context * mas, int32_t agent_idx,
                                                      llama_mas_agent_info * out_info) {
    if (!mas || !out_info) return -1;
    if (agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return -1;
    const auto & a = mas->agents[agent_idx];
    out_info->role = a->role.c_str();
    out_info->seq_id = a->seq_id;
    out_info->adapter_type = (int32_t) a->adapter_type;
    out_info->max_tokens = a->max_tokens;
    out_info->temperature = a->temperature;
    out_info->top_p = a->top_p;
    out_info->top_k = a->top_k;
    out_info->prompt_text = a->prompt_text.empty() ? nullptr : a->prompt_text.c_str();
    out_info->uuid = a->uuid.empty() ? nullptr : a->uuid.c_str();
    out_info->stop_after_prompt = a->stop_after_prompt;
    out_info->is_latent = a->is_latent;
    out_info->latent_steps = a->latent_steps;
    out_info->runtime_ms = (a->run_finished_ms > 0.0 && a->run_started_ms > 0.0)
                         ? (a->run_finished_ms - a->run_started_ms) : 0.0;
    return 0;
}

const llama_token * llama_mas_get_tokens(const llama_mas_context * mas, int32_t agent_idx,
                                         size_t * out_count) {
    if (out_count) *out_count = 0;
    if (!mas || agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return nullptr;
    const auto & agent = mas->agents[agent_idx];
    if (out_count) *out_count = agent->generated_tokens.size();
    return agent->generated_tokens.data();
}

// ---------------------------------------------------------------------------
// Toggles
// ---------------------------------------------------------------------------
void llama_mas_set_strategy(llama_mas_context * mas, llama_mas_strategy strategy) {
    if (mas) mas->strategy = strategy;
}

void llama_mas_set_broadcast_waves(llama_mas_context * mas, int32_t ** waves,
                                   int32_t * wave_sizes, int32_t n_waves) {
    if (!mas) return;
    mas->broadcast_waves.clear();
    if (!waves || !wave_sizes || n_waves <= 0) return;
    mas->broadcast_waves.reserve((size_t) n_waves);
    for (int32_t w = 0; w < n_waves; ++w) {
        std::vector<int32_t> wave;
        if (wave_sizes[w] > 0 && waves[w]) {
            wave.reserve((size_t) wave_sizes[w]);
            for (int32_t i = 0; i < wave_sizes[w]; ++i) wave.push_back(waves[w][i]);
        }
        mas->broadcast_waves.push_back(std::move(wave));
    }
}

void llama_mas_set_stream_callback(llama_mas_context * mas, llama_mas_stream_cb cb, void * user_data) {
    if (!mas) return;
    mas->stream_cb = cb;
    mas->stream_user_data = user_data;
}

void llama_mas_get_timing(const llama_mas_context * mas, llama_mas_timing * timing) {
    if (!mas || !timing) return;
    *timing = mas->last_timing;
}

void llama_mas_set_output_hidden_states(llama_mas_context * mas, bool output_hidden_states) {
    if (!mas || !mas->ctx) return;
    llama_set_output_hidden_states(mas->ctx, output_hidden_states);
}

void llama_mas_set_stop_after_prompt(llama_mas_context * mas, bool stop_after_prompt) {
    if (mas) mas->stop_after_prompt = stop_after_prompt;
}

void llama_mas_set_cuda_adapters(llama_mas_context * mas, bool use_cuda) {
    if (mas) mas->use_cuda_adapters = use_cuda;
}

bool llama_mas_get_cuda_adapters(const llama_mas_context * mas) {
    return mas ? mas->use_cuda_adapters : false;
}

void llama_mas_set_batched_decode(llama_mas_context * mas, bool use_batched) {
    if (mas) mas->use_batched_decode = use_batched;
}

bool llama_mas_get_batched_decode(const llama_mas_context * mas) {
    return mas ? mas->use_batched_decode : false;
}

void llama_mas_set_backend_sampling(llama_mas_context * mas, bool use_backend) {
    if (mas) mas->use_backend_sampling = use_backend;
}

bool llama_mas_get_backend_sampling(const llama_mas_context * mas) {
    return mas ? mas->use_backend_sampling : false;
}

void llama_mas_refresh_samplers(llama_mas_context * mas) {
    if (!mas || !mas->ctx) return;
    for (size_t i = 0; i < mas->samplers.size() && i < mas->agents.size(); ++i) {
        if (mas->samplers[i]) llama_sampler_free(mas->samplers[i]);
        mas->samplers[i] = create_agent_sampler(mas->agents[i].get(), mas->use_backend_sampling);
        if (mas->use_backend_sampling) {
            llama_set_sampler(mas->ctx, mas->agents[i]->seq_id, mas->samplers[i]);
        }
    }
}

void llama_mas_set_tokenization_cache(llama_mas_context * mas, bool use_cache) {
    if (!mas) return;
    mas->use_tokenization_cache = use_cache;
    if (!use_cache) mas->token_cache.clear();
}

bool llama_mas_get_tokenization_cache(const llama_mas_context * mas) {
    return mas ? mas->use_tokenization_cache : false;
}

void llama_mas_set_cached_shift_graph(llama_mas_context * mas, bool use_cache) {
    if (!mas || !mas->ctx) return;
    mas->use_cached_shift_graph = use_cache;
    mas->ctx->set_use_cached_shift_graph(use_cache);
}

bool llama_mas_get_cached_shift_graph(const llama_mas_context * mas) {
    return mas ? mas->use_cached_shift_graph : false;
}

void llama_mas_set_gpu_hidden_states(llama_mas_context * mas, bool use_gpu) {
    if (mas) mas->use_gpu_hidden_states = use_gpu;
}

bool llama_mas_get_gpu_hidden_states(const llama_mas_context * mas) {
    return mas ? mas->use_gpu_hidden_states : false;
}

void llama_mas_set_async_tokenization(llama_mas_context * mas, bool use_async) {
    if (mas) mas->use_async_tokenization = use_async;
}

bool llama_mas_get_async_tokenization(const llama_mas_context * mas) {
    return mas ? mas->use_async_tokenization : false;
}

void llama_mas_set_alignment_vocab_stride(llama_mas_context * mas, int32_t stride) {
    if (!mas) return;
    mas->alignment_vocab_stride = stride < 1 ? 1 : stride;
}

int32_t llama_mas_get_alignment_vocab_stride(const llama_mas_context * mas) {
    return mas ? mas->alignment_vocab_stride : 1;
}

// ── A-section alignment solve knobs ────────────────────────────────────────
void llama_mas_set_alignment_full_vocab(llama_mas_context * mas, bool on) {
    if (mas) mas->alignment_full_vocab = on;
}

bool llama_mas_get_alignment_full_vocab(const llama_mas_context * mas) {
    return mas ? mas->alignment_full_vocab : false;
}

void llama_mas_set_alignment_lowrank(llama_mas_context * mas, int32_t r) {
    if (mas) mas->alignment_lowrank = r;
}

int32_t llama_mas_get_alignment_lowrank(const llama_mas_context * mas) {
    return mas ? mas->alignment_lowrank : 0;
}

void llama_mas_set_alignment_dtype(llama_mas_context * mas, int32_t dtype) {
    if (mas) mas->alignment_dtype = dtype;
}

int32_t llama_mas_get_alignment_dtype(const llama_mas_context * mas) {
    return mas ? mas->alignment_dtype : 0;
}

void llama_mas_set_alignment_cross_progress(llama_mas_context * mas, bool on) {
    if (mas) mas->alignment_cross_progress = on;
}

bool llama_mas_get_alignment_cross_progress(const llama_mas_context * mas) {
    return mas ? mas->alignment_cross_progress : false;
}

void llama_mas_set_alignment_ridge(llama_mas_context * mas, double ridge) {
    if (!mas) return;
    // <= 0 means "use the legacy default", so store 0 and let the effective
    // helper decide. Negative values are not meaningful as a ridge.
    mas->alignment_ridge = ridge > 0.0 ? ridge : 0.0;
    std::cerr << "[MAS/alignment] ridge=" << mas_alignment_effective_ridge(mas)
              << (mas->alignment_ridge > 0.0 ? " (explicit)" : " (default)")
              << " - recompute the matrix for it to take effect" << std::endl;
}

double llama_mas_get_alignment_ridge(const llama_mas_context * mas) {
    return mas ? mas_alignment_effective_ridge(mas) : 1e-6;
}

void llama_mas_set_alignment_rescale_norm(llama_mas_context * mas, bool on) {
    if (!mas) return;
    mas->alignment_rescale_norm = on;
    std::cerr << "[MAS/alignment] norm rescale " << (on ? "enabled" : "disabled")
              << (mas->alignment_has_target_norm
                  ? " (target_norm=" + std::to_string(mas->alignment_target_norm) + ")"
                  : " (no target_norm yet - compute the matrix first)")
              << std::endl;
}

bool llama_mas_get_alignment_rescale_norm(const llama_mas_context * mas) {
    return mas ? mas->alignment_rescale_norm : false;
}

double llama_mas_get_alignment_target_norm(const llama_mas_context * mas) {
    return mas ? mas->alignment_target_norm : 0.0;
}

int llama_mas_get_latent_stats(const llama_mas_context * mas,
                               struct llama_mas_latent_stats * out) {
    if (!mas || !out) return -1;
    *out = mas->latent_stats;
    return 0;
}

void llama_mas_reset_latent_stats(llama_mas_context * mas) {
    if (!mas) return;
    mas->latent_stats = llama_mas_latent_stats{};
    std::cerr << "[MAS/latent] stats reset" << std::endl;
}

void llama_mas_set_use_alignment(llama_mas_context * mas, bool use_alignment) {
    if (!mas) return;
    mas->use_alignment_matrix = use_alignment;
    std::cerr << "[MAS/alignment] usage " << (use_alignment ? "enabled" : "disabled") << std::endl;
}

bool llama_mas_get_use_alignment(llama_mas_context * mas) {
    return mas && mas->use_alignment_matrix;
}

int32_t llama_mas_get_alignment_n_embd(llama_mas_context * mas) {
    return mas ? mas->alignment_n_embd : 0;
}

bool llama_mas_alignment_is_identity(llama_mas_context * mas) {
    return mas ? mas->alignment_is_identity : true;
}

// ---------------------------------------------------------------------------
// Hidden state access
// ---------------------------------------------------------------------------
float * llama_mas_get_agent_hidden_state(llama_mas_context * mas, int32_t agent_idx, int32_t * n_embd) {
    if (!mas || !mas->ctx || !n_embd) return nullptr;
    if (agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return nullptr;
    float * state = llama_get_hidden_states_last_token(mas->ctx);
    *n_embd = state ? llama_model_n_embd(mas->model) : 0;
    return state;
}

void * llama_mas_agent_get_hidden_state_tensor(llama_mas_context * mas, int32_t agent_idx, int32_t layer) {
    if (!mas || !mas->ctx) return nullptr;
    if (agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return nullptr;
    if (layer < 0) {
        int32_t n_layers = llama_get_hidden_states_layer_count(mas->ctx);
        layer = n_layers - 1;
    }
    return llama_get_hidden_state_tensor(mas->ctx, layer);
}


// ---------------------------------------------------------------------------
// Decode step (non-batched)
// ---------------------------------------------------------------------------
bool llama_mas_all_finished(const llama_mas_context * mas) {
    if (!mas || mas->agents.empty()) return true;
    for (const auto & agent : mas->agents) {
        if (!agent->finished) return false;
    }
    return true;
}

static // Log p(token | context) by direct log-softmax over the logits row.
//
// Computed from the RAW logits instead of the sampler because it must be the
// model's probability of the token it actually emitted, for greedy and sampled
// decoding alike. The max-subtraction is the standard stability guard.
double llama_mas_logprob_of(const float * logits, int64_t n_vocab, llama_token tok) {
    if (!logits || tok < 0 || (int64_t) tok >= n_vocab) return 0.0;
    float mx = logits[0];
    for (int64_t i = 1; i < n_vocab; ++i) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (int64_t i = 0; i < n_vocab; ++i) sum += std::exp((double)(logits[i] - mx));
    double lse = (double) mx + std::log(sum);
    return (double) logits[tok] - lse;
}

int llama_mas_decode_step_batched(llama_mas_context * mas) {
    if (!mas || !mas->ctx) return -1;
    const llama_vocab * vocab = llama_model_get_vocab(mas->model);

    std::vector<int> active;
    llama_layer_graph_context * common_layer_graph = nullptr;
    bool layer_graphs_compatible = true;
    for (int i = 0; i < (int) mas->agents.size(); ++i) {
        auto & agent = mas->agents[i];
        if (agent->finished) continue;
        llama_pos pos = (llama_pos)(agent->prompt_pos + agent->generated_len - 1);
        if (pos < 0) continue;
        if (active.empty()) {
            common_layer_graph = agent->layer_graph;
        } else if (agent->layer_graph != common_layer_graph) {
            layer_graphs_compatible = false;
            break;
        }
        active.push_back(i);
    }

    if (active.empty()) return 0;

    if (!layer_graphs_compatible) {
        // Agents with different layer graphs cannot share one decode.
        for (int idx : active) {
            auto & agent = mas->agents[idx];
            llama_pos pos = (llama_pos)(agent->prompt_pos + agent->generated_len - 1);
            llama_context_set_layer_graph(mas->ctx, agent->layer_graph);
            llama_batch batch = llama_batch_init(1, 0, 1);
            batch.token[0] = agent->generated_tokens.empty() ? 0 : agent->generated_tokens.back();
            batch.pos[0] = pos;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = agent->seq_id;
            batch.logits[0] = 1;
            batch.n_tokens = 1;
            int ret = llama_decode(mas->ctx, batch);
            llama_batch_free(batch);
            if (ret != 0) return ret;
            float * logits = llama_get_logits_ith(mas->ctx, 0);
            if (!logits) continue;
            llama_sampler * smpl = mas->samplers[idx];
            llama_token next_token = llama_sampler_sample(smpl, mas->ctx, -1);
            llama_sampler_accept(smpl, next_token);
            agent->logprob_sum += llama_mas_logprob_of(logits, (int64_t) llama_vocab_n_tokens(vocab), next_token);
            agent->logprob_n++;
            agent->generated_tokens.push_back(next_token);
            agent->generated_len++;
            if (mas->stream_cb) mas->stream_cb(idx, next_token, mas->stream_step++, mas->stream_user_data);
            if (llama_vocab_is_eog(vocab, next_token) || (int32_t) agent->generated_len >= agent->max_tokens) {
                agent->finished = true;
            }
        }
        return 0;
    }

    llama_context_set_layer_graph(mas->ctx, common_layer_graph);

    const llama_mas_adapter_weights * common_adapter = find_common_adapter(active, mas->agents);
    if (mas->use_cuda_adapters && common_adapter) {
        mas->ctx->set_mas_adapter(common_adapter);
    } else {
        mas->ctx->clear_mas_adapter();
    }

    const int n_tokens = (int) active.size();
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int b = 0; b < n_tokens; ++b) {
        auto & agent = mas->agents[active[b]];
        llama_pos pos = (llama_pos)(agent->prompt_pos + agent->generated_len - 1);
        batch.token[b] = agent->generated_tokens.empty() ? 0 : agent->generated_tokens.back();
        batch.pos[b] = pos;
        batch.n_seq_id[b] = 1;
        batch.seq_id[b][0] = agent->seq_id;
        batch.logits[b] = 1;
    }
    batch.n_tokens = n_tokens;
    int ret = llama_decode(mas->ctx, batch);
    llama_batch_free(batch);
    if (ret != 0) return ret;

    for (int b = 0; b < n_tokens; ++b) {
        auto & agent = mas->agents[active[b]];
        float * logits = llama_get_logits_ith(mas->ctx, b);
        if (!logits) continue;
        llama_sampler * smpl = mas->samplers[active[b]];
        llama_token next_token = llama_sampler_sample(smpl, mas->ctx, b);
        llama_sampler_accept(smpl, next_token);
        // Perplexity: score the emitted token against the model's own distribution.
        agent->logprob_sum += llama_mas_logprob_of(logits, (int64_t) llama_vocab_n_tokens(vocab), next_token);
        agent->logprob_n++;
        agent->generated_tokens.push_back(next_token);
        agent->generated_len++;
        if (mas->stream_cb) mas->stream_cb(active[b], next_token, mas->stream_step++, mas->stream_user_data);
        if (llama_vocab_is_eog(vocab, next_token) || (int32_t) agent->generated_len >= agent->max_tokens) {
            agent->finished = true;
        }
    }
    return 0;
}

int llama_mas_decode_step(llama_mas_context * mas) {
    if (!mas || !mas->ctx) return -1;
    if (mas->use_batched_decode) {
        return llama_mas_decode_step_batched(mas);
    }

    const llama_vocab * vocab = llama_model_get_vocab(mas->model);

    // Latent branch: hidden-state feedback. No sampling, no token ids.
    bool has_latent = false;
    for (const auto & agent : mas->agents) {
        if (!agent->finished && agent->is_latent && agent->latent_steps > 0) {
            has_latent = true;
            break;
        }
    }

    if (has_latent) {
        const int32_t n_embd = llama_model_n_embd(mas->model);

        for (int i = 0; i < (int) mas->agents.size(); ++i) {
            auto & agent = mas->agents[i];
            if (agent->finished || !agent->is_latent || agent->latent_steps <= 0) continue;

            std::vector<float> h((size_t) n_embd);
            llama_token seed = agent->generated_tokens.empty() ? 0 : agent->generated_tokens.back();

            llama_set_output_hidden_states(mas->ctx, true);

            for (int32_t s = 0; s < agent->latent_steps; ++s) {
                // Position comes from the MEMORY MODULE, not from the
                // prompt_pos/generated_len counters. Two bugs lived here:
                //
                //  1. `pos` was computed once from the counters and never
                //     advanced, so every step of the span re-wrote the SAME
                //     KV position. A latent span is supposed to append
                //     latent_steps positions; writing one cell N times made
                //     the tail lag the counter by N-1.
                //  2. those counters can be stale on entry (a previous wave
                //     or a resumed session moves them), so the first step
                //     could start at a position the KV had already passed.
                //     Measured: prompt 6 tokens (tail X = 5) but the batch
                //     asked for Y = 7 -> llama_decode hard-fails with
                //     "sequence positions remain consecutive: Y = X + 1".
                //
                // Deriving from seq_pos_max is the same rule llama_mas_agent_pos
                // uses for the turn path, and it is the only authority the batch
                // validator accepts.
                llama_memory_t mem_lat = llama_get_memory(mas->ctx);
                llama_pos kv_tail = mem_lat ? llama_memory_seq_pos_max(mem_lat, agent->seq_id) : -1;
                const llama_pos pos = (llama_pos)(kv_tail + 1);

                if (s == 0) {
                    llama_batch batch = llama_batch_init(1, 0, 1);
                    batch.token[0] = seed;
                    batch.pos[0] = pos;
                    batch.n_seq_id[0] = 1;
                    batch.seq_id[0][0] = agent->seq_id;
                    batch.logits[0] = 1; // n_outputs=0 would make the getter return NULL
                    batch.n_tokens = 1;
                    llama_context_set_layer_graph(mas->ctx, agent->layer_graph);
                    int ret = llama_decode(mas->ctx, batch);
                    llama_batch_free(batch);
                    if (ret != 0) return ret;
                } else {
                    std::vector<float> aligned((size_t) n_embd);
                    if (mas->use_alignment_matrix) {
                        llama_mas_apply_alignment(mas, h.data(), aligned.data(), n_embd);
                    } else {
                        std::memcpy(aligned.data(), h.data(), (size_t) n_embd * sizeof(float));
                    }
                    llama_batch batch = llama_batch_init(1, n_embd, 1);
                    std::memcpy(&batch.embd[0], aligned.data(), (size_t) n_embd * sizeof(float));
                    batch.pos[0] = pos;
                    batch.n_seq_id[0] = 1;
                    batch.seq_id[0][0] = agent->seq_id;
                    batch.logits[0] = 1; // this position's hidden state is the output
                    batch.n_tokens = 1;
                    llama_context_set_layer_graph(mas->ctx, agent->layer_graph);
                    int ret = llama_decode(mas->ctx, batch);
                    llama_batch_free(batch);
                    if (ret != 0) return ret;
                }

                int32_t got = 0;
                float * hs = llama_mas_get_agent_hidden_state(mas, i, &got);
                if (!hs || got != n_embd) {
                    std::cerr << "[MAS/latent] no hidden state at step " << (s + 1) << std::endl;
                    llama_set_output_hidden_states(mas->ctx, false);
                    return -1;
                }
                std::memcpy(h.data(), hs, (size_t) n_embd * sizeof(float));

                agent->generated_len++;
                if (agent->latent_steps_remaining > 0) agent->latent_steps_remaining--;
            }

            llama_set_output_hidden_states(mas->ctx, false);

            if (agent->latent_steps_remaining <= 0) {
                agent->is_latent = false; // latent phase done -> text
            }
        }
        return 0;
    }

    // Standard text generation with sampling.
    for (int i = 0; i < (int) mas->agents.size(); ++i) {
        auto & agent = mas->agents[i];
        if (agent->finished) continue;
        llama_pos pos = (llama_pos)(agent->prompt_pos + agent->generated_len - 1);
        if (pos < 0) continue;

        llama_context_set_layer_graph(mas->ctx, agent->layer_graph);
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0] = agent->generated_tokens.empty() ? 0 : agent->generated_tokens.back();
        batch.pos[0] = pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = agent->seq_id;
        batch.logits[0] = 1;
        batch.n_tokens = 1;
        int ret = llama_decode(mas->ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) return ret;

        float * logits = llama_get_logits_ith(mas->ctx, 0);
        if (!logits) continue;

        llama_sampler * smpl = mas->samplers[i];
        llama_token next_token = llama_sampler_sample(smpl, mas->ctx, -1);
        llama_sampler_accept(smpl, next_token);
        agent->logprob_sum += llama_mas_logprob_of(logits, (int64_t) llama_vocab_n_tokens(vocab), next_token);
        agent->logprob_n++;
        agent->generated_tokens.push_back(next_token);
        agent->generated_len++;

        if (mas->stream_cb) {
            mas->stream_cb(i, next_token, mas->stream_step++, mas->stream_user_data);
        }
        if (llama_vocab_is_eog(vocab, next_token) ||
            (int32_t) agent->generated_len >= agent->max_tokens) {
            agent->finished = true;
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// KV clone helpers
// ---------------------------------------------------------------------------
// Rebuild one agent's sampler with a repetition penalty.
//
// The penalty sampler must be built from the agent's CURRENT config, so this
// rewrites the fields and re-creates the chain. The old chain is deleted because
// llama_sampler_sample reads the chain's internal cursor buffer; leaving it alive
// would keep sampling with the pre-change parameters.
int llama_mas_set_repeat_penalty(llama_mas_context * mas, int32_t agent_idx,
                                 float repeat_penalty, int32_t repeat_last_n,
                                 float penalty_freq, float penalty_present) {
    if (!mas || agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return -1;
    auto & agent = mas->agents[agent_idx];
    agent->repeat_penalty = repeat_penalty > 1.0f ? repeat_penalty : 1.1f;
    agent->repeat_last_n  = repeat_last_n > 0 ? repeat_last_n : 64;
    agent->penalty_freq    = penalty_freq;
    agent->penalty_present = penalty_present;

    llama_sampler * smpl = create_agent_sampler(agent.get(), mas->use_backend_sampling);
    if (!smpl) return -1;
    llama_sampler * old = mas->samplers[agent_idx];
    mas->samplers[agent_idx] = smpl;
    if (old) llama_sampler_free(old);
    if (mas->use_backend_sampling) {
        llama_set_sampler(mas->ctx, (llama_seq_id) agent->seq_id, smpl);
    }
    return 0;
}

int llama_mas_kv_clone(llama_mas_context * mas, int32_t src_agent_idx, int32_t dst_agent_idx) {
    if (!mas || !mas->ctx) return -1;
    if (src_agent_idx < 0 || src_agent_idx >= (int32_t) mas->agents.size()) return -1;
    if (dst_agent_idx < 0 || dst_agent_idx >= (int32_t) mas->agents.size()) return -1;
    llama_memory_t mem = llama_get_memory(mas->ctx);
    if (!mem) return -1;
    auto & src = mas->agents[src_agent_idx];
    auto & dst = mas->agents[dst_agent_idx];
    llama_memory_seq_cp(mem, src->seq_id, dst->seq_id, 0, -1);
    return 0;
}

int llama_mas_kv_clone_range(llama_mas_context * mas, int32_t src_agent_idx, int32_t dst_agent_idx,
                             int32_t src_pos_start, int32_t src_pos_end, int32_t dst_pos_start) {
    if (!mas || !mas->ctx) return -1;
    if (src_agent_idx < 0 || src_agent_idx >= (int32_t) mas->agents.size()) return -1;
    if (dst_agent_idx < 0 || dst_agent_idx >= (int32_t) mas->agents.size()) return -1;
    if (src_pos_start < 0 || src_pos_end <= src_pos_start) return -1;

    llama_memory_t mem = llama_get_memory(mas->ctx);
    if (!mem) return -1;

    auto & src = mas->agents[src_agent_idx];
    auto & dst = mas->agents[dst_agent_idx];

    // Full copy + trim: ISWA caches reject partial seq_cp.
    llama_memory_seq_cp(mem, src->seq_id, dst->seq_id, -1, -1);
    llama_memory_seq_rm(mem, dst->seq_id, src_pos_end, -1);
    if (src_pos_start > 0) {
        llama_memory_seq_rm(mem, dst->seq_id, -1, src_pos_start);
    }
    if (dst_pos_start != src_pos_start) {
        llama_pos delta = dst_pos_start - src_pos_start;
        llama_memory_seq_add(mem, dst->seq_id, dst_pos_start,
                             dst_pos_start + (src_pos_end - src_pos_start), delta);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Alignment matrix Wa = W_in * W_out^†  (n_embd x n_embd)
//
// A last-layer hidden state lives in the residual stream; the input-embedding
// path expects a point in the token-embedding column space. Feeding one into the
// other is out-of-distribution. Wa is the least-squares map between them.
//
//   1. G       = W_out^T W_out + lambda*I      (n_embd x n_embd)
//   2. Ginv    = G^-1                          (Gauss-Jordan, partial pivoting)
//   3. A       = W_in W_out^T                  (n_embd x n_embd)
//   4. Wa      = A Ginv
//
// Earlier versions sampled 100 vocab rows (rank-deficient for n_embd > 100) and
// fell back to the identity for n_embd > 512 - i.e. on every real model, which
// made the whole mechanism a silent no-op. Both are gone.
// ---------------------------------------------------------------------------
int llama_mas_compute_alignment_matrix(llama_mas_context * mas) {
    if (!mas || !mas->model) return -1;

    const int32_t n_embd = llama_model_n_embd(mas->model);
    const llama_vocab * vocab = llama_model_get_vocab(mas->model);
    const int32_t n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;

    if (n_embd <= 0 || n_vocab <= 0) {
        std::cerr << "[MAS/alignment] bad dims n_embd=" << n_embd
                  << " n_vocab=" << n_vocab << std::endl;
        return -1;
    }

    // ── Why the vocabulary is subsampled ────────────────────────────────────
    // Exact cost is n_vocab * n_embd^2 MACs per matrix: at n_vocab=151936 and
    // n_embd=2048 that is 6.4e11, which took 2000+ CPU-seconds and still had not
    // finished even spread across 8 threads (measured). The Gram matrix is a sum
    // over vocab rows whose contributions fall off steeply with token frequency,
    // so a stride through the vocabulary keeps the dominant terms at a small
    // fraction of the cost. Rows are read sparsely, not skipped entirely.
    int32_t stride = mas->alignment_vocab_stride;
    if (stride < 1) stride = 1;
    // `alignment_full_vocab` forces the exact solve. The strided solve is a
    // speed hack, so an exact result must be reachable for A/B work; without
    // this the only way to get stride=1 was to set the stride field itself.
    if (mas->alignment_full_vocab) {
        if (stride != 1) {
            std::cerr << "[MAS/alignment] full_vocab=1 -> forcing stride 1 (was "
                      << stride << ")" << std::endl;
        }
        stride = 1;
    }
    if (stride > 1) {
        std::cerr << "[MAS/alignment] vocab stride=" << stride
                  << " -> ~" << ((n_vocab + stride - 1) / stride)
                  << " of " << n_vocab << " rows" << std::endl;
    }

    auto use_identity = [&](const char * why) {
        std::cerr << "[MAS/alignment] " << why << " -> identity" << std::endl;
        mas->alignment_matrix.assign((size_t) n_embd * n_embd, 0.0f);
        for (int32_t i = 0; i < n_embd; ++i) {
            mas->alignment_matrix[(size_t) i * n_embd + i] = 1.0f;
        }
        mas->alignment_n_embd = n_embd;
        mas->alignment_is_identity = true;
        mas->use_alignment_matrix = true;
        return 0;
    };

    // Non-const because the ggml helpers used below (and any future backend /
    // scheduler lookup) take non-const tensors. The model's get_tensor() hands
    // back const; nothing here mutates the tensor, so the const_cast is safe.
    ggml_tensor * tok_embd = const_cast<ggml_tensor *>(mas->model->get_tensor("token_embd.weight"));
    ggml_tensor * output   = const_cast<ggml_tensor *>(mas->model->get_tensor("output.weight"));

    if (!output) {
        output = tok_embd; // tied embeddings
        std::cerr << "[MAS/alignment] tied embeddings (no output.weight)" << std::endl;
    }
    if (!tok_embd || !output) {
        return use_identity("embedding tensors not found");
    }
    if (ggml_nelements(tok_embd) < (int64_t) n_vocab * n_embd ||
        ggml_nelements(output)   < (int64_t) n_vocab * n_embd) {
        return use_identity("embedding tensor smaller than n_vocab*n_embd");
    }

    // Dequantize ROW BY ROW, not the whole tensor in one call.
    //
    // This is the fix for an access violation (0xC0000005) seen on Qwen3-8B
    // Q4_K_M. The original code did:
    //     const int64_t n = ggml_nelements(t);
    //     std::vector<float> r(n);
    //     traits->to_float(t->data, r.data(), n);
    // which is fine for Q8_0 (type_size 34 / block 32) but breaks for
    // block-quantized types like Q4_K (type_size 144 / block 256): ggml_nelements
    // returns the LOGICAL element count (n_vocab * n_embd = 622M for Qwen3-8B),
    // while the tensor's actual data buffer holds the PACKED blocks (~350 MB).
    // Passing the logical count makes to_float walk off the end of the packed
    // buffer and write past the destination vector. Row-wise dequant respects the
    // per-row block structure and the true buffer size.
    // ── Host-residency requirement ──────────────────────────────────────────
    // Dequantising reads t->data directly, so the tensor MUST be host-resident.
    // When a model is fully offloaded (--n-gpu-layers 99) the embedding tensors
    // live in VRAM, t->data is a DEVICE pointer, and dereferencing it on the
    // host is an access violation (0xC0000005). This is what killed alignment on
    // Qwen3-8B while Qwen3-1.7B happened to survive.
    //
    // This routine does NOT yet copy a device tensor back to the host: there is
    // no backend/scheduler lookup here. A correct device->host copy needs the
    // tensor's owning backend (e.g. ggml_backend_sched_get_tensor_backend on a
    // non-const tensor) followed by ggml_backend_tensor_get, and that is the
    // intended fix for the 8B case. Until that is wired in, running alignment on
    // a fully-offloaded model will fault; compute the alignment on a
    // host-resident model (no --n-gpu-layers 99, or a CPU split) instead. The
    // const_cast above keeps the tensor non-const so such a lookup can take it.
    // Dequantise a whole tensor with ggml's own to_float.
    //
    // Two failures were fixed to get here, both silent or fatal:
    //
    // 1. ACCESS VIOLATION (0xC0000005) on Qwen3-8B. The original code assumed
    //    ggml_row_size(type, row_elements) * n_rows == ggml_nbytes(tensor) and
    //    walked rows by that stride. For output.weight (type 14) the real packed
    //    size is 3360 B/row while ggml_row_size predicts 4352 B/row, so the walk
    //    ran off the end of the buffer. Deriving strides by hand is fragile
    //    across quantisation types; ggml's to_float already knows the layout, so
    //    let it do the work over the full element count.
    //
    // 2. SILENT GARBAGE from a failed destination allocation. A 622M-float
    //    vector is 2.5 GB; if the allocation fails, r is empty and to_float
    //    writes through a null pointer. We now check the size and refuse.
    auto dequantize = [&](ggml_tensor * t) -> std::vector<float> {
        if (!t || !t->data) return {};

        const int64_t n = ggml_nelements(t);
        if (n <= 0) return {};

        std::vector<float> r((size_t) n);
        if (r.size() < (size_t) n) {
            std::cerr << "[MAS/alignment] cannot allocate " << n
                      << " floats - alignment disabled" << std::endl;
            return {};
        }

        if (t->type == GGML_TYPE_F32) {
            const size_t bytes = (size_t) n * sizeof(float);
            const size_t have = (size_t) ggml_nbytes(t);
            std::memcpy(r.data(), t->data, bytes < have ? bytes : have);
            std::cerr << "[MAS/alignment] copied f32 tensor n=" << n << std::endl;
            return r;
        }

        const ggml_type_traits * traits = ggml_get_type_traits(t->type);
        if (!traits || !traits->to_float) {
            std::cerr << "[MAS/alignment] no dequant for type " << t->type << std::endl;
            return {};
        }

        // HOST STAGING. ggml's to_float dereferences its source on the CPU, so a
        // VRAM-resident tensor must be copied down first or this is an instant
        // access violation. Model WEIGHT tensors are not in the context
        // scheduler's assignment, so ggml_backend_sched_get_tensor_backend()
        // returns null for them (that is why an earlier attempt logged
        // "backend=?" and skipped the copy). The reliable owner is t->buffer,
        // and ggml_backend_buffer_is_host() answers the question directly.
        //
        // This matters per-tensor, not per-model: on Qwen3-8B the load log says
        // token_embd.weight "cannot be used with preferred buffer type
        // CUDA_Host, using CPU instead" (CPU-resident), while output.weight is a
        // separate tensor that stays on the GPU. So one dequant was legal and the
        // other crashed - which is exactly the observed type-12-then-type-14
        // failure order.
        const void * src = t->data;
        std::vector<uint8_t> host_copy;
        if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
            const size_t nb = (size_t) ggml_nbytes(t);
            host_copy.resize(nb);
            ggml_backend_tensor_get(t, host_copy.data(), 0, nb);
            src = host_copy.data();
            std::cerr << "[MAS/alignment] staged " << (nb / 1048576.0)
                      << " MB from device buffer '"
                      << (ggml_backend_buffer_name(t->buffer) ? ggml_backend_buffer_name(t->buffer) : "?")
                      << "' to host before dequant" << std::endl;
        }

        std::cerr << "[MAS/alignment] dequant type=" << t->type
                  << " nelements=" << n << " nbytes=" << ggml_nbytes(t)
                  << " (" << (ggml_nbytes(t) / 1048576.0) << " MB packed -> "
                  << (n * 4.0 / 1048576.0) << " MB f32)"
                  << " host=" << (host_copy.empty() ? "direct" : "staged") << std::endl;

        // One call over the whole tensor: ggml handles the block layout.
        // Hand-rolled row strides are WRONG for some types - ggml_row_size(14,
        // 4096) predicts 4352 B/row while the real packed row is 3360 B/row, so
        // walking rows by that stride ran off the buffer (the original 0xC0000005).
        traits->to_float(src, r.data(), n);
        return r;
    };

    // NOTE on memory: for Qwen3-8B these two vectors are 151936 x 4096 floats =
    // 2.5 GB EACH, and the Gram matrix is another 4096^2 floats per thread part.
    // Dequantizing both and holding them while the threaded parts are alive is
    // what pushed the 8B over the edge (access violation from a failed
    // allocation, not a logic error). They are dequantized once, used, and freed
    // before the inversion. The caller can also reduce peak by lowering
    // n_threads, since parts is n_threads x n_embd^2.
    // PEAK MEMORY: each embedding copy is n_vocab * n_embd floats - 2.37 GB on
    // Qwen3-8B. Holding both while the model is resident (4.7 GB) exhausts the
    // allocator and crashes inside to_float, so they are dequantized ONE AT A
    // TIME: W_out for the Gram pass, then freed; then W_in for the cross pass.
    std::vector<float> w_out;
    std::vector<float> w_in;
    {
        w_out = dequantize(output);
        if (w_out.empty()) {
            return use_identity("output dequantization failed");
        }
    }

    const size_t n = (size_t) n_embd;
    const float * W_out = w_out.data();

    // ── Multithreaded. ──────────────────────────────────────────────────────
    // These passes were scalar single-threaded triple loops over
    // n_vocab x n_embd^2 (~6.4e11 MACs at n_embd=2048), which is minutes of
    // wall clock. That is the whole reason alignment looked "unusable" on a
    // real model.
    //
    // Threading rule here: threads are partitioned by OUTPUT row, never by the
    // reduction input. Each thread accumulates into its own private G/A block
    // and the blocks are summed at the end. Splitting the vocab loop instead
    // (the obvious first attempt) is a data race - many threads would do
    // `G[a*n+b] += ...` into the same row.
    const unsigned hw = std::thread::hardware_concurrency();
    int n_threads = (int) (hw == 0 ? 4u : (hw > 8u ? 8u : hw));
    // Peak-memory guard for the threaded parts. Each thread owns a private
    // n_embd^2 float block (parts), so the cost is n_threads * n_embd^2 * 4
    // bytes. At n_embd=2048 and 8 threads that is 128 MB - fine. At n_embd=4096
    // (Qwen3-8B) it is 512 MB, on top of the two dequantized embedding copies
    // (2.5 GB each) that are live at the same moment. That combination is what
    // produced an access violation on the 8B: the allocation failed and the
    // code wrote through a null/garbage pointer. Cap the parts memory at ~256 MB
    // by reducing threads when the model is wide.
    {
        const double bytes_per_thread = (double) n_embd * (double) n_embd * 4.0;
        const double budget = 256.0 * 1024.0 * 1024.0;  // 256 MB for parts
        int allowed = (int) (budget / (bytes_per_thread > 0 ? bytes_per_thread : 1));
        if (allowed < 1) allowed = 1;
        if (allowed < n_threads) {
            std::cerr << "[MAS/alignment] capping threads " << n_threads << " -> "
                      << allowed << " to bound parts memory (n_embd=" << n_embd
                      << ", " << (bytes_per_thread / (1024.0 * 1024.0))
                      << " MB per thread)" << std::endl;
            n_threads = allowed;
        }
    }

    // G = sum_v outer(W_out[v], W_out[v])   (output rows owned per thread)
    std::vector<float> G(n * n, 0.0f);
    {
        std::vector<std::vector<float>> parts((size_t) n_threads,
                                              std::vector<float>(n * n, 0.0f));
        std::atomic<int32_t> next_row{0};
        std::vector<std::thread> pool;
        pool.reserve((size_t) n_threads);
        for (int t = 0; t < n_threads; ++t) {
            pool.emplace_back([&, t] {
                std::vector<float> & acc = parts[(size_t) t];
                for (;;) {
                    const int32_t a = next_row.fetch_add(1, std::memory_order_relaxed);
                    if (a >= n_embd) break;
                    float * g = &acc[(size_t) a * n];
                    // This row is private to this thread, so the inner b loop
                    // can accumulate freely.
                    for (int32_t v = 0; v < n_vocab; v += stride) {
                        const float ra = W_out[(size_t) v * n + a];
                        if (ra == 0.0f) continue;
                        const float * row = W_out + (size_t) v * n;
                        // `alignment_dtype` (A3): 1 = round each product to
                        // bf16 before accumulating. bf16 keeps ~8 mantissa bits,
                        // so this is a MEASUREMENT knob for how much precision
                        // the solve actually needs - not an optimisation. The
                        // rounding is done through the float bit pattern rather
                        // than _cast_to_bfloat16 so it stays exact and portable.
                        if (mas->alignment_dtype == 1) {
                            for (int32_t b = 0; b < n_embd; ++b) {
                                float prod = ra * row[b];
                                uint32_t bits = 0;
                                std::memcpy(&bits, &prod, sizeof(bits));
                                // keep the high 16 bits, round-to-nearest-even
                                const uint32_t lsb = (bits >> 16) & 1u;
                                bits += 0x7FFFu + lsb;
                                bits &= 0xFFFF0000u;
                                float rounded = 0.0f;
                                std::memcpy(&rounded, &bits, sizeof(rounded));
                                g[b] += rounded;
                            }
                        } else {
                            for (int32_t b = 0; b < n_embd; ++b) g[b] += ra * row[b];
                        }
                    }
                }
            });
        }
        for (auto & th : pool) th.join();
        for (int32_t a = 0; a < n_embd; ++a) {
            float * g = &G[(size_t) a * n];
            for (int t = 0; t < n_threads; ++t) {
                const float * src = &parts[(size_t) t][(size_t) a * n];
                for (int32_t b = 0; b < n_embd; ++b) g[b] += src[b];
            }
        }
    }

    // Ridge: G is singular when n_vocab < n_embd. Scale to the mean diagonal so
    // the term is invariant to embedding magnitude.
    double diag_mean = 0.0;
    for (int32_t a = 0; a < n_embd; ++a) diag_mean += G[(size_t) a * n + a];
    diag_mean /= (double) n_embd;
    const float lambda = (diag_mean > 0.0)
                       ? (float) (diag_mean * mas_alignment_effective_ridge(mas))
                       : (float) mas_alignment_effective_ridge(mas);
    for (int32_t a = 0; a < n_embd; ++a) G[(size_t) a * n + a] += lambda;

    // ── `alignment_lowrank` (A2) ────────────────────────────────────────────
    // Truncate the solve to the leading `alignment_lowrank` rows/cols of the
    // Gram system. The map Wa = Ginv * A is dominated by the high-variance
    // directions of the embedding, so a rank-r truncation approximates the full
    // solve at r^2/r_full^2 of the inversion cost (the Gram/cross passes are
    // unaffected - they still see every vocab row). r <= 0 or r >= n_embd means
    // the full solve. The truncated solution is written into the leading r
    // entries of each row and the rest of the row is left as the identity's
    // contribution, which keeps the map well-defined rather than zeroing it.
    const int32_t lowrank = mas->alignment_lowrank;
    const int32_t solve_dim = (lowrank > 0 && lowrank < n_embd) ? lowrank : n_embd;
    if (solve_dim < n_embd) {
        std::cerr << "[MAS/alignment] lowrank=" << lowrank
                  << " -> solving leading " << solve_dim << " of " << n_embd
                  << " dims" << std::endl;
    }

    // Ginv via Gauss-Jordan with partial pivoting.
    std::vector<float> Ginv(n * n, 0.0f);
    for (int32_t a = 0; a < n_embd; ++a) Ginv[(size_t) a * n + a] = 1.0f;

    for (int32_t col = 0; col < solve_dim; ++col) {
        int32_t pivot = col;
        float best = std::fabs(G[(size_t) col * n + col]);
        for (int32_t r = col + 1; r < solve_dim; ++r) {
            const float cand = std::fabs(G[(size_t) r * n + col]);
            if (cand > best) { best = cand; pivot = r; }
        }
        if (best < 1e-12f) {
            return use_identity("Gram matrix singular");
        }
        if (pivot != col) {
            for (int32_t k = 0; k < n_embd; ++k) {
                std::swap(G[(size_t) col * n + k],    G[(size_t) pivot * n + k]);
                std::swap(Ginv[(size_t) col * n + k], Ginv[(size_t) pivot * n + k]);
            }
        }
        const float inv_p = 1.0f / G[(size_t) col * n + col];
        for (int32_t k = 0; k < n_embd; ++k) {
            G[(size_t) col * n + k]    *= inv_p;
            Ginv[(size_t) col * n + k] *= inv_p;
        }
        const float * grow = &G[(size_t) col * n];
        for (int32_t r = 0; r < n_embd; ++r) {
            if (r == col) continue;
            const float f = G[(size_t) r * n + col];
            if (f == 0.0f) continue;
            float * gr  = &G[(size_t) r * n];
            float * gir = &Ginv[(size_t) r * n];
            for (int32_t k = 0; k < n_embd; ++k) {
                gr[k]  -= f * grow[k];
                gir[k] -= f * Ginv[(size_t) col * n + k];
            }
        }
    }
    std::cerr << "[MAS/alignment] inverted Gram (n_embd=" << n_embd
              << ", lambda=" << lambda << ")" << std::endl;

    // G (the un-inverted Gram) is DEAD after the Gauss-Jordan inversion above:
    // only Ginv is referenced below (Wa = A * Ginv). On Qwen3-8B G is another
    // n_embd^2 floats (67 MB at 4096) that would otherwise stay resident through
    // the cross pass, where the peak working set is already at the edge. Freeing
    // it here drops the live heap before the fragile A allocation.
    {
        std::vector<float>().swap(G);
    }

    // ── Cross pass A = W_in * W_out^T ───────────────────────────────────────
    // W_in is NOT materialized as a second 2.37 GB vector. Doing so alongside
    // W_out and the resident model is what crashed: the allocator cannot satisfy
    // two simultaneous 2.37 GB float buffers. Instead each vocab row of W_in is
    // dequantized on demand into a small scratch buffer (n_embd floats = 16 KB),
    // used for that row, and discarded. Peak extra memory is one row, not one
    // whole embedding matrix.
    std::vector<float> A;
    std::vector<float> w_in_row;
    try {
        A.assign(n * n, 0.0f);
        w_in_row.assign((size_t) n_embd, 0.0f);
    } catch (const std::bad_alloc &) {
        std::cerr << "[MAS/alignment] cross pass: cannot allocate A ("
                  << (n * n * 4.0 / 1048576.0) << " MB) or row scratch - "
                  << "peak working set too large for the resident model"
                  << " -> identity" << std::endl;
        return use_identity("cross pass allocation failed");
    }
    if (A.size() < n * n) {
        std::cerr << "[MAS/alignment] cross pass: A allocation short ("
                  << A.size() << " < " << (n * n) << ") -> identity" << std::endl;
        return use_identity("cross pass A allocation short");
    }
    {
        const ggml_type_traits * in_traits = ggml_get_type_traits(tok_embd->type);
        // Row layout: these tensors are [n_vocab, n_embd] so each row is exactly
        // n_embd elements and rows are contiguous in the packed buffer at
        // ggml_row_size(type, n_embd) stride.
        const size_t in_row_bytes = ggml_row_size(tok_embd->type, (int64_t) n_embd);
        const size_t in_total = (size_t) ggml_nbytes(tok_embd);
        const uint8_t * in_base = (const uint8_t *) tok_embd->data;
        if (in_traits && in_traits->to_float && in_row_bytes > 0 &&
            in_row_bytes * (size_t) n_vocab <= in_total) {
            // Preflight the peak working set BEFORE the first row so an OOM or a
            // bad-pointer is attributed here rather than mid-loop (8B crash
            // point). A = n_embd^2 floats (64 MB at 4096); the model is resident.
            const double a_bytes   = (double) n * (double) n * 4.0;
            const double row_bytes = (double) n_embd * 4.0;
            const double wout_bytes = (double) n * (double) n_vocab * 4.0;
            std::cerr << "[MAS/alignment] cross pass preflight: A=" << (a_bytes/1048576.0)
                      << " MB, w_out=" << (wout_bytes/1048576.0)
                      << " MB, row scratch=" << (row_bytes/1024.0) << " KB, "
                      << (n_vocab/stride) << " rows x "
                      << (2.0*n*(double)n_embd) << " MACs"
                      << (mas->alignment_cross_progress ? "  [progress on]" : "") << std::endl;
            // Progress cadence: every ~5% of strided rows, only when enabled.
            int64_t next_mark = (int64_t)( (double)(n_vocab/stride) * 0.05 );
            if (next_mark < 1) next_mark = 1;
            for (int32_t v = 0; v < n_vocab; v += stride) {
                in_traits->to_float(in_base + (size_t) v * in_row_bytes,
                                    w_in_row.data(), (int64_t) n_embd);
                const float * wout = &w_out[(size_t) v * n];
                // accumulate into A row-blocks (parallelised below)
                for (int32_t i = 0; i < n_embd; ++i) {
                    const float vi = w_in_row[(size_t) i];
                    float * arow = &A[(size_t) i * n];
                    for (int32_t j = 0; j < n_embd; ++j) arow[j] += vi * wout[j];
                }
                if (mas->alignment_cross_progress) {
                    const int64_t done = v / stride;
                    if (done >= next_mark) {
                        double a_nonzero = 0.0;
                        for (size_t q = 0; q < (size_t) n * n; q += 4096) a_nonzero += A[q];
                        std::cerr << "[MAS/alignment] cross pass row " << v
                                  << "/" << n_vocab << " (" << done << "/"
                                  << (n_vocab/stride) << " strided) "
                                  << "A_probe_sum=" << a_nonzero << std::endl;
                        next_mark += (int64_t)( (double)(n_vocab/stride) * 0.05 );
                        if (next_mark < 1) next_mark = 1;
                    }
                }
            }
            std::cerr << "[MAS/alignment] cross pass via per-row W_in dequant ("
                      << "stride=" << stride << ", " << in_row_bytes
                      << " B/row)" << std::endl;
        } else {
            std::cerr << "[MAS/alignment] per-row W_in dequant unavailable - "
                      << "falling back to identity" << std::endl;
            return use_identity("W_in row layout unsupported");
        }
    }
    // W_out is dead after the cross pass.
    {
        std::vector<float>().swap(w_out);
    }


    // Wa = A * Ginv   (output rows are independent -> parallel, no accumulation)
    mas->alignment_matrix.assign(n * n, 0.0f);
    {
        std::atomic<int32_t> next_row{0};
        std::vector<std::thread> pool;
        pool.reserve((size_t) n_threads);
        for (int t = 0; t < n_threads; ++t) {
            pool.emplace_back([&] {
                for (;;) {
                    const int32_t i = next_row.fetch_add(1, std::memory_order_relaxed);
                    if (i >= n_embd) break;
                    const float * arow = &A[(size_t) i * n];
                    float * orow = &mas->alignment_matrix[(size_t) i * n];
                    for (int32_t k = 0; k < n_embd; ++k) {
                        float sum = 0.0f;
                        for (int32_t j = 0; j < n_embd; ++j) sum += arow[j] * Ginv[(size_t) j * n + k];
                        orow[k] = sum;
                    }
                }
            });
        }
        for (auto & th : pool) th.join();
    }

    std::cerr << "[MAS/alignment] n_threads=" << n_threads << std::endl;

    mas->alignment_n_embd = n_embd;
    mas->alignment_is_identity = false;
    mas->use_alignment_matrix = true;

    // target_norm = mean over vocab rows of ||W_in[v]||, exactly the quantity
    // LatentMAS uses to rescale each latent vector before feeding it back.
    // Computed here because this is where the dequantized input embedding is in
    // memory; recomputing it per latent step would mean re-dequantizing.
    {
        // target_norm is the mean L2 norm of the real token embedding rows.
        // W_in is not materialised (see the cross pass), so rows are dequantized
        // one at a time here too. Sampling every row is O(n_vocab * n_embd) which
        // is trivial next to the solve; stride is honoured to match the Gram pass
        // exactly, so the norm describes the same rows the matrix was built from.
        const ggml_type_traits * in_traits = ggml_get_type_traits(tok_embd->type);
        const size_t in_row_bytes = ggml_row_size(tok_embd->type, (int64_t) n_embd);
        const uint8_t * in_base = (const uint8_t *) tok_embd->data;
        std::vector<float> row_buf((size_t) n_embd);
        double sum_norm = 0.0;
        int32_t n_counted = 0;
        for (int32_t v = 0; v < n_vocab; v += stride) {
            if (!in_traits || !in_traits->to_float || in_row_bytes == 0) break;
            in_traits->to_float(in_base + (size_t) v * in_row_bytes,
                                row_buf.data(), (int64_t) n_embd);
            double acc = 0.0;
            for (int32_t j = 0; j < n_embd; ++j) acc += (double) row_buf[(size_t) j] * (double) row_buf[(size_t) j];
            sum_norm += std::sqrt(acc);
            ++n_counted;
        }
        mas->alignment_target_norm = n_counted > 0 ? sum_norm / (double) n_counted : 0.0;
        mas->alignment_has_target_norm = mas->alignment_target_norm > 0.0;
        std::cerr << "[MAS/alignment] target_norm=" << mas->alignment_target_norm
                  << " (mean ||W_in[v]|| over " << n_counted << " rows)"
                  << " rescale=" << (mas->alignment_rescale_norm ? "on" : "off")
                  << std::endl;
    }

    std::cerr << "[MAS/alignment] Wa=" << n_embd << "x" << n_embd
              << " from full vocab n_vocab=" << n_vocab << std::endl;

    // Both embedding buffers are released before return (w_out was freed after
    // the cross pass, w_in was never materialized).
    return 0;
}

// ---------------------------------------------------------------------------
// Alignment matrix disk cache (G)
//
// The solve is 200-380 s on Qwen3-1.7B Q8 and the matrix was per-process, so
// every server restart paid it again. Format is a tiny header + raw float32:
//
//   magic "MASALN2\0"   8 bytes
//   n_embd              int32
//   n_vocab             int32
//   stride              int32
//   ridge_bits          int32   (ridge as float bits - exact, no rounding)
//   key_len             int32
//   key                 key_len bytes (model description, may contain spaces)
//   target_norm         float64  (mean ||W_in[v]||; needed by rescale_norm)
//   Wa                  n_embd*n_embd float32
//
// A wrong magic, short read, or key mismatch is a MISS (return 1), never a
// wrong matrix: an incompatible matrix would silently corrupt every latent run.
// ---------------------------------------------------------------------------

static const char * const MAS_ALIGN_MAGIC = "MASALN2";

static std::string mas_alignment_key(const llama_mas_context * mas, int32_t n_embd,
                                     int32_t n_vocab, int32_t stride, float ridge) {
    char desc[256] = {0};
    llama_model_desc(mas->model, desc, sizeof(desc));
    uint32_t rb = 0;
    std::memcpy(&rb, &ridge, sizeof(rb));
    char buf[512];
    snprintf(buf, sizeof(buf), "%s|embd=%d|vocab=%d|stride=%d|ridge=%u",
             desc, n_embd, n_vocab, stride, rb);
    return std::string(buf);
}

int llama_mas_save_alignment_matrix(llama_mas_context * mas, const char * path) {
    if (!mas || !path || !path[0]) return -1;
    if (mas->alignment_matrix.empty()) {
        std::cerr << "[MAS/alignment] save: no matrix in memory" << std::endl;
        return -1;
    }
    const int32_t n_embd = mas->alignment_n_embd;
    if (n_embd <= 0) return -1;

    // ridge is a parameter of the solve; use the SAME helper compute uses so
    // the key always describes the matrix actually in memory.
    const float ridge = (float) mas_alignment_effective_ridge(mas);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mas->model));
    // Effective stride, not the configured one: with `alignment_full_vocab` the
    // solve runs at stride 1 while `alignment_vocab_stride` stays 16, so the
    // configured value would write a key that does not describe the matrix.
    const int32_t stride  = mas_alignment_effective_stride(mas);
    const std::string key = mas_alignment_key(mas, n_embd, n_vocab, stride, ridge);

    // Temp file + rename: an interrupted save must never leave a half-written
    // cache that a later load would happily trust.
    std::string tmp = std::string(path) + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) {
        std::cerr << "[MAS/alignment] save: cannot open " << tmp << std::endl;
        return -1;
    }
    bool ok = true;
    auto wr = [&](const void * p, size_t n) {
        if (ok && fwrite(p, 1, n, f) != n) ok = false;
    };
    wr(MAS_ALIGN_MAGIC, 8);
    wr(&n_embd, sizeof(n_embd));
    wr(&n_vocab, sizeof(n_vocab));
    wr(&stride, sizeof(stride));
    uint32_t rb = 0;
    std::memcpy(&rb, &ridge, sizeof(rb));
    wr(&rb, sizeof(rb));
    const int32_t key_len = (int32_t) key.size();
    wr(&key_len, sizeof(key_len));
    wr(key.data(), (size_t) key_len);
    // target_norm is part of the cache: a loaded matrix with target_norm=0
    // makes rescale_norm silently a no-op, which is exactly the kind of quiet
    // behaviour change this cache must not introduce.
    const double tnorm = mas->alignment_target_norm;
    wr(&tnorm, sizeof(tnorm));
    wr(mas->alignment_matrix.data(), mas->alignment_matrix.size() * sizeof(float));
    if (fclose(f) != 0) ok = false;

    if (!ok) {
        std::cerr << "[MAS/alignment] save: write failed" << std::endl;
        remove(tmp.c_str());
        return -1;
    }
    if (rename(tmp.c_str(), path) != 0) {
        std::cerr << "[MAS/alignment] save: rename failed" << std::endl;
        remove(tmp.c_str());
        return -1;
    }
    std::cerr << "[MAS/alignment] saved Wa " << n_embd << "x" << n_embd
              << " to " << path << " (key=" << key << ")" << std::endl;
    return 0;
}

int llama_mas_load_alignment_matrix(llama_mas_context * mas, const char * path) {
    if (!mas || !path || !path[0]) return -1;

    FILE * f = fopen(path, "rb");
    if (!f) return 1;  // no cache file yet = ordinary miss

    auto rd = [&](void * p, size_t n) -> bool {
        return fread(p, 1, n, f) == n;
    };

    char magic[8] = {0};
    int32_t n_embd = 0, n_vocab = 0, stride = 0, key_len = 0;
    uint32_t rb = 0;
    bool ok = rd(magic, 8) && rd(&n_embd, 4) && rd(&n_vocab, 4) &&
              rd(&stride, 4) && rd(&rb, 4) && rd(&key_len, 4);

    if (ok && (std::memcmp(magic, MAS_ALIGN_MAGIC, 8) != 0 ||
               n_embd <= 0 || key_len < 0 || key_len > 4096)) {
        ok = false;
    }

    std::string key;
    if (ok) {
        key.resize((size_t) key_len);
        ok = rd(&key[0], (size_t) key_len);
    }

    double tnorm = 0.0;
    if (ok) ok = rd(&tnorm, sizeof(tnorm));

    float ridge = 0.0f;
    if (ok) std::memcpy(&ridge, &rb, sizeof(ridge));

    // Key comparison is the correctness gate: a different model, n_embd, vocab
    // size, stride OR RIDGE must be a miss, never a silently wrong matrix.
    //
    // `ridge` above is what the FILE was written with. The comparison must be
    // against the ridge the CURRENT context would use, otherwise a cache made
    // with ridge X is accepted into a context asking for ridge Y - a wrong
    // matrix presented as a hit. (Before this fix the expected key was built
    // from the stored ridge, i.e. it always matched itself.)
    if (ok) {
        const int32_t want_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mas->model));
        const int32_t want_stride = mas_alignment_effective_stride(mas);
        const float want_ridge = (float) mas_alignment_effective_ridge(mas);
        const std::string want = mas_alignment_key(mas, n_embd, want_vocab, want_stride, want_ridge);
        if (key != want) {
            std::cerr << "[MAS/alignment] cache key mismatch -> miss" << std::endl
                      << "    file: " << key << std::endl
                      << "    want: " << want << std::endl;
            ok = false;
        }
    }

    if (ok) {
        std::vector<float> mat((size_t) n_embd * (size_t) n_embd);
        ok = rd(mat.data(), mat.size() * sizeof(float));
        if (ok) {
            // Only accept a complete, non-degenerate matrix.
            double acc = 0.0;
            for (float v : mat) acc += (double) v * (double) v;
            if (!(acc > 0.0)) {
                std::cerr << "[MAS/alignment] cached matrix is all zeros -> miss" << std::endl;
                ok = false;
            }
        }
        if (ok) {
            fclose(f);
            mas->alignment_matrix.swap(mat);
            mas->alignment_n_embd = n_embd;
            mas->alignment_is_identity = false;
            mas->use_alignment_matrix = true;
            // Restore target_norm from the cache; without it rescale_norm would
            // silently do nothing (its guard is alignment_has_target_norm).
            mas->alignment_target_norm = tnorm;
            mas->alignment_has_target_norm = tnorm > 0.0;
            std::cerr << "[MAS/alignment] loaded Wa " << n_embd << "x" << n_embd
                      << " from " << path << " (skipping the solve)"
                      << " target_norm=" << tnorm << std::endl;
            return 0;
        }
    }

    fclose(f);
    std::cerr << "[MAS/alignment] cache unusable -> miss (" << path << ")" << std::endl;
    return 1;
}

void llama_mas_set_alignment_parallel_threads(llama_mas_context * mas, int32_t n_threads) {
    if (!mas) return;
    mas->alignment_parallel_threads = n_threads < 1 ? 1 : n_threads;
}

int32_t llama_mas_get_alignment_parallel_threads(const llama_mas_context * mas) {
    return mas ? mas->alignment_parallel_threads : 1;
}

int llama_mas_apply_alignment(llama_mas_context * mas, const float * src, float * dst, int32_t n_embd) {
    if (!mas || !src || !dst) return -1;
    if (!mas->use_alignment_matrix || mas->alignment_matrix.empty()) {
        std::memcpy(dst, src, (size_t) n_embd * sizeof(float));
        return 0;
    }
    if (mas->alignment_n_embd != n_embd) return -1;

    const float * W = mas->alignment_matrix.data();

    // The projection is n_embd^2 MACs PER LATENT STEP - 16.7M at 4096, which is
    // a real fraction of the step cost (measured 72 ms/step vs 49.5 ms/token on
    // Qwen3-8B). Output rows are independent (dst[i] reads only row i and src),
    // so this parallelises exactly like the Wa multiply in the solve.
    //
    // Switchable, default OFF: `alignment_parallel_threads` <= 1 keeps the
    // original scalar loop bit-identical. Thread spawn is ~0.2 ms for a small
    // pool, negligible against a multi-ms matmul, but it is NOT free at small
    // n_embd - hence the opt-in rather than always-on.
    const int32_t nthr = mas->alignment_parallel_threads;
    if (nthr > 1 && n_embd >= 512) {
        std::atomic<int32_t> next_row{0};
        std::vector<std::thread> pool;
        pool.reserve((size_t) nthr);
        for (int32_t t = 0; t < nthr; ++t) {
            pool.emplace_back([&] {
                for (;;) {
                    const int32_t i = next_row.fetch_add(1);
                    if (i >= n_embd) break;
                    const float * row = W + (size_t) i * n_embd;
                    float sum = 0.0f;
                    for (int32_t j = 0; j < n_embd; ++j) sum += row[j] * src[j];
                    dst[i] = sum;
                }
            });
        }
        for (auto & th : pool) th.join();
    } else {
        for (int32_t i = 0; i < n_embd; ++i) {
            const float * row = W + (size_t) i * n_embd;
            float sum = 0.0f;
            for (int32_t j = 0; j < n_embd; ++j) sum += row[j] * src[j];
            dst[i] = sum;
        }
    }

    // ── Norm re-scale (LatentMAS latent_space_realign) ──────────────────────
    // The matrix only fixes DIRECTION. LatentMAS additionally rescales each
    // aligned vector to the mean norm of the real token embeddings:
    //
    //     aligned = aligned * (target_norm / ||aligned||)
    //
    // Without it the magnitude is unconstrained and drifts every step, so the
    // vector presented at layer 0 becomes steadily more off-distribution; that
    // is what made latent quality collapse after 1-2 steps here, while
    // LatentMAS reports stable results at up to 80 steps. Measured, not assumed.
    //
    // Opt-in: default off, so existing behaviour is bit-identical. Turn on with
    // llama_mas_set_alignment_rescale_norm(mas, true) or POST
    // /mas/alignment/toggle {"rescale_norm": true}.
    if (mas->alignment_rescale_norm && mas->alignment_has_target_norm) {
        double acc = 0.0;
        for (int32_t i = 0; i < n_embd; ++i) acc += (double) dst[i] * (double) dst[i];
        const double nrm = std::sqrt(acc);
        if (nrm > 1e-6) {
            const float scale = (float) (mas->alignment_target_norm / nrm);
            for (int32_t i = 0; i < n_embd; ++i) dst[i] *= scale;
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Adapter forwards (CPU reference; CUDA paths live in the mas-adapter kernels)
// ---------------------------------------------------------------------------
void llama_mas_adapter_recursive_forward(const llama_mas_adapter_weights * w,
                                         const float * hidden_in, float * hidden_out) {
    if (!w || !hidden_in || !hidden_out) return;
    const int32_t n = w->n_embd;
    std::vector<float> x((size_t) n), t1((size_t) n), t2((size_t) n);

    layer_norm(hidden_in, w->w_pre_ln, w->b_pre_ln, x.data(), n, 1e-5f);
    mat_vec_mul(w->w_proj1, x.data(), t1.data(), n, n);
    if (w->b_proj1) vec_add(t1.data(), w->b_proj1, t1.data(), n);
    for (int32_t i = 0; i < n; ++i) t1[i] = 0.5f * t1[i] * (1.0f + std::erf(t1[i] * 0.7071067811865476f));
    mat_vec_mul(w->w_proj2, t1.data(), t2.data(), n, n);
    if (w->b_proj2) vec_add(t2.data(), w->b_proj2, t2.data(), n);
    for (int32_t i = 0; i < n; ++i) hidden_out[i] = hidden_in[i] + t2[i];
    layer_norm(hidden_out, w->w_post_ln, w->b_post_ln, hidden_out, n, 1e-5f);
}

void llama_mas_adapter_cross_forward(const llama_mas_adapter_weights * w,
                                     const float * hidden_in, float * hidden_out) {
    if (!w || !hidden_in || !hidden_out) return;
    const int32_t in_dim = w->in_dim > 0 ? w->in_dim : w->n_embd;
    const int32_t out_dim = w->out_dim > 0 ? w->out_dim : w->n_embd;
    const int32_t hid = w->hidden_dim > 0 ? w->hidden_dim : out_dim * 2;

    std::vector<float> x((size_t) in_dim), t1((size_t) hid), t2((size_t) out_dim);

    layer_norm(hidden_in, w->w_cross_ln_source, w->b_cross_ln_source, x.data(), in_dim, 1e-5f);
    mat_vec_mul(w->w_cross_proj1, x.data(), t1.data(), hid, in_dim);
    if (w->b_cross_proj1) vec_add(t1.data(), w->b_cross_proj1, t1.data(), hid);
    for (int32_t i = 0; i < hid; ++i) t1[i] = 0.5f * t1[i] * (1.0f + std::erf(t1[i] * 0.7071067811865476f));
    mat_vec_mul(w->w_cross_proj2, t1.data(), t2.data(), out_dim, hid);
    if (w->b_cross_proj2) vec_add(t2.data(), w->b_cross_proj2, t2.data(), out_dim);

    if (w->w_cross_residual && w->w_cross_residual[0] != 0.0f) {
        std::vector<float> res((size_t) out_dim, 0.0f);
        mat_vec_mul(w->w_cross_residual, hidden_in, res.data(), out_dim, in_dim);
        if (w->b_cross_residual) vec_add(res.data(), w->b_cross_residual, res.data(), out_dim);
        for (int32_t i = 0; i < out_dim; ++i) t2[i] += res[i];
    }

    std::memcpy(hidden_out, t2.data(), (size_t) out_dim * sizeof(float));
    if (out_dim == in_dim) {
        layer_norm(hidden_out, w->w_cross_ln_target, w->b_cross_ln_target, hidden_out, out_dim, 1e-5f);
    }
}

void llama_mas_adapter_mean_pool(const float * states, int32_t n_tokens, int32_t n_embd, float * output) {
    if (!states || !output || n_tokens <= 0 || n_embd <= 0) return;
    for (int32_t i = 0; i < n_embd; ++i) output[i] = 0.0f;
    for (int32_t t = 0; t < n_tokens; ++t) {
        const float * s = states + (size_t) t * n_embd;
        for (int32_t i = 0; i < n_embd; ++i) output[i] += s[i];
    }
    const float inv = 1.0f / (float) n_tokens;
    for (int32_t i = 0; i < n_embd; ++i) output[i] *= inv;
}

void llama_mas_adapter_concat(const float * states, int32_t n_tokens, int32_t n_embd, float * output) {
    if (!states || !output || n_tokens <= 0 || n_embd <= 0) return;
    std::memcpy(output, states, (size_t) n_tokens * n_embd * sizeof(float));
}

// ---------------------------------------------------------------------------
// Adapter weight setters
// ---------------------------------------------------------------------------
void llama_mas_adapter_weights_set_cross_ln_source(llama_mas_adapter_weights * w, const float * ww, const float * b, int32_t in_dim) {
    if (!w || !ww || !b || w->in_dim != in_dim || in_dim <= 0) return;
    std::memcpy(w->w_cross_ln_source, ww, (size_t) in_dim * sizeof(float));
    std::memcpy(w->b_cross_ln_source, b,  (size_t) in_dim * sizeof(float));
}

void llama_mas_adapter_weights_set_cross_ln_target(llama_mas_adapter_weights * w, const float * ww, const float * b, int32_t out_dim) {
    if (!w || !ww || !b || w->out_dim != out_dim || out_dim <= 0) return;
    std::memcpy(w->w_cross_ln_target, ww, (size_t) out_dim * sizeof(float));
    std::memcpy(w->b_cross_ln_target, b,  (size_t) out_dim * sizeof(float));
}

void llama_mas_adapter_weights_set_cross_proj1(llama_mas_adapter_weights * w, const float * ww, const float * b, int32_t in_dim, int32_t hidden_dim) {
    if (!w || !ww || !b || in_dim <= 0 || hidden_dim <= 0) return;
    if (w->in_dim != in_dim || w->hidden_dim != hidden_dim) return;
    std::memcpy(w->w_cross_proj1, ww, (size_t) in_dim * hidden_dim * sizeof(float));
    std::memcpy(w->b_cross_proj1, b,  (size_t) hidden_dim * sizeof(float));
}

void llama_mas_adapter_weights_set_cross_proj2(llama_mas_adapter_weights * w, const float * ww, const float * b, int32_t hidden_dim, int32_t out_dim) {
    if (!w || !ww || !b || hidden_dim <= 0 || out_dim <= 0) return;
    if (w->hidden_dim != hidden_dim || w->out_dim != out_dim) return;
    std::memcpy(w->w_cross_proj2, ww, (size_t) hidden_dim * out_dim * sizeof(float));
    std::memcpy(w->b_cross_proj2, b,  (size_t) out_dim * sizeof(float));
}

void llama_mas_adapter_weights_set_cross_residual(llama_mas_adapter_weights * w, const float * ww, const float * b, int32_t in_dim, int32_t out_dim) {
    if (!w || !ww || !b || in_dim <= 0 || out_dim <= 0) return;
    if (w->in_dim != in_dim || w->out_dim != out_dim) return;
    std::memcpy(w->w_cross_residual, ww, (size_t) in_dim * out_dim * sizeof(float));
    std::memcpy(w->b_cross_residual, b,  (size_t) out_dim * sizeof(float));
}


// ---------------------------------------------------------------------------
// Hidden-state encoding helpers
//
// All return a malloc'd float array; free with llama_mas_free_latent().
// ---------------------------------------------------------------------------
static bool mas_decode_prompt_tokens(llama_mas_context * mas, llama_context * ctx,
                                     const std::vector<llama_token> & tokens,
                                     llama_seq_id seq_id) {
    if (!mas || !ctx || tokens.empty()) return false;
    if ((int32_t) tokens.size() > (int32_t) llama_n_batch(ctx)) {
        std::cerr << "[MAS/encode] prompt " << tokens.size()
                  << " tokens > n_batch " << llama_n_batch(ctx) << std::endl;
        return false;
    }
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t p = 0; p < tokens.size(); ++p) {
        batch.token[p] = tokens[p];
        batch.pos[p] = (llama_pos) p;
        batch.n_seq_id[p] = 1;
        batch.seq_id[p][0] = seq_id;
        batch.logits[p] = (p + 1 == tokens.size()) ? 1 : 0;
    }
    batch.n_tokens = (int32_t) tokens.size();
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret == 0;
}

float * llama_mas_encode(llama_mas_context * mas, const char * prompt,
                         int32_t * out_n_tokens, int32_t * out_n_embd) {
    if (!mas || !mas->ctx || !prompt || !out_n_tokens || !out_n_embd) return nullptr;

    const llama_vocab * vocab = llama_model_get_vocab(mas->model);
    std::vector<llama_token> tokens = tokenize_cached(mas, vocab, prompt);
    if (tokens.empty()) return nullptr;

    llama_set_output_hidden_states(mas->ctx, true);
    if (!mas_decode_prompt_tokens(mas, mas->ctx, tokens, 0)) return nullptr;

    const int32_t n_layers = llama_get_hidden_states_layer_count(mas->ctx);
    if (n_layers <= 0) return nullptr;

    float * layer = llama_get_hidden_states_layer(mas->ctx, n_layers - 1);
    const int32_t n_out = llama_get_hidden_states_layer_n_outputs(mas->ctx, n_layers - 1);
    if (!layer || n_out <= 0) return nullptr;

    const int32_t n_embd = llama_model_n_embd(mas->model);
    *out_n_tokens = n_out;
    *out_n_embd = n_embd;

    float * result = (float *) malloc((size_t) n_out * n_embd * sizeof(float));
    if (!result) return nullptr;
    std::memcpy(result, layer, (size_t) n_out * n_embd * sizeof(float));
    return result;
}

float * llama_mas_encode_agent(llama_mas_context * mas, int32_t agent_idx, const char * prompt,
                               int32_t * out_n_tokens, int32_t * out_n_embd) {
    if (!mas || !mas->ctx || agent_idx < 0 || agent_idx >= (int32_t) mas->agents.size()) return nullptr;

    auto & agent = mas->agents[agent_idx];
    llama_context_set_layer_graph(mas->ctx, agent->layer_graph);

    float * latent = llama_mas_encode(mas, prompt, out_n_tokens, out_n_embd);
    if (!latent) return nullptr;

    const int32_t n_embd = *out_n_embd;
    const int32_t n_tokens = *out_n_tokens;

    if (agent->adapter_type == LLAMA_MAS_ADAPTER_RECURSIVE && agent->adapter_weights && n_embd > 0) {
        const int32_t out_dim = agent->adapter_weights->out_dim > 0
                              ? agent->adapter_weights->out_dim : n_embd;
        float * pooled = (float *) malloc((size_t) n_embd * sizeof(float));
        float * out = (float *) malloc((size_t) out_dim * sizeof(float));
        if (pooled && out) {
            llama_mas_adapter_mean_pool(latent, n_tokens, n_embd, pooled);
            llama_mas_adapter_recursive_forward(agent->adapter_weights.get(), pooled, out);
            free(latent);
            free(pooled);
            *out_n_tokens = 1;
            *out_n_embd = out_dim;
            return out;
        }
        free(pooled);
        free(out);
    }
    return latent;
}

float ** llama_mas_encode_all_layers(llama_mas_context * mas, const char * prompt,
                                     int32_t * out_n_layers, int32_t ** out_shapes) {
    if (!mas || !mas->ctx || !prompt || !out_n_layers || !out_shapes) return nullptr;

    const llama_vocab * vocab = llama_model_get_vocab(mas->model);
    std::vector<llama_token> tokens = tokenize_cached(mas, vocab, prompt);
    if (tokens.empty()) return nullptr;

    llama_set_output_hidden_states(mas->ctx, true);
    if (!mas_decode_prompt_tokens(mas, mas->ctx, tokens, 0)) return nullptr;

    const int32_t n_layers = llama_get_hidden_states_layer_count(mas->ctx);
    if (n_layers <= 0) return nullptr;

    const int32_t n_embd = llama_model_n_embd(mas->model);

    float ** layers = (float **) calloc((size_t) n_layers, sizeof(float *));
    int32_t * shapes = (int32_t *) calloc((size_t) n_layers * 2, sizeof(int32_t));
    if (!layers || !shapes) {
        free(layers);
        free(shapes);
        return nullptr;
    }

    for (int32_t il = 0; il < n_layers; ++il) {
        float * layer = llama_get_hidden_states_layer(mas->ctx, il);
        const int32_t n_out = llama_get_hidden_states_layer_n_outputs(mas->ctx, il);
        if (!layer || n_out <= 0) continue;
        layers[il] = (float *) malloc((size_t) n_out * n_embd * sizeof(float));
        if (!layers[il]) continue;
        std::memcpy(layers[il], layer, (size_t) n_out * n_embd * sizeof(float));
        shapes[il * 2 + 0] = n_out;
        shapes[il * 2 + 1] = n_embd;
    }

    *out_n_layers = n_layers;
    *out_shapes = shapes;
    return layers;
}

void llama_mas_free_layers(float ** layers, int32_t n_layers) {
    if (!layers) return;
    for (int32_t i = 0; i < n_layers; ++i) {
        if (layers[i]) free(layers[i]);
    }
    free(layers);
}

void llama_mas_free_latent(float * latent) {
    free(latent);
}

float * llama_mas_cross_link(llama_mas_context * mas, int32_t src_agent_idx, int32_t dst_agent_idx,
                             const float * src_hidden, int32_t n_tokens, int32_t n_embd) {
    if (!mas || !src_hidden || n_tokens <= 0 || n_embd <= 0) return nullptr;
    if (src_agent_idx < 0 || src_agent_idx >= (int32_t) mas->agents.size()) return nullptr;
    if (dst_agent_idx < 0 || dst_agent_idx >= (int32_t) mas->agents.size()) return nullptr;

    auto & src = mas->agents[src_agent_idx];
    const int32_t out_dim = (src->adapter_weights && src->adapter_weights->out_dim > 0)
                          ? src->adapter_weights->out_dim : n_embd;

    float * result = (float *) malloc((size_t) n_tokens * out_dim * sizeof(float));
    if (!result) return nullptr;

    if (src->adapter_type == LLAMA_MAS_ADAPTER_CROSS && src->adapter_weights && out_dim != n_embd) {
        std::vector<float> pooled((size_t) n_embd), cross((size_t) out_dim);
        llama_mas_adapter_mean_pool(src_hidden, n_tokens, n_embd, pooled.data());
        llama_mas_adapter_cross_forward(src->adapter_weights.get(), pooled.data(), cross.data());
        for (int32_t t = 0; t < n_tokens; ++t) {
            std::memcpy(result + (size_t) t * out_dim, cross.data(), (size_t) out_dim * sizeof(float));
        }
    } else {
        std::memcpy(result, src_hidden, (size_t) n_tokens * n_embd * sizeof(float));
    }
    return result;
}

int llama_mas_project_agent_to_embeddings(llama_mas_context * mas, int32_t src_agent_idx, int32_t dst_agent_idx) {
    if (!mas || !mas->ctx) return -1;
    if (src_agent_idx < 0 || src_agent_idx >= (int32_t) mas->agents.size()) return -1;
    if (dst_agent_idx < 0 || dst_agent_idx >= (int32_t) mas->agents.size()) return -1;

    // Capture must be on, otherwise get_hidden_states_layer() returns empty and
    // the getter yields NULL.
    llama_set_output_hidden_states(mas->ctx, true);
    llama_context * ctx = mas->ctx;
    ctx->synchronize();

    float * hidden = llama_get_hidden_states_last_token(mas->ctx);
    if (!hidden) return -1;
    const int32_t n_embd = llama_model_n_embd(mas->model);

    auto & src = mas->agents[src_agent_idx];
    auto & dst = mas->agents[dst_agent_idx];

    std::vector<float> projected((size_t) n_embd);
    int32_t out_dim = n_embd;

    if (src->adapter_weights && src->adapter_weights->in_dim > 0 && src->adapter_weights->out_dim > 0) {
        out_dim = src->adapter_weights->out_dim;
        projected.resize((size_t) out_dim);
        llama_mas_adapter_cross_forward(src->adapter_weights.get(), hidden, projected.data());
    } else {
        std::memcpy(projected.data(), hidden, (size_t) n_embd * sizeof(float));
    }

    llama_mas_set_agent_input_embeddings(mas, dst_agent_idx, projected.data(), 1, out_dim);
    return 0;
}



// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
// Derive prefill vs generation timings from the agents' position counters and
// log one line. Generation is charged 10x the wall cost of a prefill token
// because decode is sequential while prefill is batched.
static void mas_compute_and_log_timing(llama_mas_context * mas, double t_run_start) {
    const double t_total = mas_time_ms() - t_run_start;

    int64_t n_prefill = 0;
    int64_t n_gen = 0;
    for (const auto & agent : mas->agents) {
        n_prefill += (int64_t) agent->prompt_pos;
        n_gen     += (int64_t) agent->generated_len;
    }

    double prefill_ms = 0.0;
    double generation_ms = t_total;
    if (n_prefill > 0 && n_gen > 0) {
        const double prefill_weight = (double) n_prefill / (n_prefill + n_gen * 10.0);
        prefill_ms = t_total * prefill_weight;
        generation_ms = t_total * (1.0 - prefill_weight);
    } else if (n_prefill > 0) {
        prefill_ms = t_total;
        generation_ms = 0.0;
    }

    mas->last_timing.prefill_ms = prefill_ms;
    mas->last_timing.generation_ms = generation_ms;
    mas->last_timing.n_prefill_tokens = n_prefill;
    mas->last_timing.n_generated_tokens = n_gen;
    mas->last_timing.prefill_tokens_per_sec = prefill_ms > 0.0 ? (n_prefill / (prefill_ms / 1000.0)) : 0.0;
    mas->last_timing.tokens_per_sec = generation_ms > 0.0 ? (n_gen / (generation_ms / 1000.0)) : 0.0;

    fprintf(stderr, "[MAS TIMING] total=%.2fms prefill=%.2fms (n=%lld, %.1f t/s) generation=%.2fms (n=%lld, %.1f t/s)\n",
            t_total, prefill_ms, (long long) n_prefill, mas->last_timing.prefill_tokens_per_sec,
            generation_ms, (long long) n_gen, mas->last_timing.tokens_per_sec);
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------
int llama_mas_run(llama_mas_context * mas, const char * task_prompt, int32_t max_tokens, llama_mas_strategy strategy) {
    if (!mas || !mas->ctx || !task_prompt) return -1;
    const llama_vocab * vocab = llama_model_get_vocab(mas->model);

    // The wave paths batch every agent's full prompt into ONE llama_decode; if
    // the sum exceeds n_batch the library asserts, so reject cleanly instead.
    {
        int64_t prefill_total = 0;
        for (const auto & agent : mas->agents) {
            prefill_total += (int64_t) agent->prompt_tokens.size();
        }
        if (prefill_total > (int64_t) llama_n_batch(mas->ctx)) {
            LLAMA_LOG_ERROR("%s: total prefill across agents %lld > n_batch %u\n",
                            __func__, (long long) prefill_total, llama_n_batch(mas->ctx));
            return -8;
        }
    }

    const double t_run_start = mas_time_ms();
    std::memset(&mas->last_timing, 0, sizeof(mas->last_timing));
    mas->stream_step = 0;

    // ---------------- CHAIN (text output of each agent feeds the next) ------
    if (strategy == LLAMA_MAS_STRATEGY_CHAIN && mas->agents.size() > 1) {
        std::string previous_output;
        for (size_t i = 0; i < mas->agents.size(); ++i) {
            auto & agent = mas->agents[i];
            agent->run_started_ms = mas_time_ms();
            agent->run_finished_ms = 0.0;

            std::string inner_prompt;
            if (i > 0 && !previous_output.empty()) {
                inner_prompt = task_prompt;
                inner_prompt += "\n\n=== PREVIOUS OUTPUT ===\n";
                inner_prompt += previous_output;
                inner_prompt += "\n=== END ===\n";
                if (!agent->prompt_text.empty()) {
                    inner_prompt += "\n=== ROLE ===\n" + agent->prompt_text + "\n";
                }
            } else {
                inner_prompt = agent->prompt_text.empty() ? task_prompt : agent->prompt_text;
            }

            std::vector<llama_token> prompt_tokens;
            if (agent->input_embeddings.empty()) {
                prompt_tokens = tokenize_cached(mas, vocab, inner_prompt);
            }

            llama_memory_t mem = llama_get_memory(mas->ctx);
            if (mem) llama_memory_seq_rm(mem, agent->seq_id, 0, -1);

            if (!agent->input_embeddings.empty()) {
                const int32_t n_emb_tok = (int32_t) agent->input_embeddings.size() / agent->input_embeddings_n_embd;
                llama_batch batch = llama_batch_init(n_emb_tok, agent->input_embeddings_n_embd, 1);
                for (size_t p = 0; p < (size_t) n_emb_tok; ++p) {
                    std::memcpy(&batch.embd[p * agent->input_embeddings_n_embd],
                                &agent->input_embeddings[p * agent->input_embeddings_n_embd],
                                agent->input_embeddings_n_embd * sizeof(float));
                    batch.pos[p] = (llama_pos) p;
                    batch.n_seq_id[p] = 1;
                    batch.seq_id[p][0] = agent->seq_id;
                    batch.logits[p] = (p + 1 == (size_t) n_emb_tok) ? 1 : 0;
                }
                batch.n_tokens = n_emb_tok;
                int ret = llama_decode(mas->ctx, batch);
                llama_batch_free(batch);
                if (ret != 0) return ret;
                agent->prompt_pos = (size_t) n_emb_tok;
            } else if (!prompt_tokens.empty()) {
                llama_batch batch = llama_batch_init((int32_t) prompt_tokens.size(), 0, 1);
                for (size_t p = 0; p < prompt_tokens.size(); ++p) {
                    batch.token[p] = prompt_tokens[p];
                    batch.pos[p] = (llama_pos) p;
                    batch.n_seq_id[p] = 1;
                    batch.seq_id[p][0] = agent->seq_id;
                    batch.logits[p] = (p + 1 == prompt_tokens.size()) ? 1 : 0;
                }
                batch.n_tokens = (int32_t) prompt_tokens.size();
                int ret = llama_decode(mas->ctx, batch);
                llama_batch_free(batch);
                if (ret != 0) return ret;
                agent->prompt_pos = prompt_tokens.size();
            }

            if (agent->stop_after_prompt) {
                agent->finished = true;
                continue;
            }

            if (!agent->is_latent && (!prompt_tokens.empty() || !agent->input_embeddings.empty())) {
                llama_sampler * smpl = mas->samplers[i];
                llama_token next = llama_sampler_sample(smpl, mas->ctx, -1);
                llama_sampler_accept(smpl, next);
                agent->generated_tokens.push_back(next);
                agent->generated_len = 1;
                if (mas->stream_cb) {
                    mas->stream_cb((int32_t) i, next, mas->stream_step++, mas->stream_user_data);
                }
                if (llama_vocab_is_eog(vocab, next) || (int32_t) agent->generated_len >= agent->max_tokens) {
                    agent->finished = true;
                }
            }

            while (!agent->finished) {
                int ret = llama_mas_decode_step(mas);
                if (ret != 0) return ret;
            }

            if (!agent->is_latent) {
                char * output = llama_mas_get_text(mas, (int32_t) i);
                if (output) { previous_output = output; llama_mas_free_text(output); }
            }
            agent->run_finished_ms = mas_time_ms();
        }
        mas_compute_and_log_timing(mas, t_run_start);
        return 0;
    }

    // ---------------- CHAIN_BROADCAST / LATENT_MAS / LATENT_MAS_NO_ALIGN ----
    if ((strategy == LLAMA_MAS_STRATEGY_CHAIN_BROADCAST ||
         strategy == LLAMA_MAS_STRATEGY_LATENT_MAS ||
         strategy == LLAMA_MAS_STRATEGY_LATENT_MAS_NO_ALIGN) && !mas->agents.empty()) {
        std::vector<std::vector<int32_t>> waves;
        if (strategy == LLAMA_MAS_STRATEGY_CHAIN_BROADCAST && !mas->broadcast_waves.empty()) {
            waves = mas->broadcast_waves;
        } else {
            for (int i = 0; i < (int) mas->agents.size(); ++i) waves.push_back({i});
        }

        for (size_t w = 0; w < waves.size(); ++w) {
            for (int idx : waves[w]) {
                if (idx < 0 || idx >= (int32_t) mas->agents.size()) return -1;
            }
        }

        for (size_t w = 0; w < waves.size(); ++w) {
            const bool is_latent_transfer =
                    (strategy != LLAMA_MAS_STRATEGY_CHAIN_BROADCAST) && w > 0;

            if (strategy == LLAMA_MAS_STRATEGY_LATENT_MAS_NO_ALIGN) {
                mas->use_alignment_matrix = false;
            } else if (strategy == LLAMA_MAS_STRATEGY_LATENT_MAS) {
                if (!mas->alignment_matrix.empty()) mas->use_alignment_matrix = true;
            }

            for (int idx : waves[w]) {
                auto & agent = mas->agents[idx];
                agent->run_started_ms = mas_time_ms();
                agent->run_finished_ms = 0.0;

                if (!is_latent_transfer) {
                    agent->input_embeddings.clear();
                    agent->input_embeddings_n_embd = 0;
                    std::string text = agent->prompt_text.empty() ? task_prompt : agent->prompt_text;
                    agent->prompt_tokens = tokenize_cached(mas, vocab, text);
                    if (agent->max_tokens <= 0) agent->max_tokens = max_tokens;
                    agent->finished = false;
                    agent->prompt_pos = 0;
                    agent->generated_len = 0;
                    agent->generated_tokens.clear();
                    agent->latent_steps_remaining = agent->latent_steps;
                    llama_sampler_reset(mas->samplers[idx]);
                } else {
                    // KV was cloned from the previous wave: no prompt decode.
                    agent->prompt_tokens.clear();
                    agent->input_embeddings.clear();
                    agent->input_embeddings_n_embd = 0;
                    if (agent->max_tokens <= 0) agent->max_tokens = max_tokens;
                    agent->finished = false;
                    agent->latent_steps_remaining = agent->latent_steps;
                    llama_sampler_reset(mas->samplers[idx]);
                }
            }

            std::vector<bool> saved_finished(mas->agents.size());
            for (size_t i = 0; i < mas->agents.size(); ++i) {
                saved_finished[i] = mas->agents[i]->finished;
                bool in_wave = false;
                for (int idx : waves[w]) if ((size_t) idx == i) { in_wave = true; break; }
                if (!in_wave) mas->agents[i]->finished = true;
            }

            // Step 1: embedding-driven agents in this wave.
            for (int idx : waves[w]) {
                auto & agent = mas->agents[idx];
                if (agent->input_embeddings.empty()) continue;
                llama_memory_t mem = llama_get_memory(mas->ctx);
                llama_memory_seq_rm(mem, agent->seq_id, 0, -1);
                const int32_t n_emb_tok = (int32_t) agent->input_embeddings.size() / agent->input_embeddings_n_embd;
                llama_batch batch = llama_batch_init(n_emb_tok, agent->input_embeddings_n_embd, 1);
                for (size_t p = 0; p < (size_t) n_emb_tok; ++p) {
                    std::memcpy(&batch.embd[p * agent->input_embeddings_n_embd],
                                &agent->input_embeddings[p * agent->input_embeddings_n_embd],
                                agent->input_embeddings_n_embd * sizeof(float));
                    batch.pos[p] = (llama_pos) p;
                    batch.n_seq_id[p] = 1;
                    batch.seq_id[p][0] = agent->seq_id;
                    batch.logits[p] = (p + 1 == (size_t) n_emb_tok) ? 1 : 0;
                }
                batch.n_tokens = n_emb_tok;
                int ret = llama_decode(mas->ctx, batch);
                llama_batch_free(batch);
                if (ret != 0) return ret;
                agent->prompt_pos = (size_t) n_emb_tok;

                llama_sampler * smpl = mas->samplers[idx];
                llama_token next = llama_sampler_sample(smpl, mas->ctx, -1);
                llama_sampler_accept(smpl, next);
                agent->generated_tokens.push_back(next);
                agent->generated_len = 1;
                if (mas->stream_cb) {
                    mas->stream_cb(idx, next, mas->stream_step++, mas->stream_user_data);
                }
                if (llama_vocab_is_eog(vocab, next) || (int32_t) agent->generated_len >= agent->max_tokens) {
                    agent->finished = true;
                }
            }

            // Step 2: token-driven agents, batched per compatible layer graph.
            std::vector<int> prompt_pending;
            for (int idx : waves[w]) {
                auto & agent = mas->agents[idx];
                if (!agent->input_embeddings.empty()) continue;
                if (is_latent_transfer) {
                    agent->finished = false;
                    continue;
                }
                llama_memory_t mem = llama_get_memory(mas->ctx);
                llama_memory_seq_rm(mem, agent->seq_id, 0, -1);
                if (agent->prompt_tokens.empty()) agent->finished = true;
                else prompt_pending.push_back(idx);
            }

            while (!prompt_pending.empty()) {
                llama_layer_graph_context * group_layer_graph = mas->agents[prompt_pending.front()]->layer_graph;
                std::vector<int> group, remaining;
                int32_t n_tokens_total = 0;
                int32_t n_tokens_all_groups = 0;
                for (int idx : prompt_pending) {
                    auto & agent = mas->agents[idx];
                    n_tokens_all_groups += (int32_t) agent->prompt_tokens.size();
                    if (agent->layer_graph == group_layer_graph) {
                        group.push_back(idx);
                        n_tokens_total += (int32_t) agent->prompt_tokens.size();
                    } else {
                        remaining.push_back(idx);
                    }
                }
                prompt_pending.swap(remaining);

                if (n_tokens_all_groups > (int32_t) llama_n_batch(mas->ctx)) {
                    LLAMA_LOG_ERROR("%s: wave prefill batch %d tokens exceeds n_batch %u\n",
                                    __func__, n_tokens_all_groups, llama_n_batch(mas->ctx));
                    return -8;
                }

                llama_context_set_layer_graph(mas->ctx, group_layer_graph);
                const llama_mas_adapter_weights * common_adapter = find_common_adapter(group, mas->agents);
                if (mas->use_cuda_adapters && common_adapter) mas->ctx->set_mas_adapter(common_adapter);
                else mas->ctx->clear_mas_adapter();

                llama_batch batch = llama_batch_init(n_tokens_total, 0, 1);
                int cursor = 0;
                std::vector<int> agent_last_cursor;
                agent_last_cursor.reserve(group.size());
                for (int idx : group) {
                    auto & agent = mas->agents[idx];
                    for (size_t p = 0; p < agent->prompt_tokens.size(); ++p) {
                        batch.token[cursor] = agent->prompt_tokens[p];
                        batch.pos[cursor] = (llama_pos) p;
                        batch.n_seq_id[cursor] = 1;
                        batch.seq_id[cursor][0] = agent->seq_id;
                        batch.logits[cursor] = (p + 1 == agent->prompt_tokens.size()) ? 1 : 0;
                        if (p + 1 == agent->prompt_tokens.size()) agent_last_cursor.push_back(cursor);
                        cursor++;
                    }
                }
                batch.n_tokens = n_tokens_total;
                int ret = llama_decode(mas->ctx, batch);
                llama_batch_free(batch);
                if (ret != 0) return ret;

                for (size_t g = 0; g < group.size(); ++g) {
                    int idx = group[g];
                    auto & agent = mas->agents[idx];
                    agent->prompt_pos = agent->prompt_tokens.size();
                    llama_sampler * smpl = mas->samplers[idx];
                    llama_token next = llama_sampler_sample(smpl, mas->ctx, agent_last_cursor[g]);
                    llama_sampler_accept(smpl, next);
                    agent->generated_tokens.push_back(next);
                    agent->generated_len++;
                    if (mas->stream_cb) {
                        mas->stream_cb(idx, next, mas->stream_step++, mas->stream_user_data);
                    }
                    if (llama_vocab_is_eog(vocab, next) || (int32_t) agent->generated_len >= agent->max_tokens) {
                        agent->finished = true;
                    }
                }
            }
            mas->ctx->clear_mas_adapter();

            // Step 3: generate the remaining tokens of this wave.
            while (true) {
                bool all_done = true;
                for (int idx : waves[w]) if (!mas->agents[idx]->finished) { all_done = false; break; }
                if (all_done) break;
                int ret = llama_mas_decode_step(mas);
                if (ret != 0) return ret;
            }

            const double wave_done_ms = mas_time_ms();
            for (int idx : waves[w]) {
                if (idx >= 0 && idx < (int32_t) mas->agents.size()) {
                    mas->agents[idx]->run_finished_ms = wave_done_ms;
                }
            }

            // Step 4: hand this wave's KV to the next wave.
            if (w + 1 < waves.size()) {
                const int last_agent_idx = waves[w].back();
                auto & last_agent = mas->agents[last_agent_idx];
                for (int next_idx : waves[w + 1]) {
                    llama_mas_kv_clone(mas, last_agent_idx, next_idx);
                    auto & next_agent = mas->agents[next_idx];
                    next_agent->input_embeddings.clear();
                    next_agent->input_embeddings_n_embd = 0;
                    next_agent->prompt_pos = last_agent->prompt_pos + last_agent->generated_len;
                    next_agent->generated_len = 0;
                    next_agent->generated_tokens = last_agent->generated_tokens;
                    next_agent->finished = false;
                    std::cerr << "[MAS/latent_mas] KV cloned " << last_agent->role
                              << " -> " << next_agent->role
                              << " total_tokens=" << next_agent->prompt_pos << std::endl;
                }
            }

            for (size_t i = 0; i < mas->agents.size(); ++i) {
                mas->agents[i]->finished = saved_finished[i];
            }
        }
        mas_compute_and_log_timing(mas, t_run_start);
        return 0;
    }

    // ---------------- LATENT_MAS_EMBEDDINGS ---------------------------------
    // Handoff is the captured hidden state as the next agent's input embedding
    // rather than a KV clone.
    if (strategy == LLAMA_MAS_STRATEGY_LATENT_MAS_EMBEDDINGS && !mas->agents.empty()) {
        std::vector<std::vector<int32_t>> waves;
        for (int i = 0; i < (int) mas->agents.size(); ++i) waves.push_back({i});

        for (size_t w = 0; w < waves.size(); ++w) {
            const bool is_latent_transfer = (w > 0);

            for (int idx : waves[w]) {
                auto & agent = mas->agents[idx];
                agent->run_started_ms = mas_time_ms();
                agent->run_finished_ms = 0.0;
                agent->input_embeddings.clear();
                agent->input_embeddings_n_embd = 0;
                if (!is_latent_transfer) {
                    std::string text = agent->prompt_text.empty() ? task_prompt : agent->prompt_text;
                    agent->prompt_tokens = tokenize_cached(mas, vocab, text);
                    agent->prompt_pos = 0;
                    agent->generated_len = 0;
                    agent->generated_tokens.clear();
                } else {
                    agent->prompt_tokens.clear();
                }
                if (agent->max_tokens <= 0) agent->max_tokens = max_tokens;
                agent->finished = false;
                agent->latent_steps_remaining = agent->latent_steps;
                llama_sampler_reset(mas->samplers[idx]);
            }

            std::vector<bool> saved_finished(mas->agents.size());
            for (size_t i = 0; i < mas->agents.size(); ++i) {
                saved_finished[i] = mas->agents[i]->finished;
                bool in_wave = false;
                for (int idx : waves[w]) if ((size_t) idx == i) { in_wave = true; break; }
                if (!in_wave) mas->agents[i]->finished = true;
            }

            // Step 1: embedding-driven agents.
            for (int idx : waves[w]) {
                auto & agent = mas->agents[idx];
                if (agent->input_embeddings.empty()) continue;
                llama_memory_t mem = llama_get_memory(mas->ctx);
                llama_memory_seq_rm(mem, agent->seq_id, 0, -1);
                const int32_t n_emb_tok = (int32_t) agent->input_embeddings.size() / agent->input_embeddings_n_embd;
                llama_batch batch = llama_batch_init(n_emb_tok, agent->input_embeddings_n_embd, 1);
                for (size_t p = 0; p < (size_t) n_emb_tok; ++p) {
                    std::memcpy(&batch.embd[p * agent->input_embeddings_n_embd],
                                &agent->input_embeddings[p * agent->input_embeddings_n_embd],
                                agent->input_embeddings_n_embd * sizeof(float));
                    batch.pos[p] = (llama_pos) p;
                    batch.n_seq_id[p] = 1;
                    batch.seq_id[p][0] = agent->seq_id;
                    batch.logits[p] = (p + 1 == (size_t) n_emb_tok) ? 1 : 0;
                }
                batch.n_tokens = n_emb_tok;
                int ret = llama_decode(mas->ctx, batch);
                llama_batch_free(batch);
                if (ret != 0) return ret;
                agent->prompt_pos = (size_t) n_emb_tok;

                llama_sampler * smpl = mas->samplers[idx];
                llama_token next = llama_sampler_sample(smpl, mas->ctx, -1);
                llama_sampler_accept(smpl, next);
                agent->generated_tokens.push_back(next);
                agent->generated_len = 1;
                if (mas->stream_cb) {
                    mas->stream_cb(idx, next, mas->stream_step++, mas->stream_user_data);
                }
                if (llama_vocab_is_eog(vocab, next) || (int32_t) agent->generated_len >= agent->max_tokens) {
                    agent->finished = true;
                }
            }

            // Step 2: token-driven agents.
            std::vector<int> prompt_pending;
            for (int idx : waves[w]) {
                auto & agent = mas->agents[idx];
                if (!agent->input_embeddings.empty()) continue;
                if (is_latent_transfer) continue;
                llama_memory_t mem = llama_get_memory(mas->ctx);
                llama_memory_seq_rm(mem, agent->seq_id, 0, -1);
                if (agent->prompt_tokens.empty()) agent->finished = true;
                else prompt_pending.push_back(idx);
            }

            while (!prompt_pending.empty()) {
                llama_layer_graph_context * group_layer_graph = mas->agents[prompt_pending.front()]->layer_graph;
                std::vector<int> group, remaining;
                int32_t n_tokens_total = 0;
                int32_t n_tokens_all_groups = 0;
                for (int idx : prompt_pending) {
                    auto & agent = mas->agents[idx];
                    n_tokens_all_groups += (int32_t) agent->prompt_tokens.size();
                    if (agent->layer_graph == group_layer_graph) {
                        group.push_back(idx);
                        n_tokens_total += (int32_t) agent->prompt_tokens.size();
                    } else {
                        remaining.push_back(idx);
                    }
                }
                prompt_pending.swap(remaining);

                if (n_tokens_all_groups > (int32_t) llama_n_batch(mas->ctx)) {
                    LLAMA_LOG_ERROR("%s: wave prefill %d tokens > n_batch %u\n",
                                    __func__, n_tokens_all_groups, llama_n_batch(mas->ctx));
                    return -8;
                }

                llama_context_set_layer_graph(mas->ctx, group_layer_graph);
                const llama_mas_adapter_weights * common_adapter = find_common_adapter(group, mas->agents);
                if (mas->use_cuda_adapters && common_adapter) mas->ctx->set_mas_adapter(common_adapter);
                else mas->ctx->clear_mas_adapter();

                llama_batch batch = llama_batch_init(n_tokens_total, 0, 1);
                int cursor = 0;
                std::vector<int> agent_last_cursor;
                agent_last_cursor.reserve(group.size());
                for (int idx : group) {
                    auto & agent = mas->agents[idx];
                    for (size_t p = 0; p < agent->prompt_tokens.size(); ++p) {
                        batch.token[cursor] = agent->prompt_tokens[p];
                        batch.pos[cursor] = (llama_pos) p;
                        batch.n_seq_id[cursor] = 1;
                        batch.seq_id[cursor][0] = agent->seq_id;
                        batch.logits[cursor] = (p + 1 == agent->prompt_tokens.size()) ? 1 : 0;
                        if (p + 1 == agent->prompt_tokens.size()) agent_last_cursor.push_back(cursor);
                        cursor++;
                    }
                }
                batch.n_tokens = n_tokens_total;
                int ret = llama_decode(mas->ctx, batch);
                llama_batch_free(batch);
                if (ret != 0) return ret;

                for (size_t g = 0; g < group.size(); ++g) {
                    int idx = group[g];
                    auto & agent = mas->agents[idx];
                    agent->prompt_pos = agent->prompt_tokens.size();
                    llama_sampler * smpl = mas->samplers[idx];
                    llama_token next = llama_sampler_sample(smpl, mas->ctx, agent_last_cursor[g]);
                    llama_sampler_accept(smpl, next);
                    agent->generated_tokens.push_back(next);
                    agent->generated_len++;
                    if (mas->stream_cb) {
                        mas->stream_cb(idx, next, mas->stream_step++, mas->stream_user_data);
                    }
                    if (llama_vocab_is_eog(vocab, next) || (int32_t) agent->generated_len >= agent->max_tokens) {
                        agent->finished = true;
                    }
                }
            }
            mas->ctx->clear_mas_adapter();

            // Step 3: finish this wave.
            while (true) {
                bool all_done = true;
                for (int idx : waves[w]) if (!mas->agents[idx]->finished) { all_done = false; break; }
                if (all_done) break;
                int ret = llama_mas_decode_step(mas);
                if (ret != 0) return ret;
            }

            const double wave_done_ms = mas_time_ms();
            for (int idx : waves[w]) {
                if (idx >= 0 && idx < (int32_t) mas->agents.size()) {
                    mas->agents[idx]->run_finished_ms = wave_done_ms;
                }
            }
            for (size_t i = 0; i < mas->agents.size(); ++i) {
                mas->agents[i]->finished = saved_finished[i];
            }

            // Step 4: capture the hidden state and pass it on as embeddings.
            if (w + 1 < waves.size()) {
                const int last_agent_idx = waves[w].back();
                auto & last_agent = mas->agents[last_agent_idx];
                int32_t n_embd = 0;
                float * hidden_state = llama_mas_get_agent_hidden_state(mas, last_agent_idx, &n_embd);
                if (!hidden_state || n_embd == 0) {
                    std::cerr << "[MAS/latent_embeddings] no hidden state from "
                              << last_agent->role << std::endl;
                    return -1;
                }
                for (int next_idx : waves[w + 1]) {
                    llama_mas_set_agent_input_embeddings(mas, next_idx, hidden_state, 1, n_embd);
                    mas->agents[next_idx]->prompt_pos = 0;
                    mas->agents[next_idx]->generated_len = 0;
                    mas->agents[next_idx]->generated_tokens.clear();
                    mas->agents[next_idx]->finished = false;
                    std::cerr << "[MAS/latent_embeddings] hidden " << last_agent->role
                              << " -> embeddings of " << mas->agents[next_idx]->role << std::endl;
                }
            }
        }
        mas_compute_and_log_timing(mas, t_run_start);
        return 0;
    }

    // ---------------- BROADCAST / ROLE / default ----------------------------
    // Prefill every agent, then decode them together.
    for (int i = 0; i < (int) mas->agents.size(); ++i) {
        auto & agent = mas->agents[i];
        agent->run_started_ms = mas_time_ms();
        agent->run_finished_ms = 0.0;
        if (agent->input_embeddings.empty()) {
            std::string text = agent->prompt_text.empty() ? task_prompt : agent->prompt_text;
            agent->prompt_tokens = tokenize_cached(mas, vocab, text);
        }
        if (agent->max_tokens <= 0) agent->max_tokens = max_tokens;
        agent->finished = false;
        agent->prompt_pos = 0;
        agent->generated_len = 0;
        agent->generated_tokens.clear();
        agent->latent_steps_remaining = agent->latent_steps;
        llama_sampler_reset(mas->samplers[i]);
    }

    // Embedding-driven agents decode alone (they cannot share a token batch).
    for (int i = 0; i < (int) mas->agents.size(); ++i) {
        auto & agent = mas->agents[i];
        if (agent->input_embeddings.empty()) continue;
        llama_memory_t mem = llama_get_memory(mas->ctx);
        llama_memory_seq_rm(mem, agent->seq_id, 0, -1);
        const int32_t n_emb_tok = (int32_t) agent->input_embeddings.size() / agent->input_embeddings_n_embd;
        llama_batch batch = llama_batch_init(n_emb_tok, agent->input_embeddings_n_embd, 1);
        for (size_t p = 0; p < (size_t) n_emb_tok; ++p) {
            std::memcpy(&batch.embd[p * agent->input_embeddings_n_embd],
                        &agent->input_embeddings[p * agent->input_embeddings_n_embd],
                        agent->input_embeddings_n_embd * sizeof(float));
            batch.pos[p] = (llama_pos) p;
            batch.n_seq_id[p] = 1;
            batch.seq_id[p][0] = agent->seq_id;
            batch.logits[p] = (p + 1 == (size_t) n_emb_tok) ? 1 : 0;
        }
        batch.n_tokens = n_emb_tok;
        int ret = llama_decode(mas->ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) return ret;
        agent->prompt_pos = (size_t) n_emb_tok;

        llama_sampler * smpl = mas->samplers[i];
        llama_token next = llama_sampler_sample(smpl, mas->ctx, -1);
        llama_sampler_accept(smpl, next);
        agent->generated_tokens.push_back(next);
        agent->generated_len = 1;
        if (mas->stream_cb) {
            mas->stream_cb(i, next, mas->stream_step++, mas->stream_user_data);
        }
        if (llama_vocab_is_eog(vocab, next) || (int32_t) agent->generated_len >= agent->max_tokens) {
            agent->finished = true;
        }
    }

    // Prefill token agents, batched per compatible layer graph.
    {
        std::vector<int> prompt_pending;
        for (int i = 0; i < (int) mas->agents.size(); ++i) {
            auto & agent = mas->agents[i];
            if (!agent->input_embeddings.empty()) continue;
            if (agent->prompt_tokens.empty()) agent->finished = true;
            else prompt_pending.push_back(i);
        }

        while (!prompt_pending.empty()) {
            llama_layer_graph_context * group_layer_graph = mas->agents[prompt_pending.front()]->layer_graph;
            std::vector<int> group, remaining;
            int32_t n_tokens_total = 0;
            int32_t n_tokens_all_groups = 0;
            for (int idx : prompt_pending) {
                auto & agent = mas->agents[idx];
                n_tokens_all_groups += (int32_t) agent->prompt_tokens.size();
                if (agent->layer_graph == group_layer_graph) {
                    group.push_back(idx);
                    n_tokens_total += (int32_t) agent->prompt_tokens.size();
                } else {
                    remaining.push_back(idx);
                }
            }
            prompt_pending.swap(remaining);

            if (n_tokens_all_groups > (int32_t) llama_n_batch(mas->ctx)) {
                LLAMA_LOG_ERROR("%s: prefill %d tokens > n_batch %u\n",
                                __func__, n_tokens_all_groups, llama_n_batch(mas->ctx));
                return -8;
            }

            llama_context_set_layer_graph(mas->ctx, group_layer_graph);
            const llama_mas_adapter_weights * common_adapter = find_common_adapter(group, mas->agents);
            if (mas->use_cuda_adapters && common_adapter) mas->ctx->set_mas_adapter(common_adapter);
            else mas->ctx->clear_mas_adapter();

            llama_batch batch = llama_batch_init(n_tokens_total, 0, 1);
            int cursor = 0;
            std::vector<int> agent_last_cursor;
            agent_last_cursor.reserve(group.size());
            for (int idx : group) {
                auto & agent = mas->agents[idx];
                for (size_t p = 0; p < agent->prompt_tokens.size(); ++p) {
                    batch.token[cursor] = agent->prompt_tokens[p];
                    batch.pos[cursor] = (llama_pos) p;
                    batch.n_seq_id[cursor] = 1;
                    batch.seq_id[cursor][0] = agent->seq_id;
                    batch.logits[cursor] = (p + 1 == agent->prompt_tokens.size()) ? 1 : 0;
                    if (p + 1 == agent->prompt_tokens.size()) agent_last_cursor.push_back(cursor);
                    cursor++;
                }
            }
            batch.n_tokens = n_tokens_total;
            int ret = llama_decode(mas->ctx, batch);
            llama_batch_free(batch);
            if (ret != 0) return ret;

            for (size_t g = 0; g < group.size(); ++g) {
                int idx = group[g];
                auto & agent = mas->agents[idx];
                agent->prompt_pos = agent->prompt_tokens.size();
                llama_sampler * smpl = mas->samplers[idx];
                llama_token next = llama_sampler_sample(smpl, mas->ctx, agent_last_cursor[g]);
                llama_sampler_accept(smpl, next);
                agent->generated_tokens.push_back(next);
                agent->generated_len++;
                if (mas->stream_cb) {
                    mas->stream_cb(idx, next, mas->stream_step++, mas->stream_user_data);
                }
                if (llama_vocab_is_eog(vocab, next) || (int32_t) agent->generated_len >= agent->max_tokens) {
                    agent->finished = true;
                }
            }
        }
        mas->ctx->clear_mas_adapter();
    }

    if (std::all_of(mas->agents.begin(), mas->agents.end(),
                    [](const std::unique_ptr<llama_mas_agent> & a) { return a->stop_after_prompt; })) {
        return 0;
    }

    while (!llama_mas_all_finished(mas)) {
        int ret = llama_mas_decode_step(mas);
        if (ret != 0) return ret;
    }

    const double t_done = mas_time_ms();
    for (auto & agent : mas->agents) {
        if (agent->run_started_ms > 0.0 && agent->run_finished_ms <= 0.0) {
            agent->run_finished_ms = t_done;
        }
    }

    mas->ctx->clear_mas_adapter();
    mas_compute_and_log_timing(mas, t_run_start);
    return 0;
}

char * llama_mas_get_text(const llama_mas_context * mas, int32_t agent_idx) {
    if (!mas || agent_idx < 0 || agent_idx >= (int32_t)mas->agents.size()) return nullptr;
    const auto & agent = mas->agents[agent_idx];
    if (agent->generated_tokens.empty()) return nullptr;
    const llama_vocab * vocab = llama_model_get_vocab(mas->model);
    int32_t n_tokens = (int32_t)agent->generated_tokens.size();
    int32_t n_chars = llama_detokenize(vocab, agent->generated_tokens.data(), n_tokens, nullptr, 0, false, true);
    if (n_chars < 0) n_chars = -n_chars;
    if (n_chars == 0) return nullptr;
    std::string text;
    text.resize(n_chars);
    int32_t written = llama_detokenize(vocab, agent->generated_tokens.data(), n_tokens, &text[0], n_chars, false, true);
    if (written < 0) written = -written;
    if (written < n_chars) text.resize(written);
    // Strip special markers the model may emit at the end of generation
    const std::vector<std::string> markers = {"<turn|>", "<eos>", "<end_of_turn>", "<start_of_turn>"};
    for (const auto & marker : markers) {
        size_t pos = text.find(marker);
        while (pos != std::string::npos) {
            text.erase(pos, marker.size());
            pos = text.find(marker);
        }
    }
    // Trim trailing whitespace
    while (!text.empty() && (text.back() == ' ' || text.back() == '\n' || text.back() == '\r' || text.back() == '\t')) {
        text.pop_back();
    }
    char * result = (char *)malloc(text.size() + 1);
    if (!result) return nullptr;
    std::memcpy(result, text.c_str(), text.size() + 1);
    return result;
}

void llama_mas_free_text(char * text) {
    free(text);
}


extern "C" LLAMA_API char * llama_generate_single(
    llama_context * ctx,
    llama_model * model,
    const char * prompt,
    int32_t max_tokens,
    float temperature,
    float top_p,
    int32_t top_k,
    float repeat_penalty
) {
    if (!ctx || !model || !prompt || max_tokens <= 0) return nullptr;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = (int)llama_vocab_n_tokens(vocab);

    // Tokenize prompt
    std::vector<llama_token> prompt_tokens;
    int n = llama_tokenize(vocab, prompt, (int32_t)std::strlen(prompt), nullptr, 0, true, true);
    if (n < 0) n = -n;
    if (n > 0) {
        prompt_tokens.resize(n);
        llama_tokenize(vocab, prompt, (int32_t)std::strlen(prompt), prompt_tokens.data(), n, true, true);
    }

    // Clear the entire KV cache (not just seq_id 0).
    // After a MAS run, other agents' seq_ids may still hold cells,
    // causing "failed to find a memory slot" on the next decode.
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);

    // Decode prompt
    if (!prompt_tokens.empty()) {
        llama_batch batch = llama_batch_init((int32_t)prompt_tokens.size(), 0, 1);
        for (size_t p = 0; p < prompt_tokens.size(); ++p) {
            batch.token[p] = prompt_tokens[p];
            batch.pos[p] = (llama_pos)p;
            batch.n_seq_id[p] = 1;
            batch.seq_id[p][0] = 0;
            batch.logits[p] = (p + 1 == prompt_tokens.size()) ? 1 : 0;
        }
        batch.n_tokens = (int32_t)prompt_tokens.size();
        int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) return nullptr;
    }

    // Build llama.cpp optimized sampler chain (SIMD, efficient top-k/top-p)
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature == 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (repeat_penalty != 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
                -1, repeat_penalty, 0.0f, 0.0f));
        }
        if (top_k > 0 && top_k < n_vocab) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        }
        if (top_p > 0.0f && top_p < 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        }
        if (temperature > 0.0f && temperature != 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        }
        llama_sampler_chain_add(smpl, llama_sampler_init_dist((uint32_t)std::chrono::steady_clock::now().time_since_epoch().count()));
    }

    std::vector<llama_token> generated;
    generated.reserve(max_tokens);

    // Pre-allocate batch for single token decode
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    batch.n_tokens = 1;

    // Generate tokens
    for (int i = 0; i < max_tokens; ++i) {
        llama_token next_token = llama_sampler_sample(smpl, ctx, -1);
        generated.push_back(next_token);
        llama_sampler_accept(smpl, next_token);

        if (llama_vocab_is_eog(vocab, next_token)) break;

        batch.token[0] = next_token;
        batch.pos[0] = (llama_pos)(prompt_tokens.size() + i);
        int ret = llama_decode(ctx, batch);
        if (ret != 0) break;
    }
    llama_batch_free(batch);

    llama_sampler_free(smpl);

    if (generated.empty()) return nullptr;

    // Detokenize
    int32_t n_chars = llama_detokenize(vocab, generated.data(), (int32_t)generated.size(), nullptr, 0, false, true);
    if (n_chars < 0) n_chars = -n_chars;
    if (n_chars == 0) return nullptr;
    std::string text;
    text.resize(n_chars);
    int32_t written = llama_detokenize(vocab, generated.data(), (int32_t)generated.size(), &text[0], n_chars, false, true);
    if (written < 0) written = -written;
    if (written < n_chars) text.resize(written);

    char * result = (char *)malloc(text.size() + 1);
    if (!result) return nullptr;
    std::memcpy(result, text.c_str(), text.size() + 1);
    return result;
}

// Custom extension: set the active layer graph on the context (moved from llama-context.cpp for export)
extern "C" LLAMA_API void llama_context_set_layer_graph(llama_context * ctx, llama_layer_graph_context * lg) {
    if (ctx) ctx->set_layer_graph(lg);
}

// Same, but forces a scheduler re-reservation. Required when a partial layer
// subset is attached after the first decode: the cached buffer assignment was
// computed for the full-layer graph and cannot hold a different node count.
extern "C" LLAMA_API void llama_context_set_layer_graph_reserve(llama_context * ctx, llama_layer_graph_context * lg) {
    if (ctx) ctx->set_layer_graph(lg, true);
}

// Attach a layer graph, sizing the scheduler for the FULL layer set (worst
// case) so a later un-graphed decode still fits. Detach -> reserve -> attach.
extern "C" LLAMA_API int llama_context_attach_layer_graph_reserved(llama_context * ctx, llama_layer_graph_context * lg) {
    if (!ctx) return -1;
    ctx->set_layer_graph(nullptr, true);
    ctx->sched_reserve();
    ctx->set_layer_graph(lg);
    return 0;
}
