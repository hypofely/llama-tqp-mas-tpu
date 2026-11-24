#pragma once

#include "llama.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

// In-graph MAS adapter weights (stored as ggml tensors on GPU)
struct llama_mas_adapter_weights_ggml {
    ggml_tensor * w_pre_ln = nullptr;
    ggml_tensor * b_pre_ln = nullptr;
    ggml_tensor * w_proj1 = nullptr;
    ggml_tensor * b_proj1 = nullptr;
    ggml_tensor * w_proj2 = nullptr;
    ggml_tensor * b_proj2 = nullptr;
    ggml_tensor * w_post_ln = nullptr;
    ggml_tensor * b_post_ln = nullptr;
    bool is_rms = true;
    int32_t n_embd = 0;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
};

struct llama_mas_adapter_weights; // forward declaration from llama-mas.h

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    void set_output_hidden_states(bool value);
    int32_t get_hidden_states_layer_count() const;
    float * get_hidden_states_layer(int32_t il);
    float * get_hidden_states_layer_ith(int32_t il, int32_t i);
    int32_t get_hidden_states_layer_n_outputs(int32_t il);

    // Layer-0 INPUT embedding capture (the space batch.embd enters at layer 0).
    float * get_inp_embd_last_token();
    // Mean-pool the layer-0 input embeddings over the last k positions into out.
    bool get_inp_embd_mean_pool(int32_t k, float * out, int32_t n_embd);
    int32_t get_inp_embd_n_outputs() const { return inp_embd_n_outputs; }
    int32_t get_inp_embd_row_size() const;



    // GPU-resident hidden state tensors (no CPU copy).

    // Valid between decode calls. Cleared on each decode.

    int32_t get_hidden_state_tensor_count() const;

    ggml_tensor * get_hidden_state_tensor(int32_t il);



    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    int32_t kv_cache_clone_to(llama_context * dst);
    int32_t kv_cache_clone_compatible_to(llama_context * dst);
    size_t  kv_cache_serialize(uint8_t * dst, size_t size);
    size_t  kv_cache_deserialize(const uint8_t * src, size_t size);
    size_t  kv_cache_deserialize_compatible(const uint8_t * src, size_t size);
    size_t  kv_cache_layer_export(int32_t il, int32_t flags, int32_t type, uint8_t * dst, size_t size);
    size_t  kv_cache_layer_import(const uint8_t * src, size_t size);

    void set_moe_router(llama_moe_router_fn router_fn, void * user_data);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<llama_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden states from all layers (2-dimensional array: [n_outputs][n_embd] per layer)
    // populated only when cparams.output_hidden_states is enabled
    std::vector<std::vector<float>> hidden_states;

    // Layer-0 INPUT capture (the space a token embedding lives in).
    //
    // The latent recurrence feeds a vector back through batch.embd, which enters
    // the graph at layer 0 - i.e. it must be in INPUT-embedding space. Capturing
    // the final layer's OUTPUT instead (hidden_states above) is a different space:
    // same width for most models, different meaning. Feeding an output-space vector
    // into the input slot is out-of-distribution and the model degrades as steps
    // accumulate. This buffer holds the post-embedding, pre-layer-0 tensor so a
    // latent step can feed back something layer 0 actually understands.
    std::vector<float> inp_embd;
    int32_t inp_embd_n_outputs = 0;

    // GPU-resident hidden state tensor pointers (no CPU copy)
    // Populated when output_hidden_states is enabled. Cleared on each decode.
    std::vector<ggml_tensor *> hidden_state_tensors;
    int32_t hidden_states_n_outputs = 0;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // Cached K-shift graph (avoids rebuilding on every shift)
    bool use_cached_shift_graph = false;
    llm_graph_result_ptr cached_shift_res;
    ggml_cgraph * cached_shift_gf = nullptr;

    // In-graph MAS adapter (injected into transformer forward pass)
    llama_mas_adapter_weights_ggml * mas_adapter = nullptr;

public:
    void set_use_cached_shift_graph(bool use) { use_cached_shift_graph = use; }
    bool get_use_cached_shift_graph() const { return use_cached_shift_graph; }

    // Cached shift graph accessors (used by llama_kv_cache)
    llm_graph_result * get_cached_shift_res() const { return cached_shift_res.get(); }
    void reset_cached_shift_res(int64_t max_nodes) { cached_shift_res.reset(new llm_graph_result(max_nodes)); }
    void reset_cached_shift_res() { if (cached_shift_res) cached_shift_res->reset(); }
    void set_cached_shift_res(llm_graph_result * res) { cached_shift_res.reset(res); }
    ggml_cgraph ** cached_shift_gf_ptr() { return &cached_shift_gf; }
    ggml_cgraph * get_cached_shift_gf() const { return cached_shift_gf; }
    void set_cached_shift_gf(ggml_cgraph * gf) { cached_shift_gf = gf; }

    // MAS adapter accessors
    void set_mas_adapter(const llama_mas_adapter_weights * weights);
    void clear_mas_adapter();
    llama_mas_adapter_weights_ggml * get_mas_adapter() const { return mas_adapter; }

private:
    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0;

    // MoE RL Router callback
    llama_moe_router_fn moe_router_fn = nullptr;
    void * moe_router_user_data = nullptr; // number of times the previous graph was reused
    // Layer graph for dynamic transformer editing (custom extension)

public:
    // set_layer_graph attaches a per-agent layer subset. `reserve` is a
    // CUSTOM fork extension (not upstream llama.cpp): when true, the next
    // sched_reserve() is forced. This is REQUIRED when the graph is attached
    // mid-process, because a layer subset changes the graph's node/leaf counts
    // and the buffer assignment computed at boot cannot hold the smaller (or
    // differently-shaped) graph - ggml_backend_sched_alloc_splits then fails
    // with GGML_ASSERT(buffer). Default false = upstream behaviour (attach
    // only, no re-reserve), which is correct when the graph is set once before
    // the first decode.
    void set_layer_graph(void * lg, bool reserve = false) {
        layer_graph_ctx = lg;
        if (reserve) {
            sched_need_reserve = true;
        }
    }

private:
    void * layer_graph_ctx = nullptr;
};
