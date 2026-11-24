// test-mas-api.cpp
// Lightweight MAS smoke test: validates struct layout, defaults, weights init/free,
// and CPU adapter forward paths (recursive/cross/mean-pool). No model required.
//
// Build: ninja test-mas-api
// Run:   bin\test-mas-api.exe
#include <iostream>
#include <cassert>
#include <cstring>
#include <cmath>
#include "llama-mas.h"

static int failures = 0;

static void check(bool cond, const char * msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << std::endl;
        ++failures;
    }
}

int main() {
    std::cout << "=== test-mas-api ===" << std::endl;

    // 1. Strategy enum completeness
    check(LLAMA_MAS_STRATEGY_ROLE == 0, "ROLE == 0");
    check(LLAMA_MAS_STRATEGY_CHAIN == 1, "CHAIN == 1");
    check(LLAMA_MAS_STRATEGY_LATENT_MAS == 5, "LATENT_MAS == 5");
    check(LLAMA_MAS_STRATEGY_LATENT_MAS_NO_ALIGN == 6, "LATENT_MAS_NO_ALIGN == 6");
    check(LLAMA_MAS_STRATEGY_LATENT_MAS_EMBEDDINGS == 7, "LATENT_MAS_EMBEDDINGS == 7");

    // 2. Adapter type enum
    check(LLAMA_MAS_ADAPTER_NONE == 0, "ADAPTER_NONE == 0");
    check(LLAMA_MAS_ADAPTER_RECURSIVE == 1, "ADAPTER_RECURSIVE == 1");
    check(LLAMA_MAS_ADAPTER_CROSS == 2, "ADAPTER_CROSS == 2");
    check(LLAMA_MAS_ADAPTER_MEAN == 3, "ADAPTER_MEAN == 3");
    check(LLAMA_MAS_ADAPTER_CAT == 4, "ADAPTER_CAT == 4");

    // 3. Weights init with n_embd=64
    llama_mas_adapter_weights * w = llama_mas_adapter_weights_init(64, 64, 64, 0);
    check(w != nullptr, "weights_init returns non-null");
    check(w->n_embd == 64, "weights n_embd == 64");
    check(w->in_dim == 64, "weights in_dim defaults to n_embd");
    check(w->out_dim == 64, "weights out_dim defaults to n_embd");
    check(w->hidden_dim == 2*64, "weights hidden_dim defaults to 2*out_dim");
    check(w->w_pre_ln  != nullptr, "w_pre_ln allocated");
    check(w->w_proj1   != nullptr, "w_proj1 allocated (n_embd^2)");
    check(w->w_post_ln != nullptr, "w_post_ln allocated");
    // Identity on diagonal
    bool identity_ok = true;
    for (int i = 0; i < 64; ++i) {
        if (w->w_pre_ln[i]  != 1.0f) identity_ok = false;
        if (w->w_proj1[i*64 + i] != 1.0f) identity_ok = false;
        if (w->w_post_ln[i] != 1.0f) identity_ok = false;
    }
    check(identity_ok, "weights default to identity on diagonal");

    // Don't leak — free here, we don't need it alive for the next sections
    llama_mas_adapter_weights_free(w);
    w = nullptr;

    // 4. Null safety
    llama_mas_adapter_weights_free(nullptr);
    check(true, "weights_free(nullptr) does not crash");

    // 5. RecursiveLink forward ran without crashing and produced finite outputs
    {
        llama_mas_adapter_weights w_r{};
        w_r.n_embd = 8;
        w_r.in_dim = 8; w_r.out_dim = 8; w_r.hidden_dim = 16;
        w_r.w_pre_ln  = (float*)calloc(8, sizeof(float));
        w_r.b_pre_ln  = (float*)calloc(8, sizeof(float));
        w_r.w_proj1   = (float*)calloc(64, sizeof(float));
        w_r.b_proj1   = (float*)calloc(8, sizeof(float));
        w_r.w_proj2   = (float*)calloc(64, sizeof(float));
        w_r.b_proj2   = (float*)calloc(8, sizeof(float));
        w_r.w_post_ln = (float*)calloc(8, sizeof(float));
        w_r.b_post_ln = (float*)calloc(8, sizeof(float));
        for (int i = 0; i < 8; ++i) {
            w_r.w_pre_ln[i]  = 1.0f;
            w_r.w_post_ln[i] = 1.0f;
            w_r.w_proj1[i*8 + i] = 1.0f;
            w_r.w_proj2[i*8 + i] = 1.0f;
        }
        float in[8];  for (int i = 0; i < 8; ++i) in[i]  = float(i);
        float out[8] = {0};
        llama_mas_adapter_recursive_forward(&w_r, in, out);
        bool finite = true;
        for (int i = 0; i < 8; ++i) {
            if (!std::isfinite(out[i])) finite = false;
        }
        check(finite, "recursive_forward produces finite output");
        // stack-allocated weights — free internal arrays manually (not the struct)
        free(w_r.w_pre_ln);  free(w_r.b_pre_ln);
        free(w_r.w_proj1);   free(w_r.b_proj1);
        free(w_r.w_proj2);   free(w_r.b_proj2);
        free(w_r.w_post_ln); free(w_r.b_post_ln);
    }

    // 6. Mean pool correctness
    {
        // 3 tokens x 4 dims
        float states[12] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        float pooled[4]  = {0};
        llama_mas_adapter_mean_pool(states, 3, 4, pooled);
        bool ok = (pooled[0]==5.0f && pooled[1]==6.0f && pooled[2]==7.0f && pooled[3]==8.0f);
        check(ok, "mean_pool averages correctly (3 tokens x 4)");
    }

    // 7. CrossLink forward with zeroed weights => zero output (layernorm eps prevents nan)
    {
        llama_mas_adapter_weights w_c{};
        w_c.in_dim    = 4;
        w_c.out_dim   = 3;
        w_c.hidden_dim = 6;
        w_c.w_cross_ln_source   = (float*)calloc(4, sizeof(float));
        w_c.b_cross_ln_source   = (float*)calloc(4, sizeof(float));
        w_c.w_cross_proj1       = (float*)calloc(24, sizeof(float)); // 6*4
        w_c.b_cross_proj1       = (float*)calloc(6, sizeof(float));
        w_c.w_cross_proj2       = (float*)calloc(18, sizeof(float)); // 3*6
        w_c.b_cross_proj2       = (float*)calloc(3, sizeof(float));
        w_c.w_cross_residual    = (float*)calloc(12, sizeof(float)); // 3*4
        w_c.b_cross_residual    = (float*)calloc(3, sizeof(float));
        w_c.w_cross_ln_target   = (float*)calloc(3, sizeof(float));
        w_c.b_cross_ln_target   = (float*)calloc(3, sizeof(float));

        float in[4] = {1,2,3,4};
        float out[3] = {99,99,99}; // sentinel
        llama_mas_adapter_cross_forward(&w_c, in, out);
        bool ok = (out[0] == 0.0f && out[1] == 0.0f && out[2] == 0.0f);
        check(ok, "cross_forward zeroed weights -> zero output");
        // manual free — stack-allocated struct
        free(w_c.w_cross_ln_source);  free(w_c.b_cross_ln_source);
        free(w_c.w_cross_proj1);      free(w_c.b_cross_proj1);
        free(w_c.w_cross_proj2);      free(w_c.b_cross_proj2);
        free(w_c.w_cross_residual);   free(w_c.b_cross_residual);
        free(w_c.w_cross_ln_target);  free(w_c.b_cross_ln_target);
    }

    // 8. Recursive forward with zeroed weights => ~0 (layernorm eps)
    {
        llama_mas_adapter_weights w_z{};
        w_z.n_embd = 4;
        w_z.in_dim = 4; w_z.out_dim = 4; w_z.hidden_dim = 8;
        w_z.w_pre_ln  = (float*)calloc(4, sizeof(float));
        w_z.b_pre_ln  = (float*)calloc(4, sizeof(float));
        w_z.w_proj1   = (float*)calloc(16, sizeof(float));
        w_z.b_proj1   = (float*)calloc(4, sizeof(float));
        w_z.w_proj2   = (float*)calloc(16, sizeof(float));
        w_z.b_proj2   = (float*)calloc(4, sizeof(float));
        w_z.w_post_ln = (float*)calloc(4, sizeof(float));
        w_z.b_post_ln = (float*)calloc(4, sizeof(float));

        float in[4] = {1,2,3,4};
        float out[4] = {99,99,99,99};
        llama_mas_adapter_recursive_forward(&w_z, in, out);
        bool ok = true;
        for (int i = 0; i < 4; ++i) if (out[i] != 0.0f) ok = false;
        check(ok, "recursive_forward zeroed weights -> zero output");
        // manual free — stack-allocated struct (don't call llama_mas_adapter_weights_free on stack memory)
        free(w_z.w_pre_ln); free(w_z.b_pre_ln); free(w_z.w_proj1); free(w_z.b_proj1);
        free(w_z.w_proj2); free(w_z.b_proj2); free(w_z.w_post_ln); free(w_z.b_post_ln);
    }

    std::cout << "\n=== result: " << failures << " failure(s) ===" << std::endl;
    return failures ? 1 : 0;
}
