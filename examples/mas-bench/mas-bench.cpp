#include "llama.h"
#include "llama-mas.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

static void print_usage(const char * prog) {
    printf("\nMAS CUDA Benchmark\n");
    printf("Usage: %s -m model.gguf [-n n_predict] [-a n_agents] [-ngl n_gpu_layers]\n", prog);
    printf("\nFlags:\n");
    printf("  -m <path>     Path to model GGUF (required)\n");
    printf("  -n <int>      Tokens to generate per agent (default: 32)\n");
    printf("  -a <int>      Number of agents (default: 4)\n");
    printf("  -ngl <int>    GPU layers (default: 99)\n");
    printf("  --no-batched  Disable batched decode\n");
    printf("  --no-cuda-adapters  Disable CUDA adapter forward\n");
    printf("  --no-backend-sampling  Disable backend (GPU) sampling\n");
}

static double elapsed_ms(const std::chrono::steady_clock::time_point & t0) {
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct bench_result {
    double setup_ms = 0;
    double prompt_ms = 0;
    double decode_ms = 0;
    double total_ms = 0;
    int tokens_generated = 0;
    double tok_per_sec = 0;
};

static bench_result run_mas_bench(
    const std::string & model_path,
    int n_agents,
    int n_predict,
    int ngl,
    bool batched,
    bool cuda_adapters,
    bool backend_sampling,
    bool tokenization_cache = true,
    bool cached_shift_graph = true
) {
    bench_result res;
    auto t0 = std::chrono::steady_clock::now();

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", model_path.c_str());
        return res;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 2048;
    cparams.n_seq_max = (uint32_t)n_agents;
    cparams.no_perf = true;

    llama_mas_context * mas = llama_mas_init(model, cparams, n_agents, batched);
    if (!mas) {
        fprintf(stderr, "llama_mas_init failed\n");
        llama_model_free(model);
        return res;
    }

    // Apply toggles
    // llama_mas_set_batched_decode(mas, batched); // Now set via init parameter
    llama_mas_set_cuda_adapters(mas, cuda_adapters);
    llama_mas_set_backend_sampling(mas, backend_sampling);
    llama_mas_set_tokenization_cache(mas, tokenization_cache);
    llama_mas_set_cached_shift_graph(mas, cached_shift_graph);

    // Create agents
    std::vector<std::string> roles = {"planner", "coder", "reviewer", "tester"};
    for (int i = 0; i < n_agents; ++i) {
        std::string role = (i < (int)roles.size()) ? roles[i] : ("agent_" + std::to_string(i));
        llama_mas_add_agent_flat(mas, role.c_str(), i, LLAMA_MAS_ADAPTER_NONE, n_predict, 0.7f, 0.9f, 40, 0, 0, 0);
    }

    res.setup_ms = elapsed_ms(t0);
    t0 = std::chrono::steady_clock::now();

    const char * task = "Write a Python function to sort a list of integers.";

    // Warm-up run (not measured)
    llama_mas_run(mas, task, 2, LLAMA_MAS_STRATEGY_BROADCAST);

    auto t_decode_start = std::chrono::steady_clock::now();
    int ret = llama_mas_run(mas, task, n_predict, LLAMA_MAS_STRATEGY_BROADCAST);
    auto t_decode_end = std::chrono::steady_clock::now();

    res.decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    res.total_ms = elapsed_ms(t0);

    // Count total tokens generated
    for (int i = 0; i < n_agents; ++i) {
        size_t count = 0;
        const llama_token * toks = llama_mas_get_tokens(mas, i, &count);
        if (toks) res.tokens_generated += (int)count;
    }

    if (res.decode_ms > 0) {
        res.tok_per_sec = res.tokens_generated / (res.decode_ms / 1000.0);
    }

    llama_mas_free(mas);
    llama_model_free(model);
    return res;
}

int main(int argc, char ** argv) {
    std::string model_path;
    int n_predict = 32;
    int n_agents = 4;
    int ngl = 99;
    bool batched = true;
    bool cuda_adapters = true;
    bool backend_sampling = true;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
            n_agents = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            ngl = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-batched") == 0) {
            batched = false;
        } else if (strcmp(argv[i], "--no-cuda-adapters") == 0) {
            cuda_adapters = false;
        } else if (strcmp(argv[i], "--no-backend-sampling") == 0) {
            backend_sampling = false;
        }
    }

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    printf("============================================================\n");
    printf("MAS CUDA Benchmark\n");
    printf("Model: %s\n", model_path.c_str());
    printf("Agents: %d  Tokens: %d  GPU layers: %d\n", n_agents, n_predict, ngl);
    printf("============================================================\n\n");

    // Baseline: all CUDA defaults
    printf("[RUN 1] CUDA defaults (all on)\n");
    bench_result r1 = run_mas_bench(model_path, n_agents, n_predict, ngl, batched, cuda_adapters, backend_sampling, true, true);
    printf("  Setup: %.1f ms | Decode: %.1f ms | Total: %.1f ms | Tokens: %d | Speed: %.2f tok/s\n\n",
           r1.setup_ms, r1.decode_ms, r1.total_ms, r1.tokens_generated, r1.tok_per_sec);

    // Comparison: disable batched decode
    printf("[RUN 2] Disable batched decode\n");
    bench_result r2 = run_mas_bench(model_path, n_agents, n_predict, ngl, false, cuda_adapters, backend_sampling, true, true);
    printf("  Setup: %.1f ms | Decode: %.1f ms | Total: %.1f ms | Tokens: %d | Speed: %.2f tok/s\n\n",
           r2.setup_ms, r2.decode_ms, r2.total_ms, r2.tokens_generated, r2.tok_per_sec);

    // Comparison: disable backend sampling
    printf("[RUN 3] Disable backend sampling\n");
    bench_result r3 = run_mas_bench(model_path, n_agents, n_predict, ngl, batched, cuda_adapters, false, true, true);
    printf("  Setup: %.1f ms | Decode: %.1f ms | Total: %.1f ms | Tokens: %d | Speed: %.2f tok/s\n\n",
           r3.setup_ms, r3.decode_ms, r3.total_ms, r3.tokens_generated, r3.tok_per_sec);

    // Comparison: disable tokenization cache
    printf("[RUN 4] Disable tokenization cache\n");
    bench_result r4 = run_mas_bench(model_path, n_agents, n_predict, ngl, batched, cuda_adapters, backend_sampling, false, true);
    printf("  Setup: %.1f ms | Decode: %.1f ms | Total: %.1f ms | Tokens: %d | Speed: %.2f tok/s\n\n",
           r4.setup_ms, r4.decode_ms, r4.total_ms, r4.tokens_generated, r4.tok_per_sec);

    // Comparison: disable cached shift graph
    printf("[RUN 5] Disable cached shift graph\n");
    bench_result r5 = run_mas_bench(model_path, n_agents, n_predict, ngl, batched, cuda_adapters, backend_sampling, true, false);
    printf("  Setup: %.1f ms | Decode: %.1f ms | Total: %.1f ms | Tokens: %d | Speed: %.2f tok/s\n\n",
           r5.setup_ms, r5.decode_ms, r5.total_ms, r5.tokens_generated, r5.tok_per_sec);

    // Comparison: all CPU fallback
    printf("[RUN 6] All CPU fallback\n");
    bench_result r6 = run_mas_bench(model_path, n_agents, n_predict, ngl, false, false, false, false, false);
    printf("  Setup: %.1f ms | Decode: %.1f ms | Total: %.1f ms | Tokens: %d | Speed: %.2f tok/s\n\n",
           r6.setup_ms, r6.decode_ms, r6.total_ms, r6.tokens_generated, r6.tok_per_sec);

    // Summary
    printf("============================================================\n");
    printf("SUMMARY\n");
    printf("============================================================\n");
    printf("%-25s %10s %10s %12s\n", "Config", "Decode(ms)", "Tok/Sec", "Speedup");
    printf("%-25s %10.1f %10.2f %12.2fx\n", "CUDA defaults", r1.decode_ms, r1.tok_per_sec, 1.0);
    printf("%-25s %10.1f %10.2f %12.2fx\n", "No batched", r2.decode_ms, r2.tok_per_sec, r2.tok_per_sec / r1.tok_per_sec);
    printf("%-25s %10.1f %10.2f %12.2fx\n", "No backend sampling", r3.decode_ms, r3.tok_per_sec, r3.tok_per_sec / r1.tok_per_sec);
    printf("%-25s %10.1f %10.2f %12.2fx\n", "No tokenization cache", r4.decode_ms, r4.tok_per_sec, r4.tok_per_sec / r1.tok_per_sec);
    printf("%-25s %10.1f %10.2f %12.2fx\n", "No cached shift graph", r5.decode_ms, r5.tok_per_sec, r5.tok_per_sec / r1.tok_per_sec);
    printf("%-25s %10.1f %10.2f %12.2fx\n", "All CPU fallback", r6.decode_ms, r6.tok_per_sec, r6.tok_per_sec / r1.tok_per_sec);
    printf("============================================================\n");

    return 0;
}
