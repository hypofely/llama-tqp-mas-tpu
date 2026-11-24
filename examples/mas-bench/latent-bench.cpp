// LatentMAS vs TextMAS Performance Benchmark
// Compares latent step batching against standard text generation

#include "llama.h"
#include "llama-mas.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>

struct BenchmarkResult {
    std::string name;
    double total_ms;
    double prefill_ms;
    double generation_ms;
    int64_t total_tokens;
    double tokens_per_sec;
    std::string output_text;
};

double get_time_ms() {
    using namespace std::chrono;
    return static_cast<double>(duration_cast<microseconds>(
        steady_clock::now().time_since_epoch()).count()) / 1000.0;
}

BenchmarkResult run_text_mas(const char* model_path, const char* task, int32_t max_tokens) {
    BenchmarkResult result;
    result.name = "TextMAS";
    
    // Load model fresh for each test
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        std::cerr << "Failed to load model for TextMAS" << std::endl;
        return result;
    }
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 8192;
    cparams.n_seq_max = 1; // CHAIN strategy processes agents sequentially
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    
    llama_mas_context* mas = llama_mas_init(model, cparams, 3, false); // CHAIN doesn't use batched decode
    if (!mas) {
        std::cerr << "Failed to init MAS for TextMAS" << std::endl;
        llama_free_model(model);
        return result;
    }
    
    // Enable CUDA optimizations
    llama_mas_set_backend_sampling(mas, true); // Move sampling to GPU
    llama_mas_set_gpu_hidden_states(mas, true); // Keep hidden states on GPU
    
    // Add 3 non-latent agents (standard text generation)
    llama_mas_add_agent_flat(mas, "Planner", 0, 1, 1024, 0.7f, 0.9f, 40, 0, 0, 0);
    llama_mas_add_agent_flat(mas, "Executor", 1, 1, 4096, 0.15f, 0.9f, 20, 0, 0, 0);
    llama_mas_add_agent_flat(mas, "Outer", 2, 1, 2048, 0.7f, 0.9f, 40, 0, 0, 0);
    
    double t0 = get_time_ms();
    int ret = llama_mas_run(mas, task, max_tokens, LLAMA_MAS_STRATEGY_CHAIN);
    double t1 = get_time_ms();
    
    result.total_ms = t1 - t0;
    result.prefill_ms = 0; // Can't access internal timing without full definition
    result.generation_ms = 0;
    result.total_tokens = 0;
    result.tokens_per_sec = 0;
    
    // Capture output text from all agents in the chain
    result.output_text = "";
    for (int agent_idx = 0; agent_idx < 3; agent_idx++) {
        char* output = llama_mas_get_text(mas, agent_idx);
        if (output) {
            result.output_text += output;
            result.output_text += "\n\n=== AGENT " + std::to_string(agent_idx) + " OUTPUT ===\n\n";
            llama_mas_free_text(output);
        }
    }
    
    llama_mas_free(mas);
    llama_free_model(model);
    return result;
}

BenchmarkResult run_latent_mas(const char* model_path, const char* task, int32_t max_tokens, int32_t latent_steps) {
    BenchmarkResult result;
    result.name = "LatentMAS_" + std::to_string(latent_steps) + "steps";
    
    // Load model fresh for each test
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        std::cerr << "Failed to load model for LatentMAS" << std::endl;
        return result;
    }
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 8192;
    cparams.n_seq_max = 1; // CHAIN strategy processes agents sequentially
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    
    llama_mas_context* mas = llama_mas_init(model, cparams, 3, false); // CHAIN doesn't use batched decode
    if (!mas) {
        std::cerr << "Failed to init MAS for LatentMAS" << std::endl;
        llama_free_model(model);
        return result;
    }
    
    // Enable CUDA optimizations
    llama_mas_set_backend_sampling(mas, true); // Move sampling to GPU
    llama_mas_set_gpu_hidden_states(mas, true); // Keep hidden states on GPU
    
    // Add 2 latent agents + 1 text agent
    llama_mas_add_agent_flat(mas, "Planner", 0, 1, 0, 0.7f, 0.9f, 40, 0, 1, latent_steps);
    llama_mas_add_agent_flat(mas, "Executor", 1, 1, 0, 0.15f, 0.9f, 20, 0, 1, latent_steps);
    llama_mas_add_agent_flat(mas, "Outer", 2, 1, 2048, 0.7f, 0.9f, 40, 0, 0, 0);
    
    double t0 = get_time_ms();
    int ret = llama_mas_run(mas, task, max_tokens, LLAMA_MAS_STRATEGY_CHAIN);
    double t1 = get_time_ms();
    
    result.total_ms = t1 - t0;
    result.prefill_ms = 0; // Can't access internal timing without full definition
    result.generation_ms = 0;
    result.total_tokens = 0;
    result.tokens_per_sec = 0;
    
    // Capture output text from the last agent (Outer)
    char* output = llama_mas_get_text(mas, 2); // Agent 2 is Outer
    if (output) {
        result.output_text = output;
        llama_mas_free_text(output);
    }
    
    llama_mas_free(mas);
    llama_free_model(model);
    return result;
}

void print_result(const BenchmarkResult& result, const BenchmarkResult& baseline) {
    std::cout << std::left << std::setw(25) << result.name << " | ";
    std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << result.total_ms << " ms | ";
    std::cout << std::setw(10) << result.tokens_per_sec << " t/s | ";
    
    if (baseline.total_ms > 0) {
        double speedup = baseline.total_ms / result.total_ms;
        std::cout << std::setw(8) << std::fixed << std::setprecision(2) << speedup << "x";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Redirect output to both console and file
    std::ofstream logfile("latent-bench-output.txt");
    std::streambuf* cout_buf = std::cout.rdbuf();
    std::cout.rdbuf(logfile.rdbuf());
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <task>" << std::endl;
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* task = argv[2];
    int32_t max_tokens = argc > 3 ? atoi(argv[3]) : 4096;
    
    std::cout << "=== LatentMAS vs TextMAS Benchmark ===" << std::endl;
    std::cout << "Task: " << task << std::endl;
    std::cout << "Max tokens: " << max_tokens << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::left << std::setw(25) << "Strategy" << " | ";
    std::cout << std::right << std::setw(10) << "Time (ms)" << " | ";
    std::cout << std::setw(10) << "Throughput" << " | ";
    std::cout << std::setw(8) << "Speedup" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    // Run TextMAS baseline
    std::cout << "Running TextMAS baseline..." << std::endl;
    std::cout.flush();
    BenchmarkResult textmas = run_text_mas(model_path, task, max_tokens);
    print_result(textmas, textmas);
    std::cout.flush();
    
    // Save TextMAS output
    std::ofstream textmas_file("textmas_output.txt");
    textmas_file << textmas.output_text;
    textmas_file.close();
    
    // Run LatentMAS with different step counts
    std::vector<int32_t> latent_steps_list = {20, 40, 60, 80};
    for (int32_t steps : latent_steps_list) {
        std::cout << "Running LatentMAS with " << steps << " steps..." << std::endl;
        std::cout.flush();
        BenchmarkResult latent = run_latent_mas(model_path, task, max_tokens, steps);
        print_result(latent, textmas);
        std::cout.flush();
        
        // Save LatentMAS output
        std::string filename = "latentmas_" + std::to_string(steps) + "_output.txt";
        std::ofstream latent_file(filename);
        latent_file << latent.output_text;
        latent_file.close();
    }
    
    std::cout << std::string(70, '-') << std::endl;
    std::cout.flush();
    
    // Restore cout
    std::cout.rdbuf(cout_buf);
    logfile.close();
    
    // Print file contents to console
    std::ifstream infile("latent-bench-output.txt");
    std::cout << infile.rdbuf();
    
    return 0;
}
