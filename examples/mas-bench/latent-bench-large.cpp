// Large-scale LatentMAS Benchmark
// Tests with more agents and higher latent steps

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

BenchmarkResult run_large_text_mas(const char* model_path, const char* task, int32_t max_tokens, int32_t n_agents) {
    BenchmarkResult result;
    result.name = "TextMAS_" + std::to_string(n_agents) + "agents";
    
    // Load model fresh for each test
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        std::cerr << "Failed to load model for TextMAS" << std::endl;
        return result;
    }
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 16384; // Larger context for more agents
    cparams.n_seq_max = 1;
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    
    llama_mas_context* mas = llama_mas_init(model, cparams, n_agents, false);
    if (!mas) {
        std::cerr << "Failed to init MAS for TextMAS" << std::endl;
        llama_free_model(model);
        return result;
    }
    
    // Add n_agents non-latent agents
    for (int32_t i = 0; i < n_agents; i++) {
        std::string role = "Agent_" + std::to_string(i);
        llama_mas_add_agent_flat(mas, role.c_str(), i, 1, 2048, 0.7f, 0.9f, 40, 0, 0, 0);
    }
    
    double t0 = get_time_ms();
    int ret = llama_mas_run(mas, task, max_tokens, LLAMA_MAS_STRATEGY_CHAIN);
    double t1 = get_time_ms();
    
    result.total_ms = t1 - t0;
    result.prefill_ms = 0;
    result.generation_ms = 0;
    result.total_tokens = 0;
    result.tokens_per_sec = 0;
    
    // Capture output text from all agents
    result.output_text = "";
    for (int agent_idx = 0; agent_idx < n_agents; agent_idx++) {
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

BenchmarkResult run_large_latent_mas(const char* model_path, const char* task, int32_t max_tokens, int32_t n_agents, int32_t latent_steps) {
    BenchmarkResult result;
    result.name = "LatentMAS_" + std::to_string(n_agents) + "agents_" + std::to_string(latent_steps) + "steps";
    
    // Load model fresh for each test
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        std::cerr << "Failed to load model for LatentMAS" << std::endl;
        return result;
    }
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 16384; // Larger context for more agents
    cparams.n_seq_max = 1;
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    
    llama_mas_context* mas = llama_mas_init(model, cparams, n_agents, false);
    if (!mas) {
        std::cerr << "Failed to init MAS for LatentMAS" << std::endl;
        llama_free_model(model);
        return result;
    }
    
    // Add n_agents: first (n_agents-1) are latent, last is text
    for (int32_t i = 0; i < n_agents; i++) {
        std::string role = "Agent_" + std::to_string(i);
        bool is_latent = (i < n_agents - 1);
        int32_t steps = is_latent ? latent_steps : 0;
        llama_mas_add_agent_flat(mas, role.c_str(), i, 1, is_latent ? 0 : 2048, 0.7f, 0.9f, 40, 0, is_latent ? 1 : 0, steps);
    }
    
    double t0 = get_time_ms();
    int ret = llama_mas_run(mas, task, max_tokens, LLAMA_MAS_STRATEGY_CHAIN);
    double t1 = get_time_ms();
    
    result.total_ms = t1 - t0;
    result.prefill_ms = 0;
    result.generation_ms = 0;
    result.total_tokens = 0;
    result.tokens_per_sec = 0;
    
    // Capture output text from the last agent (text agent)
    char* output = llama_mas_get_text(mas, n_agents - 1);
    if (output) {
        result.output_text = output;
        llama_mas_free_text(output);
    }
    
    llama_mas_free(mas);
    llama_free_model(model);
    return result;
}

void print_result(const BenchmarkResult& result, const BenchmarkResult& baseline) {
    std::cout << std::left << std::setw(35) << result.name << " | ";
    std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << result.total_ms << " ms | ";
    std::cout << std::setw(10) << result.tokens_per_sec << " t/s | ";
    
    if (baseline.total_ms > 0) {
        double speedup = baseline.total_ms / result.total_ms;
        std::cout << std::setw(8) << std::fixed << std::setprecision(2) << speedup << "x";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Redirect output to file
    std::ofstream logfile("latent-bench-large-output.txt");
    std::streambuf* cout_buf = std::cout.rdbuf();
    std::cout.rdbuf(logfile.rdbuf());
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <task>" << std::endl;
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* task = argv[2];
    int32_t max_tokens = argc > 3 ? atoi(argv[3]) : 4096;
    
    std::cout << "=== Large-Scale LatentMAS Benchmark ===" << std::endl;
    std::cout << "Task: " << task << std::endl;
    std::cout << "Max tokens: " << max_tokens << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::left << std::setw(35) << "Strategy" << " | ";
    std::cout << std::right << std::setw(10) << "Time (ms)" << " | ";
    std::cout << std::setw(10) << "Throughput" << " | ";
    std::cout << std::setw(8) << "Speedup" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    // Test with different numbers of agents
    std::vector<int32_t> agent_counts = {5, 10, 15};
    std::vector<int32_t> latent_steps_list = {100, 200, 400};
    
    BenchmarkResult baseline;
    
    // Run TextMAS baseline with 5 agents
    std::cout << "Running TextMAS baseline (5 agents)..." << std::endl;
    std::cout.flush();
    baseline = run_large_text_mas(model_path, task, max_tokens, 5);
    print_result(baseline, baseline);
    std::cout.flush();
    
    // Save baseline output
    std::ofstream baseline_file("textmas_5agents_output.txt");
    baseline_file << baseline.output_text;
    baseline_file.close();
    
    // Run LatentMAS with different agent counts and latent steps
    for (int32_t n_agents : agent_counts) {
        for (int32_t steps : latent_steps_list) {
            std::cout << "Running LatentMAS with " << n_agents << " agents, " << steps << " steps..." << std::endl;
            std::cout.flush();
            BenchmarkResult latent = run_large_latent_mas(model_path, task, max_tokens, n_agents, steps);
            print_result(latent, baseline);
            std::cout.flush();
            
            // Save output
            std::string filename = "latentmas_" + std::to_string(n_agents) + "agents_" + std::to_string(steps) + "steps_output.txt";
            std::ofstream latent_file(filename);
            latent_file << latent.output_text;
            latent_file.close();
        }
    }
    
    std::cout << std::string(70, '-') << std::endl;
    std::cout.flush();
    
    // Restore cout
    std::cout.rdbuf(cout_buf);
    logfile.close();
    
    // Print file contents to console
    std::ifstream infile("latent-bench-large-output.txt");
    std::cout << infile.rdbuf();
    
    return 0;
}
