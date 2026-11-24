// Test alignment matrix computation and dequantization
#include "llama.h"
#include "llama-mas.h"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];

    std::cout << "=== Alignment Matrix Test ===" << std::endl;
    std::cout << "Loading model: " << model_path << std::endl;

    // Load model
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_seq_max = 1;

    llama_mas_context* mas = llama_mas_init(model, cparams, 1, false);
    if (!mas) {
        std::cerr << "Failed to init MAS" << std::endl;
        llama_free_model(model);
        return 1;
    }

    // Test 1: Compute alignment matrix
    std::cout << "\nTest 1: Computing alignment matrix..." << std::endl;
    int ret = llama_mas_compute_alignment_matrix(mas);
    if (ret == 0) {
        std::cout << "  SUCCESS: Alignment matrix computed" << std::endl;
        std::cout << "  Alignment enabled: " << (llama_mas_get_use_alignment(mas) ? "yes" : "no") << std::endl;
    } else {
        std::cerr << "  FAILED: Could not compute alignment matrix" << std::endl;
    }

    // Test 2: Toggle alignment
    std::cout << "\nTest 2: Toggling alignment..." << std::endl;
    llama_mas_set_use_alignment(mas, false);
    std::cout << "  Disabled alignment: " << (llama_mas_get_use_alignment(mas) ? "yes" : "no") << std::endl;
    llama_mas_set_use_alignment(mas, true);
    std::cout << "  Re-enabled alignment: " << (llama_mas_get_use_alignment(mas) ? "yes" : "no") << std::endl;

    // Test 3: Check strategy values
    std::cout << "\nTest 3: Strategy enum values..." << std::endl;
    std::cout << "  LATENT_MAS: " << LLAMA_MAS_STRATEGY_LATENT_MAS << std::endl;
    std::cout << "  LATENT_MAS_NO_ALIGN: " << LLAMA_MAS_STRATEGY_LATENT_MAS_NO_ALIGN << std::endl;
    std::cout << "  LATENT_MAS_EMBEDDINGS: " << LLAMA_MAS_STRATEGY_LATENT_MAS_EMBEDDINGS << std::endl;

    llama_mas_free(mas);
    llama_free_model(model);

    std::cout << "\n=== All tests completed ===" << std::endl;
    return 0;
}
