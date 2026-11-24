#include "llama-layer-graph.h"
#include "llama-context.h"
#include "llama-model.h"

#include <cstring>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <sstream>

// ---------------------------------------------------------------------------
// Internal layer graph context
// ---------------------------------------------------------------------------

struct llama_layer_graph_context {
    llama_context      * ctx;
    llama_layer_graph    graph;
    bool                 capture_enabled;
    std::vector<llama_hidden_state_snapshot> snapshots;
    std::vector<double>  timings_us;  // per-node execution time
    int32_t              n_layers_orig;

    // KV-position tracking for partial / sequential execution
    llama_pos            kv_pos;       // next position to write into KV cache
    llama_seq_id         seq_id;       // sequence ID used for KV storage

    llama_layer_graph_context() : ctx(nullptr), capture_enabled(false), n_layers_orig(0),
                                   kv_pos(0), seq_id(0) {
        graph.nodes = nullptr;
        graph.n_nodes = 0;
        graph.entry = 0;
        graph.n_layers_orig = 0;
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void free_graph(llama_layer_graph & g) {
    if (g.nodes) {
        free(g.nodes);
        g.nodes = nullptr;
    }
    g.n_nodes = 0;
    g.entry = 0;
}

static void free_snapshot(llama_hidden_state_snapshot & snap) {
    if (snap.data) {
        free(snap.data);
        snap.data = nullptr;
    }
}

static int build_default_graph(llama_layer_graph_context * lg) {
    free_graph(lg->graph);

    int32_t n = lg->n_layers_orig;
    lg->graph.nodes = (llama_layer_node *)calloc(n, sizeof(llama_layer_node));
    if (!lg->graph.nodes) return -1;

    lg->graph.n_nodes = n;
    lg->graph.entry = 0;
    lg->graph.n_layers_orig = n;

    for (int32_t i = 0; i < n; i++) {
        lg->graph.nodes[i].layer_id    = i;
        lg->graph.nodes[i].enabled     = true;
        lg->graph.nodes[i].instance_id = i;
        lg->graph.nodes[i].next_count  = (i < n - 1) ? 1 : 0;
        if (i < n - 1) {
            lg->graph.nodes[i].next[0] = i + 1;
        }
    }

    // Allocate timing array
    lg->timings_us.resize(n, 0.0);

    return 0;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

llama_layer_graph_context * llama_layer_graph_init(llama_context * ctx) {
    if (!ctx) return nullptr;

    auto * lg = new llama_layer_graph_context();
    lg->ctx = ctx;

    const llama_model * model = llama_get_model(ctx);
    lg->n_layers_orig = llama_model_n_layer(model);

    if (build_default_graph(lg) != 0) {
        delete lg;
        return nullptr;
    }

    return lg;
}

void llama_layer_graph_free(llama_layer_graph_context * lg) {
    if (!lg) return;
    free_graph(lg->graph);
    for (auto & snap : lg->snapshots) free_snapshot(snap);
    lg->snapshots.clear();
    delete lg;
}

// ---------------------------------------------------------------------------
// Graph Inspection
// ---------------------------------------------------------------------------

const llama_layer_graph * llama_layer_graph_get(const llama_layer_graph_context * lg) {
    return lg ? &lg->graph : nullptr;
}

int32_t llama_layer_graph_n_layers(const llama_layer_graph_context * lg) {
    return lg ? lg->n_layers_orig : 0;
}

// ---------------------------------------------------------------------------
// Graph Mutation
// ---------------------------------------------------------------------------

void llama_layer_graph_reset(llama_layer_graph_context * lg) {
    if (!lg) return;
    build_default_graph(lg);
}

int llama_layer_graph_set_enabled(llama_layer_graph_context * lg, int32_t instance_id, bool enabled) {
    if (!lg || !lg->graph.nodes) return -1;
    for (int32_t i = 0; i < lg->graph.n_nodes; i++) {
        if (lg->graph.nodes[i].instance_id == instance_id) {
            lg->graph.nodes[i].enabled = enabled;
            return 0;
        }
    }
    return -1;
}

int llama_layer_graph_reorder(llama_layer_graph_context * lg, int32_t from_index, int32_t to_index) {
    if (!lg || !lg->graph.nodes) return -1;
    if (from_index < 0 || from_index >= lg->graph.n_nodes) return -1;
    if (to_index < 0 || to_index >= lg->graph.n_nodes) return -1;
    if (from_index == to_index) return 0;

    // Save the node being moved
    llama_layer_node moved = lg->graph.nodes[from_index];

    if (from_index < to_index) {
        // Shift left
        for (int32_t i = from_index; i < to_index; i++) {
            lg->graph.nodes[i] = lg->graph.nodes[i + 1];
        }
    } else {
        // Shift right
        for (int32_t i = from_index; i > to_index; i--) {
            lg->graph.nodes[i] = lg->graph.nodes[i - 1];
        }
    }
    lg->graph.nodes[to_index] = moved;

    // Rebuild next pointers
    for (int32_t i = 0; i < lg->graph.n_nodes; i++) {
        lg->graph.nodes[i].next_count = (i < lg->graph.n_nodes - 1) ? 1 : 0;
        if (i < lg->graph.n_nodes - 1) {
            lg->graph.nodes[i].next[0] = i + 1;
        }
    }

    return 0;
}

int32_t llama_layer_graph_duplicate(llama_layer_graph_context * lg, int32_t instance_id, int32_t position) {
    if (!lg || !lg->graph.nodes) return -1;

    // Find the source node
    int32_t src_idx = -1;
    for (int32_t i = 0; i < lg->graph.n_nodes; i++) {
        if (lg->graph.nodes[i].instance_id == instance_id) {
            src_idx = i;
            break;
        }
    }
    if (src_idx < 0) return -1;

    int32_t new_n = lg->graph.n_nodes + 1;
    auto * new_nodes = (llama_layer_node *)realloc(lg->graph.nodes, new_n * sizeof(llama_layer_node));
    if (!new_nodes) return -1;
    lg->graph.nodes = new_nodes;

    // Find max instance_id for new unique id
    int32_t max_inst = 0;
    for (int32_t i = 0; i < lg->graph.n_nodes; i++) {
        if (lg->graph.nodes[i].instance_id > max_inst) max_inst = lg->graph.nodes[i].instance_id;
    }

    int32_t insert_pos = (position >= 0 && position < lg->graph.n_nodes) ? position : lg->graph.n_nodes;

    // Shift nodes right
    for (int32_t i = lg->graph.n_nodes; i > insert_pos; i--) {
        lg->graph.nodes[i] = lg->graph.nodes[i - 1];
    }

    // Copy source node
    lg->graph.nodes[insert_pos] = lg->graph.nodes[src_idx > insert_pos ? src_idx + 1 : src_idx];
    lg->graph.nodes[insert_pos].instance_id = max_inst + 1;

    lg->graph.n_nodes = new_n;

    // Rebuild next pointers
    for (int32_t i = 0; i < lg->graph.n_nodes; i++) {
        lg->graph.nodes[i].next_count = (i < lg->graph.n_nodes - 1) ? 1 : 0;
        if (i < lg->graph.n_nodes - 1) {
            lg->graph.nodes[i].next[0] = i + 1;
        }
    }

    // Resize timings
    lg->timings_us.resize(new_n, 0.0);

    return lg->graph.nodes[insert_pos].instance_id;
}

int llama_layer_graph_remove(llama_layer_graph_context * lg, int32_t instance_id) {
    if (!lg || !lg->graph.nodes || lg->graph.n_nodes <= 1) return -1;

    int32_t rm_idx = -1;
    for (int32_t i = 0; i < lg->graph.n_nodes; i++) {
        if (lg->graph.nodes[i].instance_id == instance_id) {
            rm_idx = i;
            break;
        }
    }
    if (rm_idx < 0) return -1;

    // Shift left
    for (int32_t i = rm_idx; i < lg->graph.n_nodes - 1; i++) {
        lg->graph.nodes[i] = lg->graph.nodes[i + 1];
    }

    lg->graph.n_nodes--;

    // Rebuild next pointers
    for (int32_t i = 0; i < lg->graph.n_nodes; i++) {
        lg->graph.nodes[i].next_count = (i < lg->graph.n_nodes - 1) ? 1 : 0;
        if (i < lg->graph.n_nodes - 1) {
            lg->graph.nodes[i].next[0] = i + 1;
        }
    }

    lg->timings_us.resize(lg->graph.n_nodes);

    return 0;
}

// ---------------------------------------------------------------------------
// JSON import/export
// ---------------------------------------------------------------------------

int llama_layer_graph_load_json(llama_layer_graph_context * lg, const char * json) {
    if (!lg || !json) return -1;

    // Simple JSON parser for our format:
    // {"layers":[{"id":0,"enabled":true},...],"order":[0,1,2,3]}

    std::string s(json);

    // Parse "order" array
    auto order_pos = s.find("\"order\"");
    if (order_pos == std::string::npos) return -1;

    auto bracket_start = s.find('[', order_pos);
    auto bracket_end   = s.find(']', bracket_start);
    if (bracket_start == std::string::npos || bracket_end == std::string::npos) return -1;

    std::string order_str = s.substr(bracket_start + 1, bracket_end - bracket_start - 1);

    // Parse comma-separated integers
    std::vector<int32_t> order;
    std::istringstream iss(order_str);
    std::string token;
    while (std::getline(iss, token, ',')) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t\n\r"));
        token.erase(token.find_last_not_of(" \t\n\r") + 1);
        if (!token.empty()) {
            order.push_back(std::atoi(token.c_str()));
        }
    }

    if (order.empty()) return -1;

    // Parse "layers" array for enabled flags
    std::unordered_map<int32_t, bool> enabled_map;
    auto layers_pos = s.find("\"layers\"");
    if (layers_pos != std::string::npos) {
        // Find each layer object: {"id":N,"enabled":true/false}
        size_t pos = layers_pos;
        while (true) {
            auto id_pos = s.find("\"id\"", pos);
            if (id_pos == std::string::npos || id_pos > bracket_start) break;

            auto id_colon = s.find(':', id_pos);
            if (id_colon == std::string::npos) break;
            auto id_val_start = s.find_first_not_of(" \t\n\r", id_colon + 1);
            int32_t layer_id = std::atoi(s.c_str() + id_val_start);

            auto en_pos = s.find("\"enabled\"", id_pos);
            if (en_pos != std::string::npos && en_pos < bracket_start) {
                auto en_colon = s.find(':', en_pos);
                auto en_val_start = s.find_first_not_of(" \t\n\r", en_colon + 1);
                bool en = (s.substr(en_val_start, 4) == "true");
                enabled_map[layer_id] = en;
            }

            pos = en_pos > id_pos ? en_pos + 8 : id_pos + 8;
        }
    }

    // Rebuild graph from order
    free_graph(lg->graph);

    int32_t n = (int32_t)order.size();
    lg->graph.nodes = (llama_layer_node *)calloc(n, sizeof(llama_layer_node));
    if (!lg->graph.nodes) return -1;

    lg->graph.n_nodes = n;
    lg->graph.entry = 0;
    lg->graph.n_layers_orig = lg->n_layers_orig;

    for (int32_t i = 0; i < n; i++) {
        int32_t lid = order[i];
        lg->graph.nodes[i].layer_id    = lid;
        lg->graph.nodes[i].instance_id = i;
        lg->graph.nodes[i].next_count  = (i < n - 1) ? 1 : 0;
        if (i < n - 1) {
            lg->graph.nodes[i].next[0] = i + 1;
        }

        // Check enabled map
        auto it = enabled_map.find(lid);
        lg->graph.nodes[i].enabled = (it != enabled_map.end()) ? it->second : true;
    }

    lg->timings_us.resize(n, 0.0);

    return 0;
}

char * llama_layer_graph_export_json(const llama_layer_graph_context * lg) {
    if (!lg || !lg->graph.nodes) return nullptr;

    std::ostringstream oss;
    oss << "{\"layers\":[";

    // Build unique layer entries
    std::unordered_map<int32_t, bool> seen;
    bool first_layer = true;
    for (int32_t i = 0; i < lg->graph.n_nodes; i++) {
        int32_t lid = lg->graph.nodes[i].layer_id;
        if (seen.find(lid) != seen.end()) continue;
        seen[lid] = true;

        if (!first_layer) oss << ",";
        first_layer = false;
        oss << "{\"id\":" << lid
            << ",\"enabled\":" << (lg->graph.nodes[i].enabled ? "true" : "false")
            << "}";
    }

    oss << "],\"order\":[";
    for (int32_t i = 0; i < lg->graph.n_nodes; i++) {
        if (i > 0) oss << ",";
        oss << lg->graph.nodes[i].layer_id;
    }
    oss << "]}";

    std::string result = oss.str();
    char * c_result = (char *)malloc(result.size() + 1);
    if (!c_result) return nullptr;
    std::memcpy(c_result, result.c_str(), result.size() + 1);
    return c_result;
}

void llama_layer_graph_free_json(char * json) {
    free(json);
}

// ---------------------------------------------------------------------------
// Execution — graph-aware forward pass
// ---------------------------------------------------------------------------

// This is the core: we hook into the model's layer loop.
// The llama_context stores a pointer to the active layer graph context.
// During graph building (in models/llama.cpp), we check this pointer
// and iterate through graph nodes instead of the normal layer loop.
//
// For now, we store the graph context on the llama_context via a
// custom field. The actual graph-aware execution is implemented in
// the model files (see models/llama.cpp modifications).

// Set the active layer graph on the context (called before decode)
static void set_active_layer_graph(llama_context * ctx, llama_layer_graph_context * lg) {
    // Store as user data via the context's custom pointer
    // We use a known offset in the context structure
    // This is a custom extension to llama.cpp
    extern void llama_context_set_layer_graph(llama_context * ctx, llama_layer_graph_context * lg);
    llama_context_set_layer_graph(ctx, lg);
}

int llama_layer_graph_forward(llama_layer_graph_context * lg, const llama_token * tokens, int32_t n_tokens) {
    if (!lg || !lg->ctx || !tokens || n_tokens <= 0) return -1;

    uint32_t n_ctx = llama_n_ctx(lg->ctx);
    if (lg->kv_pos + n_tokens > (llama_pos)n_ctx) {
        LLAMA_LOG_ERROR("%s: kv_pos (%d) + n_tokens (%d) exceeds context size (%u). "
                        "Call llama_layer_graph_reset_kv_pos(lg, 0) to start a fresh sequence.\n",
                        __func__, lg->kv_pos, n_tokens, n_ctx);
        return -2;
    }

    // Set the active graph on the context
    set_active_layer_graph(lg->ctx, lg);

    if (lg->capture_enabled) {
        llama_set_output_hidden_states(lg->ctx, true);
    }

    // Build a batch and decode
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = (llama_pos)(lg->kv_pos + i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = lg->seq_id;
        batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    auto t_start = std::chrono::high_resolution_clock::now();
    int ret = llama_decode(lg->ctx, batch);
    auto t_end = std::chrono::high_resolution_clock::now();

    llama_batch_free(batch);

    // Record total timing (per-node timings are recorded in the model hook)
    if (ret == 0 && !lg->timings_us.empty()) {
        double total_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
        // Distribute evenly across nodes for now (per-node timing requires model hooks)
        double per_node = total_us / lg->graph.n_nodes;
        for (size_t i = 0; i < lg->timings_us.size(); i++) {
            lg->timings_us[i] = per_node;
        }
    }

    // Advance KV position so the next forward call appends correctly
    if (ret == 0) {
        lg->kv_pos += n_tokens;
    }

    // Clear active graph
    set_active_layer_graph(lg->ctx, nullptr);

    return ret;
}

void llama_layer_graph_reset_kv_pos(llama_layer_graph_context * lg, llama_pos pos) {
    if (!lg) return;
    lg->kv_pos = pos;
}

void llama_layer_graph_set_seq_id(llama_layer_graph_context * lg, llama_seq_id seq_id) {
    if (!lg) return;
    lg->seq_id = seq_id;
}

// ---------------------------------------------------------------------------
// Hidden State Inspector
// ---------------------------------------------------------------------------

void llama_layer_graph_set_capture(llama_layer_graph_context * lg, bool capture_enabled) {
    if (lg) lg->capture_enabled = capture_enabled;
}

int32_t llama_layer_graph_get_snapshot_count(const llama_layer_graph_context * lg) {
    return lg ? (int32_t)lg->snapshots.size() : 0;
}

const float * llama_layer_graph_get_snapshot(
    const llama_layer_graph_context * lg, int32_t index,
    int32_t * out_n_tokens, int32_t * out_n_embd, int32_t * out_layer_id)
{
    if (!lg || index < 0 || index >= (int32_t)lg->snapshots.size()) return nullptr;
    const auto & snap = lg->snapshots[index];
    if (out_n_tokens) *out_n_tokens = snap.n_tokens;
    if (out_n_embd)   *out_n_embd   = snap.n_embd;
    if (out_layer_id)  *out_layer_id  = snap.layer_id;
    return snap.data;
}

void llama_layer_graph_clear_snapshots(llama_layer_graph_context * lg) {
    if (!lg) return;
    for (auto & snap : lg->snapshots) free_snapshot(snap);
    lg->snapshots.clear();
}

// Internal: called from model hook to capture hidden state
void llama_layer_graph_capture_hidden_state(
    llama_layer_graph_context * lg,
    const float * data,
    int32_t n_tokens,
    int32_t n_embd,
    int32_t layer_id,
    int32_t instance_id)
{
    if (!lg || !lg->capture_enabled || !data) return;

    // Hard limit to prevent unbounded memory growth
    // Each snapshot is n_tokens * n_embd * 4 bytes; with 4096 tokens * 4096 embd = 64 MB per snapshot
    const size_t MAX_SNAPSHOTS = 64;
    while (lg->snapshots.size() >= MAX_SNAPSHOTS) {
        free_snapshot(lg->snapshots.front());
        lg->snapshots.erase(lg->snapshots.begin());
    }

    llama_hidden_state_snapshot snap;
    snap.n_tokens = n_tokens;
    snap.n_embd   = n_embd;
    snap.layer_id = layer_id;
    snap.instance_id = instance_id;

    size_t size = n_tokens * n_embd * sizeof(float);
    snap.data = (float *)malloc(size);
    if (!snap.data) return;

    std::memcpy(snap.data, data, size);
    lg->snapshots.push_back(snap);
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

double * llama_layer_graph_get_timings(const llama_layer_graph_context * lg, int32_t * out_count) {
    if (!lg) {
        if (out_count) *out_count = 0;
        return nullptr;
    }
    if (out_count) *out_count = (int32_t)lg->timings_us.size();

    if (lg->timings_us.empty()) return nullptr;

    double * result = (double *)malloc(lg->timings_us.size() * sizeof(double));
    if (!result) return nullptr;
    std::memcpy(result, lg->timings_us.data(), lg->timings_us.size() * sizeof(double));
    return result;
}

void llama_layer_graph_free_timings(double * timings) {
    free(timings);
}

void llama_layer_graph_reset_timings(llama_layer_graph_context * lg) {
    if (!lg) return;
    std::fill(lg->timings_us.begin(), lg->timings_us.end(), 0.0);
}

bool llama_layer_graph_is_layer_enabled(const llama_layer_graph_context * lg, int32_t layer_id) {
    if (!lg || !lg->graph.nodes) return true;
    for (int32_t in = 0; in < lg->graph.n_nodes; ++in) {
        if (lg->graph.nodes[in].layer_id == layer_id) {
            return lg->graph.nodes[in].enabled;
        }
    }
    return true; // layer not in graph -> enabled by default
}
