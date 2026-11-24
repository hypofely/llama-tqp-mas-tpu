#pragma once

#include "llama.h"
#include <vector>
#include <string>
#include <unordered_map>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration for context setter (defined in llama-context.cpp)
LLAMA_API void llama_context_set_layer_graph(struct llama_context * ctx, struct llama_layer_graph_context * lg);

// ---------------------------------------------------------------------------
// Layer Graph Node
// ---------------------------------------------------------------------------

typedef struct {
    int32_t layer_id;           // Original layer index (0..n_layers-1)
    bool    enabled;            // If false, this node is skipped (identity pass-through)
    int32_t instance_id;        // Unique instance ID (for duplicate detection)
    int32_t next_count;         // Number of outgoing edges
    int32_t next[8];            // Next node indices (supports up to 8 branches)
} llama_layer_node;

// ---------------------------------------------------------------------------
// Layer Graph
// ---------------------------------------------------------------------------

typedef struct {
    llama_layer_node * nodes;   // Array of nodes
    int32_t            n_nodes; // Number of nodes in the graph
    int32_t            entry;   // Entry node index
    int32_t            n_layers_orig; // Original number of layers in the model
} llama_layer_graph;

// ---------------------------------------------------------------------------
// Hidden State Inspector
// ---------------------------------------------------------------------------

typedef struct {
    float  * data;              // [n_tokens * n_embd] hidden state snapshot
    int32_t  n_tokens;          // Number of tokens in this snapshot
    int32_t  n_embd;            // Embedding dimension
    int32_t  layer_id;          // Which layer this snapshot is from
    int32_t  instance_id;       // Which instance (for duplicated layers)
} llama_hidden_state_snapshot;

// ---------------------------------------------------------------------------
// Layer Graph Context (wraps a llama_context with a layer graph)
// ---------------------------------------------------------------------------

struct llama_layer_graph_context;

// ── Lifecycle ──────────────────────────────────────────────────────────────

// Create a layer graph context from an existing llama_context.
// The graph starts as a default linear chain: 0→1→2→...→n_layers-1
LLAMA_API struct llama_layer_graph_context * llama_layer_graph_init(
    struct llama_context * ctx
);

LLAMA_API void llama_layer_graph_free(struct llama_layer_graph_context * lg);

// ── Graph Inspection ───────────────────────────────────────────────────────

// Get the current graph (read-only reference, do not free)
LLAMA_API const llama_layer_graph * llama_layer_graph_get(
    const struct llama_layer_graph_context * lg
);

// Get number of original layers in the model
LLAMA_API int32_t llama_layer_graph_n_layers(
    const struct llama_layer_graph_context * lg
);

// ── Graph Mutation ─────────────────────────────────────────────────────────

// Reset graph to default linear chain: 0→1→2→...→n_layers-1
LLAMA_API void llama_layer_graph_reset(struct llama_layer_graph_context * lg);

// Toggle a layer node on/off by its instance_id
LLAMA_API int llama_layer_graph_set_enabled(
    struct llama_layer_graph_context * lg,
    int32_t instance_id,
    bool enabled
);

// Reorder: move node at from_index to to_index (shifts others)
LLAMA_API int llama_layer_graph_reorder(
    struct llama_layer_graph_context * lg,
    int32_t from_index,
    int32_t to_index
);

// Duplicate a layer node (creates a new instance with same layer_id)
// Returns the new node's instance_id, or -1 on error.
LLAMA_API int32_t llama_layer_graph_duplicate(
    struct llama_layer_graph_context * lg,
    int32_t instance_id,
    int32_t position        // -1 = append to end
);

// Remove a node by instance_id
LLAMA_API int llama_layer_graph_remove(
    struct llama_layer_graph_context * lg,
    int32_t instance_id
);

// Load graph from JSON string
// Format: {"layers":[{"id":0,"enabled":true},...],"order":[0,1,2,3]}
LLAMA_API int llama_layer_graph_load_json(
    struct llama_layer_graph_context * lg,
    const char * json
);

// Export graph to JSON string (caller must free with llama_layer_graph_free_json)
LLAMA_API char * llama_layer_graph_export_json(
    const struct llama_layer_graph_context * lg
);

LLAMA_API void llama_layer_graph_free_json(char * json);

// ── Execution ──────────────────────────────────────────────────────────────

// Run one forward pass through the layer graph.
// This replaces the normal transformer forward pass.
// tokens: array of token IDs
// n_tokens: number of tokens
// Returns 0 on success.
LLAMA_API int llama_layer_graph_forward(
    struct llama_layer_graph_context * lg,
    const llama_token * tokens,
    int32_t n_tokens
);

// Reset / set the KV cache position for this layer graph.
// By default kv_pos starts at 0 and advances automatically after each
// successful llama_layer_graph_forward().  If you want to run partial
// layer graphs in sequence (e.g. layers 0-10, then 11-20) while keeping
// the same KV cache, do NOT reset kv_pos between calls.
// Use reset_kv_pos(0) only when starting a fresh sequence.
LLAMA_API void llama_layer_graph_reset_kv_pos(
    struct llama_layer_graph_context * lg,
    llama_pos pos
);

LLAMA_API void llama_layer_graph_set_seq_id(
    struct llama_layer_graph_context * lg,
    llama_seq_id seq_id
);

// ── Hidden State Inspector ─────────────────────────────────────────────────

// Enable/disable hidden state capture (off by default for performance)
LLAMA_API void llama_layer_graph_set_capture(
    struct llama_layer_graph_context * lg,
    bool capture_enabled
);

// Get the number of captured hidden state snapshots
LLAMA_API int32_t llama_layer_graph_get_snapshot_count(
    const struct llama_layer_graph_context * lg
);

// Get a specific snapshot by index (0-based)
// Returns pointer to snapshot data (caller must NOT free). Use llama_layer_graph_get_snapshot_count to iterate.
LLAMA_API const float * llama_layer_graph_get_snapshot(
    const struct llama_layer_graph_context * lg,
    int32_t index,
    int32_t * out_n_tokens,
    int32_t * out_n_embd,
    int32_t * out_layer_id
);

// Clear all captured snapshots
LLAMA_API void llama_layer_graph_clear_snapshots(
    struct llama_layer_graph_context * lg
);

// ── Query ──────────────────────────────────────────────────────────────────

// Check if a specific layer is enabled in the active layer graph.
// If no layer graph is active, returns true (all layers enabled).
LLAMA_API bool llama_layer_graph_is_layer_enabled(
    const struct llama_layer_graph_context * lg,
    int32_t layer_id
);

// ── Metrics ────────────────────────────────────────────────────────────────

// Get per-node execution time in microseconds (array of n_nodes, caller must free)
LLAMA_API double * llama_layer_graph_get_timings(
    const struct llama_layer_graph_context * lg,
    int32_t * out_count
);

LLAMA_API void llama_layer_graph_free_timings(double * timings);

// Reset timing counters
LLAMA_API void llama_layer_graph_reset_timings(
    struct llama_layer_graph_context * lg
);

#ifdef __cplusplus
}
#endif
