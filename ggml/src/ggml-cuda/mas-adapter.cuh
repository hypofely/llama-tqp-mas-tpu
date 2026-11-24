#pragma once

#include "ggml-cuda.h"

// MAS Adapter CUDA kernels
// These operate on raw float device pointers for small MLP adapters.

#ifdef __cplusplus
extern "C" {
#endif

// LayerNorm: out = (x - mean) / sqrt(var + eps) * w + b
// x, w, b, out are device pointers of size [n_embd]
void ggml_cuda_mas_layernorm(
    const float * x, const float * w, const float * b,
    float * out, int n_embd, float eps);

// GELU activation: out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void ggml_cuda_mas_gelu(
    const float * x, float * out, int n);

// Matrix-vector multiplication: out = mat * vec
// mat: [rows x cols] in row-major, vec: [cols], out: [rows]
void ggml_cuda_mas_mat_vec_mul(
    const float * mat, const float * vec, float * out,
    int rows, int cols);

// Vector add: out = a + b
void ggml_cuda_mas_vec_add(
    const float * a, const float * b, float * out, int n);

// Vector copy: out = src
void ggml_cuda_mas_vec_copy(
    const float * src, float * dst, int n);

// RMS Norm: out = x / sqrt(mean(x^2) + eps)
void ggml_cuda_mas_rms_norm(
    const float * x, float * out, int n, float eps);

// Mean pooling: output[e] = mean over tokens of states[t * n_embd + e]
GGML_BACKEND_API void ggml_cuda_mas_mean_pool(
    const float * states, float * output, int n_tokens, int n_embd);

// Concatenation: output = states (device-to-device copy)
GGML_BACKEND_API void ggml_cuda_mas_concat(
    const float * states, float * output, int n_tokens, int n_embd);

// Device memory allocation / free (wrappers to avoid cudart dependency in llama.dll)
GGML_BACKEND_API void * ggml_cuda_mas_device_malloc(size_t size);
GGML_BACKEND_API void   ggml_cuda_mas_device_free(void * ptr);

// Host <-> Device memcpy wrappers
GGML_BACKEND_API void   ggml_cuda_mas_memcpy_h2d(void * dst, const void * src, size_t size);
GGML_BACKEND_API void   ggml_cuda_mas_memcpy_d2h(void * dst, const void * src, size_t size);

// Full RecursiveLink forward on GPU:
// LayerNorm -> Linear -> GELU -> Linear -> residual -> LayerNorm
// All pointers are device pointers. temp must have size [n_embd].
GGML_BACKEND_API void ggml_cuda_mas_recursive_forward(
    const float * hidden_in, float * hidden_out,
    const float * w_pre_ln, const float * b_pre_ln,
    const float * w_proj1, const float * b_proj1,
    const float * w_proj2, const float * b_proj2,
    const float * w_post_ln, const float * b_post_ln,
    float * temp, int n_embd);

// Sampling helpers (operate on GPU logits in-place)
// penalty_tokens: device array of tokens to penalize
GGML_BACKEND_API void ggml_cuda_mas_apply_temperature_penalty(
    float * logits, int n_vocab, float temperature, float repeat_penalty,
    const int32_t * penalty_tokens, int n_penalty);

// Greedy argmax sampling on GPU (returns device pointer to token)
GGML_BACKEND_API void ggml_cuda_mas_sample_argmax(
    const float * logits, int n_vocab, int32_t * out_token);

// Top-k sampling on GPU
GGML_BACKEND_API void ggml_cuda_mas_sample_topk(
    const float * logits, int n_vocab, int k, int32_t * out_token, unsigned long long seed);

// Top-p (nucleus) sampling on GPU
GGML_BACKEND_API void ggml_cuda_mas_sample_topp(
    const float * logits, int n_vocab, float p, int32_t * out_token, unsigned long long seed);

// Device-to-device broadcast: copies one [n] vector to n_rows rows of a [n_rows x n] matrix
// All pointers are device pointers. stride = row size in bytes (n * sizeof(float)).
GGML_BACKEND_API void ggml_cuda_mas_broadcast_rows(
    const float * src, float * dst, int n_rows, int n, size_t stride);

// Batched matrix-vector: out[batch, rows] = mat[rows x cols] * vec[batch, cols]
// mat: device [rows x cols], vec: device [batch x cols], out: device [batch x rows]
GGML_BACKEND_API void ggml_cuda_mas_mat_vec_mul_batched(
    const float * mat, const float * vec, float * out,
    int batch, int rows, int cols);

// Full CrossLink forward on GPU:
// LayerNorm(source) -> Linear -> GELU -> Linear -> residual_proj -> LayerNorm(target)
// temp1 must have size [in_dim], temp2 [hidden_dim], temp3 [out_dim]
GGML_BACKEND_API void ggml_cuda_mas_cross_forward(
    const float * hidden_in, float * hidden_out,
    const float * w_cross_ln_source, const float * b_cross_ln_source,
    const float * w_cross_proj1, const float * b_cross_proj1,
    const float * w_cross_proj2, const float * b_cross_proj2,
    const float * w_cross_residual, const float * b_cross_residual,
    const float * w_cross_ln_target, const float * b_cross_ln_target,
    float * temp1, float * temp2, float * temp3,
    int in_dim, int out_dim, int hidden_dim);

#ifdef __cplusplus
}
#endif
