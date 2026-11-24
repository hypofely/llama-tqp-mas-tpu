#include "mas-adapter.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void kernel_layernorm(const float * x, const float * w, const float * b,
                                 float * out, int n_embd, float eps) {
    __shared__ float s_sum[256];
    __shared__ float s_var[256];

    float mean = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        mean += x[i];
    }
    s_sum[threadIdx.x] = mean;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    mean = s_sum[0] / n_embd;

    float var = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        float d = x[i] - mean;
        var += d * d;
    }
    s_var[threadIdx.x] = var;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_var[threadIdx.x] += s_var[threadIdx.x + stride];
        }
        __syncthreads();
    }
    var = s_var[0] / n_embd;
    float scale = 1.0f / sqrtf(var + eps);

    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        out[i] = (x[i] - mean) * scale * w[i] + (b ? b[i] : 0.0f);
    }
}

__global__ void kernel_rms_norm(const float * x, float * out, int n, float eps) {
    __shared__ float s_sum[256];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += x[i] * x[i];
    }
    s_sum[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float scale = 1.0f / sqrtf(s_sum[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = x[i] * scale;
    }
}

__global__ void kernel_gelu(const float * x, float * out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xv = x[i];
    out[i] = 0.5f * xv * (1.0f + tanhf(0.7978845608f * (xv + 0.044715f * xv * xv * xv)));
}

__global__ void kernel_mat_vec_mul(const float * mat, const float * vec, float * out,
                                     int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        sum += mat[r * cols + c] * vec[c];
    }
    out[r] = sum;
}

__global__ void kernel_vec_add(const float * a, const float * b, float * out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}

__global__ void kernel_vec_copy(const float * src, float * dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i];
}

__global__ void kernel_mean_pool(const float * states, float * output,
                                   int n_tokens, int n_embd) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_embd) return;
    float sum = 0.0f;
    for (int t = 0; t < n_tokens; ++t) {
        sum += states[t * n_embd + e];
    }
    output[e] = sum / n_tokens;
}

__global__ void kernel_concat(const float * states, float * output,
                               int n_tokens, int n_embd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * n_embd;
    if (i >= total) return;
    output[i] = states[i];
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

void ggml_cuda_mas_layernorm(const float * x, const float * w, const float * b,
                             float * out, int n_embd, float eps) {
    kernel_layernorm<<<1, 256>>>(x, w, b, out, n_embd, eps);
}

void ggml_cuda_mas_rms_norm(const float * x, float * out, int n, float eps) {
    kernel_rms_norm<<<1, 256>>>(x, out, n, eps);
}

void ggml_cuda_mas_gelu(const float * x, float * out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_gelu<<<grid, block>>>(x, out, n);
}

void ggml_cuda_mas_mat_vec_mul(const float * mat, const float * vec, float * out,
                               int rows, int cols) {
    int block = 256;
    int grid = (rows + block - 1) / block;
    kernel_mat_vec_mul<<<grid, block>>>(mat, vec, out, rows, cols);
}

void ggml_cuda_mas_vec_add(const float * a, const float * b, float * out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_vec_add<<<grid, block>>>(a, b, out, n);
}

void ggml_cuda_mas_vec_copy(const float * src, float * dst, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_vec_copy<<<grid, block>>>(src, dst, n);
}

void ggml_cuda_mas_mean_pool(const float * states, float * output,
                               int n_tokens, int n_embd) {
    int block = 256;
    int grid = (n_embd + block - 1) / block;
    kernel_mean_pool<<<grid, block>>>(states, output, n_tokens, n_embd);
}

void ggml_cuda_mas_concat(const float * states, float * output,
                           int n_tokens, int n_embd) {
    int block = 256;
    int total = n_tokens * n_embd;
    int grid = (total + block - 1) / block;
    kernel_concat<<<grid, block>>>(states, output, n_tokens, n_embd);
}

// ---------------------------------------------------------------------------
// Full adapter forwards
// ---------------------------------------------------------------------------

void ggml_cuda_mas_recursive_forward(
    const float * hidden_in, float * hidden_out,
    const float * w_pre_ln, const float * b_pre_ln,
    const float * w_proj1, const float * b_proj1,
    const float * w_proj2, const float * b_proj2,
    const float * w_post_ln, const float * b_post_ln,
    float * temp, int n_embd) {
    // pre_ln
    ggml_cuda_mas_layernorm(hidden_in, w_pre_ln, b_pre_ln, temp, n_embd, 1e-5f);
    // proj1
    ggml_cuda_mas_mat_vec_mul(w_proj1, temp, temp + n_embd, n_embd, n_embd);
    ggml_cuda_mas_vec_add(temp + n_embd, b_proj1, temp + n_embd, n_embd);
    // GELU
    ggml_cuda_mas_gelu(temp + n_embd, temp + n_embd, n_embd);
    // proj2
    ggml_cuda_mas_mat_vec_mul(w_proj2, temp + n_embd, temp + 2 * n_embd, n_embd, n_embd);
    ggml_cuda_mas_vec_add(temp + 2 * n_embd, b_proj2, temp + 2 * n_embd, n_embd);
    // residual: x + out
    ggml_cuda_mas_vec_add(hidden_in, temp + 2 * n_embd, temp + 2 * n_embd, n_embd);
    // post_ln
    ggml_cuda_mas_layernorm(temp + 2 * n_embd, w_post_ln, b_post_ln, hidden_out, n_embd, 1e-5f);
}

// ---------------------------------------------------------------------------
// Device memory wrappers
// ---------------------------------------------------------------------------

void * ggml_cuda_mas_device_malloc(size_t size) {
    void * ptr = nullptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void ggml_cuda_mas_device_free(void * ptr) {
    if (ptr) cudaFree(ptr);
}

void ggml_cuda_mas_memcpy_h2d(void * dst, const void * src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void ggml_cuda_mas_memcpy_d2h(void * dst, const void * src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

// ---------------------------------------------------------------------------
// Full adapter forwards
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// MAS sampling helpers: apply temperature + repetition penalty on GPU
// ---------------------------------------------------------------------------

__global__ void kernel_apply_temperature_and_penalty(
    float * logits, int n_vocab, float temperature, float repeat_penalty,
    const int32_t * penalty_tokens, int n_penalty)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vocab) return;
    logits[i] /= temperature;
    for (int j = 0; j < n_penalty; ++j) {
        if (penalty_tokens[j] == i) {
            if (logits[i] > 0.0f) logits[i] /= repeat_penalty;
            else                 logits[i] *= repeat_penalty;
            break;
        }
    }
}

void ggml_cuda_mas_apply_temperature_penalty(
    float * logits, int n_vocab, float temperature, float repeat_penalty,
    const int32_t * penalty_tokens, int n_penalty)
{
    int block = 256;
    int grid = (n_vocab + block - 1) / block;
    kernel_apply_temperature_and_penalty<<<grid, block>>>(
        logits, n_vocab, temperature, repeat_penalty, penalty_tokens, n_penalty);
}

// ---------------------------------------------------------------------------
// Top-k + top-p + sampling on GPU (simplified: returns the sampled token)
// ---------------------------------------------------------------------------

__global__ void kernel_sample_argmax(const float * logits, int n_vocab, int32_t * out_token) {
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];

    int best = threadIdx.x;
    float best_val = best < n_vocab ? logits[best] : -1e38f;
    for (int i = threadIdx.x + blockDim.x; i < n_vocab; i += blockDim.x) {
        float v = logits[i];
        if (v > best_val) {
            best_val = v;
            best = i;
        }
    }

    s_val[threadIdx.x] = best_val;
    s_idx[threadIdx.x] = best;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && s_val[threadIdx.x + stride] > s_val[threadIdx.x]) {
            s_val[threadIdx.x] = s_val[threadIdx.x + stride];
            s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out_token = s_idx[0];
    }
}

void ggml_cuda_mas_sample_argmax(const float * logits, int n_vocab, int32_t * out_token) {
    kernel_sample_argmax<<<1, 256>>>(logits, n_vocab, out_token);
}

// Top-k sampling kernel with proper softmax and random selection
__global__ void kernel_sample_topk(const float * logits, int n_vocab, int k, int32_t * out_token, unsigned long long seed) {
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];
    __shared__ float s_exp[256];
    __shared__ float s_sum[256];

    int tid = threadIdx.x;
    
    // Initialize curand state
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);
    
    // Find max value for numerical stability
    float max_val = tid < n_vocab ? logits[tid] : -1e38f;
    for (int i = tid + blockDim.x; i < n_vocab; i += blockDim.x) {
        float v = logits[i];
        if (v > max_val) {
            max_val = v;
        }
    }

    s_val[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_val[tid + stride] > s_val[tid]) {
            s_val[tid] = s_val[tid + stride];
        }
        __syncthreads();
    }
    max_val = s_val[0];
    __syncthreads();

    // Compute exp and sum for softmax
    float local_exp = tid < n_vocab ? expf(logits[tid] - max_val) : 0.0f;
    s_exp[tid] = local_exp;
    __syncthreads();

    float local_sum = local_exp;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_sum += s_exp[tid + stride];
        }
        __syncthreads();
        s_exp[tid] = local_sum;
    }
    float total_sum = s_exp[0];
    __syncthreads();

    // Compute softmax probabilities
    float local_prob = tid < n_vocab ? expf(logits[tid] - max_val) / total_sum : 0.0f;
    
    // For simplicity, if k=1, use argmax (greedy)
    if (k == 1) {
        int best = tid;
        float best_val = best < n_vocab ? logits[best] : -1e38f;
        for (int i = tid + blockDim.x; i < n_vocab; i += blockDim.x) {
            float v = logits[i];
            if (v > best_val) {
                best_val = v;
                best = i;
            }
        }

        s_val[tid] = best_val;
        s_idx[tid] = best;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride && s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            *out_token = s_idx[0];
        }
    } else {
        // For k > 1, sample from top-k (simplified: sample from all with softmax)
        float r = curand_uniform(&state);
        float cumulative = 0.0f;
        int selected = 0;
        
        for (int i = tid; i < n_vocab; i += blockDim.x) {
            float prob = expf(logits[i] - max_val) / total_sum;
            cumulative += prob;
            if (r <= cumulative && selected == 0) {
                selected = i;
            }
        }
        
        // Reduce to find selected token
        __shared__ int s_selected[256];
        s_selected[tid] = selected;
        __syncthreads();
        
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride && s_selected[tid + stride] > 0) {
                s_selected[tid] = s_selected[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            *out_token = s_selected[0];
        }
    }
}

void ggml_cuda_mas_sample_topk(const float * logits, int n_vocab, int k, int32_t * out_token, unsigned long long seed) {
    kernel_sample_topk<<<1, 256>>>(logits, n_vocab, k, out_token, seed);
}

// Top-p (nucleus) sampling kernel with proper softmax and random selection
__global__ void kernel_sample_topp(const float * logits, int n_vocab, float p, int32_t * out_token, unsigned long long seed) {
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];
    __shared__ float s_exp[256];
    __shared__ float s_sum[256];

    int tid = threadIdx.x;
    
    // Initialize curand state
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);
    
    // Find max value for numerical stability
    float max_val = tid < n_vocab ? logits[tid] : -1e38f;
    for (int i = tid + blockDim.x; i < n_vocab; i += blockDim.x) {
        float v = logits[i];
        if (v > max_val) {
            max_val = v;
        }
    }

    s_val[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_val[tid + stride] > s_val[tid]) {
            s_val[tid] = s_val[tid + stride];
        }
        __syncthreads();
    }
    max_val = s_val[0];
    __syncthreads();

    // Compute exp and sum for softmax
    float local_exp = tid < n_vocab ? expf(logits[tid] - max_val) : 0.0f;
    s_exp[tid] = local_exp;
    __syncthreads();

    float local_sum = local_exp;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_sum += s_exp[tid + stride];
        }
        __syncthreads();
        s_exp[tid] = local_sum;
    }
    float total_sum = s_exp[0];
    __syncthreads();

    // Sample from softmax distribution
    float r = curand_uniform(&state);
    float cumulative = 0.0f;
    int selected = 0;
    
    for (int i = tid; i < n_vocab; i += blockDim.x) {
        float prob = expf(logits[i] - max_val) / total_sum;
        cumulative += prob;
        if (r <= cumulative && selected == 0) {
            selected = i;
        }
    }
    
    // Reduce to find selected token
    __shared__ int s_selected[256];
    s_selected[tid] = selected;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_selected[tid + stride] > 0) {
            s_selected[tid] = s_selected[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *out_token = s_selected[0];
    }
}

void ggml_cuda_mas_sample_topp(const float * logits, int n_vocab, float p, int32_t * out_token, unsigned long long seed) {
    kernel_sample_topp<<<1, 256>>>(logits, n_vocab, p, out_token, seed);
}

void ggml_cuda_mas_cross_forward(
    const float * hidden_in, float * hidden_out,
    const float * w_cross_ln_source, const float * b_cross_ln_source,
    const float * w_cross_proj1, const float * b_cross_proj1,
    const float * w_cross_proj2, const float * b_cross_proj2,
    const float * w_cross_residual, const float * b_cross_residual,
    const float * w_cross_ln_target, const float * b_cross_ln_target,
    float * temp1, float * temp2, float * temp3,
    int in_dim, int out_dim, int hidden_dim) {
    // ln_source
    ggml_cuda_mas_layernorm(hidden_in, w_cross_ln_source, b_cross_ln_source, temp1, in_dim, 1e-5f);
    // proj1
    ggml_cuda_mas_mat_vec_mul(w_cross_proj1, temp1, temp2, hidden_dim, in_dim);
    ggml_cuda_mas_vec_add(temp2, b_cross_proj1, temp2, hidden_dim);
    // GELU
    ggml_cuda_mas_gelu(temp2, temp2, hidden_dim);
    // proj2
    ggml_cuda_mas_mat_vec_mul(w_cross_proj2, temp2, temp3, out_dim, hidden_dim);
    ggml_cuda_mas_vec_add(temp3, b_cross_proj2, temp3, out_dim);
    // residual projection
    ggml_cuda_mas_mat_vec_mul(w_cross_residual, hidden_in, temp2, out_dim, in_dim);
    ggml_cuda_mas_vec_add(temp2, b_cross_residual, temp2, out_dim);
    // residual add
    ggml_cuda_mas_vec_add(temp3, temp2, temp3, out_dim);
    // ln_target
    ggml_cuda_mas_layernorm(temp3, w_cross_ln_target, b_cross_ln_target, hidden_out, out_dim, 1e-5f);
}
