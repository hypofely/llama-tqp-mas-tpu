#include "common.cuh"

// Quantization: F16 -> planar/iso
void ggml_cuda_cpy_f16_planar3(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_planar4(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_iso3(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_iso4(const char * src, char * dst, int64_t ne, cudaStream_t stream);

// Dequantization: planar/iso -> F32 (for flash attention)
void ggml_cuda_dequantize_planar3(const char * src, float * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_dequantize_iso3(const char * src, float * dst, int64_t ne, cudaStream_t stream);
