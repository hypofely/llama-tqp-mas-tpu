# CUDA Flash Attention Integration for RotorQuant

## Current Status

- **CPU Implementation:** Complete and working (planar3, iso3, planar4, iso4 types added)
- **CUDA Dequantization Kernels:** Added to `cpy-planar-iso.cu` for basic quantization/dequantization
- **CUDA Flash Attention Integration:** Not yet implemented (complex template instantiation issues)

## What Has Been Completed

1. **Type Definitions:** Added `GGML_TYPE_PLANAR3_0`, `GGML_TYPE_ISO3_0`, `GGML_TYPE_PLANAR4_0`, `GGML_TYPE_ISO4_0` to `ggml.h`
2. **Block Structures:** Added RotorQuant block structures to `ggml-common.h`
3. **CPU Quantization:** Added quantization and dequantization functions for planar/iso types
4. **CUDA Basic Kernels:** Added F16 to planar/iso conversion kernels in `cpy-planar-iso.cu`
5. **CUDA Dequantization Kernels:** Added dequantization kernels in `cpy-planar-iso.cu` for general use
6. **Build System:** Updated CMakeLists.txt to include RotorQuant source files

## What Is Required for Full CUDA Flash Attention Integration

### 1. Constant Memory Access Issue

The primary blocker is accessing RotorQuant constant memory (rotation matrices, centroids) from flash attention code.

**Problem:** CUDA compiler treats `extern __constant__` declarations in header files as static definitions, causing multiple definition errors.

**Solution Options:**
- **Option A:** Move dequantization functions to `cpy-planar-iso.cu` and expose through a unified interface
- **Option B:** Create a separate header for constant declarations with proper linkage
- **Option C:** Use device functions instead of constant memory for rotation parameters
- **Option D:** Inline the rotation parameters directly in the dequantization functions

### 2. Dequantization Functions in Flash Attention

Add dequantization functions to `fattn-common.cuh`:

```cpp
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_planar3_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0);
```

**Requirements:**
- Access to `d_planar_cos`, `d_planar_sin` constant arrays
- Access to `PLANAR_CENTROIDS_3BIT` centroid array
- Implement inverse Givens rotation for dequantization
- Handle both ne=1 and ne=2 template specializations

### 3. Type Dispatch in Flash Attention

Update `get_dequantize_V()` in `fattn-common.cuh`:

```cpp
else if constexpr (type_V == GGML_TYPE_PLANAR3_0) {
    return dequantize_V_planar3_0<T, ne>;
} else if constexpr (type_V == GGML_TYPE_ISO3_0) {
    return dequantize_V_iso3_0<T, ne>;
}
```

### 4. Unquantized Type Classification

Update `fattn-vec.cuh` to treat planar/iso as unquantized types (like turbo3):

```cpp
constexpr bool K_is_unquantized = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16 || 
                                   type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO2_0 || 
                                   type_K == GGML_TYPE_TURBO4_0 || type_K == GGML_TYPE_PLANAR3_0 || 
                                   type_K == GGML_TYPE_ISO3_0);
```

### 5. Template Instances

Add template instances to `ggml-cuda/CMakeLists.txt`:

```cmake
# RotorQuant planar/iso template instances
template-instances/fattn-vec-instance-planar3_0-planar3_0.cu
template-instances/fattn-vec-instance-planar3_0-q8_0.cu
template-instances/fattn-vec-instance-q8_0-planar3_0.cu
template-instances/fattn-vec-instance-f16-planar3_0.cu
template-instances/fattn-vec-instance-planar3_0-f16.cu
# ... similar for iso3
```

### 6. Vector Dot Product Functions

Add vec_dot functions for planar/iso K cache in `fattn-common.cuh`:

```cpp
template <int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    // ...
    else if constexpr (type_K == GGML_TYPE_PLANAR3_0) {
        return vec_dot_fattn_vec_KQ_planar3_0<D, nthreads>;
    }
    // ...
}
```

**Requirements:**
- Implement quantized K cache dot product with rotation matrix application
- Similar to turbo3 implementation but with Givens rotation

## Recommended Approach

Given the complexity, the recommended approach is:

1. **Short-term:** Use CPU-only RotorQuant for now (already working)
2. **Medium-term:** Implement CUDA flash attention for planar/iso using Option D (inline parameters)
3. **Long-term:** Refactor to use proper constant memory access if needed

## Technical Challenges

1. **Constant Memory Linkage:** CUDA's treatment of `extern __constant__` in headers
2. **Template Instantiation Complexity:** Flash attention uses extensive template metaprogramming
3. **Rotation Matrix Application:** Must apply inverse rotation during dequantization
4. **Performance Optimization:** Dequantization must be efficient to maintain flash attention speed

## Files to Modify

- `ggml/src/ggml-cuda/fattn-common.cuh` - Add dequantization functions
- `ggml/src/ggml-cuda/fattn-vec.cuh` - Update type classification
- `ggml/src/ggml-cuda/CMakeLists.txt` - Add template instances
- `ggml/src/ggml-cuda/cpy-planar-iso.cu` - Potentially move dequantization functions here

## Estimated Effort

- **Low complexity:** 2-3 days (if using inline parameters)
- **High complexity:** 1-2 weeks (proper constant memory access with full optimization)
