#!/usr/bin/env bash
# Build mas-server for Android (Termux) with Vulkan backend for Adreno GPUs.
# Run this inside Termux on the device:
#   pkg install -y clang cmake git
#   ./scripts/build-termux-vulkan.sh
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-termux-vulkan}"
NPROC="${NPROC:-$(nproc)}"

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_OPENMP=OFF \
  -DGGML_NATIVE=OFF \
  -DLLAMA_OPENSSL=OFF \
  -DGGML_CPU_ARM_ARCH=armv8.5-a+fp16+i8mm

cmake --build "$BUILD_DIR" --target mas-server -j"$NPROC"

echo "Built: $BUILD_DIR/bin/mas-server"
echo "Run with: VK_ICD_FILENAMES=... $BUILD_DIR/bin/mas-server -m <model.gguf> --n-gpu-layers 99 --type-k turbo3 --type-v turbo3"
