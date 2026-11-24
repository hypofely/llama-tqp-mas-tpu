#!/usr/bin/env bash
# ============================================================================
# sync_upstream.sh — Port llama-tqp-mas-cpp custom changes on top of new upstream
# ============================================================================
# Usage:
#   ./sync_upstream.sh /path/to/new/llama.cpp
#   ./sync_upstream.sh --diff-only        # Just generate diff
#   ./sync_upstream.sh --dry-run          # Show what would be copied/applied
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIFF_FILE="$REPO_ROOT/custom.diff"
LOG_FILE="$SCRIPT_DIR/sync_upstream.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Parse args
NEW_UPSTREAM=""
DIFF_ONLY=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --diff-only) DIFF_ONLY=true ;;
        --dry-run)   DRY_RUN=true ;;
        *)           NEW_UPSTREAM="$arg" ;;
    esac
done

log "=== sync_upstream.sh starting ==="
log "Repo root: $REPO_ROOT"

# ============================================================
# Step 1: Generate diff
# ============================================================
generate_diff() {
    log "Generating custom diff..."
    
    # Find upstream base commit from git history
    BASE_COMMIT=""
    if cd "$REPO_ROOT"; then
        BASE_COMMIT=$(git log --all --oneline --grep="upstream" 2>/dev/null | head -1 | awk '{print $1}' || true)
    fi
    
    if [ -n "$BASE_COMMIT" ]; then
        log "Using upstream base commit: $BASE_COMMIT"
        cd "$REPO_ROOT"
        git diff "$BASE_COMMIT..HEAD" --no-color \
            -- . \
            ':(exclude)build*/' \
            ':(exclude).vs/' \
            ':(exclude)archive/' \
            ':(exclude)compile_commands.json' \
            > "$DIFF_FILE" 2>/dev/null || true
    else
        log "No upstream base commit found — diffing working tree"
        cd "$REPO_ROOT"
        git diff --no-color \
            -- . \
            ':(exclude)build*/' \
            ':(exclude).vs/' \
            ':(exclude)archive/' \
            > "$DIFF_FILE" 2>/dev/null || true
    fi
    
    local stats
    stats=$(git diff --stat "$BASE_COMMIT..HEAD" -- . ':(exclude)build*/' ':(exclude).vs/' ':(exclude)archive/' 2>/dev/null | tail -1 || echo "unknown")
    log "Diff generated: $stats"
    log "Diff file: $DIFF_FILE"
}

# ============================================================
# Step 2: List core new files (not in upstream)
# ============================================================
get_core_files() {
    cat <<'FILES'
src/llama-mas.cpp
src/llama-mas.h
src/llama-layer-graph.cpp
src/llama-layer-graph.h
src/llama-kv-cache.cpp
src/llama-kv-cache.h
src/llama-context.cpp
src/models/llama.cpp
include/llama.h
tools/mas-server/mas-server.cpp
tools/mas-server/mas-server-common.cpp
tools/mas-server/mas-server-common.h
tools/mas-server/server-task.cpp
tools/mas-server/server-task.h
tools/mas-server/server-queue.cpp
tools/mas-server/server-queue.h
tools/mas-server/server-models.cpp
tools/mas-server/server-models.h
tools/mas-server/server-http.cpp
tools/mas-server/server-http.h
tools/mas-server/server-context.cpp
tools/mas-server/server-context.h
tools/mas-server/server-tools.cpp
tools/mas-server/server-tools.h
tools/mas-server/CMakeLists.txt
ggml/src/ggml-cuda/mas-adapter.cu
ggml/src/ggml-cuda/mas-adapter.cuh
ggml/src/ggml-cuda/cpy-planar-iso.cu
ggml/src/ggml-cuda/cpy-planar-iso.cuh
ggml/src/ggml-cuda/cpy-utils.cuh
ggml/src/ggml-cuda/set-rows.cu
ggml/src/ggml-cuda/set-rows.cuh
ggml/src/ggml-cuda/planar-iso-constants.cuh
ggml/src/ggml-cuda/turbo-quant.cuh
ggml/src/ggml-cuda/turbo-wht.cu
ggml/src/ggml-cuda/turbo-innerq.cu
FILES
}

# ============================================================
# Step 3: Apply patch to new upstream
# ============================================================
apply_patch() {
    local target="$1"
    
    if [ ! -d "$target" ]; then
        log "ERROR: Target directory does not exist: $target"
        exit 1
    fi
    
    log "Applying patch to: $target"
    
    # 1. Copy core new files first
    local copied=0
    while IFS= read -r file; do
        [ -z "$file" ] && continue
        local src="$REPO_ROOT/$file"
        local dst="$target/$file"
        
        if [ -f "$src" ]; then
            mkdir -p "$(dirname "$dst")"
            if [ ! -f "$dst" ]; then
                cp -f "$src" "$dst"
                log "  COPIED (new): $file"
                copied=$((copied + 1))
            fi
        fi
    done < <(get_core_files)
    
    log "  Copied $copied new files"
    
    # 2. Copy CMakeLists.txt extensions (mas-server, mas-bench)
    local extras=(
        "tools/mas-server/CMakeLists.txt"
        "examples/mas-bench/mas-bench.cpp"
        "examples/mas-bench/latent-bench.cpp"
        "examples/mas-bench/test-alignment.cpp"
        "examples/mas-bench/CMakeLists.txt"
    )
    for file in "${extras[@]}"; do
        local src="$REPO_ROOT/$file"
        local dst="$target/$file"
        if [ -f "$src" ] && [ ! -f "$dst" ]; then
            mkdir -p "$(dirname "$dst")"
            cp -f "$src" "$dst"
            log "  COPIED (extra): $file"
        fi
    done
    
    # 3. Apply diff to modified files
    if [ -f "$DIFF_FILE" ]; then
        cd "$target"
        local patch_result
        patch_result=$(git apply --whitespace=nowarn "$DIFF_FILE" 2>&1) || {
            log "  PATCH conflicts detected:"
            echo "$patch_result" | while IFS= read -r line; do log "    $line"; done
            log "  Resolve conflicts manually, then re-run"
        }
        if [ $? -eq 0 ]; then
            log "  PATCH applied successfully"
        fi
    else
        log "  No diff file found, skipping patch step"
    fi
    
    # 4. Modify tools/CMakeLists.txt to add mas-server and mas-bench subdirs
    local tools_cmake="$target/tools/CMakeLists.txt"
    if [ -f "$tools_cmake" ]; then
        if ! grep -q "mas-server" "$tools_cmake" 2>/dev/null; then
            echo '' >> "$tools_cmake"
            echo 'if(LLAMA_BUILD_TOOLS)' >> "$tools_cmake"
            echo '    add_subdirectory(mas-server)' >> "$tools_cmake"
            echo '    add_subdirectory(mas-bench)' >> "$tools_cmake"
            echo 'endif()' >> "$tools_cmake"
            log "  ADDED mas-server/mas-bench to tools/CMakeLists.txt"
        fi
    fi
}

# ============================================================
# Step 4: Build verification
# ============================================================
test_build() {
    local target="$1"
    
    log "Build verification for: $target"
    
    # Check key files exist
    local checks=(
        "src/llama-mas.cpp:MAS engine"
        "src/llama-layer-graph.cpp:Layer graph"
        "tools/mas-server/mas-server.cpp:MAS server"
        "ggml/src/ggml-cuda/mas-adapter.cu:CUDA adapter"
    )
    
    for check in "${checks[@]}"; do
        local file="${check%%:*}"
        local desc="${check##*:}"
        if [ -f "$target/$file" ]; then
            log "  $desc: ✓ ($file)"
        else
            log "  $desc: MISSING ($file)"
        fi
    done
    
    # Check CMakeLists.txt
    local tools_cmake="$target/tools/CMakeLists.txt"
    if grep -q "mas-server" "$tools_cmake" 2>/dev/null; then
        log "  tools/CMakeLists.txt has mas-server: ✓"
    fi
    
    # Check include/llama.h exports
    if grep -q "llama_mas_" "$target/include/llama.h" 2>/dev/null; then
        log "  include/llama.h has MAS exports: ✓"
    fi
}

# ============================================================
# Main
# ============================================================

if [ "$DIFF_ONLY" = true ]; then
    generate_diff
    log "=== diff-only mode complete ==="
    exit 0
fi

if [ -z "$NEW_UPSTREAM" ]; then
    echo "Usage: $0 [--diff-only] /path/to/new/llama.cpp"
    echo "   Or: $0 --dry-run /path/to/new/llama.cpp"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    log "DRY RUN mode"
    generate_diff
    log "Would apply to: $NEW_UPSTREAM"
    log "Core files to copy:"
    get_core_files | while IFS= read -r f; do [ -n "$f" ] && log "  $f"; done
    exit 0
fi

generate_diff
apply_patch "$NEW_UPSTREAM"
test_build "$NEW_UPSTREAM"

log "=== sync_upstream.sh complete ==="
