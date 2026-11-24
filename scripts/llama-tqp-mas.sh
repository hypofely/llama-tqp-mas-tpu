#!/usr/bin/env bash
# ============================================================================
# llama-tqp-mas.sh — build & maintain the TurboQuant+LatentMAS llama.cpp fork
# ============================================================================
# This fork = TheTom/llama-cpp-turboquant (feature/turboquant-kv-cache) +
#             the local MAS overlay (multi-agent engine, hidden states,
#             layer graphs, KV clone, CrossLink adapters, mas-server).
#
# The repo this runs in IS the fork (it carries the whole history). The script
# keeps it fresh by merging the *latest* published state from:
#   - origin   -> https://github.com/TheTom/llama-cpp-turboquant   (TurboQuant)
#   - upstream -> https://github.com/ggml-org/llama.cpp             (upstream)
# — then builds with CUDA and runs the test suites.
#
# Windows: run under git-bash. Requires VS Build Tools (any year, auto-detected
# via vswhere) + CUDA Toolkit (nvcc) + CMake + Ninja.
#
# Usage:
#   ./scripts/llama-tqp-mas.sh fetch                 # update remotes, report drift
#   ./scripts/llama-tqp-mas.sh merge                 # merge latest TheTom fork state
#   ./scripts/llama-tqp-mas.sh merge --with-upstream # ...also try ggml-org master
#   ./scripts/llama-tqp-mas.sh build                 # CUDA build
#   ./scripts/llama-tqp-mas.sh test                  # ctest subset + server suites
#   ./scripts/llama-tqp-mas.sh all                   # fetch + merge + build + test
#   ./scripts/llama-tqp-mas.sh fresh <dir>           # reproduce from LATEST llama.cpp
#                                                     #   + TurboQuant + MAS overlay
#                                                     #   (--from-turboquant: base = TheTom fork)
#
# A merge that hits conflicts stops with guidance (left in conflict state for
# resolution; never auto-resolves); the MAS overlay files are listed in
# CUSTOM_PATCHES.md / docs/LATENTMAS_INTEGRATION.md.
# ============================================================================
set -uo pipefail

REPO_URL_ORIGIN="https://github.com/TheTom/llama-cpp-turboquant.git"
REPO_URL_UPSTREAM="https://github.com/ggml-org/llama.cpp.git"
FORK_BRANCH="feature/turboquant-kv-cache"

GIT=${GIT:-"C:/PortableGit/bin/git.exe"}
CMake=${CMAKE:-"cmake"}
NINJA=${NINJA:-""}

# ---------------------------------------------------------------------------
log()  { printf '\033[1;36m[llama-tqp-mas]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[llama-tqp-mas]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[llama-tqp-mas]\033[0m %s\n' "$*"; exit 1; }

# ---------------------------------------------------------------------------
find_repo_root() {
    local d="$PWD"
    while [ "$d" != "/" ]; do
        if [ -f "$d/CMakeLists.txt" ] && [ -d "$d/src" ] && [ -d "$d/ggml" ]; then
            echo "$d"; return 0
        fi
        d="$(dirname "$d")"
    done
    die "run this script from inside the fork checkout (or pass the repo dir as \$1)"
}

ROOT="$(find_repo_root)"
cd "$ROOT" || die "cannot cd to $ROOT"

# ---------------------------------------------------------------------------
# toolchain discovery (Windows/git-bash)
find_vcvars() {
    local vswhere="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
    if [ -x "$vswhere" ]; then
        local vsdir
        vsdir=$("$vswhere" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>/dev/null)
        if [ -n "$vsdir" ] && [ -f "$vsdir/VC/Auxiliary/Build/vcvars64.bat" ]; then
            echo "$vsdir/VC/Auxiliary/Build/vcvars64.bat"; return 0
        fi
    fi
    # fallback: classic BuildTools path
    local legacy="/c/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools"
    if [ -f "$legacy/VC/Auxiliary/Build/vcvars64.bat" ]; then
        echo "$legacy/VC/Auxiliary/Build/vcvars64.bat"; return 0
    fi
    return 1
}

run_with_vcvars() { # <vcvars.bat> <cmd...>
    local vcvars="$1"; shift
    # git-bash: run the command through cmd.exe so vcvars64.bat env applies
    local bat="$(cygpath -w "$TMPDIR/llama-tqp-vcvars.bat")" 2>/dev/null || true
    if [ -z "$bat" ]; then bat="C:\\Users\\$USERNAME\\AppData\\Local\\Temp\\llama-tqp-vcvars.bat"; fi
    {
        echo "@echo off"
        echo "call \"$(cygpath -w "$vcvars")\" >nul 2>&1"
        printf '%s ' "$@"
        echo ""
        echo "echo LLAMA_TQP_EXIT=%ERRORLEVEL%"
    } > "${TMPDIR:-/tmp}/llama-tqp-vcvars.bat"
    cmd.exe /c "$(cygpath -w "${TMPDIR:-/tmp}/llama-tqp-vcvars.bat")" 2>&1
}

# ---------------------------------------------------------------------------
cmd_fetch() {
    log "fetching latest TurboQuant (TheTom) + upstream (ggml-org)..."
    "$GIT" remote get-url origin >/dev/null 2>&1 || "$GIT" remote add origin "$REPO_URL_ORIGIN"
    "$GIT" remote get-url upstream >/dev/null 2>&1 || "$GIT" remote add upstream "$REPO_URL_UPSTREAM"
    "$GIT" fetch origin "$FORK_BRANCH" 2>&1 | tail -3
    "$GIT" fetch upstream master 2>&1 | tail -3
    local behind_ahead_origin behind_ahead_upstream
    behind_ahead_origin=$("$GIT" rev-list --left-right --count HEAD...origin/"$FORK_BRANCH" 2>/dev/null)
    behind_ahead_upstream=$("$GIT" rev-list --left-right --count HEAD...upstream/master 2>/dev/null)
    log "HEAD vs origin/$FORK_BRANCH (left=behind right=ahead): $behind_ahead_origin"
    log "HEAD vs upstream/master (left=behind right=ahead): $behind_ahead_upstream"
    log "fetch done. 'merge' will fast-forward/merge origin first."
}

cmd_merge() {
    local WITH_UPSTREAM=0
    [ "${1:-}" = "--with-upstream" ] && WITH_UPSTREAM=1
    if ! "$GIT" rev-parse --verify -q origin/"$FORK_BRANCH" >/dev/null 2>&1; then
        warn "origin/$FORK_BRANCH not fetched yet — running fetch first"
        cmd_fetch
    fi
    log "merging origin/$FORK_BRANCH (latest TurboQuant state)..."
    if ! "$GIT" merge --no-edit origin/"$FORK_BRANCH" 2>&1 | tail -8; then
        warn "conflicts while merging TurboQuant — aborting; resolve manually then re-run"
        "$GIT" merge --abort 2>/dev/null
        return 1
    fi
    if [ "$WITH_UPSTREAM" = "1" ]; then
        log "merging upstream/master (latest llama.cpp)..."
        if ! "$GIT" merge --no-edit upstream/master 2>&1 | tail -12; then
            warn "conflicts while merging upstream llama.cpp — aborted."
            warn "MAS-touched files live in: src/llama-mas.*, src/llama-layer-graph.*,"
            warn "src/llama-context.*, src/llama.cpp, src/models/llama.cpp, src/llama-kv-cache.*,"
            warn "include/llama.h, ggml/src/ggml-cuda/*mas*, tools/mas-server/, tests/*"
            "$GIT" merge --abort 2>/dev/null
            return 1
        fi
    fi
    log "merge OK."
}

cmd_build() {
    local BUILD_DIR="${BUILD_DIR:-build-cuda}"   # keep build/ (CPU + tests) untouched
    log "configuring build in $BUILD_DIR (CUDA=ON if nvcc found)..."
    local vcvars
    vcvars=$(find_vcvars) || warn "no VS vcvars found — configure may fail"
    local CUDA_FLAG=OFF
    if command -v nvcc >/dev/null 2>&1; then CUDA_FLAG=ON; fi
    if [ -n "$vcvars" ]; then
        run_with_vcvars "$vcvars" \
            "$CMake" -S "$ROOT" -B "$ROOT/$BUILD_DIR" -DGGML_CUDA=$CUDA_FLAG -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_GRAPHS=ON -DLLAMA_BUILD_TESTS=ON
    else
        "$CMake" -S "$ROOT" -B "$ROOT/$BUILD_DIR" -DGGML_CUDA=$CUDA_FLAG -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_GRAPHS=ON -DLLAMA_BUILD_TESTS=ON
    fi
    log "building mas-server, latent-bench, tests..."
    if [ -n "$vcvars" ]; then
        run_with_vcvars "$vcvars" \
            "$CMake" --build "$ROOT/$BUILD_DIR" --target mas-server latent-bench test-jinja test-json-partial test-quantize-fns test-quant-type-selection test-mas-api --config Release -j 8
    else
        "$CMake" --build "$ROOT/$BUILD_DIR" --target mas-server latent-bench --config Release -j 8
    fi
    log "build done."
}

run_server_suite() { # <model> — start mas-server, run both python suites, stop it
    local model="$1"
    local server_log="$ROOT/build/mas-server-test.log"
    ( cd "$ROOT" && ./build/bin/mas-server.exe -m "$model" -p 9000 --n-ctx 512 --type-k turbo4 --type-v turbo4 --flash-attn on --n-agents-max 8 > "$server_log" 2>&1 & )
    SERVER_PID=$!
    for i in $(seq 1 30); do
        if curl -s -m 2 http://127.0.0.1:9000/health >/dev/null 2>&1; then break; fi
        sleep 1
    done
    local health
    health=$(curl -s -m 3 http://127.0.0.1:9000/health)
    if [ -z "$health" ]; then
        warn "mas-server did not come up (see $server_log) — skipping server suites"
        return 0
    fi
    log "server up: $(echo "$health" | head -c 120)"
    log "running tool suite (test_mas_server.py)..."
    ( cd "$ROOT" && py -3.12 tools/mas-server/test_mas_server.py 2>&1 | tail -12 )
    log "running latent suite (test_mas_latent.py)..."
    ( cd "$ROOT" && py -3.12 tools/mas-server/test_mas_latent.py 2>&1 | tail -20 )
    cmd.exe /c 'taskkill /IM mas-server.exe /F' >/dev/null 2>&1 || true
}

cmd_test() {
    log "ctest (fast subset)..."
    local model="$ROOT/build/tinyllamas/stories15M-q4_0.gguf"
    if [ -f "$ROOT/build/bin/test-jinja.exe" ]; then
        ( cd "$ROOT" && ./build/bin/test-jinja 2>&1 | tail -4 )
        ( cd "$ROOT" && ./build/bin/test-json-partial 2>&1 | tail -2 )
        ( cd "$ROOT" && ./build/bin/test-quantize-fns 2>&1 | tail -2 )
        ( cd "$ROOT" && ./build/bin/test-quant-type-selection 2>&1 | tail -3 )
        ( cd "$ROOT" && ./build/bin/test-mas-api 2>&1 | tail -3 )
    else
        warn "test binaries missing — run 'build' first"
    fi
    if [ -f "$model" ]; then
        run_server_suite "$model"
    else
        warn "no tiny test model at $model — skipping mas-server suites (download stories15M-q4_0.gguf there)"
    fi
    if [ -f "$ROOT/build-codex/bin/latent-bench.exe" ]; then
        log "latent-bench (CUDA build)..."
        ( cd "$ROOT" && GGML_CUDA_FORCE_GRAPHS=1 ./build-codex/bin/latent-bench.exe "$model" "2 + 2 = ?" 48 2>&1 | tail -6 )
    fi
}

cmd_fresh() {
    local dest="$1"
    [ -n "$dest" ] || die "fresh needs a destination dir"
    [ -d "$dest" ] && die "destination exists: $dest"
    # Default: base = LATEST llama.cpp (ggml-org master), then merge TurboQuant
    # (TheTom's fork branch), then the MAS overlay. --from-turboquant starts
    # from TheTom's fork instead.
    local base_repo="$REPO_URL_UPSTREAM"
    if [ "${2:-}" = "--from-turboquant" ]; then
        base_repo="$REPO_URL_ORIGIN"
    fi
    # The MAS overlay lives ONLY as the detached worktree HEAD (no branch or
    # tag points at it) — so we must fetch it directly from this repo's gitdir.
    local src_gitdir
    src_gitdir=$(head -1 "$ROOT/.git" 2>/dev/null | sed 's/^gitdir: //')
    [ -n "$src_gitdir" ] || src_gitdir="$ROOT/.git"
    log "cloning latest llama.cpp (full history — depth-1 breaks merge ancestry) from $base_repo ..."
    "$GIT" clone "$base_repo" "$dest" || die "clone failed"
    cd "$dest" || die "cannot cd"
    if [ "$base_repo" = "$REPO_URL_UPSTREAM" ]; then
        log "merging TurboQuant (origin/$FORK_BRANCH) onto latest llama.cpp..."
        "$GIT" remote add tq "$REPO_URL_ORIGIN" || die "cannot add tq remote"
        "$GIT" fetch tq "$FORK_BRANCH" || die "cannot fetch $FORK_BRANCH from TheTom fork"
        merge_or_guide "tq/$FORK_BRANCH" "TurboQuant" || return 1
    fi
    log "merging the MAS overlay from $src_gitdir (worktree HEAD)..."
    "$GIT" fetch "$src_gitdir" HEAD || die "cannot fetch overlay HEAD from $src_gitdir"
    local OVERLAY_SHA
    OVERLAY_SHA=$("$GIT" rev-parse FETCH_HEAD)
    merge_or_guide "FETCH_HEAD" "MAS overlay" || return 1
    repair_overlay_gaps "$OVERLAY_SHA"
    grep -q "add_subdirectory(mas-server)" tools/CMakeLists.txt 2>/dev/null || \
        printf '\nadd_subdirectory(mas-server)\n' >> tools/CMakeLists.txt
    # The overlay repo tracks its machine-local build dirs (auto-commit
    # workflow). Their CMakeCache points at the OLD machine paths and breaks
    # the fresh clone's first cmake run — drop them.
    if "$GIT" ls-files | grep -q "^build/\|^build-codex/\|^build-cuda/"; then
        log "dropping tracked build dirs imported by the overlay..."
        "$GIT" rm -r --cached --ignore-unmatch build build-codex build-cuda >/dev/null 2>&1
        rm -rf build build-codex build-cuda
        "$GIT" commit -m "fresh: drop tracked build dirs (machine-local artifacts)" 2>&1 | tail -1
    fi
    log "fresh checkout ready at $dest — cd there and run ./scripts/llama-tqp-mas.sh build"
    log "note: script/overlay files under scripts/ docs/ tools/mas-server/ come with"
    log "      the overlay; the build script is at scripts/llama-tqp-mas.sh"
}

repair_overlay_gaps() { # <overlay-ref>
    # The overlay's tests/CMakeLists.txt references sources upstream has
    # renamed/deleted (get-model.cpp, export-graph-ops.cpp, test-json-partial,
    # test-regex-partial, test-mas-api...). Restore any missing from the overlay.
    local overlay_ref="$1"
    [ -z "$overlay_ref" ] && return 0
    if ! command -v python >/dev/null 2>&1; then
        warn "repair_overlay_gaps: python not found — skip; resolve missing test sources by hand"
        return 0
    fi
    python - "$overlay_ref" <<'PYEOF'
import re, subprocess, sys, os
root = os.getcwd()
overlay = sys.argv[1]
cm = open(os.path.join(root, "tests", "CMakeLists.txt"), encoding="utf-8", errors="replace").read()
refs = set(re.findall(r'llama_(?:build|build_and_test|test)\(\s*([\w\-\.]+\.cpp)', cm))
missing = [r for r in refs if not os.path.exists(os.path.join(root, "tests", r))]
if not missing:
    print("repair_overlay_gaps: all test sources present")
    sys.exit(0)
GIT = "C:/PortableGit/bin/git.exe"
restored = []
for r in missing:
    blob = subprocess.run([GIT, "show", f"{overlay}:tests/{r}"], capture_output=True, text=True)
    if blob.returncode == 0:
        with open(os.path.join(root, "tests", r), "wb") as f:
            f.write(blob.stdout.encode("utf-8"))
        restored.append(r)
    else:
        print("  not in overlay:", r)
if restored:
    subprocess.run([GIT, "add"] + [os.path.join("tests", r) for r in restored], check=True)
    subprocess.run([GIT, "commit", "-m", "fresh: restore test sources dropped by merge (from overlay)"],
                   env={**os.environ, "GIT_EDITOR": "true"}, check=True)
print("repair_overlay_gaps: restored", restored)
# tools/mas-server (server + its CMakeLists) is overlay-only; the merge can
# delete the whole dir. Restore it when missing.
if not os.path.exists(os.path.join(root, "tools", "mas-server", "mas-server.cpp")):
    os.makedirs(os.path.join(root, "tools", "mas-server"), exist_ok=True)
    for f in ("mas-server.cpp", "CMakeLists.txt", "test_mas_server.py"):
        blob = subprocess.run([GIT, "show", f"{overlay}:tools/mas-server/{f}"], capture_output=True, text=True)
        if blob.returncode == 0:
            with open(os.path.join(root, "tools", "mas-server", f), "wb") as fh:
                fh.write(blob.stdout.encode("utf-8"))
    subprocess.run(["C:/PortableGit/bin/git.exe", "add", "tools/mas-server"], check=True)
    subprocess.run(["C:/PortableGit/bin/git.exe", "commit", "-m", "fresh: restore tools/mas-server from overlay"],
                   env={**os.environ, "GIT_EDITOR": "true"}, check=True)
    print("repair_overlay_gaps: restored tools/mas-server")
PYEOF
}

merge_or_guide() { # <ref> <label>
    local ref="$1" label="$2"
    local merge_out
    merge_out=$("$GIT" merge --no-edit "$ref" 2>&1)
    local merge_rc=$?
    if [ $merge_rc -ne 0 ]; then
        printf '%s\n' "$merge_out" | tail -14
        warn "[$label] merge hit conflicts — left in conflict state for resolution."
        warn "  Resolve the conflicted files, then: git add <files> && git merge --continue"
        warn "  To accept the merged-in side for a conflicted file:  git checkout $ref -- <file>"
        warn "  TIP (tests/CMakeLists.txt conflicts): prefer the BASE version, then append:"
        warn "    llama_build_and_test(test-mas-api.cpp)"
        warn "    target_include_directories(test-mas-api PRIVATE \${PROJECT_SOURCE_DIR}/src)"
        return 1
    fi
    log "[$label] merged cleanly."
    return 0
}

cmd_all() { cmd_fetch; cmd_merge "${1:-}"; cmd_build; cmd_test; }

case "${1:-help}" in
    fetch)  cmd_fetch ;;
    merge)  shift; cmd_merge "${1:-}" ;;
    build)  cmd_build ;;
    test)   cmd_test ;;
    fresh)  shift; cmd_fresh "${1:-}" ;;
    all)    shift; cmd_all "${1:-}" ;;
    *)
        sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
esac