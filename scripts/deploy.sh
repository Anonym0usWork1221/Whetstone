#!/usr/bin/env bash
#
# Builds a release package for Linux.
#
#   ./scripts/deploy.sh                    # build, test, package for sm_75
#   ./scripts/deploy.sh --arch 86          # build for a different GPU
#   ./scripts/deploy.sh --skip-tests       # CI already ran them
#   ./scripts/deploy.sh --out /tmp/dist    # somewhere else
#
# Produces dist/whetstone-<version>-linux-x86_64-sm<arch>.tar.gz plus a
# SHA256 checksum.
#
# The CUDA architecture is part of the artifact name on purpose. Whetstone
# compiles for exactly one GPU family -- the kernels use capabilities that
# differ by architecture -- so a package built for sm_75 will not run on an
# older card, and shipping it under a generic name would be misleading.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARCH="${WHETSTONE_CUDA_ARCH:-75}"
OUT="$ROOT/dist"
SKIP_TESTS=0
KEEP_DIR=0

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarn:\033[0m %s\n' "$*" >&2; }
ok()   { printf '\033[32m  ok\033[0m %s\n' "$*"; }

usage() {
    sed -n '2,/^set -/p' "$0" | sed 's/^# \{0,1\}//; $d'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)       ARCH="${2:?--arch needs a value}"; shift 2 ;;
        --out)        OUT="${2:?--out needs a value}"; shift 2 ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        --keep-dir)   KEEP_DIR=1; shift ;;
        -h|--help)    usage ;;
        *)            die "unknown option: $1 (try --help)" ;;
    esac
done

# ---------------------------------------------------------------- preflight

info "preflight"

command -v cargo >/dev/null || die "cargo not found. Install Rust: https://rustup.rs"
ok "cargo $(cargo --version | awk '{print $2}')"

command -v nvcc >/dev/null || die \
    "nvcc not found. Whetstone needs the CUDA toolkit:
     Ubuntu/Debian:  sudo apt install nvidia-cuda-toolkit
     or set CUDA_PATH to your toolkit root."
ok "nvcc $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"

if ! nvcc --list-gpu-arch 2>/dev/null | grep -q "compute_${ARCH}"; then
    die "this nvcc cannot target sm_${ARCH}. Supported:
$(nvcc --list-gpu-arch 2>/dev/null | sed 's/compute_/  sm_/' | tr '\n' ' ')"
fi
ok "nvcc supports sm_${ARCH}"

# A GPU is not required to *build*, only to run the tests.
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
    [[ -n "$GPU" ]] && ok "gpu $GPU"
    HAVE_GPU=1
else
    warn "no nvidia-smi; building without running GPU tests"
    HAVE_GPU=0
    SKIP_TESTS=1
fi

VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' Cargo.toml | head -1)"
[[ -n "$VERSION" ]] || die "could not read version from Cargo.toml"

# Provenance. Absent in a source tarball, which is fine -- the binary reports
# "unknown" rather than claiming a commit it was not built from.
if git rev-parse --git-dir >/dev/null 2>&1; then
    GIT_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
    if [[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
        GIT_SHA="${GIT_SHA}-dirty"
        warn "working tree has uncommitted changes; marking build as dirty"
    fi
else
    GIT_SHA="unknown"
fi

NAME="whetstone-${VERSION}-linux-x86_64-sm${ARCH}"
STAGE="${OUT}/${NAME}"

info "building whetstone ${VERSION} (${GIT_SHA}) for sm_${ARCH}"

# ------------------------------------------------------------------- build

export WHETSTONE_CUDA_ARCH="$ARCH"
export WHETSTONE_GIT_SHA="$GIT_SHA"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(date -u +%s)}"

cargo build --release --locked 2>&1 | grep -Ev '^\s*(Compiling|Finished)' || true
[[ -x target/release/whetstone ]] || die "build produced no binary"
ok "built target/release/whetstone"

if [[ "$SKIP_TESTS" -eq 0 ]]; then
    info "running correctness tests"
    cargo test --release --locked 2>&1 | grep -E "test result:" | sed 's/^/  /'
    ok "correctness tests passed"

    # Performance checks are reported, not enforced. They are timing-based, so
    # running them right after a build measures machine contention as much as
    # the kernels -- a flaky release gate is worse than no gate.
    info "performance checks (informational)"
    if cargo test --release --locked -- --ignored --nocapture 2>&1 \
         | grep -E "GB/s|test result:" | sed 's/^/  /'; then
        :
    else
        warn "performance checks did not pass; not blocking the release"
    fi
else
    warn "tests skipped"
fi

# ----------------------------------------------------------------- package

info "packaging"

rm -rf "$STAGE"
mkdir -p "$STAGE"/{bin,bench,docs}

install -m 0755 target/release/whetstone "$STAGE/bin/whetstone"
install -m 0644 bench/chat.py bench/baseline_hf.py bench/reference_numpy.py \
                bench/tokenizer.py bench/prepare_tokens.py "$STAGE/bench/"
install -m 0644 README.md LICENSE "$STAGE/"
install -m 0644 docs/FORMAT.md docs/ROADMAP.md "$STAGE/docs/"
[[ -f CHANGELOG.md ]] && install -m 0644 CHANGELOG.md "$STAGE/"
[[ -f scripts/download_model.py ]] && install -m 0644 scripts/download_model.py "$STAGE/bench/"

install -m 0755 scripts/run.sh "$STAGE/run.sh"

cat > "$STAGE/VERSION" <<EOF
version:    ${VERSION}
commit:     ${GIT_SHA}
built:      $(date -u -d "@${SOURCE_DATE_EPOCH}" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)
target:     x86_64-unknown-linux-gnu
cuda arch:  sm_${ARCH}
nvcc:       $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
rustc:      $(rustc --version | awk '{print $2}')
EOF

# Runtime requirements, so a user who unpacks this on the wrong machine finds
# out immediately rather than from a linker error.
cat > "$STAGE/REQUIREMENTS.txt" <<EOF
Whetstone ${VERSION} — Linux x86_64, CUDA sm_${ARCH}

Required:
  * NVIDIA GPU with compute capability ${ARCH:0:1}.${ARCH:1:1}
    (sm_75 = Turing: RTX 2060/2070/2080, GTX 1650 Super/1660, T4, Quadro RTX)
  * NVIDIA driver new enough for the CUDA runtime below
  * libcudart.so — from the CUDA toolkit or runtime redistributable

Optional, for the Python benchmark and chat harness:
  * Python 3.10+
  * pip install torch transformers safetensors regex

Check your card:
    nvidia-smi --query-gpu=name,compute_cap --format=csv
    ./bin/whetstone probe
EOF

( cd "$OUT" && tar czf "${NAME}.tar.gz" "$NAME" )
( cd "$OUT" && sha256sum "${NAME}.tar.gz" > "${NAME}.tar.gz.sha256" )

[[ "$KEEP_DIR" -eq 1 ]] || rm -rf "$STAGE"

SIZE="$(du -h "${OUT}/${NAME}.tar.gz" | cut -f1)"

echo
info "done"
echo "  ${OUT}/${NAME}.tar.gz  (${SIZE})"
echo "  ${OUT}/${NAME}.tar.gz.sha256"
echo
echo "  verify:  sha256sum -c ${NAME}.tar.gz.sha256"
echo "  unpack:  tar xzf ${NAME}.tar.gz && cd ${NAME} && ./run.sh probe"
echo
