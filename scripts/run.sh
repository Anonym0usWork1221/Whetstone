#!/usr/bin/env bash
#
# Whetstone launcher. Ships inside the release package as ./run.sh
#
#   ./run.sh probe                          what this GPU can do
#   ./run.sh inspect  <model_dir>           architecture + roofline
#   ./run.sh convert  <model_dir> [out]     build a .wstone
#   ./run.sh verify   <file.wstone> [src]   check integrity and fidelity
#   ./run.sh run      <file.wstone> --ids   generate, and report tok/s
#   ./run.sh ppl      <file.wstone> <toks>  wikitext-2 perplexity
#   ./run.sh tune     <file.wstone>         pick the per-shape GEMV kernel
#   ./run.sh tokens   [model_dir] [out]     tokenize wikitext-2 for ppl
#   ./run.sh chat     <file.wstone>         interactive chat, tok/s per turn
#   ./run.sh hfchat   [model_dir]           the HuggingFace chat harness
#   ./run.sh bench    [model_dir]           throughput run
#   ./run.sh download [dest]                fetch the reference model
#   ./run.sh setup                          create the Python venv
#   ./run.sh doctor                         diagnose the environment
#
# The Rust subcommands need only the binary. The Python ones (chat, bench,
# tokens, download) need a virtualenv, which `setup` creates in ./.venv.
#
# A typical first run:
#
#   ./run.sh download
#   ./run.sh convert models/Qwen2.5-0.5B-Instruct model.wstone --head int4
#   ./run.sh chat    model.wstone
#   ./run.sh run     model.wstone --ids 785,6722,315,9625,374 --max-new 256 --graph
#   ./run.sh tokens  models/Qwen2.5-0.5B-Instruct wikitext2.u32
#   ./run.sh ppl     model.wstone wikitext2.u32

set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HERE/bin/whetstone"
[[ -x "$BIN" ]] || BIN="$HERE/target/release/whetstone"   # running from a source tree
BENCH="$HERE/bench"
VENV="$HERE/.venv"
PY="$VENV/bin/python"
DEFAULT_MODEL="$HERE/models/Qwen2.5-0.5B-Instruct"

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m  ok\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarn:\033[0m %s\n' "$*" >&2; }

usage() { sed -n '2,/^set -/p' "$0" | sed 's/^# \{0,1\}//; $d'; exit "${1:-0}"; }

need_bin() {
    [[ -x "$BIN" ]] || die "whetstone binary not found at $BIN
Build it with:  cargo build --release"
}

need_py() {
    if [[ ! -x "$PY" ]]; then
        die "Python environment not set up. Run:  ./run.sh setup"
    fi
}

resolve_model() {
    local m="${1:-}"
    if [[ -z "$m" ]]; then
        [[ -d "$DEFAULT_MODEL" ]] || die \
            "no model given and none at $DEFAULT_MODEL
Fetch the reference model with:  ./run.sh download"
        echo "$DEFAULT_MODEL"
    else
        [[ -d "$m" ]] || die "no such model directory: $m"
        echo "$m"
    fi
}

cmd="${1:-}"
[[ $# -gt 0 ]] && shift || true

case "$cmd" in
    probe)   need_bin; exec "$BIN" probe "$@" ;;
    inspect) need_bin; exec "$BIN" inspect "$@" ;;
    verify)  need_bin
             f="${1:?usage: run.sh verify <file.wstone> [source_model_dir]}"; shift || true
             if [[ $# -gt 0 ]]; then exec "$BIN" verify "$f" --source "$1"
             else exec "$BIN" verify "$f"; fi ;;

    convert)
        need_bin
        model="$(resolve_model "${1:-}")"; shift || true
        out="${1:-$HERE/model.wstone}"; shift || true
        exec "$BIN" convert "$model" -o "$out" "$@"
        ;;

    run)
        need_bin
        f="${1:?usage: run.sh run <file.wstone> --ids 785,6722,... [--max-new N]}"; shift || true
        exec "$BIN" run "$f" "$@"
        ;;

    ppl)
        need_bin
        f="${1:?usage: run.sh ppl <file.wstone> <tokens.u32> [--windows N]}"; shift || true
        t="${1:?usage: run.sh ppl <file.wstone> <tokens.u32> [--windows N]}"; shift || true
        exec "$BIN" ppl "$f" --tokens "$t" "$@"
        ;;

    tune)
        need_bin
        f="${1:?usage: run.sh tune <file.wstone>}"; shift || true
        exec "$BIN" tune "$f" "$@"
        ;;

    tokens)
        need_py
        model="$(resolve_model "${1:-}")"; [[ $# -gt 0 ]] && shift || true
        out="${1:-$HERE/tokens.u32}"; [[ $# -gt 0 ]] && shift || true
        exec "$PY" "$BENCH/prepare_tokens.py" --model "$model" --out "$out" "$@"
        ;;

    chat)
        # Native, not bench/chat.py: the whole point is that no Python sits in
        # the token loop, and the binary carries its own tokenizer.
        need_bin
        f="${1:?usage: run.sh chat <file.wstone> [--temperature 0] [--ctx N]}"; shift || true
        exec "$BIN" chat "$f" "$@"
        ;;

    hfchat)
        need_py
        model="$(resolve_model "${1:-}")"; [[ $# -gt 0 ]] && shift || true
        exec "$PY" "$BENCH/chat.py" --model "$model" "$@"
        ;;

    bench)
        need_py
        model="$(resolve_model "${1:-}")"; [[ $# -gt 0 ]] && shift || true
        exec "$PY" "$BENCH/chat.py" --model "$model" --bench "$@"
        ;;

    baseline)
        need_py
        model="$(resolve_model "${1:-}")"; [[ $# -gt 0 ]] && shift || true
        exec "$PY" "$BENCH/baseline_hf.py" --model "$model" "$@"
        ;;

    download)
        need_py
        dest="${1:-$HERE/models}"
        exec "$PY" "$BENCH/download_model.py" --out "$dest"
        ;;

    setup)
        info "creating Python environment in $VENV"
        command -v python3 >/dev/null || die "python3 not found"
        python3 -m venv "$VENV" || die "could not create venv"
        "$VENV/bin/pip" install --quiet --upgrade pip
        info "installing torch (large; this can take a while)"
        "$VENV/bin/pip" install --quiet torch || die "torch install failed"
        "$VENV/bin/pip" install --quiet transformers safetensors huggingface_hub regex \
            || die "dependency install failed"
        ok "environment ready"
        echo
        echo "  next:  ./run.sh download   # fetch the reference model"
        echo "         ./run.sh chat       # talk to it"
        ;;

    doctor)
        info "environment"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
                       --format=csv,noheader | sed 's/^/  gpu       /'
        else
            warn "nvidia-smi not found — no NVIDIA driver?"
        fi
        if [[ -x "$BIN" ]]; then
            "$BIN" --version | sed 's/^/  /'
        else
            warn "whetstone binary missing at $BIN"
        fi
        if [[ -x "$PY" ]]; then
            ok "python $("$PY" --version 2>&1 | awk '{print $2}')"
            "$PY" - <<'EOF' 2>/dev/null || warn "torch not importable"
import importlib
for m in ("torch", "transformers"):
    try:
        mod = importlib.import_module(m)
        print(f"  ok   {m} {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"  --   {m} not installed")
EOF
        else
            warn "no Python environment — run ./run.sh setup"
        fi
        [[ -d "$DEFAULT_MODEL" ]] && ok "model $DEFAULT_MODEL" \
            || warn "reference model not downloaded — run ./run.sh download"
        echo
        info "capability check"
        [[ -x "$BIN" ]] && "$BIN" probe --iters 20000 2>&1 | sed -n '/capabilities/,/fp8/p' | sed 's/^/  /'
        ;;

    ""|-h|--help|help) usage 0 ;;
    *) printf '\033[31merror:\033[0m unknown command: %s\n\n' "$cmd" >&2; usage 1 ;;
esac
