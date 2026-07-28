//! The `whetstone` command-line tool.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

mod bench;
mod chat;
mod convert;
mod eval;
mod inspect;
mod probe;
mod run;
mod tune;
mod verify;

/// Version string including build provenance.
///
/// A released binary compiles for exactly one GPU architecture, so the arch is
/// as much a part of its identity as the version number.
fn long_version() -> &'static str {
    concat!(
        env!("CARGO_PKG_VERSION"),
        "\ncommit:     ", env!("WHETSTONE_GIT_SHA"),
        "\nbuilt:      ", env!("WHETSTONE_BUILD_DATE"),
        "\ntarget:     ", env!("WHETSTONE_TARGET"),
        "\ncuda arch:  sm_", env!("WHETSTONE_CUDA_ARCH"),
    )
}

#[derive(Parser)]
#[command(
    name = "whetstone",
    version,
    long_version = long_version(),
    about = "A low-bit LLM inference engine that trades multiplication for bit arithmetic.",
    after_help = "Docs: https://github.com/Anonym0usWork1221/Whetstone"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Report GPU capabilities and measure every arithmetic path.
    ///
    /// Whetstone dispatches on measured facts rather than the spec sheet, and
    /// this is where those facts come from.
    Probe {
        /// Iterations per microbenchmark. Higher is less noisy.
        #[arg(long, default_value_t = 100_000)]
        iters: i32,
        /// Buffer size in MiB for the bandwidth measurement.
        #[arg(long, default_value_t = 256)]
        bandwidth_mib: usize,
    },

    /// Convert a HuggingFace checkpoint into a quantized `.wstone` file.
    ///
    /// The conversion is the whole point: a `.wstone` stores weights already in
    /// the bit layout Whetstone's kernels index, so loading is an mmap rather
    /// than a decode.
    Convert {
        /// Model directory containing config.json and model.safetensors.
        model: std::path::PathBuf,
        /// Output path.
        #[arg(short, long, default_value = "model.wstone")]
        out: std::path::PathBuf,
        /// Precision for the output projection. int4 is the largest single
        /// bandwidth win and the largest quality risk.
        #[arg(long, value_enum, default_value_t = convert::HeadPrecision::Fp16)]
        head: convert::HeadPrecision,
        /// Precision for the transformer blocks. fp16 is the lossless reference
        /// path: it is what separates an engine bug from quantization damage.
        #[arg(long, value_enum, default_value_t = convert::BodyPrecision::Int4)]
        body: convert::BodyPrecision,
        /// Memory bandwidth in GB/s for the reported ceiling.
        #[arg(long)]
        bandwidth: Option<f64>,
    },

    /// Check a `.wstone` file's integrity, and optionally its fidelity.
    Verify {
        /// The .wstone file.
        file: std::path::PathBuf,
        /// Source checkpoint, to measure quantization error against.
        #[arg(long)]
        source: Option<std::path::PathBuf>,
        /// Memory bandwidth in GB/s for the reported ceiling.
        #[arg(long)]
        bandwidth: Option<f64>,
    },

    /// Execute a `.wstone` model and report tokens/second.
    ///
    /// The prompt is given as token ids. Whetstone's tokenizer lives in
    /// `bench/tokenizer.py`; keeping ids on this interface means the engine can
    /// be timed and diffed without a tokenizer in the loop at all.
    Run {
        /// The .wstone file.
        model: std::path::PathBuf,
        /// Comma-separated prompt token ids.
        #[arg(long, required = true)]
        ids: String,
        /// Tokens to generate.
        #[arg(long, default_value_t = 128)]
        max_new: usize,
        /// KV cache capacity in tokens.
        #[arg(long, default_value_t = 2048)]
        ctx: usize,
        /// Sampling temperature. Zero is greedy, and greedy never leaves the GPU.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Nucleus mass, when sampling.
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,
        /// PRNG seed, when sampling.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Write the final-position output distribution here, for the quality
        /// gates to diff against the fp64 reference.
        #[arg(long)]
        dump_logits: Option<std::path::PathBuf>,
        /// Print only `decode_tok_s prefill_tok_s [ids]`.
        #[arg(long)]
        quiet: bool,
        /// Report a per-stage time breakdown over N steps instead of generating.
        #[arg(long, default_value_t = 0)]
        profile: usize,
        /// Capture the decode step as a CUDA graph: ~250 launches become 1.
        #[arg(long)]
        graph: bool,
        /// Force an int4 GEMV variant (see `whetstone bench`). -1 = baseline.
        #[arg(long)]
        gemv_variant: Option<i32>,
    },

    /// Interactive chat, with throughput reported per turn.
    ///
    /// The KV cache is kept across turns, so each turn only prefills its own
    /// message rather than re-sending the transcript.
    Chat {
        /// The .wstone file.
        model: std::path::PathBuf,
        /// Directory holding tokenizer.json, if the .wstone has none embedded.
        #[arg(long)]
        tokenizer: Option<std::path::PathBuf>,
        /// System prompt.
        #[arg(long)]
        system: Option<String>,
        /// KV cache capacity in tokens; the conversation lives here.
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        /// Maximum tokens per reply.
        #[arg(long, default_value_t = 512)]
        max_new: usize,
        /// Sampling temperature. 0 is greedy and never leaves the GPU.
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        /// Nucleus mass.
        #[arg(long, default_value_t = 0.8)]
        top_p: f32,
        /// PRNG seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Answer this once and exit, instead of reading from the terminal.
        #[arg(long)]
        prompt: Option<String>,
    },

    /// Perplexity over a token stream, the headline quality gate.
    ///
    /// Compare against the fp16 baseline taken with the same tokens, the same
    /// window and the same count. A perplexity quoted without those three is not
    /// comparable to anything.
    Ppl {
        /// The .wstone file.
        model: std::path::PathBuf,
        /// File of little-endian u32 token ids.
        #[arg(long)]
        tokens: std::path::PathBuf,
        /// Window length. Each window starts from an empty KV cache.
        #[arg(long, default_value_t = 2048)]
        window: usize,
        /// Maximum windows to evaluate.
        #[arg(long, default_value_t = 20)]
        windows: usize,
        /// Write the result as JSON here.
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },

    /// Dump final-position logits for a prompt set, as raw f32.
    ///
    /// The input is a JSON array of token-id arrays; the output feeds the
    /// top-1-agreement and KL comparison against the reference.
    Logits {
        /// The .wstone file.
        model: std::path::PathBuf,
        /// JSON array of prompt id arrays.
        #[arg(long)]
        prompts: std::path::PathBuf,
        /// Raw f32 output path.
        #[arg(long)]
        out: std::path::PathBuf,
        /// KV cache capacity.
        #[arg(long, default_value_t = 2048)]
        ctx: usize,
    },

    /// Sweep int4 GEMV kernel variants across every shape the model issues.
    Bench {
        /// Timed iterations per measurement.
        #[arg(long, default_value_t = 100)]
        reps: i32,
        /// Independent measurements to take the minimum of.
        #[arg(long, default_value_t = 3)]
        repeats: usize,
    },

    /// Sweep the per-shape GEMV rule by whole-generation throughput.
    ///
    /// Slower than a microbenchmark and the only method that has not misranked
    /// these kernels -- both a microbenchmark and an event profile picked rules
    /// that measured worse end to end.
    Tune {
        /// The .wstone file.
        model: std::path::PathBuf,
        /// Comma-separated prompt token ids.
        #[arg(long, default_value = "785,6722,315,9625,374")]
        ids: String,
        /// Tokens generated per sample.
        #[arg(long, default_value_t = 256)]
        tokens: usize,
        /// Samples per rule; the best is kept, as the least-contended estimate.
        #[arg(long, default_value_t = 2)]
        samples: usize,
    },

    /// Inspect a checkpoint: architecture, tensor inventory, and roofline.
    Inspect {
        /// Model directory containing config.json and model.safetensors.
        model: std::path::PathBuf,
        /// Memory bandwidth in GB/s for the roofline table. Defaults to the
        /// detected GPU's peak.
        #[arg(long)]
        bandwidth: Option<f64>,
        /// List every tensor rather than a per-layer summary.
        #[arg(long)]
        tensors: bool,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Probe { iters, bandwidth_mib } => {
            probe::run(iters, bandwidth_mib).context("probe failed")
        }
        Command::Inspect { model, bandwidth, tensors } => {
            inspect::run(&model, bandwidth, tensors).context("inspect failed")
        }
        Command::Convert { model, out, head, body, bandwidth } => {
            convert::run(&model, &out, head, body, bandwidth).context("convert failed")
        }
        Command::Verify { file, source, bandwidth } => {
            verify::run(&file, source.as_deref(), bandwidth).context("verify failed")
        }
        Command::Chat {
            model,
            tokenizer,
            system,
            ctx,
            max_new,
            temperature,
            top_p,
            seed,
            prompt,
        } => chat::run(chat::ChatArgs {
            model: &model,
            tokenizer: tokenizer.as_deref(),
            system,
            ctx,
            max_new,
            temperature,
            top_p,
            seed,
            prompt,
        })
        .context("chat failed"),
        Command::Tune { model, ids, tokens, samples } => {
            tune::run(&model, &parse_ids(&ids)?, tokens, samples).context("tune failed")
        }
        Command::Bench { reps, repeats } => {
            bench::run(reps, repeats).context("bench failed")
        }
        Command::Ppl { model, tokens, window, windows, out } => {
            eval::perplexity(&model, &tokens, window, windows, out.as_deref())
                .context("perplexity failed")
        }
        Command::Logits { model, prompts, out, ctx } => {
            eval::logits(&model, &prompts, &out, ctx).context("logit dump failed")
        }
        Command::Run {
            model,
            ids,
            max_new,
            ctx,
            temperature,
            top_p,
            seed,
            dump_logits,
            quiet,
            profile,
            graph,
            gemv_variant,
        } => {
            let parsed = parse_ids(&ids)?;
            run::run(run::RunArgs {
                model: &model,
                ids: parsed,
                max_new,
                ctx,
                temperature,
                top_p,
                seed,
                dump_logits: dump_logits.as_deref(),
                quiet,
                profile,
                graph,
                gemv_variant,
            })
            .context("run failed")
        }
    }
}

/// Parses `"785,6722,315"` into token ids, rejecting anything else.
///
/// A malformed id silently becoming 0 would produce a valid-looking generation
/// from the wrong prompt, which is exactly the class of error that is invisible
/// until someone decodes the tokens.
fn parse_ids(s: &str) -> Result<Vec<u32>> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<u32>()
                .with_context(|| format!("{t:?} is not a token id"))
        })
        .collect()
}
