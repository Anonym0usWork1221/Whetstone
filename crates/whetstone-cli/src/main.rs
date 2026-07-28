//! The `whetstone` command-line tool.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

mod convert;
mod inspect;
mod probe;
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
        Command::Convert { model, out, head, bandwidth } => {
            convert::run(&model, &out, head, bandwidth).context("convert failed")
        }
        Command::Verify { file, source, bandwidth } => {
            verify::run(&file, source.as_deref(), bandwidth).context("verify failed")
        }
    }
}
