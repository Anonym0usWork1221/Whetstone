//! `whetstone probe` — what this GPU can actually do.

use anyhow::{bail, Result};
use whetstone_kernels::Device;

pub fn run(iters: i32, bandwidth_mib: usize) -> Result<()> {
    if Device::count()? == 0 {
        bail!("no CUDA device found");
    }
    let dev = Device::default_device()?;
    let info = dev.info();

    println!("{:=<72}", "");
    println!("  {}", dev);
    println!("{:=<72}", "");
    println!(
        "  compute capability   sm_{}{}",
        info.major, info.minor
    );
    println!("  SMs                  {}", info.sm_count);
    println!("  core clock           {:.2} GHz", info.clock_khz as f64 / 1e6);
    println!(
        "  memory               {:.1} GB total, {:.1} GB free, {}-bit bus",
        info.mem_total as f64 / 1e9,
        info.mem_free as f64 / 1e9,
        info.mem_bus_width
    );
    println!("  peak bandwidth       {:.0} GB/s", info.bandwidth_gbs);
    println!("  L2                   {} KB", info.l2_bytes / 1024);
    println!("  max shared / block   {} KB", info.max_smem_per_block / 1024);

    // The capability boundaries that decide which kernels can exist at all.
    println!();
    println!("  capabilities");
    let caps = [
        ("fp16 tensor cores  (sm_70+)", info.has_tensor_cores),
        ("int8/int4 IMMA     (sm_72+)", info.has_imma),
        ("bmma .xor.popc     (sm_75+)", info.has_bmma_xor),
        ("bmma .and.popc     (sm_80+)", info.has_bmma_and),
        ("cp.async           (sm_80+)", info.has_cp_async),
        ("2:4 sparsity       (sm_80+)", info.has_sparse_tc),
        ("fp8                (sm_89+)", info.has_fp8),
    ];
    for (name, ok) in caps {
        println!("    {:<30} {}", name, if ok != 0 { "yes" } else { "no" });
    }

    // --- arithmetic paths ---------------------------------------------------
    println!();
    println!("  measuring arithmetic paths ({iters} iters) ...");
    let p = dev.probe(iters)?;

    println!();
    println!("  {:<22} {:>12} {:>12}", "path", "TOPS", "vs fp16");
    println!("  {:-<48}", "");

    let base = p.fp16_wmma_tflops;
    let rows = [
        ("wmma fp16 (TFLOPS)", p.fp16_wmma_tflops),
        ("wmma int8", p.int8_wmma_tops),
        ("wmma int4", p.int4_wmma_tops),
        ("bmma b1 xor.popc", p.bin_bmma_tops),
        ("dp4a (CUDA core)", p.dp4a_tops),
        ("popc (CUDA core)", p.popc_tops),
    ];
    // `base` is -1.0 on any card without fp16 tensor cores (sm_60/sm_61, both
    // inside the fat binary's supported set). Dividing by it printed a *negative*
    // ratio next to a perfectly real measurement, which reads as a measurement.
    for (name, v) in rows {
        if v <= 0.0 {
            println!("  {name:<22} {:>12} {:>12}", "-", "unsupported");
        } else if base > 0.0 {
            println!("  {name:<22} {v:>12.1} {:>11.2}x", v / base);
        } else {
            println!("  {name:<22} {v:>12.1} {:>12}", "n/a");
        }
    }

    println!();
    println!(
        "  XNOR identity  dot = K - 2*popcount(a^b)   {}",
        if p.xnor_identity_ok == 1 { "verified on device" } else { "FAILED" }
    );

    // --- bandwidth ----------------------------------------------------------
    println!();
    println!("  measuring achieved read bandwidth ({bandwidth_mib} MiB buffer) ...");
    let measured = dev.measure_bandwidth(bandwidth_mib << 20, 30)?;
    println!(
        "  {:.0} GB/s of {:.0} GB/s peak  ({:.0}% utilisation)",
        measured,
        info.bandwidth_gbs,
        measured / info.bandwidth_gbs * 100.0
    );

    // --- what it means ------------------------------------------------------
    println!();
    println!("{:-<72}", "");
    println!("  Reading these numbers");
    println!("{:-<72}", "");
    println!(
        "  Decode at batch=1 is bandwidth bound: every weight is read once for a\n  \
         single multiply-add, about 2 FLOP/byte against a tensor-core balance\n  \
         point near 120 FLOP/byte. The TOPS column above therefore does NOT set\n  \
         decode speed -- bytes per weight does. Use `whetstone inspect` for the\n  \
         token-rate ceiling implied by a given weight format."
    );
    if p.bin_bmma_tops > 0.0 && p.popc_tops > 0.0 {
        println!();
        println!(
            "  bmma is {:.1}x a hand-written __popc loop, so binary arithmetic is\n  \
             fastest through the tensor core -- but CUDA-core popcount is itself\n  \
             {:.1}x fp16, so it is not the dead end a naive measurement suggests.",
            p.bin_bmma_tops / p.popc_tops,
            p.popc_tops / p.fp16_wmma_tflops
        );
    }
    if p.dp4a_tops > 0.0 && p.int8_wmma_tops > 0.0 {
        println!(
            "  __dp4a is {:.1}x the int8 tensor core and {:.1}x fp16: slower than\n  \
             IMMA, but a usable decode primitive where fragments are awkward.",
            p.dp4a_tops / p.int8_wmma_tops,
            p.dp4a_tops / p.fp16_wmma_tflops
        );
    }
    println!(
        "\n  Caveats: these are dependent accumulate chains, so they sit between\n  \
         latency and issue rate. The fp16 baseline accumulates in fp32, which on\n  \
         consumer Turing is half-rate versus fp16 accumulation -- every ratio\n  \
         above is therefore ~2x flattering to the alternative."
    );
    println!();

    Ok(())
}
