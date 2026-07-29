//! Compiles Whetstone's CUDA sources into a static library and links it.
//!
//! We drive `nvcc` directly rather than going through the `cc` crate. `cc`
//! infers host-compiler flags (`-ffunction-sections`, `-fno-omit-frame-pointer`,
//! profile-driven `-O0 -G`) that nvcc rejects outright, and its `.cuda(true)`
//! mode forces `--device-c` (relocatable device code) without emitting the
//! `nvcc -dlink` step that RDC then requires. Whetstone has no cross-TU device
//! calls, so plain whole-program compilation is both correct and faster.
//!
//! # Fat binary, and why the arch gating had to change first
//!
//! Whetstone used to compile for exactly one architecture, on the argument that
//! the kernels are written against capabilities that differ by architecture
//! (`bmma.xor.popc` is sm_75+, `cp.async` is sm_80+) so a portable build would
//! be a fiction. That argument was wrong about its own code: **only `probe.cu`
//! ever used a tensor-core instruction.** The shipped decode and chunk kernels
//! are `half2` arithmetic, which every card since sm_53 has.
//!
//! So the build emits one SASS image per architecture plus a **PTX tail** at the
//! highest one, which the driver JITs onto anything newer than this toolkit
//! knows about. One archive covers Pascal through Hopper and forward.
//!
//! The gating moved with it (`common.cuh`): device code keys off
//! `__CUDA_ARCH__`, which nvcc redefines per compilation pass, and host code
//! asks the driver at run time. A `-D` from here can answer neither question in
//! a fat binary.
//!
//! ## `WHETSTONE_CUDA_ARCH`
//!
//! | value | meaning |
//! |---|---|
//! | unset / `all` | every architecture this toolkit supports, plus PTX. The release build. |
//! | `native` | just the installed GPU's. **Use this while iterating** — it is ~7× less device compilation. |
//! | `75` or `75,86` | exactly these |
//!
//! Build time is the real cost of `all`: each `.cu` is compiled once per
//! architecture. That is why `native` exists and why the chosen list is printed.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Architectures worth an image, newest last.
///
/// Anything this toolkit does not know is dropped rather than failing the
/// build, so a CUDA 12.0 install produces a Pascal-through-Hopper archive and a
/// CUDA 12.8 install additionally produces Blackwell — from the same source.
///
/// The floor is **sm_60**. Below that `half2` arithmetic is emulated, which
/// would make the one thing this engine is built around (packed fp16 GEMV)
/// slower than fp32, and reporting a working-but-pointless build as support
/// would be the same class of lie the architecture whitelist exists to prevent.
const WANTED_ARCHES: &[u32] = &[
    60,  // Pascal: P100
    61,  // Pascal: GTX 10xx, Titan Xp
    70,  // Volta: V100
    75,  // Turing: RTX 20xx, GTX 16xx, T4  <- the development card
    80,  // Ampere: A100
    86,  // Ampere: RTX 30xx, A10
    89,  // Ada: RTX 40xx, L4
    90,  // Hopper: H100
    100, // Blackwell datacentre: B100/B200   (CUDA 12.8+)
    120, // Blackwell consumer: RTX 50xx      (CUDA 12.8+)
];

/// Used when no GPU is present and no list was given. Turing, the card every
/// measurement in `research/` was taken on.
const FALLBACK_ARCH: u32 = 75;

fn main() {
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-env-changed=WHETSTONE_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_windows = target_os == "windows";

    let cuda_root = find_cuda_root(is_windows);
    let nvcc = cuda_root
        .as_deref()
        .map(|r| r.join(nvcc_rel(is_windows)))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("nvcc"));

    if Command::new(&nvcc).arg("--version").output().is_err() {
        panic!(
            "nvcc not found (tried {nvcc:?}).\n\
             Whetstone requires the CUDA toolkit.\n\
             \x20 Ubuntu:  sudo apt install nvidia-cuda-toolkit\n\
             \x20 Windows: install the CUDA Toolkit, then build from a\n\
             \x20          Developer Command Prompt so cl.exe is on PATH\n\
             or set CUDA_PATH to your toolkit root."
        );
    }

    let arches = select_arches(&nvcc);
    let gencode = gencode_flags(&arches);
    println!(
        "cargo:warning=whetstone: building device code for sm_{} + PTX {}",
        arches.iter().map(u32::to_string).collect::<Vec<_>>().join(", sm_"),
        arches.last().unwrap()
    );
    // Consumed by `whetstone --version` and `whetstone doctor`, so a user can
    // see which images their binary actually carries.
    println!(
        "cargo:rustc-env=WHETSTONE_CUDA_ARCH_LIST={}",
        arches.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
    );

    let sources = collect_cu(Path::new("cuda"));
    assert!(!sources.is_empty(), "no .cu sources found under cuda/");

    // Debug builds keep -O3 for device code: an unoptimised kernel is 10-50x
    // slower, which makes every measurement in a debug run meaningless.
    let mut objects = Vec::new();
    let mut children = Vec::new();

    for src in &sources {
        println!("cargo:rerun-if-changed={}", src.display());

        let stem = src.file_stem().unwrap().to_string_lossy();
        let obj = out_dir.join(format!("{stem}.o"));

        let mut cmd = Command::new(&nvcc);
        cmd.arg("-c")
            .arg(src)
            .arg("-o")
            .arg(&obj)
            .args(&gencode)
            .arg("-O3")
            .arg("-std=c++17")
            .arg("--use_fast_math") // maps expf/rsqrtf to the hardware SFU paths
            // The legacy default stream cannot be captured into a CUDA graph.
            // Per-thread makes every `<<<>>>` with no explicit stream target
            // `cudaStreamPerThread`, which can -- so the whole existing kernel
            // surface became graph-capturable without a stream parameter having
            // to be threaded through all of it. See cuda/graph.cu.
            .arg("--default-stream=per-thread")
            .arg("-lineinfo") // ncu/nsight attribution without -G's slowdown
            .arg("-Xptxas=-v") // register/smem usage lands in the build log
            .arg("--expt-relaxed-constexpr")
            .arg("-Icuda")
            // The *lowest* image in the archive. Reporting only -- device code
            // gates on __CUDA_ARCH__ and host code asks the driver.
            .arg(format!("-DWHETSTONE_ARCH={}", arches[0]));

        // Host-compiler flags are not portable: -fPIC and -O3 are GCC/Clang
        // spellings that MSVC rejects outright.
        if is_windows {
            cmd.arg("-Xcompiler=/O2");
        } else {
            cmd.arg("-Xcompiler=-fPIC").arg("-Xcompiler=-O3");
        }

        let child = cmd
            .spawn()
            .unwrap_or_else(|e| panic!("failed to spawn nvcc for {}: {e}", src.display()));
        children.push((child, src.clone()));
        objects.push(obj);
    }

    for (mut child, src) in children {
        let status = child
            .wait()
            .unwrap_or_else(|e| panic!("nvcc did not run for {}: {e}", src.display()));
        assert!(status.success(), "nvcc failed compiling {}", src.display());
    }

    // Archive with nvcc itself rather than `ar`.
    //
    // `nvcc --lib` emits a static library using whatever archiver the host
    // toolchain provides -- `ar` on Unix, `lib.exe` under MSVC -- so one code
    // path covers both. Calling `ar` directly worked on Linux and had no
    // equivalent on Windows.
    let lib_name = if is_windows { "whetstone_cuda.lib" } else { "libwhetstone_cuda.a" };
    let lib = out_dir.join(lib_name);
    let _ = std::fs::remove_file(&lib); // stale members would otherwise survive

    let status = Command::new(&nvcc)
        .arg("--lib")
        .arg("-o")
        .arg(&lib)
        .args(&objects)
        .status()
        .unwrap_or_else(|e| panic!("failed to run nvcc --lib: {e}"));
    assert!(status.success(), "nvcc --lib failed to archive CUDA objects");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=whetstone_cuda");

    for dir in cudart_search_dirs(cuda_root.as_deref()) {
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    // The CUDA runtime pulls in the host C++ runtime. MSVC links its own
    // automatically; GNU toolchains need it named.
    if !is_windows {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

/// Architectures to emit images for, ascending. Never empty.
fn select_arches(nvcc: &Path) -> Vec<u32> {
    let supported = toolkit_arches(nvcc);
    let request = std::env::var("WHETSTONE_CUDA_ARCH").unwrap_or_else(|_| "all".into());

    // A toolkit that cannot be interrogated must not silently produce a
    // single-architecture archive from the path documented as "the release
    // build". Failing loudly is the only honest option: the alternative ships
    // something whose `--version` says sm_75 and whose users on every other card
    // get a launch failure.
    if request.trim() == "all" || request.trim().is_empty() {
        assert!(
            !supported.is_empty(),
            "`nvcc --list-gpu-arch` produced nothing, so the set of architectures \
             this toolkit can target is unknown. Refusing to guess and emit a \
             single-arch archive from the release path. Set WHETSTONE_CUDA_ARCH \
             explicitly (e.g. =75, or =native) if this toolkit genuinely cannot \
             list its architectures."
        );
    }

    let mut chosen: Vec<u32> = match request.trim() {
        "all" | "" => WANTED_ARCHES.iter().copied().filter(|a| supported.contains(a)).collect(),
        // A card newer than the toolkit is a real configuration (an RTX 50xx on
        // CUDA 12.0), and the right answer there is the newest image the toolkit
        // *can* build plus the PTX tail, which the driver JITs. Asserting would
        // make the documented iteration path unusable on exactly the machines
        // that most need it.
        "native" => {
            let want = detect_installed_arch().unwrap_or(FALLBACK_ARCH);
            let pick = if supported.is_empty() || supported.contains(&want) {
                want
            } else {
                let top = supported.iter().copied().filter(|&a| a < want).max();
                let fallback = top.unwrap_or(FALLBACK_ARCH);
                println!(
                    "cargo:warning=whetstone: this GPU is sm_{want}, newer than any \
                     image this CUDA toolkit can build; using sm_{fallback} plus the \
                     PTX tail, which the driver will JIT."
                );
                fallback
            };
            vec![pick]
        }
        list => list
            .split(',')
            .filter_map(|t| t.trim().parse::<u32>().ok())
            .collect(),
    };

    chosen.sort_unstable();
    chosen.dedup();

    // An explicit request for something the toolkit cannot build has to fail
    // loudly. Silently dropping it would produce an archive missing exactly the
    // architecture the user asked for, which they would discover as a launch
    // failure on the target machine.
    // Only an *explicit list* is asserted against. "all" filters, and "native"
    // clamps with a warning; neither is a request for a specific architecture.
    let explicit = !matches!(request.trim(), "all" | "" | "native");
    if explicit && !supported.is_empty() {
        for a in &chosen {
            assert!(
                supported.contains(a),
                "WHETSTONE_CUDA_ARCH asks for sm_{a}, which this CUDA toolkit \
                 cannot target. It supports: {}",
                supported.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
            );
        }
    }

    if chosen.is_empty() {
        chosen.push(FALLBACK_ARCH);
    }
    chosen
}

/// `-gencode` for each architecture, plus a PTX tail at the highest.
///
/// The PTX is what makes the archive forward-compatible: a card newer than this
/// toolkit has no SASS image, so the driver JITs the PTX at first launch. It
/// costs one extra compilation and a few MB, and it is the difference between
/// "runs on GPUs released after this build" and "fails to launch".
fn gencode_flags(arches: &[u32]) -> Vec<String> {
    let mut out: Vec<String> = arches
        .iter()
        .map(|a| format!("-gencode=arch=compute_{a},code=sm_{a}"))
        .collect();
    let top = arches.last().copied().unwrap_or(FALLBACK_ARCH);
    out.push(format!("-gencode=arch=compute_{top},code=compute_{top}"));
    out
}

/// Architectures this toolkit can target, from `nvcc --list-gpu-arch`.
///
/// Asked rather than assumed: CUDA 12.0 stops at sm_90 and CUDA 12.8 adds
/// Blackwell, and hardcoding either would break the other.
fn toolkit_arches(nvcc: &Path) -> Vec<u32> {
    let Ok(out) = Command::new(nvcc).arg("--list-gpu-arch").output() else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().strip_prefix("compute_")?.parse::<u32>().ok())
        .collect()
}

/// The installed GPU's compute capability as `major*10 + minor`.
///
/// `nvidia-smi` rather than a CUDA API call, because a build script must not
/// need a working driver context -- and on a machine with several cards this
/// takes the first, which is what a `native` build means.
fn detect_installed_arch() -> Option<u32> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let first = text.lines().next()?.trim();
    let (major, minor) = first.split_once('.')?;
    Some(major.trim().parse::<u32>().ok()? * 10 + minor.trim().parse::<u32>().ok()?)
}

fn collect_cu(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(collect_cu(&p));
        } else if p.extension().is_some_and(|x| x == "cu") {
            out.push(p);
        }
    }
    out.sort(); // stable archive member order -> reproducible builds
    out
}

/// Path to nvcc relative to the toolkit root, per platform.
fn nvcc_rel(is_windows: bool) -> &'static str {
    if is_windows { "bin/nvcc.exe" } else { "bin/nvcc" }
}

fn find_cuda_root(is_windows: bool) -> Option<PathBuf> {
    // The CUDA installer sets CUDA_PATH on Windows, so this is the normal path
    // there and an explicit override everywhere else.
    if let Ok(p) = std::env::var("CUDA_PATH").or_else(|_| std::env::var("CUDA_HOME")) {
        return Some(PathBuf::from(p));
    }

    // Distro packages (Ubuntu's nvidia-cuda-toolkit) install nvcc to /usr/bin
    // with libraries in the multiarch dir, so locating nvcc implies root=/usr.
    let locator = if is_windows { "where" } else { "which" };
    if let Ok(out) = Command::new(locator).arg("nvcc").output() {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            // `where` can return several matches, one per line.
            if let Some(first) = text.lines().next() {
                let p = PathBuf::from(first.trim());
                if let Some(root) = p.parent().and_then(Path::parent) {
                    return Some(root.to_path_buf());
                }
            }
        }
    }

    ["/usr/local/cuda", "/opt/cuda", "/usr"]
        .iter()
        .map(PathBuf::from)
        .find(|p| p.join(nvcc_rel(is_windows)).exists())
}

fn cudart_search_dirs(root: Option<&Path>) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(r) = root {
        // Windows toolkit layout.
        dirs.push(r.join("lib/x64"));
        // Unix layouts, including Debian/Ubuntu multiarch.
        dirs.push(r.join("lib64"));
        dirs.push(r.join("lib"));
        dirs.push(r.join("lib/x86_64-linux-gnu"));
        dirs.push(r.join("targets/x86_64-linux/lib"));
    }
    dirs.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
    dirs.push(PathBuf::from("/usr/local/cuda/lib64"));
    dirs
}
