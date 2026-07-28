//! Compiles Whetstone's CUDA sources into a static library and links it.
//!
//! We drive `nvcc` directly rather than going through the `cc` crate. `cc`
//! infers host-compiler flags (`-ffunction-sections`, `-fno-omit-frame-pointer`,
//! profile-driven `-O0 -G`) that nvcc rejects outright, and its `.cuda(true)`
//! mode forces `--device-c` (relocatable device code) without emitting the
//! `nvcc -dlink` step that RDC then requires. Whetstone has no cross-TU device
//! calls, so plain whole-program compilation is both correct and faster.
//!
//! We compile for a single architecture on purpose. Multi-arch fat binaries
//! multiply build time, and these kernels are written against capabilities that
//! differ by architecture (`bmma.xor.popc` is sm_75+, `cp.async` is sm_80+), so
//! a "portable" build would be a fiction.
//!
//! Override with `WHETSTONE_CUDA_ARCH=86 cargo build`.

use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_ARCH: &str = "75"; // Turing. bmma.xor.popc lives here.

fn main() {
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-env-changed=WHETSTONE_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let arch = std::env::var("WHETSTONE_CUDA_ARCH").unwrap_or_else(|_| DEFAULT_ARCH.into());
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
            .arg(format!("-arch=sm_{arch}"))
            .arg("-O3")
            .arg("-std=c++17")
            .arg("--use_fast_math") // maps expf/rsqrtf to the hardware SFU paths
            .arg("-lineinfo") // ncu/nsight attribution without -G's slowdown
            .arg("-Xptxas=-v") // register/smem usage lands in the build log
            .arg("--expt-relaxed-constexpr")
            .arg("-Icuda")
            .arg(format!("-DWHETSTONE_ARCH={arch}"));

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
