//! Captures build provenance so a released binary can identify itself.
//!
//! A bug report that says "whetstone 0.1.0" is much less useful than one that
//! says which commit, which CUDA architecture, and which toolchain produced the
//! binary — especially for a project that compiles for exactly one GPU family.
//!
//! Everything here degrades gracefully: a build from a source tarball with no
//! git metadata still succeeds, it just reports `unknown`.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=WHETSTONE_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=WHETSTONE_GIT_SHA");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");

    // NOTE: the architecture list is NOT read here. `WHETSTONE_CUDA_ARCH` is a
    // *request* ("all", "native", "75,86"), not the list of images that got
    // built, and a `cargo:rustc-env` emitted by the kernels build script is not
    // visible to `env!` in this crate anyway. `whetstone-kernels` re-exports the
    // resolved list as a `pub const`, which is what `--version` prints.

    // CI passes the sha explicitly; a local build reads it from git; a tarball
    // has neither and says so.
    let sha = std::env::var("WHETSTONE_GIT_SHA").ok().or_else(git_sha).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=WHETSTONE_GIT_SHA={sha}");

    // SOURCE_DATE_EPOCH is the reproducible-builds convention; honouring it
    // means a release rebuilt from the same source produces the same string.
    let date = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .map(format_epoch)
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=WHETSTONE_BUILD_DATE={date}");

    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".into());
    println!("cargo:rustc-env=WHETSTONE_TARGET={target}");
}

fn git_sha() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if sha.is_empty() {
        return None;
    }

    // Mark a dirty tree so a binary built over uncommitted edits cannot be
    // mistaken for the tagged release.
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    Some(if dirty { format!("{sha}-dirty") } else { sha })
}

/// Formats a Unix timestamp as `YYYY-MM-DD`, without pulling in a date crate.
fn format_epoch(secs: i64) -> String {
    let days = secs.div_euclid(86_400);

    // Civil-from-days (Howard Hinnant's algorithm), shifted to a March-based
    // year so leap days land at the end and the month arithmetic stays branchless.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02}")
}
