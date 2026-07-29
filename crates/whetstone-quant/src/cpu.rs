//! Runtime CPU feature selection for the host-side hot loops.
//!
//! # Why this is not a compiler flag
//!
//! Rust's default `x86-64` target is the 2003 baseline: SSE2, and **no SSE4.1**.
//! `f32::round` has no SSE2 instruction, so on that baseline every `.round()` in
//! the quantizer compiles to a `roundf` **libm call**. The k-quant fit does 21
//! candidate grids over 32 weights and rounds each one, which is 672 library
//! calls per group of 32 — 7.4 billion of them over a 0.5 B checkpoint.
//!
//! Measured, converting Qwen2.5-0.5B on an i7-9750H, twelve threads, output
//! byte-identical in every case:
//!
//! | build | wall |
//! |---|---|
//! | baseline `x86-64`, single-threaded | 83.4 s |
//! | baseline `x86-64`, 12 threads | 17.0 s |
//! | `-C target-cpu=native`, 12 threads | **10.2 s** |
//!
//! So the instruction set is worth 1.67× on top of the threading — but
//! `target-cpu=native` bakes the build machine's ISA into the binary, and a
//! release archive that segfaults with SIGILL on a Sandy Bridge is not a
//! portable release archive. The fix is to compile the hot loop **several
//! times** and pick at run time.
//!
//! # What is dispatched, and what is not
//!
//! Only the per-row packers. They are the entire cost of a conversion (13 s of
//! the 17 s above; the whole read-widen-narrow-write path around them is 2.3 s).
//! Everything else stays on the portable baseline, where it belongs.
//!
//! # Other architectures
//!
//! `aarch64` has NEON in its baseline, so `round` is already `frintn` and there
//! is nothing to select between. Every non-x86_64 target takes the generic path.

/// Host CPU features the packers care about, resolved once.
///
/// Ordered by capability, so `>=` is a meaningful test.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Isa {
    /// The compilation target's own baseline. On x86-64 that is SSE2, where
    /// rounding is a libm call.
    Baseline,
    /// SSE4.1: `roundps` exists, so rounding is one instruction.
    Sse41,
    /// AVX2 + FMA + F16C: eight-wide `vroundps`, fused multiply-add, and
    /// single-instruction `f16 <-> f32`. Every AVX2 part shipped with all three,
    /// but they are detected separately because the ISA does not guarantee it.
    Avx2,
}

impl Isa {
    /// Short name for the banner, so a slow conversion on an old machine is
    /// self-explanatory rather than mysterious.
    pub fn name(self) -> &'static str {
        match self {
            Self::Baseline => {
                if cfg!(target_arch = "x86_64") {
                    "sse2 (baseline)"
                } else {
                    "generic"
                }
            }
            Self::Sse41 => "sse4.1",
            Self::Avx2 => "avx2+fma+f16c",
        }
    }
}

/// The best instruction set this host supports.
///
/// `is_x86_feature_detected!` caches its answer in a process-wide bitset, so
/// this is an atomic load and a bit test — cheap enough to call per row, which
/// is where the dispatch sits.
#[inline]
pub fn detect() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            return Isa::Avx2;
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return Isa::Sse41;
        }
    }
    Isa::Baseline
}

/// Compiles one `#[inline(always)]` body at several feature levels and emits a
/// wrapper that picks between them at run time.
///
/// A macro rather than three hand-written copies per packer: copies drift, and a
/// copy that drifts from its siblings is a correctness bug visible only on one
/// class of machine — which is the hardest kind to ever see.
///
/// The generated names are spelled out by the caller because concatenating
/// identifiers inside a macro is still unstable.
///
/// # Safety
///
/// Each generated function carries `#[target_feature]`, so calling it requires
/// those features. The wrapper is its only caller and invokes each one strictly
/// under the matching [`detect`] arm.
#[macro_export]
macro_rules! isa_dispatch {
    (
        body   = $body_fn:ident,
        avx2   = $avx2_fn:ident,
        sse41  = $sse41_fn:ident;
        $(#[$meta:meta])*
        $vis:vis fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) -> $ret:ty;
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma,f16c")]
        unsafe fn $avx2_fn ( $($arg : $ty),* ) -> $ret {
            $body_fn($($arg),*)
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse4.1")]
        unsafe fn $sse41_fn ( $($arg : $ty),* ) -> $ret {
            $body_fn($($arg),*)
        }

        $(#[$meta])*
        #[inline]
        $vis fn $name ( $($arg : $ty),* ) -> $ret {
            #[cfg(target_arch = "x86_64")]
            {
                match $crate::cpu::detect() {
                    // SAFETY: `detect()` just reported avx2 and fma present.
                    $crate::cpu::Isa::Avx2 => unsafe { $avx2_fn($($arg),*) },
                    // SAFETY: `detect()` just reported sse4.1 present.
                    $crate::cpu::Isa::Sse41 => unsafe { $sse41_fn($($arg),*) },
                    $crate::cpu::Isa::Baseline => $body_fn($($arg),*),
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                $body_fn($($arg),*)
            }
        }
    };
}

// --- f16 conversion -------------------------------------------------------
//
// Not part of the quantizer, but on the same critical path and with the same
// problem: `f16 <-> f32` is one instruction (`vcvtph2ps` / `vcvtps2ph`) on any
// CPU since Ivy Bridge, and a ~20-instruction bit-twiddling sequence without
// F16C. A conversion widens 15 GB of checkpoint on the way in and narrows every
// dense tensor on the way out, so it is worth the same treatment as the packer.

#[inline(always)]
fn widen_f16_body(src: &[u16], dst: &mut [f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = half::f16::from_bits(s).to_f32();
    }
}

#[inline(always)]
fn narrow_f16_body(src: &[f32], dst: &mut [u16]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = half::f16::from_f32(s).to_bits();
    }
}

crate::isa_dispatch! {
    body  = widen_f16_body,
    avx2  = widen_f16_avx2,
    sse41 = widen_f16_sse41;
    /// Widens IEEE binary16 bit patterns to `f32`, over whichever instruction
    /// set the host has. Extra elements in the longer slice are ignored.
    pub fn widen_f16(src: &[u16], dst: &mut [f32]) -> ();
}

crate::isa_dispatch! {
    body  = narrow_f16_body,
    avx2  = narrow_f16_avx2,
    sse41 = narrow_f16_sse41;
    /// Narrows `f32` to IEEE binary16 bit patterns, round-to-nearest-even.
    pub fn narrow_f16(src: &[f32], dst: &mut [u16]) -> ();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dispatch must not change arithmetic. Every f16 bit pattern round-trips
    /// exactly, and every one of the 65 536 of them is cheap to check — which is
    /// the only way to be sure an AVX2 path and a libm path agree on the
    /// subnormals and the NaN payloads rather than just on the easy values.
    #[test]
    fn f16_conversion_matches_the_scalar_reference() {
        let src: Vec<u16> = (0..=u16::MAX).collect();
        let mut wide = vec![0f32; src.len()];
        widen_f16(&src, &mut wide);
        for (i, (&bits, &w)) in src.iter().zip(&wide).enumerate() {
            let want = half::f16::from_bits(bits).to_f32();
            assert!(
                w == want || (w.is_nan() && want.is_nan()),
                "widen({bits:#06x}) = {w}, want {want} (index {i})"
            );
        }

        // Narrowing is only exact for values f16 can hold, so round-trip the
        // widened set rather than inventing f32s.
        let mut back = vec![0u16; wide.len()];
        narrow_f16(&wide, &mut back);
        for (&bits, &got) in src.iter().zip(&back) {
            let want = half::f16::from_f32(half::f16::from_bits(bits).to_f32()).to_bits();
            assert_eq!(got, want, "narrow round-trip of {bits:#06x}");
        }
    }

    /// The dispatch must not be able to pick a level the host cannot execute --
    /// the failure mode is SIGILL, which no test above this one would survive.
    #[test]
    fn detection_is_consistent_with_std() {
        let isa = detect();
        #[cfg(target_arch = "x86_64")]
        {
            if isa >= Isa::Sse41 {
                assert!(std::arch::is_x86_feature_detected!("sse4.1"));
            }
            if isa >= Isa::Avx2 {
                assert!(std::arch::is_x86_feature_detected!("avx2"));
                assert!(std::arch::is_x86_feature_detected!("fma"));
                assert!(std::arch::is_x86_feature_detected!("f16c"));
            }
        }
        assert!(!isa.name().is_empty());
    }
}
