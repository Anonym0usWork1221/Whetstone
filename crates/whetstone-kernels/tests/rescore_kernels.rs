//! Differential tests for the top-k head rescore.
//!
//! The failure modes are quiet ones. A rescore that picks the wrong rows still
//! produces a valid distribution; one that writes the right value to the wrong
//! index corrupts a logit nobody looks at until it wins; one whose threshold
//! collapses to "everything" is correct and slow, and one whose threshold
//! collapses to "nothing" is fast and does nothing at all. Each is pinned
//! separately.

use whetstone_kernels::rescore::HeadRescore;
use whetstone_kernels::{Device, DeviceBuffer};

fn gpu() -> bool {
    Device::default_device().is_ok()
}

fn f16v(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect()
}

/// A deterministic `[vocab, hidden]` head and a `[hidden]` activation.
fn fixture(vocab: usize, hidden: usize) -> (Vec<f32>, Vec<f32>) {
    let head: Vec<f32> = (0..vocab * hidden)
        .map(|i| ((i * 2_654_435_761usize) % 1009) as f32 / 504.0 - 1.0)
        .collect();
    let x: Vec<f32> = (0..hidden)
        .map(|i| ((i * 40_503usize) % 331) as f32 / 165.0 - 1.0)
        .collect();
    (head, x)
}

/// The exact fp16 logits, computed on the host in f64.
fn reference(head: &[f32], x: &[f32], vocab: usize, hidden: usize) -> Vec<f32> {
    (0..vocab)
        .map(|r| {
            let mut acc = 0f64;
            for i in 0..hidden {
                let w = half::f16::from_f32(head[r * hidden + i]).to_f32() as f64;
                let a = half::f16::from_f32(x[i]).to_f32() as f64;
                acc += w * a;
            }
            acc as f32
        })
        .collect()
}

/// The whole point: the rows that were rescored must match the fp16 reference,
/// and the rows that were not must be untouched.
#[test]
fn rescored_rows_match_the_fp16_reference_and_others_are_untouched() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (vocab, hidden, k) = (4096usize, 256usize, 64usize);
    let (head, x) = fixture(vocab, hidden);
    let exact = reference(&head, &x, vocab, hidden);

    // Stand in for a quantized head's output: the exact logits plus a
    // per-row perturbation big enough to matter but too small to reorder the
    // extremes, which is what a 4-bit head actually does.
    let approx: Vec<f32> = exact
        .iter()
        .enumerate()
        .map(|(r, &v)| v + ((r % 17) as f32 - 8.0) * 0.02)
        .collect();

    let mut logits = DeviceBuffer::from_slice(&approx).unwrap();
    let xd = DeviceBuffer::from_slice(&f16v(&x)).unwrap();
    let mut rs = HeadRescore::new(&f16v(&head), vocab, hidden, k).unwrap();
    rs.apply(&mut logits, &xd).unwrap();

    let got = logits.to_vec().unwrap();
    let count = rs.last_count().unwrap() as usize;
    let thresh = rs.last_threshold().unwrap();
    assert!(count >= k, "selected {count} rows, wanted at least {k}");

    let mut fixed = 0usize;
    for r in 0..vocab {
        if approx[r] >= thresh && fixed < rs.cap() {
            // May or may not have been rescored — the cap truncates arbitrarily
            // — so accept either the exact value or the original.
            let ok = (got[r] - exact[r]).abs() < 2e-2 * exact[r].abs().max(1.0)
                || (got[r] - approx[r]).abs() < 1e-6;
            assert!(ok, "row {r}: {} is neither exact {} nor original {}", got[r], exact[r], approx[r]);
            if (got[r] - approx[r]).abs() > 1e-6 {
                fixed += 1;
            }
        } else if approx[r] < thresh {
            assert!(
                (got[r] - approx[r]).abs() < 1e-6,
                "row {r} was below the threshold and must not have been touched: \
                 {} vs {}",
                got[r],
                approx[r]
            );
        }
    }
    assert!(fixed > 0, "nothing was rescored at all");
}

/// The argmax is what sampling reads. After a rescore it must agree with the
/// argmax of the exact fp16 logits — that is the entire quality claim, and a
/// rescore that fixed the wrong rows would leave it wrong while every value it
/// did write was correct.
#[test]
fn the_top_logit_agrees_with_fp16_after_rescoring() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (vocab, hidden, k) = (8192usize, 128usize, 32usize);
    let (head, x) = fixture(vocab, hidden);
    let exact = reference(&head, &x, vocab, hidden);

    // Perturb hard enough to move the argmax, then check the rescore moves it
    // back. Without this the test would pass on a no-op.
    let approx: Vec<f32> = exact
        .iter()
        .enumerate()
        .map(|(r, &v)| v + (((r * 7919) % 101) as f32 - 50.0) * 0.01)
        .collect();

    let argmax = |v: &[f32]| {
        v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| {
            if x > a.1 { (i, x) } else { a }
        }).0
    };
    let want = argmax(&exact);
    assert_ne!(argmax(&approx), want, "fixture does not exercise the fix");

    let mut logits = DeviceBuffer::from_slice(&approx).unwrap();
    let xd = DeviceBuffer::from_slice(&f16v(&x)).unwrap();
    let mut rs = HeadRescore::new(&f16v(&head), vocab, hidden, k).unwrap();
    rs.apply(&mut logits, &xd).unwrap();

    assert_eq!(argmax(&logits.to_vec().unwrap()), want, "rescore did not restore the argmax");
}

/// The threshold search has to adapt. A flat distribution and a spiked one give
/// wildly different logit spreads, and a fixed margin would select everything in
/// one case and nothing in the other.
#[test]
fn the_threshold_adapts_to_the_logit_spread() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (vocab, hidden, k) = (4096usize, 64usize, 64usize);
    let (head, x) = fixture(vocab, hidden);
    let hd = f16v(&head);
    let xd = DeviceBuffer::from_slice(&f16v(&x)).unwrap();

    for (name, spread) in [("spiked", 40.0f32), ("flat", 0.05f32)] {
        let logits: Vec<f32> = (0..vocab)
            .map(|r| -((r % 512) as f32) * spread / 512.0)
            .collect();
        let mut ld = DeviceBuffer::from_slice(&logits).unwrap();
        let mut rs = HeadRescore::new(&hd, vocab, hidden, k).unwrap();
        rs.apply(&mut ld, &xd).unwrap();

        let count = rs.last_count().unwrap() as usize;
        assert!(count >= k, "{name}: selected {count}, wanted at least {k}");
        // The whole point of the ladder is not selecting the entire vocabulary.
        assert!(count < vocab, "{name}: threshold admitted everything ({count})");
    }
}

/// Shapes that do not match are refused rather than read out of bounds.
#[test]
fn mismatched_shapes_are_refused() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (vocab, hidden) = (512usize, 64usize);
    let (head, x) = fixture(vocab, hidden);
    assert!(HeadRescore::new(&f16v(&head), vocab, hidden + 1, 8).is_err());
    assert!(HeadRescore::new(&f16v(&head), vocab, hidden, 0).is_err());

    let mut rs = HeadRescore::new(&f16v(&head), vocab, hidden, 8).unwrap();
    let xd = DeviceBuffer::from_slice(&f16v(&x)).unwrap();
    let mut wrong = DeviceBuffer::<f32>::zeros(vocab + 1).unwrap();
    assert!(rs.apply(&mut wrong, &xd).is_err());

    let mut right = DeviceBuffer::<f32>::zeros(vocab).unwrap();
    let short = DeviceBuffer::<u16>::zeros(hidden - 1).unwrap();
    assert!(rs.apply(&mut right, &short).is_err());
}
