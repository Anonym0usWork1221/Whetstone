//! Differential tests for the decode-step kernels.
//!
//! Each of these pins a kernel against a closed form or a CPU reference, chosen
//! so that the *specific* way the kernel could be subtly wrong is what fails.
//! Attention over a single cached position must return `v` exactly; RoPE must
//! use HuggingFace's half-rotation layout rather than adjacent even/odd pairs;
//! the argmax must order negative logits correctly and break ties the same way
//! every run. Every one of those errors produces fluent text and a plausible
//! perplexity.
//!
//! They live outside `src/` so the module stays under the project's file-length
//! limit, and they use only the public API, which is a useful constraint in its
//! own right.

use whetstone_kernels::decode::*;
use whetstone_kernels::{DeviceBuffer, Device};


fn gpu() -> bool {
    Device::default_device().is_ok()
}

fn f16v(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect()
}

fn from_f16(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect()
}

#[test]
fn rmsnorm_matches_a_cpu_reference() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let n = 896;
    let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 211) as f32 / 211.0 - 0.5) * 3.0).collect();
    let w: Vec<f32> = (0..n).map(|i| 0.5 + (i % 17) as f32 / 34.0).collect();
    let eps = 1e-6f32;

    let xd = DeviceBuffer::from_slice(&x).unwrap();
    let wd = DeviceBuffer::from_slice(&f16v(&w)).unwrap();
    let mut od = DeviceBuffer::<u16>::zeros(n).unwrap();
    rmsnorm(&xd, &wd, &mut od, eps).unwrap();
    let got = from_f16(&od.to_vec().unwrap());

    let ms: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
    let inv = 1.0 / (ms + eps as f64).sqrt();
    for i in 0..n {
        let want = (x[i] as f64 * inv * half::f16::from_f32(w[i]).to_f32() as f64) as f32;
        assert!(
            (got[i] - want).abs() < 2e-3 * want.abs().max(1.0),
            "element {i}: {} vs {want}",
            got[i]
        );
    }
}

#[test]
fn swiglu_matches_a_cpu_reference() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let n = 4864;
    let g: Vec<f32> = (0..n).map(|i| (i % 97) as f32 / 24.0 - 2.0).collect();
    let u: Vec<f32> = (0..n).map(|i| (i % 53) as f32 / 26.0 - 1.0).collect();

    let mut gu = g.clone();
    gu.extend_from_slice(&u);
    let gud = DeviceBuffer::from_slice(&gu).unwrap();
    let mut od = DeviceBuffer::<u16>::zeros(n).unwrap();
    swiglu(&gud, &mut od).unwrap();
    let got = from_f16(&od.to_vec().unwrap());

    for i in 0..n {
        let want = g[i] / (1.0 + (-g[i]).exp()) * u[i];
        assert!(
            (got[i] - want).abs() < 3e-3 * want.abs().max(1.0),
            "element {i}: {} vs {want}",
            got[i]
        );
    }
}

/// The whole binary-free path rests on this being the *half rotation*
/// layout HuggingFace uses, not adjacent even/odd pairs. A wrong layout
/// still produces fluent text, so only a direct comparison catches it.
#[test]
fn rope_uses_the_half_rotation_layout() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (n_q, n_kv, hd, max_seq, pos) = (2usize, 1usize, 64usize, 8usize, 3usize);
    let theta = 1_000_000.0f64;

    let q: Vec<f32> = (0..n_q * hd).map(|i| (i % 31) as f32 / 15.0 - 1.0).collect();
    let k: Vec<f32> = (0..n_kv * hd).map(|i| (i % 23) as f32 / 11.0 - 1.0).collect();
    let v: Vec<f32> = (0..n_kv * hd).map(|i| (i % 19) as f32 / 9.0 - 1.0).collect();

    let table = RopeTable::new(max_seq, hd, theta).unwrap();
    let mut cache = KvCache::new(n_kv, n_q, hd, max_seq).unwrap();
    let mut qkv = q.clone();
    qkv.extend_from_slice(&k);
    qkv.extend_from_slice(&v);
    let mut qd = DeviceBuffer::from_slice(&qkv).unwrap();
    let cursor = DeviceCursor::new(pos as i32).unwrap();

    rope_cache(&mut qd, &mut cache, &table, n_q, &cursor).unwrap();
    let got = qd.to_vec().unwrap();

    let half = hd / 2;
    for h in 0..n_q {
        for j in 0..half {
            let inv = theta.powf(-(j as f64) / half as f64);
            let (s, c) = (pos as f64 * inv).sin_cos();
            let (x1, x2) = (q[h * hd + j] as f64, q[h * hd + j + half] as f64);
            let w1 = (x1 * c - x2 * s) as f32;
            let w2 = (x2 * c + x1 * s) as f32;
            assert!((got[h * hd + j] - w1).abs() < 1e-5, "head {h} lo {j}");
            assert!((got[h * hd + j + half] - w2).abs() < 1e-5, "head {h} hi {j}");
        }
    }
}

/// Attention at batch=1 with a one-entry cache must return `v` exactly:
/// softmax over a single score is 1 whatever the score is. It is the one
/// case with a closed form, so it isolates the online-softmax recurrence
/// from the dot products.
#[test]
fn attention_over_one_position_returns_that_value() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (n_q, n_kv, hd, max_seq) = (14usize, 2usize, 64usize, 32usize);
    let table = RopeTable::new(max_seq, hd, 1e6).unwrap();
    let mut cache = KvCache::new(n_kv, n_q, hd, max_seq).unwrap();

    let q: Vec<f32> = (0..n_q * hd).map(|i| (i % 29) as f32 / 14.0 - 1.0).collect();
    let k: Vec<f32> = (0..n_kv * hd).map(|i| (i % 13) as f32 / 6.0 - 1.0).collect();
    let v: Vec<f32> = (0..n_kv * hd).map(|i| (i % 17) as f32 / 8.0 - 1.0).collect();

    let mut qkv = q.clone();
    qkv.extend_from_slice(&k);
    qkv.extend_from_slice(&v);
    let mut qd = DeviceBuffer::from_slice(&qkv).unwrap();
    let cursor = DeviceCursor::new(0).unwrap();
    rope_cache(&mut qd, &mut cache, &table, n_q, &cursor).unwrap();

    let mut out = DeviceBuffer::<u16>::zeros(n_q * hd).unwrap();
    attn_decode(&qd, &mut cache, &mut out, n_q, &cursor).unwrap();
    let got = from_f16(&out.to_vec().unwrap());

    for h in 0..n_q {
        let kv = h / (n_q / n_kv);
        for d in 0..hd {
            let want = half::f16::from_f32(v[kv * hd + d]).to_f32();
            assert!(
                (got[h * hd + d] - want).abs() < 2e-3,
                "head {h} dim {d}: {} vs {want}",
                got[h * hd + d]
            );
        }
    }
}

/// Two positions have a closed form too, and it exercises the rescaling
/// branch of the online softmax that the single-position case cannot.
#[test]
fn attention_over_two_positions_matches_an_explicit_softmax() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (n_q, n_kv, hd, max_seq) = (2usize, 1usize, 64usize, 16usize);
    let theta = 1_000_000.0f64;
    let mut cache = KvCache::new(n_kv, n_q, hd, max_seq).unwrap();
    let table = RopeTable::new(max_seq, hd, theta).unwrap();

    // The cache holds *rotated* keys, so the reference has to rotate too --
    // `theta^0 == 1` means the first pair of every head turns by a full
    // radian per position, however large theta is. An earlier version of
    // this test assumed a huge theta made the rotation vanish and was wrong
    // by 2% for exactly that reason.
    let half = hd / 2;
    let rotate = |x: &[f32], p: usize| -> Vec<f32> {
        let mut out = x.to_vec();
        for j in 0..half {
            let inv = theta.powf(-(j as f64) / half as f64);
            let (s, c) = (p as f64 * inv).sin_cos();
            let (x1, x2) = (x[j] as f64, x[j + half] as f64);
            out[j] = (x1 * c - x2 * s) as f32;
            out[j + half] = (x2 * c + x1 * s) as f32;
        }
        out
    };

    let mut ks = Vec::new();
    let mut vs = Vec::new();
    for p in 0..2usize {
        let k: Vec<f32> = (0..hd).map(|i| ((i + 7 * p) % 11) as f32 / 5.0 - 1.0).collect();
        let v: Vec<f32> = (0..hd).map(|i| ((i + 3 * p) % 7) as f32 / 3.0 - 1.0).collect();
        let mut qkv = vec![0f32; n_q * hd];
        qkv.extend_from_slice(&k);
        qkv.extend_from_slice(&v);
        let mut qd = DeviceBuffer::from_slice(&qkv).unwrap();
        let cursor = DeviceCursor::new(p as i32).unwrap();
        rope_cache(&mut qd, &mut cache, &table, n_q, &cursor).unwrap();
        ks.push(rotate(&k, p));
        vs.push(v);
    }

    let q: Vec<f32> = (0..n_q * hd).map(|i| (i % 5) as f32 / 2.0 - 1.0).collect();
    let qd = DeviceBuffer::from_slice(&q).unwrap();
    let mut out = DeviceBuffer::<u16>::zeros(n_q * hd).unwrap();
    let at = DeviceCursor::new(1).unwrap();  // pos 1 -> two valid entries
    attn_decode(&qd, &mut cache, &mut out, n_q, &at).unwrap();
    let got = from_f16(&out.to_vec().unwrap());

    let scale = 1.0f64 / (hd as f64).sqrt();
    for h in 0..n_q {
        let s: Vec<f64> = (0..2)
            .map(|p| {
                (0..hd)
                    .map(|d| {
                        q[h * hd + d] as f64 * half::f16::from_f32(ks[p][d]).to_f32() as f64
                    })
                    .sum::<f64>()
                    * scale
            })
            .collect();
        let m = s[0].max(s[1]);
        let e: Vec<f64> = s.iter().map(|&x| (x - m).exp()).collect();
        let z = e[0] + e[1];
        for d in 0..hd {
            let want = (e[0] * half::f16::from_f32(vs[0][d]).to_f32() as f64
                + e[1] * half::f16::from_f32(vs[1][d]).to_f32() as f64)
                / z;
            assert!(
                (got[h * hd + d] as f64 - want).abs() < 3e-3,
                "head {h} dim {d}: {} vs {want}",
                got[h * hd + d]
            );
        }
    }
}

#[test]
fn argmax_is_exact_and_breaks_ties_toward_the_lower_index() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let n = 151_936usize;
    let mut v = vec![-1.0f32; n];
    v[98_765] = 7.5;
    let d = DeviceBuffer::from_slice(&v).unwrap();
    let mut idx = DeviceBuffer::<i32>::zeros(1).unwrap();
    argmax(&d, &mut idx).unwrap();
    assert_eq!(idx.to_vec().unwrap()[0], 98_765);

    // A tie must resolve the same way every run, or a "deterministic"
    // generation stops being one.
    v[12_345] = 7.5;
    let d = DeviceBuffer::from_slice(&v).unwrap();
    argmax(&d, &mut idx).unwrap();
    assert_eq!(idx.to_vec().unwrap()[0], 12_345);
}

#[test]
fn nll_matches_a_host_log_softmax() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let n = 151_936usize;
    let l: Vec<f32> = (0..n).map(|i| ((i * 7919 % 2003) as f32) / 200.0 - 5.0).collect();
    let d = DeviceBuffer::from_slice(&l).unwrap();
    let mut acc = DeviceBuffer::<f32>::zeros(2).unwrap();

    let targets = [0u32, 1, 12_345, 151_935];
    for &t in &targets {
        nll(&d, t, &mut acc).unwrap();
    }
    let got = acc.to_vec().unwrap();
    assert_eq!(got[1] as usize, targets.len());

    let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = m + l.iter().map(|&x| ((x as f64) - m).exp()).sum::<f64>().ln();
    let want: f64 = targets.iter().map(|&t| lse - l[t as usize] as f64).sum();
    assert!(
        (got[0] as f64 - want).abs() < 1e-3 * want.abs(),
        "device nll {} vs host {want}",
        got[0]
    );
}

#[test]
fn negative_logits_are_ordered_correctly() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    // Sign-magnitude vs two's complement: a naive bit cast puts -0.1 above
    // -100.0 in the wrong direction. All-negative logits are the normal
    // case right after a cold start, so this is not a corner.
    let v: Vec<f32> = (0..4096).map(|i| -1.0 - (i as f32) * 0.01).collect();
    let d = DeviceBuffer::from_slice(&v).unwrap();
    let mut idx = DeviceBuffer::<i32>::zeros(1).unwrap();
    argmax(&d, &mut idx).unwrap();
    assert_eq!(idx.to_vec().unwrap()[0], 0);
}
