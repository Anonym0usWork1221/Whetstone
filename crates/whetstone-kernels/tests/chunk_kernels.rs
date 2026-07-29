//! Differential tests for the multi-token path.
//!
//! Every one of these pins a chunk kernel against the **single-token kernel it
//! replaces**, on identical inputs. That reference is the right one: the batch-1
//! kernels are already validated against closed forms and CPU references in
//! `decode_kernels.rs`, and every quality number the project has ever recorded
//! was produced by them. If a chunk pass and `n` sequential passes disagree, the
//! chunk pass is wrong by definition.
//!
//! The failure this is really guarding against is silent. A multi-token
//! attention that gets the causal bound off by one, or a GEMM whose token stride
//! is wrong, produces fluent text and a plausible perplexity — the same class of
//! bug that the architecture whitelist and the untied-head fix were both about.

use whetstone_kernels::{chunk, decode, gemv, DeviceBuffer, Device, QuantLinearHier};

fn gpu() -> bool {
    Device::default_device().is_ok()
}

fn f16v(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect()
}

fn from_f16(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect()
}

/// Deterministic pseudo-random floats in [-1, 1). A fixed LCG rather than a
/// crate, so a failure is reproducible without a seed to record.
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn close(got: f32, want: f32, tol: f32, what: &str) {
    let scale = want.abs().max(1.0);
    assert!(
        (got - want).abs() <= tol * scale,
        "{what}: got {got}, want {want} (tol {})",
        tol * scale
    );
}

#[test]
fn gemm_fp16_matches_n_separate_gemvs() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (in_f, out_f, n) = (896usize, 320usize, 7usize);

    let w = noise(in_f * out_f, 11);
    let x = noise(in_f * n, 22);
    let bias = noise(out_f, 33);

    let wd = DeviceBuffer::from_slice(&f16v(&w)).unwrap();
    let xd = DeviceBuffer::from_slice(&f16v(&x)).unwrap();
    let bd = DeviceBuffer::from_slice(&f16v(&bias)).unwrap();

    let mut chunked = DeviceBuffer::<f32>::zeros(n * out_f).unwrap();
    chunk::gemm_fp16(&wd, &xd, Some(&bd), &mut chunked, in_f, out_f, n, false).unwrap();
    let got = chunked.to_vec().unwrap();

    for j in 0..n {
        let xj = DeviceBuffer::from_slice(&f16v(&x[j * in_f..(j + 1) * in_f])).unwrap();
        let mut yj = DeviceBuffer::<f32>::zeros(out_f).unwrap();
        gemv::gemv_fp16_ex(&wd, &xj, Some(&bd), &mut yj, in_f, out_f, false).unwrap();
        let want = yj.to_vec().unwrap();
        for r in 0..out_f {
            close(got[j * out_f + r], want[r], 2e-3, &format!("token {j} row {r}"));
        }
    }
}

#[test]
fn gemm_fp16_accumulates_the_same_way() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (in_f, out_f, n) = (256usize, 96usize, 5usize);
    let w = noise(in_f * out_f, 44);
    let x = noise(in_f * n, 55);
    let seed_y = noise(n * out_f, 66);

    let wd = DeviceBuffer::from_slice(&f16v(&w)).unwrap();
    let xd = DeviceBuffer::from_slice(&f16v(&x)).unwrap();

    let mut chunked = DeviceBuffer::from_slice(&seed_y).unwrap();
    chunk::gemm_fp16(&wd, &xd, None, &mut chunked, in_f, out_f, n, true).unwrap();
    let got = chunked.to_vec().unwrap();

    for j in 0..n {
        let xj = DeviceBuffer::from_slice(&f16v(&x[j * in_f..(j + 1) * in_f])).unwrap();
        let mut yj = DeviceBuffer::from_slice(&seed_y[j * out_f..(j + 1) * out_f]).unwrap();
        gemv::gemv_fp16_ex(&wd, &xj, None, &mut yj, in_f, out_f, true).unwrap();
        let want = yj.to_vec().unwrap();
        for r in 0..out_f {
            close(got[j * out_f + r], want[r], 2e-3, &format!("accum token {j} row {r}"));
        }
    }
}

/// The format the engine actually ships. `in_features` deliberately spans more
/// than one 16-token slice's worth of rows so the host-side chunking loop in
/// `wst_gemm_int4_hier` is exercised too.
#[test]
fn gemm_int4_hier_matches_n_separate_gemvs() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (in_f, out_f, n) = (896usize, 256usize, 19usize); // 19 > CHUNK_NMAX

    // Packed layout: 8 nibbles per u32, one (ls|lm<<4) byte per group of 32, one
    // packed (d, dmin) half2 per row.
    let mut s = 0x1234_5678u32;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let qw: Vec<u32> = (0..out_f * in_f / 8).map(|_| rng()).collect();
    let si: Vec<u8> = (0..out_f * in_f / 32)
        .map(|_| {
            let v = rng();
            // ls must be >= 1: a zero scale index is representable but degenerate,
            // and the quantizer clamps it for exactly that reason.
            let ls = 1 + (v & 0x7) as u8;
            let lm = ((v >> 8) & 0xF) as u8;
            ls | (lm << 4)
        })
        .collect();
    let sb: Vec<u32> = (0..out_f)
        .map(|i| {
            let d = half::f16::from_f32(0.002 + (i % 7) as f32 * 1e-4).to_bits() as u32;
            let dm = half::f16::from_f32(0.01 + (i % 5) as f32 * 1e-3).to_bits() as u32;
            d | (dm << 16)
        })
        .collect();

    let q = QuantLinearHier::from_packed(&qw, &si, &sb, in_f, out_f).unwrap();

    let x = noise(in_f * n, 77);
    let xd = DeviceBuffer::from_slice(&f16v(&x)).unwrap();

    let mut chunked = DeviceBuffer::<f32>::zeros(n * out_f).unwrap();
    q.gemm_ex(&xd, None, &mut chunked, n, false).unwrap();
    let got = chunked.to_vec().unwrap();

    for j in 0..n {
        let xj = DeviceBuffer::from_slice(&f16v(&x[j * in_f..(j + 1) * in_f])).unwrap();
        let mut yj = DeviceBuffer::<f32>::zeros(out_f).unwrap();
        q.gemv_ex(&xj, None, &mut yj, false).unwrap();
        let want = yj.to_vec().unwrap();
        for r in 0..out_f {
            close(got[j * out_f + r], want[r], 3e-3, &format!("hier token {j} row {r}"));
        }
    }
}

#[test]
fn rmsnorm_chunk_matches_single() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (dim, n) = (896usize, 6usize);
    let x = noise(dim * n, 88);
    let w = noise(dim, 99);
    let eps = 1e-6f32;

    let xd = DeviceBuffer::from_slice(&x).unwrap();
    let wd = DeviceBuffer::from_slice(&f16v(&w)).unwrap();
    let mut od = DeviceBuffer::<u16>::zeros(n * dim).unwrap();
    chunk::rmsnorm_eps(&xd, &wd, &mut od, dim, n, eps).unwrap();
    let got = from_f16(&od.to_vec().unwrap());

    for j in 0..n {
        let xj = DeviceBuffer::from_slice(&x[j * dim..(j + 1) * dim]).unwrap();
        let mut oj = DeviceBuffer::<u16>::zeros(dim).unwrap();
        decode::rmsnorm(&xj, &wd, &mut oj, eps).unwrap();
        let want = from_f16(&oj.to_vec().unwrap());
        for i in 0..dim {
            assert_eq!(
                got[j * dim + i].to_bits(),
                want[i].to_bits(),
                "rmsnorm token {j} element {i}"
            );
        }
    }
}

#[test]
fn swiglu_chunk_matches_single() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (inter, n) = (4864usize, 5usize);
    let gate_up = noise(2 * inter * n, 101);

    let gd = DeviceBuffer::from_slice(&gate_up).unwrap();
    let mut od = DeviceBuffer::<u16>::zeros(n * inter).unwrap();
    chunk::swiglu(&gd, &mut od, inter, n).unwrap();
    let got = od.to_vec().unwrap();

    for j in 0..n {
        let row = &gate_up[j * 2 * inter..(j + 1) * 2 * inter];
        let rd = DeviceBuffer::from_slice(row).unwrap();
        let mut oj = DeviceBuffer::<u16>::zeros(inter).unwrap();
        decode::swiglu(&rd, &mut oj).unwrap();
        let want = oj.to_vec().unwrap();
        for i in 0..inter {
            assert_eq!(got[j * inter + i], want[i], "swiglu token {j} element {i}");
        }
    }
}

#[test]
fn argmax_chunk_matches_single() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (vocab, n) = (151936usize, 8usize);
    let mut logits = noise(vocab * n, 202);
    // Plant an unambiguous winner per row, plus an exact tie just below it so
    // the lower-index rule is actually exercised.
    for j in 0..n {
        logits[j * vocab + (j * 977 + 13) % vocab] = 9.0;
        logits[j * vocab + 5] = 8.5;
        logits[j * vocab + 6] = 8.5;
    }

    let ld = DeviceBuffer::from_slice(&logits).unwrap();
    let mut picks = DeviceBuffer::<i32>::zeros(n).unwrap();
    chunk::argmax(&ld, &mut picks, vocab, n).unwrap();
    let got = picks.to_vec().unwrap();

    for j in 0..n {
        let row = DeviceBuffer::from_slice(&logits[j * vocab..(j + 1) * vocab]).unwrap();
        let mut one = DeviceBuffer::<i32>::zeros(1).unwrap();
        decode::argmax(&row, &mut one).unwrap();
        assert_eq!(got[j], one.to_vec().unwrap()[0], "argmax row {j}");
    }
}

/// The one that matters most: rotary embedding, the cache append and causal
/// attention over a chunk, against `n` sequential single-token passes.
///
/// An off-by-one in the causal bound — query `j` seeing position `pos0+j+1`, or
/// missing its own — is invisible in generated text and shifts perplexity by an
/// amount that looks like quantization damage.
#[test]
fn rope_and_attention_chunk_match_sequential_decode() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (n_q, n_kv, hd, max_seq, n) = (14usize, 2usize, 64usize, 128usize, 9usize);
    let stride = (n_q + 2 * n_kv) * hd;

    let rope = decode::RopeTable::new(max_seq, hd, 1_000_000.0).unwrap();
    let qkv_all = noise(stride * n, 303);

    // Chunk path.
    let mut cache_a = decode::KvCache::new(n_kv, n_q, hd, max_seq).unwrap();
    let mut qkv_c = DeviceBuffer::from_slice(&qkv_all).unwrap();
    chunk::rope_cache(&mut qkv_c, &mut cache_a, &rope, n_q, 0, n, None).unwrap();
    let mut out_c = DeviceBuffer::<u16>::zeros(n * n_q * hd).unwrap();
    chunk::attn(&qkv_c, &cache_a, &mut out_c, n_q, 0, n).unwrap();
    let got = from_f16(&out_c.to_vec().unwrap());

    // Sequential path, one token at a time, into its own cache.
    let mut cache_b = decode::KvCache::new(n_kv, n_q, hd, max_seq).unwrap();
    let cursor = decode::DeviceCursor::new(0).unwrap();
    for j in 0..n {
        cursor.set(j as i32).unwrap();
        let mut qkv_1 =
            DeviceBuffer::from_slice(&qkv_all[j * stride..(j + 1) * stride]).unwrap();
        decode::rope_cache(&mut qkv_1, &mut cache_b, &rope, n_q, &cursor, None).unwrap();
        let mut out_1 = DeviceBuffer::<u16>::zeros(n_q * hd).unwrap();
        decode::attn_decode(&qkv_1, &mut cache_b, &mut out_1, n_q, &cursor).unwrap();
        let want = from_f16(&out_1.to_vec().unwrap());

        for i in 0..n_q * hd {
            close(
                got[j * n_q * hd + i],
                want[i],
                4e-3,
                &format!("attention token {j} element {i}"),
            );
        }
    }
}

/// Chunk attention starting part way into a populated cache — the speculative
/// case, where `pos0` is nonzero and the prefix was written by earlier passes.
#[test]
fn attention_chunk_resumes_mid_cache() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (n_q, n_kv, hd, max_seq) = (14usize, 2usize, 64usize, 128usize);
    let stride = (n_q + 2 * n_kv) * hd;
    let (prefix, n) = (20usize, 5usize);

    let rope = decode::RopeTable::new(max_seq, hd, 1_000_000.0).unwrap();
    let qkv_all = noise(stride * (prefix + n), 404);

    let mut cache_a = decode::KvCache::new(n_kv, n_q, hd, max_seq).unwrap();
    let mut cache_b = decode::KvCache::new(n_kv, n_q, hd, max_seq).unwrap();

    // Same prefix into both caches, through the single-token path.
    let cursor = decode::DeviceCursor::new(0).unwrap();
    for j in 0..prefix {
        cursor.set(j as i32).unwrap();
        for cache in [&mut cache_a, &mut cache_b] {
            let mut q1 =
                DeviceBuffer::from_slice(&qkv_all[j * stride..(j + 1) * stride]).unwrap();
            decode::rope_cache(&mut q1, cache, &rope, n_q, &cursor, None).unwrap();
        }
    }

    // Chunk continues cache_a from `prefix`.
    let tail = &qkv_all[prefix * stride..(prefix + n) * stride];
    let mut qkv_c = DeviceBuffer::from_slice(tail).unwrap();
    chunk::rope_cache(&mut qkv_c, &mut cache_a, &rope, n_q, prefix, n, None).unwrap();
    let mut out_c = DeviceBuffer::<u16>::zeros(n * n_q * hd).unwrap();
    chunk::attn(&qkv_c, &cache_a, &mut out_c, n_q, prefix, n).unwrap();
    let got = from_f16(&out_c.to_vec().unwrap());

    // Sequential continues cache_b the old way.
    for j in 0..n {
        let p = prefix + j;
        cursor.set(p as i32).unwrap();
        let mut q1 = DeviceBuffer::from_slice(&tail[j * stride..(j + 1) * stride]).unwrap();
        decode::rope_cache(&mut q1, &mut cache_b, &rope, n_q, &cursor, None).unwrap();
        let mut o1 = DeviceBuffer::<u16>::zeros(n_q * hd).unwrap();
        decode::attn_decode(&q1, &mut cache_b, &mut o1, n_q, &cursor).unwrap();
        let want = from_f16(&o1.to_vec().unwrap());
        for i in 0..n_q * hd {
            close(
                got[j * n_q * hd + i],
                want[i],
                4e-3,
                &format!("resumed attention token {j} element {i}"),
            );
        }
    }
}
