//! A zero-copy `safetensors` reader.
//!
//! Format: an 8-byte little-endian header length `n`, `n` bytes of JSON
//! metadata, then a tensor data region. Each JSON entry gives a dtype, a shape,
//! and a `[start, end)` byte range relative to the start of the data region.
//!
//! Whetstone reads checkpoints by `mmap`, so tensor bytes are never copied into
//! the process heap; the quantizer streams over them and writes its own format.
//!
//! Every offset in the header is treated as untrusted input. A checkpoint is a
//! file from the internet, and a bad range must produce an error rather than an
//! out-of-bounds read.

use std::collections::BTreeMap;
use std::path::Path;

use memmap2::Mmap;
use rayon::prelude::*;
use serde::Deserialize;

use crate::error::{Error, Result};

/// Element types Whetstone can read from a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// IEEE-754 binary16.
    F16,
    /// bfloat16: an f32 with the low 16 mantissa bits removed.
    BF16,
    /// IEEE-754 binary32.
    F32,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Boolean, one byte per element.
    Bool,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "F32" => Self::F32,
            "I8" => Self::I8,
            "U8" => Self::U8,
            "I32" => Self::I32,
            "I64" => Self::I64,
            "BOOL" => Self::Bool,
            other => return Err(Error::Format(format!("unsupported safetensors dtype {other:?}"))),
        })
    }

    /// Bytes occupied by one element.
    pub fn size(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::I64 => 8,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// A tensor located inside a memory-mapped checkpoint.
#[derive(Debug, Clone)]
pub struct TensorView {
    /// Tensor name, e.g. `model.layers.0.mlp.gate_proj.weight`.
    pub name: String,
    /// Element type as stored.
    pub dtype: Dtype,
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// Byte range within the file (absolute, not data-region relative).
    range: (usize, usize),
}

impl TensorView {
    /// Total element count.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Stored size in bytes.
    pub fn nbytes(&self) -> usize {
        self.range.1 - self.range.0
    }

    /// Shape as `(rows, cols)`, for the 2-D weight matrices.
    pub fn shape_2d(&self) -> Result<(usize, usize)> {
        match self.shape.as_slice() {
            [r, c] => Ok((*r, *c)),
            other => Err(Error::Shape(format!(
                "{}: expected a 2-D tensor, found shape {other:?}",
                self.name
            ))),
        }
    }
}

/// A memory-mapped safetensors checkpoint.
pub struct SafeTensors {
    mmap: Mmap,
    tensors: BTreeMap<String, TensorView>,
    metadata: BTreeMap<String, String>,
}

impl std::fmt::Debug for SafeTensors {
    /// Summarises rather than dumping: the mapping is typically ~1 GB, and
    /// every tensor descriptor would bury whatever the caller was debugging.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafeTensors")
            .field("tensors", &self.tensors.len())
            .field("data_bytes", &self.data_bytes())
            .field("mapped_bytes", &self.mmap.len())
            .finish()
    }
}

impl SafeTensors {
    /// Maps a checkpoint and parses its header.
    ///
    /// The whole header is validated up front: every declared byte range must
    /// lie inside the file, match the shape times the element size, and not
    /// overlap another tensor.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| Error::Io(format!("could not open {}: {e}", path.display())))?;

        // SAFETY: the mapping is read-only and lives as long as `Self`. The
        // usual mmap caveat applies -- external truncation while mapped is UB --
        // which we accept for checkpoint files, as every loader does.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| Error::Io(format!("could not mmap {}: {e}", path.display())))?;

        Self::from_mmap(mmap, &path.display().to_string())
    }

    fn from_mmap(mmap: Mmap, origin: &str) -> Result<Self> {
        if mmap.len() < 8 {
            return Err(Error::Format(format!("{origin}: file is too short to be safetensors")));
        }

        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;

        // A corrupt length must not become a huge slice index.
        let data_start = 8usize
            .checked_add(header_len)
            .ok_or_else(|| Error::Format(format!("{origin}: header length overflows")))?;
        if data_start > mmap.len() {
            return Err(Error::Format(format!(
                "{origin}: header claims {header_len} bytes but the file is only {} bytes",
                mmap.len()
            )));
        }

        let raw: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&mmap[8..data_start])
                .map_err(|e| Error::Format(format!("{origin}: malformed header JSON: {e}")))?;

        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        let mut spans: Vec<(usize, usize, String)> = Vec::new();
        let data_len = mmap.len() - data_start;

        for (name, value) in raw {
            if name == "__metadata__" {
                if let Ok(m) = serde_json::from_value::<BTreeMap<String, String>>(value) {
                    metadata = m;
                }
                continue;
            }

            let entry: RawEntry = serde_json::from_value(value)
                .map_err(|e| Error::Format(format!("{origin}: bad entry for {name:?}: {e}")))?;

            let dtype = Dtype::parse(&entry.dtype)?;
            let [begin, end] = entry.data_offsets;

            if end < begin {
                return Err(Error::Format(format!(
                    "{origin}: {name} has an inverted range [{begin}, {end})"
                )));
            }
            if end > data_len {
                return Err(Error::Format(format!(
                    "{origin}: {name} ends at {end} but the data region is only {data_len} bytes \
                     (file is truncated or corrupt)"
                )));
            }

            let numel: usize = entry.shape.iter().copied().try_fold(1usize, |a, b| {
                a.checked_mul(b)
            }).ok_or_else(|| Error::Format(format!("{origin}: {name} shape overflows")))?;

            let expected = numel * dtype.size();
            if end - begin != expected {
                return Err(Error::Format(format!(
                    "{origin}: {name} declares shape {:?} ({expected} bytes) but its range spans {} bytes",
                    entry.shape,
                    end - begin
                )));
            }

            spans.push((begin, end, name.clone()));
            tensors.insert(
                name.clone(),
                TensorView {
                    name,
                    dtype,
                    shape: entry.shape,
                    range: (data_start + begin, data_start + end),
                },
            );
        }

        // Overlapping tensors would mean one silently corrupts another.
        spans.sort_unstable();
        for w in spans.windows(2) {
            if w[0].1 > w[1].0 {
                return Err(Error::Format(format!(
                    "{origin}: tensors {} and {} overlap",
                    w[0].2, w[1].2
                )));
            }
        }

        Ok(Self { mmap, tensors, metadata })
    }

    /// Number of tensors in the checkpoint.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// True when the checkpoint contains no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Tensor names, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// All tensor descriptors.
    pub fn iter(&self) -> impl Iterator<Item = &TensorView> {
        self.tensors.values()
    }

    /// Header `__metadata__` entries.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Looks up a tensor descriptor.
    pub fn get(&self, name: &str) -> Result<&TensorView> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))
    }

    /// Raw bytes of a tensor, borrowed from the mapping.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.get(name)?;
        Ok(&self.mmap[t.range.0..t.range.1])
    }

    /// Reads a tensor as `f32`, converting from whatever it was stored as.
    ///
    /// This allocates. It is for the quantizer and for tests, not the hot path.
    ///
    /// Parallel over 64 Ki-element strips. Widening is trivially parallel and it
    /// is not free at scale: a 7 B checkpoint is 15 GB of bf16 in, 30 GB of f32
    /// out, and as a serial `push` loop that was minutes of the conversion. The
    /// strips also give the page-fault path more queue depth, which matters more
    /// than the arithmetic when the checkpoint is on a spinning disk.
    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.get(name)?;
        let raw = self.bytes(name)?;
        let n = t.numel();

        // `I64`/`Bool` are rejected before anything is allocated, so an
        // unreadable dtype does not cost a multi-gigabyte zeroed buffer first.
        if matches!(t.dtype, Dtype::I64 | Dtype::Bool) {
            return Err(Error::Format(format!(
                "{name}: {:?} cannot be read as f32",
                t.dtype
            )));
        }

        let width = t.dtype.size();
        let mut out = vec![0f32; n];
        const STRIP: usize = 1 << 16;

        out.par_chunks_mut(STRIP)
            .zip(raw.par_chunks(STRIP * width))
            .for_each(|(dst, src)| match t.dtype {
                Dtype::F32 => {
                    for (d, c) in dst.iter_mut().zip(src.chunks_exact(4)) {
                        *d = f32::from_le_bytes(c.try_into().unwrap());
                    }
                }
                Dtype::F16 => {
                    for (d, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                        *d = half::f16::from_le_bytes(c.try_into().unwrap()).to_f32();
                    }
                }
                Dtype::BF16 => {
                    // bf16 is the top 16 bits of an f32, so widening is a shift.
                    for (d, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                        let bits = u16::from_le_bytes(c.try_into().unwrap());
                        *d = f32::from_bits((bits as u32) << 16);
                    }
                }
                Dtype::I8 => {
                    for (d, &b) in dst.iter_mut().zip(src) {
                        *d = b as i8 as f32;
                    }
                }
                Dtype::U8 => {
                    for (d, &b) in dst.iter_mut().zip(src) {
                        *d = b as f32;
                    }
                }
                Dtype::I32 => {
                    for (d, c) in dst.iter_mut().zip(src.chunks_exact(4)) {
                        *d = i32::from_le_bytes(c.try_into().unwrap()) as f32;
                    }
                }
                Dtype::I64 | Dtype::Bool => unreachable!("rejected above"),
            });

        Ok(out)
    }

    /// Total bytes of tensor data.
    pub fn data_bytes(&self) -> usize {
        self.tensors.values().map(TensorView::nbytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Builds a valid safetensors file in a temp path, then lets the caller
    /// corrupt the header to check that validation actually fires.
    fn write_file(header: &str, data: &[u8], name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("whetstone_test_{name}.safetensors"));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(data).unwrap();
        path
    }

    #[test]
    fn reads_tensors_and_converts_dtypes() {
        // f32 [2,2] then bf16 [2]
        let mut data = Vec::new();
        for v in [1.0f32, -2.0, 3.5, 0.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        // bf16 of 1.0 is 0x3F80, of -0.5 is 0xBF00
        data.extend_from_slice(&0x3F80u16.to_le_bytes());
        data.extend_from_slice(&0xBF00u16.to_le_bytes());

        let header = r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},
                         "b":{"dtype":"BF16","shape":[2],"data_offsets":[16,20]}}"#;
        let path = write_file(header, &data, "ok");

        let st = SafeTensors::open(&path).unwrap();
        assert_eq!(st.len(), 2);
        assert_eq!(st.get("a").unwrap().shape_2d().unwrap(), (2, 2));
        assert_eq!(st.to_f32("a").unwrap(), vec![1.0, -2.0, 3.5, 0.0]);
        assert_eq!(st.to_f32("b").unwrap(), vec![1.0, -0.5]);
        assert_eq!(st.data_bytes(), 20);
        assert!(matches!(st.get("nope"), Err(Error::MissingTensor(_))));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bf16_widening_is_a_bit_shift() {
        // The conversion must be exact, not an approximation: bf16 is literally
        // the high half of an f32.
        for bits in [0x3F80u16, 0xBF00, 0x0000, 0x7F80, 0x4049] {
            let via_shift = f32::from_bits((bits as u32) << 16);
            let via_half = half::bf16::from_bits(bits).to_f32();
            assert_eq!(via_shift.to_bits(), via_half.to_bits(), "bf16 {bits:#06x}");
        }
    }

    #[test]
    fn truncated_file_is_rejected() {
        // Declares 32 bytes of data but supplies 8. This is exactly the failure
        // mode of an interrupted download, so it must be caught at open().
        let header = r#"{"a":{"dtype":"F32","shape":[2,4],"data_offsets":[0,32]}}"#;
        let path = write_file(header, &[0u8; 8], "truncated");
        let err = SafeTensors::open(&path).unwrap_err();
        assert!(
            format!("{err}").contains("truncated or corrupt"),
            "unexpected error: {err}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn shape_and_range_disagreement_is_rejected() {
        // shape [2,2] of F32 needs 16 bytes, but the range spans 8.
        let header = r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,8]}}"#;
        let path = write_file(header, &[0u8; 8], "shapemismatch");
        assert!(SafeTensors::open(&path).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn overlapping_tensors_are_rejected() {
        let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
                         "b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#;
        let path = write_file(header, &[0u8; 12], "overlap");
        let err = SafeTensors::open(&path).unwrap_err();
        assert!(format!("{err}").contains("overlap"), "unexpected error: {err}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn absurd_header_length_is_rejected_not_panicked() {
        let path = std::env::temp_dir().join("whetstone_test_badlen.safetensors");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&u64::MAX.to_le_bytes()).unwrap();
        f.write_all(b"{}").unwrap();
        drop(f);
        assert!(SafeTensors::open(&path).is_err());
        let _ = std::fs::remove_file(path);
    }
}
