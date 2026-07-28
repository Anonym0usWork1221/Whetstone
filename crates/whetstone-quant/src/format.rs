//! The `.wstone` container format.
//!
//! # Why not just use safetensors or GGUF
//!
//! Both store *tensors*. Whetstone needs to store a **pre-arranged execution
//! plan**: weights already packed into the exact bit layout its kernels index,
//! with the scale metadata interleaved the way the GEMV walks it, aligned so
//! every load is coalesced, and with the model's own configuration embedded so
//! the runtime needs no sidecar files.
//!
//! A `.wstone` file is not a checkpoint you convert at load time. The
//! conversion has already happened; loading is an `mmap` and a pointer walk.
//!
//! # Layout
//!
//! ```text
//! 0   magic        8   "WHETSTON"
//! 8   version      4   u32 LE
//! 12  flags        4   u32 LE
//! 16  header_len   8   u64 LE
//! 24  header_hash  8   u64 LE   FNV-1a of the header bytes
//! 32  header       header_len   UTF-8 JSON
//!     padding                   to ALIGN
//!     payloads                  each blob ALIGN-aligned
//! ```
//!
//! Payload alignment is 256 bytes: wide enough for any vector load the kernels
//! issue (the GEMV reads `uint4`, 16 B) and for the 128-byte cache line, with
//! room for future direct-I/O paths that require sector alignment.
//!
//! Every blob carries an FNV-1a checksum. A truncated download or a corrupt
//! byte must be an error at load time, not a garbage generation ten minutes
//! later — that failure mode is expensive to diagnose and easy to prevent.

use std::collections::BTreeMap;
use std::io::{Seek, SeekFrom, Write};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// File magic. Eight bytes so the header stays 8-byte aligned.
pub const MAGIC: &[u8; 8] = b"WHETSTON";

/// Current format version.
pub const VERSION: u32 = 1;

/// Payload alignment in bytes.
pub const ALIGN: u64 = 256;

/// Fixed prefix length before the JSON header.
const PREFIX: u64 = 32;

/// How a tensor's bytes are encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorKind {
    /// IEEE-754 binary16, row-major, dense.
    Fp16,
    /// IEEE-754 binary32, row-major, dense.
    Fp32,
    /// int4 asymmetric, groups of 128 along the input dimension.
    ///
    /// Two blobs: `qw` (`u32`, eight nibbles each) and `sz` (`u32`, an fp16
    /// scale in the low half and an fp16 zero in the high half).
    Int4G128,
    /// int4 with hierarchical scales, groups of 32 along the input dimension.
    ///
    /// Three blobs: `qw` (`u32`, eight nibbles each), `si` (`u8`, a 4-bit scale
    /// index low and a 4-bit min index high) and `sb` (`u32`, an fp16 `d` low
    /// and an fp16 `dmin` high, one per row).
    ///
    /// `scale = d*ls`, `min = -dmin*lm`, `w = q*scale + min`. Group 32 at
    /// `4 + 8/32 + 32/in_features` bits against `Int4G128`'s flat 4.25 — see
    /// `hier.rs` for the perplexity table that justifies the swap.
    Int4HierG32,
}

impl TensorKind {
    /// Bits per weight including scale metadata, for the roofline.
    pub fn bits_per_weight(self) -> f64 {
        match self {
            Self::Fp16 => 16.0,
            Self::Fp32 => 32.0,
            Self::Int4G128 => 4.0 + 32.0 / 128.0,
            // Depends on the row length, which this method does not have. The
            // roofline never uses it -- `decode_resident_bytes` sums the blobs
            // that were actually written -- so this is only a display default.
            Self::Int4HierG32 => 4.0 + 8.0 / 32.0 + 32.0 / 896.0,
        }
    }
}

/// A byte range within the payload region.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Blob {
    /// Absolute file offset.
    pub offset: u64,
    /// Length in bytes.
    pub len: u64,
    /// FNV-1a hash of the blob's bytes.
    pub hash: u64,
}

/// One tensor's directory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    /// Tensor name, matching the source checkpoint.
    pub name: String,
    /// Encoding.
    pub kind: TensorKind,
    /// Logical shape, outermost first.
    pub shape: Vec<usize>,
    /// Named byte ranges. `Int4G128` uses `qw` and `sz`; dense kinds use `data`.
    pub blobs: BTreeMap<String, Blob>,
}

impl TensorEntry {
    /// Element count.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total stored bytes.
    pub fn stored_bytes(&self) -> u64 {
        self.blobs.values().map(|b| b.len).sum()
    }

    /// Looks up a blob by name.
    pub fn blob(&self, name: &str) -> Result<&Blob> {
        self.blobs
            .get(name)
            .ok_or_else(|| Error::Format(format!("{}: no blob named {name:?}", self.name)))
    }
}

/// The JSON header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    /// Always `"wstone"`.
    pub format: String,
    /// Format version.
    pub version: u32,
    /// Producer identification, e.g. `"whetstone-quant 0.1.0"`.
    pub producer: String,
    /// The source model's `config.json`, verbatim, so no sidecar is needed.
    pub model_config: serde_json::Value,
    /// Quantization scheme description, for provenance.
    pub quant: BTreeMap<String, String>,
    /// Tensor directory, sorted by name.
    pub tensors: Vec<TensorEntry>,
    /// Non-tensor payloads, by name. Currently `tokenizer.json`.
    ///
    /// `#[serde(default)]` on both sides of the change: a reader built before
    /// this field existed ignores it, and a reader built after it accepts a file
    /// written without it. That is why adding this did not need a format version
    /// bump — the rule is that `format::VERSION` moves when a reader would
    /// *misinterpret* an old file, not merely when the header grows.
    #[serde(default)]
    pub extras: BTreeMap<String, Blob>,
}

impl Header {
    /// Finds a tensor entry.
    pub fn tensor(&self, name: &str) -> Result<&TensorEntry> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Error::Format(format!("no tensor named {name:?}")))
    }

    /// Total payload bytes across all tensors.
    pub fn payload_bytes(&self) -> u64 {
        self.tensors.iter().map(TensorEntry::stored_bytes).sum()
    }

    /// Weight bytes read on a decode step: everything except the input
    /// embedding.
    ///
    /// With tied embeddings the output projection reuses the embedding matrix,
    /// so the matrix is stored once and *is* read in full every token. It is
    /// counted here for exactly that reason — omitting it understates decode
    /// traffic by over a quarter on Qwen2.5-0.5B.
    pub fn decode_resident_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .filter(|t| t.name != "model.embed_tokens.weight" || self.tied())
            .map(TensorEntry::stored_bytes)
            .sum()
    }

    fn tied(&self) -> bool {
        self.model_config
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    }
}

/// FNV-1a-shaped, 64-bit. **The multiplier is not the FNV-1a prime.**
///
/// `0x1000_0000_01b3` is one hex digit longer than the real prime,
/// `0x100000001b3`. That was a typo, and it survived because the only thing that
/// ever checked this hash was the implementation that produced it — a
/// reimplementation of the container in Python is what surfaced it.
///
/// **Do not correct it.** The threat model is a truncated download or a bad
/// disk, not an adversary, and any odd multiplier detects that equally well;
/// changing the constant would invalidate every `.wstone` already written in
/// exchange for nothing. It is documented here so that the next person to spot
/// it does not have to rediscover why it stays.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

fn align_up(n: u64, a: u64) -> u64 {
    n.div_ceil(a) * a
}

// ------------------------------------------------------------------ writer

/// Streams a `.wstone` file to disk.
///
/// Tensors are written one at a time and the directory is patched in at the
/// end, so converting a model never needs the whole model in RAM.
pub struct Writer<W: Write + Seek> {
    out: W,
    cursor: u64,
    entries: Vec<TensorEntry>,
    extras: BTreeMap<String, Blob>,
    model_config: serde_json::Value,
    quant: BTreeMap<String, String>,
    header_reserve: u64,
}

impl<W: Write + Seek> Writer<W> {
    /// Begins a file. `header_reserve` bytes are set aside for the directory,
    /// which is written last.
    pub fn new(mut out: W, model_config: serde_json::Value, header_reserve: u64) -> Result<Self> {
        let start = align_up(PREFIX + header_reserve, ALIGN);
        out.seek(SeekFrom::Start(start))
            .map_err(|e| Error::Format(format!("seek failed: {e}")))?;

        Ok(Self {
            out,
            cursor: start,
            entries: Vec::new(),
            extras: BTreeMap::new(),
            model_config,
            quant: BTreeMap::new(),
            header_reserve,
        })
    }

    /// Records a provenance key/value in the header.
    pub fn set_quant_meta(&mut self, key: &str, value: &str) {
        self.quant.insert(key.into(), value.into());
    }

    fn write_blob(&mut self, bytes: &[u8]) -> Result<Blob> {
        let pad = align_up(self.cursor, ALIGN) - self.cursor;
        if pad > 0 {
            self.out
                .write_all(&vec![0u8; pad as usize])
                .map_err(|e| Error::Format(format!("write failed: {e}")))?;
            self.cursor += pad;
        }

        let offset = self.cursor;
        self.out
            .write_all(bytes)
            .map_err(|e| Error::Format(format!("write failed: {e}")))?;
        self.cursor += bytes.len() as u64;

        Ok(Blob { offset, len: bytes.len() as u64, hash: fnv1a(bytes) })
    }

    /// Appends an int4 group-128 tensor.
    pub fn write_int4(&mut self, name: &str, packed: &crate::PackedInt4) -> Result<()> {
        let qw_bytes: &[u8] = bytemuck_cast(&packed.qw);
        let sz_bytes: &[u8] = bytemuck_cast(&packed.sz);

        let qw = self.write_blob(qw_bytes)?;
        let sz = self.write_blob(sz_bytes)?;

        let mut blobs = BTreeMap::new();
        blobs.insert("qw".into(), qw);
        blobs.insert("sz".into(), sz);

        self.entries.push(TensorEntry {
            name: name.into(),
            kind: TensorKind::Int4G128,
            shape: vec![packed.out_features, packed.in_features],
            blobs,
        });
        Ok(())
    }

    /// Appends an int4 hierarchical-scale tensor.
    pub fn write_int4_hier(&mut self, name: &str, packed: &crate::PackedInt4Hier) -> Result<()> {
        let qw = self.write_blob(bytemuck_cast(&packed.qw))?;
        let si = self.write_blob(&packed.si)?;
        let sb = self.write_blob(bytemuck_cast(&packed.sb))?;

        let mut blobs = BTreeMap::new();
        blobs.insert("qw".into(), qw);
        blobs.insert("si".into(), si);
        blobs.insert("sb".into(), sb);

        self.entries.push(TensorEntry {
            name: name.into(),
            kind: TensorKind::Int4HierG32,
            shape: vec![packed.out_features, packed.in_features],
            blobs,
        });
        Ok(())
    }

    /// Appends a dense fp16 tensor. `data` holds raw f16 bit patterns.
    pub fn write_fp16(&mut self, name: &str, data: &[u16], shape: &[usize]) -> Result<()> {
        let expect: usize = shape.iter().product();
        if data.len() != expect {
            return Err(Error::Shape(format!(
                "{name}: {} elements for shape {shape:?} (expected {expect})",
                data.len()
            )));
        }
        let blob = self.write_blob(bytemuck_cast_u16(data))?;

        let mut blobs = BTreeMap::new();
        blobs.insert("data".into(), blob);

        self.entries.push(TensorEntry {
            name: name.into(),
            kind: TensorKind::Fp16,
            shape: shape.to_vec(),
            blobs,
        });
        Ok(())
    }

    /// Appends a non-tensor payload, such as the source `tokenizer.json`.
    ///
    /// The format's premise is that a `.wstone` needs no sidecar files. A model
    /// that cannot turn text into ids without one is only three-quarters
    /// self-contained, and the tokenizer is 7 MB against a 263 MB file.
    pub fn write_extra(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let blob = self.write_blob(bytes)?;
        self.extras.insert(name.into(), blob);
        Ok(())
    }

    /// Writes the header and finishes the file.
    pub fn finish(mut self) -> Result<Header> {
        self.entries.sort_by(|a, b| a.name.cmp(&b.name));

        let header = Header {
            format: "wstone".into(),
            version: VERSION,
            producer: concat!("whetstone-quant ", env!("CARGO_PKG_VERSION")).into(),
            model_config: self.model_config.clone(),
            quant: self.quant.clone(),
            tensors: self.entries.clone(),
            extras: self.extras.clone(),
        };

        let json = serde_json::to_vec(&header)
            .map_err(|e| Error::Format(format!("could not serialise header: {e}")))?;

        if json.len() as u64 > self.header_reserve {
            return Err(Error::Format(format!(
                "header is {} bytes but only {} were reserved; pass a larger \
                 header_reserve (tensor payloads are already written, so the \
                 directory cannot be grown in place)",
                json.len(),
                self.header_reserve
            )));
        }

        self.out
            .seek(SeekFrom::Start(0))
            .map_err(|e| Error::Format(format!("seek failed: {e}")))?;

        let mut prefix = Vec::with_capacity(PREFIX as usize);
        prefix.extend_from_slice(MAGIC);
        prefix.extend_from_slice(&VERSION.to_le_bytes());
        prefix.extend_from_slice(&0u32.to_le_bytes()); // flags
        prefix.extend_from_slice(&(json.len() as u64).to_le_bytes());
        prefix.extend_from_slice(&fnv1a(&json).to_le_bytes());
        debug_assert_eq!(prefix.len() as u64, PREFIX);

        self.out
            .write_all(&prefix)
            .and_then(|_| self.out.write_all(&json))
            .and_then(|_| self.out.flush())
            .map_err(|e| Error::Format(format!("write failed: {e}")))?;

        Ok(header)
    }
}

// ------------------------------------------------------------------ reader

/// Parses and validates a `.wstone` header from the file's leading bytes.
///
/// Everything here treats the input as hostile: a `.wstone` may arrive over the
/// network, and a bad offset must produce an error rather than an out-of-bounds
/// read.
pub fn read_header(bytes: &[u8], file_len: u64) -> Result<Header> {
    if bytes.len() < PREFIX as usize {
        return Err(Error::Format("file is too short to be a .wstone".into()));
    }
    if &bytes[0..8] != MAGIC {
        return Err(Error::Format(format!(
            "bad magic {:?}; not a .wstone file",
            String::from_utf8_lossy(&bytes[0..8])
        )));
    }

    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(Error::Format(format!(
            "file is format version {version}, this build understands {VERSION}"
        )));
    }

    let header_len = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let want_hash = u64::from_le_bytes(bytes[24..32].try_into().unwrap());

    let end = PREFIX
        .checked_add(header_len)
        .ok_or_else(|| Error::Format("header length overflows".into()))?;
    if end > file_len || end as usize > bytes.len() {
        return Err(Error::Format(format!(
            "header claims {header_len} bytes but the file holds {file_len}"
        )));
    }

    let json = &bytes[PREFIX as usize..end as usize];
    if fnv1a(json) != want_hash {
        return Err(Error::Format("header checksum mismatch; file is corrupt".into()));
    }

    let header: Header = serde_json::from_slice(json)
        .map_err(|e| Error::Format(format!("malformed header JSON: {e}")))?;

    // Validate every declared range before anyone indexes with it.
    for (name, b) in &header.extras {
        let last = b
            .offset
            .checked_add(b.len)
            .ok_or_else(|| Error::Format(format!("extra {name:?}: range overflows")))?;
        if last > file_len {
            return Err(Error::Format(format!(
                "extra {name:?} ends at {last} but the file is {file_len} bytes"
            )));
        }
    }
    for t in &header.tensors {
        for (blob_name, b) in &t.blobs {
            let last = b
                .offset
                .checked_add(b.len)
                .ok_or_else(|| Error::Format(format!("{}/{blob_name}: range overflows", t.name)))?;
            if last > file_len {
                return Err(Error::Format(format!(
                    "{}/{blob_name} ends at {last} but the file is {file_len} bytes \
                     (truncated or corrupt)",
                    t.name
                )));
            }
            if b.offset % ALIGN != 0 {
                return Err(Error::Format(format!(
                    "{}/{blob_name} is at {} which is not {ALIGN}-byte aligned",
                    t.name, b.offset
                )));
            }
        }
        let expect_blobs: &[&str] = match t.kind {
            TensorKind::Int4G128 => &["qw", "sz"],
            TensorKind::Int4HierG32 => &["qw", "si", "sb"],
            TensorKind::Fp16 | TensorKind::Fp32 => &["data"],
        };
        for want in expect_blobs {
            if !t.blobs.contains_key(*want) {
                return Err(Error::Format(format!(
                    "{}: kind {:?} requires a {want:?} blob",
                    t.name, t.kind
                )));
            }
        }
    }

    Ok(header)
}

/// Verifies every blob's checksum against the file contents.
///
/// Linear in file size, so it is opt-in rather than automatic on load.
pub fn verify_payloads(header: &Header, file: &[u8]) -> Result<()> {
    for (name, b) in &header.extras {
        let lo = b.offset as usize;
        let hi = lo + b.len as usize;
        if hi > file.len() {
            return Err(Error::Format(format!("extra {name:?}: range past end of file")));
        }
        if fnv1a(&file[lo..hi]) != b.hash {
            return Err(Error::Format(format!("extra {name:?}: checksum mismatch")));
        }
    }
    for t in &header.tensors {
        for (name, b) in &t.blobs {
            let lo = b.offset as usize;
            let hi = lo + b.len as usize;
            if hi > file.len() {
                return Err(Error::Format(format!("{}/{name}: range past end of file", t.name)));
            }
            if fnv1a(&file[lo..hi]) != b.hash {
                return Err(Error::Format(format!(
                    "{}/{name}: checksum mismatch; the file is corrupt",
                    t.name
                )));
            }
        }
    }
    Ok(())
}

// Local reinterpretation helpers. A dependency for this would be overkill, and
// both casts are from a Pod type to bytes, which is always sound.
fn bytemuck_cast(v: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding or invalid bit patterns, and the resulting
    // slice covers exactly the same bytes with a stricter-to-looser alignment
    // change (4 -> 1), which is always valid.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytemuck_cast_u16(v: &[u16]) -> &[u8] {
    // SAFETY: as above, u16 -> u8.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_config() -> serde_json::Value {
        serde_json::json!({
            "hidden_size": 896, "num_hidden_layers": 2, "vocab_size": 1024,
            "tie_word_embeddings": true
        })
    }

    #[test]
    fn round_trips_through_a_file() {
        let w: Vec<f32> = (0..128 * 8).map(|i| (i % 37) as f32 / 37.0 - 0.5).collect();
        let packed = crate::quantize_int4_g128(&w, 128, 8).unwrap();

        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = Writer::new(&mut buf, sample_config(), 64 * 1024).unwrap();
            writer.set_quant_meta("scheme", "int4-g128");
            writer.write_int4("gate", &packed).unwrap();
            writer.write_fp16("norm", &[0x3C00u16; 64], &[64]).unwrap();
            writer.finish().unwrap();
        }
        let bytes = buf.into_inner();

        let h = read_header(&bytes, bytes.len() as u64).unwrap();
        assert_eq!(h.format, "wstone");
        assert_eq!(h.version, VERSION);
        assert_eq!(h.tensors.len(), 2);
        assert_eq!(h.quant.get("scheme").map(String::as_str), Some("int4-g128"));

        let gate = h.tensor("gate").unwrap();
        assert_eq!(gate.kind, TensorKind::Int4G128);
        assert_eq!(gate.shape, vec![8, 128]);
        assert_eq!(gate.numel(), 1024);

        verify_payloads(&h, &bytes).unwrap();

        // The packed bytes must survive the round trip exactly.
        let qw = gate.blob("qw").unwrap();
        let stored = &bytes[qw.offset as usize..(qw.offset + qw.len) as usize];
        assert_eq!(stored, bytemuck_cast(&packed.qw));
    }

    #[test]
    fn extras_round_trip_and_old_files_still_load() {
        let w = vec![0.1f32; 128];
        let packed = crate::quantize_int4_g128(&w, 128, 1).unwrap();
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = Writer::new(&mut buf, sample_config(), 8192).unwrap();
            writer.write_int4("a", &packed).unwrap();
            writer.write_extra("tokenizer.json", br#"{"model":{}}"#).unwrap();
            writer.finish().unwrap();
        }
        let bytes = buf.into_inner();
        let h = read_header(&bytes, bytes.len() as u64).unwrap();
        verify_payloads(&h, &bytes).unwrap();

        let e = h.extras.get("tokenizer.json").expect("extra missing");
        assert_eq!(
            &bytes[e.offset as usize..(e.offset + e.len) as usize],
            br#"{"model":{}}"#
        );

        // A header written before `extras` existed must still parse: the field
        // is `#[serde(default)]` precisely so this needs no version bump.
        let json = serde_json::json!({
            "format": "wstone", "version": VERSION, "producer": "test",
            "model_config": sample_config(), "quant": {}, "tensors": []
        });
        let old: Header = serde_json::from_value(json).unwrap();
        assert!(old.extras.is_empty());
    }

    #[test]
    fn payloads_are_aligned_for_vector_loads() {
        let w = vec![0.1f32; 128 * 8];
        let packed = crate::quantize_int4_g128(&w, 128, 8).unwrap();
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = Writer::new(&mut buf, sample_config(), 8192).unwrap();
            writer.write_int4("a", &packed).unwrap();
            writer.write_int4("b", &packed).unwrap();
            writer.finish().unwrap();
        }
        let bytes = buf.into_inner();
        let h = read_header(&bytes, bytes.len() as u64).unwrap();
        for t in &h.tensors {
            for (name, b) in &t.blobs {
                assert_eq!(b.offset % ALIGN, 0, "{}/{name} misaligned at {}", t.name, b.offset);
            }
        }
    }

    #[test]
    fn rejects_foreign_and_corrupt_files() {
        // Not a .wstone at all.
        let mut junk = vec![0u8; 512];
        junk[0..8].copy_from_slice(b"GGUF\0\0\0\0");
        assert!(read_header(&junk, junk.len() as u64).is_err());

        // Too short.
        assert!(read_header(&[0u8; 8], 8).is_err());

        // Valid file with a flipped header byte.
        let w = vec![0.1f32; 128];
        let packed = crate::quantize_int4_g128(&w, 128, 1).unwrap();
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = Writer::new(&mut buf, sample_config(), 8192).unwrap();
            writer.write_int4("a", &packed).unwrap();
            writer.finish().unwrap();
        }
        let mut bytes = buf.into_inner();
        bytes[PREFIX as usize + 5] ^= 0xFF;
        let err = read_header(&bytes, bytes.len() as u64).unwrap_err();
        assert!(format!("{err}").contains("checksum"), "unexpected error: {err}");
    }

    #[test]
    fn truncation_is_caught_at_header_parse() {
        let w = vec![0.1f32; 128 * 8];
        let packed = crate::quantize_int4_g128(&w, 128, 8).unwrap();
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = Writer::new(&mut buf, sample_config(), 8192).unwrap();
            writer.write_int4("a", &packed).unwrap();
            writer.finish().unwrap();
        }
        let bytes = buf.into_inner();

        // Claim the file is shorter than the payloads it declares.
        let err = read_header(&bytes, 1024).unwrap_err();
        assert!(
            format!("{err}").contains("truncated") || format!("{err}").contains("holds"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn header_overflow_is_reported_not_silently_truncated() {
        let w = vec![0.1f32; 128];
        let packed = crate::quantize_int4_g128(&w, 128, 1).unwrap();
        let mut buf = Cursor::new(Vec::new());
        let mut writer = Writer::new(&mut buf, sample_config(), 16).unwrap(); // absurdly small
        writer.write_int4("a", &packed).unwrap();
        let err = writer.finish().unwrap_err();
        assert!(format!("{err}").contains("reserved"), "unexpected error: {err}");
    }

    #[test]
    fn fnv_detects_single_bit_flips() {
        let a = b"the quick brown fox";
        let mut b = *a;
        b[7] ^= 1;
        assert_ne!(fnv1a(a), fnv1a(&b));
    }
}
