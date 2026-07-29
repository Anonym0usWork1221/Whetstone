//! A checkpoint that may be one safetensors file or many.
//!
//! # Why this exists
//!
//! Every HuggingFace checkpoint above roughly 2 B parameters ships **sharded**:
//! `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`, and a
//! `model.safetensors.index.json` mapping each tensor name to the file holding
//! it. Below that threshold a single `model.safetensors` is the norm.
//!
//! Whetstone opened the single file and bailed otherwise, which was invisible for
//! as long as the only checkpoint in use was 0.5 B. It is not a small
//! limitation: it excludes **every model large enough for the engine's
//! bandwidth argument to be interesting**, which is most of the point of the
//! project. Qwen2.5-3B is the first size that hits it.
//!
//! # What it does not do
//!
//! It does not concatenate the shards or hold them in RAM. Each shard is mmapped
//! independently and a name lookup routes to the right one, so converting a 15 GB
//! checkpoint touches only the pages of the tensor being quantized — the same
//! property the single-file path already had, preserved rather than rebuilt.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::safetensors::{SafeTensors, TensorView};

/// One or more safetensors shards, addressed as a single tensor namespace.
pub struct Checkpoint {
    shards: Vec<SafeTensors>,
    /// Tensor name to shard index. Built from the index file when sharded, and
    /// from the single shard's own directory otherwise, so lookups cost the same
    /// either way.
    owner: BTreeMap<String, usize>,
    files: Vec<PathBuf>,
}

impl Checkpoint {
    /// Opens the weights in `dir`, sharded or not.
    ///
    /// Prefers a single `model.safetensors`; falls back to the shard index. The
    /// error when neither is present names both, because "no model.safetensors"
    /// is a confusing thing to be told while looking at a directory full of
    /// `model-00001-of-00004.safetensors`.
    pub fn open(dir: &Path) -> Result<Self> {
        let single = dir.join("model.safetensors");
        if single.exists() {
            let st = SafeTensors::open(&single)?;
            let owner = st.names().map(|n| (n.to_string(), 0usize)).collect();
            return Ok(Self { shards: vec![st], owner, files: vec![single] });
        }

        let index = dir.join("model.safetensors.index.json");
        if !index.exists() {
            return Err(Error::Config(format!(
                "no weights in {}: expected either model.safetensors or \
                 model.safetensors.index.json (a sharded checkpoint)",
                dir.display()
            )));
        }

        let raw = std::fs::read_to_string(&index)
            .map_err(|e| Error::Config(format!("could not read {}: {e}", index.display())))?;
        let json: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| Error::Config(format!("malformed {}: {e}", index.display())))?;
        let map = json
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                Error::Config(format!("{} has no weight_map object", index.display()))
            })?;

        // Open each distinct shard once, then point every tensor name at it.
        let mut files: Vec<PathBuf> = Vec::new();
        let mut by_file: BTreeMap<String, usize> = BTreeMap::new();
        let mut owner = BTreeMap::new();
        for (name, file) in map {
            let file = file.as_str().ok_or_else(|| {
                Error::Config(format!("{}: weight_map[{name}] is not a string", index.display()))
            })?;
            let idx = match by_file.get(file) {
                Some(&i) => i,
                None => {
                    let i = files.len();
                    files.push(dir.join(file));
                    by_file.insert(file.to_string(), i);
                    i
                }
            };
            owner.insert(name.clone(), idx);
        }

        let mut shards = Vec::with_capacity(files.len());
        for f in &files {
            if !f.exists() {
                return Err(Error::Config(format!(
                    "{} lists shard {} but it is not present; the download is \
                     incomplete",
                    index.display(),
                    f.display()
                )));
            }
            shards.push(SafeTensors::open(f)?);
        }

        // The index is metadata and the shards are the truth. A tensor the index
        // promises but no shard holds would otherwise surface much later, as a
        // missing-tensor error during conversion with no hint that the index was
        // the thing that lied.
        for (name, &i) in &owner {
            if shards[i].get(name).is_err() {
                return Err(Error::Config(format!(
                    "{} maps {name} to {}, which does not contain it",
                    index.display(),
                    files[i].display()
                )));
            }
        }

        Ok(Self { shards, owner, files })
    }

    /// Number of shards backing this checkpoint.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// The files opened, for reporting.
    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }

    /// Total bytes across every shard.
    pub fn total_bytes(&self) -> u64 {
        self.files
            .iter()
            .filter_map(|f| std::fs::metadata(f).ok())
            .map(|m| m.len())
            .sum()
    }

    /// Every tensor view, in name order.
    pub fn iter(&self) -> impl Iterator<Item = &TensorView> {
        self.owner.iter().filter_map(|(n, &i)| self.shards[i].get(n).ok())
    }

    /// Tensor count across all shards.
    pub fn len(&self) -> usize {
        self.owner.len()
    }

    /// Whether the checkpoint holds no tensors at all.
    pub fn is_empty(&self) -> bool {
        self.owner.is_empty()
    }

    /// Bytes of tensor payload, excluding headers and shard padding.
    pub fn data_bytes(&self) -> usize {
        self.iter().map(TensorView::nbytes).sum()
    }

    /// Every tensor name, in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.owner.keys().map(String::as_str)
    }

    /// Looks up a tensor's view.
    pub fn get(&self, name: &str) -> Result<&TensorView> {
        let i = *self
            .owner
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.into()))?;
        self.shards[i].get(name)
    }

    /// Reads a tensor and widens it to f32.
    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>> {
        let i = *self
            .owner
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.into()))?;
        self.shards[i].to_f32(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_with_neither_layout_says_so_clearly() {
        let dir = std::env::temp_dir().join("whetstone-ckpt-empty");
        std::fs::create_dir_all(&dir).unwrap();
        let e = match Checkpoint::open(&dir) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("an unusable directory must not open"),
        };
        // Both layouts named: being told "no model.safetensors" while looking at
        // model-00001-of-00004.safetensors is how someone concludes the download
        // is broken when it is fine.
        assert!(e.contains("model.safetensors"), "{e}");
        assert!(e.contains("index.json"), "{e}");
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn a_truncated_shard_set_is_caught_at_open() {
        let dir = std::env::temp_dir().join("whetstone-ckpt-partial");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a.weight":"model-00001-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let e = match Checkpoint::open(&dir) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("an unusable directory must not open"),
        };
        assert!(e.contains("incomplete"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
