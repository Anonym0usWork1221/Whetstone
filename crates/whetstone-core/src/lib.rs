//! Whetstone core: model configuration, checkpoint loading, and the roofline
//! model that governs the engine's design.
//!
//! # Why the roofline lives here
//!
//! At batch=1 autoregressive decode every weight is read from device memory
//! once and used for a single multiply-add. Arithmetic intensity is about
//! 2 FLOP/byte, while the GPU needs roughly 120 FLOP/byte to saturate its
//! tensor cores. Decode is therefore memory-bandwidth bound by a wide margin,
//! and the token rate is capped at:
//!
//! ```text
//! tok/s <= bandwidth / bytes_read_per_token
//! ```
//!
//! [`Roofline`] makes that bound computable, so a proposed optimization can be
//! evaluated before it is written. An idea that cuts arithmetic without cutting
//! bytes moved does not make decode faster, however impressive its throughput
//! figure looks in isolation.

#![deny(missing_docs)]

pub mod checkpoint;
pub mod config;
pub mod engine;
pub mod error;
pub mod model;
pub mod safetensors;
pub mod tokenizer;

pub use checkpoint::Checkpoint;
pub use config::{Architecture, ModelConfig, Roofline, RopeScalingConfig};
pub use engine::{Engine, Profile, RunStats, Sampler, SamplingConfig};
pub use error::{Error, Result};
pub use model::{DeviceLinear, Embedding, LayerWeights, ModelWeights};
pub use safetensors::{Dtype, SafeTensors, TensorView};
pub use tokenizer::{StreamDecoder, Tokenizer};
