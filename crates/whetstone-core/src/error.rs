//! Error types for the Whetstone core.

/// Something went wrong loading or preparing a model.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Filesystem failure.
    #[error("io: {0}")]
    Io(String),

    /// `config.json` was missing, unparseable, or described a model Whetstone
    /// cannot execute.
    #[error("config: {0}")]
    Config(String),

    /// A checkpoint file was malformed, truncated, or internally inconsistent.
    #[error("format: {0}")]
    Format(String),

    /// A tensor had a shape the caller could not use.
    #[error("shape: {0}")]
    Shape(String),

    /// A tensor the model requires was absent from the checkpoint.
    #[error("missing tensor: {0}")]
    MissingTensor(String),

    /// A failure from the CUDA layer.
    #[error(transparent)]
    Kernel(#[from] whetstone_kernels::Error),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;
