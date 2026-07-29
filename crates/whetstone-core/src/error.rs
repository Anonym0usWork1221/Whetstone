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

    /// The request is well formed but this build cannot serve it — a weight
    /// format with no kernel for the requested path, for instance.
    ///
    /// Distinct from `Config` on purpose: the model is fine, the *operation* is
    /// not available for it, and the caller can usually fall back.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A failure from the CUDA layer.
    #[error(transparent)]
    Kernel(#[from] whetstone_kernels::Error),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;
