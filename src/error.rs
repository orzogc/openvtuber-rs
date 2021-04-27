use thiserror::Error;

pub type Result<T> = core::result::Result<T, OpenVtuberError>;

#[derive(Debug, Error)]
pub enum OpenVtuberError {
    #[error("{0}")]
    Error(&'static str),
    #[error(transparent)]
    NdarrayShapeError(#[from] ndarray::ShapeError),
    #[error(transparent)]
    OpenCvError(#[from] opencv::Error),
    #[error(transparent)]
    TfLiteError(#[from] tflite::Error),
    #[error(transparent)]
    TryFromSliceError(#[from] core::array::TryFromSliceError),
}

impl From<&'static str> for OpenVtuberError {
    fn from(s: &'static str) -> Self {
        OpenVtuberError::Error(s)
    }
}
