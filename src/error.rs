use reqwest::StatusCode;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MumaTomError {
    #[error("API request failed: {url} - {message} (status: {status})")]
    ApiRequest {
        url: String,
        message: String,
        status: u16,
    },

    #[error("Rate limit exceeded: {limit} requests, retry after {retry_after}s")]
    RateLimitExceeded { limit: u32, retry_after: u32 },

    #[error("Model unavailable: {model_id}")]
    ModelUnavailable { model_id: String },

    #[error("Invalid response format: {details}")]
    InvalidResponse { details: String },

    #[error("Video processing error: {0}")]
    VideoProcessing(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, MumaTomError>;

impl MumaTomError {
    pub fn kind(&self) -> ErrorKind {
        match self {
            MumaTomError::ApiRequest { status, .. } if status.is_server_error() => {
                ErrorKind::Retryable
            }
            MumaTomError::ApiRequest { status, .. } if *status == 429 => ErrorKind::RateLimited,
            MumaTomError::RateLimitExceeded { .. } => ErrorKind::RateLimited,
            MumaTomError::ModelUnavailable { .. } => ErrorKind::Permanent,
            MumaTomError::InvalidResponse { .. } => ErrorKind::Retryable,
            MumaTomError::VideoProcessing { .. } => ErrorKind::Permanent,
            MumaTomError::Serialization { .. } => ErrorKind::Permanent,
            MumaTomError::Io { .. } => ErrorKind::Retryable,
            MumaTomError::Config { .. } => ErrorKind::Permanent,
            MumaTomError::Internal { .. } => ErrorKind::Permanent,
            _ => ErrorKind::Permanent,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Retryable,
    RateLimited,
    Permanent,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_kinds() {
        assert_eq!(
            MumaTomError::RateLimitExceeded {
                limit: 10,
                retry_after: 60
            }
            .kind(),
            ErrorKind::RateLimited
        );
        assert_eq!(
            MumaTomError::ModelUnavailable {
                model_id: "gpt-4".to_string()
            }
            .kind(),
            ErrorKind::Permanent
        );
        assert_eq!(
            MumaTomError::Io(std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout")).kind(),
            ErrorKind::Retryable
        );
    }

    #[test]
    fn test_server_error_is_retryable() {
        let error = MumaTomError::ApiRequest {
            url: "https://api.openai.com".to_string(),
            message: "Internal server error".to_string(),
            status: 500,
        };
        assert_eq!(error.kind(), ErrorKind::Retryable);
    }

    #[test]
    fn test_rate_limit_is_rate_limited() {
        let error = MumaTomError::ApiRequest {
            url: "https://api.openai.com".to_string(),
            message: "Too many requests".to_string(),
            status: 429,
        };
        assert_eq!(error.kind(), ErrorKind::RateLimited);
    }
}
