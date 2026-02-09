pub mod gemini;
pub mod openai;
pub mod unified;

use async_trait::async_trait;
use std::time::Duration;
use tokio::sync::Semaphore;
use crate::error::{MumaTomError, Result};

pub use unified::UnifiedLlmClient;

pub trait LlmClient: Send + Sync {
    async fn chat_completion(&self, request: ChatCompletionRequest) -> Result<ChatCompletionResponse>;

    async fn chat_completion_stream(&self, request: ChatCompletionRequest) -> Result<ChatCompletionStream>;

    fn model_name(&self) -> &str;
}

#[async_trait]
pub trait VlmClient: Send + Sync {
    async fn analyze_video(&self, request: VideoAnalysisRequest) -> Result<VideoAnalysisResponse>;

    async fn analyze_image(&self, request: ImageAnalysisRequest) -> Result<ImageAnalysisResponse>;

    fn model_name(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatCompletionResponse {
    pub content: String,
    pub model: String,
    pub usage: Option<TokenUsage>,
}

#[derive(Debug, Clone)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

pub struct ChatCompletionStream {
    pub rx: tokio::sync::mpsc::UnboundedReceiver<StreamChunk>,
}

#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub content: Option<String>,
    pub done: bool,
}

#[derive(Debug, Clone)]
pub struct VideoAnalysisRequest {
    pub video_path: String,
    pub prompt: String,
    pub max_frames: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct VideoAnalysisResponse {
    pub analysis: String,
    pub frames_analyzed: usize,
}

#[derive(Debug, Clone)]
pub struct ImageAnalysisRequest {
    pub image_path: String,
    pub prompt: String,
}

#[derive(Debug, Clone)]
pub struct ImageAnalysisResponse {
    pub analysis: String,
}

pub struct RetryConfig {
    pub max_attempts: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub exponential_base: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 100,
            max_delay_ms: 10000,
            exponential_base: 2.0,
        }
    }
}

pub struct RateLimiter {
    semaphore: Semaphore,
    permits_per_second: u32,
}

impl RateLimiter {
    pub fn new(requests_per_second: u32) -> Self {
        Self {
            semaphore: Semaphore::new(requests_per_second),
            permits_per_second: requests_per_second,
        }
    }

    pub async fn acquire(&self) -> Result<()> {
        let permit = self.semaphore.acquire().await.map_err(|e| {
            MumaTomError::Internal(format!("Failed to acquire rate limit permit: {}", e))
        })?;

        std::mem::drop(permit);

        tokio::time::sleep(Duration::from_millis(
            1000 / self.permits_per_second as u64
        )).await;

        Ok(())
    }
}

pub async fn retry_with_backoff<F, Fut, T>(
    config: &RetryConfig,
    operation: F,
) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut last_error = None;

    for attempt in 0..config.max_attempts {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                last_error = Some(e.clone());

                if e.kind() != crate::error::ErrorKind::Retryable
                    && e.kind() != crate::error::ErrorKind::RateLimited
                {
                    return Err(e);
                }

                if attempt < config.max_attempts - 1 {
                    let delay_ms = (config.base_delay_ms as f64
                        * config.exponential_base.powi(attempt))
                        .min(config.max_delay_ms as f64) as u64;

                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        MumaTomError::Internal("Retry loop completed without result".to_string())
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_retry_success_on_first_try() {
        let config = RetryConfig::default();
        let result = retry_with_backoff(&config, || async { Ok(42) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_retry_eventually_succeeds() {
        let config = RetryConfig {
            max_attempts: 5,
            ..Default::default()
        };

        let mut attempts = 0;
        let result = retry_with_backoff(&config, || async {
            attempts += 1;
            if attempts < 3 {
                Err(MumaTomError::ApiRequest {
                    url: "test".to_string(),
                    message: "Temporary error".to_string(),
                    status: 500,
                })
            } else {
                Ok("success".to_string())
            }
        }).await;

        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempts, 3);
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(10);

        let start = std::time::Instant::now();
        for _ in 0..5 {
            limiter.acquire().await.unwrap();
        }
        let elapsed = start.elapsed();

        assert!(elapsed >= Duration::from_millis(400));
        assert!(elapsed < Duration::from_millis(600));
    }

    #[test]
    fn test_chat_message_serialization() {
        let message = ChatMessage {
            role: MessageRole::User,
            content: "Hello, world!".to_string(),
        };

        let serialized = serde_json::to_string(&message).unwrap();
        assert!(serialized.contains("User"));
        assert!(serialized.contains("Hello, world!"));
    }

    #[test]
    fn test_retry_config_defaults() {
        let config = RetryConfig::default();
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.base_delay_ms, 100);
        assert_eq!(config.max_delay_ms, 10000);
        assert_eq!(config.exponential_base, 2.0);
    }
}
