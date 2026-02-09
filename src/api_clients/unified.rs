use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Mutex;
use std::time::Instant;
use crate::api_clients::{
    LlmClient, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream,
    OpenAiClient, GeminiClient, RateLimiter, retry_with_backoff, RetryConfig,
};
use crate::error::{MumaTomError, Result, ErrorKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAi,
    Gemini,
}

impl Provider {
    pub fn cost_per_1k_tokens(&self) -> f64 {
        match self {
            Provider::OpenAi => 30.0,
            Provider::Gemini => 1.25,
        }
    }
}

pub struct UnifiedLlmClient {
    openai: Arc<Mutex<Option<OpenAiClient>>>,
    gemini: Arc<Mutex<Option<GeminiClient>>>,
    circuit_breaker: Arc<CircuitBreaker>,
    rate_limiter: Arc<RateLimiter>,
    primary_provider: Provider,
    fallback_enabled: bool,
    cost_optimization_enabled: bool,
}

#[derive(Debug, Clone)]
struct CircuitBreaker {
    state: Arc<Mutex<CircuitState>>,
    failure_count: Arc<AtomicUsize>,
    threshold: usize,
    recovery_timeout: std::time::Duration,
    last_failure: Arc<Mutex<Option<Instant>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    fn new(threshold: usize, recovery_timeout: std::time::Duration) -> Self {
        Self {
            state: Arc::new(Mutex::new(CircuitState::Closed)),
            failure_count: Arc::new(AtomicUsize::new(0)),
            threshold,
            recovery_timeout,
            last_failure: Arc::new(Mutex::new(None)),
        }
    }

    fn can_execute(&self, provider: Provider) -> bool {
        let state = *self.state.lock().unwrap();
        matches!(state, CircuitState::Closed | CircuitState::HalfOpen)
    }

    fn record_success(&self) {
        let mut state = self.state.lock().unwrap();
        *state = CircuitState::Closed;
        self.failure_count.store(0, Ordering::SeqCst);
        *self.last_failure.lock().unwrap() = None;
    }

    fn record_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::SeqCst);
        *self.last_failure.lock().unwrap() = Some(Instant::now());

        if count >= self.threshold {
            *self.state.lock().unwrap() = CircuitState::Open;
        }
    }

    fn attempt_recovery(&self) -> bool {
        let state = *self.state.lock().unwrap();
        if state != CircuitState::Open {
            return false;
        }

        if let Some(last_failure) = *self.last_failure.lock().unwrap() {
            if last_failure.elapsed() >= self.recovery_timeout {
                *self.state.lock().unwrap() = CircuitState::HalfOpen;
                return true;
            }
        }

        false
    }
}

impl UnifiedLlmClient {
    pub fn new(
        openai: Option<OpenAiClient>,
        gemini: Option<GeminiClient>,
        requests_per_second: u32,
    ) -> Self {
        let openai = Arc::new(Mutex::new(openai));
        let gemini = Arc::new(Mutex::new(gemini));
        let circuit_breaker = Arc::new(CircuitBreaker::new(5, std::time::Duration::from_secs(60)));
        let rate_limiter = Arc::new(RateLimiter::new(requests_per_second));

        let primary_provider = if gemini.is_some() {
            Provider::Gemini
        } else if openai.is_some() {
            Provider::OpenAi
        } else {
            Provider::OpenAi
        };

        Self {
            openai,
            gemini,
            circuit_breaker,
            rate_limiter,
            primary_provider,
            fallback_enabled: true,
            cost_optimization_enabled: true,
        }
    }

    pub fn with_circuit_breaker(mut self, threshold: usize, timeout_secs: u64) -> Self {
        self.circuit_breaker = Arc::new(CircuitBreaker::new(
            threshold,
            std::time::Duration::from_secs(timeout_secs),
        ));
        self
    }

    pub fn with_fallback_enabled(mut self, enabled: bool) -> Self {
        self.fallback_enabled = enabled;
        self
    }

    pub fn with_cost_optimization(mut self, enabled: bool) -> Self {
        self.cost_optimization_enabled = enabled;
        self
    }

    pub fn set_primary_provider(mut self, provider: Provider) -> Self {
        self.primary_provider = provider;
        self
    }

    async fn get_client<T: LlmClient + Send + Sync>(&self, provider: Provider) -> Result<T> {
        match provider {
            Provider::OpenAi => {
                let openai = self.openai.lock().await;
                openai.as_ref().ok_or_else(|| {
                    MumaTomError::Internal("OpenAI client not initialized".to_string())
                }).cloned().ok_or_else(|| {
                    MumaTomError::Internal("Failed to clone OpenAI client".to_string())
                })
            }
            Provider::Gemini => {
                let gemini = self.gemini.lock().await;
                gemini.as_ref().ok_or_else(|| {
                    MumaTomError::Internal("Gemini client not initialized".to_string())
                }).cloned().ok_or_else(|| {
                    MumaTomError::Internal("Failed to clone Gemini client".to_string())
                })
            }
        }
    }

    pub async fn chat_completion(&self, request: ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        self.rate_limiter.acquire().await?;

        let providers = if self.cost_optimization_enabled && self.fallback_enabled {
            vec![Provider::Gemini, Provider::OpenAi]
        } else {
            vec![self.primary_provider]
        };

        let mut last_error = None;

        for provider in providers {
            if !self.circuit_breaker.can_execute(provider) {
                tracing::warn!("Circuit breaker open for provider: {:?}", provider);
                continue;
            }

            match self.try_provider(provider, &request).await {
                Ok(response) => {
                    self.circuit_breaker.record_success();
                    return Ok(response);
                }
                Err(e) => {
                    tracing::warn!("Provider {:?} failed: {}", provider, e);
                    self.circuit_breaker.record_failure();
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            MumaTomError::Internal("All providers failed".to_string())
        }))
    }

    async fn try_provider(
        &self,
        provider: Provider,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let retry_config = RetryConfig::default();

        retry_with_backoff(&retry_config, || async {
            match provider {
                Provider::OpenAi => {
                    let client = self.get_client::<OpenAiClient>(provider).await?;
                    client.chat_completion(request.clone()).await
                }
                Provider::Gemini => {
                    let client = self.get_client::<GeminiClient>(provider).await?;
                    client.chat_completion(request.clone()).await
                }
            }
        }).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_new() {
        let breaker = CircuitBreaker::new(3, std::time::Duration::from_secs(30));
        assert!(breaker.can_execute(Provider::OpenAi));
        assert_eq!(breaker.failure_count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let breaker = CircuitBreaker::new(3, std::time::Duration::from_secs(30));

        for _ in 0..3 {
            breaker.record_failure();
        }

        assert!(!breaker.can_execute(Provider::OpenAi));
    }

    #[test]
    fn test_circuit_breaker_recovers() {
        let breaker = CircuitBreaker::new(1, std::time::Duration::from_millis(100));

        breaker.record_failure();
        assert!(!breaker.can_execute(Provider::OpenAi));

        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        assert!(breaker.attempt_recovery());
        assert!(breaker.can_execute(Provider::OpenAi));
    }

    #[test]
    fn test_provider_cost() {
        assert_eq!(Provider::OpenAi.cost_per_1k_tokens(), 30.0);
        assert_eq!(Provider::Gemini.cost_per_1k_tokens(), 1.25);
    }

    #[tokio::test]
    async fn test_unified_client_creation() {
        let client = UnifiedLlmClient::new(None, None, 10);
        assert_eq!(client.primary_provider, Provider::OpenAi);
        assert!(client.fallback_enabled);
        assert!(client.cost_optimization_enabled);
    }

    #[tokio::test]
    async fn test_unified_client_with_providers() {
        let openai = OpenAiClient::new(
            "sk-test-key".to_string(),
            "gpt-4".to_string()
        );
        let gemini = GeminiClient::new(
            "AIza-test-key".to_string(),
            "gemini-1.5-pro"
        ).unwrap();

        let client = UnifiedLlmClient::new(Some(openai), Some(gemini), 10);
        assert_eq!(client.primary_provider, Provider::Gemini);
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let client = UnifiedLlmClient::new(None, None, 5);

        let start = std::time::Instant::now();
        for _ in 0..10 {
            client.rate_limiter.acquire().await.unwrap();
        }
        let elapsed = start.elapsed();

        assert!(elapsed >= std::time::Duration::from_secs(1));
    }
}

