use crate::error::MumaTomError;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub openai: OpenAiConfig,
    pub gemini: GeminiConfig,
    pub api: ApiConfig,
    pub caching: CachingConfig,
    pub paths: PathsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiConfig {
    pub api_key: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiConfig {
    pub api_key: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub timeout_secs: u64,
    pub retry_max_attempts: u32,
    pub requests_per_second: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    pub enable_llm_cache: bool,
    pub cache_ttl_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathsConfig {
    pub benchmark_data_path: String,
    pub train_data_path: String,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let openai = OpenAiConfig {
            api_key: std::env::var("OPENAI_API_KEY")
                .map_err(|_| ConfigError::MissingEnvVar("OPENAI_API_KEY".to_string()))?,
            model: std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string()),
        };

        let gemini = GeminiConfig {
            api_key: std::env::var("GEMINI_API_KEY")
                .map_err(|_| ConfigError::MissingEnvVar("GEMINI_API_KEY".to_string()))?,
            model: std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-1.5-pro".to_string()),
        };

        let api = ApiConfig {
            timeout_secs: std::env::var("API_TIMEOUT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(120),
            retry_max_attempts: std::env::var("RETRY_MAX_ATTEMPTS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3),
            requests_per_second: std::env::var("REQUESTS_PER_SECOND")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
        };

        let caching = CachingConfig {
            enable_llm_cache: std::env::var("ENABLE_LLM_CACHE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(true),
            cache_ttl_secs: std::env::var("CACHE_TTL_SECONDS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3600),
        };

        let paths = PathsConfig {
            benchmark_data_path: std::env::var("BENCHMARK_DATA_PATH")
                .unwrap_or_else(|_| "./data/benchmark".to_string()),
            train_data_path: std::env::var("TRAIN_DATA_PATH")
                .unwrap_or_else(|_| "./data/train".to_string()),
        };

        Ok(Self {
            openai,
            gemini,
            api,
            caching,
            paths,
        })
    }

    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.api.timeout_secs)
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigError {
    #[error("Missing environment variable: {0}")]
    MissingEnvVar(String),

    #[error("Failed to parse config: {0}")]
    ParseError(String),
}

impl From<ConfigError> for MumaTomError {
    fn from(err: ConfigError) -> Self {
        MumaTomError::ApiRequest(format!("Config error: {}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_env_missing_var() {
        std::env::remove_var("OPENAI_API_KEY");
        let result = Config::from_env();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_defaults() {
        std::env::set_var("OPENAI_API_KEY", "sk-test-key");
        std::env::set_var("GEMINI_API_KEY", "AIza-test-key");

        let config = Config::from_env().unwrap();

        assert_eq!(config.openai.model, "gpt-4o");
        assert_eq!(config.gemini.model, "gemini-1.5-pro");
        assert_eq!(config.api.timeout_secs, 120);
        assert_eq!(config.api.retry_max_attempts, 3);
        assert_eq!(config.api.requests_per_second, 10);
        assert_eq!(config.caching.enable_llm_cache, true);
        assert_eq!(config.caching.cache_ttl_secs, 3600);
    }

    #[test]
    fn test_timeout_duration() {
        std::env::set_var("OPENAI_API_KEY", "sk-test-key");
        std::env::set_var("GEMINI_API_KEY", "AIza-test-key");
        std::env::set_var("API_TIMEOUT", "60");

        let config = Config::from_env().unwrap();
        assert_eq!(config.timeout(), Duration::from_secs(60));
    }
}
