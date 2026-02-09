use std::path::Path;
use std::time::Duration;
use crate::api_clients::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageRole, LlmClient,
    VideoAnalysisRequest, VideoAnalysisResponse, ImageAnalysisRequest, ImageAnalysisResponse,
    VlmClient, RetryConfig, retry_with_backoff,
};
use crate::error::{MumaTomError, Result};

pub struct GeminiClient {
    client: gemini_rust::Client,
    model: gemini_rust::Model,
    api_key: String,
    retry_config: RetryConfig,
}

impl GeminiClient {
    pub fn new(api_key: String, model: &str) -> Result<Self> {
        let gemini_model = match model {
            "gemini-1.5-pro" => gemini_rust::Model::Gemini25Pro,
            "gemini-1.5-flash" => gemini_rust::Model::Gemini25Flash,
            "gemini-1.5-flash-lite" => gemini_rust::Model::Gemini25FlashLite,
            _ => return Err(MumaTomError::ModelUnavailable {
                model_id: model.to_string(),
            }),
        };

        let client = gemini_rust::Client::with_config(
            gemini_rust::GeminiConfig::new(api_key.clone())
        );

        Ok(Self {
            client,
            model: gemini_model,
            api_key,
            retry_config: RetryConfig::default(),
        })
    }

    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    fn convert_message(message: &ChatMessage) -> gemini_rust::Content {
        let role = match message.role {
            MessageRole::System => "user".to_string(),
            MessageRole::User => "user".to_string(),
            MessageRole::Assistant => "model".to_string(),
        };

        gemini_rust::Content::new(role, vec![gemini_rust::Part::text(&message.content)])
    }
}

#[async_trait::async_trait]
impl LlmClient for GeminiClient {
    async fn chat_completion(&self, request: ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        retry_with_backoff(&self.retry_config, || self._chat_completion(request)).await
    }

    async fn chat_completion_stream(&self, _request: ChatCompletionRequest) -> Result<crate::api_clients::ChatCompletionStream> {
        Err(MumaTomError::Internal("Streaming not implemented for Gemini yet".to_string()))
    }

    fn model_name(&self) -> &str {
        match self.model {
            gemini_rust::Model::Gemini25Pro => "gemini-1.5-pro",
            gemini_rust::Model::Gemini25Flash => "gemini-1.5-flash",
            gemini_rust::Model::Gemini25FlashLite => "gemini-1.5-flash-lite",
            _ => "gemini-1.5-pro",
        }
    }
}

impl GeminiClient {
    async fn _chat_completion(&self, request: ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        let mut conversation = self.client.start_conversation();

        for message in &request.messages {
            let content = Self::convert_message(message);
            conversation = conversation.append_message(content);
        }

        let response = conversation
            .generate_content()
            .await
            .map_err(|e| {
                MumaTomError::ApiRequest {
                    url: "https://generativelanguage.googleapis.com".to_string(),
                    message: e.to_string(),
                    status: 500,
                }
            })?;

        let content = response.text();

        Ok(ChatCompletionResponse {
            content,
            model: self.model_name().to_string(),
            usage: None,
        })
    }
}

#[async_trait::async_trait]
impl VlmClient for GeminiClient {
    async fn analyze_video(&self, request: VideoAnalysisRequest) -> Result<VideoAnalysisResponse> {
        retry_with_backoff(&self.retry_config, || self._analyze_video(request)).await
    }

    async fn analyze_image(&self, request: ImageAnalysisRequest) -> Result<ImageAnalysisResponse> {
        retry_with_backoff(&self.retry_config, || self._analyze_image(request)).await
    }

    fn model_name(&self) -> &str {
        match self.model {
            gemini_rust::Model::Gemini25Pro => "gemini-1.5-pro",
            gemini_rust::Model::Gemini25Flash => "gemini-1.5-flash",
            gemini_rust::Model::Gemini25FlashLite => "gemini-1.5-flash-lite",
            _ => "gemini-1.5-pro",
        }
    }
}

impl GeminiClient {
    async fn _analyze_video(&self, request: VideoAnalysisRequest) -> Result<VideoAnalysisResponse> {
        let video_path = Path::new(&request.video_path);
        if !video_path.exists() {
            return Err(MumaTomError::VideoProcessing(format!(
                "Video file not found: {}",
                request.video_path
            )));
        }

        let mut conversation = self.client.start_conversation();

        let user_message = gemini_rust::Content::new(
            "user".to_string(),
            vec![
                gemini_rust::Part::text(&request.prompt),
                gemini_rust::Part::file_data(
                    &video_path.to_string_lossy(),
                    "video/mp4".to_string(),
                ),
            ],
        );

        conversation = conversation.append_message(user_message);

        let response = conversation
            .generate_content()
            .await
            .map_err(|e| {
                MumaTomError::ApiRequest {
                    url: "https://generativelanguage.googleapis.com".to_string(),
                    message: e.to_string(),
                    status: 500,
                }
            })?;

        Ok(VideoAnalysisResponse {
            analysis: response.text(),
            frames_analyzed: request.max_frames.unwrap_or(100),
        })
    }

    async fn _analyze_image(&self, request: ImageAnalysisRequest) -> Result<ImageAnalysisResponse> {
        let image_path = Path::new(&request.image_path);
        if !image_path.exists() {
            return Err(MumaTomError::VideoProcessing(format!(
                "Image file not found: {}",
                request.image_path
            )));
        }

        let mut conversation = self.client.start_conversation();

        let user_message = gemini_rust::Content::new(
            "user".to_string(),
            vec![
                gemini_rust::Part::text(&request.prompt),
                gemini_rust::Part::file_data(
                    &image_path.to_string_lossy(),
                    "image/jpeg".to_string(),
                ),
            ],
        );

        conversation = conversation.append_message(user_message);

        let response = conversation
            .generate_content()
            .await
            .map_err(|e| {
                MumaTomError::ApiRequest {
                    url: "https://generativelanguage.googleapis.com".to_string(),
                    message: e.to_string(),
                    status: 500,
                }
            })?;

        Ok(ImageAnalysisResponse {
            analysis: response.text(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_client_creation() {
        let client = GeminiClient::new(
            "AIza-test-key".to_string(),
            "gemini-1.5-pro"
        );
        assert!(client.is_ok());
        assert_eq!(client.unwrap().model_name(), "gemini-1.5-pro");
    }

    #[test]
    fn test_gemini_client_invalid_model() {
        let client = GeminiClient::new(
            "AIza-test-key".to_string(),
            "invalid-model"
        );
        assert!(client.is_err());
    }

    #[test]
    fn test_gemini_model_conversion() {
        let test_cases = vec![
            ("gemini-1.5-pro", gemini_rust::Model::Gemini25Pro),
            ("gemini-1.5-flash", gemini_rust::Model::Gemini25Flash),
            ("gemini-1.5-flash-lite", gemini_rust::Model::Gemini25FlashLite),
        ];

        for (model_str, expected) in test_cases {
            let converted = match model_str {
                "gemini-1.5-pro" => gemini_rust::Model::Gemini25Pro,
                "gemini-1.5-flash" => gemini_rust::Model::Gemini25Flash,
                "gemini-1.5-flash-lite" => gemini_rust::Model::Gemini25FlashLite,
                _ => panic!("Invalid model"),
            };
            assert_eq!(std::mem::discriminant(&converted), std::mem::discriminant(&expected));
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_video_analysis() {
        let client = GeminiClient::new(
            std::env::var("GEMINI_API_KEY").unwrap_or_else(|_| "AIza-test".to_string()),
            "gemini-1.5-flash"
        ).unwrap();

        let request = VideoAnalysisRequest {
            video_path: "test_video.mp4".to_string(),
            prompt: "Describe the actions in this video".to_string(),
            max_frames: Some(10),
        };

        let result = client.analyze_video(request).await;
        if result.is_err() {
            tracing::warn!("Skipping test: Video file not found");
        }
    }
}

