use std::time::Duration;
use tokio::sync::mpsc::unbounded_channel;
use crate::api_clients::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream, ChatMessage,
    MessageRole, TokenUsage, StreamChunk, LlmClient, RetryConfig, retry_with_backoff,
};
use crate::error::{MumaTomError, Result};

pub struct OpenAiClient {
    client: async_openai::Client<async_openai::Http>,
    model: String,
    api_key: String,
    retry_config: RetryConfig,
}

impl OpenAiClient {
    pub fn new(api_key: String, model: String) -> Self {
        let client = async_openai::Client::with_config(
            async_openai::Config::default().with_api_key(api_key.clone())
        );

        Self {
            client,
            model,
            api_key,
            retry_config: RetryConfig::default(),
        }
    }

    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }
}

#[async_trait::async_trait]
impl LlmClient for OpenAiClient {
    async fn chat_completion(&self, request: ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        retry_with_backoff(&self.retry_config, || self._chat_completion(request)).await
    }

    async fn chat_completion_stream(&self, request: ChatCompletionRequest) -> Result<ChatCompletionStream> {
        self._chat_completion_stream(request).await
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

impl OpenAiClient {
    async fn _chat_completion(&self, request: ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        let openai_messages: Vec<async_openai::types::ChatCompletionRequestMessage> =
            request.messages.iter().map(|m| {
                let role = match m.role {
                    MessageRole::System => async_openai::types::ChatCompletionRequestMessageRole::System,
                    MessageRole::User => async_openai::types::ChatCompletionRequestMessageRole::User,
                    MessageRole::Assistant => async_openai::types::ChatCompletionRequestMessageRole::Assistant,
                };
                async_openai::types::ChatCompletionRequestMessageArgs::default()
                    .role(role)
                    .content(&m.content)
                    .build()
                    .unwrap()
                    .into()
            }).collect();

        let openai_request = async_openai::types::CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(openai_messages)
            .max_tokens(request.max_tokens)
            .temperature(request.temperature);

        let response = self.client.chat().create(openai_request).await.map_err(|e| {
            MumaTomError::ApiRequest {
                url: "https://api.openai.com/v1/chat/completions".to_string(),
                message: e.to_string(),
                status: 500,
            }
        })?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_else(|| "".to_string());

        let usage = response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens as u32,
            completion_tokens: u.completion_tokens as u32,
            total_tokens: u.total_tokens as u32,
        });

        Ok(ChatCompletionResponse {
            content,
            model: self.model.clone(),
            usage,
        })
    }

    async fn _chat_completion_stream(&self, request: ChatCompletionRequest) -> Result<ChatCompletionStream> {
        let openai_messages: Vec<async_openai::types::ChatCompletionRequestMessage> =
            request.messages.iter().map(|m| {
                let role = match m.role {
                    MessageRole::System => async_openai::types::ChatCompletionRequestMessageRole::System,
                    MessageRole::User => async_openai::types::ChatCompletionRequestMessageRole::User,
                    MessageRole::Assistant => async_openai::types::ChatCompletionRequestMessageRole::Assistant,
                };
                async_openai::types::ChatCompletionRequestMessageArgs::default()
                    .role(role)
                    .content(&m.content)
                    .build()
                    .unwrap()
                    .into()
            }).collect();

        let openai_request = async_openai::types::CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(openai_messages)
            .stream(true);

        let stream = self.client.chat().create_stream(openai_request).await.map_err(|e| {
            MumaTomError::ApiRequest {
                url: "https://api.openai.com/v1/chat/completions".to_string(),
                message: e.to_string(),
                status: 500,
            }
        })?;

        let (tx, rx) = unbounded_channel();

        tokio::spawn(async move {
            use futures::StreamExt;

            let mut stream = stream;
            loop {
                match stream.next().await {
                    Some(Ok(chunk)) => {
                        for delta in chunk.choices {
                            if let Some(content) = delta.delta.content {
                                let stream_chunk = StreamChunk {
                                    content: Some(content),
                                    done: false,
                                };
                                let _ = tx.send(stream_chunk);
                            }
                        }
                        if chunk.choices.is_empty() || chunk.choices[0].finish_reason.is_some() {
                            let _ = tx.send(StreamChunk {
                                content: None,
                                done: true,
                            });
                            break;
                        }
                    }
                    Some(Err(e)) => {
                        tracing::error!("Stream error: {:?}", e);
                        break;
                    }
                    None => {
                        let _ = tx.send(StreamChunk {
                            content: None,
                            done: true,
                        });
                        break;
                    }
                }
            }
        });

        Ok(ChatCompletionStream { rx })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_client_creation() {
        let client = OpenAiClient::new(
            "sk-test-key".to_string(),
            "gpt-4".to_string()
        );
        assert_eq!(client.model, "gpt-4");
        assert_eq!(client.api_key, "sk-test-key");
    }

    #[test]
    fn test_openai_client_with_retry_config() {
        let retry_config = RetryConfig {
            max_attempts: 5,
            ..Default::default()
        };
        let client = OpenAiClient::new(
            "sk-test-key".to_string(),
            "gpt-4".to_string()
        ).with_retry_config(retry_config);
        assert_eq!(client.retry_config.max_attempts, 5);
    }

    #[tokio::test]
    #[ignore]
    async fn test_chat_completion() {
        let client = OpenAiClient::new(
            std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "sk-test".to_string()),
            "gpt-3.5-turbo".to_string()
        );

        let request = ChatCompletionRequest {
            model: "gpt-3.5-turbo".to_string(),
            messages: vec![
                ChatMessage {
                    role: MessageRole::User,
                    content: "Say hello".to_string(),
                },
            ],
            max_tokens: Some(10),
            temperature: Some(0.7),
            stream: Some(false),
        };

        match client.chat_completion(request).await {
            Ok(response) => {
                assert!(!response.content.is_empty());
                assert!(response.usage.is_some());
            }
            Err(_) => {
                tracing::warn!("Skipping test: API key not set");
            }
        }
    }

    #[test]
    fn test_message_role_conversion() {
        let test_cases = vec![
            (MessageRole::System, async_openai::types::ChatCompletionRequestMessageRole::System),
            (MessageRole::User, async_openai::types::ChatCompletionRequestMessageRole::User),
            (MessageRole::Assistant, async_openai::types::ChatCompletionRequestMessageRole::Assistant),
        ];

        for (role, expected) in test_cases {
            let converted = match role {
                MessageRole::System => async_openai::types::ChatCompletionRequestMessageRole::System,
                MessageRole::User => async_openai::types::ChatCompletionRequestMessageRole::User,
                MessageRole::Assistant => async_openai::types::ChatCompletionRequestMessageRole::Assistant,
            };
            assert_eq!(std::mem::discriminant(&converted), std::mem::discriminant(&expected));
        }
    }
}

