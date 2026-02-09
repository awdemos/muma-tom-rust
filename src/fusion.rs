use crate::api_clients::{
    ChatCompletionRequest, ChatMessage, MessageRole, VideoAnalysisRequest, ImageAnalysisRequest,
    LlmClient, VlmClient, UnifiedLlmClient,
};
use crate::error::{MumaTomError, Result};
use crate::fusion::video::VideoProcessor;
use crate::models::{
    Interaction, Event, EventType, ActionEvent, UtteranceEvent, EnvironmentState, ObjectLocation,
};
use serde::{Deserialize, Serialize};

pub mod video;

pub use video::VideoProcessor;

pub struct FusionModule {
    vlm: Arc<dyn VlmClient + Send + Sync>,
    llm: Arc<dyn LlmClient + Send + Sync>,
    video_processor: VideoProcessor,
}

use std::sync::Arc;

impl FusionModule {
    pub fn new(
        vlm: Arc<dyn VlmClient + Send + Sync>,
        llm: Arc<dyn LlmClient + Send + Sync>,
        fps: f32,
    ) -> Self {
        Self {
            vlm,
            llm,
            video_processor: VideoProcessor::new(fps, None),
        }
    }

    pub async fn fuse_interaction(
        &self,
        interaction: &Interaction,
    ) -> Result<FusedInteraction> {
        let video_events = self.extract_video_events(interaction).await?;
        let text_events = self.extract_text_events(interaction).await?;

        let merged_events = self.merge_events(video_events, text_events)?;
        let fused_events = self.fill_missing_info(&merged_events, &interaction.text_path).await?;

        let initial_state = self.infer_initial_state(&fused_events)?;

        Ok(FusedInteraction {
            interaction_id: interaction.id.clone(),
            events: fused_events,
            initial_state,
            video_source_count: video_events.len(),
            text_source_count: text_events.len(),
        })
    }

    async fn extract_video_events(
        &self,
        interaction: &Interaction,
    ) -> Result<Vec<Event>> {
        let video_analysis_request = VideoAnalysisRequest {
            video_path: interaction.video_path.clone(),
            prompt: self.build_video_analysis_prompt(interaction)?,
            max_frames: Some(20),
        };

        let response = self.vlm.analyze_video(video_analysis_request).await?;
        self.parse_video_analysis(&response)
    }

    async fn extract_text_events(
        &self,
        interaction: &Interaction,
    ) -> Result<Vec<Event>> {
        let text_content = tokio::fs::read_to_string(&interaction.text_path)
            .await
            .map_err(|e| MumaTomError::VideoProcessing(format!("Failed to read text: {}", e)))?;

        let request = ChatCompletionRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                ChatMessage {
                    role: MessageRole::System,
                    content: "You are an event extractor for multi-agent interactions. Extract actions and utterances for each agent.".to_string(),
                },
                ChatMessage {
                    role: MessageRole::User,
                    content: format!(
                        "Extract events from this interaction description:\n\n{}\n\nFor each event, specify:\n- agent_id\n- type (action or utterance)\n- timestamp\n- description",
                        text_content
                    ),
                },
            ],
            max_tokens: Some(1000),
            temperature: Some(0.0),
            stream: Some(false),
        };

        let response = self.llm.chat_completion(request).await?;
        self.parse_text_events(&response.content, &interaction.agents)
    }

    fn merge_events(
        &self,
        video_events: Vec<Event>,
        text_events: Vec<Event>,
    ) -> Result<Vec<Event>> {
        let mut merged = Vec::new();
        let mut video_idx = 0;
        let mut text_idx = 0;

        loop {
            let video_next = video_events.get(video_idx).map(|e| e.timestamp);
            let text_next = text_events.get(text_idx).map(|e| e.timestamp);

            match (video_next, text_next) {
                (Some(v_ts), Some(t_ts)) if v_ts <= t_ts => {
                    merged.push(video_events[video_idx].clone());
                    video_idx += 1;
                }
                (Some(v_ts), Some(t_ts)) => {
                    merged.push(text_events[text_idx].clone());
                    text_idx += 1;
                }
                (Some(_), None) => {
                    merged.push(video_events[video_idx].clone());
                    video_idx += 1;
                }
                (None, Some(_)) => {
                    merged.push(text_events[text_idx].clone());
                    text_idx += 1;
                }
                (None, None) => break,
            }
        }

        Ok(merged)
    }

    async fn fill_missing_info(
        &self,
        events: &[Event],
        text_path: &str,
    ) -> Result<Vec<Event>> {
        let text_content = tokio::fs::read_to_string(text_path)
            .await
            .map_err(|e| MumaTomError::VideoProcessing(format!("Failed to read text: {}", e)))?;

        let events_json = serde_json::to_string(events)?;
        let events_str = events_json.lines().collect::<Vec<_>>().join("\n");

        let request = ChatCompletionRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                ChatMessage {
                    role: MessageRole::System,
                    content: "Fill in missing object names and clarify ambiguous actions using the provided context.".to_string(),
                },
                ChatMessage {
                    role: MessageRole::User,
                    content: format!(
                        "Context from text:\n\n{}\n\nEvents with potential missing info:\n\n{}\n\nFill in missing object names (replace \"some object\" or ambiguous references with specific objects from context).",
                        text_content, events_str
                    ),
                },
            ],
            max_tokens: Some(1500),
            temperature: Some(0.0),
            stream: Some(false),
        };

        let response = self.llm.chat_completion(request).await?;
        self.parse_fused_events(&response.content)
    }

    fn infer_initial_state(&self, events: &[Event]) -> Result<EnvironmentState> {
        let mut objects = std::collections::HashMap::new();
        let mut rooms = std::collections::HashSet::new();

        for event in events {
            if let EventType::Action(action) = &event.event_type {
                rooms.extend(action.locations.iter().filter_map(|loc| {
                    loc.split_whitespace()
                        .find(|word| word.ends_with("room") || word.ends_with("itchen")
                        .cloned()
                }));

                for obj in &action.objects_interacted {
                    if let Some(from_loc) = action.description
                        .find("from")
                        .and_then(|pos| {
                            action.description[pos..].split_whitespace().next()
                        })
                    {
                        if !objects.contains_key(obj) {
                            objects.insert(
                                obj.clone(),
                                ObjectLocation {
                                    object: obj.clone(),
                                    location: from_loc.to_string(),
                                    room: rooms.iter().next().unwrap_or(&"unknown".to_string()).clone(),
                                },
                            );
                        }
                    }
                }
            }
        }

        Ok(EnvironmentState {
            objects,
            rooms: rooms.into_iter().collect(),
        })
    }

    fn build_video_analysis_prompt(&self, interaction: &Interaction) -> Result<String> {
        Ok(format!(
            "Analyze this video and extract actions for both agents.\n\nAgents:\n{}\n\nExtract:\n- For each action, specify:\n  - Agent ID\n  - Action description (walk, grab, put, etc.)\n  - Objects interacted with\n  - Locations (room, surface, container)\n\nBe thorough but concise.",
            interaction.agents.iter()
                .map(|a| format!("- {}: {}", a.id, a.name))
                .collect::<Vec<_>>()
                .join("\n")
        ))
    }

    fn parse_video_analysis(&self, response: &VideoAnalysisResponse) -> Result<Vec<Event>> {
        let mut events = Vec::new();
        let mut timestamp = 0;

        for line in response.analysis.lines() {
            if let Some((agent_id, action_desc)) = self.parse_action_line(line) {
                events.push(Event {
                    timestamp,
                    agent_id: agent_id.to_string(),
                    event_type: EventType::Action(ActionEvent {
                        description: action_desc.to_string(),
                        objects_interacted: self.extract_objects(action_desc),
                        locations: self.extract_locations(action_desc),
                    }),
                });
                timestamp += 10;
            }
        }

        Ok(events)
    }

    fn parse_text_events(&self, response: &str, agents: &[crate::models::Agent]) -> Result<Vec<Event>> {
        let mut events = Vec::new();
        let mut timestamp = 0;

        for line in response.lines() {
            if let Some((agent_id, event_desc)) = self.parse_event_line(line, agents) {
                if event_desc.contains("says") || event_desc.contains("asks") || event_desc.contains("replies") {
                    events.push(Event {
                        timestamp,
                        agent_id: agent_id.to_string(),
                        event_type: EventType::Utterance(UtteranceEvent {
                            text: event_desc.to_string(),
                            target_agent_id: self.extract_target_agent(event_desc, agents),
                        }),
                    });
                } else {
                    events.push(Event {
                        timestamp,
                        agent_id: agent_id.to_string(),
                        event_type: EventType::Action(ActionEvent {
                            description: event_desc.to_string(),
                            objects_interacted: self.extract_objects(event_desc),
                            locations: self.extract_locations(event_desc),
                        }),
                    });
                }
                timestamp += 10;
            }
        }

        Ok(events)
    }

    fn parse_fused_events(&self, response: &str) -> Result<Vec<Event>> {
        serde_json::from_str(response)
            .map_err(|e| MumaTomError::Internal(format!("Failed to parse fused events: {}", e)))
    }

    fn parse_action_line(&self, line: &str) -> Option<(&str, String)> {
        if let Some(agent_part) = line.split(':').next() {
            let agent_id = agent_part.trim();
            let action_desc = line.split(':').skip(1).collect::<Vec<_>>().join(":");
            if !agent_id.is_empty() && !action_desc.is_empty() {
                return Some((agent_id, action_desc));
            }
        }
        None
    }

    fn parse_event_line(&self, line: &str, agents: &[crate::models::Agent]) -> Option<(&str, String)> {
        let line_lower = line.to_lowercase();
        
        for agent in agents {
            let agent_lower = format!("{}:", agent.id).to_lowercase();
            if line_lower.starts_with(&agent_lower) {
                let event_desc = line.strip_prefix(&format!("{}:", agent.id))
                    .unwrap_or("")
                    .trim()
                    .to_string();
                return Some((&agent.id, event_desc));
            }
            
            if line_lower.contains(&format!("\"{}\"", agent.name.to_lowercase())) 
                || line_lower.contains(&format!("{} says", agent.name.to_lowercase())) 
                || line_lower.contains(&format!("{} asks", agent.name.to_lowercase()))
            {
                return Some((&agent.id, line.trim().to_string()));
            }
        }
        None
    }

    fn extract_target_agent(&self, text: &str, agents: &[crate::models::Agent]) -> Option<String> {
        for agent in agents {
            if text.contains(&format!("to {}", agent.name)) 
                || text.contains(&format!("asks {}", agent.name))
            {
                return Some(agent.id.clone());
            }
        }
        None
    }

    fn extract_objects(&self, text: &str) -> Vec<String> {
        let objects = vec![
            "beer", "book", "magazine", "carrot", "milk", "juice", "potato", "phone", "wine",
            "apple", "banana", "cup", "plate", "glass", "remote", "keys", "bag", "box",
        ];

        let mut found = Vec::new();
        for obj in &objects {
            if text.to_lowercase().contains(obj) {
                found.push(obj.to_string());
            }
        }
        found
    }

    fn extract_locations(&self, text: &str) -> Vec<String> {
        let locations = vec![
            "coffee table", "desk", "kitchen table", "counter", "shelf",
            "fridge", "cabinet", "microwave", "drawer", "sink", "sofa",
            "living room", "kitchen", "bedroom", "bathroom",
        ];

        let mut found = Vec::new();
        for loc in &locations {
            if text.to_lowercase().contains(loc) {
                found.push(loc.to_string());
            }
        }
        found
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedInteraction {
    pub interaction_id: String,
    pub events: Vec<Event>,
    pub initial_state: EnvironmentState,
    pub video_source_count: usize,
    pub text_source_count: usize,
}

impl Default for FusionModule {
    fn default() -> Self {
        Self {
            vlm: Arc::new(crate::api_clients::gemini::GeminiClient::new(
                "AIza-test-key".to_string(),
                "gemini-1.5-pro"
            ).unwrap()),
            llm: Arc::new(crate::api_clients::openai::OpenAiClient::new(
                "sk-test-key".to_string(),
                "gpt-4o".to_string()
            )),
            video_processor: VideoProcessor::new(10.0, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_module_creation() {
        let vlm = Arc::new(crate::api_clients::gemini::GeminiClient::new(
            "AIza-test".to_string(),
            "gemini-1.5-pro"
        ).unwrap()) as Arc<dyn VlmClient + Send + Sync>;

        let llm = Arc::new(crate::api_clients::openai::OpenAiClient::new(
            "sk-test".to_string(),
            "gpt-4".to_string()
        )) as Arc<dyn LlmClient + Send + Sync>;

        let module = FusionModule::new(vlm, llm, 10.0);
        assert_eq!(module.video_processor.fps, 10.0);
    }

    #[test]
    fn test_merge_events() {
        let module = FusionModule::default();

        let video_events = vec![
            Event {
                timestamp: 0,
                agent_id: "agent1".to_string(),
                event_type: EventType::Action(ActionEvent {
                    description: "Walks to kitchen".to_string(),
                    objects_interacted: vec![],
                    locations: vec!["kitchen".to_string()],
                }),
            },
            Event {
                timestamp: 20,
                agent_id: "agent1".to_string(),
                event_type: EventType::Action(ActionEvent {
                    description: "Grabs beer".to_string(),
                    objects_interacted: vec!["beer".to_string()],
                    locations: vec!["fridge".to_string()],
                }),
            },
        ];

        let text_events = vec![
            Event {
                timestamp: 10,
                agent_id: "agent0".to_string(),
                event_type: EventType::Utterance(UtteranceEvent {
                    text: "Where is the beer?".to_string(),
                    target_agent_id: Some("agent1".to_string()),
                }),
            },
        ];

        let merged = module.merge_events(video_events, text_events).unwrap();
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].timestamp, 0);
        assert_eq!(merged[1].timestamp, 10);
        assert_eq!(merged[2].timestamp, 20);
    }

    #[test]
    fn test_extract_objects() {
        let module = FusionModule::default();

        let text = "Alice grabs the beer from the fridge and puts it on the table.";
        let objects = module.extract_objects(text);

        assert!(objects.contains(&"beer".to_string()));
        assert!(objects.len() >= 2);
    }

    #[test]
    fn test_extract_locations() {
        let module = FusionModule::default();

        let text = "John walks to the kitchen and opens the fridge.";
        let locations = module.extract_locations(text);

        assert!(locations.contains(&"kitchen".to_string()));
        assert!(locations.contains(&"fridge".to_string()));
    }

    #[test]
    fn test_infer_initial_state() {
        let module = FusionModule::default();

        let events = vec![
            Event {
                timestamp: 0,
                agent_id: "agent1".to_string(),
                event_type: EventType::Action(ActionEvent {
                    description: "Grabs beer from fridge".to_string(),
                    objects_interacted: vec!["beer".to_string()],
                    locations: vec!["fridge".to_string()],
                }),
            },
        ];

        let state = module.infer_initial_state(&events).unwrap();

        assert!(state.objects.contains_key("beer"));
        assert_eq!(state.objects.get("beer").unwrap().location, "fridge");
    }

    #[test]
    fn test_fused_interaction_serialization() {
        let fused = FusedInteraction {
            interaction_id: "test".to_string(),
            events: vec![],
            initial_state: crate::models::EnvironmentState {
                objects: std::collections::HashMap::new(),
                rooms: vec!["living_room".to_string()],
            },
            video_source_count: 5,
            text_source_count: 3,
        };

        let serialized = serde_json::to_string(&fused).unwrap();
        let deserialized: FusedInteraction = serde_json::from_str(&serialized).unwrap();

        assert_eq!(fused.interaction_id, deserialized.interaction_id);
        assert_eq!(fused.video_source_count, deserialized.video_source_count);
    }
}
