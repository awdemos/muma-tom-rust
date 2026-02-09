use std::collections::HashMap;
use std::sync::Arc;
use crate::api_clients::{ChatCompletionRequest, ChatMessage, MessageRole, LlmClient};
use crate::error::{MumaTomError, Result};
use crate::models::{
    Event, EventType, ActionEvent, UtteranceEvent, MentalState, SocialGoal,
    Hypothesis, EnvironmentState, Question,
};
use crate::inverse_planning::likelihood::LikelihoodEstimator;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PosteriorResult {
    pub hypothesis_id: String,
    pub log_likelihood: f64,
    pub prior_probability: f64,
    pub posterior: f64,
    pub normalized_posterior: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingResult {
    pub hypotheses: Vec<PosteriorResult>,
    pub best_hypothesis: Option<PosteriorResult>,
}

pub mod likelihood;

pub struct InversePlanner {
    llm: Arc<dyn LlmClient + Send + Sync>,
}

impl InversePlanner {
    pub fn new(llm: Arc<dyn LlmClient + Send + Sync>) -> Self {
        Self { llm }
    }

    pub async fn compute_posterior(
        &self,
        hypothesis: &Hypothesis,
        events: &[Event],
        initial_state: &EnvironmentState,
    ) -> Result<PosteriorResult> {
        let mut log_likelihood = 0.0f64;

        for event in events {
            match &event.event_type {
                EventType::Action(action) => {
                    let action_likelihood = self
                        .estimate_action_likelihood(
                            hypothesis.agent_id.clone(),
                            action,
                            hypothesis.initial_state.clone(),
                            hypothesis.mental_state.clone(),
                            events,
                        )
                        .await?;

                    log_likelihood += action_likelihood.ln();
                }
                EventType::Utterance(utterance) => {
                    let utterance_likelihood = self
                        .estimate_utterance_likelihood(
                            hypothesis.agent_id.clone(),
                            utterance,
                            hypothesis.initial_state.clone(),
                            hypothesis.mental_state.clone(),
                            events,
                        )
                        .await?;

                    log_likelihood += utterance_likelihood.ln();
                }
            }
        }

        let posterior = log_likelihood.exp() + hypothesis.prior_probability.ln();

        Ok(PosteriorResult {
            hypothesis_id: format!("{}:{}", hypothesis.agent_id, hypothesis.agent_id),
            log_likelihood,
            prior_probability: hypothesis.prior_probability,
            posterior,
            normalized_posterior: 0.0,
        })
    }

    pub async fn rank_hypotheses(
        &self,
        hypotheses: &[Hypothesis],
        events: &[Event],
        initial_state: &EnvironmentState,
    ) -> Result<RankingResult> {
        let mut results = Vec::new();

        for hypothesis in hypotheses {
            let posterior = self
                .compute_posterior(hypothesis, events, initial_state)
                .await?;

            results.push(posterior);
        }

        let max_posterior = results
            .iter()
            .map(|r| r.posterior)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));

        if max_posterior == f64::NEG_INFINITY {
            return Err(MumaTomError::Internal(
                "No valid hypotheses provided".to_string()
            ));
        }

        for result in &mut results {
            result.normalized_posterior = (result.posterior - max_posterior).exp();
        }

        let normalization_sum: f64 = results.iter().map(|r| r.normalized_posterior).sum();

        if normalization_sum > 0.0 {
            for result in &mut results {
                result.normalized_posterior /= normalization_sum;
            }
        }

        let best_hypothesis = results
            .iter()
            .max_by(|a, b| a.normalized_posterior.partial_cmp(&b.normalized_posterior));

        Ok(RankingResult {
            hypotheses: results,
            best_hypothesis: best_hypothesis.map(|r| r.clone()),
        })
    }

    async fn estimate_action_likelihood(
        &self,
        agent_id: String,
        action: &ActionEvent,
        initial_state: String,
        mental_state: &MentalState,
        past_events: &[Event],
    ) -> Result<_likelihood::LikelihoodResult> {
        let prompt = self.build_action_likelihood_prompt(
            &agent_id,
            action,
            &initial_state,
            mental_state,
            past_events,
        )?;

        let request = ChatCompletionRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                ChatMessage {
                    role: MessageRole::System,
                    content: "You are a Theory of Mind estimator. Evaluate the likelihood of an agent's action given their mental state and context. Return a probability between 0 and 1.".to_string(),
                },
                ChatMessage {
                    role: MessageRole::User,
                    content: prompt,
                },
            ],
            max_tokens: Some(50),
            temperature: Some(0.0),
            stream: Some(false),
        };

        let response = self.llm.chat_completion(request).await?;
        likelihood::LikelihoodEstimator::parse_likelihood(&response.content)
    }

    async fn estimate_utterance_likelihood(
        &self,
        agent_id: String,
        utterance: &UtteranceEvent,
        initial_state: String,
        mental_state: &MentalState,
        past_events: &[Event],
    ) -> Result<likelihood::LikelihoodResult> {
        let prompt = self.build_utterance_likelihood_prompt(
            &agent_id,
            utterance,
            &initial_state,
            mental_state,
            past_events,
        )?;

        let request = ChatCompletionRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                ChatMessage {
                    role: MessageRole::System,
                    content: "You are a Theory of Mind estimator. Evaluate the likelihood of an agent's utterance given their mental state and context. Return a probability between 0 and 1.".to_string(),
                },
                ChatMessage {
                    role: MessageRole::User,
                    content: prompt,
                },
            ],
            max_tokens: Some(50),
            temperature: Some(0.0),
            stream: Some(false),
        };

        let response = self.llm.chat_completion(request).await?;
        likelihood::LikelihoodEstimator::parse_likelihood(&response.content)
    }

    fn build_action_likelihood_prompt(
        &self,
        agent_id: &str,
        action: &ActionEvent,
        initial_state: &str,
        mental_state: &MentalState,
        past_events: &[Event],
    ) -> Result<String> {
        let mut prompt = format!(
            "Agent ID: {}\n",
            agent_id
        );

        prompt.push_str(&format!("Initial State: {}\n", initial_state));
        prompt.push_str(&format!("Agent's Mental State:\n"));
        prompt.push_str(&format!("  - Belief of state: {}\n", mental_state.belief_of_state));
        prompt.push_str(&format!("  - Social goal: {:?}\n", Self::social_goal_to_str(mental_state.social_goal)));
        prompt.push_str(&format!("  - Belief of other's goal: {}\n", mental_state.belief_of_other_goal));

        prompt.push_str("\nPast Events:\n");
        for event in past_events.iter().take(5) {
            match &event.event_type {
                EventType::Action(a) => {
                    prompt.push_str(&format!("  - {}: {}\n", event.agent_id, a.description));
                }
                EventType::Utterance(u) => {
                    prompt.push_str(&format!("  - {}: says \"{}\"\n", event.agent_id, u.text));
                }
            }
        }

        prompt.push_str(&format!("\nCurrent Action: {}\n", action.description));
        prompt.push_str("\nGiven this context, how likely is this action on a scale of 0 to 1?");
        prompt.push_str("Respond with just the number (e.g., 0.95, 0.23).");

        Ok(prompt)
    }

    fn build_utterance_likelihood_prompt(
        &self,
        agent_id: &str,
        utterance: &UtteranceEvent,
        initial_state: &str,
        mental_state: &MentalState,
        past_events: &[Event],
    ) -> Result<String> {
        let mut prompt = format!(
            "Agent ID: {}\n",
            agent_id
        );

        prompt.push_str(&format!("Initial State: {}\n", initial_state));
        prompt.push_str(&format!("Agent's Mental State:\n"));
        prompt.push_str(&format!("  - Belief of state: {}\n", mental_state.belief_of_state));
        prompt.push_str(&format!("  - Social goal: {:?}\n", Self::social_goal_to_str(mental_state.social_goal)));
        prompt.push_str(&format!("  - Belief of other's goal: {}\n", mental_state.belief_of_other_goal));

        prompt.push_str("\nPast Events:\n");
        for event in past_events.iter().take(5) {
            match &event.event_type {
                EventType::Action(a) => {
                    prompt.push_str(&format!("  - {}: {}\n", event.agent_id, a.description));
                }
                EventType::Utterance(u) => {
                    prompt.push_str(&format!("  - {}: says \"{}\"\n", event.agent_id, u.text));
                }
            }
        }

        prompt.push_str(&format!("\nCurrent Utterance: \"{}\"\n", utterance.text));

        if let Some(target) = &utterance.target_agent_id {
            prompt.push_str(&format!("(addressed to {})\n", target));
        }

        prompt.push_str("\nGiven this context, how likely is this utterance on a scale of 0 to 1?");
        prompt.push_str("Respond with just the number (e.g., 0.95, 0.23).");

        Ok(prompt)
    }

    fn social_goal_to_str(goal: SocialGoal) -> &'static str {
        match goal {
            SocialGoal::Helping => "Helping",
            SocialGoal::Hindering => "Hindering",
            SocialGoal::Independent => "Independent",
        }
    }
}

impl Default for InversePlanner {
    fn default() -> Self {
        let llm = Arc::new(crate::api_clients::openai::OpenAiClient::new(
            "sk-test-key".to_string(),
            "gpt-4o".to_string()
        )) as Arc<dyn LlmClient + Send + Sync>;

        Self { llm }
    }
}

    }
}

impl Default for InversePlanner {
    fn default() -> Self {
        let llm = Arc::new(crate::api_clients::openai::OpenAiClient::new(
            "sk-test-key".to_string(),
            "gpt-4o".to_string()
        )) as Arc<dyn LlmClient + Send + Sync>;

        Self { llm }
    }
}

pub mod likelihood;

    #[test]
    fn test_social_goal_to_str() {
        assert_eq!(InversePlanner::social_goal_to_str(SocialGoal::Helping), "Helping");
        assert_eq!(InversePlanner::social_goal_to_str(SocialGoal::Hindering), "Hindering");
        assert_eq!(InversePlanner::social_goal_to_str(SocialGoal::Independent), "Independent");
    }

    #[test]
    fn test_posterior_result_serialization() {
        let result = PosteriorResult {
            hypothesis_id: "agent1:agent1".to_string(),
            log_likelihood: -1.2,
            prior_probability: -0.5,
            posterior: -1.7,
            normalized_posterior: 0.8,
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: PosteriorResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.hypothesis_id, result.hypothesis_id);
        assert_eq!(deserialized.log_likelihood, result.log_likelihood);
    }

    #[test]
    fn test_ranking_result_serialization() {
        let result = RankingResult {
            hypotheses: vec![],
            best_hypothesis: None,
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let _deserialized: RankingResult = serde_json::from_str(&serialized).unwrap();
    }

    #[tokio::test]
    async fn test_build_action_likelihood_prompt() {
        let llm = Arc::new(crate::api_clients::openai::OpenAiClient::new(
            "sk-test".to_string(),
            "gpt-4".to_string()
        )) as Arc<dyn LlmClient + Send + Sync>;

        let planner = InversePlanner::new(llm);

        let action = ActionEvent {
            description: "Walks to kitchen".to_string(),
            objects_interacted: vec![],
            locations: vec!["kitchen".to_string()],
        };

        let mental_state = MentalState {
            belief_of_state: "The beer is in the fridge".to_string(),
            social_goal: SocialGoal::Helping,
            belief_of_other_goal: "John wants the beer".to_string(),
        };

        let initial_state = "The beer is in the fridge in the kitchen.";
        let past_events = vec![];

        let prompt = planner
            .build_action_likelihood_prompt("agent1", &action, &initial_state, &mental_state, &past_events)
            .unwrap();

        assert!(prompt.contains("Agent ID: agent1"));
        assert!(prompt.contains("Initial State:"));
        assert!(prompt.contains("Agent's Mental State:"));
        assert!(prompt.contains("Belief of state:"));
        assert!(prompt.contains("Social goal: Helping"));
        assert!(prompt.contains("Belief of other's goal:"));
        assert!(prompt.contains("Current Action: Walks to kitchen"));
        assert!(prompt.contains("how likely is this action"));
    }

    #[tokio::test]
    async fn test_build_utterance_likelihood_prompt() {
        let llm = Arc::new(crate::api_clients::openai::OpenAiClient::new(
            "sk-test".to_string(),
            "gpt-4".to_string()
        )) as Arc<dyn LlmClient + Send + Sync>;

        let planner = InversePlanner::new(llm);

        let utterance = UtteranceEvent {
            text: "The beer is in the fridge".to_string(),
            target_agent_id: Some("agent1".to_string()),
        };

        let mental_state = MentalState {
            belief_of_state: "The beer is on the coffee table".to_string(),
            social_goal: SocialGoal::Helping,
            belief_of_other_goal: "John wants to find the beer".to_string(),
        };

        let initial_state = "The beer is in the fridge in the kitchen.";
        let past_events = vec![];

        let prompt = planner
            .build_utterance_likelihood_prompt("agent1", &utterance, &initial_state, &mental_state, &past_events)
            .unwrap();

        assert!(prompt.contains("Agent ID: agent1"));
        assert!(prompt.contains("Current Utterance:"));
        assert!(prompt.contains("The beer is in the fridge"));
        assert!(prompt.contains("(addressed to agent1)"));
        assert!(prompt.contains("how likely is this utterance"));
    }
}

