use crate::api_clients::{ChatCompletionRequest, ChatMessage, MessageRole};
use crate::error::{MumaTomError, Result};
use crate::models::{Question, QuestionOption, MentalState, SocialGoal};
use serde::{Deserialize, Serialize};

pub struct HypothesisParser {
    system_prompt: String,
}

impl HypothesisParser {
    pub fn new() -> Self {
        Self {
            system_prompt: "You are a helpful assistant that analyzes Theory of Mind questions and extracts mental state hypotheses.".to_string(),
        }
    }

    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = prompt;
        self
    }

    pub async fn parse_question_hypotheses(
        &self,
        question: &Question,
        llm_client: &(dyn crate::api_clients::LlmClient + Send + Sync),
    ) -> Result<Vec<ParsedHypothesis>> {
        let prompt = self.build_hypothesis_prompt(question)?;
        let request = ChatCompletionRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                ChatMessage {
                    role: MessageRole::System,
                    content: self.system_prompt.clone(),
                },
                ChatMessage {
                    role: MessageRole::User,
                    content: prompt,
                },
            ],
            max_tokens: Some(2000),
            temperature: Some(0.0),
            stream: Some(false),
        };

        let response = llm_client.chat_completion(request).await?;
        let parsed: HypothesisResponse = serde_json::from_str(&response.content)
            .map_err(|e| MumaTomError::Internal(format!("Failed to parse hypothesis response: {}", e)))?;

        Ok(parsed.hypotheses)
    }

    fn build_hypothesis_prompt(&self, question: &Question) -> Result<String> {
        let mut prompt = format!(
            "Analyze this Theory of Mind question and extract hypotheses for each option.\n\nQuestion: {}\n\n",
            question.text
        );

        prompt.push_str("Options:\n");
        for (i, option) in question.options.iter().enumerate() {
            prompt.push_str(&format!(
                "{}. {}\n",
                Self::option_letter(i),
                option.text
            ));
        }

        prompt.push_str("\nFor each option, extract the three mental variables:\n");
        prompt.push_str("- belief_of_state: What does the agent believe about the physical state? (a statement about the environment)\n");
        prompt.push_str("- social_goal: What is the agent's social goal? (helping, hindering, or independent)\n");
        prompt.push_str("- belief_of_other_goal: What does the agent believe about the other agent's goal? (a statement about what the other agent wants)\n\n");

        prompt.push_str("Respond in this JSON format:\n");
        prompt.push_str("{\n");
        prompt.push_str("  \"hypotheses\": [\n");
        prompt.push_str("    {\n");
        prompt.push_str("      \"option\": \"A\",\n");
        prompt.push_str("      \"belief_of_state\": \"The beer is on the coffee table\",\n");
        prompt.push_str("      \"social_goal\": \"helping\",\n");
        prompt.push_str("      \"belief_of_other_goal\": \"John wants to find the beer\"\n");
        prompt.push_str("    }\n");
        prompt.push_str("  ]\n");
        prompt.push_str("}\n");

        Ok(prompt)
    }

    fn option_letter(index: usize) -> String {
        char::from_u32('A' as u32 + index as u32).to_string()
    }

    pub fn parse_social_goal(text: &str) -> Result<SocialGoal> {
        let lower = text.to_lowercase();
        if lower.contains("help") {
            Ok(SocialGoal::Helping)
        } else if lower.contains("hinder") {
            Ok(SocialGoal::Hindering)
        } else if lower.contains("independent") {
            Ok(SocialGoal::Independent)
        } else {
            Err(MumaTomError::Internal(format!(
                "Cannot parse social goal from: {}",
                text
            )))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HypothesisResponse {
    pub hypotheses: Vec<ParsedHypothesis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedHypothesis {
    pub option: String,
    pub belief_of_state: String,
    pub social_goal: String,
    pub belief_of_other_goal: String,
}

impl ParsedHypothesis {
    pub fn to_mental_state(&self, agent_id: &str) -> Result<MentalState> {
        let social_goal = HypothesisParser::parse_social_goal(&self.social_goal)?;

        Ok(MentalState {
            belief_of_state: self.belief_of_state.clone(),
            social_goal,
            belief_of_other_goal: self.belief_of_other_goal.clone(),
        })
    }
}

impl Default for HypothesisParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_letter_conversion() {
        assert_eq!(HypothesisParser::option_letter(0), "A");
        assert_eq!(HypothesisParser::option_letter(1), "B");
        assert_eq!(HypothesisParser::option_letter(2), "C");
    }

    #[test]
    fn test_parse_social_goal() {
        assert_eq!(
            HypothesisParser::parse_social_goal("She is trying to help him").unwrap(),
            SocialGoal::Helping
        );
        assert_eq!(
            HypothesisParser::parse_social_goal("She intends to hinder him").unwrap(),
            SocialGoal::Hindering
        );
        assert_eq!(
            HypothesisParser::parse_social_goal("She is acting independently").unwrap(),
            SocialGoal::Independent
        );

        assert!(HypothesisParser::parse_social_goal("Unknown goal").is_err());
    }

    #[test]
    fn test_hypothesis_parser_creation() {
        let parser = HypothesisParser::new();
        assert!(!parser.system_prompt.is_empty());
    }

    #[test]
    fn test_hypothesis_parser_custom_prompt() {
        let custom_prompt = "Custom system prompt".to_string();
        let parser = HypothesisParser::new().with_system_prompt(custom_prompt);
        assert_eq!(parser.system_prompt, custom_prompt);
    }

    #[test]
    fn test_parsed_hypothesis_to_mental_state() {
        let parsed = ParsedHypothesis {
            option: "A".to_string(),
            belief_of_state: "The beer is on the coffee table".to_string(),
            social_goal: "helping".to_string(),
            belief_of_other_goal: "John wants to find the beer".to_string(),
        };

        let mental_state = parsed.to_mental_state("agent1").unwrap();

        assert_eq!(mental_state.belief_of_state, "The beer is on the coffee table");
        assert_eq!(mental_state.social_goal, SocialGoal::Helping);
        assert_eq!(mental_state.belief_of_other_goal, "John wants to find the beer");
    }

    #[test]
    fn test_build_hypothesis_prompt() {
        let parser = HypothesisParser::new();
        let question = Question {
            id: "q1".to_string(),
            interaction_id: "i1".to_string(),
            question_type: crate::models::QuestionType::Belief,
            text: "What does Mary believe?".to_string(),
            options: vec![
                QuestionOption {
                    id: "A".to_string(),
                    text: "Mary believes the beer is on the coffee table".to_string(),
                    hypothesis: crate::models::MentalState {
                        belief_of_state: "The beer is on the coffee table".to_string(),
                        social_goal: SocialGoal::Helping,
                        belief_of_other_goal: "John wants the beer".to_string(),
                    },
                },
                QuestionOption {
                    id: "B".to_string(),
                    text: "Mary believes the beer is in the fridge".to_string(),
                    hypothesis: crate::models::MentalState {
                        belief_of_state: "The beer is in the fridge".to_string(),
                        social_goal: SocialGoal::Hindering,
                        belief_of_other_goal: "John wants the beer".to_string(),
                    },
                },
            ],
            correct_answer: 0,
        };

        let prompt = parser.build_hypothesis_prompt(&question).unwrap();

        assert!(prompt.contains("What does Mary believe?"));
        assert!(prompt.contains("A. Mary believes the beer is on the coffee table"));
        assert!(prompt.contains("B. Mary believes the beer is in the fridge"));
        assert!(prompt.contains("belief_of_state"));
        assert!(prompt.contains("social_goal"));
        assert!(prompt.contains("belief_of_other_goal"));
    }
}

