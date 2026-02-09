use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    pub id: String,
    pub agents: Vec<Agent>,
    pub video_path: String,
    pub text_path: String,
    pub events: Vec<Event>,
    pub initial_state: EnvironmentState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    Action(ActionEvent),
    Utterance(UtteranceEvent),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub timestamp: usize,
    pub agent_id: String,
    pub event_type: EventType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionEvent {
    pub description: String,
    pub objects_interacted: Vec<String>,
    pub locations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtteranceEvent {
    pub text: String,
    pub target_agent_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentState {
    pub objects: HashMap<String, ObjectLocation>,
    pub rooms: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectLocation {
    pub object: String,
    pub location: String,
    pub room: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SocialGoal {
    Helping,
    Hindering,
    Independent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentalState {
    pub belief_of_state: String,
    pub social_goal: SocialGoal,
    pub belief_of_other_goal: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    pub id: String,
    pub interaction_id: String,
    pub question_type: QuestionType,
    pub text: String,
    pub options: Vec<QuestionOption>,
    pub correct_answer: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuestionType {
    Belief,
    SocialGoal,
    BeliefOfGoal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionOption {
    pub id: String,
    pub text: String,
    pub hypothesis: MentalState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    pub agent_id: String,
    pub mental_state: MentalState,
    pub initial_state: String,
    pub prior_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkDataset {
    pub interactions: Vec<Interaction>,
    pub questions: Vec<Question>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub total_questions: usize,
    pub correct: usize,
    pub accuracy: f64,
    pub by_category: HashMap<String, CategoryAccuracy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryAccuracy {
    pub total: usize,
    pub correct: usize,
    pub accuracy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mental_state_serialization() {
        let mental_state = MentalState {
            belief_of_state: "The beer is on the coffee table".to_string(),
            social_goal: SocialGoal::Helping,
            belief_of_other_goal: "John wants to find the beer".to_string(),
        };

        let serialized = serde_json::to_string(&mental_state).unwrap();
        let deserialized: MentalState = serde_json::from_str(&serialized).unwrap();

        assert_eq!(mental_state.belief_of_state, deserialized.belief_of_state);
        assert_eq!(mental_state.social_goal, deserialized.social_goal);
    }

    #[test]
    fn test_question_structure() {
        let question = Question {
            id: "q1".to_string(),
            interaction_id: "i1".to_string(),
            question_type: QuestionType::Belief,
            text: "What does Mary believe?".to_string(),
            options: vec![QuestionOption {
                id: "A".to_string(),
                text: "Mary believes the beer is on the coffee table".to_string(),
                hypothesis: MentalState {
                    belief_of_state: "The beer is on the coffee table".to_string(),
                    social_goal: SocialGoal::Helping,
                    belief_of_other_goal: "John wants the beer".to_string(),
                },
            }],
            correct_answer: 0,
        };

        assert_eq!(question.options.len(), 1);
        assert_eq!(question.question_type, QuestionType::Belief);
    }

    #[test]
    fn test_event_timeline() {
        let event = Event {
            timestamp: 100,
            agent_id: "agent1".to_string(),
            event_type: EventType::Action(ActionEvent {
                description: "Walks to the kitchen".to_string(),
                objects_interacted: vec![],
                locations: vec!["kitchen".to_string()],
            }),
        };

        assert_eq!(event.timestamp, 100);
        assert_eq!(event.agent_id, "agent1");
    }
}
