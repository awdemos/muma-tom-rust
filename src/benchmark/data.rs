use crate::error::{MumaTomError, Result};
use crate::models::{BenchmarkDataset, Interaction, Question, QuestionType};
use std::path::{Path, PathBuf};

pub struct BenchmarkData {
    pub interactions: Vec<Interaction>,
    pub questions: Vec<Question>,
    pub base_path: PathBuf,
}

impl BenchmarkData {
    pub fn load(base_path: &Path) -> Result<Self> {
        if !base_path.exists() {
            return Err(MumaTomError::Internal(format!(
                "Benchmark path does not exist: {}",
                base_path.display()
            )));
        }

        let videos_path = base_path.join("videos");
        let texts_path = base_path.join("texts");
        let questions_path = base_path.join("questions");

        let interactions = Self::load_interactions(&videos_path, &texts_path)?;
        let questions = Self::load_questions(&questions_path)?;

        Self::validate_dataset(&interactions, &questions)?;

        Ok(Self {
            interactions,
            questions,
            base_path: base_path.to_path_buf(),
        })
    }

    fn load_interactions(videos_path: &Path, texts_path: &Path) -> Result<Vec<Interaction>> {
        if !videos_path.exists() {
            return Err(MumaTomError::Internal(format!(
                "Videos path does not exist: {}",
                videos_path.display()
            )));
        }

        let mut interactions = Vec::new();

        let video_entries = std::fs::read_dir(videos_path).map_err(|e| {
            MumaTomError::Internal(format!("Failed to read videos directory: {}", e))
        })?;

        for entry in video_entries {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("mp4".to_string()) {
                continue;
            }

            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| MumaTomError::Internal("Invalid video filename".to_string()))?;

            let text_path = texts_path.join(format!("{}.json", stem));
            if !text_path.exists() {
                continue;
            }

            let interaction = Self::load_interaction(&path, &text_path)?;
            interactions.push(interaction);
        }

        Ok(interactions)
    }

    fn load_interaction(video_path: &Path, text_path: &Path) -> Result<Interaction> {
        let video_path_str = video_path
            .to_str()
            .ok_or_else(|| MumaTomError::Internal("Invalid video path".to_string()))?;

        let text_content = std::fs::read_to_string(text_path)
            .map_err(|e| MumaTomError::Internal(format!("Failed to read text file: {}", e)))?;

        let _interaction_data: serde_json::Value =
            serde_json::from_str(&text_content).map_err(|e| {
                MumaTomError::Serialization(format!("Failed to parse interaction JSON: {}", e))
            })?;

        let interaction = Interaction {
            id: uuid::Uuid::new_v4().to_string(),
            agents: vec![
                crate::models::Agent {
                    id: "agent0".to_string(),
                    name: "Agent 0".to_string(),
                },
                crate::models::Agent {
                    id: "agent1".to_string(),
                    name: "Agent 1".to_string(),
                },
            ],
            video_path: video_path_str,
            text_path: text_path.to_string_lossy(),
            events: vec![],
            initial_state: crate::models::EnvironmentState {
                objects: std::collections::HashMap::new(),
                rooms: vec!["living_room".to_string(), "kitchen".to_string()],
            },
        };

        Ok(interaction)
    }

    fn load_questions(questions_path: &Path) -> Result<Vec<Question>> {
        if !questions_path.exists() {
            return Err(MumaTomError::Internal(format!(
                "Questions path does not exist: {}",
                questions_path.display()
            )));
        }

        let mut questions = Vec::new();

        let entries = std::fs::read_dir(questions_path).map_err(|e| {
            MumaTomError::Internal(format!("Failed to read questions directory: {}", e))
        })?;

        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json".to_string()) {
                continue;
            }

            let content = std::fs::read_to_string(&path).map_err(|e| {
                MumaTomError::Internal(format!("Failed to read question file: {}", e))
            })?;

            let question_data: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
                MumaTomError::Serialization(format!("Failed to parse question JSON: {}", e))
            })?;

            let question = Self::parse_question(&question_data)?;
            questions.push(question);
        }

        Ok(questions)
    }

    fn parse_question(data: &serde_json::Value) -> Result<Question> {
        let question_type = data["question_type"]
            .as_str()
            .ok_or_else(|| MumaTomError::Internal("Missing question_type".to_string()))?;

        let question_type = match question_type {
            "belief" => QuestionType::Belief,
            "social_goal" => QuestionType::SocialGoal,
            "belief_of_goal" => QuestionType::BeliefOfGoal,
            _ => {
                return Err(MumaTomError::Internal(format!(
                    "Invalid question type: {}",
                    question_type
                )))
            }
        };

        let text = data["text"]
            .as_str()
            .ok_or_else(|| MumaTomError::Internal("Missing question text".to_string()))?
            .to_string();

        let correct_answer = data["correct_answer"]
            .as_u64()
            .ok_or_else(|| MumaTomError::Internal("Missing correct_answer".to_string()))?
            as usize;

        let options = vec![];

        Ok(Question {
            id: uuid::Uuid::new_v4().to_string(),
            interaction_id: data["interaction_id"]
                .as_str()
                .unwrap_or("unknown")
                .to_string(),
            question_type,
            text,
            options,
            correct_answer,
        })
    }

    fn validate_dataset(interactions: &[Interaction], questions: &[Question]) -> Result<()> {
        if interactions.is_empty() {
            return Err(MumaTomError::Internal("No interactions loaded".to_string()));
        }

        if questions.is_empty() {
            return Err(MumaTomError::Internal("No questions loaded".to_string()));
        }

        let expected_interactions = 225;
        let expected_questions = 900;

        if interactions.len() != expected_interactions {
            tracing::warn!(
                "Expected {} interactions, found {}",
                expected_interactions,
                interactions.len()
            );
        }

        if questions.len() != expected_questions {
            tracing::warn!(
                "Expected {} questions, found {}",
                expected_questions,
                questions.len()
            );
        }

        let belief_count = questions
            .iter()
            .filter(|q| q.question_type == QuestionType::Belief)
            .count();
        let social_goal_count = questions
            .iter()
            .filter(|q| q.question_type == QuestionType::SocialGoal)
            .count();
        let belief_of_goal_count = questions
            .iter()
            .filter(|q| q.question_type == QuestionType::BeliefOfGoal)
            .count();

        tracing::info!(
            "Question distribution: belief={}, social_goal={}, belief_of_goal={}",
            belief_count,
            social_goal_count,
            belief_of_goal_count
        );

        Ok(())
    }

    pub fn get_dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset {
            interactions: self.interactions.clone(),
            questions: self.questions.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_from_nonexistent_path() {
        let result = BenchmarkData::load(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_question_belief_type() {
        let data = serde_json::json!({
            "question_type": "belief",
            "text": "What does Mary believe?",
            "correct_answer": 0,
        });

        let question = BenchmarkData::parse_question(&data).unwrap();
        assert_eq!(question.question_type, QuestionType::Belief);
        assert_eq!(question.text, "What does Mary believe?");
        assert_eq!(question.correct_answer, 0);
    }

    #[test]
    fn test_parse_question_social_goal_type() {
        let data = serde_json::json!({
            "question_type": "social_goal",
            "text": "What is Jessica's social goal?",
            "correct_answer": 1,
        });

        let question = BenchmarkData::parse_question(&data).unwrap();
        assert_eq!(question.question_type, QuestionType::SocialGoal);
    }

    #[test]
    fn test_validate_empty_dataset() {
        let interactions = vec![];
        let questions = vec![];
        let result = BenchmarkData::validate_dataset(&interactions, &questions);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_dataset_with_questions() {
        let interactions = vec![Interaction {
            id: "i1".to_string(),
            agents: vec![],
            video_path: "".to_string(),
            text_path: "".to_string(),
            events: vec![],
            initial_state: crate::models::EnvironmentState {
                objects: std::collections::HashMap::new(),
                rooms: vec![],
            },
        }];

        let questions = vec![
            Question {
                id: "q1".to_string(),
                interaction_id: "i1".to_string(),
                question_type: QuestionType::Belief,
                text: "".to_string(),
                options: vec![],
                correct_answer: 0,
            },
            Question {
                id: "q2".to_string(),
                interaction_id: "i1".to_string(),
                question_type: QuestionType::SocialGoal,
                text: "".to_string(),
                options: vec![],
                correct_answer: 0,
            },
        ];

        let result = BenchmarkData::validate_dataset(&interactions, &questions);
        assert!(result.is_ok());
    }
}
