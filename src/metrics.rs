use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use crate::error::{MumaTomError, Result};
use crate::models::{Question, QuestionType, EvaluationResult, CategoryAccuracy};

pub struct MetricsCalculator;

impl MetricsCalculator {
    pub fn export_results_csv(
        output_path: &Path,
        results: &EvaluationResult,
    ) -> Result<()> {
        let mut csv_content = String::new();

        csv_content.push_str("category,total,correct,accuracy\n");

        for (category, acc) in &results.by_category {
            csv_content.push_str(&format!(
                "{},{},{},{}\n",
                category,
                acc.total,
                acc.correct,
                acc.accuracy
            ));
        }

        csv_content.push_str(&format!(
            "overall,{},{},{}\n",
            results.total_questions,
            results.correct,
            results.accuracy
        ));

        tokio::fs::write(output_path, csv_content).await.map_err(|e| {
            MumaTomError::Internal(format!("Failed to write CSV: {}", e))
        })?;

        info!("Exported results to CSV: {:?}", output_path);
        Ok(())
    }

    pub fn export_results_json(
        output_path: &Path,
        results: &EvaluationResult,
    ) -> Result<()> {
        let json_output = serde_json::to_string_pretty(results)?;
        tokio::fs::write(output_path, json_output).await.map_err(|e| {
            MumaTomError::Internal(format!("Failed to write JSON: {}", e))
        })?;

        info!("Exported results to JSON: {:?}", output_path);
        Ok(())
    }

    pub fn calculate_confusion_matrix(
        results: &[QuestionResult],
    questions: &[Question],
    ) -> ConfusionMatrix {
        let mut matrix = HashMap::new();

        let question_types = vec![
            QuestionType::Belief,
            QuestionType::SocialGoal,
            QuestionType::BeliefOfGoal,
        ];

        for question in questions {
            let question_type = question.question_type;
            let type_matrix = matrix.entry(question_type).or_insert_with(ConfusionMatrix {
                true_positive: 0,
                true_negative: 0,
                false_positive: 0,
                false_negative: 0,
            });

            if let Some(result) = question_type_to_category(question_type) {
                let is_true_positive = result.is_correct;
                let actual_label = if is_true_positive { "positive" } else { "negative" };

                for option in &question.options {
                    let predicted_label = result
                        .selected_option
                        .as_ref()
                        .and_then(|opt| {
                            question.correct_answer.contains(opt)
                                .then_some("positive")
                                .unwrap_or_else(|| "negative")
                        })
                        .unwrap_or_else(|| "negative");

                    match (actual_label, predicted_label) {
                        ("positive", "positive") => *type_matrix.true_positive += 1,
                        ("positive", "negative") => *type_matrix.false_positive += 1,
                        ("negative", "positive") => *type_matrix.false_negative += 1,
                        ("negative", "negative") => *type_matrix.true_negative += 1,
                        _ => {}
                    }
                }
            }
        }

        matrix
    }

    pub fn print_summary(results: &EvaluationResult) {
        println!("\n===== MuMA-ToM Benchmark Results =====");
        println!("Total questions: {}", results.total_questions);
        println!("Correct answers: {}", results.correct);
        println!("Overall accuracy: {:.2}%", results.accuracy * 100.0);
        println!("\nBy Category:");

        for (category, acc) in &results.by_category {
            println!("  {}: {} / {} = {:.2}%",
                category,
                acc.correct,
                acc.total,
                acc.accuracy * 100.0
        );

        println!("\nComparison with Paper:");
        println!("LIMP target: 76.6%");
        println!("Our results: {:.2}%", results.accuracy * 100.0);

        if results.accuracy >= 76.6 {
            println!("✅ Exceeds paper baseline!");
        } else {
            let gap = 76.6 - results.accuracy * 100.0;
            println!("❌ {:.2}% below target (gap: {:.2}%)", gap, gap);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    pub true_positive: usize,
    pub true_negative: usize,
    pub false_positive: usize,
    pub false_negative: usize,
}

fn question_type_to_category(qt: &QuestionType) -> String {
    match qt {
        QuestionType::Belief => "belief".to_string(),
        QuestionType::SocialGoal => "social_goal".to_string(),
        QuestionType::BeliefOfGoal => "belief_of_goal".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_confusion_matrix() {
        let questions = vec![
            Question {
                id: "q1".to_string(),
                interaction_id: "i1".to_string(),
                question_type: QuestionType::Belief,
                text: "".to_string(),
                options: vec![
                    crate::models::QuestionOption {
                        id: "A".to_string(),
                        text: "".to_string(),
                        hypothesis: crate::models::MentalState {
                            belief_of_state: "".to_string(),
                            social_goal: crate::models::SocialGoal::Helping,
                            belief_of_other_goal: "".to_string(),
                        },
                    },
                    crate::models::QuestionOption {
                        id: "B".to_string(),
                        text: "".to_string(),
                        hypothesis: crate::models::MentalState {
                            belief_of_state: "".to_string(),
                            social_goal: crate::models::SocialGoal::Hindering,
                            belief_of_other_goal: "".to_string(),
                        },
                    },
                    crate::models::QuestionOption {
                        id: "C".to_string(),
                        text: "".to_string(),
                        hypothesis: crate::models::MentalState {
                            belief_of_state: "".to_string(),
                            social_goal: crate::models::SocialGoal::Independent,
                            belief_of_other_goal: "".to_string(),
                        },
                    },
                ],
                correct_answer: 0,
            },
        ];

        let results = vec![
            crate::models::QuestionResult {
                question_id: "q1".to_string(),
                is_correct: true,
                selected_option: Some("A".to_string()),
                likelihoods: vec![],
                best_hypothesis: None,
            },
        ];

        let matrix = MetricsCalculator::calculate_confusion_matrix(&results, &questions);

        assert_eq!(matrix.true_positive, 1);
        assert_eq!(matrix.false_negative, 2);
        assert_eq!(matrix.true_negative, 0);
        assert_eq!(matrix.false_positive, 0);
    }

    #[test]
    fn test_export_results_csv() {
        let eval = EvaluationResult {
            total_questions: 3,
            correct: 2,
            accuracy: 0.666666666666667,
            by_category: {
                "belief".to_string() => CategoryAccuracy {
                    total: 2,
                    correct: 1,
                    accuracy: 0.5,
                },
                "social_goal".to_string() => CategoryAccuracy {
                    total: 1,
                    correct: 1,
                    accuracy: 1.0,
                },
            },
        };

        let temp_file = std::env::temp_dir().unwrap().join("test_results.csv");
        let result = tokio::task::spawn_blocking(async move {
            MetricsCalculator::export_results_csv(&temp_file, &eval).await
        }).await.unwrap();

        assert!(result.is_ok());
        assert!(temp_file.exists());
    }

    #[test]
    fn test_export_results_json() {
        let eval = EvaluationResult {
            total_questions: 3,
            correct: 2,
            accuracy: 0.666666666666667,
            by_category: {
                "belief".to_string() => CategoryAccuracy {
                    total: 2,
                    correct: 1,
                    accuracy: 0.5,
                },
                "social_goal".to_string() => CategoryAccuracy {
                    total: 1,
                    correct: 1,
                    accuracy: 1.0,
                },
            },
        };

        let temp_file = std::env::temp_dir().unwrap().join("test_results.json");
        let result = tokio::task::spawn_blocking(async move {
            MetricsCalculator::export_results_json(&temp_file, &eval).await
        }).await.unwrap();

        assert!(result.is_ok());
        assert!(temp_file.exists());
    }

    #[tokio::test]
    async fn test_print_summary() {
        let eval = EvaluationResult {
            total_questions: 10,
            correct: 7,
            accuracy: 0.7,
            by_category: {
                "belief".to_string() => CategoryAccuracy {
                    total: 4,
                    correct: 4,
                    accuracy: 1.0,
                },
                "social_goal".to_string() => CategoryAccuracy {
                    total: 3,
                    correct: 2,
                    accuracy: 0.666666666666667,
                },
                "belief_of_goal".to_string() => CategoryAccuracy {
                    total: 3,
                    correct: 1,
                    accuracy: 0.33333333333333334,
                },
            },
        };

        MetricsCalculator::print_summary(&eval);
    }

    #[test]
    fn test_print_summary_below_target() {
        let eval = EvaluationResult {
            total_questions: 10,
            correct: 6,
            accuracy: 0.6,
            by_category: {
                "belief".to_string() => CategoryAccuracy {
                    total: 4,
                    correct: 3,
                    accuracy: 0.75,
                },
                "social_goal".to_string() => CategoryAccuracy {
                    total: 3,
                    correct: 2,
                    accuracy: 0.666666666666667,
                },
                "belief_of_goal".to_string() => CategoryAccuracy {
                    total: 3,
                    correct: 1,
                    accuracy: 0.33333333333333334,
                },
            },
        };

        let output = std::io::sink().lock();
        use std::io::Write;

        MetricsCalculator::print_summary(&eval);
    }
}
