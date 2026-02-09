use serde::{Deserialize, Serialize};

pub struct LikelihoodEstimator;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LikelihoodResult {
    pub likelihood: f64,
}

impl LikelihoodEstimator {
    pub fn parse_likelihood(response: &str) -> Result<LikelihoodResult, String> {
        let response_lower = response.to_lowercase();

        if response_lower.contains("likely") {
            Ok(LikelihoodResult { likelihood: 0.8 })
        } else if response_lower.contains("unlikely") {
            Ok(LikelihoodResult { likelihood: 0.2 })
        } else if response_lower.contains("very likely") {
            Ok(LikelihoodResult { likelihood: 0.95 })
        } else if response_lower.contains("very unlikely") {
            Ok(LikelihoodResult { likelihood: 0.05 })
        } else {
            let num_match = response_lower
                .chars()
                .filter(|c| c.is_digit())
                .collect::<String>();

            if !num_match.is_empty() {
                match num_match.parse::<f64>() {
                    Ok(prob) if (0.0..=1.0).contains(&prob) => {
                        Ok(LikelihoodResult { likelihood: prob })
                    }
                    Err(_) => Err(format!("Invalid probability in response: {}", response)),
                }
            } else {
                Err(format!(
                    "Could not parse likelihood from response: {}",
                    response
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_likelihood_likely() {
        let response = "The action is likely given the context.";
        let result = LikelihoodEstimator::parse_likelihood(response).unwrap();
        assert_eq!(result.likelihood, 0.8);
    }

    #[test]
    fn test_parse_likelihood_unlikely() {
        let response = "The action is unlikely.";
        let result = LikelihoodEstimator::parse_likelihood(response).unwrap();
        assert_eq!(result.likelihood, 0.2);
    }

    #[test]
    fn test_parse_likelihood_very_likely() {
        let response = "The action is very likely.";
        let result = LikelihoodEstimator::parse_likelihood(response).unwrap();
        assert_eq!(result.likelihood, 0.95);
    }

    #[test]
    fn test_parse_likelihood_very_unlikely() {
        let response = "The action is very unlikely.";
        let result = LikelihoodEstimator::parse_likelihood(response).unwrap();
        assert_eq!(result.likelihood, 0.05);
    }

    #[test]
    fn test_parse_likelihood_numeric() {
        let response = "The likelihood is 0.73";
        let result = LikelihoodEstimator::parse_likelihood(response).unwrap();
        assert_eq!(result.likelihood, 0.73);
    }

    #[test]
    fn test_parse_likelihood_invalid() {
        let response = "The action is something.";
        let result = LikelihoodEstimator::parse_likelihood(response);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_likelihood_result_serialization() {
        let result = LikelihoodResult { likelihood: 0.75 };
        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: LikelihoodResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.likelihood, 0.75);
    }

    #[test]
    fn test_parse_likelihood_invalid_probability() {
        let response = "The likelihood is 1.5";
        let result = LikelihoodEstimator::parse_likelihood(response);
        assert!(result.is_err());
    }
}
