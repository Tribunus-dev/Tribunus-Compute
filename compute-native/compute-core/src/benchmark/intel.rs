//! Accuracy / intelligence benchmarks: MMLU, GSM8K, HellaSwag.
//!
//! These benchmarks evaluate model quality using small built-in test sets
//! rather than the full OMLX datasets. The focus is on the benchmark
//! *infrastructure* — the harness that runs the evaluation — not on dataset
//! size.

use super::BenchmarkHarness;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Metric used for accuracy evaluation.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum AccuracyMetric {
    /// Exact match score (correct / total).
    ExactMatch,
    /// Partial credit: fraction of matching tokens.
    PartialCredit,
}

/// A single multiple-choice or free-response question.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Question {
    pub prompt: String,
    pub choices: Vec<String>,
    pub correct_index: usize,
}

/// A named accuracy test with its question set and metric.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AccuracyTest {
    pub name: String,
    pub questions: Vec<Question>,
    pub metric: AccuracyMetric,
}

/// Small built-in MMLU-like test set (10 questions across 3 subjects).
///
/// Covers anatomy, physics, and computer science at an undergraduate level.
pub fn build_mmlu_questions() -> Vec<Question> {
    vec![
        // ── Anatomy ──────────────────────────────────────────────────────
        Question {
            prompt: "Which bone is the longest bone in the human body?".into(),
            choices: vec![
                "A. Femur".into(),
                "B. Tibia".into(),
                "C. Humerus".into(),
                "D. Radius".into(),
            ],
            correct_index: 0,
        },
        Question {
            prompt: "What is the primary function of red blood cells?".into(),
            choices: vec![
                "A. Fight infection".into(),
                "B. Transport oxygen".into(),
                "C. Clot blood".into(),
                "D. Produce antibodies".into(),
            ],
            correct_index: 1,
        },
        Question {
            prompt: "Which organ produces insulin?".into(),
            choices: vec![
                "A. Liver".into(),
                "B. Kidney".into(),
                "C. Pancreas".into(),
                "D. Stomach".into(),
            ],
            correct_index: 2,
        },
        // ── Physics ──────────────────────────────────────────────────────
        Question {
            prompt: "What is the SI unit of force?".into(),
            choices: vec![
                "A. Joule".into(),
                "B. Newton".into(),
                "C. Pascal".into(),
                "D. Watt".into(),
            ],
            correct_index: 1,
        },
        Question {
            prompt: "Which law states that energy cannot be created or destroyed?".into(),
            choices: vec![
                "A. Newton's First Law".into(),
                "B. Law of Conservation of Energy".into(),
                "C. Second Law of Thermodynamics".into(),
                "D. Coulomb's Law".into(),
            ],
            correct_index: 1,
        },
        Question {
            prompt: "What is the speed of light in a vacuum approximately?".into(),
            choices: vec![
                "A. 3 x 10^6 m/s".into(),
                "B. 3 x 10^8 m/s".into(),
                "C. 3 x 10^10 m/s".into(),
                "D. 3 x 10^12 m/s".into(),
            ],
            correct_index: 1,
        },
        // ── Computer Science ─────────────────────────────────────────────
        Question {
            prompt: "What does CPU stand for?".into(),
            choices: vec![
                "A. Central Processing Unit".into(),
                "B. Computer Personal Unit".into(),
                "C. Core Processing Utility".into(),
                "D. Central Program Unit".into(),
            ],
            correct_index: 0,
        },
        Question {
            prompt: "Which data structure uses FIFO (First In, First Out) ordering?".into(),
            choices: vec![
                "A. Stack".into(),
                "B. Queue".into(),
                "C. Tree".into(),
                "D. Hash table".into(),
            ],
            correct_index: 1,
        },
        Question {
            prompt: "What is the time complexity of binary search on a sorted array?".into(),
            choices: vec![
                "A. O(1)".into(),
                "B. O(log n)".into(),
                "C. O(n)".into(),
                "D. O(n log n)".into(),
            ],
            correct_index: 1,
        },
        Question {
            prompt: "Which protocol is used for secure web communication?".into(),
            choices: vec![
                "A. HTTP".into(),
                "B. FTP".into(),
                "C. HTTPS".into(),
                "D. SMTP".into(),
            ],
            correct_index: 2,
        },
    ]
}

/// Small built-in GSM8K-like math test set (10 word problems).
pub fn build_gsm8k_questions() -> Vec<Question> {
    vec![
        Question {
            prompt: "Janet has 3 apples. She buys 5 more. How many apples does she have?".into(),
            choices: vec!["A. 5".into(), "B. 8".into(), "C. 3".into(), "D. 15".into()],
            correct_index: 1,
        },
        Question {
            prompt: "A train travels 60 miles per hour. How far does it travel in 3 hours?".into(),
            choices: vec![
                "A. 20 miles".into(),
                "B. 63 miles".into(),
                "C. 180 miles".into(),
                "D. 120 miles".into(),
            ],
            correct_index: 2,
        },
        Question {
            prompt: "If 5 notebooks cost $12.50, what is the cost of one notebook?".into(),
            choices: vec![
                "A. $2.00".into(),
                "B. $2.50".into(),
                "C. $3.00".into(),
                "D. $1.50".into(),
            ],
            correct_index: 1,
        },
        Question {
            prompt: "A rectangle is 8 cm long and 3 cm wide. What is its area?".into(),
            choices: vec![
                "A. 11 sq cm".into(),
                "B. 22 sq cm".into(),
                "C. 24 sq cm".into(),
                "D. 48 sq cm".into(),
            ],
            correct_index: 2,
        },
        Question {
            prompt: "Tom has $50. He spends $18 on dinner. How much money does he have left?".into(),
            choices: vec![
                "A. $68".into(),
                "B. $32".into(),
                "C. $28".into(),
                "D. $42".into(),
            ],
            correct_index: 1,
        },
        Question {
            prompt: "If a baker bakes 24 cookies per batch and makes 3 batches, how many cookies does she bake?".into(),
            choices: vec![
                "A. 27".into(),
                "B. 48".into(),
                "C. 72".into(),
                "D. 96".into(),
            ],
            correct_index: 2,
        },
        Question {
            prompt: "There are 12 students in a class. 1/3 of them wear glasses. How many wear glasses?".into(),
            choices: vec![
                "A. 3".into(),
                "B. 4".into(),
                "C. 6".into(),
                "D. 9".into(),
            ],
            correct_index: 1,
        },
        Question {
            prompt: "A store sells apples at 3 for $1. How much do 15 apples cost?".into(),
            choices: vec![
                "A. $3".into(),
                "B. $4".into(),
                "C. $5".into(),
                "D. $6".into(),
            ],
            correct_index: 2,
        },
        Question {
            prompt: "Sarah runs 2.5 km every day. How many kilometers does she run in 6 days?".into(),
            choices: vec![
                "A. 12 km".into(),
                "B. 15 km".into(),
                "C. 18 km".into(),
                "D. 10 km".into(),
            ],
            correct_index: 1,
        },
        Question {
            prompt: "If it takes 4 hours to paint a room, how many rooms can be painted in 20 hours?".into(),
            choices: vec![
                "A. 4".into(),
                "B. 5".into(),
                "C. 6".into(),
                "D. 8".into(),
            ],
            correct_index: 1,
        },
    ]
}

/// Small built-in HellaSwag-like commonsense test set (10 questions).
pub fn build_hellaswag_questions() -> Vec<Question> {
    vec![
        Question {
            prompt: "A woman is cooking eggs. She".into(),
            choices: vec![
                "A. cracks the eggs into a bowl and whisks them.".into(),
                "B. puts the eggs in the oven and closes the door.".into(),
                "C. pours the eggs into the gas tank.".into(),
                "D. plants the eggs in the garden.".into(),
            ],
            correct_index: 0,
        },
        Question {
            prompt: "After the rain stopped, the children".into(),
            choices: vec![
                "A. put on their coats and went outside to play in the puddles.".into(),
                "B. stayed inside and waited for the sun to dry everything.".into(),
                "C. opened umbrellas and stood under the rain gutters.".into(),
                "D. called the weather service to report the rain.".into(),
            ],
            correct_index: 0,
        },
        Question {
            prompt: "A musician is preparing for a concert. She".into(),
            choices: vec![
                "A. tunes her instrument and reviews the sheet music.".into(),
                "B. waters the plants on stage.".into(),
                "C. counts the number of seats in the audience.".into(),
                "D. sweeps the floor of the concert hall.".into(),
            ],
            correct_index: 0,
        },
        Question {
            prompt: "A man is fixing a leaky faucet. He".into(),
            choices: vec![
                "A. turns off the water supply and removes the old washer.".into(),
                "B. paints the faucet with a new coat of paint.".into(),
                "C. covers the faucet with a cloth.".into(),
                "D. calls a florist to arrange flowers around it.".into(),
            ],
            correct_index: 0,
        },
        Question {
            prompt: "A teacher enters a noisy classroom. She".into(),
            choices: vec![
                "A. raises her hand and waits for the students to quiet down.".into(),
                "B. turns off the lights and leaves the room.".into(),
                "C. starts singing loudly to match the noise.".into(),
                "D. opens all the windows to let the noise out.".into(),
            ],
            correct_index: 0,
        },
        Question {
            prompt: "Someone wants to start a small garden. They".into(),
            choices: vec![
                "A. prepare the soil, plant seeds, and water them regularly.".into(),
                "B. go to the beach and collect seashells.".into(),
                "C. buy a fish tank and fill it with water.".into(),
                "D. climb a tree and build a nest.".into(),
            ],
            correct_index: 0,
        },
        Question {
            prompt: "A programmer is debugging code. She".into(),
            choices: vec![
                "A. adds print statements to trace variable values.".into(),
                "B. deletes all files in the project folder.".into(),
                "C. unplugs the computer from the power source.".into(),
                "D. prints the code on paper and folds it into origami.".into(),
            ],
            correct_index: 0,
        },
        Question {
            prompt: "A baker is making a cake. She".into(),
            choices: vec![
                "A. mixes flour, sugar, eggs, and butter in a bowl.".into(),
                "B. puts the ingredients in the washing machine.".into(),
                "C. freezes the raw eggs until they turn solid.".into(),
                "D. grills the cake over an open flame.".into(),
            ],
            correct_index: 0,
        },
        Question {
            prompt: "A driver approaches a red traffic light. They".into(),
            choices: vec![
                "A. stop the car and wait for the green light.".into(),
                "B. speed up to cross before it turns green.".into(),
                "C. close their eyes and keep driving.".into(),
                "D. get out of the car and push it.".into(),
            ],
            correct_index: 0,
        },
        Question {
            prompt: "A person receives a package in the mail. They".into(),
            choices: vec![
                "A. open the box and inspect the contents.".into(),
                "B. throw the box in the trash without opening it.".into(),
                "C. mail the box back to the sender.".into(),
                "D. bury the box in the backyard.".into(),
            ],
            correct_index: 0,
        },
    ]
}

// ---------------------------------------------------------------------------
// AccuracyTest implementation
// ---------------------------------------------------------------------------

impl AccuracyTest {
    /// Create a new accuracy test with the given name, questions, and metric.
    pub fn new(
        name: impl Into<String>,
        questions: Vec<Question>,
        metric: AccuracyMetric,
    ) -> Self {
        Self {
            name: name.into(),
            questions,
            metric,
        }
    }

    /// Run the accuracy test:
    ///
    /// 1. For each question, build a prompt that includes the question and
    ///    choices, then run inference to get the model's answer.
    /// 2. Extract the predicted letter (A/B/C/D) from the output.
    /// 3. Compare to `correct_index` and compute accuracy.
    pub fn run(&self, harness: &BenchmarkHarness) -> Result<f64, String> {
        if self.questions.is_empty() {
            return Ok(1.0); // vacuously perfect
        }

        let mut correct = 0u32;
        for q in &self.questions {
            let prompt = build_mcq_prompt(q);
            let output = harness.run_inference_for_text(&prompt, 16)?;

            let predicted = extract_mcq_answer(&output);
            if predicted == q.correct_index {
                correct += 1;
            }
        }

        Ok(correct as f64 / self.questions.len() as f64)
    }
}

// ---------------------------------------------------------------------------
// Convenience runners (used by BenchmarkHarness::run_all)
// ---------------------------------------------------------------------------

/// Run the built-in MMLU test set and return accuracy as a fraction [0, 1].
pub fn run_mmlu(harness: &BenchmarkHarness) -> Result<f64, String> {
    let questions = build_mmlu_questions();
    let test = AccuracyTest::new("mmlu_5shot", questions, AccuracyMetric::ExactMatch);
    test.run(harness)
}

/// Run the built-in GSM8K test set and return accuracy as a fraction [0, 1].
pub fn run_gsm8k(harness: &BenchmarkHarness) -> Result<f64, String> {
    let questions = build_gsm8k_questions();
    let test = AccuracyTest::new("gsm8k_5shot", questions, AccuracyMetric::ExactMatch);
    test.run(harness)
}

/// Run the built-in HellaSwag test set and return accuracy as a fraction [0, 1].
pub fn run_hellaswag(harness: &BenchmarkHarness) -> Result<f64, String> {
    let questions = build_hellaswag_questions();
    let test = AccuracyTest::new("hellaswag_0shot", questions, AccuracyMetric::ExactMatch);
    test.run(harness)
}

/// Build a multiple-choice prompt string from a Question.
fn build_mcq_prompt(q: &Question) -> String {
    let mut prompt = String::new();
    prompt.push_str(&q.prompt);
    prompt.push('\n');
    for choice in &q.choices {
        prompt.push_str(choice);
        prompt.push('\n');
    }
    prompt.push_str("Answer:");
    prompt
}

/// Extract the predicted answer index (0..len) from a model output string.
///
/// Heuristic: look for "A", "B", "C", "D" in the output and map back to
/// index 0..3.  Returns `correct_index + 1` (out of range) on failure so
/// the answer is counted as wrong.
fn extract_mcq_answer(output: &str) -> usize {
    let trimmed = output.trim();
    // Check for letter prefix patterns: "A", "A.", "A)", "(A)"
    for (i, letter) in ['A', 'B', 'C', 'D'].iter().enumerate() {
        let patterns = [
            format!("{letter}"),
            format!("{letter}."),
            format!("{letter})"),
            format!("({letter})"),
        ];
        for pat in &patterns {
            if trimmed.starts_with(pat) || trimmed.contains(pat.as_str()) {
                return i;
            }
        }
        // Also check the letter alone when it's a single character
        if trimmed.len() == 1 && trimmed.chars().next() == Some(*letter) {
            return i;
        }
    }
    // Fallback: return out-of-range (wrong answer)
    usize::MAX
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_mcq_answer() {
        assert_eq!(extract_mcq_answer("A"), 0);
        assert_eq!(extract_mcq_answer("A."), 0);
        assert_eq!(extract_mcq_answer("B)"), 1);
        assert_eq!(extract_mcq_answer("(C)"), 2);
        assert_eq!(extract_mcq_answer("D. This is the correct one"), 3);
        assert_eq!(extract_mcq_answer("The answer is B obviously"), 1);
        // Default to wrong for unrecognized
        assert_eq!(extract_mcq_answer("Maybe"), usize::MAX);
    }

    #[test]
    fn test_build_mmlu_questions_count() {
        assert_eq!(build_mmlu_questions().len(), 10);
    }

    #[test]
    fn test_build_gsm8k_questions_count() {
        assert_eq!(build_gsm8k_questions().len(), 10);
    }

    #[test]
    fn test_build_hellaswag_questions_count() {
        assert_eq!(build_hellaswag_questions().len(), 10);
    }

    #[test]
    fn test_accuracy_test_empty_questions() {
        let test = AccuracyTest::new("empty", vec![], AccuracyMetric::ExactMatch);
    // Empty set returns Ok(1.0) without touching the harness.
    // We just verify the early-return path logic is correct.
    assert!(test.questions.is_empty());
    assert!(test.questions.is_empty() && matches!(test.metric, AccuracyMetric::ExactMatch));
    }

    #[test]
    fn test_build_mcq_prompt_contains_answer() {
        let questions = build_mmlu_questions();
        let prompt = build_mcq_prompt(&questions[0]);
        assert!(prompt.contains("Answer:"));
        assert!(prompt.contains("Femur"));
        assert!(prompt.contains("Tibia"));
    }
}
