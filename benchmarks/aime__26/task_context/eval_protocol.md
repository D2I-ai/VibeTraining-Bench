# AIME 2026 final-evaluation protocol

AIME 2026 is a direct, non-agentic math benchmark. Grading calls the served model through an OpenAI-compatible chat-completions endpoint; it does not run Harbor agents or task containers.

Each problem receives four independent completions by default. The evaluator requests a final `\boxed{integer}`, parses the last boxed integer (or the last complete integer as fallback), and scores each problem by its fraction of correct completions. Overall accuracy is the mean of these avg@n problem scores.

Development validation uses the separate pinned AIME 2025 benchmark. Neither AIME 2025 nor AIME 2026 questions, answers, or materialized task files may be used for training.
