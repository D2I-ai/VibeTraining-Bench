# AIME 2025 development validation protocol

This is a direct, non-agentic math evaluation. Each problem receives four independent completions by default, using pinned decoding settings from `benchmark.json`.

The evaluator asks for a final `\boxed{integer}`. It parses the last boxed integer, falling back to the last complete integer in the response. Each problem receives the fraction of correct completions as its avg@n score; reported accuracy is the mean across problems.

AIME 2025 is development validation for `aime__26`. Its questions, answers, and materialized task files must never become training data. Validation feedback may be used only to diagnose transferable behavior.
