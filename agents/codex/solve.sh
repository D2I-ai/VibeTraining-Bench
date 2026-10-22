#!/bin/bash

unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY

# Presence check only -- never echo the key value into the run log.
echo "DEBUG: CODEX_API_KEY is set: ${CODEX_API_KEY:+yes} (length: ${#CODEX_API_KEY})"

# Reasoning effort defaults to high: a multi-hour research run is bottlenecked by the
# quality of each decision, not by the API bill. VTB_CODEX_REASONING_EFFORT overrides it.
printf '%s' "$PROMPT" | codex --search exec --json -c model_reasoning_summary=detailed \
    -c model_reasoning_effort="${VTB_CODEX_REASONING_EFFORT:-high}" \
    --skip-git-repo-check --yolo --model "$AGENT_CONFIG"