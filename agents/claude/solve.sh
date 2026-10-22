#!/bin/bash
unset GEMINI_API_KEY
unset CODEX_API_KEY

export BASH_MAX_TIMEOUT_MS="36000000"

# The agent container runs as root, and Claude Code refuses --dangerously-skip-permissions
# under root unless IS_SANDBOX is set. The container is an isolated, --rm docker sandbox,
# so this is safe and expected here.
export IS_SANDBOX=1

# Extended thinking budget, in tokens. 31999 is Claude Code's top tier (the "ultrathink"
# budget) -- the right default for a multi-hour research run; 0 turns thinking off.
# VTB_MAX_THINKING_TOKENS overrides it.
export MAX_THINKING_TOKENS="${VTB_MAX_THINKING_TOKENS:-31999}"

# A resume attempt sets VTB_RESUME_NUDGE: continue the previous session rather than
# start fresh. Session state lives under $HOME/.claude, which is the run's persistent
# job directory, so a new container picks the session up.
if [ -n "${VTB_RESUME_NUDGE:-}" ]; then
    printf '%s' "$VTB_RESUME_NUDGE" | claude --continue --print --verbose --model "$AGENT_CONFIG" \
        --output-format stream-json \
        --dangerously-skip-permissions
else
    printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
        --output-format stream-json \
        --dangerously-skip-permissions
fi