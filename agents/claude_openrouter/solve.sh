#!/bin/bash

# Claude Code driving an Anthropic model over OpenRouter's Anthropic-compatible
# gateway (the "Anthropic Skin": ANTHROPIC_BASE_URL + /v1/messages).
unset GEMINI_API_KEY
unset CODEX_API_KEY

export BASH_MAX_TIMEOUT_MS="36000000"

# The agent container runs as root, and Claude Code refuses --dangerously-skip-permissions
# under root unless IS_SANDBOX is set. The container is an isolated, --rm docker sandbox,
# so this is safe and expected here.
export IS_SANDBOX=1

# OpenRouter gateway auth: the key rides as a bearer token in ANTHROPIC_AUTH_TOKEN, and
# ANTHROPIC_API_KEY must be explicitly EMPTY -- a non-empty value is sent as x-api-key and
# makes Claude Code authenticate against Anthropic directly, bypassing the gateway.
export ANTHROPIC_BASE_URL="${VTB_OPENROUTER_BASE_URL:-https://openrouter.ai/api}"
export ANTHROPIC_AUTH_TOKEN="${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set (see api_key.env)}"
export ANTHROPIC_API_KEY=""

# Reasoning effort. Claude Code's model table does not know gateway model ids, so every
# one of them (anthropic/claude-opus-5 included) gets the 32000-token per-request cap
# with thinking counted INSIDE it -- a fixed 31999-token thinking budget would leave no
# room for visible output and every heavy turn would die at the shared cap (proven on
# qwen3.8-max-0902). The effort knob is adaptive instead: the CLI and server reconcile
# the thinking budget with the request envelope themselves, so "high" is safe here.
# VTB_EFFORT_LEVEL (or CLAUDE_CODE_EFFORT_LEVEL in api_key.env) overrides it.
export CLAUDE_CODE_EFFORT_LEVEL="${VTB_EFFORT_LEVEL:-${CLAUDE_CODE_EFFORT_LEVEL:-high}}"

# Per-request max_tokens: the same unknown-id cap of 32000 applies and this variable can
# only go below it, so 32000 is written explicitly as documentation of the real ceiling.
# VTB_MAX_OUTPUT_TOKENS overrides it (downward only). A fixed thinking budget can still
# be forced with VTB_MAX_THINKING_TOKENS (forwarded by run_task.sh) -- keep it well under
# 32000 if used, it shares the envelope with output.
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${VTB_MAX_OUTPUT_TOKENS:-32000}"

# RESEARCHER may name the model without its provider prefix ("claude-opus-5", which keeps
# run directory names slash-free); the gateway needs the full id ("anthropic/claude-opus-5").
MODEL="${AGENT_CONFIG}"
case "$MODEL" in
    */*) : ;;
    *) MODEL="anthropic/${MODEL}" ;;
esac
export ANTHROPIC_MODEL="${MODEL}"

# Keep Task/Explore subagents on the model selected for this run rather than falling back
# to Claude Code's built-in fast/Haiku defaults, which a custom gateway may not serve.
export ANTHROPIC_SMALL_FAST_MODEL="${MODEL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${MODEL}"
export CLAUDE_CODE_SUBAGENT_MODEL="${MODEL}"

# A resume attempt sets VTB_RESUME_NUDGE: continue the previous session rather than
# start fresh. Session state lives under $HOME/.claude, which is the run's persistent
# job directory, so a new container picks the session up.
if [ -n "${VTB_RESUME_NUDGE:-}" ]; then
    printf '%s' "$VTB_RESUME_NUDGE" | claude --continue --print --verbose --model "$MODEL" \
        --output-format stream-json \
        --dangerously-skip-permissions
else
    printf '%s' "$PROMPT" | claude --print --verbose --model "$MODEL" \
        --output-format stream-json \
        --dangerously-skip-permissions
fi
