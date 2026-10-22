# Agent Interface

This document describes the fixed terminal-agent interface that training data and the
delivered model must support. It is not a description of the final evaluation set.

## Terminal interaction loop

The supplied `terminus-2` harness gives the model a task description and the
conversation so far. When this run ships the editable harness, you may modify its
allowlisted implementation files as described in `harness.md`, but the terminal
interaction contract remains fixed: the model must
respond with one valid OpenAI tool call requesting a shell command. The harness runs
that command, appends its output, and continues until the model signals completion or
the interaction limit is reached.

Training examples must reflect this loop:

- Emit structured tool calls, not prose commands or Markdown code blocks.
- Issue one command per turn and read its resulting output before choosing the next
  action.
- Inspect state, act, debug from errors, verify the result, and stop appropriately.
- Preserve the supported Terminus-2 multi-turn tool-calling format.

## Transferable capabilities

Prioritize capabilities that generalize across terminal work:

1. Tool-call reliability.
2. Shell fluency and safe command construction.
3. Reading command output and errors.
4. Avoiding repeated failed actions or premature completion.
5. Verifying completed work.

Keep the supplied serving template and tool-call interface compatible. A response that
is not parsed as a tool call does not execute an action.

## Service-controlled decoding

The delivered model is served through an OpenAI-compatible endpoint. Every grading run
uses fixed request settings: temperature `1.0`, top-p `1.0`, maximum output `4096`
tokens, maximum `200` turns, and
`chat_template_kwargs.enable_thinking=false`. Sampling values in a shipped
`generation_config.json` do not change these scoring requests. Include that file only
when it is needed to load the model correctly.

The endpoint has tool calling enabled and uses a parser matched to the supplied chat
template. Do not replace the chat template. The template must continue to emit the
supported tool-call format, or requests will no longer produce executable actions.

## Validation boundary

Development validation is a fixed, service-controlled diagnostic set. It is distinct
from the final evaluation set and is not a substitute for it. Use validation only to
compare transferable hypotheses; never make validation content, outcomes, trajectories,
or restricted evaluation artifacts part of training data, rules, or task-specific
heuristics.

Use `environment.md` (## Candidates, ## Grading) and `grading.md` for the
development-validation procedure.
