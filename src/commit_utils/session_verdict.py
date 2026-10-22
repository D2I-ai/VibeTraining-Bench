#!/usr/bin/env python3
"""Did the agent's last session end deliberately?

    python3 src/commit_utils/session_verdict.py SOLVE_OUT

Exit 0: the session ended with the agent's own concluding answer -- the researcher
saying it is done, a decision the resume loop honours. Exit 1: it did not (a crash, an
API failure, a thinking-only auto-quit, or a final turn whose text is not an answer).

Only the final answer counts. A claude stream is judged by its terminal `result`
record (a clean exit with text); a stream that just stops was cut off mid-work. A codex
stream is judged by the last content item once `turn.completed` is seen. Each
`===== resume attempt` marker starts a new attempt, and only the latest one may judge.
"""

from __future__ import annotations

import json
import re
import sys

# A final "answer" made of the model's own template control tokens is not an answer.
# The endpoint streamed one back as plain text -- `response<|sep|>Rich...<|open|>tools
# <|sep|>call tool="Bash" ...` -- a tool call that failed to parse. The CLI cannot act on
# it, so it closes the session as a clean success with that string as the result, and a
# test of "not an error, not empty" reads it as the researcher concluding. The run then
# delivered nothing it had meant to deliver, five hours into a 24-hour budget.
#
# Two tokens, not one: a concluding summary may legitimately quote a single `<|im_end|>`
# while discussing the chat template. A little prose beside the tokens is not enough to
# count as an answer either -- here the residue was the fragments of a tool call.
_CONTROL_TOKEN = re.compile(r"<\|[A-Za-z_][A-Za-z0-9_.\-]{0,23}\|>")
_MIN_CONTROL_TOKENS = 2
_MAX_RESIDUE_CHARS = 120


def degenerate(text: str) -> bool:
    """True when the text is control-token debris rather than the agent's answer."""
    tokens = _CONTROL_TOKEN.findall(text)
    if len(tokens) < _MIN_CONTROL_TOKENS:
        return False
    return len(_CONTROL_TOKEN.sub("", text).strip()) < _MAX_RESIDUE_CHARS


def judge(lines) -> tuple[bool, str]:
    """Return (deliberate, reason) for a solve_out stream."""
    claude_deliberate = False
    claude_degenerate = False
    saw_turn_completed = False
    codex_last_kind = None
    codex_last_text = ""

    for line in lines:
        if "===== resume attempt" in line:
            claude_deliberate = False
            claude_degenerate = False
            saw_turn_completed = False
            codex_last_kind = None
            codex_last_text = ""
            continue
        i = line.find("] ")
        if i < 0:
            continue
        try:
            record = json.loads(line[i + 2:])
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        kind = record.get("type")
        if kind == "result":
            result = (record.get("result") or "").strip()
            claude_degenerate = bool(result) and degenerate(result)
            claude_deliberate = (not record.get("is_error")) and bool(result) and not claude_degenerate
        elif kind == "item.completed":
            item = record.get("item") or {}
            item_kind = item.get("type")
            if item_kind in ("agent_message", "command_execution", "reasoning"):
                codex_last_kind = item_kind
                codex_last_text = (item.get("text") or "") if item_kind == "agent_message" else ""
        elif kind == "turn.completed":
            saw_turn_completed = True

    codex_deliberate = (
        saw_turn_completed and codex_last_kind == "agent_message" and bool(codex_last_text.strip())
    )
    if claude_deliberate or codex_deliberate:
        return True, "the session ended with the agent's own concluding answer"
    if claude_degenerate:
        return False, "the final result is control-token debris, not an answer"
    return False, "the session did not end with a concluding answer"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        with open(argv[1], errors="ignore") as handle:
            deliberate, reason = judge(handle)
    except OSError as exc:
        print(f"[resume] cannot read {argv[1]}: {exc}", file=sys.stderr)
        return 1
    if not deliberate and "debris" in reason:
        print(f"[resume] {reason}; treating the exit as unintended", file=sys.stderr)
    return 0 if deliberate else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
