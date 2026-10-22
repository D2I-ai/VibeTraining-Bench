#!/usr/bin/env python3
"""Show an agent's conversation with the grading service, from its trace.

The agent talks to grading over HTTP -- it reads grading.md, POSTs to /submit, polls
/result. Every one of those calls is a shell command in the agent's stream-json trace,
and every reply is the tool_result that follows. This pulls the two out and pairs them,
so a run's self-evaluation behaviour can be read without hand-grepping solve_out.txt.

What it is for: the platform exists to observe how an agent iterates, and how it uses
its self-evaluation budget is a large part of that -- did it submit once and stop, poll
in a tight loop, misread the scalar it got back. The grading service deliberately
returns only a scalar (the full record is archived where the agent cannot see it), so
this is the agent *side* of that exchange, which is exactly the side that shows intent.

Usage:
    python3 eval/grading/show_agent_grading.py <eval_dir>
    python3 eval/grading/show_agent_grading.py <eval_dir> --all  # every tool call, not
                                                                # just grading-related
    python3 eval/grading/show_agent_grading.py <eval_dir> --trace solve_out.txt

<eval_dir> is a run's results directory (the one holding solve_out.txt), or the trace
file itself. Pairing is by tool_use.id == tool_result.tool_use_id, so interleaved calls
stay matched.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Substrings that mark a tool call as talking to grading. Matched against the JSON of
# the call's input, so both the URL and the token-file name catch it.
GRADING_MARKERS = (
    "grading",
    "/submit",
    "/result",
    "/queue",
    "grading_token",
    "grading_url",
    ":5800",
    "job_id",
)

# How much of a request/response body to show before truncating. A trajectory can carry
# a whole file read as one tool_result; the grading exchanges themselves are small.
MAX_BODY = 800


def _load_events(trace: Path) -> list:
    """Every JSON object in a stream-json trace, one per line, bad lines skipped.

    Anthropic's stream-json prints one object per line, but the agent's own stdout is
    interleaved and sometimes prefixes a timestamp, so each line is scanned for the
    outermost braces rather than parsed whole.
    """
    events = []
    raw = trace.read_text(encoding="utf-8", errors="replace")
    for line in raw.splitlines():
        match = re.search(r"\{.*\}\s*$", line)
        if not match:
            continue
        try:
            events.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return events


def _resolve_trace(target: Path) -> Path:
    if target.is_file():
        return target
    candidate = target / "solve_out.txt"
    if candidate.is_file():
        return candidate
    raise SystemExit(f"no trace at {target} (looked for solve_out.txt inside it)")


def _is_grading(name: str, payload: str) -> bool:
    hay = f"{name} {payload}".lower()
    return any(marker in hay for marker in GRADING_MARKERS)


def _clip(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_BODY:
        return text
    return text[:MAX_BODY] + f"\n    ... [{len(text) - MAX_BODY} more chars]"


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", type=Path, help="a run's results dir, or its solve_out.txt")
    parser.add_argument("--trace", default="solve_out.txt", help="trace filename inside eval_dir")
    parser.add_argument("--all", action="store_true", help="show every tool call, not just grading ones")
    args = parser.parse_args(argv)

    target = args.eval_dir
    if target.is_dir() and args.trace != "solve_out.txt":
        target = target / args.trace
    trace = _resolve_trace(target)

    events = _load_events(trace)

    # Index tool_results by the id of the call they answer.
    results: dict = {}
    for event in events:
        if event.get("type") != "user":
            continue
        for block in (event.get("message") or {}).get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results[block.get("tool_use_id")] = block

    shown = 0
    for event in events:
        if event.get("type") != "assistant":
            continue
        for block in (event.get("message") or {}).get("content") or []:
            if block.get("type") != "tool_use":
                continue
            name = block.get("name", "?")
            payload = json.dumps(block.get("input") or {}, ensure_ascii=False)
            if not args.all and not _is_grading(name, payload):
                continue
            shown += 1
            print(f"\n[{shown}] >>> {name}")
            print("    " + _clip(payload).replace("\n", "\n    "))
            result = results.get(block.get("id"))
            if result is None:
                print("    <<< (no result recorded)")
                continue
            body = result.get("content")
            if isinstance(body, list):
                body = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in body
                )
            flag = " [is_error]" if result.get("is_error") else ""
            print(f"    <<<{flag}")
            print("    " + _clip(str(body)).replace("\n", "\n    "))

    if shown == 0:
        where = "any tool calls" if args.all else "grading-related tool calls"
        print(f"no {where} found in {trace}", file=sys.stderr)
        raise SystemExit(1)
    print(f"\n({shown} call{'s' if shown != 1 else ''} shown from {trace})")


if __name__ == "__main__":
    main()
