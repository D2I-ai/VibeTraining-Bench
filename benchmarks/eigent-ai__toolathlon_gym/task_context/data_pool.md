# Training data pool

Your training data must come from this pool. The pool is the only permitted
source of training examples; you select the subset, but you may not add data
from outside it.

## Location

The pool is the Hugging Face dataset `guoxinzju/vibetrainbench-toolathlon`
(single file `dive-mix8000.jsonl`), declared to the recipe as the pinned,
read-only offline HF-cache resource `data/toolathlon_dive_mix8000`.
Load it from that resource's declared `snapshot_path`
(`.../snapshots/<revision>/dive-mix8000.jsonl`) -- not by repo id -- and write
the subset you train on to `train.jsonl` in your workspace (a straight copy
trains on the whole pool). See the resource's `resource_config.json` for the
exact revision and offline load hint.

## What it is

8,000 multi-turn agent trajectories (~508 MB total) over MCP-style tool
servers. Each line of the JSONL is one independent trajectory:

```json
{
  "tools": "<JSON string: array of OpenAI function schemas available to the trajectory>",
  "messages": [ {"role": "...", "content": "...", "...": "..."}, ... ]
}
```

`tools` is a **string** containing a JSON array in the OpenAI
`[{"type": "function", "function": {...}}]` schema format — parse it with
`json.loads` before use. The trainer ignores it; it is there for inspection.

## Message roles

| role | one per trajectory | meaning |
|---|---|---|
| `system` | yes | tool inventory / environment description |
| `user` | yes | the task description in prose |
| `assistant` | many | model turns; carry tool calls in the standard `tool_calls` field, and embed `​...​` reasoning text before the visible answer |
| `tool` | many | a tool result, paired with the preceding assistant `tool_calls` entry via `tool_call_id` |

Turn order follows the natural agent loop:
`... assistant (tool_calls) → tool → assistant → ...` until the task is
finished. These are the **standard OpenAI chat roles** — the generic Slime SFT
entrypoint's `--input-key messages` reads this file as-is, with no role
conversion; keep this shape if you rebuild, filter or re-tokenize the file
yourself.

Length profile (Qwen3.5 tokenizer over message contents): p50 9,904 tokens,
p90 19,122, max 28,680; ~12.6 assistant turns per trajectory on average, up
to 50. Every trajectory fits a 32k-token training budget, but measure
per-trajectory length yourself before deciding how to filter or bucket them
for training.

## Rules

1. Training data must be a subset of this pool. Filtering, sampling,
   re-formatting, re-tokenizing, splitting long trajectories, and deriving
   auxiliary labels from pool contents are all allowed.
2. Do not add training examples from any other source (public datasets, the
   internet, or data you synthesise from scratch). Internet access remains
   available for documentation and tooling — not for training data.
3. Rule 3 of your task brief still applies: do not train on this benchmark's
   own test data.

## Suggested first step

Sample ~20 lines, parse `tools` and inspect the role distribution before
choosing a subset — an informed choice is usually better than training on
everything.
