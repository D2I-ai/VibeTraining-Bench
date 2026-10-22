# How your model will be scored on toolathlon-gym

This is the evaluation contract, read off the harness itself. You cannot run the
harness — scoring happens on a separate service that holds the tasks and their
checks. What follows is everything that decides your score.

toolathlon-gym is an **MCP tool-use** benchmark, not a coding one. Each task asks
your model to accomplish a realistic office/ops workflow — build a spreadsheet
from YouTube transcripts and email it, reconcile orders, plan travel into Notion —
by calling tools from **multiple MCP servers** (email, excel, word, notion,
google-calendar, google-sheets, canvas-lms, woocommerce, yahoo-finance, arxiv and
others). The services are simulated: their state lives in a database and in files,
and your changes to that state are what gets checked.

## What one task looks like

12 tasks, each needing 4–7 MCP servers. For each task:

- A container holds the toolchain and a fresh copy of the simulated world
  (database + workspace files).
- A task description states the goal in prose. It is a multi-step workflow, not a
  single call: read from several sources, transform, write into several targets.
- After your model finishes, an evaluation script inspects the final state — the
  database rows and the produced files — against the expected outcome.

## How your model is driven

Your model is put in a loop by a **CAMEL-based agent** (single-turn mode): it
receives the task plus the conversation so far, replies with tool calls, the
scaffold executes them and appends the results, until it claims done or hits the
step limit (default 100 steps, lowered on retries).

**Your model must emit valid tool calls.** The scaffold drives the model through
the OpenAI tool-calling interface (`tools=[...]`, reading `message.tool_calls`).
The tool count is large — several MCP servers' tools are exposed at once — so
choosing the RIGHT tool matters as much as emitting well-formed calls. A reply in
prose is not an action.

**Your model must finish and say so.** There is a `claim_done` tool; a run that
never claims done is judged on whatever state it left behind when the steps ran
out.

## What counts as correct

Binary, per task: **1 only if the final state passes the task's checks** —
database rows written correctly AND files produced correctly. Nothing partial.

The reported `accuracy` is the fraction of tasks scored 1 **among tasks that
reached a verdict**; tasks where the infrastructure failed are reported as
`ungraded` and excluded rather than counted against you.

## Which tasks you are scored on

Two sets that do not overlap:

- **While you work**, a submission is scored on the *validation* set — all 12
  tasks of this benchmark, in manifest order. A smaller `limit` takes the first
  N of them; you cannot select or resize the set itself.
- **Your final delivery** is scored on a separate test benchmark: 88 tasks
  disjoint from these, drawn from the official held-out final pool and absent
  from your training data. They appear in no manifest or endpoint this
  benchmark exposes, and nothing is withheld from the 12 validation tasks — so
  a validation score and the final score measure different data, not different
  slices of the same data.

Tasks run **concurrently** on the scoring service, so a validation submission
takes roughly one task's duration (up to ~35 min), not the sum.

## Decoding at scoring time

Your model is served over an OpenAI-compatible endpoint. The harness does **not**
specify temperature or top_p; they come from the `generation_config.json` you ship
inside `final_model`. The endpoint is started with tool calling enabled
(`--enable-auto-tool-choice`) and a parser matched to the chat template — if you
change the template, keep the tool-call format it emits.

## Being honest about the difficulty

These are long multi-tool workflows written for frontier models. A small model
will score near zero at first. Where training actually moves the number, in
order:

1. **Tool-call validity under a large tool list.** Dozens of tools are in
   context; picking a plausible-but-wrong one wastes a step just like a malformed
   call does.
2. **Multi-step state tracking.** Read → transform → write across services; the
   model must carry intermediate results through the conversation.
3. **Knowing when it is done.** Claiming done too early scores the incomplete
   state; never claiming done scores whatever the step limit left.

Do not train on this benchmark's tasks, their databases or their evaluation
scripts. That is contamination, and it is checked.
