# How your model will be scored on τ³-bench

This is the evaluation contract, read off the harness itself. You cannot run the harness
— scoring happens on a separate service that holds the tests. What follows is everything
that decides your score, so that you can train against the real target rather than a
guess.

τ³-bench is a **conversational** benchmark, not a coding one. Each task simulates a
customer-service interaction: your model plays an agent that must satisfy a user's request
by using a set of domain tools, while following a written policy.

Read the section "How your model is driven" below before you plan any training. The registered harness follows the τ³ Harbor adapter when `VTB_HARBOR_REPO` is available.

## What one task looks like

375 tasks across four domains — airline, retail, telecom and banking. Each task has:

- A **domain policy** your model must follow: what it is allowed to do, what it must
  refuse, what it must confirm before acting.
- A **simulated user** with a goal, played by another model. The user does not state
  everything up front. Your model has to ask.
- A set of **domain tools** (look up a booking, issue a refund, change a plan) with a
  database behind them.
- A **target end state**: which database writes should have happened, and what the user
  should have been told.

Difficulty comes from the interaction, not from algorithms: gathering the right
information, refusing what the policy forbids, and making exactly the right set of writes.

## What counts as correct

Per task, scoring combines:

- **Database state.** Did the right writes happen, and no wrong ones? An extra refund or a
  missing cancellation fails the task.
- **Communication.** Was the user actually told what they needed to know?

The reported `accuracy` is the fraction of tasks scored as correct.

The characteristic failure modes are specific to this benchmark: acting without asking,
asking forever without acting, and doing something the policy forbids because the user
insisted.

## How your model is driven

The registration follows the Harbor adapter metadata for τ³-bench. With `VTB_HARBOR_REPO` pointing at the Harbor checkout, the selected harness is the adapter-local `tau3-llm-agent`, which talks to the task's `tau3-runtime` MCP server.

1. The scaffold starts or resumes the simulated conversation.
2. Your model sends either a user-facing message or a domain tool call.
3. The runtime applies the message/tool call and returns the next user/runtime state.
4. The loop continues until the conversation is completed or the step/error limit is reached.

Your model must emit valid OpenAI-compatible tool calls when a tool action is needed. A reply in prose is only a user-facing message; it is not a database action.

## Decoding at scoring time

Your model is served over an OpenAI-compatible endpoint. The harness does **not** specify
temperature or top_p, so they come from the `generation_config.json` you ship inside
`final_model`. If it omits `temperature`, the effective value is 1.0.

The endpoint is started with tool calling enabled (`--enable-auto-tool-choice`) and a
parser matched to the chat template. If you change the chat template in `final_model`,
keep the tool-call format it emits — a template whose tool calls the server cannot parse
turns every request into a failure.

## Scoring yourself

Submit to the grading service (see `grading.md`). It runs the full validation split, so it
is not a fast inner loop.

The most useful thing you can build locally is a check of **tool-call validity, policy following, and multi-turn coherence** under a conversational tool-use loop.

Do not train on this benchmark's tasks. That is contamination, and it is checked.
