# How your model will be scored on SWE-bench Verified

This is the evaluation contract, read off the harness itself. You cannot run the harness
— scoring happens on a separate service that holds the tests. What follows is everything
that decides your score, so that you can train against the real target rather than a
guess.

SWE-bench Verified is not a question-answering benchmark. There is no prompt to complete
and no string to match. Each task is a real GitHub issue in a real Python repository, and
your model has to produce a working code change.

## What one task looks like

500 tasks, human-validated as solvable, drawn from twelve large Python projects
(matplotlib, django, sympy, scikit-learn, sphinx and others). For each one:

- A container holds the repository checked out at the commit just before the fix.
- `instruction.md` is the **issue text as it was actually filed** — a bug report with a
  summary, a reproduction snippet, expected versus actual behaviour, sometimes a
  traceback. It is not a specification. It may be vague, may contain a wrong diagnosis,
  and does not say which files to change.
- A hidden test suite exists that your model never sees. Some of its tests fail at this
  commit and must pass afterwards (**fail-to-pass**); the rest already pass and must keep
  passing (**pass-to-pass**).

Typical stated difficulty is "15 min – 1 hour" for a human familiar with the codebase.
The category is debugging.

The container has outbound network.

## How your model is driven

Your model does not receive the task as one prompt and answer once. It is put in a loop
by **mini-swe-agent**, a scaffold installed inside the task container (shipped as an
editable package in runs that enable it; `harness.md` defines it when present):

1. The scaffold sends the issue text plus the conversation so far.
2. Your model replies with **a tool call** requesting one shell command.
3. The scaffold runs it, appends the output, and goes back to step 1.
4. This continues until your model issues the submission command, or the trial dies on
   its own (an exhausted context window or an unrecoverable error). No step limit is
   configured.

**Four consequences for training:**

**Your model must emit valid tool calls.** The scaffold drives the model through the
OpenAI tool-calling interface — `tools=[...]`, `tool_choice="auto"`, and it parses
`message.tool_calls`. A reply that describes a command in prose, or wraps it in a markdown
fence, is not an action: the scaffold sees no tool call and the turn is wasted. If your
base model is weak at structured tool calls, that alone can cost every task. This is the
single highest-leverage thing to train.

**Your model has to find the code first.** The issue does not name the file. A model that
cannot navigate an unfamiliar 100k-line repository with `grep` and `find` never reaches
the part it might be good at. This is most of what separates scores on this benchmark.

**Your model has to not break things.** Pass-to-pass is half the contract. A change that
fixes the issue and regresses one unrelated test scores zero, exactly the same as changing
nothing. Small models routinely lose tasks this way — by editing too broadly, or by
"fixing" a symptom in shared code.

**Your model must decide when it is done.** It ends the task by issuing the submission
command named in the scaffold's instructions (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`).
Anything short of that leaves the loop running, and with no step limit configured a model
that never submits stops only when the context window runs out -- whatever state the
repository happens to be in at that point is what gets tested.

## What counts as correct

Binary, per task: **1 only if every fail-to-pass test passes and no pass-to-pass test
regresses.** Nothing partial.

The reported `accuracy` is the fraction of tasks scored 1.

The scoring service also records the fail-to-pass and pass-to-pass counts per task, so a
model that fixed nothing is distinguishable from one that fixed the issue but broke a
neighbour. Those counts do not enter your score, but they tell you which failure mode you
have — and the two need completely different fixes.

## Being honest about the difficulty

Frontier agents score well under 100% here. A small model will start near zero, and no
prompt-format trick changes that, because real tests run against real code. Where training
actually moves the number:

1. **Tool-call reliability.** Measure it first, on your own held-out tasks. A model that
   cannot reliably emit a well-formed call never attempts anything.
2. **Repository navigation.** `grep -rn`, reading a traceback back to its source, opening
   the right file at the right line. This is learnable from SWE-style trajectory data and
   shows up immediately.
3. **Minimal diffs.** Teaching a model to change as little as possible is worth more here
   than teaching it to write clever code, because pass-to-pass punishes breadth.
4. **Actually running the tests it can see.** The repository's own visible tests are
   available in the container. A model that runs them before declaring victory catches its
   own regressions.

## Decoding at scoring time

Your model is served over an OpenAI-compatible endpoint. Every grading run -- each
validation and the final alike -- drives the scaffold with the same fixed sampling
settings (temperature `1.0`, top-p `1.0`, top-k `-1`, repetition penalty `1.0`), passed
to the harness's config layer by the grading service on every evaluation. Sampling values
in a shipped `generation_config.json` do not change these scoring requests; include that
file only when the model needs it to load. `environment.md` names the exact values;
when this run ships the editable harness, `harness.md` additionally requires your
`run()` to keep writing that config layer -- a harness that drops it is no longer being
scored under the benchmark's conditions.

The reason is that decoding is not what this benchmark measures. On a task that is a long
chain of tool calls it moves the score about as much as a training change does, and the
baseline you are compared against was measured under one particular setting. Holding every
delivery to that same setting is what keeps "better than the base model" a statement about
training rather than about sampling. So a temperature search buys you nothing here, and the
validation runs it costs are the expensive part.

The endpoint is started with tool calling enabled (`--enable-auto-tool-choice`) and a
parser matched to the chat template. If you change the chat template in `final_model`,
keep the tool-call format it emits — a template whose tool calls the server cannot parse
turns every request into a failure.

## Scoring yourself

Submit to the grading service (see `grading.md`). It runs the operator-configured validation split by default. If the service allows it, you may request a smaller positive `limit` as a faster validation cap; that cap is applied within the service-selected validation prefix and cannot expand or choose arbitrary samples.

Do not train on the Verified split's instances or their tests. That is contamination, and it is checked. For this run, the allowed training sources are exactly the resources listed in `train_resource_manifest.lock.json` plus any derived data you create from those resources; do not download or search for extra data.
