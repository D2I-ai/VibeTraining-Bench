# Scoring your model

Scoring runs on a service, not in your container. You submit a model over HTTP and poll for the result. There is no evaluation script in your working directory, and there is nothing to install.

Before submitting, you may inspect service status:

```bash
curl -s "$GRADING_URL/health"
```

If `gpu_coordination.mode` is `shared`, training and grading may use the same GPU pool: stop any training or model-serving GPU processes before validation, then do only CPU-side log inspection, file checks, or planning while the validation job waits/runs. If `gpu_coordination.mode` is `isolated`, treat validation as an ordinary queued service job and do not change your GPU behavior based on queueing alone.

Your token is in `grading_token` in this directory. Keep it there; it identifies your run and confines submissions to your own workspace.

## Submit

```bash
curl -s -X POST "$GRADING_URL/submit" \
  -H 'Content-Type: application/json' \
  -d "{\"token\": \"$(cat grading_token)\", \"model_path\": \"output/v1\"}"
```

```json
{"job_id": "a1b2c3d4e5f60718", "queue_position": 1, "cached": false}
```

`model_path` is **relative to this directory**. Use `output/v1`, not `/home/agent/task/output/v1` -- the service resolves paths on the host filesystem, where your workspace lives somewhere else entirely, so a container-absolute path is rejected as escaping your sandbox.

`model_path` is required. You may also send a positive `limit` to request a cheaper validation signal; the service treats it as a cap and will never evaluate more than the operator-configured validation split. `mode` and `write_metrics` are operator-only and are rejected if you send them.

Submit only a model you have finished writing. The service reads the weights directly off disk, and a half-written checkpoint fails in confusing ways.

Run the submit on its own and keep the `job_id`: it is what you poll with, and a submit whose reply you did not read is how the same model gets queued twice.

## Poll

Use `printf`, never `echo`, when piping a JSON response into another command: this shell is zsh, and `echo "$var"` silently interprets `\n` / `\t` inside the string, corrupting JSON that contains them. `printf '%s'` passes the bytes unchanged.

```bash
R=$(curl -s "$GRADING_URL/result?token=$(cat grading_token)&job_id=a1b2c3d4e5f60718")
printf '%s' "$R" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))"
```

While it runs:

```json
{"job_id": "a1b2c3d4e5f60718", "state": "running",
 "progress": {"phase_label": "generating answers", "samples_finished": 29,
              "samples_outstanding": 1, "samples_unstarted": 0, "target_samples": 30,
              "percent": 96.7, "eta_seconds": 2460, "eta_basis": "observed_sample_rate",
              "eta_from_runs": null, "baseline_sample_seconds": 592,
              "sample_input_tokens": {"median": 3279433, "p90": 55823104,
                                      "max": 101620652, "counted": 29},
              "sample_errors": {"AgentTimeoutError": 9, "NonZeroAgentExitCodeError": 3},
              "outstanding_samples": [{"instance_id": "django__django-16938",
                                       "seconds_running": 2460, "turns": 1359,
                                       "no_progress_repeats": 323, "stalled": true}],
              "note": "29 of 30 samples are graded and 1 is still running ..."}}
```

Read `samples_finished`, `samples_unstarted` and `outstanding_samples` before concluding an evaluation is stuck. They are counted from the per-sample result files the harness writes as it goes, so they are ground truth, and they separate three states that look identical from outside: no progress at all, still working through samples that have not started, and waiting on a long tail while the rest are graded. Only the third is a tail, and `samples_unstarted` is what tells it from the second.

Even when `samples_finished == target_samples`, finalization may still be running, so `result` can remain absent temporarily. That is not an empty result: keep polling the same `job_id` until `state` is terminal (`done`, `failed` or `cancelled`). After cancellation, the trajectory directory may appear shortly after the terminal response; retry the path rather than resubmitting.

`eta_basis` says where the estimate came from, and they are not equally trustworthy:

- `observed_sample_rate` -- measured from this evaluation. Use it.
- `historical_median` / `historical_median_other_limit` -- **your own** previous submissions in this run. `eta_from_runs` says how many; below three the estimate carries a caveat, and one record is an anecdote, not a forecast.
- `baseline_sample_rate` -- the cold start, before anything of yours has been measured. It scales the base model's per-sample cost to your sample count, so it assumes your model is no slower than the base. A model that loops will overrun it.

No basis knows which samples are slow, so treat an ETA as a lower bound while `outstanding_samples` is non-empty. Nothing here is shared between runs: another run's timings never become your estimate.

`baseline_sample_seconds` is what one sample cost the base model, and it is the yardstick for your own rate. Divide your elapsed time by `samples_finished` and compare: several times the baseline means your model is spending that time somewhere.

`completions_observed` counts LLM calls, not samples. An agentic benchmark spends many calls per sample, so it runs far ahead of real progress -- and on harnesses that write no per-request log lines it stays at 0 for the whole evaluation. Never read 0 there as "nothing is happening"; `samples_finished` is the signal.

`sample_input_tokens` separates the two reasons an evaluation gets slow. A trial that loops is not idle -- it re-sends its whole transcript every step -- so timing does not catch it and token spend does. Read the shape:

- **The median moved.** Every sample got more expensive.
- **The median held, `p90` and `max` ran away.** A few trials are looping while the rest are normal. This is what makes a long tail.

The baseline disclosure carries `agent_result.n_input_tokens` per sample, so the same numbers can be computed for the baseline and compared. On this benchmark a healthy queue had `p90` about 4x its median against 18x for a run where a third of the trials looped -- while the medians differed only 2x.

`sample_errors` counts how the failed trials failed, as they land. A trial that repeats one command until its turns run out ends as a timeout, so timeouts climbing while the score does not is a loop showing itself while there is still time to react. A sample that was simply graded 0 is not counted here -- that is a hard sample, not a broken one. The finished result reports the same counts as `trial_errors`.

When it finishes with scalar-score-only disclosure:

```json
{"job_id": "a1b2c3d4e5f60718", "state": "done",
 "result": {"ok": true, "validation_score": 0.42}}
```

When the operator enables richer disclosure, `result.per_sample_outcomes` carries one row for **every** sample the evaluation graded, with the counts that explain each one:

```json
{"job_id": "a1b2c3d4e5f60718", "state": "done",
 "result": {"ok": true, "validation_score": 0.42,
            "per_sample_outcomes": [{"id": "task-id", "score": 0.0, "partial": 0.5,
                                     "ungraded_reason": null,
                                     "agent_stats": {"api_calls": 1363, "turns": 2711,
                                                     "exit_status": "Submitted"}}]}}
```

The `per_sample_outcomes` disclosure level exposes each task's `id`, `score`, `partial`, `correct`, `ungraded_reason` and `agent_stats`. `ungraded_reason` is normally null for graded tasks, including graded failures with score 0; it explains only why a valid score could not be produced. `agent_stats` is how the trial was spent, not what was in it: model calls, turns, the harness's own exit status and token counts. A task solved in 19 calls and one that failed after 1363 are different failures, and this is where that shows.

The rows are not truncated. Every sample that was graded is returned; the disclosure level decides *whether* they come back, not how many.

## Which job is mine

```bash
curl -s "$GRADING_URL/queue?token=$(cat grading_token)"
```

```json
{"running": {"job_id": "a1b2c3d4e5f60718", "model_path": "output/v3", "limit": 30,
             "state": "running", "started_at": 1788264709.1},
 "pending_count": 1, "own_pending": [], "own_pending_count": 0}
```

With your token, a job of yours is described in full; another run's job is reduced to its task and state. This is how to recover a `job_id` you no longer have -- if a submission's output scrolled away or was swallowed by a compound command, ask here before submitting again. Two submissions of the same model both run, both cost queue time, and the second one does not replace the first.

`own_pending` lists your queued jobs, and `running` tells you whether the job currently holding the cards is yours. An evaluation of yours that is already running is the reason the queue is not moving; cancelling something else will not help.

## Reading the full results

A successful validation, or a cancelled validation that produced samples, writes its readable record to this deterministic workspace path:

```
$HOME/grading_results/<job_id>/
├── summary.json          which samples are here, and what was withheld
└── samples/<task_id>/
    ├── agent/            the full transcript -- the whole thing, not an excerpt
    ├── result.json       the harness's own record of the trial
    ├── config.json  trial.log  artifacts/
    └── verifier/reward.txt
```

Use the `job_id` returned by `/submit`; the agent-facing `/result` response intentionally does not expose the service's host-side `results_dir`. This directory is created only for your validation submissions, not for the final official evaluation after your session.

This is the primary channel, not a convenience. A median transcript on this benchmark is over a megabyte and the longest are five, so anything returned inline has to be cut -- and the part worth reading, where a trajectory stops converging, is at the end. On disk there is nothing to cut: grep it, tail it, load the JSON and count turns.

The directory is read-only. Two files per sample are never copied there: `verifier/report.json` and `verifier/test-stdout.txt` name the hidden tests, which is the grading answer.

Only validation samples appear, and only ones actually graded. When the operator has imported a baseline validation, `$HOME/grading_baseline/` holds the same layout for the **base model** on the whole split -- the reference point for whether your training changed anything, available before your first submission.

Whether that baseline and your delivery were decoded identically is benchmark-defined:
`environment.md` (## Decoding) states this run's rule. When the benchmark pins sampling
at the service, a `generation_config.json` shipped in your delivery does not change
scoring requests.

## Reading one sample over HTTP

When a file read is inconvenient, or the sample is still running and so has no files yet, one sample can be fetched directly:

```bash
curl -s "$GRADING_URL/sample?token=$(cat grading_token)&job_id=a1b2c3d4e5f60718&sample_id=task-id"
```

```json
{"job_id": "a1b2c3d4e5f60718", "sample_id": "task-id", "running": false,
 "problem_statement": "...", "trajectory_file": "trajectory.json",
 "trajectory_excerpt": "last 12 of 593 steps, each field clipped to 500 chars\n--- step 582\naction: ...\noutput: ...",
 "agent_stats": {"api_calls": 1363, "turns": 2711, "no_progress_repeats": 916, "exit_status": "Submitted"}}
```

Any sample that job has run can be asked about, including one it is still running -- `running` says which, and for a running sample the transcript is whatever it has written so far. Anything else is a 404. A 403 means this service is configured not to disclose per-sample detail at all, in which case the finished-job directory above is your only route.

A sample that has only just started has written nothing yet, and both transcript fields come back `null`; one caught mid-write is returned with a note saying so. Neither is an error -- ask again rather than concluding the sample is broken.

`no_progress_repeats` counts actions this trial repeated without getting anywhere, and it is available while the sample is still running: a high count on a sample that has been going for twenty minutes is a loop, a low one is slow honest work, and the two call for opposite decisions. `exit_status` and the token counts only appear once the trial has finished.

You do not need this endpoint for that count alone. `progress.outstanding_samples` in a job's result already carries `turns`, `no_progress_repeats` and `stalled` for the longest-running samples, so the poll you are doing anyway answers "is anything looping". Come here when the answer is yes and you want to see *what* it is repeating.

`trajectory_excerpt` is the **last few steps**, each action and output clipped, not a slice of the file. Counted in steps because a trial that has stopped getting anywhere emits huge outputs, so the last N characters of its transcript land inside one tool result and show no complete step -- the case you are asking about is the case that would return nothing. What you get instead is the commands, which is where a repeat is visible. Pair it with `progress.outstanding_samples`, which names the ids running longest: those are the ones worth asking about.

## Evaluating particular samples

A `job_id` identifies a submission for polling; it does not select test cases. To re-test particular validation cases, send their ids as `sample_ids` in a new submission.

Once one submission has covered the whole validation split, the service knows which samples it contains, and later submissions may name some of them instead of taking the first N:

```bash
curl -s -X POST "$GRADING_URL/submit" -H 'Content-Type: application/json' \
  -d "{\"token\": \"$(cat grading_token)\", \"model_path\": \"output/v2\",
       \"sample_ids\": [\"task-id\", \"other-task-id\"]}"
```

This is what makes the samples at the back of a long run reachable: a `limit` always takes the split's first N, so without naming them the interesting ones are only ever evaluated by paying for everything in front of them. The first N are counted in the dataset's own order; id lists you read elsewhere (the baseline summary's `sample_ids`, per-sample outcomes) are alphabetical for reading and are not that order.

A named-sample result is marked `"subset": true, "comparable_to_full": false`. It is a diagnosis, not a score: the samples were chosen, so the number is not the split's number and two of them are not comparable with each other. Named ids outside the validation split are refused, and so is naming samples before the split's membership is known.

## Cancelling an evaluation

In `shared` GPU mode, an evaluation you no longer want is holding cards you could be training on. Ask for them back rather than working around it:

```bash
curl -s -X POST "$GRADING_URL/cancel" -H 'Content-Type: application/json' \
  -d "{\"token\": \"$(cat grading_token)\", \"job_id\": \"a1b2c3d4e5f60718\",
       \"reason\": \"v2 finished training; want the cards for the v3 run\"}"
```

The call returns once the evaluation is torn down and the cards report free memory, so when it answers you may start using them. `reason` is required and is kept with the run's record.

What had already been graded is kept and returned as `result.partial`, with the ids in `instance_ids_graded`. It is marked `comparable_to_full: false` and it means it: evaluation follows the dataset order, so the trials that finished first are the ones your model got through quickly, and any prefix reads higher than the whole split would. Use it to diagnose, never to compare two candidates or to choose what to deliver -- it carries no score for that purpose.

The trials that were still running are named in `instance_ids_ungraded`, and their transcripts are kept too. They scored nothing at all -- not zero -- so they are in no average, but they are usually the reason you cancelled: a sample killed at twenty minutes with a high `no_progress_repeats` is the evidence that the decision was right. `summary.json` lists them under `unfinished_sample_ids`, separately from the graded ids, and both the results directory and `/sample` will answer for them.

How much that bias costs you depends on how much was covered, so read `samples_finished` before deciding. At 2 of 10 graded a cancellation keeps only the easiest fraction; at 29 of 30, with the last sample forty minutes in, it keeps a diagnosis worth nearly as much as the full run.

The graded ids compose with named-sample submission: whatever a cancellation left unfinished can be evaluated on its own later, and samples you already know to be slow can be left out of a quick iteration.

Cancellations are counted per run and capped; `/health` reports the cap and `/submit` and `/cancel` report what you have left.

`state` is `queued`, `waiting_for_gpu`, `running`, `done`, `failed` or `cancelled`. Evaluations are serialised across all runs. In `isolated` GPU coordination mode, queueing is just service scheduling. In `shared` mode, a `resource_advice.kind` of `release_gpu` means validation is waiting for GPU memory: stop training/serving GPU processes and do CPU-only work while polling.

## What this service is not

It is not a cheap way to test every idea. A submission runs the service validation split unless you pass a smaller positive `limit`, and even a capped validation request can take minutes to tens of minutes depending on the benchmark and on how much your model generates. Use smaller caps for fast signals and reserve larger/full validation runs for promising candidates.

Read `eval_protocol.md` for the exact prompt format, answer format and scorer your model will be judged by. Matching it in your own quick tests is what makes them predictive.

## Errors

Failures come back as HTTP 4xx with an explanation:

```json
{"error": "model_path escapes the run sandbox"}
```

The message says what is wrong. Read it rather than retrying the same call.
