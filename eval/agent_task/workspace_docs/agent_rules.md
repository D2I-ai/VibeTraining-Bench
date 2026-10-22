# Agent Rules

These rules apply to every data, training, harness, validation, and delivery action in
this workspace. They exist to keep the experiment's results attributable. Benchmark- and
mode-specific documents (for example `eval_protocol.md`, `harness.md`, or a workflow
document supplied by the benchmark) refine them; they never override them.

## Integrity boundary

- Never turn benchmark or validation material into training data: no task text, verifier
  output, solutions, patches, reference answers, or validation trajectories as training
  records, prompts, labels, rules, or task-specific heuristics. Validation scores and
  your own candidates' trajectories are diagnostic material only.
- `train_resource_manifest.lock.json` is the complete allowlist of training resources.
  Do not download, clone, scrape, or call external APIs to obtain additional data.
  Derived data is fine when its provenance traces back to manifest resources.
- Submit for scoring only through the grading service described in `grading.md`. Do not
  modify the workspace-level `templates/`, the scoring code, `grading_token`, or the
  grading service. The `harness/` package, when present, is the exception: it is
  editable under `harness.md`.
- Train only the base model named by this run. Do not switch to, or distill from, an
  instruction-tuned variant of it.
- Keep all work inside this workspace and submit only workspace-relative paths. Only the
  workspace survives the run; the rest of the container filesystem is discarded.

## Time and GPU discipline

- `bash timer.sh` reports the remaining budget. Check it at startup, before committing
  to anything long, and again before delivery. When the budget ends the session is
  terminated immediately, even mid-poll.
- Before every validation submission, check `$GRADING_URL/health`. If
  `gpu_coordination.mode` is `shared`, stop training or serving GPU processes before
  submitting. While a validation is queued or running, do CPU-only inspection, data
  checks, or planning.

## Candidate discipline

- When you validate iterations, snapshot each as an immutable candidate:
  `candidates/<candidate_id>/final_model/`, plus its sibling `harness/` when the run
  ships an editable harness, with `candidate.json` (non-empty `candidate_id`) and your
  `candidate.md` notes beside them. Submit the `final_model/` path; the service pairs
  the sibling harness itself. A submitted pair is immutable -- a new idea means a new
  candidate id.
- In `candidate.md`, record the data recipe, training settings, harness change (if any),
  validation result, and selection rationale.

## Delivery

- Deliver exactly one selected candidate, as `environment.md` (## Candidates) describes.
- Leave every artifact in place; do not delete candidates, logs, or validation output.
