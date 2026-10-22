We want to train the small LLM `{model}` to excel at {benchmark}.
Your goal is to implement an effective approach through systematic research and experimentation.

## Objective

Post-train `{model}` through SFT iteration{#harness}, and improve this run's editable harness (`harness.md`) alongside it,{/harness} to maximize validation performance on {benchmark}, while preserving a clean, auditable data and training record.{#harness} The deliverable is a model+harness pair, and the two are selected together.{/harness}

- Your model is scored by a grading service which you submit to over HTTP; see `grading.md` in the current directory. There is no evaluation script here.
- Deliver the selected candidate: copy its model to `final_model/`{#harness} and its matching `harness/`{/harness} at the workspace root, and write `final_candidate.json` with the chosen `candidate_id`. When you are done, print a concise final summary of data used, SFT settings,{#harness} harness changes,{/harness} validation submissions, and why this final candidate was selected.
- Time budget: `{num_hours}` hours; `bash timer.sh` is the only thing that reports how much of it is left, and it is worth checking before committing to anything long and again before deciding you are finished. When the budget ends your session is terminated immediately, even mid-poll. What you leave in `final_model` is then scored by an official evaluation you do not control or observe -- that is the number that counts. Plan your last validation to finish before the deadline, or accept that its result lands after you are gone.

## Information on the Setup

- Read `environment.md` in this directory before anything else. It is generated for this run and names the cards, every path that matters, the HuggingFace cache, the training entrypoint and each knob that entrypoint reads, along with what validation costs and how decoding is decided. Reading it first is faster than deriving any of it.
- Then read every other document in this directory before acting: `agent_rules.md` states the integrity boundary every action answers to, `grading.md` is the scoring API, and the remaining documents carry this benchmark's contract (evaluation protocol{#harness}, harness rules{/harness}, data and training guides).
{gpu_info}

## Data

- Treat `train_resource_manifest.lock.json` as the complete list of allowed training resources. Do not download, clone, scrape, or search for additional datasets.
- Hugging Face resources in the manifest are local cache declarations only. Read their `resource_config.json` files, use only the named datasets that are already available in the mounted HF cache, and write any derived training files in the current workspace.
- You may transform, filter, rebalance, merge, synthesize, augment, distill, or self-distill training data derived from the manifest resources, as long as their provenance is traceable back to manifest resources.

## Training

- You may modify SFT hyperparameters, data-mixing choices, sequence length, batching, learning rate, epoch count, checkpoint selection, and other Slime/environment-supported training settings. You may also edit the training framework itself: job-local copies of Slime and Megatron-LM live at `framework_src/slime` and `framework_src/Megatron-LM` in this workspace (seeded from the trainer image when the container starts, and the entrypoint imports these copies via `SLIME_DIR`/`MEGATRON_DIR` -- editing the image's `/root/` copies changes nothing it runs), you run as root, and changing the loss function or the training loop is a legitimate move rather than a workaround. Two conditions apply to **every** training launch, whether or not you edited the framework:
  1. Launch through the script named by `entrypoint_hint` in `train_resource_manifest.lock.json` -- `train_resources/scripts/slime_sft/run_train.sh` for a Slime SFT recipe. It is what records each framework's commit and full diff to `logs/framework_state_<RUN_NAME>.txt` and tees job output, including per-step loss, to `logs/train_<RUN_NAME>.log`, where `<RUN_NAME>` is the run name you set. Calling `ray job submit` or `train_async.py` directly writes neither file, and a run launched that way cannot be audited afterwards.
  2. Only the current workspace survives the run; the rest of the container filesystem is discarded when the run ends. Anything you want kept -- including a framework patch you consider part of the result -- must be written into this workspace.

{#harness}
## Harness

- This run ships an editable copy of the evaluation's agent scaffold in `harness/`; `harness.md` states its contract. It is yours to improve: instruction framing, config defaults, stopping behavior, trajectory handling. The scaffold is part of the score: a better harness lifts every model you train, and a better model may deserve different scaffolding -- iterate on them together.
- The contract in `harness.md` is hard: keep the declared import path loadable and the agent class compatible, edit only the allowlisted files, and never put task material or answers into the harness.
- A submission is a pair: `candidates/<candidate_id>/final_model/` plus its sibling `candidates/<candidate_id>/harness/`, with `candidate.json` (non-empty `candidate_id`) and your `candidate.md` notes beside them. Submit the `final_model/` path; the service pairs the sibling harness itself. The pair is immutable once submitted -- a new idea means a new candidate id.
{/harness}

## Validation & feedback

- You start with the base model's validation already run for you. `$HOME/grading_baseline/` holds its score and the full transcript of every graded validation sample, so the reference point for "did my training change anything" exists before your first submission. Read it before you design your first data recipe.
- Validation is in-distribution and service-controlled. The grading service exposes at most its configured validation split. When you want a faster signal, include a positive `limit` in the submission JSON, but that is only a cap; it cannot expand the validation set or choose different benchmark samples. Once the split's membership is established by the service (`/health` reports this under `named_samples_available`), you may also name particular samples with `sample_ids` in the submission JSON instead of paying for the first N again -- use it to re-test the samples that discriminate between candidates, e.g. the ones your last version newly solved or newly failed.
- Each successful validation writes the complete trial tree, including full agent transcripts, to `$HOME/grading_results/<job_id>/`. The agent-facing result does not expose the service's host-side path. These files are the primary feedback channel -- a median transcript here exceeds a megabyte, so read them on disk rather than expecting them in an HTTP response. `grading.md` describes the layout.
- You may interrupt a validation that is no longer worth its GPU time: `POST /cancel` with a reason releases the cards within about half a minute and still hands over every sample that had finished, in the same directory layout. Cancellations are capped per run and each one is recorded with its reason. A cancelled validation's score is deliberately not comparable with a full one -- it covers the prefix of the evaluation order that finished, which is biased towards the samples the model got through quickly, so it reads high. Treat it as diagnosis, never as a candidate's score.
- Before each validation submission, check `$GRADING_URL/health`. If `gpu_coordination.mode` is `shared`, stop training or serving GPU processes so validation can use the shared GPU pool. While validation is queued or running, do CPU-only log inspection, data checks, or planning.

- You may run multiple SFT iterations, compare variants, and submit validation candidates while time remains. Keep concise notes of each candidate's data recipe, training parameters,{#harness} harness change,{/harness} validation result, and selection rationale.{#harness} Validation scores a pair -- the model you submit and the harness beside it.{/harness} Compare candidates only when they differ in the dimension you meant to test.

{setup_other}

## Rules

`agent_rules.md` in this directory states the integrity boundary, time and GPU
discipline, candidate discipline, and delivery procedure. It applies in full.

Remember: NEVER ask the user for feedback. Just execute actions which make most sense to you. We will evaluate your results on {benchmark} once you are done.
