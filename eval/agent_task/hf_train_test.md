We are running a bounded HF-data train-test of the harness, not an open-ended research run. The purpose is to verify that you can use the Hugging Face dataset described by the provided training resource config, materialize a tiny SFT file, run one training pass, and deliver a loadable model.

Read `environment.md` in this directory first. It is generated for this run and names the cards, every path that matters, and `timer.sh` for the time you have left.

Do exactly this and keep the run easy to audit.

1. Inspect `train_resource_manifest.lock.json`. It lists the provided scripts, model checkpoints, and data resource configs under `train_resources/`. Treat this manifest as the complete task input. Do not search for other datasets, do not clone repositories for data, and do not use benchmark test data.

2. Find the single `data/...` resource in the manifest and read its `resource_config.json`. Use only the Hugging Face dataset named there. For this smoke test, use the SWE training subset from that config (`swe/train`, or `subset=swe` and `split=train` if you load it with the datasets API), take the first 64 rows from that split, and materialize them into `hf_train_first64.jsonl` in the current directory.

3. Materialize `hf_train_first64.jsonl` as JSONL suitable for the provided Slime SFT entrypoint. Each line must contain a `messages` field. If an HF row already has `messages`, preserve it. Preserve `tools` and useful metadata when present, but the SFT script consumes `messages` via `--input-key messages`.

4. Start from the copied checkpoint for base model `{model}` when the manifest includes one. Use the manifest `entrypoint_hint` unless it is clearly broken. Use the manifest `env_defaults`; they are configured for one epoch and for `TRAIN_DATA=hf_train_first64.jsonl`. Run exactly one SFT job on the materialized 64 rows. Do not run broad experiments and do not run multiple training variants.

5. Write the deliverable model to a directory named `final_model` in the current directory. The model must be loadable by the existing serving stack.

6. Submit `final_model` exactly once for a validation score through the grading service described in `grading.md`. Before submitting, check `$GRADING_URL/health`: if `gpu_coordination.mode` is `shared`, stop training or serving GPU processes so validation can use the shared GPU pool. Wait for the result and print it; while validation is queued or waiting, do only CPU-side log inspection or file checks.

7. Stop. Leave `final_model` in place -- it will be scored again after you exit.

{#harness}
This run ships an editable harness (`harness.md` in this directory). Leave it
unmodified, and adapt steps 5-7 to the pair contract: put the trained model in
`candidates/<candidate_id>/final_model/` with the unmodified `harness/` copied beside
it, plus `candidate.json` and `candidate.md` as `environment.md` (## Candidates)
describes; submit that candidate's `final_model/` (a bare root `final_model/` is not a
submittable candidate here); and deliver by copying the pair to the workspace root with
`final_candidate.json` written beside them.
{/harness}

Do not modify `templates/` or the inner agent harness/scoring code. Do not switch to a different base model. Do not use external data other than the HF dataset explicitly named in the provided resource config. Any deviation makes this test useless because a failure would then be ambiguous between the harness and your changes.

{setup_other}
