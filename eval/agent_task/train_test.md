We are running a bounded train-test of the harness, not an open-ended research run. The environment may not have real internet access. Do exactly this and keep the run easy to audit.

Read `environment.md` in this directory first. It is generated for this run and names the cards, every path that matters, and `timer.sh` for the time you have left.

1. Inspect `train_resource_manifest.lock.json`. It lists the scripts, model checkpoints, and data copied into this workspace under `train_resources/`. Use the manifest's copied data resources as the complete training-data candidates; if multiple data files are provided, inspect them and choose or combine them explicitly for the single SFT job.

2. Start from the copied checkpoint for base model `{model}` when the manifest includes one. Use only the provided training resources to run exactly one SFT training job. If the manifest has an `entrypoint_hint`, prefer that script unless it is clearly broken. If the manifest has `env_defaults`, use those as the initial environment knobs for the entrypoint unless they are clearly incompatible with the visible hardware. Do not download or search for additional data, do not clone repositories for data, do not fetch unrelated models or datasets, and do not switch to a different base model.

3. Write the deliverable model to a directory named `final_model` in the current directory. The model must be loadable by the existing serving stack.

4. Submit `final_model` exactly once for a validation score through the grading service described in `grading.md`. Before submitting, check `$GRADING_URL/health`: if `gpu_coordination.mode` is `shared`, stop training or serving GPU processes so validation can use the shared GPU pool. Wait for the result and print it; while validation is queued or waiting, do only CPU-side log inspection or file checks.

5. Stop. Leave `final_model` in place -- it will be scored again in full after you exit.

{#harness}
This run ships an editable harness (`harness.md` in this directory). Leave it
unmodified, and adapt steps 3-5 to the pair contract: put the trained model in
`candidates/<candidate_id>/final_model/` with the unmodified `harness/` copied beside
it, plus `candidate.json` and `candidate.md` as `environment.md` (## Candidates)
describes; submit that candidate's `final_model/` (a bare root `final_model/` is not a
submittable candidate here); and deliver by copying the pair to the workspace root with
`final_candidate.json` written beside them.
{/harness}

Do not run broad experiments, do not iterate indefinitely, do not modify `templates/` or the inner agent harness/scoring code, do not use external data, and do not train on benchmark test data. Any deviation makes this test useless because a failure would then be ambiguous between the harness and your changes.

{setup_other}
