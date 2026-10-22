We are running a bounded connectivity test of the model+harness experiment loop, not an open-ended research run. The chain under test: edit the editable harness once, pair it with the base model, validate the pair on a few development samples, and deliver the pair. **Do not train anything** -- this run ships the base model unchanged; the mechanism under test is the harness pairing, and training would only add hours.

Read `environment.md` first for this run's facts, then `harness.md` and `grading.md`. `harness.md` defines the editable harness contract, which applies in full here. `grading.md` is the HTTP API reference for the grading service. `agent_rules.md` applies in full as well.

1. Confirm the workspace is whole. `train_resource_manifest.lock.json` lists the resources; `harness/` holds the editable harness package; the base model checkpoint is under `train_resources/models/`.

2. Prepare the deliverable model without training: copy the base model checkpoint from the model resource to `final_model/` in the workspace root. Do not run the training entrypoint.

3. Edit the harness once, so its effect is observable in the validation trajectories. Make one small, general, content-harmless change at the deepest level `harness.md` documents: the agent loop itself when it describes a loop-level edit path, otherwise the instruction or config surface it names. The change's fingerprint must show up in what the model is asked -- for example a marker sentence carried into every task's first message. Record the changed path.

4. Validate the pair once, cheaply. Snapshot `final_model/` + `harness/` as `candidates/<candidate_id>/` with `candidate.json` (non-empty `candidate_id`) and `candidate.md`, then submit `candidates/<candidate_id>/final_model` through `grading.md` with a small positive `limit` (5 is plenty; the mechanism under test is the pairing, not the score). The service is expected to pick up the sibling `harness/` automatically. Poll to a terminal state, record the job id and score in `candidate.md`, then open the returned trajectories under `$HOME/grading_results/<job_id>/` and confirm your marker appears in the first message of each graded sample. If the marker is absent, the pairing did not happen: record that and stop.

5. Deliver the pair. Copy the validated candidate's `final_model/` and `harness/` to the workspace root and write `final_candidate.json` with the chosen `candidate_id`. The official evaluation afterwards runs exactly this pair.

6. Stop. Leave every artifact in place and print a concise summary: the harness change (file + marker), the validation job id and score, where the marker was sighted, and the delivered candidate id.

Reminders: the only model delivered is the unmodified `{model}` checkpoint. The point of the pairing is that future runs co-evolve the model and the harness, so treat the pair -- not the model alone -- as the object under test. Any deviation makes this test useless, because a failure would then be ambiguous between the harness and your changes.

{gpu_info}

Time budget: `{num_hours}` hours; `bash timer.sh` reports what is left.

{setup_other}
