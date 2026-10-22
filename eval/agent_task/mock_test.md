We are smoke-testing the harness, not training anything. Do exactly this and nothing else.

Read `environment.md` in this directory first. It is generated for this run and names the cards, every path that matters, and `timer.sh` for the time you have left.

1. Materialize {model} into a directory named `final_model` from the prepared offline cache.
   Resolve `snapshot_download` with
   `revision="68c46c4b3498877f3ef123c856ecfde50c39f404"` and without `local_dir`, then copy
   the snapshot into `final_model` with symlinks dereferenced. Do not fine-tune or modify it.

2. Submit it once for a validation score, so that the mid-run scoring path gets
   exercised. Read grading.md for how; it is the same service a real run uses. Wait for
   the result and print it.

3. Stop. Leave final_model in place -- it will be scored again in full after you exit.

{#harness}
This run ships an editable harness (`harness.md` in this directory). Leave it
unmodified, and adapt steps 1-3 to the pair contract: put the downloaded model in
`candidates/smoke/final_model/` with the unmodified `harness/` copied beside it, plus
`candidate.json` and `candidate.md` as `environment.md` (## Candidates) describes;
submit `candidates/smoke/final_model` (a bare root `final_model/` is not a submittable
candidate here); and deliver by copying the pair to the workspace root with
`final_candidate.json` written beside them.
{/harness}

Do not train, do not experiment, do not iterate. Any deviation makes this test useless,
because a failure would then be ambiguous between the harness and your changes.

{setup_other}
