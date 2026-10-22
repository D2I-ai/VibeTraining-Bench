We are smoke-testing Toolathlon validation and final routing, not training. Read `environment.md` and `grading.md` first.

1. Materialize {model} into `final_model` from the prepared offline Hugging Face cache by resolving `snapshot_download` with `revision="68c46c4b3498877f3ef123c856ecfde50c39f404"` and without `local_dir`, then copying the snapshot with symlinks dereferenced. Do not modify it.
2. Submit `final_model` exactly once with `"limit": 1`, wait for the ToolGym validation result, and print it.
3. Stop and leave `final_model` in place for the launcher's one-sample Toolathlon Verified final.

Do not train, iterate, or submit twice. A zero score is acceptable; an infrastructure failure is not.

{setup_other}
