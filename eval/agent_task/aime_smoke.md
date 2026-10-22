We are smoke-testing the complete AIME experiment path, not training anything. Do exactly this and nothing else.

Read `environment.md` and `grading.md` first.

1. Materialize the model {model} into `final_model` in the current directory. This run is
   offline: call huggingface_hub's `snapshot_download` with
   `revision="68c46c4b3498877f3ef123c856ecfde50c39f404"` and without `local_dir` to resolve
   the prepared cache snapshot, then copy that directory into `final_model` with symlinks
   dereferenced (for example, `shutil.copytree(..., symlinks=False)`). Do not fine-tune or
   modify the model.

2. Submit `final_model` exactly once through the grading API. Include `"limit": 1` in the
   JSON request. Wait for validation to reach a terminal state and print the result. The
   service routes this validation to pinned AIME 2025 and uses one completion per problem.

3. Stop. Leave `final_model` in place. After you exit, the launcher performs a separate
   one-problem, one-completion official final evaluation on pinned AIME 2026.

Do not train, iterate, or make a second submission. A score of zero is an acceptable smoke
result; infrastructure errors or an ungraded problem are not.

{setup_other}
