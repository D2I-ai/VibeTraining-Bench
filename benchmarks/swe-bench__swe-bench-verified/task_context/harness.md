# Editable SWE Harness

`harness/` is the SWE task solving scaffold delivered with your model. Harbor loads
`harness.mini_swe_agent:MiniSweAgent` for this experiment's validation and final run.
It is not the Research Agent that trains the model.

## What the harness is

`harness/mini_swe_agent.py` is a byte-for-byte copy of the mini-swe-agent shim in the
installed Harbor backend package -- the same adapter that the baseline name
`--agent mini-swe-agent` resolves to. A checkout copy is deliberately not accepted for
this seed: if the backend package is absent, workspace preparation fails instead of
silently creating a different harness. Preparation also verifies the copy's SHA-256. The
shim renders the task instruction, writes the mini-swe-agent config into the
task container, launches the in-container agent there, and converts the resulting
trajectory.

The in-container agent loop is the `mini-swe-agent` pip package, installed in the task
container at the version the benchmark declares (`agent_kwargs.version` in its
`benchmark.json`). That declaration is outside your workspace and fixed for the
experiment; the `_DEFAULT_MSWEA_VERSION` fallback constant in the seeded shim is always
overridden by it, so there is no pin for you to move. The pinned package's source ships
with the workspace, read-only, at `harness_reference/`: the unpacked package tree for
reading, plus the wheel itself. Its name, version and sha256 are recorded in
`train_resource_manifest.lock.json` under `package_reference`.

The loop itself can be taken over from the shim -- see level L2 below.

## Editable files

Edit only these supplied allowlisted workspace-relative paths:

- `harness/__init__.py` for package exports;
- `harness/mini_swe_agent.py` for the shim: instruction rendering, config layering,
  CLI flags, and trajectory handling.

The workspace-level `templates/` directory contains fixed model chat templates and is
not editable.

## How to modify the harness

Use harness changes only to improve general SWE-agent behavior. Changes come in three
levels; prefer the lowest that carries your hypothesis, and change one thing per
candidate.

### L0 -- instruction and config (the default surface)

Edit what the model is asked and the config the loop runs with: the
`augmented_instruction` assembly in `run()`, and the config layers written into the
container (the shim always layers `-c mini` first, then your overrides): system and
instance templates, `agent.step_limit`, `agent.cost_limit`, `model.model_kwargs`
(temperature, max_tokens), and `MSWEA_*` environment variables.

### L1 -- shim logic

Edit the shim's own behavior: `install()` (how the task container is provisioned -- the
loop package's version comes from the benchmark declaration, not from the shim),
commands run before or after the agent (repository warm-up, environment provisioning),
config assembly, and `populate_context_post_run()` (how metrics are recovered from the
trajectory). The in-container loop is unchanged at this level.

### L2 -- in-container loop (subclassing mini-swe-agent)

Take over the agent loop itself while keeping mini-swe-agent's model, environment, and
trajectory machinery:

1. Embed a custom run module as a string in `mini_swe_agent.py` -- typically a
   `minisweagent.agents.default.DefaultAgent` subclass that overrides
   `execute_actions()` (action validation, submit interception, custom actions) or
   extends `AgentConfig`.
2. In `run()`, write the module into the task container with the same heredoc pattern
   the shim already uses for `custom.yaml`, and execute it instead of the
   `mini-swe-agent` CLI.
3. Execute it with the tool-venv interpreter, not the system python:
   `"$HOME/.local/share/uv/tools/mini-swe-agent/bin/python" custom_run.py ...`.
   `minisweagent` is installed as a uv tool and is importable only there.

Read the pinned source before writing the subclass: `harness_reference/minisweagent/`
is exactly the package the evaluation installs (`agents/default.py` holds `DefaultAgent`
and `AgentConfig`). To import or run your module locally, install the shipped wheel into
a throwaway venv (`python3 -m venv /tmp/mswea-venv && /tmp/mswea-venv/bin/pip install
harness_reference/mini_swe_agent-*.whl`). Two local checks to know about: the shim
itself cannot be imported here (its base class lives in harbor, which is installed only
on the grading host) -- check it with `python3 -m py_compile`; and the real task loop
only exists inside an evaluation, which is what the small pre-flight submission is for.

Your module must still write `$AGENT_DIR/mini-swe-agent.trajectory.json` (with
`messages` and `info.model_stats` / `info.config`): `populate_context_post_run()` reads
token usage and cost from it and converts it to ATIF. A `DefaultAgent` subclass gets
this for free; a loop written from scratch must reproduce the file.

## Boundaries and caveats

- The grading service validates the package statically only (class and base class,
  allowlist, size). A broken loop-level change fails at evaluation time, not at
  submission. After any L2 change, always pre-flight with a small validation
  submission (`limit`: 5) and read the trajectories before scaling up.
- Write loop-level code against the pinned mini-swe-agent API (the reference source at
  `harness_reference/` is exactly that version). The pin comes from the benchmark's
  declaration and is fixed for this experiment. Code written against another version
  fails at evaluation time.
- Keep `harness.mini_swe_agent:MiniSweAgent` a `BaseInstalledAgent` subclass with a
  compatible `setup()`/`run()` interface, keep the package to the two allowlisted
  files, and keep the config written into the task container loadable by the pinned
  mini-swe-agent.
- Do not add files outside the supplied `harness/` package, change the grading client,
  or put task material, verifier output, answers, or task-specific rules in the
  harness.
- The evaluation passes its own config layer (fixed sampling parameters, and a step
  limit when one applies) into the shim's `config` kwarg. Your `run()` must keep
  writing that layer into the container's config chain -- a harness that drops it is
  no longer being scored under the benchmark's conditions.
- Record every harness change in `candidate.md` (what, why, and which level).

## Candidate pairing and delivery

Every validated or final model is evaluated with its matching `harness/`. The candidate
snapshot, submission, and delivery procedure is in `environment.md` (## Candidates);
`agent_rules.md` states the discipline rules.
