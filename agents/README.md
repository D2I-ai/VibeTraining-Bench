# Research-agent adapters

An outer research agent receives the prepared workspace, GPU/time budget, benchmark contract, and validation feedback. It researches and trains a model, then leaves `task/final_model/` for grading.

Benchmark and training code select an adapter by `AGENT_CLI`; each adapter owns its CLI invocation and credentials. Configure endpoints, model IDs, and account routing through the selected adapter's supported settings.

## Selection

Smoke and formal launchers forward `AGENT_CLI` and `RESEARCHER`. The default adapter is `claude`; the researcher model must be explicitly selected for preflight/run:

```bash
AGENT_CLI=claude RESEARCHER="<your-research-model>" RUN_ID=example \
  bash scripts/run_exp/run_exp_swe.sh run
```

Built-in adapters:

| Adapter | Credential | Notes |
|---|---|---|
| `claude` | `ANTHROPIC_API_KEY` | Anthropic-compatible Claude CLI path; optional `ANTHROPIC_BASE_URL`. |
| `codex` | `OPENAI_API_KEY` / `CODEX_API_KEY` | Mapped to the Codex credential; optional `OPENAI_BASE_URL`. |
| `claude_openrouter` | `OPENROUTER_API_KEY` | Anthropic-compatible gateway; optional `VTB_OPENROUTER_BASE_URL`. |

An explicit `VTB_AGENT_ADAPTERS_DIR` selects an external adapter root instead of `agents/`; missing adapters fail without fallback. `VTB_LOCAL_ENV_FILE` loads trusted local deployment settings before resolving launcher inputs. Configuration under `.local/` is gitignored and is loaded only when explicitly selected.

## Credentials

```bash
test -f api_key.env || cp api_key.env.example api_key.env
# Edit only the credential needed by the selected adapter.
```

`api_key.env` is optional, gitignored, and sourced by the host-side launcher when present. Exported credentials or a trusted explicit `VTB_LOCAL_ENV_FILE` work without it. Only `forward_env` variables declared by the selected adapter are passed into the researcher container. Required variables fail fast; unrelated provider keys and operator tokens are not forwarded. Explicit empty values are preserved when an adapter's authentication mode requires them. Docker receives variable names, not credential-valued command arguments.

Never:

- commit `api_key.env`, auth files, access tokens, or provider URLs containing credentials;
- write an HF token into `HF_HOME`;
- bake credentials into images or task resources;
- reuse grading admin/official tokens as provider credentials;
- hard-code deployment-specific relays in shared benchmark configuration.

## Adapter layout

```text
agents/<adapter>/
├── solve.sh                  CLI entrypoint copied into the research container
├── adapter_config.json      required_env / forward_env variable names
└── human_readable_trace.py   optional parser for persistent audit traces
```

`src/run_task.sh` copies `solve.sh` to the job home and executes it inside the trainer container with the task workspace as the current directory. The adapter must:

1. fail clearly when its required credential or CLI is unavailable;
2. consume the prepared prompt/workspace without reading grading secrets;
3. respect the run timeout and continuation signal;
4. stream raw output to the job log;
5. leave all model/training artifacts inside the workspace;
6. avoid hard-coded organization URLs or model aliases;
7. never invoke the final evaluator directly.

The optional parser converts raw CLI output into `solve_parsed.txt`; the raw trace remains the audit source. Entrypoint, optional auth files, and parser all resolve from the same selected adapter directory.

Example metadata (names only, no credential values):

```json
{
  "required_env": ["ANTHROPIC_API_KEY"],
  "forward_env": ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"]
}
```

Names must be uppercase shell identifiers; required names must also be forwarded. Grading/admin/official authority and protected runtime variables are rejected. Adapter scripts and local environment files are trusted executable operator code—not configuration to load from a model-generated workspace.

## Add an adapter

1. Create `agents/<adapter>/solve.sh`.
2. Add a trace parser or document raw-trace behavior.
3. Declare only the adapter's required credentials and supported endpoint/reasoning controls in `adapter_config.json`; the runner needs no provider-specific branch.
4. Add a commented provider-neutral entry to `api_key.env.example`.
5. Pin/install the required CLI in the trainer image.
6. Verify missing credentials fail without leaking values.
7. Run a bounded smoke with explicit small validation/final limits.
8. Document adapter name, credential, supported endpoint contract, model naming, retry/continuation behavior, and output files here.

Adding an adapter is a code change; it must be reviewed together with the container and secret-forwarding changes.

## Reasoning and continuation controls

Adapters may expose bounded reasoning controls such as:

| Variable | Intended use |
|---|---|
| `VTB_MAX_THINKING_TOKENS` | Extended-thinking budget for compatible CLIs. |
| `VTB_CODEX_REASONING_EFFORT` | Codex reasoning level. |
| `VTB_RESUME_NUDGE` | Internal continuation signal after an interrupted attempt. |

An adapter should define defaults locally and reject unsupported values. Provider-specific controls must not leak into benchmark definitions.

## Network and trust boundary

The research container may reach the selected outer-agent provider and permitted public data sources during preparation, but it must not receive:

- official grading credentials;
- hidden final task IDs/data;
- benchmark verifier secrets;
- unrestricted access to benchmark-owned service internals.

The grading service and benchmark task containers enforce the evaluation boundary. See [eval/grading/README.md](../eval/grading/README.md) for role isolation and disclosure policies.
