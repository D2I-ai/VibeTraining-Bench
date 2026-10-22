# Editable Terminal-Bench Harness

`harness/` is the Terminal task solving scaffold delivered with your model. Harbor
loads `harness.terminus_2:Terminus2` for this experiment's validation and final run.
It is not the Research Agent that trains the model.

## Harness path convention

All file paths in this document are relative to the workspace root. The editable Python
package is exactly `harness/`; there is no nested `harness/harness/` directory. For
example, read or edit `harness/terminus_2.py`. If your current directory is already
`harness/`, use `terminus_2.py` rather than `harness/terminus_2.py`.

## Editable files

Edit only these supplied allowlisted workspace-relative paths:

- `harness/__init__.py` for package exports;
- `harness/terminus_2.py` for the agent loop;
- `harness/terminus_json_plain_parser.py` and
  `harness/terminus_xml_plain_parser.py` for tool-call parsing;
- `harness/tmux_session.py`, `harness/asciinema_handler.py`, and
  `harness/get-asciinema-timestamp.sh` for the bundled helpers;
- `harness/templates/terminus-json-plain.txt`,
  `harness/templates/terminus-xml-plain.txt`, and
  `harness/templates/timeout.txt` for harness prompt text.

The workspace-level `templates/` directory contains fixed model chat templates and is
not editable.

## How to modify the harness

Use harness changes only to improve general terminal-agent behavior, such as tool-call
parsing, command-loop reliability, output handling, retries, or general prompting. Keep
the package layout and the public Harbor contract intact:

- `harness.terminus_2:Terminus2` must exist;
- `Terminus2` must remain a `BaseAgent` implementation with compatible `setup()` and
  `run()` methods;
- the supported tool-loop and trajectory interfaces must continue to work.

Do not add files outside the supplied `harness/` package, change the grading client, or
put validation task material, verifier output, answers, or task-specific rules in the
harness.

## Terminal interaction contract

The harness gives the model a task description and the conversation so far. The
terminal interaction contract remains fixed: the model must respond with one valid
OpenAI tool call requesting a shell command. The harness runs that command, appends
its output, and continues until the model signals completion or the interaction limit
is reached.

## Candidate pairing and delivery

Every validated or final model is evaluated with its matching `harness/`. The candidate
snapshot, submission, and delivery procedure is in `environment.md` (## Candidates);
`agent_rules.md` states the discipline rules.
