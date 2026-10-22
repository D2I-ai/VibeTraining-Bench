#!/bin/bash
# Smoke test for the shared grading service. No GPU, no real evaluation.
#
# Kept GPU-free by two settings, not by scaffolding:
#   VTB_SELF_SERVED_BENCHMARKS=$TASK   worker serves no model (no docker, no card).
#   VTB_HARBOR=<missing>               every evaluate.py fails fast on the absent
#                                      backend tool.
# A few sections instead point VTB_REPO at a fixture repo whose evaluate.py is a
# stand-in: a hang (section 9) or a success (section 11).
#
#   bash eval/grading/smoke_test.sh
#
# A real evaluation needs a GPU and is a separate operator step (see README.md).
#
# Section 9 asserts the evaluation's HuggingFace cache overlay, so it needs
# fuse-overlayfs on the host (README setup step 1). Without it the service logs
# `[hf] fuse-overlayfs not found -- binding the cache directly` and that one check fails
# while every other section still passes -- a missing host package, not a regression.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../src/commit_utils/configure_python.sh" || exit 1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1

PORT="${SMOKE_PORT:-5899}"
# The benchmark every section registers against. Must be really installed:
# registration validates against config.known_tasks() (a benchmarks/ directory name,
# not a label), so a renamed-away task would fail every section with "unknown task".
TASK="${SMOKE_TASK:-aime__aime}"
# Honour TMPDIR: vtb-env/env.sh points it at $VTB_ROOT/tmp, and a test should not
# be the one thing that still writes to a hardcoded /tmp.
TMP="$(mktemp -d "${TMPDIR:-/tmp}/vtb_grading_smoke.XXXXXX")"
CLIENT=("$VTB_PYTHON" -m eval.grading.client --base "http://127.0.0.1:${PORT}")

PASS=0
FAIL=0
check() { # check <description> <expected-substring> <actual>
    if printf '%s' "$3" | grep -q -- "$2"; then
        echo "  PASS  $1"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $1"
        echo "        expected to contain: $2"
        echo "        got: $(printf '%s' "$3" | head -c 400)"
        FAIL=$((FAIL + 1))
    fi
}

check_absent() { # check_absent <description> <forbidden-substring> <actual>
    if printf '%s' "$3" | grep -q -- "$2"; then
        echo "  FAIL  $1"
        echo "        must NOT contain: $2"
        echo "        got: $(printf '%s' "$3" | head -c 400)"
        FAIL=$((FAIL + 1))
    else
        echo "  PASS  $1"
        PASS=$((PASS + 1))
    fi
}

cleanup() {
    # Wait for every service to finish its graceful teardown (killing the eval child,
    # unmounting the overlay) before the rm -rf below, or debris is left behind.
    for _pid in "${SRV_PID:-}" "${SRV2_PID:-}" "${SRV3_PID:-}" "${SRV4_PID:-}" "${SRV5_PID:-}"; do
        [ -n "$_pid" ] || continue
        kill "$_pid" 2>/dev/null
        for _ in $(seq 1 40); do
            kill -0 "$_pid" 2>/dev/null || break
            sleep 0.25
        done
        kill -9 "$_pid" 2>/dev/null
        wait "$_pid" 2>/dev/null
    done
    # Anchor on the whole command line: a bare -f 'sleep 987' would also match a
    # process that merely mentions the string.
    pkill -f '^sleep 987$' 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT

mkdir -p "$TMP/runA/task/output/v1" "$TMP/runA/eval" "$TMP/runB/task" "$TMP/runB/eval"
# A file that makes the submitted directory look like a model directory.
echo '{}' > "$TMP/runA/task/output/v1/config.json"

echo "=== starting service on port $PORT (state=$TMP/state) ==="
VTB_REPO="$REPO" \
VTB_GRADING_STATE="$TMP/state" \
VTB_GRADING_SETTLE=0.2 \
VTB_VAL_SOURCE=split:5 \
VTB_EVAL_GPU="" \
    VTB_SELF_SERVED_BENCHMARKS="$TASK" \
    VTB_HARBOR="$TMP/no_such_harbor" \
    "$VTB_PYTHON" -m eval.grading.server --host 127.0.0.1 --port "$PORT" > "$TMP/server.log" 2>&1 &
SRV_PID=$!

for _ in $(seq 1 40); do
    "${CLIENT[@]}" health >/dev/null 2>&1 && break
    sleep 0.25
done

# Registration is operator-only, so the client has to find this instance's admin
# secret rather than the one under the real VTB_ROOT.
export VTB_ADMIN_TOKEN_FILE="$TMP/state/admin.token"
BASE="http://127.0.0.1:${PORT}"

echo "=== 1. health ==="
OUT="$("${CLIENT[@]}" health 2>&1)"
check "health reports ok" '"status": "ok"' "$OUT"
check "health lists the task" "$TASK" "$OUT"

# The agent calls client.py by absolute path from inside its container, where the
# repo root is neither the cwd nor on sys.path -- the script form must work anywhere.
OUT="$(cd / && "$VTB_PYTHON" "$REPO/eval/grading/client.py" --base "http://127.0.0.1:${PORT}" health 2>&1)"
check "client.py runs as a plain script from an unrelated cwd" '"status": "ok"' "$OUT"

# One VTB_ROOT change must relocate everything a run produces (scratch, state, caches).
# Container images are the exception: they are docker tags, not files under VTB_ROOT.
OUT="$(cd "$REPO" && env -u VTB_GRADING_STATE -u HF_HOME \
    VTB_ROOT=/data/elsewhere "$VTB_PYTHON" - <<'PY' 2>&1
import sys
sys.path.insert(0, ".")
from eval.grading import config as c
assert str(c.STATE_DIR) == "/data/elsewhere/grading_states/direct", c.STATE_DIR
assert str(c.HF_HOME) == "/data/elsewhere/resources/data/hf_cache", c.HF_HOME
assert str(c.BASELINE_CACHE_DIR) == "/data/elsewhere/grading_states/direct/baselines", c.BASELINE_CACHE_DIR
# The serving image is a tag, so it does not move with VTB_ROOT and does not look
# like a path.
assert "elsewhere" not in c.serve_image(), c.serve_image()
assert "/" != c.serve_image()[0], c.serve_image()
# The repo is found from this file's own location, so it does not follow VTB_ROOT.
assert "elsewhere" not in str(c.REPO_ROOT), c.REPO_ROOT
print("RELOCATE_OK")
PY
)"
check "one VTB_ROOT change relocates every produced path" 'RELOCATE_OK' "$OUT"

OUT="$(cd "$REPO" && env -u HF_HOME \
    VTB_ROOT=/data/elsewhere VTB_GRADING_STATE=/fast/nvme/gs "$VTB_PYTHON" - <<'PY' 2>&1
import sys
sys.path.insert(0, ".")
from eval.grading import config as c
assert str(c.STATE_DIR) == "/fast/nvme/gs", c.STATE_DIR
assert str(c.HF_HOME) == "/data/elsewhere/resources/data/hf_cache", c.HF_HOME
print("OVERRIDE_OK")
PY
)"
check "a single path can still be overridden on its own" 'OVERRIDE_OK' "$OUT"

# Two variables lost a SWE- infix they never earned: the serve deadline and the endpoint
# apply to every benchmark in benchmarks/, most of which are not SWE. A rename that only
# reaches the new name is worse than no rename, because an operator's shell history keeps
# passing the old one and silently stops steering the run. Both must be read, and the
# current name must win when both are present.
OUT="$(cd "$REPO" && env -u VTB_SERVE_TIMEOUT VTB_SWE_SERVE_TIMEOUT=77 "$VTB_PYTHON" - <<'PY' 2>&1
import sys
sys.path.insert(0, ".")
from eval.grading import config as c
assert c.SERVE_TIMEOUT_SECONDS == 77, c.SERVE_TIMEOUT_SECONDS
print("OLD_TIMEOUT_OK")
PY
)"
check "the retired VTB_SWE_SERVE_TIMEOUT still sets the serve deadline" 'OLD_TIMEOUT_OK' "$OUT"

OUT="$(cd "$REPO" && env VTB_SERVE_TIMEOUT=88 VTB_SWE_SERVE_TIMEOUT=77 "$VTB_PYTHON" - <<'PY' 2>&1
import sys
sys.path.insert(0, ".")
from eval.grading import config as c
assert c.SERVE_TIMEOUT_SECONDS == 88, c.SERVE_TIMEOUT_SECONDS
print("NEW_TIMEOUT_WINS")
PY
)"
check "the current name wins when both serve deadlines are set" 'NEW_TIMEOUT_WINS' "$OUT"

# Same contract on the runner's side of the fence, where --api-base takes its default.
OUT="$(cd "$REPO" && env -u VTB_API_BASE VTB_SWE_API_BASE=http://old/v1 "$VTB_PYTHON" - <<'PY' 2>&1
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "benchmarks")
import registry
from eval.runner import run
spec = registry.get("swe-bench__swe-bench-verified")
assert run.build_parser(spec, 1).parse_args([]).api_base == "http://old/v1"
print("OLD_BASE_OK")
PY
)"
check "the retired VTB_SWE_API_BASE still points the runner at an endpoint" 'OLD_BASE_OK' "$OUT"

OUT="$(cd "$REPO" && env VTB_API_BASE=http://new/v1 VTB_SWE_API_BASE=http://old/v1 "$VTB_PYTHON" - <<'PY' 2>&1
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "benchmarks")
import registry
from eval.runner import run
spec = registry.get("swe-bench__swe-bench-verified")
base = run.build_parser(spec, 1).parse_args([]).api_base
assert base == "http://new/v1", base
print("NEW_BASE_WINS")
PY
)"
check "the current name wins when both endpoints are set" 'NEW_BASE_WINS' "$OUT"

# The resolved paths are operator-only: they are ingredients for naming a sandbox_root.
OUT="$(curl -s "http://127.0.0.1:${PORT}/health?admin_token=$(cat "$TMP/state/admin.token")" 2>&1)"
check "health reports the resolved paths to the operator" '"state_dir"' "$OUT"

echo "=== 2. registration ==="
TOKEN_A="$("${CLIENT[@]}" register --run-id smokeA --task "$TASK" \
    --base-model Qwen/Qwen3-1.7B-Base --sandbox-root "$TMP/runA/task" --eval-dir "$TMP/runA/eval" 2>&1)"
check "run A receives a token" '^[A-Za-z0-9_-]\{20,\}$' "$TOKEN_A"

TOKEN_B="$("${CLIENT[@]}" register --run-id smokeB --task "$TASK" \
    --base-model Qwen/Qwen3-1.7B-Base --sandbox-root "$TMP/runB/task" --eval-dir "$TMP/runB/eval" 2>&1)"
check "run B receives a token" '^[A-Za-z0-9_-]\{20,\}$' "$TOKEN_B"

OUT="$("${CLIENT[@]}" register --run-id bad --task no_such_task \
    --base-model m --sandbox-root "$TMP/runA/task" --eval-dir "$TMP/runA/eval" 2>&1)"
check "unknown task is rejected" 'unknown task' "$OUT"

echo "=== 3. path sandbox ==="
OUT="$("${CLIENT[@]}" submit --token="$TOKEN_A" --model-path /etc --mode val 2>&1)"
check "absolute path outside the sandbox is rejected" 'escapes the run sandbox' "$OUT"

OUT="$("${CLIENT[@]}" submit --token="$TOKEN_A" --model-path ../../../etc --mode val 2>&1)"
check "traversal out of the sandbox is rejected" 'escapes the run sandbox' "$OUT"

OUT="$("${CLIENT[@]}" submit --token="$TOKEN_A" --model-path output/nope --mode val 2>&1)"
check "nonexistent path is rejected" 'does not exist' "$OUT"

OUT="$("${CLIENT[@]}" submit --token deadbeef --model-path output/v1 --mode val 2>&1)"
check "malformed token is rejected client-side, before any request" 'does not look like a token' "$OUT"

# Motivates --token-file: a shell capture of an error blob (not a token) would come
# back as "403 unknown or expired token", blaming auth for a caller-side mistake.
BOGUS='{"error": "'"'"'unknown or expired token'"'"'", "_http_status": 403}'
OUT="$("${CLIENT[@]}" submit --token="$BOGUS" --model-path output/v1 --mode val 2>&1)"
check "an error blob mistaken for a token is named as such" 'does not look like a token' "$OUT"
check_absent "...and is never reported as an auth failure" 'unknown or expired token"' "$OUT"

echo "=== 3b. token via file ==="
"${CLIENT[@]}" register --run-id smokeC --task "$TASK" --base-model Qwen/Qwen3-1.7B-Base \
    --sandbox-root "$TMP/runA/task" --eval-dir "$TMP/runA/eval" --token-file "$TMP/tok" >/dev/null 2>&1
check "register --token-file writes the token" '^[A-Za-z0-9_-]\{32\}$' "$(cat "$TMP/tok" 2>&1)"
check "token file is not world readable" '600' "$(stat -c %a "$TMP/tok" 2>&1)"
OUT="$("${CLIENT[@]}" queue --token-file "$TMP/tok" 2>&1)"
check "--token-file is accepted by other subcommands" 'pending_count' "$OUT"
OUT="$("${CLIENT[@]}" queue --token-file "$TMP/nonexistent" 2>&1)"
check "a missing token file fails clearly" 'cannot read token file' "$OUT"

echo "=== 4. submission + retry ladder (backend tool missing, so all attempts fail) ==="
# stderr to a file, not OUT: --wait streams progress there while the result JSON goes
# to stdout, so 2>&1 would interleave them and break the json.load of the job id below.
OUT="$("${CLIENT[@]}" submit --token="$TOKEN_A" --model-path output/v1 --mode val \
    --wait --wait-timeout 120 2>"$TMP/submitA.err")"
check "job reaches a terminal failed state" '"state": "failed"' "$OUT"
check "ladder ran all 9 attempts (4+3+2)" '"attempts": 9' "$OUT"
JOB_A="$(printf '%s' "$OUT" | "$VTB_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["job_id"])' 2>/dev/null)"
check "the result names the job, so the assertions below read the right one" '^[0-9a-f]\{8,\}$' "$JOB_A"

# Why they failed, not just that they did: nine attempts also "fail" on a bad command
# or task dir, and the ladder assertions above would pass either way.
check "the attempts failed on the missing backend tool, and got that far" 'is not available' \
    "$(cat "$TMP/state/work/$JOB_A/eval_p1_a1.log" 2>&1)"

# The service, not the caller, sets a validation submission's sample count:
# VTB_VAL_SOURCE=split:5, so --limit 5 must appear though nothing asked for it.
CMDLINE="$(grep -m1 'phase 1 attempt 1: /' "$TMP/server.log")"
check "the service imposes its own validation limit" '\-\-limit 5' "$CMDLINE"
# No phase lowers a step budget any more, and that is the point: the samples needing the
# most steps are the hard ones, so a lowered ceiling answers them instead of measuring
# them. What replaced it is after-the-fact detection, asserted next.
check_absent "phase 2 does not lower the step budget" '--step-limit' \
    "$(grep -m1 'phase 2 attempt 1: /' "$TMP/server.log")"
check_absent "phase 3 does not lower it either" '--step-limit' \
    "$(grep -m1 'phase 3 attempt 1: /' "$TMP/server.log")"

# The replacement for that cap. It lives in eval/runner/trial_results.py rather than here,
# but it is what now keeps a runaway trial from being invisible, so the suite that used to
# assert the cap asserts this instead. Needs no GPU: it reads a message list.
OUT="$(cd "$REPO" && "$VTB_PYTHON" -c "
import sys; sys.path.insert(0,'.')
from eval.runner.trial_results import _no_progress_repeats as repeats, NO_PROGRESS_REPEATS as TH
def steps(pairs):
    ms = []
    for cmd, out in pairs:
        ms.append({'role': 'assistant', 'tool_calls': [{'function': {'arguments': cmd}}]})
        ms.append({'role': 'tool', 'content': out})
    return ms
# Polling something that keeps changing is legitimate work, however long it goes on --
# waiting on a build, watching a training log. Those must never count.
assert repeats(steps([('tail -1 train.log', 'step %d' % i) for i in range(60)])) == 1
# The same command returning the same bytes is the thing being caught.
assert repeats(steps([('ls', 'same')] * 60)) == 60
# And the threshold is categorical, not a percentile: two healthy deliveries over 951
# trials never exceeded 2, while a degenerate one reached 887.
assert repeats(steps([('ls', 'same')] * (TH - 1))) < TH <= repeats(steps([('ls', 'same')] * TH))
print('NO_PROGRESS_OK')")"
check "a stuck trial is caught by repeated action *and* output, not by a step ceiling" \
    'NO_PROGRESS_OK' "$OUT"

echo "=== 5. cross-run isolation ==="
OUT="$("${CLIENT[@]}" submit --token="$TOKEN_B" --model-path /etc --mode val 2>&1)"
check "run B cannot reach outside its own sandbox either" 'escapes the run sandbox' "$OUT"

OUT="$("$VTB_PYTHON" -c "
import json,urllib.request,urllib.parse
url='http://127.0.0.1:${PORT}/result?token='+urllib.parse.quote('$TOKEN_B')+'&job_id='+urllib.parse.quote('$JOB_A')
try:
    urllib.request.urlopen(url, timeout=10)
    print('LEAKED')
except urllib.error.HTTPError as e:
    print(e.code, e.read().decode())
" 2>&1)"
check "run B cannot read run A's job" 'unknown job_id' "$OUT"

OUT="$("${CLIENT[@]}" queue --token="$TOKEN_B" 2>&1)"
check "queue hides other runs' job ids" '"own_pending_count": 0' "$OUT"

echo "=== 6. submission log written to the run's eval dir ==="
check "submissions.json exists" 'submissions.json' "$(ls "$TMP/runA/eval" 2>&1)"
check "submissions.json records the failure" 'error' "$(cat "$TMP/runA/eval/submissions.json" 2>&1)"
check "metrics.json NOT written for a validation submission" 'no metrics' \
    "$([ -f "$TMP/runA/eval/metrics.json" ] && echo 'metrics present' || echo 'no metrics')"

echo "=== 6b. progress parsing (fixture built from real observed log lines) ==="
# Line shapes copied from a real vLLM-served evaluation. This exercises the weak path:
# inspect-ai prints no sample counter, so where a log is all there is, the access log's
# completion count is what can be counted. The agentic harnesses' attempt logs contain
# none of these lines -- the per-sample result files are the signal there, covered below.
FIX="$TMP/fixture.log"
{
  echo '[08/10/26 15:39:03] WARNING  Starting new server for            vllm.py:119'
  echo '(EngineCore_DP0 pid=1) Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:08<00:00,  8.38s/it]'
  echo '(EngineCore_DP0 pid=1) Capturing CUDA graphs (decode, FULL): 100%|##########| 35/35 [00:01<00:00, 34.95it/s]'
  echo '(APIServer pid=2) INFO:     Application startup complete.'
  echo '[08/10/26 15:39:53] WARNING  vLLM server started at             vllm.py:135'
} > "$FIX"
for _ in $(seq 1 30); do
  echo '(APIServer pid=2) INFO:     127.0.0.1:35222 - "POST /v1/chat/completions HTTP/1.1" 200 OK' >> "$FIX"
done
# One failed request must not be counted as progress.
echo '(APIServer pid=2) INFO:     127.0.0.1:35222 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error' >> "$FIX"

OUT="$(cd "$REPO" && "$VTB_PYTHON" - "$FIX" "$TMP" "$TASK" <<'PY' 2>&1
import json, os, sys, time
sys.path.insert(0, ".")
from pathlib import Path
from eval.grading import config, progress
from eval.grading import server as grading_server
from eval.runner import trial_results
from eval.grading.server import NO_PROGRESS_DURATION_LIMIT
from eval.grading.store import DurationStats

log = Path(sys.argv[1])
tmp = Path(sys.argv[2])
task_name = sys.argv[3]
reader = progress.ProgressReader(cache_ttl=0)

snap = reader.read(log)
assert snap["phase"] == "generating", snap
assert snap["completions_observed"] == 30, snap
assert snap["generating_since"] is not None, snap

# Incremental: a second read must consume only the newly appended bytes.
off_before = reader._states[str(log)].offset
with open(log, "a") as fh:
    for _ in range(10):
        fh.write('(APIServer pid=2) INFO:     127.0.0.1:35222 - "POST /v1/chat/completions HTTP/1.1" 200 OK\n')
time.sleep(0.05)
snap2 = reader.read(log)
off_after = reader._states[str(log)].offset
assert snap2["completions_observed"] == 40, snap2
assert off_after > off_before, (off_before, off_after)

# --- percentage and observed-rate ETA, for a hypothetical single-completion task ---
# Kept as a regression though no such benchmark is installed: the ETA arithmetic was
# wrong once (credited already-counted completions with zero elapsed time). At ~1
# completion/s, 20 of 40 done means ~20s remaining.
state = reader._states[str(log)]
state.completions = 20
state.generating_since = time.time() - 20.0   # 20 completions in 20s => 1/s
single_task = "_single_completion_probe"
progress.SINGLE_COMPLETION_TASKS = frozenset({single_task})
block = progress.build_progress(
    task=single_task, state="running", started_at=time.time() - 80,
    queue_position=None, attempt_phase=1, attempt_index=1, log_path=log,
    target_samples=40, historical=None, reader=reader,
)
assert block["percent"] == 50.0, block
assert block["eta_basis"] == "observed_rate", block
assert 17 <= block["eta_seconds"] <= 23, f"eta {block['eta_seconds']}s, expected ~20s: {block}"
assert block["seconds_since_last_completion"] is not None, block
progress.SINGLE_COMPLETION_TASKS = frozenset()

# A multi-completion task must not be given a percentage.
multi = progress.build_progress(
    task=task_name, state="running", started_at=time.time() - 100,
    queue_position=None, attempt_phase=1, attempt_index=1, log_path=log,
    target_samples=50, historical=None, reader=reader,
)
assert multi["percent"] is None, multi
assert "several completions per sample" in (multi["note"] or ""), multi

# Per-sample result files, where the harness writes them, replace the completion
# proxy: a real k-of-n, and the long tail named. This is the case an agent
# otherwise reads as a stalled evaluation.
sample_root = tmp / "harness_metrics_p1_a1" / "backend_output" / "2026-01-01__00-00-00"
sample_root.mkdir(parents=True)
for i in range(9):
    done_dir = sample_root / f"repo__pkg-{i}__AbCd123"
    done_dir.mkdir()
    # One sample far more expensive than the rest: the shape a looping trial makes.
    tokens = 40_000_000 if i == 8 else 2_000_000
    payload = {"agent_result": {"n_input_tokens": tokens}}
    # Two of them ended as harness failures, one of each class. A trial that repeats a
    # command until its turns run out is what a timeout is, so counting the classes as
    # they land is the loop becoming visible before the evaluation is over.
    if i in (7, 8):
        payload["exception_info"] = {
            "exception_type": "AgentTimeoutError" if i == 8 else "NonZeroAgentExitCodeError",
            "exception_message": "ran out of time",
        }
    (done_dir / "result.json").write_text(json.dumps(payload))
slow = sample_root / "repo__slow-one__ZzYy987"
slow.mkdir()
os.utime(slow, (time.time() - 2400, time.time() - 2400))

tailed = progress.build_progress(
    task=task_name, state="running", started_at=time.time() - 3000,
    queue_position=None, attempt_phase=1, attempt_index=1, log_path=log,
    target_samples=10, historical=(6204.0, True, 5), reader=reader,
    metrics_dir=tmp / "harness_metrics_p1_a1",
)
assert tailed["samples_finished"] == 9, tailed
assert tailed["samples_outstanding"] == 1, tailed
assert tailed["percent"] == 90.0, tailed
assert tailed["eta_basis"] == "observed_sample_rate", tailed
# Never shorter than the tail has already run: a mean rate would promise sooner.
assert tailed["eta_seconds"] >= 2400, tailed
assert tailed["outstanding_samples"][0]["instance_id"] == "repo__slow-one", tailed
assert "long tail rather than a stalled evaluation" in (tailed["note"] or ""), tailed
assert tailed["samples_unstarted"] == 0, tailed

# /sample serves a running sample by id, and the index it uses has to be the same walk
# that produced outstanding_samples: an id a caller can see running but cannot ask about
# is worse than not reporting it, because the 404 reads as "wrong id".
live = progress.live_trial_dirs(tmp / "harness_metrics_p1_a1")
assert set(live) == {"repo__slow-one"}, live
assert live["repo__slow-one"].name == "repo__slow-one__ZzYy987", live
assert {o["instance_id"] for o in tailed["outstanding_samples"]} == set(live), (tailed, live)
assert progress.live_trial_dirs(None) == {}, "no metrics dir yet is not an error"
assert progress.live_trial_dirs(tmp / "no-such-attempt") == {}, "missing tree is not an error"

# Loop counters ride along with the ids already reported, because a caller was otherwise
# spending one request per sample to fetch exactly this. Age alone cannot tell a loop from
# slow honest work, and that is the distinction between cancelling and waiting.
loop_agent = slow / "agent"
loop_agent.mkdir(parents=True)
stuck = []
for _ in range(25):
    stuck.append({"role": "assistant", "tool_calls": [
        {"function": {"arguments": '{"command": "ls /testbed"}'}}]})
    stuck.append({"role": "tool", "content": "same listing every time"})
(loop_agent / "trajectory.json").write_text(json.dumps({"messages": stuck}))
looped = progress.build_progress(
    task=task_name, state="running", started_at=time.time() - 3000,
    queue_position=None, attempt_phase=1, attempt_index=1, log_path=log,
    target_samples=10, historical=(6204.0, True, 5), reader=reader,
    metrics_dir=tmp / "harness_metrics_p1_a1",
)
stuck_entry = looped["outstanding_samples"][0]
assert stuck_entry["instance_id"] == "repo__slow-one", stuck_entry
assert stuck_entry["no_progress_repeats"] == 25, stuck_entry
assert stuck_entry["stalled"] is True, stuck_entry
assert stuck_entry["turns"] == 50, stuck_entry
assert "repeated one action without new output" in (looped["note"] or ""), looped["note"]
# The aggregate and the per-sample detail /sample serves must agree, or a caller comparing
# the two cannot trust either.
assert stuck_entry["no_progress_repeats"] == trial_results.agent_stats(slow)["no_progress_repeats"]
# Rate-limited rather than content-keyed: a running transcript changes on every step, so a
# cache keyed on its contents would never hit and every poll would pay the full parse.
before = len(progress._loop_cache)
progress.loop_indicators(slow, time.time())
assert len(progress._loop_cache) == before, "a second look inside the interval must be served from cache"
assert progress.loop_indicators(None, time.time()) is None

# A transcript excerpt keeps the end. The opening is setup every trial shares; a sample
# that stopped converging is repeating itself in its last actions, so truncating from the
# front withheld the only part that explains it.
_trunc = grading_server.GradingService._truncate_trajectory
long_text = "OPENING_SETUP" + ("x" * 40000) + "LAST_ACTION"
excerpt = _trunc(object(), long_text)
assert excerpt.endswith("LAST_ACTION"), excerpt[-40:]
assert "OPENING_SETUP" not in excerpt, "the head must be the part dropped"
assert excerpt.startswith("... [earlier "), excerpt[:40]
assert _trunc(object(), "short enough") == "short enough", "no marker when nothing was cut"
assert _trunc(object(), None) is None

# ...but the excerpt is counted in steps, not characters, because for a looping trial the
# two disagree in the worst direction: its outputs are huge, so the last N characters sit
# inside one tool result and contain no complete step. This fixture is that shape -- one
# command repeated with a giant identical output -- and the character tail must fail to
# show the repeat while the step tail shows it.
loop_dir = tmp / "loopy" / "agent"
loop_dir.mkdir(parents=True)
bulk = "same output line\n" * 3000
loop_msgs = []
for _ in range(20):
    loop_msgs.append({"role": "assistant", "tool_calls": [
        {"function": {"arguments": '{"command": "grep -n through models.py"}'}}]})
    loop_msgs.append({"role": "tool", "content": bulk})
(loop_dir / "trajectory.json").write_text(json.dumps({"messages": loop_msgs}))
raw = (loop_dir / "trajectory.json").read_text()
assert '"tool_calls"' not in raw[-12000:], "fixture must reproduce the zero-step character tail"
stepwise = trial_results.trajectory_tail(tmp / "loopy", 12000)
assert stepwise.count("--- step ") == 12, stepwise[:200]
assert stepwise.count("grep -n through models.py") == 12, "every step's command must survive"
assert "chars]" in stepwise, "the bulk output must be clipped, not carried"
assert len(stepwise) < 14000, f"the step tail stays near the character budget: {len(stepwise)}"
# The repeat count and the excerpt have to be reading the same notion of an action, or the
# count claims a loop the excerpt does not show.
assert trial_results.agent_stats(tmp / "loopy")["no_progress_repeats"] == 20
assert trial_results.trajectory_tail(tmp / "no-such-trial", 12000) is None
# Token spend per sample, as percentiles: a median that held while p90 and max ran
# away is a few looping trials, not a model that got uniformly more expensive. A
# single averaged number cannot express that difference.
spend = tailed["sample_input_tokens"]
assert spend["counted"] == 9, spend
assert spend["median"] == 2_000_000, spend
assert spend["max"] == 40_000_000, spend
assert spend["max"] > 10 * spend["median"], spend
# How the failures failed, counted live. A sample merely graded 0 must not appear here:
# a hard sample is not a broken one, and conflating them makes every queue look sick.
assert tailed["sample_errors"] == {"AgentTimeoutError": 1, "NonZeroAgentExitCodeError": 1}, tailed

# Concurrency-limited but early: 5 graded and 16 running out of 150 is the
# beginning, not a tail. Counting only the running samples called this a tail and
# told the agent to cancel, which would have thrown the evaluation away.
for i in range(15):
    extra = sample_root / f"repo__more-{i}__Ef3"
    extra.mkdir()
    os.utime(extra, (time.time() - 300, time.time() - 300))
early = progress.build_progress(
    task=task_name, state="running", started_at=time.time() - 600,
    queue_position=None, attempt_phase=1, attempt_index=1, log_path=log,
    target_samples=150, historical=None, reader=reader,
    metrics_dir=tmp / "harness_metrics_p1_a1",
)
assert early["samples_finished"] == 9, early
assert early["samples_unstarted"] == 125, early
assert "long tail" not in (early["note"] or ""), early
assert "have not started yet" in (early["note"] or ""), early

# Nothing graded yet, but trials are running: the container is plainly up, so the phase
# must not still read "launching". Observed live -- 16 samples were running while the
# label said launching, because the override keyed on graded samples alone.
running_only = tmp / "harness_metrics_p9_a1"
(running_only / "backend_output" / "run1").mkdir(parents=True)
for i in range(3):
    (running_only / "backend_output" / "run1" / f"repo__started-{i}__Aa1").mkdir()
no_grades = progress.build_progress(
    task=task_name, state="running", started_at=time.time() - 300,
    queue_position=None, attempt_phase=9, attempt_index=1, log_path=None,
    target_samples=30, historical=None, reader=progress.ProgressReader(),
    metrics_dir=running_only,
)
assert no_grades["samples_finished"] == 0, no_grades
assert no_grades["samples_outstanding"] == 3, no_grades
assert no_grades["phase"] == "generating", no_grades
assert "launching" not in (no_grades["phase_label"] or ""), no_grades

# --- historical ETA, exact key and fallback ---
stats = DurationStats(tmp)
run_a = "prefixA/task#agent"
stats.record(run_a, task_name, "dev", None, 240.0)
# The record count travels with the median: one observation and ten cannot be quoted
# with the same confidence, and the caller needs to know which it has.
assert stats.lookup(run_a, task_name, "dev", None) == (240.0, True, 1), stats.lookup(run_a, task_name, "dev", None)
# A limit never seen before still gets a hint, flagged inexact (an exact-key-only
# lookup meant history almost never applied).
approx = stats.lookup(run_a, task_name, "dev", 40)
assert approx == (240.0, False, 1), approx
# A task with no history returns nothing -- the inexact fallback must not invent one.
assert stats.lookup(run_a, "_never_measured", "dev", 40) is None

# Scoped to the run. The service is shared, so pooling let one run's pathology set
# another run's expectations -- which is exactly how a 981-minute record from a broken
# run became a 15.8 hour estimate for an evaluation that took 3.8.
assert stats.lookup("prefixB/task#agent", task_name, "dev", None) is None, "another run's history leaked"
# Both of a run's tokens are the same run: splitting them would leave the official
# evaluation with no history at all.
assert stats.lookup("prefixA/task#official", task_name, "dev", None) == (240.0, True, 1)
# Keys written before the run was part of the key are unreadable rather than trusted.
legacy = json.loads((tmp / "durations.json").read_text())
legacy[f"{task_name}|dev|150|c{config.eval_concurrency()}"] = [58872.0]
(tmp / "durations.json").write_text(json.dumps(legacy))
assert stats.lookup(run_a, task_name, "dev", 150) == (240.0, False, 1), "legacy key was read"

# Three records is where a median stops being a single observation wearing a
# statistic's name, so the caveat appears below that and not at or above it.
thin = progress.build_progress(
    task=task_name, state="running", started_at=time.time() - 60, queue_position=None,
    attempt_phase=0, attempt_index=0, log_path=None,
    target_samples=150, historical=(58872.0, True, 1), reader=reader,
)
assert thin["eta_from_runs"] == 1, thin
assert "too few to be a promise" in (thin["note"] or ""), thin
thick = progress.build_progress(
    task=task_name, state="running", started_at=time.time() - 60, queue_position=None,
    attempt_phase=0, attempt_index=0, log_path=None,
    target_samples=150, historical=(58872.0, True, 3), reader=reader,
)
assert thick["eta_from_runs"] == 3, thick
assert "too few to be a promise" not in (thick["note"] or ""), thick

# Cold start: a run's first submission has no history of its own, and another run's is
# not available by design, so the base model's own measured per-sample cost is what is
# left. 592s per sample over 150 samples at concurrency 16 is about 1.54 hours.
cold = progress.build_progress(
    task=task_name, state="running", started_at=time.time() - 60, queue_position=None,
    attempt_phase=0, attempt_index=0, log_path=None,
    target_samples=150, historical=None, reader=reader,
    baseline_timing={"mean_seconds": 592.3, "samples": 147},
)
assert cold["eta_basis"] == "baseline_sample_rate", cold
assert cold["baseline_sample_seconds"] == 592, cold
assert 4800 < cold["eta_seconds"] < 6400, cold
assert "assumes your model costs what the base model measured" in (cold["note"] or ""), cold
# The run's own history, however thin, still beats an assumption about a different model.
owned = progress.build_progress(
    task=task_name, state="running", started_at=time.time() - 60, queue_position=None,
    attempt_phase=0, attempt_index=0, log_path=None,
    target_samples=150, historical=(3600.0, True, 4), reader=reader,
    baseline_timing={"mean_seconds": 592.3, "samples": 147},
)
assert owned["eta_basis"] == "historical_median", owned
# Reported either way: it is the yardstick for the run's own measured rate.
assert owned["baseline_sample_seconds"] == 592, owned

# An evaluation whose trials mostly looped timed the looping, not the work, so its
# duration is not kept as an expectation for later. The measured contrast: 0 of 147 for
# a healthy queue against 53 of 149 for the run this guard was drawn from.
assert grading_server._no_progress_share({"no_progress_trials": 53, "graded": 149}) > NO_PROGRESS_DURATION_LIMIT
assert grading_server._no_progress_share({"no_progress_trials": 0, "graded": 147}) < NO_PROGRESS_DURATION_LIMIT
assert grading_server._no_progress_share({"no_progress_trials": 5, "graded": 30}) < NO_PROGRESS_DURATION_LIMIT
# Unmeasured must not read as healthy, or the guard would pass exactly the benchmarks
# that cannot report it.
assert grading_server._no_progress_share({"no_progress_trials": None, "graded": 30}) is None
assert grading_server._no_progress_share(None) is None

queued = progress.build_progress(
    task=task_name, state="queued", started_at=time.time() - 10,
    queue_position=3, attempt_phase=None, attempt_index=None, log_path=None,
    target_samples=150, historical=(240.0, True, 5), reader=reader,
)
assert queued["eta_basis"] == "historical_median", queued
assert queued["queue_position"] == 3, queued

inexact = progress.build_progress(
    task=task_name, state="queued", started_at=time.time() - 10,
    queue_position=1, attempt_phase=None, attempt_index=None, log_path=None,
    target_samples=40, historical=(240.0, False, 5), reader=reader,
)
assert inexact["eta_basis"] == "historical_median_other_limit", inexact

# Read from the benchmark's own declaration, so a wrong task name yields None. The
# config layer normalizes a non-positive declaration (aime declares -1, "no cap") to
# None -- mirror that rule here rather than pin a number.
declared_limit = json.loads(Path(f"benchmarks/{task_name}/benchmark.json").read_text()).get("default_limit")
expected_limit = declared_limit if isinstance(declared_limit, int) and declared_limit > 0 else None
assert config.default_limit(task_name) == expected_limit, (config.default_limit(task_name), expected_limit)
assert config.default_limit("no__such_task") is None
print("PROGRESS_OK")
PY
)"
check "progress parser handles the real log format" 'PROGRESS_OK' "$OUT"

echo "=== 7. baseline cache (shared across runs) ==="
OUT="$("${CLIENT[@]}" baseline --token="$TOKEN_A" --score 20.4 2>&1)"
check "baseline can be set" '"baseline_score": 20.4' "$OUT"
OUT="$("${CLIENT[@]}" baseline --token="$TOKEN_A" 2>&1)"
check "baseline can be read back" '"baseline_score": 20.4' "$OUT"

echo "=== 8. unregister ==="
OUT="$("${CLIENT[@]}" unregister --token="$TOKEN_A" 2>&1)"
check "unregister succeeds" 'unregistered' "$OUT"
OUT="$("${CLIENT[@]}" submit --token="$TOKEN_A" --model-path output/v1 --mode val 2>&1)"
check "token no longer works after unregister" 'unknown or expired token' "$OUT"

echo ""
echo "=== 9. interruption safety: service killed mid-evaluation ==="
# An evaluation child runs in its own session, so it survives the service dying and
# would keep the eval card (and its overlay mount) allocated. The attempt that must
# hang is evaluate.py, so a fixture VTB_REPO holds a $TASK whose evaluate.py is a plain
# sleep.
mkdir -p "$TMP/repoD/benchmarks/$TASK" "$TMP/fakehf" "$TMP/runD/task/output/v1" "$TMP/runD/eval"
echo '{}' > "$TMP/runD/task/output/v1/config.json"
# execvp, not subprocess: the sleep replaces this python, so `ps` shows exactly
# `sleep 987`.
cat > "$TMP/repoD/benchmarks/$TASK/evaluate.py" <<'FAKE'
import os
os.execvp("sleep", ["sleep", "987"])
FAKE

PORT2=$((PORT + 1))
CLIENT2=("$VTB_PYTHON" -m eval.grading.client --base "http://127.0.0.1:${PORT2}")
export VTB_ADMIN_TOKEN_FILE="$TMP/state2/admin.token"
VTB_REPO="$TMP/repoD" \
VTB_GRADING_STATE="$TMP/state2" \
VTB_GRADING_SETTLE=0.2 \
VTB_VAL_SOURCE=split:5 \
VTB_EVAL_GPU="" \
    VTB_SELF_SERVED_BENCHMARKS="$TASK" \
HF_HOME="$TMP/fakehf" \
    "$VTB_PYTHON" -m eval.grading.server --host 127.0.0.1 --port "$PORT2" > "$TMP/server2.log" 2>&1 &
SRV2_PID=$!

for _ in $(seq 1 40); do
    "${CLIENT2[@]}" health >/dev/null 2>&1 && break
    sleep 0.25
done

check "pid file written" "$SRV2_PID" "$(cat "$TMP/state2/service.pid" 2>&1)"

TOKEN_D="$("${CLIENT2[@]}" register --run-id smokeD --task "$TASK" \
    --base-model Qwen/Qwen3-1.7B-Base --sandbox-root "$TMP/runD/task" --eval-dir "$TMP/runD/eval" 2>&1)"
"${CLIENT2[@]}" submit --token="$TOKEN_D" --model-path output/v1 --mode val >/dev/null 2>&1
sleep 3

# Match only a process whose entire argument string is the stand-in.
standin_running() { ps -eo args= | grep -c '^sleep 987$'; }

check "stand-in evaluation child is running" '^1$' "$(standin_running)"
check "overlay was mounted for the evaluation" 'merged_huggingface' "$(mount | grep "$TMP/state2" || echo none)"

kill -TERM "$SRV2_PID" 2>/dev/null
for _ in $(seq 1 40); do
    kill -0 "$SRV2_PID" 2>/dev/null || break
    sleep 0.25
done
sleep 1

check "evaluation child was killed with the service" '^0$' "$(standin_running)"
check_absent "overlay mount was released" 'merged_huggingface' "$(mount | grep "$TMP/state2" || echo clean)"
check "shutdown killed the evaluation tree" 'killed evaluation pid' "$(cat "$TMP/server2.log")"
check "in-flight job was marked failed" 'marked in-flight job' "$(cat "$TMP/server2.log")"
check "shutdown was logged as clean" 'grading service stopped' "$(cat "$TMP/server2.log")"
check_absent "pid file was removed" '[0-9]' "$(cat "$TMP/state2/service.pid" 2>/dev/null || echo gone)"
check "the interrupted submission left a record" 'interrupted' "$(cat "$TMP/runD/eval/submissions.json" 2>&1)"

echo ""
echo "=== 10. startup sweep of debris from an unclean exit ==="
# SIGKILL cannot be handled, so leftovers must be found at startup instead.
mkdir -p "$TMP/state2/work/deadjob"/{merged_huggingface,upper_huggingface,fuse_workdir}
touch "$TMP/state2/work/deadjob/eval_p1_a1.log"
VTB_REPO="$REPO" VTB_GRADING_STATE="$TMP/state2" "$VTB_PYTHON" - <<'PY' > "$TMP/sweep.log" 2>&1
import sys
sys.path.insert(0, ".")
from eval.grading import executor
for action in executor.sweep_stale_overlays(lambda m: print(m)):
    print("ACTION", action)
PY
check "sweep removed the stale overlay scratch" 'ACTION removed stale' "$(cat "$TMP/sweep.log")"
check_absent "...and kept the attempt log as evidence" 'no log' \
    "$([ -f "$TMP/state2/work/deadjob/eval_p1_a1.log" ] && echo 'log kept' || echo 'no log')"

echo ""
echo "=== 11. the two-token contract run_task.sh depends on ==="
# run_task.sh registers twice: the agent token is confined to its own workspace (so it
# cannot name the delivered copy under results/), the official token to EVAL_DIR (so
# the verified final_model can be scored, from outside the agent container).
mkdir -p "$TMP/repoE/benchmarks/$TASK" "$TMP/fakehf3" "$TMP/runE/task/output/v1" "$TMP/runE/eval/final_model"
echo '{}' > "$TMP/runE/task/output/v1/config.json"
echo '{}' > "$TMP/runE/eval/final_model/config.json"
# A stand-in that succeeds, so --write-metrics runs end to end. It echoes its command
# line, which the final_eval.log assertion below reads.
cat > "$TMP/repoE/benchmarks/$TASK/evaluate.py" <<'FAKE'
import json
import sys

print("stand-in evaluation ran with: " + " ".join(sys.argv[1:]), flush=True)
out = None
for i, arg in enumerate(sys.argv):
    if arg == "--json-output-file" and i + 1 < len(sys.argv):
        out = sys.argv[i + 1]
if out:
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"accuracy": 0.5, "stderr": 0.1}, fh)
FAKE

PORT3=$((PORT + 2))
CLIENT3=("$VTB_PYTHON" -m eval.grading.client --base "http://127.0.0.1:${PORT3}")
export VTB_ADMIN_TOKEN_FILE="$TMP/state3/admin.token"
VTB_REPO="$TMP/repoE" \
VTB_GRADING_STATE="$TMP/state3" \
VTB_GRADING_SETTLE=0.2 \
VTB_VAL_SOURCE=split:5 \
VTB_EVAL_GPU="" \
    VTB_SELF_SERVED_BENCHMARKS="$TASK" \
    VTB_SERVE_SGLANG_IMAGE="vtb-smoke/does-not-exist:latest" \
HF_HOME="$TMP/fakehf3" \
    "$VTB_PYTHON" -m eval.grading.server --host 127.0.0.1 --port "$PORT3" > "$TMP/server3.log" 2>&1 &
SRV3_PID=$!
for _ in $(seq 1 40); do
    "${CLIENT3[@]}" health >/dev/null 2>&1 && break
    sleep 0.25
done

# A missing serving image (the default backend's, VTB_SERVE_SGLANG_IMAGE above) is
# warned about at startup, not at grading time, and must not stop a service that --
# like this one -- serves no model.
check "a missing serving image is reported at startup, not at grading time" \
    'vtb-smoke/does-not-exist:latest' "$(cat "$TMP/server3.log")"
check "...and names the command that builds it" 'build_image.sh sglang' "$(cat "$TMP/server3.log")"

"${CLIENT3[@]}" register --run-id 'smokeE#agent' --task "$TASK" --base-model Qwen/Qwen3-1.7B-Base \
    --sandbox-root "$TMP/runE/task" --eval-dir "$TMP/runE/eval" --token-file "$TMP/tok_agent" >/dev/null 2>&1
"${CLIENT3[@]}" register --run-id 'smokeE#official' --task "$TASK" --base-model Qwen/Qwen3-1.7B-Base \
    --sandbox-root "$TMP/runE/eval" --eval-dir "$TMP/runE/eval" --token-file "$TMP/tok_official" >/dev/null 2>&1

OUT="$("${CLIENT3[@]}" submit --token-file "$TMP/tok_agent" --model-path "$TMP/runE/eval/final_model" --mode final 2>&1)"
check "agent token cannot reach the delivered copy under results/" 'escapes the run sandbox' "$OUT"

OUT="$("${CLIENT3[@]}" submit --token-file "$TMP/tok_agent" --model-path output/v1 --mode val --wait --wait-timeout 120 2>&1)"
check "agent token can submit from its own workspace" '"state": "done"' "$OUT"

# Exactly what run_task.sh runs: relative to the official sandbox root (EVAL_DIR).
OUT="$("${CLIENT3[@]}" submit --token-file "$TMP/tok_official" --model-path final_model \
    --mode final --write-metrics --wait --wait-timeout 120 2>&1)"
check "official token scores the delivered copy" '"state": "done"' "$OUT"
# No baseline is asserted: no baseline was measured for this benchmark, so the
# report is null rather than a number.
check "a missing baseline is reported, not invented" '"baseline_score": null' "$OUT"
check "official metrics.json written for the scratch gate" '"accuracy": 0.5' \
    "$(cat "$TMP/runE/eval/metrics.json" 2>&1)"
check "evaluation log preserved as final_eval.log" 'stand-in evaluation ran' \
    "$(cat "$TMP/runE/eval/final_eval.log" 2>&1)"
check "a validation submission stayed out of best_final" '"mode": "val"' \
    "$(cat "$TMP/runE/eval/submissions.json" 2>&1)"

OUT="$("${CLIENT3[@]}" unregister --token-file "$TMP/tok_official" 2>&1)"
check "official token can be released" 'unregistered' "$OUT"

echo ""
echo "=== 12. progress reaches a redirected stderr while still running ==="
# Earlier assertions capture stderr into a shell variable (a pipe); run_task.sh
# redirects it to a file, which on python 3.8 is block-buffered -- unflushed progress
# lines never appeared. This checks the file path.
PROG="$TMP/progress_stderr.log"
"${CLIENT3[@]}" submit --token-file "$TMP/tok_agent" --model-path output/v1 \
    --mode val --wait --wait-timeout 120 --poll 1 > "$TMP/progress_stdout.json" 2> "$PROG"
check "progress lines land in a redirected stderr" 'grading' "$(cat "$PROG" 2>&1)"
check "...and name the phase" 'generating\|launching\|starting\|loading\|done' "$(cat "$PROG" 2>&1)"

# A final submission needs a denominator too: --limit -1 (whole dataset), whose size
# inspect records in its own log.
OUT="$(cd "$REPO" && "$VTB_PYTHON" - "$TMP" <<'PY' 2>&1
import json, sys
sys.path.insert(0, ".")
from pathlib import Path
from eval.grading import executor

work = Path(sys.argv[1]) / "fake_job"
logs = executor.inspect_log_dir(work)
logs.mkdir(parents=True, exist_ok=True)
# Shape from a real inspect log: dataset size is recorded even when --limit trimmed
# the run (a 150-sample log still reports samples=1319).
(logs / "run.json").write_text(json.dumps({
    "eval": {"dataset": {"name": "aime/aime", "samples": 1319,
                         "sample_ids": ["sample_a", "sample_b"]}},
    "samples": [{"id": "sample_a", "scores": {"match": {"value": "C"}}}],
}))
assert executor.dataset_total(work) == 1319, executor.dataset_total(work)
assert executor.dataset_total(Path(sys.argv[1]) / "no_such_job") is None
assert "inspect_logs" in str(executor.inspect_log_dir(work))
assert "tasks" not in str(executor.inspect_log_dir(work)), executor.inspect_log_dir(work)
print("TOTAL_OK")
PY
)"
check "dataset total is read from inspect's log" 'TOTAL_OK' "$OUT"
check_absent "inspect logs are not written into the checkout" 'tasks' \
    "$(cd "$REPO" && "$VTB_PYTHON" -c "
import sys; sys.path.insert(0,'.')
from pathlib import Path
from eval.grading import executor
print(executor.inspect_log_dir(Path('/somewhere/work/job1')))")"

echo ""
echo "=== 13. an agent submits a model and nothing else ==="
export VTB_ADMIN_TOKEN_FILE="$TMP/state/admin.token"
# The service decides which samples an agent is scored on, so accepting mode or
# write_metrics would hand that control back. `limit` is the one exception: a run may
# ask for fewer samples to get a faster answer, which cannot reach outside the split
# because the split is a prefix and a smaller prefix is a subset of it.
mkdir -p "$TMP/runF/task/output/v1" "$TMP/runF/eval"
AGENT_TOK="$TMP/runF/agent.token"
"${CLIENT[@]}" register --run-id runF#agent --task "$TASK" --base-model Qwen/Qwen3-1.7B-Base \
    --sandbox-root "$TMP/runF/task" --eval-dir "$TMP/runF/eval" --role agent \
    --token-file "$AGENT_TOK" >/dev/null 2>&1

for arg in "--mode final" "--mode val" "--write-metrics"; do
    OUT="$("${CLIENT[@]}" submit --token-file "$AGENT_TOK" --model-path output/v1 $arg 2>&1)"
    check "an agent may not send ${arg%% *}" 'not accepted from an agent submission' "$OUT"
done

# The split here is 5 samples (VTB_VAL_SOURCE=split:5). A bigger limit does not buy a
# bigger evaluation, and the response says which number won -- an agent that believed
# it bought 999 would later compare that against a figure that meant something else.
OUT="$("${CLIENT[@]}" submit --token-file "$AGENT_TOK" --model-path output/v1 --limit 999 2>&1)"
check "an agent's limit cannot exceed the validation split" '"limit": 5' "$OUT"
OUT="$("${CLIENT[@]}" submit --token-file "$AGENT_TOK" --model-path output/v1 --limit 2 2>&1)"
check "...but a smaller one is honoured" '"limit": 2' "$OUT"
OUT="$("${CLIENT[@]}" submit --token-file "$AGENT_TOK" --model-path output/v1 --limit 0 2>&1)"
check "...and a non-positive limit is refused" 'must be a positive integer' "$OUT"

# Bare submission is accepted, and its result is reduced to the score alone.
OUT="$("${CLIENT[@]}" submit --token-file "$AGENT_TOK" --model-path output/v1 \
    --wait --wait-timeout 120 2>&1)"
check "a bare agent submission is accepted" '"state"' "$OUT"
check_absent "...and the host model path is withheld" "$TMP" "$OUT"
check_absent "...and other runs' identifiers are withheld" '"run_id"' "$OUT"
check_absent "...and attempt internals are withheld" 'attempt_log' "$OUT"

echo ""
echo "=== 14. validation / held-out partitioning ==="
OUT="$(cd "$REPO" && "$VTB_PYTHON" - <<'PY' 2>&1
import json, sys, tempfile
sys.path.insert(0, ".")
from pathlib import Path
from eval.grading import valsplit as V

# Shape from a real agentic SWE log: ordered sample_ids, per-sample scores under a
# task-specific scorer name, and the accuracy inspect computed itself.
ids = [f"s{i:04d}" for i in range(10)]
correct = {i: (n % 2 == 0) for n, i in enumerate(ids)}   # 5 of 10
log = Path(tempfile.mkdtemp()) / "log.json"
log.write_text(json.dumps({
    "eval": {"dataset": {"samples": 10, "sample_ids": ids}},
    "results": {"scores": [{"metrics": {"accuracy": {"value": 0.5}}}]},
    "samples": [{"id": i, "scores": {"some_task_scorer": {"value": "C" if correct[i] else "I"}}}
                for i in ids],
}))
r = V.partition_log(log, 3)
assert r["reconstruction_ok"], r
assert r["accuracy_full"] == 0.5, r
assert r["val_count"] == 3 and r["heldout_count"] == 7, r
assert r["accuracy_val"] == 2 / 3, r          # s0000, s0002 correct of first 3
assert r["accuracy_heldout"] == 3 / 7, r
assert r["val_count"] + r["heldout_count"] == r["dataset_total"], r

# A scorer this code cannot read must not yield a plausible-looking number.
bad = Path(tempfile.mkdtemp()) / "bad.json"
bad.write_text(json.dumps({
    "eval": {"dataset": {"samples": 2, "sample_ids": ["a", "b"]}},
    "results": {"scores": [{"metrics": {"accuracy": {"value": 0.9}}}]},
    "samples": [{"id": "a", "scores": {"x": {"value": "C"}}},
                {"id": "b", "scores": {"x": {"value": "C"}}}],
}))
r2 = V.partition_log(bad, 1)
assert not r2["reconstruction_ok"], r2
assert r2["accuracy_heldout"] is None, r2
print("SPLIT_OK")
PY
)"
check "held-out figures are derived from one full log" 'SPLIT_OK' "$OUT"

echo ""
echo "=== 15. evaluation logs are kept, and kept away from the agent ==="
export VTB_ADMIN_TOKEN_FILE="$TMP/state/admin.token"
# A validation log records per-sample outcomes and model outputs on the split. It is
# kept for explainability, but never anywhere the agent can read.
mkdir -p "$TMP/runG/task/output/v1" "$TMP/runG/eval"

OUT="$("${CLIENT[@]}" register --run-id runG#bad --task "$TASK" --base-model Qwen/Qwen3-1.7B-Base \
    --sandbox-root "$TMP/runG/task" --eval-dir "$TMP/runG/eval" --role agent \
    --log-dir "$TMP/runG/task/logs" 2>&1)"
check "a log dir inside the sandbox is refused" 'inside the run sandbox' "$OUT"

OUT="$("${CLIENT[@]}" register --run-id runG#alsobad --task "$TASK" --base-model Qwen/Qwen3-1.7B-Base \
    --sandbox-root "$TMP/runG/task" --eval-dir "$TMP/runG/eval" --role agent \
    --log-dir "$TMP/runG/task" 2>&1)"
check "...and so is the sandbox itself" 'inside the run sandbox' "$OUT"

AGENT_TOK2="$TMP/runG/agent.token"
"${CLIENT[@]}" register --run-id runG#agent --task "$TASK" --base-model Qwen/Qwen3-1.7B-Base \
    --sandbox-root "$TMP/runG/task" --eval-dir "$TMP/runG/eval" --role agent \
    --log-dir "$TMP/runG/grading_logs" --token-file "$AGENT_TOK2" >/dev/null 2>&1
check "a sibling log dir is accepted" '32' "$(wc -c < "$AGENT_TOK2" 2>/dev/null)"

"${CLIENT[@]}" submit --token-file "$AGENT_TOK2" --model-path output/v1 \
    --wait --wait-timeout 120 >/dev/null 2>&1
# Attempts fail on the missing backend tool, so no inspect log is written: what is
# asserted is the destination -- log dir outside the sandbox, sandbox free of logs.
check "the log dir was created outside the sandbox" 'yes' \
    "$([ -d "$TMP/runG/grading_logs" ] && echo yes || echo no)"
check_absent "no evaluation log appeared inside the agent workspace" 'json' \
    "$(find "$TMP/runG/task" -name '*.json' 2>/dev/null)"

echo ""
echo "=== 16. an agent cannot reach the held-out samples at all ==="
OUT="$(cd "$REPO" && "$VTB_PYTHON" - <<'PY' 2>&1
import sys
sys.path.insert(0, ".")
from eval.grading import valsplit as V

# The validation split is a prefix, and an agent choosing a smaller limit can only
# take less of it, so the held-out remainder is unreachable however often it submits.
# An agent may also *name* samples, which the prefix alone no longer protects: that
# path is checked separately below, against the recorded split membership.
source = V.parse_val_source("split:30%")
for total in (100, 164, 1319):
    n = V.resolve_val_n(source, total)
    assert 0 < n < total, (total, n)
    ids = [f"s{i}" for i in range(total)]
    val, held = set(ids[:n]), set(ids[n:])
    assert not (val & held), total
    assert len(val) + len(held) == total, total
# An unknown dataset size yields no boundary, forcing a refusal rather than a guess.
assert V.resolve_val_n(source, None) is None
assert V.resolve_val_n(source, 0) is None
print("REACH_OK")
PY
)"
check "validation and held-out stay disjoint at every size" 'REACH_OK' "$OUT"

# Naming samples removes the prefix's protection, so it is replaced by a membership
# check against the recorded split -- and by a refusal while the split is unknown,
# since a guess there would score a submission on held-out samples.
OUT="$(cd "$REPO" && VTB_GRADING_STATE="$TMP/state16" "$VTB_PYTHON" - <<'PY' 2>&1
import sys
sys.path.insert(0, ".")
from eval.grading.store import TaskStats

stats = TaskStats()
task = "some__task"
assert stats.val_ids(task) is None, "membership must start out unknown"

# A run that left samples ungraded produces a short list. Recording it would later
# refuse legitimate validation samples, so it is not recorded at all.
stats.observe_val_ids(task, [f"s{i}" for i in range(8)], 10)
assert stats.val_ids(task) is None, "an incomplete split must not be recorded"

stats.observe_val_ids(task, [f"s{i}" for i in range(10)], 10)
known = stats.val_ids(task)
assert known == [f"s{i}" for i in range(10)], known
assert "s10" not in known and "s99" not in known

# Membership is a set: the backend's dataset order and the per-sample display order
# may differ without moving any sample across the validation boundary.
warning = stats.observe_val_ids(task, [f"s{i}" for i in range(9, -1, -1)], 10)
assert warning is None, warning
assert set(stats.val_ids(task)) == {f"s{i}" for i in range(10)}
print("MEMBERSHIP_OK")
PY
)"
check "named samples are checked against a recorded split, never a guessed one" 'MEMBERSHIP_OK' "$OUT"

echo ""
echo "=== 17. issuing a token is operator-only ==="
export VTB_ADMIN_TOKEN_FILE="$TMP/state/admin.token"
BASE="http://127.0.0.1:${PORT}"
# /register is admin-only: otherwise anything that can reach the port -- including the
# agent container, which shares the host network -- could mint a role=official token.
OUT="$(curl -s -X POST "$BASE/register" -H 'Content-Type: application/json' \
    -d "{\"run_id\":\"pwn\",\"task\":\"$TASK\",\"base_model\":\"Qwen/Qwen3-1.7B-Base\",\"sandbox_root\":\"$TMP/runA/task\",\"eval_dir\":\"$TMP/runA/eval\",\"role\":\"official\"}" 2>&1)"
check "registration without the admin token is refused" 'operator-only' "$OUT"
check_absent "...and issues no token" '"token"' "$OUT"

OUT="$(curl -s -X POST "$BASE/register" -H 'Content-Type: application/json' \
    -d "{\"admin_token\":\"wrong-secret\",\"run_id\":\"pwn2\",\"task\":\"$TASK\",\"base_model\":\"Qwen/Qwen3-1.7B-Base\",\"sandbox_root\":\"$TMP/runA/task\",\"eval_dir\":\"$TMP/runA/eval\",\"role\":\"official\"}" 2>&1)"
check "a wrong admin token is refused too" 'operator-only' "$OUT"

check "the admin token file is mode 0600" '600' \
    "$(stat -c '%a' "$TMP/state/admin.token" 2>&1)"

# /health stays open, but the resolved host paths (ingredients for a sandbox_root)
# are gated behind the admin token.
OUT="$(curl -s "$BASE/health" 2>&1)"
check "/health answers without the admin token" '"status": "ok"' "$OUT"
check_absent "...but withholds the host paths" 'vtb_root' "$OUT"
check_absent "...and the state directory" 'state_dir' "$OUT"

ADMIN="$(cat "$TMP/state/admin.token")"
OUT="$(curl -s "$BASE/health?admin_token=$ADMIN" 2>&1)"
check "...and reveals them to the operator" 'vtb_root' "$OUT"

# Rewriting a baseline rewrites every improvement measured against it.
OUT="$(curl -s -X POST "$BASE/baseline" -H 'Content-Type: application/json' \
    -d "{\"token\":\"$TOKEN_A\",\"score\":0.999}" 2>&1)"
check "rewriting a baseline needs the admin token" 'operator-only' "$OUT"

echo ""
echo "=== 18. a half-written checkpoint is refused, not evaluated ==="
# Nothing sequences the agent's write against the evaluation's read. safetensors makes
# truncation decidable from the header, so it is caught at submission, not inside vLLM.
OUT="$(cd "$REPO" && "$VTB_PYTHON" - "$TMP" <<'PY' 2>&1
import json, struct, sys
sys.path.insert(0, ".")
from pathlib import Path
from eval.grading.store import check_weights_complete

root = Path(sys.argv[1]) / "weights"

def write(name, payload_len, actual_extra):
    """Real safetensors layout: u64 header length, header JSON, then the tensors."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"w": {"dtype": "F32", "shape": [payload_len // 4],
                               "data_offsets": [0, payload_len]}}).encode()
    with open(d / "model.safetensors", "wb") as fh:
        fh.write(struct.pack("<Q", len(header)))
        fh.write(header)
        fh.write(b"\0" * actual_extra)
    return d

complete = write("complete", 4096, 4096)
assert check_weights_complete(complete) is None, check_weights_complete(complete)

truncated = write("truncated", 4096, 100)
reason = check_weights_complete(truncated)
assert reason and "still being written" in reason, reason

stub = root / "headerless"
stub.mkdir(parents=True, exist_ok=True)
(stub / "model.safetensors").write_bytes(b"\x00\x01")
assert check_weights_complete(stub), "a 2-byte file must not pass"

# A shard index naming a file that has not appeared yet.
sharded = root / "sharded"
sharded.mkdir(parents=True, exist_ok=True)
(sharded / "model.safetensors.index.json").write_text(json.dumps(
    {"weight_map": {"a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors"}}))
reason = check_weights_complete(sharded)
assert reason and "not on disk yet" in reason, reason

# Narrowness matters as much as strictness: a directory with no safetensors is a
# legitimate layout (.bin checkpoints, adapter-only dirs) and must not be refused.
other = root / "bin_only"
other.mkdir(parents=True, exist_ok=True)
(other / "pytorch_model.bin").write_bytes(b"\0" * 128)
(other / "config.json").write_text("{}")
assert check_weights_complete(other) is None, check_weights_complete(other)
print("WEIGHTS_OK")
PY
)"
check "truncation is detected from the header, complete files pass" 'WEIGHTS_OK' "$OUT"

# The refusal reaches the caller as a submission error (TOKEN_A was released in 8).
TOKEN_W="$("${CLIENT[@]}" register --run-id smokeW --task "$TASK" \
    --base-model Qwen/Qwen3-1.7B-Base --sandbox-root "$TMP/runA/task" \
    --eval-dir "$TMP/runA/eval" 2>/dev/null | tail -1)"
mkdir -p "$TMP/runA/task/output/partial"
"$VTB_PYTHON" -c "
import json, struct
from pathlib import Path
h = json.dumps({'w': {'dtype':'F32','shape':[1024],'data_offsets':[0,4096]}}).encode()
p = Path('$TMP/runA/task/output/partial/model.safetensors')
with open(p,'wb') as fh:
    fh.write(struct.pack('<Q', len(h))); fh.write(h); fh.write(b'\0'*10)
"
OUT="$("${CLIENT[@]}" submit --token="$TOKEN_W" --model-path output/partial 2>&1)"
check "submitting it is refused with the reason" 'not finished being written' "$OUT"
check_absent "...and no job was queued for it" 'job_id' "$OUT"

echo ""
echo "=== 19. held-out scoring works for a harness that is not inspect-ai ==="
# Held-out reporting is a declared contract, not an inspect-log privilege: an SWE-style
# benchmark with its own runner writes no inspect log, so it declares the shape instead.
OUT="$(cd "$REPO" && "$VTB_PYTHON" - "$TMP" <<'PY' 2>&1
import json, sys
sys.path.insert(0, ".")
from pathlib import Path
from eval.grading import valsplit as V

work = Path(sys.argv[1]) / "generic"
work.mkdir(parents=True, exist_ok=True)
metrics = work / "metrics_p1_a1.json"
metrics.write_text(json.dumps({"accuracy": 0.5}))

ids = [f"task-{i:03d}" for i in range(10)]
V.per_sample_path(metrics).write_text(json.dumps({
    "dataset_total": 10,
    "sample_ids": ids,
    # A mix of the accepted shapes: booleans and numbers, as a rubric-graded or
    # test-suite-driven harness would naturally report.
    "correct": {i: (True if n % 2 == 0 else 0.0) for n, i in enumerate(ids)},
}))

res = V.read_per_sample_file(V.per_sample_path(metrics))
assert res is not None and res.source == "per_sample.json", res
assert res.dataset_total == 10, res.dataset_total

r = V.partition(res, 3, reported_accuracy=0.5)
assert r["reconstruction_ok"], r
assert r["source"] == "per_sample.json", r
assert r["accuracy_val"] == 2 / 3 and r["accuracy_heldout"] == 3 / 7, r
assert r["val_count"] + r["heldout_count"] == 10, r

# The reconstruction check holds for the open contract too: a per-sample file that
# disagrees with its own accuracy must not publish a held-out number.
bad = V.partition(res, 3, reported_accuracy=0.9)
assert not bad["reconstruction_ok"], bad
assert bad["accuracy_heldout"] is None, bad

# A partial value counts as correct only at 1.0.
assert V.read_per_sample_file(V.per_sample_path(metrics)).correct_by_id["task-001"] is False

# Malformed or absent declarations degrade to None rather than raising.
assert V.read_per_sample_file(work / "nope.json") is None
(work / "junk.per_sample.json").write_text("not json")
assert V.read_per_sample_file(work / "junk.per_sample.json") is None
(work / "half.per_sample.json").write_text(json.dumps({"sample_ids": ["a"]}))
assert V.read_per_sample_file(work / "half.per_sample.json") is None
print("GENERIC_OK")
PY
)"
check "a declared per-sample file yields the same split" 'GENERIC_OK' "$OUT"

# A benchmark added without a ladder entry retries with an unchanged budget -- which is
# now what *every* entry does, since no phase lowers a step count. The entry is therefore
# bookkeeping rather than behaviour, and the check stays only to catch a benchmark added
# without anyone considering its retries.
OUT="$(cd "$REPO" && "$VTB_PYTHON" -c "
import sys; sys.path.insert(0,'.')
from eval.grading import config
# A hypothetical new benchmark must be reported, and its token_args must be empty.
assert config.tasks_without_ladder(['$TASK', 'swebench_variant']) == ['swebench_variant']
assert config.token_args('swebench_variant', 2) == []
assert config.token_args('swebench_variant', 3) == []
# And a laddered task no longer gets a lowered step budget in any phase.
assert config.token_args('$TASK', 1) == []
assert config.token_args('$TASK', 2) == []
assert config.token_args('$TASK', 3) == []
# Every benchmark currently installed is covered.
assert config.tasks_without_ladder() == [], config.tasks_without_ladder()
print('LADDER_OK')")"
check "a benchmark with no ladder entry is reported, and no phase lowers a step budget" 'LADDER_OK' "$OUT"

echo ""
echo "=== 20. nothing a benchmark hands the agent contains its answers ==="
# run_task.sh copies task_context/, templates/ and evaluation_code/ into the agent
# workspace. A benchmark may hand over upstream grading code, but not reference
# patches, hidden tests, or the list of tests that must pass. Checked for every
# benchmark so the next one added cannot reintroduce a leak quietly.
OUT="$(cd "$REPO" && "$VTB_PYTHON" - <<'PY' 2>&1
from pathlib import Path

# The directories run_task.sh copies into ${JOB_DIR}/task.
COPIED = ("task_context", "templates", "evaluation_code")

def answer_bearing(relative: Path) -> bool:
    """Whether this path would hand a Harbor task's answer to the agent.

    Three shapes, and the reason for each:
      solution/**    the reference patch and the script that applies it
      *.patch        the reference diff, and tests/test.patch is the hidden suite
      tests/**       the hidden tests, and tests/config.json names exactly which
                     test functions must start passing -- a specification of the
                     answer even without the tests themselves

    Not keyed on file names alone. A benchmark may legitimately ship grading code --
    an LLM-as-judge grader.py that pulls its rubric from the dataset at run time carries
    no answers, for instance -- so matching a bare name like grader.py would flag a
    benign file, and relaxing the whole check to let it pass would be the wrong repair.
    The shapes above are structural for that reason.
    """
    parts = relative.parts
    if "solution" in parts or "tests" in parts:
        return True
    return relative.name.endswith(".patch")

leaks = []
for task_dir in sorted(Path("benchmarks").iterdir()):
    if not (task_dir / "evaluate.py").is_file():
        continue
    for copied in COPIED:
        root = task_dir / copied
        if not root.is_dir():
            continue
        for entry in root.rglob("*"):
            if entry.is_file() and answer_bearing(entry.relative_to(root)):
                leaks.append(f"{task_dir.name}/{copied}/{entry.relative_to(root)}")
if leaks:
    print(f"LEAK {len(leaks)} entries, first few: {leaks[:5]}")
else:
    print("NO_ANSWER_LEAK")
PY
)"
check "no benchmark copies solutions or hidden tests into the workspace" 'NO_ANSWER_LEAK' "$OUT"

# The SWE task tree holds the answers, so it must live outside those copied paths. The
# backends materialize it under VTB_ROOT; this asserts the resolved location is not
# inside benchmarks/ at all (resolved, not hardcoded, so moving VTB_ROOT is checked).
check "the materialized SWE task tree is not under a copied directory" 'OUTSIDE' \
    "$(cd "$REPO" && "$VTB_PYTHON" -c "
import sys; sys.path.insert(0,'.')
from pathlib import Path
from eval.runner import runtime_paths
specs = runtime_paths.HUB_TASK_SPECS_DIR.resolve()
benchmarks = Path('benchmarks').resolve()
print('OUTSIDE' if benchmarks not in specs.parents and specs != benchmarks else f'INSIDE {specs}')")"

echo ""
echo "=== 21. a run cancels its own evaluation and gets the GPUs back ==="
# The retry ladder is the whole difficulty here: killing an evaluation looks exactly
# like one that failed for an infrastructure reason, and the ladder answers a failure
# with the next of its nine attempts -- which would serve the model again and take the
# cards straight back. Same hanging stand-in as section 9, so no GPU and no real work.
mkdir -p "$TMP/repoH/benchmarks/$TASK" "$TMP/fakehfH" "$TMP/runH/task/output/v1" "$TMP/runH/eval"
echo '{}' > "$TMP/runH/task/output/v1/config.json"
cat > "$TMP/repoH/benchmarks/$TASK/evaluate.py" <<'FAKE'
import os
os.execvp("sleep", ["sleep", "9871"])
FAKE

# Its own port and state: section 11's service is still listening on PORT3, and
# reusing it would silently point this client at that service's fixture instead.
PORT4=$((PORT + 3))
CLIENT4=("$VTB_PYTHON" -m eval.grading.client --base "http://127.0.0.1:${PORT4}")
export VTB_ADMIN_TOKEN_FILE="$TMP/state4/admin.token"
VTB_REPO="$TMP/repoH" \
VTB_GRADING_STATE="$TMP/state4" \
VTB_GRADING_SETTLE=0.2 \
VTB_VAL_SOURCE=split:5 \
VTB_EVAL_GPU="" \
VTB_MAX_AGENT_CANCELS=2 \
    VTB_SELF_SERVED_BENCHMARKS="$TASK" \
HF_HOME="$TMP/fakehfH" \
    "$VTB_PYTHON" -m eval.grading.server --host 127.0.0.1 --port "$PORT4" > "$TMP/server4.log" 2>&1 &
SRV4_PID=$!

for _ in $(seq 1 40); do
    "${CLIENT4[@]}" health >/dev/null 2>&1 && break
    sleep 0.25
done

AGENT_TOK_H="$TMP/runH/agent.token"
"${CLIENT4[@]}" register --run-id runH#agent --task "$TASK" --base-model Qwen/Qwen3-1.7B-Base \
    --sandbox-root "$TMP/runH/task" --eval-dir "$TMP/runH/eval" --role agent \
    --token-file "$AGENT_TOK_H" >/dev/null 2>&1

check "the cancellation policy is advertised" 'max_agent_cancels' "$("${CLIENT4[@]}" health 2>&1)"

SUB="$("${CLIENT4[@]}" submit --token-file "$AGENT_TOK_H" --model-path output/v1 2>&1)"
JOB_H="$(printf '%s' "$SUB" | "$VTB_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["job_id"])' 2>/dev/null)"
check "the submission reports the cancellation quota" 'cancel_quota' "$SUB"
sleep 3

standin_h_running() { ps -eo args= | grep -c '^sleep 9871$'; }
check "the evaluation is running" '^1$' "$(standin_h_running)"

OUT="$("${CLIENT4[@]}" cancel --token-file "$AGENT_TOK_H" --job-id "$JOB_H" \
    --reason 'training is ready and wants the cards' 2>&1)"
check "cancelling reports the job as cancelled" '"state": "cancelled"' "$OUT"
check "...and says the evaluation cards were released" '"released": true' "$OUT"
check "...and warns the partial result is not comparable" 'diagnosis only' "$OUT"
check "the evaluation child is gone" '^0$' "$(standin_h_running)"
check "the teardown was logged" 'releasing the evaluation' "$(cat "$TMP/server4.log")"

# The point of the flag: exactly one attempt ran, and the ladder did not answer the
# kill with another one.
check "the ladder stopped instead of retrying" 'cancelled by the run' "$(cat "$TMP/server4.log")"
check_absent "...and never started a second attempt" 'phase 1 attempt 2' "$(cat "$TMP/server4.log")"
check "the cancelled state reached /result" 'cancelled' \
    "$("${CLIENT4[@]}" health >/dev/null 2>&1; curl -s "http://127.0.0.1:${PORT4}/result?token=$(cat "$AGENT_TOK_H")&job_id=${JOB_H}" 2>&1)"

RECORD="$(cat "$TMP/runH/eval/submissions.json" 2>&1)"
check "the run's record says why it was cancelled" 'wants the cards' "$RECORD"
# Null on purpose: SubmissionLog.best() selects on a numeric score, so a cancelled
# evaluation cannot be chosen as a candidate whatever else is recorded beside it.
check "...and carries no score" '"score": null' "$RECORD"
check_absent "...and is not recorded as an error" '"error"' "$RECORD"

OUT="$("${CLIENT4[@]}" cancel --token-file "$AGENT_TOK_H" --job-id "$JOB_H" --reason again 2>&1)"
check "cancelling a finished job is refused" 'already cancelled' "$OUT"
OUT="$(curl -s -X POST "http://127.0.0.1:${PORT4}/cancel" -H 'Content-Type: application/json' \
    -d "{\"token\": \"$(cat "$AGENT_TOK_H")\", \"job_id\": \"$JOB_H\"}" 2>&1)"
check "cancelling without a reason is refused" 'needs a reason' "$OUT"

echo ""
echo "=== 22. a queued submission can be cancelled, and cancellations are capped ==="
# Two submissions: the first occupies the worker, so the second is still queued. A
# queued job has nothing to tear down, and the worker must skip it rather than serve a
# model nobody is waiting for.
JOB_RUN="$("${CLIENT4[@]}" submit --token-file "$AGENT_TOK_H" --model-path output/v1 2>&1 \
    | "$VTB_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["job_id"])' 2>/dev/null)"
sleep 2
JOB_QUEUED="$("${CLIENT4[@]}" submit --token-file "$AGENT_TOK_H" --model-path output/v1 2>&1 \
    | "$VTB_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["job_id"])' 2>/dev/null)"

OUT="$("${CLIENT4[@]}" cancel --token-file "$AGENT_TOK_H" --job-id "$JOB_QUEUED" \
    --reason 'queued behind a run I no longer need' 2>&1)"
check "a queued submission is cancelled without a teardown" '"was_running": false' "$OUT"
check "...and the quota shows two used" '"used": 2' "$OUT"
check "...and none remaining" '"remaining": 0' "$OUT"
check "the still-running evaluation was left alone" '^1$' "$(standin_h_running)"

# The cap is what separates handing the cards back from submitting, peeking at the
# first few samples and cancelling over and over.
OUT="$("${CLIENT4[@]}" cancel --token-file "$AGENT_TOK_H" --job-id "$JOB_RUN" \
    --reason 'one cancellation too many' 2>&1)"
check "a run past its cancellation cap is refused" 'used all 2 of its cancellations' "$OUT"
check "...and the evaluation keeps running" '^1$' "$(standin_h_running)"

kill -TERM "$SRV4_PID" 2>/dev/null
for _ in $(seq 1 40); do
    kill -0 "$SRV4_PID" 2>/dev/null || break
    sleep 0.25
done
sleep 1
SRV4_PID=""

echo ""
echo "=== 23. per-sample disclosure is complete, and bounded by what was evaluated ==="
# The cap used to be a page size: 64 rows of a 150-sample split came back and the
# other 86 were dropped with nothing in the response to say so, which left the agent
# diagnosing a prefix of an evaluation it had paid for all of.
OUT="$(cd "$REPO" && "$VTB_PYTHON" - <<'PY' 2>&1
import sys
sys.path.insert(0, ".")
from eval.grading import config, server

class FakeJob:
    job_id = "j"
    limit = 150

svc = server.GradingService.__new__(server.GradingService)
svc.log = lambda message: None
rows = [{"id": f"s{i}", "score": 0.0} for i in range(150)]

assert config.AGENT_DISCLOSURE_MAX_SAMPLES == 0, config.AGENT_DISCLOSURE_MAX_SAMPLES
disclosed = svc._disclose_rows(rows, FakeJob(), {})
assert len(disclosed) == 150, len(disclosed)

# It is still a bound: never more samples than the job was allowed to evaluate, and an
# operator may lower it further.
FakeJob.limit = 20
assert len(svc._disclose_rows(rows, FakeJob(), {})) == 20
FakeJob.limit = 150
config.AGENT_DISCLOSURE_MAX_SAMPLES = 10
assert len(svc._disclose_rows(rows, FakeJob(), {})) == 10
print("DISCLOSURE_OK")
PY
)"
check "every graded sample is disclosed, not the first 64" 'DISCLOSURE_OK' "$OUT"

# A partial is counted by the same rules as a complete run: accuracy over graded tasks,
# and trials still in flight counted as ungraded rather than as zeros.
OUT="$(cd "$REPO" && "$VTB_PYTHON" - <<'PY' 2>&1
import sys
sys.path.insert(0, ".")
from eval.runner import trial_results

records = [
    {"task_id": "a", "graded": True, "reward": 1.0},
    {"task_id": "b", "graded": True, "reward": 0.0},
    {"task_id": "c", "graded": False, "reward": None, "reason": "still running"},
]
counts = trial_results.score_records(records)
assert counts["graded"] == 2, counts
assert counts["attempted"] == 3, counts
assert counts["ungraded"] == 1, counts
# 1 of 2 graded, not 1 of 3: an unfinished trial is missing, not failed.
assert counts["accuracy"] == 0.5, counts
full = trial_results.summarise(records, benchmark="b", agent="a", model="m",
                               backend="harbor", backend_version="0", environment="docker")
assert all(full[k] == v for k, v in counts.items()), (full, counts)
print("PARTIAL_COUNT_OK")
PY
)"
check "a partial result is counted like a whole one, with no trial counted as a zero" 'PARTIAL_COUNT_OK' "$OUT"

# A cancellation keeps the trials it killed, not only the ones it finished. Those are the
# error scene: the decision to cancel is usually taken because of what they were doing, and
# a cancellation with nothing graded yet used to publish nothing at all.
OUT="$(cd "$REPO" && "$VTB_PYTHON" - <<'PY' 2>&1
import json, sys, tempfile
sys.path.insert(0, ".")
from pathlib import Path
from eval.grading import server

root = Path(tempfile.mkdtemp())
trials = {}
for name, graded in (("good", True), ("killed", False)):
    d = root / "tree" / name
    (d / "agent").mkdir(parents=True)
    (d / "agent" / "trajectory.json").write_text(json.dumps({"messages": []}))
    (d / "result.json").write_text(json.dumps({"reward": 1.0 if graded else None}))
    # Section 20's invariant applies to this path too: an unfinished trial can still have
    # a verifier directory, and it names the hidden tests.
    (d / "verifier").mkdir()
    (d / "verifier" / "report.json").write_text('{"tests_status": {"FAIL_TO_PASS": ["t"]}}')
    trials[name] = d

class FakeCtx:
    disclosure_dir = str(root / "disclosed")

class FakeJob:
    job_id = "j1"
    eval_task = "t"
    mode = "val"
    sample_ids = None
    result = {}

class FakeStats:
    def val_ids(self, task):
        return ["good", "killed"]

svc = server.GradingService.__new__(server.GradingService)
svc.log = lambda message: None
svc.task_stats = FakeStats()
svc._trial_dirs = lambda ctx, job: dict(trials)

out = svc._write_agent_disclosure(FakeCtx(), FakeJob(), ["good"], cancelled=True,
                                  ungraded_ids=["killed"])
summary = json.loads((Path(out) / "summary.json").read_text())
assert summary["sample_ids"] == ["good"], summary
# Named apart, because an unfinished trial scored nothing at all rather than zero and
# folding it into the graded ids would put it in an average it does not belong to.
assert summary["unfinished_sample_ids"] == ["killed"], summary
assert summary["samples_written"] == 2, summary
killed = Path(out) / "samples" / "killed"
assert (killed / "agent" / "trajectory.json").is_file(), sorted(killed.rglob("*"))
assert not (killed / "verifier" / "report.json").exists(), "hidden tests followed the transcript"

# Nothing graded yet is the case that matters most and used to return None.
FakeJob.job_id = "j2"
assert svc._write_agent_disclosure(FakeCtx(), FakeJob(), [], cancelled=True,
                                   ungraded_ids=["killed"]) is not None
# The official evaluation still writes nothing, cancelled or not.
FakeJob.job_id, FakeJob.mode = "j3", "final"
assert svc._write_agent_disclosure(FakeCtx(), FakeJob(), ["good"], cancelled=True,
                                   ungraded_ids=["killed"]) is None
print("CANCEL_DISCLOSURE_OK")
PY
)"
check "a cancellation discloses the trials it killed, apart from the graded ones" 'CANCEL_DISCLOSURE_OK' "$OUT"

# The grading service now copies graded trials somewhere the agent can read, which is a
# new path for section 20's invariant to be violated on: verifier/report.json carries
# tests_status.FAIL_TO_PASS, and an agent holding those test names does not need to fix
# the bug. The copy is a whitelist walked entry by entry, so this checks the whitelist
# against a fixture shaped like a real trial.
OUT="$(cd "$REPO" && "$VTB_PYTHON" - <<'PY' 2>&1
import json, sys, tempfile
sys.path.insert(0, ".")
from pathlib import Path
from eval.runner import trial_results

root = Path(tempfile.mkdtemp())
trial = root / "backend_output" / "2026-01-01__00-00-00" / "proj__proj-1__abc"
(trial / "agent").mkdir(parents=True)
(trial / "verifier").mkdir()
(trial / "agent" / "mini-swe-agent.trajectory.json").write_text(
    json.dumps({"messages": [{"role": "user", "content": "the issue text"}],
                "info": {"exit_status": "Submitted", "model_stats": {"api_calls": 7}}}))
(trial / "result.json").write_text(json.dumps({"task_name": "proj__proj-1"}))
(trial / "config.json").write_text("{}")
(trial / "trial.log").write_text("ran the agent")
# The answers, in both of the shapes a verifier writes them.
(trial / "verifier" / "report.json").write_text(
    json.dumps({"proj__proj-1": {"tests_status": {"FAIL_TO_PASS": {"success": ["tests/test_x.py::test_bug"]}}}}))
(trial / "verifier" / "test-stdout.txt").write_text("PASS_TO_PASS tests/test_y.py::test_other")
(trial / "verifier" / "reward.txt").write_text("1")

dest = root / "disclosed"
trial_results.copy_disclosable_trial(trial, dest)

leaked = [str(p) for p in dest.rglob("*") if p.is_file()
          and ("FAIL_TO_PASS" in p.read_text(errors="ignore")
               or "PASS_TO_PASS" in p.read_text(errors="ignore"))]
assert not leaked, leaked
assert not (dest / "verifier" / "report.json").exists()
assert not (dest / "verifier" / "test-stdout.txt").exists()
# The score itself is not a secret -- the agent is told it anyway.
assert (dest / "verifier" / "reward.txt").read_text() == "1"
# And what the run is meant to learn from did come through, whole.
assert trial_results.agent_trajectory_path(dest) is not None
assert trial_results.agent_stats(dest)["api_calls"] == 7
# An artefact this file does not name defaults to withheld, not disclosed.
(trial / "answers_v2.json").write_text("FAIL_TO_PASS")
trial_results.copy_disclosable_trial(trial, dest)
assert not (dest / "answers_v2.json").exists()
print("DISCLOSURE_WHITELIST_OK")
PY
)"
check "the agent-readable copy withholds the hidden tests and nothing else" 'DISCLOSURE_WHITELIST_OK' "$OUT"

# Where the copy goes is the other half: inside the sandbox the agent could rewrite what
# its own evaluation recorded, and then "what it was told" stops being checkable.
OUT="$(cd "$REPO" && VTB_GRADING_STATE="$TMP/state23" "$VTB_PYTHON" - <<'PY' 2>&1
import sys, tempfile
sys.path.insert(0, ".")
from pathlib import Path
from eval.grading.store import Registry
from eval.grading import config

root = Path(tempfile.mkdtemp())
sandbox = root / "job_dir" / "task"
sandbox.mkdir(parents=True)
reg = Registry(state_dir=root / "state")
task = sorted(config.known_tasks())[0]

for bad in (sandbox, sandbox / "results"):
    try:
        reg.register(run_id="r", task=task, base_model="m", sandbox_root=sandbox,
                     eval_dir=root / "eval", disclosure_dir=bad)
    except ValueError as exc:
        assert "inside the run sandbox" in str(exc), exc
    else:
        raise AssertionError(f"accepted a disclosure_dir at {bad}")

# A sibling of the sandbox is the supported shape, and it is created for the caller.
sibling = root / "grading_disclosure"
token = reg.register(run_id="r", task=task, base_model="m", sandbox_root=sandbox,
                     eval_dir=root / "eval", disclosure_dir=sibling)
assert sibling.is_dir()
assert reg.lookup(token).disclosure_dir == sibling.resolve()
print("DISCLOSURE_DIR_OK")
PY
)"
check "a disclosure directory inside the sandbox is refused" 'DISCLOSURE_DIR_OK' "$OUT"

# The mount is what makes it readable but not writable, so the flag has to be there.
check "the run harness mounts the disclosure directory read-only" 'grading_results:ro' \
    "$(grep -F 'grading_disclosure' -A0 "$REPO/src/run_task.sh" | grep -F ':ro' || \
       grep -F 'grading_results:ro' "$REPO/src/run_task.sh")"
check_absent "...and it is not a child of the agent-writable job directory" 'JOB_DIR}/grading_disclosure' \
    "$(grep -F 'DISCLOSURE_DIR=' "$REPO/src/run_task.sh")"

echo "=== 24. a declared validation source is checked against benchmark and service ==="
# Earlier sections re-export VTB_ADMIN_TOKEN_FILE at the servers they start; this
# section talks to the main service again, so point back at its state dir.
export VTB_ADMIN_TOKEN_FILE="$TMP/state/admin.token"
# The service under test runs VTB_VAL_SOURCE=split:5 for $TASK.

OUT="$("${CLIENT[@]}" register --run-id smokeSrcMismatch --task "$TASK" \
    --base-model Qwen/Qwen3-1.7B-Base --sandbox-root "$TMP/runA/task" --eval-dir "$TMP/runA/eval" \
    --validation-source split:150 2>&1)"
check "a declaration naming another val source is refused" "but this service runs" "$OUT"

OUT="$("${CLIENT[@]}" register --run-id smokeSrcMalformed --task "$TASK" \
    --base-model Qwen/Qwen3-1.7B-Base --sandbox-root "$TMP/runA/task" --eval-dir "$TMP/runA/eval" \
    --validation-source bogus 2>&1)"
check "an unparseable declaration is refused" "cannot parse" "$OUT"

OUT="$("${CLIENT[@]}" register --run-id smokeSrcOK --task "$TASK" \
    --base-model Qwen/Qwen3-1.7B-Base --sandbox-root "$TMP/runA/task" --eval-dir "$TMP/runA/eval" \
    --validation-source split:5 2>&1)"
check "a declaration agreeing with the service is accepted" '^[A-Za-z0-9_-]\{20,\}' "$OUT"

OUT="$("${CLIENT[@]}" register --run-id smokeHarnessBad --task "$TASK" \
    --base-model Qwen/Qwen3-1.7B-Base --sandbox-root "$TMP/runA/task" --eval-dir "$TMP/runA/eval" \
    --editable-harness bogus 2>&1)"
check "an unknown editable-harness value is refused at registration" 'editable_harness must be' "$OUT"

OUT="$("${CLIENT[@]}" register --run-id smokeHarnessOff --task "$TASK" \
    --base-model Qwen/Qwen3-1.7B-Base --sandbox-root "$TMP/runA/task" --eval-dir "$TMP/runA/eval" \
    --editable-harness off 2>&1)"
check "an editable-harness opt-out is accepted at registration" '^[A-Za-z0-9_-]\{20,\}' "$OUT"

# The recipe half: the block lives in the recipe itself, and one written for another
# benchmark refuses the run at launch. A throwaway repo keeps the fixture out of the
# real tree.
SMOKE_REPO="$TMP/smoke_repo"
mkdir -p "$SMOKE_REPO/training/recipes"
cat > "$SMOKE_REPO/training/recipes/smoke_recipe.json" <<JSON
{"name": "smoke_recipe",
 "validation": {"benchmark": "swe-bench__swe-bench-verified", "val_source": "split:150"}}
JSON

OUT="$("$VTB_PYTHON" "$REPO/src/commit_utils/resolve_validation.py" val-source-for-recipe "$SMOKE_REPO" smoke_recipe "$TASK" 2>&1)"
check "a recipe declaring another benchmark is refused at launch" "declares a validation block for benchmark" "$OUT"

OUT="$("$VTB_PYTHON" "$REPO/src/commit_utils/resolve_validation.py" val-source-for-recipe "$SMOKE_REPO" smoke_recipe swe-bench__swe-bench-verified 2>&1)"
check "a recipe for the run's benchmark yields its val source" '^split:150$' "$OUT"

echo "=== 25. a final evaluates the held-out remainder only ==="
# The held-out set is the dataset order past the validation prefix, and the order
# comes from the backend's lock file -- never the per-sample file, which lists graded
# samples alphabetically and once moved the boundary.
OUT="$(cd "$REPO" && "$VTB_PYTHON" - <<'PY' 2>&1
import sys
sys.path.insert(0, ".")
from types import SimpleNamespace
from eval.grading import valsplit
from eval.grading.server import GradingService

# Pin the benchmark-declaration layer so the repo's real benchmark.json policies
# cannot leak in: SWE gets no declaration (falls through to the service default),
# TB keeps its real task-routed declaration.
from eval.grading import config as _cfg
_orig_vs = _cfg.validation_source
_cfg.validation_source = lambda task: None if task == "swe-bench__swe-bench-verified" else _orig_vs(task)
svc = object.__new__(GradingService)
svc.val_source = valsplit.parse_val_source("split:2")
svc._benchmark_val_sources = {}
svc.task_stats = SimpleNamespace(ordered_ids=lambda _t: ["a", "b", "c", "d", "e"],
                                 total=lambda _t: None)
ctx = SimpleNamespace(task="swe-bench__swe-bench-verified")
heldout = svc._heldout_ids(ctx)
assert heldout == ["c", "d", "e"], heldout

# Unknown order: fall back to a full run rather than guess the boundary.
svc.task_stats = SimpleNamespace(ordered_ids=lambda _t: None, total=lambda _t: None)
assert svc._heldout_ids(ctx) is None

# A task-routed validation source leaves the target benchmark whole: nothing is
# held out of it, so the final covers all of it.
svc_tb = object.__new__(GradingService)
svc_tb.val_source = svc.val_source
svc_tb._benchmark_val_sources = {}
svc_tb.task_stats = svc.task_stats
ctx_tb = SimpleNamespace(task="terminal-bench__terminal-bench-2-1")
assert svc_tb._heldout_ids(ctx_tb) is None
print("HELDOUT_OK")
PY
)"
check "a final evaluates only the held-out remainder of a split validation" 'HELDOUT_OK' "$OUT"

echo "=== 26. editable-harness submissions are validated before queueing ==="
# The pairing contract lives or dies in these checks: a submission the service should
# refuse must fail here, before it costs GPU time. Pure-python fixtures; no GPU.
# The seed-source assertions must resolve the seed the way production does, so the
# sibling Harbor checkout is offered first (set_env_vars.sh semantics) when present.
if [ -d "$REPO/../harbor/adapters" ]; then
    SMOKE_HARBOR_REPO="$(realpath -m "$REPO/../harbor")"
fi
OUT="$(cd "$REPO" && VTB_HARBOR_REPO="${SMOKE_HARBOR_REPO:-}" "$VTB_PYTHON" - <<'PY' 2>&1
import json, os, shutil, sys, tempfile
sys.path.insert(0, ".")
from pathlib import Path
from types import SimpleNamespace
from eval.grading.server import harness as harness_mod
from eval.grading.server.harness import _editable_harness_submission

TB = "terminal-bench__terminal-bench-2-1"
ALLOWED = json.loads((Path("benchmarks") / TB / "benchmark.json").read_text())["editable_harness"]["allowed_files"]

def make_candidate(root):
    cand = Path(root) / "cand"
    (cand / "final_model").mkdir(parents=True)
    for rel in ALLOWED:
        p = cand / "harness" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel == "terminus_2.py":
            p.write_text("class BaseAgent:\n    pass\n\nclass Terminus2(BaseAgent):\n    pass\n")
        else:
            p.write_text("x\n")
    (cand / "candidate.json").write_text(json.dumps({"candidate_id": "cand_test"}))
    return cand, cand / "final_model"

def expect_refusal(desc, cand_model, needle, role="agent", task=TB):
    cand, model = cand_model
    try:
        _editable_harness_submission(SimpleNamespace(task=task, role=role), model)
    except ValueError as exc:
        assert needle in str(exc), f"{desc}: refused with the wrong reason: {exc}"
        return
    raise AssertionError(f"{desc}: accepted a submission that must be refused ({needle})")

root = Path(tempfile.mkdtemp())

# A valid pair is accepted and fingerprinted; a benchmark without the declaration
# passes through as a plain model submission.
cand, model = make_candidate(root / "ok")
out = _editable_harness_submission(SimpleNamespace(task=TB, role="agent"), model)
assert out["candidate_id"] == "cand_test" and out["fingerprint"]
assert _editable_harness_submission(SimpleNamespace(task="aime__aime", role="agent"), model) is None

# The same pair delivered officially reads final_candidate.json instead.
off_cand, off_model = make_candidate(root / "ok_off")
(off_cand / "candidate.json").unlink()
expect_refusal("official without final_candidate.json", (off_cand, off_model), "final_candidate.json", role="official")
(off_cand / "final_candidate.json").write_text(json.dumps({"candidate_id": "cand_test"}))
out = _editable_harness_submission(SimpleNamespace(task=TB, role="official"), off_model)
assert out["candidate_id"] == "cand_test"

bad = make_candidate(root / "noharness")
shutil.rmtree(bad[0] / "harness")
expect_refusal("harness missing beside the model", bad, "missing beside")

bad = make_candidate(root / "undeclared")
(bad[0] / "harness" / "evil.py").write_text("x\n")
expect_refusal("undeclared file in the package", bad, "undeclared file")

bad = make_candidate(root / "symlink")
(bad[0] / "harness" / "tmux_session.py").unlink()
os.symlink(bad[0] / "harness" / "terminus_2.py", bad[0] / "harness" / "tmux_session.py")
expect_refusal("symlink in the package", bad, "symlinks")

bad = make_candidate(root / "notagent")
(bad[0] / "harness" / "terminus_2.py").write_text("class Terminus2:\n    pass\n")
expect_refusal("entry no longer a BaseAgent", bad, "BaseAgent")

bad = make_candidate(root / "noid")
(bad[0] / "candidate.json").write_text(json.dumps({"candidate_id": ""}))
expect_refusal("empty candidate_id", bad, "candidate_id")

# The import path is the loader contract; a benchmark declaring anything else is refused.
orig = harness_mod.config.editable_harness
try:
    harness_mod.config.editable_harness = lambda task: {
        "enabled": True, "import_path": "elsewhere.module:Nope", "allowed_files": ALLOWED,
    }
    good = make_candidate(root / "badimport")
    expect_refusal("unsupported import path", good, "unsupported import path")
finally:
    harness_mod.config.editable_harness = orig

# The SWE declaration: another module, class, and base class, read off benchmark.json.
SWE = "swe-bench__swe-bench-verified"
SWE_ALLOWED = json.loads((Path("benchmarks") / SWE / "benchmark.json").read_text())["editable_harness"]["allowed_files"]

def make_swe_candidate(root, base="BaseInstalledAgent"):
    cand = Path(root) / "cand"
    (cand / "final_model").mkdir(parents=True)
    harness = cand / "harness"
    harness.mkdir(parents=True)
    (harness / "__init__.py").write_text("\n")
    (harness / "mini_swe_agent.py").write_text(f"class {base}:\n    pass\n\nclass MiniSweAgent({base}):\n    pass\n")
    (cand / "candidate.json").write_text(json.dumps({"candidate_id": "cand_swe"}))
    return cand, cand / "final_model"

swe_ok = make_swe_candidate(root / "swe_ok")
out = _editable_harness_submission(SimpleNamespace(task=SWE, role="agent"), swe_ok[1])
assert out["candidate_id"] == "cand_swe" and out["fingerprint"]

# A run that opted out of the editable harness at registration scores the model
# alone; no harness package is required beside it.
off_cand = Path(root) / "off_ok" / "cand"
(off_cand / "final_model").mkdir(parents=True)
assert _editable_harness_submission(
    SimpleNamespace(task=SWE, role="agent", editable_harness="off"), off_cand / "final_model"
) is None
expect_refusal(
    "swe entry with the wrong base class",
    make_swe_candidate(root / "swe_badbase", base="BaseAgent"),
    "BaseInstalledAgent",
    task=SWE,
)

# The declared import path must name a class that really exists in the source the
# workspace is seeded from -- a typo there only surfaces at evaluation time otherwise.
# Resolve through the same function the workspace prep uses. The backend-installed
# package is mandatory for SWE because that is what a baseline registered agent imports;
# directory-based development harnesses may fall back to a checkout. Machines with
# neither skip this.
import re
sys.path.insert(0, "src/commit_utils")
from prepare_agent_workspace import installed_harness_source
for bench in (SWE, TB):
    block = json.loads((Path("benchmarks") / bench / "benchmark.json").read_text())["editable_harness"]
    module_rel, cls = block["import_path"].split(":", 1)
    try:
        seed = installed_harness_source(Path(".").resolve(), block["template_source"])
    except SystemExit:
        continue
    entry = seed if seed.is_file() else seed / (module_rel.split(".", 1)[1] + ".py")
    src = entry.read_text(encoding="utf-8")
    assert re.search(rf"^class {cls}\(", src, re.M), f"declared class {cls} not in {entry}"
    if bench == SWE:
        # The L2 loop contract needs a fixed in-container API baseline, and the pin does
        # not live in the shim: harbor installs mini-swe-agent inside the task container
        # at whatever agent_kwargs.version declares (task_backends forwards it as
        # --agent-kwarg version=...), so an absent declaration installs whatever is
        # latest at evaluation time. Checked where the number actually is, because a
        # constant in the seeded shim would not be read by anything.
        pinned = str(
            (
                json.loads((Path("benchmarks") / bench / "benchmark.json").read_text()).get(
                    "agent_kwargs"
                )
                or {}
            ).get("version")
            or ""
        )
        assert re.fullmatch(r"\d+\.\d+\.\d+", pinned), \
            f"{bench} declares no exact mini-swe-agent pin in agent_kwargs.version: {pinned!r}"

print("EDITABLE_HARNESS_OK")
PY
)"
check "editable-harness submissions are validated before queueing" 'EDITABLE_HARNESS_OK' "$OUT"

echo ""
echo "=== 24. a file: split names validation, and routes final scoring ==="
# When a benchmark's test set lives elsewhere, all of its own tasks are validation.
# A JSON file declares that outright -- the ids, plus the benchmark a final score is
# measured on instead. Built like section 11: a fixture repo whose evaluate.py stands
# in, so the whole path runs without a GPU.
TESTTASK="${TASK}_routed"
mkdir -p "$TMP/repoJ/benchmarks/$TASK" "$TMP/repoJ/benchmarks/$TESTTASK" "$TMP/fakehfJ" \
    "$TMP/runJ/task/output/v1" "$TMP/runJ/task2/output/v1" "$TMP/runJ/eval/final_model" \
    "$TMP/runJ/eval2" "$TMP/runJ/logs" "$TMP/splitdir/$TASK"
ln -s "$REPO/scripts" "$TMP/repoJ/scripts"
echo '{}' > "$TMP/runJ/task/output/v1/config.json"
echo '{}' > "$TMP/runJ/task2/output/v1/config.json"
echo '{}' > "$TMP/runJ/eval/final_model/config.json"
# The split file may live anywhere: the service binds it to this benchmark by the
# directory it sits in, not by the repo layout.
cat > "$TMP/splitdir/$TASK/split.json" <<SPLIT
{"validation": {"task_ids": ["s0", "s1", "s2"]}, "test": {"benchmark": "$TESTTASK"}}
SPLIT
# Both stand-ins succeed and identify themselves: the validation one must receive
# the file's ids as --only, the routed one is what a final score is measured on.
for bench in "$TASK" "$TESTTASK"; do
    mkdir -p "$TMP/repoJ/benchmarks/$bench"
    cat > "$TMP/repoJ/benchmarks/$bench/evaluate.py" <<FAKE
import json, sys

print("$bench stand-in ran with: " + " ".join(sys.argv[1:]), flush=True)
out = None
for i, arg in enumerate(sys.argv):
    if arg == "--json-output-file" and i + 1 < len(sys.argv):
        out = sys.argv[i + 1]
if out:
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"accuracy": 0.5, "stderr": 0.1}, fh)
FAKE
done

PORT5=$((PORT + 4))
CLIENT5=("$VTB_PYTHON" -m eval.grading.client --base "http://127.0.0.1:${PORT5}")
export VTB_ADMIN_TOKEN_FILE="$TMP/stateJ/admin.token"
VTB_REPO="$TMP/repoJ" \
VTB_GRADING_STATE="$TMP/stateJ" \
VTB_GRADING_SETTLE=0.2 \
VTB_VAL_SOURCE="file:$TMP/splitdir/$TASK/split.json" \
VTB_EVAL_GPU="" \
    VTB_SELF_SERVED_BENCHMARKS="$TASK,$TESTTASK" \
HF_HOME="$TMP/fakehfJ" \
    "$VTB_PYTHON" -m eval.grading.server --host 127.0.0.1 --port "$PORT5" > "$TMP/serverJ.log" 2>&1 &
SRV5_PID=$!
for _ in $(seq 1 40); do
    "${CLIENT5[@]}" health >/dev/null 2>&1 && break
    sleep 0.25
done

OUT="$("${CLIENT5[@]}" health 2>&1)"
check "health describes the file: split" '"kind": "file"' "$OUT"
check "...and where final scoring is routed" "\"test_benchmark\": \"$TESTTASK\"" "$OUT"
check "the owning benchmark's count comes from the file" '"source": "file"' "$OUT"

"${CLIENT5[@]}" register --run-id smokeJ#agent --task "$TASK" --base-model Qwen/Qwen3-1.7B-Base \
    --sandbox-root "$TMP/runJ/task" --eval-dir "$TMP/runJ/eval" --role agent \
    --log-dir "$TMP/runJ/logs" --token-file "$TMP/tokJ_agent" >/dev/null 2>&1
"${CLIENT5[@]}" register --run-id smokeJ#official --task "$TASK" --base-model Qwen/Qwen3-1.7B-Base \
    --sandbox-root "$TMP/runJ/eval" --eval-dir "$TMP/runJ/eval" --token-file "$TMP/tokJ_official" >/dev/null 2>&1
"${CLIENT5[@]}" register --run-id smokeJ#other --task "$TESTTASK" --base-model Qwen/Qwen3-1.7B-Base \
    --sandbox-root "$TMP/runJ/task2" --eval-dir "$TMP/runJ/eval2" --role agent \
    --token-file "$TMP/tokJ_other" >/dev/null 2>&1

# The file's list is the whole split: the service names it outright (so the response
# says exactly what will run) and no count is left to send.
OUT="$("${CLIENT5[@]}" submit --token-file "$TMP/tokJ_agent" --model-path output/v1 2>&1)"
check "a file: submission runs the list, not a count" '"limit": -1' "$OUT"
check "...naming the file's ids in file order" 'ids_json s0 s1 s2' \
    "$(printf '%s' "$OUT" | "$VTB_PYTHON" -c '
import json, sys
print("ids_json", *json.load(sys.stdin)["sample_ids"])')"
OUT="$("${CLIENT5[@]}" submit --token-file "$TMP/tokJ_agent" --model-path output/v1 --limit 2 2>&1)"
check "a smaller limit takes a prefix of the list" 'ids_json s0 s1' \
    "$(printf '%s' "$OUT" | "$VTB_PYTHON" -c '
import json, sys
print("ids_json", *json.load(sys.stdin)["sample_ids"])')"

OUT="$("${CLIENT5[@]}" submit --token-file "$TMP/tokJ_agent" --model-path output/v1 --sample-ids nope 2>&1)"
check "an id outside the file's list is refused" 'not in the validation split' "$OUT"

# Naming in-list ids is the agent's own pick, so the entry is a subset rather than
# the split -- the same rule as with a prefix-based split.
OUT="$("${CLIENT5[@]}" submit --token-file "$TMP/tokJ_agent" --model-path output/v1 --sample-ids s2 s0 --wait --wait-timeout 120 2>&1)"
check "in-list ids can be named and evaluated" '"state": "done"' "$OUT"

OUT="$("${CLIENT5[@]}" submit --token-file "$TMP/tokJ_agent" --model-path output/v1 --wait --wait-timeout 120 2>&1)"
check "the whole-list validation job finishes" '"state": "done"' "$OUT"
# The service log records every attempt's full command line, which is where the
# file's ids are visible as the --only they were sent as.
check "the file's ids reached evaluate.py as --only" '--only s0 s1 s2' "$(cat "$TMP/serverJ.log")"

# Exactly what run_task.sh runs: the official token asks for a final score, and the
# split's test.benchmark is where it is measured -- not the benchmark registered
# against, and not the validation ids either.
OUT="$("${CLIENT5[@]}" submit --token-file "$TMP/tokJ_official" --model-path final_model \
    --mode final --write-metrics --wait --wait-timeout 120 2>&1)"
check "official final is scored on the routed benchmark" "\"eval_task\": \"$TESTTASK\"" "$OUT"
check "the routed run itself finished" '"state": "done"' "$OUT"
check "metrics.json carries the routed test score" '"accuracy": 0.5' \
    "$(cat "$TMP/runJ/eval/metrics.json" 2>&1)"
check "final_eval.log is the routed run's own output" "$TESTTASK stand-in ran" \
    "$(cat "$TMP/runJ/eval/final_eval.log" 2>&1)"

OUT="$("$VTB_PYTHON" - "$TMP/runJ/eval/submissions.json" <<'PY' 2>&1
import json, sys

entries = json.load(open(sys.argv[1]))
finals = [e for e in entries if e.get("mode") == "final"]
assert len(finals) == 1, finals
assert "subset" not in finals[0], finals[0]
assert finals[0]["score"] == 0.5, finals[0]
named = [e for e in entries if e.get("subset")]
assert len(named) == 1 and named[0]["sample_ids"] == ["s0", "s2"], named
assert named[0].get("comparable_to_full") is False, named[0]
print("ENTRIES_OK")
PY
)"
check "a routed final is a full score; a named pick is marked a subset" 'ENTRIES_OK' "$OUT"

OUT="$("${CLIENT5[@]}" submit --token-file "$TMP/tokJ_other" --model-path output/v1 2>&1)"
check "a file cannot declare another benchmark's split" 'does not belong to' "$OUT"

kill -TERM "$SRV5_PID" 2>/dev/null
for _ in $(seq 1 40); do
    kill -0 "$SRV5_PID" 2>/dev/null || break
    sleep 0.25
done
SRV5_PID=""

echo ""
echo "================================"
echo "  passed: $PASS   failed: $FAIL"
echo "================================"
[ "$FAIL" -eq 0 ] || { echo "server log:"; tail -30 "$TMP/server.log"; }
exit $((FAIL > 0))
