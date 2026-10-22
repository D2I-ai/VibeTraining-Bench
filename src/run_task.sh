#!/bin/bash

export EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
CLUSTER_ID="$4"
NUM_HOURS="$5"
AGENT_CONFIG="$6"
if [ -n "${7:-}" ]; then
    NUM_GPUS="$7"
elif [ -n "${VTB_AGENT_GPUS:-}" ]; then
    NUM_GPUS="$(printf '%s' "$VTB_AGENT_GPUS" | awk -F, '{print NF}')"
else
    NUM_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
    if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -lt 1 ] 2>/dev/null; then
        NUM_GPUS=1
    fi
fi

source src/commit_utils/load_launch_env.sh || exit 3
source src/commit_utils/set_env_vars.sh

# Resolve and validate the selected adapter before creating a workspace or registration.
AGENT_INPUTS="$("$VTB_PYTHON" agents/config.py shell-inputs "$AGENT" --researcher "$AGENT_CONFIG")" || exit 3
eval "$AGENT_INPUTS"
unset AGENT_INPUTS

RESULT_PREFIX_SAFE=$(echo "$MODEL_TO_TRAIN" | tr '/:[]' '____')

AGENT_CONFIG_SAFE="$RESEARCHER_SAFE"

RANDOM_UUID=$(uuidgen)

GPU_SUFFIX=""
if [ "$NUM_GPUS" -gt 1 ] 2>/dev/null; then
    GPU_SUFFIX="_${NUM_GPUS}gpu"
fi

# Which physical cards the agent container may use.
#
# This is enforced rather than suggested: the cards named here are the only ones
# `docker run --gpus` puts in the container, and they are renumbered from 0 inside it,
# so src/utils/check_cuda.py's `device_count == NUM_GPUS` check holds by construction.
# An agent cannot re-export its way onto a card it was not given.
#
# Without an explicit limit, expose every currently visible GPU. Pass NUM_GPUS (the 7th
# positional argument) or VTB_AGENT_GPUS to restrict the run; if only VTB_AGENT_GPUS is
# set, NUM_GPUS is inferred from that comma-separated list.
#
# Set VTB_AGENT_GPUS explicitly to place a run on specific cards -- for example away from
# the grading service allocation (VTB_EVAL_GPUS / VTB_EVAL_GPU) in isolated mode.
if [ -z "${VTB_AGENT_GPUS:-}" ]; then
    VTB_AGENT_GPUS=$(seq -s, 0 $((NUM_GPUS - 1)))
fi
export VTB_AGENT_GPUS

export EVAL_DIR="${VTB_RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h${GPU_SUFFIX}${VTB_EXPERIMENT_NAME}/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${CLUSTER_ID}"

mkdir -p ${EVAL_DIR}

exec 1>${EVAL_DIR}/output.log
exec 2>${EVAL_DIR}/error.log

echo "$@"

# export TMP_SUBDIR="/tmp/posttrain_container_${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"
export TMP_SUBDIR="${VTB_SCRATCH:-/tmp}/container_${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"

JOB_DIR="${TMP_SUBDIR}/job_dir"
JOB_TMP="${TMP_SUBDIR}/tmp"
OVERLAY_AGENT_HF_HOME="${VTB_AGENT_HOME}/hf_cache"
export HF_MERGED="${TMP_SUBDIR}/merged_huggingface"

echo "Preparing job directory..." 
mkdir -p "${JOB_DIR}"
mkdir -p "${JOB_TMP}"

# Runtime facts the generated environment.md states. Exported before workspace
# preparation so prepare_agent_workspace.py can quote them.
export VTB_RUN_HOURS="${NUM_HOURS}"
export GRADING_URL="http://${VTB_GRADING_HOST:-127.0.0.1}:${VTB_GRADING_PORT:-5800}"

# A half-prepared workspace must never reach the agent: a bad recipe name or resource
# path makes prepare_agent_workspace.py fail loudly, and without this guard the run
# would start anyway and the agent would flounder in an incomplete workspace.
if ! "$VTB_PYTHON" src/commit_utils/prepare_agent_workspace.py \
    --repo "${VTB_REPO}" \
    --benchmark "${EVALUATION_TASK}" \
    --model-to-train "${MODEL_TO_TRAIN}" \
    --job-dir "${JOB_DIR}" \
    --train-recipe "${VTB_TRAIN_RECIPE:-}" \
    --train-resources "${VTB_TRAIN_RESOURCES:-}"; then
    echo "FATAL: agent workspace preparation failed; refusing to start the run" >&2
    exit 3
fi

# harness_connectivity exercises the editable-harness pairing end to end; when the run
# ships no harness (the benchmark declares no editable_harness, or VTB_EDITABLE_HARNESS=off
# opted out) its steps are impossible, so refuse before the agent burns budget on it.
if [ "${VTB_PROMPT}" = "harness_connectivity" ] && [ ! -d "${JOB_DIR}/task/harness" ]; then
    echo "FATAL: VTB_PROMPT=harness_connectivity requires an editable-harness run" >&2
    echo "(benchmark declares editable_harness and VTB_EDITABLE_HARNESS is not 'off')" >&2
    exit 3
fi

# The workspace preparation decided whether this run ships an editable harness; the
# prompt renders its harness blocks from the same fact, so prompt and workspace can
# never disagree. (A hand-rendered prompt falls back to the benchmark's declaration.)
if [ -d "${JOB_DIR}/task/harness" ]; then
    VTB_HAS_HARNESS=1
else
    VTB_HAS_HARNESS=0
fi
PROMPT=$(VTB_HAS_HARNESS="$VTB_HAS_HARNESS" "$VTB_PYTHON" eval/agent_task/get_prompt.py --model-to-train "$MODEL_TO_TRAIN" --benchmark-id "$EVALUATION_TASK" --num-hours "$NUM_HOURS" --num-gpus "$NUM_GPUS" --agent "${AGENT}") || {
    echo "FATAL: prompt rendering failed; refusing to start the run" >&2
    exit 3
}
echo "$PROMPT" > "${EVAL_DIR}/prompt.md"

bash src/utils/create_timer.sh $NUM_HOURS $JOB_DIR/task/timer.sh

# ============================================================================
# Grading service: preflight and registration
# ============================================================================
# Scoring goes through the shared grading service (eval/grading/). It is not
# started here: one service serves every concurrent run, so its lifetime is
# longer than any single run's. Check it first -- discovering it is down after
# ten hours of training would waste the whole run.
#
# Its address is passed in rather than discovered. The agent container runs with
# --network host, so the service on the host loopback is reachable from inside it with
# no forwarding. grading.md refers to this by name, so it has to be injected --
# otherwise every curl in that document expands to a bare path and fails in a way that
# looks like the service is down.
# GRADING_URL itself is exported earlier, before workspace preparation, so the
# generated environment.md can quote it.

# Deliberately not exported: the agent container inherits nothing from this shell (a
# docker container starts from the image's environment plus the -e flags below), and
# these are operator-side details -- a path into the checkout, and the location of the
# token that scores the delivered model -- that had no business being in the agent's
# environment in the first place.
GRADING_CLIENT="$(pwd)/eval/grading/client.py"
RUN_LABEL="$(basename "$(dirname "$EVAL_DIR")")/$(basename "$EVAL_DIR")"

if ! "$VTB_PYTHON" "$GRADING_CLIENT" health > "${EVAL_DIR}/grading_health.json" 2>&1; then
    echo "FATAL: grading service is unreachable, refusing to start a ${NUM_HOURS}h run"
    cat "${EVAL_DIR}/grading_health.json"
    printf 'Start it with: VTB_GPU_COORDINATION_MODE=isolated VTB_EVAL_GPU=<card> %q -m eval.grading.server\n' "$VTB_PYTHON"
    exit 3
fi
"$VTB_PYTHON" - "${EVAL_DIR}/grading_health.json" "${VTB_AGENT_GPUS}" <<'PY' || true
import json
import sys
from pathlib import Path

health_path = Path(sys.argv[1])
agent_gpus = {item.strip() for item in sys.argv[2].split(",") if item.strip()}
try:
    health = json.loads(health_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
coord = health.get("gpu_coordination") or {}
mode = coord.get("mode") or "isolated"
eval_gpus = {str(item).strip() for item in (coord.get("eval_gpus") or []) if str(item).strip()}
if mode == "isolated" and agent_gpus and eval_gpus and agent_gpus & eval_gpus:
    overlap = ",".join(sorted(agent_gpus & eval_gpus))
    print(
        f"WARNING: grading service reports isolated GPU coordination, but VTB_AGENT_GPUS overlaps eval GPUs ({overlap}). "
        "Use disjoint VTB_AGENT_GPUS/VTB_EVAL_GPUS, or restart grading with VTB_GPU_COORDINATION_MODE=shared.",
        file=sys.stderr,
    )
PY

# Two tokens, because the two callers must not have the same reach.
#
#   agent token     sandbox = ${JOB_DIR}/task, the agent's own workspace. Readable
#                   by the agent. It cannot name anything under results/, which is
#                   what stops a submission of some model outside its workspace.
#   official token  sandbox = ${EVAL_DIR}, so the verified final_model copy can be
#                   scored. Kept here only: EVAL_DIR is never mounted into the agent
#                   container, so the agent cannot read this token.
AGENT_TOKEN_FILE="${JOB_DIR}/task/grading_token"
OFFICIAL_TOKEN_FILE="${EVAL_DIR}/.grading_token"

# Where the grading service leaves the full per-sample results of each validation for
# the agent to read. A sibling of JOB_DIR, not a child, for two reasons: register()
# refuses a disclosure directory inside the sandbox (there the agent could rewrite what
# its own evaluation recorded, and "what it was told" would stop being checkable), and
# reclaim_job_dir's recursive chown would otherwise walk hundreds of megabytes of
# transcripts for nothing. It is mounted into the container read-only below.
DISCLOSURE_DIR="${TMP_SUBDIR}/grading_disclosure"
mkdir -p "${DISCLOSURE_DIR}"

# The base model's own validation transcripts, if an operator has imported them
# (`client.py baseline --from-results`). Kept once under the grading state and mounted
# read-only rather than copied per run: it is ~600MB and identical for every run on the
# same base model. Absent is normal -- the mount is simply skipped.
#
# A sibling of grading_results, not a directory inside it: a nested bind mount would
# need docker to mkdir its target inside a read-only mount, which fails.
#
# The path comes from the service's own keying (BaselineCache.disclosure_dir) rather
# than being spelled out here, so the two cannot disagree about which directory belongs
# to which base model.
#
# A recipe may declare its validation inline (the validation block: benchmark +
# val_source). The run's baseline then comes from the conventionally-named
# validation resource, and the declared source must equal the service's effective
# source for this task -- a baseline measured on another split is not a reference,
# it is a wrong answer, so the run refuses to start on a mismatch rather than
# finding out from the numbers.
VALIDATION_SOURCE=""
if [ -n "${VTB_TRAIN_RECIPE:-}" ]; then
    if ! VALIDATION_SOURCE="$("$VTB_PYTHON" src/commit_utils/resolve_validation.py val-source-for-recipe "$VTB_REPO" "${VTB_TRAIN_RECIPE}" "$EVALUATION_TASK")"; then
        echo "FATAL: recipe's validation block does not match this run's benchmark (see above)"
        exit 3
    fi
fi
VALIDATION_ARGS=()
[ -n "$VALIDATION_SOURCE" ] && VALIDATION_ARGS=(--validation-source "$VALIDATION_SOURCE")

BASELINE_DISCLOSURE_ARGS=()
if [ -n "$VALIDATION_SOURCE" ]; then
    if ! BASELINE_DISCLOSURE_DIR="$("$VTB_PYTHON" src/commit_utils/resolve_validation.py baseline-for-source "$VTB_REPO" "$EVALUATION_TASK" "$MODEL_TO_TRAIN" "$VALIDATION_SOURCE" "${EVAL_DIR}/grading_health.json")"; then
        echo "FATAL: recipe's declared validation source disagrees with the grading service (see above)"
        exit 3
    fi
else
    BASELINE_DISCLOSURE_DIR="$(cd "$VTB_REPO" && "$VTB_PYTHON" -c '
import sys
sys.path.insert(0, ".")
from eval.grading.store import BaselineCache
print(BaselineCache().disclosure_dir(sys.argv[1], sys.argv[2]))
' "$EVALUATION_TASK" "$MODEL_TO_TRAIN")" || {
        echo "WARNING: could not resolve the baseline disclosure directory; it will not be mounted"
        BASELINE_DISCLOSURE_DIR=""
    }
fi
if [ -n "${BASELINE_DISCLOSURE_DIR}" ] && [ -d "${BASELINE_DISCLOSURE_DIR}" ]; then
    BASELINE_DISCLOSURE_ARGS=(-v "${BASELINE_DISCLOSURE_DIR}:${VTB_AGENT_HOME}/grading_baseline:ro")
    echo "Baseline validation transcripts mounted read-only from ${BASELINE_DISCLOSURE_DIR}"
else
    echo "No imported baseline validation for ${EVALUATION_TASK} / ${MODEL_TO_TRAIN}; none mounted"
fi

# A run may opt out of a benchmark-declared editable harness (a model-only ablation);
# the service records the flag so this run's submissions are plain model deliveries.
HARNESS_REG_ARGS=()
if [ -n "${VTB_EDITABLE_HARNESS:-}" ]; then
    HARNESS_REG_ARGS=(--editable-harness "${VTB_EDITABLE_HARNESS}")
fi

if ! "$VTB_PYTHON" "$GRADING_CLIENT" register \
        --run-id "${RUN_LABEL}#agent" \
        --task "$EVALUATION_TASK" \
        --base-model "$MODEL_TO_TRAIN" \
        --sandbox-root "${JOB_DIR}/task" \
        --eval-dir "$EVAL_DIR" \
        --role agent \
        --log-dir "${TMP_SUBDIR}/grading_logs" \
        --disclosure-dir "${DISCLOSURE_DIR}" \
        "${VALIDATION_ARGS[@]}" \
        "${HARNESS_REG_ARGS[@]}" \
        --token-file "$AGENT_TOKEN_FILE" > /dev/null; then
    echo "FATAL: could not register the agent token with the grading service"
    exit 3
fi

if ! "$VTB_PYTHON" "$GRADING_CLIENT" register \
        --run-id "${RUN_LABEL}#official" \
        --task "$EVALUATION_TASK" \
        --base-model "$MODEL_TO_TRAIN" \
        --sandbox-root "$EVAL_DIR" \
        --eval-dir "$EVAL_DIR" \
        --role official \
        "${VALIDATION_ARGS[@]}" \
        "${HARNESS_REG_ARGS[@]}" \
        --token-file "$OFFICIAL_TOKEN_FILE" > /dev/null; then
    echo "FATAL: could not register the official token with the grading service"
    exit 3
fi

echo "Grading service ready; agent token at task/grading_token"

# codex authenticates with CODEX_API_KEY, so the natural OPENAI_API_KEY (what api_key.env
# holds) is renamed to it here and OPENAI_API_KEY is then unset. That keeps the outer
# agent's key from leaking to anything inside the container and leaves OPENAI_API_KEY free
# for the eval backends to point at the served model (eval/runner/task_backends.py).
export CODEX_API_KEY="${CODEX_API_KEY:-${OPENAI_API_KEY:-}}"
unset OPENAI_API_KEY

# Copy scripts needed inside the container
cp src/utils/check_cuda.py "${JOB_DIR}/check_cuda.py"
cp src/utils/check_cuda_writing.py "${JOB_DIR}/check_cuda_writing.py"
cp src/utils/system_monitor.sh "${JOB_DIR}/system_monitor.sh"
cp src/utils/timestamp_lines.py "${JOB_DIR}/timestamp_lines.py"
cp src/commit_utils/bootstrap_mutable_framework.sh "${JOB_DIR}/bootstrap_mutable_framework.sh"
cp "${AGENT_ADAPTER_DIR}/solve.sh" "${JOB_DIR}/agent_solve.sh"
cp -r "containers/other_home_data/.codex" "${JOB_DIR}/"

# Copy agent-specific auth if present (e.g. for non-API agents)
if [ -f "${AGENT_ADAPTER_DIR}/auth.json" ]; then
    cp "${AGENT_ADAPTER_DIR}/auth.json" "${JOB_DIR}/.codex/auth.json"
fi
if [ -f "${AGENT_ADAPTER_DIR}/oauth_token" ]; then
    cp "${AGENT_ADAPTER_DIR}/oauth_token" "${JOB_DIR}/oauth_token"
fi

# Utils
with_huggingface_overlay() {
    mkdir -p "$TMP_SUBDIR/merged_huggingface"
    mkdir -p "$TMP_SUBDIR/upper_huggingface"
    mkdir -p "$TMP_SUBDIR/fuse_workdir"
    # allow_root lets the root docker daemon bind this mount into the agent container
    # (-v $HF_MERGED:...). Without it a user-owned fuse-overlayfs mountpoint is opaque
    # to root and `docker run` fails with "mkdir <path>: file exists". allow_root exposes
    # the mount only to the mounting user and root (narrower than allow_other), which is
    # enough because the container runs as root. It needs `user_allow_other` in
    # /etc/fuse.conf (a one-time host setup). If the container is ever switched to a
    # non-root --user or userns-remap, widen this to allow_other.
    fuse-overlayfs -o "lowerdir=$HF_HOME,upperdir=$TMP_SUBDIR/upper_huggingface,workdir=$TMP_SUBDIR/fuse_workdir,allow_root" "$TMP_SUBDIR/merged_huggingface"
    
    "$@"
    local exit_code=$?
    
    fusermount -u "$TMP_SUBDIR/merged_huggingface"
    rm -r "$TMP_SUBDIR/merged_huggingface"
    rm -r "$TMP_SUBDIR/upper_huggingface"
    rm -r "$TMP_SUBDIR/fuse_workdir"
    
    return $exit_code
}

# Where the datasets library may write. Its default is ${HF_HOME}/datasets, and when
# HF_HOME is the fuse-overlayfs view that default does not work: the container runs as
# root while fuse-overlayfs runs as the host account, so creating anything new in the
# merged tree fails with EPERM (modifying a path that already exists in the lower layer
# copies up fine, which is why the cache looks writable until something needs a new
# directory). A run hit this as `Operation not permitted: hub/.locks/datasets--...` and
# read it as a download problem. /tmp is a plain bind mount of JOB_TMP, so it is
# genuinely writable and thrown away with the scratch tree.
AGENT_HF_DATASETS_CACHE="/tmp/hf_datasets_cache"

HF_MOUNT_ARGS=()
run_with_agent_hf_cache() {
    mkdir -p "${JOB_TMP}/hf_cache" "${JOB_TMP}/hf_datasets_cache"
    if [ "${VTB_AGENT_HF_HOME}" != "${OVERLAY_AGENT_HF_HOME}" ]; then
        # Operator override to an isolated empty per-job cache: nothing from the host
        # cache is mounted, the agent starts from an empty HF_HOME.
        "$@"
        return $?
    fi
    # The shared cache reaches the container one of two ways, both landing at
    # OVERLAY_AGENT_HF_HOME (which is the container's HF_HOME):
    #   overlay     a fuse-overlayfs merged view, so the agent's own downloads collect in
    #               a throwaway upper layer and never touch the shared cache.
    #   plain bind  the shared cache mounted read-only, for a host without fuse-overlayfs.
    #               Its absence used to silently mount an empty dir and burn the whole run
    #               (offline + empty cache -> model never found); a read-only bind is the
    #               working substitute. Safe because the cache is fully seeded, the agent
    #               runs offline (HF_HUB_OFFLINE) so it needs no hub writes, and the
    #               datasets cache is already redirected to writable /tmp. Read-only also
    #               keeps the shared cache pristine for the grading serve reading it on
    #               the same host at the same time.
    if [ "${VTB_HF_PLAIN_BIND:-0}" = "1" ] || ! command -v fuse-overlayfs >/dev/null 2>&1; then
        if [ "${VTB_HF_PLAIN_BIND:-0}" != "1" ]; then
            echo "[hf] fuse-overlayfs not found; binding ${HF_HOME} read-only instead of overlaying it (set VTB_HF_PLAIN_BIND=1 to make this the intended path)" >&2
        fi
        HF_MOUNT_ARGS=(-v "${HF_HOME}:${OVERLAY_AGENT_HF_HOME}:ro")
        "$@"
        return $?
    fi
    HF_MOUNT_ARGS=(-v "${HF_MERGED}:${OVERLAY_AGENT_HF_HOME}")
    with_huggingface_overlay "$@"
}

with_record_the_time() {
    local begin=$(date --iso-8601=seconds)
    "$@"
    local exit_code=$?
    local end=$(date --iso-8601=seconds)
    
    local time_taken=$(( $(date --date="$end" +%s) - $(date --date="$begin" +%s) ))
    printf '%02d:%02d:%02d\n' \
        $(( time_taken / 3600 )) \
        $(( (time_taken % 3600) / 60 )) \
        $(( time_taken % 60 )) > "${EVAL_DIR}/time_taken.txt"
    
    return $exit_code
}

SOLVE_OUT="${EVAL_DIR}/solve_out.txt"

# One container name per run, so this script can always reach the container it started.
# `docker run` is only a client: killing it -- which is exactly what the timeout below
# does at the end of a run's budget -- leaves the container, the agent's training
# processes and its share of the GPUs behind, so the container has to be named and
# removed explicitly.
AGENT_CONTAINER="vtb-agent-$(printf '%s' "${EVALUATION_TASK}_${CLUSTER_ID}_${RANDOM_UUID}" | tr -c 'a-zA-Z0-9_.-' '-')"
trap 'docker rm --force "$AGENT_CONTAINER" >/dev/null 2>&1 || true' EXIT

# Forward only the selected adapter's declared variables. Docker resolves values from
# this environment by name; credential values do not appear in docker command arguments.
AGENT_ENV_NAMES="$("$VTB_PYTHON" agents/config.py env-names "$AGENT")" || exit 3
AGENT_ENV_ARGS=()
while IFS= read -r _var; do
    [ -n "$_var" ] || continue
    [ "${!_var+x}" = x ] && AGENT_ENV_ARGS+=(-e "$_var")
done <<< "$AGENT_ENV_NAMES"
unset AGENT_ENV_NAMES
AGENT_API_KEY="${VTB_API_KEY:-inspectai}"

# Forward an explicit HF_ENDPOINT override to the agent container. Downloads
# may query endpoint metadata even when files are already cached, so the
# configured endpoint must remain reachable.
HF_ENV_ARGS=()
[ -n "${HF_ENDPOINT:-}" ] && HF_ENV_ARGS+=(-e "HF_ENDPOINT=${HF_ENDPOINT}")

# $1: timeout in minutes (default: the full budget plus the serving grace).
# $2: attempt index (0 writes solve_out.txt; >0 is a resume and gets appended to it).
solve_task() {
    local timeout_min="${1:-$((NUM_HOURS * 60 + 5))}"
    local attempt="${2:-0}"
    local attempt_out="${SOLVE_OUT}"
    [ "$attempt" -gt 0 ] && attempt_out="${SOLVE_OUT}.resume${attempt}"
    # The resume nudge is only set for continuation attempts; it rides the container's
    # environment so the agent's solve.sh can switch to `claude --continue`.
    RESUME_ENV=()
    [ -n "${VTB_RESUME_NUDGE:-}" ] && RESUME_ENV=(-e "VTB_RESUME_NUDGE=${VTB_RESUME_NUDGE}")
    # --gpus takes a csv-quoted value: with a bare device=0,1 the docker CLI reads the
    # list separator as its own and rejects the flag.
    timeout --signal=TERM --kill-after=30s "${timeout_min}m" \
    docker run --rm \
        --name "$AGENT_CONTAINER" \
        --gpus "\"device=${VTB_AGENT_GPUS}\"" \
        --network host \
        --ipc host \
        --shm-size "${VTB_AGENT_SHM_SIZE:-32g}" \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --init \
        --env NVIDIA_VISIBLE_DEVICES="${VTB_AGENT_GPUS}" \
        --env HOME="${VTB_AGENT_HOME}" \
        --env HF_HOME="${VTB_AGENT_HF_HOME}" \
        --env HF_DATASETS_CACHE="${AGENT_HF_DATASETS_CACHE}" \
        --env HF_HUB_OFFLINE="${VTB_AGENT_HF_OFFLINE}" \
        --env PYTHONNOUSERSITE="1" \
        --env NUM_GPUS="${NUM_GPUS}" \
        --env VTB_VISIBLE_NUM_GPUS="${NUM_GPUS}" \
        --env VTB_MIN_CUDA_CAPABILITY="${VTB_MIN_CUDA_CAPABILITY:-8.0}" \
        --env VTB_MODEL_TO_TRAIN="${MODEL_TO_TRAIN}" \
        --env GRADING_URL="${GRADING_URL}" \
        --env PROMPT="${PROMPT}" \
        --env AGENT_CONFIG="${AGENT_CONFIG}" \
        --env VTB_API_KEY="${AGENT_API_KEY}" \
        "${AGENT_ENV_ARGS[@]}" \
        "${HF_ENV_ARGS[@]}" \
        "${RESUME_ENV[@]}" \
        -v "${JOB_DIR}:${VTB_AGENT_HOME}" \
        -v "${JOB_TMP}:/tmp" \
        -v "${DISCLOSURE_DIR}:${VTB_AGENT_HOME}/grading_results:ro" \
        "${BASELINE_DISCLOSURE_ARGS[@]}" \
        "${HF_MOUNT_ARGS[@]}" \
        -w "${VTB_AGENT_HOME}/task" \
        "${VTB_TRAINER_IMAGE}" \
        bash -c "export PATH=\"\$HOME/.local/bin:\$PATH\"; source \$HOME/bootstrap_mutable_framework.sh || exit 1; { python3 \$HOME/check_cuda.py && python3 \$HOME/check_cuda_writing.py || exit 1; bash \$HOME/system_monitor.sh & MONITOR_PID=\$!; bash \$HOME/agent_solve.sh; kill \$MONITOR_PID 2>/dev/null; } 2>&1 | python3 \$HOME/timestamp_lines.py" > "${attempt_out}" 2>&1
    local exit_code=$?

    # The timeout above reaches the client, not the container. Stop it explicitly, with
    # the same grace period, before anything on this host reads the job directory --
    # otherwise a training process outliving its budget is still writing checkpoints
    # into the tree being snapshotted below.
    docker stop --time 30 "$AGENT_CONTAINER" >/dev/null 2>&1 || true
    docker rm --force "$AGENT_CONTAINER" >/dev/null 2>&1 || true
    if [ "$attempt" -gt 0 ]; then
        printf '[%s] ===== resume attempt %s =====\n' "$(date -u +%FT%TZ)" "$attempt" >> "${SOLVE_OUT}"
        cat "${attempt_out}" >> "${SOLVE_OUT}"
        rm -f "${attempt_out}"
    fi
    return $exit_code
}

# Minutes of budget left, read from the same timer file the agent reads. A resume is
# only offered when this answers at all -- without a deadline, a retry cannot be
# bounded by the budget and could run past it.
remaining_minutes() {
    local created hours
    created=$(grep -oE '^CREATION_DATE=[0-9]+' "${JOB_DIR}/task/timer.sh" 2>/dev/null | cut -d= -f2)
    hours=$(grep -oE '^NUM_HOURS=[0-9]+' "${JOB_DIR}/task/timer.sh" 2>/dev/null | cut -d= -f2)
    [ -n "$created" ] && [ -n "$hours" ] || return 1
    echo $(( (created + hours * 3600 - $(date +%s)) / 60 ))
}

# An agent CLI can end its session cleanly after a turn with no actionable content (an
# empty result) -- a dead session, not a finished task. While budget remains, continue
# the session with a nudge instead of delivering nothing. The CLI's session state lives
# under the persistent job directory ($HOME/.claude), so a fresh container resumes it.
# Resumptions stop on delivery, on budget, on the cap, or on a churn signature (a
# resume that dies again within two minutes of container start).
#
# True when the previous session ended with the agent's own concluding text -- a
# deliberate stop. The resume loop exists for unintended exits (a crash, an API failure,
# the CLI's thinking-only auto-quit, or a final turn that came back as malformed output);
# a session whose last word was the agent's is the researcher saying it is done, and
# that decision is honoured. The judgement lives in session_verdict.py, where it is tested.
session_ended_deliberately() {
    "$VTB_PYTHON" src/commit_utils/session_verdict.py "${SOLVE_OUT}"
}

# Delivered means the official submission can be attempted: final_model is present, and
# on an editable-harness benchmark so is the selection record the delivery step below
# requires beside it. A final_model alone is not a delivery there -- a session can export
# one and die before choosing, and the delivery step then refuses it for want of
# final_candidate.json, so the run spends its budget and submits nothing.
delivery_closed() {
    [ -s "${JOB_DIR}/task/final_model/config.json" ] || return 1
    if [ -d "${JOB_DIR}/task/harness" ] && [ ! -f "${JOB_DIR}/task/final_candidate.json" ]; then
        return 1
    fi
    return 0
}

solve_with_resumes() {
    solve_task
    local exit_code=$?
    local max_resumes="${VTB_MAX_RESUMES:-10}"
    local attempt=0
    while ! delivery_closed && [ "$attempt" -lt "$max_resumes" ]; do
        local remaining
        remaining=$(remaining_minutes) || break
        if [ "$remaining" -lt 30 ]; then
            echo "[resume] ${remaining}m of budget left -- not worth restarting the agent" >&2
            break
        fi
        # Only unintended exits are resumed. A session that ended with the agent's own
        # concluding text is a deliberate stop -- not an infrastructure failure to
        # repair -- and the researcher's decision is honoured.
        if session_ended_deliberately; then
            echo "[resume] previous session ended with a deliberate final answer; not resuming" >&2
            break
        fi
        attempt=$((attempt + 1))
        if [ -s "${JOB_DIR}/task/final_model/config.json" ]; then
            echo "[resume] final_model exists but the delivery is not closed (no final_candidate.json); continuing the agent session (resume ${attempt}/${max_resumes}, ${remaining}m of budget left)" >&2
            export VTB_RESUME_NUDGE="Your previous session ended unexpectedly -- it was interrupted, or its last turn came back as malformed output that could not be acted on and was discarded. Your workspace, notes, and artifacts are intact, and a final_model/ exists, but the delivery is not complete: a candidate is a model-and-harness pair, and final_candidate.json must record the selection as environment.md (## Candidates) describes. Continue the task from where you stopped. Keep thinking short, and always end a turn with a tool call or a text answer -- a turn that is only thinking ends your session."
        else
            echo "[resume] no final_model delivered; continuing the agent session (resume ${attempt}/${max_resumes}, ${remaining}m of budget left)" >&2
            export VTB_RESUME_NUDGE="Your previous session ended unexpectedly without delivering -- it was interrupted (a crash or API failure), its last turn produced no actionable content (no text, no tool call), or that turn came back as malformed output that could not be acted on and was discarded. Your workspace, notes, and artifacts are intact. Continue the task from where you stopped. Keep thinking short, and always end a turn with a tool call or a text answer -- a turn that is only thinking ends your session."
        fi
        local started=$SECONDS
        solve_task "$((remaining + 5))" "$attempt"
        exit_code=$?
        if [ $((SECONDS - started)) -lt 120 ] && ! delivery_closed; then
            echo "[resume] resume ${attempt} died within two minutes without delivering; stopping resumptions" >&2
            break
        fi
    done
    return $exit_code
}
# Hand the job directory back to this account.
#
# The agent container runs as the trainer image's own user, root: that is where slime
# keeps its training code (/root/slime) and its site-packages, so a --user flag would
# leave the agent unable to read the thing it is here to run. The cost is that
# everything it wrote is owned by uid 0 on this host, while the rest of this script --
# the weight manifest, the final_model copy, the scratch gate's rm -rf -- runs as the
# invoking user and has to read and delete those files. Only root can hand them back,
# hence a throwaway container to do it.
#
# Named subpaths rather than the whole scratch tree: when the legacy HuggingFace overlay
# mode is enabled, a recursive chown through the full scratch tree would copy the entire
# shared cache up into the overlay's upper layer.
reclaim_job_dir() {
    docker run --rm --user 0:0 \
        -v "${TMP_SUBDIR}:/reclaim" \
        --entrypoint chown \
        "${VTB_TRAINER_IMAGE}" \
        -R "$(id -u):$(id -g)" /reclaim/job_dir /reclaim/tmp
}

echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"

run_with_agent_hf_cache with_record_the_time solve_with_resumes
SOLVE_EXIT=$?

# Before anything here touches what the agent wrote. Non-fatal: on a host where this
# account is already root there is nothing to hand back.
reclaim_job_dir >/dev/null 2>&1 || echo "[reclaim] chown of the job directory failed (non-fatal)"

echo "--- SOLVE DIAGNOSTICS ---"
echo "exit_code: $SOLVE_EXIT"
if [ $SOLVE_EXIT -eq 0 ]; then
    echo "status: exited normally"
elif [ $SOLVE_EXIT -eq 124 ]; then
    echo "status: killed by timeout (reached ${NUM_HOURS}h limit)"
elif [ $SOLVE_EXIT -gt 128 ]; then
    echo "status: killed by signal $((SOLVE_EXIT - 128)) ($(kill -l $((SOLVE_EXIT - 128)) 2>/dev/null || echo unknown))"
else
    echo "status: exited with error code $SOLVE_EXIT"
fi
echo "final_model_files: $(ls "${JOB_DIR}/task/final_model/" 2>/dev/null | wc -l)"
# Do not include the operator's hostname in diagnostics that may be published.
echo "fuse_overlayfs_alive: $(ps aux 2>/dev/null | grep fuse-overlay | grep -v grep | wc -l)"
echo "disk_job_dir: $(du -sh "${JOB_DIR}" 2>/dev/null | cut -f1)"
echo "disk_tmp: $(du -sh "${JOB_TMP}" 2>/dev/null | cut -f1)"
echo "memory: $(free -m 2>/dev/null | grep Mem | awk '{print "total=" $2 "MB used=" $3 "MB free=" $4 "MB"}')"
echo "--- END SOLVE DIAGNOSTICS ---"

# === [delivery-audit] snapshot weight fingerprints BEFORE any judge/cleanup deletes them ===
# Records sampled hashes of every checkpoint + full sha256 of final_model files, so a
# corrupted/mismatched delivery can later be traced back to its source checkpoint.
"$VTB_PYTHON" src/utils/delivery_audit.py manifest "${JOB_DIR}/task" --out "${EVAL_DIR}/weights_manifest.txt" || echo "[delivery-audit] manifest failed (non-fatal)"

# === [delivery-audit] GPU-isolation audit ===
# The allocation is enforced by the runtime: `docker run --gpus "device=..."` puts only
# those cards in the container, so an agent re-exporting CUDA_VISIBLE_DEVICES cannot
# reach another run's GPU -- it can only point CUDA at a device that is not there.
#
# It still runs, because the attempt is worth recording and because a run given the whole
# node (VTB_AGENT_GPUS covering every card) has no isolation left to rely on. The indices
# to judge against are the container's own, 0..NUM_GPUS-1: that is what CUDA and
# nvidia-smi report inside it, and therefore what appears in the trajectory.
ALLOCATED_GPU="$(seq -s, 0 $((NUM_GPUS - 1)))"
"$VTB_PYTHON" src/utils/delivery_audit.py gpu-audit "${SOLVE_OUT}" \
    --allowed-gpu "${ALLOCATED_GPU}" \
    --system-monitor "${JOB_DIR}/task/system_monitor.log" \
    --out "${EVAL_DIR}/delivery_audit.json" >> "${EVAL_DIR}/delivery_audit.log" 2>&1 \
    || echo "[delivery-audit] gpu-audit failed (non-fatal)" | tee -a "${EVAL_DIR}/delivery_audit.log"


echo "============================================"
echo "=== TASK COMPLETE, PARSING AGENT TRACE ==="
echo "============================================"

# Parse agent trace into human-readable format.
#
# Either way the trace ends up in EVAL_DIR: it is the record of what the agent actually
# did, and it has to outlive the sandbox. Without the copy below an agent with no parser
# left its trace only in scratch, which is the half that gets deleted.
TRACE_PARSER="${AGENT_ADAPTER_DIR}/human_readable_trace.py"
if [ -f "$TRACE_PARSER" ]; then
    echo "Using trace parser: $TRACE_PARSER"
    "$VTB_PYTHON" "$TRACE_PARSER" "${SOLVE_OUT}" -o "${EVAL_DIR}/solve_parsed.txt"
    cp "${EVAL_DIR}/solve_parsed.txt" "${JOB_DIR}/solve_parsed.txt"
else
    echo "Warning: No trace parser found at agents/${AGENT}/human_readable_trace.py, using raw output"
    cp "${SOLVE_OUT}" "${EVAL_DIR}/solve_parsed.txt"
    cp "${SOLVE_OUT}" "${JOB_DIR}/solve_parsed.txt"
fi


# === [delivery-audit] preserve every agent harness's native session store (raw trajectory) ===
# solve_out.txt is the merged stdout+stderr of the whole container command with timestamps
# prepended, so both it and the parsed trace lack what only the native session store carries:
# the conversation DAG (parentUuid/logicalParentUuid), context-compaction events
# (isCompactSummary/compactMetadata) and sidechain/subagent markers. Each CLI writes that store
# under the container HOME (= JOB_DIR), which disappears with scratch, so copy the raw .jsonl
# out next to the results, one subdirectory per harness. Harness-agnostic: whichever stores are
# present get archived, so a new agent only needs its store directory added to the list below.
#   Claude CLI adapters -> .claude/projects (CLI session logs)
#   codex                     -> .codex/sessions    (codex CLI rollout logs)
_archived_sessions=0
while read -r _store _label; do
    [ -n "$_store" ] || continue
    _src="${JOB_DIR}/${_store}"
    [ -d "$_src" ] || continue
    [ -n "$(find "$_src" -name '*.jsonl' -print -quit 2>/dev/null)" ] || continue
    _dst="${EVAL_DIR}/agent_sessions/${_label}"
    mkdir -p "$_dst"
    if cp -r "$_src/." "$_dst/" 2>/dev/null; then
        _archived_sessions=1
        echo "[agent-sessions] ${_label}: preserved $(find "$_dst" -name '*.jsonl' | wc -l) session file(s), $(du -sh "$_dst" | cut -f1)"
    else
        echo "[agent-sessions] ${_label}: copy failed (non-fatal)"
    fi
done <<'SESSION_STORES'
.claude/projects claude
.codex/sessions codex
SESSION_STORES
[ "$_archived_sessions" -eq 0 ] && echo "[agent-sessions] no raw session store found under ${JOB_DIR} (non-fatal)"

echo "============================="
echo "======== CLEANING UP ========"
echo "============================="

echo "Task directory contents:"
tree ${JOB_DIR}/task
echo "================================"

if [ -d "${JOB_DIR}/task/final_model" ]; then
    cp -r "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model"
    # [delivery-audit] byte-level verification of the scratch->results copy; retry once on mismatch
    if ! "$VTB_PYTHON" src/utils/delivery_audit.py verify "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model" >> "${EVAL_DIR}/delivery_audit.log" 2>&1; then
        echo "[delivery-audit] final_model copy MISMATCH, retrying once" | tee -a "${EVAL_DIR}/delivery_audit.log"
        rm -rf "$EVAL_DIR/final_model"
        cp -r "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model"
        "$VTB_PYTHON" src/utils/delivery_audit.py verify "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model" >> "${EVAL_DIR}/delivery_audit.log" 2>&1 \
            || echo "[delivery-audit] final_model copy STILL MISMATCHED after retry" | tee -a "${EVAL_DIR}/delivery_audit.log"
    fi
fi

# Candidate model copies can be large, so preserve the durable iteration records rather
# than every candidate weight in the results tree. This keeps every submission's change
# log and machine-readable identity after the scratch workspace is cleaned up.
if [ -d "${JOB_DIR}/task/candidates" ]; then
    mkdir -p "${EVAL_DIR}/candidate_records"
    while IFS= read -r -d '' candidate_record; do
        candidate_id="$(basename "$(dirname "$candidate_record")")"
        record_dir="${EVAL_DIR}/candidate_records/${candidate_id}"
        mkdir -p "$record_dir"
        cp "$candidate_record" "$record_dir/$(basename "$candidate_record")"
    done < <(find "${JOB_DIR}/task/candidates" -mindepth 2 -maxdepth 2 -type f \
        \( -name candidate.md -o -name candidate.json \) -print0)
fi

# Editable-harness benchmarks deliver a model and its agent loop as one candidate.
# The grading server resolves the harness only beside final_model, so
# freeze the selected workspace copy in the official token's sandbox before submit.
if [ -d "${JOB_DIR}/task/harness" ]; then
    if [ ! -f "${JOB_DIR}/task/final_candidate.json" ]; then
        echo "[delivery] editable harness requires task/final_candidate.json" >&2
        exit 1
    fi
    cp -r "${JOB_DIR}/task/harness" "$EVAL_DIR/harness"
    cp "${JOB_DIR}/task/final_candidate.json" "$EVAL_DIR/final_candidate.json"
fi

if [ -f "${JOB_DIR}/task/system_monitor.log" ]; then
    cp "${JOB_DIR}/task/system_monitor.log" "$EVAL_DIR/system_monitor.log"
fi

# [delivery-audit] move checkpoints to scratch quarantine instead of deleting them, so the
# scratch-gate at the end of this script can keep them if the delivery turns out broken
"$VTB_PYTHON" containers/delete_hf_models.py "${JOB_DIR}/task" --quarantine "${TMP_SUBDIR}/quarantined_models"

# Archive the workspace, minus the trainer's own checkpoint directory.
#
# Quarantine detects HF-format models through weight, config, and tokenizer files.
# Slime torch_dist checkpoints do not match those checks, so exclude checkpoints/
# from the archive to avoid copying large intermediate training artifacts.
#
# Excluding them loses nothing that is relied on: final_model was already copied and
# verified above, and if the delivery turns out broken the scratch-gate at the end of this
# script keeps the whole scratch tree, checkpoints included, where they can still be
# inspected. rsync rather than cp because the exclusion is the entire point.
#
# The mkdir is not redundant: rsync creates only the last component of its destination, so
# it fails outright if a parent is missing. $EVAL_DIR exists by now, which is the only
# reason a bare rsync would work -- and this runs after the agent's whole budget has been
# spent, which is the wrong place to depend on that.
mkdir -p "$EVAL_DIR/task"
rsync -a --exclude='checkpoints/' "${JOB_DIR}/task/" "$EVAL_DIR/task/" \
    || echo "[archive] rsync of task workspace failed" | tee -a "${EVAL_DIR}/delivery_audit.log"



echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

export REPO_ROOT="$(pwd)"

# The HuggingFace cache the smoke container uses. A container-internal path: the check
# below loads a local checkpoint and downloads nothing, so there is no reason to give it
# the shared cache.
export TMP_HF_CACHE="/tmp/hf_cache_90afd0"

# === [delivery-audit] verify the delivered model before evaluation ===
# 1) Header/EOS/tokenizer checks and sampled weight differences (host, stdlib only).
#    --fix-tokenizer converts the transformers-v5 extra_special_tokens list into
#    a mapping compatible with the serving image's tokenizer loader.
# Compared against the copy the agent was handed rather than the repo's model cache: the
# question is whether these bytes came from that checkpoint, and the cache can have been
# re-materialized since the job started. Empty when the layout does not match, and the
# audit then says outright that provenance went unchecked.
BASE_HF_DIR=""
for _cand in "${JOB_DIR}"/task/train_resources/models/*/hf_checkpoint; do
    if [ -d "$_cand" ]; then BASE_HF_DIR="$_cand"; break; fi
done
"$VTB_PYTHON" src/utils/delivery_audit.py check "$EVAL_DIR/final_model" \
    --model "${MODEL_TO_TRAIN}" --hf-home "${HF_HOME}" ${BASE_HF_DIR:+--base-dir "${BASE_HF_DIR}"} --fix-tokenizer \
    --out "${EVAL_DIR}/delivery_audit.json" >> "${EVAL_DIR}/delivery_audit.log" 2>&1 \
    || echo "[delivery-audit] check failed (non-fatal)" | tee -a "${EVAL_DIR}/delivery_audit.log"
# 2) smoke generation with transformers (bypasses the HTTP server), catches garbage weights immediately.
#
# Runs in the same serving image official evaluation uses, so a garbage-weights
# delivery fails in the same stack that would score it. Operators can still override
# the image with VTB_DELIVERY_CHECK_IMAGE.
# Runs as this account rather than root: it writes its verdict into ${EVAL_DIR}, which is
# kept, and a root-owned file in the results tree is a nuisance to clean up afterwards.
# One card, the first of this run's own allocation, so the check cannot collide with the
# grading service's reserved GPU while an evaluation is queued on it.
DELIVERY_CHECK_IMAGE="${VTB_DELIVERY_CHECK_IMAGE:-${VTB_SERVE_SGLANG_DOCKER_IMAGE:-${VTB_SERVE_SGLANG_IMAGE:-${VTB_IMAGE_NAMESPACE}/sglang:${VTB_IMAGE_TAG}}}}"
SMOKE_CONTAINER="vtb-smoke-$(printf '%s' "${RANDOM_UUID}" | tr -c 'a-zA-Z0-9_.-' '-')"
SMOKE_MOUNTS=(-v "${REPO_ROOT}:${REPO_ROOT}" -v "${EVAL_DIR}:${EVAL_DIR}")
EVAL_DIR_REAL="$(readlink -f "$EVAL_DIR")"
[ "$EVAL_DIR_REAL" != "$EVAL_DIR" ] && SMOKE_MOUNTS+=(-v "${EVAL_DIR_REAL}:${EVAL_DIR_REAL}")
if ! docker image inspect "${DELIVERY_CHECK_IMAGE}" >/dev/null 2>&1; then
    echo "[delivery-audit] delivery check image not present: ${DELIVERY_CHECK_IMAGE} (non-fatal)" | tee -a "${EVAL_DIR}/delivery_audit.log"
else
    timeout 900 docker run --rm \
        --name "$SMOKE_CONTAINER" \
        --gpus "device=${VTB_AGENT_GPUS%%,*}" \
        --ipc host \
        --shm-size "${VTB_SMOKE_SHM_SIZE:-8g}" \
        --user "$(id -u):$(id -g)" \
        --env PYTHONNOUSERSITE="1" \
        --env HOME="/tmp" \
        --env TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor" \
        --env TRITON_CACHE_DIR="/tmp/triton" \
        --env HF_HOME="${TMP_HF_CACHE}" \
        "${SMOKE_MOUNTS[@]}" \
        -w "${REPO_ROOT}" \
        "${DELIVERY_CHECK_IMAGE}" \
        python3 "${REPO_ROOT}/src/utils/delivery_audit.py" smoke "$EVAL_DIR/final_model" \
        --out "${EVAL_DIR}/delivery_audit.json" >> "${EVAL_DIR}/delivery_audit.log" 2>&1 \
        || echo "[delivery-audit] smoke test failed (non-fatal)" | tee -a "${EVAL_DIR}/delivery_audit.log"
    docker rm --force "$SMOKE_CONTAINER" >/dev/null 2>&1 || true
fi
grep -h '"verdict"' "${EVAL_DIR}/delivery_audit.json" 2>/dev/null || true

# The official score. One call: the grading service owns the retry ladder (see
# eval/grading/config.py TOKEN_LADDER) and serialises against every other run's
# evaluations, so concurrent runs cannot stack model servers onto the evaluation card.
#
# --write-metrics is what puts ${EVAL_DIR}/metrics.json in place, unchanged in
# format: the machine-readable official score, and the completion signal the
# scratch gate below checks for.
#
# The model path is relative to the official token's sandbox root (${EVAL_DIR}),
# so it resolves to the verified copy at ${EVAL_DIR}/final_model.
FINAL_LIMIT_ARGS=()
if [ "${VTB_FINAL_LIMIT:-}" -gt 0 ] 2>/dev/null; then
    FINAL_LIMIT_ARGS=(--limit "${VTB_FINAL_LIMIT}")
fi

echo "Submitting final_model for official evaluation..."
"$VTB_PYTHON" "$GRADING_CLIENT" submit \
    --token-file "$OFFICIAL_TOKEN_FILE" \
    --model-path "final_model" \
    --mode final \
    "${FINAL_LIMIT_ARGS[@]}" \
    --write-metrics \
    --wait --wait-timeout "${VTB_FINAL_WAIT_TIMEOUT:-86400}" --poll 30 \
    > "${EVAL_DIR}/grading_result.json" 2> "${EVAL_DIR}/grading_progress.log"
GRADING_EXIT=$?

echo "--- GRADING RESULT (exit ${GRADING_EXIT}) ---"
cat "${EVAL_DIR}/grading_result.json"
echo "--- END GRADING RESULT ---"

# Release both tokens. A stale entry is only a token->paths mapping and stays
# confined by its own sandbox root, but there is no reason to leave it behind.
for _tok in "$OFFICIAL_TOKEN_FILE" "${JOB_DIR}/task/grading_token"; do
    [ -f "$_tok" ] && "$VTB_PYTHON" "$GRADING_CLIENT" unregister --token-file "$_tok" > /dev/null 2>&1
done
rm -f "$OFFICIAL_TOKEN_FILE"

# === [delivery-audit] scratch gate: only discard scratch when the delivery is proven good ===
SCRATCH_OK=1
[ -f "${EVAL_DIR}/metrics.json" ] || { SCRATCH_OK=0; echo "[scratch-gate] metrics.json missing"; }
if grep -q '"verdict": "SUSPICIOUS"' "${EVAL_DIR}/delivery_audit.json" 2>/dev/null; then
    SCRATCH_OK=0; echo "[scratch-gate] delivery_audit verdict SUSPICIOUS"
fi
if grep -q 'MISMATCH' "${EVAL_DIR}/delivery_audit.log" 2>/dev/null; then
    SCRATCH_OK=0; echo "[scratch-gate] copy verification reported MISMATCH"
fi
if grep -q '"verdict": "VIOLATED"' "${EVAL_DIR}/delivery_audit.json" 2>/dev/null; then
    SCRATCH_OK=0; echo "[scratch-gate] gpu-audit verdict VIOLATED (ran on non-allocated GPU) -> score not comparable"
fi
if [ "$SCRATCH_OK" = "1" ] && [ "${VTB_KEEP_SCRATCH:-0}" != "1" ]; then
    echo "[scratch-gate] delivery verified OK -> removing scratch ${TMP_SUBDIR}"
    rm -rf "${TMP_SUBDIR}"
else
    echo "[scratch-gate] KEEPING scratch for forensics: ${TMP_SUBDIR}"
    echo "[scratch-gate] quarantined checkpoints (if any): ${TMP_SUBDIR}/quarantined_models"
    echo "${TMP_SUBDIR}" > "${EVAL_DIR}/scratch_kept.txt"
fi

echo "================================"
echo "======= EVALUATION DONE ========"
echo "================================"