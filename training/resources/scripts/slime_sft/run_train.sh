#!/usr/bin/env bash
set -euo pipefail

# Slime/Megatron SFT train-test entrypoint.
# This script owns framework orchestration. Task data paths and training knobs
# are supplied by the selected recipe's env_defaults or explicit environment overrides.

MODEL_ARG="${1:-${VTB_MODEL_TO_TRAIN:-}}"
if [ -z "${MODEL_ARG}" ]; then
  echo "Usage: VTB_MODEL_TO_TRAIN=<model-or-local-path> bash $0" >&2
  echo "   or: bash $0 <model-or-local-path>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

apply_manifest_env_defaults() {
  local manifest="${TASK_DIR}/train_resource_manifest.lock.json"
  [ -f "${manifest}" ] || return 0
  eval "$(python3 - "${manifest}" <<'PY'
import json
import os
import re
import shlex
import sys

manifest_path = sys.argv[1]
with open(manifest_path, "r", encoding="utf-8") as f:
    defaults = json.load(f).get("env_defaults") or {}

visible_num_gpus = os.environ.get("VTB_VISIBLE_NUM_GPUS")
for key, value in defaults.items():
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", str(key)):
        continue
    value = str(value)
    current = os.environ.get(key)
    if current is None:
        print(f"export {key}={shlex.quote(value)}")
        continue
    if key == "NUM_GPUS" and visible_num_gpus and current == visible_num_gpus:
        try:
            if int(value) <= int(visible_num_gpus):
                print(f"export {key}={shlex.quote(value)}")
        except ValueError:
            pass
PY
  )"
}
apply_manifest_env_defaults

SLIME_DIR="${SLIME_DIR:-${VTB_SLIME_DIR:-/root/slime}}"
MEGATRON_DIR="${MEGATRON_DIR:-${VTB_MEGATRON_DIR:-/root/Megatron-LM}}"

MODEL_NAME="${VTB_TRAIN_MODEL_NAME:-$(basename "${MODEL_ARG}")}"
RUN_NAME="${RUN_NAME:-train-test-sft}"

TRAIN_DATA="${TRAIN_DATA:-}"
if [ -z "${TRAIN_DATA}" ]; then
  TRAIN_DATA_CANDIDATES=()
  if [ -f "${TASK_DIR}/train_resource_manifest.lock.json" ]; then
    mapfile -t TRAIN_DATA_CANDIDATES < <(python3 - "${TASK_DIR}/train_resource_manifest.lock.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
task_dir = manifest_path.parent
with manifest_path.open("r", encoding="utf-8") as f:
    manifest = json.load(f)

for resource in manifest.get("resources", []):
    resource_id = str(resource.get("id") or "")
    if not resource_id.startswith("data/"):
        continue
    base = task_dir / resource["copied_to"]
    metadata = resource.get("metadata") or {}
    materialized_file = metadata.get("materialized_file")
    if isinstance(materialized_file, str) and materialized_file.strip():
        print(base / materialized_file.strip())
        continue
    for path in sorted(base.rglob("*.jsonl")):
        print(path)
PY
)
  fi
  if [ "${#TRAIN_DATA_CANDIDATES[@]}" -eq 1 ]; then
    TRAIN_DATA="${TRAIN_DATA_CANDIDATES[0]}"
  elif [ "${#TRAIN_DATA_CANDIDATES[@]}" -gt 1 ]; then
    echo "Multiple train_data candidates found; set TRAIN_DATA explicitly before running this entrypoint:" >&2
    printf '  %s\n' "${TRAIN_DATA_CANDIDATES[@]}" >&2
    exit 2
  fi
fi
if [ -n "${TRAIN_DATA}" ] && [[ "${TRAIN_DATA}" != /* ]]; then
  TRAIN_DATA="${TASK_DIR}/${TRAIN_DATA}"
fi

MODEL_RESOURCE_PATHS=()
if [ -f "${TASK_DIR}/train_resource_manifest.lock.json" ]; then
  mapfile -t MODEL_RESOURCE_PATHS < <(python3 - "${TASK_DIR}/train_resource_manifest.lock.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
task_dir = manifest_path.parent
with manifest_path.open("r", encoding="utf-8") as f:
    manifest = json.load(f)

for resource in manifest.get("resources", []):
    if resource.get("type") in {"cached_model", "local_model"}:
        copied_to = task_dir / resource["copied_to"]
        print(copied_to / "hf_checkpoint")
        print(copied_to / "torch_dist")
        break
PY
)
fi
DEFAULT_HF_CHECKPOINT="${MODEL_RESOURCE_PATHS[0]:-}"
DEFAULT_REF_LOAD="${MODEL_RESOURCE_PATHS[1]:-}"

if [ -n "${HF_CHECKPOINT:-}" ]; then
  HF_CHECKPOINT="${HF_CHECKPOINT}"
elif [ -n "${DEFAULT_HF_CHECKPOINT}" ] && [ -d "${DEFAULT_HF_CHECKPOINT}" ]; then
  HF_CHECKPOINT="${DEFAULT_HF_CHECKPOINT}"
elif [ -d "${MODEL_ARG}" ]; then
  HF_CHECKPOINT="${MODEL_ARG}"
else
  HF_CHECKPOINT="${TASK_DIR}/models/${MODEL_NAME}"
fi
if [ -n "${REF_LOAD:-}" ]; then
  REF_LOAD="${REF_LOAD}"
elif [ -n "${DEFAULT_REF_LOAD}" ] && [ -d "${DEFAULT_REF_LOAD}" ]; then
  REF_LOAD="${DEFAULT_REF_LOAD}"
else
  REF_LOAD="${TASK_DIR}/models/${MODEL_NAME}_torch_dist"
fi
SAVE_DIR="${SAVE_DIR:-${TASK_DIR}/checkpoints/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${TASK_DIR}/final_model}"
LOG_DIR="${LOG_DIR:-${TASK_DIR}/logs}"

required_training_vars=(
  SLIME_MODEL_CONFIG
  LOSS_MASK_TYPE
  NUM_GPUS
  TP_SIZE
  PP_SIZE
  CP_SIZE
  MAX_TOKENS_PER_GPU
  ROLLOUT_BATCH_SIZE
  GLOBAL_BATCH_SIZE
  NUM_EPOCH
  SAVE_INTERVAL
  SAVE_OPTIM
  LR
  MIN_LR
)
for var in "${required_training_vars[@]}"; do
  if [ -z "${!var:-}" ]; then
    echo "${var} is not set; provide it in the recipe env_defaults or explicit environment" >&2
    exit 2
  fi
done
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-}"
OBJECT_STORE_BYTES="${OBJECT_STORE_BYTES:-$((16 * 1024 ** 3))}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
if [ -n "${DASHBOARD_PORT:-}" ]; then
  DASHBOARD_PORT="${DASHBOARD_PORT}"
else
  DASHBOARD_PORT="$(python3 - <<'PY'
import socket
for port in range(8266, 8300):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit("no free Ray dashboard port in 8266..8299")
PY
)"
fi

if [ ! -d "${SLIME_DIR}" ]; then
  echo "SLIME_DIR does not exist: ${SLIME_DIR}" >&2
  exit 1
fi
if [ ! -f "${SLIME_DIR}/scripts/models/${SLIME_MODEL_CONFIG}" ]; then
  echo "missing model config: ${SLIME_DIR}/scripts/models/${SLIME_MODEL_CONFIG}" >&2
  exit 1
fi
if [ -z "${TRAIN_DATA}" ]; then
  echo "TRAIN_DATA is not set; select one JSONL from the copied data resources or set TRAIN_DATA explicitly" >&2
  exit 1
fi
if [ ! -f "${TRAIN_DATA}" ]; then
  echo "TRAIN_DATA does not exist: ${TRAIN_DATA}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${SLIME_DIR}/scripts/models/${SLIME_MODEL_CONFIG}"

if [ ! -d "${HF_CHECKPOINT}" ]; then
  echo "HF_CHECKPOINT not found; downloading ${MODEL_ARG} to ${HF_CHECKPOINT}"
  mkdir -p "$(dirname "${HF_CHECKPOINT}")"
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id=${MODEL_ARG@Q}, local_dir=${HF_CHECKPOINT@Q})
PY
fi
if [ ! -d "${REF_LOAD}" ]; then
  echo "REF_LOAD not found; converting HF checkpoint to Megatron torch_dist at ${REF_LOAD}"
  mkdir -p "$(dirname "${REF_LOAD}")"
  (cd "${SLIME_DIR}" && PYTHONPATH="${MEGATRON_DIR}" python3 tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --save "${REF_LOAD}")
fi

mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

# LOG_DIR used to be created and then never written to, which read as "logging is arranged"
# while the only copy of a loss curve was whatever the caller happened to redirect. One run
# survived solely because the agent had thought to pipe through tee; ray's own session logs
# under /tmp do carry the same lines, but /tmp is thrown away with the job sandbox.
TRAIN_LOG="${LOG_DIR}/train_${RUN_NAME}.log"

# What produced these weights. slime and Megatron live in the image, owned by root, and the
# container runs as root so the agent can both read and edit them -- editing the loss
# function is allowed. But `docker run --rm` discards that filesystem and nothing else
# records it, so a run with a modified objective would be indistinguishable from one
# without. The diff goes beside the training log, inside the task tree, which is archived
# with the run. Untracked files are listed too: a new file is also a change.
#
# `git diff HEAD`, not `git diff`. The first smoke run of this block reported 21 modified
# files under Megatron and an empty diff, because those modifications are staged: porcelain
# `M ` is index-vs-HEAD, and plain `git diff` only shows worktree-vs-index. Anything an
# agent stages -- or any tool that runs `git add` -- would have vanished from the record.
#
# Those 21 files are also the image's own: the trainer image ships Megatron already
# patched. This file therefore cannot by itself prove the agent changed anything; what it
# proves is what the code was. Two runs on the same image tag produce identical output here
# unless someone edited something, which is the comparison to make.
FRAMEWORK_STATE="${LOG_DIR}/framework_state_${RUN_NAME}.txt"
{
  echo "# recorded $(date -u '+%Y-%m-%dT%H:%M:%SZ') before launching ${RUN_NAME}"
  echo "# the image ships Megatron pre-patched, so entries here are not by themselves agent edits"
  for _framework_dir in "${SLIME_DIR}" "${MEGATRON_DIR}"; do
    echo "=== ${_framework_dir} ==="
    if [ -d "${_framework_dir}/.git" ]; then
      echo "HEAD $(git -C "${_framework_dir}" rev-parse HEAD 2>/dev/null || echo unknown)"
      echo "diff-sha256 $(git -C "${_framework_dir}" diff HEAD 2>/dev/null | sha256sum | cut -d' ' -f1)"
      echo "--- status (empty means identical to HEAD) ---"
      git -C "${_framework_dir}" status --porcelain 2>/dev/null || true
      echo "--- diffstat ---"
      git -C "${_framework_dir}" diff HEAD --stat 2>/dev/null || true
      echo "--- diff ---"
      git -C "${_framework_dir}" diff HEAD 2>/dev/null || true
    else
      echo "not a git checkout; cannot diff"
    fi
  done
} > "${FRAMEWORK_STATE}" 2>&1
echo "Framework state recorded: ${FRAMEWORK_STATE}"
echo "Training log: ${TRAIN_LOG}"

ray stop --force >/dev/null 2>&1 || true

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_LOAD}"
  --load "${SAVE_DIR}"
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
)
if [ "${SAVE_OPTIM}" != "1" ]; then
  CKPT_ARGS+=(--no-save-optim)
fi

SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data "${TRAIN_DATA}"
  --input-key messages
  --rollout-shuffle
  --num-epoch "${NUM_EPOCH}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --loss-type sft_loss
  --loss-mask-type "${LOSS_MASK_TYPE}"
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)

if [ -n "${MAX_PROMPT_LEN}" ]; then
  SFT_ARGS+=(--rollout-max-prompt-len "${MAX_PROMPT_LEN}")
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)
if [ "${DISABLE_RECOMPUTE:-0}" != "1" ]; then
  PERF_ARGS+=(
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
  )
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style cosine
  --min-lr "${MIN_LR}"
  --lr-warmup-fraction "${LR_WARMUP_FRACTION:-0.1}"
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --use-distributed-optimizer
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
if [ "${CROSS_ENTROPY_LOSS_FUSION:-0}" = "1" ]; then
  MISC_ARGS+=(--cross-entropy-loss-fusion)
fi

export no_proxy="127.0.0.1,${MASTER_ADDR}"
RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${MEGATRON_DIR}/\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"${HAS_NVLINK}\",\"no_proxy\":\"${no_proxy}\",\"MASTER_ADDR\":\"${MASTER_ADDR}\",\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\"}}"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --object-store-memory "${OBJECT_STORE_BYTES}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${DASHBOARD_PORT}"

cd "${SLIME_DIR}"
# tee, not a plain redirect: the caller still sees the run live, and pipefail is already set
# above so a failing job still fails this script rather than being swallowed by tee's exit.
# Appended with a header, because a second launch under the same RUN_NAME should extend the
# record rather than silently replace it.
echo "=== ${RUN_NAME} launched $(date -u '+%Y-%m-%dT%H:%M:%SZ') ===" >> "${TRAIN_LOG}"
ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${SLIME_DIR}/train_async.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}" 2>&1 | tee -a "${TRAIN_LOG}"

latest_iter="$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || true)"
newest_ckpt="$(find "${SAVE_DIR}" -maxdepth 1 -type d -name 'iter_*' 2>/dev/null | sort | tail -1)"
if [ -z "${newest_ckpt}" ] || ! find "${newest_ckpt}" -name '*.distcp' -print -quit 2>/dev/null | grep -q .; then
  echo "No complete torch_dist checkpoint was produced under ${SAVE_DIR}" >&2
  echo "latest_checkpointed_iteration=${latest_iter:-<none>}" >&2
  exit 1
fi

rm -rf "${OUTPUT_DIR}"
PYTHONPATH="${MEGATRON_DIR}" python3 tools/convert_torch_dist_to_hf.py \
  --input-dir "${newest_ckpt}" \
  --output-dir "${OUTPUT_DIR}" \
  --origin-hf-dir "${HF_CHECKPOINT}"

python3 "${SCRIPT_DIR}/postprocess_final_model.py" \
  --origin-hf-dir "${HF_CHECKPOINT}" \
  --final-model "${OUTPUT_DIR}"

echo "SFT checkpoint exported to ${OUTPUT_DIR}"
