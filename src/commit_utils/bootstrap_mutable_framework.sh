#!/usr/bin/env bash
# 在 agent 容器中、启动外层 agent 前 source 本脚本。
# job directory 挂载在 $HOME，因此副本会随当前任务保留，且与训练镜像和其他任务隔离。

FRAMEWORK_ROOT="${VTB_MUTABLE_FRAMEWORK_ROOT:-${HOME}/task/framework_src}"
SOURCE_SLIME_DIR="${VTB_SLIME_DIR:-/root/slime}"
SOURCE_MEGATRON_DIR="${VTB_MEGATRON_DIR:-/root/Megatron-LM}"

if [ ! -f "${FRAMEWORK_ROOT}/.ready" ] \
  || [ ! -d "${FRAMEWORK_ROOT}/slime" ] \
  || [ ! -d "${FRAMEWORK_ROOT}/Megatron-LM" ]; then
    FRAMEWORK_TMP="${FRAMEWORK_ROOT}.tmp.$$"
    rm -rf "${FRAMEWORK_TMP}" "${FRAMEWORK_ROOT}"
    mkdir -p "${FRAMEWORK_TMP}" || return 1
    cp -a "${SOURCE_SLIME_DIR}" "${FRAMEWORK_TMP}/slime" || return 1
    cp -a "${SOURCE_MEGATRON_DIR}" "${FRAMEWORK_TMP}/Megatron-LM" || return 1

    # 禁用异步 D2H checkpoint 预加载，规避 CUDA invalid argument 保存错误。
    CHECKPOINT_WRITER="${FRAMEWORK_TMP}/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py"
    if grep -Fq 'partial(self.preload_tensors, self.write_buckets, True)' "${CHECKPOINT_WRITER}"; then
        sed -i 's|partial(self\.preload_tensors, self\.write_buckets, True)|partial(self.preload_tensors, self.write_buckets, False)|' "${CHECKPOINT_WRITER}" || return 1
    fi
    grep -Fq 'partial(self.preload_tensors, self.write_buckets, False)' "${CHECKPOINT_WRITER}" || return 1

    # 禁用 actor 权重备份的 pinned memory，规避容器内 cudaHostAlloc 失败。
    TENSOR_BACKUPER="${FRAMEWORK_TMP}/slime/slime/utils/tensor_backper.py"
    if grep -Fq 'pin_memory=True' "${TENSOR_BACKUPER}"; then
        sed -i 's/pin_memory=True/pin_memory=False/' "${TENSOR_BACKUPER}" || return 1
    fi
    grep -Fq 'pin_memory=False' "${TENSOR_BACKUPER}" || return 1
    if grep -Fq 'backup_dict[name].copy_(param.detach(), non_blocking=True)' "${TENSOR_BACKUPER}"; then
        sed -i '/backup_dict\[name\]\.copy_(param\.detach(), non_blocking=True)/s/non_blocking=True/non_blocking=False/' "${TENSOR_BACKUPER}" || return 1
    fi
    grep -Fq 'backup_dict[name].copy_(param.detach(), non_blocking=False)' "${TENSOR_BACKUPER}" || return 1

    touch "${FRAMEWORK_TMP}/.ready" || return 1
    mv "${FRAMEWORK_TMP}" "${FRAMEWORK_ROOT}" || return 1
fi

export SLIME_DIR="${FRAMEWORK_ROOT}/slime"
export MEGATRON_DIR="${FRAMEWORK_ROOT}/Megatron-LM"
export VTB_SLIME_DIR="${SLIME_DIR}"
export VTB_MEGATRON_DIR="${MEGATRON_DIR}"
export PYTHONPATH="${SLIME_DIR}:${MEGATRON_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Using job-local framework sources: ${FRAMEWORK_ROOT}"
