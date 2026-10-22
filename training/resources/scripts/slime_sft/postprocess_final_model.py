#!/usr/bin/env python3
"""Postprocess Slime/Megatron SFT exports for Qwen3.5 train-test resources.

The training script should stay focused on SFT orchestration. This helper owns
only deterministic HF export artifact repair.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def restore_missing_vision_mtp(origin_hf_dir: Path, final_model: Path) -> None:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    base_index_path = origin_hf_dir / "model.safetensors.index.json"
    out_index_path = final_model / "model.safetensors.index.json"
    if not base_index_path.is_file() or not out_index_path.is_file():
        print("post-export component check skipped: missing safetensors index")
        return

    base_index = load_json(base_index_path)
    out_index = load_json(out_index_path)
    base_wm = base_index.get("weight_map") or {}
    out_wm = out_index.get("weight_map") or {}
    missing = [
        key for key in base_wm
        if key not in out_wm and ("visual" in key or key.startswith("mtp."))
    ]

    if missing:
        out_files = sorted(set(out_wm.values()))
        if not out_files or not all((final_model / name).is_file() for name in out_files):
            raise SystemExit("cannot restore missing vision/MTP tensors: final_model shards are incomplete")
        match = re.match(r"^(.*-)(\d+)-of-(\d+)(\.safetensors)$", out_files[0])
        if not match:
            raise SystemExit(f"cannot restore missing vision/MTP tensors: unexpected shard name {out_files[0]!r}")
        prefix, width, old_total, suffix = match.group(1), len(match.group(2)), int(match.group(3)), match.group(4)
        new_total = old_total + 1
        renamed = {}
        for old in out_files:
            match = re.match(r"^(.*-)(\d+)-of-(\d+)(\.safetensors)$", old)
            if not match or int(match.group(3)) != old_total:
                raise SystemExit(f"cannot restore missing vision/MTP tensors: mixed shard naming at {old!r}")
            new = f"{match.group(1)}{int(match.group(2)):0{width}d}-of-{new_total:0{width}d}{match.group(4)}"
            os.rename(final_model / old, final_model / new)
            renamed[old] = new

        tensors = {}
        for src in sorted(set(base_wm[key] for key in missing)):
            shard = load_file(origin_hf_dir / src)
            for key in missing:
                if key in shard:
                    tensors[key] = shard[key]
        if set(tensors) != set(missing):
            absent = sorted(set(missing) - set(tensors))[:5]
            raise SystemExit(f"failed to load restored tensors: missing {absent}")

        restore_name = f"{prefix}{old_total:0{width}d}-of-{new_total:0{width}d}{suffix}"
        save_file(tensors, final_model / restore_name, metadata={"format": "pt"})
        out_index["weight_map"] = {key: renamed.get(name, name) for key, name in out_wm.items()}
        out_index["weight_map"].update({key: restore_name for key in missing})
        out_index["weight_map"] = dict(sorted(out_index["weight_map"].items()))
        print(f"restored {len(missing)} missing vision/MTP tensor(s) into {restore_name}")
    else:
        print("post-export component check OK: no missing vision/MTP tensors")

    out_index.setdefault("metadata", {})["total_size"] = tensor_bytes(final_model, out_index["weight_map"])
    write_json(out_index_path, out_index)


def tensor_bytes(directory: Path, weight_map: dict) -> int:
    from safetensors import safe_open

    dtype_bytes = {
        "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
        "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    }
    total = 0
    for name in sorted(set(weight_map.values())):
        with safe_open(directory / name, framework="pt") as st:
            for key in st.keys():
                item = st.get_slice(key)
                n = 1
                for dim in item.get_shape():
                    n *= dim
                total += n * dtype_bytes[item.get_dtype()]
    return total


# The benchmark's decoding, applied to every delivery instead of chosen per run.
#
# These are the values the served model would already get with no config at all, written
# out explicitly. That is deliberate and it is what makes the comparison legal: the stored
# baseline was measured on the base checkpoint, which ships no generation_config.json, so
# it ran under exactly this. Any other choice would score every delivery under a decoding
# the baseline never ran, and "better than base" would stop meaning training.
#
# Explicit rather than left to the fallback, because a fallback is not a promise -- it
# belongs to the serving stack and can move under us between versions, silently taking the
# baseline's comparability with it.
#
# Untruncated sampling is the price. It costs some per-run variance, which a lower
# temperature with top_p truncation would reduce -- but only after the baseline is
# re-measured under it. Those two moves have to happen together or not at all.
#
# top_k disables truncation with -1 here, not 0. These values are written for the serving
# stack, and sglang's OpenAI endpoint rejects 0 outright -- `400 top_k must be -1 (disable)
# or at least 1, got 0`. It said 0 for a while, which is HuggingFace's way of disabling the
# same thing: the two libraries mean opposite values by it. The effect was that every
# request of every trial 400'd, so a delivery's first validation came back 0.0 having never
# asked the model anything, and nothing in that failure named this file. delivery_audit.py
# generates through HuggingFace instead, and translates this back on the way in.
FIXED_SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "repetition_penalty": 1.0,
}

# Every key the serving path reads as a sampling parameter, which is more than the ones
# set above: a leftover min_p or max_new_tokens would still reach the server and make the
# decoding differ between two deliveries that both went through this script.
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "max_new_tokens",
)

# Keys that read as decoding decisions but are not among the ones the serving path honours.
# `do_sample: false` is the one that actually gets shipped: it is how a config asks for
# greedy decoding under HF, it survives into an export, and on this path it is silently
# discarded -- so a file containing it next to a temperature of 0.8 describes a decoding
# nobody will use. Removed rather than left in place, so what the file says is what runs.
INERT_DECODING_KEYS = (
    "do_sample",
    "num_beams",
    "penalty_alpha",
)


def apply_fixed_generation_config(final_model: Path) -> None:
    """Give the delivery the benchmark's decoding, whatever the export produced.

    Sampling parameters are not a run's variable. On a multi-turn agent task they move
    the score about as much as a training change does, so leaving them to each run puts
    something that is not being measured in front of what is. They are written here,
    identically for every delivery, and a config found in `final_model` has its sampling
    keys replaced rather than merged.

    Everything that is not a decoding knob is kept: an export inherits `eos_token_id` and
    friends from the base checkpoint, and dropping those breaks generation outright.

    This replaces deleting a config the origin checkpoint did not provide. That deletion
    left the delivery with no file at all, which does not mean default decoding -- it means
    whatever the server falls back to, untruncated and unstated. The file is now always
    written, so there is no case where the decoding is decided somewhere else.
    """
    output = final_model / "generation_config.json"
    config = load_json(output) if output.is_file() else {}
    dropped = sorted(
        key for key in SAMPLING_KEYS + INERT_DECODING_KEYS if key in config
    )
    for key in SAMPLING_KEYS + INERT_DECODING_KEYS:
        config.pop(key, None)
    config.update(FIXED_SAMPLING)
    write_json(output, config)
    note = f", replacing {', '.join(dropped)}" if dropped else ""
    print(f"wrote the benchmark's fixed sampling parameters to generation_config.json{note}")


def make_readable_off_container(final_model: Path) -> None:
    """Let the grading service read the delivery it is about to be handed.

    The trainer container runs as root and may produce mode-0600 weights. The grading
    service reads the shared sandbox as the host account, so the checkpoint must be
    readable outside the trainer container to avoid permission errors.

    a+rX, not a+rw: the service only reads, and the checkpoint should not become writable by
    everything on a shared host.
    """
    if not final_model.is_dir():
        return
    changed = 0
    for path in [final_model, *final_model.rglob("*")]:
        try:
            mode = path.stat().st_mode
            wanted = mode | stat.S_IRGRP | stat.S_IROTH
            if path.is_dir():
                wanted |= stat.S_IXGRP | stat.S_IXOTH
            if wanted != mode:
                path.chmod(wanted)
                changed += 1
        except OSError as exc:
            print(f"could not relax permissions on {path}: {exc}")
    print(f"made {changed} path(s) under {final_model} readable outside this container")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin-hf-dir", required=True, type=Path)
    parser.add_argument("--final-model", required=True, type=Path)
    args = parser.parse_args()

    restore_missing_vision_mtp(args.origin_hf_dir, args.final_model)
    apply_fixed_generation_config(args.final_model)
    # Last, so it also covers whatever the two steps above just wrote.
    make_readable_off_container(args.final_model)


if __name__ == "__main__":
    main()
