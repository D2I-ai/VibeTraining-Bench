# AIME SFT materialization

This directory contains the deterministic conversion pipeline invoked by:

```bash
bash scripts/prepare/prepare_aime.sh training-data
```

The shell entrypoint downloads only recipe-declared files at immutable revisions, then runs `prepare.py` inside `vibetrainbench/trainer:latest`, where `pyarrow` is available. No GPU is requested and conversion has networking disabled. If the cache is already complete, `training-data --offline` checks the sources and converts them without any downloads.

The converters process sources in this fixed order:

| Source | Selection | Output rows |
|---|---|---:|
| `open-thoughts/OpenThoughts3-1.2M` | `domain == "math"` | 850,000 |
| `nvidia/Nemotron-SFT-Math-v4` | `subset == "cot"` | 285,516 |
| `open-r1/OpenR1-Math-220k` | default configuration | 93,733 |

The three converted streams are concatenated into:

```text
${VTB_ROOT}/resources/data/aime_sft/train.jsonl
```

`source_manifest.json` records source resource IDs, revisions, file patterns, row counts, preparation/converter code digests, output byte size, and the final SHA-256. Existing output is reused only after a streamed checksum, row-count, and messages-schema check. Incomplete, corrupted, or provenance-mismatched output is refused; use a fresh `VTB_ROOT` rather than overwriting data used by another run. Raw snapshot checks require 120 OpenThoughts3 shards, 12 Nemotron shards, and 10 OpenR1 default shards.

The prepared payload is copied into `$HOME/task/train_resources/data/aime/data/`; the AIME recipe's `TRAIN_DATA` points directly to its `train.jsonl`. Formal workspace preparation rechecks the payload; these full scans cost CPU/I/O time but keep memory bounded.
