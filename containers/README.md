# Container environment

VibeTraining-Bench separates research, model serving, and benchmark task execution. Docker owns all three runtime boundaries; large model/data caches remain under `VTB_ROOT` on the host.

## Host requirements

- Python 3.12 for host tools, selected by `.python-version`; `install_backend_tools.sh` provisions an isolated uv-managed runtime. Use `VTB_PYTHON` for direct commands rather than a system alias.
- NVIDIA driver and the NVIDIA Container Toolkit for GPU runs.
- A rootful Docker daemon reachable at `unix:///var/run/docker.sock` by default.
- Enough space in Docker's `data-root` for trainer/server layers and benchmark task images.
- `fuse-overlayfs` plus `user_allow_other` in `/etc/fuse.conf` for disposable Hugging Face cache overlays. A read-only bind fallback exists, but FUSE is the intended setup.

Verify the daemon before building anything:

```bash
docker info
```

Docker image storage is independent of `VTB_ROOT`. Configure `/etc/docker/daemon.json` before a large prefetch if the default filesystem is too small:

```json
{
  "data-root": "/path/on/a/large/disk/docker"
}
```

Preserve any existing daemon settings when adding `data-root`, restart Docker, and verify the resolved path with `docker info`.

## Images

Trainer/SGLang builds check their inherited Python minor against `.python-version` as well as the pinned model stack. The Tau2 base image pins Python 3.12 by digest. Independent benchmark task images and Toolathlon uv projects retain their declared upstream runtimes; the host interpreter is not mounted or forwarded into them.

| Image | Definition | Responsibility |
|---|---|---|
| `vibetrainbench/trainer:latest` | `trainer.Dockerfile` | Research-agent sandbox, Slime, Megatron-LM, training tools, and provider CLIs. |
| `vibetrainbench/sglang:latest` | `sglang.Dockerfile` | OpenAI-compatible model server used to evaluate submitted checkpoints. |
| Benchmark task images | Harbor task definitions | Isolated task environments used by Harbor-backed benchmarks. |
| `vibetrainbench/tau2-bench:<pin>` | `tau2-bench-base.Dockerfile` through `prepare_tau3.sh image` | Pinned Tau2 runtime required by Tau3 task environments. |

Build the shared images:

```bash
bash containers/build_image.sh all       # or: trainer / sglang
```

Tau3's dependency image is benchmark-specific:

```bash
bash scripts/prepare/prepare_tau3.sh image
```

Override image selection only for controlled local runs:

| Variable | Meaning |
|---|---|
| `VTB_TRAINER_IMAGE` | Full trainer image reference. |
| `VTB_SGLANG_IMAGE` | Full serving image reference. |
| `VTB_IMAGE_NAMESPACE` | Namespace used by `build_image.sh`. |
| `VTB_IMAGE_TAG` | Tag used by `build_image.sh`. |
| `TAU2_IMAGE` | Tau3's pinned Tau2 dependency image. |

Paper experiments use the repository defaults.

## Version contract

Trainer, SGLang, Tau2 Python, Toolathlon Verified, and Toolathlon Postgres upstream images are pinned by tag plus immutable manifest digest. `model-stack.txt` additionally pins exact versions that cross the trainer/server boundary, and `check_stack.py` validates them during both image builds:

- `transformers` controls checkpoint configuration;
- `tokenizers` controls tokenization and stopping behavior;
- `torch` is the compiled runtime floor shared by training and serving.

The trainer is authoritative because it writes checkpoints; the serving image must read the same format. Verify an existing image with:

```bash
docker run --rm vibetrainbench/trainer:latest python3 /opt/vtb/check_stack.py
```

A stack mismatch must be fixed by rebuilding the image, not by installing packages interactively into a running container.

The version audit also enforces one Harbor git pin, exact Toolathlon service dependencies, pinned Toolathlon service images, and matching external Toolathlon source commits. Several vendored Toolathlon MCP projects use upstream dependency ranges without local lockfiles, and Tau2 is fetched from an unmerged PR ref with its exact commit verified. A fully offline rebuild requires lockfiles and cached source exports for these dependencies.

## Hugging Face cache isolation

The shared cache defaults to:

```text
${VTB_ROOT}/resources/data/hf_cache
```

A job sees a disposable copy-on-write view created with `fuse-overlayfs`. Job-local writes land in scratch state while the shared lower layer remains unchanged. If FUSE is unavailable, the runtime logs the fallback and bind-mounts the cache read-only. Set `VTB_HF_PLAIN_BIND=1` only when intentionally selecting that fallback.

The research container runs offline by default. Populate model and dataset snapshots before launch; see [recipe resources](../training/recipes/README.md) and [environment paths](../scripts/prepare/README.md).

## File ownership

Upstream training images expect to run as root because Slime and Megatron-LM live under `/root` and compile extensions in place. `src/run_task.sh` restores ownership of named bind-mounted output paths after the job exits. Do not add `--user` to the trainer container: it makes the packaged training stack unreadable.

## Benchmark task images

SWE and Terminal use Harbor task images. Prefetch is resumable and writes metadata under `VTB_ROOT` while image layers remain in Docker's data root:

```bash
bash scripts/prepare/prepare_swe.sh tasks
bash scripts/prepare/prefetch_benchmark_images.sh \
  --benchmarks terminal-bench__terminal-bench-2-1
```

Useful controls:

| Option/environment | Purpose |
|---|---|
| `--dry-run` | Resolve task/image references without pulling. |
| `--retry-failed` | Retry images listed by the preceding prefetch. |
| `--max-per-benchmark N` | Bound a diagnostic prefetch; not a formal complete cache. |
| `VTB_PULL_RETRIES` | Pull attempts per image. |
| `VTB_PULL_BACKOFF` | Initial retry delay. |
| `VTB_DOCKER_HOST` | Explicit Docker socket override. |

Anonymous registry limits are shared by egress IP. A completed formal preparation must have no unresolved failed image pulls.

## Host-side tools

`containers/requirements.txt` pins Harbor and related host-side execution packages. Install them under `VTB_ROOT` with:

```bash
bash scripts/prepare/install_backend_tools.sh
```

They run on the host because Harbor launches task containers itself. See [scripts/prepare/README.md](../scripts/prepare/README.md) for paths and [scripts/README.md](../scripts/README.md) for benchmark workflows.
