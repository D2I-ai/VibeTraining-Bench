#!/usr/bin/env python3
"""Every container image a Harbor task tree needs, in a stable order.

Two carriers, because Harbor tasks declare their environment in one of two ways and a
prefetch that reads only the first silently does nothing for half the suites:

  task.toml `docker_image = "..."`   the task runs a published image directly.
                                    terminal-bench is like this.
  Dockerfile `FROM <image>`          the task's environment is built at run time from a
                                    base image, which is the part worth pre-pulling.
                                    swe-bench-verified is like this -- all 500 tasks,
                                    none with a docker_image, so extracting only
                                    task.toml produced an empty list.

Order is by task directory name so a run interrupted part way is at least reproducible;
it is *not* the order Harbor's own `--n-tasks` subset uses, which the export does not
record, so a partial prefetch cannot be assumed to cover a limited run.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

DOCKER_IMAGE = re.compile(r"""^\s*docker_image\s*=\s*["']([^"']+)["']""")
ARG_LINE = re.compile(
    r"^\s*ARG\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:=(?P<value>.+))?\s*$",
    re.IGNORECASE,
)
ARG_REF = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")
# `FROM [--platform=...] image[:tag] [AS stage]`. The stage alias is dropped, and a
# later `FROM <stage>` referring to it is not an image to pull -- see _images_in.
FROM_LINE = re.compile(
    r"^\s*FROM\s+(?:--[a-zA-Z-]+(?:=\S+)?\s+)*(?P<image>\S+)(?:\s+AS\s+(?P<stage>\S+))?\s*$",
    re.IGNORECASE,
)


def _images_in(dockerfile: Path, *, resolve_arg_defaults: bool = False) -> list:
    """Base images a Dockerfile pulls, resolving ARG defaults only for domestic Tau3."""
    stages: set = set()
    build_args: dict[str, str] = {}
    found = []
    try:
        text = dockerfile.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return found
    for line in text.splitlines():
        arg_match = ARG_LINE.match(line) if resolve_arg_defaults else None
        if arg_match and arg_match.group("value") is not None:
            build_args[arg_match.group("name")] = arg_match.group("value").strip().strip("\"'")
            continue
        match = FROM_LINE.match(line)
        if not match:
            continue
        image = match.group("image")
        if resolve_arg_defaults:
            image = ARG_REF.sub(
                lambda item: build_args.get(item.group("name"), item.group(0)), image
            )
        stage = match.group("stage")
        if stage:
            stages.add(stage.lower())
        # scratch is not pullable; ${...} is unresolved without build args; a stage
        # alias is local to this file.
        if image.lower() in stages or image.lower() == "scratch" or "$" in image:
            continue
        found.append(image)
    return found


def collect(root: Path, *, resolve_arg_defaults: bool = False) -> list:
    """Distinct images under one task tree, first-seen order over sorted task dirs."""
    images: list = []
    seen: set = set()

    def remember(image: str) -> None:
        if image not in seen:
            seen.add(image)
            images.append(image)

    for task_toml in sorted(root.rglob("task.toml"), key=lambda p: str(p)):
        task_dir = task_toml.parent
        try:
            lines = task_toml.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            lines = []
        for line in lines:
            match = DOCKER_IMAGE.match(line)
            if match:
                remember(match.group(1))
        for dockerfile in sorted(task_dir.rglob("Dockerfile*"), key=lambda p: str(p)):
            for image in _images_in(dockerfile, resolve_arg_defaults=resolve_arg_defaults):
                remember(image)
    return images


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Where to write the image list.")
    parser.add_argument(
        "--max-per-root",
        type=int,
        default=0,
        help="Keep at most N images per task tree. 0 means all. Useful when a suite's "
        "full image set is larger than the disk budget for a preparation window.",
    )
    parser.add_argument(
        "root",
        nargs="+",
        help="Task trees, as name=path so the report can attribute counts.",
    )
    args = parser.parse_args(argv)

    out_images: list = []
    seen: set = set()
    missing = []
    for entry in args.root:
        name, _, path = entry.partition("=")
        root = Path(path or name)
        if not root.exists():
            missing.append(f"{name} ({root})")
            continue
        # Tau3 task Dockerfiles use an ARG-pinned local task base. Resolve its default
        # in every network mode; the old domestic-only gate made the normal open-source
        # path report zero images even after a valid patch.
        images = collect(
            root,
            resolve_arg_defaults=(name == "sierra-research__tau3-bench"),
        )
        kept = images[: args.max_per_root] if args.max_per_root > 0 else images
        new = [image for image in kept if image not in seen]
        seen.update(new)
        out_images.extend(new)
        note = "" if len(kept) == len(images) else f" (capped from {len(images)})"
        print(f"[images] {name}: {len(kept)} image(s){note}, {len(new)} not already listed")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(out_images) + ("\n" if out_images else ""), encoding="utf-8")
    if missing:
        print("[images] WARN: missing task trees: " + ", ".join(missing), file=sys.stderr)
    print(f"[images] wrote {len(out_images)} distinct image(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
