#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

def read_benchmark_name(benchmark_id: str) -> str:
    """Resolve the human-readable benchmark name from the benchmark_id."""
    bench_file = Path("benchmarks") / benchmark_id / "benchmark.txt"
    if not bench_file.is_file():
        raise FileNotFoundError(f"Benchmark file not found for id '{benchmark_id}': {bench_file}")
    return bench_file.read_text(encoding="utf-8").strip()

def render_conditionals(template: str, conditions: dict) -> str:
    """Render `{#cond}...{/cond}` blocks, keeping a block only when its condition holds.

    Inline and block forms both work, so a template can keep its sentences natural.
    Same-name nesting is not supported (the backreference would pair the wrong
    markers); different conditions nest fine. Unknown or unbalanced markers fail
    loudly -- a half-rendered prompt is worse than no prompt.
    """
    pattern = re.compile(r"\{#(\w+)\}(.*?)\{/\1\}", re.S)

    def repl(match: re.Match) -> str:
        name, body = match.group(1), match.group(2)
        if name not in conditions:
            raise SystemExit(f"unknown conditional {name!r} in prompt template")
        return body if conditions[name] else ""

    previous = None
    while previous != template:
        previous = template
        template = pattern.sub(repl, template)
    leftover = re.search(r"\{[#/]\w+\}", template)
    if leftover:
        raise SystemExit(f"unbalanced conditional {leftover.group(0)!r} in prompt template")
    return template


def detect_gpu_model() -> str:
    """Best-effort accelerator model from nvidia-smi, e.g. 'Nvidia A100'.

    Runs on the host (which owns the GPUs), so it needs nvidia-smi but not torch.
    Falls back to a generic label when the model cannot be determined.
    """
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
        name = out[0].strip() if out else ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        name = ""
    if not name:
        return "Nvidia GPU"
    for family in ("H200", "H100", "H800", "A100", "A800"):
        if family in name:
            return f"Nvidia {family}"
    # Unrecognised model: keep the reported name, just normalise the vendor casing.
    return name.replace("NVIDIA ", "Nvidia ").strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--model-to-train', type=str, required=True)
    parser.add_argument('--benchmark-id', type=str, required=True)
    parser.add_argument('--num-hours', type=str, required=True)
    parser.add_argument('--num-gpus', type=int, default=1)

    args = parser.parse_args()

    benchmark_name = read_benchmark_name(args.benchmark_id)

    base_prompt = os.environ.get('VTB_PROMPT', 'workflow')

    # Templates live directly under agent_task/; workspace_docs/ holds documents that are
    # copied into the workspace, not rendered. Refuse a separator rather than silently
    # rendering one of those (or a path outside the template directory).
    if '/' in base_prompt or base_prompt.startswith('.'):
        raise SystemExit(
            f"invalid VTB_PROMPT {base_prompt!r}: name a template in eval/agent_task/ "
            "(research, workflow, harness_connectivity, train_test, hf_train_test, "
            "mock_test, tau2_smoke, terminal_smoke, aime_smoke, toolathlon_smoke)"
        )

    template_path = f'eval/agent_task/{base_prompt}.md'

    with open(template_path, 'r') as f:
        template = f.read()

    # Whether this run ships an editable harness is decided at workspace-preparation
    # time (benchmark declaration, minus a VTB_EDITABLE_HARNESS=off opt-out), and
    # run_task.sh passes the result as VTB_HAS_HARNESS -- the workspace the agent sees
    # and the prompt it reads then cannot disagree. Rendering by hand falls back to the
    # benchmark's declaration.
    has_harness_env = os.environ.get('VTB_HAS_HARNESS', '').strip().lower()
    if has_harness_env:
        has_harness = has_harness_env not in ('0', 'off', 'false', 'no')
    else:
        try:
            bench_meta = json.loads(
                Path('benchmarks', args.benchmark_id, 'benchmark.json').read_text(encoding='utf-8')
            )
        except (OSError, ValueError):
            bench_meta = {}
        has_harness = bool((bench_meta.get('editable_harness') or {}).get('enabled'))
    template = render_conditionals(template, {'harness': has_harness, 'no_harness': not has_harness})

    datetime = subprocess.run(['date', '-u'], capture_output=True, text=True).stdout.strip()

    if args.num_gpus == 1:
        gpu_info = f"- The machine is equipped with an {detect_gpu_model()} GPU."
    else:
        gpu_info = f"- The machine is equipped with {args.num_gpus} {detect_gpu_model()} GPUs."

    result = template.replace('{model}', args.model_to_train)
    result = result.replace('{benchmark}', benchmark_name)
    result = result.replace('{num_hours}', args.num_hours)
    result = result.replace('{gpu_info}', gpu_info)

    # Data-pool constraint: a benchmark may declare a curated training pool by
    # shipping benchmarks/<id>/task_context/data_pool.md, which documents where the
    # pool lives (a declared HF cache resource the recipe pins, not a file inside
    # task_context) and how the agent may use it. data_pool.md travels into the
    # workspace with the wholesale task_context copy, so keying off it -- rather than
    # off a physical train_data/ directory -- keeps the constraint intact once the
    # pool moves to Hugging Face. When present, the agent's data selection is
    # constrained to that pool; when absent, the original freedom on data sources
    # applies. Either way the template's {data_pool} slot resolves to a well-formed
    # bullet.
    pool_doc = Path("benchmarks") / args.benchmark_id / "task_context" / "data_pool.md"
    if pool_doc.is_file():
        data_pool = (
            "- Your training data must be drawn from the provided data pool only: "
            "read `data_pool.md` in the current directory for the pool location and "
            "rules, and select a subset of it for training. You may filter, sample "
            "and reformat pool contents freely, but adding training examples from "
            "any other source is not allowed."
        )
    else:
        data_pool = (
            "- You have complete freedom in your choice of data sources."
        )
    result = result.replace('{data_pool}', data_pool)

    # How the grading service behaves while a submission is in flight. This goes into
    # every prompt: every benchmark is scored by the same shared service, so every agent
    # can hit the queue.
    result = result.replace(
        "{setup_other}",
        "- A final note on the grading service: evaluations are serialised across all runs, so a "
        "submission may sit in `queued` before it starts. Poll `/result` and keep working "
        "in the meantime rather than waiting idle.\n",
    )

    result = result.replace('{datetime}', datetime)

    result += """
You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
"""
    print(result)

if __name__ == '__main__':
    main()
