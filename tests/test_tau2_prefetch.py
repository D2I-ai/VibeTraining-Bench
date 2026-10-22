from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.runner import collect_task_images, task_backends, tau3_assets  # noqa: E402

PATCH_PATH = REPO / "scripts/prepare/python/patch_export_tau3.py"
SPEC = importlib.util.spec_from_file_location("patch_tau3_export", PATCH_PATH)
assert SPEC and SPEC.loader
patcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patcher)


TASK_TOML = '''schema_version = "1.1"
[verifier]
env = { TAU2_USER_API_KEY = "${TAU2_USER_API_KEY:-}", TAU2_USER_BASE_URL = "${TAU2_USER_BASE_URL:-}", TAU2_NL_ASSERTIONS_BASE_URL = "${TAU2_NL_ASSERTIONS_BASE_URL:-}", TAU2_NL_ASSERTIONS_MODEL = "${TAU2_NL_ASSERTIONS_MODEL:-}", TAU2_NL_ASSERTIONS_LLM_ARGS_JSON = "${TAU2_NL_ASSERTIONS_LLM_ARGS_JSON:-}" }
[agent]
timeout_sec = 3600
[environment]
env = { TAU2_USER_API_KEY = "${TAU2_USER_API_KEY:-}", TAU2_USER_BASE_URL = "${TAU2_USER_BASE_URL:-}", TAU2_USER_MODEL = "${TAU2_USER_MODEL:-gpt-5.2}", TAU2_USER_REASONING_EFFORT = "${TAU2_USER_REASONING_EFFORT:-low}", TAU2_USER_TEMPERATURE = "${TAU2_USER_TEMPERATURE:-}", TAU2_USER_LLM_ARGS_JSON = "${TAU2_USER_LLM_ARGS_JSON:-}" }
'''

COMPOSE = '''services:
  tau3-runtime:
    environment:
      - TAU2_USER_API_KEY=${TAU2_USER_API_KEY:-}
      - TAU2_USER_BASE_URL=${TAU2_USER_BASE_URL:-}
      - TAU2_USER_MODEL=${TAU2_USER_MODEL:-gpt-5.2}
      - TAU2_USER_REASONING_EFFORT=${TAU2_USER_REASONING_EFFORT:-low}
'''


class Tau2ExportPatchTest(unittest.TestCase):
    def _tree(self, root: Path) -> tuple[Path, Path]:
        export = root / "export" / "tau3-bench"
        template = root / "harbor-template"
        for relative, text in {
            "environment/docker-compose.yaml": COMPOSE,
            "environment/runtime-server/server.py": "print('runtime')\n",
            "tests/evaluate.py": "print('verify')\n",
        }.items():
            path = template / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        for name in ("tau3-bench__tau3-a", "tau3-bench__tau3-b"):
            task = export / name
            task.mkdir(parents=True)
            (task / "task.toml").write_text(TASK_TOML)
            # Harbor owns static-asset patching; VTB replaces only the Dockerfiles.
            (task / "environment").mkdir()
            (task / "environment/Dockerfile").write_text("FROM python:3.12-slim\nRUN apt-get update\n")
            (task / "environment/runtime-server").mkdir()
            (task / "environment/runtime-server/Dockerfile").write_text("FROM python:3.12-slim\n")
            for relative in patcher.STATIC_ASSETS:
                destination = task / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("stale\n")
        return export, template

    @staticmethod
    def _harbor_patch(template: Path):
        def patch(task_dirs: list[Path]) -> None:
            for task in task_dirs:
                for relative in patcher.STATIC_ASSETS:
                    shutil.copy2(template / relative, task / relative)

        return patch

    def test_patch_creates_offline_tree_and_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export, template = self._tree(Path(tmp))
            patcher._patch_tree(
                export, template, 2, tau3_assets.HARBOR_COMMIT, self._harbor_patch(template)
            )
            manifest = json.loads((export / "manifest.json").read_text())
            self.assertEqual([x["task_id"] for x in manifest["tasks"]], [
                "tau3-bench__tau3-a", "tau3-bench__tau3-b"
            ])
            patch = json.loads((export / tau3_assets.PATCH_MANIFEST).read_text())
            self.assertEqual(patch["task_count"], 2)
            self.assertEqual(patch["task_base_image"], tau3_assets.TAU2_BASE_IMAGE)
            for task_id in ("tau3-bench__tau3-a", "tau3-bench__tau3-b"):
                task = export / task_id
                compose = (task / "environment/docker-compose.yaml").read_text()
                task_toml = (task / "task.toml").read_text()
                self.assertEqual(compose, COMPOSE)
                self.assertIn("TAU2_NL_ASSERTIONS_MODEL", task_toml)
                for relative in ("environment/Dockerfile", "environment/runtime-server/Dockerfile"):
                    text = (task / relative).read_text()
                    self.assertIn(tau3_assets.TAU2_BASE_IMAGE, text)
                    self.assertNotIn("apt-get", text)
                    self.assertNotIn("pip install", text)
                    self.assertNotIn("git clone", text)

    def test_validate_export_rejects_changed_task_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export, template = self._tree(Path(tmp))
            patcher._patch_tree(
                export, template, 2, tau3_assets.HARBOR_COMMIT, self._harbor_patch(template)
            )
            with mock.patch.object(tau3_assets, "TAU3_TASK_COUNT", 2):
                tau3_assets.validate_export(export)
                (export / "tau3-bench__tau3-a/task.toml").write_text("changed\n")
                with self.assertRaisesRegex(tau3_assets.Tau3AssetError, "digest mismatch"):
                    tau3_assets.validate_export(export)


class HarborLocalSourceTest(unittest.TestCase):
    def test_tau2_arg_base_is_discovered_without_network_mode_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dockerfile = Path(tmp) / "Dockerfile"
            dockerfile.write_text(patcher.MAIN_DOCKERFILE)
            self.assertEqual(
                collect_task_images._images_in(dockerfile, resolve_arg_defaults=True),
                [tau3_assets.TAU2_BASE_IMAGE],
            )

    def test_local_source_wins_over_registry_dataset(self) -> None:
        spec = SimpleNamespace(
            name=tau3_assets.TAU3_BENCHMARK,
            dataset=tau3_assets.TAU3_DATASET,
            agent="mini-swe-agent",
            agentic=True,
            environment="docker",
            agent_kwargs={},
            extra_args=[],
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            task_backends, "_binary", return_value="harbor"
        ):
            source = Path(tmp) / "tasks"
            command = task_backends.build_harbor(
                spec,
                task_source=source,
                task_ids=["tau3-bench__tau3-a"],
                output_dir=Path(tmp) / "out",
                model="openai/model",
                n_concurrent=1,
                limit=1,
            )
        self.assertIn("--path", command)
        self.assertIn(str(source), command)
        self.assertNotIn("--dataset", command)
        self.assertNotIn("--n-tasks", command)

    def test_tau_smoke_prompt_is_not_shared_with_other_benchmarks(self) -> None:
        generic = (REPO / "eval/agent_task/mock_test.md").read_text()
        tau = (REPO / "eval/agent_task/tau2_smoke.md").read_text()
        launcher = (REPO / "scripts/run_smoke/run_smoke_tau3.sh").read_text()

        self.assertIn("VTB_PROMPT=tau2_smoke", launcher)
        self.assertIn('revision="68c46c4b3498877f3ef123c856ecfde50c39f404"', generic)
        self.assertIn("without `local_dir`", generic)
        self.assertNotIn("offline huggingface_hub does not populate", generic)
        self.assertIn("offline huggingface_hub does not populate", tau)

    def test_one_hub_root_drives_tau_preparation_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "exports"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO)
            env["VTB_HUB_TASK_SPECS"] = str(hub)
            env.pop("VTB_HARBOR_DATASET_DIR", None)
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "from eval.runner import runtime_paths as p; "
                    "print(p.HUB_TASK_SPECS_DIR); print(p.TAU3_TASK_SPECS_DIR)",
                ],
                cwd=REPO,
                env=env,
                text=True,
            ).splitlines()
        self.assertEqual(output, [str(hub), str(hub / "sierra-research__tau3-bench/tau3-bench")])

    def test_tau_prefetch_publishes_only_after_patch(self) -> None:
        script = (REPO / "scripts/prepare/prefetch_benchmark_images.sh").read_text()
        self.assertIn('mktemp -d "${out_dir}.download-XXXXXX"', script)
        self.assertIn('tau_publish_targets["$stage"]="$out_dir"', script)
        self.assertLess(
            script.index("scripts/prepare/python/patch_export_tau3.py"),
            script.index('mv "$path" "$canonical"'),
        )

    def test_tau_smoke_honors_runtime_storage_overrides(self) -> None:
        launcher = (REPO / "scripts/run_smoke/run_smoke_tau3.sh").read_text()
        self.assertIn('RESULTS_DIR="${VTB_RESULTS_DIR:-$VTB_ROOT/results}"', launcher)
        self.assertIn('STATE="${VTB_GRADING_STATE:-$VTB_ROOT/', launcher)
        self.assertIn('grep -Fq -- "--path $TAU3_TASK_SPECS"', launcher)


if __name__ == "__main__":
    unittest.main()
