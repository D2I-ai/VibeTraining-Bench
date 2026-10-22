from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "commit_utils"))

import prepare_agent_workspace as paw  # noqa: E402

VERSION = "2.1.0"
WHEEL_NAME = f"mini_swe_agent-{VERSION}-py3-none-any.whl"


def make_wheel(store: Path) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    wheel = store / WHEEL_NAME
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("minisweagent/__init__.py", "# test fixture\n")
    wheel.write_bytes(buf.getvalue())
    return wheel


class EditableHarnessSeedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runtime = self.root / "runtime"
        self.checkout = self.root / "harbor-checkout"
        self._old_root = os.environ.get("VTB_ROOT")
        self._old_harbor = os.environ.get("VTB_HARBOR_REPO")
        os.environ["VTB_ROOT"] = str(self.runtime)
        os.environ["VTB_HARBOR_REPO"] = str(self.checkout)

    def tearDown(self) -> None:
        if self._old_root is None:
            os.environ.pop("VTB_ROOT", None)
        else:
            os.environ["VTB_ROOT"] = self._old_root
        if self._old_harbor is None:
            os.environ.pop("VTB_HARBOR_REPO", None)
        else:
            os.environ["VTB_HARBOR_REPO"] = self._old_harbor
        self.tmp.cleanup()

    def _adapter(self, root: Path, content: str) -> Path:
        path = root / "installed" / "mini_swe_agent.py"
        path.parent.mkdir(parents=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_backend_installed_adapter_wins_over_checkout(self) -> None:
        installed_root = (
            self.runtime
            / "tools/data/uv/tools/harbor/lib/python3.12/site-packages/harbor/agents"
        )
        installed = self._adapter(installed_root, "# baseline adapter\n")
        self._adapter(self.checkout / "src/harbor/agents", "# newer checkout adapter\n")

        source = paw.installed_harness_source(REPO, "installed-mini-swe-agent")

        self.assertEqual(source, installed)

    def test_checkout_only_mini_swe_adapter_is_refused(self) -> None:
        self._adapter(self.checkout / "src/harbor/agents", "# checkout adapter\n")

        with self.assertRaises(SystemExit):
            paw.installed_harness_source(REPO, "installed-mini-swe-agent")

    def test_editable_copy_is_byte_identical_to_baseline_adapter(self) -> None:
        installed_root = (
            self.runtime
            / "tools/data/uv/tools/harbor/lib/python3.12/site-packages/harbor/agents"
        )
        installed = self._adapter(installed_root, "class MiniSweAgent:\n    pass\n")
        task_dir = self.root / "task"
        paw.copy_editable_harness(
            REPO,
            task_dir,
            {
                "template_source": "installed-mini-swe-agent",
                "import_path": "harness.mini_swe_agent:MiniSweAgent",
                "allowed_files": ["__init__.py", "mini_swe_agent.py"],
            },
        )

        copied = task_dir / "harness/mini_swe_agent.py"
        self.assertEqual(copied.read_bytes(), installed.read_bytes())
        self.assertEqual(paw._sha256_of(copied), paw._sha256_of(installed))


class HarnessPackageReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.task_dir = self.root / "task"
        self.task_dir.mkdir()
        self.bench_dir = self.root / "bench"
        self.bench_dir.mkdir()
        (self.bench_dir / "benchmark.json").write_text(
            json.dumps({"agent_kwargs": {"version": VERSION}}), encoding="utf-8"
        )
        self._old_packages = os.environ.get("VTB_PACKAGES")
        os.environ["VTB_PACKAGES"] = str(self.root / "packages")  # package store override

    def tearDown(self) -> None:
        if self._old_packages is None:
            os.environ.pop("VTB_PACKAGES", None)
        else:
            os.environ["VTB_PACKAGES"] = self._old_packages
        self.tmp.cleanup()

    def test_mini_swe_agent_template_ships_source_and_wheel(self) -> None:
        wheel = make_wheel(self.root / "packages")
        out = paw.provision_harness_package_source(
            REPO, self.task_dir, self.bench_dir, {"template_source": "installed-mini-swe-agent"}
        )
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["name"], "mini-swe-agent")
        self.assertEqual(out["version"], VERSION)
        self.assertEqual(out["ref"], "harness_reference/")
        self.assertEqual(out["sha256"], paw._sha256_of(wheel))
        ref = self.task_dir / "harness_reference"
        # Both views of the pinned bytes: the wheel and the unpacked, readable tree.
        self.assertTrue((ref / WHEEL_NAME).is_file())
        self.assertTrue((ref / "minisweagent" / "__init__.py").is_file())

    def test_terminus_template_gets_no_reference(self) -> None:
        out = paw.provision_harness_package_source(
            REPO, self.task_dir, self.bench_dir, {"template_source": "installed-terminus-2"}
        )
        self.assertIsNone(out)
        self.assertFalse((self.task_dir / "harness_reference").exists())

    def test_missing_version_pin_is_refused(self) -> None:
        (self.bench_dir / "benchmark.json").write_text(
            json.dumps({"agent_kwargs": {}}), encoding="utf-8"
        )
        with self.assertRaises(SystemExit):
            paw.provision_harness_package_source(
                REPO, self.task_dir, self.bench_dir, {"template_source": "installed-mini-swe-agent"}
            )


class ValidationContextSmokeOptOutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.task = self.root / "task"
        self.task.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _benchmark(self, name: str) -> Path:
        benchmark = self.root / "benchmarks" / name
        benchmark.mkdir(parents=True)
        (benchmark / "benchmark.json").write_text(
            json.dumps({"validation_context": "validation/missing"}), encoding="utf-8"
        )
        return benchmark

    def test_terminal_smoke_may_skip_only_agent_diagnostics(self) -> None:
        benchmark = self._benchmark("terminal-bench__terminal-bench-2-1")
        with mock.patch.dict(
            os.environ,
            {
                "VTB_SKIP_VALIDATION_CONTEXT": "terminal-smoke",
                "VTB_PROMPT": "terminal_smoke",
                "VTB_TRAIN_RECIPE": "",
            },
        ):
            self.assertIsNone(paw.copy_validation_context(self.root, self.task, benchmark))

    def test_context_skip_is_rejected_for_other_benchmarks(self) -> None:
        benchmark = self._benchmark("swe-bench__swe-bench-verified")
        with mock.patch.dict(
            os.environ,
            {
                "VTB_SKIP_VALIDATION_CONTEXT": "terminal-smoke",
                "VTB_PROMPT": "terminal_smoke",
                "VTB_TRAIN_RECIPE": "",
            },
        ):
            with self.assertRaises(SystemExit):
                paw.copy_validation_context(self.root, self.task, benchmark)

    def test_context_skip_is_rejected_when_training(self) -> None:
        benchmark = self._benchmark("terminal-bench__terminal-bench-2-1")
        with mock.patch.dict(
            os.environ,
            {
                "VTB_SKIP_VALIDATION_CONTEXT": "terminal-smoke",
                "VTB_PROMPT": "terminal_smoke",
                "VTB_TRAIN_RECIPE": "some_recipe",
            },
        ):
            with self.assertRaises(SystemExit):
                paw.copy_validation_context(self.root, self.task, benchmark)

    def test_terminal_smoke_keeps_real_route_and_two_one_sample_limits(self) -> None:
        launcher = (REPO / "scripts/run_smoke/run_smoke_terminal.sh").read_text()
        prompt = (REPO / "eval/agent_task/terminal_smoke.md").read_text()
        self.assertIn("VTB_VAL_SOURCE='$VALIDATION_SOURCE'", launcher)
        self.assertIn("VTB_FINAL_LIMIT=1", launcher)
        self.assertIn('Include `"limit": 1`', prompt)
        self.assertNotIn("VTB_VAL_SOURCE=split:1", launcher)
        self.assertIn("startup.log", launcher)


if __name__ == "__main__":
    unittest.main()
