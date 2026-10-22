from __future__ import annotations

import importlib.util
import json
import re
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]

from eval.runner import backend_pins, tau3_assets  # noqa: E402

VERIFY_PATH = REPO / "scripts/prepare/python/verify_harbor.py"
SPEC = importlib.util.spec_from_file_location("verify_harbor", VERIFY_PATH)
assert SPEC and SPEC.loader
verify_harbor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_harbor)


class DependencyPinTest(unittest.TestCase):
    def test_harbor_has_one_authoritative_git_pin(self) -> None:
        pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        requirements = (REPO / "containers/requirements.txt").read_text(encoding="utf-8")
        self.assertNotRegex(pyproject, r'(?m)^\s*["\']harbor(?:[<>=!~@])')
        self.assertEqual(requirements.count("\nharbor @ git+"), 1)
        self.assertEqual(len(backend_pins.HARBOR_COMMIT), 40)
        self.assertEqual(tau3_assets.HARBOR_COMMIT, backend_pins.HARBOR_COMMIT)
        self.assertIn(backend_pins.HARBOR_COMMIT, backend_pins.HARBOR_PIN.requirement)

    def test_harbor_install_and_source_are_verified(self) -> None:
        installer = (REPO / "scripts/prepare/install_backend_tools.sh").read_text(encoding="utf-8")
        self.assertIn("verify-install --harbor-bin", installer)
        self.assertIn("prepare-source --checkout", installer)
        for relative in (
            "scripts/prepare/prefetch_benchmark_images.sh",
            "scripts/prepare/prepare_swe.sh",
            "scripts/prepare/prepare_terminal.sh",
        ):
            self.assertIn(
                "verify_harbor.py verify-install",
                (REPO / relative).read_text(encoding="utf-8"),
                relative,
            )

    def test_installed_harbor_requires_matching_direct_url(self) -> None:
        class FakeDistribution:
            version = "0.22.0"

            @staticmethod
            def read_text(name: str) -> str | None:
                if name != "direct_url.json":
                    return None
                return json.dumps(
                    {
                        "url": backend_pins.HARBOR_REPO_URL,
                        "vcs_info": {
                            "vcs": "git",
                            "commit_id": backend_pins.HARBOR_COMMIT,
                            "requested_revision": backend_pins.HARBOR_COMMIT,
                        },
                    }
                )

        with mock.patch.object(verify_harbor.metadata, "distribution", return_value=FakeDistribution()):
            verify_harbor.verify_install_local()

    def test_container_stack_references_do_not_drift(self) -> None:
        trainer = (REPO / "containers/trainer.Dockerfile").read_text(encoding="utf-8")
        sglang = (REPO / "containers/sglang.Dockerfile").read_text(encoding="utf-8")
        builder = (REPO / "containers/build_image.sh").read_text(encoding="utf-8")
        sglang_image = re.search(r"^ARG SGLANG_IMAGE=(\S+)$", sglang, re.MULTILINE)
        self.assertIsNotNone(sglang_image)
        image = sglang_image.group(1)
        self.assertIn(image, trainer)
        self.assertIn(image, builder)
        self.assertIn("RUN python3 /opt/vtb/check_stack.py", trainer)
        self.assertIn("RUN python3 /opt/vtb/check_stack.py", sglang)
        stack_specs = [
            line for line in (REPO / "containers/model-stack.txt").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        self.assertTrue(stack_specs)
        self.assertTrue(all(re.match(r"^[A-Za-z0-9_.-]+==[^=*<>~,]+$", spec) for spec in stack_specs))
        self.assertRegex(trainer, re.compile(r"^ARG SLIME_IMAGE=\S+@sha256:[0-9a-f]{64}$", re.MULTILINE))
        self.assertRegex(sglang, re.compile(r"^ARG SGLANG_IMAGE=\S+@sha256:[0-9a-f]{64}$", re.MULTILINE))
        tau_image = (REPO / "containers/tau2-bench-base.Dockerfile").read_text(encoding="utf-8")
        self.assertRegex(tau_image, re.compile(r"^ARG PYTHON_BASE_IMAGE=\S+@sha256:[0-9a-f]{64}$", re.MULTILINE))

    def test_tau2_runtime_versions_match_preflight(self) -> None:
        dockerfile = (REPO / "containers/tau2-bench-base.Dockerfile").read_text(encoding="utf-8")
        preflight = (REPO / "scripts/prepare/python/preflight_tau2_for_tau3.py").read_text(encoding="utf-8")
        for arg, package in (
            ("LITELLM_VERSION", "litellm"),
            ("MCP_VERSION", "mcp"),
            ("FASTMCP_VERSION", "fastmcp"),
        ):
            match = re.search(rf"^ARG {arg}=([^\s]+)$", dockerfile, re.MULTILINE)
            self.assertIsNotNone(match)
            self.assertIn(f'"org.vibetrainbench.{package}.version": "{match.group(1)}"', preflight)

    def test_toolathlon_source_and_image_pins_are_consistent(self) -> None:
        service_doc = (REPO / "eval/service/README.md").read_text(encoding="utf-8")
        for relative in (
            "eval/service/toolathlon_gym/paths.py",
            "eval/service/toolathlon_verified/paths.py",
        ):
            text = (REPO / relative).read_text(encoding="utf-8")
            commit = re.search(r'PINNED_UPSTREAM_COMMIT = "([0-9a-f]{40})"', text)
            self.assertIsNotNone(commit)
            self.assertIn(commit.group(1), service_doc)
        runner = (REPO / "eval/service/toolathlon_verified/runner.py").read_text(encoding="utf-8")
        launcher = (REPO / "eval/service/toolathlon_verified/scripts/start_service.sh").read_text(encoding="utf-8")
        image = "lockon0927/toolathlon-task-image:1016beta@sha256:4d04fe4e0a6fdb4946f51bb05120cb44a0eef980231c11252f93b62897afcb9f"
        self.assertIn(image, runner)
        self.assertIn(image, launcher)
        postgres = "postgres:15@sha256:9b1d34adbce1dd07ee6e94b4a2cf698884b89bd44a6c9c12f5da8f3acbfe4957"
        gym_runner = (REPO / "eval/service/toolathlon_gym/runner.py").read_text(encoding="utf-8")
        gym_launcher = (REPO / "eval/service/toolathlon_gym/scripts/start_service.sh").read_text(encoding="utf-8")
        self.assertIn(postgres, gym_runner)
        self.assertIn(postgres, gym_launcher)

    def test_toolathlon_host_dependencies_are_exact_and_equal(self) -> None:
        gym = (REPO / "eval/service/toolathlon_gym/requirements.txt").read_text(encoding="utf-8")
        verified = (REPO / "eval/service/toolathlon_verified/requirements.txt").read_text(encoding="utf-8")
        gym_specs = {line for line in gym.splitlines() if line and not line.startswith("#")}
        verified_specs = {line for line in verified.splitlines() if line and not line.startswith("#")}
        self.assertEqual(gym_specs, verified_specs)
        for spec in gym_specs:
            self.assertRegex(spec, r"^[A-Za-z0-9_.-]+(?:\[[^]]+\])?==[^=]+$")


if __name__ == "__main__":
    unittest.main()
