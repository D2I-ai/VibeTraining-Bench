from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval.runner import model_server

REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "src/commit_utils/configure_python.sh"


class PythonEnvironmentTest(unittest.TestCase):
    def test_version_declarations_match_images_and_bootstrap(self):
        self.assertEqual((REPO / ".python-version").read_text().strip(), "3.12")
        self.assertIn('requires-python = ">=3.12,<3.13"', (REPO / "pyproject.toml").read_text())
        bootstrap = (REPO / "scripts/prepare/install_backend_tools.sh").read_text()
        self.assertIn("< .python-version", bootstrap)
        self.assertIn('--python "$PYTHON_VERSION" --managed-python', bootstrap)
        for name in ("trainer", "sglang"):
            self.assertIn("COPY .python-version", (REPO / f"containers/{name}.Dockerfile").read_text())
        self.assertIn("python:3.12-slim@sha256:", (REPO / "containers/tau2-bench-base.Dockerfile").read_text())

    def test_selected_interpreter_ignores_shell_aliases_and_python_environment(self):
        env = {"PATH": os.environ["PATH"], "VTB_PYTHON": sys.executable,
               "PYTHONPATH": "/nonexistent/injected", "PYTHONHOME": "/nonexistent/injected"}
        script = '''set -euo pipefail
python3() { exit 91; }
python() { exit 92; }
source "$1"
"$VTB_PYTHON" -c 'import os,sys; assert "PYTHONHOME" not in os.environ; assert "PYTHONPATH" not in os.environ; assert sys.version_info[:2]==(3,12); print("ISOLATED_PYTHON_OK")'
'''
        result = subprocess.run(["bash", "-c", script, "test", str(HELPER)], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ISOLATED_PYTHON_OK", result.stdout)

    def test_old_interpreter_fails_before_repository_code_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            interpreter = Path(tmp) / "old-python"
            interpreter.write_text('#!/bin/sh\nprintf "FATAL: Python 3.12 required; selected interpreter is 3.8\\n" >&2\nexit 1\n')
            interpreter.chmod(0o755)
            result = subprocess.run(["bash", "-c", 'source "$1" || exit 1; echo UNSAFE_CONTINUE', "test", str(HELPER)],
                                    env={"PATH": os.environ["PATH"], "VTB_PYTHON": str(interpreter)}, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("UNSAFE_CONTINUE", result.stdout)

    def test_host_serving_uses_current_interpreter_but_docker_uses_image_python(self):
        server = model_server.ModelServer("/tmp/model", served_name="fixture")
        server.runtime = "host"
        with mock.patch.object(model_server, "_host_has_sglang", return_value=True):
            self.assertEqual(server._serve_command(9000)[0], sys.executable)
        self.assertEqual(server._payload(9000)[0], "python3")
        with mock.patch.object(model_server.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
            self.assertTrue(model_server._host_has_sglang())
            self.assertEqual(run.call_args.args[0][0], sys.executable)

    def test_service_dependencies_are_exact_and_no_user_site_install_is_used(self):
        for name in ("toolathlon_gym", "toolathlon_verified"):
            text = (REPO / f"eval/service/{name}/scripts/start_service.sh").read_text()
            self.assertNotIn("pip install --user", text)
            self.assertIn("verify_python_environment.py --requirements", text)

    def test_host_launchers_freeze_interpreter_across_detached_phases(self):
        for family in ("run_exp", "run_smoke"):
            for path in (REPO / f"scripts/{family}").glob("*.sh"):
                text = path.read_text()
                self.assertNotIn("python3 -m eval.grading", text, path.name)
                self.assertIn("VTB_PYTHON='$VTB_PYTHON'", text, path.name)
                self.assertIn("'$VTB_PYTHON'", text, path.name)


if __name__ == "__main__":
    unittest.main()
