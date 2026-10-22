from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
# The trainer also installs an unrelated package named agents; load our CLI by path.
_spec = importlib.util.spec_from_file_location("vtb_agent_config", REPO / "agents/config.py")
assert _spec and _spec.loader
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)


class AdapterConfigurationTest(unittest.TestCase):
    def _adapter(self, root, name="custom"):
        path = root / name
        path.mkdir(parents=True)
        (path / "solve.sh").write_text("exit 0\n")
        (path / "adapter_config.json").write_text(json.dumps({
            "required_env": ["CUSTOM_API_KEY"],
            "forward_env": ["CUSTOM_API_KEY", "CUSTOM_BASE_URL"],
        }))
        return path

    def test_public_adapter_and_scoped_credentials(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test", "OPENAI_API_KEY": "unrelated"}, clear=True):
            path, cfg = config.validate("claude", "provider/model-A")
            self.assertEqual(path, REPO / "agents/claude")
            self.assertEqual(config.forwarded_environment(cfg), {"ANTHROPIC_API_KEY": "test"})

    def test_local_adapter_requires_explicit_root_and_has_no_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = self._adapter(root)
            with mock.patch.dict(os.environ, {"VTB_AGENT_ADAPTERS_DIR": tmp, "CUSTOM_API_KEY": "test"}, clear=True):
                self.assertEqual(config.validate("custom", "model")[0], adapter)
                with self.assertRaisesRegex(ValueError, "missing"):
                    config.resolve_adapter("claude")
            with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
                config.resolve_adapter("custom")

    def test_missing_researcher_and_key_fail_without_leaking_values(self):
        with mock.patch.dict(os.environ, {"OTHER_SECRET": "never-print-this"}, clear=True):
            for researcher in ("", "model"):
                with self.assertRaises(ValueError) as caught:
                    config.validate("claude", researcher)
                self.assertNotIn("never-print-this", str(caught.exception))

    def test_forbidden_variables_and_unsafe_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = self._adapter(root)
            with mock.patch.dict(os.environ, {"VTB_AGENT_ADAPTERS_DIR": tmp}, clear=True):
                for key in ("VTB_ADMIN_TOKEN_FILE", "OFFICIAL_TOKEN", "GRADING_TOKEN", "PATH", "VTB_PYTHON", "PYTHONHOME", "bad-name"):
                    (adapter / "adapter_config.json").write_text(json.dumps({"required_env": [], "forward_env": [key]}))
                    with self.assertRaises(ValueError):
                        config.read_config("custom")
                for name in ("../custom", "/custom", "custom;true", ".."):
                    with self.assertRaises(ValueError):
                        config.resolve_adapter(name)

    def test_model_slug_matches_safe_run_path(self):
        self.assertEqual(config.model_slug("provider/model:v2[small]"), "provider_model_v2_small_")
        for value in ("", "model\nname", "x'y"):
            with self.assertRaises(ValueError):
                config.model_slug(value)

    def test_explicit_empty_optional_settings_are_preserved(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test", "ANTHROPIC_AUTH_TOKEN": ""}, clear=True):
            _, cfg = config.validate("claude", "model")
            self.assertIn("ANTHROPIC_AUTH_TOKEN", config.forwarded_environment(cfg))

    def test_recorded_result_path_can_be_read_without_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            env = {"PATH": os.environ["PATH"], "ANTHROPIC_API_KEY": "test"}
            args = [sys.executable, str(REPO / "agents/config.py"), "record-result", "claude",
                    "--researcher", "provider/model", "--state", str(state), "--default-result-dir", str(state / "results")]
            subprocess.run(args, check=True, env=env, capture_output=True)
            out = subprocess.check_output([sys.executable, str(REPO / "agents/config.py"), "result-path", "claude",
                                           "--state", str(state)], env={"PATH": os.environ["PATH"]}, text=True)
            self.assertEqual(out.strip(), str(state / "results"))

    def test_optional_tool_provider_uses_configured_endpoint(self):
        path = REPO / "eval/service/toolathlon_verified/overlay/utils/api_model/model_provider.py"
        text = path.read_text()
        function = text[text.index("class CustomModelProviderQwenOfficial"):text.index("class CustomModelProviderKimiOfficial")]
        clients = []
        class Base:
            pass
        fake = type("Fake", (), {"qwen_official_key": "test"})()
        namespace = {"ModelProvider": Base, "Model": object, "global_configs": fake,
                     "AsyncOpenAI": lambda **kwargs: clients.append(kwargs) or kwargs,
                     "OpenAIChatCompletionsModelWithRetry": lambda **kwargs: kwargs}
        exec(function, namespace)
        provider = namespace["CustomModelProviderQwenOfficial"]()
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "VTB_TOOL_MODEL_BASE_URL"):
            provider.get_model("model")
        with mock.patch.dict(os.environ, {"VTB_TOOL_MODEL_BASE_URL": "http://example.invalid/v1"}, clear=True):
            provider.get_model("model")
        self.assertEqual(clients[0]["base_url"], "http://example.invalid/v1")


class LauncherAdapterTest(unittest.TestCase):
    def _phase(self, suffix, action, family="run_smoke", local=False, missing_model=False, execute=False, credential_file=True):
        # Strip configuration sourcing ONLY in this isolated mock harness, not production.
        text = (REPO / f"scripts/{family}/{family}_{suffix}.sh").read_text()
        text = text.replace('source "$REPO/src/commit_utils/launcher_agent.sh"', 'source "$REAL_REPO/src/commit_utils/launcher_agent.sh"')
        text = text[text.index("\n", text.index("set -euo pipefail")) + 1:]
        text = re.sub(r'REPO="\$\(cd .*?&& pwd\)"', 'REPO="$TEST_REPO"', text)
        # The harness checkout intentionally has no real credentials or benchmark .env.
        preflight_start = text.index("preflight() {") if family == "run_smoke" else -1
        stubs = r'''
set -euo pipefail
python3() {
    if [[ "$1" = */agents/config.py ]]; then shift; command "$TEST_PYTHON" "$REAL_REPO/agents/config.py" "$@"; return; fi
    if [ "$1" = -m ] && [ "$2" = eval.grading.client ]; then
        printf '%s\n' '{"status": "ok","validation":{"raw":"task:aime__aime","resolved_counts":{"aime__26":{"validation_task":"aime__aime","count":30},"terminal-bench__terminal-bench-2-1":{"validation_task":"openthoughts__openthoughts-tblite-val50","count":50}}}}'
        return
    fi
    return 0
}
docker() { return 0; }
ss() { :; }
sleep() { :; }
bash() {
    if [ "${1:-}" = src/run_task.sh ]; then
        printf 'MOCK_RUN adapter=%s model=%s root=%s\\n' "$3" "$7" "$VTB_AGENT_ADAPTERS_DIR"
        return
    fi
    return 0
}
command() { if [ "$1" = -v ]; then printf '/bin/true\n'; else builtin command "$@"; fi; }
tmux() {
    case "$1" in
        has-session) return 1 ;;
        new-session)
            printf 'MOCK_TMUX %s\n' "$*"
            if [ "${EXECUTE_TMUX:-0}" = 1 ]; then
                builtin command bash -c "${@: -1}"
            fi
            ;;
        *) return 1 ;;
    esac
}
'''
        # Bypass real setup/preflight bodies; retain researcher gate and command assembly.
        if family == "run_smoke":
            end = text.index("\n}\n", preflight_start) + 3
            text = text[:preflight_start] + 'preflight() { [ "${1:-}" = runtime ] || require_researcher; }\n' + text[end:]
        text = text.replace('validate_health >/dev/null', 'true')
        text = re.sub(r'(?<![A-Za-z_])health >/dev/null', 'true', text)
        text = text.replace('require_tau3\n', 'true\n')
        with tempfile.TemporaryDirectory() as tmp:
            interpreter = Path(tmp) / "mock-python3"
            interpreter.write_text('#!/bin/bash\npython3 "$@"\n')
            interpreter.chmod(0o755)
            env = {"PATH": os.environ["PATH"], "REAL_REPO": str(REPO), "TEST_REPO": tmp,
                   "VTB_PYTHON": str(interpreter), "TEST_PYTHON": sys.executable,
                   "VTB_ROOT": tmp, "VTB_GRADING_STATE": tmp + "/state", "SMOKE_ID": "test",
                   "RESEARCHER": "provider/model:v2", "ANTHROPIC_API_KEY": "test",
                   "VTB_TOOLATHLON_URL": "http://example.invalid", "VTB_TOOLATHLON_VERIFIED_URL": "http://example.invalid",
                   "VTB_MODEL_ADVERTISE_HOST": "example.invalid", "VTB_AGENT_ADAPTERS_DIR": str(REPO / "agents"),
                   "EXECUTE_TMUX": "1" if execute else "0"}
            if execute:
                helper = Path(tmp) / "src/commit_utils/load_launch_env.sh"
                helper.parent.mkdir(parents=True)
                helper.write_text((REPO / "src/commit_utils/load_launch_env.sh").read_text())
                (helper.parent / "configure_python.sh").write_text((REPO / "src/commit_utils/configure_python.sh").read_text())
                (Path(tmp) / ".python-version").write_text((REPO / ".python-version").read_text())
                if credential_file:
                    (Path(tmp) / "api_key.env").write_text('export ANTHROPIC_API_KEY=test\n')
            if action in {"run", "verify"}:
                state = Path(tmp) / "state"
                state.mkdir()
                if action == "verify":
                    result_dir = Path(tmp) / "recorded-results"
                    result_dir.mkdir()
                    (result_dir / "metrics.json").write_text('{}')
                    (state / "smoke_run.json").write_text(json.dumps({"adapter": "claude", "researcher": "provider/model:v2", "result_dir": str(result_dir)}))
                    archive = state / "archive"
                    archive.mkdir()
                    (archive / "backend_command.txt").write_text(f'harbor run --path {tmp}/hub_task_specs/sierra-research__tau3-bench/tau3-bench')
                    (archive / "tau3_runtime_state.json").write_text('{"bootstrap_complete": true}')
                    (state / "service.log").write_text('mock healthy service')
            if missing_model:
                env.pop("RESEARCHER")
            if local:
                root = Path(tmp) / "private"
                root.mkdir()
                AdapterConfigurationTest()._adapter(root, "custom")
                file = Path(tmp) / "local.env"
                file.write_text(f'export VTB_AGENT_ADAPTERS_DIR="{root}"\nexport AGENT_CLI=custom\nexport CUSTOM_API_KEY=test\n')
                env["VTB_LOCAL_ENV_FILE"] = str(file)
            stubs += '\nexport -f python3 docker ss sleep bash command tmux\n'
            result = subprocess.run(["bash", "-c", stubs + text, "test", action], env=env,
                                    capture_output=True, text=True, timeout=10)
        return result

    def test_all_smoke_runs_use_selected_model_and_adapter(self):
        for suffix in ("swe", "terminal", "tau3", "toolathlon", "aime"):
            with self.subTest(suffix=suffix):
                result = self._phase(suffix, "run")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("'claude'", result.stdout)
                self.assertIn("'provider/model:v2'", result.stdout)
                self.assertIn("claude_provider_model_v2_1h", result.stdout)
                self.assertIn("VTB_AGENT_ADAPTERS_DIR=", result.stdout)

    def test_local_adapter_runs_for_all_benchmarks(self):
        for suffix in ("swe", "terminal", "tau3", "toolathlon", "aime"):
            result = self._phase(suffix, "run", local=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("'custom'", result.stdout)
            self.assertIn("custom_provider_model_v2_1h", result.stdout)

    def test_detached_shell_loads_local_adapter_configuration(self):
        result = self._phase("swe", "run", local=True, execute=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MOCK_RUN adapter=custom model=provider/model:v2", result.stdout)
        self.assertIn("/private", result.stdout)

    def test_toolathlon_detached_run_without_credential_file(self):
        for local in (False, True):
            with self.subTest(local=local):
                result = self._phase("toolathlon", "run", local=local, execute=True, credential_file=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                adapter = "custom" if local else "claude"
                self.assertIn(f"MOCK_RUN adapter={adapter} model=provider/model:v2", result.stdout)

    def test_run_requires_model_but_idle_service_does_not(self):
        result = self._phase("swe", "run", missing_model=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("MOCK_TMUX", result.stdout)
        result = self._phase("swe", "service", missing_model=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MOCK_TMUX", result.stdout)

    def test_all_idle_smoke_services_and_health_without_model(self):
        for suffix in ("swe", "terminal", "tau3", "toolathlon", "aime"):
            for phase in ("service", "health", "cleanup", "verify"):
                with self.subTest(suffix=suffix, phase=phase):
                    result = self._phase(suffix, phase, missing_model=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertNotIn("MOCK_RUN", result.stdout)

    def test_all_formal_services_and_health_without_model(self):
        for suffix in ("swe", "terminal", "tau3", "toolathlon", "aime"):
            for phase in ("service", "health"):
                result = self._phase(suffix, phase, family="run_exp", missing_model=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_formal_runs_use_sanitized_model_prefix(self):
        for suffix in ("swe", "terminal", "tau3", "toolathlon", "aime"):
            result = self._phase(suffix, "run", family="run_exp")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("'provider/model:v2'", result.stdout)
            self.assertIn("claude_provider_model_v2_", result.stdout)


if __name__ == "__main__":
    unittest.main()
