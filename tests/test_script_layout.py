from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


class ScriptLayoutTest(unittest.TestCase):
    def test_python_is_only_in_action_python_directories(self) -> None:
        allowed = {
            SCRIPTS / "prepare/python",
            SCRIPTS / "run_smoke/python",
        }
        for path in SCRIPTS.rglob("*.py"):
            self.assertTrue(any(root == path.parent or root in path.parents for root in allowed), path)
        for directory in SCRIPTS.rglob("*"):
            if not directory.is_dir():
                continue
            self.assertFalse(
                any(directory.glob("*.py")) and any(directory.glob("*.sh")),
                f"mixed Python and shell scripts: {directory}",
            )

    def test_published_experiment_defaults(self) -> None:
        expected = {
            "swe": ("24", False, "slime_sft_swe"),
            "terminal": ("24", False, "slime_sft_terminal"),
            "tau3": ("16", True, "slime_sft_tau"),
            "toolathlon": ("16", True, "slime_sft_toolathlon"),
            "aime": ("16", True, "slime_sft_aime"),
        }
        for suffix, (hours, fixed_harness, recipe) in expected.items():
            text = (SCRIPTS / f"run_exp/run_exp_{suffix}.sh").read_text(encoding="utf-8")
            self.assertIn(f'HOURS="${{HOURS:-{hours}}}"', text)
            self.assertIn("NUM_GPUS=\"${NUM_GPUS:-8}\"", text)
            self.assertIn("0,1,2,3,4,5,6,7", text)
            self.assertIn(recipe, text)
            self.assertIn("gpu_suffix", text)
            self.assertIn("VTB_GRADING_HOST", text)
            self.assertIn("VTB_SERVE_EXTRA_ARGS", text)
            self.assertEqual("VTB_EDITABLE_HARNESS=off" in text, fixed_harness)
            self.assertIn("scripts/prepare/configure_evaluation_environment.sh", text)

    def test_grading_state_defaults_are_grouped_and_overridable(self) -> None:
        for suffix in ("swe", "terminal", "tau3", "toolathlon", "aime"):
            formal = (SCRIPTS / f"run_exp/run_exp_{suffix}.sh").read_text()
            smoke = (SCRIPTS / f"run_smoke/run_smoke_{suffix}.sh").read_text()
            self.assertIn(f'GRADING_STATE="${{VTB_GRADING_STATE:-$VTB_ROOT/grading_states/formal/{suffix}}}"', formal)
            self.assertIn(f'STATE="${{VTB_GRADING_STATE:-$VTB_ROOT/grading_states/smoke/{suffix}_${{SMOKE_ID}}}}"', smoke)

    def test_python_state_paths_follow_runtime_root_and_explicit_override(self) -> None:
        env = {key: value for key, value in os.environ.items()
               if key not in {"VTB_GRADING_STATE", "VTB_ROOT", "VTB_REPO"}}
        env["VTB_ROOT"] = "/data/elsewhere"
        code = (
            "from eval.runner import runtime_paths as r; "
            "from eval.grading import config as c; "
            "assert c.STATE_DIR == r.GRADING_STATE_DIR; "
            "print(c.STATE_DIR); print(c.BASELINE_CACHE_DIR)"
        )
        result = subprocess.check_output([sys.executable, "-c", code], cwd=REPO, env=env, text=True)
        self.assertEqual(result.splitlines(), [
            "/data/elsewhere/grading_states/direct",
            "/data/elsewhere/grading_states/direct/baselines",
        ])
        env["VTB_GRADING_STATE"] = "/custom/legacy-state"
        result = subprocess.check_output([sys.executable, "-c", code], cwd=REPO, env=env, text=True)
        self.assertEqual(result.splitlines(), ["/custom/legacy-state", "/custom/legacy-state/baselines"])

    def test_client_admin_lookup_matches_direct_state_default(self) -> None:
        from eval.grading import client
        args = argparse.Namespace(admin_token_file=None)
        paths = []
        def read_token(path, *args, **kwargs):
            paths.append(path)
            return "test-token"
        with mock.patch.dict(os.environ, {"VTB_ROOT": "/data/elsewhere"}, clear=True), \
                mock.patch.object(Path, "read_text", read_token):
            client._admin_token(args)
        self.assertEqual(paths, [Path("/data/elsewhere/grading_states/direct/admin.token")])
        paths.clear()
        with mock.patch.dict(os.environ, {"VTB_GRADING_STATE": "/custom/legacy-state"}, clear=True), \
                mock.patch.object(Path, "read_text", read_token):
            client._admin_token(args)
        self.assertEqual(paths, [Path("/custom/legacy-state/admin.token")])

    def test_publish_action_surfaces_are_minimal(self) -> None:
        suffixes = {"swe", "terminal", "tau3", "toolathlon", "aime"}
        tau_prepare = SCRIPTS / "prepare/prepare_tau3.sh"
        self.assertIn("prepare_tau3.sh <datasets|image|tasks|preflight|all>", tau_prepare.read_text(encoding="utf-8"))
        self.assertEqual(
            {path.stem[len("prepare_"):] for path in (SCRIPTS / "prepare").glob("prepare_*.sh")},
            suffixes,
        )
        for suffix in ("swe", "toolathlon"):
            prepare_path = f"scripts/prepare/prepare_{suffix}.sh"
            self.assertIn(prepare_path, (SCRIPTS / f"run_exp/run_exp_{suffix}.sh").read_text(encoding="utf-8"))
            self.assertIn(prepare_path, (SCRIPTS / f"run_smoke/run_smoke_{suffix}.sh").read_text(encoding="utf-8"))
        self.assertEqual(
            {path.name for path in (SCRIPTS / "run_exp").iterdir() if path.is_file()},
            {f"run_exp_{suffix}.sh" for suffix in suffixes},
        )
        self.assertEqual(
            {path.name for path in (SCRIPTS / "run_smoke").glob("*.sh")},
            {f"run_smoke_{suffix}.sh" for suffix in suffixes},
        )

    def test_benchmark_entrypoint_suffixes(self) -> None:
        suffixes = {"swe", "terminal", "tau3", "toolathlon", "aime"}
        self.assertEqual(
            {path.stem[len("run_exp_"):] for path in (SCRIPTS / "run_exp").glob("run_exp_*.sh")},
            suffixes,
        )
        self.assertEqual(
            {path.stem[len("run_smoke_"):] for path in (SCRIPTS / "run_smoke").glob("run_smoke_*.sh")},
            suffixes,
        )

    def test_removed_script_names_have_no_tracked_references(self) -> None:
        removed = (
            "run_research.sh",
            "run_tau2_smoke.sh",
            "prepare_tau2.sh",
            "preflight_tau2.py",
            "patch_tau3_export.py",
            "set_eval_env.sh",
            "run_eval.sh",
            "add_hub_benchmark.sh",
            "test_toolathlon_submit.sh",
            "import_baseline_tau3.sh",
            "run_validation_tau3.sh",
            "run_final_tau3.sh",
            "run_full_sweep_toolathlon.py",
            "generate_validation_context_terminal.py",
            "baseline_eval",
            "evaluate_model.sh",
            "serve_model.sh",
            "prepare_tau2_for_tau3.sh",
            "build_tau2_base_image_for_tau3.sh",
            "slime_sft_swe_hf",
            "slime_sft_terminal_debug64",
            "slime_sft_tau_preexp",
            "slime_sft_toolathlon_pool",
            "slime_sft_swe_hf_debug64",
            "slime_sft_swe_debug64",
            "slime_sft_tau_debug64",
            "train_test/qwen3_5_9b",
            "data/custom/",
            "data/hf/",
        )
        pattern = re.compile("|".join(re.escape(name) for name in removed))
        offenders = []
        for root_name in ("README.md", "scripts", "eval", "benchmarks", "training", "src", "tests"):
            root = REPO / root_name
            paths = [root] if root.is_file() else root.rglob("*")
            for path in paths:
                if path.resolve() == Path(__file__).resolve():
                    continue
                if not path.is_file() or path.suffix not in {".md", ".py", ".sh", ".json", ".yaml", ".yml"}:
                    continue
                if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                    offenders.append(str(path.relative_to(REPO)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
