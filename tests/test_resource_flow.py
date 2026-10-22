from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval.grading import config, valsplit
from eval.grading.store import BaselineCache
from eval.runner import resource_assets as assets
from src.commit_utils import prepare_agent_workspace as workspace
from src.commit_utils import resolve_validation

REPO = Path(__file__).resolve().parents[1]


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, REPO / f"scripts/prepare/python/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cache_check = load_helper("verify_recipe_cache")
model_prepare = load_helper("prepare_model")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ResourceFlowTest(unittest.TestCase):
    def test_all_short_baselines_resolve_and_mount(self):
        for path in (REPO / "training/resources/validation").glob("*/resource_config.json"):
            cfg = json.loads(path.read_text())
            if cfg["type"] != "validation_baseline":
                continue
            source = valsplit.expand_val_source(REPO, cfg["val_source"])
            found = valsplit.find_baseline_resource(REPO, cfg["benchmark"], cfg["base_model"], source)
            self.assertEqual(found["id"], cfg["id"])
            with tempfile.TemporaryDirectory() as tmp:
                health = Path(tmp) / "health.json"
                write_json(health, {"validation": {"resolved_counts": {cfg["benchmark"]: {"validation_source": source}}}})
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    resolve_validation._baseline_for_source(REPO, cfg["benchmark"], cfg["base_model"], cfg["val_source"], health)
                self.assertEqual(Path(out.getvalue().strip()), valsplit.resolve_resource_path(REPO, cfg["materialized_path"]))
                with mock.patch.object(config, "REPO_ROOT", REPO), mock.patch.object(config, "VAL_SOURCE_RAW", source):
                    cache = BaselineCache(Path(tmp) / "cache")
                    self.assertEqual(cache.get(cfg["benchmark"], cfg["base_model"]), cfg["score"])
                    self.assertEqual(cache.baseline_payload_dir(cfg["benchmark"], cfg["base_model"], source), Path(out.getvalue().strip()))

    def test_ambiguous_baselines_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            cfg = {"type": "validation_baseline", "benchmark": "bench", "base_model": "model", "val_source": "split:1"}
            for name in ("one", "two"):
                write_json(repo / f"training/resources/validation/{name}/resource_config.json", cfg)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                valsplit.find_baseline_resource(repo, "bench", "model", "split:1")

    def test_configless_script_is_copied_to_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            source = repo / "training/resources/scripts/example/run.sh"
            source.parent.mkdir(parents=True)
            source.write_text("true\n")
            result = workspace.copy_training_resource(repo, repo / "task", "scripts/example")
            self.assertEqual(result["copied_to"], "train_resources/scripts/example")
            self.assertTrue((repo / "task" / result["copied_to"] / "run.sh").is_file())

    def test_snapshot_files_and_custom_hf_home_are_checked(self):
        with tempfile.TemporaryDirectory(prefix="cache with spaces ") as tmp:
            repo = Path(tmp)
            cfg = {"id": "data/example", "type": "hf_cache_dataset", "repo_id": "org/data", "revision": "a" * 40,
                   "snapshot_path": "${HF_HOME}/hub/datasets--org--data/snapshots/" + "a" * 40,
                   "data_files": {"train": "data/shard-{0,1}.parquet"}}
            write_json(repo / "training/resources/data/example/resource_config.json", cfg)
            with mock.patch.dict(os.environ, {"HF_HOME": str(repo / "custom cache")}, clear=True):
                snapshot = assets.resolve_path(repo, cfg["snapshot_path"])
                snapshot.mkdir(parents=True)
                self.assertTrue(assets.resource_errors(repo, cfg["id"]))
                first = snapshot / "data/shard-0.parquet"
                first.parent.mkdir(exist_ok=True)
                first.write_bytes(b"test")
                errors = assets.resource_errors(repo, cfg["id"])
                self.assertEqual(len(errors), 1)
                self.assertIn("shard-1.parquet", errors[0])
                (snapshot / "data/shard-1.parquet").write_bytes(b"test")
                self.assertEqual(assets.resource_errors(repo, cfg["id"]), [])
                result = workspace.copy_training_resource(repo, repo / "task", cfg["id"])
                dest = repo / "task" / result["copied_to"]
                self.assertEqual([p.name for p in dest.iterdir()], ["resource_config.json"])

    def test_download_argv_uses_manifest_and_preserves_space_paths(self):
        with tempfile.TemporaryDirectory(prefix="download with spaces ") as tmp:
            repo = Path(tmp)
            cfg = {"id": "data/tau", "type": "hf_cache_dataset", "repo_id": "org/tau", "revision": "a" * 40,
                   "snapshot_path": "${HF_HOME}/snapshot", "data_files": {"train": "corpus/data.jsonl"}}
            write_json(repo / "training/resources/data/tau/resource_config.json", cfg)
            write_json(repo / "training/recipes/example.json", {"resources": ["data/tau"]})
            with mock.patch.object(cache_check, "REPO", repo), mock.patch.dict(os.environ, {"HF_HOME": str(repo / "cache with spaces")}, clear=True), mock.patch.object(cache_check.subprocess, "run") as run, mock.patch.object(assets, "dataset_errors", return_value=[]):
                self.assertEqual(cache_check.download_recipe("example"), [])
                argv = run.call_args.args[0]
                self.assertIn(str(repo / "cache with spaces") + ":/hf", argv)
                self.assertIn("a" * 40, argv)
                self.assertEqual(argv[argv.index("--include") + 1:], ["corpus/data.jsonl"])
                self.assertNotIn("eval/tau3_val_test_split.json", argv)

    def test_formal_recipe_and_published_payload_alignment(self):
        with mock.patch.object(assets, "model_errors", return_value=[]), \
                mock.patch.object(assets, "dataset_errors", return_value=[]), \
                mock.patch.object(assets, "materialized_dataset_errors", return_value=[]):
            for suffix in ("swe", "terminal", "tau", "toolathlon", "aime"):
                self.assertEqual(assets.recipe_errors(REPO, f"qwen3_5_9b/slime_sft_{suffix}"), [])

    def test_tau_download_selects_each_training_pattern_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"HF_HOME": tmp}), mock.patch.object(cache_check.subprocess, "run") as run, mock.patch.object(assets, "dataset_errors", return_value=[]):
            self.assertEqual(cache_check.download_recipe("qwen3_5_9b/slime_sft_tau"), [])
            argv = run.call_args.args[0]
            self.assertEqual(argv.count("--include"), 3)
            self.assertIn("ec268e4c203bb9e015ce759813f57fae599c88c4", argv)
            self.assertFalse(any("eval/" in arg for arg in argv))

    def test_tau_nemotron_is_pinned_and_split_is_not_training(self):
        _, cfg = assets.read_resource(REPO, "data/tau_data")
        self.assertEqual(cfg["revision"], "ec268e4c203bb9e015ce759813f57fae599c88c4")
        self.assertEqual(len(cfg["data_files"]), 3)
        self.assertTrue(any("Nemotron" in name for name in cfg["data_files"].values()))
        self.assertFalse(any(name.startswith("eval/") for name in cfg["data_files"].values()))
        self.assertNotIn("pending", cfg)

    def test_terminal_and_aime_resources_are_formally_declared(self):
        self.assertFalse(any(p.is_file() for p in (REPO / "training/resources/data/terminalbench_rts_reward1_debug64").glob("**/*")))
        for name in ("terminal", "aime"):
            _, cfg = assets.read_resource(REPO, f"data/{name}")
            self.assertNotIn("pending", cfg)
            self.assertFalse(any("pending" in error for error in assets.resource_errors(REPO, f"data/{name}")))

    def test_terminal_and_aime_sources_are_pinned(self):
        _, terminal = assets.read_resource(REPO, "data/terminal")
        self.assertEqual(terminal["repo_id"], "prt66/terminal_data")
        self.assertEqual(terminal["revision"], "dd833a625e032842b2d84bd2b993b7c40e301f3c")
        self.assertEqual(len(terminal["data_files"]), 3)
        expected = {
            "data/aime_open_thoughts3": "61bcf9d4eb38b30295efc2021227a63cc5bb34c8",
            "data/aime_nemotron_math": "84d42ad0cb960f07f951b9baa9ed2b46a5a18c66",
            "data/aime_openr1_math": "e4e141ec9dea9f8326f4d347be56105859b2bd68",
        }
        for resource_id, revision in expected.items():
            _, cfg = assets.read_resource(REPO, resource_id)
            self.assertEqual(cfg["revision"], revision)
        recipe = json.loads(assets.recipe_file(REPO, "qwen3_5_9b/slime_sft_aime").read_text())
        self.assertTrue(set(expected).issubset(recipe["resources"]))
        self.assertEqual(recipe["env_defaults"]["TRAIN_DATA"], "train_resources/data/aime/data/train.jsonl")

    def test_aime_download_selects_three_pinned_sources(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"HF_HOME": tmp}), \
                mock.patch.object(cache_check.subprocess, "run") as run, \
                mock.patch.object(assets, "dataset_errors", return_value=[]):
            self.assertEqual(cache_check.download_recipe("qwen3_5_9b/slime_sft_aime"), [])
            self.assertEqual(run.call_count, 3)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertTrue(all("--revision" in command and "--include" in command for command in commands))

    def test_model_normalization_and_conversion_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            cfg = {"id": "models/example", "repo_id": "org/model", "revision": "a" * 40,
                   "eos_token": "<|im_end|>", "eos_token_id": 42,
                   "hf_checkpoint_path": str(repo / "hf"), "torch_dist_path": str(repo / "torch_dist")}
            checkpoint = repo / "hf"
            write_json(checkpoint / "tokenizer_config.json", {"eos_token": "bad", "chat_template": "<|im_end|>", "added_tokens_decoder": {"42": {"content": "<|im_end|>"}}})
            write_json(checkpoint / "config.json", {"text_config": {"eos_token_id": 0}})
            (checkpoint / "model.safetensors").write_bytes(b"weights")
            model_prepare.normalize_checkpoint(checkpoint, cfg)
            self.assertEqual(assets.model_errors(repo, cfg, require_conversion=False), [])
            self.assertTrue(assets.model_errors(repo, cfg))
            converted = repo / "torch_dist"
            converted.mkdir()
            (converted / "weights.distcp").write_bytes(b"converted")
            write_json(converted / assets.MODEL_SOURCE_FILE, assets.model_signature(cfg))
            self.assertEqual(assets.model_errors(repo, cfg), [])
            cfg["revision"] = "b" * 40
            self.assertTrue(assets.model_errors(repo, cfg))

    def test_offline_model_materialization_preserves_shared_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            snapshot = repo / "hf/snapshots" / ("a" * 40)
            cfg = {"id": "models/example", "type": "cached_model", "repo_id": "org/model", "revision": "a" * 40,
                   "eos_token": "<|im_end|>", "eos_token_id": 42,
                   "snapshot_path": str(snapshot), "hf_checkpoint_path": str(repo / "fresh/hf_checkpoint"),
                   "torch_dist_path": str(repo / "fresh/torch_dist")}
            write_json(snapshot / "tokenizer_config.json", {"eos_token": "old", "chat_template": "<|im_end|>",
                       "added_tokens_decoder": {"42": {"content": "<|im_end|>"}}})
            write_json(snapshot / "config.json", {"eos_token_id": 0})
            write_json(snapshot / "tokenizer.json", {})
            (snapshot / "model.safetensors").write_bytes(b"weights")
            with mock.patch.object(model_prepare, "REPO", repo), mock.patch.object(model_prepare.subprocess, "run") as run:
                model_prepare.materialize_model(cfg, offline=True)
                run.assert_not_called()
            self.assertEqual(json.loads((snapshot / "tokenizer_config.json").read_text())["eos_token"], "old")
            self.assertFalse((snapshot / assets.MODEL_SOURCE_FILE).exists())
            self.assertEqual(assets.model_errors(repo, cfg, require_conversion=False), [])
            self.assertTrue(assets.model_errors(repo, cfg))
            converted = Path(cfg["torch_dist_path"])
            converted.mkdir()
            (converted / "weights.distcp").write_bytes(b"converted fixture")
            write_json(converted / assets.MODEL_SOURCE_FILE, assets.model_signature(cfg))
            write_json(repo / "training/resources/models/example/resource_config.json", cfg)
            task = repo / "job/task"
            result = workspace.copy_training_resource(repo, task, cfg["id"])
            model = task / result["copied_to"]
            self.assertEqual((model / "hf_checkpoint/model.safetensors").read_bytes(), b"weights")
            self.assertEqual((model / "torch_dist/weights.distcp").read_bytes(), b"converted fixture")

    def test_unknown_model_provenance_refused_before_network_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            checkpoint = repo / "legacy"
            checkpoint.mkdir()
            cfg = {"repo_id": "org/model", "revision": "a" * 40, "eos_token": "<|im_end|>", "eos_token_id": 42,
                   "hf_checkpoint_path": str(checkpoint), "snapshot_path": str(repo / "absent")}
            with mock.patch.object(model_prepare, "REPO", repo), mock.patch.object(model_prepare.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "unknown/different provenance"):
                    model_prepare.materialize_model(cfg)
                run.assert_not_called()
            self.assertEqual(list(checkpoint.iterdir()), [])

    def test_model_root_override_matches_workspace_resolver(self):
        with mock.patch.dict(os.environ, {"VTB_ROOT": "/runtime", "VTB_MODEL_RESOURCE_ROOT": "/fresh/models"}, clear=True):
            _, cfg = assets.read_resource(REPO, "models/qwen3_5_9b_base")
            host = assets.resolve_path(REPO, cfg["hf_checkpoint_path"])
            workspace_path = workspace.resolve_declared_path(REPO, cfg["hf_checkpoint_path"], "hf_checkpoint_path", cfg["id"])
            self.assertEqual(host, Path("/fresh/models/hf_checkpoint"))
            self.assertEqual(host, workspace_path)

    def test_materialized_aime_payload_reaches_recipe_train_data_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            payload = repo / "prepared/aime"
            payload.mkdir(parents=True)
            train = payload / "train.jsonl"
            train.write_text('{"messages": [{"role": "user", "content": "test"}]}\n')
            write_json(payload / "source_manifest.json", {"materialized_file": "train.jsonl", "rows": 1,
                       "bytes": train.stat().st_size, "sha256": assets.sha256_file(train), "sources": []})
            cfg = {"id": "data/aime", "type": "hf_dataset", "materialized_path": str(payload),
                   "materialized_file": "train.jsonl", "num_rows": 1}
            write_json(repo / "training/resources/data/aime/resource_config.json", cfg)
            task = repo / "job/task"
            result = workspace.copy_training_resource(repo, task, "data/aime")
            self.assertEqual(result["copied_to"], "train_resources/data/aime/data")
            self.assertEqual((task / result["copied_to"] / "train.jsonl").read_bytes(), train.read_bytes())
            recipe = json.loads(assets.recipe_file(REPO, "qwen3_5_9b/slime_sft_aime").read_text())
            self.assertTrue((task / recipe["env_defaults"]["TRAIN_DATA"]).is_file())

    def test_wildcard_shards_require_declared_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train-0.parquet").write_bytes(b"fixture")
            cfg = {"id": "data/example", "revision": "a" * 40, "snapshot_path": str(root),
                   "data_files": {"train": "train-*.parquet"}, "file_count_check": {"train-*.parquet": 2}}
            self.assertTrue(any("file count mismatch" in error for error in assets.dataset_errors(REPO, cfg)))
            (root / "train-1.parquet").write_bytes(b"fixture")
            self.assertEqual(assets.dataset_errors(REPO, cfg), [])

    def test_snapshot_check_does_not_accept_other_cached_revisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            snapshot = repo / "cache" / ("b" * 40)
            snapshot.mkdir(parents=True)
            (snapshot / "model.safetensors").write_bytes(b"weights")
            for name in ("tokenizer_config.json", "tokenizer.json", "config.json"):
                write_json(snapshot / name, {})
            cfg = {"snapshot_path": str(snapshot), "revision": "a" * 40}
            self.assertTrue(any("revision mismatch" in error for error in assets.snapshot_errors(repo, cfg)))

    def test_aime_baseline_counts_are_measured_not_divided(self):
        rows = [json.loads(p.read_text()) for p in (REPO / "benchmarks/aime__26/baseline/samples").glob("*/direct_trajectory.json")]
        cfg = json.loads((REPO / "training/resources/validation/aime_baseline/resource_config.json").read_text())
        self.assertEqual(len(rows), cfg["graded"])
        self.assertEqual(sum(r["correct_completions"] == r["samples_per_task"] for r in rows), cfg["solved"])
        self.assertAlmostEqual(sum(r["avg_at_n"] for r in rows) / len(rows), cfg["score"])

    def test_formal_launchers_gate_resources_without_running_downloads(self):
        for suffix in ("swe", "terminal", "tau3", "toolathlon", "aime"):
            text = (REPO / f"scripts/run_exp/run_exp_{suffix}.sh").read_text()
            self.assertIn('verify_recipe_cache.py', text)
            self.assertIn("HF_HOME='$HF_HOME'", text)
            self.assertIn("VTB_MODEL_RESOURCE_ROOT='$VTB_MODEL_RESOURCE_ROOT'", text)
            self.assertNotIn("--download", text)
        tool = (REPO / "scripts/prepare/prepare_toolathlon.sh").read_text()
        self.assertIn("all) download_datasets; preflight --formal", tool)
        for path in (REPO / "scripts/prepare").glob("prepare_*.sh"):
            self.assertNotIn("local hf_dl=", path.read_text())


if __name__ == "__main__":
    unittest.main()
