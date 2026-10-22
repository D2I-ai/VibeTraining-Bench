from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval.runner import resource_assets as assets

REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO / "scripts/prepare/python/aime_sft_pipeline"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, PIPELINE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


math_converter = load_module("convert_math_to_slime_sft")
cot_converter = load_module("convert_cot_to_slime_sft")
openr1_converter = load_module("convert_default_to_slime_sft")
prepare = load_module("prepare")


class AimeSftPipelineTest(unittest.TestCase):
    def test_open_thoughts_conversion_preserves_reasoning(self):
        messages = math_converter.convert_conversations([
            {"from": "human", "value": "Solve x."},
            {"from": "gpt", "value": "<think>work</think>answer"},
        ])
        self.assertTrue(messages[0]["content"].endswith(math_converter.REASONING_INSTRUCTION))
        self.assertEqual(messages[1], {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "work",
        })

    def test_nemotron_conversion_keeps_message_fields(self):
        messages = [{"role": "user", "content": "Question"}, {
            "role": "assistant", "content": "Answer", "reasoning_content": "Reason"
        }]
        row = {
            "uuid": "id",
            "messages": messages,
            "subset": "cot",
            "expected_answer": "Answer",
            "source": "source",
            "license": "license",
        }
        record = cot_converter.build_record(row, "shard.parquet", 0)
        self.assertEqual(record["messages"][1]["reasoning_content"], "Reason")
        self.assertTrue(record["messages"][0]["content"].endswith(cot_converter.REASONING_INSTRUCTION))

    def test_openr1_conversion_extracts_thinking(self):
        messages = openr1_converter.convert_messages([
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "<think>Reason</think>Answer"},
        ])
        self.assertEqual(messages[1]["reasoning_content"], "Reason")
        self.assertEqual(messages[1]["content"], "Answer")

    def test_materialized_dataset_provenance_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            hf_home = repo / "hf"
            signatures = []
            for item in prepare.SOURCES:
                resource_id = str(item["resource"])
                revision = "a" * 40
                snapshot = hf_home / f"snapshot-{len(signatures)}"
                (snapshot / "data").mkdir(parents=True)
                (snapshot / "data/train-0.parquet").write_bytes(b"fixture")
                cfg = {
                    "id": resource_id,
                    "type": "hf_cache_dataset",
                    "repo_id": f"org/source-{len(signatures)}",
                    "revision": revision,
                    "snapshot_path": str(snapshot),
                    "data_files": {"train": "data/train-*.parquet"},
                }
                path = repo / "training/resources" / resource_id / "resource_config.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(cfg), encoding="utf-8")
                signatures.append({
                    "resource": resource_id,
                    "repo_id": cfg["repo_id"],
                    "revision": cfg["revision"],
                    "data_files": cfg["data_files"],
                    "rows": item["rows"],
                    "converter": item["converter"],
                    "converter_sha256": assets.sha256_file(PIPELINE / str(item["converter"])),
                })
            output = repo / "materialized"
            output.mkdir()
            (output / "train.jsonl").write_text('{"messages": [{"role": "user", "content": "test"}]}\n', encoding="utf-8")
            (output / "source_manifest.json").write_text(json.dumps({
                "materialized_file": "train.jsonl",
                "rows": 1,
                "bytes": (output / "train.jsonl").stat().st_size,
                "sha256": assets.sha256_file(output / "train.jsonl"),
                "sources": signatures,
            }), encoding="utf-8")
            cfg = {
                "id": "data/aime",
                "type": "hf_dataset",
                "materialized_path": str(output),
                "materialized_file": "train.jsonl",
                "source_manifest": "source_manifest.json",
                "source_resources": [item["resource"] for item in prepare.SOURCES],
                "num_rows": 1,
            }
            self.assertEqual(assets.materialized_dataset_errors(repo, cfg), [])
            with mock.patch.object(prepare, "TOTAL_ROWS", 1):
                prepare.validate_existing(output, prepare.source_signatures(repo, hf_home))
            manifest = json.loads((output / "source_manifest.json").read_text())
            (output / "train.jsonl").write_text('{"messages": [{"role": "user", "content": "best"}]}\n', encoding="utf-8")
            self.assertTrue(any("SHA-256 mismatch" in error for error in assets.materialized_dataset_errors(repo, cfg)))
            manifest["rows"] -= 1
            (output / "source_manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(assets.materialized_dataset_errors(repo, cfg))

    def test_invalid_messages_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text('{"messages": []}\n')
            self.assertTrue(assets.jsonl_errors(path, rows=1, sha256=assets.sha256_file(path)))

    def test_changed_preparation_script_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            script = repo / "prepare.py"
            script.write_text("first\n")
            cfg = {"preparation_scripts": ["prepare.py"]}
            before = assets.preparation_signature(repo, cfg)
            script.write_text("second\n")
            self.assertNotEqual(before, assets.preparation_signature(repo, cfg))

    def test_parquet_pipeline_materializes_and_reuses_valid_output(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow integration test runs inside the trainer image")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            hf_home = repo / "hf"
            sources = tuple(dict(item, rows=1) for item in prepare.SOURCES)
            rows = [
                [{"difficulty": 1, "source": "fixture", "domain": "math", "conversations": [
                    {"from": "human", "value": "Question"}, {"from": "gpt", "value": "<think>Reason</think>Answer"}]},
                 {"difficulty": 1, "source": "fixture", "domain": "code", "conversations": []}],
                [{"uuid": "cot-id", "subset": "cot", "messages": [
                    {"role": "user", "content": "Question", "reasoning_content": None},
                    {"role": "assistant", "content": "Answer", "reasoning_content": "Reason"}],
                  "expected_answer": "Answer", "source": "fixture", "license": "test"}],
                [{"uuid": "r1-id", "answer": "Answer", "source": "fixture", "problem_type": "test",
                  "question_type": "test", "correctness_count": 1, "messages": [
                      {"role": "user", "content": "Question"},
                      {"role": "assistant", "content": "<think>Reason</think>Answer"}]}],
            ]
            for item, records in zip(sources, rows):
                snapshot = hf_home / str(item["resource"]).replace("/", "_") / ("a" * 40)
                (snapshot / "data").mkdir(parents=True)
                pq.write_table(pa.Table.from_pylist(records), snapshot / "data/train-0.parquet")
                cfg = {"id": item["resource"], "type": "hf_cache_dataset", "repo_id": "org/test",
                       "revision": "a" * 40, "snapshot_path": str(snapshot),
                       "data_files": {"train": "data/train-*.parquet"}, "file_count_check": {"data/train-*.parquet": 1}}
                path = repo / "training/resources" / str(item["resource"]) / "resource_config.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(cfg))
            target = repo / "scripts/prepare/python/aime_sft_pipeline"
            shutil.copytree(PIPELINE, target, ignore=shutil.ignore_patterns("__pycache__"))
            cfg_path = repo / "training/resources/data/aime/resource_config.json"
            cfg_path.parent.mkdir(parents=True)
            cfg = json.loads((REPO / "training/resources/data/aime/resource_config.json").read_text())
            cfg["num_rows"] = 3
            cfg["materialized_path"] = str(repo / "output")
            cfg_path.write_text(json.dumps(cfg))
            with mock.patch.object(prepare, "SOURCES", sources), mock.patch.object(prepare, "TOTAL_ROWS", 3):
                prepare.materialize(repo, hf_home, repo / "output")
                prepare.materialize(repo, hf_home, repo / "output")
            self.assertEqual(assets.materialized_dataset_errors(repo, cfg), [])
            manifest = json.loads((repo / "output/source_manifest.json").read_text())
            self.assertEqual(manifest["rows"], 3)
            self.assertEqual(manifest["preparation_sha256"], assets.preparation_signature(repo, cfg))

    def test_prepare_entrypoint_invokes_containerized_pipeline(self):
        text = (REPO / "scripts/prepare/prepare_aime.sh").read_text()
        self.assertIn("aime_sft_pipeline/prepare.py", text)
        self.assertIn("prepare_training_data", text)
        self.assertIn('\"$VTB_PYTHON\" scripts/prepare/python/verify_recipe_cache.py --download "$recipe"', text)
        self.assertIn('--check-sources "$recipe"', text)
        self.assertIn('--network none', text)


if __name__ == "__main__":
    unittest.main()
