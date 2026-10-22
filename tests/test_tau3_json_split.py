from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.grading import config, valsplit  # noqa: E402
from eval.grading.server import GradingService  # noqa: E402


TAU3 = "sierra-research__tau3-bench"
VAL_1 = "tau3-bench__tau3-airline-val-1"
VAL_2 = "tau3-bench__tau3-airline-val-2"
TEST_1 = "tau3-bench__tau3-airline-test-1"
TEST_2 = "tau3-bench__tau3-airline-test-2"


class Tau3JsonSplitTest(unittest.TestCase):
    def write_split(self, root: Path, payload: dict) -> Path:
        path = root / "split.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_parser_redacts_membership_from_public_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_split(Path(tmp), {"val": [VAL_1, VAL_2], "test": [TEST_1, TEST_2]})
            source = valsplit.parse_val_source(f"json:{path}")
            self.assertEqual(source.kind, "json")
            self.assertEqual(source.json_sha256, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(source.to_json()["val_count"], 2)
            self.assertNotIn("val", source.to_json())
            self.assertEqual(source.operator_json()["test"], [TEST_1, TEST_2])

    def test_parser_rejects_non_tau_and_invalid_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute JSON"):
            valsplit.parse_val_source("json:split.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid = self.write_split(root, {"val": ["other-task"], "test": [TEST_1]})
            with self.assertRaisesRegex(ValueError, "invalid Tau3"):
                valsplit.parse_val_source(f"json:{invalid}")
            overlap = self.write_split(root, {"val": [VAL_1], "test": [VAL_1]})
            with self.assertRaisesRegex(ValueError, "both val and test"):
                valsplit.parse_val_source(f"json:{overlap}")

    def test_explicit_and_test_only_partitioning(self) -> None:
        results = valsplit.SampleResults([VAL_1, TEST_1], {VAL_1: True, TEST_1: False}, source="test")
        split = valsplit.partition(results, reported_accuracy=0.5, val_ids=[VAL_1], test_ids=[TEST_1])
        self.assertTrue(split["reconstruction_ok"])
        self.assertEqual(split["accuracy_val"], 1.0)
        self.assertEqual(split["accuracy_heldout"], 0.0)

        partial = valsplit.SampleResults([TEST_1], {TEST_1: True}, source="test")
        test_only = valsplit.partition_test_only(partial, [TEST_1, TEST_2], reported_accuracy=1.0)
        self.assertTrue(test_only["reconstruction_ok"])
        self.assertEqual(test_only["accuracy_test"], 0.5)
        self.assertEqual(test_only["test_missing_ids"], [TEST_2])

    def test_benchmark_source_expands_runtime_root(self) -> None:
        original_cache = dict(config._validation_source_cache)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config._validation_source_cache.clear()
                with mock.patch.object(config, "VTB_ROOT", root), mock.patch.object(
                    config,
                    "_registry_spec_attr",
                    return_value="json:${VTB_ROOT}/tau3_val_test_split.json",
                ):
                    self.assertEqual(
                        config.validation_source(TAU3),
                        f"json:{root}/tau3_val_test_split.json",
                    )
        finally:
            config._validation_source_cache.clear()
            config._validation_source_cache.update(original_cache)

    def test_service_routes_benchmark_json_source_for_tau3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_split(Path(tmp), {"val": [VAL_1, VAL_2], "test": [TEST_1, TEST_2]})
            source = valsplit.parse_val_source(f"json:{path}")
            service = object.__new__(GradingService)
            service.val_source = valsplit.parse_val_source("split:30%")
            service._benchmark_val_sources = {
                TAU3: source,
                "swe-bench__swe-bench-verified": source,
            }
            previous = config.VAL_SOURCE_RAW
            try:
                config.VAL_SOURCE_RAW = ""
                self.assertEqual(service._validation_plan(TAU3), (TAU3, None, [VAL_1, VAL_2]))
                self.assertEqual(service._heldout_ids(SimpleNamespace(task=TAU3)), [TEST_1, TEST_2])
                self.assertEqual(service._resolve_sample_ids(TAU3, [VAL_2], known_ids=[VAL_1, VAL_2]), [VAL_2])
                with self.assertRaisesRegex(ValueError, "only valid"):
                    service._validation_plan("swe-bench__swe-bench-verified")
            finally:
                config.VAL_SOURCE_RAW = previous


if __name__ == "__main__":
    unittest.main()
