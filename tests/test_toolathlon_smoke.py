from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "scripts/run_smoke/python/verify_smoke_toolathlon.py"
SPEC = importlib.util.spec_from_file_location("verify_smoke_toolathlon", PATH)
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class ToolathlonSmokeVerifierTest(unittest.TestCase):
    def _tree(self, root: Path, samples: int = 1) -> tuple[Path, Path]:
        state, results = root / "state", root / "results"
        state.mkdir(); results.mkdir()
        val, final = "val123", "final456"
        (state / "service.log").write_text(
            f"[submit] queued job={val} role=agent task=eigent-ai__toolathlon_gym eval_task=eigent-ai__toolathlon_gym mode=val limit=-1 position=1 samples={samples}\n"
            f"[worker] finish job={val} state=done score=0.0\n"
            f"[submit] queued job={final} role=official task=eigent-ai__toolathlon_gym eval_task=hkust-nlp__toolathlon_verified mode=final limit=1 position=1\n"
            f"[worker] finish job={final} state=done score=0.0\n",
            encoding="utf-8",
        )
        metrics = {"accuracy": 0.0, "graded": 1, "attempted": 1, "ungraded": 0}
        for mode, job in (("val", val), ("final", final)):
            archive = state / "archive/run" / f"{mode}_{job}"
            archive.mkdir(parents=True)
            (archive / "metrics_p1_a1.json").write_text(json.dumps(metrics), encoding="utf-8")
        (results / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return state, results

    def test_accepts_file_split_one_sample_without_backend_returncode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, results = self._tree(Path(tmp))
            self.assertEqual(verifier.verify(state, results), {"agent": 0.0, "official": 0.0})

    def test_rejects_more_than_one_validation_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, results = self._tree(Path(tmp), samples=12)
            with self.assertRaises(ValueError):
                verifier.verify(state, results)


if __name__ == "__main__":
    unittest.main()
