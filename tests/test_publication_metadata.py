from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("publication_sanitizer", REPO / "scripts/prepare/python/sanitize_publication.py")
assert _spec and _spec.loader
sanitizer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sanitizer)


class PublicationMetadataTest(unittest.TestCase):
    def test_personal_paths_and_login_prompts_are_redacted(self):
        # Construct synthetic identities so the test itself contains no personal paths.
        before = "/home/" + "developer/project /mnt/" + "colleague/run /Users/" + "researcher/code"
        after = sanitizer.sanitize_text(before)
        self.assertEqual(after, "/home/user/project /mnt/runtime/run /Users/user/code")
        self.assertEqual(sanitizer.sanitize_text("developer" + "@workstation:~/repo$ "), "user@host:~/repo$ ")
        self.assertEqual(sanitizer.sanitize_text("root" + "@container-id:/app# "), "root@host:/app# ")
        self.assertEqual(sanitizer.sanitize_text(after), after)

    def test_scores_ids_and_public_attribution_are_preserved(self):
        before = {"trial_uri": "file:///mnt/" + "operator/run/sample", "trials_dir": "/mnt/" + "operator/run",
                  "task_id": {"name": "task-1", "ref": "sha256:abc"}, "task_checksum": "digest",
                  "verifier_result": {"rewards": {"reward": 0.5}}, "started_at": "2026-01-01",
                  "source": "https://github.com/organization/project", "author": "Paper Author",
                  "email": "customer@example.com", "hostname": "private-machine"}
        after = sanitizer.sanitize_document(before, artifact_metadata=True)
        self.assertEqual(after["trial_uri"], ".")
        self.assertEqual(after["trials_dir"], "..")
        self.assertEqual(after["hostname"], "redacted")
        for key in ("task_id", "task_checksum", "verifier_result", "started_at", "source", "author", "email"):
            self.assertEqual(before[key], after[key])

    def test_container_paths_and_task_file_uris_are_unchanged(self):
        text = "/home/agent/task /root/slime /workspace/task file:///app/secrets/example.txt"
        self.assertEqual(sanitizer.sanitize_text(text), text)

    def test_json_cleanup_retains_semantic_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(json.dumps({"trial_uri": "file:///mnt/" + "operator/run", "score": 0.0}))
            _, result = sanitizer.cleaned_content(path, Path("benchmarks/example/baseline/samples/sample/result.json"))
            self.assertEqual(json.loads(result), {"trial_uri": ".", "score": 0.0})

    def test_published_tree_is_sanitized(self):
        if (REPO / ".git").exists():
            self.assertEqual(sanitizer.process(REPO), [])
        else:
            # Clean archives have no Git index and no ignored runtime directories.
            for path in (REPO / "benchmarks").rglob("*"):
                if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                    before, after = sanitizer.cleaned_content(path, path.relative_to(REPO))
                    self.assertEqual(before, after, str(path.relative_to(REPO)))


if __name__ == "__main__":
    unittest.main()
