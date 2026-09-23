import argparse
import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "publish_prepared_dataset", Path(__file__).parents[1] / "scripts/publish_prepared_dataset.py"
)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


class FakeGitHub:
    def __init__(self):
        self.current = "a" * 40
        self.calls = []
        self.wrong_blob_hash = False

    def head(self, repository, branch):
        return self.current

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "create_blob":
            data = (base64.b64decode(arguments["content"]) if arguments["encoding"] == "base64"
                    else arguments["content"].encode())
            return {"sha": "bad" if self.wrong_blob_hash else publisher.git_blob_sha(data)}
        if name == "fetch":
            return {"tree": {"sha": "b" * 40}}
        if name == "create_tree":
            return {"sha": "c" * 40}
        if name == "create_commit":
            return {"sha": "d" * 40}
        if name == "update_ref":
            self.current = arguments["sha"]
            return {"object": {"sha": self.current}}
        raise AssertionError(name)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "shard.gz").write_bytes(b"\x1f\x8b\xffsample")
        self.file_list = self.root / "files.json"
        self.write_list([{"path": "datasets/example/shard.gz", "local_path": "shard.gz"}])
        self.args = argparse.Namespace(
            repository="owner/repo", branch="main", expected_head="a" * 40,
            checkpoint=self.root / "checkpoint.json", message="Publish dataset", workers=3,
            stage_only=False,
        )

    def write_list(self, rows):
        self.file_list.write_text(json.dumps(rows))

    def test_rejects_path_traversal_duplicate_and_wrong_digest(self):
        for target in ("../data", "/data", ".git/config", "x/../data", "x//data"):
            self.write_list([{"path": target, "local_path": "shard.gz"}])
            with self.assertRaises(ValueError):
                publisher.load_files(self.file_list)
        row = {"path": "data", "local_path": "shard.gz"}
        self.write_list([row, row])
        with self.assertRaises(ValueError):
            publisher.load_files(self.file_list)
        self.write_list([{**row, "sha256": "incorrect"}])
        with self.assertRaises(ValueError):
            publisher.load_files(self.file_list)

    def test_understands_actual_hosted_fetch_envelope(self):
        payload = {"object": {"sha": "a" * 40}}
        result = {"content": [{"type": "text", "text": json.dumps({
            "structuredContent": {"content": json.dumps(payload)}
        })}]}
        self.assertEqual(publisher.unwrap(result), payload)

    def test_binary_publish_checks_hash_and_updates_without_force(self):
        files = publisher.load_files(self.file_list)
        github = FakeGitHub()
        self.assertEqual(publisher.publish(files, self.args, github), "d" * 40)
        blob = next(args for name, args in github.calls if name == "create_blob")
        self.assertEqual(blob["encoding"], "base64")
        self.assertEqual(base64.b64decode(blob["content"]), (self.root / "shard.gz").read_bytes())
        update = next(args for name, args in github.calls if name == "update_ref")
        self.assertFalse(update["force"])
        tree = next(args for name, args in github.calls if name == "create_tree")
        self.assertEqual(tree["base_tree_sha"], "b" * 40)
        self.assertEqual(tree["tree_elements"][0]["sha"], files[0]["git_sha"])
        call_count = len(github.calls)
        self.assertEqual(publisher.publish(files, self.args, github), "d" * 40)
        self.assertEqual(len(github.calls), call_count)

    def test_rejects_wrong_remote_blob_hash_before_tree_or_commit(self):
        github = FakeGitHub()
        github.wrong_blob_hash = True
        with self.assertRaisesRegex(ValueError, "hash differs"):
            publisher.publish(publisher.load_files(self.file_list), self.args, github)
        self.assertEqual([name for name, _ in github.calls], ["create_blob"])

    def test_rejects_stale_branch_without_writes(self):
        github = FakeGitHub()
        github.current = "e" * 40
        with self.assertRaisesRegex(ValueError, "Branch changed"):
            publisher.publish(publisher.load_files(self.file_list), self.args, github)
        self.assertEqual(github.calls, [])

    def test_stage_only_uploads_blobs_then_normal_publish_resumes_without_reupload(self):
        github = FakeGitHub()
        files = publisher.load_files(self.file_list)
        self.args.stage_only = True
        self.assertIsNone(publisher.publish(files, self.args, github))
        self.assertEqual([name for name, _ in github.calls], ["create_blob"])
        self.assertEqual(github.current, "a" * 40)
        checkpoint = json.loads(self.args.checkpoint.read_text())
        self.assertEqual(checkpoint["status"], "blobs_staged")
        self.assertNotIn("tree", checkpoint)
        self.assertNotIn("commit", checkpoint)
        self.args.stage_only = False
        self.assertEqual(publisher.publish(files, self.args, github), "d" * 40)
        self.assertEqual([name for name, _ in github.calls].count("create_blob"), 1)
        self.assertEqual(github.current, "d" * 40)


if __name__ == "__main__":
    unittest.main()
