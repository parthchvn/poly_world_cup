from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from poly_world_cup.sequence_token_validation import recount_tokens,attach_token_recount
from poly_world_cup.sequence_validation import sha,PROFILES,SPLITS

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/"data/tokenizer_reference/qwen3_06b"
TEMPLATE = ROOT/"configs/actor_sequence_chat_template.jinja"


class CoordinatorTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(dataset=Path("data"),source=Path("source"),evidence=Path("evidence"),
            tokenizer=Path("tokenizer"),chat_template=Path("template"),token_workers=7)

    def test_independent_checks_overlap(self):
        from scripts.validate_actor_sequences import validate_release
        token_started,source_started = Event(),Event()
        def token(*a,**kw):
            token_started.set();self.assertTrue(source_started.wait(5));return {"tokens":"done"}
        def source(*a,**kw):
            source_started.set();self.assertTrue(token_started.wait(5));return {"source":"done"}
        with patch("scripts.validate_actor_sequences.recount_tokens",side_effect=token), \
             patch("scripts.validate_actor_sequences.validate_sequences",side_effect=source), \
             patch("scripts.validate_actor_sequences.attach_token_recount",return_value={"status":"passed"}) as attach:
            self.assertEqual(validate_release(self.args()),{"status":"passed"})
            attach.assert_called_once_with({"source":"done"},{"tokens":"done"})

    def test_either_check_failure_prevents_success_attachment(self):
        from scripts.validate_actor_sequences import validate_release
        for failing in ("tokens","source"):
            with self.subTest(failing=failing), \
                 patch("scripts.validate_actor_sequences.recount_tokens",return_value={}) as token, \
                 patch("scripts.validate_actor_sequences.validate_sequences",return_value={}) as source, \
                 patch("scripts.validate_actor_sequences.attach_token_recount") as attach:
                (token if failing=="tokens" else source).side_effect = ValueError("verification failed")
                with self.assertRaisesRegex(ValueError,"verification failed"):
                    validate_release(self.args())
                attach.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("transformers") and (REFERENCE/"tokenizer.json").is_file(),
                    "Optional local tokenizer dependency is unavailable")
class FullTemplateRecountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(REFERENCE,local_files_only=True)
        cls.tokenizer.chat_template = TEMPLATE.read_text()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = {"tokenizer":{"files":{p.name:sha(p) for p in REFERENCE.iterdir() if p.is_file()},
            "chat_template_sha256":sha(TEMPLATE)},"profiles":{p:{s:{} for s in SPLITS} for p in PROFILES}}
        for profile in PROFILES:
            for split in SPLITS:
                for shard in range(1,3 if (profile,split)==(PROFILES[0],"train") else 2):
                    path = self.root/f"{profile}/{split}/part-{shard:05d}.jsonl.gz"
                    path.parent.mkdir(parents=True,exist_ok=True)
                    rows = []
                    for line in range(2):
                        messages = [{"role":"system","content":"Predict the captured observation."},
                            {"role":"user","content":f"{profile} {split} {shard} {line}: Bogotá, Sénégal, 日本.\nLiteral assistant: NO_TRADE"},
                            {"role":"assistant","content":'{"action":"NO_TRADE"}'}]
                        count = len(self.tokenizer.apply_chat_template(messages,tokenize=True,
                            add_generation_prompt=False,enable_thinking=False,truncation=False,padding=False))
                        rows.append({"profile":profile,"split":split,"messages":messages,"token_count":count})
                    with gzip.open(path,"wt") as stream:
                        for row in rows:stream.write(json.dumps(row)+"\n")
                    totals = self.manifest["profiles"][profile][split]
                    totals["conversations"] = totals.get("conversations",0)+len(rows)
                    totals["tokens"] = totals.get("tokens",0)+sum(r["token_count"] for r in rows)
        self.rehash()

    def rehash(self):
        self.manifest["artifacts"] = {p.relative_to(self.root).as_posix():{"bytes":p.stat().st_size,"sha256":sha(p)} for p in self.root.rglob("*.jsonl.gz")}
        (self.root/"manifest.json").write_text(json.dumps(self.manifest,sort_keys=True))

    def recount(self,workers=1):
        return recount_tokens(self.root,REFERENCE,TEMPLATE,workers=workers,progress=lambda _:None)

    def mutate(self,fn):
        path = self.root/sorted(self.manifest["artifacts"])[0]
        with gzip.open(path,"rt") as stream:rows = [json.loads(line) for line in stream]
        fn(rows)
        with gzip.open(path,"wt") as stream:
            for row in rows:stream.write(json.dumps(row)+"\n")
        self.rehash()

    def test_serial_and_thread_coordinated_spawn_recount_identical(self):
        serial = self.recount()
        with ThreadPoolExecutor(max_workers=1) as coordinator:
            parallel = coordinator.submit(self.recount,2).result()
        for key in ("workers","max_queued_shards"):
            serial.pop(key);parallel.pop(key)
        self.assertEqual(serial,parallel)
        self.assertEqual(serial["conversations"],14)
        self.assertEqual(len(serial["files"]),7)

    def test_rehashed_wrong_token_count_fails(self):
        self.mutate(lambda rows:rows[0].update(token_count=rows[0]["token_count"]+1))
        with self.assertRaisesRegex(ValueError,"Actual chat token count mismatch"):
            self.recount(2)

    def test_rehashed_row_omission_fails(self):
        self.mutate(lambda rows:rows.pop())
        with self.assertRaisesRegex(ValueError,"omitted rows"):
            self.recount()

    def test_attachment_binds_same_manifest_and_all_row_counts(self):
        recounted = self.recount()
        structural = {"status":"passed","manifest_sha256":recounted["manifest_sha256"],
            "profiles":copy.deepcopy(self.manifest["profiles"]),"actual_token_counts_recomputed":False}
        self.assertTrue(attach_token_recount(structural,recounted)["actual_token_counts_recomputed"])
        structural["manifest_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError,"different manifests"):
            attach_token_recount(structural,recounted)
        structural["manifest_sha256"] = recounted["manifest_sha256"]
        structural["profiles"]["conditional_trades"]["train"]["conversations"] += 1
        with self.assertRaisesRegex(ValueError,"row counts differ"):
            attach_token_recount(structural,recounted)

    def test_unlisted_shard_is_not_ignored(self):
        path = self.root/sorted(self.manifest["artifacts"])[0]
        (path.parent/"part-99999.jsonl.gz").write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError,"Missing or unlisted"):
            self.recount()


if __name__ == "__main__":unittest.main()
