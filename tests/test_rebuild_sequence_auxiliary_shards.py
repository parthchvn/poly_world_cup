import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from poly_world_cup.sequence_validation import PROFILES,SPLITS,canonical
from poly_world_cup.sft import _JSONLines,_Shards
from scripts.rebuild_sequence_auxiliary_shards import rebuild_index


class IndexReconstructionTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)/"dataset"
        self.output=Path(temporary.name)/"repair"
        rows={}
        for profile in PROFILES:
            for split in SPLITS:
                path=self.root/profile/split/"part-00001.jsonl.gz"
                writer=_JSONLines(path)
                for actor in ("alice","bob"):
                    for chunk in (0,1):
                        row={"actor_id":actor,"profile":profile,"split":split,"chunk_index":chunk,
                            "sequence_id":f"{actor}/{split}/{profile}/{chunk}","messages":[{"role":"user","content":f"{actor} {chunk}"}]}
                        rows[actor,split,profile,chunk]=row
                        writer.write(row)
                writer.close()
        writer=_Shards(self.root/"actor_index",100000)
        for actor in ("alice","bob"):
            for split in ("train","validation","test"):
                for profile in ("conditional_trades","scheduled_windows"):
                    for chunk in (0,1):
                        row=rows[actor,split,profile,chunk]
                        entry={k:row[k] for k in ("sequence_id","actor_id","profile","split","chunk_index")}
                        entry.update(path=f"{profile}/{split}/part-00001.jsonl.gz",
                            line=(0 if actor=="alice" else 2)+chunk+1,sha256=hashlib.sha256(canonical(row).encode()).hexdigest())
                        writer.write(entry)
        writer.close()
        (self.root/"manifest.json").write_text(json.dumps({"profiles":{p:{s:{"conversations":4} for s in SPLITS} for p in PROFILES}}))

    def test_six_stream_merge_recreates_original_actor_split_profile_order(self):
        result=rebuild_index(self.root,self.output)
        self.assertEqual(result["rows"],24)
        self.assertEqual(result["files"][0]["old_status"],"intact_exact_match")
        self.assertEqual((self.root/"actor_index/part-00001.jsonl.gz").read_bytes(),
                         (self.output/"actor_index/part-00001.jsonl.gz").read_bytes())

    def test_truncated_index_is_reconstructed_with_every_original_compressed_byte_preserved(self):
        path=self.root/"actor_index/part-00001.jsonl.gz"
        original=path.read_bytes()
        path.write_bytes(original[:-20])
        result=rebuild_index(self.root,self.output)
        self.assertEqual(result["files"][0]["old_status"],"truncated_prefix_preserved")
        self.assertEqual((self.output/"actor_index/part-00001.jsonl.gz").read_bytes(),original)

    def test_conversation_change_cannot_silently_rewrite_intact_index(self):
        path=self.root/"conditional_trades/train/part-00001.jsonl.gz"
        import gzip
        with gzip.open(path,"rt") as stream:rows=[json.loads(line) for line in stream]
        path.unlink()
        rows[0]["messages"][0]["content"]="changed context"
        writer=_JSONLines(path)
        for row in rows:writer.write(row)
        writer.close()
        with self.assertRaisesRegex(ValueError,"differs from existing compressed prefix"):
            rebuild_index(self.root,self.output)

    def test_bounded_prefix_excludes_active_shards_and_checks_right_boundary(self):
        import gzip
        path=self.root/"actor_index/part-00001.jsonl.gz"
        with gzip.open(path,"rt") as stream:entries=[json.loads(line) for line in stream]
        path.unlink()
        writer=_Shards(self.root/"actor_index",6)
        for entry in entries:writer.write(entry)
        writer.close()
        for profile in PROFILES:
            for split in SPLITS:
                (self.root/profile/split/"part-00002.jsonl.gz").write_bytes(b"an active incomplete gzip stream")
        result=rebuild_index(self.root,self.output,last_shard=1,shard_size=6)
        self.assertEqual(result["rows"],6)
        self.assertTrue(result["right_boundary_matches"])
        self.assertEqual(result["right_boundary_sequence_id"],entries[6]["sequence_id"])
        self.assertEqual(result["files"][0]["old_status"],"intact_exact_match")

    def test_bounded_prefix_requires_a_closed_successor_record(self):
        for profile in PROFILES:
            for split in SPLITS:
                (self.root/profile/split/"part-00002.jsonl.gz").write_bytes(b"active")
        with self.assertRaisesRegex(ValueError,"do not cover the entire requested index prefix"):
            rebuild_index(self.root,self.output,last_shard=1,shard_size=24)


if __name__=="__main__":unittest.main()
