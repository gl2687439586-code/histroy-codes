"""Offline checks for repeat isolation, fail-fast behavior and honest aggregation."""
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import repeat500
from analyze_repeats import analyze
from prepare_repeat500 import join_official, load_queries


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class RepeatTests(unittest.TestCase):
    def test_uploaded500_exact_membership(self):
        cfg = repeat500.settings()
        rows = repeat500.validate_inputs(cfg)
        self.assertEqual(len(rows), 500)
        self.assertEqual(rows[0]["sample_id"], "5ae7a8175542993210983ed8")
        self.assertFalse(any("answer" in row or "supporting_facts" in row or "context" in row for row in rows))

    def test_independent_dirs_cache_and_clean_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = repeat500.settings()
            cfg["batch_dir"] = Path(tmp)
            caches = []
            for number in (1, 2, 3):
                with patch.dict(os.environ, {"RUN_DIR": "/old/run", "LIMIT": "5", "EXPOSURES": "50,1500", "TOP_K": "10", "RETRIEVAL_CACHE": "/old/cache"}):
                    env = repeat500.infrastructure.env_for_run(repeat500.run_cfg(cfg, number))
                self.assertEqual(env["TOP_K"], "5")
                self.assertEqual(env["LIMIT"], "0")
                self.assertEqual(env["EXPOSURES"], "")
                self.assertEqual(env["GPU_IDS"], "2,3")
                self.assertEqual(env["SERVER_MAX_TOTAL_TOKENS"], "24497")
                self.assertEqual(env["CONTEXT_LENGTH"], "40960")
                self.assertEqual(env["REPAIR_MAX_NEW_TOKENS"], "0")
                caches.append(env["RETRIEVAL_CACHE"])
                self.assertIn(f"repeat_{number}", env["RUN_DIR"])
            self.assertEqual(len(set(caches)), 3)

    def test_input_join_uses_original_outer_strip_and_rejects_content_edit(self):
        original = [{"_id": "a", "question": " Q? ", "answer": "A", "context": [["T", ["Evidence"]]], "supporting_facts": [["T", 0]]}]
        questions, _ = join_official([{"query_id": "a", "query": "Q?"}], original)
        self.assertEqual(questions[0]["question"], "Q?")
        with self.assertRaises(RuntimeError):
            join_official([{"query_id": "a", "query": "Different?"}], original)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.jsonl"
            dump(path, [{"query_id": "a", "query": "Q?"}, {"query_id": "a", "query": "Q?"}])
            with self.assertRaises(RuntimeError):
                load_queries(path, expected=2)

    def test_changed_protocol_rejects_resume(self):
        with self.assertRaisesRegex(RuntimeError, "protocol"):
            repeat500.compare_plan({"protocol": {"temperature": 0}}, {"protocol": {"temperature": 0.6}})

    def test_missing_services_start_only_gpu23_with_new_session_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = repeat500.settings()
            cfg["service_dir"] = Path(tmp)
            cfg["repo"] = Path(tmp) / "SmartSearch"
            source = cfg["repo"] / "src/flashrag"
            source.mkdir(parents=True)
            (source / "__init__.py").write_text("", encoding="utf-8")
            free = type("Result", (), {"stdout": "24000"})()
            with patch("full7405.sys.platform", "linux"), patch("full7405.shutil.which", return_value="tmux"), \
                 patch("full7405.listening", return_value=False), patch("full7405.command", return_value=free), \
                 patch("full7405.service_env", return_value={}), patch("full7405.tmux_start") as starts, \
                 patch("full7405.wait_for_health"), patch("full7405.model_probe", return_value={"passed": True}), \
                 patch("full7405.retriever_probe", return_value={"passed": True}):
                repeat500.infrastructure.services(cfg, create=True)
            self.assertEqual([call.args[0] for call in starts.call_args_list],
                             ["hotpot_repeat500_model_g2", "hotpot_repeat500_model_g3", "hotpot_repeat500_retriever"])
            self.assertEqual(starts.call_args_list[2].args[4]["PYTHONPATH"], str(cfg["repo"] / "src"))
            self.assertNotIn("PYTHONPATH", starts.call_args_list[0].args[4])
            self.assertNotIn("PYTHONPATH", starts.call_args_list[1].args[4])

    def test_worker_runs_sequentially_then_offline_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = repeat500.settings()
            cfg["batch_dir"] = Path(tmp)
            for number in (1, 2, 3):
                (Path(tmp) / f"repeat_{number}").mkdir()
            with patch("repeat500.verify_frozen"), patch("repeat500.infrastructure.services", return_value={}), \
                 patch("repeat500.infrastructure.command") as offline, patch("repeat500.subprocess.call", return_value=0) as calls:
                self.assertEqual(repeat500.worker(cfg, False), 0)
            self.assertEqual(calls.call_count, 6)
            self.assertEqual([Path(call.kwargs["env"]["RUN_DIR"]).name for call in calls.call_args_list],
                             ["repeat_1", "repeat_1", "repeat_2", "repeat_2", "repeat_3", "repeat_3"])
            self.assertEqual(offline.call_count, 1)
            self.assertEqual(json.loads((Path(tmp) / "BATCH_STATUS.json").read_text())["stage"], "complete")

    def test_pipeline_failure_preserved_and_later_runs_not_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = repeat500.settings()
            cfg["batch_dir"] = Path(tmp)
            with patch("repeat500.verify_frozen"), patch("repeat500.infrastructure.services", return_value={}), \
                 patch("repeat500.subprocess.call", return_value=1) as calls:
                with self.assertRaisesRegex(RuntimeError, "later repeats"):
                    repeat500.worker(cfg, False)
            self.assertEqual(calls.call_count, 1)
            self.assertEqual(json.loads((Path(tmp) / "repeat_1/PIPELINE_STATUS.json").read_text())["stage"], "failed")
            self.assertFalse((Path(tmp) / "repeat_2").exists())

    def test_resume_skips_completed_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = repeat500.settings()
            cfg["batch_dir"] = Path(tmp)
            first = Path(tmp) / "repeat_1"
            first.mkdir()
            (first / "PIPELINE_STATUS.json").write_text('{"stage":"complete"}')
            with patch("repeat500.verify_frozen"), patch("repeat500.infrastructure.services", return_value={}), \
                 patch("repeat500.infrastructure.command"), patch("repeat500.subprocess.call", return_value=0) as calls:
                repeat500.worker(cfg, True)
            self.assertEqual(calls.call_count, 4)
            self.assertTrue(all(call.kwargs["env"]["RETRY_FAILURES"] == "1" for call in calls.call_args_list))

    def test_same500_mean_not_best_of_three_and_common_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            for number in (1, 2, 3):
                run = batch / f"repeat_{number}"
                run.mkdir()
                (run / "PIPELINE_STATUS.json").write_text('{"stage":"complete"}')
                dump(run / "dataset/questions.no_labels.jsonl", [{"sample_id": "a"}, {"sample_id": "b"}])
                dump(run / "roots/roots.merged.jsonl.gz", [{"sample_id": sid, "query": f"q{number}", "root_full_context_sha256": f"e{number}"} for sid in ("a", "b")])
                rows = []
                for sid in ("a", "b"):
                    for group in ("100", "200", "FULL"):
                        if number == 2 and sid == "b" and group == "200":
                            continue
                        value = 1 if number == 1 else 0
                        rows.append({"sample_id": sid, "condition_group": group, "answer_em": value,
                                     "answer_f1": value, "answer_acc": value, "standalone_llm_input_tokens": 90,
                                     "standalone_llm_output_tokens": 10, "visible_evidence_tokens_total": 100,
                                     "search_calls": 1})
                dump(run / "analysis/scored_records.jsonl.gz", rows)
            result = analyze(batch, expected=2)
            self.assertEqual(result["common_conditions"], ["100", "FULL"])
            self.assertAlmostEqual(result["mean_and_sample_std"]["FULL"]["em"]["mean"], 1/3)
            self.assertEqual(len(result["crossrun_oracle_transfer"]), 18)


if __name__ == "__main__":
    unittest.main(verbosity=2)
