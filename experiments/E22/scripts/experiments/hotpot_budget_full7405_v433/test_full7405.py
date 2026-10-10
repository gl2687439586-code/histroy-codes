#!/usr/bin/env python3
"""Offline operational tests: no GPU/network/model invocation."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from full7405 import (PROTOCOL, assert_resume, env_for_run, package_hashes,
                      read_dataset, recursive_pools, require_empty_run, settings, snapshot, model_probe)
from summarize_full_tokens import aggregate, oracle_accounting


class Full7405Tests(unittest.TestCase):
    def test_protocol_and_clean_environment(self):
        with patch.dict(os.environ, {"TOP_K": "10", "MAX_SEARCHES": "9", "LIMIT": "500", "EXPOSURES": "50,1500", "RETRIEVAL_CACHE": "/old/cache.sqlite"}):
            cfg = settings()
            env = env_for_run(cfg)
        self.assertEqual(env["TOP_K"], "5")
        self.assertEqual(env["LIMIT"], "0")
        self.assertEqual(env["EXPOSURES"], "")
        self.assertEqual(env["SERVER_MAX_TOTAL_TOKENS"], "24497")
        self.assertEqual(env["GPU_IDS"], "2,3")
        self.assertEqual(env["CONTEXT_LENGTH"], "40960")
        self.assertTrue(env["RETRIEVAL_CACHE"].startswith(str(cfg["run_dir"])))
        self.assertEqual(PROTOCOL["questions"], 7405)

    def test_reject_subset_duplicates_and_label_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            q, l = Path(tmp) / "q.jsonl", Path(tmp) / "l.jsonl"
            rows = [{"sample_id": str(i), "sample_index": i, "question": f"Q{i}"} for i in range(3)]
            q.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
            l.write_text("\n".join(json.dumps({"sample_id": str(i)}) for i in range(3)), encoding="utf-8")
            self.assertEqual(len(read_dataset(q, l, 3)), 3)
            with self.assertRaises(RuntimeError):
                read_dataset(q, l)  # full-run guard rejects any subset
            rows[0]["answer"] = "leaked"
            q.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                read_dataset(q, l, 3)
            del rows[0]["answer"]
            rows[1]["sample_id"] = rows[0]["sample_id"]
            q.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                read_dataset(q, l, 3)

    def test_new_run_does_not_reuse_old_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            require_empty_run(run)
            (run / "old_record.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                require_empty_run(run)
            self.assertTrue((run / "old_record.json").exists())
            with self.assertRaises(RuntimeError):
                assert_resume({"run_dir": run}, {})

    def test_read_only_empty_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "absent"
            cfg = {"run_dir": run, "session": "test"}
            with patch("full7405.shutil.which", return_value=None):
                result = snapshot(cfg)
            self.assertEqual(result["roots"], 0)
            self.assertIsNone(result["expected_trajectories"])
            self.assertFalse(run.exists())

    def test_token_no_double_counting_and_strict_fallback(self):
        row = {"sample_id": "x", "condition_id": "b00100", "condition_group": "100",
               "search_calls": 2, "visible_evidence_tokens_total": 200,
               "standalone_llm_input_tokens": 500, "standalone_llm_output_tokens": 80,
               "continuation_llm_input_tokens": 480, "continuation_llm_output_tokens": 70,
               "answer_em": 1, "answer_f1": 1, "answer_acc": 1}
        tables, _ = aggregate([row], 1)
        self.assertEqual(tables["budget"][0]["mean_llm_total_tokens_per_question"], 580)
        with self.assertRaises(RuntimeError):
            aggregate([row, row], 1)
        with self.assertRaises(RuntimeError):
            aggregate([dict(row, local_fallback=True)], 1)

    def test_pool_and_package_integrity(self):
        self.assertEqual(list(recursive_pools({"internal_states": [{"max_total_num_tokens": 25000}, {"max_total_num_tokens": 24497}]})), [25000, 24497])
        self.assertEqual(len(package_hashes()["prompts.py"]), 64)

    def test_service_requires_real_generation_and_sufficient_pool(self):
        cfg = {"model_path": Path("/models/Qwen3-8B")}
        opts = {"--context-length": "40960", "--dtype": "bfloat16", "--tp": "1", "--dp": "1",
                "--attention-backend": "triton", "--sampling-backend": "pytorch",
                "--chunked-prefill-size": "4096", "--max-running-requests": "2",
                "--disable-cuda-graph": True, "--disable-overlap": True,
                "--trust-remote-code": True, "--mem-fraction-static": "0.84"}
        process = (123, ["python", "-m", "sglang.launch_server"], opts, {"CUDA_VISIBLE_DEVICES": "2"})
        identity = {"model_path": str(cfg["model_path"])}
        with patch("full7405.process_for_port", return_value=process):
            with patch("full7405.http", side_effect=[None, identity, {"max_total_num_tokens": 24497}, {"text": "OK"}]) as mock:
                self.assertTrue(model_probe(cfg, 9103, 2)["passed"])
                self.assertEqual(mock.call_count, 4)
            with patch("full7405.http", side_effect=[None, identity, {"max_total_num_tokens": 9905}]) as mock:
                with self.assertRaisesRegex(RuntimeError, "observed pool"):
                    model_probe(cfg, 9103, 2)
                self.assertEqual(mock.call_count, 3)
            with patch("full7405.http", side_effect=[None, identity, {"max_total_num_tokens": 24497}, {"text": ""}]):
                with self.assertRaisesRegex(RuntimeError, "no text"):
                    model_probe(cfg, 9103, 2)

    def test_oracle_reports_selected_costs_not_sweep_costs(self):
        base = {"sample_id": "x", "answer_f1": 1.0, "answer_em": 1, "answer_acc": 1,
                "search_calls": 2, "visible_evidence_tokens_total": 200,
                "standalone_llm_input_tokens": 500, "standalone_llm_output_tokens": 100}
        rows = [dict(base, condition_group="100"),
                dict(base, condition_group="200", standalone_llm_input_tokens=400, standalone_llm_output_tokens=300),
                dict(base, condition_group="FULL", standalone_llm_input_tokens=200, standalone_llm_output_tokens=200)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.jsonl"
            path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
            means, selected = oracle_accounting(path, {"100", "200", "FULL"})
        self.assertEqual(selected[0]["selected_condition"], "200")
        self.assertEqual(selected[1]["selected_condition"], "FULL")
        self.assertEqual(means[1]["mean_llm_total_tokens_per_question"], 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
