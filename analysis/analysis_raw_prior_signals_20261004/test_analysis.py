"""Tests for feature leakage boundaries, folds and output arithmetic."""
import json
import unittest
import numpy as np
import analyze as a


class AuditTests(unittest.TestCase):
    def test_math_and_model(self):
        a.self_test()

    def test_features_ignore_gold_and_future(self):
        root=next(a.rows(a.RAW/'input/old500/roots/roots.merged.jsonl.gz'))
        before=a.root_features(root)
        altered=dict(root)
        altered.update(gold_answers=['LEAK'],supporting_facts=['LEAK'],type='LEAK',level='LEAK',
                       final_answer='LEAK',search_calls=999,answer_em=999,decision_events=[{'LEAK':True}])
        self.assertEqual(before,a.root_features(altered))

    def test_folds_cover_ids_once(self):
        for seed in a.SEEDS:
            folds=np.array_split(np.random.default_rng(seed).permutation(500),5)
            self.assertEqual(sorted(np.concatenate(folds).tolist()),list(range(500)))
            for test in folds:
                train=np.setdiff1d(np.arange(500),test)
                self.assertFalse(set(train)&set(test))

    def test_saved_results(self):
        r=json.loads((a.OUT/'CROSS_RUN_POLICIES.json').read_text(encoding='utf-8'))
        self.assertEqual(len(r),72)
        for p in r:
            self.assertEqual(sum(p['actions'].values()),500)
            m=p['means']
            self.assertAlmostEqual(m['total_tokens'],m['standalone_llm_input_tokens']+m['standalone_llm_output_tokens'])
            self.assertTrue(0<=m['answer_em']<=1)
        s=json.loads((a.OUT/'ORACLE_SETS.json').read_text(encoding='utf-8'))
        self.assertEqual(s['any_shared_correct_budget']+s['both_any_but_no_shared_correct_budget'],s['both_any_correct'])
        self.assertEqual(s['old_any_correct'],242)
        self.assertEqual(s['new_any_correct'],249)
        f=json.loads((a.OUT/'FIXED_BUDGETS.json').read_text(encoding='utf-8'))
        self.assertEqual(f['new500']['fixed']['FULL']['answer_em'],.382)
        self.assertAlmostEqual(f['new500']['fixed']['FULL']['total_tokens'],4504.178)


if __name__=='__main__':unittest.main()
