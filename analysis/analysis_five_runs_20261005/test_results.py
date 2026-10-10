"""Independent arithmetic checks on exported arrays and audit counters."""
import json,unittest
import numpy as np
import analyze as a

class ResultTests(unittest.TestCase):
    def test_source_alignment(self):
        ids,rr,roots,maps,Y,C,I,O,E,S,F,RC=a.load()
        self.assertEqual(Y.shape,(5,500,7,3))
        for r,name in enumerate(a.RUNS):
            raw={(x['sample_id'],x['budget']):x for x in a.rows(a.OUT/(name+'_compact.jsonl.gz'))}
            self.assertEqual(len(raw),len(rr[r]))
            for i,sid in enumerate(ids):
                for j,b in enumerate(a.B):
                    x=raw[sid,b]
                    self.assertEqual(C[r,i,j],x['total_tokens'])
                    np.testing.assert_allclose(Y[r,i,j],[x['scores'][k] for k in a.FIELDS],rtol=0,atol=1e-12)
        self.assertTrue((C==I+O).all())
        self.assertTrue((C>=RC[:,:,None]).all())
        self.assertTrue((Y[F.astype(bool)]==0).all())
    def test_oracle_and_frozen_transfer(self):
        z=np.load(a.OUT/'matched_arrays.npz');Y=z['Y'];C=z['C'];q=np.arange(500)
        for mi in range(3):
            for r in range(5):
                pick=a.rank_budgets(Y[r,:,:,mi],C[r])[:,0]
                np.testing.assert_equal(Y[r,q,pick,mi],Y[r,:,:,mi].max(1))
        # Independent Python min selection uses no held-out outcome/cost.
        tr=Y[:2,:,:,0].mean(0);tc=C[:2].mean(0)
        chosen=np.array([min(range(7),key=lambda b:(-tr[i,b],tc[i,b],b)) for i in q])
        np.testing.assert_equal(chosen,a.rank_budgets(tr,tc)[:,0])
        self.assertAlmostEqual(Y[2:][:,q,chosen,0].mean(),.404)
        self.assertAlmostEqual(C[2:][:,q,chosen].mean(),3322.220666666667)
    def test_sets_and_bootstrap_unit(self):
        z=np.load(a.OUT/'matched_arrays.npz');y=z['Y'][:,:,:,0]
        self.assertEqual(int(y.max((0,2)).sum()),277)
        self.assertEqual(int(y.max(2).min(0).sum()),203)
        self.assertEqual(int(y.min(0).max(1).sum()),181)
        self.assertEqual(int(y.min((0,2)).sum()),66)
        self.assertEqual(a.IDX.shape,(10000,500))
        self.assertTrue((a.IDX<500).all())
        d=json.loads((a.OUT/'set_transfer.json').read_text())['EM']
        self.assertAlmostEqual(d['old_ranked_top1']['mean_bundle_tokens']['mean'],3322.220666666667)
        self.assertLess(d['old_ranked_top1']['oracle_hit']['mean'],d['old_ranked_top2']['oracle_hit']['mean'])
    def test_no_silent_audit_failures(self):
        for r in a.RUNS:
            report=json.loads((a.OUT/(r+'_raw_audit.json')).read_text())
            for k in ['reference_tokenizer_input_tokens_mismatches','reference_tokenizer_output_tokens_mismatches',
                      'reference_evidence_not_exact_token_prefix','reference_evidence_token_count_mismatches',
                      'visible_not_even_nfc_original_text_prefix_rounds']:
                self.assertEqual(report['warnings'].get(k,0),0)

if __name__=='__main__':unittest.main()
