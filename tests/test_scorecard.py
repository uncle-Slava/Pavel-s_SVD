"""Black-box SVD scorecard: verdicts, truncation, values-only and failure isolation."""
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from svdbench.laboratory import MatrixCase
from algorithm_laboratory.scorecard import evaluate, compare, summary, save, DEFAULT_SUITE

DEVICE=os.environ.get('SVD_TEST_DEVICE','cpu')
torch.set_num_threads(1)
SMALL=[dict(name='tall',m=20,n=12,family='gaussian'),
       dict(name='complex_wide',m=10,n=16,family='gaussian',dtype='complex128'),
       dict(name='low_rank',m=24,n=18,family='low_rank',rank=3,tail=1e-6,output_rank=3)]


class Scorecard(unittest.TestCase):
    def run_small(self,fn,**kw):
        return evaluate(fn,SMALL,device=DEVICE,repeat=2,warmup=0,verbose=False,**kw)

    def test_full_svd_passes_every_case_including_truncated(self):
        rows=self.run_small(lambda a,rank:torch.linalg.svd(a,full_matrices=False))
        self.assertTrue(all(r['ok'] and r['status']=='done' for r in rows),rows)
        self.assertEqual([r['rank'] for r in rows],[12,10,3])
        self.assertLess(rows[2]['residual_over_optimal_tail'],1.01)

    def test_numpy_output_builtins_and_suite(self):
        rows=compare({'numpy':'numpy','jacobi':'jacobi','mine':'mine'},SMALL[:1],device=DEVICE,
                     repeat=1,warmup=0,verbose=False)
        self.assertTrue(all(r['ok'] for r in rows),rows)
        self.assertEqual(len(DEFAULT_SUITE),8)

    def test_wrong_answers_fail_and_errors_are_isolated(self):
        calls=[]
        def bad(a,rank):
            calls.append(rank)
            if len(calls)==1:raise RuntimeError('boom')
            u,s,vh=torch.linalg.svd(a,full_matrices=False)
            return u,s*1.01,vh
        rows=self.run_small(bad)
        self.assertEqual(rows[0]['status'],'error');self.assertIn('boom',rows[0]['error'])
        self.assertFalse(any(r['ok'] for r in rows))
        self.assertEqual(rows[1]['status'],'done')
        self.assertEqual(summary(rows)[0]['failed_cases'],['tall','complex_wide','low_rank'])

    def test_values_only_and_custom_cases(self):
        case=MatrixCase.from_spectrum([4,2,1,0],m=6,n=5)
        rows=evaluate(lambda a,rank:torch.linalg.svdvals(a),[case],device=DEVICE,repeat=1,warmup=0,verbose=False)
        self.assertTrue(rows[0]['ok'],rows);self.assertNotIn('reconstruction_relative_fro',rows[0])
        rows=self.run_small(lambda a,rank:torch.linalg.svd(a,full_matrices=False)[1][:rank-1])
        self.assertTrue(all(r['status']=='error' for r in rows))

    def test_save_csv_and_json(self):
        rows=self.run_small('torch')
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn('reconstruction_relative_fro',save(rows,Path(tmp)/'r.csv').read_text())
            self.assertIn('"ok": true',save(rows,Path(tmp)/'r.json').read_text())


if __name__=='__main__':unittest.main()
