import unittest
import numpy as np
import torch
from svdbench.broad import factory, grade
from svdbench.cli import to_host
from svdbench.matrices import make_matrix
from svdbench.metrics import reference, validate


class BroadBenchmark(unittest.TestCase):
    def test_library_controls_real_and_complex(self):
        torch.set_num_threads(1)
        for dtype in ('float64','complex128'):
            for method in ('randomized_library','sparse_randomized_library','gram_library','qr_library'):
                with self.subTest(dtype=dtype,method=method):
                    a=make_matrix(dict(m=20,n=12,family='sparse',density=.5,dtype=dtype,seed=23))
                    spec=dict(method=method,rank=4,oversample=8,iterations=2,seed=1)
                    result=to_host(factory(torch.from_numpy(a),spec)())
                    self.assertEqual(result[1].shape,(4,))
                    row=dict(status='ok',metrics=validate(a,result,reference(a)))
                    cfg=dict(case=dict(m=20,n=12,dtype=dtype),target_rank=4)
                    self.assertTrue(grade(row,cfg)['passed'],row)

    def test_partial_quality_gate_rejects_inaccurate_spectrum(self):
        w=dict(orthogonality_u_fro_over_sqrt_cols=0,orthogonality_v_fro_over_sqrt_rows=0,
               residual_over_optimal_tail=1.001,resolved_spectrum_max_relative_error=.02,
               singular_values_nonnegative_sorted=True)
        cfg=dict(case=dict(m=100,n=50,dtype='float32'),target_rank=4)
        self.assertFalse(grade(dict(status='ok',metrics={'worst':w}),cfg)['passed'])
        w['resolved_spectrum_max_relative_error']=.005
        self.assertTrue(grade(dict(status='ok',metrics={'worst':w}),cfg)['passed'])


if __name__=='__main__':unittest.main()
