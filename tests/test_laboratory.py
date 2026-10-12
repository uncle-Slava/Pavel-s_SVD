"""Laboratory data integrity, instrumentation neutrality and numerical contracts."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from svdbench.laboratory import MatrixCase, Trace, NullTrace, Run, accuracy, benchmark, describe, basis_quality, residual_arrays, check_accuracy
from algorithm_laboratory.algorithms import randomized, jacobi, orthogonalize
from algorithm_laboratory.my_algorithm import solve as mine
from algorithm_laboratory.power import summarize_power

DEVICE=os.environ.get('SVD_TEST_DEVICE','cpu')
torch.set_num_threads(1)


class Laboratory(unittest.TestCase):
    def test_custom_spectrum_roundtrip_and_batch(self):
        case=MatrixCase.from_spectrum([3,3,1,0],m=7,n=5,dtype='complex128')
        np.testing.assert_allclose(case.reference[0][1][:4],[3,3,1,0],atol=2e-15)
        with tempfile.TemporaryDirectory() as tmp:
            case.save(tmp)
            self.assertEqual(MatrixCase.load(Path(tmp)/'input.npy').sha256,case.sha256)
        batch=MatrixCase.create(m=7,n=5,batch=2,family='gaussian')
        result=torch.linalg.svd(batch.tensor(DEVICE),full_matrices=False)
        self.assertLess(accuracy(batch,result)['worst']['reconstruction_relative_fro'],1e-12)

    def test_tracing_does_not_change_result_and_snapshots_are_immutable(self):
        case=MatrixCase.create(m=24,n=13,rank=3,tail=1e-6,dtype='complex128')
        a=case.tensor(DEVICE)
        expected=randomized(a,NullTrace(),rank=3,iterations=2,seed=19)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'run'
            with Trace(folder,case,DEVICE) as trace:
                actual=randomized(a,trace,rank=3,iterations=2,seed=19)
                marker=torch.ones(2,2,device=DEVICE)
                event=trace.checkpoint('immutable',tensors={'marker':marker})
                marker.zero_()
            for x,y in zip(expected,actual):torch.testing.assert_close(x,y,rtol=0,atol=0)
            run=Run(folder)
            np.testing.assert_array_equal(run.tensor(event['event_id'],'marker'),np.ones((2,2)))
            self.assertEqual(len(run.select(name='iterate')),3)
            self.assertIn('relative_error',run.select(name='final')[0]['tensors'])
            self.assertEqual(run.metadata['input_sha256'],case.sha256)
            stages=run.stage_statistics(by_iteration=True)
            self.assertTrue(any(s['name']=='orthogonalize_left' and s['iteration']==2 for s in stages))

    def test_null_trace_omits_diagnostic_core_solves(self):
        case=MatrixCase.create(m=15,n=9,rank=2)
        calls=[]
        def core(b):
            calls.append(tuple(b.shape));return torch.linalg.svd(b,full_matrices=False)
        randomized(case.tensor(DEVICE),NullTrace(),rank=2,iterations=3,core_solver=core)
        self.assertEqual(len(calls),1)

    def test_jacobi_real_complex_tall_wide_zero_and_rank_deficient(self):
        for dtype in ('float64','complex128'):
            for shape in ((11,7),(7,11),(1,1)):
                for family in ('gaussian','low_rank','zero'):
                    with self.subTest(dtype=dtype,shape=shape,family=family):
                        case=MatrixCase.create(m=shape[0],n=shape[1],dtype=dtype,family=family,rank=1,tail=0)
                        result=jacobi(case.tensor(DEVICE))
                        w=accuracy(case,result)['worst']
                        self.assertLess(w['reconstruction_relative_fro'],1e-12)
                        self.assertLess(w['orthogonality_u_fro_over_sqrt_cols'],1e-12)
                        self.assertLess(w['orthogonality_v_fro_over_sqrt_rows'],1e-12)

    def test_dense_sparse_products_and_custom_template(self):
        case=MatrixCase.create(m=17,n=11,family='sparse',density=.6,dtype='complex128')
        dense=randomized(case.tensor(DEVICE),rank=4,oversample=7,primitives='library')
        sparse=randomized(case.tensor(DEVICE,sparse=True),rank=4,oversample=7,primitives='library')
        torch.testing.assert_close(dense[1],sparse[1],atol=1e-12,rtol=1e-12)
        self.assertLess(accuracy(case,sparse)['worst']['resolved_spectrum_max_relative_error'],1e-12)
        case=MatrixCase.create(m=17,n=11,rank=3,tail=0)
        self.assertLess(accuracy(case,mine(case.tensor(DEVICE),rank=3))['worst']['reconstruction_relative_fro'],1e-12)

    def test_budget_every_and_failed_run_survive_reload(self):
        case=MatrixCase.create(m=4,n=3,family='gaussian')
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'run'
            with self.assertRaisesRegex(RuntimeError,'deliberate'):
                with Trace(folder,case,DEVICE,snapshot_budget_bytes=0,every=2,max_events=2) as trace:
                    self.assertIsNone(trace.checkpoint('skipped',iteration=1))
                    event=trace.checkpoint('kept',iteration=2,tensors={'A':case.tensor(DEVICE)})
                    self.assertEqual(event['tensors']['A']['snapshot_status'],'budget_exceeded')
                    trace.record('scalar',number=float('nan'))
                    trace.record('dropped',number=2)
                    raise RuntimeError('deliberate')
            run=Run(folder)
            self.assertEqual(run.metadata['status'],'failed')
            self.assertEqual(run.metadata['events_dropped'],1)
            self.assertIsNone(run.select(name='scalar')[0]['scalars']['number'])
            with self.assertRaisesRegex(ValueError,'not saved'):run.tensor(0,'A')
            with self.assertRaises(FileExistsError):Trace(folder,case)

    def test_stage_errors_and_no_observations_in_timer(self):
        case=MatrixCase.create(m=4,n=3,family='gaussian')
        with tempfile.TemporaryDirectory() as tmp:
            with Trace(Path(tmp)/'run',case,DEVICE) as trace:
                with self.assertRaisesRegex(RuntimeError,'after measured'):
                    with trace.stage('broken'):trace.checkpoint('bad')
                trace.checkpoint('recovered',tensors={'A':case.array})
            run=Run(Path(tmp)/'run')
            self.assertEqual(run.select(name='broken')[0]['status'],'error')
            if DEVICE.startswith('cuda'):
                self.assertIsNotNone(run.select(name='broken')[0]['device_event_ms'])

    def test_quality_checks_repeated_boundary_and_wrong_shapes(self):
        case=MatrixCase.from_spectrum([4,4,1],m=5,n=3)
        q=torch.from_numpy(case.reference[0][0][:,:1].copy())
        quality=basis_quality(case,q)
        self.assertFalse(quality['leading_subspace_defined'])
        self.assertIsNone(quality['largest_principal_angle_radians'])
        with self.assertRaisesRegex(ValueError,'shape'):
            accuracy(case,(q,torch.ones(2),torch.ones(2,3)))

    def test_timing_samples_and_power_integration(self):
        a=torch.eye(4,device=DEVICE)
        result=benchmark(lambda:a@a,DEVICE,repeat=5,warmup=1)
        self.assertEqual(result['wall_ms']['n'],5)
        self.assertIsNotNone(result['wall_ms']['median_bootstrap_95_interval'])
        self.assertFalse(result['inspection_enabled'])
        with self.assertRaises(ValueError):benchmark(lambda:a,repeat=0)
        self.assertIsNone(describe([1,2])['median_bootstrap_95_interval'])
        rows=[dict(uuid='test',monotonic_s=t,power_w=p) for t,p in [(0,100),(1,20),(2,40),(3,500)]]
        power=summarize_power(rows,.5,2.5)[0]
        self.assertEqual(power['gross_energy_joules'],30)
        self.assertEqual(power['covered_seconds'],1)

    def test_partial_sweeps_residual_arrays_and_limits(self):
        case=MatrixCase.create(m=15,n=9,family='gaussian')
        factors=jacobi(case.tensor(DEVICE),max_sweeps=1,require_convergence=False)
        metrics=accuracy(case,factors)
        residual=residual_arrays(case,factors)
        self.assertEqual(residual['left_triplet_norms'].shape,(9,))
        self.assertFalse(check_accuracy(metrics,orthogonality_u_fro_over_sqrt_cols=1e-12)['passed'])
        self.assertTrue(check_accuracy(metrics,reconstruction_relative_fro=1e-12)['passed'])

    def test_divergent_checkpoint_retains_arrays(self):
        case=MatrixCase.create(m=5,n=3,family='gaussian')
        with tempfile.TemporaryDirectory() as tmp:
            with Trace(Path(tmp)/'run',case) as trace:
                factors=(np.full((5,3),np.nan),np.ones(3),np.eye(3))
                event=trace.checkpoint('diverged',factors=factors)
                self.assertIn('validation_error',event)
                self.assertEqual(event['tensors']['U']['nonfinite'],15)
            self.assertTrue(np.isnan(Run(Path(tmp)/'run').tensor(0,'U')).all())


if __name__=='__main__':unittest.main()
