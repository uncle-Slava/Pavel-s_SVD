import unittest

import numpy as np
import torch

from svdbench.algorithms import Unsupported, prepare
from svdbench.matrices import make_matrix
from svdbench.metrics import reference, validate


class NumericalContract(unittest.TestCase):
    def solve(self, a, spec):
        return tuple(None if x is None else x.resolve_conj().resolve_neg().numpy() for x in prepare(torch.from_numpy(a), spec, 11)())

    def test_exact_rectangular_complex_and_batches(self):
        for m, n, b, dtype in [(40, 16, 1, "float64"), (16, 40, 1, "complex128"), (24, 16, 3, "float64")]:
            a = make_matrix(dict(m=m, n=n, batch=b, dtype=dtype, family="gaussian"))
            for name in ["default", "qr_svd"]:
                metric = validate(a, self.solve(a, dict(name=name)), reference(a))["worst"]
                self.assertLess(metric["reconstruction_relative_fro"], 1e-12)
                self.assertLess(metric["orthogonality_u_fro_over_sqrt_cols"], 1e-12)

    def test_randomized_recovers_known_rank(self):
        a = make_matrix(dict(m=100, n=60, family="low_rank", rank=5, tail=0, dtype="float64"))
        out = self.solve(a, dict(name="randomized", rank=5, oversample=4, iterations=1))
        metric = validate(a, out, reference(a))["worst"]
        self.assertEqual(out[0].shape, (100, 5))
        self.assertLess(metric["reconstruction_relative_fro"], 1e-12)

    def test_lowrank_error_compared_with_tail(self):
        a = np.diag(np.array([4., 3., 2., 1.]))
        out = (np.eye(4)[:, :2], np.array([4., 3.]), np.eye(4)[:2])
        metric = validate(a, out, reference(a))["worst"]
        self.assertAlmostEqual(metric["residual_over_optimal_tail"], 1)
        self.assertAlmostEqual(metric["right_subspace_leakage_fro_over_sqrt_k"], 0)

    def test_signs_and_cluster_rotations_are_valid(self):
        a = np.diag([3., 3., 1.])
        q = np.array([[0., -1.], [1., 0.]])
        u = np.eye(3)[:, :2] @ q
        out = (u, np.array([3., 3.]), u.T)
        self.assertAlmostEqual(validate(a, out, reference(a))["worst"]["reconstruction_relative_fro"], 1/np.sqrt(19))
        cut = (np.eye(3)[:, :1], np.array([3.]), np.eye(3)[:1])
        self.assertIsNone(validate(a, cut, reference(a))["worst"]["right_subspace_leakage_fro_over_sqrt_k"])

    def test_zero_and_gram_nullspace(self):
        a = np.zeros((8, 4))
        exact = validate(a, self.solve(a, {"name": "default"}), reference(a))["worst"]
        self.assertEqual(exact["reconstruction_relative_fro"], 0)
        gram = validate(a, self.solve(a, {"name": "gram"}), reference(a))["worst"]
        self.assertEqual(gram["orthogonality_u_fro_over_sqrt_cols"], 1)

    def test_driver_does_not_fallback_to_cpu(self):
        with self.assertRaises(Unsupported):
            prepare(torch.eye(4), {"name": "gesvdj"}, 0)

    def test_spectrum_checks_actual_fp32_input(self):
        a = make_matrix(dict(m=24, n=16, family="geometric", condition=1e6, dtype="float32"))
        out = tuple(x for x in reference(a)[0])
        self.assertEqual(validate(a, out, reference(a))["worst"]["spectrum_error_over_sigma1"], 0)


if __name__ == "__main__":
    unittest.main()
