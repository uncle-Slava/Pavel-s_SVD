"""svdlab front door: blocks, recipes, file methods, the command line and native builds."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
import torch
from svdlab import blocks, check, recipes
from svdlab.__main__ import main
from svdlab.scorecard import QUICK_SUITE, label_of, load_file
from svdlab import native

DEVICE = os.environ.get('SVD_TEST_DEVICE', 'cpu')
ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def matrix(m, n, dtype=torch.float64, seed=0):
    return torch.randn(m, n, dtype=dtype, generator=torch.Generator().manual_seed(seed)).to(DEVICE)


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
        result = fn(*args, **kw)
    return result, out.getvalue() + err.getvalue()


class Blocks(unittest.TestCase):
    def close(self, x, y, tol=1e-12):
        self.assertLess(float(torch.linalg.norm(x - y) / torch.linalg.norm(y)), tol)

    def test_qr_and_bidiagonalization_reconstruct(self):
        for dtype in (torch.float64, torch.complex128):
            a = matrix(12, 7, dtype)
            q, r = blocks.qr(a)
            self.close(q @ r, a)
            self.assertTrue(torch.allclose(r, r.triu()))
            q, d, e, p = blocks.bidiagonalize(a)
            b = torch.diag(d) + torch.diag(e, 1)
            self.close(q @ b.to(dtype) @ p.mH, a)
            u, s, v = blocks.bidiagonal_svd(d, e)
            self.close(u @ torch.diag(s) @ v.mT, b)

    def test_tall_only_blocks_explain_wide_input(self):
        with self.assertRaisesRegex(ValueError, 'a.mH'):
            blocks.qr(matrix(3, 5))

    def test_hermitian_and_triangular_blocks(self):
        a = matrix(9, 6)
        g = a.mT @ a
        w, v = blocks.hermitian_eig(g)
        self.close(v @ torch.diag(w) @ v.mT, g)
        l = blocks.cholesky(g)
        self.close(l @ l.mT, g)
        x = blocks.solve_lower(l, a.mT)
        self.close(l @ x, a.mT)
        q, d, e = blocks.tridiagonalize(g)
        self.close(q @ (torch.diag(d) + torch.diag(e, 1) + torch.diag(e, -1)) @ q.mT, g)


class Recipes(unittest.TestCase):
    def test_recipes_pass_the_quick_suite(self):
        for recipe in (recipes.golub_kahan, recipes.qr_then_svd, recipes.randomized, recipes.lanczos, recipes.jacobi):
            rows = check(recipe, QUICK_SUITE, device=DEVICE, repeat=1, warmup=0, verbose=False)
            self.assertTrue(all(r['ok'] for r in rows), (recipe.__name__, [(r['case'], r['error']) for r in rows]))

    def test_a_replaced_step_is_used(self):
        calls = []

        def counting_qr(y):
            calls.append(y.shape)
            return blocks.torch_qr(y)
        u, s, vh = recipes.randomized(matrix(30, 20), 5, qr=counting_qr, power_iterations=1)
        self.assertEqual(len(calls), 3)
        self.assertEqual((u.shape, s.shape, vh.shape), ((30, 5), (5,), (5, 20)))


class Lanczos(unittest.TestCase):
    def test_block_gives_orthonormal_bases_and_the_bidiagonal(self):
        for dtype in (torch.float64, torch.complex128):
            a = matrix(30, 18, dtype, seed=1)
            u, alpha, beta, v = blocks.lanczos_bidiagonalize(a, 10)
            self.assertEqual((u.shape, alpha.shape, beta.shape, v.shape), ((30, 10), (10,), (9,), (18, 10)))
            eye = torch.eye(10, dtype=dtype, device=DEVICE)
            self.assertLess(float((u.mH @ u - eye).abs().max()), 1e-12)
            self.assertLess(float((v.mH @ v - eye).abs().max()), 1e-12)
            b = (torch.diag(alpha) + torch.diag(beta, 1)).to(dtype)
            self.assertLess(float((u.mH @ a @ v - b).abs().max() / alpha.abs().max()), 1e-12)
        u, alpha, beta, v = blocks.lanczos_bidiagonalize(matrix(5, 4), 1)
        self.assertEqual((u.shape, alpha.shape, beta.shape, v.shape), ((5, 1), (1,), (0,), (4, 1)))

    def test_lanczos_svd_is_built_on_the_block(self):
        from svdbench.standalone import iterative
        from svdbench.standalone.solvers import svd as standalone_svd
        a = matrix(40, 25, seed=3)
        u, s, v, info = iterative.lanczos_svd(iterative.DenseOperator(a), rank=6)
        ub, _, _, vb, block_info = iterative.lanczos_bidiagonalize(iterative.DenseOperator(a), info['steps'])
        vb = torch.cat([vb, block_info['next_v']], 1)
        core = ub.mH @ (a @ vb)                   # same order as lanczos_svd: matmat first
        uc, sc, _ = standalone_svd(core, 'jacobi')
        torch.testing.assert_close(s, sc[:6], rtol=0, atol=0)
        torch.testing.assert_close(u, ub @ uc[:, :6], rtol=0, atol=0)
        self.assertEqual(info['breakdown_restarts'], block_info['breakdown_restarts'])

    def test_recipe_low_rank_and_full_rank(self):
        cases = [dict(name='low_rank', m=90, n=60, family='low_rank', rank=8, tail=1e-7, output_rank=8),
                 dict(name='gaussian', m=40, n=30, family='gaussian')]
        rows = check(recipes.lanczos, cases[:1], device=DEVICE, repeat=1, warmup=0, verbose=False)
        self.assertTrue(rows[0]['ok'], rows[0])
        full = lambda a, rank: recipes.lanczos(a, rank, steps=min(a.shape))
        rows = check(full, cases[1:], device=DEVICE, repeat=1, warmup=0, verbose=False)
        self.assertTrue(rows[0]['ok'], rows[0])
        u, s, vh = recipes.lanczos(matrix(20, 50), 5)               # wide input through the adjoint
        self.assertEqual((u.shape, s.shape, vh.shape), ((20, 5), (5,), (5, 50)))

    def test_swapping_the_bidiagonal_svd(self):
        calls = []

        def torch_bidiagonal_svd(d, e):
            calls.append(len(d))
            u, s, vh = torch.linalg.svd(torch.diag(d) + torch.diag(e, 1))
            return u, s, vh.mT
        svd = lambda a, rank: recipes.lanczos(a, rank, bidiagonal_svd=torch_bidiagonal_svd)
        rows = check(svd, [dict(name='low_rank', m=90, n=60, family='low_rank', rank=8, tail=1e-7, output_rank=8)],
                     device=DEVICE, repeat=1, warmup=0, verbose=False)
        self.assertTrue(rows[0]['ok'], rows[0])
        self.assertTrue(calls and all(k == 24 for k in calls))


class Files(unittest.TestCase):
    def write(self, text, name='method.py'):
        folder = Path(tempfile.mkdtemp())
        (folder / name).write_text(text)
        return folder / name

    def test_file_function_lookup(self):
        path = self.write('import torch\ndef other(a, rank):\n    return torch.linalg.svd(a, full_matrices=False)\n'
                          'def svd(a, rank):\n    return other(a, rank)\n')
        self.assertEqual(load_file(path).__name__, 'svd')
        self.assertEqual(load_file(path, 'other').__name__, 'other')
        self.assertEqual(label_of(f'{path}:other'), 'other')
        self.assertEqual(label_of(str(path)), 'method')
        self.assertEqual(label_of('svdlab.recipes:randomized'), 'recipes.randomized')
        with self.assertRaisesRegex(AttributeError, 'Name it svd'):
            load_file(self.write('def helper(a, rank):\n    pass\n'))

    def test_command_line_exit_codes(self):
        good = str(ROOT / 'examples' / 'my_svd.py')
        bad = str(self.write('import torch\ndef svd(a, rank):\n    u, s, vh = torch.linalg.svd(a, full_matrices=False)\n'
                             '    return u, s * 1.01, vh\n', 'scaled.py'))
        args = ['check', '--quick', '--case', 'gaussian_tall', '--repeat', '1', '--device', DEVICE]
        code, out = quiet(main, [*args, good, '--compare', 'torch'])
        self.assertEqual(code, 0, out)
        self.assertIn('my_svd: 1/1 ok', out)
        code, out = quiet(main, [*args, bad])
        self.assertEqual(code, 1, out)
        code, out = quiet(main, [*args, good, '--compare', bad])   # a failing baseline does not fail the run
        self.assertEqual(code, 0, out)
        code, out = quiet(main, [*args, 'missing_file.py'])
        self.assertEqual(code, 2)
        self.assertIn('does not exist', out)
        csv = Path(tempfile.mkdtemp()) / 'rows.csv'
        quiet(main, [*args, good, '--csv', str(csv)])
        self.assertIn('gaussian_tall', csv.read_text())

    def test_python_example(self):
        from importlib import util
        spec = util.spec_from_file_location('example_my_svd', ROOT / 'examples' / 'my_svd.py')
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = check(module.svd, QUICK_SUITE, device=DEVICE, repeat=1, warmup=0, verbose=False)
        self.assertTrue(all(r['ok'] for r in rows), [(r['case'], r['error']) for r in rows])


@unittest.skipIf(native.missing_tools(), f'native build tools missing: {native.missing_tools()}')
class Native(unittest.TestCase):
    def test_cpp_example_builds_and_passes(self):
        fn = load_file(ROOT / 'examples' / 'native' / 'my_svd.cpp')
        rows = check(fn, QUICK_SUITE, device=DEVICE, repeat=1, warmup=0, verbose=False)
        self.assertTrue(all(r['ok'] for r in rows), [(r['case'], r['error']) for r in rows])

    @unittest.skipUnless(DEVICE.startswith('cuda') and not native.missing_tools(cuda=True), 'needs CUDA and nvcc')
    def test_cuda_example_builds_and_passes_real_cases(self):
        fn = load_file(ROOT / 'examples' / 'native' / 'my_svd.cu')
        rows = check(fn, [c for c in QUICK_SUITE if 'complex' not in c['dtype']], device=DEVICE, repeat=1,
                     warmup=0, verbose=False)
        self.assertTrue(all(r['ok'] for r in rows), [(r['case'], r['error']) for r in rows])


if __name__ == '__main__':
    unittest.main()
