"""The tutorial examples behave as docs/tutorials describe; the notebooks run end to end.

Notebook execution takes a few minutes and needs nbclient and ipykernel, so it is opt-in:
    SVDLAB_TEST_NOTEBOOKS=1 python -m unittest tests.test_tutorials -v
"""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock
import torch
from svdlab import check, native
from svdlab.scorecard import QUICK_SUITE, load_file

DEVICE = os.environ.get('SVD_TEST_DEVICE', 'cpu')
ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def failed(rows):
    return sorted(r['case'] for r in rows if not r['ok'])


class Examples(unittest.TestCase):
    def test_gram_svd_fails_where_the_tutorial_says_and_refined_passes(self):
        path = ROOT / 'examples' / 'gram_svd.py'
        rows = check(load_file(path), QUICK_SUITE, device=DEVICE, repeat=1, warmup=0, verbose=False)
        self.assertEqual(failed(rows), ['clustered', 'ill_conditioned_1e8'])
        rows = check(load_file(path, 'refined'), QUICK_SUITE, device=DEVICE, repeat=1, warmup=0, verbose=False)
        self.assertEqual(failed(rows), [], [(r['case'], r['error']) for r in rows])

    @unittest.skipIf(native.missing_tools(), f'native build tools missing: {native.missing_tools()}')
    def test_minimal_native_example_exports_svd_and_qr(self):
        path = ROOT / 'examples' / 'native' / 'minimal.cpp'
        rows = check(load_file(path), QUICK_SUITE, device=DEVICE, repeat=1, warmup=0, verbose=False)
        self.assertEqual(failed(rows), [], [(r['case'], r['error']) for r in rows])
        q, r = native.build(path).qr(torch.randn(7, 4, dtype=torch.float64, device=DEVICE))
        self.assertEqual(tuple(q.shape), (7, 4))
        self.assertTrue(torch.allclose(r, r.triu()))


class Tables(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec('pandas') is None, 'needs pandas')
    def test_table_keeps_the_gpu_check_columns(self):
        """The notebooks select passed/cpu_ok/gpu_ok/speedup from table(gpu_check(...))."""
        from svdlab import gpu_check, table
        device = DEVICE if DEVICE.startswith('cuda') else 'cpu'
        with mock.patch('torch.cuda.is_available', return_value=True):   # gpu_check on two CPU runs
            rows = gpu_check('torch', QUICK_SUITE[:2], device=device, repeat=1, warmup=0, verbose=False)
        df = table(rows)
        for column in ('method', 'case', 'passed', 'cpu_ok', 'gpu_ok', 'output_device', 'on_gpu', 'speedup',
                       'max_singular_value_difference_over_sigma1'):
            self.assertIn(column, df.columns)
        self.assertEqual(list(df['case']), [c['name'] for c in QUICK_SUITE[:2]])


@unittest.skipUnless(os.environ.get('SVDLAB_TEST_NOTEBOOKS'), 'set SVDLAB_TEST_NOTEBOOKS=1 to execute the notebooks')
class Notebooks(unittest.TestCase):
    def test_notebooks_run_without_errors(self):
        import nbformat
        from nbclient import NotebookClient
        for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
            with self.subTest(notebook=path.name):
                nb = nbformat.read(path, as_version=4)
                NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': str(path.parent)}}).execute()
