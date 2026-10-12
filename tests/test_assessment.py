import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("assess", Path(__file__).parents[1] / "tools/assess.py")
assessor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assessor)


class AssessmentContract(unittest.TestCase):
    def test_info_success_is_not_numerical_success(self):
        row = {"status": "ok", "metrics": {"worst": {
            "singular_values_nonnegative_sorted": True, "reconstruction_relative_fro": 27.0}}}
        self.assertTrue(assessor.assess(row, {"reconstruction_relative_fro": 1e-3}))

    def test_missing_metric_is_not_a_pass(self):
        row = {"status": "ok", "metrics": {"worst": {"singular_values_nonnegative_sorted": True}}}
        self.assertTrue(assessor.assess(row, {"reconstruction_relative_fro": 1e-3}))

    def test_good_metric_passes_explicit_tolerance(self):
        row = {"status": "ok", "metrics": {"worst": {
            "singular_values_nonnegative_sorted": True, "reconstruction_relative_fro": 1e-7}}}
        self.assertEqual(assessor.assess(row, {"reconstruction_relative_fro": 1e-3}), [])
