"""Tools for editable, step-by-step numerical experiments."""
from .matrices import MatrixCase
from .diagnostics import accuracy, spectrum_errors, orthogonality, basis_quality, tensor_statistics, residual_arrays, check_accuracy
from .trace import Trace, NullTrace, Run
from .statistics import benchmark, describe

__all__ = ['MatrixCase', 'Trace', 'NullTrace', 'Run', 'benchmark', 'describe',
           'accuracy', 'spectrum_errors', 'orthogonality', 'basis_quality', 'tensor_statistics', 'residual_arrays', 'check_accuracy']
