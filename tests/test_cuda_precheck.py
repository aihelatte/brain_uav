"""Small regression checks for the standalone diagnostic runner."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch


def load_script():
    spec = importlib.util.spec_from_file_location('cuda_precheck', Path(__file__).resolve().parents[1] / 'scripts' / 'cuda_precheck.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCudaPrecheck(unittest.TestCase):
    def test_cuda_unavailable_fails_before_creating_outputs(self):
        runner = load_script()
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'output'
            with patch.object(torch.cuda, 'is_available', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                    runner.main(['--model', 'ann', '--bc-checkpoint', 'missing.pt', '--output-dir', str(output)])
            self.assertFalse(output.exists())

    def test_state_comparison_detects_mutation(self):
        runner = load_script()
        runner.check_equal({'a': [torch.tensor([1.0])]}, {'a': [torch.tensor([1.0])]})
        with self.assertRaises(AssertionError):
            runner.check_equal({'a': torch.tensor([1.0])}, {'a': torch.tensor([2.0])})

    def test_cpu_requires_explicit_selection(self):
        runner = load_script()
        args = runner.parser().parse_args(['--model', 'ann', '--bc-checkpoint', 'x.pt', '--output-dir', 'out'])
        self.assertEqual(args.device, 'cuda')
