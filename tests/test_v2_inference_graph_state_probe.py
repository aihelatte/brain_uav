"""CPU process/report contracts only; not CUDA capture evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock
import json

from brain_uav.scripts import probe_v2_inference_graph_state as probe


class TestInferenceGraphStateProbe(TestCase):
    def results(self):
        return dict(zip(probe.CASES, (
            dict(empty_capture_succeeded=True),
            dict(compiled_graph_verified=True, stage='normal_empty_capture', error=dict(
                message='Inplace update to inference tensor outside InferenceMode is not allowed.')),
            dict(compiled_graph_verified=True, empty_capture_succeeded=True,
                registration_rng_unchanged=True, early_graph_retained_through_final_capture=True))))

    def test_candidate_requires_graph_evidence_rng_and_retained_graph(self):
        results = self.results()
        self.assertTrue(probe.verdict(results).startswith('supports_early_registration'))
        for case in probe.CASES[1:]:
            results[case]['compiled_graph_verified'] = False
            self.assertEqual(probe.verdict(results),'insufficient_cuda_graph_evidence')
            results[case]['compiled_graph_verified'] = True
        for field in ('registration_rng_unchanged','early_graph_retained_through_final_capture'):
            results[probe.CASES[2]][field] = False
            self.assertEqual(probe.verdict(results),'reproduced_candidate_not_validated')
            results[probe.CASES[2]][field] = True
        results[probe.CASES[1]]['empty_capture_succeeded'] = True
        self.assertEqual(probe.verdict(results),'not_reproduced_on_simplified_model')

    def test_fresh_children_failure_isolation_and_partial_evidence(self):
        commands = []
        results = self.results()
        def child(command, **kwargs):
            commands.append(command)
            self.assertEqual(kwargs['env']['TORCH_SHOW_CPP_STACKTRACES'],'1')
            self.assertFalse(kwargs['check'])
            case = command[command.index('--case')+1]
            if case == 'plain':
                raise OSError('synthetic spawn failure')
            output = Path(command[-1]); output.mkdir()
            probe.write(output/'result.json',results[case])
            return SimpleNamespace(returncode=1 if case == probe.CASES[1] else 0)
        with TemporaryDirectory() as directory, mock.patch.object(probe.subprocess,'run',side_effect=child):
            self.assertEqual(probe.main(['--output-dir',directory]),1)
            report = json.loads((Path(directory)/'summary.json').read_text())
            self.assertEqual(len(commands),3)
            self.assertEqual(len({command[-1] for command in commands}),3)
            self.assertEqual(report['status'],'control_failed')
            self.assertIn('traceback',report['results']['plain']['error'])
            self.assertTrue(report['results'][probe.CASES[2]]['empty_capture_succeeded'])

    def test_no_cuda_reports_failure_with_original_context(self):
        with TemporaryDirectory() as directory, mock.patch.object(probe.torch.cuda,'is_available',return_value=False):
            self.assertEqual(probe.worker('plain',Path(directory)),1)
            report = json.loads((Path(directory)/'result.json').read_text())
            self.assertFalse(report['empty_capture_succeeded'])
            self.assertEqual(report['stage'],'require_cuda')
            self.assertEqual(report['error']['type'],'RuntimeError')
            self.assertIn('CUDA unavailable',report['error']['message'])
            self.assertIn('Traceback',report['error']['traceback'])
