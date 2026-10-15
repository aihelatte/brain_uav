"""CPU/mock lifecycle checks only, not CUDA registration/capture validation."""
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from unittest import TestCase, mock

import torch

from test_v2_td3 import TestV2TD3 as Fixture


class TestGPUComputeRegistration(TestCase):
    def engine(self):
        fixture = Fixture(); fixture.setUp()
        return fixture.make_engine()

    @contextmanager
    def cuda(self, *, changed_rng=False):
        current, stream, graph = mock.Mock(), mock.Mock(), object()
        before = torch.tensor([1, 2], dtype=torch.uint8)
        after = before + 1 if changed_rng else before.clone()
        with ExitStack() as stack:
            for name, value in (
                ('device', lambda device: nullcontext()),
                ('current_stream', lambda device: current),
                ('Stream', mock.Mock(return_value=stream)),
                ('CUDAGraph', mock.Mock(return_value=graph)),
                ('graph', mock.Mock(side_effect=lambda *a, **kw: nullcontext())),
                ('synchronize', mock.Mock()),
                ('get_rng_state', mock.Mock(side_effect=[before, after])),
            ):
                stack.enter_context(mock.patch.object(torch.cuda, name, value))
            yield current, stream, graph

    def test_disabled_does_not_touch_cuda(self):
        e = self.engine()
        with mock.patch.object(torch.cuda, 'CUDAGraph', side_effect=AssertionError('disabled')):
            e._ensure_gpu_compute_graph_registration()
        self.assertIsNone(e._gpu_compute_registration_graph)

    def test_registration_precedes_compile_and_survives_configuration_and_load(self):
        from brain_uav.scripts.train_v2_td3 import default_v2_cuda_graph_compilation
        e = self.engine(); initial = deepcopy(e.checkpoint_state_dict())
        e.gpu_td3_compute = True; e.device = torch.device('cuda:1')
        e.reduce_update_stat_syncs = True
        options = default_v2_cuda_graph_compilation(model='ann', resolved_device='cuda')
        options.pop('fused_adam'); options.pop('reduce_update_stat_syncs')
        methods = ('enable_shared_relations_compile','enable_actor_loss_compile',
            'enable_action_inference_compile','enable_critic_loss_compile',
            'enable_actor_guidance_context_compile','enable_target_block_compile','enable_actor_compile')
        with self.cuda() as (current, stream, graph), ExitStack() as stack:
            def compile_scope(**kwargs):
                self.assertIs(e._gpu_compute_registration_graph, graph)
                self.assertFalse(torch.is_inference_mode_enabled())
                return ()
            for method in methods:
                stack.enter_context(mock.patch.object(e, method, side_effect=compile_scope))
            stack.enter_context(mock.patch.object(e, 'set_frozen_critic_strategy',
                side_effect=lambda strategy: setattr(e, 'frozen_critic_strategy', strategy)))
            for _ in range(2):
                e.configure_compilation(**options)
            torch.cuda.CUDAGraph.assert_called_once_with()
            torch.cuda.Stream.assert_called_once_with(device=torch.device('cuda:1'))
            stream.wait_stream.assert_called_once_with(current)
            current.wait_stream.assert_called_once_with(stream)
            e.device = torch.device('cpu')  # Actual loader on local CPU fixture.
            old_executor = mock.Mock()
            e._gpu_update_executor = old_executor
            e.load_checkpoint_state_dict(initial)
            old_executor.invalidate.assert_called_once_with()
            self.assertIs(e._gpu_compute_registration_graph, graph)
            self.assertIsNone(e._gpu_update_executor)
            e.device = torch.device('cuda:1')
            e._ensure_gpu_compute_graph_registration()
            torch.cuda.CUDAGraph.assert_called_once_with()
            e.device = torch.device('cuda:0')
            with self.assertRaisesRegex(RuntimeError, 'device changed'):
                e._ensure_gpu_compute_graph_registration()

    def test_rng_change_fails_without_publishing_graph_or_retry(self):
        e = self.engine(); e.gpu_td3_compute = True; e.device = torch.device('cuda:0')
        with self.cuda(changed_rng=True):
            with self.assertRaisesRegex(RuntimeError, 'changed RNG'):
                e._ensure_gpu_compute_graph_registration()
            self.assertIsNone(e._gpu_compute_registration_graph)
            with self.assertRaisesRegex(RuntimeError, 'previously failed'):
                e._ensure_gpu_compute_graph_registration()
            torch.cuda.CUDAGraph.assert_called_once_with()

    def test_inference_mode_registration_is_rejected(self):
        e = self.engine(); e.gpu_td3_compute = True; e.device = torch.device('cuda:0')
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'outside InferenceMode'):
            e._ensure_gpu_compute_graph_registration()
