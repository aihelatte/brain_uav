"""Small CPU checks for actor-update graph wiring, not CUDA capture tests."""
import unittest
from unittest import mock

import torch

import test_v2_td3 as ann_fixture
import test_v2_snn_td3 as snn_fixture
from brain_uav.scripts import profile_v2_td3, train_v2_td3, run_v2_td3_curriculum


class TestActorUpdateGraph(unittest.TestCase):
    def make_engine(self, snn=False):
        fixture = snn_fixture.TestV2SNNTD3() if snn else ann_fixture.TestV2TD3()
        fixture.setUp()
        if snn:
            return fixture.make_engine(bc=fixture.make_actor())
        engine = fixture.make_engine(bc_reference_actor=fixture.make_bc_reference(0.01))
        fixture.fill_replay(engine, counts=(0, 7), next_counts=(10, 0))
        return engine

    def configure(self, engine):
        engine.device = torch.device('cuda')
        calls = []
        def compile_fn(fn, **kwargs):
            calls.append((fn, kwargs))
            return fn
        with mock.patch('torch.compile', side_effect=compile_fn):
            metadata = engine.configure_compilation(
                compile_actors=True, compile_critic_block=True,
                frozen_critic_strategy='compiled_no_grad_context',
                cuda_graph_updates=True, cuda_graph_actor_update=True,
            )
        engine.device = torch.device('cpu')
        return calls, metadata

    def test_snn_monitor_validation_compile_without_cudagraphs(self):
        for snn in (False, True):
            with self.subTest(snn=snn):
                engine = self.make_engine(snn)
                calls, _ = self.configure(engine)
                actor_calls = {
                    fn.__name__: (fn, kwargs)
                    for fn, kwargs in calls
                    if fn.__name__ in (
                        '_online_actor_context_tensors',
                        '_online_actor_monitor_context_tensors',
                        '_online_actor_validation_context_tensors',
                        '_compute_full_forward_tensors',
                    )
                }
                if snn:
                    self.assertEqual(set(actor_calls), {
                        '_online_actor_context_tensors',
                        '_online_actor_monitor_context_tensors',
                        '_online_actor_validation_context_tensors',
                    })
                    self.assertEqual(len({fn.__code__ for fn, _ in actor_calls.values()}), 3)
                    self.assertEqual(
                        actor_calls['_online_actor_context_tensors'][1]['options'],
                        {'triton.cudagraphs': True},
                    )
                    for role in ('monitor', 'validation'):
                        kwargs = actor_calls[f'_online_actor_{role}_context_tensors'][1]
                        self.assertEqual(kwargs['backend'], 'inductor')
                        self.assertEqual(kwargs['options'], {'triton.cudagraphs': False})
                else:
                    self.assertEqual(set(actor_calls), {'_compute_full_forward_tensors'})
                    self.assertEqual(
                        actor_calls['_compute_full_forward_tensors'][1]['options'],
                        {'triton.cudagraphs': True},
                    )

    def test_wiring_and_real_cpu_gradient_boundaries(self):
        for snn in (False, True):
            with self.subTest(snn=snn):
                engine = self.make_engine(snn)
                calls, metadata = self.configure(engine)
                graph_calls = [
                    kw for _, kw in calls
                    if kw.get('options') == {'triton.cudagraphs': True}
                ]
                self.assertEqual(len(graph_calls), 4)  # critic + three new blocks
                self.assertTrue(metadata['cuda_graph_actor_update'])
                self.assertIsNone(engine._compiled_action_inference)
                reference_before = {k: v.clone() for k, v in engine.bc_reference_actor.state_dict().items()}
                actor_before = [p.detach().clone() for p in engine.actor.parameters()]
                with mock.patch('torch.compiler.cudagraph_mark_step_begin'):
                    metrics = engine.update_once(total_steps=2, bc_lambda=1.5)
                    batch = engine.replay.sample(2).obs
                    expected = engine.critic1.encode_context(batch).detach().clone()
                    with engine._actor_critic_guidance(batch, shared_relations=None, profile_sections=False) as context:
                        torch.testing.assert_close(context, expected)
                        self.assertFalse(context.requires_grad)
                self.assertTrue(metrics.actor_updated)
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, engine.actor.parameters())))
                self.assertTrue(any(p.grad is not None for p in engine.actor.parameters()))
                self.assertTrue(all(p.grad is None for p in engine.bc_reference_actor.parameters()))
                for k, v in reference_before.items():
                    torch.testing.assert_close(engine.bc_reference_actor.state_dict()[k], v)
                if snn:
                    self.assertEqual(engine.actor.snn_head.lif1.v, 0.0)
                    self.assertEqual(engine.bc_reference_actor.snn_head.lif2.v, 0.0)
                    engine.actor.snn_head.lif1.v = 0.25
                gradients = [None if p.grad is None else p.grad.clone() for p in engine.actor.parameters()]
                with mock.patch('torch.compiler.cudagraph_mark_step_begin'):
                    engine.warmup_actor_compile((batch,))
                for parameter, saved in zip(engine.actor.parameters(), gradients):
                    if saved is None:
                        self.assertIsNone(parameter.grad)
                    else:
                        torch.testing.assert_close(parameter.grad, saved)
                if snn:
                    self.assertEqual(engine.actor.snn_head.lif1.v, 0.25)

    def test_constraints_and_parser_defaults(self):
        engine = self.make_engine()
        with self.assertRaisesRegex(ValueError, 'cuda_graph_actor_update requires'):
            engine.configure_compilation(cuda_graph_actor_update=True)
        for module in (profile_v2_td3, train_v2_td3, run_v2_td3_curriculum):
            parser = module.build_parser()
            option = next(a for a in parser._actions if a.dest == 'cuda_graph_actor_update')
            self.assertFalse(option.default)
            self.assertIn('--cuda-graph-actor-update', option.option_strings)

    def test_missing_actor_backward_evidence_is_not_hidden(self):
        engine = self.make_engine()
        self.configure(engine)
        engine.device = torch.device('cuda')
        batch = engine.replay.sample(2).obs
        def prepare(batch, scope):
            return lambda: scope
        def profile(replay):
            scope = replay()
            return {'cuda_graph_launch_count': int(scope != 'actor_backward'), 'event_names': ['cudaGraphLaunch']}
        with mock.patch.object(engine, 'warmup_full_compile'), mock.patch.object(engine, 'warmup_actor_compile'), mock.patch.object(engine, '_prepare_update_cuda_graph_verification_scope', side_effect=prepare), mock.patch.object(engine, '_profile_cuda_graph_replay', side_effect=profile):
            with self.assertRaisesRegex(RuntimeError, 'actor_backward'):
                engine.verify_update_cuda_graph_capture((batch,))
