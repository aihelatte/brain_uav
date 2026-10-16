"""CPU diagnostic contract tests; no CUDA capture/performance claims."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest import mock
import torch
from brain_uav.scripts import profile_v2_td3 as profile
from brain_uav.trainers import v2_td3 as td3
from brain_uav.observations import V2ObservationBatch


class TestTD3DiagnosticFlow(unittest.TestCase):
    def config(self):
        return SimpleNamespace(policy_delay=2, actor_freeze_steps=25000,
            spike_regularization=td3.V2SpikeRegularizationConfig(), max_steps=500000)

    def engine(self):
        return SimpleNamespace(actor=torch.nn.Linear(2, 1), critic1=torch.nn.Linear(2, 1),
            critic2=torch.nn.Linear(2, 1),
            _validate_and_clip_gradients=td3.V2TD3UpdateEngine._validate_and_clip_gradients)

    def test_increasing_steps_and_reference_contract(self):
        steps = profile._td3_block_correctness_steps(self.config(), 8)
        self.assertEqual(steps, [1, 2, 25000, 25002, 25003, 50000, 75000, 75002])
        self.assertTrue(all(a < b for a, b in zip(steps, steps[1:])))
        with self.assertRaises(ValueError): profile._td3_block_correctness_steps(self.config(), 7)
        contract = dict(version=profile.TD3_BLOCK_CORRECTNESS_VERSION, stage_steps=steps)
        with self.assertRaisesRegex(ValueError, 'Obsolete'):
            profile._validate_td3_block_reference({'snapshots': [{}] * 8}, contract, 8)
        payload = dict(contract=contract, snapshots=[{}] * 8)
        profile._validate_td3_block_reference(payload, contract, 8)
        with self.assertRaises(ValueError): profile._validate_td3_block_reference(payload, contract, 9)
        with self.assertRaisesRegex(ValueError, 'Obsolete'):
            profile._validate_td3_block_reference({**payload, 'contract': {**contract, 'version': 2}}, contract, 8)

    def test_gradient_storage_none_and_failure_context(self):
        engine = self.engine()
        engine.actor.weight.grad = torch.ones_like(engine.actor.weight)
        identity = engine.actor.weight.grad
        expected, metadata = profile._td3_gradient_snapshot(engine, actor_backward=True)
        self.assertIs(engine.actor.weight.grad, identity)
        self.assertTrue(expected['grad.actor.1.none'])
        engine.actor.weight.grad.add_(3)
        actual, evidence = profile._td3_gradient_snapshot(engine, actor_backward=False)
        self.assertEqual(expected['grad.actor.0'].tolist(), [[1., 1.]])
        self.assertEqual(evidence['grad.actor.0']['source'], 'retained_no_backward')
        report = {}
        context = dict(snapshot_index=4, stage_steps=25003, actor_backward=False, gradients=evidence)
        result = profile._compare_td3_block_snapshot(expected, actual, context=context, report=report)
        self.assertEqual(result['unused_retained_gradient_fields'], 1)
        observation = report['retained_gradient_observations'][0]
        self.assertEqual(observation['parameter_name'], 'actor.weight')
        self.assertEqual(observation['maximum_absolute_error'], 3.)
        self.assertFalse(observation['consumed_by_update'])
        self.assertIs(engine.actor.weight.grad, identity)
        # Identical numerical difference is still fatal for a current backward.
        current_context = {**context, 'actor_backward': True, 'gradients': metadata}
        with self.assertRaises(AssertionError):
            profile._compare_td3_block_snapshot(expected, actual, context=current_context, report=report)
        self.assertTrue(report['first_snapshot_failure']['actor_backward'])
        # None/presence remains strict even for retained storage.
        missing = dict(actual); missing.pop('grad.actor.1.none'); missing['grad.actor.1'] = torch.ones(1)
        with self.assertRaises(AssertionError):
            profile._compare_td3_block_snapshot(expected, missing, context=context, report={})
        # Do not hide weight, optimizer, target, count or statistic changes.
        for key in ('actor.weight', 'optimizer.actor.moment', 'actor_target.weight',
                    'counts.actor_update_count', 'metrics.critic_loss', 'grad.critic1.0'):
            before = {**expected, key: torch.tensor(1.)}
            after = {**actual, key: torch.tensor(2.)}
            with self.assertRaises(AssertionError):
                profile._compare_td3_block_snapshot(before, after, context=context, report={})
        # Across adjacent critic-only steps, actor state must be exactly unchanged.
        before = {**expected, 'actor.weight': torch.tensor(1.)}
        after = {**actual, 'actor.weight': torch.tensor(1.000001)}
        with self.assertRaises(AssertionError):
            profile._compare_td3_block_snapshot(before, after,
                context={**context, 'comparison': 'adjacent_steps'}, report={})

    def test_inf_injection_calls_original_and_restores_for_actor_and_critic(self):
        for component in ('actor', 'critic'):
            engine = self.engine()
            original = engine._validate_and_clip_gradients
            module = engine.actor if component == 'actor' else engine.critic1
            # First parameter has no gradient: inject into an actual existing gradient.
            module.bias.grad = torch.ones_like(module.bias)
            before = deepcopy(module.state_dict())
            evidence = {}
            with self.assertRaisesRegex(FloatingPointError, 'Non-finite ' + component):
                with profile._inject_diagnostic_inf_gradient(engine, component=component, evidence=evidence):
                    engine._validate_and_clip_gradients(list(module.parameters()), max_norm=1.,
                        component=component, total_steps=75002)
            self.assertIs(engine._validate_and_clip_gradients, original)
            self.assertEqual(evidence['hits'], 1)
            self.assertTrue(evidence['confirmed_inf'])
            self.assertEqual(evidence['parameter_name'], ('actor' if component == 'actor' else 'critic1') + '.bias')
            self.assertIsNone(module.weight.grad)
            for name, value in module.state_dict().items(): torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_real_backward_injection_preserves_protected_state(self):
        from test_v2_merged_update_blocks import TestMergedUpdateBlocks
        fixture = TestMergedUpdateBlocks()
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for model in ('ann', 'snn'):
                for component in ('actor', 'critic'):
                    engine = fixture.engine(model)
                    before = deepcopy(engine.checkpoint_state_dict())
                    evidence = {}
                    with self.assertRaisesRegex(FloatingPointError, 'Non-finite ' + component):
                        with profile._inject_diagnostic_inf_gradient(engine, component=component, evidence=evidence):
                            engine.update_once(total_steps=75002)
                    after = deepcopy(engine.checkpoint_state_dict())
                    fixture.compare(before['actor_state_dict'], after['actor_state_dict'])
                    fixture.compare(before['actor_optimizer_state_dict'], after['actor_optimizer_state_dict'])
                    self.assertEqual(after['update_counts']['actor_update_count'], 0)
                    if component == 'critic':
                        fixture.compare(before, after)
                    else:
                        # Existing update_count/critic_update_count advance only after the actor path succeeds.
                        self.assertEqual(after['update_counts']['critic_update_count'], 0)
                        self.assertTrue(all(float(state['step']) == 1. for state in
                            after['critic_optimizer_state_dict']['state'].values()))
                        self.assertEqual(after['update_counts']['critic_target_update_count'], 1)
                    self.assertEqual(evidence['hits'], 1)
                    self.assertTrue(evidence['confirmed_inf'])
                    self.assertNotIn('_validate_and_clip_gradients', engine.__dict__)
        finally:
            torch.set_num_threads(threads)

    def test_injection_not_reached_is_rejected_and_restored(self):
        engine = self.engine(); original = engine._validate_and_clip_gradients
        with self.assertRaisesRegex(AssertionError, 'never reached'):
            with profile._inject_diagnostic_inf_gradient(engine, component='actor', evidence={}): pass
        self.assertIs(engine._validate_and_clip_gradients, original)

    def test_critic_probe_distinct_storage_target_mode_and_repeated_callable(self):
        batch = V2ObservationBatch(torch.zeros(2, td3.EGO_FEATURE_DIM),
            torch.zeros(2, td3.GOAL_FEATURE_DIM), torch.zeros(2, 2, td3.ZONE_FEATURE_DIM),
            torch.ones(2, 2, dtype=torch.bool))
        parameter = torch.nn.Parameter(torch.ones(()))
        calls, target_calls = [], []
        def target(obs, **kwargs):
            target_calls.append(torch.is_grad_enabled())
            return torch.zeros(2, 2)
        def front(current, following, actions, target_input, reward, done, success):
            for a, b in zip(current, following):
                self.assertNotEqual(a.data_ptr(), b.data_ptr())
            self.assertEqual(len({t.data_ptr() for t in (reward, done, success)}), 3)
            self.assertFalse(target_input.requires_grad)
            calls.append((current, following))
            return None, None, parameter.square(), None, None
        engine = SimpleNamespace(device=torch.device('cpu'), action_dim=2,
            policy_noise=.1, noise_clip=.2, model_type='snn', actor_target=target,
            action_low=torch.full((2,), -1.), action_high=torch.ones(2),
            _mark_cuda_graph_step_begin=lambda: None, _build_shared_relations=lambda obs: None,
            critic1=SimpleNamespace(zone_set_encoder=SimpleNamespace(
                prepare_tensor_forward_arguments=lambda *args, **kw: args)),
            _compiled_critic_front=front, _require_finite_loss=lambda loss, **kw: None,
            critic_optimizer=torch.optim.SGD([parameter], lr=.1))
        probe = td3.V2TD3UpdateEngine._prepare_update_cuda_graph_verification_scope(engine, batch, 'critic_update_front')
        for _ in range(4):
            probe.prepare_inputs()
            probe()
        self.assertEqual(len(calls), 4)
        self.assertEqual(target_calls, [False] * 4)
        self.assertEqual(parameter.item(), 1.)

    def test_actor_probe_context_outside_profile_cannot_supply_replay_evidence(self):
        batch = V2ObservationBatch(torch.zeros(2, td3.EGO_FEATURE_DIM),
            torch.zeros(2, td3.GOAL_FEATURE_DIM), torch.zeros(2, 1, td3.ZONE_FEATURE_DIM),
            torch.ones(2, 1, dtype=torch.bool))
        observed = False
        auxiliary_launches = 0
        context_calls, objective_calls = [], []
        critic = torch.nn.Module()
        critic.head = torch.nn.Linear(3, 1)
        critic.head.weight.requires_grad_(False)  # Restore mixed original flags too.
        context_buffer = torch.ones(2, 1)
        def encode_context(*args, **kwargs):
            nonlocal auxiliary_launches
            context_calls.append(observed)
            self.assertFalse(torch.is_grad_enabled())
            if observed: auxiliary_launches += 1
            return context_buffer
        critic.encode_context = encode_context
        def objective(module, context, actions, *args):
            self.assertTrue(all(not p.requires_grad for p in module.head.parameters()))
            self.assertFalse(context.requires_grad)
            self.assertNotEqual(context.data_ptr(), context_buffer.data_ptr())
            objective_calls.append(observed)
            return (module.head(torch.cat((context, actions), dim=1)).sum(),)
        cpu_engine = SimpleNamespace(device=torch.device('cpu'), action_dim=2,
            _mark_cuda_graph_step_begin=lambda: None, _build_shared_relations=lambda obs: None,
            critic1=critic, frozen_critic_strategy='compiled_no_grad_context',
            action_low=-torch.ones(2), action_high=torch.ones(2), actor_rl_scale_alpha=2.5,
            terminal_geo_lambda=0., terminal_geo_radius=250., terminal_geo_regularization_enabled=False,
            actor=SimpleNamespace(scales=SimpleNamespace(horizontal_span=2625., vertical_span=599.9, gamma_max=.6)),
            model_type='ann', spike_regularization=td3.V2SpikeRegularizationConfig(),
            _compiled_actor_objective=objective)
        cpu_engine._actor_critic_guidance = lambda *a, **kw: td3.V2TD3UpdateEngine._actor_critic_guidance(cpu_engine, *a, **kw)
        probe = td3.V2TD3UpdateEngine._prepare_update_cuda_graph_verification_scope(cpu_engine, batch, 'actor_objective')
        def observe(replay):
            nonlocal observed
            observed = True
            try: replay()
            finally: observed = False
            return dict(cuda_graph_launch_count=auxiliary_launches if replay is probe else 1, event_names=[])
        verifier = SimpleNamespace(cuda_graph_updates=True, device=SimpleNamespace(type='cuda'),

            critic1=critic, critic2=torch.nn.Linear(2, 1),
            _compiled_target_block=None, _compiled_target_critic_td=None,
            _compiled_actor_loss=None, cuda_graph_actor_update=False,
            update_count=0, critic_update_count=0, critic_target_update_count=0,
            actor_update_count=0, last_total_steps=0,
            _actor_graph_snn_memory_snapshot=lambda: [], warmup_full_compile=lambda batches: None,
            _prepare_update_cuda_graph_verification_scope=lambda batch, scope: probe if scope == 'actor_objective' else lambda: None,
            _profile_cuda_graph_replay=observe)
        with mock.patch('torch.cuda.get_rng_state_all', return_value=[]), mock.patch('torch.cuda.set_rng_state_all'):
            with self.assertRaises(RuntimeError) as failure:
                td3.V2TD3UpdateEngine.verify_update_cuda_graph_capture(verifier, [batch])
        self.assertEqual(context_calls, [False] * 4)
        self.assertEqual(objective_calls, [False, False, False, True])
        self.assertEqual(auxiliary_launches, 0)
        evidence = failure.exception.cuda_graph_evidence['scope_evidence']['actor_objective']
        self.assertFalse(evidence['verified'])
        self.assertEqual(evidence['shapes'][0]['cuda_graph_launch_count'], 0)
        self.assertEqual([p.requires_grad for p in critic.head.parameters()], [False, True])
        self.assertTrue(all(p.grad is None for p in critic.head.parameters()))

    def test_same_callable_warmup_and_missing_replay_evidence(self):
        batch = SimpleNamespace(batch_size=2, max_zone_count=1)
        counts, profiled = {}, []
        def prepare(batch, scope):
            token = object(); counts[token] = 0
            def replay(): counts[token] += 1
            replay.token = token
            replay.scope = scope
            replay.prepare_inputs = lambda: None
            return replay
        def observe(replay):
            profiled.append((replay.scope, counts[replay.token]))
            replay()
            return dict(cuda_graph_launch_count=0 if replay.scope == 'critic_update_front' else 1, event_names=[])
        engine = SimpleNamespace(cuda_graph_updates=True, device=SimpleNamespace(type='cuda'),

            critic1=torch.nn.Linear(2, 1), critic2=torch.nn.Linear(2, 1),
            _compiled_target_block=None, _compiled_target_critic_td=None,
            _compiled_actor_loss=None, cuda_graph_actor_update=False,
            update_count=0, critic_update_count=0, critic_target_update_count=0,
            actor_update_count=0, last_total_steps=0,
            _actor_graph_snn_memory_snapshot=lambda: [], warmup_full_compile=lambda batches: None,
            _prepare_update_cuda_graph_verification_scope=prepare, _profile_cuda_graph_replay=observe)
        with mock.patch('torch.cuda.get_rng_state_all', return_value=[]), mock.patch('torch.cuda.set_rng_state_all'):
            with self.assertRaises(RuntimeError) as failure:
                td3.V2TD3UpdateEngine.verify_update_cuda_graph_capture(engine, [batch])
        self.assertIn(('critic_update_front', 3), profiled)
        evidence = failure.exception.cuda_graph_evidence
        self.assertFalse(evidence['verified'])
        self.assertEqual(evidence['status'], 'replay_not_observed')
        self.assertEqual(evidence['scope_evidence']['critic_update_front']['shapes'][0]['cuda_graph_launch_count'], 0)


if __name__ == '__main__': unittest.main()
