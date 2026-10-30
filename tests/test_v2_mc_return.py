"""Local CPU checks for opt-in behavior-trajectory return supervision."""
import unittest
from copy import deepcopy
from dataclasses import replace
from unittest import mock
from tempfile import TemporaryDirectory
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
import torch
import test_v2_formal_training as fixtures
import test_v2_replay_buffer as replay_fixtures
from brain_uav.config import ScenarioConfig, RewardConfig
from brain_uav.trainers.v2_replay_buffer import V2ReplayBuffer
from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
from brain_uav.trainers import v2_formal_training as formal
from brain_uav.trainers.v2_gpu_compute import GPUCompute, tensors, clone_batch
from brain_uav.trainers.v2_gpu_update import GPUUpdate


def engine(weight=.1, **overrides):
    base = fixtures._engine(ScenarioConfig())
    kwargs = dict(actor=base.actor, critic1=base.critic1, critic2=base.critic2, replay=base.replay,
        actor_lr=.001, critic_lr=.001, gamma=.5, tau=.005, policy_noise=.015, noise_clip=.03,
        policy_delay=2, batch_size=2, action_low=base.action_low, action_high=base.action_high,
        target_q_aggregation='mean', mc_return_weight=weight, mc_return_max_age_steps=100000,
        actor_freeze_steps=25000, critic_grad_clip_norm=1., actor_grad_clip_norm=1.)
    kwargs['bc_reference_actor'] = base.bc_reference_actor
    kwargs.update(overrides)
    return V2TD3UpdateEngine(**kwargs)


def populated_engine(weight=.1):
    e = engine(weight)
    from test_v2_td3 import _observation
    for i in range(2):
        obs = _observation(i, scales=e.actor.scales)
        slot = e.replay.add(obs, [0., 0.], i+1., obs, i == 1,
            **(dict(collected_stage_step=i+1) if weight else {}))
        if weight: e.replay.fill_episode_returns([slot], [3.5-i])
    return e

from brain_uav.trainers.v2_mc_return import (
    validate_mc_return_config, discounted_episode_returns, critic_loss_components,
)


class TestMCReturn(unittest.TestCase):
    def test_ring_generation_success_copy_and_fresh_age_mask(self):
        replay = V2ReplayBuffer(2, 2, 2, success_replay_fraction=1., success_batch_fraction=.5, seed=7)
        replay.enable_mc_returns()
        o = replay_fixtures._observation(0)
        first = replay.add(o, [0, 0], 0., o, False, collected_stage_step=1)
        second = replay.add(o, [0, 0], -2., o, True, collected_stage_step=2)
        replacement = replay.add(o, [0, 0], 3., o, False, collected_stage_step=3)
        replay.fill_episode_returns([first, second], [0., -2.])
        self.assertFalse(replay.mc_valid[first[0]])
        self.assertEqual(replay.collected_stage_step[first[0]], 3)
        # Success copy uses episode metadata even though first primary slot was overwritten.
        replay.add_success_transition(o, [0, 0], 0., o, True, collected_stage_step=1, mc_return=0.)
        self.assertTrue(replay.success_mc_valid[0]); self.assertEqual(replay.success_mc_return[0], 0.)
        self.assertEqual(replay.success_collected_stage_step[0], 1)
        replay.fill_episode_returns([replacement], [3.])
        # Force deterministic indices; no additional random sampling for MC.
        with mock.patch.object(replay, '_sample_primary_indices', return_value=np.array([1])):
            for step, expected in ((2, [1., 1.]), (100001, [1., 1.]), (100002, [1., 0.]), (1, [0., 1.])):
                b = replay.sample(2, current_stage_step=step)
                self.assertEqual(b.mc_return_mask.flatten().tolist(), expected)
                self.assertEqual(b.reward.flatten().tolist(), [-2., 0.])
        replay.add(o, [0, 0], 9., o, False, collected_stage_step=10)
        self.assertFalse(replay.mc_valid[second[0]])
        self.assertEqual(replay.mc_return[second[0]], 0.)

    def test_exact_age_boundaries(self):
        r = V2ReplayBuffer(2, 2, 0, success_batch_fraction=0.)
        r.enable_mc_returns(); o = replay_fixtures._observation(0)
        slot = r.add(o, [0, 0], 0, o, True, collected_stage_step=5)
        r.fill_episode_returns([slot], [0.])
        for age, expected in ((-1, 0), (0, 1), (99999, 1), (100000, 1), (100001, 0)):
            b = r.sample(1, current_stage_step=5+age)
            self.assertEqual(b.mc_return_mask.item(), expected)

    def test_mc_metadata_preserves_sampling_rng_and_source_independence(self):
        left, right = [V2ReplayBuffer(8, 2, 2, seed=7, success_replay_fraction=.5,
            success_batch_fraction=.5, failure_sample_bias=2.) for _ in range(2)]
        left.enable_mc_returns()
        refs = []
        for i in range(4):
            obs = replay_fixtures._observation(i % 3, offset=i)
            refs.append(left.add(obs, [0, 0], i, obs, i == 3, collected_stage_step=i+1))
            right.add(obs, [0, 0], i, obs, i == 3)
            if i < 2:
                left.add_success_transition(obs, [0, 0], i, obs, True, collected_stage_step=i+1, mc_return=-i)
                right.add_success_transition(obs, [0, 0], i, obs, True)
        left.fill_episode_returns(refs, [0., -1., -2., -3.])
        for step in (4, 100002, 100005):
            a = left.sample(4, current_stage_step=step)
            b = right.sample(4)
            for value, expected in zip(tensors(a)[:14], tensors(b)):
                self.assertTrue(torch.equal(value, expected))
            self.assertEqual(left.rng.bit_generator.state, right.rng.bit_generator.state)
        old_g = left.mc_return.copy(); old_mask = left.mc_valid.copy()
        a.mc_return.fill_(123.); a.mc_return_mask.zero_()
        np.testing.assert_array_equal(left.mc_return, old_g)
        np.testing.assert_array_equal(left.mc_valid, old_mask)

    def test_compiled_update_matches_eager_without_extra_forwards(self):
        a, b = populated_engine(), populated_engine()
        b.load_checkpoint_state_dict(deepcopy(a.checkpoint_state_dict()))
        b._compiled_critic_loss = torch.compile(b._compute_twin_critic_loss_tensors, backend='eager', fullgraph=True)
        batch = a._sample_training_batch(2)
        state = deepcopy(b.critic1.state_dict()); rng = torch.get_rng_state().clone()
        b.warmup_full_compile((batch.obs,))
        for name, value in state.items(): self.assertTrue(torch.equal(value, b.critic1.state_dict()[name]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(b.update_count, 0)
        with mock.patch.object(a, '_sample_training_batch', return_value=batch), \
                mock.patch.object(b, '_sample_training_batch', return_value=batch), \
                mock.patch('torch.randn_like', new=lambda t: torch.zeros_like(t)):
            left = a.update_once(total_steps=2)
            right = b.update_once(total_steps=2)
        self.assertEqual(left, right)
        for net in ('critic1', 'critic2'):
            for p, q in zip(getattr(a, net).parameters(), getattr(b, net).parameters()):
                self.assertTrue(torch.equal(p, q)); self.assertTrue(torch.equal(p.grad, q.grad))
        # Loss helper only consumes the already-computed Qs, never calls a network.
        with mock.patch.object(a.critic1, '_compute_full_forward_tensors', wraps=a.critic1._compute_full_forward_tensors) as first, \
                mock.patch.object(a.critic2, '_compute_full_forward_tensors', wraps=a.critic2._compute_full_forward_tensors) as second:
            u = GPUCompute.__new__(GPUCompute); u.e = a
            a._compute_twin_critic_loss_tensors(*u.arguments(batch.obs), batch.action, batch.reward,
                                               batch.mc_return, batch.mc_return_mask)
        self.assertEqual((first.call_count, second.call_count), (1, 1))

    def test_eager_compiled_shared_loss_and_graph_inputs(self):
        e = populated_engine()
        b = e._sample_training_batch(2)
        u = GPUCompute.__new__(GPUCompute); u.e = e
        args = u.arguments(b.obs)
        target = torch.tensor([[1.], [2.]])
        eager = e._compute_twin_critic_loss_tensors(*args, b.action, target, b.mc_return, b.mc_return_mask)
        compiled = torch.compile(e._compute_twin_critic_loss_tensors, backend='eager', fullgraph=True)
        result = compiled(*args, b.action, target, b.mc_return, b.mc_return_mask)
        for a, value in zip(eager, result): self.assertTrue(torch.equal(a, value))
        self.assertTrue(torch.equal(result[-1], critic_loss_components(*result[:2], target, .1, b.mc_return, b.mc_return_mask)[0]))
        for zones in (0, 1, 2):
            obs = replay_fixtures._observation(zones)
            r = V2ReplayBuffer(2, 2, 2); r.enable_mc_returns()
            slot = r.add(obs, [0, 0], 1., obs, True, collected_stage_step=1)
            r.fill_episode_returns([slot], [2.])
            b = r.sample(1, current_stage_step=1).to('cpu')
            stable = clone_batch(b)
            self.assertEqual(len(tensors(stable)), 16)
            for a, value in zip(tensors(b), tensors(stable)):
                self.assertTrue(torch.equal(a, value)); self.assertNotEqual(a.data_ptr(), value.data_ptr()) if a.numel() else None
            # Same zip-copy used by the real executor, new runtime mask values.
            next_batch = r.sample(1, current_stage_step=100002)
            for dst, src in zip(tensors(stable), tensors(next_batch)): dst.copy_(src)
            self.assertEqual(stable.mc_return_mask.item(), 0.)
        before = GPUUpdate.static_options(e)
        e.mc_return_max_age_steps += 1
        self.assertNotEqual(before, GPUCompute.static_options(e))
        with self.assertRaisesRegex(ValueError, 'requires'):
            u.key(replace(b, mc_return=None, mc_return_mask=None), 'critic_only')

    def test_mc_inputs_capture_restore_and_graph_key_are_runtime(self):
        import test_v2_gpu_compute as gpu_fixtures
        e = populated_engine()
        sample = e.replay.sample
        e.replay.sample = lambda size: sample(size, current_stage_step=2)
        # Run the real capture/restore function with its existing CPU graph fixture.
        with mock.patch.object(gpu_fixtures.TestGPUCompute, 'engine', return_value=e):
            gpu_fixtures.TestGPUCompute().test_actual_capture_restore_cpu_mock_preserves_values_none_and_masks()
        b = e.replay.sample(2)
        u = GPUCompute.__new__(GPUCompute)
        u.e = e; u.zone_limit = e.replay.zone_storage_capacity
        u.bound_device = torch.device('cpu'); u.published = [None, None]
        key = u.key(b, 'critic_only')
        b.mc_return.fill_(99.); b.mc_return_mask.zero_()
        self.assertEqual(u.key(b, 'critic_only'), key)

    def test_snn_mc_update_and_snapshot_strict_diagnostic_loading(self):
        import test_v2_spike_regularization as snn_fixtures
        old = snn_fixtures.TestSpikeRegularization().engine(delay=2)
        e = V2TD3UpdateEngine(old.actor, old.critic1, old.critic2, old.replay,
            .001, .001, old.gamma, old.tau, old.policy_noise, old.noise_clip, 2, 2,
            old.action_low, old.action_high, target_q_aggregation='mean', mc_return_weight=.1,
            terminal_geo_regularization_enabled=False, spike_regularization=old.spike_regularization)
        for i in range(len(e.replay)): e.replay.collected_stage_step[i] = i+1
        e.replay.fill_episode_returns([(i, int(e.replay.write_id[i])) for i in range(len(e.replay))],
                                     [1.] * len(e.replay))
        m = e.update_once(total_steps=75000)
        self.assertTrue(m.actor_updated); self.assertEqual(m.mc_valid_count, 2.)
        self.assertEqual(m.spike_beta_eff, 1.)
        self.assertAlmostEqual(m.critic_loss, m.critic_td_loss + .1*m.critic_mc_loss, places=5)
        cfg = formal.V2FormalTrainingConfig(stage='easy', target_q_aggregation='mean', mc_return_weight=.1,
            spike_regularization=e.spike_regularization, terminal_geo_regularization_enabled=False)
        payload = formal.build_v2_periodic_snapshot(e, cfg, stage_steps=300123, scenario=ScenarioConfig(),
            rewards=RewardConfig(), uav_collision_radius=0., seed_manifest={'base_seed': 7},
            initialization_source={'kind': 'v2_bc_best', 'path': 'bc.pt', 'model_type': 'snn'})
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pt'; formal.save_v2_periodic_snapshot(path, payload)
            from brain_uav.scripts.diagnose_v2_policy import load_diagnostic_checkpoint
            self.assertEqual(load_diagnostic_checkpoint(path)['formal_config']['mc_return_weight'], .1)

    def test_default_off_sampling_rng_update_and_old_engine_checkpoint(self):
        a, b = populated_engine(0.), populated_engine(0.)
        saved = deepcopy(a.checkpoint_state_dict())
        saved['algorithm_config'].pop('mc_return_weight'); saved['algorithm_config'].pop('mc_return_max_age_steps')
        b.load_checkpoint_state_dict(saved)
        self.assertFalse(a.replay.mc_return_enabled)
        rng = torch.get_rng_state()
        left = a.update_once(total_steps=2)
        torch.set_rng_state(rng)
        right = b.update_once(total_steps=2)
        self.assertEqual(left, right)
        self.assertEqual(a.replay.rng.bit_generator.state, b.replay.rng.bit_generator.state)
        for p, q in zip(a.critic1.parameters(), b.critic1.parameters()):
            self.assertTrue(torch.equal(p, q)); self.assertTrue(torch.equal(p.grad, q.grad))
        self.assertIsNone(left.critic_mc_loss)

    def test_formal_completion_and_budget_tail_preserve_order(self):
        for goal, env_steps, budget in ((True, 1, 1), (False, 2, 2), (False, 10, 2)):
            e = engine()
            scenario = ScenarioConfig(max_steps=env_steps)
            cfg = formal.V2FormalTrainingConfig(stage='easy', max_steps=budget, batch_size=2,
                target_q_aggregation='mean', mc_return_weight=.1, gamma=.5)
            trainer = formal.V2FormalStageTrainer(scenario, RewardConfig(), cfg, e,
                scenario_sources={'easy': fixtures._Source(fixtures._scenario_payload(0, goal=goal))},
                validation_runner=lambda actor: self.fail('unexpected fixed validation'))
            order = []
            original_fill, original_update = e.replay.fill_episode_returns, e.update_once
            def fill(*a): order.append('fill'); return original_fill(*a)
            def update(**kw):
                order.append('update')
                self.assertFalse(e.replay.mc_valid[:len(e.replay)].any())
                return original_update(**kw)
            with mock.patch.object(e.replay, 'fill_episode_returns', side_effect=fill), mock.patch.object(e, 'update_once', side_effect=update):
                result = trainer.run()
            if env_steps <= budget:
                self.assertTrue(e.replay.mc_valid[:len(e.replay)].all())
                rewards = e.replay.reward[:len(e.replay), 0].tolist()
                np.testing.assert_array_equal(e.replay.mc_return[:len(e.replay)], np.array(discounted_episode_returns(rewards, .5), dtype=np.float32))
                if 'update' in order: self.assertLess(order.index('update'), order.index('fill'))
                self.assertEqual(len(result.episodes), 1)
            else:
                self.assertFalse(e.replay.mc_valid.any()); self.assertEqual(result.episodes, [])
            if goal:
                self.assertEqual(e.replay.success_collected_stage_step[0], 1)
                self.assertEqual(e.replay.success_mc_return[0], e.replay.mc_return[0])

    def test_mc_update_metrics_and_missing_fields(self):
        e = populated_engine()
        m = e.update_once(total_steps=2)
        self.assertEqual(m.mc_valid_count, 2.)
        self.assertAlmostEqual(m.critic_loss, m.critic_td_loss + .1 * m.critic_mc_loss, places=5)
        m = e.update_once(total_steps=100003)
        self.assertEqual(m.mc_valid_count, 0.); self.assertEqual(m.critic_mc_loss, 0.)
        self.assertEqual(m.critic_loss, m.critic_td_loss)
        self.assertTrue(m.critic_updated)
        e.replay.mc_return_enabled = False
        with self.assertRaisesRegex(ValueError, 'storage'): e.update_once(total_steps=100004)

    def test_real_gpu_shared_compute_cpu_matches_eager_three_modes(self):
        # Real compute_batch arithmetic, CPU optimizer adapter; no CUDA claim.
        for mode, step, freeze in (('critic_only', 1, 4), ('frozen_targets', 2, 4), ('actor', 2, 0)):
            with self.subTest(mode=mode):
                e, other = populated_engine(), populated_engine()
                e.actor_freeze_steps = other.actor_freeze_steps = freeze
                other.load_checkpoint_state_dict(deepcopy(e.checkpoint_state_dict()))
                b = e._sample_training_batch(step)
                u = GPUCompute.__new__(GPUCompute)
                u.e = other; u.diagnostic = True; u.compute_diagnostic = True; u.tensor_evidence = {}
                u.actor_params = list(other.actor.parameters())
                u.critic_params = list(other.critic1.parameters()) + list(other.critic2.parameters())
                u.initialize_actor_loss_scalars()
                class CPUAdam:
                    def __init__(self, opt): self.opt = opt
                    def step(self, error):
                        if not bool(error): self.opt.step()
                u.adams = (CPUAdam(other.actor_optimizer), CPUAdam(other.critic_optimizer))
                u.injections = torch.zeros(4); u.noise = torch.zeros_like(b.action)
                u.coefficients = torch.tensor([0., 0., other.policy_noise, other.noise_clip])
                u.failure_fraction = torch.tensor(b.sample_failure_fraction)
                with mock.patch.object(e, '_sample_training_batch', return_value=b), \
                        mock.patch('torch.randn_like', new=lambda t: torch.zeros_like(t)):
                    expected = e.update_once(total_steps=step)
                values = u.compute(b, mode)
                m = u.metrics(values[:-2].detach().tolist(), step, 0., mode != 'critic_only', mode == 'actor')
                for name in ('critic_loss', 'critic_td_loss', 'critic_mc_loss', 'mc_valid_count', 'mc_valid_fraction'):
                    self.assertAlmostEqual(getattr(expected, name), getattr(m, name), places=5)
                for name in ('actor', 'critic1', 'critic2', 'actor_target', 'critic1_target', 'critic2_target'):
                    for a, value in zip(getattr(e, name).parameters(), getattr(other, name).parameters()):
                        torch.testing.assert_close(a, value, rtol=1e-4, atol=1e-5)
                        self.assertEqual(a.grad is None, value.grad is None)
                        if a.grad is not None: torch.testing.assert_close(a.grad, value.grad, rtol=1e-4, atol=1e-5)
                for a, value in zip(e.critic_optimizer.state.values(), other.critic_optimizer.state.values()):
                    for key in a: torch.testing.assert_close(a[key], value[key], rtol=1e-4, atol=1e-5)
                self.assertEqual(len(values), 29)  # existing status/metrics + five MC + two bank publication flags

    def test_formal_snapshot_legacy_strict_loading_and_configuration(self):
        e = populated_engine()
        config = formal.V2FormalTrainingConfig(stage='easy', target_q_aggregation='mean', mc_return_weight=.1, gamma=e.gamma)
        result = formal.V2FormalTrainingResult.empty('easy', global_steps_start=0)
        result.status = 'failed'; result.stop_reason = 'max_steps'
        common = dict(scenario=ScenarioConfig(), rewards=RewardConfig(), uav_collision_radius=0.,
            seed_manifest={'base_seed': 7}, initialization_source={'kind': 'v2_bc_best', 'path': 'bc.pt'})
        build_formal = lambda cfg: formal.build_v2_formal_checkpoint(e, result, cfg,
            validation_pool_metadata=dict(path='pool.json', format_version=1, curriculum_level='easy',
                master_seed=20260904, stage_seed=1, scenario_count=100, content_digest='digest'), **common)
        with self.assertRaisesRegex(ValueError, 'MC return'): build_formal(replace(config, mc_return_weight=0.))
        for payload, load in ((build_formal(config), formal.load_v2_formal_checkpoint),
                (formal.build_v2_periodic_snapshot(e, config, stage_steps=300123, **common), formal.load_v2_periodic_snapshot)):
            with TemporaryDirectory() as directory:
                path = Path(directory) / 'snapshot.pt'; torch.save(payload, path)
                from brain_uav.scripts.diagnose_v2_policy import load_diagnostic_checkpoint
                restored = load_diagnostic_checkpoint(path)
                self.assertEqual(restored['formal_config']['mc_return_weight'], .1)
                self.assertEqual(restored['engine_checkpoint']['algorithm_config']['mc_return_max_age_steps'], 100000)
            for name, bad_value in (('mc_return_weight', True), ('mc_return_weight', float('nan')),
                                    ('mc_return_max_age_steps', 1.5), ('mc_return_max_age_steps', False)):
                bad = deepcopy(payload)
                bad['formal_config'][name] = bad_value
                bad['engine_checkpoint']['algorithm_config'][name] = bad_value
                with self.assertRaises(ValueError): load(bad)
            bad = deepcopy(payload); bad['formal_config']['mc_return_weight'] = .2
            with self.assertRaisesRegex(ValueError, 'MC return'): load(bad)
            legacy = deepcopy(payload)
            for values in (legacy['formal_config'], legacy['engine_checkpoint']['algorithm_config']):
                values.pop('mc_return_weight'); values.pop('mc_return_max_age_steps')
            self.assertEqual(load(legacy)['formal_config']['mc_return_weight'], 0.)
        state = deepcopy(e.checkpoint_state_dict())
        with mock.patch.object(e, 'invalidate_gpu_update', wraps=e.invalidate_gpu_update) as invalidate:
            e.load_checkpoint_state_dict(state)
            invalidate.assert_called()
        bad = deepcopy(state); bad['algorithm_config']['mc_return_weight'] = 0.
        with self.assertRaisesRegex(ValueError, 'algorithm_config'): e.load_checkpoint_state_dict(bad)
        with self.assertRaisesRegex(ValueError, 'legacy gpu_td3_update'): engine(gpu_td3_update=True)
        with self.assertRaisesRegex(ValueError, 'explicit'): engine(target_q_aggregation='min')

    def test_cli_api_and_actual_builder_propagation(self):
        import inspect
        from brain_uav.scripts import train_v2_td3 as single, run_v2_td3_curriculum as course
        import test_run_v2_td3_curriculum as course_fixtures
        for fn in (single.run_v2_td3_stage, course.run_v2_curriculum):
            self.assertEqual(inspect.signature(fn).parameters['mc_return_weight'].default, 0.)
            self.assertEqual(inspect.signature(fn).parameters['mc_return_max_age_steps'].default, 100000)
            self.assertEqual(inspect.signature(fn).parameters['target_q_aggregation'].default, 'min')
        extra = ['--target-q-aggregation', 'mean', '--mc-return-weight', '.1', '--mc-return-max-age-steps', '100000']
        with redirect_stdout(StringIO()), mock.patch.object(single, 'run_v2_td3_stage', return_value={'passed': True}) as run:
            single.main(['--stage', 'easy', '--init-checkpoint', 'bc.pt', '--output', 'o.pt',
                '--metrics-out', 'm.json', '--validation-pool', 'pool.json', '--device', 'cpu', *extra])
        self.assertEqual(run.call_args.kwargs['mc_return_weight'], .1)
        self.assertEqual(run.call_args.kwargs['mc_return_max_age_steps'], 100000)
        with redirect_stdout(StringIO()), mock.patch.object(course, 'run_v2_curriculum', return_value={'passed': True}) as run:
            course.main(['--bc-checkpoint', 'bc.pt', '--output-root', 'out', '--validation-pool-dir', 'pool', '--device', 'cpu', *extra])
        self.assertEqual(run.call_args.kwargs['mc_return_weight'], .1)
        with TemporaryDirectory() as directory:
            root = Path(directory); bc = root / 'bc.pt'; bc.write_bytes(b'fixture')
            prepared = course_fixtures._prepared_initialization(ScenarioConfig(), source=bc)
            cfg = formal.V2FormalTrainingConfig(stage='easy', target_q_aggregation='mean', mc_return_weight=.1,
                                                batch_size=2, replay_capacity=8)
            components = formal.build_v2_stage_engine(ScenarioConfig(), cfg, init_checkpoint=bc, device='cpu', prepared_initialization=prepared)
            self.assertEqual(components.engine.mc_return_weight, .1)
            self.assertTrue(components.engine.replay.mc_return_enabled)
            calls = []
            def stage(**kw):
                calls.append(kw)
                return dict(stage=kw['stage'], passed=True, checkpoint=str(kw['output']), steps=1, global_steps_end=1)
            from types import SimpleNamespace
            with mock.patch.object(course, 'prepare_v2_stage_initialization', return_value=prepared), \
                    mock.patch.object(course, 'prepare_v2_validation_pools', return_value={s: root / f'{s}.json' for s in ('easy', 'medium', 'hard')}), \
                    mock.patch.object(course, 'load_v2_validation_pool', return_value=SimpleNamespace(master_seed=20260904,
                        stage_seed=1, scenario_count=100, content_digest='digest')):
                result = course.run_v2_curriculum(bc_checkpoint=bc, output_root=root/'run', validation_pool_dir=root/'pool',
                    target_q_aggregation='mean', mc_return_weight=.1, mc_return_max_age_steps=100000,
                    device='cpu', stage_runner=stage)
            self.assertEqual([c['mc_return_weight'] for c in calls], [.1]*3)
            self.assertEqual([c['mc_return_max_age_steps'] for c in calls], [100000]*3)
            self.assertEqual(result['mc_return_weight'], .1)

    def test_returns_and_validation(self):
        self.assertEqual(discounted_episode_returns([1, 2, 3], .5), [2.75, 3.5, 3.])
        self.assertEqual(discounted_episode_returns([0, -2], .5), [-1., -2.])
        for rewards, gamma in (([float('nan')], .5), ([1], float('inf')), ([1], -1)):
            with self.assertRaises(ValueError): discounted_episode_returns(rewards, gamma)
        for weight in (True, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): validate_mc_return_config(weight, 100000, 'mean')
        for age in (True, 0, -1, 1.5, '100'):
            with self.assertRaises(ValueError): validate_mc_return_config(.1, age, 'mean')
        with self.assertRaises(ValueError): validate_mc_return_config(.1, 100000, 'min')
        for kwargs in (dict(mc_return_weight=True), dict(mc_return_weight=float('nan')),
                       dict(mc_return_max_age_steps=1.5), dict(mc_return_max_age_steps=False)):
            with self.assertRaises(ValueError): engine(**kwargs) if 'mc_return_weight' not in kwargs else engine(weight=kwargs['mc_return_weight'])
            with self.assertRaises(ValueError): formal.V2FormalTrainingConfig(stage='easy', target_q_aggregation='mean', **kwargs)

    def test_loss_and_gradients(self):
        q1 = torch.tensor([[2.], [10.]], requires_grad=True)
        q2 = torch.tensor([[4.], [20.]], requires_grad=True)
        y = torch.tensor([[1.], [2.]], requires_grad=True)
        g = torch.tensor([[3.], [100.]], requires_grad=True)
        total, td, mc, count, fraction = critic_loss_components(q1, q2, y, .1, g, torch.tensor([[1.], [0.]]))
        self.assertEqual(td.item(), 199.)
        self.assertEqual(mc.item(), 2.)
        self.assertAlmostEqual(total.item(), 199.2, places=4)
        self.assertEqual((count.item(), fraction.item()), (1., .5))
        total.backward()
        self.assertIsNotNone(q1.grad); self.assertIsNotNone(q2.grad)
        self.assertIsNone(y.grad); self.assertIsNone(g.grad)
        a = torch.tensor([[4500.]])
        b = torch.tensor([[5500.]])
        self.assertEqual(critic_loss_components(a, b, a, .1, torch.tensor([[5000.]]), torch.ones(1, 1))[2].item(), 500000.)
        self.assertEqual(critic_loss_components(a, b, a, .1, torch.zeros(1, 1), torch.zeros(1, 1))[2].item(), 0.)
        self.assertEqual(critic_loss_components(q1, q2, y, 0.)[0].item(), 199.)


if __name__ == '__main__': unittest.main()
