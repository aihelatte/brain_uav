"""Small CPU contracts for projected TD3 loss, schedules and logical counts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import torch
from brain_uav.models import V2ANNPolicyActor, V2ANNCritic, V2SNNPolicyActor
from brain_uav.models.v2_policy_config import V2SpikeRegularizationConfig, projection_from_architecture
from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine, _actor_loss_tensor_block
from brain_uav.utils.v2_operation_counts import count_v2_actor_operations, summarize_operation_counts
from brain_uav.observations import collate_v2_observations
import test_v2_snn_td3 as snn_fixture
from test_v2_ann import _observation


class TestSpikeRegularization(unittest.TestCase):
    def engine(self, beta=1., freeze=0, delay=1):
        fixture = snn_fixture.TestV2SNNTD3()
        fixture.setUp()
        old = fixture.make_engine()
        actor = V2SNNPolicyActor(old.actor.scales, 2, 8, old.actor.action_limit, projection_dim=4)
        with torch.no_grad():
            actor.snn_head.fc1.bias.fill_(3.)
        return V2TD3UpdateEngine(actor, old.critic1, old.critic2, old.replay,
            .001, .001, .99, .25, .01, .02, delay, 2,
            old.action_low, old.action_high, actor_freeze_steps=freeze,
            terminal_geo_regularization_enabled=False,
            spike_regularization=V2SpikeRegularizationConfig(beta_max=beta))

    def test_schedule_validation_and_stage_local_reset(self):
        config = V2SpikeRegularizationConfig(1.)
        self.assertEqual([config.beta(s) for s in (0, 25000, 50000, 75000, 500000)], [0., 0., .5, 1., 1.])
        self.assertEqual(config.beta(0), 0.)
        for kw in ({'beta_max':True}, {'beta_max':float('nan')}, {'beta_max':-1},
                   {'target_rate':1.1}, {'target_rate':float('inf')},
                   {'ramp_start_steps':True}, {'ramp_end_steps':25000}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                V2SpikeRegularizationConfig(**kw)
        with self.assertRaises(ValueError):
            config.validate_model('ann')
        V2SpikeRegularizationConfig().validate_model('ann')

    def test_actual_loss_global_rate_and_compiled_tensor_parity(self):
        engine = self.engine()
        batch = engine.replay.sample(2)
        compiled = torch.compile(_actor_loss_tensor_block, backend='eager', fullgraph=True)
        engine.cache_actor_loss_coefficients = True
        for steps in (25000, 50000, 75000):
            engine._compiled_actor_loss = None
            eager = engine._compute_actor_loss_terms(batch.obs, batch.line_to_goal_safe, bc_lambda=0., stage_steps=steps)
            gradients = torch.autograd.grad(eager.actor_loss, tuple(engine.actor.parameters()), allow_unused=True)
            engine._compiled_actor_loss = compiled
            other = engine._compute_actor_loss_terms(batch.obs, batch.line_to_goal_safe, bc_lambda=0., stage_steps=steps)
            other_gradients = torch.autograd.grad(other.actor_loss, tuple(engine.actor.parameters()), allow_unused=True)
            torch.testing.assert_close(eager.actor_loss, other.actor_loss)
            torch.testing.assert_close(eager.spike_loss, torch.relu(eager.spike_rate - .17).square())
            self.assertEqual(eager.spike_beta, engine.spike_regularization.beta(steps))
            for a, b in zip(gradients, other_gradients):
                if a is None:
                    self.assertIsNone(b)
                else:
                    torch.testing.assert_close(a, b)
        for rate in (.1, .17, .3):
            r = torch.tensor(rate, requires_grad=True)
            loss = torch.relu(r - .17).square()
            self.assertAlmostEqual(loss.item(), max(0., rate - .17)**2, places=7)
        # Average all spike slots before penalty; per-sample penalties differ.
        rates = torch.tensor([0., .3])
        self.assertEqual(torch.relu(rates.mean() - .17).square().item(), 0.)
        self.assertGreater(torch.relu(rates - .17).square().mean().item(), 0.)

    def test_beta_zero_update_equivalence_freeze_and_delay(self):
        torch.manual_seed(2)
        first = self.engine(beta=0., freeze=25000, delay=2)
        second = self.engine(beta=0., freeze=25000, delay=2)
        second.load_checkpoint_state_dict(deepcopy(first.checkpoint_state_dict()))
        second._compiled_actor_loss = torch.compile(_actor_loss_tensor_block, backend='eager', fullgraph=True)
        for step in (1, 25000, 25001, 75000):
            rng = torch.random.get_rng_state()
            a = first.update_once(total_steps=step)
            torch.random.set_rng_state(rng)
            b = second.update_once(total_steps=step)
            self.assertEqual(a.actor_updated, b.actor_updated)
            self.assertAlmostEqual(a.actor_loss, b.actor_loss, places=6)
            for name, weight in first.actor.state_dict().items():
                torch.testing.assert_close(weight, second.actor.state_dict()[name])
            if a.actor_updated:
                self.assertEqual(a.weighted_spike_regularization, 0.)
            else:
                self.assertIsNone(a.actor_batch_spike_rate_l1)
                self.assertIsNone(a.spike_beta_eff)
        self.assertEqual(first.actor_update_count, 1)

    def test_counts_projection_policy_head_and_finite_set_activity(self):
        engine = self.engine()
        batch = collate_v2_observations([_observation(0)])
        for cls in (V2ANNPolicyActor, V2SNNPolicyActor):
            actor = cls(engine.actor.scales, 2, 192, engine.actor.action_limit, projection_dim=32).eval()
            with torch.no_grad():
                if cls is V2SNNPolicyActor:
                    actor.snn_head.fc1.weight.zero_()
                    actor.snn_head.fc1.bias.fill_(10.)
            action, counts = count_v2_actor_operations(actor, batch)
            torch.testing.assert_close(action, actor(batch))
            self.assertEqual(counts['projection_macs'], 4096)
            self.assertEqual(counts['policy_head_macs'], 43392 if cls is V2ANNPolicyActor else 6528)
            self.assertEqual(counts['actor_contraction_ops'], 2*counts['macs'] + counts['acs'])
            if cls is V2SNNPolicyActor:
                self.assertEqual(counts['acs'], 4 * 192 * 192)
                self.assertEqual(counts['policy_head_ops'], 13056 + 147456)
                stats = summarize_operation_counts([counts, counts])
                self.assertEqual(stats['spike_rate_l1'], 1.)
                self.assertEqual(stats['evaluation_unobserved_neuron_fraction']['l1'], 0.)

    def test_new_engine_schema_round_trip_and_strict_failures(self):
        engine = self.engine()
        payload = engine.checkpoint_state_dict()
        self.assertEqual(payload['format_version'], 2)
        self.assertEqual(projection_from_architecture(payload['architecture']), 4)
        engine.load_checkpoint_state_dict(payload)
        for mutate in ('projection', 'version', 'missing'):
            bad = deepcopy(payload)
            if mutate == 'projection':
                bad['architecture']['policy_head']['projection_dim'] = 7
            elif mutate == 'version':
                bad['format_version'] = 1
            else:
                del bad['algorithm_config']['spike_regularization']
            with self.assertRaises(ValueError):
                engine.load_checkpoint_state_dict(bad)

    def test_legacy_engine_without_new_config_keeps_disabled_semantics(self):
        fixture = snn_fixture.TestV2SNNTD3()
        fixture.setUp()
        engine = fixture.make_engine()
        payload = deepcopy(engine.checkpoint_state_dict())
        payload['algorithm_config'].pop('spike_regularization')
        engine.load_checkpoint_state_dict(payload)
        self.assertEqual(engine.spike_regularization.beta_max, 0.)
        # Existing required fields remain required in the legacy path.
        payload['algorithm_config'].pop('gamma')
        with self.assertRaises(ValueError):
            engine.load_checkpoint_state_dict(payload)

    def test_episode_log_distinguishes_training_batch_from_terminal_sample(self):
        from brain_uav.config import ScenarioConfig, RewardConfig
        from brain_uav.trainers.v2_td3 import V2TD3UpdateMetrics
        from brain_uav.trainers.v2_formal_training import V2FormalStageTrainer, V2FormalTrainingConfig
        from brain_uav.trainers.v2_reporting import V2ExperimentReporter
        from test_v2_formal_training import _engine, _Source, _scenario_payload, _validation_result
        scenario = ScenarioConfig(max_steps=1)
        old = _engine(scenario)
        actor = V2SNNPolicyActor(old.actor.scales, 2, 8, old.actor.action_limit, projection_dim=4)
        engine = V2TD3UpdateEngine(actor, old.critic1, old.critic2, old.replay,
            .001, .001, .99, .005, .01, .02, 1, 1, old.action_low, old.action_high,
            bc_reference_actor=deepcopy(actor), spike_regularization=V2SpikeRegularizationConfig(1.))
        config = V2FormalTrainingConfig(stage='easy', max_steps=1, warmup_steps=0, batch_size=1,
            replay_capacity=8, actor_freeze_steps=0, spike_regularization=engine.spike_regularization)
        for updated in (False, True):
            metrics = V2TD3UpdateMetrics(actor_updated=updated,
                actor_batch_spike_rate_l1=.35 if updated else None,
                spike_regularization_loss=.0324 if updated else None,
                spike_beta_eff=.5 if updated else None,
                weighted_spike_regularization=.0162 if updated else None)
            with tempfile.TemporaryDirectory() as tmp:
                reporter = V2ExperimentReporter(Path(tmp), stage='easy', model_type='snn',
                    scenario=scenario, rewards=RewardConfig(), uav_collision_radius=0., max_steps=1)
                try:
                    trainer = V2FormalStageTrainer(scenario, RewardConfig(), config, engine,
                        scenario_sources={'easy':_Source(_scenario_payload(0, goal=True))},
                        validation_runner=lambda actor:_validation_result(False), reporter=reporter)
                    with (mock.patch.object(engine, 'update_once', return_value=metrics),
                          mock.patch.object(engine, 'select_action', return_value=old._action_high_cpu * 0)):
                        result = trainer.run()
                    reporter.close()
                    row = json.loads((Path(tmp)/'episodes.jsonl').read_text().strip())
                    self.assertEqual(row['actor_batch_spike_rate_l1'], .35 if updated else None)
                    self.assertEqual(row['weighted_spike_regularization'], .0162 if updated else None)
                    self.assertIsNotNone(row['episode_terminal_spike_rate_l1'])
                finally:
                    reporter.close()
