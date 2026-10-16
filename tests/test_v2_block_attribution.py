"""CPU measurement contracts only, no CUDA performance claims."""
import ast
from pathlib import Path
from copy import deepcopy
import unittest
from unittest import mock

import torch

from brain_uav.scripts import profile_v2_td3 as profile
from brain_uav.trainers import v2_td3 as td3
import test_v2_td3 as td3_fixture
from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig


class TestBlockAttribution(unittest.TestCase):
    def test_disabled_clock_and_aggregation_reset(self):
        with mock.patch.object(td3, 'perf_counter', side_effect=AssertionError('clock sampled')):
            timer = td3._OptionalUpdateDetailTimer(None, False, names=('merged_bindings',))
            with timer.section('merged_bindings'):
                pass
            timer.finish()
        summary = profile._BlockAttributionSummary()
        with mock.patch.object(td3, 'perf_counter', side_effect=[1., 1.25, 2., 2.5]):
            timer = td3._OptionalUpdateDetailTimer(summary.record, False, names=('merged_bindings',))
            for _ in range(2):
                with timer.section('merged_bindings'):
                    pass
            timer.finish()
        region = summary.report(True)['regions']['merged_bindings']
        self.assertEqual(region['calls'], 2)
        self.assertEqual(region['cpu_wall_seconds'], .75)
        self.assertEqual(region['mean_cpu_wall_seconds_per_call'], .375)
        summary.reset()  # warmup boundary
        self.assertEqual(summary.report(True)['updates'], 0)
        self.assertEqual(summary.report(True)['regions'], {})

    def test_compute_labels_exclude_auxiliary_context_and_forward(self):
        tree = ast.parse(Path(td3.__file__).read_text())
        regions = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            for item in node.items:
                expr = item.context_expr
                if (isinstance(expr, ast.Call) and expr.args
                        and isinstance(expr.args[0], ast.Constant)
                        and expr.args[0].value in ('actor_q_head_objective', 'critic_equivalent_compute')):
                    regions.append((expr.args[0].value, ast.unparse(node)))
        actor = [body for label, body in regions if label == 'actor_q_head_objective']
        self.assertEqual(len(actor), 3)
        for body in actor:
            self.assertNotIn('self.actor(', body)
            self.assertNotIn('bc_reference_actor(', body)
            self.assertNotIn('prepare_tensor_forward_arguments', body)
            self.assertNotIn('guidance.__enter__', body)
        self.assertTrue(any('objective(' in body for body in actor))
        self.assertTrue(any('forward_from_context' in body for body in actor))
        for label, body in regions:
            if label == 'critic_equivalent_compute':
                self.assertNotIn('_build_shared_relations', body)
                self.assertNotIn('prepare_tensor_forward_arguments', body)
                self.assertNotIn('self.actor_target(', body)  # SNN actor outside; ANN inside compiled target block

    def assert_tree_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_tree_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_post_measurement_noise_uses_each_profile_step(self):
        config = V2FormalTrainingConfig(stage='easy', max_steps=500000)
        for step in (75001, 75002, 75003):
            expected = config.noise_schedule.values(step, max_steps=config.max_steps)[1:]
            self.assertEqual(profile._post_measurement_target_noise(config, step, 75000), expected)
        with mock.patch.object(type(config.noise_schedule), 'values', side_effect=AssertionError('schedule called')):
            self.assertEqual(profile._post_measurement_target_noise(config, 123, 0),
                (config.noise_schedule.policy_initial, config.noise_schedule.clip_initial))
        source = Path(profile.__file__).read_text()
        self.assertIn('_post_measurement_target_noise(\n                    formal_config, profile_total_steps, controlled_stage_step_offset)', source)

    def test_recording_does_not_change_update(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            fixture = td3_fixture.TestV2TD3()
            fixture.setUp()
            a, b = (fixture.make_engine(policy_delay=1) for _ in range(2))
            b.load_checkpoint_state_dict(deepcopy(a.checkpoint_state_dict()))
            fixture.fill_replay(a)
            batch = a.replay.sample(a.batch_size)
            for engine in (a, b):
                engine.replay.sample = lambda size: batch
            rng = torch.random.get_rng_state()
            metric_a = a.update_once(total_steps=50000)
            torch.random.set_rng_state(rng)
            summary = profile._BlockAttributionSummary()
            metric_b = b.update_once(total_steps=50000, block_attribution_recorder=summary.record)
            self.assertTrue(metric_a.actor_updated)
            self.assertEqual(metric_a, metric_b)
            self.assert_tree_equal(a.checkpoint_state_dict(), b.checkpoint_state_dict())
            for name in ('actor', 'critic1', 'critic2', 'actor_target', 'critic1_target', 'critic2_target'):
                for first, second in zip(getattr(a, name).parameters(), getattr(b, name).parameters()):
                    self.assert_tree_equal(first.grad, second.grad)
            self.assertEqual(summary.updates, 1)
            self.assertEqual(summary.calls['merged_bindings'], 0)
            self.assertEqual(summary.calls['critic_equivalent_backward'], 1)
            self.assertEqual(summary.calls['actor_q_head_objective'], 2)
        finally:
            torch.set_num_threads(previous)

    def test_window_callback_and_post_measurement_profiler_settings(self):
        source = Path(profile.__file__).read_text()
        self.assertIn('block_attribution.record if block_attribution_wall and measured_phase and not profile_sections else None', source)
        self.assertIn('profile_total_steps = controlled_stage_step_offset + step_number + offset + 1', source)
        self.assertIn('bc_lambda=v2_bc_lambda(total_steps - 1, schedule=bc_schedule)', source)

    def test_cli_default_off_and_rejects_unrelated_mode(self):
        args = profile.build_parser().parse_args(['--model', 'snn', '--bc-checkpoint', 'unused.pt', '--output-dir', 'unused'])
        with self.assertRaises(SystemExit):
            profile.main(['--model', 'snn', '--bc-checkpoint', 'unused.pt', '--output-dir', 'unused',
                '--td3-block-attribution', 'wall'])


if __name__ == '__main__':
    unittest.main()
