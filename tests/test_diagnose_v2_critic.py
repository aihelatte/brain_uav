"""Tiny CPU fixtures only; no production twenty-scene diagnostic."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from brain_uav.config import RewardConfig
from brain_uav.models import V2SNNPolicyActor
from brain_uav.observations import V2ObservationScales
from brain_uav.scripts import diagnose_v2_critic as d
from brain_uav.trainers.v2_formal_training import save_v2_formal_checkpoint
from test_diagnose_v2_policy import write_inputs
from test_v2_static_no_fly_env import _config, _scenario


class TestCriticDiagnostic(unittest.TestCase):
    def test_discount_includes_candidate_and_stops_at_real_terminal(self):
        self.assertEqual(d.discounted_return([10., 4., -2.], .5), 11.5)
        self.assertEqual(d.discounted_return([7.], .99), 7.)
        self.assertEqual(d.discounted_return([], .99), 0.)
        with self.assertRaises(ValueError): d.discounted_return([float('nan')], .99)
        with self.assertRaises(ValueError): d.discounted_return([1.], 2.)

    def test_selection_exact_empty_count_unique_seeded_and_ordered(self):
        records = [{'scenario_id': str(i), 'sequence_index': i,
            'payload': {'zones': [] if i%14==0 else [1]}} for i in range(98)]
        pool = SimpleNamespace(scenarios=records)
        a = d.select_scenarios(pool)
        self.assertEqual(a, d.select_scenarios(pool))
        self.assertEqual(len(a), 20)
        self.assertEqual(sum(not r['payload']['zones'] for r in a), 7)
        self.assertEqual(len({r['scenario_id'] for r in a}), 20)
        self.assertEqual([r['sequence_index'] for r in a], sorted(r['sequence_index'] for r in a))
        with self.assertRaises(ValueError): d.select_scenarios(SimpleNamespace(scenarios=records[1:]))

    def test_normalized_perturbations_clip_and_exact_dedup(self):
        limit = np.array([.14, .2], np.float32)
        actions, proposals = d.candidate_actions(limit, limit, 'A', limit)
        self.assertLessEqual(len(actions), 6)
        self.assertEqual(len(actions), 3)
        self.assertEqual(proposals[1]['candidate_index'], 0)
        self.assertTrue(proposals[1]['deduplicated'])
        self.assertTrue(any(p['clipped'] for p in proposals))
        np.testing.assert_array_equal(np.array(actions[1]['action'], np.float32), limit * np.array([.75,1], np.float32))
        np.testing.assert_array_equal(np.array(actions[2]['action'], np.float32), limit * np.array([1,.75], np.float32))
        self.assertTrue(d.near_tie(1., 1.+1e-5))
        self.assertFalse(d.near_tie(1., 1.1))

    def test_ranking_reverse_and_all_ties_with_own_baseline_bias(self):
        rows = [{'candidate_index': i, 'Q1': q, 'G': g, 'outcome': outcome, 'critic_disagreement': 2.}
            for i,(q,g,outcome) in enumerate(((3.,1.,'collision'),(2.,2.,'goal'),(1.,3.,'timeout')))]
        result = d.ranking(rows, 1)
        self.assertEqual(result['pair_counts']['reversed'], 3)
        self.assertEqual(result['effective_non_tied_pairs'], 3)
        self.assertEqual(result['G_regret_at_Q1_best'], 2.)
        self.assertEqual(result['own_Q1_minus_G'], 0.)
        for r in rows: r.update(Q1=1., G=1.)
        result = d.ranking(rows, 0)
        self.assertEqual(result['pair_counts']['both_tied'], 3)
        self.assertEqual(result['pair_counts']['reversed'], 0)
        self.assertEqual(result['effective_non_tied_pairs'], 0)
        self.assertEqual(result['ranking_information'], 'insufficient')
        self.assertEqual(result['G_regret_range_over_Q1_ties'], [0.,0.])

    def test_primary_q_ranking_and_best_ties_invariant_to_common_shift(self):
        for q_values, own_index in (([1.,0.], 0), ([1., 1.+2**-11, 0.], 0),
                                     ([1., 1.+2**-11, 0.], 2)):
            rows = [{'candidate_index': i, 'Q1': q, 'G': i*100., 'outcome': 'timeout',
                     'critic_disagreement': 0.} for i,q in enumerate(q_values)]
            original = d.ranking(rows, own_index)
            for shift in (-100000., 100000.):
                shifted = deepcopy(rows)
                for row in shifted: row['Q1'] += shift
                result = d.ranking(shifted, own_index)
                for key in ('pair_counts', 'effective_non_tied_pairs', 'ranking_information',
                            'Q1_best_candidate', 'Q1_best_tie_candidates', 'G_regret_at_Q1_best',
                            'G_regret_range_over_Q1_ties', 'relative_to_own'):
                    self.assertEqual(result[key], original[key], (q_values, own_index, shift, key))
                self.assertEqual(result['own_Q1_minus_G'], original['own_Q1_minus_G'] + shift)
            if len(rows)==2:
                self.assertEqual(original['pair_counts']['reversed'], 1)
                self.assertEqual(original['G_regret_at_Q1_best'], 100.)
            else:
                self.assertEqual(original['Q1_best_tie_candidates'], [0,1])

    def test_summary_exposes_effective_pairs_all_ties_and_information_gap(self):
        rows = [{'candidate_index': i, 'Q1': 1.-i, 'G': i*100., 'outcome': 'timeout',
                 'critic_disagreement': 0.} for i in range(2)]
        reversed_result = d.ranking(rows, 0)
        for row in rows: row.update(Q1=0., G=0.)
        tied_result = d.ranking(rows, 0)
        probes = [{'empty_scene': True, 'rankings': {'A': tied_result, 'B': tied_result}}]
        summary = d.compact_summary([], probes, [])
        self.assertEqual(summary['A']['all']['effective_non_tied_pairs'], 0)
        self.assertEqual(summary['A']['all']['probes_without_effective_pairs'], 1)
        self.assertEqual(summary['A']['all']['ranking_information'], 'insufficient')
        self.assertEqual(summary['A']['all']['pair_counts_descriptive_not_independent_samples'],
                         dict(concordant=0,reversed=0,Q_tied=0,G_tied=0,both_tied=1))
        probes.append({'empty_scene': False, 'rankings': {'A': reversed_result, 'B': reversed_result}})
        summary = d.compact_summary([], probes, [])
        self.assertEqual(summary['A']['all']['effective_non_tied_pairs'], 1)
        self.assertEqual(summary['A']['all']['probes_without_effective_pairs'], 1)
        self.assertEqual(summary['A']['all']['pair_counts_descriptive_not_independent_samples']['reversed'], 1)
        self.assertEqual(summary['A']['empty']['ranking_information'], 'insufficient')

    def tiny_data(self):
        scenario = _config(max_steps=5)
        return {'scenario': scenario, 'rewards': RewardConfig(), 'gamma': .5,
            'checkpoint_payload': {'formal_config': {'seed': 23}},
            'pool': SimpleNamespace(uav_collision_radius=0.), 'device': torch.device('cpu')}

    def test_prefix_restores_history_remaining_time_and_branch_independence(self):
        data = self.tiny_data()
        record = {'payload': _scenario([0.,0.,10.,0.,0.], [60.,0.,10.], [])}
        env = d.make_env(data)
        actions = [np.array([.01,.02], np.float32), np.array([-.01,.01], np.float32)]
        obs, state = d.reset_replay(env, record, actions)
        expected = d.digest(state)
        self.assertEqual(state['steps'], 2)
        self.assertEqual(len(state['recent_progress']), 2)
        np.testing.assert_array_equal(state['prev_action'], actions[-1])
        other = d.make_env(data)
        obs2, state2 = d.reset_replay(other, record, actions, expected)
        self.assertEqual(state, state2)
        with mock.patch.object(d, 'actor_action', return_value=np.zeros(2, np.float32)):
            a = d.fly(data, env, obs, first_action=np.zeros(2, np.float32))
            b = d.fly(data, other, obs2, first_action=np.zeros(2, np.float32))
        self.assertEqual(a, b)
        self.assertEqual(a['episode_length'], 3)
        self.assertEqual(a['final_steps'], 5)
        self.assertEqual(a['outcome'], 'timeout')
        self.assertEqual(a['discounted_return'], d.discounted_return(a['rewards'], .5))
        fresh = d.make_env(data)
        _, after = d.reset_replay(fresh, record, actions, expected)
        self.assertEqual(after, state)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            d.reset_replay(fresh, record, actions, 'bad')
        with self.assertRaisesRegex(ValueError, 'terminated'):
            d.reset_replay(fresh, record, actions + [np.zeros(2)]*5)

    def test_strict_reconstruction_rejects_critic_weights_buffers_and_input_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ckpt, pool_path, payload = write_inputs(root)
            loaded = d.load_models(ckpt, pool_path, 'cpu')
            self.assertFalse(loaded['actor'].training)
            for i, critic in enumerate(loaded['critics'],1):
                self.assertFalse(critic.training)
                self.assertTrue(all(not p.requires_grad for p in critic.parameters()))
                for key, tensor in critic.state_dict().items():
                    self.assertTrue(torch.equal(tensor, payload['engine_checkpoint'][f'critic{i}_state_dict'][key]))
            altered = deepcopy(loaded)
            altered['checkpoint_payload']['reward_config']['goal_bonus'] = 12345
            with self.assertRaisesRegex(ValueError, 'reward_config'): d.compare_inputs(loaded, altered)
            for target, buffer in (('critic1_state_dict', False), ('critic2_state_dict', True), ('actor_state_dict', False)):
                bad = deepcopy(payload)
                state = bad['engine_checkpoint'][target]
                key = (next(iter(dict(loaded['critics'][1].named_buffers()))) if buffer
                       else next(k for k in state if state[k].is_floating_point() and state[k].numel()>0))
                state[key].flatten()[0] = float('nan') if not buffer else 999.
                path = root/f'{target}_{buffer}.pt'
                torch.save(bad, path)
                with self.assertRaises(ValueError): d.load_models(path, pool_path, 'cpu')
            with self.assertRaises(FileExistsError):
                d.run(checkpoint_a=ckpt, checkpoint_b=ckpt, validation_pool=pool_path, output_dir=root)

    def test_snn_forward_repeated_calls_reset_state(self):
        from test_v2_snn_td3 import _observation
        scales = V2ObservationScales(100.,0.,50.,.6)
        actor = V2SNNPolicyActor(scales,2,8,torch.tensor([.14,.2]),time_window=2).eval().requires_grad_(False)
        data = {'actor':actor,'device':torch.device('cpu')}
        a = d.actor_action(data, _observation(0,scales))
        d.actor_action(data, _observation(2,scales))
        b = d.actor_action(data, _observation(0,scales))
        np.testing.assert_array_equal(a,b)
        for node in (actor.snn_head.lif1,actor.snn_head.lif2):
            self.assertEqual(float(torch.count_nonzero(node.v)) if isinstance(node.v,torch.Tensor) else node.v,0.)

    def test_tiny_pipeline_persists_common_probe_q_and_complete_own_returns(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ckpt, pool_path, _ = write_inputs(root)
            a = d.load_models(ckpt, pool_path, 'cpu')
            b = d.load_models(ckpt, pool_path, 'cpu')
            record = a['pool'].scenarios[1]  # timeout in two steps, not production twenty-scene run
            output = root/'out'
            with mock.patch.object(d, 'load_v2_validation_pool', return_value=SimpleNamespace(
                    content_digest=d.POOL_DIGEST, scenario_count=100)), \
                    mock.patch.object(d, 'select_scenarios', return_value=[record]), \
                    mock.patch.object(d, 'load_models', side_effect=[a,b]), \
                    mock.patch('torch.optim.Adam.step', side_effect=AssertionError('no optimizer step')), \
                    mock.patch.object(torch.Tensor, 'backward', side_effect=AssertionError('no backward')):
                result = d.run(checkpoint_a=ckpt, checkpoint_b=ckpt, validation_pool=pool_path, output_dir=output)
            self.assertTrue(result['execution_completed'])
            self.assertEqual(result['baseline_episodes'], 2)
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['selection']['scenarios'][0]['scenario_id'], record['scenario_id'])
            self.assertEqual(manifest['inputs']['A']['status'], 'failed')
            branches = [json.loads(line) for line in (output/'branches.jsonl').read_text().splitlines()]
            self.assertLessEqual(len(branches),12)
            for row in branches:
                self.assertEqual(row['G'], d.discounted_return(row['continuation']['rewards'], a['gamma']))
                self.assertEqual(row['continuation']['final_steps'], 2)
            self.assertTrue(any(r['baseline_suffix_reused'] for r in branches))
            self.assertNotIn('passed', result)

    def test_bad_pool_refused_and_runtime_error_preserves_incomplete_report(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'out'
            kwargs = dict(checkpoint_a='a', checkpoint_b='b', validation_pool='pool', output_dir=output)
            with mock.patch.object(d, 'load_v2_validation_pool', return_value=SimpleNamespace(content_digest='wrong')), \
                    mock.patch.object(d, 'load_models') as models:
                with self.assertRaisesRegex(ValueError, 'digest'): d.run(**kwargs)
                models.assert_not_called()
                self.assertFalse(output.exists())
            ckpt, pool_path, _ = write_inputs(root)
            a = d.load_models(ckpt, pool_path, 'cpu')
            record = a['pool'].scenarios[1]
            with mock.patch.object(d, 'load_v2_validation_pool', return_value=SimpleNamespace(content_digest=d.POOL_DIGEST, scenario_count=100)), \
                    mock.patch.object(d, 'select_scenarios', return_value=[record]), \
                    mock.patch.object(d, 'load_models', side_effect=[a,a]), \
                    mock.patch.object(d, 'actor_action', side_effect=ValueError('Non-finite actor action')):
                with self.assertRaisesRegex(ValueError, 'Non-finite'):
                    d.run(checkpoint_a=ckpt, checkpoint_b=ckpt, validation_pool=pool_path, output_dir=output)
            self.assertTrue((output/'manifest.json').is_file())
            result = json.loads((output/'summary.json').read_text())
            self.assertFalse(result['execution_completed'])
            self.assertEqual(result['error']['type'], 'ValueError')
            self.assertNotIn('passed', result)


if __name__ == '__main__':
    unittest.main()
