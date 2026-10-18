"""CPU/mock record propagation checks; no CUDA or experiment acceptance."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
import torch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.scripts import train_v2_td3 as stage
from brain_uav.trainers import v2_reporting as reporting, v2_validation as validation
from brain_uav.trainers.v2_formal_training import (
    V2FormalStageTrainer, V2FormalTrainingConfig, load_v2_formal_checkpoint,
)
from test_v2_formal_training import _engine, _Source, _scenario_payload, _validation_result
from test_v2_validation import _pool
from test_run_v2_td3_curriculum import _prepared_initialization


class TestTerminalGoalDistance(TestCase):
    def trainer(self, *, reporter=None, budget=1, episode_steps=1, goal=True):
        scenario = ScenarioConfig(max_steps=episode_steps, goal_radius=.2)
        engine = _engine(scenario)
        payload = _scenario_payload(0)
        payload['goal'] = [1.25 if goal else 100., .125, 100.]
        trainer = V2FormalStageTrainer(scenario, RewardConfig(),
            V2FormalTrainingConfig(stage='easy', max_steps=budget, replay_capacity=16,
                                  batch_size=2, warmup_steps=10), engine,
            scenario_sources={'easy': _Source(payload)},
            validation_runner=lambda actor: _validation_result(False), reporter=reporter)
        engine.select_action = mock.Mock(return_value=np.zeros(2, dtype=np.float32))
        return trainer, scenario, engine

    def test_training_selects_last_segment_for_goal_and_endpoint_for_each_failure(self):
        for outcome in ('goal', 'collision', 'boundary', 'ground', 'timeout'):
            with self.subTest(outcome=outcome):
                trainer, _, _ = self.trainer(goal=outcome == 'goal')
                original = trainer.env.step
                last_info = {}
                def step(action):
                    obs, reward, terminated, truncated, info = original(action)
                    if outcome != 'goal':
                        info = dict(info, outcome=outcome, segment_goal_distance=.01)
                    last_info.update(info)
                    return obs, reward, outcome != 'timeout', outcome == 'timeout', info
                with mock.patch.object(trainer.env, 'step', side_effect=step):
                    result = trainer.run()
                record = result.to_dict()['episodes'][0]
                expected = last_info['segment_goal_distance' if outcome == 'goal' else 'goal_distance']
                self.assertEqual(record['terminal_goal_distance'], expected)
                self.assertNotEqual(last_info['goal_distance'], last_info['segment_goal_distance'])
                if outcome == 'goal':
                    self.assertGreater(expected, 0.)
                self.assertEqual(record['terminal_position'], last_info['terminal_position'])
                self.assertEqual(result.outcome_counts[outcome], 1)
                self.assertFalse(result.passed_validation)

    def test_budget_cutoff_does_not_add_terminal_record(self):
        trainer, _, _ = self.trainer(budget=1, episode_steps=10, goal=False)
        result = trainer.run()
        self.assertEqual(result.episodes, [])
        self.assertEqual(sum(result.outcome_counts.values()), 0)
        self.assertFalse(result.passed_validation)
        self.assertEqual(result.stop_reason, 'max_steps_without_validation')

    def test_training_records_reach_jsonl_metrics_stage_end_and_old_checkpoint_loads(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            trainer, scenario, engine = self.trainer()
            reporter = reporting.V2ExperimentReporter(root/'episodes_report', stage='easy', model_type='ann',
                scenario=scenario, rewards=RewardConfig(), uav_collision_radius=0., max_steps=1)
            stack.callback(reporter.close)
            trainer.reporter = reporter
            stack.enter_context(mock.patch.object(reporting, 'export_v2_trajectory_views'))
            stack.enter_context(mock.patch.object(reporting, '_plot_training_windows'))
            result = trainer.run()
            reporter.close()
            rows = [json.loads(line) for line in (root/'episodes_report/episodes.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['terminal_goal_distance'], result.episodes[0]['terminal_goal_distance'])
            prepared = _prepared_initialization(scenario, actor=engine.actor, source=root/'bc.pt')
            components = SimpleNamespace(engine=engine, scenario_generators={}, selector=None,
                exploration_rng=np.random.default_rng(7), seed_manifest={'base_seed':7},
                initialization_source={'kind':'v2_bc_best','path':str(root/'bc.pt')})
            stack.enter_context(mock.patch.object(stage, 'prepare_v2_stage_initialization', return_value=prepared))
            stack.enter_context(mock.patch.object(stage, 'load_v2_validation_pool', return_value=SimpleNamespace(
                scenario_count=100, curriculum_level='easy', master_seed=20260904, stage_seed=11, content_digest='fixture')))
            stack.enter_context(mock.patch.object(stage, 'build_v2_stage_engine', return_value=components))
            stack.enter_context(mock.patch.object(stage, '_configure_stage_compilation', return_value={}))
            stack.enter_context(mock.patch.object(stage, 'V2FormalStageTrainer', return_value=SimpleNamespace(
                run=lambda:result, execution_configuration={'cpu_deduplication': {}})))
            summary = stage.run_v2_td3_stage(stage='easy', init_checkpoint=root/'bc.pt', output=root/'model.pt',
                metrics_out=root/'metrics.json', validation_pool=root/'pool.json', device='cpu', max_stage_steps=1)
            metrics = json.loads((root/'metrics.json').read_text())
            end = json.loads((root/'metrics_reports/stage_end.json').read_text())
            for records in (metrics['result']['episodes'], end['episodes']):
                self.assertEqual(records, result.to_dict()['episodes'])
            checkpoint = load_v2_formal_checkpoint(Path(summary['checkpoint']))
            self.assertIn('terminal_goal_distance', checkpoint['training_result']['episodes'][0])
            legacy = deepcopy(checkpoint)
            del legacy['training_result']['episodes'][0]['terminal_goal_distance']
            loaded = load_v2_formal_checkpoint(legacy)
            self.assertNotIn('terminal_goal_distance', loaded['training_result']['episodes'][0])

    def test_validation_results_and_files_keep_distance_with_or_without_reporter(self):
        scenario = ScenarioConfig(max_steps=1, goal_radius=.2)
        actor = _engine(scenario).actor
        for parameter in actor.parameters():
            parameter.data.zero_()
        pool = _pool('easy', scenario, (1.25, 100.))
        scenarios = deepcopy(pool.scenarios)
        for record in scenarios:
            record['payload']['goal'][1] = .125
        pool = replace(pool, scenarios=scenarios)
        results = []
        for enabled in (False, True):
            with self.subTest(reporter=enabled), TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                reporter = reporting.V2ExperimentReporter(root/'report', stage='easy', model_type='ann',
                    scenario=scenario, rewards=RewardConfig(), uav_collision_radius=0., max_steps=1) if enabled else None
                if reporter is not None:
                    stack.callback(reporter.close)
                    reporter.prepare_validation_candidate(1, global_steps=17)
                stack.enter_context(mock.patch.object(reporting, 'export_v2_trajectory_views'))
                result = validation.evaluate_v2_fixed_validation(actor, pool, RewardConfig(), reporter=reporter,
                                                               max_failures=1).to_dict()
                results.append(result)
                goal, timeout = result['scenarios']
                self.assertEqual(goal['outcome'], 'goal')
                self.assertAlmostEqual(goal['terminal_goal_distance'], .125)
                endpoint = np.asarray(goal['terminal_position'])
                self.assertGreater(np.linalg.norm(endpoint-np.asarray(pool.scenarios[0]['payload']['goal'])), .2)
                self.assertEqual(timeout['outcome'], 'timeout')
                expected = np.linalg.norm(np.asarray(timeout['terminal_position'])-np.asarray(pool.scenarios[1]['payload']['goal']))
                self.assertAlmostEqual(timeout['terminal_goal_distance'], expected, places=5)
                self.assertEqual(result['goal_count'], 1)
                self.assertEqual(result['failure_count'], 1)
                self.assertTrue(result['passed'])
                if reporter is not None:
                    reporter.close()
                    summaries = list((root/'report').rglob('summary.json'))
                    self.assertEqual(len(summaries), 1)
                    self.assertEqual(json.loads(summaries[0].read_text())['scenarios'], result['scenarios'])
                    rows = [json.loads(line) for line in next((root/'report').rglob('scenarios.jsonl')).read_text().splitlines()]
                    self.assertEqual([r['terminal_goal_distance'] for r in rows],
                                     [r['terminal_goal_distance'] for r in result['scenarios']])
        self.assertEqual(results[0], results[1])
