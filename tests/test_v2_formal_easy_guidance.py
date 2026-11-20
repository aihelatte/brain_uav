"""CPU checks for the formal easy climb warning and safe-direct objective."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brain_uav.config import RewardConfig
from brain_uav.geometry import NoFlyZone, Sphere
from brain_uav.models.v2_snn import V2SNNPolicyActor
from brain_uav.scripts import run_v2_formal_easy_baseline as run
from brain_uav.scripts import train_v2_td3 as single_stage
from brain_uav.scripts import v2_formal_easy_guidance as reporting
from brain_uav.trainers import v2_formal_training as formal
from brain_uav.trainers.v2_td3 import _actor_loss_tensor_block
from test_v2_static_no_fly_env import _config, _scenario, _v2_api
import test_v2_formal_training as fixture


class Guidance(unittest.TestCase):
    def env(self, state, goal, zones=(), *, speed=2.5, **reward):
        env_type, _, _ = _v2_api()
        env = env_type(_config(world_xy=1250., world_z_max=600., speed=speed),
                       RewardConfig(**reward), seed=17,
                       fixed_scenarios=[_scenario(state, goal, zones)])
        return env, env.reset()[0]

    def test_upper_warning_only_changes_upper_face(self):
        for gamma in (.3, 0., -.3):
            env, _ = self.env((0, 0, 510, gamma, 0), (500, 0, 510),
                              boundary_directional_warning_enabled=True,
                              upper_boundary_directional_warning_distance=120.)
            expected = 90. * (1 - 90. / 120.) ** 2 * max(
                0., np.sin(gamma) / np.sin(env.scenario.gamma_max))
            self.assertAlmostEqual(env._boundary_warning_penalty(env.state[:3]), expected, places=5)
        env, _ = self.env((1220, 0, 510, 0, 0), (500, 0, 510),
                          boundary_directional_warning_enabled=True,
                          upper_boundary_directional_warning_distance=120.)
        self.assertAlmostEqual(env._boundary_warning_penalty(env.state[:3]), 22.5)

    def test_pre_action_band_and_obstacle(self):
        def zone(center):
            return NoFlyZone('z', Sphere(center, 50.))

        cases = [
            ((0, 0, 200, 0, 0), (400, 0, 200), (), 1),
            ((0, 0, 200, 0, 0), (401, 0, 200), (), 2),
            ((0, 0, 200, 0, 0), (600, 0, 200), (zone((300, 200, 200)),), 2),
            ((0, 0, 200, 0, 0), (600, 0, 200), (zone((300, 0, 200)),), 0),
            ((400, 120, 200, 0, 0), (600, 0, 200), (zone((300, 0, 200)),), 1),
            ((0, 0, 599.9, .55, 0), (600, 0, 300), (), 0),
            ((0, 0, .2, -.55, 0), (600, 0, 300), (), 0),
        ]
        for state, goal, zones, expected in cases:
            with self.subTest(state=state, goal=goal, expected=expected):
                env, obs = self.env(state, goal, zones)
                before = env.state.copy()
                rng = deepcopy(env.rng.bit_generator.state)
                self.assertEqual(env.direct_guidance_band(obs, radius=400., clearance=40.), expected)
                np.testing.assert_array_equal(env.state, before)
                self.assertEqual(env.steps, 0)
                self.assertEqual(rng, env.rng.bit_generator.state)

    def test_unobstructed_line_but_reference_step_hits_zone_or_boundary(self):
        sphere = NoFlyZone('z', Sphere((20., 117., 200.), 24.))
        env, obs = self.env((0, 0, 200, 0, np.pi / 2), (600, 0, 200), (sphere,), speed=100.)
        self.assertTrue(env.line_to_goal_is_safe(clearance=40.))
        self.assertEqual(env.direct_guidance_band(obs, radius=400., clearance=40.), 0)
        _, _, _, _, info = env.step(np.array([0., -.2], dtype=np.float32))
        self.assertEqual(info['outcome'], 'collision')
        env, obs = self.env((1249.5, 0, 200, 0, 0), (600, 0, 200))
        self.assertEqual(env.direct_guidance_band(obs, radius=400., clearance=40.), 0)

    def test_independent_means_and_empty_masks(self):
        actions = torch.tensor([[1., 1.], [2., 2.], [2., 2.], [9., 9.]])
        ego = torch.zeros(4, 6)
        goal = torch.zeros(4, 6)
        low = torch.tensor([-.14, -.2])

        def loss(mask):
            return _actor_loss_tensor_block(actions, torch.zeros(4), None, ego, goal,
                torch.tensor(mask).reshape(-1, 1), low, -low, torch.tensor(2.5),
                torch.tensor(0.), torch.tensor(3000.), 2500., 600., .6, 400., True,
                far_weight=.5)[0].item()

        self.assertEqual(loss([1, 2, 2, 0]), 9000.)
        self.assertEqual(loss([1, 0, 0, 0]), 3000.)
        self.assertEqual(loss([0, 2, 2, 0]), 6000.)
        self.assertEqual(loss([0, 0, 0, 0]), 0.)
        compiled = torch.compile(_actor_loss_tensor_block, backend='eager', fullgraph=True)
        parameters = (actions.requires_grad_(), torch.zeros(4), None, ego, goal,
                      torch.tensor([[1.], [2.], [2.], [0.]]), low, -low,
                      torch.tensor(2.5), torch.tensor(0.), torch.tensor(3000.),
                      2500., 600., .6, 400., True)
        result = compiled(*parameters, far_weight=.5)[0]
        self.assertEqual(result.item(), 9000.)
        result.backward()
        self.assertEqual(actions.grad[-1].abs().sum().item(), 0.)

    def test_six_jobs_and_seed11_control(self):
        for batch, seeds in [('main', {7, 23}), ('seed11', {11})]:
            jobs = run.jobs(run.GUIDANCE_EXPERIMENT, batch)
            self.assertEqual({(j['model'], j['seed']) for j in jobs},
                             {(m, s) for m in ('ann', 'snn') for s in seeds})
            for job in jobs:
                spec, config = run.resolved(job, run.GUIDANCE_EXPERIMENT)
                self.assertEqual(config.terminal_geo_radius, 400.)
                self.assertEqual(config.terminal_geo_far_weight, .5)
                self.assertEqual(config.near_goal_radius, 250.)
                self.assertEqual(spec['reward_config']['upper_boundary_directional_warning_distance'], 120.)
        for job in run.jobs(run.GUIDANCE_CONTROL, 'seed11'):
            _, config = run.resolved(job, run.GUIDANCE_CONTROL)
            self.assertEqual(config.terminal_geo_radius, 250.)
            self.assertEqual(config.terminal_geo_far_weight, 0.)

    def test_plan_only_and_release_requires_the_job_set(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / 'absent'
            args = run.parser().parse_args([
                '--root', str(root), '--ann-bc', 'a', '--snn-bc', 's',
                '--validation-pool', 'p', '--experiment', run.GUIDANCE_EXPERIMENT,
            ])
            self.assertEqual(len(run.plan(args)['jobs']), 6)
            self.assertFalse(root.exists())
            with self.assertRaisesRegex(ValueError, 'batch workers'):
                run.release({}, root, None, run.GUIDANCE_EXPERIMENT, batch='seed11')

    def test_worker_checks_guidance_in_real_single_stage_entry(self):
        class ReachedInitialization(Exception):
            pass

        real_stage = single_stage.run_v2_td3_stage
        for job in run.jobs(run.GUIDANCE_EXPERIMENT, 'seed11'):
            for overrides in ({}, {'terminal_geo_radius': 250.}, {'terminal_geo_far_weight': 0.}):
                with self.subTest(model=job['model'], overrides=overrides), TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    args = run.parser().parse_args([
                        'worker', '--root', tmp, '--job', job['id'], '--batch', 'seed11',
                        '--experiment', run.GUIDANCE_EXPERIMENT, '--ann-bc', str(root / 'ann.pt'),
                        '--snn-bc', str(root / 'snn.pt'), '--validation-pool', str(root / 'pool.json'),
                    ])
                    paths = dict(ann_bc=str(args.ann_bc.resolve()), snn_bc=str(args.snn_bc.resolve()),
                                 pool=str(args.validation_pool.resolve()))
                    run.atomic_json(root / 'manifest.json', dict(
                        plan=run.plan(args), inputs={'paths': paths}, batch_id='guidance-test',
                    ))

                    def stage_with_wrong_argument(**kwargs):
                        return real_stage(**{**kwargs, **overrides})

                    inject = (patch.object(single_stage, 'run_v2_td3_stage', side_effect=stage_with_wrong_argument)
                              if overrides else nullcontext())
                    # Stop after the real consistency check, before the checkpoint or CUDA is opened.
                    with (patch.object(run, 'check_inputs'),
                          patch.object(single_stage, 'resolve_training_device', return_value='cuda'),
                          patch('brain_uav.trainers.v2_gpu_update.require_cuda_abi'),
                          patch.object(single_stage, 'prepare_v2_stage_initialization',
                                       side_effect=ReachedInitialization) as prepare,
                          inject):
                        if overrides:
                            with self.assertRaisesRegex(ValueError, 'Explicit formal config differs'):
                                run.worker(args)
                            prepare.assert_not_called()
                        else:
                            with self.assertRaises(ReachedInitialization):
                                run.worker(args)
                            prepare.assert_called_once()
                            config = prepare.call_args.args[0]
                            self.assertEqual((config.terminal_geo_radius, config.terminal_geo_far_weight), (400., .5))
                            self.assertEqual(prepare.call_args.kwargs['model_type'], job['model'])

    def test_ann_and_snn_formal_builder_trainer_replay_and_checkpoint(self):
        scenario = _config(world_xy=1250., world_z_max=600., goal_radius=410., max_steps=1)
        payload = _scenario((0, 0, 200, 0, 0), (401, 0, 200), (), level='easy')
        rewards = RewardConfig(boundary_directional_warning_enabled=True,
                               upper_boundary_directional_warning_distance=120.)
        for job in run.jobs(run.GUIDANCE_EXPERIMENT, 'seed11'):
            model = job['model']
            with self.subTest(model=model):
                _, config = run.resolved(job, run.GUIDANCE_EXPERIMENT)
                config = replace(config, replay_capacity=32, batch_size=2, max_steps=1)
                source = {}
                model_seed = formal.derive_v2_component_seed(config.seed, 'easy', 'model')
                actor = fixture._engine(scenario).actor
                if model == 'snn':
                    actor = V2SNNPolicyActor(actor.scales, 2, 8, actor.action_limit, time_window=4)
                prepared = formal.V2PreparedStageInitialization(
                    stage='easy', model_seed=model_seed, verified_initialization_source=source,
                    snn_time_window=4 if model == 'snn' else None, scenario_config=scenario,
                    reward_config=rewards, uav_collision_radius=0., model_type=model,
                    bc_initialization=formal.V2BCFormalInitialization(
                        actor=actor, scenario_config=scenario, uav_collision_radius=0., model_type=model),
                    formal_checkpoint=None, torch_rng_state=torch.get_rng_state().clone(),
                )
                engine = formal.build_v2_stage_engine(
                    scenario, config, init_checkpoint=source, device='cpu', model_type=model,
                    prepared_initialization=prepared,
                ).engine
                trainer = formal.V2FormalStageTrainer(
                    scenario, rewards, config, engine,
                    scenario_sources={'easy': fixture._Source(payload)},
                    validation_runner=lambda _: fixture._validation_result(False),
                )
                trainer.run()
                self.assertEqual(engine.replay.line_to_goal_safe[0], 2)
                self.assertEqual(engine.replay.success_line_to_goal_safe[0], 2)
                batch = engine.replay.sample(1)
                self.assertEqual(batch.line_to_goal_safe.item(), 2.)
                loss = engine._terminal_geo_loss(batch.obs, torch.ones(1, 2), batch.line_to_goal_safe)
                self.assertEqual(loss.item(), .5)
                snapshot = formal.build_v2_periodic_snapshot(
                    engine, config, stage_steps=1, scenario=scenario, rewards=rewards,
                    uav_collision_radius=0., seed_manifest={'base_seed': config.seed},
                    initialization_source={'kind': 'test'}, bc_schedule='smooth-mid-strong',
                )
                restored = formal.load_v2_periodic_snapshot(snapshot, expected_model_type=model)
                self.assertEqual(restored['formal_config']['terminal_geo_far_weight'], .5)
                self.assertEqual(restored['reward_config']['upper_boundary_directional_warning_distance'], 120.)
                tampered = deepcopy(restored)
                tampered['engine_checkpoint']['algorithm_config']['terminal_geo_far_weight'] = .25
                with self.assertRaisesRegex(ValueError, 'terminal_geo_far_weight'):
                    formal.load_v2_periodic_snapshot(tampered, expected_model_type=model)
                engine.load_checkpoint_state_dict(restored['engine_checkpoint'])
                terminal = formal.build_v2_formal_checkpoint(
                    engine, trainer.result, config, scenario=scenario, rewards=rewards,
                    uav_collision_radius=0., seed_manifest={'base_seed': config.seed},
                    validation_pool_metadata=dict(path='test', content_digest='test', format_version=1,
                        curriculum_level='easy', master_seed=7, stage_seed=11, scenario_count=100),
                    initialization_source=dict(kind='v2_bc_best', path='test', model_type=model),
                    bc_schedule='smooth-mid-strong',
                )
                with TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'model_failed.pt'
                    formal.save_v2_formal_checkpoint(path, terminal)
                    loaded = formal.load_v2_formal_checkpoint(path, expected_model_type=model)
                self.assertEqual(loaded['formal_config']['terminal_geo_radius'], 400.)
                self.assertEqual(loaded['engine_checkpoint']['algorithm_config']['terminal_geo_far_weight'], .5)

    def test_source_manifest_matches_current_sources_for_both_modes(self):
        sealed = reporting.read_json(run.GUIDANCE_SOURCE_MANIFEST)['sha256']
        for experiment in run.GUIDANCE_EXPERIMENTS:
            self.assertEqual(run.identity(experiment)['sha256'], sealed)

    def test_diagnostic_scenarios_and_paired_outcomes(self):
        for record in reporting.diagnostic_records():
            payload = record['payload']
            env_type, _, _ = _v2_api()
            env = env_type(_config(world_xy=1250., world_z_max=600.), RewardConfig(),
                           fixed_scenarios=[payload])
            env.reset()
            name = record['scenario_id']
            self.assertEqual(not env.zones, name.startswith('empty'))
            self.assertEqual(env.line_to_goal_is_safe(clearance=40.), not name.startswith('detour'))
        cap = [dict(axis='z', direction='positive')]
        before = [dict(scenario_id=str(i), outcome='goal') for i in range(100)]
        before[0].update(outcome='boundary', boundary_violations=cap)
        before[1].update(outcome='timeout')
        after = deepcopy(before)
        after[0].update(outcome='goal')
        after[1].update(outcome='collision')
        result = reporting.paired_outcomes(before, list(reversed(after)))
        self.assertEqual(result['transitions']['upper_boundary']['to_goal'], 1)
        self.assertEqual(result['transitions']['timeout']['to_other_failure'], 1)
        with self.assertRaises(ValueError):
            reporting.paired_outcomes(before, after[1:])

    def test_obstacle_clearance_recovers_in_same_environment(self):
        sphere = NoFlyZone('blocker', Sphere((300., 0., 200.), 50.))
        env, obs = self.env((0, 0, 200, 0, 0), (600, 0, 200), (sphere,))
        self.assertEqual(env.direct_guidance_band(obs, radius=400., clearance=40.), 0)
        env.state[:2] = (400., 120.)
        self.assertEqual(env.direct_guidance_band(env._get_obs(), radius=400., clearance=40.), 1)

    def test_collect_requires_all_groups_before_publishing(self):
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            args = run.parser().parse_args([
                'collect', '--launch', '--root', str(work / 'main'),
                '--second-batch', str(work / 'seed11'), '--baseline-root', str(work / 'old'),
                '--control-root', str(work / 'control'), '--ann-bc', 'a', '--snn-bc', 's',
                '--validation-pool', 'p', '--experiment', run.GUIDANCE_EXPERIMENT, '--batch', 'main',
            ])
            scenes = [dict(scenario_id=str(i), outcome='goal', episode_return=float(i)) for i in range(100)]
            lock = reporting.read_json(run.GUIDANCE_CONFIG)['historical_artifact_sha256']
            actual_hash = run.file_hash

            def hashes(path):
                if path.is_relative_to(args.baseline_root):
                    name = path.relative_to(args.baseline_root).as_posix()
                    if name in lock:
                        return lock[name]
                return actual_hash(path)

            def records(_args, root, experiment, batch, _pool, _inputs, **kwargs):
                result = {}
                for job in run.jobs(experiment, batch):
                    model, seed = job['model'], job['seed']
                    path = root / (job['id'] + '.pt')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.exists():
                        path.write_bytes(job['id'].encode())
                    record = dict(seed=seed, execution_completed=True, evaluation=dict(
                        scenarios=scenes, goal_count=100), checkpoint=str(path),
                        checkpoint_sha256=hashes(path), actual_steps=500000, formal_passed=False)
                    result[(model, seed)] = (record, {}, {'categories': {}})
                return root.name, result

            def incomplete(*a, **kw):
                identity, result = records(*a, **kw)
                if identity == 'seed11':
                    result.pop(('ann', 11))
                return identity, result

            with patch.object(run, 'inputs', return_value={}), patch.object(run, 'file_hash', hashes), \
                 patch('brain_uav.trainers.v2_validation.load_v2_validation_pool', return_value=SimpleNamespace()), \
                 patch.object(reporting, 'validated_batch', side_effect=incomplete):
                with self.assertRaises((KeyError, ValueError)):
                    run.collect(args)
                self.assertFalse((args.root / 'selected').exists())
                self.assertFalse((args.root / 'selection_summary.json').exists())

            with patch.object(run, 'inputs', return_value={}), patch.object(run, 'file_hash', hashes), \
                 patch('brain_uav.trainers.v2_validation.load_v2_validation_pool', return_value=SimpleNamespace()), \
                 patch.object(reporting, 'validated_batch', side_effect=records):
                result = run.collect(args)
                self.assertEqual({m['selected']['seed'] for m in result['models'].values()}, {7})
                self.assertTrue((args.root / 'selected' / 'snn' / 'baseline.pt').exists())
                self.assertEqual(run.collect(args), result)
