"""CPU checks only: target smoothing and diagnostic-only episode snapshots."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
from types import SimpleNamespace

from brain_uav.config import ScenarioConfig, RewardConfig
from brain_uav.v2_curriculum import V2NoiseSchedule
from brain_uav.scripts import train_v2_td3 as single
from brain_uav.scripts import run_v2_td3_curriculum as curriculum
from brain_uav.scripts import diagnose_v2_policy as policy
from brain_uav.scripts import diagnose_v2_critic as critic
from brain_uav.trainers import v2_formal_training as formal
from test_v2_formal_training import _engine, _Source, _scenario_payload, _validation_result
from test_diagnose_v2_policy import write_inputs
from test_run_v2_td3_curriculum import _prepared_initialization


class TestTargetNoiseExperiment(TestCase):
    def test_noise_scale(self):
        original = V2NoiseSchedule()
        self.assertEqual(original.with_target_noise_scale(), original)
        for scale in (.5, 1., 2.):
            schedule = original.with_target_noise_scale(scale)
            for step in (0, 1, 75000, 249999, 250000, 499999):
                before = original.values(step, max_steps=500000)
                after = schedule.values(step, max_steps=500000)
                self.assertEqual(after, (before[0], before[1]*scale, before[2]*scale))
        for bad in (0, -1, float('nan'), float('inf'), True):
            with self.assertRaises((TypeError, ValueError)):
                original.with_target_noise_scale(bad)

    def test_cli_forwarding(self):
        cases = [(single, 'run_v2_td3_stage', ['--stage', 'easy', '--init-checkpoint', 'bc.pt',
            '--output', 'out.pt', '--metrics-out', 'metrics.json', '--validation-pool', 'pool.json']),
            (curriculum, 'run_v2_curriculum', ['--bc-checkpoint', 'bc.pt', '--output-root', 'run',
             '--validation-pool-dir', 'pool'])]
        for module, name, required in cases:
            defaults = module.build_parser().parse_args(required)
            self.assertEqual(defaults.target_noise_scale, 1.)
            self.assertEqual(defaults.weight_snapshot_steps, ())
            with mock.patch.object(module, name, return_value={'passed': False}) as run:
                module.main(required + ['--device', 'cpu', '--target-noise-scale', '.5',
                    '--weight-snapshot-steps', '300000', '350000', '400000', '450000'])
            self.assertEqual(run.call_args.kwargs['target_noise_scale'], .5)
            self.assertEqual(run.call_args.kwargs['weight_snapshot_steps'],
                             (300000, 350000, 400000, 450000))

    def test_api_builds_scaled_formal_config_before_initialization(self):
        for scale in (.5, 2.):
            with mock.patch.object(single, 'prepare_v2_stage_initialization', side_effect=RuntimeError('stop')) as init:
                with self.assertRaisesRegex(RuntimeError, 'stop'):
                    single.run_v2_td3_stage(stage='easy', init_checkpoint='bc.pt', output='out.pt',
                        metrics_out='metrics.json', validation_pool='pool.json', device='cpu',
                        target_noise_scale=scale)
            self.assertEqual(init.call_args.args[0].noise_schedule,
                             V2NoiseSchedule().with_target_noise_scale(scale))
        with TemporaryDirectory() as directory:
            path = Path(directory)/'bc.pt'
            path.touch()
            with mock.patch.object(curriculum, 'prepare_v2_stage_initialization', side_effect=RuntimeError('stop')) as init:
                with self.assertRaisesRegex(RuntimeError, 'stop'):
                    curriculum.run_v2_curriculum(bc_checkpoint=path, output_root=Path(directory)/'out',
                        validation_pool_dir=Path(directory)/'pool', device='cpu', target_noise_scale=2.)
            self.assertEqual(init.call_args.args[0].noise_schedule, V2NoiseSchedule().with_target_noise_scale(2.))

    def test_curriculum_forwards_to_each_stage(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            bc = base/'bc.pt'
            bc.touch()
            prepared = _prepared_initialization(ScenarioConfig(), source=bc)
            run = mock.Mock(side_effect=lambda **kw: dict(stage=kw['stage'], passed=True, checkpoint=str(kw['output']),
                                                         steps=1, global_steps_end=kw['global_steps_start']+1))
            with mock.patch.object(curriculum, 'prepare_v2_stage_initialization', return_value=prepared), \
                    mock.patch.object(curriculum, 'prepare_v2_validation_pools',
                                      return_value={s: base/f'{s}.json' for s in ('easy','medium','hard')}), \
                    mock.patch.object(curriculum, 'load_v2_validation_pool',
                                      return_value=SimpleNamespace(content_digest='digest', scenario_count=100,
                                                                   master_seed=20260904, stage_seed=11)):
                curriculum.run_v2_curriculum(bc_checkpoint=bc, output_root=base/'out',
                    validation_pool_dir=base/'pools', device='cpu', stage_runner=run,
                    target_noise_scale=2., weight_snapshot_steps=(300000,350000,400000,450000))
            self.assertEqual(run.call_count, 3)
            for call in run.call_args_list:
                self.assertEqual(call.kwargs['target_noise_scale'], 2.)
                self.assertEqual(call.kwargs['weight_snapshot_steps'], (300000,350000,400000,450000))

    def make_trainer(self, *, goal=False, **kwargs):
        scenario = ScenarioConfig(max_steps=3)
        config = formal.V2FormalTrainingConfig(stage='easy', max_steps=7,
            batch_size=2, replay_capacity=16, window_episode_count=99,
            early_stop_min_steps=99, noise_schedule=V2NoiseSchedule().with_target_noise_scale(.5))
        return formal.V2FormalStageTrainer(scenario, RewardConfig(), config, _engine(scenario),
            scenario_sources={'easy': _Source(_scenario_payload(0, goal=goal))},
            validation_runner=lambda actor: self.fail('weight snapshots must not validate'), **kwargs)

    def test_actual_boundary_no_validation_budget_and_actual_noise(self):
        saved = []
        trainer = self.make_trainer(weight_snapshot_steps=(2, 4, 7, 9),
            weight_snapshot_sink=lambda a,b: saved.append((a,b)))
        with mock.patch.object(trainer.engine, 'set_target_noise', wraps=trainer.engine.set_target_noise) as noise:
            result = trainer.run()
        self.assertEqual(saved, [(2, 3), (4, 6)])
        self.assertEqual(result.weight_snapshots, [dict(requested_step=a, actual_step=b) for a,b in saved])
        self.assertEqual(result.stage_steps, 7)
        self.assertEqual(len(result.episodes), 2)
        for step, call in enumerate(noise.call_args_list):
            _, policy_noise, noise_clip = trainer.config.noise_schedule.values(step, max_steps=7)
            self.assertEqual(call.kwargs, dict(policy_noise=policy_noise, noise_clip=noise_clip))
        self.assertEqual(result.validation_records, [])

    def test_early_stop(self):
        trainer = self.make_trainer(goal=True, weight_snapshot_steps=(2, 4), weight_snapshot_sink=mock.Mock())
        trainer.controller.window_episode_count = 1
        trainer.controller.required_consecutive = 1
        trainer.controller.early_stop_min_steps = 0
        trainer.validation_runner = lambda actor: _validation_result(True)
        result = trainer.run()
        self.assertTrue(result.passed_validation)
        self.assertEqual(result.stage_steps, 1)
        trainer.weight_snapshot_sink.assert_not_called()

    def test_strict_diagnostic_load_and_old_result(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            _, pool_path, payload = write_inputs(base)
            old = deepcopy(payload)
            del old['training_result']['weight_snapshots']
            self.assertEqual(formal.load_v2_formal_checkpoint(old, require_passed=False)['status'], 'failed')
            for scale in (.5, 2.):
                schedule = V2NoiseSchedule().with_target_noise_scale(scale)
                payload['formal_config']['noise_schedule'] = formal.asdict(schedule)
                path = base / f'formal_{scale}.pt'
                formal.save_v2_formal_checkpoint(path, payload)
                self.assertEqual(formal.load_v2_formal_checkpoint(path, require_passed=False)
                                 ['formal_config']['noise_schedule'], formal.asdict(schedule))
                snapshot = {key: payload[key] for key in ('stage', 'engine_checkpoint',
                    'formal_config', 'scenario_config', 'reward_config', 'uav_collision_radius',
                    'seed_manifest', 'bc_schedule', 'initialization_source')}
                snapshot.update(format=formal.V2_PERIODIC_SNAPSHOT_FORMAT,
                    format_version=formal.V2_PERIODIC_SNAPSHOT_VERSION, stage_steps=300003)
                path = base / f'snapshot_{scale}.pt'
                formal.save_v2_periodic_snapshot(path, snapshot)
                data = policy.load_diagnostic_inputs(model='ann', checkpoint=path, validation_pool=pool_path, device='cpu')
                self.assertEqual(data['checkpoint_payload']['stage_steps'], 300003)
                self.assertNotIn('passed_validation', data['checkpoint_payload'])
                critic_data = critic.load_models(path, pool_path, 'cpu')
                self.assertEqual(len(critic_data['critics']), 2)
                self.assertEqual(critic.compare_inputs(critic_data, critic_data), [])
                with self.assertRaises(ValueError):
                    formal.load_v2_formal_checkpoint(path)
                damaged = deepcopy(snapshot)
                damaged['formal_config']['noise_schedule']['clip_initial'] = -1.
                with self.assertRaises(ValueError):
                    formal.load_v2_periodic_snapshot(damaged)
