"""CPU-only formal execution-option wiring checks."""
import contextlib
import inspect
import io
from unittest import TestCase, mock

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.scripts import train_v2_td3 as stage
from brain_uav.scripts import run_v2_td3_curriculum as curriculum
from brain_uav.trainers.v2_formal_training import V2FormalStageTrainer, V2FormalTrainingConfig
import test_v2_formal_training as fixture
import test_v2_validation as validation_fixture
from brain_uav.trainers import v2_validation as validation
from brain_uav.models import V2SNNPolicyActor
from brain_uav.observations import V2ObservationScales
import torch


class TestFormalCPUDeduplication(TestCase):
    def test_stage_and_curriculum_cli_default_and_explicit(self):
        for enabled in (False, True):
            extra = ['--cpu-deduplication'] if enabled else ['--no-cpu-deduplication']
            for module, argv, target in (
                (stage, ['--stage','easy','--init-checkpoint','bc.pt','--output','out.pt',
                         '--metrics-out','out.json','--validation-pool','pool.json'], 'run_v2_td3_stage'),
                (curriculum, ['--bc-checkpoint','bc.pt','--output-root','run',
                              '--validation-pool-dir','pools'], 'run_v2_curriculum'),
            ):
                with self.subTest(module=module.__name__, enabled=enabled):
                    self.assertEqual(module.build_parser().parse_args(argv + extra).cpu_deduplication, enabled)
                    with mock.patch.object(module, target, return_value={'passed': True}) as runner, \
                         contextlib.redirect_stdout(io.StringIO()):
                        module.main(argv + ['--device','cpu'] + extra)
                    self.assertEqual(runner.call_args.kwargs['cpu_deduplication'], enabled)
        for function in (stage.run_v2_td3_stage, curriculum.run_v2_curriculum,
                         V2FormalStageTrainer):
            self.assertIs(inspect.signature(function).parameters['cpu_deduplication'].default, True)

    def test_trainer_environment_action_and_execution_record(self):
        for enabled in (False, True):
            scenario = ScenarioConfig(max_steps=1)
            engine = fixture._engine(scenario)
            config = V2FormalTrainingConfig(stage='easy', max_steps=2,
                replay_capacity=16, batch_size=2, warmup_steps=3)
            trainer = V2FormalStageTrainer(scenario, RewardConfig(), config, engine,
                scenario_sources={'easy': fixture._Source(fixture._scenario_payload(0))},
                validation_runner=lambda actor: fixture._validation_result(False),
                cpu_deduplication=enabled)
            with mock.patch.object(engine, 'select_action', wraps=engine.select_action) as action:
                trainer.run()
            self.assertEqual(trainer.env.cpu_deduplication, enabled)
            for call in action.call_args_list:
                self.assertEqual(call.kwargs.get('cpu_deduplication', False), enabled)
            self.assertEqual(trainer.execution_configuration['cpu_deduplication'], {
                'requested': enabled, 'training_environment': enabled,
                'training_action_input': enabled})
            self.assertNotIn('cpu_deduplication', config.to_dict())

    def test_fixed_validation_uses_existing_reuse_and_preserves_snn_state_and_rng(self):
        scenario = ScenarioConfig(max_steps=1)
        scales = V2ObservationScales(scenario.world_xy, scenario.world_z_min,
                                    scenario.world_z_max, scenario.gamma_max)
        actor = V2SNNPolicyActor(scales, 2, 8, torch.tensor([
            scenario.delta_gamma_max, scenario.delta_psi_max]), time_window=2)
        pool = validation_fixture._pool('easy', scenario, (1., 100.))
        results = []
        for enabled in (False, True):
            before = torch.get_rng_state().clone()
            with mock.patch.object(validation, 'V2StaticNoFlyTrajectoryEnv',
                    wraps=validation.V2StaticNoFlyTrajectoryEnv) as env, \
                 mock.patch.object(validation, 'collate_v2_observations',
                    wraps=validation.collate_v2_observations) as collate:
                results.append(validation.evaluate_v2_fixed_validation(
                    actor, pool, RewardConfig(), cpu_deduplication=enabled).to_dict())
            self.assertEqual(env.call_args.kwargs['cpu_deduplication'], enabled)
            self.assertTrue(collate.called)
            for call in collate.call_args_list:
                self.assertEqual(call.kwargs.get('deduplicate_copies', False), enabled)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(actor.snn_head.lif1.v, 0.)
            self.assertEqual(actor.snn_head.lif2.v, 0.)
        self.assertEqual(results[0], results[1])
