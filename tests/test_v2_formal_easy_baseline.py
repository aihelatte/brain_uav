"""CPU/tiny synthesis only; no real source models, MPS or CUDA."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import Mock,patch
import numpy as np
import torch

from brain_uav.scripts import run_v2_formal_easy_baseline as run
from brain_uav.trainers import v2_formal_training as formal
from brain_uav.trainers.v2_formal_acc import FormalACC,load_state
from brain_uav.config import ScenarioConfig,RewardConfig
import test_v2_formal_training as f


class FormalEasy(unittest.TestCase):
    def test_locked_four_configs_and_default_no_processes(self):
        configs=[run.resolved(j) for j in run.jobs()]
        self.assertEqual([c.seed for _,c in configs],[7,23,7,23])
        for spec,cfg in configs:
            self.assertEqual(cfg.max_steps,500000);self.assertEqual(cfg.failure_sample_bias,2.)
            self.assertEqual(spec['reward_config']['breakthrough_clearance_gate_enabled'],False)
            self.assertTrue(spec['non_blocking_batch_transfer']);self.assertFalse(spec['gpu_compute_non_blocking_coefficients'])
            self.assertEqual(cfg.noise_schedule.values(499999,max_steps=500000),(.005,.006,.012))
        with TemporaryDirectory() as tmp,patch.object(run,'launch',side_effect=AssertionError),patch.object(run,'inputs',side_effect=AssertionError):
            args=run.parser().parse_args(['--root',tmp,'--ann-bc','a','--snn-bc','s','--validation-pool','p'])
            self.assertFalse(run.plan(args)['launch']);self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_formal_cli_acc_and_nonblocking_forward_without_changing_defaults(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from brain_uav.scripts import train_v2_td3 as entry
        argv=['--stage','easy','--init-checkpoint','bc','--output','model','--metrics-out','metrics','--validation-pool','pool',
              '--device','cpu','--acc-eta','.01','--non-blocking-batch-transfer']
        with patch.object(entry,'run_v2_td3_stage',return_value={'passed':False}) as stage,redirect_stdout(StringIO()):
            entry.main(argv)
        self.assertEqual(stage.call_args.kwargs['acc_eta'],.01)
        self.assertTrue(stage.call_args.kwargs['non_blocking_batch_transfer'])
        args=entry.build_parser().parse_args(argv[:-3])
        self.assertIsNone(args.acc_eta);self.assertFalse(args.non_blocking_batch_transfer)

    def test_gate_failure_report_is_incomplete_and_keeps_original_error(self):
        from brain_uav.scripts import v2_formal_easy_cuda_gate as gate
        with TemporaryDirectory() as tmp,patch.object(gate,'_check',side_effect=RuntimeError('CUDA fixture error')):
            path=Path(tmp)/'gate.json'
            with self.assertRaisesRegex(RuntimeError,'CUDA fixture error'):gate.check(None,None,None,None,None,None,output=path)
            report=json.loads(path.read_text());self.assertFalse(report['strict_passed']);self.assertFalse(report['execution_completed'])
            self.assertIn('CUDA fixture error',report['error'])

    def test_acc_complete_real_episode_early_stop_and_tail(self):
        for tail in (False,True):
            scenario=ScenarioConfig(max_steps=10 if tail else 1);engine=f._engine(scenario)
            cfg=formal.V2FormalTrainingConfig(stage='easy',acc_eta=.01,max_steps=1 if tail else 4,
                early_stop_min_steps=1,window_episode_count=1,consecutive_qualified_windows=1,max_failures_per_window=0,
                replay_capacity=16,batch_size=2)
            hook=FormalACC(engine,cfg)
            seen=[]
            original=hook.episode
            def episode(trainer,transitions):
                for t in transitions:np.testing.assert_array_equal(t.action,trainer.env.prev_action)
                seen.append(len(transitions));original(trainer,transitions)
            hook.episode=episode
            validate=Mock(return_value=f._validation_result(True))
            trainer=formal.V2FormalStageTrainer(scenario,RewardConfig(),cfg,engine,
                scenario_sources={'easy':f._Source(f._scenario_payload(0,goal=not tail))},validation_runner=validate)
            result=trainer.run()
            self.assertIsNone(trainer.online_research)
            if tail:
                self.assertFalse(seen);self.assertEqual(len(hook.controller.episodes),0);self.assertEqual(result.episodes,[])
            else:
                self.assertTrue(result.passed_validation);self.assertEqual(result.stage_steps,1);validate.assert_called_once()
                self.assertEqual(len(hook.controller.episodes),1)
                bank=hook.controller.episodes[0];self.assertEqual(bank[-1]['G'],bank[-1]['reward'])

    def test_acc_feedback_snapshot_formal_roundtrip_and_tamper(self):
        scenario=ScenarioConfig();engine=f._engine(scenario)
        cfg=formal.V2FormalTrainingConfig(stage='easy',acc_eta=.01,replay_capacity=16,batch_size=2)
        hook=FormalACC(engine,cfg)
        from test_v2_td3 import _observation
        obs=_observation(0,scales=engine.actor.scales)
        transition=SimpleNamespace(observation=obs,action=np.zeros(2,dtype=np.float32),reward=1.,done=True,collected_stage_step=25001)
        trainer=SimpleNamespace(engine=engine,result=SimpleNamespace(stage_steps=25001))
        before=engine.research_beta;hook.episode(trainer,[transition]);self.assertNotEqual(before,engine.research_beta)
        self.assertEqual(engine.research_beta,hook.controller.beta)
        engine.set_target_noise(policy_noise=.006,noise_clip=.012)
        kw=dict(scenario=scenario,rewards=RewardConfig(),uav_collision_radius=0.,seed_manifest={'base_seed':7},initialization_source={'kind':'v2_bc_best','path':'fixture'})
        snapshot=formal.build_v2_periodic_snapshot(engine,cfg,stage_steps=25001,**kw)
        loaded=formal.load_v2_periodic_snapshot(snapshot);other=f._engine(scenario);restore=load_state(other,cfg,loaded)
        self.assertEqual(other.research_beta,engine.research_beta);self.assertEqual(restore.controller.state()['controls'],hook.controller.controls)
        self.assertEqual((other.policy_noise,other.noise_clip),(.006,.012))
        result=formal.V2FormalTrainingResult.empty('easy',global_steps_start=0);result.stage_steps=25001;result.status='failed';result.stop_reason='max_steps'
        cp=formal.build_v2_formal_checkpoint(engine,result,cfg,validation_pool_metadata={'path':'test','content_digest':'test',
            'format_version':1,'curriculum_level':'easy','master_seed':7,'stage_seed':11,'scenario_count':100},**kw)
        formal.load_v2_formal_checkpoint(cp,require_passed=False)
        with TemporaryDirectory() as tmp:
            periodic=Path(tmp)/'periodic.pt';terminal=Path(tmp)/'terminal.pt'
            formal.save_v2_periodic_snapshot(periodic,snapshot)
            formal.save_v2_formal_checkpoint(terminal,cp)
            for artifact in (formal.load_v2_periodic_snapshot(periodic),formal.load_v2_formal_checkpoint(terminal)):
                state=artifact['acc_state']['controller']
                self.assertEqual(state['beta'],engine.research_beta)
                np.testing.assert_array_equal(state['episodes'][0][0]['action'],transition.action)
        for original,loader in ((snapshot,formal.load_v2_periodic_snapshot),(cp,lambda p:formal.load_v2_formal_checkpoint(p,require_passed=False))):
            for mutation in ('G','beta','eta','config','method'):
                bad=deepcopy(original)
                if mutation=='G':bad['acc_state']['controller']['episodes'][0][0]['G']+=1
                elif mutation=='beta':bad['engine_checkpoint']['online_research_beta']+=.1
                elif mutation=='eta':bad['acc_state']['controller']['eta']=.1
                elif mutation=='method':bad['acc_state']['method']='mean'
                else:bad['formal_config']['acc_eta']=None
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):loader(bad)

    def test_exact_mps_gate_and_safe_cleanup(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);processes={j['id']:Mock(pid=i+10) for i,j in enumerate(run.jobs())}
            for job in run.jobs():
                p=processes[job['id']];p.poll.return_value=None
                from brain_uav.v2_curriculum import derive_v2_component_seed
                seeds={'base_seed':job['seed'],**{f'{k}_seed':derive_v2_component_seed(job['seed'],'easy',k)
                    for k in ('model','replay','curriculum','exploration','easy_generator')}}
                run.atomic_json(root/job['id']/'ready.json',dict(pid=p.pid,job=job,resolved=run.resolved(job)[0],synchronized=True,seed_manifest=seeds))
            session=Mock();session.verify.return_value={'effective':'mps'}
            run.release(processes,root,session);session.verify.assert_called_once_with(processes)
            self.assertTrue((root/'release.json').exists())
            (root/'release.json').unlink();session.verify.side_effect=RuntimeError('foreign PID')
            with self.assertRaises(RuntimeError):run.release(processes,root,session)
            self.assertFalse((root/'release.json').exists())
            session.before_terminate.side_effect=RuntimeError('response700')
            with self.assertRaises(RuntimeError):run.stop_owned(processes,session)
            for p in processes.values():p.terminate.assert_not_called();p.kill.assert_not_called()
            session.before_terminate.side_effect=None;run.stop_owned(processes,session)
            for p in processes.values():p.terminate.assert_called_once()

    def test_terminal_selection_rules_failures_and_incomplete(self):
        def row(seed,passed=False,goals=1,ret=1.,steps=500000):
            return dict(seed=seed,formal_passed=passed,execution_completed=True,actual_steps=steps,
                evaluation=dict(goal_count=goals,scenarios=[dict(scenario_id=str(i),episode_return=ret) for i in range(100)]))
        cases=[(row(7),row(23,passed=True)),(row(7),row(23,goals=2)),(row(7),row(23,ret=2)),(row(7),row(23,steps=400000))]
        for a,b in cases:self.assertEqual(run.choose([a,b])['seed'],23)
        self.assertEqual(run.choose([row(7),row(23)])['seed'],7)
        bad=row(23);bad['execution_completed']=False
        with self.assertRaises(ValueError):run.choose([row(7),bad])

    def test_parent_dispatch_failure_retains_service_and_never_signals_on_unknown_termination(self):
        from brain_uav.scripts import v2_mps_diagnostic as mps
        import sys
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'batch';a=run.parser().parse_args(['launch','--launch','--gate-only','--root',str(root),
                '--ann-bc','a','--snn-bc','s','--validation-pool','p'])
            session=Mock();session.env={};session.before_terminate.side_effect=RuntimeError('terminate_client response700')
            session.finish.return_value={'stopped':False,'cleanup_error':'clients remain'}
            processes=[Mock(pid=i+10) for i in range(4)]
            for p in processes:p.poll.return_value=None
            with patch.object(run.provenance,'PYTHON',sys.executable),patch.object(run,'inputs',return_value={}), \
                 patch.object(mps,'Session',return_value=session),patch.object(mps,'capability'), \
                 patch.object(run.subprocess,'Popen',side_effect=processes) as start, \
                 patch.object(run,'wait_ready'),patch.object(run,'release',side_effect=RuntimeError('bad PID')):
                with self.assertRaisesRegex(RuntimeError,'bad PID'):run.launch(a)
            self.assertEqual(start.call_count,4)
            self.assertTrue(all('--gate-only' in call.args[0] for call in start.call_args_list))
            for p in processes:p.terminate.assert_not_called();p.kill.assert_not_called()
            result=json.loads((root/'batch_summary.json').read_text());self.assertFalse(result['execution_completed'])
            self.assertIn('response700',result['cleanup_error']);session.finish.assert_called_once()


if __name__=='__main__':unittest.main()
