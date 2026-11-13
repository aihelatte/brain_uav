"""Four reviewed LR candidates: CPU builders and mocked processes, never CUDA."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock,patch

import torch
from brain_uav.scripts import run_v2_formal_easy_baseline as run
from brain_uav.trainers import v2_formal_training as formal
import test_v2_formal_training as fixture
import test_v2_formal_easy_review as review_fixtures


class LRFinal(unittest.TestCase):
    def args(self,root,mode='launch'):
        return run.parser().parse_args([mode,'--root',str(root),'--ann-bc',str(root/'ann.pt'),
            '--snn-bc',str(root/'snn.pt'),'--validation-pool',str(root/'pool.json'),
            '--experiment',run.LR_EXPERIMENT,'--source-manifest',str(run.LR_SOURCE_MANIFEST)])

    def test_only_requested_learning_rates_change_and_old_defaults_remain(self):
        self.assertEqual(len(run.jobs()),4)
        for job in run.jobs():
            before,cfg=run.resolved(job);after,actual=run.resolved(job,run.LR_EXPERIMENT)
            field='actor_lr' if job['model']=='ann' else 'critic_lr'
            new=1.2e-4 if job['model']=='ann' else 2.5e-4
            old=1.55e-4 if job['model']=='ann' else 1e-3
            self.assertEqual(getattr(cfg,field),old);self.assertEqual(getattr(actual,field),new)
            expected=deepcopy(before);expected['formal_config'][field]=new
            for key in ('experiment','baseline_commit','learning_rate_changes'):after.pop(key)
            self.assertEqual(after,expected)
        with TemporaryDirectory() as tmp:
            args=self.args(Path(tmp));args.experiment='baseline'
            self.assertEqual(run.plan(args)['jobs'],[run.resolved(j)[0] for j in run.jobs()])
        with self.assertRaises(ValueError):run.resolved(run.jobs()[0],'not-an-experiment')

    def test_real_cli_plan_no_side_effects(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'absent'
            command=[sys.executable,'-B','-m','brain_uav.scripts.run_v2_formal_easy_baseline','--root',str(root),
                '--ann-bc','absent_ann','--snn-bc','absent_snn','--validation-pool','absent_pool',
                '--experiment',run.LR_EXPERIMENT,'--source-manifest',str(run.LR_SOURCE_MANIFEST)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=30,check=True)
            plan=json.loads(result.stdout)
            self.assertFalse(root.exists());self.assertFalse(plan['launch'])
            self.assertEqual(plan['jobs'],[run.resolved(j,run.LR_EXPERIMENT)[0] for j in run.jobs()])

    def test_worker_to_real_cpu_builder_and_optimizer(self):
        from brain_uav.models.v2_snn import V2SNNPolicyActor
        from brain_uav.trainers.v2_formal_training import V2BCFormalInitialization
        from brain_uav.config import RewardConfig
        for job in run.jobs():
            with self.subTest(job=job['id']),TemporaryDirectory() as tmp:
                root=Path(tmp);a=self.args(root,'worker');a.job=job['id']
                spec,cfg=run.resolved(job,run.LR_EXPERIMENT)
                paths={key:str(path.resolve()) for key,path in [('ann_bc',a.ann_bc),('snn_bc',a.snn_bc),('pool',a.validation_pool)]}
                run.atomic_json(root/'manifest.json',dict(plan=run.plan(a),inputs={'paths':paths},batch_id='lr'))
                def stage(**kwargs):
                    self.assertEqual(kwargs['formal_config'],cfg)
                    self.assertEqual((kwargs['actor_lr'],kwargs['critic_lr']),(cfg.actor_lr,cfg.critic_lr))
                    self.assertEqual(kwargs['init_checkpoint'],getattr(a,job['model']+'_bc'))
                    scenario=fixture.ScenarioConfig();actor=fixture._engine(scenario).actor
                    if job['model']=='snn':
                        actor=V2SNNPolicyActor(actor.scales,2,8,actor.action_limit,time_window=4)
                    actor.eval()
                    # Keep exact LR/config values; reduce only fixture replay storage, no updates run.
                    small=replace(cfg,replay_capacity=8,batch_size=2)
                    source={}
                    prepared=formal.V2PreparedStageInitialization(stage='easy',
                        model_seed=formal.derive_v2_component_seed(cfg.seed,'easy','model'),
                        verified_initialization_source=source,snn_time_window=4 if job['model']=='snn' else None,
                        scenario_config=scenario,reward_config=RewardConfig(),uav_collision_radius=0.,model_type=job['model'],
                        bc_initialization=V2BCFormalInitialization(actor=actor,scenario_config=scenario,uav_collision_radius=0.,model_type=job['model']),
                        formal_checkpoint=None,torch_rng_state=torch.get_rng_state().clone())
                    engine=formal.build_v2_stage_engine(scenario,small,init_checkpoint=source,device='cpu',
                        model_type=job['model'],prepared_initialization=prepared).engine
                    self.assertEqual(engine.actor_optimizer.param_groups[0]['lr'],cfg.actor_lr)
                    self.assertEqual(engine.critic_optimizer.param_groups[0]['lr'],cfg.critic_lr)
                    self.assertEqual(engine.checkpoint_state_dict()['actor_optimizer_state_dict']['param_groups'][0]['lr'],cfg.actor_lr)
                    return dict(passed=False,execution_configuration={'gpu_td3_compute':{'effective':True},
                        'non_blocking_batch_transfer':{'actual_execution':{'effective':True}}})
                with patch.object(run,'check_inputs'),patch('brain_uav.scripts.train_v2_td3.run_v2_td3_stage',side_effect=stage) as call:
                    run.worker(a)
                call.assert_called_once()
                self.assertTrue(json.loads((root/job['id']/'worker_summary.json').read_text())['execution_completed'])

    def test_dispatch_and_worker_configuration_mismatch(self):
        from brain_uav.scripts import v2_mps_diagnostic as mps
        with TemporaryDirectory() as tmp:
            a=self.args(Path(tmp)/'batch');a.gate_only=True
            session=Mock(env={});session.finish.return_value={'stopped':True}
            workers=[Mock(pid=i+10) for i in range(4)]
            for p in workers:p.poll.return_value=None
            with patch.object(run.provenance,'PYTHON',sys.executable),patch.object(run,'inputs',return_value={}), \
                patch.object(mps,'Session',return_value=session),patch.object(mps,'capability'), \
                patch.object(run.subprocess,'Popen',side_effect=workers) as spawn,patch.object(run,'wait_ready'), \
                patch.object(run,'release',side_effect=RuntimeError('test release stop')) as release:
                with self.assertRaisesRegex(RuntimeError,'test release stop'):run.launch(a)
            self.assertEqual(spawn.call_count,4)
            self.assertEqual({c.args[0][c.args[0].index('--job')+1] for c in spawn.call_args_list},{j['id'] for j in run.jobs()})
            for c in spawn.call_args_list:self.assertEqual(c.args[0][c.args[0].index('--experiment')+1],run.LR_EXPERIMENT)
            self.assertEqual(release.call_args.args[-1],run.LR_EXPERIMENT)
        with TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.args(root,'worker');a.job='ann_seed7'
            run.atomic_json(root/'manifest.json',dict(plan={'gate_only':False,'experiment':'baseline'},inputs={}))
            with patch.object(run,'check_inputs'),patch('brain_uav.scripts.train_v2_td3.run_v2_td3_stage') as stage:
                with self.assertRaisesRegex(ValueError,'mode differs'):run.worker(a)
                stage.assert_not_called()

    def test_gate_reports_cannot_cross_experiments_or_parameters(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.args(root/'train');a.gate_report=root/'gate'
            evidence=dict(source={'sha256':{'source':'same'}},hashes={'input':'same'})
            review_fixtures.Review().gate_fixture(a.gate_report,evidence)
            with self.assertRaisesRegex(ValueError,'input/source/batch'):run.validate_gate(a,evidence)
            manifest=json.loads((a.gate_report/'manifest.json').read_text())
            manifest['plan']['experiment']=run.LR_EXPERIMENT
            run.atomic_json(a.gate_report/'manifest.json',manifest)
            # Same source/input hashes cannot make baseline LR reports valid for candidate LR.
            with self.assertRaisesRegex(ValueError,'identity'):run.validate_gate(a,evidence)
            for job in run.jobs():
                p=a.gate_report/job['id']/'cuda_gate.json';v=json.loads(p.read_text())
                v['identity']['resolved']=run.resolved(job,run.LR_EXPERIMENT)[0];run.atomic_json(p,v)
            run.atomic_json(a.gate_report/'batch_summary.json',dict(execution_completed=True,gate_bindings=run.gate_bindings(a.gate_report,manifest)))
            self.assertEqual(len(run.validate_gate(a,evidence)['bindings']),4)

    def test_source_manifest_is_independent_and_complete(self):
        old=json.loads(run.SOURCE_MANIFEST.read_text())
        actual=run.identity(run.LR_EXPERIMENT)
        self.assertNotEqual(old['sha256'],actual['sha256'])
        self.assertEqual({p.relative_to(run.REPO).as_posix() for p in (run.REPO/'src').rglob('*.py')},
            {p for p in actual['sha256'] if p.startswith('src/')})
        self.assertIn(run.LR_CONFIG.relative_to(run.REPO).as_posix(),actual['sha256'])
        self.assertEqual(json.loads(run.LR_SOURCE_MANIFEST.read_text())['sha256'],actual['sha256'])
        with TemporaryDirectory() as tmp:
            a=self.args(Path(tmp));a.source_manifest=run.SOURCE_MANIFEST
            with self.assertRaisesRegex(ValueError,'Actual source files'):run.inputs(a)

    def test_release_checks_candidate_configuration_and_full_seed_manifest(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);processes={};session=Mock();session.verify.return_value={'own_clients':True}
            for index,job in enumerate(run.jobs()):
                spec,cfg=run.resolved(job,run.LR_EXPERIMENT);pid=100+index
                process=Mock(pid=pid);process.poll.return_value=None;processes[job['id']]=process
                seeds={'base_seed':cfg.seed,**{name+'_seed':formal.derive_v2_component_seed(cfg.seed,cfg.stage,name)
                    for name in ('model','replay','curriculum','exploration','easy_generator')}}
                run.atomic_json(root/job['id']/'ready.json',dict(pid=pid,job=job,resolved=spec,seed_manifest=seeds,synchronized=True))
            run.release(processes,root,session,run.LR_EXPERIMENT)
            self.assertEqual(len(json.loads((root/'release.json').read_text())['pids']),4)
            (root/'release.json').unlink();session.verify.reset_mock()
            path=root/'snn_seed23/ready.json';ready=json.loads(path.read_text())
            ready['resolved']=run.resolved(run.jobs()[-1])[0];run.atomic_json(path,ready)
            with self.assertRaisesRegex(ValueError,'readiness'):run.release(processes,root,session,run.LR_EXPERIMENT)
            self.assertFalse((root/'release.json').exists());session.verify.assert_not_called()

    def test_collect_uses_candidate_configuration_and_rejects_baseline_selection(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            with review_fixtures.Review().terminal_batch(root) as (a,raws):
                a.experiment=run.LR_EXPERIMENT
                manifest=json.loads((root/'manifest.json').read_text());manifest['plan']['experiment']=run.LR_EXPERIMENT
                run.atomic_json(root/'manifest.json',manifest)
                for job in run.jobs():
                    raws[str(root/job['id']/'model_failed.pt')]['formal_config']=run.resolved(job,run.LR_EXPERIMENT)[1].to_dict()
                self.assertTrue(run.collect(a)['execution_completed'])
                original=(root/'selection_summary.json').read_bytes()
                a.experiment='baseline'
                with self.assertRaisesRegex(ValueError,'experiment'):run.collect(a)
                self.assertEqual((root/'selection_summary.json').read_bytes(),original)


if __name__=='__main__':unittest.main()
