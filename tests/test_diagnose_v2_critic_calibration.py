"""Tiny CPU fixtures; never the twenty-scene/1000-update experiment."""
from copy import deepcopy
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from brain_uav.scripts import diagnose_v2_critic_calibration as d
from brain_uav.scripts import diagnose_v2_critic as b
from brain_uav.config import RewardConfig
from test_v2_static_no_fly_env import _config, _scenario


class TinyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear = torch.nn.Linear(8,1)
    def forward(self, obs, action):
        return self.linear(torch.cat((obs.ego_features,action),dim=1))


class TestCalibration(unittest.TestCase):
    def fixture(self):
        data = dict(scenario=_config(max_steps=2), rewards=RewardConfig(),
            checkpoint_payload=dict(formal_config=dict(seed=23)),
            pool=SimpleNamespace(uav_collision_radius=0.),gamma=.5)
        scene = dict(payload=_scenario([0,0,10,0,0],[60,0,10],[]))
        env = b.make_env(data); obs,info = env.reset(options={'scenario':scene['payload']})
        result = b.fly(data,env,obs,record_probe_hashes=True,initial_info=info)
        result.update(scenario_id='one',empty_scene=True)
        return data,scene,result

    def test_returns_include_current_reward_and_done_truncated_boundary(self):
        self.assertEqual(d.returns([10.,4.,-2.],[False,False,True],.5),[11.5,3.,-2.])
        for terminated,truncated in ((True,False),(False,True)):
            self.assertEqual(d.returns([7.],[terminated or truncated],.99),[7.])
        for rewards,done in (([1,2],[True,True]),([1],[False]),([float('nan')],[True])):
            with self.assertRaises(ValueError): d.returns(rewards,done,.99)
        y = d.td_target(torch.tensor([3.,4.]),torch.tensor([1.,0.]),.5,
                        torch.tensor([-100.,6.]),torch.tensor([-200.,8.]))
        self.assertTrue(torch.equal(y,torch.tensor([3.,7.])))
        with self.assertRaises(ValueError): d.td_target(torch.tensor([float('nan')]),torch.ones(1),.99,torch.ones(1),torch.ones(1))

    def test_split_stratified_deterministic_without_leakage_or_outcome_selection(self):
        rows = [dict(scenario_id=str(i),empty_scene=i<7,outcome='goal') for i in range(20)]
        split = d.split_scenes(rows)
        self.assertEqual((len(split['train']),len(split['holdout'])),(16,4))
        self.assertFalse(set(split['train']) & set(split['holdout']))
        self.assertEqual(sum(r['empty_scene'] for r in rows if r['scenario_id'] in split['holdout']),2)
        for r in rows: r.update(outcome='collision',Q=-1e10,G=1e10)
        self.assertEqual(split,d.split_scenes(list(reversed(rows))))

    def test_real_linear_replay_exact_and_mismatch_rejected(self):
        with mock.patch.object(b,'actor_action',return_value=np.zeros(2,dtype=np.float32)):
            data,scene,record = self.fixture()
        with mock.patch.object(b,'actor_action',side_effect=AssertionError('must not fly')):
            rows = d.replay_episode(data,scene,record)
            self.assertEqual(len(rows),2)
            self.assertTrue(rows[-1]['done'])
            self.assertEqual(rows[0]['G'],record['discounted_return'])
            for key in ('pre_action_hashes','states','rewards','final_info'):
                changed=deepcopy(record)
                if key=='pre_action_hashes': changed[key][0]='wrong'
                elif key=='states': changed[key][1][0]+=1
                elif key=='rewards': changed[key][0]+=1
                else: changed[key]['steps']+=1
                with self.subTest(key=key),self.assertRaises(ValueError): d.replay_episode(data,scene,changed)

    def test_joint_adam_same_initial_batches_frozen_networks_and_nonfinite(self):
        with mock.patch.object(b,'actor_action',return_value=np.zeros(2,dtype=np.float32)):
            data,scene,record = self.fixture()
        rows = d.replay_episode(data,scene,record)
        critics = [TinyCritic(),TinyCritic()]
        params = [p for c in critics for p in c.parameters()]
        opt = torch.optim.Adam(params,lr=.001)
        # Nonempty checkpoint Adam moments/step, not a fresh optimizer fixture.
        sum(p.square().sum() for p in params).backward(); opt.step(); opt.zero_grad(set_to_none=True)
        saved = deepcopy(opt.state_dict())
        data.update(critics=critics,actor=torch.nn.Linear(1,1).requires_grad_(False),device='cpu')
        data['checkpoint_payload']['engine_checkpoint']=dict(critic_optimizer_state_dict=saved,
            algorithm_config=dict(critic_grad_clip_norm=1.))
        targets=[torch.nn.Linear(1,1).requires_grad_(False) for _ in range(3)]
        for r in rows: r['y']=r['G'] # Identical labels must produce identical fits/curves.
        with mock.patch.object(d,'make_fit',wraps=d.make_fit) as make:
            result=d.fit(data,targets,rows,dict(train=['one'],holdout=[]),budget=2,evaluations=(0,1,2))
        self.assertEqual(make.call_count,2)
        self.assertEqual(result['initial']['fixed_td'],result['initial']['mc_return'])
        copies,restored,parameters=d.make_fit(critics,saved)
        self.assertEqual(len(restored.param_groups),1)
        self.assertEqual(len(parameters),len(params))
        self.assertEqual(d.state_hash(restored.state_dict()['state']),d.state_hash(saved['state']))
        self.assertTrue(result['frozen_unchanged'])
        for step in (0,1,2):
            a,c=[r for r in result['curve'] if r['updates']==step]
            self.assertEqual(a['metrics'],c['metrics'])
        self.assertEqual(len(result['gradient_norms']['fixed_td']),2)
        rows[0]['G']=float('inf')
        with self.assertRaises(ValueError): d.fit(data,targets,rows,dict(train=['one'],holdout=[]),budget=1,evaluations=(0,))

    def test_strict_target_weights_not_online_and_missing_rejected(self):
        models = [torch.nn.Linear(1,1) for _ in range(3)]
        names=('actor_target_state_dict','critic1_target_state_dict','critic2_target_state_dict')
        payload={name:{k:torch.full_like(v,7.) for k,v in m.state_dict().items()} for name,m in zip(names,models)}
        data=dict(actor=models[0],critics=models[1:],checkpoint_payload=dict(engine_checkpoint=payload))
        targets=d.load_targets(data)
        self.assertTrue(all(not p.requires_grad for m in targets for p in m.parameters()))
        self.assertTrue(all(torch.equal(p,torch.full_like(p,7.)) for m in targets for p in m.parameters()))
        del payload[names[0]]
        with self.assertRaises(KeyError): d.load_targets(data)

    def test_labels_adjacent_contrasts_terminal_and_independent_frozen_noise(self):
        class Actor(torch.nn.Module):
            def __init__(self, value):
                super().__init__(); self.register_buffer('value',torch.tensor(value))
            def forward(self, obs): return self.value.expand(obs.batch_size,2)
        with mock.patch.object(b,'actor_action',return_value=np.zeros(2,dtype=np.float32)):
            data,scene,record = self.fixture()
        rows=d.replay_episode(data,scene,record)
        data.update(actor=Actor(0.),critics=[TinyCritic(),TinyCritic()],device='cpu')
        data['checkpoint_payload']['engine_checkpoint']=dict(
            algorithm_config=dict(policy_noise=.006,noise_clip=.012),action_low=[-.14,-.2],action_high=[.14,.2])
        targets=[Actor(.01),TinyCritic(),TinyCritic()]
        rng=torch.random.get_rng_state().clone()
        first=d.compute_labels(data,targets,rows)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        old=deepcopy([{k:v for k,v in r.items() if k not in ('observation','next_observation','action')} for r in rows])
        self.assertEqual(first,d.compute_labels(data,targets,rows))
        self.assertEqual(old,[{k:v for k,v in r.items() if k not in ('observation','next_observation','action')} for r in rows])
        self.assertEqual(rows[-1]['y'],float(torch.tensor(rows[-1]['reward'],dtype=torch.float32)))
        self.assertEqual(rows[-1]['target_critic_shift'],0.)
        for row in rows:
            self.assertAlmostEqual(row['y']-row['online_noiseless_y'],
                row['target_critic_shift']+row['target_actor_shift']+row['smoothing_shift'],places=4)
            self.assertLessEqual(row['min_vs_mean_shift'],0.)
        data['checkpoint_payload']['engine_checkpoint']['algorithm_config']['policy_noise']=.015
        with self.assertRaisesRegex(ValueError,'terminal policy_noise'): d.compute_labels(data,targets,rows)

    def test_scene_equal_and_transition_weighted_statistics_are_distinct(self):
        rows=[dict(scenario_id=s,empty_scene=True,done=True,G=0.,y=0.) for s in ('a','a','b')]
        stats=d.statistics(rows,[[0.,0.],[0.,0.],[6.,6.]],dict(train=['a','b'],holdout=[]))
        first=stats[0]['critic1']
        self.assertEqual(first['transition_weighted']['Q_minus_G'],2.)
        self.assertEqual(first['scene_equal']['Q_minus_G'],3.)
        self.assertTrue(any(r.get('information')=='insufficient' for r in stats))

    def test_output_refuses_overwrite_and_failure_report(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=SimpleNamespace(output_dir=root,checkpoint=root/'missing',history_dir=root,device='cpu')
            with self.assertRaises(FileExistsError): d.run(args)
            args.output_dir=root/'new'
            with self.assertRaises(FileNotFoundError): d.run(args)
            import json
            self.assertFalse(json.loads((args.output_dir/'summary.json').read_text())['execution_completed'])

    def test_run_success_persists_initial_split_noise_and_fit_outputs(self):
        import json
        class Actor(torch.nn.Module):
            def forward(self, obs): return obs.ego_features.new_zeros((obs.batch_size,2))
        with mock.patch.object(b,'actor_action',return_value=np.zeros(2,dtype=np.float32)):
            data,scene,record=self.fixture()
        template=d.replay_episode(data,scene,record)[0]
        with TemporaryDirectory() as tmp:
            root=Path(tmp); history=root/'history'; history.mkdir()
            checkpoint=root/'checkpoint.pt'; checkpoint.write_bytes(b'mock checkpoint input')
            checkpoint_hash=d.file_sha256(checkpoint)
            pool_path=root/'pool.json'; pool_path.write_text('{}')
            records=[dict(model='B',scenario_id=str(i),sequence_index=i,empty_scene=i<7) for i in range(20)]
            (history/'baseline_episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
            (history/'manifest.json').write_text(json.dumps(dict(inputs=dict(B=dict(sha256=checkpoint_hash)),
                pool=dict(content_digest=b.POOL_DIGEST),selection=dict(scenarios=records))),encoding='utf-8')
            pool=SimpleNamespace(content_digest=b.POOL_DIGEST,scenarios=[dict(scenario_id=r['scenario_id'],
                sequence_index=r['sequence_index'],payload=dict(zones=[] if r['empty_scene'] else [1])) for r in records])
            data.update(model='snn',device='cpu',actor=Actor(),critics=[TinyCritic(),TinyCritic()])
            data['checkpoint_payload'].update(stage='easy',formal_config=dict(seed=23,critic_lr=.001),
                training_result=dict(stage_steps=500000),engine_checkpoint=dict(
                    algorithm_config=dict(batch_size=64,policy_noise=.006,noise_clip=.012),
                    critic_optimizer_state_dict=dict(param_groups=[dict(lr=.001)]),
                    action_low=[-.14,-.2],action_high=[.14,.2]))
            args=SimpleNamespace(output_dir=root/'output',checkpoint=checkpoint,history_dir=history,
                                 validation_pool=pool_path,device='cpu')
            manifest_path=args.output_dir/'manifest.json'
            def replay(data_arg,scene_arg,record_arg):
                first=json.loads(manifest_path.read_text(encoding='utf-8'))
                self.assertEqual(len(first['split']['train']),16)
                self.assertNotIn('noise_sha256',first)
                return [dict(deepcopy(template),scenario_id=record_arg['scenario_id'],empty_scene=record_arg['empty_scene'])]
            targets=[Actor(),TinyCritic(),TinyCritic()]
            result=dict(frozen_unchanged=True,initial={'mock':'identical'},curve=[dict(updates=1000,fit='fixed_td')])
            def fit(data_arg,targets_arg,rows,split):
                saved=json.loads(manifest_path.read_text(encoding='utf-8'))
                self.assertEqual(len(saved['noise_sha256']),64)
                self.assertEqual(saved['split'],split)
                self.assertTrue(all('y' in row and 'smoothing_noise' in row for row in rows))
                return result
            with mock.patch.object(d,'CHECKPOINT_SHA256',checkpoint_hash), \
                    mock.patch.object(d,'load_v2_validation_pool',return_value=pool), \
                    mock.patch.object(b,'load_models',return_value=data), \
                    mock.patch.object(d,'load_targets',return_value=targets), \
                    mock.patch.object(d,'replay_episode',side_effect=replay) as replay_call, \
                    mock.patch.object(d,'compute_labels',wraps=d.compute_labels) as labels, \
                    mock.patch.object(d,'fit',side_effect=fit) as fit_call, \
                    mock.patch.object(d.subprocess,'run',return_value=SimpleNamespace(stdout='mock source')), \
                    mock.patch.object(d,'_write_json',wraps=d._write_json) as writes:
                summary=d.run(args)
            self.assertEqual(replay_call.call_count,20) # Mocked replay, no real twenty-scene flight.
            labels.assert_called_once(); fit_call.assert_called_once()
            self.assertEqual(sum(call.args[0]==manifest_path for call in writes.call_args_list),1)
            self.assertEqual(json.loads((args.output_dir/'fit_curves.json').read_text()),result)
            saved=json.loads((args.output_dir/'summary.json').read_text())
            self.assertTrue(saved['execution_completed']); self.assertEqual(saved,summary)
            self.assertEqual(saved['transitions'],20)
