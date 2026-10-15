"""CPU mechanisms only: not CUDA capture, graph-address or numerical evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock
import json
import random

import numpy as np
import torch

from brain_uav.scripts import v2_ann_common_state_diagnostic as a
from brain_uav.scripts import v2_gpu_compute_diagnostic as d
from test_v2_td3 import TestV2TD3 as Fixture


class TestANNCommonState(TestCase):
    def engine(self):
        fixture = Fixture(); fixture.setUp()
        engine = fixture.make_engine(policy_delay=2)
        fixture.fill_replay(engine, counts=(0, 2), next_counts=(0, 2))
        return engine

    def test_snapshot_is_numeric_owned_and_preserves_none(self):
        source = torch.ones(3, requires_grad=True)*2
        saved = a.owned(dict(tensor=source, absent=None))
        self.assertFalse(saved['tensor'].requires_grad)
        self.assertIsNone(saved['tensor'].grad_fn)
        self.assertNotEqual(source.data_ptr(), saved['tensor'].data_ptr())
        torch.testing.assert_close(source, saved['tensor'], rtol=0, atol=0)
        self.assertIsNone(saved['absent'])
        source.add_(1)
        self.assertTrue(torch.equal(saved['tensor'], torch.full((3,), 2.)))

    def test_real_update_warmup_restores_adam_grad_refs_counts_and_rng(self):
        e = self.engine()
        e.update_once(total_steps=2, bc_lambda=0.)  # Existing nonempty Adam state.
        before = a.values(e)
        params = list(e.actor.parameters())+list(e.critic1.parameters())+list(e.critic2.parameters())
        refs = [p.grad for p in params]
        tensors = [p for p in params]+[v for o in (e.actor_optimizer,e.critic_optimizer)
            for s in o.state.values() for v in s.values() if isinstance(v,torch.Tensor)]
        addresses = [t.data_ptr() for t in tensors]
        with a.restored_warmup(e):
            e.update_once(total_steps=4, bc_lambda=0.)
            random.random(); np.random.random(); torch.rand(1)
        self.assertTrue(d.state_difference(before, a.values(e), exact=True)['passed'])
        self.assertTrue(all(p.grad is ref for p,ref in zip(params,refs)))
        self.assertEqual(addresses, [t.data_ptr() for t in tensors])

    def test_noise_observer_uses_actual_draw_and_rejects_different_rng(self):
        e = self.engine()
        state = a.rng_state(e)
        expected = torch.randn_like(torch.ones(2,2))
        focus = a.Focus('compute',Path('.'),Path('.'))
        focus.bundle = dict(noise=expected)
        a.restore_rng(e,state)
        with focus.observe_noise(0):
            actual = torch.randn_like(torch.ones(2,2))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        with self.assertRaises(AssertionError), focus.observe_noise(0):
            torch.randn_like(torch.ones(2,2))

    def test_warmup_restores_unpublished_adam_membership_and_bank_choice(self):
        e = self.engine()
        p = next(e.actor.parameters())
        bank = torch.ones_like(p)
        executor = SimpleNamespace(persistent=lambda: [], banks=[[[bank]]],
            published=[None,None], native_states=[[None]])
        e._gpu_update_executor = executor
        with a.restored_warmup(e):
            state = {'step':torch.tensor(1.)}
            e.actor_optimizer.state[p] = state
            executor.native_states[0][0] = state
            executor.published[0] = 0
            bank.add_(2)
        self.assertNotIn(p,e.actor_optimizer.state)
        self.assertIsNone(executor.native_states[0][0])
        self.assertEqual(executor.published,[None,None])
        self.assertTrue(torch.equal(bank,torch.ones_like(bank)))

    def test_history_hook_exports_only_26_and_detaches_all_evidence(self):
        e = self.engine()
        e.actor_freeze_steps = 25000
        batch = e.replay.sample(2)
        with TemporaryDirectory() as directory:
            path = Path(directory)/'state.pt'
            focus = a.Focus('history',Path(directory),path)
            focus.contract = dict(schedule='smooth-final')
            focus.before_step(e,25,75026,batch,None,None)
            self.assertIsNone(focus.bundle)
            focus.before_step(e,26,75028,batch,None,None)
            saved_weight = next(iter(focus.bundle['engine']['actor_state_dict'].values())).clone()
            with focus.observe_noise(26):
                torch.randn_like(batch.action)
            with self.assertRaises(a.HistoryReady):
                focus.after_step(e,26,{'example':torch.ones(1,requires_grad=True)})
            bundle = torch.load(path,weights_only=False)
            self.assertEqual(bundle['total'],75028)
            self.assertEqual(bundle['index'],26)
            self.assertTrue(bundle['post_clip_key'].startswith('grad.critic2.'))
            self.assertIsNone(bundle['continuous_after']['example'].grad_fn)
            with torch.no_grad(): next(e.actor.parameters()).add_(1)
            torch.testing.assert_close(next(iter(bundle['engine']['actor_state_dict'].values())),saved_weight)

    def test_common_state_worker_warmup_then_exact_alignment_then_one_update_cpu(self):
        from brain_uav.scripts.profile_v2_td3 import v2_bc_lambda
        e = self.engine()
        e.bc_reference_actor = e._frozen_target(e.actor,device=e.device)
        batch = e.replay.sample(2)
        with TemporaryDirectory() as directory:
            root = Path(directory); (root/'other').mkdir()
            focus = a.Focus('baseline',root,root/'other/common_state.pt')
            focus.contract = dict(schedule='smooth-final')
            focus.before_step(e,26,75028,batch,None,None)  # Replay hook is intentionally inert.
            history = a.Focus('history',root,focus.source); history.contract = focus.contract
            history.before_step(e,26,75028,batch,None,None)
            focus.bundle = history.bundle
            rng = a.rng_state(e)
            focus.bundle['noise'] = torch.randn_like(batch.action)
            a.restore_rng(e,rng)
            calls=[]
            def update(total,source):
                calls.append(total)
                e.replay.sample = lambda size: source
                with focus.observe_noise(0):
                    e.update_once(total_steps=total,bc_lambda=v2_bc_lambda(total,schedule='smooth-final'))
                return a.values(e)
            def profile(callback):
                callback()
                return {'cuda_graph_launch_count':1}  # CPU orchestration mock only.
            with mock.patch.object(e,'_profile_cuda_graph_replay',side_effect=profile):
                focus.replay(e,update)
            self.assertEqual(calls,[75028,75028])
            self.assertEqual(e.update_count,1)
            result = torch.load(root/'other/baseline_result.pt',weights_only=False)
            self.assertTrue(result['alignment']['passed'])
            self.assertEqual(e.actor_update_count,1)

    def test_target_report_and_update_delta_detect_small_difference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); (root/'other').mkdir(); (root/'review').mkdir()
            key = 'tensor.gradient.pre_clip.'+a.TARGET
            expected = torch.ones(64); expected[19] = -.0035659894347190857
            actual = expected.clone(); actual[19] = -.003552636131644249
            for worker,tensor in (('baseline',expected),('compute',actual)):
                torch.save(dict(snapshot={key:tensor,'grad.critic2.3':tensor},
                    deltas={'critic2.weight':torch.tensor([0. if worker=='baseline' else 2e-5])},
                    precision={}, alignment={'passed':True}, graph_evidence={'cuda_graph_launch_count':1}),
                    root/'other'/f'{worker}_result.pt')
            torch.save(dict(post_clip_key='grad.critic2.3'), root/'other/common_state.pt')
            self.assertFalse(a.compare(root))
            report = json.loads((root/'review/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['target']['pre_clip']['mismatches'],1)
            self.assertFalse(report['parameter_deltas']['passed'])
            self.assertGreater(report['target']['pre_clip']['index19']['absolute_error'],
                report['target']['pre_clip']['index19']['allowed_error'])

    def test_interpretation_requires_continuous_baseline_reproduction(self):
        for continuous in ('missing', 'different', 'equal'):
            with self.subTest(continuous=continuous), TemporaryDirectory() as directory:
                root = Path(directory); (root/'other').mkdir(); (root/'review').mkdir()
                key = 'tensor.gradient.pre_clip.'+a.TARGET
                snapshot = {key:torch.ones(64), 'grad.critic2.3':torch.ones(64)}
                result = dict(snapshot=snapshot, deltas={'critic2.weight':torch.zeros(1)},
                    precision={}, alignment={'passed':True}, graph_evidence={'cuda_graph_launch_count':1})
                for worker in ('baseline','compute'):
                    torch.save(result,root/'other'/f'{worker}_result.pt')
                bundle = dict(post_clip_key='grad.critic2.3')
                if continuous != 'missing':
                    bundle['continuous_after'] = {name:t.clone() for name,t in snapshot.items()}
                    if continuous == 'different':
                        bundle['continuous_after'][key][19] += 1.
                torch.save(bundle,root/'other/common_state.pt')
                self.assertTrue(a.compare(root))  # Independent same-state result remains intact.
                report = json.loads((root/'review/summary.json').read_text(encoding='utf-8'))
                self.assertTrue(report['comparison']['passed'])
                if continuous == 'equal':
                    self.assertIn('支持历史状态差异参与原失败',report['interpretation'])
                else:
                    self.assertIn('连续基准未复现，原因定位证据不足',report['interpretation'])
                    self.assertNotIn('支持历史状态差异参与原失败',report['interpretation'])

    def test_existing_history_reaches_26_without_checkpoint_or_timing(self):
        from brain_uav.scripts import profile_v2_td3 as p, train_v2_td3 as single
        from brain_uav.config import RewardConfig
        e = self.engine(); e.actor_freeze_steps = 25000
        e.bc_reference_actor = e._frozen_target(e.actor,device=e.device)
        config = SimpleNamespace(to_dict=lambda: {},policy_delay=2,actor_freeze_steps=25000,
            spike_regularization=e.spike_regularization,batch_size=e.batch_size,max_steps=500000,
            noise_schedule=SimpleNamespace(values=lambda *a,**kw:(0.,.01,.02)))
        observation = e.replay.sample(2).obs
        with TemporaryDirectory() as directory:
            root = Path(directory); (root/'other/history').mkdir(parents=True); (root/'review').mkdir()
            bc = root/'bc.pt'; bc.write_bytes(b'fixture')
            pool = root/'pool.json'; pool.write_text('{}')
            focus = a.Focus('history',root,root/'other/common_state.pt')
            args = SimpleNamespace(model='ann',seed=7,bc_checkpoint=bc,validation_pool=pool,
                gpu_td3_compute=False,snn_time_window=4,correctness_updates=8,
                comparison_bc_schedule='smooth-final',_ann_common_state=focus)
            with mock.patch.object(p,'build_v2_stage_engine',return_value=SimpleNamespace(engine=e)), \
                    mock.patch.object(single,'_configure_stage_compilation',return_value={}), \
                    mock.patch.object(p,'_compile_warmup_batches',return_value=[observation]), \
                    mock.patch.object(p,'_numeric_diagnostic_observation_batches',return_value=(observation,observation)), \
                    mock.patch.object(d.subprocess,'run',return_value=SimpleNamespace(stdout='fixture')), \
                    mock.patch.object(torch.cuda,'get_rng_state',return_value=torch.tensor([1],dtype=torch.uint8)):
                report = d.correctness(args,prepared=SimpleNamespace(reward_config=RewardConfig()),
                    config=config,pool=SimpleNamespace(content_digest='fixture'),device='cpu',
                    output=root/'other/history',full_pool_digest='fixture')
            self.assertEqual(len(report['checks']),27)
            self.assertEqual(report['checks'][-1]['total_steps'],75028)
            self.assertEqual(report['location']['index'],26)
            self.assertIsNone(report['location']['failure'])
            self.assertNotIn('load_rebuild_verified',report)
            self.assertTrue(focus.source.is_file())
            prep = json.loads((root/'review/history_preparation.json').read_text())
            self.assertFalse(prep['full_acceptance'])

    def test_driver_serial_processes_no_snn_no_timing_and_failure_stops(self):
        for failed_worker in (None,'baseline'):
            with self.subTest(failed_worker=failed_worker), TemporaryDirectory() as directory:
                output = Path(directory)/'new'
                calls=[]
                def child(command,**kwargs):
                    calls.append(command)
                    return SimpleNamespace(returncode=int(command[-1]==failed_worker))
                with mock.patch.object(a.subprocess,'run',side_effect=child), \
                        mock.patch.object(a,'compare',return_value=True):
                    args=['--model','ann','--device','cuda','--output-dir',str(output)]
                    if failed_worker:
                        with self.assertRaisesRegex(RuntimeError,'baseline'):
                            a.main(args)
                    else:
                        a.main(args)
                self.assertEqual([c[-1] for c in calls],
                    ['history','baseline'] if failed_worker else ['history','baseline','compute'])
                self.assertTrue(all('snn' not in c and '--gpu-compute-comparison' not in c for c in calls))
                self.assertEqual(json.loads((output/'review/driver.json').read_text())['completed'],
                    failed_worker is None)
