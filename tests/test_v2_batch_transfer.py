"""CPU/mock contracts only; CUDA transfer/update/speed remain server checks."""
from unittest import TestCase, mock
import tempfile
import numpy as np
import torch
from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
from brain_uav.scripts import v2_batch_transfer_diagnostic as diagnostic
from test_v2_td3 import TestV2TD3 as Fixture, _observation
from brain_uav.observations import collate_v2_observations


class TestBatchTransfer(TestCase):
    def setUp(self):
        self.fixture=Fixture(); self.fixture.setUp()

    def test_to_default_and_flag_reach_all_fourteen_tensors(self):
        e=self.fixture.make_engine(); self.fixture.fill_replay(e,counts=(0,2),next_counts=(2,0))
        batch=e.replay.sample(2); original=torch.Tensor.to
        for enabled in (False,True):
            calls=[]
            def transfer(t,*a,**kw):
                calls.append(kw['non_blocking']); return original(t,*a,**kw)
            with mock.patch.object(torch.Tensor,'to',new=transfer):
                actual=batch.to('cpu',non_blocking=True) if enabled else batch.to('cpu')
            self.assertEqual(calls,[enabled]*14)
            self.assertEqual(batch.sample_failure_fraction,actual.sample_failure_fraction)
            for name,tensor in diagnostic.fields(batch).items():
                torch.testing.assert_close(tensor,diagnostic.fields(actual)[name],rtol=0,atol=0)

    def test_engine_defaults_types_device_and_conflicts(self):
        e=self.fixture.make_engine()
        self.assertFalse(e.non_blocking_batch_transfer)
        self.assertFalse(e.local_optimization_metadata()['non_blocking_batch_transfer']['effective'])
        kwargs=dict(actor=e.actor,critic1=e.critic1,critic2=e.critic2,replay=e.replay,
            actor_lr=.001,critic_lr=.001,gamma=.99,tau=.01,policy_noise=.1,noise_clip=.2,
            policy_delay=2,batch_size=2,action_low=e.action_low,action_high=e.action_high)
        with self.assertRaisesRegex(TypeError,'must be a bool'):
            V2TD3UpdateEngine(**kwargs,non_blocking_batch_transfer=1)
        with self.assertRaisesRegex(ValueError,'requires CUDA'):
            V2TD3UpdateEngine(**kwargs,non_blocking_batch_transfer=True)
        for other in ('pinned_batch_transfer','gpu_td3_update'):
            with self.assertRaisesRegex(ValueError,'conflicts'):
                V2TD3UpdateEngine(**kwargs,device='cuda',non_blocking_batch_transfer=True,**{other:True})
        self.assertNotIn('non_blocking_batch_transfer',e.checkpoint_state_dict()['algorithm_config'])

    def test_cpu_samples_dynamic_masks_order_and_owned_source_storage(self):
        obs=collate_v2_observations([_observation(n,scales=self.fixture.scales) for n in (0,1,2)])
        batches=diagnostic.cpu_batches(obs,obs,2,7)
        repeated=diagnostic.cpu_batches(obs,obs,2,7)
        self.assertEqual([b.obs.max_zone_count for b in batches],[0,2,2])
        self.assertFalse(torch.equal(batches[1].reward,batches[2].reward))
        for a,b in zip(batches,repeated):
            for name,t in diagnostic.fields(a).items():
                torch.testing.assert_close(t,diagnostic.fields(b)[name],rtol=0,atol=0)
        e=self.fixture.make_engine(); self.fixture.fill_replay(e)
        batch=e.replay.sample(2); saved=batch.reward.clone()
        e.replay.reward.fill(99.)
        torch.testing.assert_close(batch.reward,saved,rtol=0,atol=0)

    def test_worker_translates_only_transfer_flag_and_defaults(self):
        from brain_uav.scripts.profile_v2_td3 import build_parser
        args=build_parser().parse_args(['--model','snn','--device','cuda','--bc-checkpoint','bc.pt',
            '--validation-pool','pool.json','--output-dir','unused','--batch-transfer-worker','correctness',
            '--non-blocking-batch-transfer'])
        with mock.patch('brain_uav.scripts.profile_v2_td3.run_local_optimization_comparison',return_value={}) as run:
            diagnostic.run(args)
        passed=run.call_args.args[0]
        self.assertTrue(passed.non_blocking_batch_transfer)
        self.assertEqual(passed.td3_block_choice,'nonblocking')
        self.assertTrue(passed.td3_block_correctness)
        self.assertFalse(passed.gpu_td3_update); self.assertFalse(passed.pinned_batch_transfer)
        self.assertFalse(passed.compile_actor_objective); self.assertFalse(passed.compile_critic_update_front)
        self.assertEqual(passed.correctness_updates,8)

    def test_failed_correctness_skips_all_timing_processes(self):
        from brain_uav.scripts.profile_v2_td3 import build_parser
        with tempfile.TemporaryDirectory() as directory:
            args=build_parser().parse_args(['--model','ann','--device','cuda','--bc-checkpoint','bc.pt',
                '--validation-pool','pool.json','--output-dir',directory,'--batch-transfer-comparison'])
            calls=[]
            def process(command,**kwargs):
                calls.append(command)
                return mock.Mock(returncode=1 if '--non-blocking-batch-transfer' in command else 0)
            with mock.patch('subprocess.run',new=process),self.assertRaisesRegex(RuntimeError,'timing skipped'):
                diagnostic.run(args)
            self.assertEqual(len(calls),2)
            self.assertTrue(all(c[c.index('--batch-transfer-worker')+1]=='correctness' for c in calls))
