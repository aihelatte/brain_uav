"""CPU mechanism checks only; never CUDA capture/performance evidence."""
from copy import deepcopy
from unittest import TestCase, mock
import numpy as np
import torch
from brain_uav.trainers.v2_gpu_replay import DeviceReplay
from brain_uav.trainers.v2_gpu_update import GPUUpdate, DeviceAdam, checked_clip
from brain_uav.trainers.v2_replay_buffer import V2ReplayBuffer
from brain_uav.observations import collate_v2_observations, V2ObservationBatch
from test_v2_td3 import TestV2TD3, _observation
from test_v2_spike_regularization import TestSpikeRegularization


class TestGPUUpdate(TestCase):
    def test_controlled_mixed_sampler_oracle_ring_events_and_rng(self):
        from brain_uav.scripts.v2_gpu_td3_diagnostic import controlled_cpu_sample, controlled_cpu_indices, expected_rows
        f=TestV2TD3(); f.setUp()
        r=V2ReplayBuffer(8,2,3,seed=9,failure_sample_bias=2.,near_goal_sample_bias=2.,success_batch_fraction=.5)
        d=DeviceReplay(r,'cpu'); refs=[]; transitions=[]
        for i in range(10):
            obs=_observation(i%4,scales=f.scales)
            transition=(obs,np.zeros(2,np.float32),float(i),obs,i==9)
            refs.append(r.add(*transition,near_goal=i%2==0)); transitions.append(transition)
        self.assertEqual(r.mark_failure_slots(refs[:2]),0)
        r.mark_failure_slots(refs[-3:])
        for size in (0,1,2):
            while r.success_size<size:
                with d.success_upload_batch():
                    r.add_success_transition(*transitions[r.success_size],near_goal=True)
            if size==2: r.mark_success_slots(refs[-2:])
            state=deepcopy(r.rng.bit_generator.state); tree=d.tree.clone()
            count=min(2,r.success_size)
            draws=np.array([.05,.25,.55,.95],np.float64); ranks=np.zeros(4,np.int64)
            p,s,b=controlled_cpu_sample(r,draws,ranks,4)
            pi,si=controlled_cpu_indices(r,draws,ranks,4)
            np.testing.assert_array_equal(p,pi); np.testing.assert_array_equal(s,si)
            self.assertEqual(state,r.rng.bit_generator.state)
            self.assertEqual(len(set(p)),4-count); self.assertEqual(len(set(s)),count)
            rows,failed=d.sample_tensors(torch.from_numpy(draws),torch.from_numpy(ranks),
                torch.tensor(4-count),torch.tensor(r.success_size),return_status=True)
            self.assertFalse(bool(failed))
            torch.testing.assert_close(rows,expected_rows(d,p,s),rtol=0,atol=0)
            torch.testing.assert_close(tree,d.tree,rtol=0,atol=0)
            torch.testing.assert_close(b.reward,rows[:,d.layout['reward'][0]],rtol=0,atol=0)
            obs,_,_=d.unpack(rows)
            padded=torch.zeros_like(obs.presence_mask); padded[:,:b.obs.presence_mask.shape[1]]=b.obs.presence_mask
            self.assertTrue(torch.equal(padded,obs.presence_mask))
        d.detach()

    def test_sampling_cost_entry_is_standalone_and_records_cuda_failure(self):
        import tempfile, json
        from pathlib import Path
        from brain_uav.scripts.profile_v2_td3 import build_parser
        from brain_uav.scripts.v2_gpu_td3_diagnostic import run, REFERENCE_VERSION
        self.assertEqual(REFERENCE_VERSION,2)
        with tempfile.TemporaryDirectory() as directory:
            args=build_parser().parse_args(['--model','ann','--device','cuda','--gpu-td3-sampling-cost',
                '--replay-capacity','500000','--output-dir',directory])
            self.assertIsNone(args.bc_checkpoint)
            with mock.patch('brain_uav.trainers.v2_gpu_update.require_cuda_abi',side_effect=RuntimeError('CUDA unavailable')):
                with self.assertRaisesRegex(RuntimeError,'CUDA unavailable'): run(args)
            report=json.loads((Path(directory)/'summary.json').read_text())
            self.assertFalse(report['passed']); self.assertFalse(report['completed'])
            self.assertEqual(report['checks'],[]); self.assertEqual(report['location']['stage'],'capability')
            self.assertIn('RuntimeError: CUDA unavailable',report['failure']['traceback'])

    def test_sampling_cost_validates_capture_outputs_only_after_replay_and_sync(self):
        import tempfile
        from contextlib import ExitStack, contextmanager, nullcontext
        from brain_uav.scripts.profile_v2_td3 import build_parser
        from brain_uav.scripts.v2_gpu_td3_diagnostic import sampling_cost
        # CPU/mock execution of the real entry point. Capture returns invalid
        # output buffers; only replay produces values, then sync makes them
        # observable. This models ordering, not CUDA capture correctness.
        graphs=[]; capturing=[None]; events=[]
        original_sample=DeviceReplay.sample_tensors
        original_tensor=torch.tensor
        original_close=torch.testing.assert_close
        class Graph:
            def __init__(self):
                self.rows=None; self.replays=0; self.synced=False; graphs.append(self)
            def replay(self):
                self.rows.copy_(self.expected); self.failed.fill_(False)
                self.replays+=1; self.synced=False; events.append('replay')
        @contextmanager
        def capture(graph,**kwargs):
            capturing[0]=graph
            try: yield
            finally: capturing[0]=None; events.append('capture_complete')
        def sample(adapter,*args,**kwargs):
            rows,failed=original_sample(adapter,*args,**kwargs)
            if capturing[0] is not None:
                graph=capturing[0]
                graph.rows=rows; graph.failed=failed; graph.expected=rows.detach().clone()
                rows.fill_(float('nan')); failed.fill_(True)
            return rows,failed
        def synchronize():
            for graph in graphs:
                if graph.replays: graph.synced=True
            events.append('sync')
        def close(actual,expected,**kwargs):
            for graph in graphs:
                if actual is graph.rows:
                    self.assertGreater(graph.replays,0)
                    self.assertTrue(graph.synced,'Captured output checked without replay completion synchronization.')
                    events.append('validate')
            return original_close(actual,expected,**kwargs)
        def tensor(*args,**kwargs):
            if kwargs.get('device')=='cuda': kwargs['device']='cpu'
            return original_tensor(*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            args=build_parser().parse_args(['--model','ann','--device','cuda','--gpu-td3-sampling-cost',
                '--replay-capacity','500000','--sampling-warmup','1','--sampling-replays','2',
                '--repeats','1','--output-dir',directory])
            stack.enter_context(mock.patch('brain_uav.trainers.v2_gpu_update.require_cuda_abi'))
            stack.enter_context(mock.patch('brain_uav.trainers.v2_replay_buffer.V2ReplayBuffer',
                new=lambda capacity,*a,**kw: V2ReplayBuffer(96,*a,**kw)))
            stack.enter_context(mock.patch('brain_uav.trainers.v2_gpu_replay.DeviceReplay',
                new=lambda replay,device: DeviceReplay(replay,'cpu')))
            stack.enter_context(mock.patch.object(DeviceReplay,'sample_tensors',new=sample))
            stack.enter_context(mock.patch.object(torch.Tensor,'cuda',new=lambda self:self))
            stack.enter_context(mock.patch('torch.tensor',new=tensor))
            stream=mock.Mock()
            stack.enter_context(mock.patch('torch.cuda.Stream',return_value=stream))
            stack.enter_context(mock.patch('torch.cuda.current_stream',return_value=stream))
            stack.enter_context(mock.patch('torch.cuda.stream',new=lambda stream: nullcontext()))
            stack.enter_context(mock.patch('torch.cuda.CUDAGraph',new=Graph))
            stack.enter_context(mock.patch('torch.cuda.graph',new=capture))
            stack.enter_context(mock.patch('torch.cuda.synchronize',new=synchronize))
            for name in ('reset_peak_memory_stats','memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
                stack.enter_context(mock.patch('torch.cuda.'+name,return_value=0))
            stack.enter_context(mock.patch('torch.cuda.get_device_name',return_value='mock; not CUDA evidence'))
            stack.enter_context(mock.patch('subprocess.check_output',return_value='mock'))
            stack.enter_context(mock.patch('torch.testing.assert_close',new=close))
            report=sampling_cost(args)
        self.assertTrue(report['passed']); self.assertEqual(len(graphs),2)
        self.assertEqual(events.count('validate'),4) # first replay + end of measurement, both pools
        for index,event in enumerate(events):
            if event=='capture_complete':
                first_check=events.index('validate',index)
                self.assertEqual(events[index+1:first_check],['sync','replay','sync'])

    def test_old_gpu_reference_is_explicitly_rejected(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from brain_uav.scripts.v2_gpu_td3_diagnostic import correctness
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.pt'; torch.save({'contract':{'version':1}},path)
            args=SimpleNamespace(seed=7,bc_checkpoint=Path('bc.pt'),model='snn',gpu_td3_update=True,gpu_td3_reference=path)
            engine=mock.Mock(); engine.checkpoint_state_dict.return_value={}
            with mock.patch('brain_uav.scripts.profile_v2_td3.build_v2_stage_engine',return_value=SimpleNamespace(engine=engine)):
                with self.assertRaisesRegex(ValueError,'Old GPU correctness reference rejected'):
                    correctness(args,None,SimpleNamespace(reward_config=None),None,
                        {'fused_adam':True,'reduce_update_stat_syncs':True},{'contract':{'version':2}})
            engine.load_checkpoint_state_dict.assert_not_called()

    def test_gpu_compilation_warmup_uses_no_grad_and_skips_old_backward(self):
        from types import SimpleNamespace
        from brain_uav.scripts.train_v2_td3 import _configure_stage_compilation
        f=TestV2TD3(); f.setUp(); e=f.make_engine()
        e.gpu_td3_update=True  # Test scheduling only, not CPU execution support.
        e.configure_compilation=mock.Mock(return_value={})
        e.warmup_shared_relations_compile=mock.Mock()
        for name in ('warmup_actor_compile','warmup_full_compile','warmup_actor_loss_compile','verify_update_cuda_graph_capture'):
            setattr(e,name,mock.Mock(side_effect=AssertionError('Old backward/capture warmup must not run.')))
        e.warmup_action_inference_compile=mock.Mock()
        observed=[]
        hook=e.actor.register_forward_hook(lambda m,a,out: observed.append((torch.is_grad_enabled(),out.requires_grad)))
        pool=SimpleNamespace(stage_seed=7,scenarios=[{'payload':{'zones':[1]}}])
        prepared=SimpleNamespace(scenario_config=None,reward_config=None,uav_collision_radius=0.)
        env=mock.Mock(); env.reset.return_value=(_observation(1,scales=e.actor.scales),{})
        try:
            with mock.patch('brain_uav.scripts.train_v2_td3.V2StaticNoFlyTrajectoryEnv',return_value=env):
                _configure_stage_compilation(e,pool,prepared,compile_critic_encoder=False,
                    compile_target_encoders=False,frozen_critic_strategy='compiled_no_grad_context',
                    compile_actors=True,compile_critic_block=True,
                    compile_target_block=True,compile_shared_relations=True,compile_actor_loss=True,
                    compile_action_inference=True)
        finally: hook.remove()
        self.assertTrue(observed)
        self.assertTrue(all(pair==(False,False) for pair in observed))
        self.assertTrue(all(p.grad is None for p in e.actor.parameters()))
        e.warmup_action_inference_compile.assert_called_once()

    def test_replay_events_generations_sampler_restore_and_padding(self):
        f=TestV2TD3(); f.setUp()
        r=V2ReplayBuffer(8,2,3,seed=9,failure_sample_bias=2.)
        d=DeviceReplay(r,'cpu'); refs=[]
        for i in range(10):
            o=_observation(i%4,scales=f.scales)
            refs.append(r.add(o,np.zeros(2,np.float32),float(i),o,False))
        self.assertEqual(r.mark_failure_slots(refs[:2]),0)
        self.assertEqual(r.mark_failure_slots(refs[-3:]),3)
        torch.testing.assert_close(d.weights,torch.from_numpy(r.sample_weight))
        tree=d.tree.clone()
        # Fixed random draws exactly replay the existing primary algorithm.
        state=deepcopy(r.rng.bit_generator.state)
        draws=r.rng.random(4); r.rng.bit_generator.state=state
        expected=r._sample_primary_indices(4)
        rows=d.sample_tensors(torch.from_numpy(draws),torch.zeros(4,dtype=torch.long),torch.tensor(4),torch.tensor(0))
        slot,_=d.layout['reward']
        torch.testing.assert_close(rows[:,slot].flatten(),torch.from_numpy(r.reward[expected].flatten()),rtol=0,atol=0)
        torch.testing.assert_close(tree,d.tree,rtol=0,atol=0)
        _,failed=d.sample_tensors(torch.full((4,),float('nan'),dtype=torch.float64),
            torch.zeros(4,dtype=torch.long),torch.tensor(4),torch.tensor(0),return_status=True)
        self.assertTrue(bool(failed))
        torch.testing.assert_close(tree,d.tree,rtol=0,atol=0)
        obs,_,_=d.unpack(rows)
        self.assertEqual(obs.zone_features.shape,(4,3,19))
        for i in range(4):
            self.assertEqual(int(obs.presence_mask[i].sum()),int(r.zone_count[expected[i]]))
        original=r.add; d.detach()
        self.assertNotEqual(r.add,original)

    def test_uniform_success_no_replacement_order_and_events(self):
        f=TestV2TD3(); f.setUp()
        r=V2ReplayBuffer(16,2,2,success_replay_fraction=.5,success_batch_fraction=.5)
        d=DeviceReplay(r,'cpu')
        with d.success_upload_batch():
            for i in range(8):
                o=_observation(i%3,scales=f.scales)
                r.add(o,np.zeros(2,np.float32),i,o,False)
                r.add_success_transition(o,np.zeros(2,np.float32),100+i,o,True)
            with self.assertRaisesRegex(RuntimeError,'Finish episode'):
                d.controls(4)
        self.assertEqual(d.submissions,42) # eight primary writes, one success upload
        rows=d.sample_tensors(torch.tensor([.1,.2,.3,.4],dtype=torch.float64),
            torch.tensor([0,0,0,0]),torch.tensor(2),torch.tensor(8))
        slot,_=d.layout['reward']; rewards=rows[:,slot].flatten().tolist()
        self.assertTrue(all(x<100 for x in rewards[:2]))
        self.assertEqual(rewards[2:],[100.,107.])
        self.assertGreater(d.seconds['episode_sync'],0)

    def test_default_conflicts_abi_and_load_invalidation(self):
        f=TestV2TD3(); f.setUp(); e=f.make_engine()
        self.assertFalse(e.gpu_td3_update)
        payload=deepcopy(e.checkpoint_state_dict())
        executor=mock.Mock(); e._gpu_update_executor=executor
        e.load_checkpoint_state_dict(payload)
        executor.invalidate.assert_called_once(); self.assertIsNone(e._gpu_update_executor)
        self.assertNotIn('gpu_td3_update',payload['algorithm_config'])
        budget=GPUUpdate.memory_budget(e,500000)
        self.assertEqual(budget['capacity'],500000)
        self.assertEqual(budget['success_capacity'],125000)
        self.assertFalse(budget['fit_validated'])
        self.assertGreater(budget['total_resident_reservation_bytes'],budget['replay_bytes'])
        from brain_uav.scripts.train_v2_td3 import build_parser
        # Parser default doesn't enable the experiment.
        self.assertFalse(build_parser().get_default('gpu_td3_update'))
        from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
        args=(e.actor,e.critic1,e.critic2,e.replay,.001,.001,.99,.01,.01,.02,2,2,e.action_low,e.action_high)
        with self.assertRaisesRegex(ValueError,'conflicts'):
            V2TD3UpdateEngine(*args,gpu_td3_update=True,compile_actor_objective=True)
        with self.assertRaisesRegex(ValueError,'requires CUDA'):
            V2TD3UpdateEngine(*args,gpu_td3_update=True)
        with self.assertRaises(TypeError):
            V2TD3UpdateEngine(*args,gpu_td3_update=1)

    def test_serial_suite_failure_skips_candidate_and_timing(self):
        import tempfile
        from pathlib import Path
        from brain_uav.scripts.profile_v2_td3 import build_parser
        from brain_uav.scripts.v2_gpu_td3_diagnostic import run
        with tempfile.TemporaryDirectory() as directory:
            args=build_parser().parse_args(['--model','ann','--bc-checkpoint','ann.pt',
                '--snn-bc-checkpoint','snn.pt','--validation-pool','pool.json',
                '--output-dir',directory,'--device','cuda','--gpu-td3-suite'])
            calls=[]
            def failed(command,**kwargs):
                calls.append(command)
                return mock.Mock(returncode=1)
            with mock.patch('subprocess.run',new=failed),self.assertRaisesRegex(RuntimeError,'failures'):
                run(args)
            self.assertEqual(len(calls),2)
            self.assertEqual([c[c.index('--model')+1] for c in calls],['ann','snn'])
            self.assertTrue(all('correctness' in c and 'timing' not in c for c in calls))
            import json
            saved=json.loads((Path(directory)/'summary.json').read_text())
            self.assertFalse(saved['passed'])
            self.assertEqual(saved['configurations'][1]['skipped'],'Baseline correctness unavailable')

    def test_sampling_correctness_failure_skips_gpu_timing(self):
        import tempfile, json
        from pathlib import Path
        from brain_uav.scripts.profile_v2_td3 import build_parser
        from brain_uav.scripts.v2_gpu_td3_diagnostic import run
        with tempfile.TemporaryDirectory() as directory:
            args=build_parser().parse_args(['--model','ann','--bc-checkpoint','ann.pt',
                '--snn-bc-checkpoint','snn.pt','--validation-pool','pool.json',
                '--output-dir',directory,'--device','cuda','--gpu-td3-suite'])
            calls=[]
            def result(command,**kwargs):
                calls.append(command)
                enabled='--gpu-td3-update' in command
                return mock.Mock(returncode=1 if enabled else 0)
            with mock.patch('subprocess.run',new=result),self.assertRaisesRegex(RuntimeError,'failures'):
                run(args)
            self.assertEqual(len(calls),6)
            self.assertTrue(all('correctness' in c for c in calls if '--gpu-td3-update' in c))
            saved=json.loads((Path(directory)/'summary.json').read_text())
            for item in saved['configurations']:
                if item['group']=='gpu':
                    self.assertNotIn('timing',item); self.assertFalse(item['passed'])

    def test_curriculum_forwards_flag_all_stages_without_training(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from brain_uav.config import ScenarioConfig
        from brain_uav.scripts.run_v2_td3_curriculum import run_v2_curriculum
        from brain_uav.scripts.train_v2_td3 import default_v2_cuda_graph_compilation
        from test_run_v2_td3_curriculum import _prepared_initialization,derive_validation_stage_seed
        calls=[]
        def stage(**kwargs):
            calls.append(kwargs)
            return dict(stage=kwargs['stage'],passed=True,checkpoint=str(kwargs['output']),
                metrics=str(kwargs['metrics_out']),steps=1,outcome_counts={'goal':1},
                validation_result={'passed':True},stop_reason='fixed_validation_passed')
        def pool(path,**kwargs):
            level=Path(path).stem
            return SimpleNamespace(curriculum_level=level,master_seed=20260904,
                stage_seed=derive_validation_stage_seed(20260904,level),scenario_count=100,content_digest=level+'-test')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); bc=root/'bc.pt'; bc.write_bytes(b'test double; never loaded')
            prepared=_prepared_initialization(ScenarioConfig(),source=bc)
            prefix='brain_uav.scripts.run_v2_td3_curriculum.'
            with mock.patch(prefix+'prepare_v2_stage_initialization',return_value=prepared), \
                 mock.patch(prefix+'prepare_v2_validation_pools',return_value={s:root/(s+'.json') for s in ('easy','medium','hard')}), \
                 mock.patch(prefix+'load_v2_validation_pool',side_effect=pool), \
                 mock.patch(prefix+'resolve_training_device',return_value='cuda'):
                run_v2_curriculum(bc_checkpoint=bc,output_root=root/'run',validation_pool_dir=root/'pools',
                    device='cuda',seed=7,max_stage='hard',gpu_td3_update=True,stage_runner=stage,
                    **default_v2_cuda_graph_compilation(model='ann',resolved_device='cuda'))
        self.assertEqual([c['stage'] for c in calls],['easy','medium','hard'])
        self.assertTrue(all(c['gpu_td3_update'] and c['fused_adam'] for c in calls))
        self.assertEqual(calls[1]['init_checkpoint'],calls[0]['output'])
        self.assertEqual(calls[2]['init_checkpoint'],calls[1]['output'])

    def test_real_raw_update_cpu_same_shapes_equivalence(self):
        # Outer CUDA capture is NOT mocked into a success. Exercise only real
        # tensor math using CPU native Adam as a local arithmetic reference.
        for model in ('ann','snn'):
            f=TestV2TD3(); f.setUp()
            a=f.make_engine(terminal_enabled=True) if model=='ann' else TestSpikeRegularization().engine(freeze=25000,delay=2)
            if model=='ann': f.fill_replay(a)
            b=f.make_engine(terminal_enabled=True) if model=='ann' else TestSpikeRegularization().engine(freeze=25000,delay=2)
            if model=='ann': f.fill_replay(b)
            b.load_checkpoint_state_dict(deepcopy(a.checkpoint_state_dict()))
            u=GPUUpdate.__new__(GPUUpdate); u.e=b
            u.diagnostic=True; u.tensor_evidence={}
            u.actor_params=list(b.actor.parameters()); u.critic_params=list(b.critic1.parameters())+list(b.critic2.parameters())
            u.initialize_actor_loss_scalars()
            class CPUAdam:
                def __init__(self,opt): self.opt=opt
                def step(self,error):
                    if not bool(error): self.opt.step()
            u.adams=(CPUAdam(b.actor_optimizer),CPUAdam(b.critic_optimizer))
            u.replay=DeviceReplay(b.replay,'cpu')
            u.coefficients=torch.tensor([0.,.5,b.policy_noise,b.noise_clip])
            u.injections=torch.zeros(4); u.fixed_mode=torch.tensor(True); u.use_fixed_rows=True
            for step in (24999,25000,25002,25003,50000,75000):
                if step>=50000:
                    from brain_uav.scripts.v2_gpu_td3_diagnostic import controlled_cpu_sample
                    draws=np.array([.2,.8],np.float64); ranks=np.zeros(2,np.int64)
                    _,_,batch=controlled_cpu_sample(a.replay,draws,ranks,2)
                    u.draws=torch.from_numpy(draws); u.ranks=torch.from_numpy(ranks)
                    u.primary_count=torch.tensor(2); u.success_size=torch.tensor(0)
                    u.replay_size=torch.tensor(b.replay.size)
                    u.weight_total=torch.tensor(b.replay.total_sample_weight,dtype=torch.float64)
                    u.use_fixed_rows=False; u.fixed_mode.fill_(False)
                else: batch=a.replay.sample(2)
                # Isolate real TD3 formula/optimizer equivalence from changing
                # GEMM dimensions. Dynamic-vs-fixed padding is a separate server
                # acceptance gate, not proved by this arithmetic test.
                from dataclasses import replace
                def padded(obs):
                    z=torch.zeros(2,b.replay.zone_storage_capacity,19)
                    mask=torch.zeros(2,b.replay.zone_storage_capacity,dtype=torch.bool)
                    z[:,:obs.zone_features.shape[1]]=obs.zone_features
                    mask[:,:obs.presence_mask.shape[1]]=obs.presence_mask
                    return V2ObservationBatch(obs.ego_features,obs.goal_features,z,mask)
                batch=replace(batch,obs=padded(batch.obs),next_obs=padded(batch.next_obs))
                # Real row layout; no invented training loss.
                rows=torch.zeros(2,u.replay.width)
                for prefix,obs in (('',batch.obs),('next_',batch.next_obs)):
                    for suffix,tensor in (('ego_features',obs.ego_features),('goal_features',obs.goal_features)):
                        rows[:,u.replay.layout[prefix+suffix][0]]=tensor
                    zone=torch.zeros(2,b.replay.zone_storage_capacity,19)
                    zone[:,:obs.zone_features.shape[1]]=obs.zone_features
                    rows[:,u.replay.layout[prefix+'zone_features'][0]]=zone.flatten(1)
                    rows[:,u.replay.layout[prefix+'zone_count'][0]]=obs.presence_mask.sum(1,keepdim=True)
                for name in ('action','reward','done','success','near_goal','line_to_goal_safe'):
                    rows[:,u.replay.layout[name][0]]=getattr(batch,name)
                u.fixed_rows=rows; u.noise=torch.full_like(batch.action,.2)
                u.coefficients[1]=b.spike_regularization.beta(step)
                actor=step%b.policy_delay==0 and step>b.actor_freeze_steps
                mode='actor' if actor else 'frozen_targets' if step%b.policy_delay==0 else 'critic_only'
                tensors={}
                with mock.patch.object(a.replay,'sample',return_value=batch), mock.patch('torch.randn_like',new=lambda t: torch.full_like(t,.2)):
                    expected=a.update_once(total_steps=step,diagnostic_tensor_recorder=lambda n,t:tensors.update({n:t.detach().clone()}))
                with mock.patch.object(torch.Tensor, 'new_tensor', side_effect=AssertionError('CPU scalar inside compute')):
                    actual=u.operation(mode)
                torch.testing.assert_close(u.tensor_evidence[mode]['replay.rows'],rows,rtol=0,atol=0)
                if actor and model=='snn':
                    from brain_uav.scripts.v2_gpu_td3_diagnostic import complete_actor_evidence
                    self.assertNotIn('actor.weighted_spike',tensors)
                    complete_actor_evidence(tensors,model)
                    left={n:t for n,t in tensors.items() if n.startswith('actor.')}
                    right={n:t for n,t in u.tensor_evidence[mode].items() if n.startswith('actor.')}
                    self.assertEqual(left.keys(),right.keys())
                    for name in left: torch.testing.assert_close(left[name],right[name],rtol=1e-4,atol=1e-5)
                    self.assertIn('replay.rows',u.tensor_evidence[mode])
                self.assertEqual(int(actual[0]),0)
                torch.testing.assert_close(actual[1],torch.tensor(expected.critic_loss),rtol=1e-4,atol=1e-5)
                for ma,mb in zip((a.actor,a.critic1,a.critic2,a.actor_target,a.critic1_target,a.critic2_target),
                                (b.actor,b.critic1,b.critic2,b.actor_target,b.critic1_target,b.critic2_target)):
                    for (name,pa),pb in zip(ma.named_parameters(),mb.parameters()):
                        with self.subTest(model=model,step=step,module=type(ma).__name__,parameter=name):
                            torch.testing.assert_close(pa,pb,rtol=1e-4,atol=1e-5)
                if actor:
                    torch.testing.assert_close(actual[2],torch.tensor(expected.actor_loss),rtol=1e-4,atol=1e-5)
                for left,right in zip((a.actor,a.critic1,a.critic2),(b.actor,b.critic1,b.critic2)):
                    for pa,pb in zip(left.parameters(),right.parameters()):
                        if pa.grad is None: self.assertIsNone(pb.grad)
                        else: torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-4,atol=1e-5)
                for oa,ob in ((a.actor_optimizer,b.actor_optimizer),(a.critic_optimizer,b.critic_optimizer)):
                    for sa,sb in zip(oa.state.values(),ob.state.values()):
                        self.assertEqual(sa.keys(),sb.keys())
                        for key in sa:
                            torch.testing.assert_close(sa[key],sb[key],rtol=0 if key=='step' else 1e-4,atol=0 if key=='step' else 1e-5)

    def test_device_adam_does_not_publish_missing_states_before_commit(self):
        p=torch.nn.Parameter(torch.tensor([1.]))
        opt=torch.optim.Adam([p],fused=True)
        a=DeviceAdam(opt)
        self.assertFalse(opt.state)
        with mock.patch('torch._fused_adam_') as kernel:
            a.step(torch.tensor(0.))
            kernel.assert_not_called()
        self.assertEqual(a.steps[0].item(),0)
        a.publish(); self.assertIs(opt.state[p]['exp_avg'],a.m[0])

    def test_loss_error_priority_and_clip(self):
        p=torch.nn.Parameter(torch.ones(2)); p.grad=torch.tensor([3.,4.])
        error,norm=checked_clip([p],torch.tensor(1.),torch.tensor(0),1,2,1.)
        self.assertEqual(error.item(),0); self.assertEqual(norm.item(),5.)
        torch.testing.assert_close(p.grad,torch.tensor([3.,4.])/(5.+1e-6))
        p.grad.fill_(float('inf'))
        error,_=checked_clip([p],torch.tensor(float('nan')),torch.tensor(0),1,2,1.)
        self.assertEqual(error.item(),1)

    def test_padding_forward_and_gradient_presence_empty_and_mixed(self):
        f=TestV2TD3(); f.setUp(); e=f.make_engine()
        for n in (0,1,2):
            batch=collate_v2_observations([_observation(n,scales=e.actor.scales),_observation(0,scales=e.actor.scales)])
            z=torch.zeros(2,10,19); mask=torch.zeros(2,10,dtype=torch.bool)
            z[:,:n]=batch.zone_features; mask[:,:n]=batch.presence_mask
            padded=V2ObservationBatch(batch.ego_features,batch.goal_features,z,mask)
            for m in (e.actor,e.critic1):
                outputs=[]; presence=[]
                for b in (batch,padded):
                    m.zero_grad(set_to_none=True)
                    out=m(b) if m is e.actor else m(b,torch.zeros(2,2))
                    out.sum().backward(); outputs.append(out.detach().clone())
                    presence.append([p.grad is not None for p in m.parameters()])
                torch.testing.assert_close(*outputs,rtol=1e-4,atol=1e-5)
                self.assertEqual(*presence)
                self.assertTrue(all(presence[0]))
        snn=TestSpikeRegularization().engine().actor
        for n in (0,1,2):
            b=collate_v2_observations([_observation(n,scales=snn.scales),_observation(0,scales=snn.scales)])
            snn.zero_grad(set_to_none=True)
            action,rate=snn.forward_with_activity(b)
            (action.sum()+torch.relu(rate-.17).square()).backward()
            self.assertTrue(all(p.grad is not None for p in snn.parameters()))
            self.assertEqual(snn.snn_head.lif1.v,0.)
            self.assertEqual(snn.snn_head.lif2.v,0.)
