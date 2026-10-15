"""CPU wiring/mechanism checks only, not manual CUDA capture evidence."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock
import inspect
import json
import torch

from brain_uav.trainers.v2_gpu_compute import GPUCompute, clone_batch, preserving_clip, tensors
from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
from test_v2_td3 import TestV2TD3 as Fixture, _observation


class TestGPUCompute(TestCase):
    def engine(self):
        f = Fixture(); f.setUp()
        e = f.make_engine()
        for n in (0,2):
            observation = _observation(n,scales=e.actor.scales)
            e.replay.add(observation,[0.,0.],.1,observation,False)
        return e

    def test_default_conflicts_cpu_and_checkpoint_schema(self):
        e = self.engine()
        self.assertFalse(e.gpu_td3_compute)
        self.assertNotIn('gpu_td3_compute',e.checkpoint_state_dict()['algorithm_config'])
        kw = dict(actor=e.actor,critic1=e.critic1,critic2=e.critic2,replay=e.replay,
            actor_lr=.001,critic_lr=.001,gamma=.99,tau=.1,policy_noise=.01,noise_clip=.02,
            policy_delay=2,batch_size=2,action_low=e.action_low,action_high=e.action_high)
        for extra, message in ((dict(gpu_td3_compute=1),'must be a bool'),
                (dict(gpu_td3_compute=True),'requires CUDA'),
                (dict(gpu_td3_compute=True,gpu_td3_update=True),'conflicts'),
                (dict(gpu_td3_compute=True,non_blocking_batch_transfer=True),'conflicts'),
                (dict(gpu_td3_compute=True,compile_actor_objective=True),'conflicts')):
            with self.subTest(extra=extra), self.assertRaisesRegex((TypeError,ValueError),message):
                V2TD3UpdateEngine(**kw,**extra)

    def test_cli_and_api_defaults_explicit(self):
        from brain_uav.scripts import train_v2_td3 as single, run_v2_td3_curriculum as curriculum, profile_v2_td3 as profile
        from brain_uav.trainers.v2_formal_training import build_v2_stage_engine
        for function in (single.run_v2_td3_stage,curriculum.run_v2_curriculum,build_v2_stage_engine):
            self.assertIs(inspect.signature(function).parameters['gpu_td3_compute'].default,False)
        args = profile.build_parser().parse_args(['--model','snn','--output-dir','unused','--gpu-td3-compute'])
        self.assertTrue(args.gpu_td3_compute)
        self.assertFalse(profile.build_parser().parse_args(['--model','ann','--output-dir','unused']).gpu_td3_compute)

    def test_clone_independent_dynamic_shapes_and_masks(self):
        e = self.engine()
        batch = e.replay.sample(2)
        cloned = clone_batch(batch)
        self.assertEqual(batch.sample_failure_fraction,cloned.sample_failure_fraction)
        for a,b in zip(tensors(batch),tensors(cloned)):
            self.assertEqual(a.shape,b.shape); self.assertEqual(a.dtype,b.dtype)
            self.assertFalse(b.requires_grad)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            if a.numel(): self.assertNotEqual(a.data_ptr(),b.data_ptr())

    def test_nonfinite_gradient_unchanged_and_priority(self):
        p = torch.nn.Parameter(torch.ones(2)); p.grad = torch.tensor([float('inf'),3.])
        for loss,code in ((1.,2),(float('nan'),1)):
            before = p.grad.clone()
            error,_ = preserving_clip([p],torch.tensor(loss),torch.tensor(0),1,2,1.)
            self.assertEqual(int(error),code); self.assertTrue(torch.equal(p.grad,before))

    def test_real_common_compute_cpu_same_shapes_loss_gradient_and_update(self):
        # Exactly the same RAW arithmetic boundary, no replay adapter at all.
        from brain_uav.trainers.v2_gpu_update import DeviceAdam
        e = self.engine(); other = self.engine()
        other.load_checkpoint_state_dict(deepcopy(e.checkpoint_state_dict()))
        batch = e.replay.sample(2)
        e.replay.sample = lambda size: batch
        u = GPUCompute.__new__(GPUCompute)
        u.e = other; u.diagnostic = False; u.compute_diagnostic = False
        u.actor_params = list(other.actor.parameters())
        u.critic_params = list(other.critic1.parameters())+list(other.critic2.parameters())
        class CPUAdam:
            def __init__(self,opt): self.opt=opt
            def step(self,error):
                if not bool(error): self.opt.step()
        u.adams = (CPUAdam(other.actor_optimizer),CPUAdam(other.critic_optimizer))
        u.injections = torch.zeros(4)
        u.coefficients = torch.tensor([0.,0.,other.policy_noise,other.noise_clip])
        u.noise = torch.zeros_like(batch.action)
        u.failure_fraction = torch.tensor(batch.sample_failure_fraction)
        def noise(t,**kw): return torch.zeros_like(t)
        with mock.patch('torch.randn_like',new=noise):
            expected = e.update_once(total_steps=2)
        values = u.compute(batch,'actor')
        self.assertEqual(int(values[0]),0)
        self.assertAlmostEqual(expected.critic_loss,float(values[1]),places=5)
        self.assertAlmostEqual(expected.actor_loss,float(values[2]),places=5)
        for name in ('actor','critic1','critic2','actor_target','critic1_target','critic2_target'):
            for a,b in zip(getattr(e,name).parameters(),getattr(other,name).parameters()):
                torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
                self.assertEqual(a.grad is None,b.grad is None)
                if a.grad is not None: torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-5)
        self.assertFalse(hasattr(u,'replay'))

    def test_snn_regularization_uses_real_same_forward_and_resets(self):
        from test_v2_spike_regularization import TestSpikeRegularization
        e = TestSpikeRegularization().engine(delay=2)
        other = TestSpikeRegularization().engine(delay=2)
        options = GPUCompute.static_options(other)
        other.actor.snn_head.time_window += 1
        self.assertNotEqual(GPUCompute.static_options(other),options)
        other.actor.snn_head.time_window -= 1
        other.load_checkpoint_state_dict(deepcopy(e.checkpoint_state_dict()))
        batch = e.replay.sample(2); e.replay.sample = lambda size: batch
        u = GPUCompute.__new__(GPUCompute)
        u.e=other; u.diagnostic=False; u.compute_diagnostic=False
        u.actor_params=list(other.actor.parameters()); u.critic_params=list(other.critic1.parameters())+list(other.critic2.parameters())
        class CPUAdam:
            def __init__(self,opt): self.opt=opt
            def step(self,error):
                if not bool(error): self.opt.step()
        u.adams=(CPUAdam(other.actor_optimizer),CPUAdam(other.critic_optimizer))
        u.injections=torch.zeros(4); u.noise=torch.zeros_like(batch.action)
        u.coefficients=torch.tensor([0.,1.,other.policy_noise,other.noise_clip]); u.failure_fraction=torch.tensor(batch.sample_failure_fraction)
        def noise(t,**kw): return torch.zeros_like(t)
        with mock.patch('torch.randn_like',new=noise): expected=e.update_once(total_steps=75002)
        values=u.compute(batch,'actor')
        self.assertAlmostEqual(expected.actor_batch_spike_rate_l1,float(values[10]),places=6)
        self.assertAlmostEqual(expected.spike_regularization_loss,float(values[8]),places=6)
        for a,b in zip(e.actor.parameters(),other.actor.parameters()):
            torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-5)
        self.assertEqual(other.actor.snn_head.lif1.v,0.)
        self.assertEqual(other.actor.snn_head.lif2.v,0.)

    def test_load_invalidates_without_serializing_executor(self):
        e=self.engine(); state=deepcopy(e.checkpoint_state_dict())
        execution=mock.Mock(); e._gpu_update_executor=execution
        generation=e._gpu_update_generation
        e.load_checkpoint_state_dict(state)
        execution.invalidate.assert_called_once()
        self.assertIsNone(e._gpu_update_executor)
        self.assertGreater(e._gpu_update_generation,generation)

    def test_stage_dispatch_samples_once_then_uploads_and_generates_noise(self):
        e=self.engine(); batch=e.replay.sample(2)
        e.gpu_td3_compute=True; e._gpu_update_formal_configured=True
        calls=[]
        def sample(size): calls.append('sample'); return batch
        e.replay.sample=sample
        def noise(t,**kw): calls.append('noise'); return torch.zeros_like(t)
        class Executor:
            diagnostic=False
            last_sections={'gpu_compute_input_copy':0.,'gpu_compute_to_result':0.}
            def step_batch(self,actual,normal,total,bc,**kw):
                calls.append('compute'); self.actual=actual
                return SimpleNamespace(actor_updated=False)
        executor=Executor(); e._gpu_update_executor=executor
        with mock.patch('torch.randn_like',new=noise): e.update_once(total_steps=1)
        self.assertEqual(calls,['sample','noise','compute'])
        torch.testing.assert_close(executor.actual.reward,batch.reward,rtol=0,atol=0)

    def test_actual_capture_restore_cpu_mock_preserves_values_none_and_masks(self):
        from contextlib import ExitStack, nullcontext
        e=self.engine(); batch=e.replay.sample(2)
        u=GPUCompute.__new__(GPUCompute)
        u.e=e; u.diagnostic=False; u.compute_diagnostic=False
        u.models=(e.actor,e.critic1,e.critic2,e.actor_target,e.critic1_target,e.critic2_target,e.bc_reference_actor)
        u.actor_params=list(e.actor.parameters()); u.critic_params=list(e.critic1.parameters())+list(e.critic2.parameters())
        u.params=(u.actor_params,u.critic_params)
        u.banks=[[[torch.ones_like(p)*3 for p in params] for _ in range(2)] for params in u.params]
        class Adam:
            def __init__(self,params):
                self.params=params
                self.values=[torch.zeros_like(p) for p in params]
            def state_tensors(self): return self.values
            def step(self,error):
                with torch.no_grad():
                    for p,v in zip(self.params,self.values):
                        if p.grad is not None: p.add_(.001); v.add_(1.)
        u.adams=(Adam(u.actor_params),Adam(u.critic_params))
        u.graphs={}; u.tensor_evidence={}; u.graph_limit=8*(e.replay.zone_storage_capacity+1)**2
        u.zone_limit=e.replay.zone_storage_capacity; u.bound_device=torch.device('cpu')
        u.published=[0,0]; u.injections=torch.zeros(4)
        u.noise=torch.zeros_like(batch.action); u.coefficients=torch.tensor([0.,0.,.01,.02])
        u.capture_seconds=0.; u.captures=0
        u.stream=SimpleNamespace(wait_stream=lambda other: None)
        u.actor_params[0].grad=u.banks[0][0][0]
        original_ref=u.actor_params[0].grad
        before=[p.detach().clone() for p in u.persistent()]
        original_mask=batch.obs.presence_mask.clone()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(torch.cuda,'mem_get_info',return_value=(1024**3,1024**3)))
            stack.enter_context(mock.patch.object(torch.cuda,'current_stream',return_value=u.stream))
            stack.enter_context(mock.patch.object(torch.cuda,'stream',return_value=nullcontext()))
            stack.enter_context(mock.patch.object(torch.cuda,'graph',return_value=nullcontext()))
            stack.enter_context(mock.patch.object(torch.cuda,'CUDAGraph',return_value=object()))
            stack.enter_context(mock.patch.object(torch.cuda,'synchronize'))
            stack.enter_context(mock.patch.object(torch.cuda,'get_rng_state',return_value=torch.tensor([1],dtype=torch.uint8)))
            stack.enter_context(mock.patch.object(torch.cuda,'set_rng_state'))
            u.capture_shape(u.key(batch,'actor'),batch)
        for p,value in zip(u.persistent(),before): torch.testing.assert_close(p,value,rtol=0,atol=0)
        self.assertIs(u.actor_params[0].grad,original_ref)
        torch.testing.assert_close(original_ref,torch.full_like(original_ref,3.),rtol=0,atol=0)
        self.assertIsNone(u.actor_params[1].grad)
        torch.testing.assert_close(batch.obs.presence_mask,original_mask,rtol=0,atol=0)
        self.assertEqual(u.captures,1)

    def test_curriculum_passes_compute_all_three_stages(self):
        from brain_uav.config import ScenarioConfig
        from brain_uav.scripts.run_v2_td3_curriculum import run_v2_curriculum
        from brain_uav.scripts.train_v2_td3 import default_v2_cuda_graph_compilation
        from test_run_v2_td3_curriculum import _prepared_initialization, derive_validation_stage_seed
        calls=[]
        def stage(**kw):
            calls.append(kw)
            return dict(stage=kw['stage'],passed=True,checkpoint=str(kw['output']),metrics=str(kw['metrics_out']),
                steps=1,outcome_counts={'goal':1},validation_result={'passed':True},stop_reason='fixed_validation_passed')
        def pool(path,**kw):
            level=Path(path).stem
            return SimpleNamespace(curriculum_level=level,master_seed=20260904,
                stage_seed=derive_validation_stage_seed(20260904,level),scenario_count=100,content_digest=level)
        with TemporaryDirectory() as directory:
            root=Path(directory); bc=root/'bc.pt'; bc.write_bytes(b'never loaded; mock only')
            prefix='brain_uav.scripts.run_v2_td3_curriculum.'
            with mock.patch(prefix+'prepare_v2_stage_initialization',return_value=_prepared_initialization(ScenarioConfig(),source=bc)), \
                 mock.patch(prefix+'prepare_v2_validation_pools',return_value={s:root/(s+'.json') for s in ('easy','medium','hard')}), \
                 mock.patch(prefix+'load_v2_validation_pool',side_effect=pool), \
                 mock.patch(prefix+'resolve_training_device',return_value='cuda'):
                run_v2_curriculum(bc_checkpoint=bc,output_root=root/'run',validation_pool_dir=root/'pools',device='cuda',
                    seed=7,max_stage='hard',gpu_td3_compute=True,stage_runner=stage,
                    **default_v2_cuda_graph_compilation(model='ann',resolved_device='cuda'))
        self.assertEqual([c['stage'] for c in calls],['easy','medium','hard'])
        self.assertTrue(all(c['gpu_td3_compute'] and not c['gpu_td3_update'] for c in calls))
        self.assertEqual(calls[1]['init_checkpoint'],calls[0]['output'])
        self.assertEqual(calls[2]['init_checkpoint'],calls[1]['output'])

    def test_cached_binding_checks_reject_parameter_and_optimizer_replacement(self):
        e=self.engine(); u=GPUCompute.__new__(GPUCompute)
        u.e=e; u.engine_device=e.device; u.bound_device=e.device; u.generation=e._gpu_update_generation
        u.models=(e.actor,e.critic1,e.critic2,e.actor_target,e.critic1_target,e.critic2_target,e.bc_reference_actor)
        u.optimizers=(e.actor_optimizer,e.critic_optimizer)
        u.params=(list(e.actor.parameters()),list(e.critic1.parameters())+list(e.critic2.parameters()))
        u.module_bindings=[]; u.buffer_bindings=[]; u.encoder_bindings=[]; u.action_bindings=[]; u.registrations=[]
        for model in u.models:
            if model is not None:
                for parent in model.modules():
                    for name,p in parent._parameters.items():
                        if p is not None: u.registrations.append((parent,name,p,p.data_ptr(),p.shape,p.stride(),p.requires_grad))
        u.optimizer_groups=tuple(o.param_groups[0] for o in u.optimizers)
        u.optimizer_states=tuple(o.state for o in u.optimizers)
        u.optimizer_options=tuple(u.options(o) for o in u.optimizers)
        u.static_configuration=u.static_options(e)
        u.native_states=[[o.state.get(p) for p in params] for o,params in zip(u.optimizers,u.params)]
        u.state_handles=[[None]*len(params) for params in u.params]
        u.validate_binding()
        parent,name,p,*_=u.registrations[0]
        parent._parameters[name]=torch.nn.Parameter(p.detach().clone())
        with self.assertRaisesRegex(RuntimeError,'parameter/address'): u.validate_binding()
        parent._parameters[name]=p
        e.actor_optimizer.state[p]={'step':torch.tensor(0.)}
        with self.assertRaisesRegex(RuntimeError,'Adam state replaced'): u.validate_binding()

    def test_shape_keys_use_bank_addresses_not_mask_values(self):
        e = self.engine(); batch = e.replay.sample(2)
        u = GPUCompute.__new__(GPUCompute)
        u.e=e; u.bound_device=torch.device('cpu'); u.zone_limit=e.replay.zone_storage_capacity
        u.published=[0,1]
        key=u.key(batch,'actor')
        changed=replace(batch,obs=replace(batch.obs,presence_mask=~batch.obs.presence_mask))
        self.assertEqual(u.key(changed,'actor'),key)
        u.published=[1,0]
        self.assertNotEqual(u.key(batch,'actor'),key)
        self.assertNotEqual(u.key(batch,'critic_only'),key)

    def test_serial_correctness_failure_skips_timing(self):
        from brain_uav.scripts.profile_v2_td3 import build_parser
        from brain_uav.scripts.v2_gpu_compute_diagnostic import run
        with TemporaryDirectory() as directory:
            args=build_parser().parse_args(['--model','snn','--device','cuda','--gpu-compute-comparison',
                '--bc-checkpoint','bc.pt','--validation-pool','pool.json','--output-dir',directory])
            calls=[]
            def subprocess(command,**kw):
                calls.append(command)
                return SimpleNamespace(returncode=0 if len(calls)==1 else 1)
            with mock.patch('brain_uav.scripts.v2_gpu_compute_diagnostic.subprocess.run',side_effect=subprocess):
                with self.assertRaisesRegex(RuntimeError,'correctness failed'): run(args)
            self.assertEqual(len(calls),2)
            self.assertTrue(all('correctness' in call for call in calls))
            report=json.loads((Path(directory)/'summary.json').read_text())
            self.assertFalse(report['passed'])
