"""Local semantic/dispatch checks only; no CUDA capture or performance evidence."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import inspect
import json
import tempfile
import unittest
from unittest import mock
import torch
from brain_uav.observations import collate_v2_observations
from brain_uav.trainers.v2_td3 import (
    V2TD3UpdateEngine, _actor_loss_tensor_block, _actor_objective_tensor_block,
)
from test_v2_td3 import TestV2TD3, _observation
from test_v2_spike_regularization import TestSpikeRegularization
from brain_uav.scripts import profile_v2_td3 as profile


class TestMergedUpdateBlocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def engine(self, model='ann'):
        if model == 'snn':
            return TestSpikeRegularization().engine(freeze=25000, delay=2)
        fixture = TestV2TD3(); fixture.setUp()
        engine = fixture.make_engine(actor_freeze_steps=25000, policy_delay=2,
                                     terminal_enabled=True)
        fixture.fill_replay(engine)
        return engine

    def compare(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a: self.compare(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b): self.compare(x, y)
        elif isinstance(a, float):
            self.assertAlmostEqual(a, b, delta=1e-5 + abs(a)*1e-4)
        else:
            self.assertEqual(a, b)

    def test_semantic_updates_ann_snn_all_blocks_and_history(self):
        # Force only the new tensor functions through Dynamo/eager on CPU.
        # Production flags still refuse CPU; this does NOT test CUDA Graphs.
        for model in ('ann', 'snn'):
            torch.manual_seed(17)
            engines = [self.engine(model) for _ in range(4)]
            initial = deepcopy(engines[0].checkpoint_state_dict())
            for engine, (c, a) in zip(engines, ((False,False),(True,False),(False,True),(True,True))):
                engine.load_checkpoint_state_dict(initial)
                engine.reduce_update_stat_syncs = True
                engine.frozen_critic_strategy = 'compiled_no_grad_context'
                engine.cache_actor_loss_coefficients = True
                engine._compiled_actor_loss = torch.compile(_actor_loss_tensor_block, backend='eager', fullgraph=True)
                if c:
                    engine._compiled_critic_front = torch.compile(engine._compute_critic_front_tensors, backend='eager', fullgraph=True)
                if a:
                    engine._compiled_actor_objective = torch.compile(_actor_objective_tensor_block, backend='eager', fullgraph=True)
            history = []
            for step, zones in zip((1,2,25000,25002,50000,75000), (0,2,1,0,3,1)):
                sampled = engines[0].replay.sample(2)
                obs = collate_v2_observations([_observation(n, scales=engines[0].actor.scales) for n in (zones, 0)])
                batch = replace(sampled, obs=obs, next_obs=obs)
                metrics=[]
                for engine in engines:
                    engine.replay.sample = lambda size,b=batch:b
                    torch.manual_seed(step)
                    metrics.append(asdict(engine.update_once(total_steps=step, bc_lambda=0.)))
                for other, metric in zip(engines[1:], metrics[1:]):
                    self.compare(metrics[0], metric)
                    self.compare(engines[0].checkpoint_state_dict(), other.checkpoint_state_dict())
                    for name in ('actor', 'critic1', 'critic2'):
                        self.compare([p.grad for p in getattr(engines[0], name).parameters()], [p.grad for p in getattr(other,name).parameters()])
                    self.assertTrue(all(p.grad is None for m in (other.actor_target,other.critic1_target,other.critic2_target) for p in m.parameters()))
                history.append(deepcopy(metrics))
            self.assertEqual(history[0][0]['actor_updated'],False)
            if model=='snn':
                self.assertEqual([history[i][3]['spike_beta_eff'] for i in (3,4,5)], [0.00004,.5,1.])

    def test_actor_objective_gradient_and_dynamic_coefficients(self):
        engine=self.engine()
        critic=engine.critic1
        for p in critic.parameters(): p.requires_grad_(False)
        compiled=torch.compile(_actor_objective_tensor_block,backend='eager',fullgraph=True)
        context=torch.randn(2,critic.zone_set_encoder.output_dim)
        batch=engine.replay.sample(2)
        observed=[]
        for beta,bc in ((0.,0.),(.5,1.5),(1.,.2)):
            actions=torch.zeros(2,2,requires_grad=True)
            rate=torch.tensor(.3,requires_grad=True)
            terms=compiled(critic,context,actions,torch.ones_like(actions),batch.obs.ego_features,
                batch.obs.goal_features,batch.line_to_goal_safe,engine.action_low,engine.action_high,
                torch.tensor(2.5),torch.tensor(bc),torch.tensor(0.),1.,1.,.6,250.,False,
                rate,torch.tensor(beta),.17)
            terms[0].backward()
            self.assertIsNotNone(actions.grad)
            self.assertAlmostEqual(rate.grad.item(),beta*2*(.3-.17),places=6)
            self.assertTrue(all(p.grad is None for p in critic.parameters()))
            observed.append(terms[0].item())
        self.assertNotEqual(observed[0],observed[1])

    def test_default_flags_reject_cpu_and_conflicts_before_side_effects(self):
        engine=self.engine()
        self.assertFalse(engine.compile_critic_update_front)
        self.assertFalse(engine.compile_actor_objective)
        with mock.patch.object(engine, '_validate_merged_bindings', side_effect=AssertionError('disabled guard')):
            engine.update_once(total_steps=1)
        for name in ('compile_critic_update_front','compile_actor_objective'):
            kw=dict(actor=engine.actor,critic1=engine.critic1,critic2=engine.critic2,replay=engine.replay,
                actor_lr=.001,critic_lr=.001,gamma=.99,tau=.005,policy_noise=.1,noise_clip=.05,
                policy_delay=2,batch_size=2,action_low=engine.action_low,action_high=engine.action_high)
            with self.assertRaises(TypeError): V2TD3UpdateEngine(**kw,**{name:1})
            with self.assertRaises(ValueError): V2TD3UpdateEngine(**kw,**{name:True})
        from brain_uav.scripts import train_v2_td3 as stage, run_v2_td3_curriculum as course
        from brain_uav.trainers.v2_formal_training import build_v2_stage_engine
        for fn in (stage.run_v2_td3_stage,course.run_v2_curriculum,build_v2_stage_engine,profile._run_diagnostic_level):
            for name in ('compile_critic_update_front','compile_actor_objective'):
                self.assertIs(inspect.signature(fn).parameters[name].default,False)
        for parser in (stage.build_parser(),course.build_parser(),profile.build_parser()):
            opts={a.dest:a.default for a in parser._actions}
            self.assertIs(opts['compile_critic_update_front'],False)
            self.assertIs(opts['compile_actor_objective'],False)
        # Require formal compilation without touching checkpoint/files.
        with mock.patch.object(stage,'prepare_v2_stage_initialization') as prepare:
            with self.assertRaises(ValueError):
                stage.run_v2_td3_stage(stage='easy',init_checkpoint='absent.pt',validation_pool='absent.json',
                    output='absent.pt',metrics_out='absent.json',device='cpu',compile_critic_update_front=True)
            prepare.assert_not_called()

    def test_checkpoint_clear_and_binding_replacements(self):
        engine=self.engine()
        engine._compiled_critic_front=object();engine._compiled_actor_objective=object()
        payload=deepcopy(engine.checkpoint_state_dict())
        engine.load_checkpoint_state_dict(payload)
        self.assertIsNone(engine._compiled_critic_front)
        self.assertIsNone(engine._compiled_actor_objective)
        engine._merged_compile_options={}
        engine._rebuild_merged_blocks()
        engine._merged_device=engine.device
        original=engine.critic1
        engine.critic1=deepcopy(original)
        with self.assertRaisesRegex(RuntimeError,'model replacement'): engine._validate_merged_bindings()
        engine.critic1=original
        original.head[0].weight=torch.nn.Parameter(original.head[0].weight.clone())
        with self.assertRaisesRegex(RuntimeError,'parameter replacement'): engine._validate_merged_bindings()

    def test_cached_registration_guards_without_parameter_walks(self):
        from contextlib import ExitStack
        engine = self.engine()
        engine._merged_compile_options = {}
        engine._merged_device = engine.device
        engine._rebuild_merged_blocks()
        models = (engine.actor, engine.critic1, engine.critic2, engine.actor_target,
                  engine.critic1_target, engine.critic2_target)
        with ExitStack() as stack:
            for model in models:
                stack.enter_context(mock.patch.object(model, 'parameters', side_effect=AssertionError('recursive parameter walk')))
                stack.enter_context(mock.patch.object(model, 'modules', side_effect=AssertionError('recursive module walk')))
            engine._validate_merged_bindings()
        original_head = engine.critic1.head
        engine.critic1.head = deepcopy(original_head)
        with self.assertRaisesRegex(RuntimeError, 'module replacement'): engine._validate_merged_bindings()
        engine.critic1.head = original_head
        parameters = original_head[0]._parameters
        original_head[0]._parameters = dict(parameters)
        with self.assertRaisesRegex(RuntimeError, 'parameter replacement'): engine._validate_merged_bindings()
        original_head[0]._parameters = parameters
        items = tuple(parameters.items())
        value = parameters.pop('weight'); parameters['weight'] = value
        with self.assertRaisesRegex(RuntimeError, 'parameter replacement'): engine._validate_merged_bindings()
        parameters.clear(); parameters.update(items)
        original_data = original_head[0].weight.data
        original_head[0].weight.data = original_data.to(torch.float64)
        with self.assertRaisesRegex(RuntimeError, 'float32'): engine._validate_merged_bindings()
        original_head[0].weight.data = original_data
        engine._merged_device = torch.device('meta')
        with self.assertRaisesRegex(RuntimeError, 'configured CUDA'): engine._validate_merged_bindings()
        engine._merged_device = engine.device
        original_head[0].register_parameter('unexpected', None)
        with self.assertRaisesRegex(RuntimeError, 'parameter replacement'): engine._validate_merged_bindings()
        del original_head[0]._parameters['unexpected']
        engine._validate_merged_bindings()
        # Loading rebinds the registration cache without changing the checkpoint schema.
        previous = engine._merged_bindings
        engine.load_checkpoint_state_dict(deepcopy(engine.checkpoint_state_dict()))
        self.assertIsNot(engine._merged_bindings, previous)
        engine._validate_merged_bindings()

    def test_serial_suite_order_failure_isolation_and_baseline_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=profile.build_parser().parse_args(['--model','ann','--bc-checkpoint','ann.pt',
                '--snn-bc-checkpoint','snn.pt','--output-dir',tmp,'--validation-pool','pool.json',
                '--device','cuda','--td3-block-suite'])
            calls=[]
            def run(command,**kwargs):
                model=command[command.index('--model')+1];choice=command[command.index('--td3-block-choice')+1]
                correct='--td3-block-correctness' in command
                calls.append((model,choice,correct))
                out=Path(command[command.index('--output-dir')+1]);out.mkdir(parents=True)
                if correct and model=='ann' and choice=='C': return SimpleNamespace(returncode=1)
                if correct and choice=='baseline': (out/'reference.pt').write_bytes(b'mock reference')
                if not correct:
                    (out/'local_optimization_summary.json').write_text(json.dumps({'valid_for_speed_comparison':True,
                        'comparison':{choice:{'seconds_per_step':.04,'stdev_wall_seconds':.001}}}))
                return SimpleNamespace(returncode=0)
            with mock.patch('subprocess.run',side_effect=run): result=profile.run_td3_block_suite(args)
            self.assertEqual(calls,[('ann','baseline',True),('ann','baseline',False),('ann','C',True),
                ('ann','A',True),('ann','A',False),('snn','baseline',True),('snn','baseline',False),
                ('snn','C',True),('snn','C',False),('snn','A',True),('snn','A',False)])
            self.assertTrue(result['models']['ann']['C']['timing_skipped'])
            self.assertFalse(result['combination_automatically_tested'])
            self.assertTrue(all(choice!='CA' for _,choice,_ in calls))

    def test_curriculum_passes_blocks_to_all_stages_and_summary(self):
        from brain_uav.scripts import run_v2_td3_curriculum as course
        from brain_uav.scripts.train_v2_td3 import default_v2_cuda_graph_compilation
        from brain_uav.config import ScenarioConfig
        from test_run_v2_td3_curriculum import _prepared_initialization
        from brain_uav.trainers.v2_validation import derive_validation_stage_seed
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/'bc.pt'; checkpoint.write_bytes(b'mocked BC')
            calls=[]
            def runner(**kwargs):
                calls.append(kwargs)
                return dict(stage=kwargs['stage'],passed=True,checkpoint=str(kwargs['output']),
                    steps=1,global_steps_end=kwargs['global_steps_start']+1)
            def load(path,**kwargs):
                stage=Path(path).stem
                return SimpleNamespace(master_seed=20260904,stage_seed=derive_validation_stage_seed(20260904,stage),
                    scenario_count=100,content_digest=stage)
            pools={stage:root/f'{stage}.json' for stage in ('easy','medium','hard')}
            with mock.patch.object(course,'resolve_training_device',return_value='cuda'), \
                 mock.patch.object(course,'prepare_v2_stage_initialization',return_value=_prepared_initialization(ScenarioConfig(),source=checkpoint)), \
                 mock.patch.object(course,'prepare_v2_validation_pools',return_value=pools), \
                 mock.patch.object(course,'load_v2_validation_pool',side_effect=load):
                result=course.run_v2_curriculum(bc_checkpoint=checkpoint,output_root=root/'out',
                    validation_pool_dir=root/'pools',device='cuda',stage_runner=runner,
                    compile_critic_update_front=True,compile_actor_objective=True,
                    **default_v2_cuda_graph_compilation(model='ann',resolved_device='cuda'))
            self.assertEqual(len(calls),3)
            self.assertTrue(all(c['compile_critic_update_front'] and c['compile_actor_objective'] for c in calls))
            self.assertTrue(result['compilation_request']['compile_critic_update_front'])
            self.assertTrue(result['compilation_request']['compile_actor_objective'])

    def test_failure_barriers_preserve_partial_update_semantics(self):
        for failure in ('critic_nan','actor_inf'):
            torch.manual_seed(9)
            original=self.engine()
            initial=deepcopy(original.checkpoint_state_dict())
            batch=original.replay.sample(2)
            states=[]
            for c,a in ((False,False),(True,False),(False,True)):
                engine=self.engine();engine.load_checkpoint_state_dict(initial)
                engine.frozen_critic_strategy='compiled_no_grad_context'
                engine.cache_actor_loss_coefficients=True
                engine._compiled_actor_loss=_actor_loss_tensor_block
                if c: engine._compiled_critic_front=engine._compute_critic_front_tensors
                if a: engine._compiled_actor_objective=_actor_objective_tensor_block
                chosen=replace(batch,reward=torch.full_like(batch.reward,float('nan'))) if failure=='critic_nan' else batch
                engine.replay.sample=lambda size,b=chosen:b
                before=deepcopy(engine.checkpoint_state_dict())
                hook=list(engine.actor.parameters())[-1].register_hook(lambda g:torch.full_like(g,float('inf'))) if failure=='actor_inf' else None
                try:
                    torch.manual_seed(51)
                    with self.assertRaises(FloatingPointError): engine.update_once(total_steps=75000)
                finally:
                    if hook is not None: hook.remove()
                after=deepcopy(engine.checkpoint_state_dict())
                if failure=='critic_nan': self.compare(before,after)
                else:
                    self.compare(before['actor_state_dict'],after['actor_state_dict'])
                    self.compare(before['actor_optimizer_state_dict'],after['actor_optimizer_state_dict'])
                    self.assertEqual(after['update_counts']['critic_target_update_count'],1)
                    self.assertEqual(after['update_counts']['actor_update_count'],0)
                states.append(after)
            self.compare(states[0],states[1]);self.compare(states[0],states[2])

    def test_suite_cpu_rejected(self):
        args=profile.build_parser().parse_args(['--model','ann','--bc-checkpoint','a.pt',
            '--snn-bc-checkpoint','s.pt','--output-dir','unused','--device','cpu','--td3-block-suite'])
        with mock.patch('subprocess.run') as run:
            with self.assertRaises(ValueError): profile.run_td3_block_suite(args)
            run.assert_not_called()


if __name__ == '__main__': unittest.main()
