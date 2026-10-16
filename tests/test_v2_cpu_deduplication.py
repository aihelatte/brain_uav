"""Exact CPU reuse/ownership tests; never CUDA or speed evidence."""
from copy import deepcopy
from unittest import TestCase, mock
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import random
import numpy as np
import torch

from brain_uav.config import RewardConfig
from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.geometry import Sphere, Ellipsoid, Box, TriangularPyramid, QuadrangularPyramid, NoFlyZone
from brain_uav.observations.v2_batch import collate_v2_observations
from test_v2_static_no_fly_env import _config, _scenario
import test_v2_td3 as td3_fixture
from test_v2_td3 import _observation


class TestCPUDeduplication(TestCase):
    def env(self, zones, enabled, *, gate=False, y=0.):
        return V2StaticNoFlyTrajectoryEnv(_config(), RewardConfig(
            breakthrough_clearance_gate_enabled=gate, descent_three_band_enabled=True,
            boundary_directional_warning_enabled=True), seed=7, uav_collision_radius=.25,
            fixed_scenarios=[_scenario([-5.,y,10.,0.,0.], [30.,y,10.], zones)],
            cpu_deduplication=enabled)

    def assert_observation(self, a, b):
        for name in ('ego_features','goal_features','zone_features','presence_mask'):
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))

    def test_five_shapes_empty_and_fixed_actions_are_exact(self):
        shapes=[None,Sphere([4.,0.,10.],1.),Ellipsoid([4.,0.,10.],2.,1.,1.),
            Box([4.,0.,10.],2.,2.,2.),TriangularPyramid([4.,0.,0.],8.,8.,20.),
            QuadrangularPyramid([4.,0.,0.],8.,8.,20.)]
        for shape in shapes:
            for y in (0.,10.):
                for gate in (False,True):
                    with self.subTest(shape=type(shape).__name__,y=y,gate=gate):
                        zones=[] if shape is None else [NoFlyZone('zone',shape,.2)]
                        a,b=[self.env(zones,on,gate=gate,y=y) for on in (False,True)]
                        oa,ia=a.reset(); ob,ib=b.reset()
                        self.assert_observation(oa,ob); self.assertEqual(ia,ib)
                        if shape is not None:
                            self.assertEqual(a.line_to_goal_is_safe(),y!=0.)
                            if y!=0.: self.assertFalse(a.line_to_goal_is_safe(clearance=40.))
                        old=deepcopy(ob)
                        for i in range(12):
                            # Preserve pre-action ordering and distinct safety radii.
                            for clearance in (40.,0.,.5):
                                self.assertEqual(a.line_to_goal_is_safe(clearance=clearance),
                                    b.line_to_goal_is_safe(clearance=clearance))
                            ra=a.step(np.array([0.,0.],dtype=np.float32))
                            rb=b.step(np.array([0.,0.],dtype=np.float32))
                            self.assert_observation(ra[0],rb[0])
                            self.assertEqual(ra[1:],rb[1:])
                            np.testing.assert_array_equal(a.state,b.state)
                            self.assertFalse(hasattr(b,'_active_zone_point_clearances'))
                            if ra[2] or ra[3]: break
                        self.assert_observation(old,ob)
                        pa=a.set_goal([40.,y,10.]); pb=b.set_goal([40.,y,10.])
                        self.assert_observation(pa[0],pb[0]); self.assertEqual(pa[1],pb[1])
                        oa,ia=a.reset(); ob,ib=b.reset()
                        self.assert_observation(oa,ob); self.assertEqual(ia,ib)
                        # No reward context survives external replacement.
                        a.zones=[]; b.zones=[]
                        a.uav_collision_radius=.75; b.uav_collision_radius=.75
                        ra=a.step(np.array([0.,0.],dtype=np.float32))
                        rb=b.step(np.array([0.,0.],dtype=np.float32))
                        self.assert_observation(ra[0],rb[0]); self.assertEqual(ra[1:],rb[1:])

    def test_reward_reuse_reduces_calls_and_has_exact_context_fallback(self):
        zones=[NoFlyZone('zone',Ellipsoid([10.,3.,10.],2.,1.,1.),.2)]
        calls=[]
        for enabled in (False,True):
            e=self.env(zones,enabled); e.reset()
            e.recent_progress=[1.]*e.rewards.progress_window_size
            with mock.patch.object(NoFlyZone,'point_clearance',autospec=True,
                    side_effect=NoFlyZone.point_clearance) as query:
                e.step(np.array([0.,0.],dtype=np.float32))
                calls.append(query.call_count)
        self.assertEqual(calls,[1,0])
        e=self.env(zones,True); e.reset()
        point=e.state[:3].copy()
        context=(point,e._zone_query_identity(),e.uav_collision_radius,(123.,))
        e._active_zone_point_clearances=context
        self.assertEqual(e._nearest_zone_surface_clearance(point),123.)
        # A float64 point that rounds to the same float32 is still a different query.
        shifted=point.astype(np.float64); shifted[0]+=1e-8
        with mock.patch.object(e,'min_zone_clearance',wraps=e.min_zone_clearance) as fallback:
            e._nearest_zone_surface_clearance(shifted)
            fallback.assert_called_once()
        e.uav_collision_radius+=.1
        self.assertEqual(e._nearest_zone_surface_clearance(point),e.min_zone_clearance(point))
        e.uav_collision_radius=context[2]
        e.zones[0]._safety_margin+=.1
        self.assertEqual(e._nearest_zone_surface_clearance(point),e.min_zone_clearance(point))
        e.zones[0]=NoFlyZone('replacement',Sphere([5.,2.,10.],1.),.3)
        self.assertEqual(e._nearest_zone_surface_clearance(point),e.min_zone_clearance(point))
        del e._active_zone_point_clearances
        self.assertEqual(e._nearest_zone_surface_clearance(point),e.min_zone_clearance(point))

    def test_single_batch_ownership_and_copy_reduction(self):
        f=td3_fixture.TestV2TD3(); f.setUp(); scales=f.make_engine().actor.scales
        for n in (0,1,5):
            observation=_observation(n,scales=scales)
            before=deepcopy(observation)
            a=collate_v2_observations([observation])
            b=collate_v2_observations([observation],deduplicate_copies=True)
            allocated=[]
            stack=np.stack
            def stacked(*a,**kw):
                value=stack(*a,**kw); allocated.append(value); return value
            with mock.patch('numpy.stack',side_effect=stacked):
                owned=collate_v2_observations([observation],deduplicate_copies=True)
            self.assertTrue(np.shares_memory(owned.ego_features.numpy(),allocated[0]))
            self.assertTrue(np.shares_memory(owned.goal_features.numpy(),allocated[1]))
            old=collate_v2_observations([observation],deduplicate_copies=True)
            for name in ('ego_features','goal_features','zone_features','presence_mask'):
                torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
                self.assertFalse(np.shares_memory(getattr(b,name).numpy(),getattr(observation,name)))
                getattr(b,name).zero_()
                torch.testing.assert_close(getattr(a,name),getattr(old,name),rtol=0,atol=0)
            self.assert_observation(observation,before)
            for name in ('ego_features','goal_features','zone_features','presence_mask'):
                source=getattr(observation,name); source.setflags(write=True); source[:]=0
                torch.testing.assert_close(getattr(a,name),getattr(old,name),rtol=0,atol=0)
        with self.assertRaises(ValueError):
            collate_v2_observations([_observation(1,scales=scales)]*2,deduplicate_copies=True)

    def test_actions_and_rng_are_exact_with_explicit_flag(self):
        import test_v2_spike_regularization as spike_fixture
        f=td3_fixture.TestV2TD3(); f.setUp(); e=f.make_engine()
        for e in (e,spike_fixture.TestSpikeRegularization().engine()):
            for n in (0,2):
                observation=_observation(n,scales=e.actor.scales)
                rngs=[np.random.default_rng(7),np.random.default_rng(7)]
                cpu_rng=torch.get_rng_state().clone()
                actions=[e.select_action(observation,exploration_noise=.1,exploration_rng=rng,
                    cpu_deduplication=enabled) for rng,enabled in zip(rngs,(False,True))]
                np.testing.assert_array_equal(*actions)
                self.assertEqual(rngs[0].bit_generator.state,rngs[1].bit_generator.state)
                self.assertTrue(torch.equal(cpu_rng,torch.get_rng_state()))

    def test_evidence_is_owned_exact_and_tracks_rng(self):
        from brain_uav.scripts.v2_cpu_dedup_diagnostic import step_evidence, compare_evidence
        f=td3_fixture.TestV2TD3(); f.setUp(); e=f.make_engine()
        obs=_observation(0,scales=e.actor.scales); rng=np.random.default_rng(7)
        state=np.zeros(5,dtype=np.float32); action=np.zeros(2,dtype=np.float32)
        def record(): return step_evidence(e,rng,1,True,obs,action,obs,1.,False,False,{'outcome':'running'},state,True)
        a=record(); b=record()
        self.assertTrue(compare_evidence([a],[b])['passed'])
        rng.random(); c=record()
        self.assertFalse(compare_evidence([a],[c])['passed'])
        self.assertEqual(compare_evidence([a],[c])['first_difference']['field'],'rng')
        action[0]=.1
        self.assertNotEqual(a['action'],record()['action'])
        self.assertEqual(a['action'],[0.,0.])

    def test_existing_loop_records_exact_rollout_without_gpu_instrumentation(self):
        from brain_uav.scripts import profile_v2_td3 as p
        from brain_uav.scripts.v2_cpu_dedup_diagnostic import compare_evidence
        from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig
        evidence=[]
        for enabled in (False,True):
            random.seed(7); np.random.seed(7); torch.manual_seed(7)
            f=td3_fixture.TestV2TD3(); f.setUp(); e=f.make_engine()
            e.replay.rng=np.random.default_rng(7)
            e.bc_reference_actor=e._frozen_target(e.actor,device=e.device)
            e.gpu_td3_compute=True
            e._gpu_update_executor=SimpleNamespace(begin_measurement=lambda:None,
                invalidate=lambda:None,metadata=lambda:{'requested':True})
            original=e.update_once
            def cpu_update(**kw):
                self.assertNotIn('diagnostic_timing_recorder',kw)
                e.gpu_td3_compute=False
                try: return original(**kw)
                finally: e.gpu_td3_compute=True
            e.update_once=cpu_update
            rows=[]
            pool=SimpleNamespace(stage_seed=7,scenarios=[dict(scenario_id='fixed',payload=_scenario(
                [-5.,0.,10.,0.,0.],[30.,0.,10.],[NoFlyZone('e',Ellipsoid([10.,3.,10.],2.,1.,1.))]))])
            with mock.patch.object(p,'build_v2_stage_engine',return_value=SimpleNamespace(
                    engine=e,exploration_rng=np.random.default_rng(7))), \
                    mock.patch('torch.profiler.profile',side_effect=AssertionError('Unexpected profiler')):
                result=p._run_diagnostic_level(level='easy',pool=pool,prepared=SimpleNamespace(
                    scenario_config=_config(),reward_config=RewardConfig(breakthrough_clearance_gate_enabled=False,
                        descent_three_band_enabled=True,boundary_directional_warning_enabled=True),uav_collision_radius=.25),
                    formal_config=V2FormalTrainingConfig(stage='easy',batch_size=e.batch_size,replay_capacity=64,actor_freeze_steps=0),
                    bc_checkpoint=Path('unused'),model='ann',snn_time_window=4,device=torch.device('cpu'),
                    warmup_steps=8,measured_steps=3,gpu_td3_compute=True,
                    cpu_deduplication=enabled,deduplication_evidence=rows)
            self.assertEqual(result['cpu_deduplication'],enabled)
            self.assertFalse(result['environment_geometry_diagnostic']['enabled'])
            self.assertEqual(sum(row['measured'] for row in rows),3)
            evidence.append(rows)
        comparison=compare_evidence(*evidence)
        self.assertTrue(comparison['passed'],comparison)

    def test_server_driver_gates_timing_and_retains_failure_report(self):
        from brain_uav.scripts import profile_v2_td3 as p, v2_cpu_dedup_diagnostic as d
        with TemporaryDirectory() as directory:
            root=Path(directory)/'run'
            args=p.build_parser().parse_args(['--model','snn','--device','cuda','--gpu-td3-compute',
                '--bc-checkpoint','bc.pt','--validation-pool','pool.json','--output-dir',str(root)])
            calls=[]
            def worker(command,**kw):
                calls.append(command)
                return SimpleNamespace(returncode=1)
            with mock.patch.object(d.subprocess,'run',side_effect=worker):
                with self.assertRaises(RuntimeError): d.run(args)
            report=json.loads((root/'review/summary.json').read_text(encoding='utf-8'))
            self.assertFalse(report['passed'])
            self.assertEqual(len(calls),1)
            self.assertIn('--cpu-deduplication-evidence',calls[0])
            self.assertNotIn('--cpu-deduplication',calls[0])
            self.assertTrue((root/'review/correctness_A.log').exists())

    def test_server_driver_exact_gate_and_serial_timing(self):
        from brain_uav.scripts import profile_v2_td3 as p, v2_cpu_dedup_diagnostic as d
        for mismatch in (True,False):
            with self.subTest(mismatch=mismatch), TemporaryDirectory() as directory:
                root=Path(directory)/'run'
                args=p.build_parser().parse_args(['--model','snn','--device','cuda','--gpu-td3-compute',
                    '--bc-checkpoint','bc.pt','--validation-pool','pool.json','--output-dir',str(root),
                    '--steps-per-level','3','--repeats','2','--gpu-compute-environment-diagnostic'])
                calls=[]
                def worker(command,**kw):
                    calls.append(command)
                    output=Path(command[command.index('--output-dir')+1]); (output/'other').mkdir(parents=True)
                    enabled='--cpu-deduplication' in command
                    evidence='--cpu-deduplication-evidence' in command
                    instrumented='--gpu-compute-environment-diagnostic' in command
                    data={k:'same' for k in ('checkpoint','checkpoint_sha256','pool_sha256','original_pool_digest',
                        'selected_pool_digest','config','reward_config','software','compilation_requested','scenario_ids','bc_schedule')}
                    data['scenario_shapes']=[{'shape_types':['ellipsoid']}]
                    timing={k:1. for k in ('action_inference_wall_seconds','pre_action_geometry_wall_seconds',
                        'environment_step_wall_seconds','td3_update_wall_seconds','total_wall_seconds')}
                    item=dict(measured_steps=3,scenario_coverage=[],timing=timing,
                        batch_transfer_window_completion={'td3_wall_including_final_window_wait_seconds':1.},
                        critic_encoder_compile={'valid_for_speed_comparison':True},
                        local_optimizations={'gpu_td3_compute':{'effective':True,'measurement_new_captures':0}},
                        server_load_before={},server_load_after={})
                    rows=[dict(item,repeat=i) for i in range(int(command[command.index('--repeats')+1]))]
                    data.update(cpu_deduplication=enabled,cpu_deduplication_evidence=evidence,
                        valid_for_speed_comparison=not evidence and not instrumented,runs={'compute':rows})
                    (output/'other/local_optimization_summary.json').write_text(json.dumps(data))
                    if evidence:
                        steps=[dict(measured=True,rng='different' if mismatch and enabled else 'same')]*3
                        (output/'other/rollout_0.json').write_text(json.dumps({'steps':steps}))
                    if instrumented:
                        (output/'review').mkdir()
                        (output/'review/environment_summary.json').write_text('{}')
                    return SimpleNamespace(returncode=0)
                with mock.patch.object(d.subprocess,'run',side_effect=worker):
                    if mismatch:
                        with self.assertRaisesRegex(RuntimeError,'timing skipped'): d.run(args)
                        self.assertEqual(len(calls),2)
                    else:
                        report=d.run(args)
                        self.assertTrue(report['passed'])
                        self.assertIn('both A and B', report['ab_changes']['common'][0])
                        self.assertEqual(len(report['ab_changes']['switch']), 3)
                        self.assertEqual(len(calls),5)
                        self.assertEqual([('--cpu-deduplication-evidence' in c) for c in calls],
                            [True,True,False,False,False])
                        self.assertEqual([('--gpu-compute-environment-diagnostic' in c) for c in calls],
                            [False,False,False,False,True])
                        self.assertTrue((root/'review/environment_summary.json').exists())
