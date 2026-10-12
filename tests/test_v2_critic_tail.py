"""Small local guards; real capture is explicitly CUDA-only."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import torch
from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
from brain_uav.trainers.v2_critic_tail import CriticUpdateTail
import test_v2_td3 as fixture


class TestCriticTail(unittest.TestCase):
    def engine(self):
        f = fixture.TestV2TD3(); f.setUp()
        return f.make_engine()

    def test_defaults_and_unsupported_configuration(self):
        e = self.engine()
        self.assertFalse(e.cuda_graph_critic_tail)
        for value, error in ((1, TypeError), (True, ValueError)):
            with self.assertRaises(error):
                V2TD3UpdateEngine(e.actor, e.critic1, e.critic2, e.replay,
                    .001, .001, .99, .005, .1, .05, 2, 2, e.action_low, e.action_high,
                    cuda_graph_critic_tail=value)
        from brain_uav.scripts import train_v2_td3, run_v2_td3_curriculum, profile_v2_td3
        for module in (train_v2_td3, run_v2_td3_curriculum, profile_v2_td3):
            action = next(a for a in module.build_parser()._actions if a.dest == 'cuda_graph_critic_tail')
            self.assertIs(action.default, False)
        self.assertFalse(e.local_optimization_metadata()['cuda_graph_critic_tail']['effective'])

    def test_checkpoint_invalidates_tail_and_baseline_freeze_targets(self):
        f = fixture.TestV2TD3(); f.setUp()
        e = f.make_engine(actor_freeze_steps=25000); f.fill_replay(e)
        e._critic_tail = object()
        e.load_checkpoint_state_dict(deepcopy(e.checkpoint_state_dict()))
        self.assertIsNone(e._critic_tail)
        for step in (1, 2):
            result = e.update_once(total_steps=step)
            self.assertEqual(result.critic_targets_updated, step == 2)
            self.assertFalse(result.actor_updated)
        self.assertEqual(e.critic_target_update_count, 1)
        e._critic_tail = object()
        e.load_network_state_dicts(deepcopy(e.checkpoint_state_dict()))
        self.assertIsNone(e._critic_tail)

    def test_error_checks_precede_tail_creation(self):
        from dataclasses import replace
        from unittest import mock
        f = fixture.TestV2TD3(); f.setUp()
        e = f.make_engine(); f.fill_replay(e)
        # Exercise the shared CPU error barrier only; never attempt a CUDA capture here.
        e.cuda_graph_critic_tail = True
        batch = e.replay.sample(2)
        bad = replace(batch, reward=torch.full_like(batch.reward, float('nan')))
        with mock.patch.object(e.replay, 'sample', return_value=bad):
            with self.assertRaises(FloatingPointError):
                e.update_once(total_steps=2)
        self.assertIsNone(e._critic_tail)
        self.assertFalse(e.critic_optimizer.state)
        hook = next(e.critic1.parameters()).register_hook(lambda g: torch.full_like(g, float('inf')))
        try:
            with mock.patch.object(e.replay, 'sample', return_value=batch):
                with self.assertRaises(FloatingPointError):
                    e.update_once(total_steps=2)
        finally:
            hook.remove()
        self.assertIsNone(e._critic_tail)
        self.assertFalse(e.critic_optimizer.state)
        self.assertEqual(e.critic_target_update_count, 0)

    def test_curriculum_transmits_flag_with_local_optimizations_off(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest import mock
        from brain_uav.scripts import run_v2_td3_curriculum as curriculum
        from brain_uav.config import ScenarioConfig
        from test_run_v2_td3_curriculum import _prepared_initialization
        from brain_uav.trainers.v2_validation import derive_validation_stage_seed
        with TemporaryDirectory() as tmp:
            root = Path(tmp); checkpoint = root / 'bc.pt'; checkpoint.write_bytes(b'mocked')
            calls = []
            def runner(**kwargs):
                calls.append(kwargs)
                return {'stage': kwargs['stage'], 'passed': True, 'checkpoint': str(kwargs['output']),
                        'steps': 1, 'global_steps_end': kwargs['global_steps_start'] + 1}
            pools = {stage: root / (stage + '.json') for stage in ('easy', 'medium', 'hard')}
            def load(path, **kwargs):
                stage = Path(path).stem
                return SimpleNamespace(master_seed=20260904, stage_seed=derive_validation_stage_seed(20260904, stage),
                                       scenario_count=100, content_digest=stage)
            with mock.patch.object(curriculum, 'prepare_v2_stage_initialization',
                    return_value=_prepared_initialization(ScenarioConfig(), source=checkpoint)), \
                 mock.patch.object(curriculum, 'prepare_v2_validation_pools', return_value=pools), \
                 mock.patch.object(curriculum, 'load_v2_validation_pool', side_effect=load):
                curriculum.run_v2_curriculum(bc_checkpoint=checkpoint, output_root=root/'out',
                    validation_pool_dir=root/'pools', device='cpu', stage_runner=runner, cuda_graph_critic_tail=True)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(c['cuda_graph_critic_tail'] for c in calls))
            self.assertTrue(all(not c['window_update_statistics'] and not c['reuse_update_buffers'] for c in calls))

    @unittest.skipUnless(torch.cuda.is_available(), 'real CUDA capture pending server validation')
    def test_minimal_capture_shared_adam_and_alternating_replay(self):
        # Standalone feasibility check before formal compiled ANN/SNN diagnostic.
        device = torch.device('cuda')
        def make():
            models = [torch.nn.Linear(3, 2, device=device) for _ in range(4)]
            e = SimpleNamespace(critic1=models[0], critic2=models[1],
                critic1_target=models[2], critic2_target=models[3], device=device, tau=.005)
            parameters = list(e.critic1.parameters()) + list(e.critic2.parameters())
            e.critic_optimizer = torch.optim.Adam(parameters, lr=.001, fused=True, capturable=True)
            def soft(source, target):
                with torch.no_grad():
                    torch._foreach_mul_(list(target.parameters()), 1-e.tau)
                    torch._foreach_add_(list(target.parameters()), list(source.parameters()), alpha=e.tau)
            e._soft_update = soft
            return e
        old, new = make(), make()
        for name in ('critic1', 'critic2', 'critic1_target', 'critic2_target'):
            getattr(new, name).load_state_dict(getattr(old, name).state_dict())
        tail = CriticUpdateTail(new)
        for index in range(6):
            for e in (old, new):
                e.critic_optimizer.zero_grad(set_to_none=True)
                for group in e.critic_optimizer.param_groups:
                    for p in group['params']:
                        p.grad = torch.full_like(p, .01 * (index + 1))
            original = [p.grad for g in new.critic_optimizer.param_groups for p in g['params']]
            old.critic_optimizer.step()
            if index % 2:
                old._soft_update(old.critic1, old.critic1_target)
                old._soft_update(old.critic2, old.critic2_target)
            tail.step(bool(index % 2))
            torch.cuda.synchronize()
            self.assertTrue(all(p.grad is g for p, g in zip(
                [p for group in new.critic_optimizer.param_groups for p in group['params']], original)))
            for name in ('critic1', 'critic2', 'critic1_target', 'critic2_target'):
                for a, b in zip(getattr(old, name).parameters(), getattr(new, name).parameters()):
                    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
            for a, b in zip(old.critic_optimizer.state.values(), new.critic_optimizer.state.values()):
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=1e-4, atol=1e-5)
        failed = make()
        saved = [p.detach().clone() for m in (failed.critic1, failed.critic2, failed.critic1_target, failed.critic2_target) for p in m.parameters()]
        for g in failed.critic_optimizer.param_groups:
            for p in g['params']:
                p.grad = torch.ones_like(p)
        def fail_target(*args):
            raise RuntimeError('controlled warmup failure')
        failed._soft_update = fail_target
        failed_tail = CriticUpdateTail(failed)
        with self.assertRaisesRegex(RuntimeError, 'controlled warmup'):
            failed_tail.step(True)
        self.assertFalse(failed.critic_optimizer.state)
        for p, value in zip([p for m in (failed.critic1, failed.critic2, failed.critic1_target, failed.critic2_target) for p in m.parameters()], saved):
            torch.testing.assert_close(p, value, rtol=0, atol=0)
        self.assertFalse(failed_tail.graphs)
        self.assertEqual(tail.replays, 6)
        p = new.critic_optimizer.param_groups[0]['params'][0]
        p.grad = None
        before = p.detach().clone()
        with self.assertRaisesRegex(RuntimeError, 'presence'):
            tail.step(False)
        torch.testing.assert_close(p, before, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
