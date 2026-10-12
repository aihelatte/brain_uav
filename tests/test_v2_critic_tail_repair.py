"""CPU copy/guard checks only; graph replay is a stub, not CUDA evidence."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest import mock
import torch
from brain_uav.trainers.v2_critic_tail import CriticUpdateTail


class TestCriticTailRepair(unittest.TestCase):
    def tail(self, *, missing=False):
        models = [torch.nn.Linear(2, 2) for _ in range(4)]
        engine = SimpleNamespace(critic1=models[0], critic2=models[1],
            critic1_target=models[2], critic2_target=models[3], tau=.005)
        parameters = list(models[0].parameters()) + list(models[1].parameters())
        engine.critic_optimizer = torch.optim.Adam(parameters)
        for p in parameters:
            p.grad = torch.ones_like(p)
        engine.critic_optimizer.step()  # Synthetic nonempty Adam state for address guards.
        for index, p in enumerate(parameters):
            p.grad = torch.full_like(p, index + .25)
        if missing:
            parameters[1].grad = None
        tail = CriticUpdateTail(engine)
        tail.buffers = tuple(None if p.grad is None else torch.zeros_like(p) for p in parameters)
        tail._bind_replay_inputs(parameters)
        tail.graphs = {False: mock.Mock(), True: mock.Mock()}
        tail.signature = tail._signature()
        return tail

    def test_one_foreach_copy_preserves_mapping_none_and_gradient_objects(self):
        tail = self.tail(missing=True)
        gradients = tuple(p.grad for p in tail._parameters)
        original = tuple(None if g is None else g.clone() for g in gradients)
        with mock.patch('brain_uav.trainers.v2_critic_tail.torch._foreach_copy_', wraps=torch._foreach_copy_) as copy:
            for targets in (False, True):
                tail.step(targets)
            self.assertEqual(copy.call_count, 2)
            self.assertEqual(len(copy.call_args.args[0]), 3)
        for p, g, saved, buffer in zip(tail._parameters, gradients, original, tail.buffers):
            self.assertIs(p.grad, g)
            if g is None:
                self.assertIsNone(buffer)
            else:
                torch.testing.assert_close(g, saved, rtol=0, atol=0)
                torch.testing.assert_close(buffer, saved, rtol=0, atol=0)
        # A new backward can replace source objects; copied values must be fresh.
        for p in tail._parameters:
            if p.grad is not None:
                p.grad = torch.full_like(p, 7.)
        tail.step(False)
        for buffer in tail._copy_buffers:
            torch.testing.assert_close(buffer, torch.full_like(buffer, 7.), rtol=0, atol=0)

    def test_mutations_rejected_before_copy_and_replay(self):
        def parameter_replace(t):
            t.engine.critic1.weight = torch.nn.Parameter(t.engine.critic1.weight.detach().clone())
        def parameter_address(t):
            p = t.engine.critic1.weight
            p.data = p.data.clone()
        def parameter_stride(t):
            p = t.engine.critic1.weight
            p.data = p.data.t()
        def parameter_dtype(t):
            p = t.engine.critic1.weight
            p.data = p.data.double()
        def state_tensor(t):
            p = t._parameters[0]
            state = t.engine.critic_optimizer.state[p]
            state['exp_avg'] = state['exp_avg'].clone()
        def state_dictionary(t):
            p = t._parameters[0]
            t.engine.critic_optimizer.state[p] = dict(t.engine.critic_optimizer.state[p])
        def state_mapping(t):
            optimizer = t.engine.critic_optimizer
            optimizer.state = deepcopy(optimizer.state)
        def optimizer_replace(t):
            t.engine.critic_optimizer = torch.optim.Adam(t._parameters)
        def target_replace(t):
            t.engine.critic1_target = deepcopy(t.engine.critic1_target)
        def config(t):
            t.engine.critic_optimizer.param_groups[0]['lr'] *= 2
        def tau(t):
            t.engine.tau *= 2
        def gradient_presence(t):
            t._parameters[0].grad = None
        def gradient_stride(t):
            p = t._parameters[0]
            p.grad = p.grad.t()
        for mutation in (parameter_replace, parameter_address, parameter_stride, parameter_dtype,
                         state_tensor, state_dictionary, state_mapping, optimizer_replace,
                         target_replace, config, tau, gradient_presence, gradient_stride):
            with self.subTest(mutation=mutation.__name__):
                tail = self.tail()
                mutation(tail)
                with mock.patch('brain_uav.trainers.v2_critic_tail.torch._foreach_copy_') as copy:
                    with self.assertRaisesRegex(RuntimeError, 'before|invalidate'):
                        tail.step(True)
                    copy.assert_not_called()
                tail.graphs[True].replay.assert_not_called()
                self.assertEqual(tail.replays, 0)

    def test_no_gradient_stays_absent_and_config_never_repr_tensors(self):
        tail = self.tail()
        for p in tail._parameters:
            p.grad = None
        tail.buffers = tuple(None for p in tail._parameters)
        tail._bind_replay_inputs(tail._parameters)
        tail.signature = tail._signature()
        with mock.patch('brain_uav.trainers.v2_critic_tail.torch._foreach_copy_') as copy:
            tail.step(False)
            copy.assert_not_called()
        tail.engine.critic_optimizer.param_groups[0]['lr'] = torch.tensor(.001)
        with mock.patch.object(torch.Tensor, '__repr__', side_effect=AssertionError('device repr')):
            with self.assertRaisesRegex(RuntimeError, 'scalar optimizer'):
                tail.step(False)


if __name__ == '__main__':
    unittest.main()
