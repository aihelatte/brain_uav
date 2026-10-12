"""CPU-only checks of timer bookkeeping, not CUDA capture evidence."""
from types import SimpleNamespace
from unittest import TestCase, main, mock
import torch
from brain_uav.trainers.v2_critic_tail import CriticUpdateTail


class TestCriticTailTiming(TestCase):
    def tail(self):
        models = [torch.nn.Linear(2, 1) for _ in range(4)]
        engine = SimpleNamespace(critic1=models[0], critic2=models[1],
            critic1_target=models[2], critic2_target=models[3], tau=.005)
        parameters = list(models[0].parameters()) + list(models[1].parameters())
        engine.critic_optimizer = torch.optim.Adam(parameters)
        for p in parameters:
            p.grad = torch.ones_like(p)
        tail = CriticUpdateTail(engine)
        tail.buffers = tuple(torch.zeros_like(p) for p in parameters)
        tail.graphs = {False: mock.Mock(), True: mock.Mock()}
        tail.signature = tail._signature()
        return tail

    def test_disabled_does_not_read_clock_or_accumulate(self):
        tail = self.tail()
        with mock.patch('brain_uav.trainers.v2_critic_tail.perf_counter', side_effect=AssertionError('clock sampled')):
            tail.step(False)
            tail.step(True)
        report = tail.internal_timing_report()
        self.assertFalse(report['enabled'])
        self.assertEqual(report['calls'], 0)
        self.assertTrue(all(v == 0 for v in report['total_seconds'].values()))
        self.assertEqual(tail.replays, 2)

    def test_enabled_aggregates_intervals_and_resets(self):
        tail = self.tail()
        tail.capture_seconds = 12.5
        tail.reset_internal_timing(enabled=True)
        with mock.patch('brain_uav.trainers.v2_critic_tail.perf_counter', side_effect=[0., 2., 5., 6., 10., 20., 22., 25., 26., 30.]) as clock:
            tail.step(False)
            tail.step(True)
        self.assertEqual(clock.call_count, 10)
        report = tail.internal_timing_report()
        self.assertEqual(report['calls'], 2)
        self.assertEqual(report['total_seconds'], dict(checks=4., gradient_copy_submission=6., replay_submission=8., total=20., uncovered=2.))
        self.assertEqual(report['average_seconds']['total'], 10.)
        self.assertEqual(report['capture_and_warmup_seconds'], 12.5)
        self.assertIn('not pure GPU compute', report['note'])
        tail.reset_internal_timing(enabled=True)
        self.assertEqual(tail.internal_timing_report()['calls'], 0)
        self.assertEqual(tail.internal_timing_report()['total_seconds']['total'], 0)
        self.assertEqual(tail.capture_seconds, 12.5)
        tail.reset_internal_timing()
        self.assertFalse(tail.internal_timing_report()['enabled'])

    def test_first_capture_call_excluded_from_steady_state(self):
        tail = self.tail()
        graphs = tail.graphs
        tail.graphs = {}
        tail.reset_internal_timing(enabled=True)
        # Stub capture only to test exclusion bookkeeping; no CUDA claim.
        def capture(parameters):
            tail.graphs = graphs
        with mock.patch.object(tail, '_capture', side_effect=capture), \
             mock.patch('brain_uav.trainers.v2_critic_tail.perf_counter', side_effect=AssertionError('capture mixed into steady state')):
            tail.step(False)
        self.assertEqual(tail.internal_timing_report()['calls'], 0)
        self.assertEqual(tail.replays, 1)


if __name__ == '__main__':
    main()
