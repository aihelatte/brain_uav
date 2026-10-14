"""Experimental device replay. CPU owns event metadata, never sampled batches.

Weighted primary sampling is the existing sequential Fenwick PPSWOR algorithm.
Success sampling uses partial Fisher-Yates: same uniform/no-replacement law,
but a different NumPy RNG consumption contract from Generator.choice.
"""
from time import perf_counter
from contextlib import contextmanager
import numpy as np
import torch
from brain_uav.observations import EGO_FEATURE_DIM, GOAL_FEATURE_DIM, ZONE_FEATURE_DIM
from brain_uav.observations.v2_batch import V2ObservationBatch
from .v2_replay_buffer import V2ReplayBuffer


class DeviceReplay:
    """Incremental mirror; only public replay mutations are supported.

    CPU replay remains authoritative for slot generations and episode marking.
    Monkey-patching/restoring these methods or writing its arrays directly is
    unsupported. Detach this adapter before rebinding/reloading replay.
    """
    fields = ('ego_features', 'goal_features', 'zone_features', 'zone_count',
              'next_ego_features', 'next_goal_features', 'next_zone_features',
              'next_zone_count', 'action', 'reward', 'done', 'near_goal',
              'line_to_goal_safe')

    def __init__(self, replay, device):
        if not isinstance(replay, V2ReplayBuffer):
            raise TypeError('Device replay requires V2ReplayBuffer.')
        self.cpu, self.device = replay, torch.device(device)
        self.seconds = dict(transition_upload=0., episode_sync=0.)
        self.submissions = 0
        self._success_batch=None
        self.layout = {}
        width = 0
        for name in self.fields:
            array = getattr(replay, name)
            count = int(np.prod(array.shape[1:]))
            self.layout[name] = (slice(width, width + count), array.shape[1:])
            width += count
        self.layout['success'] = (slice(width, width + 1), (1,)); width += 1
        self.layout['failure'] = (slice(width, width + 1), (1,)); width += 1
        self.width = width
        self.primary = torch.zeros((replay.capacity, width), device=device,dtype=torch.float32)
        self.success = torch.zeros((max(1, replay.success_capacity), width), device=device,dtype=torch.float32)
        self.tree = torch.as_tensor(replay._sampling_weight_tree._tree.copy(), device=device)
        self.weights = torch.as_tensor(replay.sample_weight.copy(), device=device)
        self._original = {}
        if replay.size:
            self._upload(range(replay.size), False)
        if replay.success_size:
            self._upload(range(replay.success_size), True)
        for name in ('add', 'add_success_transition', 'mark_success_slots', 'mark_failure_slots'):
            self._original[name] = getattr(replay, name)
            setattr(replay, name, self._hook(name))

    def detach(self):
        for name, function in self._original.items():
            setattr(self.cpu, name, function)
        self._original.clear()

    @contextmanager
    def success_upload_batch(self):
        if self._success_batch is not None:
            raise RuntimeError('Nested GPU episode submissions are unsupported.')
        self._success_batch=set()
        try:
            yield
        finally:
            start=perf_counter()
            indices=sorted(self._success_batch)
            self._success_batch=None
            self._upload(indices,True)
            self.seconds['episode_sync']+=perf_counter()-start

    def _hook(self, name):
        def event(*args, **kwargs):
            start = perf_counter()
            success = name == 'add_success_transition'
            if name == 'add':
                indices = [self.cpu.position]
            elif success:
                indices = [self.cpu.success_position] if self.cpu.success_capacity else []
            else:
                refs = args[0] if args else kwargs['slot_refs']
                # The original method still decides success/failure and generation.
                indices = sorted({int(i) for i, generation in refs
                    if 0 <= i < self.cpu.capacity and self.cpu.write_id[i] == generation})
            result = self._original[name](*args, **kwargs)
            if success and self._success_batch is not None:
                self._success_batch.update(indices)
            else:
                self._upload(indices, success)
            self.seconds['episode_sync' if name.startswith('mark_') or success
                         else 'transition_upload'] += perf_counter() - start
            return result
        return event

    def _upload(self, indices, success):
        indices = list(indices)
        if not indices:
            return
        rows = np.zeros((len(indices), self.width), np.float32)
        for name in self.fields:
            slot, _ = self.layout[name]
            rows[:, slot] = getattr(self.cpu, ('success_' if success else '') + name)[indices].reshape(len(indices), -1)
        rows[:, self.layout['success'][0]] = 1 if success else self.cpu.success[indices, None]
        rows[:, self.layout['failure'][0]] = 0 if success else self.cpu.failure[indices, None]
        index = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        (self.success if success else self.primary).index_copy_(0, index, torch.as_tensor(rows, device=self.device))
        self.submissions += 2
        if not success:
            nodes = set()
            for i in indices:
                j = i + 1
                while j <= self.cpu.capacity:
                    nodes.add(j); j += j & -j
            nodes = sorted(nodes)
            node_index = torch.as_tensor(nodes, device=self.device)
            self.tree.index_copy_(0, node_index, torch.as_tensor(self.cpu._sampling_weight_tree._tree[nodes].copy(), device=self.device))
            self.weights.index_copy_(0, index, torch.as_tensor(self.cpu.sample_weight[indices].copy(), device=self.device))
            self.submissions += 3

    def controls(self, batch_size):
        """Small RNG inputs, no CPU selected indices or observation assembly."""
        if self._success_batch is not None:
            raise RuntimeError('Finish episode submission before GPU sampling.')
        if batch_size > self.cpu.size:
            raise ValueError('GPU sample exceeds replay size.')
        success_count = min(int(batch_size * self.cpu.success_batch_fraction), self.cpu.success_size)
        primary_count = batch_size - success_count
        draws = np.zeros(batch_size, np.float64)
        draws[:primary_count] = self.cpu.rng.random(primary_count)
        ranks = np.zeros(batch_size, np.int64)
        for i in range(success_count):
            ranks[i] = self.cpu.rng.integers(self.cpu.success_size - i)
        return draws, ranks, success_count

    def sample_tensors(self, draws, ranks, primary_count, success_size, *,
                       size=None, expected_total=None, return_status=False):
        """Capture-compatible O(batch * log(capacity)), not O(capacity).

        Restore touched Fenwick nodes in reverse order using original values,
        avoiding accumulated rounding from adding removed weights back.
        Counts are scalar device inputs, so successful episode events don't
        require another graph. All loops have static batch/capacity bounds.
        """
        ranks=ranks.to(torch.long)
        primary_count=primary_count.to(torch.long)
        success_size=success_size.to(torch.long)
        n = draws.numel()
        maximum_success=int(n*self.cpu.success_batch_fraction)
        success_count=n-primary_count
        primary_indices, success_indices, undo = [], [], []
        zero = torch.zeros((), dtype=torch.long, device=self.device)
        failed=(primary_count<0) | (primary_count>n) | (success_count>maximum_success) | (success_count>success_size)
        represented_total=None
        size=draws.new_tensor(self.cpu.size) if size is None else size
        for i in range(n):
            active = i < primary_count
            total = self.tree.new_zeros(())
            j = self.cpu.capacity
            while j:
                total = total + self.tree[j]; j -= j & -j
            if i==0:
                represented_total=total
                if expected_total is not None:
                    threshold=(1e-12*torch.maximum(total.abs(),expected_total.abs())).clamp_min(1e-9)
                    failed=failed | (~torch.isfinite(total)) | (~torch.isfinite(expected_total)) | (total<=0) | ((total-expected_total).abs()>threshold)
            failed=failed | (active & ((~torch.isfinite(total)) | (total<=0)
                | (~torch.isfinite(draws[i])) | (draws[i]<0) | (draws[i]>=1)))
            remaining = draws[i] * total
            index = zero
            bit = 1 << (self.cpu.capacity.bit_length() - 1)
            while bit:
                candidate = index + bit
                value = self.tree.index_select(0,candidate.clamp(max=self.cpu.capacity).reshape(1)).reshape(())
                take = (candidate <= self.cpu.capacity) & (value <= remaining)
                index = torch.where(take, candidate, index)
                remaining = torch.where(take, remaining - value, remaining)
                bit >>= 1
            failed=failed | (active & (index>=size))
            index = torch.where(active, index.clamp(max=self.cpu.capacity-1), zero)
            primary_indices.append(index)
            j = index + 1
            weight = torch.where(active, self.weights.index_select(0,index.reshape(1)).reshape(()), self.weights.new_zeros(()))
            failed=failed | (active & ((~torch.isfinite(weight)) | (weight<=0)))
            # Mask padded iterations; each write remembers its pre-write value.
            for _ in range(self.cpu.capacity.bit_length()):
                valid = (j <= self.cpu.capacity) & active
                safe = j.clamp(max=self.cpu.capacity)
                old = self.tree.index_select(0,safe.reshape(1)).reshape(()).clone()
                undo.append((safe, old))
                self.tree.index_copy_(0,safe.reshape(1),torch.where(valid,old-weight,old).reshape(1))
                j = j + (j & -j)
        for index, value in reversed(undo):
            self.tree.index_copy_(0,index.reshape(1),value.reshape(1))
        restored_total=self.tree.new_zeros(())
        j=self.cpu.capacity
        while j:
            restored_total=restored_total+self.tree[j]; j-=j&-j
        failed=failed | (restored_total!=represented_total)
        # Partial Fisher-Yates on a sparse swap table of at most batch entries.
        keys, values = [], []
        for i in range(maximum_success):
            rank = ranks[i]
            failed=failed | ((i<success_count) & ((rank<0) | (rank>=success_size-i)))
            last = (success_size - i - 1).clamp_min(0)
            picked, replacement = rank, last
            for key, value in zip(keys, values):
                picked = torch.where(rank == key, value, picked)
                replacement = torch.where(last == key, value, replacement)
            keys.append(rank); values.append(replacement)
            success_indices.append(picked.clamp(min=0, max=max(0, self.cpu.success_capacity-1)))
        position = torch.arange(n, device=self.device)
        p = self.primary[torch.stack(primary_indices)]
        if maximum_success:
            local_success=(position-primary_count).clamp(min=0,max=maximum_success-1)
            s=self.success[torch.stack(success_indices)[local_success]]
            rows=torch.where((position < primary_count)[:, None],p,s)
        else:
            rows=p
        return (rows,failed) if return_status else rows

    def unpack(self, rows):
        def value(name):
            slot, shape = self.layout[name]
            return rows[:, slot].reshape((rows.shape[0], *shape))
        def obs(prefix):
            counts = value(prefix + 'zone_count').reshape(-1)
            mask = torch.arange(self.cpu.zone_storage_capacity, device=self.device)[None] < counts[:, None]
            # Dataclass validation is metadata-only; raw tensor math bypasses
            # compiled encoders and has one manual capture owner.
            return V2ObservationBatch(value(prefix+'ego_features'), value(prefix+'goal_features'),
                                      value(prefix+'zone_features'), mask)
        return obs(''), obs('next_'), value

    def gather_indices(self, primary_indices, success_indices):
        """Controlled diagnostic input, same pool ordering; no CPU assembly.

        Caller supplies validated in-range indices. This does not replace
        production sampling or consume its RNG.
        """
        for indices in (primary_indices,success_indices):
            if indices.dtype!=torch.long or indices.ndim!=1 or indices.device!=self.primary.device:
                raise ValueError('Fixed indices must be one-dimensional device int64 tensors.')
        return torch.cat((self.primary.index_select(0,primary_indices),
                          self.success.index_select(0,success_indices)),dim=0)
