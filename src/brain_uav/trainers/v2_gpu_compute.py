"""Default-off CPU replay + raw full TD3 compute, manual CUDA Graph owner.

No DeviceReplay is constructed. Dynamic input shapes remain intact. Graphs
bind physical gradient-bank addresses; publication only happens at the final
status read. Direct optimizer-state mutation must invalidate before mutation.
"""
from dataclasses import replace
from time import perf_counter
import torch

from brain_uav.observations.v2_batch import V2ObservationBatch
from .v2_replay_buffer import V2ReplayBatch
from .v2_gpu_update import GPUUpdate, DeviceAdam, first_error, require_cuda_abi


def preserving_clip(parameters, loss, error, loss_code, gradient_code, max_norm):
    gradients = [p.grad for p in parameters if p.grad is not None]
    error = first_error(error, ~torch.isfinite(loss).all(), loss_code)
    if not gradients:
        return error, loss.new_zeros(())
    if max_norm is None:
        finite = torch.stack([torch.isfinite(g).all() for g in gradients]).all()
        return first_error(error, ~finite, gradient_code), loss.new_zeros(())
    norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients, 2)), 2)
    error = first_error(error, ~torch.isfinite(norm), gradient_code)
    # Inf must never be multiplied by zero, even temporarily.
    coefficient = torch.where(error == 0, (max_norm/(norm+1e-6)).clamp(max=1.), torch.ones_like(norm))
    torch._foreach_mul_(gradients, coefficient)
    return error, norm


def tensors(batch):
    return [getattr(o, n) for o in (batch.obs, batch.next_obs)
            for n in ('ego_features', 'goal_features', 'zone_features', 'presence_mask')] + [
        getattr(batch, n) for n in ('action', 'reward', 'done', 'success', 'near_goal', 'line_to_goal_safe')]


def clone_batch(batch):
    values = [t.detach().clone() for t in tensors(batch)]
    return V2ReplayBatch(V2ObservationBatch(*values[:4]), values[8], values[9],
        V2ObservationBatch(*values[4:8]), *values[10:], sample_failure_fraction=batch.sample_failure_fraction)


class GPUCompute(GPUUpdate):
    clip_gradients = staticmethod(preserving_clip)
    bank_publication = True

    def __init__(self, engine, *, diagnostic=False):
        started = perf_counter()
        require_cuda_abi()
        self.e, self.diagnostic = engine, diagnostic
        self.engine_device = engine.device
        self.compute_diagnostic = diagnostic
        self.models = (engine.actor, engine.critic1, engine.critic2, engine.actor_target,
                       engine.critic1_target, engine.critic2_target, engine.bc_reference_actor)
        if not all(m.training for m in self.models[:3]):
            raise ValueError('GPU compute requires training online models.')
        if engine.model_type == 'snn' and engine.actor.backend != 'torch':
            raise ValueError('GPU compute requires existing torch SNN backend.')
        self.actor_params = list(engine.actor.parameters())
        self.critic_params = list(engine.critic1.parameters()) + list(engine.critic2.parameters())
        self.params = (self.actor_params, self.critic_params)
        if any(not p.requires_grad for params in self.params for p in params):
            raise ValueError('GPU compute freezing uses counters; trainable parameters must require grad.')
        parameter_sets = [{id(p) for p in m.parameters()} for m in self.models if m is not None]
        if any(not a.isdisjoint(b) for i,a in enumerate(parameter_sets) for b in parameter_sets[i+1:]):
            raise ValueError('GPU compute models must not share parameters.')
        self.bound_device = torch.device('cuda', engine.device.index if engine.device.index is not None else torch.cuda.current_device())
        self.registrations = []
        self.module_bindings = []
        self.buffer_bindings = []
        self.encoder_bindings = []
        # Cache parents once, not recursively rediscover six models each step.
        for model in self.models:
            if model is not None:
                encoder = model.zone_set_encoder
                self.encoder_bindings.append((encoder,encoder.scales,encoder.uav_radius,encoder.config))
                if encoder.scales != engine.actor.scales or encoder.uav_radius != engine.actor.zone_set_encoder.uav_radius:
                    raise ValueError('GPU compute shared relation geometry differs between models.')
                for parent in model.modules():
                    self.module_bindings.extend((parent,name,child) for name,child in parent._modules.items())
                    self.buffer_bindings.extend((parent,name,t,t.data_ptr(),t.shape,t.dtype,t.device)
                        for name,t in parent._buffers.items() if t is not None)
                    for name, p in parent._parameters.items():
                        if p is not None:
                            if p.device != self.bound_device or p.dtype != torch.float32:
                                raise ValueError('GPU compute supports FP32 CUDA only.')
                            self.registrations.append((parent, name, p, p.data_ptr(),p.shape,p.stride(),p.requires_grad))
        self.action_bindings = [(name,getattr(engine,name),getattr(engine,name).data_ptr()) for name in ('action_low','action_high')]
        self.optimizers = (engine.actor_optimizer, engine.critic_optimizer)
        for optimizer, params in zip(self.optimizers, self.params):
            if tuple(map(id, optimizer.param_groups[0]['params'])) != tuple(map(id,params)):
                raise ValueError('GPU compute optimizer/model binding differs.')
        self.optimizer_groups = tuple(o.param_groups[0] for o in self.optimizers)
        self.optimizer_states = tuple(o.state for o in self.optimizers)
        self.optimizer_options = tuple(self.options(o) for o in self.optimizers)
        self.static_configuration = self.static_options(engine)
        self.adams = tuple(DeviceAdam(o) for o in self.optimizers)
        self.native_states = [[o.state.get(p) for p in params] for o,params in zip(self.optimizers,self.params)]
        self.state_handles = [[(a.m[i],a.v[i],a.steps[i],a.m[i].data_ptr(),a.v[i].data_ptr(),a.steps[i].data_ptr())
            for i in range(len(a.params))] for a in self.adams]
        self.banks = [[[torch.zeros_like(p) for p in params] for _ in range(2)] for params in self.params]
        self.published = [None, None]
        self.graphs, self.tensor_evidence = {}, {}
        self.generation = engine._gpu_update_generation
        # Each current/next zone shape has 2 critic-bank variants in each
        # nonactor mode and 4 critic/actor-bank variants in actor mode.
        self.zone_limit = engine.replay.zone_storage_capacity
        self.graph_limit = 8*(self.zone_limit+1)**2
        self.capture_seconds = self.copy_seconds = self.compute_seconds = 0.
        self.ready_to_result_seconds = 0.
        self.captures = self.replays = self.handoffs = 0
        self.mode_replays = {'critic_only':0,'frozen_targets':0,'actor':0}
        self.coefficients = torch.zeros(4, device=engine.device)
        self.noise = torch.zeros(engine.batch_size, engine.action_dim, device=engine.device)
        self.injections = torch.zeros(4, device=engine.device)
        if engine._gpu_update_stream is None:
            engine._gpu_update_stream = torch.cuda.Stream(device=engine.device)
        self.stream = engine._gpu_update_stream
        if self.stream.device != self.bound_device:
            raise ValueError('GPU compute device migration requires a fresh engine stream.')
        self.initialization_seconds = perf_counter()-started
        self.begin_measurement()

    @staticmethod
    def static_options(e):
        values = GPUUpdate.static_options(e)
        for model in (e.actor,e.critic1,e.critic2,e.actor_target,e.critic1_target,e.critic2_target,e.bc_reference_actor):
            if model is not None:
                values += (model.hidden_dim,model.action_dim,getattr(model,'projection_dim',None))
        for actor in (e.actor,e.actor_target,e.bc_reference_actor):
            if actor is not None and hasattr(actor,'snn_head'):
                head = actor.snn_head
                values += (actor.training,actor.time_window,actor.tau,actor.backend,
                           head.time_window,head.tau,head.context_dim,head.hidden_dim,head.action_dim)
                for node in (head.lif1,head.lif2):
                    values += (node.tau,node.v_threshold,node.v_reset,node.detach_reset,node.step_mode,node.backend,
                               type(node.surrogate_function),node.surrogate_function.alpha)
        return values

    def validate_binding(self):
        e = self.e
        if (self.generation != e._gpu_update_generation or
            self.models != (e.actor,e.critic1,e.critic2,e.actor_target,e.critic1_target,e.critic2_target,e.bc_reference_actor) or
            self.optimizers != (e.actor_optimizer,e.critic_optimizer) or e.device != self.engine_device):
            raise RuntimeError('GPU compute binding changed; invalidate before rebinding.')
        for parent,name,child in self.module_bindings:
            if parent._modules.get(name) is not child:
                raise RuntimeError('GPU compute module replaced; invalidate before mutation.')
        for parent,name,t,address,shape,dtype,device in self.buffer_bindings:
            if (parent._buffers.get(name) is not t or t.data_ptr()!=address or t.shape!=shape or t.dtype!=dtype or t.device!=device):
                raise RuntimeError('GPU compute buffer replaced; invalidate before mutation.')
        for encoder,scales,radius,config in self.encoder_bindings:
            if encoder.scales!=scales or encoder.uav_radius!=radius or encoder.config!=config:
                raise RuntimeError('GPU compute encoder configuration changed; invalidate first.')
        for name,t,address in self.action_bindings:
            if getattr(e,name) is not t or t.data_ptr()!=address or t.device!=self.bound_device or t.dtype!=torch.float32:
                raise RuntimeError('GPU compute action bounds rebound; invalidate first.')
        for parent, name, p, address, shape, stride, requires_grad in self.registrations:
            if (parent._parameters.get(name) is not p or p.device != self.bound_device or p.dtype != torch.float32 or
                    p.data_ptr() != address or p.shape != shape or p.stride() != stride or p.requires_grad != requires_grad):
                raise RuntimeError('GPU compute parameter/address/device/dtype changed; invalidate before mutation.')
        if self.static_options(e) != self.static_configuration:
            raise RuntimeError('GPU compute configuration changed; invalidate first.')
        for o, group, state, options, params in zip(self.optimizers, self.optimizer_groups, self.optimizer_states, self.optimizer_options, self.params):
            if (len(o.param_groups) != 1 or o.param_groups[0] is not group or o.state is not state or
                self.options(o) != options or len(group['params']) != len(params) or
                any(a is not b for a,b in zip(group['params'],params))):
                raise RuntimeError('GPU compute optimizer rebound; invalidate before loading.')
        # Cached handles only, no state_dict/named_parameters/tree signatures.
        for group,o in enumerate(self.optimizers):
            for p,expected,handles in zip(self.params[group],self.native_states[group],self.state_handles[group]):
                actual = o.state.get(p)
                if actual is not expected:
                    raise RuntimeError('GPU compute Adam state replaced; invalidate before mutation.')
                if actual:
                    m,v,step,mp,vp,sp = handles
                    if (len(actual) != 3 or actual.get('exp_avg') is not m or
                        actual.get('exp_avg_sq') is not v or actual.get('step') is not step or
                        m.data_ptr()!=mp or v.data_ptr()!=vp or step.data_ptr()!=sp or
                        m.device!=self.bound_device or v.device!=self.bound_device or step.device!=self.bound_device or
                        m.dtype!=torch.float32 or v.dtype!=torch.float32 or step.dtype!=torch.float32 or
                        m.shape!=p.shape or v.shape!=p.shape or step.shape!=torch.Size([])):
                        raise RuntimeError('GPU compute Adam tensor binding changed; invalidate before mutation.')

    def key(self, batch, mode):
        if batch.batch_size != self.e.batch_size:
            raise ValueError('GPU compute supports configured batch size only.')
        if max(batch.obs.max_zone_count, batch.next_obs.max_zone_count) > self.zone_limit:
            raise ValueError('GPU compute dynamic shape exceeds replay zone capacity.')
        for t in tensors(batch):
            if t.device != self.bound_device or t.dtype not in (torch.float32, torch.bool) or t.layout != torch.strided:
                raise ValueError('GPU compute batch device/dtype/layout mismatch.')
        work = tuple(0 if i is None else 1-i for i in self.published)
        if mode != 'actor': work = (0, work[1])  # inactive actor bank never enters graph.
        return (batch.obs.max_zone_count, batch.next_obs.max_zone_count, mode, *work)

    def compute(self, batch, mode):
        def value(name):
            if name == 'failure':
                return batch.reward.new_full(batch.reward.shape, batch.sample_failure_fraction)
            return getattr(batch, name)
        # fraction is a runtime input, not a captured Python value.
        def runtime_value(name):
            return self.failure_fraction.expand_as(batch.reward) if name == 'failure' else value(name)
        result = self.compute_batch(batch.obs, batch.next_obs, runtime_value,
            batch.reward.new_zeros((), dtype=torch.bool), mode)
        return torch.cat((result, self.publication.float()))

    def capture_shape(self, key, batch):
        if len(self.graphs) >= self.graph_limit:
            raise RuntimeError('GPU compute bounded shape cache exhausted; no fallback.')
        # No eviction: old graphs share bank/Adam addresses. Reject low free VRAM.
        if torch.cuda.mem_get_info(self.e.device)[0] < 256*1024**2:
            raise MemoryError('GPU compute graph reserve exhausted; no fallback.')
        started = perf_counter()
        stable = clone_batch(batch)
        self.failure_fraction = batch.reward.new_tensor(batch.sample_failure_fraction)
        fraction = self.failure_fraction
        saved = [t.detach().clone() for t in self.persistent()]
        bank_tensors = [g for group in self.banks for bank in group for g in bank]
        bank_saved = [g.detach().clone() for g in bank_tensors]
        refs = [p.grad for params in self.params for p in params]
        grad_saved = [None if g is None else g.detach().clone() for g in refs]
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(self.e.device)
        counts = tuple(getattr(self.e,n) for n in ('update_count','critic_update_count','critic_target_update_count','actor_update_count','last_total_steps'))
        mode, actor_bank, critic_bank = key[2:]
        old_flags = self.injections.detach().clone()
        self.injections.zero_()
        self.stream.wait_stream(torch.cuda.current_stream(self.e.device))

        def bind(presence):
            for group, bank in enumerate((actor_bank,critic_bank)):
                for p,g,active in zip(self.params[group],self.banks[group][bank],presence[group]):
                    p.grad = g if active else None

        def reset_persistent():
            with torch.no_grad():
                for dst,src in zip(self.persistent(),saved): dst.copy_(src)

        try:
            with torch.cuda.stream(self.stream):
                discovered = []
                # With .grad=None discover real participation, also on the
                # other mask extreme. Tensor-only V2 masks have no Python branch.
                masks = (stable.obs.presence_mask, stable.next_obs.presence_mask)
                mask_saved = [m.detach().clone() for m in masks]
                for extreme in (None, False, True):
                    for params in self.params:
                        for p in params: p.grad = None
                    if extreme is not None:
                        for m in masks: m.fill_(extreme)
                    self.compute(stable, mode)
                    found = tuple(tuple(p.grad is not None for p in params) for params in self.params)
                    if discovered and found != discovered[0]:
                        raise RuntimeError('Unsupported mask-dependent gradient participation.')
                    discovered.append(found)
                    reset_persistent()
                for m,v in zip(masks,mask_saved): m.copy_(v)
                presence = discovered[0]
                for _ in range(3):
                    bind(presence); self.compute(stable,mode)
                reset_persistent()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.stream):
                    bind(presence)
                    output = self.compute(stable,mode)
            torch.cuda.current_stream(self.e.device).wait_stream(self.stream)
            torch.cuda.synchronize(self.e.device)
            self.assert_snn_reset()
        except BaseException as exc:
            exc.gpu_compute_context = {'phase':'capture', 'key':list(key), 'completed_graphs':len(self.graphs)}
            raise
        finally:
            torch.cuda.current_stream(self.e.device).wait_stream(self.stream)
            torch.cuda.synchronize(self.e.device)
            with torch.no_grad():
                reset_persistent()
                for dst,src in zip(bank_tensors,bank_saved): dst.copy_(src)
                for p,ref,src in zip([p for params in self.params for p in params],refs,grad_saved):
                    p.grad = ref
                    if ref is not None: ref.copy_(src)
                self.injections.copy_(old_flags)
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng,self.e.device)
        for dst,src in zip(self.persistent(),saved):
            torch.testing.assert_close(dst,src,rtol=0,atol=0,equal_nan=True)
        for p,ref,src in zip([p for params in self.params for p in params],refs,grad_saved):
            if p.grad is not ref: raise RuntimeError('Capture changed public gradient reference.')
            if ref is not None: torch.testing.assert_close(ref,src,rtol=0,atol=0,equal_nan=True)
        if counts != tuple(getattr(self.e,n) for n in ('update_count','critic_update_count','critic_target_update_count','actor_update_count','last_total_steps')):
            raise RuntimeError('Capture changed counters.')
        self.graphs[key] = (graph,stable,fraction,output,presence,self.tensor_evidence.get(mode,{}))
        self.capture_seconds += perf_counter()-started
        self.captures += 1

    def assert_snn_reset(self):
        if self.e.model_type == 'snn':
            for actor in (self.e.actor,self.e.actor_target,self.e.bc_reference_actor):
                if actor is not None:
                    for node in (actor.snn_head.lif1,actor.snn_head.lif2):
                        if isinstance(node.v,torch.Tensor):
                            if bool(torch.count_nonzero(node.v)): raise RuntimeError('SNN reset failed.')
                        elif node.v != 0: raise RuntimeError('SNN reset failed.')

    def step_batch(self, batch, noise, total_steps, bc_lambda, *, recorder=None):
        start = perf_counter()
        self.validate_binding()
        targets = total_steps % self.e.policy_delay == 0
        actor = targets and total_steps > self.e.actor_freeze_steps
        mode = 'actor' if actor else 'frozen_targets' if targets else 'critic_only'
        key = self.key(batch,mode)
        self.noise.copy_(noise)
        self.coefficients.copy_(torch.tensor([bc_lambda,self.e.spike_regularization.beta(total_steps),self.e.policy_noise,self.e.noise_clip],device=self.e.device))
        if key not in self.graphs: self.capture_shape(key,batch)
        graph,stable,fraction,output,presence,evidence = self.graphs[key]
        copy_started = perf_counter()
        floats, booleans = [], []
        for dst,src in zip(tensors(stable),tensors(batch)):
            (booleans if dst.dtype == torch.bool else floats).append((dst,src))
        for pairs in (floats,booleans):
            torch._foreach_copy_([p[0] for p in pairs],[p[1] for p in pairs])
        fraction.fill_(batch.sample_failure_fraction)
        copy_elapsed = perf_counter()-copy_started
        self.copy_seconds += copy_elapsed
        replay_started = perf_counter()
        graph.replay(); self.replays += 1
        self.mode_replays[mode] += 1
        values = output.cpu().tolist()  # sole terminal state/statistics barrier
        compute_elapsed = perf_counter()-replay_started
        self.compute_seconds += compute_elapsed
        self.handoffs += 2  # coefficient submission + combined status/statistics read; batch/noise copies separately accounted
        for group,publish in enumerate((values[-1],values[-2])):
            if publish:
                self.published[group] = key[3+group]
                for p,g,active in zip(self.params[group],self.banks[group][key[3+group]],presence[group]):
                    p.grad = g if active else None
        error = int(values[0])
        for group, committed in ((1,error in (0,3,4)),(0,actor and error == 0)):
            if committed:
                a = self.adams[group]
                for i,(p,state,active) in enumerate(zip(a.params,a.states,presence[group])):
                    if active:
                        a.optimizer.state[p] = state
                        self.native_states[group][i] = state
        if recorder is not None:
            for name,t in evidence.items(): recorder(name,t)
        if error:
            if error >= 3 and targets: self.e.critic_target_update_count += 1
            component = 'critic' if error < 3 else 'actor'
            self.ready_to_result_seconds += perf_counter()-start
            raise FloatingPointError(f'Non-finite {component} {"loss" if error in (1,3) else "gradient"} at total_steps={total_steps}; GPU compute key={key}.')
        e = self.e
        e.update_count += 1; e.critic_update_count += 1
        e.critic_target_update_count += int(targets); e.actor_update_count += int(actor)
        e.last_total_steps = total_steps
        metric = self.metrics(values[:-2],total_steps,bc_lambda,targets,actor)
        metric = replace(metric, sample_failure_fraction=batch.sample_failure_fraction)
        self.last_sections = dict(gpu_compute_input_copy=copy_elapsed,
                                  gpu_compute_to_result=compute_elapsed)
        self.ready_to_result_seconds += perf_counter()-start
        return metric

    def invalidate(self):
        self.graphs.clear(); self.tensor_evidence.clear()

    def begin_measurement(self):
        self.measurement_captures_before = self.captures
        self.measurement_copy_before = self.copy_seconds
        self.measurement_compute_before = self.compute_seconds
        self.measurement_ready_before = self.ready_to_result_seconds
        self.measurement_replays_before = self.replays

    def metadata(self):
        return dict(requested=True,effective=bool(self.graphs),graphs=len(self.graphs),graph_limit=self.graph_limit,
            capture_seconds=self.capture_seconds,initialization_seconds=self.initialization_seconds,
            replays=self.replays,measurement_new_captures=self.captures-self.measurement_captures_before,
            replay_submissions_by_mode=dict(self.mode_replays),
            captured_current_next_shapes=sorted({(key[0],key[1]) for key in self.graphs}),
            input_copy_submission_seconds=self.copy_seconds-self.measurement_copy_before,
            replay_to_result_wall_seconds=self.compute_seconds-self.measurement_compute_before,
            gpu_batch_ready_to_result_wall_seconds=self.ready_to_result_seconds-self.measurement_ready_before,
            state_reads=self.replays, cpu_gpu_handoffs=self.handoffs,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self.e.device),
            gradient_bank_bytes=sum(g.numel()*g.element_size() for group in self.banks for bank in group for g in bank),
            padding='original CPU replay dynamic cropping, unchanged',
            sampler='original CPU replay; no DeviceReplay or success upload',
            capture_owner='one manual outer CUDA Graph; raw tensor functions, no nested Inductor replay',
            lifecycle='loads invalidate; cached parent registrations detect parameter replacement/address changes; direct nested optimizer-state mutation requires invalidate before mutation',
            validity_checks='nonrecursive cached parent/parameter/buffer/Adam handles; no repr, named_parameters or state_dict scan per step',
            exception_temporary_state='unconsumed backward may run in private bank; public gradients/Adam/targets/counts protected; true-model safety requires CUDA correctness')
