"""Opt-in full TD3 update, one manual CUDA capture owner, PyTorch 2.5.1/cu124.

No CPU loss/gradient decision between backward and parameter commits. Models
are raw tensor forwards, not the separately compiled inference callables.
Direct external mutation of models/optimizers/replay is unsupported: call
engine.invalidate_gpu_update() BEFORE modifying/rebinding them.
"""
from time import perf_counter
from math import isfinite
import numpy as np
import torch
from torch.nn import functional as F
from .v2_gpu_replay import DeviceReplay


def require_cuda_abi():
    if not torch.cuda.is_available():
        raise RuntimeError('gpu_td3_update requires CUDA; no CPU fallback.')
    if torch.__version__.split('+')[0] != '2.5.1' or torch.version.cuda != '12.4':
        raise RuntimeError('gpu_td3_update requires validated PyTorch 2.5.1+cu124 ABI.')
    if not hasattr(torch, '_fused_adam_'):
        raise RuntimeError('Internal fused Adam capability missing.')


def first_error(error, failed, code):
    return torch.where((error == 0) & failed, code, error)


class DeviceAdam:
    """Native optimizer state objects remain the checkpoint authority.

    Per-parameter None is excluded, never substituted by a zero gradient.
    Warmup proves all supported parameters participate; unknown participation
    raises before capture, rather than creating a silently incomplete graph.
    """
    def __init__(self, optimizer):
        if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
            raise ValueError('GPU updater supports one native Adam group only.')
        self.optimizer = optimizer
        self.group = optimizer.param_groups[0]
        g = self.group
        if (not g.get('fused') or g.get('amsgrad') or g.get('maximize')
                or g.get('weight_decay', 0) != 0 or g.get('differentiable')):
            raise ValueError('GPU updater requires existing fused, unscaled native Adam.')
        self.params = list(g['params'])
        self.states = []
        self.published = False
        for p in self.params:
            state = optimizer.state.get(p, {})
            if not state:
                state = dict(step=torch.zeros((), device=p.device,dtype=torch.float32),
                             exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
            if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError('Unsupported Adam state.')
            if (state['step'].shape != torch.Size([]) or state['step'].dtype != torch.float32
                    or state['step'].device != p.device):
                raise ValueError('Adam step must be a bound FP32 device scalar.')
            for name in ('exp_avg','exp_avg_sq'):
                t=state[name]
                if t.shape!=p.shape or t.device!=p.device or t.dtype!=p.dtype or t.layout!=torch.strided:
                    raise ValueError('Adam state shape/device/dtype/layout differs: '+name)
            self.states.append(state)
        self.m = [state['exp_avg'] for state in self.states]
        self.v = [state['exp_avg_sq'] for state in self.states]
        self.steps = [state['step'] for state in self.states]

    def publish(self):
        # First successful commit attaches the SAME captured state tensors.
        # Frozen/error paths leave previously absent optimizer state absent.
        if self.published:
            return
        for p, state in zip(self.params, self.states):
            self.optimizer.state[p] = state
        self.published = True

    @torch.no_grad()
    def step(self, failed):
        active = [i for i, p in enumerate(self.params) if p.grad is not None]
        if not active:
            return
        steps = [self.steps[i] for i in active]
        torch._foreach_add_(steps, 1)
        torch._fused_adam_([self.params[i] for i in active], [self.params[i].grad for i in active],
            [self.m[i] for i in active], [self.v[i] for i in active], [], steps,
            amsgrad=False, lr=self.group['lr'], beta1=self.group['betas'][0],
            beta2=self.group['betas'][1], weight_decay=0., eps=self.group['eps'],
            maximize=False, grad_scale=None, found_inf=failed)
        torch._foreach_sub_(steps, [failed] * len(steps))

    def state_tensors(self):
        return self.m + self.v + self.steps


def checked_clip(parameters, loss, error, loss_code, gradient_code, max_norm):
    gradients = [p.grad for p in parameters if p.grad is not None]
    error = first_error(error, ~torch.isfinite(loss).all(), loss_code)
    if max_norm is None:
        finite = torch.stack([torch.isfinite(g).all() for g in gradients]).all()
        return first_error(error, ~finite, gradient_code), loss.new_zeros(())
    norms = torch._foreach_norm(gradients, 2)
    norm = torch.linalg.vector_norm(torch.stack(norms), 2)
    error = first_error(error, ~torch.isfinite(norm), gradient_code)
    torch._foreach_mul_(gradients, (max_norm / (norm + 1e-6)).clamp(max=1.))
    return error, norm


@torch.no_grad()
def gated_targets(online, target, tau, error):
    # Original foreach mul then foreach add order; condition selects persistent
    # writes, not NaN*0, zero LR, or a fake optimizer skip.
    values = torch._foreach_mul(target, 1. - tau)
    torch._foreach_add_(values, online, alpha=tau)
    for p, value in zip(target, values):
        p.copy_(torch.where(error == 0, value, p))


class GPUUpdate:
    def __init__(self, engine, *, diagnostic=False):
        initialization_started=perf_counter()
        require_cuda_abi()
        self.e = engine
        if type(diagnostic) is not bool: raise TypeError('diagnostic must be bool.')
        self.diagnostic=diagnostic
        self.tensor_evidence={}
        expected_device = torch.device('cuda', engine.device.index if engine.device.index is not None else torch.cuda.current_device())
        self.models = (engine.actor, engine.critic1, engine.critic2, engine.actor_target,
                       engine.critic1_target, engine.critic2_target, engine.bc_reference_actor)
        if not all(m.training for m in self.models[:3]):
            raise ValueError('GPU update capture requires online actor/critics in training mode.')
        if engine.model_type=='snn' and engine.actor.backend!='torch':
            raise ValueError('GPU updater capability is validated only for the existing torch SNN backend.')
        self.optimizers = (engine.actor_optimizer, engine.critic_optimizer)
        self.bound_device=engine.device
        self.optimizer_groups=tuple(o.param_groups[0] for o in self.optimizers)
        self.optimizer_states=tuple(o.state for o in self.optimizers)
        self.optimizer_options=tuple(self.options(o) for o in self.optimizers)
        self.optimizer_param_lists=tuple(g['params'] for g in self.optimizer_groups)
        self.static_configuration=self.static_options(engine)
        if any(p.dtype != torch.float32 or p.device != expected_device
               for m in self.models if m is not None for p in m.parameters()):
            raise ValueError('GPU updater supports bound FP32 CUDA models only.')
        encoder=engine.actor.zone_set_encoder
        if any(m.zone_set_encoder.scales!=encoder.scales or
               m.zone_set_encoder.uav_radius!=encoder.uav_radius
               for m in self.models if m is not None):
            raise ValueError('Shared relations require identical observation scales and collision radius.')
        self.actor_params = list(engine.actor.parameters())
        self.critic_params = list(engine.critic1.parameters()) + list(engine.critic2.parameters())
        self.initialize_actor_loss_scalars()
        if any(not p.requires_grad for p in self.actor_params+self.critic_params):
            raise ValueError('GPU updater requires the existing dense trainable topology; actor freezing uses stage counters.')
        parameter_sets=[{id(p) for p in m.parameters()} for m in self.models if m is not None]
        if any(not a.isdisjoint(b) for i,a in enumerate(parameter_sets) for b in parameter_sets[i+1:]):
            raise ValueError('GPU update models must not share Parameters.')
        for optimizer,params in ((engine.actor_optimizer,self.actor_params),(engine.critic_optimizer,self.critic_params)):
            bound=[p for group in optimizer.param_groups for p in group['params']]
            if tuple(map(id,bound))!=tuple(map(id,params)):
                raise ValueError('Optimizer parameters do not match current models; rebind before GPU initialization.')
        self.budget = self.memory_budget(engine)
        free, _ = torch.cuda.mem_get_info(engine.device)
        self.budget['free_bytes_before_initialization']=free
        if self.budget['initialization_reservation_bytes'] > free:
            raise MemoryError('Insufficient VRAM for requested replay, models, Adam and graph reservation; capacity not reduced.')
        self.adams = (DeviceAdam(engine.actor_optimizer), DeviceAdam(engine.critic_optimizer))
        self.replay = DeviceReplay(engine.replay, engine.device)
        try:
            b = engine.batch_size
            self.control = torch.zeros(2*b+8, dtype=torch.float64, device=engine.device)
            self.draws = self.control[:b]
            self.ranks = self.control[b:2*b]
            self.primary_count = self.control[2*b]
            self.success_size = self.control[2*b+1]
            self.primary_count.fill_(b)
            self.noise = torch.zeros(b, engine.actor.action_dim, device=engine.device,dtype=torch.float32)
            # bc, spike beta, policy noise and noise clip are runtime device inputs.
            self.coefficients = self.control[2*b+2:2*b+6]
            self.replay_size=self.control[-2]
            self.weight_total=self.control[-1]
            self.replay_size.fill_(engine.replay.size)
            self.weight_total.fill_(engine.replay.total_sample_weight)
            self.injections = torch.zeros(4, device=engine.device,dtype=torch.float32)
            self.fixed_rows = torch.zeros(b, self.replay.width, device=engine.device,dtype=torch.float32)
            self.fixed_mode = torch.zeros((), dtype=torch.bool, device=engine.device)
            self.use_fixed_rows = False
            self.graphs, self.outputs = {}, {}
            self.capture_seconds, self.replays = 0., 0
            self.generation = engine._gpu_update_generation
            self.capture()
        except BaseException:
            self.replay.detach()
            raise
        self.initialization_seconds=perf_counter()-initialization_started
        self.measurement_replays_before=0
        self.measurement_captures_before=len(self.graphs)
        self.handoffs=0

    @staticmethod
    def memory_budget(engine, capacity=None):
        r = engine.replay
        capacity = r.capacity if capacity is None else capacity
        success_capacity = r.success_capacity if capacity == r.capacity else int(capacity*r.success_replay_fraction)
        # Actual packed FP32 row layout, double Fenwick/weights, sampled row
        # and O(batch log capacity) undo + O(batch^2) sparse swaps.
        width = sum(int(np.prod(getattr(r, f).shape[1:])) for f in DeviceReplay.fields) + 2
        replay = 4 * width * (capacity + max(1, success_capacity)) + 8*(2*capacity+1)
        models = [engine.actor, engine.critic1, engine.critic2, engine.actor_target,
                  engine.critic1_target, engine.critic2_target, engine.bc_reference_actor]
        model_bytes = sum(t.numel()*t.element_size() for m in models if m is not None
                          for t in (*m.parameters(),*m.buffers()))
        train_bytes = sum(p.numel()*p.element_size() for m in models[:3] for p in m.parameters())
        adam_steps=4*sum(1 for m in models[:3] for _ in m.parameters())
        batch = engine.batch_size * width * 4 * 4
        sampler = engine.batch_size * (capacity.bit_length()*24 + engine.batch_size*24)
        # Graph/private and activation allocations depend on CUDA allocator and
        # model. Explicit conservative reservation, then measure actual peak;
        # this is not a fit proof based solely on the 0.605 GiB payload.
        reserve = max(2*1024**3, 12*(model_bytes + train_bytes + batch + sampler))
        return dict(capacity=capacity, success_capacity=success_capacity,
                    zones=r.zone_storage_capacity, replay_bytes=replay, model_bytes=model_bytes,
                    adam_and_gradients_bytes=3*train_bytes+adam_steps, batch_bytes=batch,
                    sampler_workspace_bytes=sampler, graph_temporary_reservation_bytes=reserve,
                    initialization_reservation_bytes=replay + 3*train_bytes + adam_steps + batch + sampler + reserve,
                    total_resident_reservation_bytes=model_bytes+replay+3*train_bytes+adam_steps+batch+sampler+reserve,
                    fit_validated=False, graph_private_bytes='measured after capture; reservation is conservative, not exact')

    def invalidate(self):
        self.replay.detach()
        self.graphs.clear(); self.outputs.clear()
        self.tensor_evidence.clear()
        if hasattr(self,'actor_grad_buffers'): self.actor_grad_buffers.clear()

    def validate_binding(self):
        # Supported mutation entrypoints increment generation. Direct tensor or
        # nested Parameter mutation is explicitly unsupported; no recursive
        # signature scan is added to the steady path.
        e = self.e
        if (self.generation != e._gpu_update_generation or
            self.models != (e.actor,e.critic1,e.critic2,e.actor_target,e.critic1_target,e.critic2_target,e.bc_reference_actor)
                or self.optimizers != (e.actor_optimizer,e.critic_optimizer)
                or self.replay.cpu is not e.replay or e.device!=self.bound_device):
            raise RuntimeError('GPU updater binding changed; explicitly invalidate before rebinding.')
        if self.static_options(e)!=self.static_configuration:
            raise RuntimeError('GPU update static configuration/training mode changed; invalidate before changing it.')
        for o,group,state,options,params in zip(self.optimizers,self.optimizer_groups,self.optimizer_states,self.optimizer_options,self.optimizer_param_lists):
            if len(o.param_groups)!=1 or o.param_groups[0] is not group or o.state is not state or group['params'] is not params or self.options(o)!=options:
                raise RuntimeError('GPU optimizer binding/options changed; invalidate before loading or changing optimizer settings.')

    @staticmethod
    def options(optimizer):
        if len(optimizer.param_groups)!=1:
            raise ValueError('GPU updater supports one Adam group.')
        g=optimizer.param_groups[0]
        numeric=(g.get('lr'),g.get('eps'),g.get('weight_decay'),*g.get('betas',()))
        if any(type(value) not in (float,int) or not isfinite(value) for value in numeric):
            raise ValueError('GPU optimizer options must be finite Python scalars, not device tensors.')
        return tuple(g.get(k) for k in ('lr','betas','eps','weight_decay','fused','capturable','amsgrad','maximize'))

    @staticmethod
    def static_options(e):
        values=(e.target_q_aggregation,e.mc_return_weight,e.mc_return_max_age_steps,e.gamma,e.tau,e.actor_grad_clip_norm,e.critic_grad_clip_norm,e.actor_rl_scale_alpha,
                e.terminal_geo_regularization_enabled,e.terminal_geo_radius,e.terminal_geo_lambda,
                e.spike_regularization.target_rate,e.actor.scales,
                *(m.training for m in (e.actor,e.critic1,e.critic2,e.actor_target,e.critic1_target,e.critic2_target)))
        if any(isinstance(value,torch.Tensor) for value in values):
            raise ValueError('GPU static configuration must not contain device tensor controls.')
        return values

    def arguments(self, obs):
        encoder = self.e.actor.zone_set_encoder
        clean, _, valid, relations, pair_mask = encoder._compute_shared_relation_tensors(
            obs.ego_features, obs.zone_features, obs.presence_mask)
        return (obs.ego_features, obs.goal_features, clean, obs.presence_mask, valid, relations, pair_mask)

    def actor_forward(self, actor, args):
        if self.e.model_type == 'ann':
            return actor._compute_full_forward_tensors(*args), self.noise.new_zeros(())
        with actor.reset_state_context():
            context = actor.zone_set_encoder._compute_policy_context_tensors(*args)
            raw, first, second = actor.snn_head._forward_impl(actor.projection(context))
            if self.diagnostic and actor is self.e.actor:
                self._activity_evidence={'activity.l1':first.detach().bool().clone(),
                                         'activity.l2':second.detach().bool().clone()}
            return raw.tanh()*actor.action_limit, (first.mean() if actor is self.e.actor else self.noise.new_zeros(()))

    def operation(self, mode):
        if self.use_fixed_rows:
            sampled=self.fixed_rows
            sample_error=torch.zeros((),dtype=torch.bool,device=self.e.device)
        else:
            sampled,sample_error=self.replay.sample_tensors(
                self.draws,self.ranks,self.primary_count,self.success_size,
                size=self.replay_size,expected_total=self.weight_total,return_status=True)
        rows = torch.where(self.fixed_mode, self.fixed_rows, sampled)
        obs, following, value = self.replay.unpack(rows)
        return self.compute_batch(obs, following, value, sample_error, mode, rows=rows)

    clip_gradients = staticmethod(checked_clip)
    bank_publication = False

    def initialize_actor_loss_scalars(self):
        # CPU scalar uploads must finish before any manual graph capture.
        like = self.actor_params[0]
        self.actor_loss_alpha = torch.tensor(self.e.actor_rl_scale_alpha,
            device=like.device, dtype=like.dtype)
        self.actor_loss_terminal_lambda = torch.tensor(
            self.e.terminal_geo_lambda if self.e.terminal_geo_regularization_enabled else 0.,
            device=like.device, dtype=like.dtype)
        # Both values are covered by static_options(), checked before replay.

    def compute_batch(self, obs, following, value, sample_error, mode, *, rows=None):
        # Shared REAL TD3 arithmetic. CPU-batch executor never calls operation()
        # or DeviceReplay: no sampler, packed rows or fixed padding in its graph.
        from .v2_td3 import _actor_loss_tensor_block
        e = self.e
        coefficients=self.coefficients.float()
        evidence={}
        def record(name,value):
            if self.diagnostic: evidence[name]=value.detach().clone()
        if rows is not None: record('replay.rows', rows)
        record('replay.mask', obs.presence_mask)
        record('replay.next_mask', following.presence_mask)
        current, next_args = self.arguments(obs), self.arguments(following)
        with torch.no_grad():
            next_action, _ = self.actor_forward(e.actor_target, next_args)
            noise = (self.noise*coefficients[2]).clamp(-coefficients[3],coefficients[3])
            next_action = torch.maximum(torch.minimum(next_action+noise,e.action_high),e.action_low)
            _, tq1, tq2, target = e._compute_target_critics_td_tensors(
                *next_args,next_action,value('reward'),value('done'))
        for p in self.critic_params:
            if p.grad is not None: p.grad.zero_()
        q1 = e.critic1._compute_full_forward_tensors(*current,value('action'))
        q2 = e.critic2._compute_full_forward_tensors(*current,value('action'))
        mc_stats = ()
        if e.mc_return_weight > 0:
            from .v2_mc_return import critic_loss_components
            loss, td_loss, mc_loss, mc_count, mc_fraction = critic_loss_components(
                q1, q2, target, e.mc_return_weight, value('mc_return'), value('mc_return_mask'))
            mc_stats = (td_loss, mc_loss, mc_loss * e.mc_return_weight, mc_count, mc_fraction)
        else:
            loss = F.mse_loss(q1,target)+F.mse_loss(q2,target)
        for name,value_tensor in (('target.noise',noise),('target.action',next_action),
            ('target.q1',tq1),('target.q2',tq2),('target.td',target),
            ('critic.q1',q1),('critic.q2',q2),('critic.loss',loss)):
            record(name,value_tensor)
        loss = torch.where(self.injections[0] != 0, torch.full_like(loss,float('nan')),loss)
        if self.bank_publication:
            publish_critic = torch.isfinite(loss) & ~sample_error
            publish_actor = torch.zeros_like(publish_critic)
        loss.backward()
        if self.diagnostic:
            for module_name in ('critic1','critic2'):
                for name,p in getattr(e,module_name).named_parameters():
                    if p.grad is not None: record('gradient.pre_clip.'+module_name+'.'+name,p.grad)
        if self.critic_params[0].grad is not None:
            g = self.critic_params[0].grad
            g.copy_(torch.where(self.injections[1] != 0,torch.full_like(g,float('inf')),g))
        error = torch.where(sample_error,5,0).to(torch.long)
        error, critic_norm = self.clip_gradients(self.critic_params,loss,error,1,2,e.critic_grad_clip_norm)
        self.adams[1].step((error != 0).float())
        if mode != 'critic_only':
            for online, target_model in ((e.critic1,e.critic1_target),(e.critic2,e.critic2_target)):
                gated_targets(list(online.parameters()),list(target_model.parameters()),e.tau,error)
        terms = [loss.new_zeros(())]*8
        rate, actor_norm = loss.new_zeros(()), loss.new_zeros(())
        if mode == 'actor':
            # Frozen context and detached head weights preserve Q->action
            # gradients without writing any critic parameter gradient.
            with torch.no_grad():
                context = e.critic1.zone_set_encoder._compute_policy_context_tensors(*current)
            for p in self.actor_params:
                if p.grad is not None: p.grad.zero_()
            actions, rate = self.actor_forward(e.actor,current)
            if getattr(self, 'compute_diagnostic', False): record('actor.action', actions)
            if self.diagnostic and e.model_type=='snn': evidence.update(self._activity_evidence)
            with torch.no_grad():
                reference = None if e.bc_reference_actor is None else self.actor_forward(e.bc_reference_actor,current)[0]
            q = torch.func.functional_call(e.critic1.head,
                {name:p.detach() for name,p in e.critic1.head.named_parameters()},
                (torch.cat((context,actions),dim=-1),))
            terms = _actor_loss_tensor_block(actions,q,reference,obs.ego_features,obs.goal_features,
                value('line_to_goal_safe'),e.action_low,e.action_high,
                self.actor_loss_alpha,coefficients[0],
                self.actor_loss_terminal_lambda,
                e.actor.scales.horizontal_span,e.actor.scales.vertical_span,e.actor.scales.gamma_max,
                e.terminal_geo_radius,e.terminal_geo_regularization_enabled,
                rate if e.model_type=='snn' else None,coefficients[1],
                e.spike_regularization.target_rate,return_spike_terms=True)
            actor_loss = torch.where(self.injections[2]!=0,torch.full_like(terms[0],float('nan')),terms[0])
            for name,tensor in zip(('actor_loss','rl_actor_loss','scaled_rl_actor_loss','actor_rl_scale','bc_loss','terminal_geo_loss'),terms[:6]):
                record('actor.'+name,tensor)
            record('actor.spike_rate',rate); record('actor.spike_loss',terms[6])
            record('actor.spike_beta',coefficients[1])
            if e.model_type=='snn': record('actor.weighted_spike',terms[7])
            if self.bank_publication: publish_actor = (error == 0) & torch.isfinite(actor_loss)
            actor_loss.backward()
            if self.diagnostic:
                for name,p in e.actor.named_parameters():
                    if p.grad is not None: record('gradient.pre_clip.actor.'+name,p.grad)
            if self.actor_params[0].grad is not None:
                g=self.actor_params[0].grad
                g.copy_(torch.where(self.injections[3]!=0,torch.full_like(g,float('inf')),g))
            error, actor_norm = self.clip_gradients(self.actor_params,actor_loss,error,3,4,e.actor_grad_clip_norm)
            self.adams[0].step((error != 0).float())
            gated_targets(self.actor_params,list(e.actor_target.parameters()),e.tau,error)
        with torch.no_grad():
            if self.bank_publication: self.publication = torch.stack((publish_critic, publish_actor))
            if self.diagnostic: self.tensor_evidence[mode]=evidence
            td=(q1-target).abs(); failure=(value('success')<.5).float()
            return torch.stack((error.float(),loss,*terms,rate,actor_norm,
                value('success').mean(),value('failure').mean(),value('near_goal').mean(),
                q1.mean(),q1.std(unbiased=False),q1.abs().max(),target.mean(),td.mean(),
                (td*failure).sum()/failure.sum().clamp_min(1.),critic_norm,*mc_stats))

    def persistent(self):
        return [p for m in self.models if m is not None for p in m.parameters()] + [x for a in self.adams for x in a.state_tensors()]

    def capture(self):
        start=perf_counter(); e=self.e
        saved=[p.detach().clone() for p in self.persistent()]
        original_actor_grads=[p.grad for p in self.actor_params]
        cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state(e.device)
        # One pre-capture ownership boundary, not a reassignment per graph.
        # Prior Inductor backward outputs must not become graph gradient inputs.
        for p in self.actor_params+self.critic_params:
            p.grad = None
        self.use_fixed_rows=True  # Capture construction must not consume replay/RNG.
        if e._gpu_update_stream is None:
            e._gpu_update_stream=torch.cuda.Stream(device=e.device)
        stream=e._gpu_update_stream
        expected_device=torch.device('cuda',e.device.index if e.device.index is not None else torch.cuda.current_device())
        if stream.device!=expected_device:
            raise ValueError('GPU device migration requires a fresh engine; old gradient stream cannot be reused.')
        stream.wait_stream(torch.cuda.current_stream(e.device))
        capture_mode='allocate_gradients'
        def restore():
            with torch.no_grad():
                for p,v in zip(self.persistent(),saved): p.copy_(v)
        try:
            with torch.cuda.stream(stream):
                # Allocate backward-owned gradients once using the real loss;
                # then all three captures accumulate into these same addresses.
                self.operation('actor'); restore()
                for name,p in [(f'actor.{n}',p) for n,p in e.actor.named_parameters()]+[(f'critic.{i}',p) for i,p in enumerate(self.critic_params)]:
                    if p.grad is None:
                        raise RuntimeError('Unsupported missing gradient during capture: '+name)
                self.use_fixed_rows=False
                # Initialize valid sampler controls; warmups never consume CPU RNG.
                self.draws.fill_(.1); self.primary_count.fill_(e.batch_size)
                for mode in ('critic_only','frozen_targets','actor'):
                    capture_mode=mode
                    for _ in range(3): self.operation(mode)
                    restore()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream): self.outputs[mode]=self.operation(mode)
                    self.graphs[mode]=graph
                    restore()
            torch.cuda.current_stream(e.device).wait_stream(stream)
            torch.cuda.synchronize(e.device)
            # Do not reassign .grad after capture: captured addresses stay live.
            # Retained warmup gradients are nonconsumed and overwritten at the
            # next corresponding backward. Checkpoint state has no gradients.
            for p,v in zip(self.persistent(),saved):
                if not torch.equal(p,v): raise RuntimeError('Capture changed persistent state.')
            if not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state(e.device)):
                raise RuntimeError('GPU capture consumed RNG.')
            self.actor_grad_buffers=[p.grad for p in self.actor_params]
            # Keep warmup buffers alive; expose None until the first actual
            # actor backward. Restoring these SAME objects cannot change graph
            # gradient addresses or turn missing gradients into Adam updates.
            self.actor_grad_bound=not all(g is None for g in original_actor_grads)
            if not self.actor_grad_bound:
                for p in self.actor_params: p.grad=None
            self.budget.update(fit_validated=True,allocated_bytes=torch.cuda.memory_allocated(e.device),
                reserved_bytes=torch.cuda.memory_reserved(e.device),peak_allocated_bytes=torch.cuda.max_memory_allocated(e.device))
        except BaseException as exc:
            exc.gpu_update_context=dict(phase='warmup_capture',model=e.model_type,
                mode=capture_mode,completed_captures=list(self.graphs),
                capture_seconds=perf_counter()-start,persistent_restore_completed=False)
            try:
                torch.cuda.current_stream(e.device).wait_stream(stream)
                torch.cuda.synchronize(e.device); restore()
                exc.gpu_update_context['persistent_restore_completed']=True
            except BaseException as cleanup:
                cleanup.gpu_update_context={**exc.gpu_update_context,'restore_error':str(cleanup)}
                raise cleanup from exc
            finally:
                self.invalidate()
            raise
        finally:
            self.capture_seconds=perf_counter()-start

    def step(self, total_steps, bc_lambda, timing_recorder=None):
        from .v2_td3 import V2TD3UpdateMetrics
        start=perf_counter(); self.validate_binding(); e=self.e
        draws,ranks,success=self.replay.controls(e.batch_size)
        control=np.concatenate((draws,ranks.astype(np.float64),
            np.array([e.batch_size-success,e.replay.success_size,bc_lambda,
                      e.spike_regularization.beta(total_steps),e.policy_noise,e.noise_clip,
                      e.replay.size,e.replay.total_sample_weight],np.float64)))
        self.control.copy_(torch.from_numpy(control))  # One host->device control submission.
        self.handoffs+=1
        # Same torch RNG call and placement as old update; explicit noise can be
        # supplied in correctness through a fixed randn_like callable.
        self.noise.copy_(torch.randn_like(self.noise))
        targets=total_steps%e.policy_delay==0
        actor=targets and total_steps>e.actor_freeze_steps
        mode='actor' if actor else 'frozen_targets' if targets else 'critic_only'
        if actor and not self.actor_grad_bound:
            for p,g in zip(self.actor_params,self.actor_grad_buffers): p.grad=g
            self.actor_grad_bound=True
        self.graphs[mode].replay(); self.replays+=1
        values=self.outputs[mode].cpu().tolist()  # Single terminal status/stats barrier.
        self.handoffs+=1
        error=int(values[0])
        if error:
            if error==5:
                raise RuntimeError(f'GPU replay cumulative weight/RNG/slot validation failed at total_steps={total_steps}; mode={mode}.')
            if error>=3:
                self.adams[1].publish()
                if targets: e.critic_target_update_count+=1
            component='critic' if error<3 else 'actor'
            kind='loss' if error in (1,3) else 'gradient'
            raise FloatingPointError(f'Non-finite {component} {kind} at total_steps={total_steps}; GPU mode={mode}.')
        self.adams[1].publish()
        if actor: self.adams[0].publish()
        e.update_count+=1; e.critic_update_count+=1
        e.critic_target_update_count+=int(targets); e.actor_update_count+=int(actor)
        e.last_total_steps=total_steps
        m = self.metrics(values, total_steps, bc_lambda, targets, actor)
        if timing_recorder:
            from .v2_td3 import V2_TD3_UPDATE_TIMING_SECTIONS
            sections={name:0. for name in V2_TD3_UPDATE_TIMING_SECTIONS}
            sections['gpu_td3_sample_to_result']=perf_counter()-start
            timing_recorder(sections)
        return m

    def metrics(self, values, total_steps, bc_lambda, targets, actor):
        from .v2_td3 import V2TD3UpdateMetrics
        e = self.e
        _,critic,*_=values
        # output: error, critic, eight actor terms, rate, norm, 3 fractions,
        # seven critic reductions and critic norm.
        m=V2TD3UpdateMetrics(critic_loss=critic,
            **(dict(zip(('critic_td_loss', 'critic_mc_loss', 'critic_mc_weighted_loss',
                         'mc_valid_count', 'mc_valid_fraction'), values[22:27])) if e.mc_return_weight > 0 else {}),
            actor_loss=values[2] if actor else 0.,rl_actor_loss=values[3] if actor else 0.,
            scaled_rl_actor_loss=values[4] if actor else 0.,actor_rl_scale=values[5] if actor else 1.,
            bc_loss=values[6] if actor else 0.,terminal_geo_loss=values[7] if actor else 0.,
            bc_lambda=bc_lambda if actor else 0.,terminal_geo_lambda=e.terminal_geo_lambda if actor and e.terminal_geo_regularization_enabled else 0.,
            actor_batch_spike_rate_l1=values[10] if actor and e.model_type=='snn' else None,
            spike_regularization_loss=values[8] if actor and e.model_type=='snn' else None,
            spike_beta_eff=e.spike_regularization.beta(total_steps) if actor and e.model_type=='snn' else None,
            weighted_spike_regularization=values[9] if actor and e.model_type=='snn' else None,
            actor_grad_norm=values[11],sample_success_fraction=values[12],sample_failure_fraction=values[13],sample_near_goal_fraction=values[14],
            critic_q1_mean=values[15],critic_q1_std=values[16],critic_q1_max_abs=values[17],critic_target_q_mean=values[18],
            critic_td_error_mean=values[19],critic_failure_td_error_mean=values[20],
            critic_updated=True,critic_targets_updated=targets,actor_updated=actor)
        return m

    def metadata(self):
        return dict(requested=True,effective=bool(self.graphs),capture_owner='manual outer CUDA Graph; raw model tensor functions',
            capture_seconds=self.capture_seconds,initialization_seconds=self.initialization_seconds,
            replays=self.replays,memory=self.budget,
            measurement_replays=self.replays-self.measurement_replays_before,
            measurement_new_captures=len(self.graphs)-self.measurement_captures_before,
            event_seconds=dict(self.replay.seconds),event_submissions=self.replay.submissions,
            event_seconds_scope='CPU replay mutation, packing and upload submission wall time; includes original metadata work, not pure DMA time',
            cpu_gpu_handoffs=self.handoffs, handoff_scope='one packed control upload and one terminal read per update; event transfers separate; target noise remains a CUDA RNG launch',
            formal_capacity_budget=dict(self.budget) if self.e.replay.capacity==500000 else self.memory_budget(self.e,500000),
            random_stream='Primary NumPy float64 draws preserved; success partial Fisher-Yates replaces NumPy choice consumption. Torch target noise unchanged.',
            padding='Fixed zone capacity, masked; equality and overhead require server validation',
            exception_temporary_state='Device graph may compute otherwise unconsumed gradients and consume target noise before sampling/loss failure is returned. Persistent commits are gated; temporary gradients are not rolled back.',
            lifecycle='Call invalidate_gpu_update before direct model/parameter/device/optimizer mutation. Loads invalidate automatically.')

    def begin_measurement(self):
        self.measurement_replays_before=self.replays
        self.measurement_captures_before=len(self.graphs)
        self.handoffs=0
        self.replay.seconds={name:0. for name in self.replay.seconds}
        self.replay.submissions=0
