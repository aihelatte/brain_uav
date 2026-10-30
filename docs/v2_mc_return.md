# Mean TD + recent behavior-return auxiliary supervision

Opt-in API/CLI settings (single stage and curriculum):

```text
target_q_aggregation='mean', mc_return_weight=0.1, mc_return_max_age_steps=100000
--target-q-aggregation mean --mc-return-weight 0.1 --mc-return-max-age-steps 100000
```

Defaults remain `min`, weight `0.0`, age `100000`. Positive MC weight requires
explicit `mean`; legacy `gpu_td3_update` is rejected. Formal GPUCompute remains
supported. This option does not enable any transfer or MPS experiment.

Only naturally completed exploratory training episodes supply labels:
`G[t] = reward[t] + gamma * G[t+1]`, with the last return equal to its reward.
The original replay-write → update → episode-finalization order is retained.
Consequently, labels filled at finalization first become available to later
updates. Budget-truncated unfinished episodes never receive complete labels.
These are realized exploratory behavior returns, not exact Q labels for the
current deterministic actor. No validation trajectories supervise training.

The main ring checks `(slot, write_id)` before filling. Success copies use the
episode's own G and original one-based stage step even when primary slots were
overwritten. Scalar return/valid/collection-step arrays are allocated only when
enabled; transitions, sampling weights and RNG calls are unchanged.

Each sample computes its mask on CPU afresh:
`valid AND 0 <= current_stage_step - collected_stage_step <= max_age_steps`.
Expired samples remain in the full TD loss. Returns and mask are runtime FP32
`[B,1]` batch inputs, including GPUCompute clone/upload/fixed-input copies; label
values do not enter the graph cache key. Weight and age join static binding checks.

`TD = MSE(Q1,y) + MSE(Q2,y)`;
`MC = sum(mask * ((Q1-G)^2 + (Q2-G)^2)) / max(sum(mask),1)`;
`critic_loss = TD + weight * MC`.
Both terms reuse the same batch and Q forwards, with one original backward,
joint clipping and Adam step. Inactive residuals are selected out before squaring.
No-valid-label MC is exactly zero. Weight zero uses the original loss/data path.

Update metrics and enabled episode records distinguish `critic_td_loss`,
`critic_mc_loss` (unweighted), `critic_mc_weighted_loss`, `critic_loss` (actual
backward total), `mc_valid_count` and `mc_valid_fraction`. Episode quantities are
means across that episode's updates; no-update episodes use null MC statistics.
Execution reports distinguish requested/configured values from observed updates
and retain existing GPUCompute replay evidence and real initial-state hashes.

Algorithm/formal configs persist both settings in existing checkpoint/snapshot
formats. Missing historical fields mean weight zero / age 100000. Strict loaders
reject invalid settings and formal/engine conflicts. Network-only stage
initialization retains its existing distinction from strict engine loading;
this does not add full replay/environment/RNG training resume support.

Local checks: `PYTHONPATH=src:tests python -m unittest test_v2_mc_return -v`
(Windows uses `src;tests`). CPU/mock and eager-backend compilation checks are
not CUDA Graph, server performance or training acceptance. Real CUDA correctness,
runtime mask updates across graph replays and training overhead remain untested.
