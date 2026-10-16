# V2 acceleration cleanup review

Pre-cleanup working-source snapshot: `E:\wurenji\code_snapshots\v2_pre_cleanup_20261016_135312`.
The `source/` directory includes the adopted uncommitted ellipsoid-cache default.
`manifest.json` records per-file hashes, HEAD, configuration and excluded untracked paths.
No models or experiment artifacts were copied. Keep the snapshot and manifest together.

Formal API `cpu_deduplication=True`; explicit False disables it.
Formal API `gpu_td3_compute=None` resolves once: supported CUDA ABI, fused Adam and
the existing full formal compilation configuration enable it. Explicit False stays off;
explicit True on unsupported configuration raises. Python callers with eager compilation
options remain on original updates with the reason recorded; CLI retains its existing
formal compilation defaults. No capture/runtime fallback is added.
CLI exposes positive/negative forms of these same two options.
Engine remains explicit bool. Fixed validation inherits the stage CPU option and keeps
noiseless inference and SNN reset. Diagnostic options remain explicit/default-off.
Execution configuration is separate from algorithm config and checkpoint schema.
Configuration selection and observed executor metadata are recorded separately;
an interrupted stage writes a small execution-error report when the run loop is active.

Removed: pinned staging, relation aggregation reorder, old single-observation fast path,
critic-tail graphs, merged critic-front/actor-objective compilation, their exclusive
entrypoints/tests, and three one-off capability/state probes with their exclusive tests.
Kept: original CPU replay/transfer/update, adopted compilation, GPUCompute, CPU dedup,
ellipsoid exact cache, pyramid combined solve, protection/restore/reset paths.
Deferred: window statistics, reusable buffers, nonblocking transfer, GPUUpdate/DeviceReplay.
GPUCompute still inherits real arithmetic and helpers from GPUUpdate; old diagnostic
modules still supply shared snapshot/flatten/input functions. No broad extraction was made.
AGENTS.md was preserved; its removed experimental flag descriptions now need separate review.

## Acceptance draft

After authorized synchronization of current source and the saved `source/` plus manifest:

```bash
BASELINE_SOURCE=/home/zj/projects/my_project/outputs/pre_cleanup_snapshot/source \
  bash docs/v2_acceleration_acceptance.sh
```

The script runs ANN then SNN serially, A then B. A compares pre-cleanup adopted options
with current default-equivalent explicit options. B switches GPUCompute and CPU dedup
OFF/ON; exact ellipsoid cache and the global pyramid solve stay ON on both sides.
It never reattributes these common improvements to the switch delta.
Correctness gates are separate:
- GPU computation retains the existing controlled same-initial-state/batch/noise/schedule
  lifecycle check, original rtol=1e-4/atol=1e-5, failure protection, recovery and checkpoint checks.
- CPU deduplication uses the SAME observation and actor for input-copy OFF/ON,
  restores identical RNG before inference and requires exact action and post-inference RNG.
  An evidence-only shadow environment uses the same reset and each same fixed action,
  requiring exact state, reward, termination, observation/mask, info and environment RNG.
  The second rollout replays the first rollout's recorded actions. Predicted actions after
  different TD3 paths, update counters and global training-RNG hashes are diagnostic evidence,
  not an exact cross-path trajectory gate. Fixed-action evidence is never throughput timing.
- A additionally compares the pre/post adopted GPUCompute snapshots directly using
  original tolerances, without new exceptions. Both workers export existing owned CPU
  snapshots via a small evidence wrapper; their engine imports follow each source revision.
  No update formulas, error barriers or saved baseline source files are changed.
  Shared fixed-action rollout evidence also checks pre/post environment regression.
- The formal-default wiring test actually calls CLI -> stage -> engine builder/trainer
  construction with both optimization parameters omitted, supported CUDA ABI mocked;
  explicit OFF and CPU are checked separately. A's explicit flags alone do not prove defaults.

Reports retain independent `strict_passed` and `engineering_accepted`, and record
`acceptance_basis` / `timing_acceptance_basis`. Default `--acceptance-basis strict`
blocks timing on any strict failure. ANN's historical gradient excess remains a strict
failure with prior engineering acceptance only; it is not converted into a strict pass.
Explicit `--acceptance-basis ann-existing-engineering` currently BLOCKS with a saved
explanation: no complete machine-verifiable accepted-field/state/evidence contract exists
in this driver, so known accepted deviations cannot reliably be separated from new failures.
No model-name exemption or relaxed tolerance is implemented. This prevents ANN timing
until that evidence gap is reviewed and resolved. Other serial jobs can still report.

BC/pool hashes, seed7, P1off/P2/P3on, 500k schedule, smooth-final offset75000, batch64,
128 warmup / 256 measured / 2 repeats are fixed. Initial compile/capture/warmup cost is
separate from measured time. New measured graphs/captures invalidate speed conclusions.
Full-step time includes final completion/readback; nested times must not be added.
Detailed environment instrumentation is a final separate worker, never throughput evidence.
Small summaries/errors/logs are under review/, detailed tensors under other/.
The three-scene pool does not cover all five geometries or long-run stability.
No server/CUDA validation has been performed for this cleanup.
