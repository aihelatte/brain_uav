#!/usr/bin/env bash
# Strict gates only: CPU same-input/fixed-action; GPU controlled lifecycle; A direct cross-version snapshots.
# Historical ANN engineering selection remains blocked pending a verifiable accepted-scope contract.
# Review-only draft. Run only after code/saved-source synchronization is authorized.
set -u
set -o pipefail
cd /home/zj/projects/my_project || exit 1
: "${BASELINE_SOURCE:?Set to the synchronized pre-cleanup snapshot source directory}"
test -f "$BASELINE_SOURCE/../manifest.json" || exit 1
root="outputs/v2_acceleration_acceptance_$(date +%Y%m%d_%H%M%S_%N)"
mkdir -p "$root"
failed=0
for model in ann snn; do
  if [[ "$model" == ann ]]; then
    bc=outputs/v2_projected_head_round1/ann_20261012_102839/bc/bc_v2_ann_best.pt
  else
    bc=outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt
  fi
  for comparison in A B; do
    job="$root/${model}_${comparison}"
    extra=()
    if [[ "$comparison" == A ]]; then extra=(--baseline-source "$BASELINE_SOURCE"); fi
    PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u \
      -m brain_uav.scripts.v2_cpu_dedup_diagnostic \
      --acceptance-comparison "$comparison" --acceptance-basis strict "${extra[@]}" \
      --model "$model" --device cuda --gpu-td3-compute \
      --gpu-compute-environment-diagnostic \
      --bc-checkpoint "$bc" \
      --validation-pool outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json \
      --expected-pool-digest 2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b \
      --validation-seed 20260904 --seed 7 --batch-size 64 --replay-capacity 500000 \
      --scenario-count 3 --snn-time-window 4 \
      --comparison-bc-schedule smooth-final --controlled-stage-step-offset 75000 \
      --warmup-steps 128 --steps-per-level 256 --repeats 2 --output-dir "$job" \
      2>&1 | tee "$job.log"
    rc=${PIPESTATUS[0]}
    printf 'exit_code=%s\n' "$rc" | tee -a "$job.log"
    mkdir -p "$job/review"
    mv -- "$job.log" "$job/review/driver.log"
    if ((rc != 0)); then failed=1; fi
  done
done
exit "$failed"
