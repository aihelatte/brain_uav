#!/usr/bin/env bash
# Plan only unless --launch; never changes Git or starts tmux.
uav_mc_cuda() {
  local mode="" manifest="" root start rc
  case "$#:$1" in
    1:*) manifest="$1" ;;
    2:--launch) mode="--launch"; manifest="$2" ;;
    *) printf 'usage: bash scripts/validate_v2_mc_return_cuda.sh [--launch] SOURCE_MANIFEST\n' >&2; return 2 ;;
  esac
  cd /home/zj/projects/my_project || return 1
  test -f "$manifest" || { printf 'Missing source manifest: %s\n' "$manifest" >&2; return 1; }
  root="outputs/v2_mc_return_cuda_$(date +%Y%m%d_%H%M%S_%N)"
  if [ -n "$mode" ]; then
    mkdir -p "$root/review" || return 1
  else
    root="outputs/v2_mc_return_cuda_PLAN"
  fi
  local args=(--model snn --seed 7 --batch-size 64
    --source-manifest "$manifest"
    --bc-checkpoint outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt
    --expected-bc-sha256 7b2b266fdf32cdb8b4eb220ef26a45008717a0af1538f8d478019f37ddc9adbc
    --validation-pool outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json
    --expected-pool-digest 2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b
    --output-dir "$root/other")
  if [ -z "$mode" ]; then
    PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u -B -m brain_uav.scripts.diagnose_v2_mc_return_cuda "${args[@]}"
    return "$?"
  fi
  start=$(date +%s)
  PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u -B -m brain_uav.scripts.diagnose_v2_mc_return_cuda "${args[@]}" --launch 2>&1 | tee "$root/review/driver.log"
  rc=${PIPESTATUS[0]}
  printf 'exit_code=%s\nprocess_wall_seconds=%s\noutput_root=%s\n' "$rc" "$(( $(date +%s)-start ))" "$root" | tee -a "$root/review/driver.log"
  if [ -f "$root/other/summary.json" ]; then
    cp -- "$root/other/summary.json" "$root/review/summary.json" || return 1
  fi
  return "$rc"
}
uav_mc_cuda "$@"
