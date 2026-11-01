#!/usr/bin/env bash
# Fixed-actor critic diagnostic only. No formal training or curriculum promotion.
repo=/home/zj/projects/my_project
python_bin=/home/zj/envs/uav_torch/bin/python
required=67b9a64
pool="$repo/outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json"
history="$repo/outputs/v2_critic_reliability_round1/20261026_171846_464722553"
ann_checkpoint="$repo/outputs/v2_ann_critic_lr_round2/20261020_221413/ann_clr_2e-4_seed23/v2_ann_td3_easy_failed.pt"
snn_checkpoint="$repo/outputs/v2_snn_critic_lr_confirm_round1/20261024_102749/snn_clr_1e-3_seed23/v2_snn_td3_easy_failed.pt"
socket="${V2_TMUX_SOCKET:-${TMUX%%,*}}"

tmux_do() {
  local -a options=()
  if [ -n "$socket" ]; then options=(-S "$socket"); fi
  env -u TMUX -u TMUX_PANE tmux "${options[@]}" "$@"
}

preflight() {
  cd "$repo" || return 1
  git merge-base --is-ancestor "$required" HEAD || return 1
  command -v tmux >/dev/null || return 1
  test -x "$python_bin" || return 1
  sha256sum -c <<HASHES
c48c94a8fa1d7a2ca2a833cfcf9b103e61655cf089c140237d1ded886449e8af  $ann_checkpoint
c5cb20f60c8181fa03bc1fcd31901cfeae218b1eecd414b186abd15f5710f36d  $snn_checkpoint
ebf605a59e1e88590d1fe8d7ab39eabb190e097085a20b8a4c38e5e890a4edcc  $pool
743421e4789f5826b34517f289dc3922afec9ad0fe6a2f8917a5efca278ccb45  $history/manifest.json
f079b141b1e78125f70a4d4ebd3ec9354878405d70a88ac37b49ab9ff86e684d  $history/baseline_episodes.jsonl
HASHES
  local rc=$?
  test "$rc" -eq 0 || return "$rc"
  PYTHONPATH=src "$python_bin" -u -B - "$pool" "$ann_checkpoint" "$snn_checkpoint" <<'PREFLIGHT'
import argparse, sys, torch
from pathlib import Path
from brain_uav.trainers.v2_validation import load_v2_validation_pool
from brain_uav.scripts.diagnose_v2_wd3_acc import source_data
pool_path, ann, snn = map(Path, sys.argv[1:])
pool = load_v2_validation_pool(pool_path, expected_level='easy', expected_count=100,
                               expected_master_seed=20260904)
if pool.content_digest != '2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b':
    raise RuntimeError('Pool content digest mismatch; nothing launched.')
for model, checkpoint in [('ann', ann), ('snn', snn)]:
    source_data(argparse.Namespace(model=model, checkpoint=checkpoint,
                                  validation_pool=pool_path, device='cpu'))
    print('Strict source reconstruction OK:', model, flush=True)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable; nothing launched.')
if torch.ones(1, device='cuda').sum().item() != 1:
    raise RuntimeError('CUDA allocation check failed; nothing launched.')
print('Preflight OK. CUDA:', torch.cuda.get_device_name(0), flush=True)
PREFLIGHT
}

await_gate() {
  local root="$1" gate="$2"
  while true; do
    if [ -f "$root/ABORT" ]; then echo 'Batch aborted; no new work started.'; return 1; fi
    if [ -f "$root/$gate" ]; then return 0; fi
    sleep 1
  done
}

run_phase() {
  local root="$1" model="$2" phase="$3" arm="${4:-}" checkpoint out begin finished rc
  local -a extra=()
  checkpoint="$ann_checkpoint"
  if [ "$model" = snn ]; then checkpoint="$snn_checkpoint"; fi
  out="$root/${model}_${phase}"
  case "$phase" in
    collect) if [ "$model" = snn ]; then extra=(--score-history "$history"); fi ;;
    prepare) extra=(--independent-history "$root/${model}_collect/other") ;;
    fit) out="$root/${model}_${arm}"; extra=(--prepared-dir "$root/${model}_prepare/other" --arm "$arm") ;;
    *) return 2 ;;
  esac
  mkdir -p "$out/review" || return 1
  begin=$(date +%s)
  PYTHONPATH=src "$python_bin" -u -B -m brain_uav.scripts.diagnose_v2_wd3_acc \
    --mode "$phase" --model "$model" --checkpoint "$checkpoint" --validation-pool "$pool" \
    --updates 10000 --device cuda --output-dir "$out/other" "${extra[@]}" --launch \
    2>&1 | tee "$out/review/driver.log"
  rc=${PIPESTATUS[0]}
  finished=$(date +%s)
  printf 'exit_code=%s\nprocess_wall_seconds=%s\noutput_dir=%s\n' \
    "$rc" "$((finished-begin))" "$out" | tee "$out/review/timing.txt" | tee -a "$out/review/driver.log"
  local name
  for name in manifest.json summary.json evidence.json; do
    if [ -f "$out/other/$name" ]; then cp "$out/other/$name" "$out/review/$name" || return 1; fi
  done
  return "$rc"
}

prepare_worker() {
  local root="$1" model="$2" rc
  cd "$repo" || { printf '1\n' > "$root/DONE_prepare_$model"; touch "$root/ABORT"; return 1; }
  await_gate "$root" PREP_GO || return 1
  run_phase "$root" "$model" collect
  rc=$?
  if [ "$rc" -eq 0 ]; then run_phase "$root" "$model" prepare; rc=$?; fi
  printf '%s\n' "$rc" > "$root/DONE_prepare_$model"
  if [ "$rc" -ne 0 ]; then touch "$root/ABORT"; fi
  return "$rc"
}

fit_worker() {
  local root="$1" model="$2" arm="$3" rc
  cd "$repo" || { printf '1\n' > "$root/DONE_${model}_$arm"; touch "$root/ABORT"; return 1; }
  await_gate "$root" FIT_GO || return 1
  run_phase "$root" "$model" fit "$arm"
  rc=$?
  printf '%s\n' "$rc" > "$root/DONE_${model}_$arm"
  return "$rc"
}

supervise() {
  local root="$1" session="$2" cmd model arm name failed=0
  cd "$repo" || return 1
  tmux_do set-window-option -t "$session:supervisor" remain-on-exit on || { touch "$root/ABORT"; return 1; }
  for model in ann snn; do
    printf -v cmd '%q ' env "V2_TMUX_SOCKET=$socket" bash "$root/launcher.sh" --prepare "$root" "$model"
    tmux_do new-window -d -t "$session:" -n "prep_$model" "$cmd" || { touch "$root/ABORT"; return 1; }
    tmux_do set-window-option -t "$session:prep_$model" remain-on-exit on || { touch "$root/ABORT"; return 1; }
  done
  for model in ann snn; do
    for arm in min mean wd3 acc; do
      name="${model}_$arm"
      printf -v cmd '%q ' env "V2_TMUX_SOCKET=$socket" bash "$root/launcher.sh" --fit "$root" "$model" "$arm"
      tmux_do new-window -d -t "$session:" -n "$name" "$cmd" || { touch "$root/ABORT"; return 1; }
      tmux_do set-window-option -t "$session:$name" remain-on-exit on || { touch "$root/ABORT"; return 1; }
    done
  done
  touch "$root/PREP_GO" || { touch "$root/ABORT"; return 1; }
  echo 'Started ANN/SNN collection and preparation concurrently; fits are waiting.'
  while [ ! -f "$root/DONE_prepare_ann" ] || [ ! -f "$root/DONE_prepare_snn" ]; do
    if [ -f "$root/ABORT" ]; then echo 'Preparation failed; fits were not released.'; return 1; fi
    sleep 2
  done
  if [ "$(cat "$root/DONE_prepare_ann")" != 0 ] || [ "$(cat "$root/DONE_prepare_snn")" != 0 ]; then
    touch "$root/ABORT"; echo 'Preparation failed; fits were not released.'; return 1
  fi
  touch "$root/FIT_GO" || { touch "$root/ABORT"; return 1; }
  echo 'Released 8/8 independent critic fits. 10000 updates each; actor stays frozen.'
  local done_count
  while true; do
    if [ -f "$root/ABORT" ]; then echo 'Batch aborted while waiting for fits; inspect worker logs.'; return 1; fi
    done_count=0
    for model in ann snn; do
      for arm in min mean wd3 acc; do
        if [ -f "$root/DONE_${model}_$arm" ]; then done_count=$((done_count+1)); fi
      done
    done
    if [ "$done_count" -eq 8 ]; then break; fi
    sleep 5
  done
  for model in ann snn; do
    for arm in min mean wd3 acc; do
      name="${model}_$arm"
      if [ "$(cat "$root/DONE_$name")" != 0 ]; then failed=$((failed+1)); fi
    done
  done
  printf 'completed_fits=8\nfailed_fits=%s\noutput_root=%s\n' "$failed" "$root" | tee "$root/batch_status.txt"
  return "$failed"
}

launch() {
  local session=uav_wd3_acc_r1 root cmd
  cd "$repo" || return 1
  if tmux_do has-session -t "$session" 2>/dev/null; then
    echo "Session already exists: $session. Nothing new started."; return 1
  fi
  preflight || return 1
  root="$repo/outputs/v2_wd3_acc_round1/$(date +%Y%m%d_%H%M%S_%N)"
  mkdir -p "$root" || return 1
  cp "${BASH_SOURCE[0]}" "$root/launcher.sh" || return 1
  git rev-parse HEAD > "$root/git_head.txt" || return 1
  git status --short > "$root/git_status.txt" || return 1
  printf -v cmd '%q ' env "V2_TMUX_SOCKET=$socket" bash "$root/launcher.sh" --supervise "$root" "$session"
  tmux_do new-session -d -s "$session" -n supervisor "$cmd" || return 1
  printf 'Supervisor started.\noutput_root=%s\nsession=%s\n' "$root" "$session"
  printf 'Inside tmux: tmux '; if [ -n "$socket" ]; then printf -- '-S %q ' "$socket"; fi
  printf 'switch-client -t %s\n' "$session"
  printf 'Outside tmux: tmux '; if [ -n "$socket" ]; then printf -- '-S %q ' "$socket"; fi
  printf 'attach -t %s\n' "$session"
}

case "${1:-}" in
  --launch) launch ;;
  --supervise) supervise "$2" "$3" ;;
  --prepare) prepare_worker "$2" "$3" ;;
  --fit) fit_worker "$2" "$3" "$4" ;;
  *) printf 'Plan only: ANN/SNN collect+prepare concurrently, then eight concurrent 10000-update critic fits.\nRun: bash %q --launch\n' "${BASH_SOURCE[0]}" ;;
esac
