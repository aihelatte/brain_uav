#!/usr/bin/env bash
# No global shell options; all commands run in this script process.
if [[ "$1" == "--worker-run" ]]; then
  root="$2"; job="$3"; py="$4"; timeout="$5"; seed="${6:-23}"
  repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
  cd -- "$repo" || exit 1
  mkdir -p -- "$root/$job/review" || exit 1
  start=$SECONDS
  PYTHONPATH=src "$py" -u -B -m brain_uav.scripts.run_v2_online_wd3_acc worker \
    --root "$root" --job "$job" --timeout "$timeout" --seed "$seed" --launch 2>&1 | tee "$root/$job/review/driver.log"
  rc=${PIPESTATUS[0]}
  printf '{"exit_code":%s,"process_wall_seconds":%s}\n' "$rc" "$((SECONDS-start))" > "$root/$job/review/process.json"
  printf 'exit_code=%s process_wall_seconds=%s logs=%s\n' "$rc" "$((SECONDS-start))" "$root/$job/review"
  # Keep the result visible without changing global/session tmux options.
  sleep infinity
else
  repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
  cd -- "$repo" || exit 1
  PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u -B -m brain_uav.scripts.run_v2_online_wd3_acc "$@"
fi
