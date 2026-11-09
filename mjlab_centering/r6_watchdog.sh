#!/usr/bin/env bash
# Restart a keeper whose hunter has gone silent. The orchestrator prints a status line every poll (20 s), so a log that
# has not moved for 15 min means the process is stuck even if systemd still calls the unit "active" (it cannot see a hang).
set -u
cd "$(dirname "$0")"
check() {
  local unit=$1 log=$2
  [ -f "$log" ] || return 0
  local age=$(( $(date +%s) - $(stat -c %Y "$log") ))
  if [ "$age" -gt 900 ]; then
    echo "[watchdog] $(date -u +%H:%M:%S) $unit log stale ${age}s; restarting"
    systemctl --user restart "$unit"
  fi
}
check r6-width-after.service results/r6_width_after.log
check r6-pilot-after.service results/r6_pilot_after.log
check r7.service results/orch_r7.log
