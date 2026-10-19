#!/bin/bash
# Sep 20 launcher daemon: waits for 00:01:30 UTC then fires fire_all_sep20.py (all queues).
set -u
LOG=/home/z/my-project/fire_all_sep20.log
TARGET="2026-09-20 00:01:30"
echo "[$(date -u '+%F %T')] daemon armed, waiting for $TARGET UTC" >> "$LOG"

while true; do
  NOW=$(date -u '+%Y-%m-%d %H:%M:%S')
  if [[ "$NOW" > "$TARGET" || "$NOW" == "$TARGET" ]]; then
    break
  fi
  sleep 30
done

echo "[$(date -u '+%F %T')] window reached — firing all queues" >> "$LOG"
/home/z/.venv/bin/python /home/z/my-project/scripts/fire_all_sep20.py all >> "$LOG" 2>&1
echo "[$(date -u '+%F %T')] launcher exited" >> "$LOG"
