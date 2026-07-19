#!/usr/bin/env bash
#
# run_weekly_digest.sh — cron entry point for news-crawler.
#
# What it does each run:
#   1. Runs crawl.sh for all configured groups (HPC News, Troitsky Variant, ...)
#   2. crawl.sh itself exports digests/YYYY-MM-DD-hpc-digest.md and uploads it
#      to Drive via rclone (this was added to crawl.sh separately)
#   3. Fetches/processes the arXiv digest
#   4. Runs digest processing script
#   5. Logs everything with timestamps, refuses to run if a previous run is
#      still active (or crashed and left a stale lock)
#
# --- One-time setup ---
#   1. Edit REPO_DIR below to the actual path on this machine.
#   2. Make sure `rclone config` has been run once, with a remote named
#      "gdrive" pointing at your Google Drive.
#   3. Make sure the Python venv/deps from requirements.txt are installed
#      (adjust the VENV_DIR line below, or remove that block if you install
#      deps globally instead of in a venv).
#   4. chmod +x run_weekly_digest.sh crawl.sh export_digest.py
#   5. Add to crontab (`crontab -e`):
#        0 6 * * MON  /path/to/news-crawler/run_weekly_digest.sh
#      (runs Monday 6am; pick whatever day/time suits — just make sure it's
#      BEFORE whenever you plan to ask Claude to generate that week's ideas)

set -euo pipefail

REPO_DIR="/path/to/news-crawler"          # Base dir
VENV_DIR="$REPO_DIR/.venv"                # remove it if not using a venv
ARXIV_MAIL_DIR="/path/to/arxiv/eml"       # path to the dir with arXiv eml files
ARXIV_BACK_DIR="/path/to/backups"         # path to directory, which holds backup files
LOCK_FILE="/tmp/hpc-news-crawler.lock"
LOG_DIR="$REPO_DIR/cron-logs"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%F-%H%M%S).log"

# Send all output (stdout+stderr) to both the terminal (useful when testing
# by hand) and a timestamped log file (useful for cron, where output is easy
# to lose track of).
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== $(date -Is) starting weekly HPC crawl ==="

# --- Lock so overlapping cron runs can't stomp on each other ---
if [ -e "$LOCK_FILE" ]; then
    echo "Lock file $LOCK_FILE already exists — a previous run may still be" \
         "active, or crashed without cleaning up. Exiting without running."
    echo "If you're sure nothing else is running, delete the lock file and retry."
    exit 1
fi
trap 'rm -f "$LOCK_FILE"; echo "=== $(date -Is) finished (exit $?) ==="' EXIT
touch "$LOCK_FILE"

cd "$REPO_DIR"

# --- Activate venv if present ---
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# --- Run the crawl (this also exports + uploads the digest, see crawl.sh) ---
if ./crawl.sh all; then
    echo "crawl.sh completed successfully"
else
    status=$?
    echo "crawl.sh FAILED with exit code $status"
    exit "$status"
fi

if ./arxiv.py --output arxiv_summaries.json $ARXIV_MAIL_DIR; then
    echo "arXiv news updated"
    mv "$ARXIV_MAIL_DIR/*.eml" "$ARXIV_BACK_DIR"
else
    status=$?
    echo "arxiv.py FAILED with exit code $status"
    exit "$status"
fi

# Run digest processing
./export_digest.py --config config.yaml --arxiv arxiv_summaries.json --min-rate 6

# --- Prune old logs so cron-logs/ doesn't grow forever ---
find "$LOG_DIR" -name '*.log' -mtime +60 -delete

echo "=== $(date -Is) done ==="
