#!/usr/bin/env bash
#
# run_weekly_digest.sh — cron entry point for news-crawler.
#
# What it does each run:
#   1. Runs crawl.sh for all configured groups (HPC News, Troitsky Variant, ...)
#   2. crawl.sh itself exports digests/YYYY-MM-DD-hpc-digest.md and uploads it
#      to Drive via rclone (this was added to crawl.sh separately)
#   3. Caches the two Drive-only sources (episode index sheet, outline doc)
#      to drive-cache/ locally via rclone export
#   4. Runs generate_digest_email.py, which uses the Anthropic API for the
#      dedup/angle judgment call and the Gmail API to create a draft —
#      fully unattended, no Claude session or approval prompts needed
#   5. Logs everything with timestamps, refuses to run if a previous run is
#      still active (or crashed and left a stale lock)
#
# What it does NOT do:
#   - Fetch/process the arXiv digest. That's still a manual step: paste the
#     digest email into a text file and run
#       ./arxiv.py <file> --output arxiv_summaries.json
#       ./export_digest.py --config config.yaml --arxiv arxiv_summaries.json --min-rate 6
#     whenever you have one. It will merge into (and re-upload) that day's
#     digest file, whether you run it before or after this script.
#   - Send the email. generate_digest_email.py only ever creates a Gmail
#     draft (gmail.compose scope) — review and send happens by hand.
#
# --- One-time setup ---
#   1. Edit REPO_DIR below to the actual path on this machine.
#   2. Make sure `rclone config` has been run once, with a remote named
#      "gdrive" pointing at your Google Drive.
#   3. Make sure the Python venv/deps are installed:
#        pip install scrapy markdownify rich pyyaml feedparser \
#          anthropic google-auth google-auth-oauthlib google-api-python-client
#      (adjust the VENV_DIR line below, or remove that block if you install
#      deps globally instead of in a venv).
#   4. Run ./gmail_oauth_setup.py once, interactively, to create
#      gmail_token.json (see that file's docstring for the one-time Google
#      Cloud Console steps needed first).
#   5. Set ANTHROPIC_API_KEY in the environment cron will actually run in —
#      crontab doesn't inherit your interactive shell's env vars, so either
#      add an `export ANTHROPIC_API_KEY=...` line directly into this script
#      (below, before the crawl step), or set it in /etc/environment, or
#      source a separate env file at the top of this script. Whichever you
#      pick, don't commit the key itself to the repo.
#   6. chmod +x run_weekly_digest.sh crawl.sh export_digest.py generate_digest_email.py
#   7. Add to crontab (`crontab -e`):
#        0 6 * * MON  /path/to/news-crawler/run_weekly_digest.sh
#      (runs Monday 6am; pick whatever day/time suits)
#
# Note on the rclone export step below: I haven't been able to verify the
# exact output filename rclone produces against a live Drive account from
# here — the mv lines are my best understanding of rclone's --drive-export-
# formats naming convention, but rclone versions can differ. First real run,
# check `ls drive-cache/` afterward and adjust the mv lines if the actual
# filename doesn't match what's expected.
# ./generate_digest_email.py  --digest-dir digests --backend "local" --local-model localhost:8090 --sheet-csv drive-cache/episode-sheet.csv --outline-txt drive-cache/outline.txt --to you@example.com --send --mail-method smtp --smtp-host smtp.example.com --smtp-port 587 --smtp-user you --smtp-from you@example.com

set -euo pipefail

REPO_DIR="$HOME/projects/crawler"    # Base dir
# SMTP_PASSWORD (and any other secrets) live in .env, kept out of git
set -a; source "$REPO_DIR/.env"; set +a

LOCK_FILE="/tmp/hpc-news-crawler.lock"
LOG_DIR="$REPO_DIR/cron-logs"
VENV_DIR="$REPO_DIR/venv"                 # remove it if not using a venv
ARXIV_MAIL_DIR="$HOME/tmp"           # path to the dir with arXiv eml files
ARXIV_BACK_DIR="$HOME/tmp/back"
LLAMA_MODEL="$HOME/projects/GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
LLAMA_MODEL="$HOME/.cache/llama.cpp/unsloth_Qwen3.6-35B-A3B-GGUF_Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
MODEL="--local-model localhost:8090"
LOCAL_MAIL_TO="$(id -un)"                      # local user for mail/mailx fallback if SMTP is down
SEND="--send --mail-method smtp --smtp-host smtp.example.com --smtp-port 587 --smtp-user you --smtp-from you@example.com --local-mail-to $LOCAL_MAIL_TO"
# use emprty to draft email, not send it

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%F-%H%M%S).log"
LLAMA_LOG="$LOG_DIR/llama-$(date +%F-%H%M%S).log"

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

cleanup() {
  local status=$?
  rm -f "$LOCK_FILE"
  if [[ ${LLAMA_PID:-0} != 0 ]]; then
    kill $LLAMA_PID 2>/dev/null || true
  fi
  echo "=== $(date -Is) finished (exit $status) ==="
}

trap cleanup EXIT QUIT
touch "$LOCK_FILE"

arxiv() {
  if ./arxiv.py ${MODEL} --output arxiv_summaries.json $ARXIV_MAIL_DIR; then
    echo "arXiv news updated"
    mv "$ARXIV_MAIL_DIR"/*.eml "$ARXIV_BACK_DIR" 2>/dev/null || true
  else
    status=$?
    echo "arXiv processing failed: $status"
    exit "$status"
  fi
}

cd "$REPO_DIR"

# --- Activate venv if present ---
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

export DOWNLOAD_TIMEOUT=30

# --- Start the Qwen llama.cpp server up front: ai-summary.py (via crawl.sh,
# config.yaml ai.model = localhost:8090), arxiv.py and, without an Anthropic
# key, generate_digest_email.py all use it. cleanup() stops it on exit.
echo -n "Waiting for llama server to start"
llama-server --model ${LLAMA_MODEL} \
  -fa on -ctk q4_1 -ctv q4_1 \
  -c 64000 --port 8090 >$LLAMA_LOG 2>&1 & LLAMA_PID=$!
export LLAMA_PID  # generate_digest_email.py stops it before sending mail
while sleep 5; do
  echo -n "."
  if grep -q 'starting the main loop' "$LLAMA_LOG"; then
    echo " started!"
    break
  fi
  if ! kill -0 $LLAMA_PID 2>/dev/null; then
    echo " llama.cpp start failed, see $LLAMA_LOG"
    LLAMA_PID=0
    exit 1
  fi
done

if [ "${NOCRAWL:-0}" == "0" ]; then
  # --- Run the crawl (this also exports + uploads the digest, see crawl.sh) ---
  if ./crawl.sh all; then
      echo "crawl.sh completed successfully"
  else
      status=$?
      echo "crawl.sh FAILED with exit code $status"
      exit "$status"
  fi
fi

# --- Suggest config.yaml fixes for selectors broken by site redesigns
# (uses the llama server started above; must run before generate_digest_email.py stops it)
echo "=== Selector fix suggestions ==="
./fix-selectors.py --config config.yaml || echo "fix-selectors.py failed ($?)"

# --- Cache the two Drive-only sources (sheet + outline doc) locally.
# Reuses the same rclone remote already configured for digest upload.
# --drive-export-formats converts native Google Sheets/Docs to plain
# csv/txt on download (rclone appends the export extension itself —
# don't add .gsheet/.gdoc to the source path, that's not a real rclone
# convention, just reference the file by its plain Drive name).
mkdir -p drive-cache
if command -v rclone &>/dev/null; then
    rclone copy --drive-export-formats csv \
        "gdrive:15_HPC podcast/HPC Book podcasts plan sheet.csv" drive-cache/ --update
    mv "drive-cache/HPC Book podcasts plan sheet.csv" drive-cache/episode-sheet.csv 2>/dev/null || true

    rclone copy --drive-export-formats txt \
        "gdrive:15_HPC podcast/HPC Book podcasts plan.txt" drive-cache/ --update
    mv "drive-cache/HPC Book podcasts plan.txt" drive-cache/outline.txt 2>/dev/null || true
else
    echo "rclone not found — cannot refresh drive-cache/, using whatever is already there" >&2
fi

echo "==============================================================="
# --- Generate the episode-ideas email via the Anthropic + Gmail APIs,
# fully unattended (no Claude session / Cowork approval needed).
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    arxiv
    ./generate_digest_email.py \
        --digest-dir digests \
        --backend "local" \
        ${MODEL} \
        --sheet-csv drive-cache/episode-sheet.csv \
        --outline-txt drive-cache/outline.txt \
        $SEND --to you@example.com
else
    arxiv
    ./generate_digest_email.py \
        --digest-dir digests \
        --sheet-csv drive-cache/episode-sheet.csv \
        --outline-txt drive-cache/outline.txt \
        $SEND --to you@example.com
fi

# --- Prune old logs so cron-logs/ doesn't grow forever ---
find "$LOG_DIR" -name '*.log' -mtime +60 -delete

echo "=== $(date -Is) done ==="

## Run digest processing
#./export_digest.py --config config.yaml --arxiv arxiv_summaries.json --min-rate 6

