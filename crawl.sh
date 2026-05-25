#!/usr/bin/env bash
#
# ADDED=... Custom path to added.txt
# 

CRAWL_BASE="texts"
ARCH="arch"
export CRAWL_BASE
source ./.venv/bin/activate
if [ -z "${ADDED}" ]; then ADDED="added.txt"; fi

if [ "${1:none}" != nocrawl ]; then

  declare -a CR=( news )

  mv "$ADDED" "${ARCH}/${ADDED}-$(date +%F)" 2>/dev/null || true
  mv "summary.json" "s${ARCH}/ummary.json-$(date +%F)" 2>/dev/null || true

  declare -a args
  if [ "${1:-none}" = none ]; then
    args=()
  else
    args=( "-a" "only=$1" )
  fi

  for cr in "${CR[@]}"; do
    scrapy crawl "$cr" "${args[@]}" || echo "Ooops! ($?)"
  done
fi

./ai-summary.py summary.json "$ADDED"

