#!/usr/bin/env bash
#
# ADDED=... Custom path to added.txt (legacy, ignored in --config mode)
#

CRAWL_BASE="texts"
ARCH="arch"
export CRAWL_BASE
source ./.venv/bin/activate

declare -a args
if [ "${1:-none}" = none ]; then
  args=()
else
  args=( "-a" "only=$1" )
fi

if [ "${1:-none}" != nocrawl ]; then
  scrapy crawl news "${args[@]}" || echo "Ooops! ($?)"
fi

./ai-summary.py --config config.yaml --archive "$ARCH"
