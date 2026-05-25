#!/usr/bin/env bash
#

CRAWL_BASE="texts"
ARCH="arch"
export CRAWL_BASE

source ./.venv/bin/activate

if [ "${1:-none}" != nocrawl ]; then
  scrapy crawl news -a "only=Troitsky variant" || echo "Ooops! ($?)"
fi

./ai-summary.py --config config.yaml --archive "$ARCH" --group "Troitsky Variant"
