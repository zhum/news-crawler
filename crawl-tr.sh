#!/usr/bin/env bash
#

CRAWL_BASE="texts"
ARCH="arch"
export CRAWL_BASE

source ./.venv/bin/activate

# Read the Troitsky Variant group name from config (second group, index 1)
if command -v yq &>/dev/null; then
    GROUP=$(yq -r '.groups[1].name' config.yaml)
else
    GROUP=$(python3 -c "
import yaml
with open('config.yaml') as f:
    c = yaml.safe_load(f)
print(c.get('groups', [])[1].get('name', 'Troitsky Variant'))
")
fi

if [ "${1:-none}" != nocrawl ]; then
  scrapy crawl news -a "group=$GROUP" || echo "Ooops! ($?)"
fi

./ai-summary.py --config config.yaml --archive "$ARCH" --group "$GROUP"
