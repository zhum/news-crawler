#!/usr/bin/env bash
#

CRAWL_BASE="texts"
ARCH="arch"
YQ=/usr/bin/yq
export CRAWL_BASE

source ./venv/bin/activate

# Read the Troitsky Variant group name from config (second group, index 1)
if command -v $YQ &>/dev/null; then
    GROUP=$($YQ -r '.groups[1].name' config.yaml)
else
    GROUP=$(python3 -c "
import yaml
with open('config.yaml') as f:
    c = yaml.safe_load(f)
print(c.get('groups', [])[1].get('name', 'Troitsky Variant'))
")
fi

if [ "${1:-none}" != nocrawl ]; then
    # Archive added_file before crawl so the spider starts fresh
    if command -v $YQ &>/dev/null; then
        ADDED_FILE=$($YQ -r '.groups[1].output.added_file' config.yaml)
    else
        ADDED_FILE=$(python3 -c "
import yaml
with open('config.yaml') as f:
    c = yaml.safe_load(f)
print(c.get('groups', [])[1].get('output', {}).get('added_file', 'added-tr.txt'))
")
    fi
    [ -f "$ADDED_FILE" ] && mv "$ADDED_FILE" "$ARCH/${ADDED_FILE}-$(date +%F)" 2>/dev/null || true

    scrapy crawl news -a "group=$GROUP" || echo "Ooops! ($?)"
fi

./ai-summary.py --config config.yaml --archive "$ARCH" --group "$GROUP"
