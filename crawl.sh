#!/usr/bin/env bash
#
# Usage:
#   ./crawl.sh                  # interactive group selection
#   ./crawl.sh all              # all groups
#   ./crawl.sh "HPC News"       # one group
#   ./crawl.sh "HPC News,Troitsky Variant"  # subset (comma-separated)
#   ./crawl.sh nocrawl [group]  # skip crawl, run summary only
#

CRAWL_BASE="texts"
ARCH="arch"
export CRAWL_BASE
source ./.venv/bin/activate

# Read group names from config.yaml
mapfile -t ALL_GROUPS < <(python3 -c "
import yaml
with open('config.yaml') as f:
    c = yaml.safe_load(f)
for g in c.get('groups', []):
    print(g['name'])
")

NOCRAWL=false
if [ "${1:-none}" = "nocrawl" ]; then
    NOCRAWL=true
    shift
fi

declare -a SELECTED_GROUPS

if [ $# -gt 0 ]; then
    # Group(s) passed on command line
    if [ "$1" = "all" ]; then
        SELECTED_GROUPS=("${ALL_GROUPS[@]}")
    else
        IFS=',' read -ra SELECTED_GROUPS <<< "$1"
    fi
else
    # Interactive multi-select with bash select
    echo "Select groups to process (multiple allowed):"
    PS3="Add group: "
    while true; do
        select choice in "${ALL_GROUPS[@]}" "(all)" "(done)"; do
            case "$choice" in
                "(done)")
                    break 2
                    ;;
                "(all)")
                    SELECTED_GROUPS=("${ALL_GROUPS[@]}")
                    break 2
                    ;;
                "")
                    echo "Invalid selection." >&2
                    ;;
                *)
                    already=false
                    for s in "${SELECTED_GROUPS[@]}"; do
                        [[ "$s" == "$choice" ]] && already=true && break
                    done
                    if [ "$already" = false ]; then
                        SELECTED_GROUPS+=("$choice")
                    fi
                    echo "Selected: [${SELECTED_GROUPS[*]}]" >&2
                    ;;
            esac
            break
        done
    done
fi

if [ ${#SELECTED_GROUPS[@]} -eq 0 ]; then
    echo "No groups selected. Exiting." >&2
    exit 0
fi

echo "Groups: ${SELECTED_GROUPS[*]}"

# Build comma-separated list for the spider's -a group= parameter
GROUPS_CSV=$(IFS=','; echo "${SELECTED_GROUPS[*]}")

if [ "$NOCRAWL" = false ]; then
    scrapy crawl news -a "group=$GROUPS_CSV" || echo "Ooops! ($?)"
fi

for grp in "${SELECTED_GROUPS[@]}"; do
    ./ai-summary.py --config config.yaml --archive "$ARCH" --group "$grp"
done
