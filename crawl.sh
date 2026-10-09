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
YQ=/usr/bin/yq
export CRAWL_BASE
source ./venv/bin/activate

# Read group names from config.yaml
if command -v $YQ &>/dev/null; then
    mapfile -t ALL_GROUPS < <($YQ -r '.groups[].name' config.yaml)
else
    mapfile -t ALL_GROUPS < <(python3 -c "
import yaml
with open('config.yaml') as f:
    c = yaml.safe_load(f)
for g in c.get('groups', []):
    print(g['name'])
")
fi

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
    # Archive each group's added_file before crawl so the spider starts fresh
    for grp in "${SELECTED_GROUPS[@]}"; do
        # if command -v $YQ &>/dev/null; then
        #     added_file=$($YQ -r --arg g "$grp" \
        #         '.groups[] | select(.name == $g) | .output.added_file' config.yaml)
        # else
            added_file=$(GROUP_NAME="$grp" python3 -c "
import yaml, os
with open('config.yaml') as f:
    c = yaml.safe_load(f)
name = os.environ['GROUP_NAME']
for g in c.get('groups', []):
    if g['name'] == name:
        print(g.get('output', {}).get('added_file', 'added.txt'))
        break
")
        # fi
        [ -f "$added_file" ] && mv "$added_file" "$ARCH/${added_file}-$(date +%F)" 2>/dev/null || true
    done

    scrapy crawl news -a "group=$GROUPS_CSV" || echo "Ooops! ($?)"
fi

for grp in "${SELECTED_GROUPS[@]}"; do
    if ! ./ai-summary.py --config config.yaml --archive "$ARCH" --group "$grp"; then
        echo "ai-summary.py failed for group '$grp' — not exporting a stale digest." >&2
        exit 1
    fi
done

# Combine this run's summaries into one dated digest and push to Drive.
# Requires: rclone configured with a remote named "gdrive" (run `rclone config`
# once, pick Google Drive, and share the "15_HPC podcast" folder's ID or just
# browse to it — see README section "Drive digest upload" for the one-time setup).
./export_digest.py --config config.yaml --min-rate 6
if command -v rclone &>/dev/null; then
    rclone copy digests/ "gdrive:15_HPC podcast/digests/" --update
else
    echo "rclone not found — digest written locally to digests/ but not uploaded to Drive" >&2
fi
