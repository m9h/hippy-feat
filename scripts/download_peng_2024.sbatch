#!/bin/bash
#SBATCH --job-name=dl-peng-2024
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/data/datasets/peng_2024_neurofeedback/logs/%x-%j.out

# Download Peng, Wammes, Nguyen, Iordan, Norman, Turk-Browne (2024)
# "Inducing representational change in the hippocampus through real-time
# neurofeedback" — Phil Trans R Soc B 379:20230091.
#
# Most recent PCML closed-loop neurofeedback dataset with full subject data
# released. ~21.93 GB across 19 subject .zip files + README. Useful reference
# for validating Variant G's variance-aware decoder against the lab's
# current classifier-evidence-based feedback approach on matched-paradigm
# data (Princeton 3T, RT-Cloud infrastructure, visual cortex + hippocampus).
#
# Dryad: https://doi.org/10.5061/dryad.kd51c5bg2
# Code:  https://github.com/KailongPeng/real_time_neurofeedback
#
# Dryad bulk-download API returns a single tar archive for the dataset.

set -euo pipefail

DOI="10.5061/dryad.kd51c5bg2"
OUT=/data/datasets/peng_2024_neurofeedback
mkdir -p "$OUT/logs" "$OUT/raw"
cd "$OUT"

# Dryad exposes a bulk archive endpoint keyed by the DOI. It returns a tar
# containing README + all per-subject zips. Some versions of the API produce
# a tar stream; others produce a pointer URL — we handle both.
DOI_URL_ENC=$(printf '%s' "$DOI" | sed 's|/|%2F|g; s|:|%3A|g')
ARCHIVE_URL="https://datadryad.org/api/v2/datasets/doi%3A${DOI_URL_ENC}/download"

echo "Authenticated download via Dryad OAuth2 + /api/v2/files/{id}/download."

# Dryad uses OAuth2 client_credentials. The token at ~/.dryad-token is
# the API token (which Dryad calls a "secret" in their docs); we
# exchange it for an access_token, then use Bearer auth on the file
# download endpoints. Endpoint: /oauth/token with grant_type=client_credentials.
DRYAD_SECRET=$(cat ~/.dryad-token)
ACCESS_JSON=$(curl -fsSL -X POST \
    -d "client_id=$DRYAD_SECRET&grant_type=client_credentials&client_secret=$DRYAD_SECRET" \
    https://datadryad.org/oauth/token 2>&1) || ACCESS_JSON=""
ACCESS_TOKEN=$(echo "$ACCESS_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('access_token',''))" 2>/dev/null)

if [ -z "$ACCESS_TOKEN" ]; then
    echo "OAuth client_credentials exchange failed — assuming the token is"
    echo "already a long-lived bearer token (personal API token)."
    ACCESS_TOKEN="$DRYAD_SECRET"
fi
echo "auth token resolved (length: ${#ACCESS_TOKEN})"

UA="Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36"
curl -fsSL -A "$UA" "https://datadryad.org/api/v2/datasets/doi%3A${DOI_URL_ENC}/versions" \
    -o versions.json

# Build list of (name, file_id, size) tuples from the API.
python3 - <<'PY' > /tmp/peng_dl_list.tsv
import json, urllib.request, sys
with open("versions.json") as f:
    versions = json.load(f)
latest = versions["_embedded"]["stash:versions"][-1]
files_url = "https://datadryad.org" + latest["_links"]["stash:files"]["href"]
req = urllib.request.Request(files_url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    listing = json.load(resp)
for f in listing["_embedded"]["stash:files"]:
    name = f["path"]
    file_id = f["_links"]["self"]["href"].rstrip("/").split("/")[-1]
    size = f.get("size", 0)
    sys.stdout.write(f"{name}\t{file_id}\t{size}\n")
PY

TOTAL=$(awk -F'\t' '{sum+=$3} END {print sum/1e9}' /tmp/peng_dl_list.tsv)
COUNT=$(wc -l < /tmp/peng_dl_list.tsv)
echo "fetching $COUNT files, total ${TOTAL} GB"

while IFS=$'\t' read -r name file_id size; do
    out="raw/$name"
    if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ]; then
        echo "SKIP $name (already complete)"
        continue
    fi
    # Authenticated endpoint — bypasses Anubis bot-challenge.
    url="https://datadryad.org/api/v2/files/$file_id/download"
    size_gb=$(awk -v s="$size" 'BEGIN{printf "%.2f", s/1e9}')
    echo "GET $name ($size_gb GB) id=$file_id"
    curl -fsSL -A "$UA" \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        --retry 5 --retry-delay 15 --retry-all-errors \
        --connect-timeout 30 --max-time 1800 \
        -o "$out" "$url" || {
            echo "FAILED $name — will be retried next job"
            continue
        }
    # Sanity check: did we actually get the file (not a 4kB HTML challenge)?
    actual=$(stat -c%s "$out")
    if [ "$actual" -lt 100000 ] && [ "$size" -gt 100000 ]; then
        echo "  WARN: $name shrank to $actual bytes (expected $size); likely auth challenge"
        head -c 200 "$out"
        rm -f "$out"
        echo "  → token may be wrong; aborting"
        exit 2
    fi
    echo "  $name done -> $actual bytes"
done < /tmp/peng_dl_list.tsv

echo "---"
du -sh "$OUT/raw/"
ls "$OUT/raw/" | head
echo "---"
# Unpack per-subject zips in place
for z in "$OUT/raw"/sub*.zip; do
    [ -f "$z" ] || continue
    sub=$(basename "$z" .zip)
    mkdir -p "$OUT/subjects/$sub"
    echo "unpacking $sub"
    unzip -q -o "$z" -d "$OUT/subjects/$sub/"
done
echo "Done. Unpacked subjects:"
ls "$OUT/subjects/" 2>/dev/null | head
