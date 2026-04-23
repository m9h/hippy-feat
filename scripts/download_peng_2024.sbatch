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

echo "Attempting bulk download: $ARCHIVE_URL"
if curl -fsSL -o peng_2024_dryad.tar "$ARCHIVE_URL"; then
    echo "Bulk archive downloaded; unpacking..."
    tar -xf peng_2024_dryad.tar -C raw/ || {
        echo "tar failed — keeping archive for manual inspection"
    }
    ls -la raw/ | head
    rm -f peng_2024_dryad.tar
else
    echo "Bulk endpoint failed — falling back to per-file downloads via Dryad API"
    # List files in the dataset via the API, then pull each one.
    curl -fsSL "https://datadryad.org/api/v2/datasets/doi%3A${DOI_URL_ENC}/versions" \
        -o versions.json
    # Get the latest version's files endpoint
    python3 - <<'PY'
import json, urllib.request, os
with open("versions.json") as f:
    versions = json.load(f)
latest = versions["_embedded"]["stash:versions"][-1]
files_url = "https://datadryad.org" + latest["_links"]["stash:files"]["href"]
with urllib.request.urlopen(files_url) as resp:
    files = json.load(resp)
for f in files["_embedded"]["stash:files"]:
    name = f["path"]
    stream_url = "https://datadryad.org" + f["_links"]["stash:file-download"]["href"]
    out_path = os.path.join("raw", name)
    if os.path.exists(out_path):
        print(f"SKIP {name}")
        continue
    print(f"GET {name} from {stream_url}")
    urllib.request.urlretrieve(stream_url, out_path)
PY
fi

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
