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

echo "Per-file download via Dryad API (bulk endpoint requires auth)."
curl -fsSL "https://datadryad.org/api/v2/datasets/doi%3A${DOI_URL_ENC}/versions" \
    -o versions.json
python3 - <<'PY'
import json, urllib.request, os, sys
with open("versions.json") as f:
    versions = json.load(f)
latest = versions["_embedded"]["stash:versions"][-1]
files_url = "https://datadryad.org" + latest["_links"]["stash:files"]["href"]
with urllib.request.urlopen(files_url) as resp:
    listing = json.load(resp)
files = listing["_embedded"]["stash:files"]
total_bytes = sum(f.get("size", 0) for f in files)
print(f"fetching {len(files)} files, total {total_bytes/1e9:.2f} GB")
for i, f in enumerate(files, 1):
    name = f["path"]
    # Link key on current Dryad API is "stash:download" (the /files/{id}/download endpoint)
    stream_url = "https://datadryad.org" + f["_links"]["stash:download"]["href"]
    out_path = os.path.join("raw", name)
    if os.path.exists(out_path) and os.path.getsize(out_path) == f.get("size", 0):
        print(f"[{i}/{len(files)}] SKIP {name} (already downloaded)")
        continue
    print(f"[{i}/{len(files)}] GET {name} ({f.get('size',0)/1e9:.2f} GB) from {stream_url}")
    urllib.request.urlretrieve(stream_url, out_path)
    print(f"  done -> {os.path.getsize(out_path)/1e9:.2f} GB")
print("all files downloaded")
PY

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
