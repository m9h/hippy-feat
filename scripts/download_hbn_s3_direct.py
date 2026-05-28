#!/usr/bin/env python
"""Direct S3 download of HBN-EEG RestingState data, bypassing eegdash registry.

Why: eegdash's metadata index for HBN releases R7 (ds005511) and R11
(ds005516) returns 0 RestingState subjects — but the actual .set files
ARE on OpenNeuro's public S3 bucket (741 and 567 files respectively).
This is an eegdash registry gap, not a data gap.

What it fetches (per release) into ``<cache>/dsXXXXXX/`` in BIDS layout:
    dataset_description.json         (mandatory for BIDS)
    participants.tsv                  (subject demographics)
    phenotype/                        (HBN bifactor + CBCL scores, if present)
    sub-NDARxxxxxxx/
        eeg/
            *_task-RestingState_eeg.set
            *_task-RestingState_eeg.json
            *_task-RestingState_channels.tsv
            *_task-RestingState_events.tsv
            *_task-RestingState_eeg.fdt   (if present; many are missing)
            *_task-RestingState_electrodes.tsv  (if present)

Output is BIDS-compliant so downstream tools (mne_bids, braindecode, or
eegdash with ``download=False``) can read it directly.

Use boto3 with unsigned requests since OpenNeuro's S3 bucket is public.
Runs on CPU only — no GPU needed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


HBN_RELEASE_TO_DS = {
    1: "ds005505", 2: "ds005506", 3: "ds005507", 4: "ds005508",
    5: "ds005509", 6: "ds005510", 7: "ds005511", 8: "ds005512",
    9: "ds005514", 10: "ds005515", 11: "ds005516",
}


def _is_relevant(key: str, task: str) -> bool:
    """Filter S3 keys to per-subject task-relevant files + BIDS metadata."""
    if key.endswith("/"):
        return False
    base = key.split("/")[-1]
    # BIDS-level metadata (root of dataset)
    if "/" not in key.lstrip("ds005xxx/"):
        # bare-root files like dataset_description.json, participants.tsv
        # ds00xxxx/foo.ext
        return base in ("dataset_description.json", "participants.tsv",
                         "participants.json", "README", "CHANGES", "LICENSE")
    # phenotype scores
    if "/phenotype/" in key:
        return True
    # task-specific subject files
    if f"task-{task}_" in base or f"task-{task}." in base:
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="/data/derivatives/eegdash_cache")
    ap.add_argument("--releases", type=int, nargs="+", required=True,
                    help="HBN release numbers (1..11)")
    ap.add_argument("--task", default="RestingState")
    ap.add_argument("--bucket", default="openneuro.org")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent download threads")
    args = ap.parse_args()

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from concurrent.futures import ThreadPoolExecutor, as_completed

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    grand_total_bytes = 0
    grand_t0 = time.time()

    for rn in args.releases:
        if rn not in HBN_RELEASE_TO_DS:
            print(f"[skip] unknown release R{rn}", flush=True)
            continue
        ds_id = HBN_RELEASE_TO_DS[rn]
        prefix = f"{ds_id}/"
        print(f"\n=== R{rn} ({ds_id}) — listing {args.bucket}/{prefix} ===",
              flush=True)
        t0 = time.time()

        # Enumerate all keys under the dataset.
        paginator = s3.get_paginator("list_objects_v2")
        keys: list[tuple[str, int]] = []
        for page in paginator.paginate(Bucket=args.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if _is_relevant(key, args.task):
                    keys.append((key, obj["Size"]))
        print(f"  {len(keys)} relevant keys "
              f"({sum(s for _, s in keys) / 1e9:.1f} GB total)", flush=True)

        # Download in parallel; skip files already present + correct size.
        def fetch_one(key_size):
            key, size = key_size
            # Strip the leading "dsXXXXXX/" so files land at cache/dsXXXXXX/...
            local = cache_dir / key
            if local.exists() and local.stat().st_size == size:
                return ("skip", key, size)
            local.parent.mkdir(parents=True, exist_ok=True)
            tmp = local.with_suffix(local.suffix + ".tmp")
            try:
                s3.download_file(args.bucket, key, str(tmp))
                tmp.rename(local)
                return ("ok", key, size)
            except Exception as e:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
                return ("fail", key, f"{type(e).__name__}: {str(e)[:80]}")

        ok = skipped = failed = 0
        bytes_fetched = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(fetch_one, ks) for ks in keys]
            for i, fut in enumerate(as_completed(futures), 1):
                status, key, info = fut.result()
                if status == "ok":
                    ok += 1
                    bytes_fetched += info
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1
                    if failed <= 5:
                        print(f"  [fail] {key}: {info}", flush=True)
                if i % 100 == 0:
                    dt = time.time() - t0
                    print(f"  {i}/{len(keys)} processed  "
                          f"({ok} new + {skipped} cached + {failed} fail; "
                          f"{bytes_fetched / 1e9:.2f} GB, {dt:.0f}s)",
                          flush=True)

        dt = time.time() - t0
        print(f"[R{rn} done] ok={ok} skipped={skipped} failed={failed}  "
              f"({bytes_fetched / 1e9:.2f} GB, {dt:.0f}s)", flush=True)
        grand_total_bytes += bytes_fetched

    grand_dt = time.time() - grand_t0
    print(f"\n=== ALL DONE  {grand_total_bytes / 1e9:.2f} GB "
          f"({grand_dt / 60:.1f} min) ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
