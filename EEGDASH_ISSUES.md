# EEGDash issues encountered — discussion notes for SCCN/UCSD

*Documented 2026-05-27 while building an SAE-based mechanistic-interpretability pipeline on top of NeuralBench-EEG v1.0 (REVE/LaBraM activations on HBN-EEG resting state). Tested versions: `eegdash==0.7.2`, `braindecode==1.5.1`, `mne==1.12.1`, `mne-bids==0.18.0`. Happy to file as GitHub issues / PRs if useful.*

The issues below are roughly in order of impact. The top three each cost us multiple hours to diagnose and have clean fixes. The rest are smaller papercuts.

---

## 1. `EEGChallengeDataset` / `EEG2025R{N}[MINI]` route to non-existent NEMARDatasets repos

**Severity:** blocking — these classes can never download their data.

**Reproducer:**
```python
from eegdash.dataset import EEG2025R1MINI
ds = EEG2025R1MINI(cache_dir="./data")
# StorageAccessError: Could not resolve NEMAR pointer
#   EEG2025r1mini/sub-NDARFK610GY5/eeg/sub-NDARFK610GY5_task-RestingState_eeg.bdf
#   from GitHub raw: HTTP 404 Not Found
```

**Root cause:** `eegdash/dataset/dataset.py` line 839 constructs the dataset_id as `f"EEG2025r{release[1:]}"` (yielding `EEG2025r1mini`, `EEG2025r1`, …). The resolver then builds `https://raw.githubusercontent.com/NEMARDatasets/EEG2025r1mini/HEAD/...` — but the **NEMARDatasets GitHub org has no repos with that naming**. It has `nm000NNN` (NEMAR-canonical, e.g. `nm000103` = HBN_EEG_NC) and `on00NNNN` (OpenNeuro mirrors, e.g. `on005505` = the R1 dataset).

**The mapping IS in the codebase**, just not used for routing:

```python
# eegdash/const.py:37
RELEASE_TO_OPENNEURO_DATASET_MAP = {"R1": "ds005505", "R2": "ds005506", …}
```

The class imports it for *validation* of release names but routes via the NEMAR path anyway.

**Suggested fix (~3-line change):** in `EEGChallengeDataset.__init__`, replace

```python
dataset_id = f"EEG2025r{release[1:]}"
```

with

```python
dataset_id = RELEASE_TO_OPENNEURO_DATASET_MAP[release]
```

Then it routes through `s3://openneuro.org/dsXXXXXX/` (already supported by your `STORAGE_CONFIGS["openneuro"]` backend in `_source_inference.py`) instead of the non-existent NEMAR path.

If the intent is to route through NEMAR specifically, the repo naming would need to be either `nm000103` (full HBN, 447 subjects) for everyone, or the per-release subset repos would need to be created/published.

**Workaround we used:**
```python
from eegdash import EEGDashDataset
from eegdash.const import RELEASE_TO_OPENNEURO_DATASET_MAP, SUBJECT_MINI_RELEASE_MAP

ds = EEGDashDataset(
    cache_dir=cache_dir,
    query={
        "dataset": RELEASE_TO_OPENNEURO_DATASET_MAP[f"R{n}"],
        "task": "RestingState",
        "subject": {"$in": list(SUBJECT_MINI_RELEASE_MAP[f"R{n}"])},  # for mini
    },
)
```

This works for R1–R6, R8, R9, R10 — see issue 2 for the two exceptions.

---

## 2. MongoDB registry returns 0 subjects for R7 (ds005511) and R11 (ds005516) despite ~1300 files on S3

**Severity:** blocking — 2 of 11 HBN releases unfetchable via eegdash.

**Reproducer:**
```python
ds = EEGDashDataset(
    cache_dir="...",
    query={"dataset": "ds005511", "task": "RestingState"},
)
print(len(ds.datasets))  # 0
```

**Verification that the data IS there:**
```bash
$ aws --no-sign-request s3 ls s3://openneuro.org/ds005511/ --recursive \
       | grep -c RestingState_eeg
741
$ aws --no-sign-request s3 ls s3://openneuro.org/ds005516/ --recursive \
       | grep -c RestingState_eeg
567
```

OpenNeuro's S3 has 741 RestingState files in ds005511 and 567 in ds005516. The corresponding NEMARDatasets repos (`on005511`, `on005516`) also exist on GitHub (HTTP 200). It's specifically the **eegdash MongoDB index** for these two datasets that's empty.

**Workaround we used:** bypass eegdash entirely with anonymous boto3 against OpenNeuro:

```python
import boto3
from botocore import UNSIGNED
from botocore.config import Config
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket="openneuro.org", Prefix="ds005511/"):
    for obj in page.get("Contents", []):
        if "RestingState_eeg" in obj["Key"]:
            s3.download_file("openneuro.org", obj["Key"], local_path)
```

Got 1482 R7 keys and 1134 R11 keys downloaded (763 + 695 new + cached, 0 failed) at ~50 MB/s with 8 workers. Then loaded them via `mne_bids` directly into a `braindecode.datasets.BaseConcatDataset` — works fine, the data is well-formed BIDS, eegdash's metadata just didn't include it.

**Suggestion:** re-run whatever ingestion pipeline populates the eegdash MongoDB for ds005511 + ds005516, or document that R7 + R11 need the local-BIDS path.

---

## 3. `DataIntegrityError` on a single recording kills the entire joblib preprocess pass

**Severity:** medium — one bad subject scuttles ~184 good ones.

**Reproducer:**
```python
from braindecode.preprocessing import preprocess, Preprocessor
ds = EEGDashDataset(cache_dir=..., query={"dataset": "ds005507", "task": "RestingState"})
preprocess(ds, [Preprocessor("filter", l_freq=0.5, h_freq=99.5)])
# eegdash.dataset.exceptions.DataIntegrityError:
#   Primary data file not found on S3:
#   s3://openneuro.org/ds005507/sub-NDARVG597HNL/eeg/sub-NDARVG597HNL_task-RestingState_eeg.set
```

ds005507 advertises 184 RestingState subjects in eegdash's registry, but `sub-NDARVG597HNL`'s `.set` file is genuinely absent on the OpenNeuro mirror. eegdash raises `DataIntegrityError` inside `_download_required_files`, which propagates up through `braindecode.preprocessing.preprocess`'s joblib `Parallel(n_jobs=-1)` call and kills the whole loop. Loses 183 good subjects worth of work.

**Workaround we used:** pre-validate every subject upfront, drop the failures, then call `preprocess` on the filtered list:

```python
from mne._fiff.pick import _picks_to_idx
valid_idx = []
for i, ds_rec in enumerate(concat_ds.datasets):
    try:
        ds_rec._ensure_raw()
        picks = _picks_to_idx(ds_rec.raw.info, None, "data_or_ica",
                              exclude=(), allow_empty=True)
        if len(picks) == 0:
            continue  # no data channels — would crash filter()
        valid_idx.append(i)
    except Exception:
        continue  # DataIntegrityError, FileNotFoundError, …
    finally:
        ds_rec._raw = None  # release memory (see issue 4)
concat_ds = concat_ds.split(valid_idx)["0"]
```

This catches:
- `DataIntegrityError` when the S3 file is missing
- The `picks=NoneNone yielded no channels` failure mode (happens on at least one R10 = ds005515 subject — channels.tsv has no `type=EEG` rows)

**Suggestion:** either (a) make `DataIntegrityError` per-subject-recoverable in `_download_required_files`, or (b) document the pre-validate pattern. Option (a) is preferable since downstream users would likely run into this on any release with missing files.

---

## 4. `_ensure_raw()` cache_dir on `rec._raw` causes OOM for predownload loops

**Severity:** medium — instant OOM if you try to walk a whole release.

**Reproducer:**
```python
ds = EEGDashDataset(cache_dir=..., query={"dataset": "ds005508"})  # 322 subjects
for rec in ds.datasets:
    rec._ensure_raw()  # downloads .set + loads mne.Raw into rec._raw
# By subject 100 you're holding ~15 GB of mne.Raw objects in RAM.
# OOM-killed by slurm cgroups at 322 subjects × ~150 MB ≈ 48 GB.
```

`_ensure_raw()` does the right thing for *one-at-a-time* iteration but the loaded Raw stays cached on `rec._raw` indefinitely. For a "download all subjects to local disk" workflow we don't need any Raw in memory — just the cached `.set` file.

**Workaround:** explicitly `rec._raw = None` after each iteration.

**Suggestion:** either (a) split `_ensure_raw()` into `_download_files()` + `_load_raw()` so callers can do file-only fetch without the load, or (b) add a `keep_raw=True` kwarg that defaults to current behavior.

---

## 5. Documentation gap — `download=False` + `records=...` for local BIDS

`EEGDashDataset.__init__` accepts `download: bool = True` and `records: list[dict] | None = None`. These suggest a "local-BIDS, no-MongoDB" mode is supported, but I couldn't find example usage in `eegdash.org` docs, the GitHub README, or the tutorials. Schema for `records` (what fields are required?) isn't documented either.

We ended up bypassing eegdash for local-BIDS loading and going straight to `mne_bids.read_raw_bids` → `braindecode.datasets.BaseDataset` to handle ds005511 + ds005516. Worked fine, but if `EEGDashDataset(download=False)` is the intended path for that use case, it would benefit from a documented example.

---

## 6. Channel-type defaults to `misc` for HBN .set files, breaks strict `pick_types(eeg=True)`

**Severity:** low — a footgun for users who try strict EEG-only picking.

When `mne.io.read_raw_eeglab` loads HBN's GSN-128 .set files, channels come back with type "misc" rather than "eeg". `raw.filter(...)` handles this fine (defaults to `picks="data_or_ica"`), but `mne.pick_types(eeg=True)` returns an empty list, which surprises users writing pre-validation logic.

Not really an eegdash bug — this is upstream from EEGLAB's `.set` channel-type metadata — but worth a note in the docs if you're recommending users do channel-validity checks (e.g., "use `_picks_to_idx(info, None, 'data_or_ica')` not `pick_types(eeg=True)`"). 

---

## 7. Architecture-config mismatch on CBraMod / LUNA HF checkpoints (NeuralBench-adjacent)

Not strictly eegdash, but related to the FM-loading flow you reference: `braindecode.models.CBraMod.from_pretrained("braindecode/cbramod-pretrained", n_outputs=1)` constructs a default-arg model with `d_model=256`, but the released checkpoint has `d_model=576`. State-dict load raises 50+ shape mismatches. Same pattern for LUNA. Workaround: skip the constructor entirely and wrap each weight in a synthetic `nn.Linear` for analysis. The released checkpoints would benefit from a sidecar `config.json` documenting the architecture kwargs, or a smarter `from_pretrained` that reads them from the checkpoint metadata.

---

## Offer to help

Items 1, 3, and 4 each look like contained PRs we could file with eegdash. Item 2 is a data-pipeline issue on your side (MongoDB ingestion). Happy to:

- File the seven as GitHub issues with the reproducers above
- Open a draft PR for #1 (the `EEGChallengeDataset` → OpenNeuro routing fix is ~3 lines)
- Open a draft PR for #3 (per-subject error tolerance in `preprocess` — would need a wider design discussion since it changes joblib behavior)
- Share our `scripts/download_hbn_s3_direct.py` (anonymous boto3 → BIDS layout) — drop-in alternative for users who hit #2

Whichever combination would be most useful — let me know.

— *(Morgan G. Hough, `mhough` on GitHub, hippy-feat / jaxoccoli project)*
