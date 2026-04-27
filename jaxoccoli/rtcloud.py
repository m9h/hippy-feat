"""rt-cloud bridge for jaxoccoli.

Adapter layer between the rt-cloud framework (Princeton CML / Brainiak)
and `jaxoccoli.realtime.RTPipeline`. Exposes a factory that returns
`(pipeline, on_dicom_callback)` matching rt-cloud's expected callable
interface for per-DICOM-arrival processing.

Example rt-cloud project script:

    from jaxoccoli.rtcloud import make_rtcloud_decoder
    import nibabel as nib

    pipeline, on_dicom = make_rtcloud_decoder(
        mask_path="/path/to/chosenMask.npy",
        onsets_sec=[6.0, 13.5, 21.0, 28.5, ...],
        tr=1.5,
        post_stim_window_sec=12.0,
    )

    # Inside rt-cloud's per-volume callback:
    def on_volume_arrival(nifti_path, tr_index):
        vol = nib.load(nifti_path).get_fdata()
        result = on_dicom(vol, tr_index)
        if result is not None:
            # Write feedback signal back to the experiment
            send_to_experiment(result["beta_mean"], result["snr"])
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .realtime import (
    RTPipeline,
    RTPipelineConfig,
    confidence_mask,
)


def _load_mask(path: str | Path) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        m = np.load(p)
    else:
        try:
            import nibabel as nib
        except ImportError as e:
            raise ImportError(
                "nibabel is required to load NIfTI masks; install with "
                "`pip install jaxoccoli[neuro]`"
            ) from e
        m = nib.load(p).get_fdata()
    return np.asarray(m, dtype=bool)


def make_rtcloud_decoder(
    mask_path: str | Path,
    onsets_sec: Sequence[float],
    tr: float,
    *,
    post_stim_window_sec: float = 12.0,
    max_trs: int = 200,
    prior_mean: Optional[np.ndarray] = None,
    prior_var: Optional[np.ndarray] = None,
    include_drift: bool = True,
    warmup: bool = True,
) -> tuple[RTPipeline, Callable[[np.ndarray, int], Optional[dict]]]:
    """Build a Variant G real-time decoder for an rt-cloud project.

    Args:
        mask_path: path to a 3-D mask file. Either `.npy` (boolean or
            non-zero-int) or NIfTI (any data type; non-zero treated as True).
        onsets_sec: sequence of event onset times in seconds, run-relative.
        tr: repetition time in seconds.
        post_stim_window_sec: how long after a probe onset to wait before
            emitting the posterior. 12 s captures the HRF tail.
        max_trs: pad-to length for static JIT shapes. Set to the maximum
            run length in TRs (e.g. 200 for a 5-min run @ TR=1.5).
        prior_mean: optional (P,) prior mean — typically the average β from
            a held-out training session. P is determined by include_drift.
        prior_var: optional (P,) prior variance. Smaller variance shrinks
            harder toward the prior; pass training-set per-coefficient
            variance for a Gaussian prior matching the empirical Bayes
            recommendation in Eklund et al. 2014.
        include_drift: whether to add a cosine drift regressor.
        warmup: trigger JIT compilation immediately (cost: ~3 s once).

    Returns:
        (pipeline, on_volume_callback). The callback signature is
            on_volume(volume_3d: np.ndarray, tr_index: int) -> dict | None
        and returns None until a probe trial completes, then a dict with
        keys `probe_trial`, `n_trs_used`, `beta_mean`, `beta_var`, `snr`.
    """
    mask = _load_mask(mask_path)
    config = RTPipelineConfig(
        tr=tr,
        mask=mask,
        onsets_sec=np.asarray(onsets_sec, dtype=np.float32),
        post_stim_window_sec=post_stim_window_sec,
        max_trs=max_trs,
        prior_mean=prior_mean,
        prior_var=prior_var,
        include_drift=include_drift,
    )
    pipeline = RTPipeline(config)
    if warmup:
        pipeline.precompute()
    return pipeline, pipeline.on_volume


__all__ = ["make_rtcloud_decoder", "confidence_mask"]
