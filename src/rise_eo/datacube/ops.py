"""Datacube ops — pure transforms over (t, H, W, bands) + metadata.

Spec: specs/04-datacube-ops.md. Folds in the L2A-relevant subset of
core/datacube_ops.py. Each op: (datacube, metadata, **kwargs) -> (datacube, metadata),
so they compose via `run_ops`.

Dropped vs legacy: run_s2cloudless / CMK-based apply_cloud_mask (L1C-only) +
run_s2cloudless_core[_chunkwise] (s2cloudless dependency). See LIMITATIONS.md, "Legacy".
"""

from __future__ import annotations

import copy
import datetime

import numba
import numpy as np
import pandas as pd

from rise_eo import config

MOSAIC_SCHEMES = ("calendar",)
MONTH = "month"


def check_mosaic_scheme(mosaic_scheme: str) -> None:
    """Raise `ValueError` unless `mosaic_scheme` is `"calendar"`."""
    if mosaic_scheme not in MOSAIC_SCHEMES:
        raise ValueError(
            f"mosaic_scheme={mosaic_scheme!r} is not supported: only \"calendar\" is "
            "accepted (the legacy \"acquisition\" scheme was removed)."
        )


def check_mosaic_days(mosaic_days, startdate=None):
    """Return `mosaic_days` as an `int` or `"month"`, raising `ValueError` for anything else.

    With `startdate` and `"month"`, `startdate` must be the 1st of a month at 00:00 UTC.
    """
    if isinstance(mosaic_days, (int, np.integer)) and not isinstance(mosaic_days, bool):
        return int(mosaic_days)
    if mosaic_days != MONTH:
        raise ValueError(f'mosaic_days must be an int or "month", got {mosaic_days!r}.')
    if startdate is not None:
        start = _dt2ts(startdate).tz_convert("UTC")
        if start != start.normalize() or start.day != 1:
            raise ValueError(
                f'mosaic_days="month" needs startdate on the 1st of a month at 00:00 UTC, got {start}.'
            )
    return MONTH


def parse_mosaic_days(value):
    """`"month"` or an integer, read back from a CLI argument or a CSV cell."""
    if isinstance(value, str) and value.strip() != MONTH:
        value = int(value)
    return check_mosaic_days(value.strip() if isinstance(value, str) else value)


def run_ops(datacube, metadata, sequence):
    """Run a sequence of (func, kwargs) ops, threading (datacube, metadata)."""
    for func, kwargs in sequence:
        datacube, metadata = func(datacube=datacube, metadata=metadata, **kwargs)
    return datacube, metadata


def apply_cloud_mask_scl(datacube, metadata, *, mask_classes, mask_band="SCL",
                         mask_value=0):
    """Set pixels to `mask_value` where the categorical mask band (`mask_band`,
    default `"SCL"`) has a value in `mask_classes`, across every
    non-mask band. The mask band itself is left untouched (drop it separately).

    `mask_band` generalizes this beyond S2 (`mask_type="categorical_classes"`) — any
    source whose QA band encodes
    "masked" as a small set of discrete values (not just SCL) can reuse this op
    by declaring its own `mask_band`."""
    band_indices = {band: i for i, band in enumerate(metadata["bands"])}
    if mask_band not in band_indices:
        raise ValueError(f"{mask_band} band not present in datacube")

    idx_to_modify = [i for b, i in band_indices.items() if b != mask_band]

    mask_col = datacube[:, :, :, band_indices[mask_band]]
    selected = datacube[:, :, :, idx_to_modify]
    selected[np.where(np.isin(mask_col, mask_classes))] = mask_value
    datacube[:, :, :, idx_to_modify] = selected
    return datacube, metadata


def drop_bands(datacube, metadata, *, bands_to_drop):
    """Slice out `bands_to_drop`; update `metadata['bands']`."""
    band_indices = {band: i for i, band in enumerate(metadata["bands"])}
    bands_to_keep = [b for b in band_indices if b not in bands_to_drop]
    idx_to_keep = [band_indices[b] for b in bands_to_keep]
    datacube = datacube[:, :, :, idx_to_keep]
    metadata["bands"] = bands_to_keep
    return datacube, metadata


def median_mosaic(datacube, metadata, *, startdate: datetime.datetime,
                  enddate: datetime.datetime, mosaic_days=20, mask_value=0,
                  mosaic_scheme: str | None = None):
    """Median-mosaic `timestamps` into calendar windows over `[startdate, enddate)`, one slice per window.

    Every window is emitted (an empty one is all-`mask_value`) and labelled by its start, so the
    `timestamps` axis depends only on `(startdate, enddate, mosaic_days)`. `mosaic_scheme` must be
    `"calendar"`.
    """
    if mosaic_scheme is None:
        mosaic_scheme = config.MOSAIC_SCHEME
    check_mosaic_scheme(mosaic_scheme)
    mosaic_days = check_mosaic_days(mosaic_days, startdate)
    if mosaic_days != MONTH and mosaic_days < 1:
        return datacube, metadata

    timestamps = metadata["timestamps"]
    labels, index_intervals = _calendar_windows(
        timestamps=timestamps, startdate=startdate, enddate=enddate,
        mosaic_days=mosaic_days,
    )

    dtype = datacube.dtype
    datacube = datacube.astype(float)
    datacube[np.where(datacube == mask_value)] = np.nan

    mosaiced = _mosaic_from_intervals(datacube, index_intervals, mask_value)
    mosaiced = mosaiced.astype(dtype)

    md = copy.deepcopy(metadata)
    md["mosaic_index_intervals"] = index_intervals
    md["previous_timestamps"] = timestamps
    md["timestamps"] = labels
    md["data_shape_desc"] = ("timestamps", "height", "width", "bands")
    return mosaiced, md


def area_median(datacube, metadata=None, *, mask_value=0):
    """Collapse H×W to a single median pixel per timestamp (deploy helper)."""
    dtype = datacube.dtype
    _, height, width, _ = datacube.shape

    datacube = datacube.astype(float)
    datacube[np.where(datacube == mask_value)] = np.nan
    out = np.expand_dims(np.nanmedian(datacube, axis=(1, 2)), axis=(1, 2))
    out[np.isnan(out)] = mask_value
    out = out.astype(dtype)

    md = None
    if metadata is not None:
        md = copy.deepcopy(metadata)
        md["previous_height_width"] = (height, width)
    return out, md


# --- private helpers (faithful port; preserve legacy bucket behavior) ---------

def _dt2ts(dt: datetime.datetime, tz="UTC") -> pd.Timestamp:
    """Localize a naive datetime to a tz-aware pd.Timestamp (default UTC).

    The catalog stores tz-aware (UTC) acquisition timestamps, but user
    startdate/enddate arrive tz-naive. Comparison ops (== < > >=) raise when
    mixing tz-aware and tz-naive, so we attach UTC here. Attaching the tz is
    easier on a pd.Timestamp than on a datetime; hour/minute granularity is
    irrelevant since S2 revisit is ~5 days.
    """
    if dt.tzinfo is None:
        return pd.Timestamp(dt, tz=tz)
    return pd.Timestamp(dt)


def _is_sorted(seq) -> bool:
    return all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))


def _checked_bounds(timestamps, startdate, enddate):
    """`(startdate, enddate)` as tz-aware `pd.Timestamp`s, after checking that `timestamps`
    is sorted and lies within them."""
    startdate = _dt2ts(startdate)
    enddate = _dt2ts(enddate)

    if not _is_sorted(timestamps):
        raise ValueError("timestamps is not sorted.")
    if startdate > timestamps[0]:
        raise ValueError("startdate must be before or on the first timestamp")
    if enddate < timestamps[-1]:
        raise ValueError("enddate must be after or on the last timestamp")
    return startdate, enddate


def mosaic_windows(startdate, enddate, mosaic_days):
    """The `(lo, hi)` pairs of `mosaic_days`-long (or calendar-month) windows tiling
    `[startdate, enddate)`; the last one may run past `enddate`."""
    check_mosaic_days(mosaic_days, startdate)
    step = pd.DateOffset(months=1) if mosaic_days == MONTH else datetime.timedelta(days=mosaic_days)
    windows = []
    lo = startdate
    while lo < enddate:
        windows.append((lo, lo + step))
        lo = windows[-1][1]
    return windows


def _calendar_windows(timestamps, startdate, enddate, mosaic_days=20):
    """Fixed calendar windows `[lo, lo+mosaic_days)` tiling `[startdate, enddate)`
. Returns `(labels, index_intervals)`:

    - `labels[k]` = the window-start `pd.Timestamp` (tz-aware UTC).
    - `index_intervals[k]` = `(min_i, max_i)` of the sorted `timestamps` falling in
      window k, or `None` when the window is empty.

    Windows are half-open `[lo, hi)` so a timestamp on a boundary lands in the *later*
    window; the final window is upper-inclusive so a timestamp exactly at `enddate`
    (allowed by the guard) is not dropped. Data-independent: the labels/count depend
    only on `(startdate, enddate, mosaic_days)`."""
    startdate, enddate = _checked_bounds(timestamps, startdate, enddate)

    windows = mosaic_windows(startdate, enddate, mosaic_days)

    labels, intervals = [], []
    for k, (lo, hi) in enumerate(windows):
        last = k == len(windows) - 1
        members = [i for i, ts in enumerate(timestamps)
                   if lo <= ts and (ts <= hi if last else ts < hi)]
        labels.append(lo)
        intervals.append((min(members), max(members)) if members else None)
    return labels, intervals


def _mosaic_from_intervals(datacube, index_intervals, mask_value):
    """Median-mosaic each interval into one slice, in order. A `None` interval (an
    empty calendar window) becomes an all-`mask_value` slice, so the output timestamp
    axis has one slot per interval. `datacube` already has `mask_value` set to NaN;
    an all-NaN result (fully-masked pixel or empty window) is restored to `mask_value`.
    Non-empty intervals go through the single numba core call (legacy path)."""
    n_out = len(index_intervals)
    _, height, width, n_bands = datacube.shape
    out = np.full((n_out, height, width, n_bands), np.nan)

    nonempty = [(k, iv) for k, iv in enumerate(index_intervals) if iv is not None]
    if nonempty:
        positions = np.array([k for k, _ in nonempty])
        ranges = np.array([iv for _, iv in nonempty])
        out[positions] = _median_mosaic_core(datacube, ranges)

    out[np.isnan(out)] = mask_value
    return out


@numba.njit()
def _median_mosaic_core(datacube: np.ndarray, ts_index_ranges: np.ndarray):
    n_ts, height, width, n_bands = datacube.shape
    n_mosaiced = ts_index_ranges.shape[0]
    out = np.full((n_mosaiced, height, width, n_bands), np.nan)
    for t in numba.prange(n_mosaiced):
        for h in numba.prange(height):
            for w in numba.prange(width):
                for b in numba.prange(n_bands):
                    out[t, h, w, b] = np.nanmedian(
                        datacube[ts_index_ranges[t][0]: ts_index_ranges[t][1] + 1,
                                 h, w, b]
                    )
    return out
