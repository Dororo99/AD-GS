"""Train/eval splitting compatible with SplatAD's NeurAD data parsers.

The reference implementation is
nerfstudio/data/dataparsers/ad_dataparser.py::_get_linspaced_indices.
It applies the split independently to every sensor, includes both temporal
endpoints in the training set, and uses the complement for evaluation.
"""

import math
from typing import Tuple

import numpy as np


def sensor_time_bounds(*time_stamps: np.ndarray) -> Tuple[float, float]:
    """Return finite, positive bounds shared by every supplied sensor stream.

    SplatAD subtracts the minimum over cameras and lidars before applying the
    train/eval split. AD-GS additionally needs a unit time domain, so the same
    all-sensor minimum and maximum define its [0, 1] normalization.
    """
    arrays = []
    for values in time_stamps:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("sensor timestamps must be one-dimensional")
        if values.size:
            if not np.isfinite(values).all():
                raise ValueError("sensor timestamps must be finite")
            arrays.append(values)
    if not arrays:
        raise ValueError("at least one non-empty sensor timestamp array is required")

    start = min(float(values.min()) for values in arrays)
    end = max(float(values.max()) for values in arrays)
    if not end > start:
        raise ValueError("sensor timestamps must span a positive duration")
    return start, end


def normalize_sensor_times(
    values: np.ndarray,
    sensor_time_min: float,
    sensor_time_max: float,
    atol: float = 1e-4,
) -> np.ndarray:
    """Map timestamps into the shared all-sensor [0, 1] interval.

    The tolerance admits float32 PLY timestamps while still rejecting values
    from an unrelated time origin. Tiny endpoint roundoff is clipped.
    """
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("timestamps to normalize must be finite")
    if not (
        np.isfinite(sensor_time_min)
        and np.isfinite(sensor_time_max)
        and sensor_time_max > sensor_time_min
    ):
        raise ValueError("sensor time bounds must be finite and increasing")
    if values.size and (
        float(values.min()) < sensor_time_min - atol
        or float(values.max()) > sensor_time_max + atol
    ):
        raise ValueError(
            "timestamps fall outside the shared sensor interval "
            "[{}, {}]".format(sensor_time_min, sensor_time_max)
        )
    normalized = (values - sensor_time_min) / (sensor_time_max - sensor_time_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    return float(normalized) if normalized.ndim == 0 else normalized


def linspace_train_local_indices(
    num_samples: int, train_split_fraction: float = 0.5
) -> np.ndarray:
    """Return SplatAD's local training indices for one sensor."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if not 0.0 < train_split_fraction <= 1.0:
        raise ValueError("train_split_fraction must be in (0, 1]")
    if num_samples == 0:
        return np.empty(0, dtype=np.int64)
    if train_split_fraction == 1.0:
        return np.arange(num_samples, dtype=np.int64)

    num_train = math.ceil(num_samples * train_split_fraction)
    return np.linspace(0, num_samples - 1, num_train, dtype=np.int64)


def split_indices_by_sensor(
    sensor_ids: np.ndarray, train_split_fraction: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Return global train/eval indices using SplatAD's sensor-wise split."""
    sensor_ids = np.asarray(sensor_ids)
    if sensor_ids.ndim != 1:
        raise ValueError("sensor_ids must be a one-dimensional array")
    if sensor_ids.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()

    if train_split_fraction == 1.0:
        all_indices = np.arange(sensor_ids.size, dtype=np.int64)
        # SplatAD intentionally uses all samples for both train and eval here.
        return all_indices, all_indices.copy()

    train_chunks = []
    for sensor_id in np.unique(sensor_ids):
        sensor_sample_indices = np.flatnonzero(sensor_ids == sensor_id)
        local_train = linspace_train_local_indices(
            len(sensor_sample_indices), train_split_fraction
        )
        train_chunks.append(sensor_sample_indices[local_train])

    train_indices = np.concatenate(train_chunks).astype(np.int64, copy=False)
    eval_indices = np.setdiff1d(
        np.arange(sensor_ids.size, dtype=np.int64),
        train_indices,
        assume_unique=True,
    )
    return train_indices, eval_indices


def get_splatad_split(
    sensor_ids: np.ndarray, train_split_fraction: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility alias returning SplatAD train and eval indices."""
    return split_indices_by_sensor(sensor_ids, train_split_fraction)


def splatad_is_val_mask(
    sensor_ids: np.ndarray, train_split_fraction: float = 0.5
) -> np.ndarray:
    """Return an evaluation mask aligned with the input sample order."""
    sensor_ids = np.asarray(sensor_ids)
    train_indices, eval_indices = split_indices_by_sensor(
        sensor_ids, train_split_fraction
    )
    if train_split_fraction == 1.0:
        return np.zeros(sensor_ids.size, dtype=np.bool_)

    is_val = np.zeros(sensor_ids.size, dtype=np.bool_)
    is_val[eval_indices] = True
    if np.intersect1d(train_indices, eval_indices).size:
        raise AssertionError("SplatAD train/eval split overlaps")
    return is_val


def normalized_train_frame_gap(
    time_stamps: np.ndarray,
    sensor_ids: np.ndarray,
    is_val: np.ndarray,
    normalization_time_stamps: np.ndarray = None,
) -> float:
    """Median train-camera gap on the shared all-sensor unit time domain.

    normalization_time_stamps should contain all camera and LiDAR times. It
    remains optional only for callers without a separate LiDAR stream.
    """
    time_stamps = np.asarray(time_stamps, dtype=np.float64)
    sensor_ids = np.asarray(sensor_ids)
    is_val = np.asarray(is_val, dtype=np.bool_)
    if not (
        time_stamps.ndim == sensor_ids.ndim == is_val.ndim == 1
        and len(time_stamps) == len(sensor_ids) == len(is_val)
    ):
        raise ValueError("time_stamps, sensor_ids and is_val must be aligned 1-D arrays")
    if len(time_stamps) < 2:
        return 1.0

    if normalization_time_stamps is None:
        normalization_time_stamps = time_stamps
    sensor_time_min, sensor_time_max = sensor_time_bounds(
        np.asarray(normalization_time_stamps, dtype=np.float64)
    )
    normalized_times = normalize_sensor_times(
        time_stamps, sensor_time_min, sensor_time_max
    )

    gaps = []
    for sensor_id in np.unique(sensor_ids):
        sensor_train_times = np.sort(
            normalized_times[(sensor_ids == sensor_id) & ~is_val]
        )
        positive_gaps = np.diff(sensor_train_times)
        gaps.extend(positive_gaps[positive_gaps > 0.0].tolist())
    if not gaps:
        return 1.0
    return float(np.median(np.asarray(gaps, dtype=np.float64)))
