#!/usr/bin/env python3
"""Compact, backward-compatible storage helpers for AD-GS image priors.

New depth and mask priors are ZIP-compressed NumPy archives.  Depth is stored
as float16 and promoted to float32 on read.  The nonzero foreground pattern of
semantic/sky masks is stored losslessly at one bit per pixel and restored as a
boolean array; instance IDs are deliberately not part of the training
semantics.  Existing ``.npy`` files and flow ``.npz`` files remain readable.
"""

from __future__ import absolute_import

import os
from pathlib import Path
import uuid
import zipfile

import numpy as np


FORMAT_VERSION = 1
FORMAT_KEY = "adgs_prior_format"
KIND_KEY = "kind"
DATA_KEY = "data"
SHAPE_KEY = "shape"
PACKED_KEY = "packed"
DEPTH_ERROR_KEY = "max_abs_error"
MAX_NORMALIZED_DEPTH_ABS_ERROR = 5e-4


def _stored_kind(kind):
    if kind in ("semantic", "sky"):
        return kind + "_binary_nonzero"
    return kind


def _text_scalar(value):
    value = np.asarray(value).item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def prior_candidates(path):
    """Return the preferred path followed by its legacy/compact counterpart."""
    path = Path(path)
    if path.suffix == ".npy":
        return (path.with_suffix(".npz"), path)
    if path.suffix == ".npz":
        return (path, path.with_suffix(".npy"))
    return (path.with_suffix(".npz"), path.with_suffix(".npy"))


def resolve_prior_path(path, required=True):
    """Resolve a compact ``.npz`` first, then a legacy ``.npy`` prior."""
    candidates = prior_candidates(path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if required:
        raise FileNotFoundError(
            "Prior does not exist (checked {}): {}".format(
                ", ".join(str(item) for item in candidates), path
            )
        )
    return None


def prior_exists(path):
    return resolve_prior_path(path, required=False) is not None


def _write_npz(path, **arrays):
    """Atomically replace *path* only after a complete, durable NPZ write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (".{}.tmp-{}".format(path.name, uuid.uuid4().hex))
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _archive_header(kind):
    return {
        FORMAT_KEY: np.asarray(FORMAT_VERSION, dtype=np.int16),
        KIND_KEY: np.asarray(_stored_kind(kind)),
    }


def save_depth_prior(path, array):
    """Store normalized DPT depth as float16 with a checked error bound."""
    array = np.asarray(array)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError("Depth must be numeric, got {}".format(array.dtype))
    if not np.isfinite(array).all():
        raise ValueError("Depth contains non-finite values")
    if array.size and (np.min(array) < 0.0 or np.max(array) > 1.0):
        raise ValueError("DPT depth must be normalized to [0, 1]")
    values = array.astype(np.float16)
    max_abs_error = float(
        np.max(np.abs(values.astype(np.float32) - array.astype(np.float32)))
        if array.size else 0.0
    )
    if max_abs_error > MAX_NORMALIZED_DEPTH_ABS_ERROR:
        raise ValueError(
            "float16 depth error {} exceeds {}".format(
                max_abs_error, MAX_NORMALIZED_DEPTH_ABS_ERROR
            )
        )
    payload = _archive_header("depth")
    payload[DATA_KEY] = values
    payload[DEPTH_ERROR_KEY] = np.asarray(max_abs_error, dtype=np.float32)
    return _write_npz(path, **payload)


def save_mask_prior(path, array, kind):
    """Losslessly store a mask's binary nonzero pattern as packed bits."""
    if kind not in ("semantic", "sky"):
        raise ValueError("Mask kind must be semantic or sky, got {}".format(kind))
    mask = np.asarray(array) != 0
    payload = _archive_header(kind)
    payload[SHAPE_KEY] = np.asarray(mask.shape, dtype=np.int64)
    payload[PACKED_KEY] = np.packbits(mask.reshape(-1))
    return _write_npz(path, **payload)


def save_flow_prior(path, flow):
    """Store flow with the legacy ``flow`` key using ZIP compression."""
    return _write_npz(path, flow=np.asarray(flow, dtype=object))


def _load_compact_archive(archive, expected_kind):
    files = set(archive.files)
    if FORMAT_KEY not in files:
        raise ValueError("Not an AD-GS compact prior archive")
    version = int(np.asarray(archive[FORMAT_KEY]).item())
    if version != FORMAT_VERSION:
        raise ValueError("Unsupported AD-GS prior format version {}".format(version))
    kind = _text_scalar(archive[KIND_KEY])
    stored_expected_kind = _stored_kind(expected_kind)
    if kind != stored_expected_kind:
        raise ValueError(
            "Prior kind mismatch: expected {}, archive contains {}".format(
                stored_expected_kind, kind
            )
        )
    if expected_kind == "depth":
        return np.asarray(archive[DATA_KEY], dtype=np.float32)
    shape = tuple(int(value) for value in np.asarray(archive[SHAPE_KEY]).tolist())
    count = int(np.prod(shape, dtype=np.int64))
    unpacked = np.unpackbits(np.asarray(archive[PACKED_KEY], dtype=np.uint8))[:count]
    if unpacked.size != count:
        raise ValueError("Packed mask is shorter than its declared shape")
    return unpacked.astype(np.bool_).reshape(shape)


def load_array_prior(path, kind):
    """Load depth/semantic/sky from compact NPZ or legacy NPY."""
    if kind not in ("depth", "semantic", "sky"):
        raise ValueError("Unsupported array prior kind: {}".format(kind))
    resolved = resolve_prior_path(path)
    loaded = np.load(str(resolved), allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            array = _load_compact_archive(loaded, kind)
        finally:
            loaded.close()
    else:
        array = np.asarray(loaded)
        if kind == "depth":
            array = array.astype(np.float32, copy=False)
    return array


def load_depth_prior(path):
    return load_array_prior(path, "depth")


def load_mask_prior(path, kind="semantic"):
    return load_array_prior(path, kind)


def load_flow_prior(path):
    """Load the unchanged legacy flow object-array API from an NPZ archive."""
    resolved = resolve_prior_path(path)
    archive = np.load(str(resolved), allow_pickle=True)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Flow prior must be an NPZ archive: {}".format(resolved))
    try:
        if "flow" not in archive.files:
            raise ValueError("Flow key is missing: {}".format(resolved))
        return archive["flow"]
    finally:
        archive.close()


def is_compact_prior(path, kind):
    """Return whether *path* is already in the canonical compressed format."""
    path = Path(path)
    if path.suffix != ".npz" or not path.is_file():
        return False
    try:
        with zipfile.ZipFile(str(path), "r") as zip_archive:
            members = zip_archive.infolist()
    except (OSError, zipfile.BadZipFile):
        return False
    if not members or not all(
        member.compress_type == zipfile.ZIP_DEFLATED for member in members
    ):
        return False
    if kind == "flow":
        return any(member.filename == "flow.npy" for member in members)
    if kind not in ("depth", "semantic", "sky"):
        return False
    try:
        archive = np.load(str(path), allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile):
        return False
    if not isinstance(archive, np.lib.npyio.NpzFile):
        return False
    try:
        if FORMAT_KEY not in archive.files or KIND_KEY not in archive.files:
            return False
        return (
            int(np.asarray(archive[FORMAT_KEY]).item()) == FORMAT_VERSION
            and _text_scalar(archive[KIND_KEY]) == _stored_kind(kind)
        )
    finally:
        archive.close()


def compact_prior(source, target, kind):
    """Read a legacy or compact prior and write a canonical compact archive."""
    if kind == "depth":
        return save_depth_prior(target, load_depth_prior(source))
    if kind in ("semantic", "sky"):
        return save_mask_prior(target, load_mask_prior(source, kind), kind)
    if kind == "flow":
        return save_flow_prior(target, load_flow_prior(source))
    raise ValueError("Unsupported prior kind: {}".format(kind))
