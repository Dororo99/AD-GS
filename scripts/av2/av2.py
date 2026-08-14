#!/usr/bin/env python3
"""Convert an Argoverse 2 sensor log to the AD-GS multi-camera format.

This converter intentionally follows the SplatAD camera order and its per-sensor
LINSPACE train/eval split.  It consumes the official AV2 SDK representation and
writes camera-major images, ``meta.npz``, and a colored, train-only LiDAR PLY.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

# Executing a file named ``av2.py`` otherwise shadows the installed ``av2``
# package because the script directory is first on sys.path.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR
]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.geometry.se3 import SE3
from av2.structures.sweep import Sweep
from av2.utils.io import read_city_SE3_ego

from scripts.splatad_split import (  # noqa: E402
    get_splatad_split,
    normalized_train_frame_gap,
    sensor_time_bounds,
)


CAMERA_ORDER: Tuple[str, ...] = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
)

# Match SplatAD's AV2 dataparser exactly: these cameras discard the bottom
# 250 pixels, while the remaining four cameras use the complete image.
BOTTOM_CROP: Mapping[str, int] = {
    "ring_front_center": 250,
    "ring_front_left": 0,
    "ring_front_right": 0,
    "ring_rear_left": 250,
    "ring_rear_right": 250,
    "ring_side_left": 0,
    "ring_side_right": 0,
}


@dataclass(frozen=True)
class CameraSample:
    """One camera observation before it is serialized to AD-GS."""

    camera_id: int
    camera_name: str
    frame_id: int
    timestamp_ns: int
    source_path: Path
    width: int
    height: int
    K: np.ndarray
    R: np.ndarray
    T: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one AV2 sensor log into AD-GS camera-major data."
    )
    parser.add_argument(
        "src",
        type=Path,
        help="AV2 root (for example data/av2), sensor root, or sensor/train root.",
    )
    parser.add_argument(
        "dst",
        type=Path,
        help="Processed AV2 output root. The scene UUID is created below it.",
    )
    parser.add_argument("scene", help="AV2 log UUID to convert.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--train_fraction", type=float, default=0.5)
    parser.add_argument(
        "--downsample_ratio",
        type=float,
        default=1.0,
        help="Deterministic per-sweep LiDAR retention ratio in (0, 1].",
    )
    parser.add_argument(
        "--max_camera_lidar_delta_ms",
        type=float,
        default=75.0,
        help="Fail if no train image is sufficiently close to a train LiDAR sweep.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate raw files, poses, ordering, crop sizes, and exact split without writing.",
    )
    return parser.parse_args()


def _resolve_split_root(src: Path, split: str) -> Path:
    """Accept the common AV2 root spellings without guessing a missing path."""

    src = src.expanduser().resolve()
    candidates = (
        src / "sensor" / split,
        src / split,
        src,
    )
    for candidate in candidates:
        if candidate.is_dir():
            # A split root contains UUID log directories. Avoid accepting data/av2
            # itself merely because it is non-empty.
            if any(path.is_dir() and (path / "sensors").is_dir() for path in candidate.iterdir()):
                return candidate
    raise FileNotFoundError(
        f"Cannot locate AV2 sensor/{split} below {src}. "
        f"Expected {src / 'sensor' / split} or an explicit split directory."
    )


def _require_scene_layout(split_root: Path, scene: str) -> Path:
    log_dir = split_root / scene
    required = (
        log_dir / "calibration" / "egovehicle_SE3_sensor.feather",
        log_dir / "calibration" / "intrinsics.feather",
        log_dir / "city_SE3_egovehicle.feather",
        log_dir / "annotations.feather",
        log_dir / "sensors" / "lidar",
    )
    missing = [str(path) for path in required if not path.exists()]
    for camera_name in CAMERA_ORDER:
        camera_dir = log_dir / "sensors" / "cameras" / camera_name
        if not camera_dir.is_dir():
            missing.append(str(camera_dir))
    if missing:
        raise FileNotFoundError("Incomplete AV2 log; missing:\n  " + "\n  ".join(missing))
    return log_dir


def _validate_timestamps(name: str, timestamps_ns: np.ndarray) -> None:
    if timestamps_ns.ndim != 1 or timestamps_ns.size < 2:
        raise ValueError(f"{name}: expected at least two timestamps, got {timestamps_ns.shape}")
    gaps = np.diff(timestamps_ns)
    if np.any(gaps <= 0):
        raise ValueError(f"{name}: timestamps are not strictly increasing")
    median_gap = float(np.median(gaps))
    max_gap = float(np.max(gaps))
    if max_gap > median_gap * 1.5:
        raise ValueError(
            f"{name}: internal frame gap {max_gap / 1e6:.3f} ms exceeds "
            f"1.5x the median {median_gap / 1e6:.3f} ms; a frame may be missing"
        )


def _validate_partition(
    sensor_ids: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    train_fraction: float,
) -> None:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    eval_indices = np.asarray(eval_indices, dtype=np.int64)
    if train_fraction == 1.0:
        all_indices = np.arange(sensor_ids.size, dtype=np.int64)
        if not (
            np.array_equal(train_indices, all_indices)
            and np.array_equal(eval_indices, all_indices)
        ):
            raise AssertionError(
                "SplatAD's fraction=1 split must use every observation for both splits"
            )
        return
    if np.intersect1d(train_indices, eval_indices).size:
        raise AssertionError("SplatAD train/eval indices overlap")
    if not np.array_equal(
        np.sort(np.concatenate((train_indices, eval_indices))),
        np.arange(sensor_ids.size, dtype=np.int64),
    ):
        raise AssertionError("SplatAD train/eval indices do not partition all observations")
    for sensor_id in np.unique(sensor_ids):
        sensor_global = np.flatnonzero(sensor_ids == sensor_id)
        expected = math.ceil(sensor_global.size * train_fraction)
        actual = np.intersect1d(sensor_global, train_indices).size
        if actual != expected:
            raise AssertionError(
                f"sensor {sensor_id}: expected {expected} train samples, got {actual}"
            )


def _load_sources(
    loader: AV2SensorDataLoader,
    scene: str,
) -> Tuple[
    Dict[str, List[Path]],
    Dict[str, np.ndarray],
    Dict[str, PinholeCamera],
    np.ndarray,
]:
    paths_by_camera: Dict[str, List[Path]] = {}
    timestamps_by_camera: Dict[str, np.ndarray] = {}
    cameras: Dict[str, PinholeCamera] = {}
    counts: Dict[str, int] = {}

    for camera_name in CAMERA_ORDER:
        paths = list(loader.get_ordered_log_cam_fpaths(scene, camera_name))
        if not paths:
            raise FileNotFoundError(f"{scene}/{camera_name}: no camera images")
        if any(not path.is_file() for path in paths):
            missing = [str(path) for path in paths if not path.is_file()]
            raise FileNotFoundError(
                f"{scene}/{camera_name}: SDK returned missing files:\n  " + "\n  ".join(missing)
            )
        timestamps_ns = np.asarray([int(path.stem) for path in paths], dtype=np.int64)
        _validate_timestamps(camera_name, timestamps_ns)
        pinhole = loader.get_log_pinhole_camera(scene, camera_name)
        crop = BOTTOM_CROP[camera_name]
        if crop >= int(pinhole.intrinsics.height_px):
            raise ValueError(
                f"{camera_name}: bottom crop {crop} is invalid for "
                f"height {pinhole.intrinsics.height_px}"
            )
        if not np.isfinite(pinhole.intrinsics.K).all():
            raise ValueError(f"{camera_name}: non-finite camera intrinsics")
        if pinhole.intrinsics.K[0, 0] <= 0 or pinhole.intrinsics.K[1, 1] <= 0:
            raise ValueError(f"{camera_name}: non-positive focal length")

        paths_by_camera[camera_name] = paths
        timestamps_by_camera[camera_name] = timestamps_ns
        cameras[camera_name] = pinhole
        counts[camera_name] = len(paths)

    count_values = np.asarray(list(counts.values()), dtype=np.int64)
    count_summary = ", ".join(f"{name}={counts[name]}" for name in CAMERA_ORDER)
    print(f"Camera counts: {count_summary}")
    if np.ptp(count_values) > 1:
        raise ValueError(
            "Camera counts differ by more than one observation; refusing to hide a likely "
            f"missing channel sample: {count_summary}"
        )
    if np.ptp(count_values) == 1:
        print(
            "[NOTICE] Camera channels differ by one endpoint observation. "
            "Each channel will use its own local ordinal LINSPACE split."
        )

    lidar_timestamps_ns = np.asarray(
        loader.get_ordered_log_lidar_timestamps(scene), dtype=np.int64
    )
    _validate_timestamps("lidar", lidar_timestamps_ns)
    for timestamp_ns in lidar_timestamps_ns:
        lidar_path = loader.get_lidar_fpath(scene, int(timestamp_ns))
        if lidar_path is None or not lidar_path.is_file():
            raise FileNotFoundError(
                f"{scene}/lidar: missing sweep for timestamp {int(timestamp_ns)}"
            )

    return paths_by_camera, timestamps_by_camera, cameras, lidar_timestamps_ns


def _validate_image_sizes(
    paths_by_camera: Mapping[str, Sequence[Path]],
    cameras: Mapping[str, PinholeCamera],
) -> None:
    for camera_name in CAMERA_ORDER:
        expected = (
            int(cameras[camera_name].intrinsics.width_px),
            int(cameras[camera_name].intrinsics.height_px),
        )
        for path in paths_by_camera[camera_name]:
            try:
                with Image.open(path) as image:
                    actual = image.size
            except Exception as exc:
                raise ValueError(f"Cannot read AV2 image {path}: {exc}") from exc
            if actual != expected:
                raise ValueError(
                    f"{path}: image size {actual} does not match calibration {expected}"
                )


def _make_camera_samples(
    paths_by_camera: Mapping[str, Sequence[Path]],
    timestamps_by_camera: Mapping[str, np.ndarray],
    cameras: Mapping[str, PinholeCamera],
    poses: Mapping[int, SE3],
    reference_SE3_city: SE3,
) -> List[CameraSample]:
    """Flatten observations in SplatAD's camera-major sensor order."""

    samples: List[CameraSample] = []
    # The reference Argoverse2 parser consumes one complete chronological
    # camera stream before moving to the next sensor.  The split helper is
    # sensor-aware either way, but matching this global order also preserves
    # AD-GS's fixed-seed camera sampling schedule.
    for camera_id, camera_name in enumerate(CAMERA_ORDER):
        paths = paths_by_camera[camera_name]
        for frame_id in range(len(paths)):
            timestamp_ns = int(timestamps_by_camera[camera_name][frame_id])
            if timestamp_ns not in poses:
                raise RuntimeError(
                    f"{camera_name} frame {frame_id}: no ego pose for {timestamp_ns}"
                )
            pinhole = cameras[camera_name]
            local_SE3_cam = (
                reference_SE3_city.compose(poses[timestamp_ns]).compose(pinhole.ego_SE3_cam)
            )
            cam_SE3_local = local_SE3_cam.inverse().transform_matrix
            R = np.asarray(cam_SE3_local[:3, :3], dtype=np.float32)
            T = np.asarray(cam_SE3_local[:3, 3], dtype=np.float32)
            determinant = float(np.linalg.det(R))
            if not np.isfinite(R).all() or not np.isfinite(T).all():
                raise ValueError(f"{camera_name} frame {frame_id}: non-finite camera pose")
            if not np.isclose(determinant, 1.0, atol=1e-3):
                raise ValueError(
                    f"{camera_name} frame {frame_id}: invalid rotation determinant {determinant}"
                )
            samples.append(
                CameraSample(
                    camera_id=camera_id,
                    camera_name=camera_name,
                    frame_id=frame_id,
                    timestamp_ns=timestamp_ns,
                    source_path=paths[frame_id],
                    width=int(pinhole.intrinsics.width_px),
                    height=int(pinhole.intrinsics.height_px) - BOTTOM_CROP[camera_name],
                    K=np.asarray(pinhole.intrinsics.K, dtype=np.float32),
                    R=R,
                    T=T,
                )
            )
    return samples


def _build_metadata(
    samples: Sequence[CameraSample],
    lidar_timestamps_ns: np.ndarray,
    reference_timestamp_ns: int,
    train_fraction: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    camera_ids = np.asarray([sample.camera_id for sample in samples], dtype=np.int32)
    train_indices, eval_indices = get_splatad_split(camera_ids, train_fraction)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    eval_indices = np.asarray(eval_indices, dtype=np.int64)
    _validate_partition(camera_ids, train_indices, eval_indices, train_fraction)

    is_val_list = np.ones(len(samples), dtype=np.bool_)
    is_val_list[train_indices] = False
    if train_fraction < 1.0 and not np.array_equal(
        np.flatnonzero(is_val_list), np.sort(eval_indices)
    ):
        raise AssertionError("is_val_list is not the exact complement of SplatAD train indices")

    # AV2 stores both 32-beam lidars in one feather sweep, while SplatAD exposes
    # them as two sensor-major observation streams.  Their timestamps and local
    # ordinals are identical, so split both streams independently and deduplicate
    # the selected physical sweep ids before loading the feather files.
    num_physical_lidar_sweeps = int(lidar_timestamps_ns.size)
    lidar_sensor_ids = np.repeat(
        np.arange(2, dtype=np.int32), num_physical_lidar_sweeps
    )
    lidar_physical_sweep_ids = np.tile(
        np.arange(num_physical_lidar_sweeps, dtype=np.int32), 2
    )
    lidar_observation_timestamps_ns = np.tile(lidar_timestamps_ns, 2)
    lidar_train_observation_indices, lidar_eval_observation_indices = (
        get_splatad_split(lidar_sensor_ids, train_fraction)
    )
    lidar_train_observation_indices = np.asarray(
        lidar_train_observation_indices, dtype=np.int64
    )
    lidar_eval_observation_indices = np.asarray(
        lidar_eval_observation_indices, dtype=np.int64
    )
    _validate_partition(
        lidar_sensor_ids,
        lidar_train_observation_indices,
        lidar_eval_observation_indices,
        train_fraction,
    )
    lidar_is_val_list = np.ones(lidar_sensor_ids.size, dtype=np.bool_)
    lidar_is_val_list[lidar_train_observation_indices] = False
    per_sensor_lidar_is_val = lidar_is_val_list.reshape(
        2, num_physical_lidar_sweeps
    )
    if not np.array_equal(
        per_sensor_lidar_is_val[0], per_sensor_lidar_is_val[1]
    ):
        raise AssertionError(
            "Synchronous AV2 LiDAR streams produced different LINSPACE ordinals"
        )
    physical_lidar_is_val_list = per_sensor_lidar_is_val[0].copy()
    lidar_train_indices = np.flatnonzero(
        ~physical_lidar_is_val_list
    ).astype(np.int64)

    camera_timestamps_ns = np.asarray(
        [sample.timestamp_ns for sample in samples], dtype=np.int64
    )
    camera_nearest_lidar_ids = np.empty(len(samples), dtype=np.int32)
    camera_nearest_lidar_delta_ns = np.empty(len(samples), dtype=np.int64)
    camera_nearest_train_lidar_ids = np.empty(len(samples), dtype=np.int32)
    camera_nearest_train_lidar_delta_ns = np.empty(len(samples), dtype=np.int64)
    train_lidar_timestamps_ns = lidar_timestamps_ns[lidar_train_indices]
    for sample_index, timestamp_ns in enumerate(camera_timestamps_ns):
        lidar_id = int(np.argmin(np.abs(lidar_timestamps_ns - timestamp_ns)))
        train_lidar_local_id = int(
            np.argmin(np.abs(train_lidar_timestamps_ns - timestamp_ns))
        )
        train_lidar_id = int(lidar_train_indices[train_lidar_local_id])
        camera_nearest_lidar_ids[sample_index] = lidar_id
        camera_nearest_lidar_delta_ns[sample_index] = abs(
            int(lidar_timestamps_ns[lidar_id]) - int(timestamp_ns)
        )
        camera_nearest_train_lidar_ids[sample_index] = train_lidar_id
        camera_nearest_train_lidar_delta_ns[sample_index] = abs(
            int(lidar_timestamps_ns[train_lidar_id]) - int(timestamp_ns)
        )

    # These global camera indices are also the sequential output image IDs. The
    # explicit Mx7 mappings avoid idx % num_cam when one AV2 channel contains an
    # additional endpoint observation.
    lidar_nearest_camera_indices = np.empty(
        (lidar_timestamps_ns.size, len(CAMERA_ORDER)), dtype=np.int32
    )
    lidar_nearest_train_camera_indices = np.empty_like(lidar_nearest_camera_indices)
    lidar_nearest_camera_delta_ns = np.empty_like(
        lidar_nearest_camera_indices, dtype=np.int64
    )
    lidar_nearest_train_camera_delta_ns = np.empty_like(
        lidar_nearest_camera_indices, dtype=np.int64
    )
    for camera_id, _ in enumerate(CAMERA_ORDER):
        camera_global = np.flatnonzero(camera_ids == camera_id)
        train_camera_global = camera_global[~is_val_list[camera_global]]
        if not train_camera_global.size:
            raise ValueError(f"camera {camera_id}: no train observations")
        camera_times = camera_timestamps_ns[camera_global]
        train_camera_times = camera_timestamps_ns[train_camera_global]
        for lidar_id, lidar_timestamp_ns in enumerate(lidar_timestamps_ns):
            nearest_local = int(np.argmin(np.abs(camera_times - lidar_timestamp_ns)))
            nearest_train_local = int(
                np.argmin(np.abs(train_camera_times - lidar_timestamp_ns))
            )
            nearest_global = int(camera_global[nearest_local])
            nearest_train_global = int(train_camera_global[nearest_train_local])
            lidar_nearest_camera_indices[lidar_id, camera_id] = nearest_global
            lidar_nearest_train_camera_indices[lidar_id, camera_id] = nearest_train_global
            lidar_nearest_camera_delta_ns[lidar_id, camera_id] = abs(
                int(camera_timestamps_ns[nearest_global]) - int(lidar_timestamp_ns)
            )
            lidar_nearest_train_camera_delta_ns[lidar_id, camera_id] = abs(
                int(camera_timestamps_ns[nearest_train_global]) - int(lidar_timestamp_ns)
            )

    time_stamps = np.asarray(
        [(sample.timestamp_ns - reference_timestamp_ns) / 1e9 for sample in samples],
        dtype=np.float64,
    )
    physical_lidar_time_stamps = (
        lidar_timestamps_ns - np.int64(reference_timestamp_ns)
    ).astype(np.float64) / 1e9
    lidar_time_stamps = np.tile(physical_lidar_time_stamps, 2)
    normalization_time_stamps = np.concatenate(
        [time_stamps, lidar_time_stamps]
    )
    sensor_time_min, sensor_time_max = sensor_time_bounds(
        normalization_time_stamps
    )
    frame_gap = normalized_train_frame_gap(
        time_stamps,
        camera_ids,
        is_val_list,
        normalization_time_stamps=normalization_time_stamps,
    )
    image_widths = np.asarray(
        [sample.width for sample in samples], dtype=np.int32
    )
    image_heights = np.asarray(
        [sample.height for sample in samples], dtype=np.int32
    )

    metadata = {
        "R": np.stack([sample.R for sample in samples]).astype(np.float32),
        "T": np.stack([sample.T for sample in samples]).astype(np.float32),
        "K": np.stack([sample.K for sample in samples]).astype(np.float32),
        "camera_ids": camera_ids,
        "camera_names": np.asarray(CAMERA_ORDER, dtype=np.str_),
        "camera_names_per_image": np.asarray(
            [sample.camera_name for sample in samples], dtype=np.str_
        ),
        "frame_ids": np.asarray([sample.frame_id for sample in samples], dtype=np.int32),
        "image_file_names": np.asarray(
            [f"{index:06d}.jpg" for index in range(len(samples))], dtype=np.str_
        ),
        "camera_timestamps_ns": camera_timestamps_ns,
        "time_stamps": time_stamps,
        "is_val_list": is_val_list,
        "dataset_type": np.asarray("av2", dtype=np.str_),
        "image_widths": image_widths,
        "image_heights": image_heights,
        # Keep concise aliases for external consumers while AD-GS uses the
        # image_* names shared by the nuScenes and Waymo converters.
        "widths": image_widths,
        "heights": image_heights,
        "camera_order": np.asarray(CAMERA_ORDER, dtype=np.str_),
        "camera_crop_bottom": np.asarray(
            [BOTTOM_CROP[name] for name in CAMERA_ORDER], dtype=np.int32
        ),
        "camera_sample_counts": np.bincount(
            camera_ids, minlength=len(CAMERA_ORDER)
        ).astype(np.int32),
        "camera_train_counts": np.bincount(
            camera_ids[~is_val_list], minlength=len(CAMERA_ORDER)
        ).astype(np.int32),
        "camera_layout": np.asarray("camera_major", dtype=np.str_),
        "split_type": np.asarray("linspace", dtype=np.str_),
        "split_scope": np.asarray("per_sensor", dtype=np.str_),
        "train_split_fraction": np.asarray(train_fraction, dtype=np.float32),
        "frame_gap": np.asarray(frame_gap, dtype=np.float32),
        "sensor_time_min": np.asarray(sensor_time_min, dtype=np.float64),
        "sensor_time_max": np.asarray(sensor_time_max, dtype=np.float64),
        "sensor_time_duration": np.asarray(
            sensor_time_max - sensor_time_min, dtype=np.float64
        ),
        "time_normalization_scope": np.asarray(
            "all_cameras_all_lidars", dtype=np.str_
        ),
        "reference_timestamp_ns": np.asarray(reference_timestamp_ns, dtype=np.int64),
        "lidar_names": np.asarray(("lidar_up", "lidar_down"), dtype=np.str_),
        "lidar_sensor_ids": lidar_sensor_ids,
        "lidar_frame_ids": lidar_physical_sweep_ids.copy(),
        "lidar_physical_sweep_ids": lidar_physical_sweep_ids,
        "lidar_timestamps_ns": lidar_observation_timestamps_ns.astype(np.int64),
        "lidar_time_stamps": lidar_time_stamps.astype(np.float64),
        "lidar_is_val_list": lidar_is_val_list,
        "physical_lidar_timestamps_ns": lidar_timestamps_ns.astype(np.int64),
        "physical_lidar_time_stamps": physical_lidar_time_stamps.astype(
            np.float64
        ),
        "physical_lidar_is_val_list": physical_lidar_is_val_list,
        "camera_nearest_lidar_ids": camera_nearest_lidar_ids,
        "camera_nearest_lidar_delta_ns": camera_nearest_lidar_delta_ns,
        "camera_nearest_train_lidar_ids": camera_nearest_train_lidar_ids,
        "camera_nearest_train_lidar_frame_ids": camera_nearest_train_lidar_ids.copy(),
        "camera_nearest_train_lidar_delta_ns": camera_nearest_train_lidar_delta_ns,
        "lidar_nearest_camera_indices": lidar_nearest_camera_indices,
        "lidar_nearest_camera_frame_ids": np.asarray(
            [
                [samples[index].frame_id for index in row]
                for row in lidar_nearest_camera_indices
            ],
            dtype=np.int32,
        ),
        "lidar_nearest_camera_delta_ns": lidar_nearest_camera_delta_ns,
        "lidar_nearest_train_camera_indices": lidar_nearest_train_camera_indices,
        "lidar_nearest_train_camera_frame_ids": np.asarray(
            [
                [samples[index].frame_id for index in row]
                for row in lidar_nearest_train_camera_indices
            ],
            dtype=np.int32,
        ),
        "lidar_nearest_train_camera_delta_ns": lidar_nearest_train_camera_delta_ns,
        "pcd_uses_train_lidar_only": np.asarray(True),
        "pcd_color_uses_train_cameras_only": np.asarray(True),
    }
    return metadata, train_indices, lidar_train_indices


def _validate_sensor_mappings(
    samples: Sequence[CameraSample],
    lidar_timestamps_ns: np.ndarray,
    metadata: Mapping[str, np.ndarray],
    train_lidar_indices: np.ndarray,
    max_camera_lidar_delta_ms: float,
) -> None:
    """Verify every serialized camera/LiDAR association before any output write."""

    num_images = len(samples)
    num_lidar = int(lidar_timestamps_ns.size)
    num_lidar_observations = num_lidar * 2
    num_cameras = len(CAMERA_ORDER)
    camera_ids = np.asarray(metadata["camera_ids"], dtype=np.int32)
    camera_timestamps_ns = np.asarray(
        metadata["camera_timestamps_ns"], dtype=np.int64
    )
    is_val = np.asarray(metadata["is_val_list"], dtype=np.bool_)

    expected_camera_names = np.asarray(CAMERA_ORDER, dtype=np.str_)
    if not np.array_equal(metadata["camera_names"], expected_camera_names):
        raise AssertionError("camera_names does not match the SplatAD camera order")
    per_image_names = np.asarray(metadata["camera_names_per_image"], dtype=np.str_)
    expected_per_image_names = np.asarray(
        [sample.camera_name for sample in samples], dtype=np.str_
    )
    if not np.array_equal(per_image_names, expected_per_image_names):
        raise AssertionError("camera_names_per_image is not image-aligned")

    if camera_ids.shape != (num_images,) or is_val.shape != (num_images,):
        raise AssertionError("Camera metadata arrays are not image-aligned")
    if camera_timestamps_ns.shape != (num_images,):
        raise AssertionError("Camera timestamp metadata is not image-aligned")
    expected_shapes = {
        "K": (num_images, 3, 3),
        "R": (num_images, 3, 3),
        "T": (num_images, 3),
        "frame_ids": (num_images,),
        "time_stamps": (num_images,),
        "image_widths": (num_images,),
        "image_heights": (num_images,),
        "image_file_names": (num_images,),
        "lidar_sensor_ids": (num_lidar_observations,),
        "lidar_frame_ids": (num_lidar_observations,),
        "lidar_physical_sweep_ids": (num_lidar_observations,),
        "lidar_timestamps_ns": (num_lidar_observations,),
        "lidar_time_stamps": (num_lidar_observations,),
        "lidar_is_val_list": (num_lidar_observations,),
        "physical_lidar_timestamps_ns": (num_lidar,),
        "physical_lidar_time_stamps": (num_lidar,),
        "physical_lidar_is_val_list": (num_lidar,),
    }
    for key, expected_shape in expected_shapes.items():
        if np.asarray(metadata[key]).shape != expected_shape:
            raise AssertionError(
                f"{key} must have shape {expected_shape}, got "
                f"{np.asarray(metadata[key]).shape}"
            )
    expected_lidar_sensor_ids = np.repeat(
        np.arange(2, dtype=np.int32), num_lidar
    )
    expected_lidar_frame_ids = np.tile(
        np.arange(num_lidar, dtype=np.int32), 2
    )
    expected_lidar_timestamps_ns = np.tile(lidar_timestamps_ns, 2)
    if not np.array_equal(
        np.asarray(metadata["lidar_names"], dtype=np.str_),
        np.asarray(("lidar_up", "lidar_down"), dtype=np.str_),
    ):
        raise AssertionError("lidar_names does not match SplatAD AV2 sensor order")
    if not np.array_equal(
        np.asarray(metadata["lidar_sensor_ids"], dtype=np.int32),
        expected_lidar_sensor_ids,
    ):
        raise AssertionError("lidar_sensor_ids is not sensor-major [up, down]")
    if not np.array_equal(
        np.asarray(metadata["lidar_frame_ids"], dtype=np.int32),
        expected_lidar_frame_ids,
    ):
        raise AssertionError("lidar_frame_ids is not local-ordinal sensor-major")
    if not np.array_equal(
        np.asarray(metadata["lidar_physical_sweep_ids"], dtype=np.int32),
        expected_lidar_frame_ids,
    ):
        raise AssertionError("lidar_physical_sweep_ids is inconsistent")
    if not np.array_equal(
        np.asarray(metadata["lidar_timestamps_ns"], dtype=np.int64),
        expected_lidar_timestamps_ns,
    ):
        raise AssertionError("LiDAR observation timestamps are not sensor-major")
    if not np.array_equal(
        np.asarray(metadata["physical_lidar_timestamps_ns"], dtype=np.int64),
        lidar_timestamps_ns,
    ):
        raise AssertionError("physical LiDAR timestamps are inconsistent")
    lidar_is_val = np.asarray(metadata["lidar_is_val_list"], dtype=np.bool_)
    expected_lidar_train, _ = get_splatad_split(
        expected_lidar_sensor_ids,
        float(np.asarray(metadata["train_split_fraction"]).item()),
    )
    expected_lidar_is_val = np.ones(num_lidar_observations, dtype=np.bool_)
    expected_lidar_is_val[np.asarray(expected_lidar_train, dtype=np.int64)] = False
    if not np.array_equal(lidar_is_val, expected_lidar_is_val):
        raise AssertionError("LiDAR split is not per-sensor SplatAD LINSPACE")
    per_sensor_lidar_is_val = lidar_is_val.reshape(2, num_lidar)
    physical_lidar_is_val = np.asarray(
        metadata["physical_lidar_is_val_list"], dtype=np.bool_
    )
    if not (
        np.array_equal(per_sensor_lidar_is_val[0], per_sensor_lidar_is_val[1])
        and np.array_equal(physical_lidar_is_val, per_sensor_lidar_is_val[0])
    ):
        raise AssertionError("Physical and per-sensor LiDAR split masks differ")
    frame_gap = float(np.asarray(metadata["frame_gap"]).item())
    if not np.isfinite(frame_gap) or frame_gap <= 0.0:
        raise AssertionError("frame_gap must be positive and finite")
    expected_file_names = np.asarray(
        [f"{index:06d}.jpg" for index in range(num_images)], dtype=np.str_
    )
    if not np.array_equal(metadata["image_file_names"], expected_file_names):
        raise AssertionError("image_file_names is not sequential and image-aligned")

    camera_nearest_lidar = np.asarray(
        metadata["camera_nearest_lidar_ids"], dtype=np.int64
    )
    camera_nearest_delta = np.asarray(
        metadata["camera_nearest_lidar_delta_ns"], dtype=np.int64
    )
    if camera_nearest_lidar.shape != (num_images,):
        raise AssertionError("camera_nearest_lidar_ids has an invalid shape")
    if np.any((camera_nearest_lidar < 0) | (camera_nearest_lidar >= num_lidar)):
        raise AssertionError("camera_nearest_lidar_ids contains an out-of-range sweep")
    expected_camera_delta = np.abs(
        lidar_timestamps_ns[camera_nearest_lidar] - camera_timestamps_ns
    )
    if not np.array_equal(camera_nearest_delta, expected_camera_delta):
        raise AssertionError("camera_nearest_lidar_delta_ns is inconsistent")

    camera_nearest_train_lidar = np.asarray(
        metadata["camera_nearest_train_lidar_ids"], dtype=np.int64
    )
    camera_nearest_train_lidar_frames = np.asarray(
        metadata["camera_nearest_train_lidar_frame_ids"], dtype=np.int64
    )
    camera_nearest_train_delta = np.asarray(
        metadata["camera_nearest_train_lidar_delta_ns"], dtype=np.int64
    )
    if not (
        camera_nearest_train_lidar.shape
        == camera_nearest_train_lidar_frames.shape
        == camera_nearest_train_delta.shape
        == (num_images,)
    ):
        raise AssertionError("camera-to-train-LiDAR arrays are not image-aligned")
    if not np.array_equal(
        camera_nearest_train_lidar, camera_nearest_train_lidar_frames
    ):
        raise AssertionError("train LiDAR ids and frame ids are inconsistent")
    if np.any(
        (camera_nearest_train_lidar < 0)
        | (camera_nearest_train_lidar >= num_lidar)
    ):
        raise AssertionError("camera_nearest_train_lidar_ids is out of range")
    if np.any(physical_lidar_is_val[camera_nearest_train_lidar]):
        raise AssertionError("camera_nearest_train_lidar_ids selects eval LiDAR")
    expected_train_lidar_delta = np.abs(
        lidar_timestamps_ns[camera_nearest_train_lidar] - camera_timestamps_ns
    )
    if not np.array_equal(
        camera_nearest_train_delta, expected_train_lidar_delta
    ):
        raise AssertionError("camera_nearest_train_lidar_delta_ns is inconsistent")

    mapping_specs = (
        (
            "lidar_nearest_camera_indices",
            "lidar_nearest_camera_frame_ids",
            "lidar_nearest_camera_delta_ns",
            False,
        ),
        (
            "lidar_nearest_train_camera_indices",
            "lidar_nearest_train_camera_frame_ids",
            "lidar_nearest_train_camera_delta_ns",
            True,
        ),
    )
    for index_key, frame_key, delta_key, must_be_train in mapping_specs:
        indices = np.asarray(metadata[index_key], dtype=np.int64)
        frame_ids = np.asarray(metadata[frame_key], dtype=np.int64)
        deltas = np.asarray(metadata[delta_key], dtype=np.int64)
        expected_shape = (num_lidar, num_cameras)
        if (
            indices.shape != expected_shape
            or frame_ids.shape != expected_shape
            or deltas.shape != expected_shape
        ):
            raise AssertionError(
                f"{index_key} and companion arrays must be {expected_shape}"
            )
        if np.any((indices < 0) | (indices >= num_images)):
            raise AssertionError(f"{index_key} contains an out-of-range image")

        for lidar_id in range(num_lidar):
            for camera_id in range(num_cameras):
                image_id = int(indices[lidar_id, camera_id])
                sample = samples[image_id]
                if sample.camera_id != camera_id:
                    raise AssertionError(
                        f"{index_key}[{lidar_id}, {camera_id}] selects "
                        f"camera {sample.camera_id}"
                    )
                if frame_ids[lidar_id, camera_id] != sample.frame_id:
                    raise AssertionError(f"{frame_key} is inconsistent with frame_ids")
                expected_delta = abs(
                    sample.timestamp_ns - int(lidar_timestamps_ns[lidar_id])
                )
                if deltas[lidar_id, camera_id] != expected_delta:
                    raise AssertionError(f"{delta_key} is inconsistent with timestamps")
                if must_be_train and is_val[image_id]:
                    raise AssertionError(f"{index_key} selects an eval image")

    train_delta = np.asarray(
        metadata["lidar_nearest_train_camera_delta_ns"], dtype=np.int64
    )
    max_delta_ns = int(round(max_camera_lidar_delta_ms * 1e6))
    selected_delta = train_delta[np.asarray(train_lidar_indices, dtype=np.int64)]
    bad = np.argwhere(selected_delta > max_delta_ns)
    if bad.size:
        train_row, camera_id = bad[0]
        lidar_id = int(train_lidar_indices[int(train_row)])
        raise RuntimeError(
            f"LiDAR frame {lidar_id} -> {CAMERA_ORDER[int(camera_id)]}: nearest "
            f"train image is {selected_delta[train_row, camera_id] / 1e6:.3f} ms "
            f"away (limit {max_camera_lidar_delta_ms:.3f} ms)"
        )


def _write_images(samples: Sequence[CameraSample], image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=False)
    for image_id, sample in enumerate(samples):
        destination = image_dir / f"{image_id:06d}.jpg"
        crop = BOTTOM_CROP[sample.camera_name]
        if crop == 0:
            shutil.copy2(sample.source_path, destination)
            continue
        with Image.open(sample.source_path) as image:
            image = image.convert("RGB")
            cropped = image.crop((0, 0, sample.width, sample.height))
            cropped.save(destination, format="JPEG", quality=95, subsampling=0)


def _write_binary_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, times: np.ndarray) -> None:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Invalid xyz shape {xyz.shape}")
    if rgb.shape != xyz.shape or times.shape != (xyz.shape[0], 1):
        raise ValueError(
            f"PLY array shape mismatch: xyz={xyz.shape}, rgb={rgb.shape}, t={times.shape}"
        )
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("t", "<f4"),
        ]
    )
    vertices = np.empty(xyz.shape[0], dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz.T.astype(np.float32)
    vertices["nx"] = 0.0
    vertices["ny"] = 0.0
    vertices["nz"] = 0.0
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T.astype(np.uint8)
    vertices["t"] = times[:, 0].astype(np.float32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float t\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as ply_file:
        ply_file.write(header)
        vertices.tofile(ply_file)


def _build_train_lidar_ply(
    loader: AV2SensorDataLoader,
    scene: str,
    output_path: Path,
    samples: Sequence[CameraSample],
    lidar_timestamps_ns: np.ndarray,
    train_lidar_indices: np.ndarray,
    lidar_nearest_train_camera_indices: np.ndarray,
    cameras: Mapping[str, PinholeCamera],
    poses: Mapping[int, SE3],
    reference_SE3_city: SE3,
    reference_timestamp_ns: int,
    downsample_ratio: float,
    max_camera_lidar_delta_ms: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    xyz_parts: List[np.ndarray] = []
    rgb_parts: List[np.ndarray] = []
    time_parts: List[np.ndarray] = []
    max_delta_ns = int(round(max_camera_lidar_delta_ms * 1e6))

    for sweep_number, lidar_index in enumerate(train_lidar_indices, start=1):
        timestamp_ns = int(lidar_timestamps_ns[lidar_index])
        if timestamp_ns not in poses:
            raise RuntimeError(f"LiDAR frame {lidar_index}: no ego pose for {timestamp_ns}")
        lidar_path = loader.get_lidar_fpath(scene, timestamp_ns)
        if lidar_path is None:
            raise FileNotFoundError(f"No LiDAR file for timestamp {timestamp_ns}")
        sweep = Sweep.from_feather(lidar_path)
        points_ego = np.asarray(sweep.xyz, dtype=np.float64)
        if points_ego.ndim != 2 or points_ego.shape[1] != 3 or not np.isfinite(points_ego).all():
            raise ValueError(f"{lidar_path}: invalid LiDAR xyz array {points_ego.shape}")

        color_sum = np.zeros((points_ego.shape[0], 3), dtype=np.float32)
        color_count = np.zeros(points_ego.shape[0], dtype=np.uint8)
        lidar_pose = poses[timestamp_ns]

        for camera_id, camera_name in enumerate(CAMERA_ORDER):
            camera_sample_index = int(
                lidar_nearest_train_camera_indices[lidar_index, camera_id]
            )
            camera_sample = samples[camera_sample_index]
            if (
                camera_sample.camera_id != camera_id
                or camera_sample.camera_name != camera_name
            ):
                raise AssertionError(
                    "LiDAR-camera map points to the wrong channel: "
                    f"expected {camera_name}, got {camera_sample.camera_name}"
                )
            delta_ns = abs(camera_sample.timestamp_ns - timestamp_ns)
            if delta_ns > max_delta_ns:
                raise RuntimeError(
                    f"LiDAR {timestamp_ns} -> {camera_name}: nearest train image is "
                    f"{delta_ns / 1e6:.3f} ms away (limit {max_camera_lidar_delta_ms:.3f} ms)"
                )
            camera_pose = poses[camera_sample.timestamp_ns]
            uv, _, valid = cameras[camera_name].project_ego_to_img_motion_compensated(
                points_ego,
                city_SE3_ego_cam_t=camera_pose,
                city_SE3_ego_lidar_t=lidar_pose,
            )
            valid = np.asarray(valid, dtype=np.bool_)
            valid &= np.isfinite(uv).all(axis=1)
            valid &= uv[:, 0] >= 0
            valid &= uv[:, 0] < camera_sample.width
            valid &= uv[:, 1] >= 0
            valid &= uv[:, 1] < camera_sample.height
            valid_indices = np.flatnonzero(valid)
            if not valid_indices.size:
                continue
            pixels = np.rint(uv[valid_indices]).astype(np.int64)
            pixels[:, 0] = np.clip(pixels[:, 0], 0, camera_sample.width - 1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, camera_sample.height - 1)
            with Image.open(camera_sample.source_path) as image:
                image_rgb = np.asarray(image.convert("RGB"))[: camera_sample.height, : camera_sample.width]
            color_sum[valid_indices] += image_rgb[pixels[:, 1], pixels[:, 0]].astype(
                np.float32
            )
            color_count[valid_indices] += 1

        visible = color_count > 0
        if not np.any(visible):
            raise RuntimeError(f"LiDAR frame {lidar_index}: no points project into train cameras")
        local_SE3_ego = reference_SE3_city.compose(lidar_pose)
        xyz = local_SE3_ego.transform_point_cloud(points_ego[visible]).astype(np.float32)
        rgb = np.rint(color_sum[visible] / color_count[visible, None]).astype(np.uint8)

        if downsample_ratio < 1.0:
            keep_count = max(1, int(math.floor(xyz.shape[0] * downsample_ratio)))
            choice = np.sort(rng.choice(xyz.shape[0], size=keep_count, replace=False))
            xyz = xyz[choice]
            rgb = rgb[choice]

        relative_time = (timestamp_ns - reference_timestamp_ns) / 1e9
        xyz_parts.append(xyz)
        rgb_parts.append(rgb)
        time_parts.append(np.full((xyz.shape[0], 1), relative_time, dtype=np.float32))
        print(
            f"LiDAR train sweep {sweep_number}/{len(train_lidar_indices)}: "
            f"frame={int(lidar_index)} visible_points={xyz.shape[0]}"
        )

    if not xyz_parts:
        raise RuntimeError("SplatAD split produced no train LiDAR points")
    xyz = np.concatenate(xyz_parts, axis=0)
    rgb = np.concatenate(rgb_parts, axis=0)
    times = np.concatenate(time_parts, axis=0)
    _write_binary_ply(output_path, xyz, rgb, times)
    print(f"Wrote train-only colored LiDAR PLY: {output_path} ({xyz.shape[0]} points)")


def _summary(
    scene: str,
    samples: Sequence[CameraSample],
    metadata: Mapping[str, np.ndarray],
    lidar_timestamps_ns: np.ndarray,
) -> Dict[str, object]:
    camera_ids = np.asarray(metadata["camera_ids"])
    is_val = np.asarray(metadata["is_val_list"])
    per_camera = {}
    for camera_id, camera_name in enumerate(CAMERA_ORDER):
        selected = camera_ids == camera_id
        per_camera[camera_name] = {
            "total": int(selected.sum()),
            "train": int(np.logical_and(selected, ~is_val).sum()),
            "eval": int(np.logical_and(selected, is_val).sum()),
            "bottom_crop": BOTTOM_CROP[camera_name],
        }
    return {
        "scene": scene,
        "dataset_type": "av2",
        "layout": "camera-major",
        "camera_order": list(CAMERA_ORDER),
        "camera_observations": len(samples),
        "camera_split": per_camera,
        "lidar_observations": int(
            np.asarray(metadata["lidar_sensor_ids"]).size
        ),
        "physical_lidar_sweeps": int(lidar_timestamps_ns.size),
        "lidar_train": int((~np.asarray(metadata["lidar_is_val_list"])).sum()),
        "lidar_eval": int(np.asarray(metadata["lidar_is_val_list"]).sum()),
    }


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError("--train_fraction must be in (0, 1]")
    if not 0.0 < args.downsample_ratio <= 1.0:
        raise ValueError("--downsample_ratio must be in (0, 1]")
    if args.max_camera_lidar_delta_ms <= 0.0:
        raise ValueError("--max_camera_lidar_delta_ms must be positive")

    split_root = _resolve_split_root(args.src, args.split)
    log_dir = _require_scene_layout(split_root, args.scene)
    loader = AV2SensorDataLoader(split_root, split_root)
    if args.scene not in loader.get_log_ids():
        raise FileNotFoundError(
            f"Scene {args.scene} is not indexed by AV2 SDK under {split_root}"
        )

    paths_by_camera, timestamps_by_camera, cameras, lidar_timestamps_ns = _load_sources(
        loader, args.scene
    )
    _validate_image_sizes(paths_by_camera, cameras)
    poses = read_city_SE3_ego(log_dir)
    required_timestamps = set(lidar_timestamps_ns.tolist())
    for timestamps_ns in timestamps_by_camera.values():
        required_timestamps.update(timestamps_ns.tolist())
    missing_poses = sorted(timestamp for timestamp in required_timestamps if timestamp not in poses)
    if missing_poses:
        preview = ", ".join(str(timestamp) for timestamp in missing_poses[:10])
        raise RuntimeError(
            f"{args.scene}: missing {len(missing_poses)} required ego poses; first: {preview}"
        )

    reference_timestamp_ns = min(required_timestamps)
    reference_SE3_city = poses[reference_timestamp_ns].inverse()
    samples = _make_camera_samples(
        paths_by_camera,
        timestamps_by_camera,
        cameras,
        poses,
        reference_SE3_city,
    )
    metadata, train_camera_indices, train_lidar_indices = _build_metadata(
        samples,
        lidar_timestamps_ns,
        reference_timestamp_ns,
        args.train_fraction,
    )
    _validate_sensor_mappings(
        samples=samples,
        lidar_timestamps_ns=lidar_timestamps_ns,
        metadata=metadata,
        train_lidar_indices=train_lidar_indices,
        max_camera_lidar_delta_ms=args.max_camera_lidar_delta_ms,
    )
    summary = _summary(args.scene, samples, metadata, lidar_timestamps_ns)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.validate_only:
        print("Validation complete; --validate_only wrote no files.")
        return

    scene_output = args.dst.expanduser().resolve() / args.scene
    if scene_output.exists() and any(scene_output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output scene {scene_output}. "
            "Use a fresh output directory."
        )
    scene_output.mkdir(parents=True, exist_ok=True)
    _write_images(samples, scene_output / "image")
    np.savez_compressed(scene_output / "meta.npz", **metadata)
    _build_train_lidar_ply(
        loader=loader,
        scene=args.scene,
        output_path=scene_output / "points3d.ply",
        samples=samples,
        lidar_timestamps_ns=lidar_timestamps_ns,
        train_lidar_indices=train_lidar_indices,
        lidar_nearest_train_camera_indices=metadata[
            "lidar_nearest_train_camera_indices"
        ],
        cameras=cameras,
        poses=poses,
        reference_SE3_city=reference_SE3_city,
        reference_timestamp_ns=reference_timestamp_ns,
        downsample_ratio=args.downsample_ratio,
        max_camera_lidar_delta_ms=args.max_camera_lidar_delta_ms,
        seed=args.seed,
    )
    print(f"AV2 conversion complete: {scene_output}")


if __name__ == "__main__":
    main()
