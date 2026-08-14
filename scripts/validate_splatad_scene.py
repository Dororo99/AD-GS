#!/usr/bin/env python3
"""Fail-fast validation for an AD-GS scene prepared with SplatAD's split."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

try:
    from scripts.prior_storage import (
        load_depth_prior,
        load_flow_prior,
        load_mask_prior,
        resolve_prior_path,
    )
    from scripts.splatad_split import (
        normalize_sensor_times,
        normalized_train_frame_gap,
        sensor_time_bounds,
        splatad_is_val_mask,
    )
except ImportError:  # Direct ``python scripts/validate_splatad_scene.py`` execution.
    from prior_storage import (
        load_depth_prior,
        load_flow_prior,
        load_mask_prior,
        resolve_prior_path,
    )
    from splatad_split import (
        normalize_sensor_times,
        normalized_train_frame_gap,
        sensor_time_bounds,
        splatad_is_val_mask,
    )


CAMERAS = {
    "waymo": ("FRONT", "FRONT_LEFT", "FRONT_RIGHT"),
    "nuscenes": (
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ),
    "av2": (
        "ring_front_center",
        "ring_front_left",
        "ring_front_right",
        "ring_rear_left",
        "ring_rear_right",
        "ring_side_left",
        "ring_side_right",
    ),
}

LIDARS = {
    "waymo": ("TOP",),
    "nuscenes": ("LIDAR_TOP",),
    "av2": ("lidar_up", "lidar_down"),
}


def _scalar(meta, key):
    return np.asarray(meta[key]).item()


def _nearest_distances(values, reference):
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    insertion = np.searchsorted(reference, values)
    left = reference[np.clip(insertion - 1, 0, len(reference) - 1)]
    right = reference[np.clip(insertion, 0, len(reference) - 1)]
    return np.minimum(np.abs(values - left), np.abs(values - right))


def _image_names(image_dir):
    names = sorted(
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not names:
        raise ValueError("No images in {}".format(image_dir))
    return names


def _validate_flow_package(
    flow_package,
    expected_shape,
    sensor_time_min,
    sensor_time_max,
    image_name,
):
    """Validate the exact six-field AD-GS flow payload before training."""
    if len(flow_package) != 6:
        raise ValueError(
            "{} flow must contain time, K, R, T, flow, visibility".format(
                image_name
            )
        )

    target_time, K, R, T, flow, visibility = flow_package
    normalize_sensor_times(target_time, sensor_time_min, sensor_time_max)
    height, width = expected_shape
    arrays = {
        "K": (np.asarray(K), ((3, 3),)),
        "R": (np.asarray(R), ((3, 3),)),
        "T": (np.asarray(T), ((3,), (3, 1))),
        "flow": (np.asarray(flow), ((2, height, width),)),
        "visibility": (np.asarray(visibility), ((height, width),)),
    }
    for field, (array, allowed_shapes) in arrays.items():
        if array.shape not in allowed_shapes:
            raise ValueError(
                "{} {} shape {} is not {}".format(
                    image_name, field, array.shape, allowed_shapes
                )
            )
        try:
            finite = np.isfinite(array).all()
        except TypeError:
            finite = False
        if not finite:
            raise ValueError(
                "{} {} contains non-finite/non-numeric values".format(
                    image_name, field
                )
            )
    visibility = arrays["visibility"][0]
    if visibility.size and (
        float(np.min(visibility)) < 0.0
        or float(np.max(visibility)) > 1.0
    ):
        raise ValueError(
            "{} visibility must lie in [0, 1]".format(image_name)
        )


def _missing_training_artifacts(scene, point_properties):
    """Return top-level outputs required before expensive per-file checks."""
    missing = []
    if "obj" not in point_properties:
        missing.append("points3d.ply[obj]")
    for folder in ("depth", "semantic", "sky", "flow"):
        if not (scene / folder).is_dir():
            missing.append(folder + "/")
    if not (scene / "colmap.ply").is_file():
        missing.append("colmap.ply")
    return missing


def _training_readiness_error(scene, dataset, missing):
    work = scene / ".adgs-priors-work"
    message = (
        "Scene {} is not training-ready; missing {}. The obj field is "
        "generated from semantic masks by the full prior pipeline and must "
        "not be zero-filled. Run `bash scripts/prepare_splatad_priors.sh {} "
        "{} PHYSICAL_GPU`.".format(
            scene, ", ".join(missing), dataset, scene
        )
    )
    if work.is_dir():
        message += (
            " Interrupted staging exists at {}; restarting it requires the "
            "explicit OVERWRITE=1 option, which removes that manifest-marked "
            "staging directory first.".format(work)
        )
    return message


def validate(scene, expected_dataset, metadata_only=False):
    scene = scene.resolve()
    meta_path = scene / ("cameras.npz" if expected_dataset == "waymo" else "meta.npz")
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    image_dir = scene / "image"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    image_names = _image_names(image_dir)

    with np.load(str(meta_path), allow_pickle=True) as meta:
        required = {
            "K", "R", "T", "time_stamps", "camera_ids", "camera_names",
            "is_val_list", "dataset_type", "split_type",
            "train_split_fraction", "frame_gap", "lidar_time_stamps",
            "lidar_sensor_ids", "lidar_names", "lidar_is_val_list",
            "sensor_time_min", "sensor_time_max",
            "sensor_time_duration", "time_normalization_scope",
        }
        missing = sorted(required.difference(meta.files))
        if missing:
            raise ValueError("Missing metadata keys: {}".format(missing))

        dataset_type = str(_scalar(meta, "dataset_type")).lower()
        if dataset_type != expected_dataset:
            raise ValueError(
                "Expected dataset {}, metadata says {}".format(
                    expected_dataset, dataset_type
                )
            )
        if str(_scalar(meta, "split_type")).lower() != "linspace":
            raise ValueError("split_type must be LINSPACE")
        fraction = float(_scalar(meta, "train_split_fraction"))
        if not np.isclose(fraction, 0.5):
            raise ValueError("train_split_fraction must be 0.5, got {}".format(fraction))

        expected_names = CAMERAS[expected_dataset]
        actual_names = tuple(str(name) for name in meta["camera_names"].tolist())
        if actual_names != expected_names:
            raise ValueError(
                "Camera order mismatch: expected {}, got {}".format(
                    expected_names, actual_names
                )
            )

        count = len(image_names)
        aligned = (
            "K", "R", "T", "time_stamps", "camera_ids", "is_val_list",
            "frame_ids", "image_heights", "image_widths",
        )
        for key in aligned:
            if key not in meta.files or len(meta[key]) != count:
                raise ValueError("{} is not aligned to {} images".format(key, count))

        camera_ids = np.asarray(meta["camera_ids"], dtype=np.int64)
        if set(camera_ids.tolist()) != set(range(len(expected_names))):
            raise ValueError("camera_ids are not contiguous in configured camera order")
        times = np.asarray(meta["time_stamps"], dtype=np.float64)
        is_val = np.asarray(meta["is_val_list"], dtype=np.bool_)
        expected_val = splatad_is_val_mask(camera_ids, fraction)
        if not np.array_equal(is_val, expected_val):
            raise ValueError("Camera split is not exact sensor-wise SplatAD LINSPACE")

        camera_summary = {}
        for camera_id, camera_name in enumerate(expected_names):
            selected = camera_ids == camera_id
            sensor_times = times[selected]
            if len(sensor_times) < 2 or np.any(np.diff(sensor_times) <= 0):
                raise ValueError("{} timestamps are not increasing".format(camera_name))
            train_count = int(np.count_nonzero(selected & ~is_val))
            expected_train = int(math.ceil(np.count_nonzero(selected) * 0.5))
            if train_count != expected_train:
                raise ValueError("{} train count mismatch".format(camera_name))
            camera_summary[camera_name] = {
                "total": int(np.count_nonzero(selected)),
                "train": train_count,
                "val": int(np.count_nonzero(selected & is_val)),
            }

        heights = np.asarray(meta["image_heights"], dtype=np.int64)
        widths = np.asarray(meta["image_widths"], dtype=np.int64)
        for index, name in enumerate(image_names):
            with Image.open(str(image_dir / name)) as image:
                if image.size != (int(widths[index]), int(heights[index])):
                    raise ValueError("Image dimension mismatch: {}".format(name))

        lidar_times = np.asarray(meta["lidar_time_stamps"], dtype=np.float64)
        lidar_sensor_ids = np.asarray(meta["lidar_sensor_ids"], dtype=np.int64)
        lidar_is_val = np.asarray(meta["lidar_is_val_list"], dtype=np.bool_)
        if not (
            lidar_times.shape
            == lidar_sensor_ids.shape
            == lidar_is_val.shape
        ) or len(lidar_times) < 2:
            raise ValueError("Invalid LiDAR metadata")
        expected_lidar_names = LIDARS[expected_dataset]
        actual_lidar_names = tuple(
            str(name) for name in meta["lidar_names"].tolist()
        )
        if actual_lidar_names != expected_lidar_names:
            raise ValueError(
                "LiDAR order mismatch: expected {}, got {}".format(
                    expected_lidar_names, actual_lidar_names
                )
            )
        if not np.array_equal(
            np.unique(lidar_sensor_ids),
            np.arange(len(expected_lidar_names), dtype=np.int64),
        ):
            raise ValueError("lidar_sensor_ids are not contiguous in sensor order")
        lidar_summary = {}
        for lidar_sensor_id, lidar_name in enumerate(expected_lidar_names):
            selected = lidar_sensor_ids == lidar_sensor_id
            sensor_times = lidar_times[selected]
            if len(sensor_times) < 2 or np.any(np.diff(sensor_times) <= 0):
                raise ValueError(
                    "{} timestamps are not increasing".format(lidar_name)
                )
            lidar_summary[lidar_name] = {
                "total": int(np.count_nonzero(selected)),
                "train": int(np.count_nonzero(selected & ~lidar_is_val)),
                "val": int(np.count_nonzero(selected & lidar_is_val)),
            }
        expected_lidar_val = splatad_is_val_mask(
            lidar_sensor_ids, fraction
        )
        if not np.array_equal(lidar_is_val, expected_lidar_val):
            raise ValueError("LiDAR split is not exact SplatAD LINSPACE")
        train_lidar_times = lidar_times[~lidar_is_val]

        sensor_time_min, sensor_time_max = sensor_time_bounds(times, lidar_times)
        if str(_scalar(meta, "time_normalization_scope")) != "all_cameras_all_lidars":
            raise ValueError("time_normalization_scope must cover every camera and lidar")
        stored_bounds = np.asarray(
            [
                float(_scalar(meta, "sensor_time_min")),
                float(_scalar(meta, "sensor_time_max")),
                float(_scalar(meta, "sensor_time_duration")),
            ],
            dtype=np.float64,
        )
        expected_bounds = np.asarray(
            [
                sensor_time_min,
                sensor_time_max,
                sensor_time_max - sensor_time_min,
            ],
            dtype=np.float64,
        )
        if not np.allclose(stored_bounds, expected_bounds, rtol=1e-9, atol=1e-9):
            raise ValueError(
                "Stored sensor time bounds {} do not match {}".format(
                    stored_bounds.tolist(), expected_bounds.tolist()
                )
            )
        expected_frame_gap = normalized_train_frame_gap(
            times,
            camera_ids,
            is_val,
            normalization_time_stamps=np.concatenate([times, lidar_times]),
        )
        frame_gap = float(_scalar(meta, "frame_gap"))
        if not np.isclose(frame_gap, expected_frame_gap, rtol=1e-5, atol=1e-7):
            raise ValueError(
                "frame_gap {} does not match all-sensor formula {}".format(
                    frame_gap, expected_frame_gap
                )
            )
        normalized_camera_times = normalize_sensor_times(
            times, sensor_time_min, sensor_time_max
        )
        normalized_lidar_times = normalize_sensor_times(
            lidar_times, sensor_time_min, sensor_time_max
        )

    ply_path = scene / "points3d.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)
    vertices = PlyData.read(str(ply_path))["vertex"]
    properties = set(vertices.data.dtype.names)
    if "t" not in properties:
        raise ValueError("points3d.ply has no timestamp field")
    point_times = np.asarray(vertices["t"], dtype=np.float64)
    if point_times.size == 0:
        raise ValueError("points3d.ply is empty")
    if np.any(_nearest_distances(point_times, train_lidar_times) > 1e-4):
        raise ValueError("points3d.ply contains held-out/non-train LiDAR sweeps")
    normalized_point_times = normalize_sensor_times(
        point_times, sensor_time_min, sensor_time_max
    )

    if not metadata_only:
        missing_artifacts = _missing_training_artifacts(scene, properties)
        if missing_artifacts:
            raise ValueError(
                _training_readiness_error(
                    scene, expected_dataset, missing_artifacts
                )
            )
        for folder, prefix in (
            ("depth", ""),
            ("semantic", "mask_"),
            ("sky", "mask_"),
        ):
            root = scene / folder
            missing = [
                name for name in image_names
                if resolve_prior_path(
                    root / (prefix + Path(name).stem + ".npz"), required=False
                ) is None
            ]
            if missing:
                raise ValueError(
                    "{}/ is missing {} files (first: {})".format(
                        folder, len(missing), missing[0]
                    )
                )
            for index, name in enumerate(image_names):
                prior_path = root / (prefix + Path(name).stem + ".npz")
                expected_shape = (int(heights[index]), int(widths[index]))
                if folder == "depth":
                    array = load_depth_prior(prior_path)
                    if array.dtype != np.float32 or not np.isfinite(array).all():
                        raise ValueError("Invalid depth dtype/values: {}".format(name))
                    if array.shape not in (expected_shape, expected_shape + (1,)):
                        raise ValueError("Depth dimension mismatch: {}".format(name))
                else:
                    array = load_mask_prior(prior_path, folder)
                    if array.shape != expected_shape:
                        raise ValueError("{} dimension mismatch: {}".format(folder, name))

        flow_dir = scene / "flow"
        train_indices = np.flatnonzero(~is_val)
        train_names = [image_names[index] for index in train_indices]
        missing_flow = [
            name for name in train_names
            if resolve_prior_path(
                flow_dir / (Path(name).stem + ".npz"), required=False
            ) is None
        ]
        if missing_flow:
            raise ValueError(
                "flow/ is missing {} train files (first: {})".format(
                    len(missing_flow), missing_flow[0]
                )
            )
        for index, name in zip(train_indices, train_names):
            flow = load_flow_prior(flow_dir / (Path(name).stem + ".npz"))
            for flow_package in flow:
                _validate_flow_package(
                    flow_package,
                    (int(heights[index]), int(widths[index])),
                    sensor_time_min,
                    sensor_time_max,
                    name,
                )
        if not (scene / "colmap.ply").is_file():
            raise FileNotFoundError(scene / "colmap.ply")

    result = {
        "scene": str(scene),
        "dataset": expected_dataset,
        "cameras": camera_summary,
        "camera_total": len(image_names),
        "camera_train": int(np.count_nonzero(~is_val)),
        "camera_val": int(np.count_nonzero(is_val)),
        "lidar_total": int(len(lidar_times)),
        "lidar_train": int(np.count_nonzero(~lidar_is_val)),
        "lidar_val": int(np.count_nonzero(lidar_is_val)),
        "lidars": lidar_summary,
        "time_normalization": {
            "scope": "all_cameras_all_lidars",
            "sensor_min": sensor_time_min,
            "sensor_max": sensor_time_max,
            "duration": sensor_time_max - sensor_time_min,
            "camera_normalized_min": float(normalized_camera_times.min()),
            "camera_normalized_max": float(normalized_camera_times.max()),
            "lidar_normalized_min": float(normalized_lidar_times.min()),
            "lidar_normalized_max": float(normalized_lidar_times.max()),
            "point_normalized_min": float(normalized_point_times.min()),
            "point_normalized_max": float(normalized_point_times.max()),
            "frame_gap": frame_gap,
        },
        "priors_checked": not metadata_only,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    parser.add_argument("--dataset", required=True, choices=sorted(CAMERAS))
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    validate(args.scene, args.dataset, args.metadata_only)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
